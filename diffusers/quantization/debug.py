# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import torch
from config import get_int8_config
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionControlNetPipeline
)
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import check_lora, filter_func, load_calib_prompts, set_fmha

from diffusers.models.attention_processor import Attention

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


DATA_DIR = "/home/georgel/data/chore_project/bytedance/sd_controlnet"

MODEL_ID = {
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd_controlnet": "RV_V51_noninpainting",
    "sdsr": "sd-x2-latent-upscaler",
}

SDXL_FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Half",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        ## Disable quantization for layers in ControlNet
        if "control" in name:
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            elif isinstance(module, Attention):
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
        elif isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 1.5 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            if quant_level >= 4:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            down_samples = [
                    down_sample * conditioning_scale
                    for down_sample in down_samples
                ]
            mid_sample *= conditioning_scale

            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred

def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            # output_type="latent",
            image=[torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_0.pt"), torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_1.pt")],
            generator=torch.manual_seed(11557),
            num_inference_steps=kwargs["n_steps"],
            guidance_scale=7.5,
            # strength=1.0,
            controlnet_conditioning_scale=[1.0, 1.0]).images


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="sd_controlnet",
        choices=[
            "sdxl-1.0",
            "sdxl-turbo",
            "sd1.5",
            "sd3-medium",
            "sd_controlnet",
            "sdsr",
        ],
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Number of denoising steps, for SDXL-turbo, use 1-4 steps",
    )

    # Calibration and quantization parameters
    parser.add_argument("--format", type=str, default="fp8", choices=["int8", "fp8"])
    parser.add_argument("--percentile", type=float, default=1.0, required=False)
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
        help=(
            "Ways to collect the amax of each layers, for example, min-max means min(max(step_0),"
            " max(step_1), ...)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=1.0,
        type=float,
        choices=[0.0, 1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 0: None, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=None, help="Will export the ONNX if not None"
    )
    parser.add_argument("--simulate", action='store_true')
    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    if args.model == "sd1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16, safety_checker=None
        )
    elif args.model == "sd_controlnet":
        controlnets = []
        controlnets.append(ControlNetModel.from_pretrained(f"{DATA_DIR}/control_v11p_sd15_inpaint", torch_dtype=torch.float16))
        controlnets.append(ControlNetModel.from_pretrained(f"{DATA_DIR}/7f2f69197050967007f6bbd23ab5e52f0384162a", torch_dtype=torch.float16))
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            f"{DATA_DIR}/RV_V51_noninpainting",
            controlnet = controlnets,
            safety_checker = None,
            torch_dtype = torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    elif args.model == "sdsr":
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained("/root/checkpoints/sd-x2-latent-upscaler", torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    if args.model == "sd_controlnet":
        backbone = UNet2DConditionControlNetModel(pipe.unet, torch.nn.ModuleList(pipe.controlnet.nets))
    elif args.model != "sd3-medium":
        backbone = pipe.unet
    else:
        backbone = pipe.transformer

    if args.quant_level == 0.0 and args.model == "sd_controlnet":
        if args.simulate:
            images = pipe(prompt=["on the table, in a modern room, soft and bright light, high detailed"],
                        negative_prompt=[
                            "top view, float things, extra connection, adjunct, appendages, stand, bracket, " + \
                            "bad anatomy, text, wood, grid, brown, gray, bubble, high saturation, sunlight, " + \
                            "sun, stripe, spot, empty background, wheel, lowres, ugly, bad hands, cropped, "  + \
                            "worst quality, baby, body, human, brand, bad face"],
                        # output_type="latent",
                        image=[
                            torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_0.pt"), 
                            torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_1.pt")],
                        generator=torch.manual_seed(3),
                        num_inference_steps=args.n_steps,
                        guidance_scale=7.5,
                        # strength=1.0,
                        controlnet_conditioning_scale=[1.0, 1.0]).images
            images[0].save("test.debug.fp16.png")
        modelopt_export_sd(backbone, f"{str(args.onnx_dir)}", args.model)
        return
    elif args.quant_level == 4.0:
        assert args.format != "int8", "We only support fp8 for Level 4 Quantization"
        # assert args.model == "sdxl-1.0", "We only support fp8 for SDXL on Level 4"
        set_fmha(backbone)

    if not args.restore_from:
        # This is a list of prompts
        cali_prompts = load_calib_prompts(
            args.batch_size,
            "./calib/calib_prompts.txt",
        )
        extra_step = (
            1 if args.model == "sd1.5" or args.model == "sdsr" or args.model == "sd_controlnet" else 0
        )  # Depending on the scheduler. some schedulers will do n+1 steps
        if args.format == "int8":
            # Making sure to use global_min in the calibrator for SD 1.5
            assert args.collect_method != "default"
            if args.model == "sd1.5" or args.model == "sdsr" or args.model == "sd_controlnet":
                args.collect_method = "global_min"
            quant_config = get_int8_config(
                backbone,
                args.quant_level,
                args.alpha,
                args.percentile,
                args.n_steps + extra_step,
                collect_method=args.collect_method,
            )
        elif args.format == "fp8":
            if args.collect_method == "default":
                quant_config = SDXL_FP8_DEFAULT_CONFIG
            else:
                raise NotImplementedError

        def forward_loop(backbone):
            if args.model == "sd_controlnet":
                pipe.unet = backbone.unet
                pipe.controlnet = MultiControlNetModel(backbone.controlnets)
            elif args.model != "sd3-medium":
                pipe.unet = backbone
            else:
                pipe.transformer = backbone
            do_calibrate(
                pipe=pipe,
                calibration_prompts=cali_prompts,
                calib_size=args.calib_size,
                n_steps=args.n_steps,
            )

        # All the LoRA layers should be fused
        check_lora(backbone)
        mtq.quantize(backbone, quant_config, forward_loop)
        # mto.save(backbone, f"{args.exp_name}")
    else:
        mto.restore(backbone, args.restore_from)
    quantize_lvl(backbone, args.quant_level)
    mtq.disable_quantizer(backbone, filter_func)
    if args.simulate:
        images = pipe(prompt=["on the table, in a modern room, soft and bright light, high detailed"],
                      negative_prompt=[
                          "top view, float things, extra connection, adjunct, appendages, stand, bracket, " + \
                          "bad anatomy, text, wood, grid, brown, gray, bubble, high saturation, sunlight, " + \
                          "sun, stripe, spot, empty background, wheel, lowres, ugly, bad hands, cropped, "  + \
                          "worst quality, baby, body, human, brand, bad face"],
                      # output_type="latent",
                      image=[
                          torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_0.pt"), 
                          torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_1.pt")],
                      generator=torch.manual_seed(3),
                      num_inference_steps=args.n_steps,
                      guidance_scale=7.5,
                      # strength=1.0,
                      controlnet_conditioning_scale=[1.0, 1.0]).images
        images[0].save(f"test.debug.{args.format}.1.png")

    print(backbone)
    if args.onnx_dir is not None:
        if args.format == "fp8":
            generate_fp8_scales(backbone)
        modelopt_export_sd(backbone, f"{str(args.onnx_dir)}", args.model)
    if args.simulate:
        images = pipe(prompt=["on the table, in a modern room, soft and bright light, high detailed"],
                      negative_prompt=[
                          "top view, float things, extra connection, adjunct, appendages, stand, bracket, " + \
                          "bad anatomy, text, wood, grid, brown, gray, bubble, high saturation, sunlight, " + \
                          "sun, stripe, spot, empty background, wheel, lowres, ugly, bad hands, cropped, "  + \
                          "worst quality, baby, body, human, brand, bad face"],
                      # output_type="latent",
                      image=[
                          torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_0.pt"), 
                          torch.load(f"{DATA_DIR}/test_control_temp_input/input_image_1.pt")],
                      generator=torch.manual_seed(3),#2946901),
                      num_inference_steps=args.n_steps,
                      guidance_scale=7.5,
                      # strength=1.0,
                      controlnet_conditioning_scale=[1.0, 1.0]).images
        images[0].save(f"test.debug.{args.format}.2.png")


if __name__ == "__main__":
    main()
