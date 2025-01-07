# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.ops import compute_padding
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures
from compressai.zoo.image_vbr import model_architectures as architectures_vbr

#gcs
import numpy as np 
from compressai.datasets import FeatureFolder
import copy

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".npy",
)

architectures.update(architectures_vbr)


def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

#gcs, note that the mse computed here does not consider the clip operation, this is not the actual mse
def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, trun_low: float = -5.0, trun_high: float = 5.0
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {} 
    interval = trun_high - trun_low
    org = (org * interval + trun_low).clamp(trun_low, trun_high)  # verified, exactly same as original feature before scaling but after clip!!!
    # print('scale', [f"{x:.8f}" for x in org[0, 0, 0, :10]])
    rec = (rec * interval + trun_low).clamp(trun_low, trun_high)
    metrics["mse"] = (org - rec).pow(2).mean().item()   #verified, exactly same as the numpy computation in mse.py
    return metrics


@torch.no_grad()
#gcs
def inference(model, x, vbr_stage=None, vbr_scale=None):
    # x = x.unsqueeze(0)    #gcs, no need for feature, already [N,C,H,W] 4 dims

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)

    start = time.time()
    out_enc = (
        model.compress(x_padded)
        if vbr_scale is None
        else model.compress(x_padded, stage=vbr_stage, s=0, inputscale=vbr_scale)
    )
    enc_time = time.time() - start

    start = time.time()
    out_dec = (
        model.decompress(out_enc["strings"], out_enc["shape"])
        if vbr_scale is None
        else model.decompress(
            out_enc["strings"],
            out_enc["shape"],
            stage=vbr_stage,
            s=0,
            inputscale=vbr_scale,
        )
    )
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # input images are 8bit RGB for now
    num_points = x.size(0) * x.size(1) * x.size(2) * x.size(3)
    bpfp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_points

    #gcs, return rec_feat
    rec_feat = out_dec["x_hat"]
    return rec_feat, {
        #gcs
        "bpfp": bpfp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, vbr_stage=None, vbr_scale=None):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = (
        model.forward(x)
        if vbr_scale is None
        else model.forward(x, stage=vbr_stage, inputscale=vbr_scale)
    )
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True, progress=False
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    
    #gcs, remove "module."
    # if arch == "bmshj2018-hyperprior":
    #     prefix = "module."
    #     state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    # print(state_dict.keys())
    model_cls = architectures[arch]
    if arch in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
        net = model_cls.from_state_dict(state_dict, vr_entbttlnck=True)
        if not no_update:
            net.update(force=True, scale=net.Gain[-1])
    else:
        net = model_cls.from_state_dict(state_dict)
        if not no_update:
            net.update(force=True)

    return net.eval()


def eval_model(
    model: nn.Module,
    outputdir: Path,
    inputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    vbr_stage=None,
    vbr_scale=None,
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    is_vbr_model = args["architecture"].endswith("-vbr")
    #gcs
    model_type = args['model_type']; task = args["task"]; trun_flag = args["trun_flag"]; trun_low = args["trun_low"]
    trun_high = args['trun_high']; quant_type = args['quant_type']; qsamples = args['qsamples']; bit_depth = args['bit_depth']

    for filepath in filepaths:
        #gcs, load feature in the same way of training
        feat = np.load(filepath)
        org_feat = copy.deepcopy(feat)
        N, C, H, W = feat.shape
        #gcs, preprocessing
        if not args["half"]: feat = feat.astype(np.float32)
        if trun_flag == True: feat = FeatureFolder.truncation(feat, trun_low, trun_high)
        feat = FeatureFolder.uniform_quantization(feat, trun_low, trun_high, bit_depth)
        feat = FeatureFolder.packing(feat, model_type)
        x = torch.from_numpy(feat).to(device)
        x = x.unsqueeze(0); x = x.unsqueeze(0) # reshape to [1,1,H,W]

        if not entropy_estimation:
            if args["half"]:
                model = model.half()
                x = x.half()
            #gcs, return rec_feat
            rec_feat, rv = (
                        inference(model, x)
                        if not is_vbr_model
                        else inference(model, x, vbr_stage, vbr_scale)
            )
            #gcs, postprocessing
            rec_feat = rec_feat.squeeze(0); rec_feat = rec_feat.squeeze(0)
            rec_feat = rec_feat.cpu().detach().numpy()
            rec_feat = FeatureFolder.unpacking(rec_feat, [N, C, H, W], model_type)
            rec_feat = FeatureFolder.uniform_dequantization(rec_feat, trun_low, trun_high, bit_depth)
            # Feature dtype
            dtype = np.float16 if model_type=='sd3' else np.float32
            rec_feat = rec_feat.astype(dtype)
            # Compute MSE
            feat_mse = np.mean((rec_feat-org_feat)**2)
            rv["mse"] = feat_mse
        else:
            rv = (
                inference_entropy_estimation(model, x)
                if not is_vbr_model
                else inference_entropy_estimation(model, x, vbr_stage, vbr_scale)
            )
        for k, v in rv.items():
            metrics[k] += v
        if args["per_image"]:
            if not Path(outputdir).is_dir():
                raise FileNotFoundError("Please specify output directory")

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(
                inputdir
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            #gcs, save rec_feat
            rec_feat_name = output_subdir / f"{filepath.stem}.npy"
            np.save(rec_feat_name, rec_feat)

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

# Custom type parsing function
def parse_truncation(value):
    try:
        # Try to parse as a float
        return float(value)
    except ValueError:
        # Try to parse as a list of floats (comma-separated)
        try:
            return [float(x) for x in value.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid input for --truncation: {value}. Must be a float or a list of floats."
            )

def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    #gcs, add model_type, trun_flag, trun_low, trun_high, quant_type, qsamples, bit_depth
    parent_parser.add_argument(
        "-model_type",
        "--model_type",
        type=str,
        default="sd3",
        help="Please input the model_type.",
    )
    parent_parser.add_argument(
        "-task",
        "--task",
        type=str,
        default="tti",
        help="Please input the task.",
    )
    parent_parser.add_argument(
        "-trun_flag",
        "--trun_flag",
        type=bool,
        default=True,
        help="Please input the trun_flag.",
    )
    parent_parser.add_argument(
        "-trun_low",
        "--trun_low",
        type=parse_truncation,
        default=-5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parent_parser.add_argument(
        "-trun_high",
        "--trun_high",
        type=parse_truncation,
        default=5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parent_parser.add_argument(
        "-quant_type",
        "--quant_type",
        type=str,
        default="uniform",
        help="Please input the quant_type.",
    )
    parent_parser.add_argument(
        "-qsamples",
        "--qsamples",
        type=int,
        default=0,
        help="Please input the qsamples.",
    )
    parent_parser.add_argument(
        "-bit_depth",
        "--bit_depth",
        type=int,
        default=1,
        help="Please input the bit_depth.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )
    # Options for variable bitrate (vbr) models
    parent_parser.add_argument(
        "--vbr_quantstep",
        dest="vbr_quantstepsizes",
        type=str,
        default="10.0000,7.1715,5.1832,3.7211,2.6833,1.9305,1.3897,1.0000",
        help="Quantization step sizes for variable bitrate (vbr) model. Floats [10.0 , 1.0] (example: 10.0,8.0,6.0,3.0,1.0)",
    )
    parent_parser.add_argument(
        "--vbr_tr_stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="Stage in vbr model training. \
            1: Model behaves/runs like a regular single-rate \
            model without using any vbr tool (use for training/testing a model for single/highest lambda). \
            2: Model behaves/runs like a vbr model using vbr tools. (use for post training stage=1 result.)",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        type=str,
        default="1",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


def main(argv):  # noqa: C901
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    is_vbr_model = args.architecture.endswith("-vbr")

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        args.qualities = [int(q) for q in args.qualities.split(",") if q]
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        if is_vbr_model:
            opts += (0,)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d} "
    else:
        runs = args.checkpoint_paths
        opts = (args.architecture, args.no_update)
        if is_vbr_model:
            opts += (args.checkpoint_paths[0],)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s} "

    if is_vbr_model:
        if args.source == "checkpoint":
            assert (
                len(args.checkpoint_paths) <= 1
            ), "Use only one checkpoint for vbr model."
        scales = [1.0 / float(q) for q in args.vbr_quantstepsizes.split(",") if q]
        runs = sorted(scales)
        runs = torch.tensor(runs)
        log_fmt = "\rEvaluating quant step {run:5.2f} "
        model = load_func(*opts)
        # set some arch specific params for vbr
        model.no_quantoffset = False
        if args.architecture in ["mbt2018-vbr"]:
            model.scl2ctx = True
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            runs = runs.to("cuda")

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(
                log_fmt.format(*opts, run=(run if not is_vbr_model else 1.0 / run))
            )
            sys.stderr.flush()
        if not is_vbr_model:
            model = load_func(*opts, run)
        else:
            # update bottleneck for every new quant_step if vbr bottleneck is used in the model
            if (
                args.architecture in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]
                and args.vbr_tr_stage == 2
            ):
                model.update(force=True, scale=run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            run_ = run if not is_vbr_model else args.checkpoint_paths[0]
            cpt_name = Path(run_).name[: -len(".tar.pth")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available() and not is_vbr_model:
            model = model.to("cuda")
        args_dict = vars(args)
        metrics = eval_model(
            model,
            args.output_directory,
            args.dataset,
            filepaths,
            trained_net=trained_net,
            description=description,
            vbr_stage=None if not is_vbr_model else args.vbr_tr_stage,
            vbr_scale=None if not is_vbr_model else run,
            **args_dict,
        )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{args.architecture}-{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
