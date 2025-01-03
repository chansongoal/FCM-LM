# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Union

import torch
import torch.nn as nn

from .backbones import _make_dinov2_model
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    IMAGENET1K = "IMAGENET1K"


def _make_dinov2_linear_classification_head(
    *,
    arch_name: str = "vit_large",
    patch_size: int = 14,
    embed_dim: int = 1024,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")
    # if isinstance(weights, str):
    #     try:
    #         weights = Weights[weights]
    #     except KeyError:
    #         raise AssertionError(f"Unsupported weights: {weights}")

    linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

    if pretrained:
        print('Load head checkpoint from local path: ', weights)
        state_dict = torch.load(weights)
        linear_head.load_state_dict(state_dict, strict=True)

    return linear_head


class _LinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module, layers: int = 1):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.layers = layers

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)

    def forward_backbone(self, x):
        if self.layers == 1:
            x = self.backbone.cls_backbone(x)   # Extract features with the defined cls_backbone
            return x
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True, is_encode=True)
            return x
        else:
            assert False, f"Unsupported number of layers: {self.layers}"

    def forward_head(self, feature_list):
        if self.layers == 1:
            x_norm = self.backbone.norm(feature_list[0])     # Perform norm in the head

            cls_token = x_norm[:, 0]
            patch_tokens = x_norm[:, self.backbone.num_register_tokens + 1 :]
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
        elif self.layers == 4:
            outputs = [self.backbone.norm(out) for out in feature_list]
            class_tokens = [out[:, 0] for out in outputs]
            outputs = [out[:, 1 + self.backbone.num_register_tokens:] for out in outputs]

            x = tuple(zip(outputs, class_tokens))

            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim = 1)
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)


def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_large",
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str, list[str]] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    **kwargs,
):
    backbone = _make_dinov2_model(
        arch_name=arch_name,
        pretrained=pretrained,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        weights=weights[0], # Specify the backbone checkpoint 
        **kwargs,
    )

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    linear_head = _make_dinov2_linear_classification_head(
        arch_name=arch_name,
        patch_size=patch_size,
        embed_dim=embed_dim,
        layers=layers,
        pretrained=pretrained,
        weights=weights[1], # Specify the head checkpoint
        num_register_tokens=num_register_tokens,
    )

    return _LinearClassifierWrapper(backbone=backbone, linear_head=linear_head, layers=layers)


def dinov2_vits14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitb14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitl14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitg14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str, list[str]] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vits14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
