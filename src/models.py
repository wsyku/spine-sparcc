"""BME classification and SPARCC regression model definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


def _resnet_backbone(name: str, input_channels: int = 1) -> tuple[nn.Module, int]:
    if name == "resnet18":
        network = resnet18(weights=None)
    elif name == "resnet34":
        network = resnet34(weights=None)
    else:
        raise ValueError(f"Only resnet18/resnet34 are used, got {name!r}")
    if input_channels != 3:
        old = network.conv1
        network.conv1 = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(network.conv1.weight, mode="fan_out", nonlinearity="relu")
    feature_dim = int(network.fc.in_features)
    return nn.Sequential(*list(network.children())[:-1]), feature_dim


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    activation: str,
    layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    activation_class = nn.GELU if activation == "gelu" else nn.ReLU
    for hidden in hidden_dims:
        layers.append(nn.Linear(current, int(hidden)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden)))
        layers.append(activation_class())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        current = int(hidden)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class SegmentClassificationHead(nn.Module):
    """E02: adjacent mean -> LN -> concatenate segment embedding -> pair MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(512)
        self.pair_classifier = _mlp(528, (128,), 0.2, "relu", False)
        self.segment_embedding = nn.Embedding(3, 16)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)
        self.temperature = 0.5

    def forward(
        self,
        slice_features: torch.Tensor,
        slice_mask: torch.Tensor,
        segment_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_features = 0.5 * (slice_features[:, :-1] + slice_features[:, 1:])
        pair_features = self.pair_norm(pair_features)
        pair_mask = slice_mask[:, :-1] & slice_mask[:, 1:]
        valid_sample = pair_mask.any(dim=1)
        embedding = self.segment_embedding(segment_id.long())
        expanded = embedding.unsqueeze(1).expand(-1, pair_features.shape[1], -1)
        pair_logits = self.pair_classifier(
            torch.cat([pair_features, expanded], dim=-1)
        ).squeeze(-1)
        masked = (pair_logits / self.temperature).masked_fill(
            ~pair_mask, torch.finfo(pair_logits.dtype).min
        )
        pair_weights = torch.softmax(masked, dim=1).masked_fill(~pair_mask, 0.0)
        pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(pair_weights.dtype).eps
        )
        segment_logit = torch.sum(pair_weights * pair_logits, dim=1)
        segment_logit = torch.where(valid_sample, segment_logit, torch.zeros_like(segment_logit))
        return {
            "cls_logit": segment_logit,
            "prob_bme": torch.sigmoid(segment_logit),
            "pair_logits": pair_logits,
            "pair_probabilities": torch.sigmoid(pair_logits).masked_fill(~pair_mask, 0.0),
            "pair_mask": pair_mask,
            "pair_valid_sample": valid_sample,
            "aggregation_weights": pair_weights,
        }


class BMEClassifierNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder, self.feature_dim = _resnet_backbone("resnet18")
        self.classifier = _mlp(512, (), 0.3, "relu", False)
        self.segment_head = SegmentClassificationHead()

    def forward(
        self,
        image: torch.Tensor,
        segment_id: torch.Tensor,
        slice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch, channels, depth, height, width = image.shape
        if slice_mask is None:
            slice_mask = torch.ones(batch, depth, dtype=torch.bool, device=image.device)
        else:
            slice_mask = slice_mask.to(device=image.device, dtype=torch.bool)
        flat = image.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        valid_flat = slice_mask.reshape(-1)
        valid_features = self.encoder(flat[valid_flat]).flatten(1)
        flat_features = valid_features.new_zeros(batch * depth, self.feature_dim)
        flat_features[valid_flat] = valid_features
        features = flat_features.view(batch, depth, self.feature_dim)
        valid_logits = self.classifier(valid_features).squeeze(-1)
        flat_logits = valid_logits.new_zeros(batch * depth)
        flat_logits[valid_flat] = valid_logits
        slice_logits = flat_logits.view(batch, depth)
        segment = self.segment_head(features, slice_mask, segment_id)
        return {
            **segment,
            "slice_logits": slice_logits,
            "slice_probabilities": torch.sigmoid(slice_logits),
            "segment_head_enabled": True,
            "reg_score": segment["cls_logit"] * 0.0,
            "final_score": segment["cls_logit"] * 0.0,
        }


class BMEClassifier(nn.Module):
    """Wrapper names intentionally match the trained checkpoint state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.model = BMEClassifierNetwork()
        self.classification_head = self.model.classifier

    def forward(
        self,
        raw_image: torch.Tensor,
        probability: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        segment_id: Optional[torch.Tensor] = None,
        slice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if segment_id is None:
            raise ValueError("segment_id is required")
        return self.model(raw_image * probability, segment_id, slice_mask)


class SPARCCRegressorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slice_encoder, self.feature_dim = _resnet_backbone("resnet34")
        self.slice_norm = nn.LayerNorm(512)
        self.slice_dropout = nn.Dropout(0.1)
        self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.shared = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.segment_embedding = nn.Embedding(3, 512)
        self.base_head = _mlp(128, (256, 128, 64), 0.3, "gelu", True)
        self.cls_head = None

    def forward(
        self,
        image: torch.Tensor,
        segment_id: torch.Tensor,
        slice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch, channels, depth, height, width = image.shape
        flat = image.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        features = self.slice_encoder(flat).flatten(1).view(batch, depth, self.feature_dim)
        features = features + self.segment_embedding(segment_id.long()).unsqueeze(1)
        features = self.slice_dropout(self.slice_norm(features))
        if slice_mask is None:
            slice_mask = torch.ones(batch, depth, dtype=torch.bool, device=image.device)
        else:
            slice_mask = slice_mask.to(device=image.device, dtype=torch.bool)
        attention_logits = self.attn(features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~slice_mask, torch.finfo(attention_logits.dtype).min
        )
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
        shared = self.shared(pooled)
        score = F.softplus(self.base_head(shared).squeeze(-1))
        zeros = score * 0.0
        return {
            "cls_logit": None,
            "prob_bme": torch.ones_like(score),
            "base_score": score,
            "correction": zeros,
            "reg_score": score,
            "final_score": score,
            "ordinal_logits": None,
            "ordinal_probabilities": None,
            "attention_weights": attention_weights,
        }


class SPARCCRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SPARCCRegressorNetwork()
        self.classification_head = self.model.cls_head

    def forward(
        self,
        raw_image: torch.Tensor,
        probability: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        segment_id: Optional[torch.Tensor] = None,
        slice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if segment_id is None:
            raise ValueError("segment_id is required")
        return self.model(raw_image * probability, segment_id, slice_mask)


def load_model(checkpoint_path: str | Path, task: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    model: nn.Module = BMEClassifier() if task == "classifier" else SPARCCRegressor()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
