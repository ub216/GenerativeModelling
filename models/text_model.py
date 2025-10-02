from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection

import helpers.custom_types as custom_types


class TextModel(nn.Module):
    def __init__(
        self,
        device: custom_types.DeviceType,
        model_id: str = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(
            device
        )
        self.dim = self.text_model.config.projection_dim
        self.device = device

    def forward(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(
            self.device
        )
        outputs = self.text_model(**text_inputs)

        return outputs.text_embeds
