from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel

import helpers.custom_types as custom_types


class ClipTextModel(nn.Module):
    def __init__(
        self,
        device: custom_types.DeviceType,
        model_id: str = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device)
        self.dim = self.text_model.config.projection_dim
        self.device = device

    def forward(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP models typically have a max length of 77
            return_tensors="pt",
        ).to(self.device)
        outputs = self.text_model(**text_inputs)

        return outputs.text_embeds


class T5TextModel(nn.Module):
    def __init__(
        self,
        device: custom_types.DeviceType,
        model_id: str = "google/t5-v1_1-small",
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = T5EncoderModel.from_pretrained(model_id).to(device)
        self.dim = self.text_model.config.d_model  # 512 for small, 768 for base
        self.device = device
        self.max_length = max_length

    def forward(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.text_model(**text_inputs)

        return outputs.last_hidden_state  # (B, seq_len, dim)
