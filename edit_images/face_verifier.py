import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceVerifier:
    def __init__(self, device="cuda", image_size: int = 160):
        self.device = device
        # Load once, keep in memory
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        # The user freezes the resnet weights to ensure no accidental training occurs
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.image_size = image_size  # FaceNet expected input size

    def preprocess(self, img: np.ndarray):
        # resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        # standardize (Whitening) - This is what facenet-pytorch expects
        img = (img.astype(np.float32) - 127.5) / 128.0
        # to Tensor
        img = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        )
        return img

    @torch.no_grad()
    def get_similarity(self, face1: np.ndarray, face2: np.ndarray) -> float:
        t1 = self.preprocess(face1)
        t2 = self.preprocess(face2)

        emb1 = self.get_embedding_differentiable(t1)
        emb2 = self.get_embedding_differentiable(t2)

        # L2 Normalization
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine Similarity
        similarity = (emb1 * emb2).sum().item()
        return similarity

    def get_embedding_differentiable(
        self,
        x_tensor: torch.Tensor,
        h_crop_percent: float = 0.7,
        w_crop_percent: float = 0.5,
    ) -> torch.Tensor:
        """
        The method computes the embedding for a batch of tensors (B, C, H, W)
        where x_tensor is in range [-1, 1] (Diffusion model output).
        """
        # The code applies a center crop to simulate the "tight" bounding box required by FaceNet.
        # This removes background and focuses on facial features before resizing.
        h, w = x_tensor.shape[2], x_tensor.shape[3]
        crop_h, crop_w = int(h * h_crop_percent), int(w * w_crop_percent)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2

        x_cropped = x_tensor[
            :, :, start_y : start_y + crop_h, start_x : start_x + crop_w
        ]

        # The cropped tensor is then resized to 160x160 using bilinear interpolation
        # to match the input requirements of InceptionResnetV1.
        x_resized = F.interpolate(
            x_cropped,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Diffusion output is [-1, 1]. FaceNet expects whitened inputs.
        # As established, [-1, 1] maps cleanly to the expected statistical range.
        emb = self.resnet(x_resized)
        return F.normalize(emb, p=2, dim=1)
