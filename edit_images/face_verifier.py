import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceVerifier:
    def __init__(self, device="cuda"):
        self.device = device
        # Load once, keep in memory
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def preprocess(self, img: np.ndarray):
        # resize
        img = cv2.resize(img, (160, 160))
        # standardize (Whitening) - This is what facenet-pytorch expects
        img = (img - 127.5) / 128.0
        # to Tensor
        img = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        )
        return img

    @torch.no_grad()
    def get_similarity(self, face1: np.ndarray, face2: np.ndarray) -> float:
        t1 = self.preprocess(face1)
        t2 = self.preprocess(face2)

        emb1 = self.resnet(t1)
        emb2 = self.resnet(t2)

        # L2 Normalization
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine Similarity
        similarity = (emb1 * emb2).sum().item()
        return similarity
