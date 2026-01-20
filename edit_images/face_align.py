import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


def _to_rgb_uint8(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_bgr_uint8(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def estimate_similarity_transform(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray:
    """
    Computes a 2x3 similarity transform (rotation, scale, translation)
    that preserves aspect ratio.
    """
    assert src_pts.shape == dst_pts.shape and src_pts.shape[0] >= 3
    # LMEDS is robust to landmark outliers
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Failed to estimate similarity transform")
    return M.astype(np.float32)


class FaceAligner:
    def __init__(
        self,
        image_size: int = 64,
        device: str = "cuda",
        relaxation_scale: float = 0.975,  # 0.925-0.975 matches CelebA distribution
        min_face_size: int = 40,
    ):
        self.image_size = image_size
        self.device = device
        self.mtcnn = MTCNN(
            image_size=None, keep_all=True, min_face_size=min_face_size, device=device
        )

        # original celebA template (178x218)
        train_template = np.array(
            [[69, 111], [108, 111], [88, 135], [71, 152], [105, 152]], dtype=np.float32
        )

        # simulate the training preprocessing steps:
        # training logic: Take 178x218 -> Center Crop to 178x178 -> Resize to 64x64
        # this means 20 pixels were removed from the top ( (218-178)/2 )
        top_offset = (218 - 178) / 2

        # Shift Y coordinates up by the offset
        cropped_template = train_template.copy()
        cropped_template[:, 1] -= top_offset

        # normalise the crop box (178x178) to [0, image_size)
        # now we divide by 178 for BOTH X and Y because it's a square crop
        self.target_landmarks = (cropped_template / 178.0) * image_size
        self.relaxation_scale = relaxation_scale

    def detect(self, img_bgr: np.ndarray):
        img_rgb = _to_rgb_uint8(img_bgr)
        boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
        return boxes, probs, landmarks

    def align_largest(self, img_bgr, debug_save=False):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _, lms = self.mtcnn.detect(img_rgb, landmarks=True)

        if boxes is None or lms is None:
            return None, None

        # pick largest face
        idx = np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        src_lm = lms[idx].astype(np.float32)

        # we shrink the target template toward the center of the image_sizeximage_size square.
        # This forces the warp to include MORE surrounding context (zoom out)
        # to make the face fit into that smaller central target.
        # This is needed because the training data (celebA) had faces that aren't tightly cropped.
        target_center = np.array([self.image_size / 2, self.image_size / 2])

        # Note: scale < 1.0 = Zoom Out (Relaxed), scale > 1.0 = Zoom In (Tight)
        relaxed_target_lm = (
            self.target_landmarks - target_center
        ) * self.relaxation_scale + target_center

        # compute similarity transform
        # map original landmarks to the relaxed target
        M = estimate_similarity_transform(src_lm, relaxed_target_lm)

        aligned = cv2.warpAffine(
            img_rgb,
            M,
            (self.image_size, self.image_size),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # DEBUG: Visualization to confirm mouth/eyes are where they should be
        if debug_save:
            debug_img = aligned.copy()
            for pt in relaxed_target_lm:
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
            cv2.imwrite(
                "debug_alignment.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            )

        meta = {"M": M, "orig_size": (img_bgr.shape[1], img_bgr.shape[0])}
        return aligned, meta

    def unalign_and_paste(self, edited_rgb, original_bgr, meta, feather=10):
        # Invert the mapping: Aligned image_sizeximage_size -> Original Resolution Space
        Minv = cv2.invertAffineTransform(meta["M"])
        warped_edit = cv2.warpAffine(
            edited_rgb,
            Minv,
            meta["orig_size"],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask[5:-5, 5:-5] = 1.0  # Inset to prevent edge artifacts

        mask_w = cv2.warpAffine(
            mask,
            Minv,
            meta["orig_size"],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        if feather > 0:
            k = feather * 2 + 1
            mask_w = cv2.GaussianBlur(mask_w, (k, k), 0)

        mask_w = np.clip(mask_w[..., None], 0, 1)
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        composite = (mask_w * warped_edit + (1 - mask_w) * orig_rgb).astype(np.uint8)
        return _to_bgr_uint8(composite)
