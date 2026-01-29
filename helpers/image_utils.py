from typing import Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(
    image_size: Tuple[int, int],
    degree: int = 5,
    translate: Tuple[float, float] = (0.05, 0.05),
    scale: Tuple[float, float] = (0.95, 1.05),
    shear: float = 1.0,
) -> transforms.Compose:

    rotate = transforms.RandomAffine(
        degrees=degree,
        translate=None,
        scale=None,
        shear=None,
        interpolation=InterpolationMode.BILINEAR,
        fill=None,
    )
    translate = transforms.RandomAffine(
        degrees=0,
        translate=translate,
        scale=None,
        shear=None,
        interpolation=InterpolationMode.BILINEAR,
        fill=None,
    )
    scale = transforms.RandomAffine(
        degrees=0,
        translate=None,
        scale=scale,
        shear=None,
        interpolation=InterpolationMode.BILINEAR,
        fill=None,
    )
    shear = transforms.RandomAffine(
        degrees=0,
        translate=None,
        scale=None,
        shear=shear,
        interpolation=InterpolationMode.BILINEAR,
        fill=None,
    )
    color_jitter = transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.1
    )
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    flip = transforms.RandomHorizontalFlip(p=0.5)

    geometric_transform_list = [rotate, translate, scale, shear]
    geometric_transform_list = [
        transforms.RandomApply([t], p=0.5) for t in geometric_transform_list
    ]
    geometric_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Pad(int(max(image_size) * 0.15), padding_mode="edge"),
        ]
        + geometric_transform_list
        + [transforms.CenterCrop(image_size)]
    )
    photometric_transform = transforms.Compose(
        [
            transforms.RandomApply([color_jitter], p=0.5),
            transforms.RandomApply([gaussian_blur], p=0.1),
        ]
    )
    transform = transforms.Compose(
        [
            geometric_transform,
            flip,
            photometric_transform,
            transforms.ToTensor(),
        ]
    )
    return transform
