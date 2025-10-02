from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader

from loaders import GeneratedDataset


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for models. All models should inherit from this class
    and implement the sample method.
    """

    def __init__(self):
        super().__init__()
        self.has_conditional_generation = False

    @abstractmethod
    def sample(self, num_samples, device, *args, **kwargs):
        """
        Generate samples from the model.
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device to perform computation on.
            *args, **kwargs: Additional arguments specific to the model.
        Returns:
            Tensor of generated samples.
        """
        pass

    def wrap_sampler_to_loader(self, num_samples, device, img_size, batch_size=16):
        """
        Wrap the model's sampling method into a DataLoader for evaluation.
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device to perform computation on.
            img_size (int): Size of the generated images (assumed square).
            batch_size (int): Batch size for the DataLoader.
        Returns:
            DataLoader yielding generated samples.
        """
        dataset = GeneratedDataset(self, num_samples, device, img_size, batch_size)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return dataloader
