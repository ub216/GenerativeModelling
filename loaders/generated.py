import torch
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, model, num_samples, device, img_size, batch_size):
        self.model = model
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    @torch.no_grad()
    def __getitem__(self, *args, **kwargs):
        sample = self.model.sample(1, self.device, self.img_size, batch_size=1).squeeze(
            0
        )  # (C,H,W)
        return sample, 0  # label not needed


"""
def get_generated_dataloader(model, num_samples, device, img_size, batch_size=64):
    dataset = GeneratedDataset(model, num_samples, device, img_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader
"""
