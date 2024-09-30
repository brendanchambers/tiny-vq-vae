# notes, bc 2024-09-24 exploring the autoencoder.py example

# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/bc/github/brendanchambers/tiny-vq-vae/local_packages/vector-quantize-pytorch/vector_quantize_pytorch")
from vector_quantize_pytorch import VectorQuantize
from dataclasses import dataclass

@dataclass
class Args():
    lr: float = 3e-4
    train_iter: int = 100
    num_codes: int = 256
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_out_dir: str = "/home/bc/github/brendanchambers/tiny-vq-vae/models/checkpoints/explore-autoencoder-fashion-mnist_it100_lr3e-4_n256"
config = Args()

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VectorQuantize(dim=32, accept_image_fmap = True, **vq_kwargs),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        return

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss


def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(config.device), y.to(config.device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / config.num_codes * 100:.3f}"
        )
    return model


if __name__ == "__main__":
        
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = DataLoader(
        datasets.FashionMNIST(
            root="~/data/fashion_mnist", train=True, download=True, transform=transform
        ),
        batch_size=256,
        shuffle=True,
    )

    print("baseline")
    torch.random.manual_seed(config.seed)
    model = SimpleVQAutoEncoder(codebook_size=config.num_codes).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model = train(model, train_dataset, train_iterations=config.train_iter)


    # write model out
    import os
    import json
    import dataclasses
    os.makedirs(config.model_out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{config.model_out_dir}/checkpoint.pt")
    with open(f"{config.model_out_dir}/config.json", 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=1)