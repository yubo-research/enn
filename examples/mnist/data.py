from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
    seed: int | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_mnist_subset(
    n_train: int = 1000,
    n_test: int = 200,
    data_dir: str = "./data",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)

    train_indices = torch.randperm(len(train_dataset), generator=generator)[:n_train]
    test_indices = torch.randperm(len(test_dataset), generator=generator)[:n_test]

    train_images = torch.stack([train_dataset[i][0] for i in train_indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    test_images = torch.stack([test_dataset[i][0] for i in test_indices])
    test_labels = torch.tensor([test_dataset[i][1] for i in test_indices])

    return train_images, train_labels, test_images, test_labels
