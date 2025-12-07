from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).float().mean()
    return loss.item(), accuracy.item()
