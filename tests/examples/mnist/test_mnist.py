from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

examples_path = Path(__file__).parent.parent.parent.parent / "examples"
if str(examples_path) not in sys.path:
    sys.path.insert(0, str(examples_path))


def test_mnist_model_forward_shape():
    from mnist.mnist_model import MNISTModel

    model = MNISTModel()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10)


def test_mnist_model_num_parameters():
    from mnist.mnist_model import MNISTModel

    model = MNISTModel()
    num_params = model.num_parameters()
    assert num_params > 100_000
    assert num_params < 500_000


def test_mnist_model_deterministic_with_seed():
    from mnist.mnist_model import MNISTModel

    torch.manual_seed(42)
    model1 = MNISTModel()
    x1 = torch.randn(2, 1, 28, 28)
    out1 = model1(x1)

    torch.manual_seed(42)
    model2 = MNISTModel()
    x2 = torch.randn(2, 1, 28, 28)
    out2 = model2(x2)

    assert torch.allclose(out1, out2)


def test_evaluate_batch():
    from mnist.evaluate import evaluate_batch
    from mnist.mnist_model import MNISTModel

    model = MNISTModel()
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    loss, accuracy = evaluate_batch(model, images, labels)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


def test_model_gradients_disabled_during_eval():
    from mnist.evaluate import evaluate_batch
    from mnist.mnist_model import MNISTModel

    model = MNISTModel()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))

    evaluate_batch(model, images, labels)

    for param in model.parameters():
        assert param.grad is None


@pytest.mark.slow
def test_mnist_data_loading():
    from mnist.data import get_mnist_subset

    train_images, train_labels, test_images, test_labels = get_mnist_subset(
        n_train=100, n_test=20
    )

    assert train_images.shape == (100, 1, 28, 28)
    assert train_labels.shape == (100,)
    assert test_images.shape == (20, 1, 28, 28)
    assert test_labels.shape == (20,)

    assert train_labels.min() >= 0
    assert train_labels.max() <= 9


@pytest.mark.slow
def test_mnist_model_on_real_data():
    from mnist.data import get_mnist_subset
    from mnist.evaluate import evaluate_batch
    from mnist.mnist_model import MNISTModel

    train_images, train_labels, _, _ = get_mnist_subset(n_train=32, n_test=10)
    model = MNISTModel()

    loss, accuracy = evaluate_batch(model, train_images, train_labels)

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
    assert accuracy < 0.5


@pytest.mark.slow
def test_get_mnist_loaders():
    from mnist.data import get_mnist_loaders

    train_loader, test_loader = get_mnist_loaders(batch_size=32)

    train_batch = next(iter(train_loader))
    assert train_batch[0].shape == (32, 1, 28, 28)
    assert train_batch[1].shape == (32,)

    test_batch = next(iter(test_loader))
    assert test_batch[0].shape == (32, 1, 28, 28)
    assert test_batch[1].shape == (32,)


@pytest.mark.slow
def test_get_mnist_loaders_reproducible_with_seed():
    from mnist.data import get_mnist_loaders

    train_loader1, _ = get_mnist_loaders(batch_size=16, seed=42)
    batch1 = next(iter(train_loader1))

    train_loader2, _ = get_mnist_loaders(batch_size=16, seed=42)
    batch2 = next(iter(train_loader2))

    assert torch.allclose(batch1[0], batch2[0])
    assert torch.equal(batch1[1], batch2[1])


@pytest.mark.slow
def test_evaluate_model():
    from mnist.data import get_mnist_loaders
    from mnist.evaluate import evaluate_model
    from mnist.mnist_model import MNISTModel

    _, test_loader = get_mnist_loaders(batch_size=64)
    model = MNISTModel()

    loss, accuracy = evaluate_model(model, test_loader)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
