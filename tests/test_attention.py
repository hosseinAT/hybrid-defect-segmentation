import pytest
import torch
from visionseg.model import DefectNet, WindowAttention, defect_loss, partition, reverse
from visionseg.run import selection


def test_partition_reversal_preserves_token_order():
    x = torch.arange(2 * 8 * 12 * 3).reshape(2, 8, 12, 3)
    assert torch.equal(reverse(partition(x, 4), 4, 8, 12, 2), x)


@pytest.mark.parametrize("shift", [0, 2])
def test_padding_keeps_output_and_gradients_finite(shift):
    model = WindowAttention(dim=8, heads=2, shift=shift)
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = model(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.qkv.weight.grad.abs().sum() > 0


def test_shift_does_not_connect_opposite_image_boundaries():
    torch.manual_seed(12)
    attention = WindowAttention(dim=8, heads=2, shift=2).eval()
    x = torch.randn(1, 8, 8, 8)
    changed = x.clone()
    changed[:, :, :, -1] += torch.arange(8).reshape(1, 8, 1)
    # Without the cyclic-boundary mask, the left edge sees the last column.
    torch.testing.assert_close(attention(x)[:, :, :, 0], attention(changed)[:, :, :, 0])


@pytest.mark.parametrize("variant", ["cnn", "attention", "fusion", "hybrid"])
def test_variants_support_non_square_input_and_empty_masks(variant):
    torch.set_num_threads(2)
    model = DefectNet(variant)
    x = torch.randn(1, 1, 33, 35)
    target = torch.zeros_like(x)
    output = model(x)
    assert output.shape == target.shape
    loss = defect_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad.abs().sum() > 0


def test_validation_threshold_selection_uses_confusion_counts():
    import numpy as np
    probabilities = np.array([.2, .4, .6, .8])
    targets = np.array([0, 1, 1, 1])
    score, threshold = selection(probabilities, targets, [.3, .5, .7])
    assert score == 1. and threshold == .3
