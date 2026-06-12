"""Regression tests for the segmap binary-mask logic.

The bug we are guarding against: Replicator's semantic_segmentation
annotator (default colorize=True) returns (N, H, W, 4) RGBA where
unlabeled geometry is painted (0, 0, 0, 255) and labeled prims get a
non-black color.  Summing all four channels marks every visible pixel
as labeled because alpha=255 everywhere.  The correct mask is
extracted from the R channel: (R > 0) ⇔ labeled.

Source-of-truth lines:
- env obs path:   pursuit_evasion_env.py:_compute_obs_tensor (segmap branch)
- wandb logger:   scripts/skrl/train.py:WandbFPVVideoLogger.step (segmap branch)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers replicating the source-of-truth logic
# ---------------------------------------------------------------------------


def env_segmap_to_mask(segmap: torch.Tensor) -> torch.Tensor:
    """Replicates the env's obs-path processing.

    Mirrors lines in pursuit_evasion_env.py:_compute_obs_tensor.  Kept
    in lockstep so a test failure here flags real divergence.
    """
    if segmap.dim() == 3:
        segmap = segmap.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    elif segmap.dim() == 4 and segmap.shape[1] != 1:
        segmap = segmap.permute(0, 3, 1, 2)[:, :1]  # take R channel
    return (segmap.float() > 0).float()


def wandb_logger_segmap_to_mask(seg_raw: torch.Tensor) -> torch.Tensor:
    """Replicates the wandb logger's per-frame processing for env 0."""
    seg = seg_raw[0].detach().cpu()
    if seg.dim() == 3 and seg.shape[-1] in (3, 4):
        seg = seg[..., 0].float()
    elif seg.dim() == 3 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1).float()
    else:
        seg = seg.float()
    return (seg > 0).to(torch.uint8) * 255


# ---------------------------------------------------------------------------
# Synthetic RGBA inputs (colorize=True default)
# ---------------------------------------------------------------------------


def _make_rgba(n: int = 2, h: int = 8, w: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (N, H, W, 4) RGBA tensor with a labeled-opponent blob in env 0.

    Returns (rgba, expected_mask_for_env0) where expected_mask is (H, W)
    with 1.0 at labeled pixels.
    """
    rgba = torch.zeros((n, h, w, 4), dtype=torch.uint8)
    # Replicator paints unlabeled geometry with (0, 0, 0, 255) — alpha-only.
    rgba[..., 3] = 255
    # In env 0, place a 3x3 "labeled opponent" blob at (2:5, 2:5).
    # Replicator picks a non-black color per class; here we use a vivid red.
    rgba[0, 2:5, 2:5, 0] = 200  # R channel non-zero
    rgba[0, 2:5, 2:5, 1] = 30   # some G
    rgba[0, 2:5, 2:5, 2] = 30   # some B
    expected = torch.zeros((h, w), dtype=torch.float32)
    expected[2:5, 2:5] = 1.0
    return rgba, expected


def _make_class_ids(n: int = 2, h: int = 8, w: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (N, H, W, 1) int32 class-ID tensor (the colorize=False format)."""
    ids = torch.zeros((n, h, w, 1), dtype=torch.int32)
    ids[0, 2:5, 2:5, 0] = 7  # opponent class id
    expected = torch.zeros((h, w), dtype=torch.float32)
    expected[2:5, 2:5] = 1.0
    return ids, expected


# ---------------------------------------------------------------------------
# Env obs path
# ---------------------------------------------------------------------------


def test_env_obs_rgba_marks_only_labeled_pixels():
    rgba, expected = _make_rgba()
    mask = env_segmap_to_mask(rgba)
    # Expect (N, 1, H, W) with 1.0 ONLY at the labeled blob in env 0.
    assert mask.shape == (2, 1, 8, 8)
    assert torch.equal(mask[0, 0], expected)
    # Env 1 has no labeled prim → all zero.
    assert torch.all(mask[1] == 0)


def test_env_obs_alpha_only_pixel_does_not_leak_through_mask():
    """Direct regression for the original bug: alpha=255 must NOT light the mask."""
    rgba = torch.zeros((1, 4, 4, 4), dtype=torch.uint8)
    rgba[..., 3] = 255  # full-frame "unlabeled" color from Replicator
    mask = env_segmap_to_mask(rgba)
    assert mask.sum().item() == 0, "alpha-only pixels must not be flagged as labeled"


def test_env_obs_class_id_path_still_works():
    """colorize=False legacy path: (N, H, W, 1) int IDs, mask = (id > 0)."""
    ids, expected = _make_class_ids()
    mask = env_segmap_to_mask(ids)
    assert mask.shape == (2, 1, 8, 8)
    assert torch.equal(mask[0, 0], expected)
    assert torch.all(mask[1] == 0)


# ---------------------------------------------------------------------------
# WandB logger path (single env 0)
# ---------------------------------------------------------------------------


def test_wandb_logger_uses_r_channel_not_alpha():
    rgba, expected = _make_rgba()
    mask = wandb_logger_segmap_to_mask(rgba)  # uint8 0/255
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.uint8
    # The labeled region must be 255, the rest must be 0.
    assert torch.equal((mask > 0).float(), expected)
    assert mask.max().item() == 255
    assert mask.min().item() == 0


def test_wandb_logger_full_alpha_unlabeled_frame_is_all_black():
    """Without the R-channel fix this frame would be all-white (the bug)."""
    rgba = torch.zeros((1, 16, 16, 4), dtype=torch.uint8)
    rgba[..., 3] = 255
    mask = wandb_logger_segmap_to_mask(rgba)
    assert mask.sum().item() == 0


def test_wandb_logger_handles_3channel_rgb_no_alpha():
    """Camera may also return (H, W, 3) RGB without alpha — must still extract R."""
    rgb = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)
    rgb[0, 1, 1, 0] = 100  # one labeled pixel
    mask = wandb_logger_segmap_to_mask(rgb)
    assert mask[1, 1].item() == 255
    rgb_off = mask.clone()
    rgb_off[1, 1] = 0
    assert rgb_off.sum().item() == 0


def test_wandb_logger_handles_class_id_format():
    ids, expected = _make_class_ids()
    mask = wandb_logger_segmap_to_mask(ids)
    assert torch.equal((mask > 0).float(), expected)
