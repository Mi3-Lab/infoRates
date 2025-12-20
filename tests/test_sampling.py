import numpy as np
from info_rates.sampling.temporal import apply_aliasing, subsample


def test_apply_aliasing_basic():
    frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    out = apply_aliasing(frames, frame_percent=50, stride=2)
    assert len(out) <= 50 and len(out) > 0


def test_subsample_basic():
    frames = np.random.randint(0, 255, (80, 224, 224, 3), dtype=np.uint8)
    out = subsample(frames, coverage=25, stride=4)
    assert len(out) > 0
