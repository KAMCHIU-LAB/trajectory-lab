import numpy as np
import pytest
import torch

from trajectorylab.data import Episode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(0)


@pytest.fixture
def episodes():
    rng = np.random.default_rng(4)
    result = []
    for i in range(30):
        length = 8
        result.append(
            Episode(
                str(i),
                rng.normal(size=(length + 1, 3)).astype(np.float32),
                rng.uniform(-1, 1, (length, 1)).astype(np.float32),
                np.full(length, i - 30, dtype=np.float32),
                np.zeros(length, dtype=bool),
                np.asarray([False] * (length - 1) + [True]),
            )
        )
    return result
