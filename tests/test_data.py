import copy

import numpy as np
import pytest

from trajectorylab.data import (
    VIEWS,
    Dataset,
    Normalizer,
    flatten_observation,
    returns_to_go,
    select_view,
    split_episodes,
)


def test_roundtrip_and_fingerprint(tmp_path, episodes):
    dataset = Dataset(episodes, {"source": "test"})
    dataset.save(tmp_path)
    restored = Dataset.load(tmp_path)
    assert restored.digest == dataset.digest
    assert np.array_equal(restored.episodes[-1].observations, episodes[-1].observations)
    manifest = tmp_path / "dataset.json"
    manifest.write_text(manifest.read_text().replace('"source": "test"', '"source": "changed"'))
    with pytest.raises(ValueError, match="fingerprint"):
        Dataset.load(tmp_path)


@pytest.mark.parametrize("defect", ["nan", "boundary", "alignment", "action"])
def test_invalid_episodes_are_rejected(episodes, defect):
    episode = copy.deepcopy(episodes[0])
    if defect == "nan":
        episode.observations[0, 0] = np.nan
    elif defect == "boundary":
        episode.terminated[2] = True
    elif defect == "alignment":
        episode.observations = episode.observations[:-1]
    else:
        episode.actions[0] = 2.0
    with pytest.raises(ValueError):
        episode.validate()


def test_split_has_no_identity_or_content_leakage(episodes):
    duplicate = copy.deepcopy(episodes[0])
    duplicate.id = "different-id-same-content"
    parts = split_episodes(episodes + [duplicate], 42)
    content = [{e.digest for e in values} for values in parts.values()]
    assert not content[0] & content[1]
    assert not content[0] & content[2]
    assert not content[1] & content[2]
    repeated = split_episodes(episodes + [duplicate], 42)
    assert [[e.id for e in v] for v in parts.values()] == [
        [e.id for e in v] for v in repeated.values()
    ]


@pytest.mark.parametrize("view", VIEWS)
def test_views_preserve_episodes_and_match_budget(episodes, view):
    result = select_view(episodes, view, 80, 42, tolerance=0.0)
    assert sum(map(len, result)) == 80
    assert len({e.id for e in result}) == len(result)
    assert all(any(e is original for original in episodes) for e in result)


def test_impossible_budget_fails_instead_of_silently_shrinking(episodes):
    with pytest.raises(ValueError, match="complete episodes"):
        select_view(episodes, "uniform", 81, 0, tolerance=0)


def test_top_return_really_selects_the_highest_returns(episodes):
    result = select_view(episodes, "top_return", 80, 0, tolerance=0)
    assert [e.id for e in result] == [str(i) for i in range(29, 19, -1)]


def test_normalizer_uses_only_its_training_input():
    train = np.asarray([[0.0, 1], [2.0, 1]], dtype=np.float32)
    normalizer = Normalizer.fit(train)
    assert np.allclose(normalizer.mean, [1, 1])
    assert np.isfinite(normalizer(np.asarray([[1000, 1]], dtype=np.float32))).all()
    assert np.allclose(normalizer.mean, [1, 1])
    assert np.allclose(Normalizer.from_state(normalizer.state())(train), normalizer(train))


def test_rtg_and_observation_key_order():
    assert np.array_equal(returns_to_go(np.asarray([1, 2, 3])), [6, 5, 3])
    assert np.array_equal(flatten_observation({"z": [2, 3], "a": [1]}), [1, 2, 3])
