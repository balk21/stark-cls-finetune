"""ft_shots (k-shot): which frames are chosen, that they are fine-tuned on in one batch, the experiment name
and that the evaluation leaves them out. Without a network or GPU."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from lib.test.tracker.ft_sampling import shot_frames
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.evaluation import evaluate_sequence, got10k_ious
from stark_ft.test.tracker_factory import extra_shot_frames, shot_samples

BOX = [10.0, 10.0, 40.0, 30.0]
FAR = [200.0, 200.0, 10.0, 10.0]  # no overlap with BOX
GT = [BOX] + [BOX if i % 3 else None for i in range(1, 100)]  # the target is absent in every third frame
KSHOT = dict(ft_mode="init", ft_shots=5, ft_pos_jitter=False, update_interval=99999, seed=42, sequences=["bull"])


def test_shot_frames():
    assert shot_frames(GT, 1, 42) == [0]
    frames = shot_frames(GT, 5, 42)
    assert frames[0] == 0 and len(frames) == 5 and frames == sorted(set(frames))
    assert all(GT[i] is not None for i in frames)
    # np.random.default_rng(seed).choice over the visible frames: same seed -> same frames
    eligible = [i for i in range(1, len(GT)) if GT[i] is not None]
    assert frames == [0] + sorted(np.random.default_rng(42).choice(eligible, 4, replace=False).tolist())
    assert shot_frames(GT, 10, 42)[:1] == [0] and len(shot_frames(GT, 10, 42)) == 10
    try:
        shot_frames([BOX, None, BOX], 5, 0)  # one visible frame besides the first
        raise AssertionError("no error")
    except ValueError:
        pass


def test_config():
    cfg = ExperimentConfig(**KSHOT).validate()
    assert cfg.shots() == 5
    assert cfg.experiment_name == "st101_init_pos_5shot_lr0.0001_ep15_noupd_s42_nojit_bull"
    assert replace(cfg, ft_shots=1).experiment_name == "st101_init_pos_lr0.0001_ep15_noupd_s42_nojit_bull"
    assert extra_shot_frames(cfg, GT) == shot_frames(GT, 5, 42)[1:]
    # no effect without fine-tuning on the first frame: not in the name, no extra frames
    for other in (replace(cfg, ft_mode="none"), replace(cfg, ft_mode="online", ft_epochs_init=0)):
        assert other.shots() == 1 and "shot" not in other.experiment_name and extra_shot_frames(other, GT) == []
    assert extra_shot_frames(replace(cfg, ft_shots=1), GT) == []
    for bad in (dict(ft_shots=0), dict(ft_shots=5, dataset="got10k_test")):
        try:
            ExperimentConfig(**bad).validate()
            raise AssertionError(f"accepted: {bad}")
        except ValueError:
            pass


def test_shot_samples():
    cfg = ExperimentConfig(**KSHOT)
    images = [f"frame{i}.jpg" for i in range(len(GT))]
    frames, samples = shot_samples(cfg, GT, images, lambda path: path)
    assert frames == shot_frames(GT, 5, 42)[1:]
    assert samples == [(f"frame{i}.jpg", BOX) for i in frames]
    assert shot_samples(replace(cfg, ft_shots=1), GT, images, lambda path: path) == ([], [])


def _tracker(samples):
    from lib.test.tracker.stark_st_ft import STARK_ST_FT
    t = STARK_ST_FT.__new__(STARK_ST_FT)  # without __init__: no network
    t.ft_pos_jitter, t.ft_samples, t.params = False, samples, SimpleNamespace(search_factor=5.0)
    t._get_hs = lambda image, box: (image.shape[1], tuple(box))  # "hs" = (image width, box)
    calls = []
    t._run_finetune = lambda hs, labels, epochs, session, coverage, frame: calls.append((hs, labels, epochs, frame))
    return t, calls


def test_all_shots_in_one_batch():
    a, b = np.zeros((300, 400, 3)), np.zeros((300, 500, 3))
    t, calls = _tracker("pos")
    t._finetune_on_frames([(a, BOX), (b, [50.0, 60.0, 20.0, 20.0])], 15, session="init", frame=0)
    assert calls == [([(400, tuple(BOX)), (500, (50.0, 60.0, 20.0, 20.0))], [1.0, 1.0], 15, 0)]
    t, calls = _tracker("posneg")  # every frame: positive, then its negative
    t._finetune_on_frames([(a, BOX), (b, BOX)], 15, session="init", frame=0)
    assert calls[0][1] == [1.0, 0.0, 1.0, 0.0] and [hs[0] for hs in calls[0][0]] == [400, 400, 500, 500]


def test_evaluation_leaves_out_the_shot_frames():
    gt = [BOX] * 11
    pred = [None] + [BOX] * 10
    pred[3] = FAR  # wrong box in frame 3
    scores = [float("nan")] + [0.9] * 10
    every, _ = evaluate_sequence(gt, pred, scores, 0.35, 0.5)
    without, records = evaluate_sequence(gt, pred, scores, 0.35, 0.5, exclude=[3])
    assert every["AP50"] < 1.0 and every["frames"] == 10
    assert without["AP50"] == 1.0 and without["frames"] == 9 and 3 not in [r[0] for r in records]
    assert len(got10k_ious(gt, pred, (640, 480))) == 10
    assert list(got10k_ious(gt, pred, (640, 480), exclude=[3])) == [1.0] * 9


if __name__ == "__main__":  # without pytest: python -m tests.test_ft_shots
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
