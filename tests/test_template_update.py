"""Template update rules of STARK-ST ('stark' and 'max'), without a network or GPU."""
from stark_ft.test.config import ExperimentConfig

FAR = [500.0, 500.0, 20.0, 20.0]  # no overlap with BOX
BOX = [100.0, 100.0, 20.0, 20.0]


def _tracker(mode, interval, conf_thr=0.5, iou_thr=0.5, max_updates=-1):
    from lib.test.tracker.stark_st import STARK_ST
    t = STARK_ST.__new__(STARK_ST)  # without __init__: no network
    t.update_mode, t.update_intervals = mode, [interval]
    t.update_conf_thr, t.update_iou_thr, t.max_template_updates = conf_thr, iou_thr, max_updates
    t.z_dict_list = ["frame0", "frame0"]
    t.template_update_count, t.update_frames, t.interval_best, t.template_source = 0, [], {}, None
    t._template = lambda image, box: f"frame{image}"  # the "image" is the frame number here
    t.state, t.frame_id = list(BOX), 0
    return t


def _run(t, frames):
    """frames: [(conf, box), ...] for frames 1, 2, ... -> {frame: (template, source frame)} of the updates."""
    updates = {}
    for conf, box in frames:
        t.frame_id += 1
        prev, t.state = t.state, list(box)
        done = t._update_max(t.frame_id, conf, prev) if t.update_mode == "max" else t._update_stark(t.frame_id, conf)
        if done:
            updates[t.frame_id] = (t.z_dict_list[1], t.template_source[0])
    return updates


def test_stark_rule():
    frames = [(0.9, BOX)] * 9 + [(0.6, BOX)] + [(0.9, BOX)] * 9 + [(0.4, BOX)]
    assert _run(_tracker("stark", 10), frames) == {10: ("frame10", 10)}


def test_max_takes_the_best_frame_of_the_interval():
    confs = [0.99] * 10 + [0.6, 0.7, 0.95, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] + [0.3] * 10 + [0.6] * 9 + [0.7]
    updates = _run(_tracker("max", 10), [(c, BOX) for c in confs])
    # frames 1-10 are skipped; 11-20 -> frame 13 at frame 20; 21-30 no candidate (no update); 31-40 -> frame 40
    assert updates == {20: ("frame13", 13), 40: ("frame40", 40)}


def test_max_iou_check():
    # frame 12 has the highest score but jumps away from the box of the previous candidate (frame 11); frame 13 is
    # compared with frame 11 (the last candidate), not with frame 12
    frames = [(0.9, BOX)] * 10 + [(0.6, BOX), (0.99, FAR), (0.7, BOX), (0.9, FAR)] + [(0.1, BOX)] * 6
    assert _run(_tracker("max", 10), frames) == {20: ("frame13", 13)}
    assert _run(_tracker("max", 10, iou_thr=0.0), frames) == {20: ("frame12", 12)}  # 0 = no IoU check


def test_max_limit_and_threshold():
    frames = [(0.9, BOX)] * 40
    assert sorted(_run(_tracker("max", 10, max_updates=1), frames)) == [20]
    assert _run(_tracker("max", 10, conf_thr=0.9), frames) == {}  # the score must be greater than the threshold


def test_config():
    cfg = ExperimentConfig(ft_mode="none", update_mode="max", update_conf_thr=0.8, sequences=["bull"])
    assert cfg.validate().experiment_name == "st101_base_max100_uthr0.8_bull"
    assert ExperimentConfig(ft_mode="none", update_mode="max", update_iou_thr=0).experiment_name == \
        "st101_base_max100_uiou0"
    assert ExperimentConfig(ft_mode="none", update_mode="max", update_interval=99999).experiment_name == \
        "st101_base_noupd"
    for bad in (dict(update_mode="best"), dict(update_iou_thr=1.5),
                dict(model="stark_s", model_config="baseline", ft_mode="none", update_mode="max")):
        try:
            ExperimentConfig(**bad).validate()
            raise AssertionError(f"accepted: {bad}")
        except ValueError:
            pass


if __name__ == "__main__":  # without pytest: python -m tests.test_template_update
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
