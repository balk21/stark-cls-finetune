# Google Colab

**English** | [Türkçe](colab.tr.md)

| Notebook | |
|---|---|
| Train | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/train.ipynb) |
| Test | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/test.ipynb) |
| Compare | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/compare.ipynb) |

Choose a GPU runtime (*Runtime → Change runtime type*; A100 or L4 for training) and run the cells in order. The first
cell clones the repository and calls `nb.init(colab_drive=...)`, which mounts Google Drive and restores the same `vot1`
environment as on other machines (created with micromamba the first time, ~5 min, then cached on Drive).

Everything that should survive the session is kept in the Drive folder given by `colab_drive`
(default `MyDrive/LOKAP`):

| Drive folder | Content |
|---|---|
| `cache/` | the environment (7.8 GB) and every VOT sequence a test has used (`votlt2019_sequences/<sequence>.tar`; all 50: 17.6 GB), restored to the local disk when needed |
| `checkpoints/` | downloaded / your own checkpoints (`checkpoints/stark_st2/<model_config>/`), copied to the local disk |
| `train_archives/{coco,got10k}/` | training dataset archives, extracted to the local disk in every session |
| `outputs/` | test experiments (interrupted ones resume) |
| `training/` | training runs (interrupted ones resume) |

Notes:

- Open the notebooks from the badges above. A notebook opened earlier keeps its old cells; the first cell updates the
  code (`git pull`), not the notebook.
- If a session was open while the code was updated, use *Runtime → Disconnect and delete runtime* and start again.
- Results depend on the GPU type (TF32 on A100 / L4, not on T4); see [test.md](test.md#results-across-gpus).
- Training runs for days; every finished epoch is saved on Drive, so a new session resumes the run. Colab compute
  units are consumed per hour.

## GOT-10k from Google Drive

The GOT-10k download links (sent by e-mail after registration) may point to Google Drive, e.g. `full_data.zip`
(70.7 GB). Such shared files often reach Google's daily download limit ("Quota exceeded"); then neither a download nor
a shortcut works. Use your own copy instead (made inside Drive, nothing is downloaded; needs 70.7 GB of Drive space):

1. Open the link in the browser (with the Google account used in Colab) → *Add shortcut to Drive*.
2. Right-click the shortcut → *Make a copy*.
3. Move the copy into `<colab_drive>/train_archives/got10k/` (the data step of `train.ipynb` creates this folder),
   delete the shortcut, and run `train.ipynb` with `got10k_sources=[]`.

The archive is then read from Drive and only the train videos are extracted to the local disk (≈ 74 GB, about half an
hour) at the start of every session, because reading the images from Drive during training would be far slower.
The preparation checks the free disk space first.
