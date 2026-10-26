# Training on a rented GPU machine

**English** | [Türkçe](server.tr.md)

Scripts in [`scripts/`](../scripts); run them in the repository folder. Every script can be run again after an
interruption: it continues where it stopped.

## 1. Preparation

```bash
git clone https://github.com/balk21/stark-cls-finetune.git && cd stark-cls-finetune
scripts/setup.sh /workspace/data    # tools, vot1 environment (~10 min), paths, check
```

`/workspace/data`: a folder on the large disk (`df -h`). Without it, the data goes into the repository folder.

Only the datasets you train on; they can run at the same time (separate terminals or tmux windows):

| Dataset | Command | Download → on disk |
|---|---|---|
| COCO | `scripts/get_coco.sh` | 19 GB → 20 GB |
| GOT-10k | `scripts/get_got10k.sh <Google Drive link \| path of full_data.zip>` | 70.7 GB → 74 GB |
| LaSOT | `scripts/get_lasot.sh` | 248 GB → 248 GB |

- Archives are deleted after extraction; `KEEP_ARCHIVES=1 scripts/get_coco.sh` keeps them.
- GOT-10k link: a copy of `full_data.zip` in your own Google Drive, shared as "anyone with the link"
  ([colab.md](colab.md#got-10k-from-google-drive)).
- Folders: `<data>/train/{coco, got10k/train, lasot}`, training runs in `<data>/training/`.

## 2. Training

```bash
scripts/train.sh --set 'datasets=[got10k, coco, lasot]'                # stage 1, in the tmux session "train"
scripts/train.sh --set 'datasets=[got10k, coco, lasot]' --set stage=2  # stage 2, after stage 1
tmux attach -t train                                                   # watch it (leave it running: Ctrl-b d)
scripts/stark.sh train-report st101_lasot+got10k+coco_stage1           # progress
```

- The same command again resumes the run from its last finished epoch.
- `train.sh` takes the arguments of `python -m stark_ft train` ([train.md](train.md#parameters)), e.g.
  `--set num_workers=12`, `--set model_config=baseline`.

Backups (the disk of a rented machine is lost with it):

```bash
rclone config                                                              # once, e.g. Google Drive as "gdrive"
scripts/backup.sh st101_lasot+got10k+coco_stage1 gdrive:LOKAP/training     # every 30 min, tmux session backup-<run>
scripts/restore.sh st101_lasot+got10k+coco_stage1 gdrive:LOKAP/training    # on a new machine; then the same train.sh
```

Resuming on another machine needs the same parameters and the same training code
([train.md](train.md#run-folder-and-resuming)).
