from share import *
import glob
import os
import csv
import subprocess
from datetime import datetime
import torch
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict
from visa_dataloader import VisaDataset
from pytorch_lightning.callbacks import ModelCheckpoint

# protocol change: PL's default checkpoint writer (_atomic_save) first
# serializes the whole 8GB+ checkpoint into an in-memory io.BytesIO buffer,
# then writes that buffer to disk -- doubling peak RAM use at save time.
# This machine has 23.7GB RAM total; that doubling crashed a save with a
# torch.save zip-container corruption error ("unexpected pos ... vs ...").
# Write straight to disk instead, avoiding the extra in-memory copy.
import lightning_fabric.plugins.io.torch_io as _torch_io

def _direct_save(checkpoint, filepath):
    torch.save(checkpoint, filepath)

_torch_io._atomic_save = _direct_save

# protocol change: user travels while this trains unattended, so each
# completed epoch is a real, honestly-timestamped progress marker -- log
# it to a csv and commit+push just that file. Checkpoints stay out of git
# (too large); this gives a reviewable, evidence-linked record of when
# training actually progressed, at whatever real cadence epochs land.
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EPOCH_LOG_PATH = os.path.join(REPO_DIR, 'method3_diad', 'source', 'epoch_log.csv')

class GitCommitOnEpochEnd(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        epoch_loss = trainer.callback_metrics.get('train/loss_epoch')
        epoch_loss = float(epoch_loss) if epoch_loss is not None else ''
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')

        file_exists = os.path.exists(EPOCH_LOG_PATH)
        with open(EPOCH_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'global_step', 'train_loss_epoch', 'timestamp'])
            writer.writerow([epoch, global_step, epoch_loss, timestamp])

        try:
            subprocess.run(['git', 'add', EPOCH_LOG_PATH], cwd=REPO_DIR, check=True, timeout=30)
            msg = f'DiAD auto-log: epoch {epoch} done (global_step={global_step}, loss_epoch={epoch_loss})'
            commit = subprocess.run(['git', 'commit', '-m', msg], cwd=REPO_DIR, timeout=30)
            if commit.returncode == 0:
                subprocess.run(['git', 'push'], cwd=REPO_DIR, timeout=180)
            else:
                print(f'[epoch-log auto-commit] nothing to commit or commit failed (rc={commit.returncode})')
        except Exception as e:
            # Never let a git/network hiccup kill an unattended training run.
            print(f'[epoch-log auto-commit] failed, will retry next epoch: {e}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministirc = True
    torch.backends.cudnn.benchmark = False

# Configs
resume_path = 'C:/ai_local/diad_models/diad.ckpt'

setup_seed(1)
batch_size = 2
logger_freq = 3000000000000
learning_rate = 1e-5
only_mid_control = True
# protocol change: iteration speed kept degrading through a training run
# (3.7s/it -> ~16-19s/it) with GPU at 100% util but only ~30W draw -- GPU
# starved waiting on data, not compute-bound. data_path was still under
# OneDrive (only checkpoints had been moved out earlier); moved the 5GB
# dataset to a local-only directory to remove that I/O contention.
data_path = 'C:/ai_local/diad_dataset/'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
train_dataset, test_dataset = MVTecDataset('train',data_path), MVTecDataset('test',data_path)
# train_dataset, test_dataset = VisaDataset('train',data_path), VisaDataset('test',data_path)
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=True)

# protocol change: checkpoints (8GB+ each) must not live under the
# OneDrive-synced repo folder -- OneDrive trying to upload them in the
# background stalled training for hours (one step took 5h43m). Save to
# a local-only directory instead, same fix as resume_path above.
ckpt_dir = 'C:/ai_local/diad_val_ckpt/'
ckpt_callback_val_loss = ModelCheckpoint(monitor='val_acc', dirpath=ckpt_dir, mode='max')
# protocol change: one epoch takes ~10h on this 8GB laptop GPU, so the
# 25-epoch validation/checkpoint cadence above would never save anything
# in a single session. Add a step-based checkpoint so partial progress
# survives an interruption (thermal shutdown, session end, etc.).
# save_top_k bounded (was -1, unbounded) so 8GB checkpoints don't pile up forever.
# monitor=None only allows 0/1/-1, so keep just the most recent one.
ckpt_callback_periodic = ModelCheckpoint(dirpath=ckpt_dir, filename='step_{step}', every_n_train_steps=50, save_top_k=1)
logger = ImageLogger(batch_frequency=logger_freq)
# protocol change: the built-in validation_step runs the same full
# DDIM-sampling eval as test.py (which we already run separately for
# H3/H4 judgment) over all 1725 images, but embedded inside the training
# process where GPU memory is already occupied by optimizer state --
# ~48s/image here vs ~2-4s/image in a dedicated test.py run. We don't
# use its 'val_acc' output for anything, so disable it outright instead
# of eating this cost every 25 epochs.
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger,ckpt_callback_val_loss,ckpt_callback_periodic,GitCommitOnEpochEnd()], accumulate_grad_batches=4, check_val_every_n_epoch=25, limit_val_batches=0)

# Resume from the latest local step checkpoint if one exists, so a
# restart continues training instead of starting over from diad.ckpt.
existing_ckpts = sorted(
    glob.glob(os.path.join(ckpt_dir, 'step_step=*.ckpt')),
    key=os.path.getmtime,
)
resume_ckpt_path = existing_ckpts[-1] if existing_ckpts else None
if resume_ckpt_path:
    print(f'Resuming full trainer state from {resume_ckpt_path}')
    # PL's ckpt_path restore calls model.load_state_dict(..., strict=True)
    # internally with no override hook. The saved checkpoint includes the
    # eval-time ResNet50 feature extractor (attached lazily on first
    # validation), which this freshly-constructed model doesn't have yet
    # -- harmless key mismatch, so force strict=False for this one call.
    _orig_load_state_dict = model.load_state_dict
    model.load_state_dict = lambda state_dict, strict=True: _orig_load_state_dict(state_dict, strict=False)

# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=resume_ckpt_path)
