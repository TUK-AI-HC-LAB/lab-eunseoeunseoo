from share import *
import glob
import os
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
data_path = 'C:/Users/kelly/OneDrive/바탕 화면/For Labs/No_Submit/Dataset/'

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
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger,ckpt_callback_val_loss,ckpt_callback_periodic], accumulate_grad_batches=4, check_val_every_n_epoch=25)

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