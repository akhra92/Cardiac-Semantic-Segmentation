import os
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
import torch``'
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import config as cfg
from dataloader import get_loaders
from train import Trainer
from test import Inference
from utils import transforms, Plot
from model import UNet

trn_loader, val_loader, test_loader, num_classes = get_loaders(root=cfg.ROOT, transforms=transforms,
							       batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)


if cfg.MODEL_TYPE == 'Pretrained Model':
    model = smp.Segformer(encoder_name='resnet34', encoder_weights='imagenet', classes=cfg.NUM_CLASSES)
elif cfg.MODEL_TYPE == 'Custom Model':
    model = UNet(in_channels=3, out_channels=64, num_classes=cfg.NUM_CLASSES, up_method='tr_conv')
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = cfg.LEARNING_RATE)

trainer = Trainer(
    model=model,
    tr_dl=trn_loader,
    val_dl=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=cfg.DEVICE,
    n_cls=cfg.NUM_CLASSES
)

if __name__ == '__main__':
    history = trainer.run(epochs=cfg.NUM_EPOCHS, save_prefix="cardiac")
    Plot(history)
    inference_runner = Inference(model_path=cfg.MODEL_PATH, device=cfg.DEVICE)
    inference_runner.run(test_loader, n_ims=15)