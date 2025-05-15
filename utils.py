import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
import config as cfg

transforms = A.Compose( [A.Resize(cfg.IMG_H, cfg.IMG_W),
			 A.augmentations.transforms.Normalize(mean = cfg.MEAN, std = cfg.STD),
			 ToTensorV2(transpose_mask = True)], is_check_shapes = False)

class Metrics():

    def __init__(self, pred, gt, loss_fn, eps = 1e-10, n_cls = 2):

        self.pred, self.gt = torch.argmax(pred, dim = 1), gt.squeeze(1) # (batch, width, height)
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred

    def to_contiguous(self, inp): return inp.contiguous().view(-1) # memory

    def PA(self):

        with torch.no_grad():
            match = torch.eq(self.pred, self.gt).int() #  acc += (self.pred == self.gt)

        return float(match.sum()) / float(match.numel()) # number of elements 3 x 224 x 224

    def mIoU(self): # mean intersection over union

        with torch.no_grad():

            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []

            for c in range(self.n_cls): # 0, 1

                match_pred = pred == c
                match_gt   = gt == c

                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan) # not a value

                else:

                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item() # tensor -> float
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    iou = intersect / (union + self.eps) # numeric stability
                    iou_per_class.append(iou)

            return np.nanmean(iou_per_class) # nanmean -> nan ni chiqarib tashlab average ni hisoblaydi

    def loss(self): return self.loss_fn(self.pred_, self.gt.long()) # int

def tic_toc(start_time = None): return time() if start_time == None else time() - start_time

class Plot:
    def __init__(self, history, save_dir='plots', model_name=cfg.MODEL_TYPE):
        self.history = history
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_all()

    def plot_metric(self, metric1, metric2, label1, label2, title, ylabel, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history[metric1], label=label1)
        plt.plot(self.history[metric2], label=label2)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_all(self):
        self.plot_metric(
            metric1="tr_iou", metric2="val_iou",
            label1="Train IoU", label2="Validation IoU",
            title=f"Mean Intersection Over Union (mIoU) Learning Curve of {self.model_name}",
            ylabel="mIoU Score",
            filename="iou_curve.png"
        )
        self.plot_metric(
            metric1="tr_pa", metric2="val_pa",
            label1="Train Pixel Accuracy", label2="Validation Pixel Accuracy",
            title=f"Pixel Accuracy (PA) Learning Curve of {self.model_name}",
            ylabel="PA Score",
            filename="pa_curve.png"
        )
        self.plot_metric(
            metric1="tr_loss", metric2="val_loss",
            label1="Train Loss", label2="Validation Loss",
            title=f"Loss Learning Curve of {self.model_name}",
            ylabel="Loss Value",
            filename="loss_curve.png")