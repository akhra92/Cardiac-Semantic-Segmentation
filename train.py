import torch
from tqdm import tqdm
from utils import Metrics
import os
import numpy as np

class Trainer:
    def __init__(self, model, tr_dl, val_dl, loss_fn, optimizer, device, n_cls, save_path="saved_models", early_stop_threshold=5, threshold=0.005):

        self.model = model
        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.n_cls = n_cls
        self.save_path = save_path
        self.early_stop_threshold = early_stop_threshold
        self.threshold = threshold

        os.makedirs(self.save_path, exist_ok=True)

    def run(self, epochs, save_prefix):

        self.model.to(self.device)

        tr_loss, tr_pa, tr_iou = [], [], []
        val_loss, val_pa, val_iou = [], [], []
        best_loss, not_improve = np.inf, 0

        print("Starting training process...")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Training Phase
            self.model.train() # self.model.eval() -> validation va test (inference, deployment)
            train_metrics = self._process_epoch(self.tr_dl, is_training=True)

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._process_epoch(self.val_dl, is_training=False)

            # Log Metrics
            tr_loss.append(train_metrics["loss"])
            tr_iou.append(train_metrics["iou"])
            tr_pa.append(train_metrics["pa"])
            val_loss.append(val_metrics["loss"])
            val_iou.append(val_metrics["iou"])
            val_pa.append(val_metrics["pa"])

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Train Loss: {train_metrics['loss']:.3f} | Train PA: {train_metrics['pa']:.3f} | Train IoU: {train_metrics['iou']:.3f} |")
            print(f"Val Loss:   {val_metrics['loss']:.3f} | Val PA:   {val_metrics['pa']:.3f} | Val IoU:   {val_metrics['iou']:.3f} |")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            # Save the Best Model
            if best_loss > (val_metrics["loss"] + self.threshold):
                print(f"Validation loss decreased from {best_loss:.3f} to {val_metrics['loss']:.3f}. Saving model...")
                best_loss = val_metrics["loss"]
                not_improve = 0
                torch.save(self.model.state_dict(), f"{self.save_path}/{save_prefix}_best_model.pt")
            else:
                not_improve += 1
                print(f"No improvement for {not_improve} epoch(s).")
                if not_improve >= self.early_stop_threshold:
                    print(f"Early stopping: No improvement for {self.early_stop_threshold} epochs.")
                    break

        return {
            "tr_loss": tr_loss, "tr_iou": tr_iou, "tr_pa": tr_pa,
            "val_loss": val_loss, "val_iou": val_iou, "val_pa": val_pa,
        }

    def _process_epoch(self, dataloader, is_training):

        phase = "Train" if is_training else "Validation"
        print(f"{phase} phase started...")

        total_loss, total_iou, total_pa = 0, 0, 0
        for ims, gts in tqdm(dataloader, desc=f"{phase} Progress"):
            ims, gts = ims.to(self.device), gts.to(self.device)

            if is_training:
                preds = self.model(ims) # classification (logits); segmentation (predicted segmentation mask)
                metrics = Metrics(preds, gts, self.loss_fn, n_cls=self.n_cls)
                loss = metrics.loss() # computes loss


                self.optimizer.zero_grad() # zero grad
                loss.backward() # backprop
                self.optimizer.step() # optimization
            else: # validation
                with torch.no_grad():
                    preds = self.model(ims)
                    metrics = Metrics(preds, gts, self.loss_fn, n_cls=self.n_cls)
                    loss = metrics.loss()

            # Accumulate Metrics
            total_loss += loss.item()
            total_iou += metrics.mIoU()
            total_pa += metrics.PA()

        num_batches = len(dataloader) # to compute mean
        return {
            "loss": total_loss / num_batches,
            "iou": total_iou / num_batches,
            "pa": total_pa / num_batches,
        }