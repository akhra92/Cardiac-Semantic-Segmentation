import torch
import segmentation_models_pytorch as smp
import config as cfg
from dataloader import get_loaders
from train import Trainer
from test import Inference
from utils import transforms, Plot
from model import UNet


def build_model():
    if cfg.MODEL_TYPE == 'Pretrained Model':
        return smp.Segformer(encoder_name='resnet34', encoder_weights='imagenet', classes=cfg.NUM_CLASSES)
    if cfg.MODEL_TYPE == 'Custom Model':
        return UNet(in_channels=3, out_channels=64, num_classes=cfg.NUM_CLASSES, up_method='tr_conv')
    raise ValueError(f"Unknown MODEL_TYPE {cfg.MODEL_TYPE!r}. Expected 'Pretrained Model' or 'Custom Model'.")


if __name__ == '__main__':
    trn_loader, val_loader, test_loader, num_classes = get_loaders(
        root=cfg.ROOT, transforms=transforms,
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
    )

    model = build_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.LEARNING_RATE)

    trainer = Trainer(
        model=model,
        tr_dl=trn_loader,
        val_dl=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=cfg.DEVICE,
        n_cls=cfg.NUM_CLASSES,
    )

    history = trainer.run(epochs=cfg.NUM_EPOCHS, save_prefix="cardiac")
    Plot(history)
    inference_runner = Inference(model_path=cfg.MODEL_PATH, device=cfg.DEVICE)
    inference_runner.run(test_loader, n_samples=5)
