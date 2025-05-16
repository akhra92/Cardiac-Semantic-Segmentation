import torch
import onnx
from model import UNet
import config as cfg
import numpy as np
from PIL import Image
from torchvision import transforms as T
import onnxruntime as ort

def run_onnx():

    preprocess = T.Compose([T.Resize((cfg.IMG_H, cfg.IMG_W)),
                            T.Grayscale(num_output_channels=3),
                            T.ToTensor(),
                            T.Normalize(mean=cfg.MEAN, std=cfg.STD)])

    model = UNet(in_channels=3, out_channels=64, num_classes=2, up_method='tr_conv')
    model.load_state_dict(torch.load('./saved_models/best_model.pth'))
    model.eval

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        'model.onnx',
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    onnx_model = onnx.load('model.onnx')
    onnx.checker.check_model(onnx_model)
    print('ONNX model is ready!')    
    

if __name__ == '__main__':
    run_onnx()