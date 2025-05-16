import torch
import argparse
import streamlit as st
import torchvision.transforms as T
from model import UNet
import timm
import config as cfg
from PIL import Image, ImageFont
import numpy as np
st.set_page_config(layout='wide')


def get_transforms():
    return T.Compose([T.Resize((cfg.IMG_H, cfg.IMG_W)),
                      T.Grayscale(num_output_channels=3),
                      T.ToTensor(),
                      T.Normalize(mean=cfg.MEAN, std=cfg.STD)])


def load_model(num_classes, checkpoint_path):
    m = UNet(in_channels=3, out_channels=64, num_classes=num_classes, up_method='tr_conv')
    m.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return m.eval()


def tn_2_np(t):
    invTrans = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    rgb = True if len(t.shape) == 3 else False
    return (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (t * 255).detach().cpu().numpy().astype(np.uint8)


def predict(m, path, tfs):
    im = Image.open(path)
    im.save(path)

    return im, torch.max(m(tfs(im).unsqueeze(0)).data, 1)[1]


def run(args):
    tfs = get_transforms()
    
    default_path = './sample_images/2.png'

    m = load_model(cfg.NUM_CLASSES, args.checkpoint_path)
    st.title('Medical Image Segmentation')
    file = st.file_uploader('Please upload your image')
    im, pred = predict(m =m, path=file, tfs=tfs) if file else predict(m=m, path=default_path, tfs=tfs)
    st.write(f'Input Image: ')
    st.image(im)
    pred = tn_2_np(pred.squeeze(0))
    st.write(f'Output Mask: ')
    st.image(pred)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Medical Image Segmentation Demo')

    parser.add_argument('-cp', '--checkpoint_path', type=str, default='./saved_models/best_model.pth')

    args = parser.parse_args()

    run(args)


