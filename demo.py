import torch
import argparse
import streamlit as st
from utils import transforms
from model import UNet
import _timm
import config as cfg
from PIL import Image, ImageFont
st.set_page_config(layout='wide')

def load_model(num_classes, checkpoint_path):
    m = UNet(in_channels=3, out_channels=64, num_classes=cfg.NUM_CLASSES, up_method='tr_conv')

