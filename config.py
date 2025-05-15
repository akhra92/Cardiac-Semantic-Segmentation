import torch

ROOT = './datasets/cardiac'
MEAN, STD, IMG_H, IMG_W = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
MODEL_PATH = "saved_models/cardiac_best_model.pt"
MODEL_TYPE = 'Pretrained Model' 				# choose Custom Model or Pretrained Model, must be string
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'