"""Configuration for Bird Flight Classifier"""
from pathlib import Path

import torch

# ============== UPDATE THIS PATH ==============
ROOT_DIR = "C:/Users/jorda/PycharmProjects/SuperFlier"
# ==============================================

# Folder structure
FLY_FOLDER = Path(ROOT_DIR + "/data/fly")
NOFLY_FOLDER = Path(ROOT_DIR + "/data/nofly")
TEST_FOLDER = Path(ROOT_DIR + "/data/test")
CSV_FILENAME = Path(ROOT_DIR + "/bird_labels.csv")
MODEL_SAVE_DIR = Path(ROOT_DIR + "/models")
RESULTS_DIR = Path(ROOT_DIR + "/results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5 backbone options (ResNet50-tier)
MODELS = ["resnet50", "efficientnet_b3", "densenet121"]

# Sample images for visual evaluation (5 fly, 5 nofly - set after running pre-process)
SAMPLE_FLY_IDS = [132, 164, 3, 196, 189]
SAMPLE_NOFLY_IDS = [361, 791, 790, 913, 952]

# Training (Set from first round of 5 default models)
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 384
NUM_WORKERS = 0
USE_WEIGHTED_LOSS = True
DROPOUT = 0.2