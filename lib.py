# ─────────────────────────────────────────────
# Standard Libraries
# ─────────────────────────────────────────────
import os
import sys
import math
import time
import random
import copy
import gzip
import shutil
import inspect
import itertools
import gc
from timeit import timeit
from collections import defaultdict

# ─────────────────────────────────────────────
# Third-party Libraries
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import psutil

# ─────────────────────────────────────────────
# OpenCV (with Colab display patch)
# ─────────────────────────────────────────────
import cv2
from google.colab.patches import cv2_imshow

# ─────────────────────────────────────────────
# PyTorch Core
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.quantization as quant
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# TorchVision
# ─────────────────────────────────────────────
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)

# ─────────────────────────────────────────────
# PyTorch Image Models (TIMM)
# ─────────────────────────────────────────────
import timm

# ─────────────────────────────────────────────
# Model Analysis Tools
# ─────────────────────────────────────────────
from ptflops import get_model_complexity_info
