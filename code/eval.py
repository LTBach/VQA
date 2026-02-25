import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
import warnings
from pprint import pprint

# self-defined
import model.fusion_net as fusion_net
import model.answer_net as answer_net
from model import Vector, SimpleClassifier
from config import cfg
from torchlight import initialize_exp, set_seed, snapshot, get_dump_path, show_params
from utils import unseen_mask, freeze_layer, cosine_sim, Metrics, instance_bce_with_logits
from data import fvqa
import copy
import main
from main import Runner
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Config loading...
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    set_seed(cfg.random_seed)

    # Environment initialization...
    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)
    if not cfg.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    torch.cuda.set_device(cfg.gpu_id)

    # Run...
    runner = Runner(cfg)
    runner.load_model()
    runner.eval()
