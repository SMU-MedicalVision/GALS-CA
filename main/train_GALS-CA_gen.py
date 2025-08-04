import os
import time
import glob
import math
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from Nii_utils import *
from dataset.Dataset_ide import Dataset_harmonize_t1_t2f_t1c
from models.Networks_ide.model import EfficientNet
from models.Networks_ide.Validation_inference import Model_Validation, Model_Inference



def main(opt):
    train_writer = SummaryWriter(join(opt.save_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(join(opt.save_dir, 'log/val'), flush_secs=2)
    print(opt.save_dir)
    net = []
    return net


def pred(opt, net=None):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epoch', type=int, default=100, help='all_epochs')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--lr_max', type=float, default=5e-4, help='max learning rate')
    parser.add_argument('--bs', type=int, default=2, help='training input batch size')
    parser.add_argument('--num_threads', type=int, default=1, help='# threads for loading dataset')

    # -------------------- Inference settings
    parser.add_argument('--val_bs', type=int, default=4, help='Val/Test batch size')
    parser.add_argument('--save_dir', type=str, default='', help="./main/trained_models/GALS-CA_cla/")  # Path for saving model parameters

    # -------------------- Data settings
    parser.add_argument('--data_dim', type=str, default='2D')
    parser.add_argument('--ImageSize', type=int, default=424, help='Spatial dimension cropped to 424 * 424')
    parser.add_argument("--gen_save_dir", type=str, default='', help="./main/trained_models/GALS-CA_gen/")
    parser.add_argument('--CT_max', type=int, default=255, help='max value of preprocessed CT image')
    parser.add_argument('--CT_min', type=int, default=0, help='min value of preprocessed CT image')
    parser.add_argument('--preloading', type=bool, default=True, help='preloading the image')

    # -------------------- Model settings
    parser.add_argument('--model_name', type=str, default='EfficientNet_b0')
    parser.add_argument('--inchannel', type=int, default=3, help='input channel')
    parser.add_argument('--classes', type=int, default=1, help='')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout rate 0~1 ')

    # -------------------- Loss function
    parser.add_argument('--do_flood', type=bool, default=True, help='do flood loss')
    parser.add_argument('--flood', type=float, default=0.1, help='flood loss threshold')
    parser.add_argument('--warmup', action='store_false')
    parser.add_argument('--warm_up_epochs', type=int, default=5, help='warm_up_epochs')

    # -------------------- Quick test settings
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')
    opt = parser.parse_args()
    # torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    setup_seed(opt.seed)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GALS-CA_cla Start")
    if opt.quick_test:
        opt.max_epoch = 10
        opt.ref_timestep = 10

    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        opt.save_dir = './main/trained_models/GALS-CA_cla/bs{}_ImageSize{}_epoch{}_seed{}_{}'.format(opt.bs, opt.ImageSize, opt.max_epoch, opt.seed, current_time)
        os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)

    if not opt.inference_only:
        net = main(opt)
        pred(opt, net)
    else:
        pred(opt)

    if not opt.inference_only:
        print("GALS-CA_cla Training Done")
    else:
        print("GALS-CA_cla Inference Done")
    print("-------------------------------------------")
    print(f"Attention !! Results can be viewed here :\n {opt.save_dir}")
    print("-------------------------------------------")