import argparse
from pathlib import Path
import os
import shutil
import open3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import time

from SpareNet.Inference_model import SpareNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="Initialize network from the weights file", default="/home/halperin/ML3D/SpareNet/SpareNet.pth")
    parser.add_argument("--test_mode", default="ML3D", help="default, vis, render, kitti", type=str)
    parser.add_argument("--sample_limit", default=10, help="How many inputs should be transalted?", type=int)
    return parser.parse_args()

def create_model(args):
    m = SpareNet(args=vars(args))
    m.test()
    
    
if __name__ == "__main__":
    
    # Set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_arguments()
    create_model(args)
