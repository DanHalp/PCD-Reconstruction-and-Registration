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
    parser.add_argument("--ckpt", help="Initialize network from the weights file", default="/home/halperin/ML3D/SpareNet/pretrained/sparenet.pth")
    parser.add_argument("--model", help="which model to use for completion", default="sparenet")
    parser.add_argument("--test_mode", default="ML3D", help="default, vis, render, kitti", type=str)
    parser.add_argument("--sample_limit", default=1, help="How many inputs should be transalted?", type=int)
    parser.add_argument("--random_samples", default=False, help="How many inputs should be transalted?", type=bool)
    parser.add_argument("--save", default=False, help="How many inputs should be transalted?", type=bool)
    parser.add_argument("--outputs", default="/home/halperin/ML3D/Outputs/Completion", help="How many inputs should be transalted?", type=str)
    return parser.parse_args()

def create_model(args):
    m = SpareNet(args=vars(args))
    m.test()
    
    # Perfom DCP on predicted:
    path = Path(args.outputs) / args.model / "point_clouds"
    samples_list = os.listdir(path)
    bound = args.sample_limit if args.sample_limit != -1 else len(samples_list)
    for i in range(bound):
        path = path / samples_list[i]
        pcds = os.listdir(path)
    
    
    
if __name__ == "__main__":
    
    # Set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_arguments()
    create_model(args)
    
    # TODO:
    # - Perform completion on random subset of points from gt, and not the partial scan.
    # - Test DCP on Shapenet
    # - Change parser to a cfg file, to allow better integration of the two repos.
    # - Perform reconstruction and DCP.