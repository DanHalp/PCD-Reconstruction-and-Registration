import argparse
from pathlib import Path
import os
import shutil
import open3d
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import time
from Configs.pr_config import CFG
import torch
from SpareNet.Inference_model import SpareNet
from DCP.inference_model import DCP_MODEL
from easydict import EasyDict as edict
from SpareNet.configs.base_config import cfg as sparenet_cfg
from DCP.main import main_func as dcp_main


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()
        
    def write_file(self, text):
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def make_dir(path):

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def create_model():

    
    if "PROJECT" not in CFG:
        CFG.PROJECT = edict()
    CFG.update(sparenet_cfg)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    CFG.PROJECT.device = device
    CFG.TRAIN.batch_size = CFG.TRANSFORM.batch_size
    
    ################################# Training #################################
    # model = DCP_MODEL(args=CFG, partiton="train")

    # model.train()
    # dcp_main(CFG)
    
    ################################# Testing + Metrics saving #################################
    
    cloud_type = "Reconstruction" if CFG.RECONSTRUCTION.active else CFG.TRANSFORM.pc_type
    pc_path = Path(CFG.TRANSFORM.output) / CFG.TRANSFORM.dataset_name / cloud_type
    make_dir(pc_path)
    metrics_path = Path(CFG.TRANSFORM.output) / "Metrics" / cloud_type
    make_dir(metrics_path)
    CFG.TRANSFORM.output = pc_path
    CFG.TRANSFORM.metrics_path = metrics_path
    
    time = datetime.now().strftime('run_%H_%M_%d_%m_%Y')
    metrics_file = IOStream(metrics_path / f"metrics_{CFG.TRANSFORM.dataset_name}_{cloud_type}_{time}.csv")
    metrics_file.write_file("Iteration,ID,Loss,Cycle_Loss,MSE,RMSE,MAE,rot_MSE,rot_RMSE,rot_MAE,trans_MSE,trans_RMSE,trans_MAE")
    xxx_file = IOStream(metrics_path / f"transformation_{CFG.TRANSFORM.dataset_name}_{cloud_type}_{time}.csv")
    xxx_file.write_file("Sample_ID;type;rotation_ab;translation_ab;rotation_ba;translation_ba")


    model = DCP_MODEL(args=CFG)
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.test_print(metrics_file, xxx_file)

    metrics_file.close()
    xxx_file.close()
   

    
    
    
if __name__ == "__main__":
    
    # Set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    create_model()
    
    # TODO:
    # - Perform completion on random subset of points from gt, and not the partial scan.
    # - Test DCP on Shapenet
    # - Change parser to a cfg file, to allow better integration of the two repos.
    # - Perform reconstruction and DCP.