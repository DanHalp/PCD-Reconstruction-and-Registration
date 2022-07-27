import argparse
from pathlib import Path
import os
import shutil
import open3d
from scipy.spatial.transform import Rotation as Rot
from Configs.pr_config import * 
import torch
from DCP.inference_model import DCP_MODEL
from easydict import EasyDict as edict
from SpareNet.configs.base_config import cfg as sparenet_cfg
from DCP.main import main_func as dcp_main


def create_model():

    ################################# General Update #################################
    if "PROJECT" not in CFG:
        CFG.PROJECT = edict()
    CFG.update(sparenet_cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    CFG.PROJECT.device = device
    CFG.TRAIN.batch_size = CFG.TRANSFORM.batch_size
    
    ################################# Training #################################
    if CFG.PROJECT.task == TASK.TRAIN:
        model = DCP_MODEL(args=CFG, partiton=TASK.TRAIN)
        model.train()
    
    ################################# Testing + Metrics saving #################################
    elif CFG.PROJECT.task == TASK.TEST:
        model = DCP_MODEL(args=CFG)
        torch.cuda.empty_cache()
        with torch.no_grad():
            # print(model.test_and_print(CFG.TRANSFORM.pc_type));exit()
            model.compare_all()

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