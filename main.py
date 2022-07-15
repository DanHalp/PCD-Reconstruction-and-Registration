import argparse
from pathlib import Path
import os
import shutil
import open3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import time
from Configs.pr_config import CFG

from SpareNet.Inference_model import SpareNet



def create_model():


    m = SpareNet(args=CFG)
    m.test()
    
    # Perfom DCP on predicted:
    # path = Path(args.TRNSFORM.output) / args.model / "point_clouds"
    # samples_list = os.listdir(path)
    # bound = args.sample_limit if args.sample_limit != -1 else len(samples_list)
    # for i in range(bound):
    #     path = path / samples_list[i]
    #     pcds = os.listdir(path)

    
    
    
    
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