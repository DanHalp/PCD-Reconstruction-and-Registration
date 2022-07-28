import os
from pathlib import Path
import shutil

import torch
from easydict import EasyDict as edict
from enum import Enum, IntEnum
import numpy as np


CFG = edict()

###################### MagicNumbers ######################
class DATASETS(Enum):
    SHAPENET = 0
    MODELNET = 1

class PC_TYPES(IntEnum):
    GT_CLOUD = 0
    PARTIAL_CLOUD = 1
    RECONSTRUCTED_CLOUD = 2


PC_TYPES_LIST = ["gtcloud", "partial_cloud", "reconstructed"]

class TASK(Enum):
    TRAIN = 0
    TEST = 1

###################### Helper classes ######################

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op
        self.schedulers = [torch.optim.lr_scheduler.StepLR(opti, step_size=CFG.TRANSFORM.optim_scheduler_step_size, 
                                                             gamma=CFG.TRANSFORM.optim_scheduler_gamma) for opti in op]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    
    def schedule_step(self):
        for s in self.schedulers:
            s.step()

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
        
###################### Helper functions ######################
def make_dir(path, delete=False):

    if delete and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def write_pcd_to_obj(file_path, file_content, c=None):
    if c is None:
        c = (0,0,0)
    with open(file_path, "w") as output:
        for v in file_content:
            x, y, z = v
            output.write(f"v {x} {y} {z} {c[0]} {c[1]} {c[2]}\n")
    output.close()
    
###################### SpareNet ######################
CFG.RECONSTRUCTION = edict()
CFG.RECONSTRUCTION.model_path = "/home/danha/PCD-Reconstruction-and-Registration/data/Checkpoints/SpareNet/SpareNet.pth"
CFG.RECONSTRUCTION.model = ["sparenet"][0]
CFG.RECONSTRUCTION.test_mode = ["ML3D", "default"][0]
CFG.RECONSTRUCTION.gpu = ["0"][0]
CFG.RECONSTRUCTION.test_mode = ["ML3D", "default"][0]
CFG.RECONSTRUCTION.sample_limit = [1][0]
CFG.RECONSTRUCTION.random_samples = [False, True][0]
CFG.RECONSTRUCTION.save = [False, True][0]
CFG.RECONSTRUCTION.output = "/home/danha/PCD-Reconstruction-and-Registration/Outputs/"
CFG.RECONSTRUCTION.local_dir = "/home/danha/PCD-Reconstruction-and-Registration/SpareNet"
CFG.RECONSTRUCTION.active = [True, False][1]



######################### DCP #########################
CFG.TRANSFORM = edict()
CFG.TRANSFORM.output = CFG.RECONSTRUCTION.output
CFG.TRANSFORM.current_out_dir = CFG.RECONSTRUCTION.output
CFG.TRANSFORM.test_folder_name = ["Comparison"][0]
CFG.TRANSFORM.model_path = Path("/home/danha/PCD-Reconstruction-and-Registration/Checkpoints/ShapeNet/run_00_48_28_07_2022/model.best_epoch_1_19200_valloss_0.04304520973021075.t7")
# CFG.TRANSFORM.model_path = Path("/home/danha/PCD-Reconstruction-and-Registration/DCP/pretrained/dcp_v2.t7")
CFG.TRANSFORM.exp_name = ["dcp_v1", "dcp_v2"][1]
CFG.TRANSFORM.model = ['dcp'][0]
CFG.TRANSFORM.emb_nn = ['pointnet', 'dgcnn'][1]
CFG.TRANSFORM.pointer = ['identity', 'transformer'][1]
CFG.TRANSFORM.head = ['mlp', 'svd'][1]
CFG.TRANSFORM.emb_dims = 512
CFG.TRANSFORM.n_blocks = 1
CFG.TRANSFORM.n_heads = 4
CFG.TRANSFORM.ff_dims = 1024
CFG.TRANSFORM.dropout = 0.1
CFG.TRANSFORM.test_batch_size = 20
CFG.TRANSFORM.iterations = 1
CFG.TRANSFORM.use_sgd = [False, True][0]
CFG.TRANSFORM.momentum = 0.9
CFG.TRANSFORM.no_cuda = [False, True][0]
CFG.TRANSFORM.seed = 1234
CFG.TRANSFORM.eval = [True, False][0]
CFG.TRANSFORM.cycle = [True, False][1]
CFG.TRANSFORM.gaussian_noise = [True, False][0]
CFG.TRANSFORM.unseen = [True, False][1]
CFG.TRANSFORM.num_points = [512, 1024, 2048, 3072, 4096, 16384][1]
CFG.TRANSFORM.factor = 4
CFG.TRANSFORM.pc_type = PC_TYPES_LIST[PC_TYPES.PARTIAL_CLOUD]   # Relevant only to ShapeNet
CFG.TRANSFORM.dataset = [DATASETS.SHAPENET, DATASETS.MODELNET][0]
CFG.TRANSFORM.dataset_name = "ShapeNet" if  CFG.TRANSFORM.dataset == DATASETS.SHAPENET else "ModelNet"
CFG.TRANSFORM.fixed_dataset_indices = [np.array([1, 944,  869,  465,  997,  906,  735, 1068,  725, 1138,  573,  374,  482,  912, 737, 1175,  210,  351,  705, 11]), None][1]

CFG.TRANSFORM.dataset_fraction = [1, 1e-1, 1e-2, 1e-3][0]
CFG.TRANSFORM.batch_size = 12
CFG.TRANSFORM.lr = 5e-3
CFG.TRANSFORM.epochs = 10
CFG.TRANSFORM.val_every = 75
CFG.TRANSFORM.optim_scheduler_step_size = 100
CFG.TRANSFORM.optim_scheduler_gamma = 0.5
CFG.TRANSFORM.early_stop_thresh = 7
CFG.TRANSFORM.full_shape_p = 0.4
CFG.TRANSFORM.reg = 1e-3

#### ICP ###
CFG.TRANSFORM.icp_iter = 100
CFG.TRANSFORM.use_icp = [False, True][1]



CFG.PROJECT = edict()
CFG.PROJECT.task = [TASK.TRAIN, TASK.TEST][1]