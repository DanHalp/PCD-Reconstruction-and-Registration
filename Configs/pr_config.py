from easydict import EasyDict as edict
from enum import Enum
import numpy as np


CFG = edict()

###################### MagicNumbers ######################
class DATASETS(Enum):
    SHAPENET = 0
    MODELNET = 1

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
CFG.RECONSTRUCTION.active = [True, False][0]



######################### DCP #########################
CFG.TRANSFORM = edict()
CFG.TRANSFORM.output = CFG.RECONSTRUCTION.output
CFG.TRANSFORM.test_folder_name = ["Comparison"][0]
CFG.TRANSFORM.model_path = "/home/danha/PCD-Reconstruction-and-Registration/DCP/pretrained/dcp_v2.t7"
CFG.TRANSFORM.exp_name = ["dcp_v1", "dcp_v2"][1]
CFG.TRANSFORM.model = ['dcp'][0]
CFG.TRANSFORM.emb_nn = ['pointnet', 'dgcnn'][1]
CFG.TRANSFORM.pointer = ['identity', 'transformer'][1]
CFG.TRANSFORM.head = ['mlp', 'svd'][1]
CFG.TRANSFORM.emb_dims = 512
CFG.TRANSFORM.n_blocks = 1
CFG.TRANSFORM.n_heads = 4
CFG.TRANSFORM.ff_dims = 1024
CFG.TRANSFORM.dropout = 0.0
CFG.TRANSFORM.batch_size = 16
CFG.TRANSFORM.test_batch_size = 20
CFG.TRANSFORM.iterations = 5
CFG.TRANSFORM.epochs = 1
CFG.TRANSFORM.val_every = 5
CFG.TRANSFORM.use_sgd = [True, False][1]
CFG.TRANSFORM.lr = 1e-1
CFG.TRANSFORM.momentum = 0.9
CFG.TRANSFORM.no_cuda = [True, False][1]
CFG.TRANSFORM.seed = 1234
CFG.TRANSFORM.eval = [True, False][0]
CFG.TRANSFORM.cycle = [True, False][1]
CFG.TRANSFORM.gaussian_noise = [True, False][1]
CFG.TRANSFORM.unseen = [True, False][1]
CFG.TRANSFORM.num_points = [512, 1024, 2048, 3072, 4096, 16384][1]
CFG.TRANSFORM.factor = 4
CFG.TRANSFORM.pc_type = ["gtcloud", "partial_cloud"][1]   # Relevant only to ShapeNet
CFG.TRANSFORM.dataset = [DATASETS.SHAPENET, DATASETS.MODELNET][0]
CFG.TRANSFORM.dataset_name = "ShapeNet" if  CFG.TRANSFORM.dataset == DATASETS.SHAPENET else "ModelNet"
CFG.TRANSFORM.fixed_dataset_indices = [np.array([295, 944,  869,  465,  997,  906,  735, 1068,  725, 1138,  573,  374,  482,  912, 737, 1175,  210,  351,  705, 11]), None][0]
