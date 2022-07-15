from easydict import EasyDict as edict

CFG = edict()

###################### SpareNet ######################
CFG.RECONSTRUCTION = edict()
CFG.RECONSTRUCTION.ckpt = "/home/danha/PCD-Reconstruction-and-Registration/data/Checkpoints/SpareNet/SpareNet.pth"
CFG.RECONSTRUCTION.model = ["sparenet"][0]
CFG.RECONSTRUCTION.test_mode = ["ML3D", "default"][0]
CFG.RECONSTRUCTION.gpu = ["0"][0]
CFG.RECONSTRUCTION.test_mode = ["ML3D", "default"][0]
CFG.RECONSTRUCTION.sample_limit = [1][0]
CFG.RECONSTRUCTION.random_samples = [False, True][0]
CFG.RECONSTRUCTION.save = [False, True][0]
CFG.RECONSTRUCTION.output = "/home/danha/PCD-Reconstruction-and-Registration/Outputs/Completion"
CFG.RECONSTRUCTION.local_dir = "/home/danha/PCD-Reconstruction-and-Registration/SpareNet"


######################### DCP #########################
CFG.TRANSORM = edict()
CFG.TRANSORM.output = CFG.RECONSTRUCTION.output
CFG.TRANSORM.ckpt = "/home/danha/PCD-Reconstruction-and-Registration/DCP/pretrained/dcp_v1.t7"
CFG.TRANSORM.exp_name = ["exp"][0]
CFG.TRANSORM.model = ['dcp'][0]
CFG.TRANSORM.emb_nn = ['pointnet', 'dgcnn'][0]
CFG.TRANSORM.pointer = ['identity', 'transformer'][1]
CFG.TRANSORM.head = ['mlp', 'svd'][1]
CFG.TRANSORM.emb_dims = 512
CFG.TRANSORM.n_blocks = 1
CFG.TRANSORM.n_heads = 4
CFG.TRANSORM.ff_dims = 1024
CFG.TRANSORM.dropout = 0.0
CFG.TRANSORM.batch_size = 32
CFG.TRANSORM.test_batch_size = 10
CFG.TRANSORM.epochs = 250
CFG.TRANSORM.use_sgd = [True, False][1]
CFG.TRANSORM.lr = 1e-1
CFG.TRANSORM.momentum = 0.9
CFG.TRANSORM.no_cuda = [True, False][1]
CFG.TRANSORM.seed = 1234
CFG.TRANSORM.eval = [True, False][1]
CFG.TRANSORM.cycle = [True, False][1]
CFG.TRANSORM.gaussian_noise = [True, False][1]
CFG.TRANSORM.unseen = [True, False][1]
CFG.TRANSORM.num_points = 1024
CFG.TRANSORM.dataset = ['modelnet40', "ShapeNet"]
CFG.TRANSORM.factor = 4
CFG.TRANSORM.model_path = CFG.TRANSORM.ckpt