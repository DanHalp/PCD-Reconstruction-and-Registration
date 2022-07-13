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
