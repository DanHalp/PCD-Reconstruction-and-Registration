import argparse
from genericpath import exists
from pathlib import Path
import os
from secrets import choice
import shutil
import open3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import time
from distutils.dir_util import copy_tree

def write_pcd_to_obj(file_path, file_content):
    with open(file_path, "w") as output:
        for v in file_content:
            x, y, z = v
            output.write(f"v {x} {y} {z}\n")
    output.close()

def write_pcd(file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc, write_ascii=True)

def create_obj_files(args, num=1):
    
    dir = Path("/home/halperin/ML3D/dataset") / "obj_files"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    
    files = []
    path = Path(args.target_path)
    for s in os.listdir(path):
        orig_files = path / s / args.original_name
        rotated_files = path / s / args.target_name
        for c in os.listdir(orig_files):
            class_name = orig_files / c
            for f in os.listdir(class_name):
                files.append((str(class_name / f), str(rotated_files / c / f)))
                
    
    indices = np.random.choice(np.arange(len(files)), num)
    
    for j, i in enumerate(indices):
        orig = np.asarray(open3d.io.read_point_cloud(files[i][0]).points)
        target = np.asarray(open3d.io.read_point_cloud(files[i][1]).points)
        write_pcd_to_obj(dir / f"{j + 1}_1.obj", orig)
        write_pcd_to_obj(dir / f"{j + 1}_2.obj", target)
            
        

def parse_args():
    """
    config the parameter
    """
    parser = argparse.ArgumentParser(description="The argument parser of R2Net runner")

    # choose model
    parser.add_argument("--data_path", type=str, default="/home/halperin/ML3D/dataset/ShapeNetCompletion", help="Original dataset")
    parser.add_argument("--target_path", type=str, default="/home/halperin/ML3D/dataset/ShapeNetTransform", help="Where to create the new dataset")
    parser.add_argument("--target_name", type=str, default="rotated", help="What name would diffrintiate the orig from the target")
    parser.add_argument("--original_name", type=str, default="original", help="What name would diffrintiate the orig from the target")
    parser.add_argument("--target_type", type=str, default="pcd", help="What is the type of the data")

    return parser.parse_args()

def create_transposed_data(args):
    dataset_path = Path(args.data_path)
    trans_data_path = Path(args.target_path)
    dirs = os.listdir(dataset_path)
    for split in tqdm(dirs, desc="Splits"):
        comp_dir = dataset_path / split / "complete"
        classes = os.listdir(comp_dir)
        for c in tqdm(classes, leave=False, desc="classes"):
            files = os.listdir(comp_dir / c)
            for f in tqdm(files, leave=False, desc="files"):
                f = Path(f)
                new_data_orig = trans_data_path / split / args.original_name  / c 
                os.makedirs(new_data_orig, exist_ok=True)
                
                orig_file = comp_dir / c / f
                shutil.copy2(orig_file, new_data_orig)  # Move the original pc to the new folder.
                
                new_data_rot = trans_data_path / split / args.target_name / c 
                os.makedirs(new_data_rot, exist_ok=True)
                
                orig_file = new_data_orig / f
                target_file = new_data_rot / f
                
                orig_pcd = open3d.io.read_point_cloud(str(orig_file))
                R = Rot.random().as_matrix()
                t = np.random.rand(3)
                trans = np.matmul(np.asarray(orig_pcd.points), R) + t
                write_pcd(str(target_file), trans)
                time.sleep(0.01)
            # return    
            time.sleep(0.01)

def create_partial_data(args):
    dataset_path = Path(args.data_path)
    trans_data_path = Path(args.target_path)
    dirs = os.listdir(dataset_path)
    for split in tqdm(dirs, desc="Splits"):
        comp_dir = dataset_path / split / "partial"
        new_data_orig = trans_data_path / split / "partial"
        if os.path.exists(new_data_orig):
            shutil.rmtree(new_data_orig)
        os.makedirs(new_data_orig, exist_ok=True)
        copy_tree(str(comp_dir), str(new_data_orig))
        


if __name__ == '__main__':
    # This code won't run if this file is imported.
    args = parse_args()
    # create_transposed_data(args)
    # create_obj_files(args, 3)
    create_partial_data(args)
                