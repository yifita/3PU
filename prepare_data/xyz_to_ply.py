import numpy as np
import os
import sys
from argparse import ArgumentParser
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../code'))
from utils.pc_util import save_ply

parser = ArgumentParser()
parser.add_argument('txtfiles', nargs='+')
args =parser.parse_args()
file_paths = args.txtfiles

for f in file_paths:
    print(os.path.basename(f))
    points = np.loadtxt(f, delimiter=' ').astype(np.float32)
    if points.shape[1] == 6:
        save_ply(points[:, :3], os.path.splitext(f)[0]+'.ply', normals=points[:, 3:])
    else:
        save_ply(points, os.path.splitext(f)[0]+'.ply')
