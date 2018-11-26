from argparse import ArgumentParser
import os
from glob import glob
import subprocess
from multiprocessing import Pool, Lock
import sys
import itertools

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../code'))
from utils.misc import get_filenames

parser = ArgumentParser()
parser.add_argument("--pred", type=str, nargs="+", required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gt", type=str, required=True, help="mesh")
parser.add_argument("--nuc", action="store_true")
parser.add_argument("--no-p2f", dest="p2f", action="store_false")
FLAGS = parser.parse_args()
PRED_DIR = FLAGS.pred
GT_DIR = FLAGS.gt
NAME = FLAGS.name

globallock = Lock()
FNULL = open(os.devnull, 'w')

def processfile_p2f(gt_pred_pair):
    gt, pred = gt_pred_pair
    args = ['./build/evaluation', gt, pred]
    rv = subprocess.call(args)
    globallock.acquire()
    os.path.isfile(gt)
    os.path.isfile(pred)
    if rv == 0:
        print("File '{} {}' processed.".format(gt, pred))
    else:
        print("Error when processing file '{} {}'".format(gt, pred))
    globallock.release()

def processfile_nuc(gt_pred_pair):
    gt, pred = gt_pred_pair
    args = ['./build/nuc', gt, pred]
    # rv = subprocess.call(args, stdout=FNULL)
    rv = subprocess.call(args)
    globallock.acquire()
    os.path.isfile(gt)
    os.path.isfile(pred)
    if rv == 0:
        print("File '{} {}' processed.".format(gt, pred))
    else:
        print("Error when processing file '{} {}'".format(gt, pred))
    globallock.release()


if __name__ == '__main__':
    gt_paths = get_filenames(GT_DIR, "off")
    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
    print(gt_names)
    pred_paths = [glob(os.path.join(d, "**", NAME), recursive=True) for d in PRED_DIR]
    pred_paths = list(itertools.chain(*pred_paths))
    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert(ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))
    print("%d files to run" % len(gt_pred_pairs))
    p = Pool(processes=1)
    if FLAGS.nuc:
        p.map(processfile_nuc, gt_pred_pairs)
    if FLAGS.p2f:
        p.map(processfile_p2f, gt_pred_pairs)
    p.close()
