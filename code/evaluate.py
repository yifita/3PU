import argparse
import os
import numpy as np
import tensorflow as tf
from glob import glob
import re
import csv
from collections import OrderedDict

from model_utils import get_cd_loss, get_hausdorff_loss
from utils.misc import get_filenames
from utils import pc_util
from utils.pc_util import load, save_ply_property
from utils.tf_util2 import normalize_point_cloud
from tf_ops.CD import tf_nndistance
from tf_ops.grouping.tf_grouping import knn_point


parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, nargs="+", required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gt", type=str, required=True, help="mesh")
FLAGS = parser.parse_args()
PRED_DIR = FLAGS.pred
GT_DIR = FLAGS.gt
NAME = FLAGS.name

gt_paths = get_filenames(GT_DIR, ("ply", "pcd", "xyz"))
gt_names = [os.path.basename(p)[:-4] for p in gt_paths]


pred_placeholder = tf.placeholder(tf.float32, [1, None, 3])
gt_placeholder = tf.placeholder(tf.float32, [1, None, 3])
pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_placeholder)
gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_placeholder)

# B, P_predict, 1
# cd_forward, _ = knn_point(1, gt_tensor, pred_tensor)
# cd_backward, _ = knn_point(1, pred_tensor, gt_tensor)
# cd_forward = cd_forward[0, :, 0]
# cd_backward = cd_backward[0, :, 0]
cd_forward, _, cd_backward, _ = tf_nndistance.nn_distance(pred_tensor, gt_tensor)
cd_forward = cd_forward[0, :]
cd_backward = cd_backward[0, :]

with tf.Session() as sess:
    fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"] + ["nuc_%d" % d for d in range(7)]
    print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
    for D in PRED_DIR:
        avg_md_forward_value = 0
        avg_md_backward_value = 0
        avg_hd_value = 0
        counter = 0
        pred_paths = glob(os.path.join(D, "**", NAME), recursive=True)

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

        # print("total inputs ", len(gt_pred_pairs))
        tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
        if tag:
            tag = tag.groups()[0]
        else:
            tag = D

        print("{:60s}".format(tag), end=' ')
        global_p2f = []
        global_density = []
        with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
            writer.writeheader()
            for gt_path, pred_path in gt_pred_pairs:
                row = {}
                gt = load(gt_path)[:, :3]
                gt = gt[np.newaxis, ...]
                pred = pc_util.load(pred_path)
                pred = pred[:, :3]
                row["name"] = os.path.basename(pred_path)
                pred = pred[np.newaxis, ...]
                cd_forward_value, cd_backward_value = sess.run([cd_forward, cd_backward], feed_dict={pred_placeholder:pred, gt_placeholder:gt})
                save_ply_property(np.squeeze(pred), cd_forward_value, pred_path[:-4]+"_cdF.ply", property_max=0.003, cmap_name="jet")
                save_ply_property(np.squeeze(gt), cd_backward_value, pred_path[:-4]+"_cdB.ply", property_max=0.003, cmap_name="jet")
                # cd_backward_value = cd_forward_value = 0.0
                md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
                hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
                cd_backward_value = np.mean(cd_backward_value)
                cd_forward_value = np.mean(cd_forward_value)
                # row["CD_forward"] = np.mean(cd_forward_value)
                # row["CD_backwar"] = np.mean(cd_backward_value)
                row["CD"] = cd_forward_value+cd_backward_value

                row["hausdorff"] = hd_value
                avg_md_forward_value += cd_forward_value
                avg_md_backward_value += cd_backward_value
                avg_hd_value += hd_value
                if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
                    point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
                    if point2mesh_distance.size == 0:
                        continue
                    point2mesh_distance = point2mesh_distance[:, 3]
                    row["p2f avg"] = np.nanmean(point2mesh_distance)
                    row["p2f std"] = np.nanstd(point2mesh_distance)
                    global_p2f.append(point2mesh_distance)
                if os.path.isfile(pred_path[:-4] + "_density.xyz"):
                    density = load(pred_path[:-4] + "_density.xyz")
                    global_density.append(density)
                    std = np.std(density, axis=0)
                    for i in range(7):
                        row["nuc_%d" % i] = std[i]
                writer.writerow(row)
                counter += 1

            row = OrderedDict()

            avg_md_forward_value /= counter
            avg_md_backward_value /= counter
            avg_hd_value /= counter
            # row["CD_forward"] = avg_md_forward_value
            # row["CD_backward"] = avg_md_backward_value
            row["CD"] = avg_md_forward_value+avg_md_backward_value
            row["hausdorff"] = avg_hd_value
            if global_p2f:
                global_p2f = np.concatenate(global_p2f, axis=0)
                mean_p2f = np.nanmean(global_p2f)
                std_p2f = np.nanstd(global_p2f)
                row["p2f avg"] = mean_p2f
                row["p2f std"] = std_p2f
            if global_density:
                global_density = np.concatenate(global_density, axis=0)
                nuc = np.std(global_density, axis=0)
                for i in range(7):
                    row["nuc_%d" % i] = std[i]

            writer.writerow(row)
            print("|".join(["{:>15.8f}".format(d) for d in row.values()]))
