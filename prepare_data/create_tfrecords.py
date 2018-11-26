import os.path as osp
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../code'))
import tensorflow as tf
from utils.misc import get_filenames
from utils.pc_util import load, normalize_point_cloud
from tf_ops.grouping.tf_grouping import knn_point
from pprint import pprint


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


if __name__ == '__main__':
    num_point = 312
    shard = 10
    out_dir = "/mnt/external/points/data/SketchFab"
    # dirs = [
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_5000",
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_10000",
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_20000",
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_40000",
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_80000",
    #     "/mnt/external/points/data/SketchFab/train_points/poisson_160000",
    #     # "/mnt/external/points/data/SketchFab/train_points/poisson_320000",
    # ]
    # npoints = [5000, 10000, 20000, 40000, 80000, 160000]
    dirs = [
        "/mnt/external/points/data/SketchFab/scan_train/56",
        "/mnt/external/points/data/SketchFab/scan_train/79",
        "/mnt/external/points/data/SketchFab/scan_train/112",
        "/mnt/external/points/data/SketchFab/scan_train/158",
        "/mnt/external/points/data/SketchFab/scan_train/224",
        "/mnt/external/points/data/SketchFab/scan_train/316",
    ]
    npoints = [5000, 10000, 20000, 40000, 80000, 160000]
    # npoints = [10000, 20000, 40000, 80000, 160000]
    ratios = [n // npoints[0] for n in npoints]
    num_patch_per_shape = int(npoints[0] / num_point * 10)
    print("num_patch_per_shape:", num_patch_per_shape)
    # npoins = [156, 312, 625, 1250, 2500, 5000]
    datasets = ["res_%d" % n for n in npoints]
    # file_paths = get_filenames(osp.join(dirs[0], 'train'), PC_EXTENSIONS)
    file_paths = get_filenames(osp.join(dirs[0]), ["ply", "pcd", "xyz"])
    file_paths = [f for f in file_paths if "train" in f]
    file_paths.sort()

    train_relpath = file_paths[:int(len(file_paths))]
    # keep relpath
    train_relpath = [f[len(dirs[0]):] for f in train_relpath]

    pprint(train_relpath)
    print("train:", len(dirs), len(train_relpath))

    train_relpath_shards = [train_relpath[start:start+shard] for start in range(0, len(train_relpath), shard)]

    # placeholder for point cloud
    input_placeholder = tf.placeholder(tf.float32, [1, None, 3])
    num_in_point_placeholder = tf.placeholder(tf.int32, [])
    seed_points_placeholder = tf.placeholder(tf.float32, [1, num_patch_per_shape, 3])

    _, knn_index = knn_point(num_in_point_placeholder, input_placeholder, seed_points_placeholder)
    # [batch_size, 1, num_gt_point/up_ratio, 3]
    input_patches = tf.gather_nd(input_placeholder, knn_index)
    input_patches = tf.squeeze(input_patches, axis=0)

    with tf.Session() as sess:
        for i, train_relpath in enumerate(train_relpath_shards):
            print("shard {}".format(i))
            print(train_relpath)
            with tf.python_io.TFRecordWriter(
                    os.path.join(out_dir, "{}_p{}_shard{}.tfrecord".format("_".join(datasets), num_point, i))) as writer:
                for p in train_relpath:
                    seed_points = None
                    centroid = furthest_distance = None
                    example = {}
                    for i, zipped in enumerate(zip(ratios, npoints, dirs, datasets)):
                        ratio, npc, dirname, dset = zipped
                        point_path = dirname + p
                        pc = load(point_path)[:, :3]
                        if seed_points is None:
                            seed_idx = np.random.choice(np.arange(pc.shape[0]),
                                size=[num_patch_per_shape], replace=False)
                            seed_points = pc[seed_idx, ...]
                        input_patches_value = sess.run(input_patches,
                            feed_dict={input_placeholder:pc[np.newaxis, ...],
                                       num_in_point_placeholder:num_point*ratio,
                                       seed_points_placeholder:seed_points[np.newaxis, ...]})
                        if furthest_distance is None or centroid is None:
                            input_patches_value, centroid, furthest_distance = normalize_point_cloud(input_patches_value)
                        else:
                            input_patches_value = (input_patches_value - centroid)/furthest_distance
                        example[dset] = input_patches_value
                    # each example [N, P, 3] to N examples
                    for i in range(num_patch_per_shape):
                        # [save_ply(example[k][i], "./{}_{}.ply".format(i, k)) for k in example]
                        features = {k: _floats_feature(v[i].flatten().tolist()) for k, v in example.items()}
                        tfexample = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(tfexample.SerializeToString())
