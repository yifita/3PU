import tensorflow as tf
import os
import numpy as np
import h5py
import re
import itertools

from tf_ops.grouping.tf_grouping import knn_point
from utils import logger

is_2D = False


def load_patch_data(h5_filename,
        up_ratio=4, step_ratio=2, num_point=2048, patch_size=None, use_randominput=True, norm=True):
    # load h5 file, infer dataset names poisson_x from h5 file name.
    # given up_ratio, load {poisson_x|x~=max(x)/u, 1<u<=up_ratio} (inputs) and {poisson_x|x=max(x)}
    # radius and names
    # return dict(u=[N, max(x)/u, 3], u/4=[N, max(x)/(u/4), 3], ..., gt=[N, max(x), 3], data_radius=radius, name=name)
    h5_filepath = os.path.join(h5_filename)
    num_points = re.findall("\d+", os.path.basename(h5_filepath)[:-5])
    num_points = list(map(int, num_points))
    num_points.sort()
    num_points = np.asarray(num_points)
    if num_point is None:
        max_num_points = np.max(num_points)
        # all points smaller or equal to max_num_points//up_ratio
        num_in_point = num_points[:np.searchsorted(num_points, max_num_points//up_ratio, side="right")]
        if patch_size is not None:
            num_in_point = num_in_point[np.searchsorted(num_in_point, patch_size, side="left"):]
    else:
        num_in_point = [num_points[np.searchsorted(num_points, num_point)]]

    f = h5py.File(h5_filepath, "r")
    tag = re.findall("_([A-Za-z]+)_", os.path.basename(h5_filepath))[-1]

    all_data = []
    all_label = []
    all_radius = []
    all_name = []
    for num in num_in_point:
        data = f[tag+"_%d" % num][:, :, 0:3]
        logger.info("input point_num %d" % data.shape[1])

        radius = np.ones(shape=(len(data)))
        centroid = np.mean(data[:, :, 0:3], axis=1, keepdims=True)
        data[:, :, 0:3] = data[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(data[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        data[:, :, 0:3] = data[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        label = {}
        for x in range(1, int(np.log2(up_ratio)//np.log2(step_ratio))+1):
            r = step_ratio**x
            closest_larger_equal = num_points[np.searchsorted(num_points, num*r)]
            label["x%d" % r] = f[tag+"_%d" % closest_larger_equal][:, :, :3]
            label["x%d" % r][:, :, 0:3] = label["x%d" % r][:, :, 0:3] - centroid
            label["x%d" % r][:, :, 0:3] = label["x%d" % r][:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
            logger.info("gt (ratio %d), point_num %d " % (r, label["x%d" % r].shape[1]))

        name = f["name"][:]
        name = [n.decode() for n in name]

        global is_2D
        if np.all(data[:,:,2] == 0):
            is_2D = True
            logger.info("2D dataset")
        else:
            is_2D = False
            logger.info("3D dataset")

        # print(("load object names {}".format(name)))
        logger.info(("total %d samples" % (data.shape[0])))
        all_data.append(data)
        all_label.append(label)
        all_radius.append(radius)
        all_name.append(name)

    f.close()

    return all_data, all_label, all_radius, all_name

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keepdims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


class Fetcher(object):
    """docstring for Fetcher"""
    def __init__(self, input_pointcloud, label_pointclouds, data_radius, batch_size, num_in_point, step_ratio, up_ratio,
            jitter=False, jitter_max=0.05, jitter_sigma=0.03, drop_out=1.0, **kwargs):
        super(Fetcher, self).__init__()
        self.label_pointclouds = label_pointclouds
        self.input_pointcloud = input_pointcloud
        self.data_radius = data_radius
        self.batch_size = batch_size
        self.jitter = jitter
        self.jitter_max = jitter_max
        self.jitter_sigma = jitter_sigma
        self.drop_out = drop_out
        self.ratio = tf.Variable(step_ratio, name="ratio", trainable=False)
        self.num_in_point = tf.Variable(num_in_point, name="num_in_point", trainable=False)
        self.num_gt_point = self.num_in_point*self.ratio
        self.ratio_placeholder = tf.placeholder(tf.int32, [])
        self.update_ratio = tf.assign(self.ratio, self.ratio_placeholder)
        self.num_batches = 300
        self.dataset = None
        self.placeholders = []

        for label, input in zip(label_pointclouds, input_pointcloud):
            input_placeholder = tf.placeholder(tf.float32, input.shape)
            label_placeholder = tf.placeholder(tf.float32, [input.shape[0], None, 3])
            radius_placeholder = tf.placeholder(tf.float32, [input.shape[0]])
            if self.dataset is None:
                self.dataset = tf.data.Dataset.from_tensors({"label": label_placeholder, "input":input_placeholder, "radius":radius_placeholder})
            else:
                self.dataset = self.dataset.concatenate(tf.data.Dataset.from_tensors({"label": label_placeholder, "input":input_placeholder, "radius":radius_placeholder}))
            self.placeholders.append({"label": label_placeholder, "input":input_placeholder, "radius":radius_placeholder})

        self.dataset = self.dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(x), cycle_length=len(self.placeholders))
        # sess.run(self.next_element)
        assert(num_in_point <= input.shape[1])
        self.dataset = self.dataset.map(self.shape_to_patch, num_parallel_calls=16).map(self.augment_data, num_parallel_calls=8)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(500))
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def fetch(self, sess):
        # return label, input, radius
        return sess.run(self.next_element)

    def initialize(self, sess, ratio, *args, **kwargs):
        sess.run(self.update_ratio, feed_dict={self.ratio_placeholder: ratio})
        logger.info("data_provider: ratio %d" % self.ratio.eval(), bold=True)
        feed_dict = [[(pls["label"], self.label_pointclouds[i]["x%d" % ratio]), (pls["input"], self.input_pointcloud[i]), (pls["radius"], self.data_radius[i])]
            for i, pls in enumerate(self.placeholders)]
        feed_dict = dict(itertools.chain.from_iterable(feed_dict))
        sess.run(self.iterator.initializer,
            feed_dict=feed_dict)

    def shape_to_patch(self, label_input_radius):
        # random sample points as seed
        # knn on seeds
        # normalize patch and update radius
        input_pc, label_pc, radius = label_input_radius['input'], label_input_radius['label'], label_input_radius['radius']
        label_pc = tf.expand_dims(label_pc, 0)
        input_pc = tf.expand_dims(input_pc, 0)
        if self.jitter:
            input_pc, centroid, furthest_distance = normalize_point_cloud(input_pc)
            input_pc = self.jitter_perturbation_point_cloud(input_pc, sigma=self.jitter_sigma, clip=self.jitter_max)
        rnd_pts = tf.random_uniform((1, self.batch_size), dtype=tf.int32, maxval=tf.shape(input_pc)[1])
        rnd_pts = tf.batch_gather(label_pc, rnd_pts)  # [1, batch_size, 3]
        _, knn_index = knn_point(self.num_gt_point, label_pc, rnd_pts)  # [1, batch_size, num_gt_point, 2]
        label_patches = tf.gather_nd(label_pc, knn_index)  # [1, batch_size, num_gt_point, 3]
        _, knn_index = knn_point(self.num_in_point, input_pc, rnd_pts)
        input_patches = tf.gather_nd(input_pc, knn_index)  # [1, batch_size, num_gt_point/up_ratio, 3]

        label_patches = tf.squeeze(label_patches, axis=0)  # [batch_size, num_gt_point, 3]
        input_patches = tf.squeeze(input_patches, axis=0)

        label_patches, centroid, furthest_distance = self.normalize_point_cloud(label_patches)
        input_patches = (input_patches - centroid)/furthest_distance
        radius = tf.constant(np.ones((self.batch_size)), dtype=tf.float32)

        return {"label": label_patches, "input": input_patches, "radius": radius}

    def augment_data(self, label_input_radius):
        input_patches, label_patches, radius = label_input_radius['input'], label_input_radius['label'], label_input_radius['radius']
        input_patches, label_patches = self.rotate_point_cloud_and_gt(input_patches, label_patches)
        input_patches, label_patches, scales = self.random_scale_point_cloud_and_gt(input_patches, label_patches,
            scale_low=0.8, scale_high=1.2)
        radius = radius*scales
        if self.drop_out < 1:
            # randomly discard input
            num_point = input_patches.shape[1].value
            point_idx = tf.random_shuffle(tf.range(num_point))[:int(num_point*self.drop_out)]
            input_patches = tf.gather(input_patches, point_idx, axis=1)
        return input_patches, label_patches, radius, self.ratio

    def jitter_perturbation_point_cloud(self, batch_data, sigma=0.005, clip=0.02):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        assert(clip > 0)
        jittered_data = tf.clip_by_value(sigma * tf.random_normal(tf.shape(batch_data)), -1 * clip, clip)
        jittered_data = tf.concat([batch_data[:, :, :3] + jittered_data[:, :, :3], batch_data[:, :, 3:]], axis=-1)
        return jittered_data

    def rotate_point_cloud_and_gt(self, batch_data, batch_gt=None):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        batch_size, num_point, num_channels = batch_data.get_shape().as_list()
        angles = tf.random_uniform((batch_size, 3), dtype=tf.float32) * 2 * np.pi
        cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
        sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
        one  = tf.ones_like(cos_x,  dtype=tf.float32)
        zero = tf.zeros_like(cos_x, dtype=tf.float32)
        # [B, 3, 3]
        Rx = tf.stack(
            [tf.concat([one,   zero,   zero], axis=1),
             tf.concat([zero,  cos_x, sin_x], axis=1),
             tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

        Ry = tf.stack(
            [tf.concat([cos_y, zero, -sin_y], axis=1),
             tf.concat([zero,  one,    zero], axis=1),
             tf.concat([sin_y, zero,  cos_y], axis=1)], axis=1)

        Rz = tf.stack(
            [tf.concat([cos_z,  sin_z, zero], axis=1),
             tf.concat([-sin_z, cos_z, zero], axis=1),
             tf.concat([zero,   zero,   one], axis=1)], axis=1)

        if is_2D:
            rotation_matrix = Rz
        else:
            rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

        if num_channels > 3:
            batch_data = tf.concat(
                [tf.matmul(batch_data[:, :, :3], rotation_matrix),
                 tf.matmul(batch_data[:, :, 3:], rotation_matrix),
                 batch_data[:, :, 6:]], axis=-1)
        else:
            batch_data = tf.matmul(batch_data, rotation_matrix)

        if batch_gt is not None:
            if num_channels > 3:
                batch_gt = tf.concat(
                    [tf.matmul(batch_gt[:, :, :3], rotation_matrix),
                     tf.matmul(batch_gt[:, :, 3:], rotation_matrix),
                     batch_gt[:, :, 6:]], axis=-1)
            else:
                batch_gt = tf.matmul(batch_gt, rotation_matrix)

        return batch_data, batch_gt

    def rotate_perturbation_point_cloud(self, batch_data, angle_sigma=0.03, angle_clip=0.09):
        """ Randomly perturb the point clouds by small rotations
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        batch_size, num_point, num_channels = batch_data.get_shape().as_list()
        angles = tf.clip_by_value(tf.random_normal((batch_size, 3))*angle_sigma, -angle_clip, angle_clip)

        cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
        sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
        one  = tf.ones_like(cos_x,  dtype=tf.float32)
        zero = tf.zeros_like(cos_x, dtype=tf.float32)
        # [B, 3, 3]
        Rx = tf.stack(
            [tf.concat([one,   zero,   zero], axis=1),
             tf.concat([zero,  cos_x, sin_x], axis=1),
             tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

        Ry = tf.stack(
            [tf.concat([cos_y, zero, -sin_y], axis=1),
             tf.concat([zero,  one,    zero], axis=1),
             tf.concat([sin_y, zero,  cos_y], axis=1)], axis=1)

        Rz = tf.stack(
            [tf.concat([cos_z,  sin_z, zero], axis=1),
             tf.concat([-sin_z, cos_z, zero], axis=1),
             tf.concat([zero,   zero,   one], axis=1)], axis=1)

        if is_2D:
            rotation_matrix = Rz
        else:
            rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

        if num_channels > 3:
            batch_data = tf.concat(
                [tf.matmul(batch_data[:, :, :3], rotation_matrix),
                 tf.matmul(batch_data[:, :, 3:], rotation_matrix),
                 batch_data[:, :, 6:]], axis=-1)
        else:
            batch_data = tf.matmul(batch_data, rotation_matrix)

        return batch_data

    def random_scale_point_cloud_and_gt(self, batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.get_shape().as_list()
        scales = tf.random_uniform((B, 1, 1), minval=scale_low, maxval=scale_high, dtype=tf.float32)

        batch_data = tf.concat([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

        if batch_gt is not None:
            batch_gt = tf.concat([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

        return batch_data, batch_gt, tf.squeeze(scales)

    def normalize_point_cloud(self, pc):
        """
        pc [N, P, 3]
        """
        centroid = tf.reduce_mean(pc, axis=1, keepdims=True)
        pc = pc - centroid
        furthest_distance = tf.reduce_max(
            tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
        pc = pc / furthest_distance
        return pc, centroid, furthest_distance


if __name__ == '__main__':
    import time
    from utils.pc_util import save_ply
    up_ratio = 16
    batch_size = 10
    num_point = 156
    h5_filename = "h5_data/chair_train_poisson_156_poisson_310_poisson_625_poisson_1250_poisson_2500_poisson_5000.hdf5"
    input_data, label, radius, name = load_patch_data(h5_filename=h5_filename, num_point=None, up_ratio=up_ratio, step_ratio=4)
    # input_data, label, radius, name = load_patch_data(num_point=5000, up_ratio=16, step_ratio=4)
    with tf.device('/cpu:0'):
        # fetcher = Fetcher(input_data, label, radius, batch_size=10,
        #     step_ratio=4, up_ratio=16, num_in_point=1024)
        fetcher = Fetcher(input_data, label, radius, batch_size=batch_size,
            step_ratio=4, up_ratio=up_ratio, num_in_point=num_point, jitter=True)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for ratio in (4,):
            fetcher.initialize(sess, ratio)
            print("x%d" % ratio)
            for cnt in range(0, 5*batch_size, batch_size):
                start = time.time()
                data = fetcher.fetch(sess)
                input, gt, radius = data
                assert len(input) == len(gt)
                end = time.time()
                print((cnt, end - start))
                for i in range(len(input)):
                    # while True:
                    #     cmd = show3d.showpoints(input[i, :, 0:3])
                    #     if cmd == ord(' '):
                    #         break
                    #     elif cmd == ord('q'):
                    #         break
                    # if cmd == ord('q'):
                    #     break
                    save_ply(input[i], "./x%d_%d_in.ply" % (ratio, cnt+i))
                    save_ply(gt[i], "./x%d_%d_gt.ply" % (ratio, cnt+i))
                    # plot_pcd_three_views("input_gt_%d_%d.png" % (cnt, i), [input[i, ...], gt[i, ...]], ["input", "gt"])
