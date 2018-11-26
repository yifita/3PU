import tensorflow as tf
from utils import tf_util
import numpy as np
from model_utils import extract_patch_for_next_level, gen_1d_grid, gen_grid, knn_point, farthest_point_sample, \
    exponential_distance


def get_gen_model(point_cloud, is_training, scope="generator", bradius=1.0, gt_point_cloud=None,
                  reuse=tf.AUTO_REUSE, use_rv=False, use_bn=False, use_ibn=False, max_up_ratio=16,
                  use_normal=False, bn_decay=None, up_ratio=4, step_ratio=4, idx=None, no_res=False,
                  knn=16, growth_rate=12, dense_n=1, use_l0_points=False, adaptive_receptive_field=True,
                  h=0.05, fm_knn=3, **kwargs):

    comp = growth_rate*2
    l0_xyz = point_cloud[:, :, 0:3]
    batch_size = l0_xyz.shape[0]
    init_num_point = l0_xyz.get_shape()[1].value
    if adaptive_receptive_field:
        max_num_point = min(init_num_point, 312)
    else:
        max_num_point = init_num_point * max_up_ratio  # extract_patch will always be falses

    def crop_input_and_gt(l0_xyz, l0_points, gt, scale):
        # During training, if point cloud too large, extract (B, 1) patches for next level
        extract_patch = (l0_xyz.shape[1] > max_num_point)
        if extract_patch:
            # scale / step_ratio = last_scale, up_ratio // last_scale = remaining_scale
            gt_k = max_num_point * up_ratio // scale * step_ratio
            l0_xyz, l0_points, gt = extract_patch_for_next_level(l0_xyz, max_num_point, batch_features=l0_points, gt_xyz=gt, gt_k=gt_k, is_training=is_training)
            return l0_xyz, l0_points, gt
        else:
            return l0_xyz, l0_points, gt

    def body(global_idx, scale, l0_xyz, l0_points, bradius, gt, *args):
        scale_idx = int(np.log(scale)/np.log(step_ratio)) - 1
        gt_original = gt

        with tf.variable_scope("level_%d" % (scale_idx+1), reuse=tf.AUTO_REUSE):
            num_point_unaltered = l0_xyz.shape[1]
            if scale_idx > 0:
                l0_features = None if not use_l0_points else l0_points
                l0_xyz, l0_features, gt = crop_input_and_gt(l0_xyz, l0_features, gt, scale)
                if gt is not None:
                    gt = tf.stop_gradient(gt)
                else:
                    gt = gt_original
                l0_xyz_normalized, centroid, furthest_distance = tf_util.normalize_point_cloud(l0_xyz)
                bradius = tf.constant(1.0, shape=[l0_xyz.shape[0]], dtype=tf.float32)
                if not use_l0_points:
                    l0_features = tf.expand_dims(l0_xyz_normalized, axis=2)
                if len(l0_features.shape) == 3:
                    l0_features = tf.expand_dims(l0_features, axis=2)
            else:
                l0_features = tf.expand_dims(l0_xyz, axis=2)
                l0_xyz_normalized = l0_xyz

            num_point = l0_xyz.get_shape()[1].value or max_num_point
            l0_features = tf_util.conv2d(l0_features, 24, [1, 1],
                padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
            l0_features = tf.squeeze(l0_features, axis=2)

            # encoding layer
            l1_features, l1_idx = tf_util.dense_conv1(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)
            l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

            l2_features = tf_util.conv1d(l1_features, comp, 1,   # 24
                padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)
            l2_features, l2_idx = tf_util.dense_conv1(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

            l3_features = tf_util.conv1d(l2_features, comp, 1,   # 48
                padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 48
            l3_features, l3_idx = tf_util.dense_conv1(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

            l4_features = tf_util.conv1d(l3_features, comp, 1,   # 48
                padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 48
            l4_features, l3_idx = tf_util.dense_conv1(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

            l4_features = tf.expand_dims(l4_features, axis=2)

            if scale_idx > 0:
                with tf.name_scope("skip_connection"):
                    lower_scale_idx = scale_idx - 1
                    skip_feature = tf.get_collection("SKIP_FEATURES_%d_%d" % (global_idx, lower_scale_idx))
                    skip_feature_xyz = tf.get_collection("SKIP_FEATURE_XYZ_%d_%d" % (global_idx, lower_scale_idx))
                    assert(len(skip_feature_xyz) == 1)
                    skip_feature_xyz = skip_feature_xyz.pop()
                    skip_feature = skip_feature.pop()
                    if not is_training and skip_feature_xyz.shape[0] != l0_xyz.shape[0]:
                        skip_feature_xyz = tf.tile(skip_feature_xyz, [l0_xyz.shape[0].value, 1, 1])
                        skip_feature = tf.tile(skip_feature, [l0_xyz.shape[0].value, 1, 1, 1])
                    if fm_knn > 1:
                        # find closest k point in spatial
                        dist, idx = knn_point(fm_knn, skip_feature_xyz, l0_xyz, sort=True, unique=True)
                        knn_feature = tf.gather_nd(tf.squeeze(skip_feature, axis=2), idx)  # B, N, k, C
                        knn_xyz = tf.gather_nd(skip_feature_xyz, idx)  # B, N, k,
                        # compute distance in feature and sptial sapce
                        # knn_feature = tf.stop_gradient(knn_feature)
                        d, f_average_weight = exponential_distance(l4_features, knn_feature, scope="feature_matching_fweight")
                        tf.contrib.summary.histogram("f_average_distance_{}_{}".format(scale_idx, lower_scale_idx), d)
                        d, s_average_weight = exponential_distance(tf.expand_dims(l0_xyz, axis=2), knn_xyz, scope="feature_matching_sweight")
                        tf.contrib.summary.histogram("s_average_distance_{}_{}".format(scale_idx, lower_scale_idx), d)
                        s_average_weight = tf.stop_gradient(s_average_weight)
                        f_average_weight = tf.stop_gradient(f_average_weight)
                        # weights: B, P, K, 1
                        tf.contrib.summary.histogram("f_average_weight_{}_{}".format(scale_idx, lower_scale_idx), f_average_weight)
                        tf.contrib.summary.histogram("s_average_weight_{}_{}".format(scale_idx, lower_scale_idx), s_average_weight)
                        average_weight = (f_average_weight * s_average_weight)
                        # average_weight = s_average_weight
                        average_weight = average_weight / tf.reduce_sum(average_weight+1e-5, axis=2, keepdims=True)
                        knn_feature = tf.reduce_sum(average_weight * knn_feature, axis=2, keepdims=True)
                    else:
                        dist, idx = knn_point(1, skip_feature_xyz, l0_xyz, sort=True, unique=True)
                        knn_feature = tf.gather_nd(tf.squeeze(skip_feature, axis=2), idx)  # B, N, 1, C

                    l4_features = 0.2*knn_feature + l4_features

            tf.add_to_collection("SKIP_FEATURE_XYZ_%d_%d" % (global_idx, scale_idx),
                tf.concat(tf.split(l0_xyz, l0_xyz.shape[0]//batch_size, axis=0), axis=1))
            tf.add_to_collection("SKIP_FEATURES_%d_%d" % (global_idx, scale_idx),
                tf.concat(tf.split(l4_features, l4_features.shape[0]//batch_size, axis=0), axis=1))

            with tf.variable_scope('up_layer', reuse=reuse):
                if not np.isscalar(bradius):
                    bradius_expand = tf.expand_dims(tf.expand_dims(bradius, axis=-1), axis=-1)
                else:
                    bradius_expand = bradius

                if step_ratio < 4:
                    grid = gen_1d_grid(step_ratio)
                    expansion_ratio = step_ratio
                else:
                    grid = gen_grid(np.round(np.sqrt(step_ratio)).astype(np.int32))
                    expansion_ratio = (np.round(np.sqrt(step_ratio))**2).astype(np.int32)

                num_point = tf.shape(l0_xyz)[1]
                grid = tf.tile(tf.expand_dims(grid, 0), [l0_xyz.shape[0], num_point, 1])  # [batch_size, num_point*4, 2])
                grid = tf.expand_dims(grid*bradius_expand, axis=2)

                # [B, N, 1, 1, 256+3] -> [B, N, 4, 1, 256+3] -> [B, num_point*4, 1, 256+3]
                new_feature = tf.reshape(
                    tf.tile(tf.expand_dims(l4_features, 2), [1, 1, expansion_ratio, 1, 1]), [l4_features.shape[0], num_point*expansion_ratio, 1, l4_features.shape[-1]])
                new_feature = tf.concat([new_feature, grid], axis=-1)

                new_feature = tf_util.conv2d(new_feature, 128, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='up_layer1', bn_decay=bn_decay)

                new_feature = tf_util.conv2d(new_feature, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=use_bn, is_training=is_training,
                                             scope='up_layer2',
                                             bn_decay=bn_decay)

            # get the xyz
            new_xyz = tf_util.conv2d(new_feature, 64, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer1', bn_decay=bn_decay)

            new_xyz = tf_util.conv2d(new_xyz, 3, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='fc_layer2', bn_decay=bn_decay,
                                    activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3

            new_xyz = tf.squeeze(new_xyz, axis=2)  # B*(2N)*3
            new_feature = tf.squeeze(new_feature, axis=2)  # B*(2N)*3
            if not no_res:
                new_xyz += tf.reshape(tf.tile(tf.expand_dims(l0_xyz_normalized, 2), [1, 1, expansion_ratio, 1]), [l0_xyz_normalized.shape[0], num_point*expansion_ratio, -1])  # B, N, 4, 3
            if scale_idx > 0:
                new_xyz = (new_xyz * furthest_distance) + centroid
                if not is_training and (new_xyz.shape[0] != batch_size):
                    new_xyz = tf.concat(tf.split(new_xyz, new_xyz.shape[0]//batch_size, axis=0), axis=1)
                    output_num = num_point_unaltered*step_ratio
                    # resample to get sparser points idx [B, P, 1]
                    idx = tf.expand_dims(farthest_point_sample(output_num, new_xyz), axis=-1)
                    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, output_num, 1))
                    new_xyz = tf.gather_nd(new_xyz, tf.concat([batch_indices, idx], axis=-1))
                    new_feature = tf.gather_nd(new_feature, tf.concat([batch_indices, idx], axis=-1))
                bradius = furthest_distance
            return new_xyz, new_feature, bradius, gt, l0_xyz

    if is_training:
        assert(gt_point_cloud is not None)
    else:
        gt_point_cloud = l0_xyz

    cases = []

    def create_body(i):
        def recursive_call_body(sub_i):
            if sub_i == 0:
                xyz, points, out_radius, out_gt, out_l0_xyz = body(i, step_ratio, l0_xyz, None, bradius, gt_point_cloud)

                return xyz, points, out_radius, out_gt, out_l0_xyz
            else:
                xyz, points, out_radius, out_gt, out_l0_xyz = body(i, step_ratio**(sub_i+1), *recursive_call_body(sub_i-1))
                return xyz, points, out_radius, out_gt, out_l0_xyz

        return recursive_call_body(i)

    for i in range(int(np.log(max_up_ratio) / np.log(step_ratio))):
        cases.append((tf.equal(up_ratio, step_ratio**(i+1)), lambda i=i: create_body(i)))

    with tf.name_scope(scope):
        xyz, points, bradius, gt, l0_xyz = tf.case(cases)

    total_param_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("===Total number of parameters: %d===" % total_param_num)
    return xyz, points, l0_xyz, gt, bradius
