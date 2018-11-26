import traceback
import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import importlib
import socket
import model_utils
import mixed_data_provider
import curriculum_data_provider
import curriculumn_record_provider as record_data_provider
from utils import pc_util, tf_util, logger
from tf_ops.sampling.tf_sampling import farthest_point_sample


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test',
                    help='train or test [default: train]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--id', help="experiment name, prepended to log_dir")
parser.add_argument('--log_dir', default='../model', help='Log dir [default: log]')
parser.add_argument('--model', default='model_microscope', help='model name')
parser.add_argument('--root_dir', default='../', help='project root, data and h5_data diretories')
parser.add_argument('--result_dir', help='result directory')
parser.add_argument('--restore', help='model to restore from')
parser.add_argument('--num_point', type=int, help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--num_shape_point', type=int, help="Number of points per shape")
parser.add_argument('--no-repulsion', dest='repulsion', action='store_false', help='activate repulsion loss')
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 2]')
parser.add_argument('--max_epoch', type=int, default=160, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=28, help='Batch Size during training [default: 32]')
parser.add_argument('--h5_data', help='h5 file for training')
parser.add_argument('--record_data', help='record file for training')
parser.add_argument('--test_data', default='data/test_data/sketchfab_poisson/poisson_5000/*.xyz', help='h5 file for training')
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--restore_epoch', type=int)
parser.add_argument('--stage_steps', type=int, default=15000, help="number of updates per curriculums stage")
parser.add_argument('--step_ratio', type=int, default=2, help="upscale ratio per step")
parser.add_argument('--patch_num_ratio', type=float, default=3)
parser.add_argument('--jitter', action="store_true", help="jitter augmentation")
parser.add_argument('--jitter_sigma', type=float, default=0.0025, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.005, help="jitter augmentation")
parser.add_argument('--drop_out', type=float, default=1.0, help="drop_out ratio. default 1.0 (no drop out) ")
parser.add_argument('--knn', type=int, default=32, help="neighbood size for edge conv")
parser.add_argument('--dense_n', type=int, default=3, help="number of dense layers")
parser.add_argument('--block_n', type=int, default=3, help="number of dense blocks")
parser.add_argument('--fm_knn', type=int, default=5, help="number of neighboring points for feature matching")
parser.add_argument('--no_adaptive_receptive_field', dest="adaptive_receptive_field", action="store_false", help="use last feature")
parser.add_argument('--cd_threshold', default=2.0, type=float, help="threshold for cd")
parser.add_argument('--repulsion_weight', default=1.0, type=float, help="repulsion_weight")
parser.add_argument('--fidelity_weight', default=50.0, type=float, help="repulsion_weight")

USE_DATA_NORM = True
USE_RANDOM_INPUT = True

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
ASSIGN_MODEL_PATH = FLAGS.restore
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point or int(FLAGS.num_shape_point * FLAGS.drop_out)
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MODEL_DIR = os.path.join(FLAGS.log_dir, FLAGS.id)
ROOT_DIR = FLAGS.root_dir
MODEL_GEN = importlib.__import__(FLAGS.model)
USE_REPULSION_LOSS = FLAGS.repulsion
TRAIN_H5 = FLAGS.h5_data
TRAIN_RECORD = FLAGS.record_data
TEST_DATA = FLAGS.test_data
STAGE_STEPS = FLAGS.stage_steps
STEP_RATIO = FLAGS.step_ratio
RESTORE_EPOCH = FLAGS.restore_epoch
NUM_SHAPE_POINT = FLAGS.num_shape_point
PATCH_NUM_RATIO = FLAGS.patch_num_ratio
JITTER = FLAGS.jitter
ADAPTIVE_RECEPTIVE_FIELD = FLAGS.adaptive_receptive_field
FM_KNN = FLAGS.fm_knn
if TRAIN_H5 is not None:
    if NUM_SHAPE_POINT:
        data_provider = curriculum_data_provider
    else:
        data_provider = mixed_data_provider
elif TRAIN_RECORD is not None:
    data_provider = record_data_provider

assert(NUM_SHAPE_POINT is not None or NUM_POINT is not None)

print((socket.gethostname()))
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX


class Network(object):

    def get_next_ratio(self, step):
        idx = min((step + 1 + STAGE_STEPS) // (2*STAGE_STEPS)+1, np.log2(UP_RATIO)/np.log2(STEP_RATIO))
        ratio = STEP_RATIO ** idx
        return ratio

    def get_loss(self, pred, gt, radius, fidelity_loss_weight=50, repulsion_loss_weight=1, cd_forward_weight=1.0, threshold=None, scope="loss"):
        with tf.name_scope(scope):
            gen_fidelity_loss = model_utils.get_cd_loss(pred, gt, radius, cd_forward_weight, threshold)[0]*fidelity_loss_weight
            if USE_REPULSION_LOSS:
                gen_repulsion_loss = model_utils.get_repulsion_loss(pred)*repulsion_loss_weight
            else:
                gen_repulsion_loss = 0.0

        return gen_fidelity_loss, gen_repulsion_loss

    def build_graph(self, is_training=True, scope='generator'):
        self.train_writer = tf.contrib.summary.create_file_writer(os.path.join(MODEL_DIR, 'train'), flush_millis=10000)
        self.pointclouds_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3), name='pointclouds_input')
        self.pointclouds_radius = tf.placeholder(tf.float32, shape=(BATCH_SIZE), name='pointclouds_radius')
        self.model_up_ratio = tf.placeholder(tf.int32, shape=(), name="model_up_ratio")
        self.model_up_ratio_idx = tf.cast(tf.log(tf.cast(self.model_up_ratio, tf.float32)) / tf.log(tf.cast(STEP_RATIO, tf.float32)), tf.int32) - 1
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        tf.contrib.summary.scalar("learning_rate", self.learning_rate)

        # create the model depending on model_up_ratio
        if PHASE == "test":
            self.pred, _, self.input, self.gt, self.bradius = MODEL_GEN.get_gen_model(self.pointclouds_input, is_training,
                                    gt_point_cloud=None, scope=scope,
                                    up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
                                    max_up_ratio=UP_RATIO,
                                    bradius=self.pointclouds_radius,
                                    knn=FLAGS.knn, dense_n=FLAGS.dense_n, n_blocks=FLAGS.block_n,
                                    adaptive_receptive_field=ADAPTIVE_RECEPTIVE_FIELD,
                                    fm_knn=FM_KNN,
                                    reuse=None, use_normal=False, use_bn=False, use_ibn=False,
                                    bn_decay=None)
        else:
            with self.train_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(50):
                self.step = tf.train.get_global_step()
                self.pointclouds_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, 3), name='pointclouds_gt')
                self.pred, _, self.input, self.gt, self.bradius = MODEL_GEN.get_gen_model(self.pointclouds_input, is_training,
                                        gt_point_cloud=self.pointclouds_gt, scope=scope,
                                        up_ratio=self.model_up_ratio, step_ratio=STEP_RATIO,
                                        max_up_ratio=UP_RATIO,
                                        bradius=self.pointclouds_radius,
                                        knn=FLAGS.knn, dense_n=FLAGS.dense_n, n_blocks=FLAGS.block_n,
                                        fm_knn=FM_KNN,
                                        adaptive_receptive_field=ADAPTIVE_RECEPTIVE_FIELD,
                                        reuse=None, use_normal=False, use_bn=False, use_ibn=False,
                                        bn_decay=None)

                def compute_blend_value(i):
                    print("compute_blend_value %d" % i)
                    if i == 0:
                        blend = 1.0
                    else:
                        blend = tf.cast(self.step - STAGE_STEPS*(1+2*(i-1)), dtype=tf.float32)  # initial stage use 1.0 directly
                        blend = blend / (2*STAGE_STEPS)
                        blend = tf.clip_by_value(blend*2, 0, 1)
                    tf.contrib.summary.scalar("blend_%d" % i, blend)
                    return blend

                # get total loss function
                def add_loss_and_summary(i, blend):
                    cd_forward_weight = 1.0
                    # care about distance within this threshold because gt and input could mismatch
                    if i > 0:
                        threshold = tf.cond(blend > 0.6, lambda: tf.constant(FLAGS.cd_threshold), lambda: tf.constant(100.0))
                    else:
                        threshold = tf.constant(100.0)

                    cd_loss, repulsion_loss = self.get_loss(self.pred, self.gt, self.bradius,
                        fidelity_loss_weight=FLAGS.fidelity_weight, repulsion_loss_weight=FLAGS.repulsion_weight,
                        cd_forward_weight=cd_forward_weight, threshold=threshold)
                    tf.contrib.summary.scalar("loss/cd_loss_%d" % i, cd_loss)
                    tf.contrib.summary.scalar("loss/repulsion_loss_%d" % i, repulsion_loss)
                    # tf.contrib.summary.scalar("loss/anisotropic_loss_%d" % i, anisotropic_loss)
                    weight = (np.log(UP_RATIO) / np.log(STEP_RATIO) - i)
                    pc_loss = tf.add_n([cd_loss, repulsion_loss]) * weight
                    tf.contrib.summary.scalar("loss/pc_loss_%d" % i, pc_loss)
                    print("add_loss_and_summary %d, cd_forward_weight %.1f, weight %.2f" % (i, cd_forward_weight, weight))
                    return pc_loss

                cases = [(tf.equal(self.model_up_ratio_idx, i), lambda i=i: compute_blend_value(i)) for i in range(int(np.log(UP_RATIO) / np.log(STEP_RATIO)))]
                self.blend = tf.case(cases, name="blend", default=lambda: tf.constant(1.0))
                # make optimizer
                update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9)

                with tf.control_dependencies(update_ops):
                    def create_train_op(i):
                        cd_forward_weight = 1.0
                        # care about distance within this threshold because gt and input could mismatch
                        if i > 0:
                            threshold = tf.cond(self.blend > 0.6, lambda: tf.constant(2.0), lambda: tf.constant(100.0))
                        else:
                            threshold = tf.constant(100.0)

                        cd_loss, repulsion_loss = self.get_loss(self.pred, self.gt, self.bradius, cd_forward_weight=cd_forward_weight, threshold=threshold)
                        tf.contrib.summary.scalar("loss/cd_loss_%d" % i, cd_loss)
                        tf.contrib.summary.scalar("loss/repulsion_loss_%d" % i, repulsion_loss)
                        # tf.contrib.summary.scalar("loss/anisotropic_loss_%d" % i, anisotropic_loss)
                        weight = (np.log(UP_RATIO) / np.log(STEP_RATIO) - i)
                        pc_loss = tf.cond(self.blend > 0.4, lambda: tf.add_n([cd_loss, repulsion_loss]) * weight, lambda: cd_loss*weight)
                        # pc_loss = tf.add_n([cd_loss, repulsion_loss]) * weight
                        tf.contrib.summary.scalar("loss/pc_loss_%d" % i, pc_loss)
                        print("add_loss_and_summary %d, cd_forward_weight %.1f, weight %.2f" % (i, cd_forward_weight, weight))
                        combined = self.blend >= 1.0
                        var_all = []
                        for k in range(i+1):
                            var_all += [var for var in tf.trainable_variables() if ('level_%d' % (k+1)) in var.name]
                        gen_tvars_level = [var for var in tf.trainable_variables() if ('level_%d' % (i+1)) in var.name]

                        # gen_train = tf.cond(combined, lambda: compute_and_update_gradients(pc_loss, var_all), lambda: compute_and_update_gradients(pc_loss, gen_tvars_level))
                        grad_var_all = [(grad, var) for grad, var in optimizer.compute_gradients(pc_loss, var_all) if grad is not None]
                        grad_var_level = [(grad, var) for grad, var in optimizer.compute_gradients(pc_loss, gen_tvars_level) if grad is not None]

                        grad_var_all = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grad_var_all]
                        grad_var_level = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grad_var_level]

                        gen_train = tf.cond(combined,
                            lambda: optimizer.apply_gradients(grad_var_all, global_step=self.step),
                            lambda: optimizer.apply_gradients(grad_var_level, global_step=self.step))

                        return pc_loss, gen_train

                    cases = [(tf.equal(self.model_up_ratio_idx, i), lambda i=i: create_train_op(i)) for i in range(int(np.log(UP_RATIO) / np.log(STEP_RATIO)))]
                    self.pc_loss, self.gen_train = tf.case(cases)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.blend)
                    self.total_loss = self.pc_loss  # * self.blend  + self.regularization_loss
                    self.merged = tf.contrib.summary.all_summary_ops()

    def train(self,assign_model_path=None):
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        if TRAIN_H5 is not None:
            if NUM_SHAPE_POINT:

                input_data, gt_data, data_radius, _ = data_provider.load_patch_data(
                    h5_filename=TRAIN_H5, up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
                    num_point=NUM_SHAPE_POINT,  # roughly 1/4 of the complete model
                    norm=USE_DATA_NORM, use_randominput=USE_RANDOM_INPUT)
            else:

                input_data, gt_data, data_radius, _ = data_provider.load_patch_data(
                    h5_filename=TRAIN_H5, up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
                    num_point=NUM_SHAPE_POINT, patch_size=NUM_POINT,
                    norm=USE_DATA_NORM, use_randominput=USE_RANDOM_INPUT)

            self.fetchworker = data_provider.Fetcher(input_data, gt_data, data_radius, BATCH_SIZE,
                num_in_point=NUM_POINT, step_ratio=STEP_RATIO, up_ratio=UP_RATIO,
                jitter=JITTER, jitter_sigma=FLAGS.jitter_sigma, jitter_max=FLAGS.jitter_max,
                drop_out=FLAGS.drop_out)
        elif TRAIN_RECORD is not None:
            self.fetchworker = data_provider.Fetcher(
                TRAIN_RECORD, batch_size=BATCH_SIZE,
                step_ratio=STEP_RATIO, up_ratio=UP_RATIO, num_in_point=NUM_POINT, num_shape_point=NUM_SHAPE_POINT,
                jitter=JITTER, drop_out=FLAGS.drop_out, jitter_max=FLAGS.jitter_max, jitter_sigma=FLAGS.jitter_sigma
            )
        else:
            raise(ValueError)

        with tf.Session(config=config) as self.sess, self.train_writer.as_default():
            tf.global_variables_initializer().run()
            tf.contrib.summary.initialize(graph=tf.get_default_graph())

            ### assign the generator with another model file
            # restore the model
            self.saver = tf.train.Saver(max_to_keep=None)
            if assign_model_path is not None:
                logger.info("Load pre-train model from %s" % (assign_model_path), bold=True)
                # assign_saver = tf.train.Saver(
                #     var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
                # assign_saver.restore(self.sess, assign_model_path)
                # self.saver.restore(self.sess, assign_model_path)
                tf_util.optimistic_restore(self.sess, assign_model_path)
                self.restore_epoch = RESTORE_EPOCH or 0
                logger.info(("Resume training from %d epoch model from %s" % (self.restore_epoch, assign_model_path)), bold=True)
                if RESTORE_EPOCH is not None:
                    tf.assign(self.step, RESTORE_EPOCH*self.fetchworker.num_batches).eval()
            else:
                self.restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(MODEL_DIR)
                if checkpoint_path is not None:
                    # self.saver.restore(self.sess, checkpoint_path)
                    tf_util.optimistic_restore(self.sess, checkpoint_path)
                    try:
                        self.restore_epoch = RESTORE_EPOCH or int(self.step.eval() / self.fetchworker.num_batches)
                    except Exception:
                        self.restore_epoch = RESTORE_EPOCH or 0
                    logger.info(("Resume training from %d epoch model from %s" % (self.restore_epoch, checkpoint_path)), bold=True)

            global LOG_FOUT
            if self.restore_epoch == 0:
                LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'w')
                LOG_FOUT.write(str(socket.gethostname()) + '\n')
                LOG_FOUT.write(str(FLAGS) + '\n')
            else:
                LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'a')

            self.total_steps = self.fetchworker.num_batches * MAX_EPOCH
            self.stage_steps = STAGE_STEPS
            self.last_max_ratio = self.get_next_ratio(self.step.eval())
            self.last_is_combined = self.blend.eval(feed_dict={self.model_up_ratio: self.last_max_ratio}) >= 1.0
            self.fetchworker.initialize(self.sess, self.last_max_ratio, self.last_is_combined)

            for epoch in tqdm(list(range(self.restore_epoch, MAX_EPOCH + 1))):
                log_string('**** EPOCH %03d ****\t' % (epoch))
                self.train_one_epoch(epoch)
                if epoch % 10 == 0:
                    self.saver.save(self.sess, os.path.join(MODEL_DIR, "model"))
                    # self.eval_per_epoch(epoch, TEST_DATA)
            self.saver.save(self.sess, os.path.join(MODEL_DIR, "final"))

    def eval_per_epoch(self, epoch, input_folder):
        step = self.step.eval()
        max_ratio = self.get_next_ratio(step)
        ratio_idx = int(np.log2(max_ratio)/np.log2(STEP_RATIO)) - 1
        start = time.time()
        samples = glob(input_folder, recursive=True)
        samples.sort()
        for i in range(ratio_idx):
            ratio = STEP_RATIO**(i+1)
            save_path = os.path.join(MODEL_DIR, "eval", "epoch_%d" % epoch, 'knn_p%d_s%d_x%d' % (NUM_POINT, NUM_SHAPE_POINT, ratio))
            if len(samples)>50:
                samples = samples[:50]
            for point_path in samples:
                data = pc_util.load(point_path, count=NUM_SHAPE_POINT)
                data = data[:,0:3]
                ## get the edge information
                logger.info(os.path.basename(point_path))
                input_list, pred_list = self.pc_prediction(data, self.sess, ratio=ratio, patch_num_ratio=PATCH_NUM_RATIO)

                input_pc = np.concatenate(input_list, axis=0)
                pred_pc = np.concatenate(pred_list,axis=0)

                path = os.path.join(save_path, point_path.split('/')[-1][:-4]+'.ply')
                pc_util.save_ply(pred_pc, path[:-4]+'.ply')
                pc_util.save_ply(input_pc, path[:-4]+'_input.ply')
                # if len(input_list) > 1:
                #     counter = 0
                #     for in_p, pred_p in zip(input_list, pred_list):
                #         pc_util.save_ply(in_p, os.path.join(save_path, point_path.split('/')[-1][:-4]+"_input_patch_%d.ply" % counter))
                #         pc_util.save_ply(pred_p, os.path.join(save_path, point_path.split('/')[-1][:-4]+"_pred_patch_%d.ply" % counter))
                #         counter += 1
        end = time.time()
        logger.info("Evaluation time: %.2f" % (end-start))

    def train_one_epoch(self, epoch):
        loss_sum = []
        fetch_time = 0
        run_time = 0
        # self.last_max_ratio = self.get_next_ratio(self.step.eval())
        blend_val = self.blend.eval(feed_dict={self.model_up_ratio: self.last_max_ratio})
        # self.last_is_combined = (blend_val >= 1.0)
        for batch_idx in range(self.fetchworker.num_batches):
            try:
                step = self.step.eval()
                new_max_ratio = self.get_next_ratio(step)
                blend_val = self.blend.eval(feed_dict={self.model_up_ratio: new_max_ratio})
                new_is_combined = not ((blend_val < 1.0) or (new_max_ratio != self.last_max_ratio))
                lr_decay = 0.5 if new_is_combined else 1.0
                if new_max_ratio != self.last_max_ratio or new_is_combined != self.last_is_combined:
                    self.fetchworker.initialize(self.sess, new_max_ratio, new_is_combined)
                    logger.info("new upsample ratio: %d, new_is_combined: %s, blend: %.2f" % (
                        new_max_ratio, str(new_is_combined), blend_val), bold=True)
                    log_string("new upsample ratio: %d, new_is_combined: %s, blend: %.2f\n" % (
                        new_max_ratio, str(new_is_combined), blend_val))
                    self.saver.save(self.sess, os.path.join(MODEL_DIR, "x%d" % self.last_max_ratio), global_step=epoch)
                    self.last_max_ratio = new_max_ratio
                    self.last_is_combined = new_is_combined
                start = time.time()
                batch_input_data, batch_data_gt, radius, ratio = self.fetchworker.fetch(self.sess)
                end = time.time()
                fetch_time += end - start
                feed_dict = {self.pointclouds_input: batch_input_data,
                             self.pointclouds_gt: batch_data_gt[:, :, 0:3],
                             self.pointclouds_radius: radius,
                             self.model_up_ratio: ratio,
                             self.learning_rate: BASE_LEARNING_RATE*lr_decay}
                # at beginning only evaluate first output to save time
                fetch_list = [self.gen_train, self.pc_loss]

                if step % 50 == 0:  # visualize and fetch summary only every 50 epochs
                    fetch_list.extend([self.merged, self.input, self.pred, self.gt])
                    start = time.time()
                    _, pc_loss, summary, input_val, pred_val, gt_val = self.sess.run(fetch_list, feed_dict=feed_dict)
                    end = time.time()

                    # save training results
                    if step % 500 == 0:
                        os.makedirs(os.path.join(MODEL_DIR, "training_ply"), exist_ok=True)
                        if ratio > STEP_RATIO:
                            pc_util.save_ply(batch_input_data[0, :, 0:3],
                                os.path.join(MODEL_DIR, "training_ply", "step_%06d_x%d_initial_input.ply" % (step, ratio)))
                            pc_util.save_ply(batch_data_gt[0, :, 0:3],
                                os.path.join(MODEL_DIR, "training_ply", "step_%06d_x%d_initial_gt.ply" % (step, ratio)))
                        pc_util.save_ply(input_val[0, :, 0:3],
                            os.path.join(MODEL_DIR, "training_ply", "step_%06d_x%d_input.ply" % (step, ratio)))
                        pc_util.save_ply(pred_val[0, :, 0:3],
                            os.path.join(MODEL_DIR, "training_ply", "step_%06d_x%d_pred.ply" % (step, ratio)))
                        pc_util.save_ply(gt_val[0, :, 0:3],
                            os.path.join(MODEL_DIR, "training_ply", "step_%06d_x%d_gt.ply" % (step, ratio)))

                else:
                    start = time.time()
                    _, pc_loss = self.sess.run(fetch_list, feed_dict=feed_dict)
                    end = time.time()

                run_time += end - start
                loss_sum.append(pc_loss)

            except (ValueError, RuntimeError):
                logger.warn(traceback.print_exc())

        loss_sum = np.array(loss_sum)
        log_string('step: %d cd_loss: %.4f\n' % (step, loss_sum.mean()))
        logger.info('blend: %.2f, datatime/ep:%.4f runtime/ep:%.4f loss:%.4f' % (blend_val, fetch_time, run_time, loss_sum.mean()))

    def patch_prediction(self, patch_point, sess, ratio):
        # normalize the point clouds
        patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
        patch_point = np.expand_dims(patch_point,axis=0)
        pred = sess.run(self.pred,
            feed_dict={self.pointclouds_input: patch_point,
                       self.pointclouds_radius: np.ones(BATCH_SIZE, dtype=np.float32),
                       self.model_up_ratio: ratio})

        pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
        return pred

    def pc_prediction(self, pc, sess, patch_num_ratio=3, ratio=UP_RATIO):
        ## get patch seed from farthestsampling
        points = tf.convert_to_tensor(np.expand_dims(pc,axis=0),dtype=tf.float32)
        start= time.time()
        seed1_num = int(pc.shape[0] / NUM_POINT * patch_num_ratio)

        ## FPS sampling
        seed = farthest_point_sample(seed1_num, points).eval()[0]
        seed_list = seed[:seed1_num]
        print("farthest distance sampling cost", time.time() - start)
        print("number of patches: %d" % len(seed_list))
        input_list = []
        up_point_list=[]

        patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, NUM_POINT)
        for point in tqdm(patches, total=len(patches)):
            up_point = self.patch_prediction(point, sess, ratio)
            input_list.append(point)
            up_point_list.append(up_point)

        return input_list, up_point_list

    def test_hierarical_prediction(self, input_folder=None, save_path=None):
        _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
        logger.info(restore_model_path, bold=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            tf_util.optimistic_restore(sess, restore_model_path)
            total_time = 0
            samples = glob(input_folder, recursive=True)
            samples.sort()
            # if len(samples)>100:
            #     samples = samples[:100]
            for point_path in samples:
                start = time.time()
                data = pc_util.load(point_path, count=NUM_SHAPE_POINT)
                num_shape_point = data.shape[0]
                data = data[:,0:3]
                is_2D = np.all(data[:,2] == 0)
                data, centroid, furthest_distance = pc_util.normalize_point_cloud(data)
                if FLAGS.drop_out < 1:
                    idx = farthest_point_sample(int(num_shape_point*FLAGS.drop_out), data[np.newaxis,...]).eval()[0]
                    data = data[idx, 0:3]
                if JITTER:
                    data = pc_util.jitter_perturbation_point_cloud(data[np.newaxis,...], sigma=FLAGS.jitter_sigma, clip=FLAGS.jitter_max, is_2D=is_2D)
                    data = data[0, ...]
                ## get the edge information
                logger.info(os.path.basename(point_path))
                input_list, pred_list = self.pc_prediction(data, sess, patch_num_ratio=PATCH_NUM_RATIO)
                end = time.time()
                print("total time: ",end-start)
                pred_pc = np.concatenate(pred_list,axis=0)
                pred_pc = (pred_pc * furthest_distance) + centroid
                data = (data * furthest_distance) + centroid
                folder = os.path.basename(os.path.dirname(point_path))
                path = os.path.join(save_path, folder, point_path.split('/')[-1][:-4]+'.ply')
                # pc_util.save_ply(pred_pc, path[:-4]+'_overlapped.ply')
                pc_util.save_ply(data, path[:-4]+'_input.ply')
                idx = farthest_point_sample(int(num_shape_point*FLAGS.drop_out)*UP_RATIO, pred_pc[np.newaxis,...]).eval()[0]
                pred_pc = pred_pc[idx, 0:3]
                # pred_pc, _, _ = pc_util.normalize_point_cloud(pred_pc)
                # pred_pc = (pred_pc * furthest_distance) + centroid
                pc_util.save_ply(pred_pc, path[:-4]+'.ply')

                # if len(input_list) > 1:
                #     counter = 0
                #     for in_p, pred_p in zip(input_list, pred_list):
                #         pc_util.save_ply(in_p*furthest_distance+centroid, path[:-4]+"_input_patch_%d.ply" % counter)
                #         pc_util.save_ply(pred_p*furthest_distance+centroid, path[:-4]+"_pred_patch_%d.ply" % counter)
                #         counter += 1

            print(total_time/len(samples))

    def visualize(self, input_folder=None, save_path=None):
        _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
        logger.info(restore_model_path, bold=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            tf_util.optimistic_restore(sess, restore_model_path)
            samples = glob(input_folder, recursive=True)
            samples.sort()
            if len(samples)>100:
                samples = samples[:100]
            for point_path in samples:
                start = time.time()
                data = pc_util.load(point_path, count=NUM_SHAPE_POINT)
                num_shape_point = data.shape[0]
                data = data[:,0:3]
                is_2D = np.all(data[:,2] == 0)
                data, centroid, furthest_distance = pc_util.normalize_point_cloud(data)
                if FLAGS.drop_out < 1:
                    idx = farthest_point_sample(int(num_shape_point*FLAGS.drop_out), data[np.newaxis,...]).eval()[0]
                    data = data[idx, 0:3]
                if JITTER:
                    data = pc_util.jitter_perturbation_point_cloud(data[np.newaxis,...], sigma=FLAGS.jitter_sigma, clip=FLAGS.jitter_max, is_2D=is_2D)
                    data = data[0, ...]
                ## get the edge information
                logger.info(os.path.basename(point_path))
                mid = data[(np.abs(data[:, 2]) < np.amax(data[:, 2])*0.2), :]
                idx = farthest_point_sample(5, mid[np.newaxis, ...]).eval()[0]
                # idx = np.random.choice(data.shape[0], 5, replace=False)
                patches = pc_util.extract_knn_patch(mid[idx, :], data, NUM_POINT)
                end = time.time()
                print("total time: ",end-start)
                path = os.path.join(save_path, point_path.split('/')[-1][:-4]+'.ply')
                total_levels = int(np.log2(UP_RATIO)/np.log2(STEP_RATIO))
                for p in range(patches.shape[0]):
                    patch = patches[p]
                    for i in range(1, total_levels+1):
                        patch_result = self.patch_prediction(patch, sess, STEP_RATIO**i)
                        pc_util.save_ply((patch_result*furthest_distance)+centroid, path[:-4]+"_p_%d_%d.ply" % (p, i))
                    pc_util.save_ply((patch*furthest_distance)+centroid, path[:-4]+"_p_%d_%d.ply" % (p, 0))
                pc_util.save_ply((data*furthest_distance)+centroid, path[:-4]+"_input.ply")

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


if __name__ == "__main__":
    np.random.seed(240)
    tf.set_random_seed(240)
    input_folder = os.path.join(ROOT_DIR, TEST_DATA)
    # sub_dir = os.path.relpath(os.path.dirname(input_folder), os.path.join(ROOT_DIR, "data", "test_data"))
    append_name = []
    if NUM_POINT is None:
        append_name += ["pWhole"]
    else:
        append_name += ["p%d" % NUM_POINT]
    if NUM_SHAPE_POINT is None:
        append_name += ["sWhole"]
    else:
        append_name += ["s%d" % NUM_SHAPE_POINT]
    if JITTER:
        append_name += ["s{}".format("{:.4f}".format(FLAGS.jitter_sigma).replace(".", ""))]
    else:
        append_name += ["clean"]
    if FLAGS.drop_out < 1:
        append_name += ["d{}".format("{:.2f}".format(FLAGS.drop_out).replace(".", ""))]

    result_path = FLAGS.result_dir or os.path.join(MODEL_DIR, 'result', 'x%d' % (UP_RATIO), "_".join(append_name))
    if PHASE == 'train':
        # copy the code
        code_dir = os.path.join(MODEL_DIR, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(code_dir)
        os.system('cp -r * %s' % code_dir)  # bkp of model def
        network = Network()
        network.build_graph(is_training=True)
        network.train(ASSIGN_MODEL_PATH)
        del network
        tf.reset_default_graph()
        network = Network()
        BATCH_SIZE = 1
        network.build_graph(is_training=False)
        logger.info("saving to {}".format(result_path))
        network.test_hierarical_prediction(input_folder=input_folder, save_path=result_path)
        LOG_FOUT.close()

    elif PHASE == "test":
        network = Network()
        BATCH_SIZE = 1
        network.build_graph(is_training=False)
        logger.info("saving to {}".format(result_path))
        network.test_hierarical_prediction(input_folder=input_folder, save_path=result_path)
