import argparse
from glob import glob

import tensorflow as tf

from model_py3_tfrecord import denoiser
from utils_py3_tfrecord import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu', dest='num_gpu', type=str, default="0", help='choose which gpu')
parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--test_set', dest='test_set', default='KodakTrueColor', help='dataset for testing')
args = parser.parse_args()

def denoiser_test(denoiser):
    test_files_gt = glob('./data/CDM/{}/groundtruth/*'.format(args.test_set))
    test_files_bl = glob('./data/CDM/{}/bilinear/*'.format(args.test_set))
    denoiser.test(test_files_gt, test_files_bl, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def main(_):
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    numPatches = 0

    if args.use_gpu:
        print("GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
        gpu_options = tf.GPUOptions(allow_growth = True)#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95) #
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Session() as sess:
            model = denoiser(sess)
            if args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
