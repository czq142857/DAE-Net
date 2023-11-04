import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--shapes_per_epoch", action="store", dest="shapes_per_epoch", default=2500*5, type=int, help="Control the number of shapes to train in each epoch, regardless of the number of shapes in the training set.")
parser.add_argument("--resolution", action="store", dest="resolution", default=32, type=int, help="The resolution of voxels where the training points are sampled from. 16, 32, or 64.")
parser.add_argument("--branch_num", action="store", dest="branch_num", default=16, type=int, help="Number of part templates, i.e., MLP branches")
parser.add_argument("--z_dim", action="store", dest="z_dim", default=4, type=int, help="Part deformation MLP input per-shape feature size")
parser.add_argument("--ef_dim", action="store", dest="ef_dim", default=16, type=int, help="CNN shape encoder hidden layer size")
parser.add_argument("--df_dim", action="store", dest="df_dim", default=64, type=int, help="Part deformation MLP hidden layer size")
parser.add_argument("--gf_dim", action="store", dest="gf_dim", default=64, type=int, help="Part template neural implicit MLP hidden layer size")
parser.add_argument("--occupancy_loss_weight", action="store", dest="occupancy_loss_multiplier", default=0.1, type=float, help="Weight of the loss to encourage binary occupancy")
parser.add_argument("--sparse_loss_weight", action="store", dest="sparse_loss_multiplier", default=0.01, type=float, help="Weight of the regularization loss for part sparsity")
parser.add_argument("--affine_loss_weight", action="store", dest="affine_loss_multiplier", default=0.0, type=float, help="Weight of the regularization loss for shape-specific part affine transformation")
parser.add_argument("--deform_loss_weight", action="store", dest="deform_loss_multiplier", default=100.0, type=float, help="Weight of the regularization loss for shape-specific part deformation")

parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0002, type=float, help="Learning rate for adam")
parser.add_argument("--shape_batch_size", action="store", dest="shape_batch_size", default=16, type=int, help="Number of training shapes per mini-batch")

parser.add_argument("--data_dir", action="store", dest="data_dir", default="../bae_net_data/03001627_chair/", help="Root directory of dataset")
parser.add_argument("--data_file", action="store", dest="data_file", default="03001627_vox", help="Which split to use")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the samples")

parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training")
parser.add_argument("--test", action="store_true", dest="test", default=False, help="True for testing, will output colored segmented reconstructions for shapes specified in data_file")
parser.add_argument("--template", action="store_true", dest="template", default=False, help="True for visualizing learned part templates")
parser.add_argument("--iou", action="store_true", dest="iou", default=False, help="True for computing segmentation IOU")
parser.add_argument("--cluster", action="store_true", dest="cluster", default=False, help="True for getting shape clusters based on part existence scores")

parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="Which GPU to use")
FLAGS = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


import dataset
from model import DAE_NET

import torch

if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)

dae_net = DAE_NET(FLAGS)

if FLAGS.train:
	dae_dataset = dataset.hdf5_dataset(FLAGS.data_dir, FLAGS.data_file, FLAGS.resolution, FLAGS.shapes_per_epoch)
	dae_dataloader = torch.utils.data.DataLoader(dae_dataset, batch_size=FLAGS.shape_batch_size, shuffle=False, num_workers=8) #shuffle disabled
	dae_net.train(FLAGS,dae_dataloader)

elif FLAGS.test:
	dae_net.test(FLAGS)

elif FLAGS.template:
	dae_net.template(FLAGS)

elif FLAGS.iou:
	dae_net.iou(FLAGS)

elif FLAGS.cluster:
	dae_dataset = dataset.hdf5_dataset(FLAGS.data_dir, FLAGS.data_file, FLAGS.resolution)
	dae_dataloader = torch.utils.data.DataLoader(dae_dataset, batch_size=1, shuffle=False, num_workers=8)
	dae_net.cluster(FLAGS,dae_dataloader)
