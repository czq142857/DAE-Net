import os
import time
import numpy as np
import h5py
import cv2
from skimage import measure

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

#colors of different branches (segmented parts)
color_list = [
	"255 0 0",     "0 255 0",     "0 0 255",     "255 255 0",   "255 0 255",   "0 255 255",   "32 32 32",    "160 160 160",
	"255 128 128", "128 255 128", "128 128 255", "255 255 128", "255 128 255", "128 255 255", "96 96 96",    "224 224 224",
	"255 64 64",   "64 255 64",   "64 64 255",   "255 255 64",  "255 64 255",  "64 255 255",  "64 64 64",    "192 192 192",
	"255 192 192", "192 255 192", "192 192 255", "255 255 192", "255 192 255", "192 255 255", "128 128 128", "255 255 255",
]

class MLP_G(nn.Module):
	def __init__(self, input_dim, gf_dim, branch_num):
		super(MLP_G, self).__init__()
		self.input_dim = input_dim
		self.df_dim = gf_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.linear_d1 = nn.Conv1d(self.input_dim*self.branch_num, self.df_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_d2 = nn.Conv1d(self.df_dim*self.branch_num,                 self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_d3 = nn.Conv1d(self.df_dim*self.branch_num,                 self.input_dim*self.branch_num, 1, groups=self.branch_num, bias=True)

		self.linear_1 = nn.Conv1d(self.input_dim*self.branch_num, self.gf_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_2 = nn.Conv1d(self.gf_dim*self.branch_num,    self.gf_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_3 = nn.Conv1d(self.gf_dim*self.branch_num,    self.branch_num,             1, groups=self.branch_num, bias=True)
		#nn.Conv1d(A*B, C*B, 1, groups=B)
		#weight [C*B, A, 1]
		#bias [C*B]

	def init(self, branch_id):
		nn.init.xavier_uniform_(self.linear_d1.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_d1.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_d2.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_d2.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_d3.weight[branch_id*self.input_dim:(branch_id+1)*self.input_dim])
		nn.init.constant_(self.linear_d3.bias[branch_id*self.input_dim:(branch_id+1)*self.input_dim],0)

		nn.init.normal_(self.linear_1.weight[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], mean=0.0, std= (2/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_1.bias[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], 0)
		nn.init.normal_(self.linear_2.weight[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], mean=0.0, std= (2/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_2.bias[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], 0)
		nn.init.constant_(self.linear_3.weight[branch_id:(branch_id+1)], -(3.14*4/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_3.bias[branch_id:(branch_id+1)], 0)

	def forward(self, points):
		#points [N, 3*B, P]

		out = self.linear_d1(points)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_d2(out)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_d3(out) + points

		out = self.linear_1(out)
		out = F.relu(out)

		out = self.linear_2(out)
		out = F.relu(out)

		out = self.linear_3(out)
		out = torch.sigmoid(out)

		return out

class MLP_D(nn.Module):
	def __init__(self, input_dim, z_dim, df_dim, branch_num):
		super(MLP_D, self).__init__()
		self.input_dim = input_dim
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.branch_num = branch_num

		self.linear_1 = nn.Conv1d((self.input_dim+self.z_dim)*self.branch_num, self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_2 = nn.Conv1d(self.df_dim*self.branch_num,                 self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_3 = nn.Conv1d(self.df_dim*self.branch_num,                 self.input_dim*self.branch_num, 1, groups=self.branch_num, bias=True)

	def init(self, branch_id):
		nn.init.xavier_uniform_(self.linear_1.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_1.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_2.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_2.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_3.weight[branch_id*self.input_dim:(branch_id+1)*self.input_dim])
		nn.init.constant_(self.linear_3.bias[branch_id*self.input_dim:(branch_id+1)*self.input_dim],0)

	def forward(self, zs, points):
		#points [N, P, 3]
		affine_mat = zs[:,:,self.z_dim:self.z_dim+12]-0.5 #[N, B, 12]
		zs = zs[:,:,:self.z_dim] #[N, B, Z]

		N,P,_ = points.size()
		points = points.permute(0,2,1) #[N, 3, P]
		pointsx = torch.cat([points,torch.ones(N,1,P,device=points.device)],1) #[N, 4, P]
		ps = []
		pzs = []
		for i in range(self.branch_num):
			ps.append(affine_mat[:,i].view(N,3,4)@pointsx+points) #[N, 3, P]
			pzs.append(ps[i]) #[N, 3, P]
			pzs.append(zs[:,i].view(N,self.z_dim,1).repeat(1,1,P)) #[N, Z, P]
		ps = torch.cat(ps,1) #[N, 3*B, P]
		pzs = torch.cat(pzs,1) #[N, (3+Z)*B, P]

		out = self.linear_1(pzs)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_2(out)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		offsets = self.linear_3(out) #[N, 3*B, P]

		return affine_mat, ps, offsets

class generator(nn.Module):
	def __init__(self, z_dim, df_dim, gf_dim, branch_num):
		super(generator, self).__init__()
		self.point_dim = 3
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.deformers = MLP_D(self.point_dim,self.z_dim,self.df_dim,self.branch_num)
		self.generators = MLP_G(self.point_dim,self.gf_dim,self.branch_num)

	def init(self, branch_id):
		self.deformers.init(branch_id)
		self.generators.init(branch_id)

	def forward(self, points, zs, matrices, out_sum=False, out_branch=False, affine_only=False):
		if out_branch and not out_sum:
			#points [N,P,3]
			out = self.generators(points.permute(0,2,1).repeat(1,self.branch_num,1)) #[N,B,P]
			out = out.permute(0,2,1) #[N,P,B]
			return out

		#points [N,P,3]
		#zs [N,B,Z]
		#matrices [N,B,1]
		N,P,_ = points.size()
		affine_mat, affine_points, deformed_offsets = self.deformers(zs,points) #[N, 3*B, P]
		if affine_only:
			deformed_points = affine_points
		else:
			deformed_points = affine_points + deformed_offsets
		out = self.generators(deformed_points) #[N,B,P]
		deformed_offsets = deformed_offsets.view(N, self.branch_num, 3, P) * out.detach().view(N, self.branch_num, 1, P) #mask out empty space
		out = out.permute(0,2,1) #[N,P,B]

		if out_branch:
			return out

		matrices = matrices.view(N,1,self.branch_num).repeat(1,P,1) #[N,P,B]
		out = out*matrices
		out_sum = torch.sum(out,-1,keepdim=True)
		out_max = torch.max(out,-1,keepdim=True)[0]

		return out_sum, out_max, affine_mat, deformed_offsets

class encoder(nn.Module):
	def __init__(self, z_dim, ef_dim, branch_num):
		super(encoder, self).__init__()
		self.z_dim = z_dim + 12 # + affine transform matrix
		self.ef_dim = ef_dim
		self.branch_num = branch_num

		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.norm_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.norm_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.norm_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.norm_4 = nn.InstanceNorm3d(self.ef_dim*8)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*16, 4, stride=1, padding=0, bias=True)

		self.linear_1 = nn.Linear(self.ef_dim*16, self.branch_num, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.branch_num*self.z_dim, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)

	def init(self, branch_id):
		nn.init.normal_(self.linear_1.weight[branch_id:(branch_id+1)], mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias[branch_id:(branch_id+1)], 0)
		nn.init.normal_(self.linear_2.weight[branch_id*self.z_dim:(branch_id+1)*self.z_dim], mean=0.0, std=0.02)
		nn.init.normal_(self.linear_2.bias[branch_id*self.z_dim:(branch_id+1)*self.z_dim], mean=0.0, std=0.1)
		#only scale and translate when init
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+1] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+2] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+4] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+6] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+8] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+9] = 0

	def forward(self, inputs):
		out = inputs.float()
		out = F.leaky_relu(self.norm_1(self.conv_1(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_2(self.conv_2(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_3(self.conv_3(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_4(self.conv_4(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.conv_5(out), negative_slope=0.02, inplace=True)
		out = out.view(-1, self.ef_dim*16)

		out1 = self.linear_1(out)
		out1 = out1.view(-1, self.branch_num, 1)
		out1 = torch.sigmoid(out1-8)

		out2 = self.linear_2(out)
		out2 = out2.view(-1, self.branch_num, self.z_dim)
		out2 = torch.sigmoid(out2)

		return out1, out2


class dae_network(nn.Module):
	def __init__(self, z_dim, ef_dim, df_dim, gf_dim, branch_num):
		super(dae_network, self).__init__()
		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.encoder = encoder(self.z_dim, self.ef_dim, self.branch_num)
		self.generator = generator(self.z_dim, self.df_dim, self.gf_dim, self.branch_num)



class DAE_NET(object):
	def __init__(self, config):
		self.branch_num = config.branch_num
		self.z_dim = config.z_dim
		self.ef_dim = config.ef_dim
		self.df_dim = config.df_dim
		self.gf_dim = config.gf_dim

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			print("ERROR: GPU not available!!!")
			exit(-1)

		#build model
		self.dae_network = dae_network(self.z_dim, self.ef_dim, self.df_dim, self.gf_dim, self.branch_num)
		self.dae_network.to(self.device)

		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 40
		self.checkpoint_path = os.path.join("checkpoint", config.data_file.split('_')[0])
		self.checkpoint_name='DAE_NET.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0

		#for testing
		self.cell_grid_size = 4
		self.frame_grid_size = 32
		self.output_size = self.cell_grid_size*self.frame_grid_size #=128, output voxel grid size in testing
		self.test_size = 64 #related to testing batch_size, adjust according to gpu memory size
		self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change

		#get coords for testing
		cell_t = np.linspace(0, self.cell_grid_size-1, self.cell_grid_size, dtype = np.int32)
		self.cell_x, self.cell_y, self.cell_z = np.meshgrid(cell_t,cell_t,cell_t, sparse=False, indexing='ij')
		frame_t = np.linspace(0, self.frame_grid_size-1, self.frame_grid_size, dtype = np.int32)
		self.frame_x, self.frame_y, self.frame_z = np.meshgrid(frame_t,frame_t,frame_t, sparse=False, indexing='ij')
		self.cell_x = np.reshape(self.cell_x,[-1]).astype(np.int32)
		self.cell_y = np.reshape(self.cell_y,[-1]).astype(np.int32)
		self.cell_z = np.reshape(self.cell_z,[-1]).astype(np.int32)
		self.frame_x = np.reshape(self.frame_x,[-1]).astype(np.int32)
		self.frame_y = np.reshape(self.frame_y,[-1]).astype(np.int32)
		self.frame_z = np.reshape(self.frame_z,[-1]).astype(np.int32)
		self.frame_coords = np.concatenate([self.frame_x[:,None],self.frame_y[:,None],self.frame_z[:,None]],1)
		self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/self.frame_grid_size-0.5
		self.cell_coords = np.zeros([self.frame_grid_size,self.frame_grid_size,self.frame_grid_size,self.cell_grid_size*self.cell_grid_size*self.cell_grid_size,3],np.int32)
		for i in range(self.frame_grid_size):
			for j in range(self.frame_grid_size):
				for k in range(self.frame_grid_size):
					self.cell_coords[i,j,k,:,0] = self.cell_x+i*self.cell_grid_size
					self.cell_coords[i,j,k,:,1] = self.cell_y+j*self.cell_grid_size
					self.cell_coords[i,j,k,:,2] = self.cell_z+k*self.cell_grid_size
		self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.output_size-0.5



	def train(self, config, dataloader_train):

		total_training_epochs = 20 * self.branch_num
		revive_period = 10
		start_reg_epoch = 1
		real_data_len = dataloader_train.dataset.data_len
		alive_threshold = real_data_len*0.1
		branch_age = np.zeros([self.branch_num],np.int32)

		start_time = time.time()
		epoch = 0
		previous_saved_epoch = 0
		previous_avg_iou = 0
		while epoch < total_training_epochs:

			if epoch%revive_period==0:
				#age
				branch_age[:] = branch_age[:]+1

				#compute live dead
				vitality = np.zeros([self.branch_num],np.float32)
				counter = 0
				if epoch>0:
					with torch.no_grad():
						self.dae_network.eval()
						for idx, data in enumerate(dataloader_train, 0):
							points_, values_, voxels_ = data
							voxels = voxels_.to(self.device)
							t_vector, d_vector = self.dae_network.encoder(voxels)
							t_vector = t_vector.detach().cpu().numpy()
							if counter+len(t_vector)>=real_data_len:
								vitality = vitality + np.sum(t_vector[:real_data_len-counter,:,0],0)
								break
							else:
								vitality = vitality + np.sum(t_vector[:,:,0],0)
								counter += len(t_vector)

				#revive dead
				counter = 0
				for cid in range(self.branch_num):
					if vitality[cid]<alive_threshold:
						counter += 1
						self.dae_network.encoder.init(cid)
						self.dae_network.generator.init(cid)
						branch_age[cid] = 0
				print("revived dead:", counter)

				#randomly revive live
				cid = np.argmax(branch_age)
				if branch_age[cid]>0:
					counter += 1
					self.dae_network.encoder.init(cid)
					self.dae_network.generator.init(cid)
					branch_age[cid] = 0

				#get a new optimizer
				self.optimizer = torch.optim.Adam(self.dae_network.parameters(), lr=config.learning_rate)


			self.dae_network.train()
			avg_loss_s1 = 0 #reconstruction loss
			avg_iou = 0 #reconstruction IOU
			avg_num = 0

			for idx, data in enumerate(dataloader_train, 0):

				points_, values_, voxels_ = data
				points = points_.to(self.device)
				values = values_.to(self.device)
				voxels = voxels_.to(self.device)

				self.dae_network.zero_grad()

				t_vector, d_vector = self.dae_network.encoder(voxels)

				points_out_sum, points_out_max, affine_mat, deformed_offsets = self.dae_network.generator(points, d_vector, t_vector, out_sum=True)

				errS1 = torch.mean( torch.where(values>0.5,1-points_out_sum,torch.clamp(points_out_sum,min=0.1))**2 )
				errS2 = torch.mean( torch.where(values>0.5,1-points_out_max,torch.clamp(points_out_max,min=0.1))**2 ) * config.occupancy_loss_multiplier

				with torch.no_grad():
					points_out_bin = (points_out_max>0.5).float()
					intersection = torch.sum(points_out_bin*values)
					iou = intersection/(torch.sum(points_out_bin)+torch.sum(values)-intersection)

				exp_weight = float(epoch%revive_period)/(revive_period-1)
				exp_weight = np.exp( np.log(1)*exp_weight + np.log(0.001)*(1-exp_weight) )

				errS3 = 0
				errS4 = 0
				errS5 = 0

				if config.sparse_loss_multiplier>0:
					errS3 = torch.mean( -(torch.mean(t_vector,0)-1)**2 ) * (config.sparse_loss_multiplier*exp_weight)

				if config.affine_loss_multiplier>0:
					errS4 = torch.mean( affine_mat**2 ) * (config.affine_loss_multiplier*exp_weight)

				if config.deform_loss_multiplier>0:
					errS5 = torch.mean( deformed_offsets**2 ) * (config.deform_loss_multiplier*exp_weight)

				loss = errS1 + errS2 + errS3 + errS4 + errS5

				loss.backward()
				self.optimizer.step()

				avg_loss_s1 += errS1.item()
				avg_iou += iou.item()
				avg_num += 1


			if epoch%revive_period==revive_period-1:
				avg_iou = avg_iou/avg_num
				print("epoch: %d, time: %4.4f, recon_iou: %.6f, loss: %.6f" % (epoch, time.time() - start_time, avg_iou, avg_loss_s1/avg_num))

				if (1-avg_iou)<(1-previous_avg_iou)*1.05:
					previous_saved_epoch = epoch
					previous_avg_iou = avg_iou

					#------ save checkpoint ------
					if not os.path.exists(self.checkpoint_path):
						os.makedirs(self.checkpoint_path)
					#save checkpoint
					save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
					torch.save(self.dae_network.state_dict(), save_dir)
					#delete checkpoint
					self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
					if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
						if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
							os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
					#update checkpoint manager
					self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
					#write file
					checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
					with open(checkpoint_txt, 'w') as fout:
						for i in range(self.max_to_keep):
							pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
							if self.checkpoint_manager_list[pointer] is not None:
								fout.write(self.checkpoint_manager_list[pointer]+"\n")
					print(" [*] Save checkpoint", epoch)

					#------ save samples ------
					with torch.no_grad():
						self.dae_network.eval()
						counter = 0
						for idx, data in enumerate(dataloader_train, 0):
							points_, values_, voxels_ = data
							for t in range(len(voxels_)):
								self.voxel2seg(voxels_[t:t+1].to(self.device), config.sample_dir+"/"+str(epoch)+"_"+str(counter)+".ply")
								counter += 1
								if counter>=4: break
							if counter>=4: break

				else:
					#------ save samples ------
					with torch.no_grad():
						self.dae_network.eval()
						counter = 0
						for idx, data in enumerate(dataloader_train, 0):
							points_, values_, voxels_ = data
							for t in range(len(voxels_)):
								self.voxel2seg(voxels_[t:t+1].to(self.device), config.sample_dir+"/"+str(epoch)+"_"+str(counter)+"_discarded"+".ply")
								counter += 1
								if counter>=4: break
							if counter>=4: break

					#------ revert checkpoint ------
					#load checkpoint
					model_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(previous_saved_epoch)+".pth")
					self.dae_network.load_state_dict(torch.load(model_dir))
					print(" [*] Revert checkpoint", previous_saved_epoch)

			epoch += 1


	def test(self, config):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			with open(checkpoint_txt) as fin:
				model_dir = fin.readline().strip()
			self.dae_network.load_state_dict(torch.load(model_dir))
			start_epoch = int(model_dir.split('-')[-1].split('.')[0])+1
			print(" [*] Load SUCCESS", start_epoch-1)
		else:
			print(" [!] Load failed")
			exit(-1)

		from sklearn.neighbors import KDTree

		with torch.no_grad():
			self.dae_network.eval()

			data_hdf5_name = config.data_dir+'/'+config.data_file+'.hdf5'
			data_txt_name = config.data_dir+'/'+config.data_file+'.txt'

			if os.path.exists(config.data_dir+'/'+"points"):
				output_GT = True
			else:
				output_GT = False

			#load input voxels
			data_dict = h5py.File(data_hdf5_name, 'r')
			data_voxels = data_dict['voxels'][:,:,:,:,0]
			data_dict.close()

			if output_GT:
				#load point clouds and labels
				gt_seg_points, gt_seg_labels, gt_seg_point_num, gt_seg_part_num, gt_seg_obj_names = utils.parse_txt_list(data_txt_name, config.data_dir+'/'+"points")

			for idx in range(len(data_voxels)):

				if output_GT:
					seg_point_num = gt_seg_point_num[idx]
					seg_points = gt_seg_points[idx,:seg_point_num]
					seg_labels = gt_seg_labels[idx,:seg_point_num]

				voxel = data_voxels[idx:idx+1,None,...].astype(np.float32)
				voxel_ = torch.from_numpy(voxel).to(self.device)

				self.voxel2seg(voxel_, config.sample_dir+"/"+str(idx)+"_affine_transformed.ply", affine_only=True)
				self.voxel2seg(voxel_, config.sample_dir+"/"+str(idx)+"_deformed.ply", affine_only=False)


				#ground truth
				points_out = voxel_[0,0].detach().cpu().numpy()
				verts, faces, _, _ = measure.marching_cubes(0.5-points_out[:,:,:], 0)
				verts = (verts+0.5)/64-0.5

				if output_GT:
					kd_tree = KDTree(seg_points, leaf_size=8)
					_, closest_idx = kd_tree.query(verts)
					labels = seg_labels[np.reshape(closest_idx,[-1])]
					colors = []
					for i in range(len(verts)):
						colors.append( [int(j) for j in color_list[labels[i]].split()] )
					colors = np.array(colors, np.int32)
					utils.write_ply_triangle_color(config.sample_dir+"/"+str(idx)+"_ground_truth.ply", verts, colors, faces)
				else:
					utils.write_ply_triangle(config.sample_dir+"/"+str(idx)+"_ground_truth.ply", verts, faces)


				#segmentation results
				t_vector, d_vector = self.dae_network.encoder(voxel_)
				t_vector = t_vector.detach().cpu().numpy() #[1,branch_num,1]
				t_vector = np.reshape(t_vector, [1,self.branch_num])

				seg_points = verts.astype(np.float32)
				seg_points_ = torch.from_numpy(seg_points[None,...]).to(self.device)
				pred_values = self.dae_network.generator(seg_points_, d_vector, None, out_sum=True, out_branch=True)
				pred_values = pred_values.detach().cpu().numpy()
				pred_values = np.reshape(pred_values, [len(verts),self.branch_num])
				pred_values = pred_values * t_vector

				pred_branch_labels = np.argmax(pred_values, 1)

				try:
					valid_labels = np.max(pred_values, 1)>0.4
					valid_seg_points = seg_points[valid_labels]
					valid_pred_branch_labels = pred_branch_labels[valid_labels]
					kd_tree = KDTree(valid_seg_points, leaf_size=8)
					_, closest_idx = kd_tree.query(seg_points)
					pred_branch_labels = valid_pred_branch_labels[np.reshape(closest_idx,[-1])]
				except:
					pass

				colors = []
				for i in range(len(verts)):
					colors.append( [int(j) for j in color_list[pred_branch_labels[i]].split()] )
				colors = np.array(colors, np.int32)

				utils.write_ply_triangle_color(config.sample_dir+"/"+str(idx)+"_ours_segmentation_on_GT.ply", verts, colors, faces)

				print(idx)


	def template(self, config):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			with open(checkpoint_txt) as fin:
				model_dir = fin.readline().strip()
			self.dae_network.load_state_dict(torch.load(model_dir))
			start_epoch = int(model_dir.split('-')[-1].split('.')[0])+1
			print(" [*] Load SUCCESS", start_epoch-1)
		else:
			print(" [!] Load failed")
			exit(-1)

		with torch.no_grad():
			self.dae_network.eval()

			pred_voxel = np.zeros([self.output_size+2,self.output_size+2,self.output_size+2,self.branch_num],np.float32)
			queue = []
			for i in range(self.frame_grid_size):
				for j in range(self.frame_grid_size):
					for k in range(self.frame_grid_size):
						queue.append((i,j,k))

			cell_batch_size = self.cell_grid_size**3
			cell_batch_num = int(self.test_point_batch_size/cell_batch_size)

			while len(queue)>0:
				batch_num = min(len(queue),cell_batch_num)
				point_list = []
				points = []
				for i in range(batch_num):
					point = queue.pop(0)
					point_list.append(point)
					points.append(self.cell_coords[point[0],point[1],point[2]])
				points = np.concatenate(points, axis=0)*2 #shrink parts by 2, avoid clamp
				points = torch.from_numpy(points[None,...]).to(self.device)
				pred_values = self.dae_network.generator(points,None,None,out_branch=True)
				pred_values = pred_values.detach().cpu().numpy()
				pred_values = np.reshape(pred_values, [-1,self.branch_num])
				for i in range(batch_num):
					point = point_list[i]
					values = pred_values[i*cell_batch_size:(i+1)*cell_batch_size]
					x_coords = self.cell_x+point[0]*self.cell_grid_size
					y_coords = self.cell_y+point[1]*self.cell_grid_size
					z_coords = self.cell_z+point[2]*self.cell_grid_size
					pred_voxel[x_coords+1,y_coords+1,z_coords+1] = values

			for cid in range(self.branch_num):
				try:
					verts, faces, _, _ = measure.marching_cubes(0.5-pred_voxel[:,:,:,cid], 0)
					verts = (verts-0.5)/self.output_size-0.5

					#output ply
					output_name = config.sample_dir+"/"+"part_"+str(cid)+".ply"
					fout = open(output_name, 'w')
					fout.write("ply\n")
					fout.write("format ascii 1.0\n")
					fout.write("element vertex "+str(len(verts))+"\n")
					fout.write("property float x\n")
					fout.write("property float y\n")
					fout.write("property float z\n")
					fout.write("property uchar red\n")
					fout.write("property uchar green\n")
					fout.write("property uchar blue\n")
					fout.write("element face "+str(len(faces))+"\n")
					fout.write("property uchar red\n")
					fout.write("property uchar green\n")
					fout.write("property uchar blue\n")
					fout.write("property list uchar int vertex_index\n")
					fout.write("end_header\n")

					color = color_list[cid]

					for i in range(len(verts)):
						fout.write(str(verts[i,0])+" "+str(verts[i,1])+" "+str(verts[i,2])+" "+color+"\n")

					for i in range(len(faces)):
						fout.write(color+" 3 "+str(faces[i,0])+" "+str(faces[i,1])+" "+str(faces[i,2])+"\n")

					fout.close()

					print(cid, "template")
				except:
					print(cid, "empty")


	def iou(self, config, use_post_processing=True):
		#list checkpoints
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			with open(checkpoint_txt) as fin:
				models_dir = [d.strip() for d in fin.readlines()]
		else:
			print(" [!] Load failed")
			exit(-1)

		if use_post_processing:
			from sklearn.neighbors import KDTree

		with torch.no_grad():

			self.dae_network.eval()

			data_hdf5_name = config.data_dir+'/'+config.data_file+'.hdf5'
			data_txt_name = config.data_dir+'/'+config.data_file+'.txt'

			#load input voxels
			data_dict = h5py.File(data_hdf5_name, 'r')
			data_voxels = data_dict['voxels'][:,:,:,:,0]
			data_dict.close()

			#load point clouds and labels
			gt_seg_points, gt_seg_labels, gt_seg_point_num, gt_seg_part_num, gt_seg_obj_names = utils.parse_txt_list(data_txt_name, config.data_dir+'/'+"points")


			for model_dir in models_dir:

				#load checkpoint
				self.dae_network.load_state_dict(torch.load(model_dir))
				start_epoch = int(model_dir.split('-')[-1].split('.')[0])+1
				print(" [*] Load SUCCESS", start_epoch-1)

				#to store predicted labels
				pred_branch = np.zeros(gt_seg_labels.shape, np.int32)
				pred_labels = np.zeros(gt_seg_labels.shape, np.int32)

				zs = []

				#obtain branch labels (template labels)
				for idx in range(len(data_voxels)):
					seg_point_num = gt_seg_point_num[idx]
					seg_points = gt_seg_points[idx,:seg_point_num]

					voxel = data_voxels[idx:idx+1,None,...].astype(np.float32)
					voxel_ = torch.from_numpy(voxel).to(self.device)

					t_vector, d_vector = self.dae_network.encoder(voxel_)
					t_vector = t_vector.detach().cpu().numpy() #[1,branch_num,1]
					t_vector = np.reshape(t_vector, [1,self.branch_num])

					seg_points_ = torch.from_numpy(seg_points[None,...]).to(self.device)
					pred_values = self.dae_network.generator(seg_points_, d_vector, None, out_sum=True, out_branch=True)
					pred_values = pred_values.detach().cpu().numpy()
					pred_values = np.reshape(pred_values, [seg_point_num,self.branch_num])
					pred_values = pred_values * t_vector

					zs.append(t_vector)

					pred_branch_labels = np.argmax(pred_values, 1)
					pred_branch[idx,:seg_point_num] = pred_branch_labels

					if use_post_processing:
						try:
							valid_labels = np.max(pred_values, 1)>0.4
							valid_seg_points = seg_points[valid_labels]
							valid_pred_branch_labels = pred_branch_labels[valid_labels]
							kd_tree = KDTree(valid_seg_points, leaf_size=8)
							_, closest_idx = kd_tree.query(seg_points)
							pred_branch_labels = valid_pred_branch_labels[np.reshape(closest_idx,[-1])]
							pred_branch[idx,:seg_point_num] = pred_branch_labels
						except:
							pass


				#map branch labels to ground truth labels
				#use simple voting
				#store how many points vote for <branch i -> part j>
				poll = np.zeros([self.branch_num,gt_seg_part_num], np.int32)
				for idx in range(len(data_voxels)):
					seg_point_num = gt_seg_point_num[idx]
					seg_labels = gt_seg_labels[idx,:seg_point_num]

					pred_branch_labels = pred_branch[idx,:seg_point_num]

					for i in range(self.branch_num):
						for j in range(gt_seg_part_num):
							poll[i,j] += np.sum((pred_branch_labels==i)&(seg_labels==j))
				mapping = np.argmax(poll,1)
				#print("initial mapping", mapping)


				#evaluation
				pred_labels = mapping[pred_branch]
				shape_mIOU = [None] * len(data_voxels)
				for idx in range(len(data_voxels)):
					seg_point_num = gt_seg_point_num[idx]
					seg_labels = gt_seg_labels[idx,:seg_point_num]

					pred_part_labels = pred_labels[idx,:seg_point_num]

					part_ious = [0.0] * gt_seg_part_num
					for i in range(gt_seg_part_num):
						if (np.sum(seg_labels==i) == 0) and (np.sum(pred_part_labels==i) == 0): # part is not present, no prediction as well
							part_ious[i] = 1.0
						else:
							part_ious[i] = np.sum(( seg_labels==i ) & ( pred_part_labels==i )) / float(np.sum( ( seg_labels==i ) | ( pred_part_labels==i ) ))

					shape_mIOU[idx] = np.mean(part_ious)

				category_mIOU = np.round(np.mean(shape_mIOU)*1000.0)/10.0
				#print("initial iou", category_mIOU)

				#optimize the mapping
				for x in range(self.branch_num):
					original_map = mapping[x]

					for y in range(gt_seg_part_num):
						if y==original_map:
							continue

						mapping[x] = y

						#evaluation
						pred_labels = mapping[pred_branch]
						shape_mIOU = [None] * len(data_voxels)
						for idx in range(len(data_voxels)):
							seg_point_num = gt_seg_point_num[idx]
							seg_labels = gt_seg_labels[idx,:seg_point_num]

							pred_part_labels = pred_labels[idx,:seg_point_num]

							part_ious = [0.0] * gt_seg_part_num
							for i in range(gt_seg_part_num):
								if (np.sum(seg_labels==i) == 0) and (np.sum(pred_part_labels==i) == 0): # part is not present, no prediction as well
									part_ious[i] = 1.0
								else:
									part_ious[i] = np.sum(( seg_labels==i ) & ( pred_part_labels==i )) / float(np.sum( ( seg_labels==i ) | ( pred_part_labels==i ) ))

							shape_mIOU[idx] = np.mean(part_ious)

						temp_category_mIOU = np.round(np.mean(shape_mIOU)*1000.0)/10.0

						if temp_category_mIOU>category_mIOU:
							category_mIOU = temp_category_mIOU
						else:
							mapping[x] = original_map

				#print("final mapping", mapping)
				print("final iou", category_mIOU)

				#zs are part existence scores
				#analyze zs and save:
				#1. the number of learned part templates that are actually used in shapes
				#2. after clustering by part existence scores, how many groups are there
				#3. how many groups contain more than 2 shapes
				#4. how many groups contain more than 4 shapes
				#...

				zs = np.concatenate(zs,0)
				zs = (zs>0.5).astype(np.uint8)
				part_num_0 = np.sum(np.any(zs,0))
				unique_zs,unique_idxs,counts = np.unique(zs,return_index=True,return_counts=True,axis=0)
				unique_zs_len_0 = np.sum(counts>0)
				unique_zs_len_2 = np.sum(counts>2)
				unique_zs_len_4 = np.sum(counts>4)
				unique_zs_len_8 = np.sum(counts>8)
				unique_zs_len_16 = np.sum(counts>16)
				unique_zs_len_32 = np.sum(counts>32)
				unique_zs_len_64 = np.sum(counts>64)

				fout = open(model_dir+".iou."+str(category_mIOU)+".txt",'w')
				fout.write(str(part_num_0)+"\n")
				fout.write(str(unique_zs_len_0)+"\n")
				fout.write(str(unique_zs_len_2)+"\n")
				fout.write(str(unique_zs_len_4)+"\n")
				fout.write(str(unique_zs_len_8)+"\n")
				fout.write(str(unique_zs_len_16)+"\n")
				fout.write(str(unique_zs_len_32)+"\n")
				fout.write(str(unique_zs_len_64)+"\n")
				fout.close()


	def cluster(self, config, dataloader_test, use_actual_part_existence=False):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			with open(checkpoint_txt) as fin:
				model_dir = fin.readline().strip()
			self.dae_network.load_state_dict(torch.load(model_dir))
			start_epoch = int(model_dir.split('-')[-1].split('.')[0])+1
			print(" [*] Load SUCCESS", start_epoch-1)
		else:
			print(" [!] Load failed")
			exit(-1)


		if use_actual_part_existence:
			#get coords for testing
			test_grid_size = config.resolution
			frame_t = np.linspace(0, test_grid_size-1, test_grid_size, dtype = np.int32)
			frame_x, frame_y, frame_z = np.meshgrid(frame_t,frame_t,frame_t, sparse=False, indexing='ij')
			frame_x = np.reshape(frame_x,[-1]).astype(np.int32)
			frame_y = np.reshape(frame_y,[-1]).astype(np.int32)
			frame_z = np.reshape(frame_z,[-1]).astype(np.int32)
			test_coords = np.concatenate([frame_x[:,None],frame_y[:,None],frame_z[:,None]],1)
			test_coords = (test_coords.astype(np.float32)+0.5)/test_grid_size-0.5


		with torch.no_grad():
			self.dae_network.eval()

			if use_actual_part_existence:
				points = torch.from_numpy(test_coords).to(self.device).view(1,test_grid_size**3,3)

			#compute live dead
			zs = []
			for idx, data in enumerate(dataloader_test, 0):

				if idx%100==99: print(idx)

				points_, values_, voxels_ = data

				t_vector, d_vector = self.dae_network.encoder(voxels_.to(self.device))
				t_vector = t_vector.detach().cpu().numpy()
				t_vector = np.reshape(t_vector, [1,self.branch_num])

				if use_actual_part_existence:
					pred_values = self.dae_network.generator(points, d_vector, None, out_sum=True, out_branch=True)
					pred_values = pred_values.detach().cpu().numpy()
					pred_values = np.reshape(pred_values, [-1,self.branch_num])
					pred_values = pred_values * t_vector
					z = np.sum((pred_values>0.5).astype(np.int32),0)
					z = (z[None,...]>10).astype(np.uint8) #part exists if at least 10 voxels are occupied by this part
				else:
					z = (t_vector>0.4).astype(np.uint8) #part exists if part existence score > 0.4

				zs.append(z)

			zs = np.concatenate(zs,0)

			unique_zs,unique_idxs,counts = np.unique(zs,return_index=True,return_counts=True,axis=0)
			print(len(unique_zs))
			for lowerbound in range(100):
				get_list = []
				get_list_c = []
				for i in range(len(unique_zs)):
					if counts[i]>lowerbound:
						get_list.append(unique_idxs[i])
						get_list_c.append(counts[i])
				if len(get_list)<=64: break #visualize the first 64 groups, ranked by the number of shapes they contain
			print(lowerbound,len(get_list))

			for idx, data in enumerate(dataloader_test, 0):

				points_, values_, voxels_ = data

				if idx in get_list:

					self.voxel2seg(voxels_.to(self.device), config.sample_dir+"/"+str(idx)+"_"+str(get_list_c[get_list.index(idx)])+".ply")

					#ground truth
					points_out = voxels_[0,0].detach().cpu().numpy()
					verts, faces, _, _ = measure.marching_cubes(0.5-points_out[:,:,:], 0)
					verts = (verts+0.5)/64-0.5
					utils.write_ply_triangle(config.sample_dir+"/"+str(idx)+"_"+str(get_list_c[get_list.index(idx)])+"_gt.ply", verts, faces)


	def voxel2seg(self,voxel,output_name,affine_only=False):

		t_vector, d_vector = self.dae_network.encoder(voxel)
		t_vector = t_vector.detach().cpu().numpy() #[1,branch_num,1]
		t_vector = np.reshape(t_vector, [1,1,1,self.branch_num])

		pred_voxel = np.zeros([self.output_size+2,self.output_size+2,self.output_size+2,self.branch_num],np.float32)
		queue = []
		for i in range(self.frame_grid_size):
			for j in range(self.frame_grid_size):
				for k in range(self.frame_grid_size):
					queue.append((i,j,k))

		cell_batch_size = self.cell_grid_size**3
		cell_batch_num = int(self.test_point_batch_size/cell_batch_size)

		while len(queue)>0:
			batch_num = min(len(queue),cell_batch_num)
			point_list = []
			points = []
			for i in range(batch_num):
				point = queue.pop(0)
				point_list.append(point)
				points.append(self.cell_coords[point[0],point[1],point[2]])
			points = np.concatenate(points, axis=0)
			points = torch.from_numpy(points[None,...]).to(self.device)
			pred_values = self.dae_network.generator(points, d_vector, None, out_sum=True, out_branch=True, affine_only=affine_only)
			pred_values = pred_values.detach().cpu().numpy()
			pred_values = np.reshape(pred_values, [-1,self.branch_num])
			for i in range(batch_num):
				point = point_list[i]
				values = pred_values[i*cell_batch_size:(i+1)*cell_batch_size]
				x_coords = self.cell_x+point[0]*self.cell_grid_size
				y_coords = self.cell_y+point[1]*self.cell_grid_size
				z_coords = self.cell_z+point[2]*self.cell_grid_size
				pred_voxel[x_coords+1,y_coords+1,z_coords+1] = values

		pred_voxel = pred_voxel * t_vector

		vertices_num = 0
		triangles_num = 0
		vertices_list = []
		triangles_list = []
		vertices_num_list = [0]
		for cid in range(self.branch_num):
			if np.max(pred_voxel[:,:,:,cid])>0.5:
				vertices, triangles, _, _ = measure.marching_cubes(0.5-pred_voxel[:,:,:,cid], 0)
				vertices_num += len(vertices)
				triangles_num += len(triangles)
				vertices_list.append(vertices)
				triangles_list.append(triangles)
				vertices_num_list.append(vertices_num)
			else:
				vertices_num += 0
				triangles_num += 0
				vertices_list.append(None)
				triangles_list.append(None)
				vertices_num_list.append(vertices_num)

		#output ply
		fout = open(output_name, 'w')
		fout.write("ply\n")
		fout.write("format ascii 1.0\n")
		fout.write("element vertex "+str(vertices_num)+"\n")
		fout.write("property float x\n")
		fout.write("property float y\n")
		fout.write("property float z\n")
		fout.write("property uchar red\n")
		fout.write("property uchar green\n")
		fout.write("property uchar blue\n")
		fout.write("element face "+str(triangles_num)+"\n")
		fout.write("property uchar red\n")
		fout.write("property uchar green\n")
		fout.write("property uchar blue\n")
		fout.write("property list uchar int vertex_index\n")
		fout.write("end_header\n")

		for cid in range(self.branch_num):
			color = color_list[cid]
			if vertices_list[cid] is not None:
				vertices = (vertices_list[cid]-0.5)/self.output_size-0.5
				for i in range(len(vertices)):
					fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")

		for cid in range(self.branch_num):
			color = color_list[cid]
			if triangles_list[cid] is not None:
				triangles = triangles_list[cid] + vertices_num_list[cid]
				for i in range(len(triangles)):
					fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")

		fout.close()
