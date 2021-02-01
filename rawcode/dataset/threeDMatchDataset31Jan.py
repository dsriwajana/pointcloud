import torch
import open3d as o3d
import numpy as np
import os
import pickle

def randomly_rotation_matrix(augment_rotation=1):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    return Rx @ Ry @ Rz

class ThreeDMatchTrainDataset(torch.utils.data.Dataset):

  def __init__(self, 
               root):
    #declare variables
    self.root = root

    #locations of files
    pts_filename = os.path.join(self.root, '3DMatch_train_0.030_points.pkl')
    keypts_filename = os.path.join(self.root, '3DMatch_train_0.030_keypts.pkl')

    #load files
    if os.path.exists(pts_filename) and os.path.exists(keypts_filename):
      with open(pts_filename, 'rb') as file:
        data = pickle.load(file)
        self.points = [*data.values()]
        self.ids_list = [*data.keys()]
      with open(keypts_filename, 'rb') as file:
        self.correspondences = pickle.load(file)
      print(f"Load PKL file from {self.root}")
    else:
      print("PKL file not found.") 

    #store correspondence pairs (src name -> tgt name)
    self.src_to_tgt = {}
    for idpair in self.correspondences.keys():
      src = idpair.split("@")[0]
      tgt = idpair.split("@")[1]
      if src not in self.src_to_tgt.keys():
        self.src_to_tgt[src] = [tgt]
      else:
        self.src_to_tgt[src] += [tgt]

  def __getitem__(self, index):
    #get source from index, get target randomly
    src_name = list(self.src_to_tgt.keys())[index]
    tgt_name = np.random.choice(self.src_to_tgt[src_name])

    #-----------generate patch pair from point clouds-------------#
    
    ##match pairs of point clouds
    corr = self.correspondences[f"{src_name}@{tgt_name}"]
    randomIndex = np.random.randint(len(corr))
    corrPair = corr[randomIndex]

    #load source and target point clouds 
    src_index = self.ids_list.index(src_name)
    tgt_index = self.ids_list.index(tgt_name)
    src_points = self.points[src_index]
    tgt_points = self.points[tgt_index]

    if np.random.choice([True, False]):
      #key points location
      src_keypt = np.array(src_points[corrPair[0]])
      tgt_keypt = np.array(tgt_points[corrPair[1]])

      #create patches
      src_pcd = make_point_cloud(src_points)
      tgt_pcd = make_point_cloud(tgt_points)
      src_patch = src_pcd.translate(src_keypt.dot(-1))
      tgt_patch = tgt_pcd.translate(tgt_keypt.dot(-1))
      src_patch = src_patch.voxel_down_sample(voxel_size = 0.05)
      tgt_patch = tgt_patch.voxel_down_sample(voxel_size = 0.05)
      src_patch.rotate(randomly_rotation_matrix(),[0,0,0])
      tgt_patch.rotate(randomly_rotation_matrix(),[0,0,0])
      bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
      src_patch = src_patch.crop(bbox)
      tgt_patch = tgt_patch.crop(bbox)

    #currently patches are not good
      print("acquiring corresponded patches")
      return src_patch.points, tgt_patch.points, 1

    else:
      #key points location
      src_keypt = np.array(src_points[corrPair[0]])
      tgt_keypt = np.array(tgt_points[corrPair[1]])
      count = 0
      while True:
        random_index = np.random.randint(len(tgt_points))
        random_pt = tgt_points[random_index]
        if np.sum((tgt_keypt[:] - random_pt[:]) ** 2) > 1:
          break
        if count == 100:
          return [0,0,0],[0,0,0]

      #create patches
      src_pcd = make_point_cloud(src_points)
      tgt_pcd = make_point_cloud(tgt_points)
      src_patch = src_pcd.translate(src_keypt.dot(-1))
      tgt_patch = tgt_pcd.translate(random_pt.dot(-1))
      src_patch = src_patch.voxel_down_sample(voxel_size = 0.05)
      tgt_patch = tgt_patch.voxel_down_sample(voxel_size = 0.05)
      src_patch.rotate(randomly_rotation_matrix(),[0,0,0])
      tgt_patch.rotate(randomly_rotation_matrix(),[0,0,0])
      bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
      src_patch = src_patch.crop(bbox)
      tgt_patch = tgt_patch.crop(bbox)

      print("acquiring uncorresponded patches")
      return src_patch.points, tgt_patch.points, 0


  #have not tried 
  def __len__(self):
    return len(self.src_to_tgt.keys())
    
