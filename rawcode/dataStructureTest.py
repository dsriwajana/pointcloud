import ocnn
import os
import torch
import numpy as np
from tqdm import tqdm
from config import parse_args
from modelnet import ModelNet40

import random
import math
import plotly.graph_objects as go
import plotly.express as px

# from transforms.py import TransformPoints, Points2Octree

def get_dataloader(flags, train=True):
  transform = ocnn.TransformCompose(flags)
  dataset = ModelNet40(flags.location, train, transform, in_memory=False)
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=flags.batch_size, shuffle=train, pin_memory=True,
      num_workers=flags.num_workers, collate_fn=ocnn.collate_octrees)
  return data_loader


def load_modelnet40(suffix='points'):
    points, labels = [], []
    folders = sorted(os.listdir(root))
    assert len(folders) == 40
    for idx, folder in enumerate(folders):
      subfolder = 'train' if train else 'test'
      current_folder = os.path.join(root, folder, subfolder)
      filenames = sorted(os.listdir(current_folder))
      for filename in filenames:
        if filename.endswith(suffix):
          filename_abs = os.path.join(current_folder, filename)
          if in_memory:
            points.append(np.fromfile(filename_abs, dtype=np.uint8))
          else:
            points.append(filename_abs)
          labels.append(idx)
    return points, labels, folders


if __name__ == "__main__":
  # configs
  FLAGS = parse_args()
  root = FLAGS.DATA.train.location
  train = True
  in_memory = False
  points,labels,folder = load_modelnet40()
  print(points)
  print(labels)
  print(folder)
  pointsFromFile = np.fromfile(points[10], np.uint8)
  pointsFromFile = torch.from_numpy(pointsFromFile)
  transform = ocnn.TransformCompose(FLAGS.DATA.train)
  dataset = ModelNet40(root, train=True, transform=transform, in_memory=False)
  sampleOctreeWithLabels = dataset.__getitem__(2000)
  sampleOctreeWithLabels = sampleOctreeWithLabels
  print(sampleOctreeWithLabels)
  sampleOctree = sampleOctreeWithLabels[0].cuda()
  print(sampleOctree)
  print(sampleOctree.size())
  print('feature')
  print(ocnn.octree_property(sampleOctree, 'feature', FLAGS.MODEL.depth))
  print(ocnn.octree_property(sampleOctree, 'feature', FLAGS.MODEL.depth).size())
  print('key')
  print(ocnn.octree_property(sampleOctree, 'key', FLAGS.MODEL.depth))
  print(ocnn.octree_property(sampleOctree, 'key', FLAGS.MODEL.depth).size())
  print('xyz')
  print(ocnn.octree_property(sampleOctree, 'xyz', FLAGS.MODEL.depth))
  print(ocnn.octree_property(sampleOctree, 'xyz', FLAGS.MODEL.depth).size())
  print('index')
  print(ocnn.octree_property(sampleOctree, 'index', FLAGS.MODEL.depth))
  print(ocnn.octree_property(sampleOctree, 'index', FLAGS.MODEL.depth).size())
  print(ocnn.octree_property(sampleOctree, 'label', FLAGS.MODEL.depth))
  torch.save(ocnn.octree_property(sampleOctree, 'key', FLAGS.MODEL.depth), '/file.pt')
  x= torch.load('/file.pt')
  print(x)

  locationOfPoints = '/root/O-CNN/tensorflow/script/dataset/ModelNet40/ModelNet40.points/table/test/table_0393.upgrade.smp.points'
  pointsFromFile = np.fromfile(locationOfPoints, np.uint8)
  pointsFromFile = torch.from_numpy(pointsFromFile)
  transform = ocnn.TransformCompose(FLAGS.DATA.train)
  print(pointsFromFile[0:10])
  print(pointsFromFile.size())
  points = ocnn.NormalizePoints('sphere')(pointsFromFile)
  points = ocnn.TransformPoints(**FLAGS.DATA.train)(points)
  octreeWhat = ocnn.Points2Octree(**FLAGS.DATA.train)(points)
  print(ocnn.octree_property(octreeWhat.cuda(), 'label', FLAGS.MODEL.depth))
  torch.save(ocnn.octree_property(octreeWhat.cuda(), 'key', FLAGS.MODEL.depth), '/table_0393_key.pt')
  torch.save(ocnn.octree_property(octreeWhat.cuda(), 'feature', FLAGS.MODEL.depth), '/table_0393_feature.pt')
  x= torch.load('/chair_0890.pt')
  print(x)
  # print(octree.size())
  # transform = ocnn.TransformCompose(FLAGS.DATA.train)
  # dataset = ModelNet40(root, train=True, transform=transform, in_memory=False)

  # # data
  # train_loader = get_dataloader(FLAGS.DATA.train, train=True)
  # test_loader = get_dataloader(FLAGS.DATA.test,  train=False)

  # # model
  # flags_model = FLAGS.MODEL
  # model = ocnn.LeNet(flags_model.depth, flags_model.channel, flags_model.nout)
  # model.cuda()
  # a=0
  # for i, data in enumerate(train_loader, 0):
  #   # get the inputs
  #   octrees, labels = data[0].cuda(), data[1].cuda()
  #   a+=1
  #   if a==10:
  #     break
  # print(octrees)
  # print(octrees.size())
  # data = ocnn.octree_property(octrees, 'feature', FLAGS.MODEL.depth)
  # print(data)
  # print(data.size())
  # # print('key')
  # print(ocnn.octree_property(octrees, 'key', FLAGS.MODEL.depth))
  # print(ocnn.octree_property(octrees, 'key', FLAGS.MODEL.depth).size())
  # print('xyz')
  # print(ocnn.octree_property(octrees, 'xyz', FLAGS.MODEL.depth))
  # print(ocnn.octree_property(octrees, 'xyz', FLAGS.MODEL.depth).size())
  # print('index')
  # print(ocnn.octree_property(octrees, 'index', FLAGS.MODEL.depth))
  # print(ocnn.octree_property(octrees, 'index', FLAGS.MODEL.depth).size())
  # print('label')
  # print(ocnn.octree_property(octrees, 'label', FLAGS.MODEL.depth))

