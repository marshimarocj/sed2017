import os
import itertools
import sys
sys.path.append('../')

import numpy as np
from sklearn.cluster import KMeans

import api.db
import sample


'''func
'''


'''expr
'''
def sample_data_for_center():
  root_dir = '/data1/jiac/sed' # uranus
  ft_root_dir = os.path.join(root_dir, 'c3d')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  num_sample = 10000
  out_file = os.path.join(ft_root_dir, 'sample.%d.npy'%num_sample)

  names = [
    'LGW_20071101_E1_CAM1',
    'LGW_20071106_E1_CAM2',
    'LGW_20071107_E1_CAM3',
    'LGW_20071108_E1_CAM5',
  ]

  rs = sample.ReservoirSampling(num_sample)

  for name in names:
    ft_dir = os.path.join(ft_root_dir, name)
    c3d_db = api.db.C3DFtDb(ft_dir)
    # vgg_db = api.db.VGG19FtDb(ft_dir)
    print name

    for chunk in c3d_db.chunks:
    # for chunk in vgg_db.chunks:
      print chunk
      fts = c3d_db.load_chunk(chunk)
      # fts = vgg_db.load_chunk(chunk)
      shape = fts.shape
      for i, j, k in itertools.product(range(shape[0]), range(shape[2]), range(shape[3])):
        rs.addData(np.array(fts[i, :, j, k]))
      del fts

  data = rs.pool
  np.save(out_file, data)


def cluster_centers():
  # root_dir = '/data1/jiac/sed' # uranus
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  root_dir = '/home/jiac/data2/sed' # gpu9
  ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  sample_file = os.path.join(ft_root_dir, 'sample.10000.npy')
  out_file = os.path.join(ft_root_dir, 'center.32.npy')

  num_center = 32
  kmeans = KMeans(n_clusters=num_center)

  data = np.load(sample_file)
  kmeans.fit(data)

  cluster_centers = kmeans.cluster_centers_
  np.save(out_file, cluster_centers_)


if __name__ == '__main__':
  # sample_data_for_center()
  cluster_centers()
