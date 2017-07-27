import os
import sys
sys.path.append('../')

import numpy as np

import api.db
import sample


'''func
'''


'''expr
'''
def sample_data_for_center():
  # root_dir = '/data1/jiac/sed' # uranus
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  root_dir = '/home/jiac/data2/sed' # gpu9
  ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  num_sample = 10000
  out_file = os.path.join(root_dir, 'sample.%d.npy'%num_sample)

  names = [
    'LGW_20071101_E1_CAM1',
    'LGW_20071106_E1_CAM2',
    'LGW_20071107_E1_CAM3',
    'LGW_20071108_E1_CAM5',
  ]

  rs = sample.ReservoirSampling(num_sample)

  for name in names:
    ft_dir = os.path.join(ft_root_dir, name)
    # c3d_db = api.db.C3DFtDb(ft_dir)
    vgg_db = api.db.VGG19FtDb(ft_dir)
    print name

    # for chunk in c3d_db.chunks:
    for chunk in vgg_db.chunks:
      fts = c3d_db.load_chunk(chunk)
      fts = np.moveaxis(fts, (0, 1, 2, 3), (0, 3, 1, 2))
      dim_ft = fts.shape[-1]
      fts = fts.reshape((-1, dim_ft))
      for i in range(fts.shape[0]):
        rs.addData(fts[i])

  data = rs.pool
  np.save(out_file, data)


if __name__ == '__main__':
  sample_data_for_center()
