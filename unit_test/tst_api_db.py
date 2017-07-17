import os
import sys
sys.path.append('../')

import api.db


'''func
'''
def tst_c3d_ftdb():
  ft_root_dir = '/data/liujiang/cj_data/sed/c3d' # 198
  ft_dir = os.path.join(ft_root_dir, 'LGW_20071101_E1_CAM1')
  ft_gap = 16
  chunk_gap = 10000

  ftdb = api.db.FtDb(ft_dir, ft_gap, chunk_gap)
  fts = ftdb.load_chunk(chunk_gap*5)

  print fts.shape


def tst_paf_ftdb():
  ft_root_dir = '/mnt/sdd/jiac/data/sed/paf/1.0' # gpu1
  ft_dir = os.path.join(ft_root_dir, 'LGW_20071107_E1_CAM2')
  ft_gap = 5
  chunk_gap = 7500

  ftdb = api.db.FtDb(ft_dir, ft_gap, chunk_gap)
  fts = ftdb.load_chunk(chunk_gap*3)

  print fts.shape


'''expr
'''


if __name__ == '__main__':
  tst_c3d_ftdb()
