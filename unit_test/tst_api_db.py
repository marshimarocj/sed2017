import os


'''func
'''
def tst_c3d_ftdb():
  ft_root_dir = '/data/liujiang/cj_data/sed/c3d' # 198
  ft_dir = os.path.join(ft_root_dir, 'LGW_20071101_E1_CAM1')
  ft_gap = 16
  chunk_gap = 10000

  ftdb = FtDb(ft_dir, ft_gap, chunk_gap)
  fts = ftdb.load_chunk(10)

  print fts.shape


'''expr
'''


if __name__ == '__main__':
  tst_c3d_ftdb()
