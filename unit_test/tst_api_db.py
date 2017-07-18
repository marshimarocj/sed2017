import os
import sys
sys.path.append('../')

import api.db


'''func
'''



'''expr
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


def tst_vgg19_ftdb():
  ft_root_dir = '/home/jiaac/hdd/sed/vgg19_pool5'
  ft_dir = os.path.join(ft_root_dir, 'LGW_20071101_E1_CAM3')
  ft_gap = 5
  chunk_gap = 7500

  ftdb = api.db.FtDb(ft_dir, ft_gap, chunk_gap)
  fts = ftdb.load_chunk(chunk_gap*2)

  print fts.shape


def tst_track_db():
  root_dir = '/data1/jiac/sed' # uranus
  tracking_dir = os.path.join(root_dir, 'tracking')
  video_name = 'LGW_20071107_E1_CAM3'
  direction = 'forward'
  track_len = 25
  track_map_file = os.path.join(tracking_dir, '%s.%s.%d.map'%(video_name, direction, track_len))
  track_file = os.path.join(tracking_dir, '%s.%s.%d.npz'%(video_name, direction, track_len))

  trackdb = api.db.TrackDb(track_map_file, track_file, track_len)
  
  print len(trackdb.valid_trackletids)
  print trackdb.track_len
  print trackdb.tracks.shape
  # print trackdb.frame_box2trackletid


if __name__ == '__main__':
  # tst_c3d_ftdb()
  # tst_paf_ftdb()
  # tst_vgg_ftdb()
  tst_track_db()
