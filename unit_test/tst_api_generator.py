import os
import sys
sys.path.append('../')

import api.db
import api.generator


'''func
'''
def c3d_threshold_func(qbegin, qend, tbegin, tend):
  ibegin = max(tbegin, qbegin)
  iend = min(tend, qend)
  return  iend - ibegin >= 8


'''expr
'''


def tst_c3d_toi():
  root_dir = '/data1/jiac/sed' # uranus
  tracking_root_dir = os.path.join(root_dir, 'tracking')
  c3d_root_dir = os.path.join(root_dir, 'c3d')
  video_name = 'LGW_20071107_E1_CAM3'

  direction = 'forward'
  track_len = 25
  track_map_file = os.path.join(tracking_root_dir, '%s.%d.%s.map'%(video_name, track_len, direction))
  track_file = os.path.join(tracking_root_dir, '%s.%d.%s.npz'%(video_name, track_len, direction))
  track_db = api.db.TrackDb(track_map_file, track_file, track_len)

  c3d_dir = os.path.join(c3d_root_dir, video_name)
  c3d_db = api.db.C3DFtDb(c3d_dir)

  chunk_idx = 1
  centers = api.db.get_c3d_centers()
  ft_in_track_generator = api.generator.duration_ft_in_track_generator(
    track_db, c3d_db, centers, c3d_db.chunk_gap*chunk_idx, c3d_threshold_func)

  cnt = 0
  for ft_in_track in ft_in_track_generator:
    print ft_in_track.id, ft_in_track.fts.shape, len(set(ft_in_track.frames))
    cnt += 1
    if cnt == 100:
      break


def tst_vgg_toi():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  tracking_root_dir = os.path.join(root_dir, 'tracking', 'person')
  vgg_root_dir = os.path.join(root_dir, 'raw_ft', 'vgg19_pool5')
  video_name = 'LGW_20071107_E1_CAM2'

  direction = 'forward'
  track_len = 25
  track_map_file = os.path.join(tracking_root_dir, '%s.%d.%s.map'%(video_name, track_len, direction))
  track_file = os.path.join(tracking_root_dir, '%s.%d.%s.npz'%(video_name, track_len, direction))
  track_db = api.db.TrackDb(track_map_file, track_file, track_len)

  vgg_dir = os.path.join(vgg_root_dir, video_name)
  vgg_db = api.db.VGG19FbDb(vgg_dir)

  chunk_idx = 1
  centers = api.db.get_vgg19_centers()
  ft_in_track_generator = api.generator.instant_ft_in_track_generator(
    track_db, vgg_db, centers, vgg_db.chunk_gap*chunk_idx)

  cnt = 0
  for ft_in_track in ft_in_track_generator:
    print ft_in_track.id, ft_in_track.fts.shape, len(set(ft_in_track.frames))
    cnt += 1
    if cnt == 100:
      break


if __name__ == '__main__':
  # tst_c3d_toi()
  tst_vgg_toi()
