import os
import sys
sys.path.append('../')

import api.db
import api.generator


'''func
'''


'''expr
'''
def tst_c3d_toi():
  root_dir = '/data1/jiac/sed' # uranus
  tracking_root_dir = os.path.join(root_dir, 'tracking')
  c3d_root_dir = os.path.join(root_dir, 'c3d')
  video_name = 'LGW_20071107_E1_CAM3'

  toi_threshold = 0.5

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
    track_db, c3d_db, centers, c3d_db.chunk_gap*chunk_idx, tiou_threshold)

  cnt = 0
  for ft_in_track in ft_in_track_generator:
    print ft_in_track.id, ft_in_track.fts.shape, len(set(ft_in_track.frames))
    cnt += 1
    if cnt == 100:
      break


if __name__ == '__main__':
  tst_c3d_toi()
