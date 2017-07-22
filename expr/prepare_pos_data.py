import os
import sys
sys.path.append('../')

import numpy as np

import api.db
import api.generator


'''func
'''
def load_track_label_file(file):
  id2event = {}
  with open(file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      id = int(data[0])
      event = data[1]
      id2event[id] = event

  return id2event


def c3d_threshold_func(qbegin, qend, tbegin, tend):
  ibegin = max(tbegin, qbegin)
  iend = min(tend, qend)
  return  iend - ibegin >= 8


'''expr
'''
def prepare_pos_c3d():
  root_dir = '/data1/jiac/sed' # uranus
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_label_dir = os.path.join(root_dir, 'tracklet_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d')
  out_dir = os.path.join(root_dir, 'c3d', 'pos')

  direction = 'forward'
  track_len = 25

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        names.append(name)

  c3d_centers = api.db.get_c3d_centers()

  for name in names:
    track_label_file = os.path.join(track_label_dir, '%s.%d.%s.pos'%(name, track_len, direction))
    id2event = load_track_label_file(track_label_file)
    pos_trackids = id2event.keys()

    track_file = os.path.join(track_dir, '%s.%d.%s.npz'%(name, track_len, direction))
    track_map_file = os.path.join(track_dir, '%s.%d.%s.map'%(name, track_len, direction))
    track_db = api.db.TrackDb(track_map_file, track_file, track_len, pos_trackids)

    ft_dir = os.path.join(ft_root_dir, name)
    c3d_db = api.db.C3DFtDb(ft_dir)

    pos_c3d_in_track_generator = api.generator.crop_duration_ft_in_track(
      track_db, c3d_db, c3d_centers, c3d_threshold_func)
    fts = []
    frames = []
    centers = []
    ids = []
    for ft_in_track in pos_c3d_in_track_generator:
      num = len(ft_in_track.frames)
      fts.append(ft_in_track.fts)
      frames.extend(ft_in_track.frames)
      centers.append(ft_in_track.centers)
      ids.extend([num*ft_in_track.id])

    fts = np.concatenate(fts, 0)
    frames = np.array(frames, dtype=np.int32)
    centers = np.concatenate(centers, 0)
    ids = np.array(ids, dtype=np.int32)
    out_file = os.path.join(out_dir, name + '.npz')
    np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


if __name__ == '__main__':
  prepare_pos_c3d()
