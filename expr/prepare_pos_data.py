import os
import argparse
import sys
sys.path.append('../')

import numpy as np

import api.db
import api.generator


'''func
'''
def load_pos_track_label_file(file):
  valid_events = set(['CellToEar', 'Embrace', 'Pointing', 'PersonRuns'])
  id2event = {}
  with open(file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      event = data[1]
      if event in valid_events:
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
  # lst_files = [
  #   os.path.join(root_dir, 'dev08-1.lst'),
  #   os.path.join(root_dir, 'eev08-1.lst'),
  # ]
  track_label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d')
  out_dir = os.path.join(root_dir, 'c3d', 'track_group')

  # track_len = 25
  # track_len = 50
  track_lens = [25, 50]

  # names = []
  # for lst_file in lst_files:
  #   with open(lst_file) as f:
  #     for line in f:
  #       line = line.strip()
  #       name, _ = os.path.splitext(line)
  #       names.append(name)

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  c3d_centers = api.db.get_c3d_centers()

  for track_len in track_lens:
    track_label_file = os.path.join(track_label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    id2event = load_pos_track_label_file(track_label_file)
    pos_trackids = id2event.keys()

    db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file, pos_trackids)

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
      ids.extend(num*[ft_in_track.id])

    fts = np.concatenate(fts, 0)
    frames = np.array(frames, dtype=np.int32)
    centers = np.concatenate(centers, 0)
    ids = np.array(ids, dtype=np.int32)
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
    np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


def generate_script():
  root_dir = '/data1/jiac/sed' # uranus
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]

  num_process = 10

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' in name:
          continue
        names.append(name)

  num = len(names)
  gap = (num + num_process - 1) / num_process

  for i in range(0, num, gap):
    idx = i/gap
    out_file = 'prepare_pos_data.%d.sh'%idx
    with open(out_file, 'w') as fout:
      for j in range(i, min(i+gap, num)):
        name = names[j]
        cmd = [
          'python', 'prepare_pos_data.py', name
        ]
        fout.write(' '.join(cmd) + '\n')


def prepare_pos_vgg19():
  root_dir = '' # gpu9


if __name__ == '__main__':
  prepare_pos_c3d()
  # generate_script()
