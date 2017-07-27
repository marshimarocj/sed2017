import os
import argparse
import random
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
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d')
  out_dir = os.path.join(root_dir, 'c3d', 'track_group')

  # track_len = 25
  # track_len = 50
  track_lens = [25, 50]

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  c3d_centers = api.db.get_c3d_centers()

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    id2event = load_pos_track_label_file(label_file)
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


def shuffle_neg():
  root_dir = '/data1/jiac/sed' # uranus
  label_dir = os.path.join(root_dir, 'pseudo_label')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]

  track_lens = [25, 50]

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' in name:
          continue
        names.append(name)

  for name in names:
    print name
    for track_len in track_lens:
      neg_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg'%(name, track_len))
      ids = []
      with open(neg_file) as f:
        for line in f:
          line = line.strip()
          ids.append(int(line))
      random.shuffle(ids)

      num = len(ids)
      gap = (num + 9)/10
      for i in range(0, 10):
        _ids = ids[gap*i:gap*(i+1)]
        out_file = '%s.%d'%(neg_file, i)
        with open(out_file, 'w') as fout:
          for id in _ids:
            fout.write('%d\n'%id)


def prepare_neg_c3d():
  root_dir = '/data1/jiac/sed' # uranus
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d')
  out_dir = os.path.join(root_dir, 'c3d', 'track_group')

  track_lens = [25, 50]
  neg_split = 0

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  c3d_centers = api.db.get_c3d_centers()

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg.%d'%(name, track_len, neg_split))
    neg_trackids = []
    with open(label_file) as f:
      for line in f:
        line = line.strip()
        neg_trackids.append(int(line))

    db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file, neg_trackids)

    ft_dir = os.path.join(ft_root_dir, name)
    c3d_db = api.db.C3DFtDb(ft_dir)

    neg_c3d_in_track_generator = api.generator.crop_duration_ft_in_track(
      track_db, c3d_db, c3d_centers, c3d_threshold_func)
    fts = []
    frames = []
    centers = []
    ids = []
    for ft_in_track in neg_c3d_in_track_generator:
      num = len(ft_in_track.frames)
      fts.append(ft_in_track.fts)
      frames.extend(ft_in_track.frames)
      centers.append(ft_in_track.centers)
      ids.extend(num*[ft_in_track.id])

    fts = np.concatenate(fts, 0)
    frames = np.array(frames, dtype=np.int32)
    centers = np.concatenate(centers, 0)
    ids = np.array(ids, dtype=np.int32)
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, neg_split))
    np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


def generate_script():
  # root_dir = '/data1/jiac/sed' # uranus
  root_dir = '/home/jiac/data2/sed' # gpu9
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
    out_file = 'prepare_toi_data.%d.sh'%idx
    with open(out_file, 'w') as fout:
      for j in range(i, min(i+gap, num)):
        name = names[j]
        cmd = [
          'python', 'prepare_toi_data.py', name
        ]
        fout.write(' '.join(cmd) + '\n')


def prepare_pos_vgg19():
  root_dir = '/home/jiac/data2/sed' # gpu9
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  out_dir = os.path.join(ft_root_dir, 'track_group')

  track_lens = [25, 50]

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  vgg_centers = api.db.get_vgg19_centers()

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    id2event = load_pos_track_label_file(label_file)
    pos_trackids = id2event.keys()

    db_file = os.path.join(track_dir,'%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file, pos_trackids)

    ft_dir = os.path.join(ft_root_dir, name)
    vgg_db = api.db.VGG19FtDb(ft_dir)

    pos_vgg_in_track_generator = api.generator.crop_instant_ft_in_track(
      track_db, vgg_db, vgg_centers)
    fts = []
    frames = []
    centers = []
    ids = []
    for ft_in_track in pos_vgg_in_track_generator:
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


def prepare_neg_vgg19():
  root_dir = '/home/jiac/data2/sed' # gpu9
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  out_dir = os.path.join(ft_root_dir, 'track_group')

  track_lens = [25, 50]
  neg_split = 0

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  vgg_centers = api.db.get_vgg19_centers()

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg.%d'%(name, track_len, neg_split))
    neg_trackids = []
    with open(label_file) as f:
      for line in f:
        line = line.strip()
        neg_trackids.append(int(line))

    db_file = os.path.join(track_dir,'%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file, neg_trackids)

    ft_dir = os.path.join(ft_root_dir, name)
    vgg_db = api.db.VGG19FtDb(ft_dir)

    neg_vgg_in_track_generator = api.generator.crop_instant_ft_in_track(
      track_db, vgg_db, vgg_centers)
    fts = []
    frames = []
    centers = []
    ids = []
    for ft_in_track in neg_vgg_in_track_generator:
      num = len(ft_in_track.frames)
      fts.append(ft_in_track.fts)
      frames.extend(ft_in_track.frames)
      centers.append(ft_in_track.centers)
      ids.extend(num*[ft_in_track.id])

    fts = np.concatenate(fts, 0)
    frames = np.array(frames, dtype=np.int32)
    centers = np.concatenate(centers, 0)
    ids = np.array(ids, dtype=np.int32)
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, neg_split))
    np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


if __name__ == '__main__':
  # prepare_pos_c3d()
  # generate_script()
  # prepare_pos_vgg19()
  # shuffle_neg()
  # prepare_neg_c3d()
  prepare_neg_vgg19()