import os
import argparse
import random
import itertools
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


def flow_threshold_func(qbegin, qend, tbegin, tend):
  return qbegin >= tbegin and qend <= tend


def _prepare_neg_ft(label_file, track_db_file, ft_dir, out_file, ft='c3d'):
  neg_trackids = []
  with open(label_file) as f:
    for line in f:
      line = line.strip()
      neg_trackids.append(int(line))

  track_db = api.db.TrackDb()
  track_db.load(track_db_file, neg_trackids)

  if ft == 'c3d':
    ft_db = api.db.C3DFtDb(ft_dir)
    center_grid = api.db.C3DFtCenters()
    threshold_func = c3d_threshold_func
  elif ft == 'flow':
    ft_db = api.db.FlowFtDb(ft_dir)
    center_grid = api.db.FlowFtCenters()
    threshold_func = flow_threshold_func
  elif ft == 'vgg':
    ft_db = api.db.VGG19FtDb(ft_dir)
    center_grid = api.db.VggFtCenters()

  if ft == 'c3d' or ft == 'flow':
    neg_ft_in_track_generator = api.generator.crop_duration_ft_in_track(
      track_db, ft_db, center_grid, threshold_func)
  elif ft == 'vgg':
    neg_ft_in_track_generator = api.generator.crop_instant_ft_in_track(
      track_db, ft_db, center_grid)
  fts = []
  frames = []
  centers = []
  ids = []
  for ft_in_track in neg_ft_in_track_generator:
    num = len(ft_in_track.frames)
    fts.append(ft_in_track.fts)
    frames.extend(ft_in_track.frames)
    centers.append(ft_in_track.centers)
    ids.extend(num*[ft_in_track.id])

  fts = np.concatenate(fts, 0)
  frames = np.array(frames, dtype=np.int32)
  centers = np.concatenate(centers, 0)
  ids = np.array(ids, dtype=np.int32)
  np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


'''expr
'''
def prepare_pos_ft():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data/sed' # danny
  # root_dir = '/home/jiac/data2/sed' # gpu9
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  # out_dir = os.path.join(root_dir, 'c3d', 'track_group')
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  # out_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'track_group')

  # track_len = 25
  # track_len = 50
  track_lens = [25, 50]

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  # center_grid = api.db.C3DFtCenters()
  # threshold_func = c3d_threshold_func
  center_grid = api.db.FlowFtCenters()
  threshold_func = flow_threshold_func
  # center_grid = api.db.VggFtCenters()

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    id2event = load_pos_track_label_file(label_file)
    pos_trackids = id2event.keys()

    db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file, pos_trackids)

    ft_dir = os.path.join(ft_root_dir, name)
    # ft_db = api.db.C3DFtDb(ft_dir)
    ft_db = api.db.FlowFtDb(ft_dir)
    # ft_db = api.db.VggFtDb(ft_dir)

    pos_ft_in_track_generator = api.generator.crop_duration_ft_in_track(
      track_db, ft_db, center_grid, threshold_func)
    # pos_ft_in_track_generator = api.generator.crop_instant_ft_in_track(
    #   track_db, ft_db, center_grid)
    fts = []
    frames = []
    centers = []
    ids = []
    for ft_in_track in pos_ft_in_track_generator:
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


def prepare_neg_ft():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # root_dir = '/home/jiac/data/sed' # danny
  # root_dir = '/home/jiac/data2/sed' # gpu9
  root_dir = '/home/jiac/data/sed2017' # rocks
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  # out_dir = os.path.join(root_dir, 'c3d', 'track_group')
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  # out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  out_dir = os.path.join('/data/MM22/jiac/sed2017', 'twostream', 'feat_anet_flow_6frame', 'track_group')
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  # out_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'track_group')

  track_lens = [25, 50]
  # neg_split = 1

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  parser.add_argument('neg_split', type=int)
  args = parser.parse_args()
  name = args.name
  neg_split = args.neg_split

  for track_len in track_lens:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg.%d'%(name, track_len, neg_split))
    track_db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    ft_dir = os.path.join(ft_root_dir, name)
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, neg_split))
    _prepare_neg_ft(label_file, track_db_file, ft_dir, out_file, ft='flow')


def check_track_group_npzfile():
  root_dir = '/home/jiac/data/sed2017' # rocks
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  kmeans_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  lst_file = os.path.join(root_dir, 'dev08-1.lst')
  track_lens = [25, 50]
  outfile = 'prepare_toi_data.failed.sh'

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      if 'CAM4' not in line:
        name, _ = os.path.splitext(line)
        names.append(name)

  with open(outfile, 'w') as fout:
    for name in names:
      for track_len in track_lens:
        for split in range(1, 10):
          file = os.path.join(ft_root_dir, '%s.%d.forward.backward.square.neg.0.50.%s.npz'%(name, track_len, split))
          try:
            data = np.load(file)
          except:
            cmd = ['python', 'prepare_toi_data.py', name, str(split), str(track_len)]
            fout.write(' '.join(cmd) + '\n')


def prepare_neg_ft_missing():
  root_dir = '/home/jiac/data/sed2017' # rocks
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  out_dir = os.path.join('/data/MM22/jiac/sed2017', 'twostream', 'feat_anet_flow_6frame', 'track_group')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  parser.add_argument('neg_split', type=int)
  parser.add_argument('track_len', type=int)
  args = parser.parse_args()
  name = args.name
  neg_split = args.neg_split
  track_len = args.track_len

  label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg.%d'%(name, track_len, neg_split))
  track_db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
  ft_dir = os.path.join(ft_root_dir, name)
  out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, neg_split))
  _prepare_neg_ft(label_file, track_db_file, ft_dir, out_file, ft='flow')


def remove_neg_data_in_dev_for_consistency():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_file = os.path.join(root_dir, 'dev08-1.lst')
  toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      if 'CAM4' not in name:
        names.append(name)

  track_lens = [25, 50]

  for name in names:
    for track_len in track_lens:
      for split in range(1, 10):
        file = os.path.join(toi_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, split))
        if os.path.exists(file):
          os.remove(file)


def prepare_neg_ft_on_all_splits():
  root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data2/sed' # gpu9
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d')
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  lst_files = [
    # os.path.join(root_dir, 'eev08-1.lst'),
    os.path.join(root_dir, 'dev08-1.lst'),
  ]
  out_dir = os.path.join(ft_root_dir, 'track_group')

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  # track_len = 25
  track_lens = [25, 50]

  parser = argparse.ArgumentParser()
  parser.add_argument('neg_split', type=int)
  args = parser.parse_args()
  neg_split = args.neg_split

  for name, track_len in itertools.product(names, track_lens):
    print name
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg.%d'%(name, track_len, neg_split))
    track_db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    ft_dir = os.path.join(ft_root_dir, name)
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, neg_split))
    _prepare_neg_ft(label_file, track_db_file, ft_dir, out_file, ft='c3d')
    # _prepare_neg_ft(label_file, track_db_file, ft_dir, out_file, ft='vgg')


def generate_script():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data/sed' # danny
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # root_dir = '/home/jiac/data/sed' # gpu9
  root_dir = '/home/jiac/data3/sed' # gpu4
  lst_files = [
    # os.path.join(root_dir, 'dev08-1.lst'),
    # os.path.join(root_dir, 'eev08-1.lst'),
    os.path.join(root_dir, '2017.refined.lst')
    # os.path.join(root_dir, 'video', '2017.refined.lst')
  ]

  num_process = 5
  # num_process = 3

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        # name, _ = os.path.splitext(line)
        name = line
        # pos = line.find('.')
        # name = line[:pos]
        if 'CAM4' in name:
          continue
        names.append(name)

  num = len(names)
  gap = (num + num_process - 1) / num_process

  for i in range(0, num, gap):
    idx = i/gap
    # out_file = 'prepare_toi_data.%d.sh'%idx
    out_file = 'prepare_toi_data.tst.%d.sh'%idx
    with open(out_file, 'w') as fout:
      for j in range(i, min(i+gap, num)):
        name = names[j]
        cmd = [
          'python', 'prepare_toi_data.py', name
        ]
        fout.write(' '.join(cmd) + '\n')


def gen_script_rocks():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]

  out_file = 'prepare_toi_data.sh'
  with open(out_file, 'w') as fout:
  #   with open(lst_files[0]) as f:
  #     for line in f:
  #       line = line.strip()
  #       name, _ = os.path.splitext(line)
  #       if 'CAM4' in name:
  #         continue
  #       cmd = [
  #         'python', 'prepare_toi_data.py', name, '0'
  #       ]
  #       fout.write(' '.join(cmd) + '\n')
    with open(lst_files[0]) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' in name:
          continue
        # for s in range(1):
        for s in range(1, 10):
          cmd = [
            'python', 'prepare_toi_data.py', name, str(s)
          ]
          fout.write(' '.join(cmd) + '\n')


def retrieve_failed_jobs():
  root_dir = '/home/jiac/data/sed2017' # rocks
  file = 'prepare_toi_data.sh'
  out_file = 'prepare_toi_data.failed.sh'

  cmds = []
  with open(file) as f:
    for i, line in enumerate(f):
      job_err_file = 'toi.job.err-%d'%(i+1)
      if os.path.getsize(job_err_file) > 0:
        line = line.strip()
        cmds.append(line)

  with open(out_file, 'w') as fout:
    for cmd in cmds:
      fout.write(cmd + '\n')


def prepare_toi_ft_for_tst():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data2/sed' # gpu9
  # root_dir = '/home/jiac/data3/sed' # gpu4
  track_dir = os.path.join(root_dir, 'tracking', 'tst2017')
  # track_dir = os.path.join(root_dir, 'tst2017', 'tracking')
  ft_root_dir = os.path.join(root_dir, 'c3d', 'tst2017')
  out_dir = os.path.join(root_dir, 'c3d', 'tst2017', 'track_group')
  # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'tst2017')
  # out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'tst2017', 'track_group')
  # ft_root_dir = os.path.join(root_dir, 'tst2017', 'vgg19_pool5_fullres')
  # out_dir = os.path.join(root_dir, 'tst2017', 'vgg19_pool5_fullres', 'track_group')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  center_grid = api.db.C3DFtCenters()
  threshold_func = c3d_threshold_func
  # center_grid = api.db.FlowFtCenters()
  # threshold_func = flow_threshold_func
  # center_grid = api.db.VggFtCenters()

  db_file = os.path.join(track_dir, '%s.25.forward.square.npz'%name)
  track_db = api.db.TrackDb()
  track_db.load(db_file)

  ft_dir = os.path.join(ft_root_dir, name + '.mov.deint')
  ft_db = api.db.C3DFtDb(ft_dir)
  # ft_dir = os.path.join(ft_root_dir, name)
  # ft_db = api.db.FlowFtDb(ft_dir)
  # ft_db = api.db.VGG19FtDb(ft_dir)

  out_file = os.path.join(out_dir, '%s.25.forward.square.npz'%name)
  if os.path.exists(out_file):
    return

  ft_in_track_generator = api.generator.crop_duration_ft_in_track(
    track_db, ft_db, center_grid, threshold_func)
  # ft_in_track_generator = api.generator.crop_instant_ft_in_track(
  #   track_db, ft_db, center_grid)
  fts = []
  frames = []
  centers = []
  ids = []
  for ft_in_track in ft_in_track_generator:
    num = len(ft_in_track.frames)
    fts.append(ft_in_track.fts)
    frames.extend(ft_in_track.frames)
    centers.append(ft_in_track.centers)
    ids.extend(num*[ft_in_track.id])

  fts = np.concatenate(fts, 0)
  frames = np.array(frames, dtype=np.int32)
  centers = np.concatenate(centers, 0)
  ids = np.array(ids, dtype=np.int32)
  out_file = os.path.join(out_dir, '%s.25.forward.square.npz'%name)
  np.savez_compressed(out_file, fts=fts, frames=frames, centers=centers, ids=ids)


if __name__ == '__main__':
  # prepare_pos_ft()
  # generate_script()
  gen_script_rocks()
  # retrieve_failed_jobs()
  # prepare_pos_vgg19()
  # shuffle_neg()
  # prepare_neg_ft()
  # check_track_group_npzfile()
  # prepare_neg_ft_missing()
  # remove_neg_data_in_dev_for_consistency()
  # prepare_neg_ft_on_all_splits()
  # prepare_neg_vgg19()
  # prepare_toi_ft_for_tst()
