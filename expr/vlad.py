import os
import itertools
import cPickle
import argparse
import sys
sys.path.append('../')

import numpy as np
from sklearn.cluster import KMeans

import api.db
import sample


'''func
'''
def encode(fts, kmeans):
  center_idxs = kmeans.predict(fts)
  centers = kmeans.cluster_centers_
  diffs = []
  for i in range(kmeans.n_clusters):
    idx = np.nonzero(center_idxs==i)[0]
    diff = fts[idx] - np.expand_dims(centers[i], 0)
    if diff.shape[0] > 0:
      diff = np.sum(diff, 0)
    else:
      diff = np.zeros(centers[i].shape)
    diffs.append(diff)
  vlad = np.concatenate(diffs)
  norm = np.linalg.norm(vlad)
  if norm > 0:
    vlad /= norm
  return vlad


def _encode_vlad(file, out_file, kmeans):
  data = np.load(file)
  fts = data['fts']
  centers = data['centers']
  ids = data['ids']
  num = ids.shape[0]

  prev_id = ids[0]
  _fts = []
  out_vlads = []
  out_ids = []
  for i in range(num):
    id = ids[i]
    if id != prev_id:
      vlad = encode(np.array(_fts), kmeans)
      out_vlads.append(vlad)
      out_ids.append(prev_id)
      prev_id = id
      del _fts
      _fts = []
    _fts.append(fts[i])
  if len(_fts) > 0:
    vlad = encode(np.array(_fts), kmeans)
    out_vlads.append(vlad)
    out_ids.append(id)
    del _fts
    _fts = []

  np.savez_compressed(out_file, vlads=out_vlads, ids=out_ids)


'''expr
'''
def sample_data_for_center():
  # root_dir = '/data1/jiac/sed' # uranus
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  root_dir = '/home/jiac/data/sed' # danny
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  num_sample = 10000
  out_file = os.path.join(ft_root_dir, 'sample.%d.npy'%num_sample)

  names = [
    'LGW_20071101_E1_CAM1',
    'LGW_20071106_E1_CAM2',
    'LGW_20071107_E1_CAM3',
    'LGW_20071108_E1_CAM5',
  ]

  rs = sample.ReservoirSampling(num_sample)

  for name in names:
    ft_dir = os.path.join(ft_root_dir, name)
    # ft_db = api.db.C3DFtDb(ft_dir)
    # ft_db = api.db.VGG19FtDb(ft_dir)
    ft_db = api.db.FlowFtDb(ft_dir)
    print name

    for chunk in ft_db.chunks:
      print chunk
      fts = ft_db.load_chunk(chunk)
      shape = fts.shape
      for i, j, k in itertools.product(range(shape[0]), range(shape[2]), range(shape[3])):
        rs.addData(np.array(fts[i, :, j, k]))
      del fts

  data = rs.pool
  np.save(out_file, data)


def cluster_centers():
  # root_dir = '/data1/jiac/sed' # uranus
  # ft_root_dir = os.path.join(root_dir, 'c3d')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres')
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  root_dir = '/home/jiac/data/sed' # danny
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  sample_file = os.path.join(ft_root_dir, 'sample.10000.npy')
  out_file = os.path.join(ft_root_dir, 'kmeans.center.32.pkl')

  num_center = 32
  kmeans = KMeans(n_clusters=num_center)

  data = np.load(sample_file)
  kmeans.fit(data)

  # cluster_centers = kmeans.cluster_centers_
  # np.save(out_file, cluster_centers)
  with open(out_file, 'w') as fout:
    cPickle.dump(kmeans, fout)


def encode_vlad():
  # root_dir = '/data1/jiac/sed' # uranus
  # ft_root_dir = os.path.join(root_dir, 'c3d', 'track_group')
  # kmeans_file = os.path.join(root_dir, 'c3d', 'kmeans.center.32.pkl')
  # ft_root_dir = os.path.join(root_dir, 'c3d', 'tst2017', 'track_group')
  # kmeans_file = os.path.join(root_dir, 'c3d', 'kmeans.center.32.pkl')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # # ft_root_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'track_group')
  # # kmeans_file = os.path.join(root_dir, 'vgg19_pool5_fullres', 'kmeans.center.32.pkl')
  # # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'tst2017', 'track_group')
  # # kmeans_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  # ft_root_dir = os.path.join(root_dir, 'c3d', 'tst2017', 'track_group')
  # kmeans_file = os.path.join(root_dir, 'c3d', 'kmeans.center.32.pkl')
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  # kmeans_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  root_dir = '/home/jiac/data/sed2017' # rocks
  # # root_dir = '/home/jiac/data/sed' # danny
  # ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  # kmeans_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    # os.path.join(root_dir, 'eev08-1.lst'),
    # os.path.join(root_dir, '2017.refined.lst'),
  ]
  # out_dir = os.path.join(root_dir, 'c3d', 'vlad')
  # out_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')
  # out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'tst2017', 'vlad')
  # out_dir = os.path.join(root_dir, 'c3d', 'tst2017', 'vlad')

  track_lens = [25, 50]
  # track_lens = [25]

  with open(kmeans_file) as f:
    kmeans = cPickle.load(f)

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        if 'CAM4' not in line:
          name, _ = os.path.splitext(line)
          names.append(name)
        # names.append(line)

  for name in names:
    for track_len in track_lens:
      files = [
        # os.path.join(ft_root_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len)),
        # os.path.join(ft_root_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len)),
        os.path.join(ft_root_dir, '%s.%d.forward.backward.square.neg.0.50.%s.npz'%(name, track_len, split)) for split in range(1, 10)
        # os.path.join(ft_root_dir, '%s.%d.forward.square.npz'%(name, track_len))
      ]
      out_files = [
        # os.path.join(out_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len)),
        # os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len)),
        os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, split)) for split in range(1, 10)
        # os.path.join(out_dir, '%s.%d.forward.square.npz'%(name, track_len))
      ]
      print name

      for file, out_file in zip(files, out_files):
        if os.path.exists(out_file):
          continue

        _encode_vlad(file, out_file, kmeans)


def gen_script():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_file = 'vlad.sh'

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  with open(out_file, 'w') as fout:
    for name in names:
      cmd = ['python', 'vlad.py', name]
      fout.write(' '.join(cmd) + '\n')


def encode_vlad_rocks():
  root_dir = '/home/jiac/data/sed2017' # rocks
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  kmeans_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()

  name = args.name

  # track_lens = [25, 50]
  track_lens = [50]

  with open(kmeans_file) as f:
    kmeans = cPickle.load(f)

  for track_len in track_lens:
    file = os.path.join(ft_root_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len))
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len))

    _encode_vlad(file, out_file, kmeans)


if __name__ == '__main__':
  # sample_data_for_center()
  # sample_data_for_twostream_sync_center()
  # cluster_centers()
  encode_vlad()
  # gen_script()
  # encode_vlad_rocks()
