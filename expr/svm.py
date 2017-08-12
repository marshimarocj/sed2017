import os
import cPickle
import random
import argparse
import itertools
from ctypes import *

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import liblinear

import sample


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


def prepare_pos_instances(label_file, pos_ft_file, name,
    pos_fts, pos_labels, pos_tids, pos_names):
  tid2lid = {}
  with open(label_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      tid = int(data[0])
      label = data[1]
      if label in event2lid:
        lid = event2lid[label]
        tid2lid[tid] = lid

  data = np.load(pos_ft_file)
  fts = data['vlads']
  ids = data['ids']

  num = ids.shape[0]
  for i in range(num):
    ft = fts[i]
    id = ids[i]
    if id in tid2lid:
      lid = tid2lid[id]
      pos_fts.append(ft)
      pos_labels.append(lid)
      pos_tids.append(id)
      pos_names.append(name)


def load_sampled_neg_ids(neg_id_file):
  track_len2name2ids = {25: {}, 50: {}}
  with open(neg_id_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      track_len = int(data[0])
      id = int(data[1])
      name = data[2]
      if name not in track_len2name2ids[track_len]:
        track_len2name2ids[track_len][name] = set()
      track_len2name2ids[track_len][name].add(id)

  return track_len2name2ids


def prepare_neg_instances(neg_ft_file, name, name2ids,
    neg_fts, neg_ids, neg_names):
  data = np.load(neg_ft_file)
  fts = data['vlads']
  ids = data['ids']

  num = ids.shape[0]
  for i in range(num):
    ft = np.array(fts[i])
    id = ids[i]
    if id in name2ids[name]:
      neg_fts.append(ft)
      neg_ids.append(id)
      neg_names.append(name)


'''expr
'''
def prepare_trn_tst_pos_data():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_files = [
  #   os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
  #   os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
    os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz')
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  # ft_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

  track_lens = [25, 50]

  for s in range(2):
    lst_file = lst_files[s]
    out_file = out_files[s]

    names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        if 'CAM4' in line:
          continue
        name, _ = os.path.splitext(line)
        names.append(name)

    pos_fts = []
    pos_labels = []
    pos_tids = []
    pos_names = []
    for name in names:
      print name
      for track_len in track_lens:
        label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))

        pos_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))

        prepare_pos_instances(label_file, pos_ft_file, name,
          pos_fts, pos_labels, pos_tids, pos_names)
    np.savez_compressed(out_file, fts=pos_fts, labels=pos_labels, ids=pos_tids, names=pos_names)


def sample_neg_ids():
  root_dir = '/data1/jiac/sed' # uranus
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pos_files = [
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
  ]
  multiplier = 5
  ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  out_files = [
    os.path.join(root_dir, 'expr', 'neg.dev08.5.lst'),
    os.path.join(root_dir, 'expr', 'neg.eev08.5.lst'),
  ]

  track_lens = [25, 50]

  for s in range(2):
    lst_file = lst_files[s]
    pos_file = pos_files[s]
    out_file = out_files[s]

    data = np.load(pos_file)
    num_pos = data['labels'].shape[0]
    num_neg = num_pos*multiplier

    rs = sample.ReservoirSampling(num_neg)

    names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        if 'CAM4' in line:
          continue
        name, _ = os.path.splitext(line)
        names.append(name)

    for name in names:
      print name
      for track_len in track_lens:
        neg_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len))
        data = np.load(neg_ft_file)
        fts = data['vlads']
        ids = data['ids']

        num = ids.shape[0]
        for i in range(num):
          ft = np.array(fts[i])
          id = ids[i]
          rs.addData((track_len, id, name))
    data = rs.pool
    with open(out_file, 'w') as fout:
      for d in data:
        fout.write('%d %d %s\n'%(d[0], d[1], d[2]))


def prepare_trn_tst_neg_data():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  neg_id_files = [
    os.path.join(root_dir, 'expr', 'neg.dev08.5.lst'),
    os.path.join(root_dir, 'expr', 'neg.eev08.5.lst'),
  ]
  out_files = [
    # os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz')
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.5.npz')
    os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.neg.5.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.npz')
  ]
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  # ft_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

  track_lens = [25, 50]

  for s in range(2):
    lst_file = lst_files[s]
    neg_id_file = neg_id_files[s]
    out_file = out_files[s]

    track_len2name2ids = load_sampled_neg_ids(neg_id_file)

    names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        if 'CAM4' in line:
          continue
        name, _ = os.path.splitext(line)
        names.append(name)

    neg_fts = []
    neg_ids = []
    neg_names = []
    for name in names:
      print name
      for track_len in track_lens:
        neg_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len))
        name2ids = track_len2name2ids[track_len]
        prepare_neg_instances(neg_ft_file, name, name2ids,
          neg_fts, neg_ids, neg_names)
      print len(neg_ids)

    np.savez_compressed(out_file, fts=neg_fts, ids=neg_ids, names=neg_names)


def prepare_tst_pos_data_with_tracklen_fixed():
  # root_dir = '/data1/jiac/sed' # uranus
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_file = os.path.join(root_dir, 'eev08-1.lst')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_len = 25
  # track_len = 50
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.%d.npz'%track_len)
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.%d.npz'%track_len)
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      if 'CAM4' in line:
        continue
      name, _ = os.path.splitext(line)
      names.append(name)

  pos_fts = []
  pos_labels = []
  pos_tids = []
  pos_names = []
  for name in names:
    print name
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    pos_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
    prepare_pos_instances(label_file, pos_ft_file, name,
      pos_fts, pos_labels, pos_tids, pos_names)
  np.savez_compressed(out_file, fts=pos_fts, labels=pos_labels, ids=pos_tids, names=pos_names)


def prepare_tst_neg_data_with_tracklen_fixed():
  # root_dir = '/data1/jiac/sed' # uranus
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_file = os.path.join(root_dir, 'eev08-1.lst')
  neg_id_file = os.path.join(root_dir, 'expr', 'neg.eev08.5.lst')
  track_len = 25
  # track_len = 50
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.%d.npz'%track_len)
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.%d.npz'%track_len)
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

  track_len2name2ids = load_sampled_neg_ids(neg_id_file)

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      if 'CAM4' in line:
        continue
      name, _ = os.path.splitext(line)
      names.append(name)

  name2ids = track_len2name2ids[track_len]

  neg_fts = []
  neg_ids = []
  neg_names = []
  for name in names:
    print name
    neg_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len))
    prepare_neg_instances(neg_ft_file, name, name2ids,
      neg_fts, neg_ids, neg_names)

  np.savez_compressed(out_file, fts=neg_fts, ids=neg_ids, names=neg_names)


def prepare_trn_data():
  # root_dir = '/data1/jiac/sed' # uranus
  # pos_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz')
  # neg_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz')
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.npz')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # pos_trn_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz')
  # neg_trn_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.5.npz')
  # out_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.npz')
  root_dir = '/home/jiac/data/sed2017' # rocks
  # pos_trn_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.pos.npz')
  # neg_trn_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.neg.5.npz')
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.npz')
  pos_trn_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz')
  neg_trn_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.npz')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.npz')

  data = np.load(pos_trn_file)
  pos_fts = data['fts']
  pos_labels = data['labels']
  pos_ids = data['ids']
  pos_names = data['names']
  num_pos = pos_fts.shape[0]

  data = np.load(neg_trn_file)
  neg_fts = data['fts']
  neg_ids = data['ids']
  neg_names = data['names']
  num_neg = neg_fts.shape[0]

  dim_ft = pos_fts.shape[1]

  fts = np.zeros((num_pos+num_neg, dim_ft), dtype=np.float32)
  labels = np.zeros((num_pos+num_neg,), dtype=np.int32)
  ids = np.zeros((num_pos+num_neg,), dtype=np.int32)
  names = np.concatenate([pos_names, neg_names])

  idxs = np.arange(num_pos+num_neg)
  np.random.shuffle(idxs)
  fts[idxs < num_pos] = pos_fts[idxs[idxs < num_pos]]
  fts[idxs >= num_pos] = neg_fts[idxs[idxs >= num_pos] - num_pos]
  labels[idxs < num_pos]= pos_labels[idxs[idxs < num_pos]]
  ids[idxs < num_pos] = pos_ids[idxs[idxs < num_pos]]
  ids[idxs >= num_pos] = neg_ids[idxs[idxs >= num_pos] - num_pos]
  names = names[idxs]

  np.savez_compressed(out_file, fts=fts, labels=labels, ids=ids, names=names)


def prepare_trn_txt():
  root_dir = '/home/jiac/data/sed2017' # rocks
  trn_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.npz')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.trn')

  data = np.load(trn_file)
  fts = data['fts']
  labels = data['labels']
  num = fts.shape[0]

  with open(out_file, 'w') as fout:
    for i in range(num):
      fout.write('%d '%labels[i])
      ft = fts[i]
      for j, ele in enumerate(ft):
        if ele > 0:
          fout.write('%d:%f '%(j+1, ft[j]))
      fout.write('\n')


def prepare_trn_with_neg_sample():
  # root_dir = '/home/jiac/data/sed2017' # rocks
  # pos_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.pos.npz')
  # lst_file = os.path.join(root_dir, 'dev08-1.lst')
  # ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')
  # neg_splits = [0, 1, 2, 3, 4]
  # s = '_'.join([str(d) for d in neg_splits])
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'vlad.neg.%s.trn'%s)

  root_dir = '/data1/jiac/sed' # uranus
  pos_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz')
  lst_file = os.path.join(root_dir, 'dev08-1.lst')
  ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  # neg_splits = [0]
  neg_splits = [0, 1, 2, 3, 4]
  s = '_'.join([str(d) for d in neg_splits])
  out_file = os.path.join(root_dir, 'expr', 'c3d', 'vlad.neg.%s.trn'%s)

  track_lens = [25, 50]

  with open(out_file, 'w') as fout:
    data = np.load(pos_file)
    fts = data['fts']
    labels = data['labels']

    num = labels.shape[0]
    for i in range(num):
      ft = fts[i]
      label = labels[i]
      fout.write('%d '%label)
      dim_ft = ft.shape[0]
      for j in range(dim_ft):
        if ft[j] > 0:
          fout.write('%d:%f '%(j+1, ft[j]))
      fout.write('\n')

    names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

    for name, split, track_len in itertools.product(names, neg_splits, track_lens):
      print name
      file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, split))
      data = np.load(file)

      fts = data['vlads']
      num = fts.shape[0]
      for i in range(num):
        ft = fts[i]
        dim_ft = ft.shape[0]
        fout.write('0 ')
        for j in range(dim_ft):
          if ft[j] > 0:
            fout.write('%d:%f '%(j+1, ft[j]))
        fout.write('\n')


def prepare_trn_early_fusion_data():
  # root_dir = '/home/jiac/data2/sed' # gpu9
  root_dir = '/home/jiac/data/sed2017' # rocks
  pos_trn_files = [
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz'),
  ]
  neg_trn_files = [
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.neg.5.npz')
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.npz')
  ]
  # out_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'dev08.vlad.npz')
  out_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.npz')

  id2fts = {}
  id2label = {}
  id2cnt = {}

  ft_dims = []
  for pos_trn_file in pos_trn_files:
    data = np.load(pos_trn_file)
    pos_fts = data['fts']
    pos_labels = data['labels']
    pos_ids = data['ids']
    pos_names = data['names']
    num_pos = pos_fts.shape[0]
    ft_dims.append(pos_fts.shape[1])
    unique_keys = set()
    for i in range(num_pos):
      id = pos_ids[i]
      label = pos_labels[i]
      ft = pos_fts[i]
      name = pos_names[i]
      key = '%s_%d'%(name, id)
      if key in unique_keys:
        continue
      unique_keys.add(key)
      if key not in id2fts:
        id2fts[key] = []
        id2cnt[key] = 0
      id2fts[key].append(ft)
      id2label[key] = label
      id2cnt[key] += 1

  for neg_trn_file in neg_trn_files:
    data = np.load(neg_trn_file)
    neg_fts = data['fts']
    neg_ids = data['ids']
    neg_names = data['names']
    num_neg = neg_fts.shape[0]
    unique_keys = set()
    for i in range(num_neg):
      id = neg_ids[i]
      label = 0
      ft = neg_fts[i]
      name = neg_names[i]
      key = '%s_%d'%(name, id)
      if key in unique_keys:
        continue
      unique_keys.add(key)
      if key not in id2fts:
        id2fts[key] = []
        id2cnt[key] = 0
      id2fts[key].append(ft)
      id2label[key] = label
      id2cnt[key] += 1

  print len(id2cnt)
  valid_ids = []
  for id in id2cnt:
    cnt = id2cnt[id]
    if cnt == 2:
      valid_ids.append(id)
  print len(valid_ids)

  random.shuffle(valid_ids)

  num = len(valid_ids)
  fts = np.zeros((num, sum(ft_dims)), dtype=np.float32)
  labels = np.zeros((num,), dtype=np.int32)
  for i, id in enumerate(valid_ids):
    fts[i] = np.concatenate(id2fts[id])
    labels[i] = id2label[id]
  print fts.shape

  ids = []
  names = []
  for id in valid_ids:
    data = id.split('_')
    names.append(data[0])
    ids.append(int(data[1]))
  np.savez_compressed(out_file, fts=fts, labels=labels, ids=ids, names=names)


def prepare_val_early_fusion_data():
  # root_dir = '/home/jiac/data2/sed' # gpu9
  root_dir = '/home/jiac/data/sed2017' # rocks
  pos_tst_files = [
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.25.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.25.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.50.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.50.npz'),
  ]
  neg_tst_files = [
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.npz')
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.25.npz'),
    # os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.25.npz')
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.50.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.50.npz')
  ]
  # out_pos_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.pos.npz')
  # out_neg_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.neg.5.npz')
  # out_pos_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'eev08.vlad.pos.npz')
  # out_neg_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'eev08.vlad.neg.5.npz')
  # out_pos_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.npz')
  # out_neg_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.npz')
  # out_pos_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.25.npz')
  # out_neg_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.25.npz')
  out_pos_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.50.npz')
  out_neg_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.50.npz')

  tst_files = [
    pos_tst_files,
    neg_tst_files,
  ]
  out_files = [
    out_pos_file,
    out_neg_file,
  ]

  for s in range(2):
    out_file = out_files[s]
    id2fts = {}
    if s == 0:
      id2label = {}
    id2cnt = {}

    ft_dims = []
    for tst_file in tst_files[s]:
      data = np.load(tst_file)
      pos_fts = data['fts']
      if s == 0:
        pos_labels = data['labels']
      pos_ids = data['ids']
      pos_names = data['names']
      num_pos = pos_fts.shape[0]
      ft_dims.append(pos_fts.shape[1])
      unique_keys = set()
      for i in range(num_pos):
        id = pos_ids[i]
        if s == 0:
          label = pos_labels[i]
        ft = pos_fts[i]
        name = pos_names[i]
        key = '%s_%d'%(name, id)
        if key in unique_keys:
          continue
        unique_keys.add(key)
        if key not in id2fts:
          id2fts[key] = []
          id2cnt[key] = 0
        id2fts[key].append(ft)
        if s == 0:
          id2label[key] = label
        id2cnt[key] += 1

    valid_ids = []
    for id in id2cnt:
      cnt = id2cnt[id]
      if cnt == 2:
        valid_ids.append(id)

    num = len(valid_ids)
    fts = np.zeros((num, sum(ft_dims)), dtype=np.float32)
    if s == 0:
      labels = np.zeros((num,), dtype=np.int32)
    for i, id in enumerate(valid_ids):
      fts[i] = np.concatenate(id2fts[id])
      if s == 0:
        labels[i] = id2label[id]

    ids = []
    names = []
    for id in valid_ids:
      data = id.split('_')
      names.append(data[0])
      ids.append(int(data[1]))
    if s == 0:
      np.savez_compressed(out_file, fts=fts, labels=labels, ids=ids, names=names)
    else:
      np.savez_compressed(out_file, fts=fts, ids=ids, names=names)


def train_model():
  # root_dir = '/data1/jiac/sed' # uranus
  # trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # trn_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'vgg19', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # trn_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  root_dir = '/home/jiac/data/sed2017' # rocks
  trn_file = os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'lr.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # trn_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'dev08.vlad.npz')
  # out_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # trn_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'dev08.vlad.npz')
  # # out_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # out_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'svm.prob.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  data = np.load(trn_file)
  fts = data['fts']
  labels = data['labels']

  print 'load complete'

  # model = LinearSVC(verbose=1)
  # model = SVC(verbose=1, probability=True)
  model = LogisticRegression(solver='lbfgs', verbose=1)
  model.fit(fts, labels)

  with open(out_file, 'w') as fout:
    cPickle.dump(model, fout)


def train_final_model():
  root_dir = '/home/jiac/data/sed2017' # rocks
  trn_files = [
    # os.path.join(root_dir, 'expr', 'c3d.flow', 'dev08.vlad.npz'),
    # os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.npz'),
  ]
  # out_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'svm.final.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'svm.final.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'svm.final.prob.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  out_file = os.path.join(root_dir, 'expr', 'twostream', 'lr.final.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  fts = []
  labels = []
  for trn_file in trn_files:
    data = np.load(trn_file)
    fts.append(data['fts'])
    labels.append(data['labels'])

  print 'load complete'

  fts = np.concatenate(fts, axis=0)
  labels = np.concatenate(labels, axis=0)

  print 'merge complete'

  # model = LinearSVC(verbose=1)
  # model = SVC(verbose=1, probability=True)
  model = LogisticRegression(solver='lbfgs', verbose=1)
  model.fit(fts, labels)

  with open(out_file, 'w') as fout:
    cPickle.dump(model, fout)


def val_model():
  # root_dir = '/data1/jiac/sed' # uranus
  # # pos_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
  # # neg_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz')
  # # pos_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.25.npz')
  # # neg_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.25.npz')
  # pos_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.50.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.50.npz')
  # model_file = os.path.join(root_dir, 'expr', 'c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  # root_dir = '/home/jiac/data2/sed' # gpu9
  # pos_val_file = os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.5.npz')
  # model_file = os.path.join(root_dir, 'expr', 'vgg19', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # pos_val_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.neg.5.npz')
  # model_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  root_dir = '/home/jiac/data/sed2017' # rocks
  # pos_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.npz')
  # pos_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.25.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.25.npz')
  # pos_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.50.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.50.npz')
  # model_file = os.path.join(root_dir, 'expr', 'twostream', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # pos_val_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'eev08.vlad.neg.5.npz')
  # model_file = os.path.join(root_dir, 'expr', 'vgg19.flow', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # pos_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.npz')
  # pos_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.25.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.25.npz')
  pos_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.pos.50.npz')
  neg_val_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'eev08.vlad.neg.5.50.npz')
  model_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  with open(model_file) as f:
    model = cPickle.load(f)

  data = np.load(pos_val_file)
  fts = data['fts']
  pos_labels = data['labels']

  pos_predicts = model.decision_function(fts)

  del fts
  data = np.load(neg_val_file)
  fts = data['fts']
  neg_predicts = model.decision_function(fts)
  num_neg = neg_predicts.shape[0]
  neg_labels = np.zeros((num_neg,), dtype=np.int32)

  predicts = np.concatenate([pos_predicts, neg_predicts], axis=0)
  labels = np.concatenate([pos_labels, neg_labels])

  events = {}
  for event in event2lid:
    lid = event2lid[event]
    events[lid] = event

  for c in range(1, 5):
    ap = average_precision_score(labels == c, predicts[:, c])
    print events[c], ap


def val_model_on_full():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_file = os.path.join(root_dir, 'eev08-1.lst')

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      if 'CAM4' in line:
        continue
      name, _ = os.path.splitext(line)
      names.append(name)


def gen_predict_script():
  # root_dir = '/home/jiac/data/sed2017' # rocks
  root_dir = '/data1/jiac/sed' # uranus
  lst_file = os.path.join(root_dir, 'eev08-1.lst')
  out_file = 'predict_eev.sh'

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      if 'CAM4' not in name:
        names.append(name)

  with open(out_file, 'w') as fout:
    for name in names:
      cmd = ['python', 'svm.py', name]
      fout.write(' '.join(cmd) + '\n')


def predict_on_eev():
  # root_dir = '/home/jiac/data/sed2017' # rocks
  # vlad_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')
  # model_file = os.path.join(root_dir, 'expr', 'twostream', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  root_dir = '/data1/jiac/sed' # uranus
  vlad_dir = os.path.join(root_dir, 'c3d', 'vlad')
  model_file = os.path.join(root_dir, 'expr', 'c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()

  name = args.name

  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08_full', name + '.npz')
  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08_full', name + '.raw.npz')
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08_full', name + '.npz')
  out_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08_full', name + '.raw.npz')

  with open(model_file) as f:
    model = cPickle.load(f)

  files = [os.path.join(vlad_dir, '%s.25.forward.backward.square.pos.0.75.npz'%name)] + \
    [os.path.join(vlad_dir, '%s.25.forward.backward.square.neg.0.50.%d.npz'%(name, split)) for split in range(10)]
  ids = []
  predicts = []
  for file in files:
    print file
    data = np.load(file)
    _ids = data['ids']
    _vlads = data['vlads']
    _predicts = model.decision_function(_vlads)
    # _predicts = np.exp(_predicts)
    # _predicts = _predicts / np.sum(_predicts, axis=1, keepdims=True)
    ids.append(_ids)
    predicts.append(_predicts)
  ids = np.concatenate(ids, 0)
  predicts = np.concatenate(predicts, 0)
  np.savez_compressed(out_file, predicts=predicts, ids=ids)


def predict_liblinear_on_eev():
  # root_dir = '/home/jiac/data/sed2017' # rocks
  # vlad_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')
  # model_file = os.path.join(root_dir, 'expr', 'twostream', 'vlad.neg.0.model')
  root_dir = '/data1/jiac/sed' # uranus
  vlad_dir = os.path.join(root_dir, 'c3d', 'vlad')
  model_file = os.path.join(root_dir, 'expr', 'c3d', 'vlad.neg.0.model')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()

  name = args.name

  # out_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08_full', name + '.neg.0.raw.npz')
  out_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08_full', name + '.neg.0.raw.npz')

  model = liblinear.liblinear.load_model(model_file)

  files = [os.path.join(vlad_dir, '%s.25.forward.backward.square.pos.0.75.npz'%name)] + \
    [os.path.join(vlad_dir, '%s.25.forward.backward.square.neg.0.50.%d.npz'%(name, split)) for split in range(10)]
  ids = []
  predicts = []
  for file in files:
    print file
    data = np.load(file)
    _ids = data['ids']
    _vlads = data['vlads']
    num = _ids.shape[0]
    for i in range(num):
      id = _ids[i]
      vlad = _vlads[i]
      ft_dict = {}
      for j, ele in enumerate(vlad):
        if ele > 0:
          ft_dict[j+1] = ele
      x, _ = liblinear.gen_feature_nodearray(ft_dict)
      # label = liblinear.liblinear.predict(model, x)
      dec_values = (c_double * 5)()
      liblinear.liblinear.predict_values(model, x, dec_values)
      predict = dec_values[:5]

      ids.append(id)
      predicts.append(predict)
  np.savez_compressed(out_file, predicts=predicts, ids=ids)


def predict_on_tst2017():
  root_dir = '/home/jiac/data2/sed' # gpu9
  lst_file = os.path.join(root_dir, '2017.refined.lst')
  vlad_dirs = [
    os.path.join(root_dir, 'c3d', 'tst2017', 'vlad'),
    os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'tst2017', 'vlad'),
  ]
  # model_file = os.path.join(root_dir, 'expr', 'flow', 'svm.final.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # out_dir = os.path.join(root_dir, 'expr', 'flow', 'tst2017')
  model_file = os.path.join(root_dir, 'expr', 'c3d.flow', 'svm.final.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  out_dir = os.path.join(root_dir, 'expr', 'c3d.flow', 'tst2017')

  with open(model_file) as f:
    model = cPickle.load(f)

  with open(lst_file) as f:
    for line in f:
      name = line.strip()
      print name

      id2fts = {}
      id2cnt = {}
      for vlad_dir in vlad_dirs:
        file = os.path.join(vlad_dir, '%s.25.forward.square.npz'%name)
        data = np.load(file)
        ids = data['ids']
        vlads = data['vlads']
        num = ids.shape[0]
        unique_ids = set() # hack, guard against duplicate bug
        for i in range(num):
          id = ids[i]
          if id in unique_ids:
            continue
          if id not in id2fts:
            id2fts[id] = []
            id2cnt[id] = 0
          id2fts[id].append(vlads[i])
          id2cnt[id] += 1

      vlads = []
      ids = []
      for id in id2fts:
        if id2cnt[id] == len(vlad_dirs):
          ft = np.concatenate(id2fts[id])
          vlads.append(ft)
          ids.append(id)

      predicts = model.decision_function(vlads)
      predicts = np.exp(predicts)
      predicts = predicts / np.sum(predicts, axis=1, keepdims=True)

      out_file = os.path.join(out_dir, name + '.npz')
      np.savez_compressed(out_file, predicts=predicts, ids=ids)


def eval_full():
  root_dir = '/home/jiac/data/sed2017' # rocks
  # root_dir = '/data1/jiac/sed' # uranus
  lst_file = os.path.join(root_dir, 'eev08-1.lst')
  predict_dir = os.path.join(root_dir, 'expr', 'twostream', 'eev08_full')
  pos_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz')
  # predict_dir = os.path.join(root_dir, 'expr', 'c3d', 'eev08_full')
  # pos_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')

  videonames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      videoname, _ = os.path.splitext(line)
      if 'CAM4' not in videoname:
        videonames.append(videoname)

  data = np.load(pos_file)
  ids = data['ids']
  names = data['names']
  labels = data['labels']
  num = ids.shape[0]
  pos_key2label = {}
  for i in range(num):
    key = '%s_%d'%(names[i], ids[i])
    pos_key2label[key] = labels[i]

  predicts = []
  labels = []
  for videoname in videonames:
    print videoname
    # predict_file = os.path.join(predict_dir, videoname + '.npz')
    # predict_file = os.path.join(predict_dir, videoname + '.raw.npz')
    predict_file = os.path.join(predict_dir, videoname + '.neg.0.raw.npz')
    data = np.load(predict_file)
    _predicts = data['predicts']
    _ids = data['ids']
    num = _ids.shape[0]
    predicts.append(_predicts)
    for i in range(num):
      key = '%s_%d'%(videoname, _ids[i])
      if key in pos_key2label:
        label = pos_key2label[key]
      else:
        label = 0
      labels.append(label)

  predicts = np.concatenate(predicts, 0)
  labels = np.array(labels)

  events = {}
  for event in event2lid:
    lid = event2lid[event]
    events[lid] = event

  for c in range(1, 5):
    ap = average_precision_score(labels == c, predicts[:, c])
    print events[c], ap


if __name__ == '__main__':
  # prepare_trn_tst_pos_data()
  # sample_neg_ids()
  # prepare_tst_pos_data_with_tracklen_fixed()
  # prepare_tst_neg_data_with_tracklen_fixed()
  # prepare_trn_tst_neg_data()
  # prepare_trn_data()
  prepare_trn_txt()
  # prepare_trn_with_neg_sample()
  # prepare_trn_early_fusion_data()
  # prepare_val_early_fusion_data()
  # train_model()
  # train_final_model()
  # val_model()
  # predict_on_eev()
  # predict_liblinear_on_eev()
  # gen_predict_script()
  # eval_full()
  # predict_on_tst2017()
