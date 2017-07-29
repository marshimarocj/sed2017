import os
import cPickle
import random

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score

import sample


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


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

        pos_ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
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
    np.savez_compressed(out_file, fts=pos_fts, labels=pos_labels, ids=pos_tids, names=pos_names)


def prepare_trn_tst_neg_data():
  # root_dir = '/data1/jiac/sed' # uranus
  # root_dir = '/home/jiac/data2/sed' # uranus
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pos_files = [
  #   os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
  #   os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
    os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.npz')
  ]
  multiplier = 5
  out_files = [
    # os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.%d.npz'%multiplier),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.%d.npz'%multiplier)
    # os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.%d.npz'%multiplier),
    # os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.%d.npz'%multiplier)
    os.path.join(root_dir, 'expr', 'twostream', 'dev08.vlad.neg.%d.npz'%multiplier),
    os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.%d.npz'%multiplier)
  ]
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  # ft_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'vlad')

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
          rs.addData((ft, id, name))

    data = rs.pool
    neg_fts = [d[0] for d in data]
    neg_ids = [d[1] for d in data]
    neg_names = [d[2] for d in data]
    np.savez_compressed(out_file, fts=neg_fts, ids=neg_ids, names=neg_names)


def prepare_trn_data():
  # root_dir = '/data1/jiac/sed' # uranus
  # pos_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz')
  # neg_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz')
  # out_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.npz')
  root_dir = '/home/jiac/data2/sed' # gpu9
  pos_trn_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz')
  neg_trn_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.5.npz')
  out_file = os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.npz')

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


def prepare_trn_early_fusion_data():
  root_dir = '/home/jiac/data2/sed' # gpu9
  pos_trn_files = [
    os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
  ]
  neg_trn_files = [
    os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.5.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz')
  ]
  out_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'dev08.vlad.npz')

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
    for i in range(num_pos):
      id = pos_ids[i]
      label = pos_labels[i]
      ft = pos_fts[i]
      name = pos_names[i]
      key = '%s_%d'%(name, id)
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
    for i in range(num_neg):
      id = neg_ids[i]
      label = 0
      ft = neg_fts[i]
      name = neg_names[i]
      key = '%s_%d'%(name, id)
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
  root_dir = '/home/jiac/data2/sed' # gpu9
  pos_tst_files = [
    os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz'),
  ]
  neg_tst_files = [
    os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.5.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz')
  ]
  out_pos_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.pos.npz')
  out_neg_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.neg.5.npz')

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
      for i in range(num_pos):
        id = pos_ids[i]
        if s == 0:
          label = pos_labels[i]
        ft = pos_fts[i]
        name = pos_names[i]
        key = '%s_%d'%(name, id)
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
  root_dir = '/home/jiac/data2/sed' # gpu9
  trn_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'dev08.vlad.npz')
  out_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

  data = np.load(trn_file)
  fts = data['fts']
  labels = data['labels']

  print 'load complete'

  model = LinearSVC(verbose=1)
  model.fit(fts, labels)

  with open(out_file, 'w') as fout:
    cPickle.dump(model, fout)


def val_model():
  # root_dir = '/data1/jiac/sed' # uranus
  # pos_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.5.npz')
  # model_file = os.path.join(root_dir, 'expr', 'c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  # root_dir = '/home/jiac/data2/sed' # gpu9
  # pos_val_file = os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
  # neg_val_file = os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.5.npz')
  # model_file = os.path.join(root_dir, 'expr', 'vgg19', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')
  root_dir = '/home/jiac/data2/sed' # gpu9
  pos_val_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.pos.npz')
  neg_val_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'eev08.vlad.neg.5.npz')
  model_file = os.path.join(root_dir, 'expr', 'vgg19.c3d', 'svm.CellToEar.Embrace.Pointing.PersonRuns.pkl')

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


if __name__ == '__main__':
  # prepare_trn_tst_pos_data()
  prepare_trn_tst_neg_data()
  # prepare_trn_data()
  # prepare_trn_early_fusion_data()
  # prepare_val_early_fusion_data()
  # train_model()
  # val_model()
