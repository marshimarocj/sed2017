import os

import numpy as np

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
  root_dir = '/home/jiac/data2/sed' # gpu9
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_files = [
  #   os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
  #   os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
    os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  ft_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')

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
  root_dir = '/home/jiac/data2/sed' # uranus
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pos_files = [
  #   os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
  #   os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
    os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.pos.npz')
  ]
  multiplier = 5
  out_files = [
    # os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.%d.npz'%multiplier),
    # os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.%d.npz'%multiplier)
    os.path.join(root_dir, 'expr', 'vgg19', 'dev08.vlad.neg.%d.npz'%multiplier),
    os.path.join(root_dir, 'expr', 'vgg19', 'eev08.vlad.neg.%d.npz'%multiplier)
  ]
  # ft_dir = os.path.join(root_dir, 'c3d', 'vlad')
  ft_dir = os.path.join(root_dir, 'vgg19_pool5_fullres', 'vlad')

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
  root_dir = '/data1/jiac/sed' # uranus
  pos_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz')
  neg_trn_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.5.npz')
  out_file = os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.npz')

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
  names = np.zeros((num_pos+num_neg,), dtype=np.int32)

  idxs = np.random.shuffle(range(num_pos+num_neg))
  fts[idxs < num_pos] = pos_fts[idxs[idxs < num_pos]]
  fts[idxs >= num_pos] = neg_fts[idxs[idxs >= num_pos] - num_pos]
  labels[idxs < num_pos]= pos_labels[idxs[idxs < num_pos]]
  ids[idxs < num_pos] = pos_ids[idxs[idxs < num_pos]]
  ids[ids >= num_pos] = neg_ids[idxs[idxs >= num_pos] - num_pos]
  names[idxs < num_pos] = pos_names[idxs[idxs < num_pos]]
  names[idxs >= num_pos] = neg_names[idxs[idxs >= num_pos] - num_pos]

  np.savez_compressed(out_file, fts=fts, labels=labels, ids=ids, names=names)


def train_model():
  root_dir = '/data1/jiac/sed' # uranus


if __name__ == '__main__':
  # prepare_trn_tst_pos_data()
  # prepare_trn_tst_neg_data()
  prepare_trn_data()
