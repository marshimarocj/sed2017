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
  root_dir = '/data1/jiac/sed' # uranus
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_files = [
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.pos.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.pos.npz')
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')
  ft_dir = os.path.join(root_dir, 'c3d', 'vlad')

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
  out_files = [
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.neg.%d.npz'%multiplier),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.neg.%d.npz'%multiplier)
  ]
  ft_dir = os.path.join(root_dir, 'c3d', 'vlad')

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
        data = np.load(pos_ft_file)
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


if __name__ == '__main__':
  prepare_trn_tst_pos_data()
