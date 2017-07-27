import os

import numpy as np


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
    os.path.join(root_dir, 'expr', 'c3d', 'dev08.vlad.npz'),
    os.path.join(root_dir, 'expr', 'c3d', 'eev08.vlad.npz')
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
        name, _ = os.path.splitext(line)
        names.append(name)

    pos_fts = []
    pos_labels = []
    pos_tids = []
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
    np.savez_compressed(out_file, fts=pos_fts, labels=pos_labels, ids=pos_tids)


if __name__ == '__main__':
  prepare_trn_tst_pos_data()
