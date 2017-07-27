import os
import argparse

import numpy as np


'''func
'''


'''expr
'''
def interpolate_to_align():
  root_dir = '/home/jiac/data/sed'
  # lst_files = [
  #   os.path.join(root_dir, 'dev08-1.lst'),
  #   os.path.join(root_dir, 'eev08-1.lst'),
  # ]
  ft_root_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame')
  out_root_dir = os.path.join('/data/extDisk1/jiac/sed', 'twostream', 'feat_anet_flow_5frame')
 
  # names = []
  # for lst_file in lst_files:
  #   with open(lst_file) as f:
  #     for line in f:
  #       line = line.strip()
  #       if 'CAM4' in line:
  #         continue
  #       name, _  = os.path.splitext(line)
  #       names.append(name)

  parser = argparse.ArgumentParser()
  parser.add_argument('name');
  args = parser.parse_args()
  name = args.name

  src_ft_gap = 6
  dst_ft_gap = 5
  chunk_gap = 7500

  # for name in names:
  out_dir = os.path.join(out_root_dir, name)
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  print name
  ft_dir = os.path.join(ft_root_dir, name)
  names = os.listdir(ft_dir)
  chunks = [int(name.split('.')[0]) for name in names]
  chunks = sorted(chunks)

  base_frame = 0
  last_frame = 0
  last_ft = None
  fts = []
  for chunk in chunks:
    print chunk
    chunk_file = os.path.join(ft_dir, '%d.npz'%chunk)
    data = np.load(chunk_file)
    _fts = data['fts']
    num = _fts.shape[0]
    for i in range(num):
      frame = last_frame + src_ft_gap
      last_idx = last_frame / dst_ft_gap
      current_idx = frame / dst_ft_gap
      if last_ft is None: # initial case
        fts.append(_fts[i])
      else:
        for j in range(last_idx, current_idx):
          interpolate_frame = (j+1)*dst_ft_gap
          if interpolate_frame % chunk_gap == 0:
            _chunk = interpolate_frame/chunk_gap - 1
            out_file = os.path.join(out_dir, '%d.npz'%(_chunk*chunk_gap))
            np.savez_compressed(out_file, fts=fts)
            del fts
            fts = []
          a = (frame - interpolate_frame) / float(frame - last_frame)
          b = (interpolate_frame - last_frame) / float(frame - last_frame)
          ft = last_ft * a + _fts[i] * b
          fts.append(ft)
      last_ft = _fts[i]
      last_frame = frame

  if len(fts) > 0:
    _chunk = interpolate_frame / chunk_gap
    out_file = os.path.join(out_dir, '%d.npz'%(_chunk*chunk_gap))
    np.savez_compressed(out_file, fts=fts)
    del fts
    fts = []


def gen_script():
  root_dir = '/home/jiac/data/sed'
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        if 'CAM4' in line:
          continue
        name, _  = os.path.splitext(line)
        names.append(name)

  num_process = 4
  gap = (len(names) + num_process - 1) / num_process
  for i in range(num_process):
    _names = names[i*gap:(i+1)*gap]
    out_file = '%d.sh'%i
    with open(out_file, 'w') as fout:
      for name in _names:
        cmd = ['python', 'twostream.py', name]
        fout.write(' '.join(cmd) + '\n')


if __name__ == '__main__':
  interpolate_to_align()
  # gen_script()
