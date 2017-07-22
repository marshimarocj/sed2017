import os

import cv2
import numpy as np


'''func
'''
def resize_paf(paf, dst_size):
  num_channel = paf.shape[0]
  out = []
  for i in range(0, num_channel, 3):
    part = paf[i:i+3]
    num = part.shape[0]
    if num < 3:
      part = np.concatenate([part, np.zeros((1)+part.shape[1:])], axis=0)
    part = np.moveaxis(part, [0, 1, 2], [2, 0, 1])
    _part = cv2.resize(part, (dst_size))
    out.append(_part[:, :, :num])
  paf = np.concatenate(out, axis=2)
  paf = np.moveaxis(paf, [0, 1, 2], [1, 2, 0])
  return paf


'''expr
'''
def merge():
  root_dir = '/home/jiac/data/sed' # danny
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  paf_root_dirs = [
    os.path.join(root_dir, 'paf', '1.0'),
    os.path.join(root_dir, 'paf', '1.5'),
  ]
  out_root_dir = os.path.join(root_dir, 'paf', 'merge_1.0_1.5')

  dst_size = (135, 108)

  videonames = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' in name:
          continue
        videonames.append(name)

  for videoname in videonames:
    paf_dirs = [os.path.join(paf_root_dir, videoname) for paf_root_dir in paf_root_dirs]
    out_dir = os.path.join(out_root_dir, videoname)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    print videoname

    names = os.listdir(paf_dirs[0])
    for name in names:
      paf_file = os.path.join(paf_dirs[0], name)
      paf1 = np.load(paf_file)['fts']
      paf_file = os.path.join(paf_dirs[1], name)
      paf1_5 = np.load(paf_file)['fts']

      num = paf1.shape[0]
      paf_merge = np.zeros(paf1_5.shape, dtype=np.float32)
      for i in range(num):
        r_paf1 = resize_paf(paf1[i], dst_size)
        paf_merge[i] = (r_paf1 + paf1_5[i])/2.0
      paf_merge[paf_merge < 1e-3] = 0

      out_file = os.path.join(out_dir, name)
      np.savez_compressed(out_file, fts=paf_merge)


if __name__ == '__main__':
  merge()
