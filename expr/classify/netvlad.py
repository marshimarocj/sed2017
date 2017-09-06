import os
import cPickle
import sys
sys.path.append('../')

import numpy as np

import model.netvlad


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


def gen_proto_cfg(num_ft, dim_ft, num_center):
  return {
    'dim_ft': dim_ft,
    'num_ft': num_ft,
    'num_center': num_center,
    'dim_output': 2048,
    'trn_neg2pos_in_batch': 10,
    'val_neg2pos_in_batch': 1,
  }


def gen_model_cfg(proto_cfg):
  return {
    'proto': proto_cfg,
    'num_class': 5,
  }


'''expr
'''
def generate_label2lid_file():
  root_dir = '/home/jiac/data/sed' # xiaojun
  out_file = os.path.join(root_dir, 'meta', 'label2lid.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump(event2lid, fout)


def class_instance_stat():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  out_file = os.path.join(root_dir, 'meta', 'label_cnt.pkl')

  track_lens = [25, 50]
  out = {}
  for lst_file in lst_files:
    video_names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        video_names.append(name)

    for track_len in track_lens:
      for name in video_names:
        if 'CAM4' in name:
          continue
        print track_len, name
        label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
        id2label = {}
        with open(label_file) as f:
          for line in f:
            line = line.strip()
            data = line.split(' ')
            id = int(data[0])
            label = data[1]
            id2label[id] = label

        label2cnt = {}

        print 'pos'
        ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
        data = np.load(ft_file)
        ids = data['ids']
        pos_ids = set(ids.tolist())
        for id in pos_ids:
          label = id2label[id]
          if label not in label2cnt:
            label2cnt[label] = 0
          label2cnt[label] += 1

        for n in range(10):
          print n
          ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, n))
          data = np.load(ft_file)
          if 'ids' in data:
            ids = data['ids']
            neg_ids = set(ids.tolist())
            label2cnt[n] = len(neg_ids)
          else:
            label2cnt[n] = 0

        out['%s_%d'%(name, track_len)] = label2cnt
  with open(out_file, 'w') as fout:
    cPickle.dump(out, fout)


def num_descriptor_toi_stat():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')

  track_lens = [25, 50]
  for track_len in track_lens:
    nums = []
    for lst_file in lst_files:
      with open(lst_file) as f:
        for line in f:
          line = line.strip()
          name, _ = os.path.splitext(line)
          if 'CAM4' in name:
            continue

          ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
          data = np.load(ft_file)
          if 'ids' not in data:
            continue
          ids = data['ids']
          id2cnt = {}
          for id in ids:
            if id not in id2cnt:
              id2cnt[id] = 0
            id2cnt[id] += 1
          nums += id2cnt.values()
    print track_len, np.median(nums), np.mean(nums), np.percentile(nums, 10), np.percentile(nums, 90)


def prepare_lst_files():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_files = [
    os.path.join(root_dir, 'meta', 'trn.lst'),
    os.path.join(root_dir, 'meta', 'val.lst'),
  ]

  with open(lst_files[0]) as f, open(out_files[0], 'w') as fout:
    for line in f:
      if 'CAM4' not in line:
        fout.write(line)
  with open(lst_files[1]) as f, open(out_files[1], 'w') as fout:
    for l, line in f:
      if l == 5:
        break
      if 'CAM4' not in line:
        fout.write(line)


def prepare_cfg():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
  ]


def tst_reader():
  root_dir = '/home/jiac/data/sed' # xiaojun


if __name__ == "__main__":
  # generate_label2lid_file()
  # class_instance_stat()
  # num_descriptor_toi_stat()
  prepare_lst_files()
