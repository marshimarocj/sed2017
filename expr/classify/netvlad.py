import os
import cPickle
import sys
import json
sys.path.append('../../')

import numpy as np
from sklearn.cluster import KMeans

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
    'trn_neg2pos_in_batch': 5,
    'val_neg2pos_in_batch': 1,
  }


def gen_model_cfg(proto_cfg):
  return {
    'proto': proto_cfg,
    'num_class': 5,

    'learning_rate': 1e-4,
    'monitor_iter': 50,
    'trn_batch_size': 32,
    'tst_batch_size': 128,
    'val_iter': -1,
    'num_epoch': 20,
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
    for l, line in enumerate(f):
      if l == 5:
        break
      if 'CAM4' not in line:
        fout.write(line)


def prepare_cfg():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'meta', 'trn.lst'),
    os.path.join(root_dir, 'meta', 'val.lst'),
  ]
  ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  label2lid_file = os.path.join(root_dir, 'meta', 'label2lid.pkl')
  num_center = 16
  init_weight_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.%d.npz'%num_center)
  out_dir = os.path.join(root_dir, 'model', 'netvlad')
  num_ft = 100
  dim_ft = 1024
  neg_lst = [0]
  track_lens = [50]

  out_prefix = os.path.join(out_dir, 'netvlad.%s.%s'%(
    '_'.join([str(d) for d in neg_lst]), '_'.join([str(d) for d in track_lens])))
  if not os.path.exists(out_prefix):
    os.mkdir(out_prefix)

  proto_cfg = gen_proto_cfg(num_ft, dim_ft, num_center)
  model_cfg = gen_model_cfg(proto_cfg)
  model_cfg_file = '%s.model.json'%out_prefix
  with open(model_cfg_file, 'w') as fout:
    json.dump(model_cfg, fout, indent=2)

  path_cfg = {
    'trn_video_lst_file': lst_files[0],
    'val_video_lst_file': lst_files[1],
    'ft_track_group_dir': ft_toi_dir, 
    'label_dir': label_dir,
    'label2lid_file': label2lid_file,
    'output_dir': out_prefix,
    'neg_lst': neg_lst,
    'track_lens': track_lens,
    'init_weight_file': init_weight_file,
  }
  path_cfg_file = '%s.path.json'%out_prefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def tst_reader():
  root_dir = '/home/jiac/data/sed' # xiaojun
  model_cfg_file = os.path.join(root_dir, 'model', 'netvlad', 'netvlad.0.50.model.json')
  path_cfg_file = os.path.join(root_dir, 'model', 'netvlad', 'netvlad.0.50.path.json')

  model_cfg = model.netvlad.ModelCfg()
  model_cfg.load(model_cfg_file)

  path_cfg = model.netvlad.PathCfg()
  path_cfg.load(path_cfg_file)

  reader = model.netvlad.Reader(
    # path_cfg.val_video_lst_file, path_cfg.ft_track_group_dir, path_cfg.label_dir,
    path_cfg.trn_video_lst_file, path_cfg.ft_track_group_dir, path_cfg.label_dir,
    path_cfg.label2lid_file, model_cfg, 
    neg_lst=path_cfg.neg_lst, track_lens=path_cfg.track_lens)

  print 'init complete'
  print reader.pos_fts.shape, reader.pos_masks.shape, reader.pos_labels.shape
  print reader.pos_idxs[:10]

  # batch_size = 100
  # for epoch in range(20):
  #   print 'epoch', epoch
  #   for fts, masks, labels in reader.yield_trn_batch(batch_size):
  #     print fts.shape, masks.shape, labels.shape
  #   # print np.where(labels > 0)[1]

  # batch_size = 100
  # for fts, masks in reader.yield_tst_batch(batch_size):
  #   print fts.shape, masks.shape


def prepare_init_center_file():
  root_dir = '/home/jiac/data/sed' # xiaojun
  center_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.16.pkl')
  out_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.16.npz')

  with open(center_file) as f:
    kmeans = cPickle.load(f)
  centers = kmeans.cluster_centers_
  centers = centers.T

  np.savez_compressed(out_file, centers=centers)


def prepare_neg_for_val():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_group_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val')

  track_lens = [25, 50]

  for lst_file in lst_files:
    video_names = []
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          video_names.append(name)

    for video_name in video_names:
      print video_name
      for track_len in track_lens:
        pos_ft_file = os.path.join(track_group_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(video_name, track_len))
        data = np.load(pos_ft_file)
        ids = set(data['ids'].tolist())
        num_pos = len(ids)
        del data

        neg_ft_file = os.path.join(track_group_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        data = np.load(neg_ft_file)
        ids = data['ids']
        fts = data['fts']
        centers = data['centers']
        frames = data['frames']

        num = idx.shape[0]
        previd = ids[0]
        cnt = 0
        for i in range(num):
          if ids[i] != previd:
            cnt += 1
            previd = ids[i]
            if cnt == num_pos:
              break
        out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        np.save(out_file, ids=ids[:i], fts=fts[:i], centers=centers[:i], frames=frames[:i])


if __name__ == "__main__":
  # generate_label2lid_file()
  # class_instance_stat()
  # num_descriptor_toi_stat()
  # prepare_lst_files()
  # prepare_cfg()
  # tst_reader()
  # prepare_init_center_file()
  prepare_neg_for_val()
