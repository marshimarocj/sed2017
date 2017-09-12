import os
import cPickle
import sys
import json
sys.path.append('../../')

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
import mxnet as mx

import model.netvlad


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


def select_best_epoch(file):
  with open(file) as f:
    data = cPickle.load(f)
  best_epoch = data[0]['epoch'] 
  min_loss = data[0]['loss']
  for d in data:
    if d['loss'] < min_loss:
      min_loss = d['loss']
      best_epoch = d['epoch']

  return best_epoch


def select_best_epoch_from_dir(log_dir):
  names = os.listdir(log_dir)
  min_loss = 1e10
  best_epoch = -1
  for name in names:
    if 'val_metrics' in name:
      file = os.path.join(log_dir, name)
      with open(file) as f:
        data = json.load(f)
        if data['loss'] < min_loss:
          best_epoch = data['epoch']

  return best_epoch


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
    # 'num_epoch': 20,
    'num_epoch': 10,
  }


def gen_focal_loss_model_cfg(proto_cfg, num_class, alphas):
  return {
    'proto': proto_cfg,
    'num_class': num_class,
    'gamma': 2,
    'alphas': alphas,

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
      if 'CAM4' not in line:
        fout.write(line)


def prepare_cfg():
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # root_dir = '/usr0/home/jiac/data/sed' # aladdin3 
  root_dir = '/home/jiac/data/sed' # danny
  video_lst_files = [
    os.path.join(root_dir, 'meta', 'trn.lst'),
    # os.path.join(root_dir, 'meta', 'debug.lst'),
    os.path.join(root_dir, 'meta', 'val.lst'),
  ]
  # trn_neg_lst_file = os.path.join(root_dir, 'meta', 'trn_neg.lst')
  # trn_neg_lst_file = os.path.join(root_dir, 'meta', 'trn_neg.25.lst')
  # trn_neg_lst_file = os.path.join(root_dir, 'meta', 'trn_neg.50.lst')
  trn_neg_lst_file = os.path.join(root_dir, 'meta', 'trn_neg.25.tfrecords.lst')
  trn_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split')
  val_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val')
  tst_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_tst')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  label2lid_file = os.path.join(root_dir, 'meta', 'label2lid.pkl')
  num_center = 16
  # num_center = 32
  # num_center = 8
  # init_weight_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.%d.npz'%num_center)
  init_weight_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.norm.%d.npz'%num_center)
  out_dir = os.path.join(root_dir, 'expr', 'netvlad')
  num_ft = 100
  dim_ft = 1024
  tst_neg_lst = [0]
  # track_lens = [25, 50]
  # track_lens = [50]
  track_lens = [25]
  gamma = 2

  # out_prefix = os.path.join(out_dir, 'netvlad.0.%s'%(
  #   '_'.join([str(d) for d in track_lens])))
  # out_prefix = os.path.join(out_dir, 'netvlad.0.%s.%d'%(
  # out_prefix = os.path.join(out_dir, 'netvlad.l2norm_output.0.%s.%d'%(
  # out_prefix = os.path.join(out_dir, 'netvlad.l2norm_input.0.%s.%d'%(
  # out_prefix = os.path.join(out_dir, 'netvlad.l2norm_input.dropout.0.%s.%d'%(
  # out_prefix = os.path.join(out_dir, 'netvlad.l2norm_input.l2norm_output.0.%s.%d'%(
  out_prefix = os.path.join(out_dir, 'netvlad.l2norm_input.dropin.0.%s.%d'%(
    '_'.join([str(d) for d in track_lens]), num_center))
  if not os.path.exists(out_prefix):
    os.mkdir(out_prefix)

  proto_cfg = gen_proto_cfg(num_ft, dim_ft, num_center)
  proto_cfg['l2_norm_input'] = True
  # proto_cfg['l2_norm_output'] = True
  # proto_cfg['dropin'] = True
  model_cfg = gen_model_cfg(proto_cfg)
  model_cfg['trn_batch_size'] = 32
  model_cfg['tst_batch_size'] = 128
  # model_cfg['dropout'] = True
  model_cfg['num_epoch'] = 20
  model_cfg_file = '%s.model.json'%out_prefix
  with open(model_cfg_file, 'w') as fout:
    json.dump(model_cfg, fout, indent=2)

  path_cfg = {
    'trn_video_lst_file': video_lst_files[0],
    'val_video_lst_file': video_lst_files[1],
    'trn_neg_lst_file': trn_neg_lst_file,
    'trn_ft_track_group_dir': trn_ft_toi_dir, 
    'val_ft_track_group_dir': val_ft_toi_dir, 
    'tst_ft_track_group_dir': tst_ft_toi_dir,
    'label_dir': label_dir,
    'label2lid_file': label2lid_file,
    'output_dir': out_prefix,
    'tst_neg_lst': tst_neg_lst,
    'track_lens': track_lens,
    'init_weight_file': init_weight_file,
    # 'model_file': os.path.join(out_prefix, 'model', 'epoch-9'),
  }
  path_cfg_file = '%s.path.json'%out_prefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def prepare_focalloss_cfg():
  root_dir = '/data1/jiac/sed' # uranus
  video_lst_files = [
    os.path.join(root_dir, 'meta', 'trn.lst'),
    os.path.join(root_dir, 'meta', 'val.lst'),
  ]
  trn_neg_lst_file = os.path.join(root_dir, 'meta', 'trn_neg.25.tfrecords.lst')
  trn_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split')
  val_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val')
  tst_ft_toi_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_tst')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  label2lid_file = os.path.join(root_dir, 'meta', 'label2lid.pkl')
  num_center = 16
  # num_center = 32
  # num_center = 8
  # init_weight_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.%d.npz'%num_center)
  init_weight_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.norm.%d.npz'%num_center)
  out_dir = os.path.join(root_dir, 'expr', 'netvlad')
  num_ft = 100
  dim_ft = 1024
  tst_neg_lst = [0]
  track_lens = [25]
  gamma = 2
  num_class = 5

  out_prefix = os.path.join(out_dir, 'netvlad.l2norm_input.focalloss.0.%s.%d.%d'%(
    '_'.join([str(d) for d in track_lens]), num_center, gamma))
  # out_prefix = os.path.join(out_dir, 'netvlad.focalloss.0.%s.%d.%d'%(
    # '_'.join([str(d) for d in track_lens]), num_center, gamma))
  if not os.path.exists(out_prefix):
    os.mkdir(out_prefix)

  proto_cfg = gen_proto_cfg(num_ft, dim_ft, num_center)
  proto_cfg['l2_norm_input'] = True
  # proto_cfg['l2_norm_output'] = True
  # proto_cfg['dropin'] = True
  neg2pos = proto_cfg['trn_neg2pos_in_batch']
  alphas = [neg2pos] + [1./(num_class-1) for _ in range(num_class-1)]
  alphas = np.array(alphas)
  alphas = 1./alphas
  alphas = self.num_class / np.sum(alphas) * alphas
  model_cfg = gen_focal_loss_model_cfg(proto_cfg, num_class, alphas)
  model_cfg['trn_batch_size'] = 32
  model_cfg['tst_batch_size'] = 128
  model_cfg['gamma'] = gamma
  model_cfg['learning_rate'] = 1e-4
  # model_cfg['dropout'] = True
  model_cfg['num_epoch'] = 20
  model_cfg_file = '%s.model.json'%out_prefix
  with open(model_cfg_file, 'w') as fout:
    json.dump(model_cfg, fout, indent=2)

  path_cfg = {
    'trn_video_lst_file': video_lst_files[0],
    'val_video_lst_file': video_lst_files[1],
    'trn_neg_lst_file': trn_neg_lst_file,
    'trn_ft_track_group_dir': trn_ft_toi_dir, 
    'val_ft_track_group_dir': val_ft_toi_dir, 
    'tst_ft_track_group_dir': tst_ft_toi_dir,
    'label_dir': label_dir,
    'label2lid_file': label2lid_file,
    'output_dir': out_prefix,
    'tst_neg_lst': tst_neg_lst,
    'track_lens': track_lens,
    'init_weight_file': init_weight_file,
    # 'model_file': os.path.join(out_prefix, 'model', 'epoch-9'),
  }
  path_cfg_file = '%s.path.json'%out_prefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def tst_trn_reader():
  root_dir = '/home/jiac/data/sed' # xiaojun
  # model_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.50.model.json')
  # path_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.50.path.json')
  model_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.25.model.json')
  path_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.25.path.json')

  model_cfg = model.netvlad.ModelCfg()
  model_cfg.load(model_cfg_file)

  path_cfg = model.netvlad.PathCfg()
  path_cfg.load(path_cfg_file)

  reader = model.netvlad.TrnReader(
    path_cfg.trn_video_lst_file, path_cfg.trn_neg_lst_file,
    path_cfg.trn_ft_track_group_dir, path_cfg.label_dir,
    path_cfg.label2lid_file, model_cfg, 
    track_lens=path_cfg.track_lens)

  print 'init complete'
  print reader.pos_fts.shape, reader.pos_masks.shape, reader.pos_labels.shape
  print reader.pos_idxs[:10]

  batch_size = 100
  for epoch in range(20):
    print 'epoch', epoch
    for fts, masks, labels in reader.yield_trn_batch(batch_size):
      print fts.shape, masks.shape, labels.shape
    # print np.where(labels > 0)[1]

  # batch_size = 100
  # for fts, masks in reader.yield_tst_batch(batch_size):
  #   print fts.shape, masks.shape


def tst_val_reader():
  root_dir = '/home/jiac/data/sed' # xiaojun
  model_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.50.model.json')
  path_cfg_file = os.path.join(root_dir, 'expr', 'netvlad', 'netvlad.0.50.path.json')

  model_cfg = model.netvlad.ModelCfg()
  model_cfg.load(model_cfg_file)

  path_cfg = model.netvlad.PathCfg()
  path_cfg.load(path_cfg_file)

  reader = model.netvlad.ValReader(
    path_cfg.val_video_lst_file, path_cfg.val_ft_track_group_dir, path_cfg.label_dir,
    path_cfg.label2lid_file, model_cfg,
    track_lens=path_cfg.track_lens)

  print 'init complete'
  print reader.pos_fts.shape, reader.pos_masks.shape, reader.pos_labels.shape
  print reader.pos_idxs[:10]

  batch_size = 100
  for fts, masks, labels in reader.yield_val_batch(batch_size):
    print fts.shape, masks.shape, labels.shape


def prepare_init_center_file():
  root_dir = '/home/jiac/data/sed' # xiaojun
  # center_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.16.pkl')
  # out_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.16.npz')
  # center_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.pkl')
  # out_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.32.npz')
  # center_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.8.pkl')
  # out_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.8.npz')
  center_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.norm.16.pkl')
  out_file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'kmeans.center.norm.16.npz')

  with open(center_file) as f:
    kmeans = cPickle.load(f)
  centers = kmeans.cluster_centers_
  centers = centers.T

  np.savez_compressed(out_file, centers=centers)


def prepare_neg_for_val():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    # os.path.join(root_dir, 'dev08-1.lst'),
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

        # out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(video_name, track_len))
        # os.symlink(pos_ft_file, out_file)

        neg_ft_file = os.path.join(track_group_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        data = np.load(neg_ft_file)
        ids = data['ids']
        fts = data['fts']
        centers = data['centers']
        frames = data['frames']

        num = num_pos*2
        previd = ids[0]
        cnt = 0
        for i in range(ids.shape[0]):
          if ids[i] != previd:
            cnt += 1
            previd = ids[i]
            if cnt == num:
              break
        out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        np.savez_compressed(out_file, ids=ids[:i], fts=fts[:i], centers=centers[:i], frames=frames[:i])


def split_neg_for_trn():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_group_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split')

  # num_id_in_chunk = 1000
  num_id_in_chunk = 500
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
        neg_ft_file = os.path.join(track_group_dir, 
          '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        data = np.load(neg_ft_file)
        ids = data['ids']
        fts = data['fts']
        frames = data['frames']
        centers = data['centers']
        num = ids.shape[0]

        _ids, _fts, _frames, _centers = [], [], [], []
        chunk = 0
        previd = ids[0]
        cnt = 0
        for i in range(num):
          if ids[i] != previd:
            previd = ids[i]
            cnt += 1

            if cnt % num_id_in_chunk == 0:
              out_file = os.path.join(out_dir, 
                '%s.%d.forward.backward.square.neg.0.50.0.%d.npz'%(video_name, track_len, chunk))
              np.savez_compressed(out_file, ids=_ids, fts=_fts, frames=_frames, centers=_centers)
              chunk += 1
              del _ids, _fts, _frames, _centers
              _ids, _fts, _frames, _centers = [], [], [], []
          _ids.append(ids[i])
          _fts.append(fts[i])
          _frames.append(frames[i])
          _centers.append(centers[i])
        out_file = os.path.join(out_dir,
          '%s.%d.forward.backward.square.neg.0.50.0.%d.npz'%(video_name, track_len, chunk))
        np.savez_compressed(out_file, ids=_ids, fts=_fts, frames=_frames, centers=_centers)


def lnk_pos_for_trn():
  root_dir = '/home/jiac/data/sed' # xiaojun
  # lst_file = os.path.join(root_dir, 'meta', 'trn.lst')
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  # src_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group')
  src_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split')
  dst_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val')

  track_lens = [25, 50]

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      for track_len in track_lens:
        src_file = os.path.join(src_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
        dst_file = os.path.join(dst_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
        os.symlink(src_file, dst_file)


def gen_neg_lst_for_trn():
  # root_dir = '/home/jiac/data/sed' # xiaojun
  root_dir = '/home/jiac/data/sed' # danny
  lst_file = os.path.join(root_dir, 'meta', 'trn.lst')
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split')
  # out_file = os.path.join(root_dir, 'meta', 'trn_neg.lst')
  out_file = os.path.join(root_dir, 'meta', 'trn_neg.25.tfrecords.lst')

  video_names = set()
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      video_names.add(name)

  names = os.listdir(ft_dir)
  out = []
  for name in names:
    if 'tfrecords' not in name:
      continue
    pos = name.find('.')
    video_name = name[:pos]
    if video_name in video_names and 'neg' in name:
      out.append(name)

  with open(out_file, 'w') as fout:
    for name in out:
      fout.write(name + '\n')


def neg_lst_split_by_track_len():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_file = os.path.join(root_dir, 'meta', 'trn_neg.lst')
  # lst_file = os.path.join(root_dir, 'meta', 'val_neg.lst')
  out_files = [
    os.path.join(root_dir, 'meta', 'trn_neg.25.lst'),
    os.path.join(root_dir, 'meta', 'trn_neg.50.lst'),
    # os.path.join(root_dir, 'meta', 'val_neg.25.lst'),
    # os.path.join(root_dir, 'meta', 'val_neg.50.lst'),
  ]

  with open(lst_file) as f, open(out_files[0], 'w') as fout25, open(out_files[1], 'w') as fout50:
    for line in f:
      line = line.strip()
      data = line.split('.')
      if data[1] == '25':
        fout25.write(line + '\n')
      else:
        fout50.write(line + '\n')


def prepare_tst_files():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  ft_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group') 
  pos_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.pos.25.npz')
  neg_file = os.path.join(root_dir, 'expr', 'twostream', 'eev08.vlad.neg.5.25.npz')
  out_dir = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_tst') 

  track_lens = [25, 50]

  name2ids = {}
  for file in [pos_file, neg_file]:
    data = np.load(file)
    ids = data['ids']
    names = data['names']
    for id, name in zip(ids, names):
      if name not in name2ids:
        name2ids[name] = set()
      name2ids[name].add(id)
  print name2ids.keys()

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      print name
      if name not in name2ids:
        continue
      valid_ids = name2ids[name]

      for track_len in track_lens:
        print name, track_len
        src_files = [
          os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len)),
          os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.0.npz'%(name, track_len)),
        ]
        dst_files = [
          os.path.join(out_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len)),
          os.path.join(out_dir, '%s.%d.forward.backward.square.neg.0.50.0.5.npz'%(name, track_len)),
        ]

        for src_file, dst_file in zip(src_files, dst_files):
          data = np.load(src_file)
          frames = data['frames']
          fts = data['fts']
          centers = data['centers']
          ids = data['ids']
          num = ids.shape[0]

          out_frames = []
          out_fts = []
          out_centers = []
          out_ids = []
          for i in range(num):
            if ids[i] in valid_ids:
              out_frames.append(frames[i])
              out_fts.append(fts[i])
              out_centers.append(centers[i])
              out_ids.append(ids[i])
          np.savez_compressed(dst_file, 
            frames=out_frames, fts=out_fts, centers=out_centers, ids=out_ids)
      break


def gen_tst_script():
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  root_dir = '/home/jiac/data/sed' # danny
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  # expr_name = 'netvlad.0.50'
  # expr_name = 'netvlad.0.25'
  # expr_name = 'netvlad.0.25.32'
  # expr_name = 'netvlad.0.25.8'
  # expr_name = 'netvlad.l2norm.0.25.16'
  # expr_name = 'netvlad.l2norm_input.0.25.16'
  # expr_name = 'netvlad.0.25_50'
  # expr_name = 'netvlad.l2norm_input.dropout.0.25.16'
  expr_name = 'netvlad.l2norm_input.l2norm_output.0.25.16'
  expr_dir = os.path.join(root_dir, 'expr', 'netvlad', expr_name)
  model_cfg_file = '%s.model.json'%expr_dir
  path_cfg_file = '%s.path.json'%expr_dir
  out_file = '../../driver/tst.sh'

  gpu = 0

  val_file = os.path.join(expr_dir, 'log', 'val_metrics.pkl')
  # best_epoch = select_best_epoch(val_file)
  best_epoch = select_best_epoch_from_dir(os.path.join(expr_dir, 'log'))
  # best_epoch = 9

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      if 'CAM4' not in name:
        names.append(name)

  with open(out_file, 'w') as fout:
    fout.write('export CUDA_VISIBLE_DEVICES=%d\n'%gpu)
    for name in names:
      cmd = [
        # 'python', 'netvlad.py', 
        'python', 'netvlad_clean.py', 
        model_cfg_file, path_cfg_file, 
        '--is_train', '0',
        '--best_epoch' , str(best_epoch),
        '--tst_video_name', name,
      ]
      fout.write(' '.join(cmd) + '\n')


def eval():
  # root_dir = '/home/jiac/data/sed' # xiaojun
  # root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  root_dir = '/home/jiac/data/sed' # danny
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  # expr_name = 'netvlad.0.50'
  # expr_name = 'netvlad.0.25'
  # expr_name = 'netvlad.0.25.32'
  # expr_name = 'netvlad.0.25.8'
  # expr_name = 'netvlad.l2norm_input.0.25.16'
  # expr_name = 'netvlad.0.25_50'
  # expr_name = 'netvlad.l2norm_input.dropout.0.25.16'
  expr_name = 'netvlad.l2norm_input.l2norm_output.0.25.16'
  predict_dir = os.path.join(root_dir, 'expr', 'netvlad', expr_name, 'pred')

  # best_epoch = 0
  # best_epoch = 5
  best_epoch = 2
  # best_epoch = 9
  # best_epoch = 1

  predicts = []
  labels = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      if 'CAM4' in name:
        continue

      predict_file = os.path.join(predict_dir, 'epoch-%d.%s.npz'%(best_epoch, name))
      data = np.load(predict_file)
      _logits = data['logits']
      _predicts = mx.nd.softmax(mx.nd.array(_logits)).asnumpy()
      _labels = data['labels']
      predicts.append(_predicts)
      labels.append(_labels)
  predicts = np.concatenate(predicts, axis=0)
  labels = np.concatenate(labels, axis=0)

  events = {}
  for event in event2lid:
    lid = event2lid[event]
    events[lid] = event

  for c in range(1, 5):
    ap = average_precision_score(labels[:, c], predicts[:, c])
    print events[c], ap*100


if __name__ == "__main__":
  # generate_label2lid_file()
  # class_instance_stat()
  # num_descriptor_toi_stat()
  # prepare_lst_files()
  # prepare_cfg()
  prepare_focalloss_cfg()
  # tst_trn_reader()
  # tst_val_reader()
  # prepare_init_center_file()
  # prepare_neg_for_val()
  # split_neg_for_trn()
  # lnk_pos_for_trn()
  # gen_neg_lst_for_trn()
  # neg_lst_split_by_track_len()
  # prepare_tst_files()
  # gen_tst_script()
  # eval()
