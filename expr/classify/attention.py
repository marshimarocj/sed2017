import os
import cPickle
import sys
import json
sys.path.append('../../')

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
import mxnet as mx

import model.attention


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4,
}


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


def gen_proto_cfg(num_ft, dim_ft):
  return {
    'dim_ft': dim_ft,
    'num_ft': num_ft,
    'num_attention': 5,
    'trn_neg2pos_in_batch': 5,
    'val_neg2pos_in_batch': 1,
  }


def gen_model_cfg(proto_cfg, num_hidden):
  return {
    'proto': proto_cfg,
    'num_class': 5,
    'num_hidden': num_hidden,

    'learning_rate': 1e-4,
    'monitor_iter': 50,
    'trn_batch_size': 32,
    'tst_batch_size': 128,
    'val_iter': -1,
    'num_epoch': 20,
  }


'''expr
'''
def prepare_cfg():
  root_dir = '/home/jiac/data/sed' # danny
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
  out_dir = os.path.join(root_dir, 'expr', 'attention')
  num_ft = 100
  dim_ft = 1024
  num_hidden = 512
  tst_neg_lst = [0]
  track_lens = [25]

  out_prefix = os.path.join(out_dir, 'attention.l2norm_input.0.%s.%d'%(
    '_'.join([str(d) for d in track_lens]), num_hidden))
  if not os.path.exists(out_prefix):
    os.mkdir(out_prefix)

  proto_cfg = gen_proto_cfg(num_ft, dim_ft)
  proto_cfg['l2_norm_input'] = True
  model_cfg = gen_model_cfg(proto_cfg, num_hidden)
  model_cfg['trn_batch_size'] = 32
  model_cfg['tst_batch_size'] = 128
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
  }
  path_cfg_file = '%s.path.json'%out_prefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_tst_script():
  root_dir = '/home/jiac/data/sed' # danny
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  expr_name = 'attention.l2norm_input.0.25.512'
  expr_dir = os.path.join(root_dir, 'expr', 'attention', expr_name)
  model_cfg_file = '%s.model.json'%expr_dir
  path_cfg_file = '%s.path.json'%expr_dir
  out_file = '../../driver/tst.sh'

  gpu = 0

  best_epoch = select_best_epoch_from_dir(os.path.join(expr_dir, 'log'))
  print best_epoch

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
        'python', 'attention.py', 
        model_cfg_file, path_cfg_file, 
        '--is_train', '0',
        '--best_epoch' , str(best_epoch),
        '--tst_video_name', name,
      ]
      fout.write(' '.join(cmd) + '\n')


def eval():
  root_dir = '/home/jiac/data/sed' # danny
  lst_file = os.path.join(root_dir, 'meta', 'val.lst')
  expr_name = 'attention.l2norm_input.0.25.512'
  predict_dir = os.path.join(root_dir, 'expr', 'attention', expr_name, 'pred')

  best_epoch = 2

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


if __name__ == '__main__':
  # prepare_cfg()
  # gen_tst_script()
  eval()
