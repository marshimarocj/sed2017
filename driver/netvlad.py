import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import model.netvlad
import common


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  parser.add_argument('--is_wb', dest='is_wb', type=int, default=True)
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--tst_video_name', dest='tst_video_name')

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.netvlad.PathCfg()
  return common.gen_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(path_cfg, model_cfg_file):
  model_cfg = model.netvlad.ModelCfg()
  model_cfg.load(model_cfg_file)
  data = np.load(path_cfg.init_weight_file)
  model_cfg.proto_cfg.centers = data['centers']
  model_cfg.proto_cfg.alpha = data['alpha']

  return model_cfg


def load_and_fill_wbmodel_cfg(path_cfg, model_cfg_file):
  model_cfg = model.netvlad.ModelWBCfg()
  model_cfg.load(model_cfg_file)
  data = np.load(path_cfg.init_weight_file)
  model_cfg.proto_cfg.centers = data['centers']

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  if bool(opts.is_wb):
    model_cfg = load_and_fill_wbmodel_cfg(path_cfg, opts.model_cfg_file)
    _model = model.netvlad.NetVladWBModel(model_cfg)
  else:
    model_cfg = load_and_fill_model_cfg(path_cfg, opts.model_cfg_file)
    _model = model.netvlad.NetVladModel(model_cfg)

  if opts.is_train:
    with open(os.path.join(path_cfg.log_dir, 'cfg.pkl'), 'w') as fout:
      cPickle.dump(model_cfg, fout)
      cPickle.dump(path_cfg, fout)
      cPickle.dump(opts, fout)

    trntst = model.netvlad.TrnTst(model_cfg, path_cfg, _model)

    trn_reader = model.netvlad.TrnReader(
      path_cfg.trn_video_lst_file, path_cfg.trn_neg_lst_file, path_cfg.trn_ft_track_group_dir, path_cfg.label_dir,
      path_cfg.label2lid_file, model_cfg, track_lens=path_cfg.track_lens)
    val_reader = model.netvlad.ValReader(
      path_cfg.val_video_lst_file, path_cfg.val_ft_track_group_dir, path_cfg.label_dir,
      path_cfg.label2lid_file, model_cfg, track_lens=path_cfg.track_lens)
    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction, resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction)
  else:
    path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred',
      'epoch-%d.%s.npz'%(opts.best_epoch, opts.tst_video_name))
    path_cfg.log_file = None

    trntst = model.netvlad.TrnTst(model_cfg, path_cfg, _model)

    tst_reader = model.netvlad.TstReader(
      opts.tst_video_name, path_cfg.trn_ft_track_group_dir, path_cfg.label_dir,
      path_cfg.label2lid_file, model_cfg,
      track_lens=path_cfg.track_lens)
    trntst.test(tst_reader, memory_fraction=opts.memory_fraction)
