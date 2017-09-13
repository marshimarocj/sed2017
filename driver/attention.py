import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import model.attention
import common


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--tst_video_name', dest='tst_video_name')

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.attention.PathCfg()
  return common.gen_dir_struct_info(path_cfg, path_cfg_file)


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = model.attention.ModelCfg()
  model_cfg.load(opts.model_cfg_file)
  _model = model.attention.AttentionModel(model_cfg)

  if opts.is_train:
    with open(os.path.join(path_cfg.log_dir, 'cfg.pkl'), 'w') as fout:
      cPickle.dump(model_cfg, fout)
      cPickle.dump(path_cfg, fout)
      cPickle.dump(opts, fout)

    trntst = model.attention.TrnTst(model_cfg, path_cfg, _model)

    trn_reader = model.attention.TrnReader(
      path_cfg.trn_video_lst_file, path_cfg.trn_neg_lst_file, path_cfg.trn_ft_track_group_dir, path_cfg.label_dir,
      path_cfg.label2lid_file, model_cfg, track_lens=path_cfg.track_lens)
    val_reader = model.attention.ValReader(
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

    print opts.tst_video_name

    trntst = model.attention.TrnTst(model_cfg, path_cfg, _model)

    tst_reader = model.attention.TstReader(
      opts.tst_video_name, path_cfg.tst_ft_track_group_dir, path_cfg.label_dir,
      path_cfg.label2lid_file, model_cfg,
      track_lens=path_cfg.track_lens)
    trntst.test(tst_reader, memory_fraction=opts.memory_fraction)
