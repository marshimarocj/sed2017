import os


'''func
'''


'''expr
'''
def missing_videos_in_preprocess():
  root_dir = '/home/jiac/data/sed/video' # xiaojun
  lst_file = os.path.join(root_dir, 'video2017.lst')
  preprocess_root_dir = os.path.join(root_dir, 'dev09_preprocess')

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')
      name = line[:pos]
      preprocess_dir = os.path.join(preprocess_root_dir, name)
      if not os.path.exists(preprocess_dir):
        print name


def mkdir_for_c3d_sync():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          out_dir = os.path.join(root_dir, 'c3d', name)
          os.mkdir(out_dir)


def eval():
  root_dir = '/data1/jiac/sed' # mercurial
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
  # missing_videos_in_preprocess()
  # mkdir_for_c3d_sync()
  eval()
