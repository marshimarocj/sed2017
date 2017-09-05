import os
import cPickle


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

        ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.pos.0.75.npz'%(name, track_len))
        data = np.load(ft_file)
        ids = data['ids']
        pos_ids = set(ids.tolist())
        label2cnt = {}
        for id in pos_ids:
          label = id2label[id]
          if label not in label2cnt:
            label2cnt[label] = 0
          label2cnt[label] += 1

        for n in range(10):
          ft_file = os.path.join(ft_dir, '%s.%d.forward.backward.square.neg.0.50.%d.npz'%(name, track_len, n))
          data = np.load(ft_file)
          ids = data['ids']
          neg_ids = set(ids.tolist())
          label2cnt[n] = len(neg_ids)

        out['%s_%d'%(name, track_len)] = label2cnt
  with open(out_file, 'w') as fout:
    cPickle.dump(out, fout)


if __name__ == "__main__":
  # generate_label2lid_file()
  class_instance_stat()
