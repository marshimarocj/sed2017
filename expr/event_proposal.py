import os
import cPickle
import argparse
import sys
sys.path.append('../')

import numpy as np

import api.db


'''func
'''


'''expr
'''
def flow_dstrb_in_events():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_label_dir = os.path.join(root_dir, 'tracklet_label')
  opticalflow_pool_dir = os.path.join(root_dir, 'toi_max_opticalflow')

  directions = ['forward', 'backward']
  track_lens = [25, 50, 100]

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        names.append(name)

  # direction = directions[0]
  # track_len = track_lens[0]
  direction = directions[0]
  track_len = track_lens[2]

  event2maxflow = {}
  for name in names:
    track_label_file = os.path.join(track_label_dir, '%s.%d.%s.pos'%(name, track_len, direction))
    opticalflow_pool_file = os.path.join(opticalflow_pool_dir, '%s.%d.%s.npz'%(name, track_len, direction))
    if not os.path.exists(opticalflow_pool_file):
      continue
    print name

    id2event = {}
    with open(track_label_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        id = int(data[0])
        event = data[1]
        id2event[id] = event

    data = np.load(opticalflow_pool_file)
    for key in data:
      id = int(key)
      maxflow = float(data[key])

      if id in id2event:
        event = id2event[id]
      else:
        event = 'Background'
      if event not in event2maxflow:
        event2maxflow[event] = []
      event2maxflow[event].append(maxflow)

  for event in event2maxflow:
    maxflows = event2maxflow[event]
    print event, np.mean(maxflows), np.median(maxflows), np.percentile(maxflows, 10), np.percentile(maxflows, 90)


def gen_normalize_script():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]

  num_process = 10

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  total = len(names)
  gap = (total + num_process - 1) / num_process

  for i in range(0, total, gap):
    _names = names[i:i+gap]
    out_file = '%d.sh'%(i/gap)
    with open(out_file, 'w') as fout:
      for name in _names:
        cmd = ['python', 'event_proposal.py', name]
        fout.write(' '.join(cmd) + '\n')


def normalize_opticalflow():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  pool_opticalflow_dir = os.path.join(root_dir, 'toi_max_opticalflow')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  # for name in names:
  #   print name

  pool_opticalflow_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npz')
  data = np.load(pool_opticalflow_file)
  num = len(data.keys())
  max_val = np.zeros((num,), dtype=np.float32)
  for key in data:
    val = data[key]
    id = int(key)
    max_val[id] = val
  out_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npy')
  np.save(out_file, max_val)


def filter_out_proposals():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pool_opticalflow_dir = os.path.join(root_dir, 'toi_max_opticalflow')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  threshold = 6.

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  event2preserved = {}
  event2discarded = {}
  for name in names:
    print name

    label_file = os.path.join(label_dir, name + '.pkl')
    with open(label_file) as f:
      pseudo_pos_labels = cPickle.load(f)
    tid2label = {}
    for pseudo_pos_label in pseudo_pos_labels:
      tid = pseudo_pos_label['tid']
      tid2label[tid] = pseudo_pos_label

    track_file = os.path.join(track_dir, name + '.25.forward.npz')
    track_map_file = os.path.join(track_dir, name + '.25.forward.map')
    track_db = api.db.TrackDb(track_map_file, track_file, 25)

    pool_opticalflow_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npz')
    data = np.load(pool_opticalflow_file)
    for key in data:
      tid = int(key)
      max_val = data[key]
      if tid not in tid2label:
        continue

      label = tid2label[tid]
      event = label['event']
      beg = label['beg']
      end = label['end']
      track = track_db.trackid2track[tid]
      center_frame = track.start_frame + 25/2
      pos = (center_frame - beg) / float(end-beg)

      if max_val >= threshold:
        if event not in event2preserved:
          event2preserved[event] = []
        event2preserved[event].append(pos)
      else:
        if event not in event2discarded:
          event2discarded[event] = []
        event2discarded[event].append(pos)

  print 'preserved'
  for event in event2preserved:
    pos = event2perserved[event]
    print event, np.mean(pos), np.median(pos), np.percentile(pos, 10), np.percentile(pos, 90)
  print 'discarded'
  for event in event2discarded:
    pos = event2discarded[event]
    print event, np.mean(pos), np.median(pos), np.percentile(pos, 10), np.percentile(pos, 90)


if __name__ == '__main__':
  # flow_dstrb_in_events()
  # filter_out_proposals()
  normalize_opticalflow()
  # gen_normalize_script()
