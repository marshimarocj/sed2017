import os

import numpy as np


'''func
'''


'''expr
'''
def flow_dstrb_in_pos_events():
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

  direction = directions[0]
  track_len = track_lens[0]

  event2maxflow = {}
  for name in names:
    track_label_file = os.path.join(track_label_dir, '%s.%d.%s.pos')
    opticalflow_pool_file = os.path.join(opticalflow_pool_dir, '%s.%d.%s.npz'%(name, track_len, direction))
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
      maxflow = data[key][0]

      if id in id2event:
        event = id2event[id]
      else:
        event = 'Background'
      if event not in event2maxflow:
        event2maxflow[event] = []
      event2maxflow[event].append(maxflow)

  for event in event2maxflow:
    maxflows = event2maxflow[event]
    print event, np.mean(maxflows), np.median(maxflows), np.percentile(maxflows, 10), np.percentile(maxflows. 90)


if __name__ == '__main__':
  flow_dstrb_in_pos_events()
