import os
import cPickle
import argparse
import sys
sys.path.append('../')

import numpy as np
from scipy.stats import pearsonr

import api.db


'''func
'''
def forward_backward_threshold(qbegin, qend, tbegin, tend):
  return (qbegin == tbegin) and (qend == tend)


def threshold_25_50(qbegin, qend, tbegin, tend):
  ibegin = max(qbegin, tbegin)
  iend = min(qend, tend)
  return (ibegin == qbegin) and (iend == qend)


def calc_iou(box_lhs, boxs, union_or_min=False):
  xmin = np.maximum(box_lhs[0], boxs[:,0])
  ymin = np.maximum(box_lhs[1], boxs[:,1])
  xmax = np.minimum(box_lhs[2], boxs[:,2])
  ymax = np.minimum(box_lhs[3], boxs[:,3])
  intersect_width = xmax - xmin
  intersect_height = ymax - ymin
  is_intersect = np.logical_and(intersect_width > 0, intersect_height > 0)
  area_intersect = np.where(is_intersect, 
    intersect_width * intersect_height, np.zeros(intersect_width.shape))
  area = (box_lhs[2] - box_lhs[0]) * (box_lhs[3] - box_lhs[1])
  areas = (boxs[:, 3] - boxs[:, 1])*(boxs[:, 2] - boxs[:, 0])

  if area == 0:
    iou = np.zeros(area_intersect.shape)
  else:
    if union_or_min:
      iou = area_intersect / (float(area) + areas - area_intersect)
    else:
      iou = area_intersect / \
        np.where(np.minimum(float(area), areas) == 0, 
          1, np.minimum(float(area), areas))

  return iou


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
  # root_dir = '/home/jiac/data/sed' # gpu4
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pool_opticalflow_dir = os.path.join(root_dir, 'toi_max_opticalflow')

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()
  name = args.name

  # names = []
  # for lst_file in lst_files:
  #   with open(lst_file) as f:
  #     for line in f:
  #       line = line.strip()
  #       name, _ = os.path.splitext(line)
  #       if 'CAM4' not in name:
  #         names.append(name)

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

  threshold = 3.

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
  for name in names[:-1]:
  # for name in names[:10]:
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

    pool_opticalflow_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npy')
    data = np.load(pool_opticalflow_file)
    for tid, max_val in enumerate(data):
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
    pos = event2preserved[event]
    print event, np.histogram(pos, range=(0, 1.))[0]
    # print event, np.mean(pos), np.median(pos), np.percentile(pos, 10), np.percentile(pos, 90)
  print 'discarded'
  for event in event2discarded:
    pos = event2discarded[event]
    print event, np.histogram(pos, range=(0, 1.))[0]
    # print event, np.mean(pos), np.median(pos), np.percentile(pos, 10), np.percentile(pos, 90)


def event_recall_change_filter_out_proposals():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pool_opticalflow_dir = os.path.join(root_dir, 'toi_max_opticalflow')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  threshold = 3.
  # threshold = 6.

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  num_events = 0
  num_recalled_events = 0
  for name in names[:-1]:
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

    pool_opticalflow_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npy')
    data = np.load(pool_opticalflow_file)
    events = set()
    recalled_events = set()
    for tid, max_val in enumerate(data):
      if tid not in tid2label:
        continue

      label = tid2label[tid]
      event = label['event']
      beg = label['beg']
      end = label['end']
      eventid = '%d_%d_%s'%(beg, end, event)
      events.add(eventid)

      if max_val >= threshold:
        recalled_events.add(eventid)
    num_events += len(events)
    num_recalled_events += len(recalled_events)
    print len(recalled_events), len(events)

  print num_recalled_events, num_events


def correlation_between_opticalflow_and_boxsize():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  pool_opticalflow_dir = os.path.join(root_dir, 'toi_max_opticalflow')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  max_vals = []
  areas = []
  for name in names[:-1]:
  # for name in names[:10]:
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

    pool_opticalflow_file = os.path.join(pool_opticalflow_dir, name + '.25.forward.npy')
    data = np.load(pool_opticalflow_file)
    for tid, max_val in enumerate(data):
      if tid not in tid2label:
        continue

      label = tid2label[tid]
      track = track_db.trackid2track[tid]
      bboxs = track.track
      area = np.mean((bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0]))

      max_vals.append(max_val)
      areas.append(area)

  coef, pval = pearsonr(areas, max_vals)
  print coef, pval


def intersect_backward_forward_tracks():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  # track_len = 25
  track_len = 50
  threshold = 0.5

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  for name in names:
    track_file = os.path.join(track_dir, '%s.%d.forward.npz'%(name, track_len))
    track_map_file = os.path.join(track_dir, '%s.%d.forward.map'%(name, track_len))
    forward_track_db = api.db.TrackDb(track_map_file, track_file, track_len)

    track_file = os.path.join(track_dir, '%s.%d.backward.npz'%(name, track_len))
    track_map_file = os.path.join(track_dir, '%s.%d.backward.map'%(name, track_len))
    backward_track_db = api.db.TrackDb(track_map_file, track_file, track_len)

    backward_tracks = backward_track_db.trackid2track

    new_trackid_from_backward_db = []
    for trackid in backward_tracks:
      track = backward_tracks[trackid]
      start_frame = track.start_frame
      end_frame = start_frame + backward_track_db.track_len
      forward_tracks = forward_track_db.query_by_time_interval(start_frame, end_frame, forward_backward_threshold)
      if len(forward_tracks) == 0:
        new_trackid_from_backward_db.append(trackid)
        continue

      start_bbox = track.track[0]
      end_bbox = track.track[-1]

      start_forward_bboxs = np.array([d.track[0] for d in forward_tracks])
      end_forward_bboxs = np.array([d.track[-1] for d in forward_tracks])

      start_ious = calc_iou(start_bbox, start_forward_bboxs, True)
      end_ious = calc_iou(end_bbox, end_forward_bboxs, True)
      is_duplicate = np.logical_and(start_ious >= threshold, end_ious >= threshold)
      if np.sum(is_duplicate) == 0:
        new_trackid_from_backward_db.append(trackid)

    print name, len(new_trackid_from_backward_db), len(backward_track_db.trackid2track), len(forward_track_db.trackid2track)
    out_file = os.path.join(track_dir, '%s.%d.backward.diff'%(name, track_len))
    with open(out_file, 'w') as fout:
      for tid in new_trackid_from_backward_db:
        fout.write('%d\n'%tid)


# def intersect_25_50_tracks():
#   root_dir = '/usr0/home/jiac/data/sed' # aladdin1
#   lst_files = [
#     os.path.join(root_dir, 'dev08-1.lst'),
#     os.path.join(root_dir, 'eev08-1.lst'),
#   ]
#   track_dir = os.path.join(root_dir, 'tracking', 'person')

#   threshold = 0.5

#   names = []
#   for lst_file in lst_files:
#     with open(lst_file) as f:
#       for line in f:
#         line = line.strip()
#         name, _ = os.path.splitext(line)
#         if 'CAM4' not in name:
#           names.append(name)

#   for name in names:
#     db_file = os.path.join(track_dir, '%s.50.forward.backward.npz'%name)
#     track_50_db = api.db.TrackDb()
#     track_50_db.load(db_file)

#     db_file = os.path.join(track_dir, '%s.25.forward.backward.npz'%name)
#     track_25_db = api.db.TrackDb()
#     track_25_db.load(db_file)

#     tracks_25 = track_25_db.trackid2track

#     new_trackid_from_25 = []
#     for trackid in tracks_25:
#       track = tracks_25[trackid]
#       start_frame = track.start_frame
#       end_frame = start_frame + track.track_len
#       tracks_50 = track_50_db.query_by_time_interval(start_frame, end_frame, threshold_25_50)
#       if len(tracks_50) == 0:
#         new_trackid_from_25.append(trackid)
#         continue

#       start_bbox = track.track[0]
#       end_bbox = track.track[-1]

#       start_50_bboxs = np.array(
#         [d.track[track.start_frame - d.start_frame] for d in tracks_50])
#       end_50_bboxs = np.array(
#         [d.track[track.start_frame - d.start_frame + 24] for d in tracks_50])

#       start_ious = calc_iou(start_bbox, start_50_bboxs, True)
#       end_ious = calc_iou(end_bbox, end_50_bboxs, True)
#       is_duplicate = np.logical_and(start_ious >= threshold, end_ious >= threshold)
#       if np.sum(is_duplicate) == 0:
#         new_trackid_from_25.append(trackid)

#     print name, len(new_trackid_from_25), len(tracks_25), len(track_50_db.trackid2track)
#     out_file = os.path.join(track_dir, '%s.forward.backward.25.diff'%name)
#     with open(out_file, 'w') as fout:
#       for tid in new_trackid_from_25:
#         fout.write('%d\n'%tid)


def merge_track_db():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  track_len = 50
  # track_len = 25
  threshold = 0.5

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  for name in names:
    print name

    track_file = os.path.join(track_dir, '%s.%d.forward.npz'%(name, track_len))
    track_map_file = os.path.join(track_dir, '%s.%d.forward.map'%(name, track_len))
    forward_track_db = api.db.TrackDb()
    forward_track_db.load_v0(track_map_file, track_file)

    track_file = os.path.join(track_dir, '%s.%d.backward.npz'%(name, track_len))
    track_map_file = os.path.join(track_dir, '%s.%d.backward.map'%(name, track_len))
    backward_track_db = api.db.TrackDb()
    backward_track_db.load_v0(track_map_file, track_file)

    merge_track_db = forward_track_db
    base_trackid = max(merge_track_db.trackid2track.keys()) + 1

    diff_file = os.path.join(track_dir, '%s.%d.backward.diff'%(name, track_len))
    with open(diff_file) as f:
      for i, line in enumerate(f):
        line = line.strip()
        id = int(line)
        track = backward_track_db.trackid2track[id]
        nid = base_trackid + i
        merge_track_db.add_track(nid, track)

    out_file = os.path.join(track_dir, '%s.%d.forward.backward.npz'%(name, track_len))
    merge_track_db.save(out_file)


if __name__ == '__main__':
  # flow_dstrb_in_events()
  # filter_out_proposals()
  # event_recall_change_filter_out_proposals()
  # normalize_opticalflow()
  # gen_normalize_script()
  # correlation_between_opticalflow_and_boxsize()
  # intersect_backward_forward_tracks()
  intersect_25_50_tracks()
  # merge_track_db()
