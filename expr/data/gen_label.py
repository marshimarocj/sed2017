import os
import json
import cPickle
import shutil
import itertools
import sys
sys.path.append('../../')

import numpy as np

import api.db


'''func
'''
class Label(object):
  def __init__(self, event, beg, end):
    self.boxs = []
    self.frames = []
    self.event = event
    self.beg = beg
    self.end = end


class PseudoPosLabel(object):
  def __init__(self, tid, beg, end, event):
    self.tid = tid
    self.riou = 0.
    self.intersect_frames = []
    self.beg = beg
    self.end = end
    self.event = event


def load_bboxs(file):
  rh = 240
  rw = 320
  h = 576
  w = 720

  video2eventid2label = {}
  with open(file) as f:
    data = json.load(f)
  for d in data:
    frame = d['begin'] + int(d['second_in_frame'])
    video_name = 'LGW_%s_%s_%s'%(d['date'], d['E'], d['camera'])
    labels = d['label']
    event = d['event']
    beg = d['begin']
    end = d['end']
    eventid = '%s_%d_%d_%s'%(video_name, beg, end, event)
    if video_name not in video2eventid2label:
      video2eventid2label[video_name] = {}
    if eventid not in video2eventid2label[video_name]:
      video2eventid2label[video_name][eventid] = Label(event, beg, end)
    for label in labels:
      box = label['bndbox']
      box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
      box[0] = box[0] * w / rw
      box[2] = box[2] * w / rw
      box[1] = box[1] * h / rh
      box[3] = box[3] * h / rh
      video2eventid2label[video_name][eventid].boxs.append(box)
      video2eventid2label[video_name][eventid].frames.append(frame)

  video2labels = {}
  for video in video2eventid2label:
    eventid2label = video2eventid2label[video]
    video2labels[video] = eventid2label.values()

  return video2labels


def gen_groundtruth_threshold_func(track_len):
  def groundtruth_threshold_func(qbegin, qend, tbegin, tend):
    ibegin = max(tbegin, qbegin)
    iend = min(tend, qend)

    return iend-qbegin >= min(track_len/2, qend - qbegin)

  return groundtruth_threshold_func


def calc_iou(box_lhs, boxs, union_or_left=False):
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
    # if union_or_min:
    if union_or_left:
      iou = area_intersect / (float(area) + areas - area_intersect)
    else:
      # iou = area_intersect / \
      #   np.where(np.minimum(float(area), areas) == 0, 
      #     1, np.minimum(float(area), areas))
      iou = area_intersect / \
        np.where(float(area) == 0, 
          1, float(area))

  return iou


'''expr
'''
def find_track_interval_intersected_with_bbox():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  bbox_file = os.path.join(root_dir, 'box_label', 'train.label.json')
  track_dir = os.path.join(root_dir, 'tracking', 'person')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_dir = os.path.join(root_dir, 'pseudo_label')

  # direction = 'forward'
  # direction = 'backward'
  track_len = 25
  # track_len = 50
  groundtruth_threshold_func = gen_groundtruth_threshold_func(track_len)
  # iou_threshold = 0.5
  iou_threshold = 0.75

  video2labels = load_bboxs(bbox_file)

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  for name in names:
    labels = video2labels[name]
    # track_file = os.path.join(track_dir, '%s.%d.%s.npz'%(name, track_len, direction))
    # track_map_file = os.path.join(track_dir, '%s.%d.%s.map'%(name, track_len, direction))
    db_file = os.path.join(track_dir, '%s.%d.forward.backward.npz'%(name, track_len))
    # db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    print name

    # track_db = api.db.TrackDb(track_map_file, track_file, track_len)
    track_db = api.db.TrackDb()
    track_db.load(db_file)

    pseudo_pos_labels = {}
    for label in labels:
      qbegin = label.beg
      qend = label.end
      tracks = track_db.query_by_time_interval(qbegin, qend, groundtruth_threshold_func)
      interval_trackids = set([track.id for track in tracks])

      num_box = len(label.boxs)
      for i in range(num_box):
        lbox = label.boxs[i]
        frame = label.frames[i]
        tracks = track_db.query_by_frame(frame)
        for track in tracks:
          if track.id not in interval_trackids:
            continue
          tbox = track.track[frame - track.start_frame]
          tboxs = np.expand_dims(tbox, 0)
          iou = calc_iou(lbox, tboxs)
          if iou >= iou_threshold:
            if track.id not in pseudo_pos_labels:
              pseudo_pos_labels[track.id] = PseudoPosLabel(track.id, label.beg, label.end, label.event)
            pseudo_pos_labels[track.id].riou = max(iou, pseudo_pos_labels[track.id].riou)
            pseudo_pos_labels[track.id].intersect_frames.append(frame)

    out = []
    for tid in pseudo_pos_labels:
      pseudo_pos_label = pseudo_pos_labels[tid]
      out.append({
        'tid': pseudo_pos_label.tid,
        'riou': pseudo_pos_label.riou,
        'intersect_frames': pseudo_pos_label.intersect_frames,
        'beg': pseudo_pos_label.beg,
        'end': pseudo_pos_label.end,
        'event': pseudo_pos_label.event
      })

    out_file = os.path.join(out_dir, '%s.%d.forward.backward.%.2f.interval.pkl'%(name, track_len, iou_threshold))
    # out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.%.2f.interval.pkl'%(name, track_len, iou_threshold))
    with open(out_file, 'w') as fout:
      cPickle.dump(out, fout)


def find_track_frame_intersected_with_bbox():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  bbox_file = os.path.join(root_dir, 'box_label', 'train.label.json')
  track_dir = os.path.join(root_dir, 'tracking', 'person')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_dir = os.path.join(root_dir, 'pseudo_label')

  track_len = 25
  # track_len = 50
  iou_threshold = 0.5

  video2labels = load_bboxs(bbox_file)

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  # names = ['LGW_20071112_E1_CAM2']
  # for name in names:
  for name in names[17:]:
    labels = video2labels[name]
    # db_file = os.path.join(track_dir, '%s.%d.forward.backward.npz'%(name, track_len))
    db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    print name

    track_db = api.db.TrackDb()
    track_db.load(db_file)

    pseudo_pos_labels = {}
    for label in labels:
      num_box = len(label.boxs)
      for i in range(num_box):
        lbox = label.boxs[i]
        frame = label.frames[i]
        tracks = track_db.query_by_frame(frame)
        for track in tracks:
          tbox = track.track[frame - track.start_frame]
          tboxs = np.expand_dims(tbox, 0)
          iou = calc_iou(lbox, tboxs)
          if iou >= iou_threshold:
            if track.id not in pseudo_pos_labels:
              pseudo_pos_labels[track.id] = PseudoPosLabel(track.id, label.beg, label.end, label.event)
            pseudo_pos_labels[track.id].riou = max(iou, pseudo_pos_labels[track.id].riou)
            pseudo_pos_labels[track.id].intersect_frames.append(frame)

    out = []
    for tid in pseudo_pos_labels:
      pseudo_pos_label = pseudo_pos_labels[tid]
      out.append({
        'tid': pseudo_pos_label.tid,
        'riou': pseudo_pos_label.riou,
        'intersect_frames': pseudo_pos_label.intersect_frames,
        'beg': pseudo_pos_label.beg,
        'end': pseudo_pos_label.end,
        'event': pseudo_pos_label.event
      })

    # out_file = os.path.join(out_dir, '%s.%d.forward.backward.frame.pkl'%(name, track_len))
    out_file = os.path.join(out_dir, '%s.%d.forward.backward.square.%.2f.frame.pkl'%(name, track_len, iou_threshold))
    with open(out_file, 'w') as fout:
      cPickle.dump(out, fout)


def normalize_match_name():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')

  direction = 'forward'
  track_len = 25

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  for name in names:
    src_file = os.path.join(label_dir, '%s.pkl'%name)
    dst_file = os.path.join(label_dir, '%s.%d.%s.pkl'%(name, track_len, direction))
    # print src_file, dst_file
    shutil.move(src_file, dst_file)


def event_matched_tracks():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')

  # direction = 'forward'
  # track_len = 25
  track_len = 50

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  # event2num_tracks = {}
  for name in names:
    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.interval.pkl'%(name, track_len))
    with open(label_file) as f:
      pseudo_pos_labels = cPickle.load(f)
    eventid2trackids = {}
    for pseudo_pos_label in pseudo_pos_labels:
      tid = pseudo_pos_label['tid']
      beg = pseudo_pos_label['beg']
      end = pseudo_pos_label['end']
      event = pseudo_pos_label['event']
      eventid = '%d_%d_%s'%(beg, end, event)
      if eventid not in eventid2trackids:
        eventid2trackids[eventid] = []
      eventid2trackids[eventid].append(tid)

    # for eventid in eventid2trackids:
    #   data = eventid.split('_')
    #   event = data[2]
    #   num_track = len(eventid2trackids[eventid])
    #   if event not in event2num_tracks:
    #     event2num_tracks[event] = []
    #   event2num_tracks[event].append(num_track)

    out_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.interval.match.pkl'%(name, track_len))
    with open(out_file, 'w') as fout:
      cPickle.dump(eventid2trackids, fout)

  # for event in event2num_tracks:
  #   num_tracks = event2num_tracks[event]
  #   print event, np.mean(num_tracks), np.median(num_tracks), np.percentile(num_tracks, 10), np.percentile(num_tracks, 90)


def recall():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  bbox_file = os.path.join(root_dir, 'box_label', 'train.label.json')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')

  # directions = ['forward']
  # directions = ['backward']
  # directions = ['forward', 'backward']
  # track_lens = [25]
  # track_lens = [50]
  track_lens = [25, 50]

  names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          names.append(name)

  with open(bbox_file) as f:
    data = json.load(f)
  video2events = {}
  for d in data:
    video = 'LGW_%s_%s_%s'%(d['date'], d['E'], d['camera'])
    event = d['event']
    beg = d['begin']
    end = d['end']
    eventid = '%s_%d_%d'%(event, beg, end)
    if video not in video2events:
      video2events[video] = set()
    video2events[video].add(eventid)

  total = 0
  hit = 0
  for name in names:
    events = video2events[name]
    recalled_events = set()
    for track_len in track_lens:
      file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.interval.pkl'%(name, track_len))
      # file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.interval.pkl'%(name, track_len))
      # file = os.path.join(label_dir, '%s.%d.forward.backward.0.50.interval.pkl'%(name, track_len))
      # file = os.path.join(label_dir, '%s.%d.forward.backward.0.75.interval.pkl'%(name, track_len))
      with open(file) as f:
        pseudo_pos_labels = cPickle.load(f)
      for pseudo_pos_label in pseudo_pos_labels:
        event = pseudo_pos_label['event']
        beg = pseudo_pos_label['beg']
        end = pseudo_pos_label['end']
        eventid = '%s_%d_%d'%(event, beg, end)
        recalled_events.add(eventid)
    total += len(events)
    hit += len(recalled_events)
    print name, len(recalled_events) / float(len(events))
  print hit / float(total), hit, total


def generate_pos_neg_lst():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking', 'person')

  track_len = 25
  # track_len = 50

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

    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.interval.pkl'%(name, track_len))
    out_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
    with open(label_file) as f:
      pseudo_pos_labels = cPickle.load(f)
    with open(out_file, 'w') as fout:
      for label in pseudo_pos_labels:
        tid = label['tid']
        event = label['event']
        fout.write('%d %s\n'%(tid, event))

    label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.frame.pkl'%(name, track_len))
    tids = set()
    with open(label_file) as f:
      for label in pseudo_pos_labels:
        tid = label['tid']
        tids.add(tid)

    db_file = os.path.join(track_dir, '%s.%d.forward.backward.square.npz'%(name, track_len))
    track_db = api.db.TrackDb()
    track_db.load(db_file)
    out_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.50.neg'%(name, track_len))
    with open(out_file, 'w') as fout:
      for tid in track_db.trackid2track:
        if tid not in tids:
          fout.write('%d\n'%tid)


def refine_label_for_cell2ear():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  bbox_file = os.path.join(root_dir, 'box_label', 'train.label.json')
  out_file = os.path.join(root_dir, 'box_label', 'train.label.cell2ear.refine.json')

  threshold = 62

  with open(bbox_file) as f:
    data = json.load(f)
  out = []
  for d in data:
    if d['event'] == 'CellToEar':
      begin = d['begin']
      end = d['end']
      duration = end - begin
      if duration <= threshold:
        out.append(d)
    else:
      out.append(d)
  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


if __name__ == '__main__':
  # find_track_interval_intersected_with_bbox()
  # find_track_frame_intersected_with_bbox()
  # generate_pos_neg_lst()
  # recall()
  # normalize_match_name()
  # event_matched_tracks()
  refine_label_for_cell2ear()
