import os
import json
import cPickle
import sys
sys.path.append('../')

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

    return iend-qbegin >= track_len/2

  return groundtruth_threshold_func


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
def find_track_intersected_with_bbox():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  bbox_file = os.path.join(root_dir, 'box_label', 'train.label.json')
  track_dir = os.path.join(root_dir, 'tracking', 'person')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_dir = os.path.join(root_dir, 'pseudo_label')

  direction = 'forward'
  track_len = 25
  groundtruth_threshold_func = gen_groundtruth_threshold_func(track_len)
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

  for name in names:
    labels = video2labels[name]
    track_file = os.path.join(track_dir, '%s.%d.%s.npz'%(name, track_len, direction))
    track_map_file = os.path.join(track_dir, '%s.%d.%s.map'%(name, track_len, direction))
    print name

    track_db = api.db.TrackDb(track_map_file, track_file, track_len)

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

    out_file = os.path.join(out_dir, name + '.pkl')
    with open(out_file, 'w') as fout:
      cPickle.dump(out, fout)


if __name__ == '__main__':
  find_track_intersected_with_bbox()
