import os
import sys
sys.path.append('../')

import numpy as np
import cv2

import api.db
import api.generator


'''func
'''
def c3d_threshold_func(qbegin, qend, tbegin, tend):
  ibegin = max(tbegin, qbegin)
  iend = min(tend, qend)
  return  iend - ibegin >= 8


def load_pos_track_label_file(file):
  valid_events = set(['CellToEar', 'Embrace', 'Pointing', 'PersonRuns'])
  id2event = {}
  with open(file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      event = data[1]
      if event in valid_events:
        id = int(data[0])
        event = data[1]
        id2event[id] = event

  return id2event


'''expr
'''


def tst_c3d_toi():
  root_dir = '/data1/jiac/sed' # uranus
  tracking_root_dir = os.path.join(root_dir, 'tracking')
  c3d_root_dir = os.path.join(root_dir, 'c3d')
  video_name = 'LGW_20071107_E1_CAM3'

  direction = 'forward'
  track_len = 25
  track_map_file = os.path.join(tracking_root_dir, '%s.%d.%s.map'%(video_name, track_len, direction))
  track_file = os.path.join(tracking_root_dir, '%s.%d.%s.npz'%(video_name, track_len, direction))
  track_db = api.db.TrackDb(track_map_file, track_file, track_len)

  c3d_dir = os.path.join(c3d_root_dir, video_name)
  c3d_db = api.db.C3DFtDb(c3d_dir)

  chunk_idx = 1
  centers = api.db.get_c3d_centers()
  ft_in_track_generator = api.generator.crop_duration_ft_in_track(
    track_db, c3d_db, centers, c3d_db.chunk_gap*chunk_idx, c3d_threshold_func)

  cnt = 0
  for ft_in_track in ft_in_track_generator:
    print ft_in_track.id, ft_in_track.fts.shape, len(set(ft_in_track.frames))
    cnt += 1
    if cnt == 100:
      break


def tst_vgg_toi():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  tracking_root_dir = os.path.join(root_dir, 'tracking', 'person')
  vgg_root_dir = os.path.join(root_dir, 'raw_ft', 'vgg19_pool5')
  video_name = 'LGW_20071107_E1_CAM2'

  direction = 'forward'
  track_len = 25
  track_map_file = os.path.join(tracking_root_dir, '%s.%d.%s.map'%(video_name, track_len, direction))
  track_file = os.path.join(tracking_root_dir, '%s.%d.%s.npz'%(video_name, track_len, direction))
  track_db = api.db.TrackDb(track_map_file, track_file, track_len)

  vgg_dir = os.path.join(vgg_root_dir, video_name)
  vgg_db = api.db.VGG19FtDb(vgg_dir)

  chunk_idx = 1
  centers = api.db.get_vgg19_centers()
  ft_in_track_generator = api.generator.crop_instant_ft_in_track(
    track_db, vgg_db, centers, vgg_db.chunk_gap*chunk_idx)

  cnt = 0
  for ft_in_track in ft_in_track_generator:
    id = ft_in_track.id
    track = track_db.trackid2track[id]
    area = np.mean((track.track[:, 2] - track.track[:, 0]) * (track.track[:, 3] - track.track[:, 1]))
    print ft_in_track.id, ft_in_track.fts.shape, len(set(ft_in_track.frames)), area
    cnt += 1
    if cnt == 100:
      break


def tst_opticalflow_toi():
  pass


def tst_viz_tracklet():
  # root_dir = '/usr0/home/jiac/data/sed' #aladdin1
  root_dir = '/data'
  preprocess_dir = os.path.join(root_dir, 'video', 'preprocess')
  label_dir = os.path.join(root_dir, 'pseudo_label')
  track_dir = os.path.join(root_dir, 'tracking', 'person')
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  out_root_dir = os.path.join(root_dir, 'viz_pos')

  # video_name = 'LGW_20071101_E1_CAM1'
  video_names = []
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          video_names.append(name)

  for video_name in video_names:
    clip_dir = os.path.join(preprocess_dir, video_name, 'clip_6000_100')
    clip_lst_file = os.path.join(preprocess_dir, video_name, 'clip_6000_100.lst')
    clip_db = api.db.ClipDb(clip_dir, clip_lst_file)

    label_file = os.path.join(label_dir, '%s.50.forward.backward.square.0.75.pos'%video_name)
    id2event = load_pos_track_label_file(label_file)
    pos_ids = id2event.keys()

    track_db_file = os.path.join(track_dir, '%s.50.forward.backward.square.npz'%video_name)
    track_db = api.db.TrackDb()
    track_db.load(track_db_file, valid_trackids=pos_ids)

    crop_clip_in_track_generator = api.generator.crop_clip_in_track(clip_db, track_db)

    out_dir = os.path.join(out_root_dir, video_name)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    for trackid, imgs in crop_clip_in_track_generator:
      event = id2event[trackid]
      out_file = os.path.join(out_dir, '%s.%d.mp4'%(event, trackid))
      print out_file

      shape = imgs[0].shape
      fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
      # fourcc = cv2.cv.CV_FOURCC('H', '2', '6', '4')
      writer = cv2.VideoWriter(out_file, fourcc, 25, (shape[1], shape[0]))
      for img in imgs:
        writer.write(img)
      writer.release()


if __name__ == '__main__':
  # tst_c3d_toi()
  # tst_vgg_toi()
  tst_viz_tracklet()
