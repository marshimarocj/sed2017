from collections import deque

import cv2
import numpy as np


class FtInTrack(object):
  def __init__(self, id, cached_ft_in_track):
    self._id = id
    self._fts = np.array([d['ft'] for d in cached_ft_in_track])
    self._frames = [d['frame'] for d in cached_ft_in_track]
    self._centers = np.array([d['center'] for d in cached_ft_in_track])

  @property
  def id(self):
    return self._id

  @property
  def fts(self):
    return self._fts

  @property
  def frames(self):
    return self._frames

  @property
  def centers(self):
    return self._centers


# one pass of ftdb to generate features in the tracklets from trackdb
# this is a generator and yield FtInTrack object
def instant_ft_in_track_generator(trackdb, ftdb, centers, chunk):
  fts = ftdb.load_chunk(chunk)
  shape = fts.shape

  cache = {}
  q = deque()
  for f in range(shape[0]):
    frame = chunk + ftdb.ft_gap * f

    # update or insert new tracklets
    # start_frame, tracks = trackdb.query_by_frame(frame)
    # if start_frame != -1:
    #   boxs = tracks[frame-start_frame, ::]
    #   is_xy = ftdb.query_center_in_box(centers, boxs)
    #   center_idxs, box_idxs = np.where(is_xy)
    #   for center_idx, box_idx in zip(center_idxs, box_idxs):
    #     key = '%d %d'%(start_frame, box_idx)
    #     if key not in trackdb.frame_box2trackletid:
    #       continue
    #     trackletid = trackdb.frame_box2trackletid[key]
    #     if trackletid not in cache:
    #       cache[trackletid] = []
    #       q.append((trackletid, start_frame))
    #     r = center_idx/shape[3]
    #     c = center_idx%shape[3]
    #     cache[trackletid].append({
    #       'ft': fts[f, :, r, c],
    #       'frame': frame,
    #       'center': centers[center_idx] 
    #     })
    tracks = trackdb.query_by_frame(frame)
    for track in tracks:
      id = track.id
      box = track.track[frame-track.start_frame]
      boxs = np.expand_dims(box, 0)
      is_xy = ftdb.query_center_in_box(centers, boxs)
      center_idxs, box_idxs = np.where(is_xy)
      for center_idx, box_idx in zip(center_idxs, box_idxs):
        if id not in cache:
          cache[id] = []
          q.append((id, track.start_frame))
        r = center_idx/shape[3]
        c = center_idx%shape[3]
        cache[id].append({
          'ft': fts[f, :, r, c],
          'frame': frame,
          'center': centers[center_idx]
        })

    # remove old tracklets
    while len(q) > 0:
      if q[0][1] + trackdb.track_len > frame:
        break
      d = q.pop()
      id = d[0]
      _fts = cache[id]
      # _copy_fts = {
      #   'ft': np.array([d['ft'] for d in _fts]),
      #   'frame': [d['frame'] for d in _fts],
      #   'center': np.array([d['center'] for d in _fts])
      # }
      ft_in_track = FtInTrack(id, _fts)
      del cache[id]
      # yield (id, _copy_fts)
      yield ft_in_track

  # remove rest tracklets
  while len(q) > 0:
    d = q.pop()
    id = d[0]
    _fts = cache[id]
    # _copy_fts = {
    #   'ft': np.array([d['ft'] for d in _fts]),
    #   'frame': [d['frame'] for d in _fts],
    #   'center': np.array([d['center'] for d in _fts])
    # }
    ft_in_track = FtInTrack(id, _fts)
    del cache[id]
    yield ft_in_track


# one pass of ftdb to generate features in the tracklets from trackdb
# for duration feature, we consider that the feature is counted as included in the tracklet 
# if more than half the feature duration intersects with the tracklet time interval
def duration_ft_in_track_generator(trackdb, ftdb, centers, chunk, tiou_threshold):
  # ft_duration = ftdb.ft_duration
  # track_len = trackdb.track_len

  fts = ftdb.load_chunk(chunk)
  shape = fts.shape

  cache = {}
  q = deque()
  for f in range(shape[0]):
    frame = chunk + ftdb.ft_gap * f

    # update or insert new tracklets
    # start_frame, tracks = trackdb.query_by_frame(frame)
    # _frame = frame + ft_duration
    # if start_frame != -1 and start_frame + track_len - frame >= ft_duration/2:
    #   boxs = tracks[frame-start_frame, ::]
    #   is_xy = ftdb.query_center_in_box(centers, boxs)
    #   center_idxs, box_idxs = np.where(is_xy)
    #   for center_idx, box_idx in zip(center_idxs, box_idxs):
    #     key = '%d %d'%(start_frame, box_idx)
    #     if key not in trackdb.frame_box2trackletid:
    #       continue
    #     trackletid = trackdb.frame_box2trackletid[key]
    #     if trackletid not in cache:
    #       cache[trackletid] = []
    #       q.append((trackletid, start_frame))
    #     r = center_idx/shape[3]
    #     c = center_idx%shape[3]
    #     cache[trackletid].append({
    #       'ft': fts[f, :, r, c],
    #       'frame': frame,
    #       'center': centers[center_idx] 
    #     })
    tracks = trackdb.query_by_tiou_threshold(frame, frame + ftdb.ft_duration, tiou_threshold)
    for track in tracks:
      id = track.id
      if frame - track.start_frame >= track.track.shape[0]:
        print track.id, track.start_frame, track.track.shape, frame, ftdb.ft_duration
      box = track.track[frame-track.start_frame]
      boxs = np.expand_dims(box, 0)
      is_xy = ftdb.query_center_in_box(centers, boxs)
      center_idxs, box_idxs = np.where(is_xy)
      for center_idx, box_idx in zip(center_idxs, box_idxs):
        if id not in cache:
          cache[id] = []
          q.append((id, track.start_frame))
        r = center_idx/shape[3]
        c = center_idx%shape[3]
        cache[id].append({
          'ft': fts[f, :, r, c],
          'frame': frame,
          'center': centers[center_idx]
        })

    # remove old tracklets
    while len(q) > 0:
      if q[0][1] + trackdb.track_len > frame:
        break
      d = q.pop()
      id = d[0]
      _fts = cache[id]
      # _copy_fts = {
      #   'ft': np.array([d['ft'] for d in _fts]),
      #   'frame': [d['frame'] for d in _fts],
      #   'center': np.array([d['center'] for d in _fts])
      # } 
      ft_in_track = FtInTrack(id, _fts)
      del cache[id]
      # yield (trackletid, _copy_fts)
      yield ft_in_track

  # remove rest tracklets
  while len(q) > 0:
    d = q.pop()
    id = d[0]
    _fts = cache[id]
    # _copy_fts = {
    #   'ft': np.array([d['ft'] for d in _fts]),
    #   'frame': [d['frame'] for d in _fts],
    #   'center': np.array([d['center'] for d in _fts])
    # } 
    ft_in_track = FtInTrack(id, _fts)
    del cache[id]
    # yield (trackletid, _copy_fts)
    yield ft_in_track


# the size of img in yielded imgs may not be the same!
def clip_in_track_generator(clipdb, trackdb):
  clip2trackletids = {}
  for start_frame in trackdb.start_frames:
    clips = clipdb.query_tracklet(start_frame, trackdb.track_len)
    _, tracks = trackdb.query_by_frame(start_frame)
    trackletids = []
    for i, track in enumerate(tracks):
      frame_box = '%d %d'%(start_frame, i)
      if frame_box in trackdb.frame_box2trackletid:
        trackletid = trackdb.frame_box2trackletid[frame_box]
        trackletids.append(trackletid)
    for clip in clips:
      clip_name = '%d %d'%(clip[0], clip[1])
      if clip_name not in clip2track_idxs:
        clip2trackletids[clip_name] = []
      clip2trackletids[clip_name].extend(trackletids)

  for clip_name in clip2track_idxs:
    clip_file = clipdb.query_clip(clip_name)
    trackletids = clip2track_idxs[clip_name]
    trackletids = set(trackletids)

    pos = clip_name.find('_')
    base_frame = int(clip_name[:pos])
    cap = cv2.VideoCapture(clip_file)
    frame = base_frame

    cache = {}
    q = deque()
    while True:
      ret, img = cap.read()
      if ret == 0:
        break

      # update or insert new tracklets
      start_frame, tracks = trackdb.query_by_frame(frame)
      if start_frame != -1:
        boxs = tracks[frame-start_frame, ::]
        for box_idx, box in enumerate(boxs):
          key = '%d %d'%(start_frame, box_idx)
          if key not in trackdb.frame_box2trackletid:
            continue
          trackletid = trackdb.frame_box2trackletid[key]
          if trackletid in trackletids:
            if trackletid not in cache:
              cache[trackletid] = []
              q.append((trackletid, start_frame))
            cache[trackletid].append(
              img[box[1]:box[3], box[0]:box[2]])

      # remove old tracklets
      while len(q) > 0:
        if q[0][1] + trackdb.track_len > frame:
          break
        d = q.pop()
        trackletid = d[0]
        imgs = cache[trackletid]
        del cache[trackletid]
        yield (trackletid, imgs)

      frame += 1

    # remove rest tracklets
    while len(q) > 0:
      d = q.pop()
      trackletid = d[0]
      imgs = cache[trackletid]
      del cache[trackletid]
      yield (trackletid, imgs)
