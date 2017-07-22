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


# it's a generator
# one pass of ftdb to generate features in the tracklets from trackdb
# this is a generator and yield FtInTrack object
def crop_instant_ft_in_track(trackdb, ftdb, centers, chunk):
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


# it's a generator
# one pass of ftdb to generate features in the tracklets from trackdb
# as for duration feature, the feature is counted as included in the tracklet 
# only if two conditions are satisfied:
# 1. the feature time interval intersects with the track interval
# 2. the intersection pases threshold_func, refer to TrackDb.query_by_time_interval for threshold_func
def crop_duration_ft_in_track(trackdb, ftdb, centers, threshold_func):
  chunks = ftdb.chunks
  for chunk in chunks:
    print 'chunk:', chunk
    one_chunk_generator = _crop_duration_ft_in_track(trackdb, ftdb, centers, chunk, threshold_func)
    for ft_in_track in one_chunk_generator:
      yield ft_in_track


def _crop_duration_ft_in_track(trackdb, ftdb, centers, chunk, threshold_func):
  fts = ftdb.load_chunk(chunk)
  shape = fts.shape

  cache = {}
  q = deque()
  for f in range(shape[0]):
    frame = chunk + ftdb.ft_gap * f

    # update or insert new tracklets
    tracks = trackdb.query_by_time_interval(frame, frame + ftdb.ft_duration, threshold_func)
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
      ft_in_track = FtInTrack(id, _fts)
      del cache[id]
      yield ft_in_track

  # remove rest tracklets
  while len(q) > 0:
    d = q.pop()
    id = d[0]
    _fts = cache[id]
    ft_in_track = FtInTrack(id, _fts)
    del cache[id]
    yield ft_in_track


# it's a generator
def crop_clip_in_track(clipdb, trackdb):
  # clip2trackletids = {}
  # for start_frame in trackdb.start_frames:
  #   clips = clipdb.query_tracklet(start_frame, trackdb.track_len)
  #   _, tracks = trackdb.query_by_frame(start_frame)
  #   trackletids = []
  #   for i, track in enumerate(tracks):
  #     frame_box = '%d %d'%(start_frame, i)
  #     if frame_box in trackdb.frame_box2trackletid:
  #       trackletid = trackdb.frame_box2trackletid[frame_box]
  #       trackletids.append(trackletid)
  #   for clip in clips:
  #     clip_name = '%d %d'%(clip[0], clip[1])
  #     if clip_name not in clip2track_idxs:
  #       clip2trackletids[clip_name] = []
  #     clip2trackletids[clip_name].extend(trackletids)

  clip_name2trackids = {}
  trackid2track = trackdb.trackid2track
  for trackid in trackid2track:
    track = trackid2track[trackid]
    clip_names = clipdb.query_track(track.start_frame, trackdb.track_len)
    for clip_name in clip_names:
      if clip_name not in clip_name2trackids:
        clip_name2trackids[clip_name] = []
      clip_name2trackids[clip_name].append(trackid)

  # for clip_name in clip2track_idxs:
  #   clip_file = clipdb.query_clip(clip_name)
  #   trackletids = clip2track_idxs[clip_name]
  #   trackletids = set(trackletids)

  #   pos = clip_name.find('_')
  #   base_frame = int(clip_name[:pos])
  #   cap = cv2.VideoCapture(clip_file)
  #   frame = base_frame
  for clip_name in clip_name2trackids:
    clip_file = clipdb.query_clip(clip_name)
    trackids = clip_name2trackids[clip_name]
    trackids = set(trackids)

    # data = clip_name.split('_')
    # base_frame = int(data[1])
    base_frame, _ = clipdb.get_beg_end_from_clip_name(clip_name)
    cap = cv2.VideoCapture(clip_file)
    frame = base_frame

    cache = {}
    q = deque()
    while True:
      ret, img = cap.read()
      if ret == 0:
        break

      # # update or insert new tracklets
      # start_frame, tracks = trackdb.query_by_frame(frame)
      # if start_frame != -1:
      #   boxs = tracks[frame-start_frame, ::]
      #   for box_idx, box in enumerate(boxs):
      #     key = '%d %d'%(start_frame, box_idx)
      #     if key not in trackdb.frame_box2trackletid:
      #       continue
      #     trackletid = trackdb.frame_box2trackletid[key]
      #     if trackletid in trackletids:
      #       if trackletid not in cache:
      #         cache[trackletid] = []
      #         q.append((trackletid, start_frame))

      #       cache[trackletid].append(img[box[1]:box[3], box[0]:box[2]])

      # update or insert new tracklets
      tracks = trackdb.query_by_frame(frame)
      for track in tracks:
        id = track.id
        if id not in trackids:
          continue

        box = track.track[frame-track.start_frame]

        if id not in cache:
          cache[id] = []
          q.append((id, track.start_frame))

        _img = np.zeros((box[3]-box[1], box[2]-box[0]), dtype=np.uint8)
        rstart = -box[1] if box[1] < 0 else 0
        rend = box[3] - _img.shape[0] if box[3] > _img.shape[0] else box[3]-box[1]
        cstart = -box[0] if box[0] < 0 else 0
        cend = box[2] - _img.shape[1] if box[2] > _img.shape[1] else box[2]-box[0]
        _img[rstart:rend, cstart:cend] = img[max(box[1], 0):box[3], max(box[0], 0):box[2]]

        cache[id].append(_img)

      # remove old tracklets
      while len(q) > 0:
        if q[0][1] + trackdb.track_len > frame:
          break
        d = q.pop()
        trackid = d[0]
        imgs = cache[trackid]
        del cache[trackid]
        yield (trackid, imgs)

      frame += 1

    # remove rest tracklets
    while len(q) > 0:
      d = q.pop()
      trackid = d[0]
      imgs = cache[trackid]
      del cache[trackid]
      yield (trackid, imgs)
