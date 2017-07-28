from collections import deque
import heapq

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


# it's a generator and yields FtInTrack object
# one pass of ftdb to generate features in the tracklets from trackdb
def crop_instant_ft_in_track(trackdb, ftdb, centers):
  chunks = ftdb.chunks
  for chunk in chunks:
    print 'chunk:', chunk
    one_chunk_generator = _crop_instant_ft_in_track(trackdb, ftdb, centers, chunk)
    for ft_in_track in one_chunk_generator:
      yield ft_in_track


def _crop_instant_ft_in_track(trackdb, ftdb, centers, chunk):
  fts = ftdb.load_chunk(chunk)
  shape = fts.shape

  cache = {}
  pq = []
  for f in range(shape[0]):
    frame = chunk + ftdb.ft_gap * f

    # update or insert new tracklets
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
          end_frame = track.start_frame + track.track_len
          heapq.heappush(pq, (end_frame, id))
        r = center_idx/shape[3]
        c = center_idx%shape[3]
        cache[id].append({
          'ft': fts[f, :, r, c],
          'frame': frame,
          'center': centers[center_idx]
        })

    # remove old tracklets
    while len(pq) > 0:
      if pq[0][0] > frame:
        break
      d = heapq.heappop(pq)
      id = d[1]
      _fts = cache[id]
      ft_in_track = FtInTrack(id, _fts)
      del cache[id]
      yield ft_in_track

  # remove rest tracklets
  while len(pq) > 0:
    d = heapq.heappop(pq)
    id = d[1]
    _fts = cache[id]
    ft_in_track = FtInTrack(id, _fts)
    del cache[id]
    yield ft_in_track


# it's a generator and yields FtInTrack object
# one pass of ftdb to generate features in the tracklets from trackdb
# as for duration feature, the feature is counted as included in the tracklet 
# only if two conditions are satisfied:
# 1. the feature time interval intersects with the track interval
# 2. the intersection pases threshold_func, refer to TrackDb.query_by_time_interval for threshold_func
def crop_duration_ft_in_track(trackdb, ftdb, center_grid, threshold_func):
  chunks = ftdb.chunks
  for chunk in chunks:
    print 'chunk:', chunk
    one_chunk_generator = _crop_duration_ft_in_track(trackdb, ftdb, center_grid, chunk, threshold_func)
    for ft_in_track in one_chunk_generator:
      yield ft_in_track


def _crop_duration_ft_in_track(trackdb, ftdb, center_grid, chunk, threshold_func):
  fts = ftdb.load_chunk(chunk)
  shape = fts.shape

  cache = {}
  pq = []
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
      is_xy = center_grid.center_in_box(boxs)
      center_idxs, box_idxs = np.where(is_xy)
      for center_idx, box_idx in zip(center_idxs, box_idxs):
        if id not in cache:
          cache[id] = []
          end_frame = track.start_frame + track.track_len
          heapq.heappush(pq, (end_frame, id))
        r = center_idx/shape[3]
        c = center_idx%shape[3]
        cache[id].append({
          'ft': fts[f, :, r, c],
          'frame': frame,
          'center': center_grid.centers[center_idx]
        })

    # remove old tracklets
    while len(pq) > 0:
      if pq[0][0] > frame:
        break
      d = heapq.heappop(pq)
      id = d[1]
      _fts = cache[id]
      ft_in_track = FtInTrack(id, _fts)
      del cache[id]
      yield ft_in_track

  # remove rest tracklets
  while len(pq) > 0:
    d = heapq.heappop(pq)
    id = d[1]
    _fts = cache[id]
    ft_in_track = FtInTrack(id, _fts)
    del cache[id]
    yield ft_in_track


# it's a generator and yields (trackid, imgs)
def crop_clip_in_track(clipdb, trackdb):
  clip_name2trackids = {}
  trackid2track = trackdb.trackid2track
  for trackid in trackid2track:
    track = trackid2track[trackid]
    clip_names = clipdb.query_track(track.start_frame, track.track_len)
    for clip_name in clip_names:
      if clip_name not in clip_name2trackids:
        clip_name2trackids[clip_name] = []
      clip_name2trackids[clip_name].append(trackid)

  for clip_name in clip_name2trackids:
    clip_file = clipdb.query_clip_file(clip_name)
    trackids = clip_name2trackids[clip_name]
    trackids = set(trackids)

    base_frame, _ = clipdb.get_beg_end_from_clip_name(clip_name)
    cap = cv2.VideoCapture(clip_file)
    frame = base_frame

    cache = {}
    pq = []
    while True:
      ret, img = cap.read()
      if ret == 0:
        break

      # update or insert new tracklets
      tracks = trackdb.query_by_frame(frame)
      for track in tracks:
        id = track.id
        if id not in trackids:
          continue

        box = track.track[frame-track.start_frame]
        box = box.astype(np.int32)

        if id not in cache:
          cache[id] = []
          end_frame = track.start_frame + track.track_len
          heapq.heappush(pq, (end_frame, id))

        _img = np.zeros((box[3]-box[1], box[2]-box[0], 3), dtype=np.uint8)
        rstart = -box[1] if box[1] < 0 else 0
        # guard agains box[1] >= img.shape[0]
        rend = max(img.shape[0] - box[1], 0) if box[3] > img.shape[0] else box[3]-box[1]
        cstart = -box[0] if box[0] < 0 else 0
        # guard against box[0] >= img.shape[1]
        cend = max(img.shape[1] - box[0], 0) if box[2] > img.shape[1] else box[2]-box[0]
        # print box
        # print rstart, rend, cstart, cend
        _img[rstart:rend, cstart:cend] = img[max(box[1], 0):box[3], max(box[0], 0):box[2]]

        cache[id].append(_img)

      # remove old tracklets
      while len(pq) > 0:
        if pq[0][0] > frame:
          break
        d = heapq.heappop(pq)
        trackid = d[1]
        imgs = cache[trackid]
        del cache[trackid]
        yield (trackid, imgs)

      frame += 1

    # remove rest tracklets
    while len(pq) > 0:
      d = heapq.heappop(pq)
      trackid = d[1]
      imgs = cache[trackid]
      del cache[trackid]
      yield (trackid, imgs)
