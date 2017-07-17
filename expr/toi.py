import os
from collections import deque
import bisect


'''func
'''
class TrackDb(object):
  def __init__(self, track_map_file, track_file, track_len, valid_trackletids=None):
    self._valid_trackletids = valid_trackletids
    if self._valid_trackletids is not None:
      self._valid_trackletids = set(self._valid_trackletids)

    self._track_len = track_len

    self._frame_box2trackletid = {}
    start_frames = set()
    with open(track_map_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        id = int(data[0])
        if self._valid_trackletids is None or id in self._valid_trackletids:
          fields = data[1].split('_')
          start_frame = int(fields[0])
          start_frames.add(start_frame)
          boxid = int(fields[1])
          key = '%d %d'%(start_frame, boxid)
          self._frame_box2trackletid[key] = id

    self._start_frames = sorted(list(start_frames))

    self._tracks = []
    data = np.load(track_file)
    for start_frame in self._start_frames:
      track = data['%010d'%start_frame]
      track[:, :, 2] += track[:, :, 0]
      track[:, :, 3] += track[:, :, 1]
      track[:, :, 0] = np.maximum(track[:, :, 0], np.zeros(track.shape[:2]))
      track[:, :, 1] = np.maximum(track[:, :, 1], np.zeros(track.shape[:2]))
      self._tracks.append(track)

  @property
  def valid_trackletids(self):
    return self._valid_trackletids

  @property
  def track_len(self):
    return self._track_len

  @property
  def start_frames(self):
    return self._start_frames

  @property
  def tracks(self):
    return self._tracks

  @property
  def frame_box2trackletid(self):
    return self._frame_box2trackletid


def load_ft(ft_file):
  data = np.load(file)
  shape = data['shape']
  keys = data['keys']
  ft = np.zeros((shape[0], shape[1]*shape[2]*shape[3]), dtype=np.float32)
  ft[keys[:, 0], keys[:, 1]] = data['values']
  ft = np.reshape(ft, shape)

  return ft


class ToiInChunk(object):
  def __init__(self, ft_dir, ft_gap, trackdb):
    self.ft_dir = ft_dir
    self.ft_gap = ft_gap
    self.trackdb = trackdb

  # return centers in np.array
  @property
  def centers(self):
    raise NotImplementedException("""please customize centers""")

  def _is_centers_in_box(self, centers, boxs):
    is_x = np.logical_and(
      np.expand_dims(centers[:, 0], 1) >= np.expand_dims(boxs[:, 1], 0), 
      np.expand_dims(centers[:, 0], 1) < np.expand_dims(boxs[:, 3], 0)
    )
    is_y = np.logical_and(
      np.expand_dims(centers[:, 1], 1) >= np.expand_dims(boxs[:, 0], 0), 
      np.expand_dims(centers[:, 1], 1) < np.expand_dims(boxs[:, 2], 0)
    )
    is_xy = np.logical_and(is_x, is_y)

    return is_xy

  def yield_toi_fts(self, chunk, overlap=False):
    ft_file = os.path.join(self.ft_dir, '%d.npz'%chunk)
    fts = load_ft(ft_file)

    if overlap:
      ft_file = os.path.join(self.ft_dir, '%d.overlap.npz'%chunk)
      overlap_fts = load_ft(ft_file)
      fts = np.concatenate([fts, overlap_fts], axis=0)

    start_idx = bisect.bisect_right(self.trackdb.start_frames, start_frame)-1
    track = self.trackdb.tracks[start_idx]

    shape = fts.shape
    for f in range(shape[0]):
      ft_frame = chunk + self.ft_gap*f

      # update or insert new tracklets
      start_idx = bisect.bisect_right(self.trackdb.start_frames, ft_frame)-1
      if start_idx != -1:
        start_frame = self.trackdb.start_frames[start_idx]
        if ft_frame - start_frame < self.trackdb.track_len and ft_frame >= start_frame:
          track = self.trackdb.tracks[start_idx]
          boxs = track[ft_frame-start_frame, ::]
          is_xy = is_centers_in_boxs(self.centers, boxs)
          center_idxs,  box_idxs = np.where(is_xy)
          for center_idx, box_idx in zip(center_idxs, box_idxs):
            key = '%d %d'%(start_frame, box_idx)
            if key not in self.trackdb.frame_box2trackletid:
              continue
            trackletid = self.trackdb.frame_box2trackletid[key]
            if trackletid not in cache: # insert new tracklets
              cache[trackletid] = []
              q.append((trackletid, start_frame))
            cache[trackletid].append(
              fts[f, :, center_idx/shape[3], center_idx%shape[3]])

      # remove old tracklets
      while len(q) > 0:
        if q[0][1] + self.trackdb.track_len > ft_frame:
          break
        d = q.pop()
        trackletid = d[0]
        _fts = cache[trackletid]
        _fts = np.array(_fts)
        del cache[trackletid]
        yield (trackletid, _fts)

    # remove rest tracklets
    while len(q) > 0:
      d = q.pop()
      trackletid = d[0]
      _fts = cache[trackletid]
      _fts = np.array(_fts)
      del cache[trackletid]
      yield (trackletid, _fts)


def one_pass(chunk_gap, max_frame, toi_chunk):
  for chunk in range(0, max_frame, chunk_gap):
    for trackletid, fts in toi_chunk.yield_toi_fts(chunk):
      yield (trackletid, fts)


'''expr
'''



if __name__ == '__main__':
  pass
