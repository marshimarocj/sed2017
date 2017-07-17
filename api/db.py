import os
import bisect

import numpy as np


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

  def query_by_frame(self, frame):
    start_idx = bisect.bisect_right(self.start_frames, frame)-1
    if start_idx != -1:
      start_frame = self.start_frames[start_idx]
      if frame - start_frame < self.track_len and frame >= start_frame:
        tracks = self.tracks[start_idx]
        return (start_frame, tracks)
    else:
      return (-1, None)


class ClipDb(object):
  def __init__(self, clip_dir, clip_lst_file):
    self._clip_dir = clip_dir
    self._beg_ends = []
    with open(clip_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        frames = [int(d) for d in name.split('_')]
        self._beg_ends.append((frames[0], frames[1]))

    self._index = IntervalTree()
    for i, beg_end in enumerate(self._beg_ends):
      self._index[beg_end[0]:beg_end[1]] = i

  @property
  def index(self):
    return self._index

  @property
  def beg_ends(self):
    return self._beg_ends

  def query_tracklet(self, start_frame, track_len):
    clips = self.index[start_frame:start_frame + track_len]
    out = []
    for clip in clips:
      clip_idx = clip.data
      out.append(self.beg_ends[clip_idx])
    return out

  def query_clip(self, clip_name):
    return os.path.join(self._clip_dir, clip_name + '.mp4')


class FtDb(object):
  def __init__(self, ft_dir, ft_gap, chunk_gap):
    self.ft_dir = ft_dir
    self.ft_gap = ft_gap
    self.chunk_gap = chunk_gap

  @staticmethod
  def _decompress_chunk(file):
    data = np.load(file)
    shape = data['shape']
    keys = data['keys']
    ft = np.zeros((shape[0], shape[1]*shape[2]*shape[3]), dtype=np.float32)
    ft[keys[:, 0], keys[:, 1]] = data['values']
    ft = np.reshape(ft, shape)

    return ft

  def load_chunk(self, chunk, overlap=False):
    chunk_file = os.path.join(self.ft_dir, '%d.npz'%chunk)
    fts = self._decompress_chunk(chunk_file)

    if overlap:
      overlap_file = os.path.join(self.ft_dir, '%d.overlap.npz'%chunk)
      overlap_fts = self._decompress_chunk(overlap_file)
      fts = np.concatenate([fts, overlap_fts], axis=0)

    return fts

  @staticmethod
  def query_center_in_box(centers, boxs):
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


def get_vgg19_centers(shape, sample=1):
  centers = [
    (32*sample*i + 31, 32*sample*j + 31) for i in range(shape[2]) for j in range(shape[3])
  ]
  centers = np.array(centers)

  return centers


def get_c3d_centers(shape, sample=1):
  centers = [
    (16*sample*i, 16*sample*j) for i in range(shape[2]) for j in range(shape[3])
  ]
  centers = np.array(centers)

  return centers
