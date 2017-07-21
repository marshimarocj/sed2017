import os

import numpy as np
from intervaltree import IntervalTree


class Track(object):
  def __init__(self, id, track, start_frame):
    self._id = id
    self._track = track
    self._start_frame = start_frame

  # id of the track
  @property
  def id(self):
    return self._id

  # np.array in shape (track_len, 4)
  # 4 corresponds to xmin, ymin, xmax, ymax
  @property
  def track(self):
    return self._track

  # start frame of the track
  @property
  def start_frame(self):
    return self._start_frame


# Note: 
# for simplicity, the bounding boxs in the the same track is of the same size at different frames
# in return, the coordinate of the bounding boxs may be <0 or >= img_width/img_height
class TrackDb(object):
  def __init__(self, track_map_file, track_file, track_len, valid_trackids=None):
    self._track_len = track_len

    frame_box2trackid = {}
    with open(track_map_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        id = int(data[0])
        if valid_trackids is None or id in valid_trackids:
          fields = data[1].split('_')
          start_frame = int(fields[0])
          boxid = int(fields[1])
          key = '%d %d'%(start_frame, boxid)
          frame_box2trackid[key] = id

    self._trackid2track = {}
    self._index = IntervalTree()
    data = np.load(track_file)
    for key in data:
      start_frame = int(key)
      tracks = data[key]
      tracks[:, :, 2] += tracks[:, :, 0]
      tracks[:, :, 3] += tracks[:, :, 1]
      # tracks[:, :, 0] = np.maximum(tracks[:, :, 0], np.zeros(tracks.shape[:2]))
      # tracks[:, :, 1] = np.maximum(tracks[:, :, 1], np.zeros(tracks.shape[:2]))
      num_track = tracks.shape[1]
      for i in range(num_track):
        frame_box = '%d %d'%(start_frame, i)
        if frame_box in frame_box2trackid:
          trackid = frame_box2trackid[frame_box]
          track = Track(trackid, tracks[:, i, :], start_frame)
          self._trackid2track[trackid] = track
          self._index[start_frame:start_frame + track_len] = track

  # returns an int
  @property
  def track_len(self):
    return self._track_len

  @property
  def trackid2track(self):
    return self._trackid2track

  # return list of Track objects whose time interval covers the input frame
  def query_by_frame(self, frame):
    results = self._index[frame]
    results = [d.data for d in results]
    return results

  # return list of Track objects who intersects with query interval
  # and pass threhold_func
  # threshold_func: (qbegin, qend, tbegin, tend) -> bool
  def query_by_time_interval(self, qbegin, qend, threshold_func):
    intersects = self._index[qbegin:qend]
    results = []
    for intersect in intersects:
      track = intersect.data
      tbegin = track.start_frame
      tend = tbegin + self.track_len

      if threshold_func(qbegin, qend, tbegin, tend):
        results.append(track)

    return results


class ClipDb(object):
  def __init__(self, clip_dir, clip_lst_file):
    self._clip_dir = clip_dir
    # self._beg_ends = []
    self._clip_name2beg_end = {}
    self._index = IntervalTree()

    with open(clip_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        frames = [int(d) for d in name.split('_')[1:]]
        self._clip_name2beg_end[name] = (frames[0], frames[1])
        self._index[frames[0]:frames[1]] = name
        # self._beg_ends.append((frames[0], frames[1]))

    # for i, beg_end in enumerate(self._beg_ends):
    #   self._index[beg_end[0]:beg_end[1]] = i

  # @property
  # def index(self):
  #   return self._index

  # @property
  # def beg_ends(self):
  #   return self._beg_ends

  @property
  def clip_name2beg_end(self):
    return self._clip_name2beg_end

  @staticmethod
  def get_beg_end_from_clip_name(clip_name):
    data = clip_name.split('_')
    return (int(data[1]), int(data[2]))

  # return list of clipnames that contain the query track
  def query_track(self, start_frame, track_len):
    clips = self._index[start_frame:start_frame + track_len]
    out = []
    for clip in clips:
      clip_name = clip.data
      out.append(clip_name)
    return out

  # return full path of clip file
  def query_clip_file(self, clip_name):
    return os.path.join(self._clip_dir, clip_name + '.mp4')


class FtDb(object):
  def __init__(self, ft_dir, ft_gap, chunk_gap):
    self._ft_dir = ft_dir
    self._ft_gap = ft_gap
    self._chunk_gap = chunk_gap

  @property
  def ft_gap(self):
    return self._ft_gap

  @property
  def chunk_gap(self):
    return self._chunk_gap

  @staticmethod
  def _decompress_chunk(file):
    data = np.load(file)
    if 'fts' in data: # dense storage
      ft = data['fts']
    else: # sparse storage
      shape = data['shape']
      keys = data['keys']
      ft = np.zeros((shape[0], shape[1]*shape[2]*shape[3]), dtype=np.float32)
      ft[keys[:, 0], keys[:, 1]] = data['values']
      ft = np.reshape(ft, shape)

    return ft

  def load_chunk(self, chunk, overlap=False):
    chunk_file = os.path.join(self._ft_dir, '%d.npz'%chunk)
    fts = self._decompress_chunk(chunk_file)

    if overlap:
      overlap_file = os.path.join(self._ft_dir, '%d.overlap.npz'%chunk)
      overlap_fts = self._decompress_chunk(overlap_file)
      fts = np.concatenate([fts, overlap_fts], axis=0)

    return fts

  @staticmethod
  def query_center_in_box(centers, boxs):
    # say there are m centers and n boxs
    # this is a trick to generate (m,n) comparison efficiently with numpy
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


# just an alias
class InstantFtDb(FtDb):
  pass


class DurationFtDb(FtDb):
  @property
  def ft_duration(self):
    return self._ft_duration


class C3DFtDb(DurationFtDb):
  def __init__(self, ft_dir):
    self._ft_gap = 16
    self._chunk_gap = 10000
    self._ft_duration = 16
    DurationFtDb.__init__(self, ft_dir, self._ft_gap, self._chunk_gap)


class PAFFtDb(InstantFtDb):
  def __init__(self, ft_dir):
    self._ft_gap = 5
    self._chunk_gap = 7500
    InstantFtDb.__init__(self, ft_dir, self._ft_gap, self._chunk_gap)


class VGG19FtDb(InstantFtDb):
  def __init__(self, ft_dir):
    self._ft_gap = 5
    self._chunk_gap = 7500
    InstantFtDb.__init__(self, ft_dir, self._ft_gap, self._chunk_gap)


def get_vgg19_centers():
  grid_h = 9
  grid_w = 11
  centers = [
    (64*i + 32, 64*j + 32) for i in range(grid_h) for j in range(grid_w)
  ]
  centers = np.array(centers)

  return centers


def get_c3d_centers():
  grid_h = 36
  grid_w = 45
  centers = [
    (16*i + 8, 16*j + 8) for i in range(grid_h) for j in range(grid_w)
  ]
  centers = np.array(centers)

  return centers
