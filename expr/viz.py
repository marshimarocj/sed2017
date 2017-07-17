import os
import bisect

from intervaltree import Interval, IntervalTree

from toi import TrackDb


'''func
'''
class ToiVideo(object):
  def __init__(self, video_file, trackdb, base_frame=0):
    self.video_file = video_file
    self.trackdb = trackdb
    self.base_frame = base_frame

  def _load_tracks(self, track_file):
    tracks = []
    data = np.load(track_file)
    for start_frame in self.start_frames:
      track = data['%010d'%start_frame]
      track[:, :, 2] += track[:, :, 0]
      track[:, :, 3] += track[:, :, 1]
      track[:, :, 0] = np.maximum(track[:, :, 0], np.zeros(track.shape[:2]))
      track[:, :, 1] = np.maximum(track[:, :, 1], np.zeros(track.shape[:2]))
      tracks.append(track)

    return tracks

  def yield_tracks(self):
    cap = cv2.VideoCapture(self.video_file)
    cache = {}
    q = deque()
    frame = self.base_frame
    while True:
      ret, img = cap.read()
      if ret == 0:
        break

      # update or insert new tracklets
      start_idx = bisect.bisect_right(self.start_frames, frame)-1
      if start_idx != -1:
        start_frame = start_frames[start_idx]
        if frame - start_frame < self.track_len and frame >= start_frame:
          track = self.tracks[start_idx]
          boxs = track[frame-start_frame, ::]
          for i, box in enumerate(boxs):
            key = '%d %d'%(start_frame, i)
            if key not in self.frame_box2trackletid:
              continue
            trackletid = self.frame_box2trackletid[key]
            if trackletid not in cache:
              cache[trackletid] = []
              q.append((trackletid, start_frame))
            _img = img[boxs[1]:boxs[3], boxs[0]:boxs[2]]
            cache[trackletid].append(_img)

      # remove old tracklets
      while len(q) > 0:
        if q[0][1] + self.track_len > frame:
          break
        d = q.pop()
        trackletid = d[0]
        _imgs = cache[trackletid] # the size of _img in _imgs may not be the same!
        yield (trackletid, _imgs)

      frame += 1

    # remove old tracklets
    while len(q) > 0:
      if q[0][1] + self.track_len > frame:
        break
      d = q.pop()
      trackletid = d[0]
      _imgs = cache[trackletid] # the size of _img in _imgs may not be the same!
      yield (trackletid, _imgs)


class ClipDb(object):
  def __init__(clip_dir, clip_lst_file):
    self._clip_dir = clip_dir
    self._beg_ends = []
    with open(clip_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        frames = [int(d) for d in name.split('_')]
        self._beg_ends.append((frames[0], frames[1]))

    self._index = IntervalTree()
    for i, beg_end in enumerate(beg_ends):
      self._index[beg_end[0]:beg_end[1]] = i

  @property
  def index(self):
    return self._index

  @property
  def beg_ends(self):
    return self._beg_ends


def clip2tracks(clip_lst_file, track_db):
  beg_ends = []
  with open(clip_lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      frames = [int(d) for d in name.split('_')]
      beg_ends.append((frames[0], frames[1]))

  t = IntervalTree()
  for i, beg_end in enumerate(beg_ends):
    t[beg_end[0]:beg_end[1]] = i


def viz_tracklet(clip_dir, clip_lst_file, 
    track_map_file, track_file, track_len, trackletids):
  beg_ends = []
  with open(clip_lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      frames = [int(d) for d in name.split('_')]
      beg_ends.append((frames[0], frames[1]))

  t = IntervalTree()
  for i, beg_end in enumerate(beg_ends):
    t[beg_end[0]:beg_end[1]] = i

  start_frames, _ = load_track_map_file(track_map_file, valid_trackletids=trackletids)

  clip_idx2track_idxs = {}
  for i, track in enumerate(tracks):
    start_frame = track.start_frame
    end_frame = track.start_frame + track.track_len

    clips = t[start_frame:end_frame]
    for clip in clips:
      clip_idx = clip.data
      if clip_idx not in clip_idx2track_idxs:
        clip_idx2track_idxs[clip_idx] = []
      clip_idx2track_idxs[clip_idx].append(i)

  for clip_idx in clip_idx2track_idxs:
    beg_end = beg_ends[clip_idx]
    clip_file = os.path.join(clip_dir, '%d_%d.mp4'%(beg, end))
    track_idxs = clip_idx2track_idxs[clip_idx]
    _tracks = []
    for track_idx in track_idxs:
      _tracks.append(tracks[track_idx])
    _tracks = sorted(_tracks, key=lambda x:x.start_frame)




'''expr
'''


if __name__ == '__main__':
  pass
