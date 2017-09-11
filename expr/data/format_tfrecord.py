import os

import numpy as np
import tensorflow as tf


'''func
'''
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


'''expr
'''
def transform_by_grouping():
  root_dir = '/home/jiac/data/sed' # danny
  ft_dirs = [
    # os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split'),
    os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val'),
    # os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_tst'),
  ]
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  label_dir = os.path.join(root_dir, 'pseudo_label')

  video2track_len2pos_id2lid = {}
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        video2track_len2pos_id2lid[name] = {}

        for track_len in [25, 50]:
          video2track_len2pos_id2lid[name][track_len] = {}
          label_file = os.path.join(label_dir, '%s.%d.forward.backward.square.0.75.pos'%(name, track_len))
          with open(label_file) as f_label:
            for line in f_label:
              line = line.strip()
              data = line.split(' ')
              id = int(data[0])
              lid = event2lid[data[1]]
              video2track_len2pos_id2lid[name][track_len][id] = lid

  dim_ft = 1024
  dim_center = 2
  for ft_dir in ft_dirs:
    names = os.listdir(ft_dir)
    for name in names:
      if 'tfrecords' in name:
        continue
      _name, _ = os.path.splitext(name)
      data = _name.split('.')
      video_name = data[0]
      track_len = int(data[1])
      pos_id2lid = video2track_len2pos_id2lid[video_name][track_len]
      src_file = os.path.join(ft_dir, name)
      dst_file = os.path.join(ft_dir, _name + '.tfrecords')
      print name

      data = np.load(src_file)
      ids = data['ids']
      fts = data['fts']
      frames = data['frames']
      centers = data['centers']
      num = ids.shape[0]

      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) 
      with tf.python_io.TFRecordWriter(dst_file, options=options) as writer:
        if ids.shape[0] == 0:
          continue
        prev_id = ids[0]
        ft_in_track = []
        frame_in_track = []
        center_in_track = []
        for i in range(num):
          id = int(ids[i])
          if id != prev_id:
            _fts = np.array(ft_in_track, dtype=np.float32).tostring()
            _frames = np.array(frame_in_track, dtype=np.float32).tostring()
            _centers = np.array(center_in_track, dtype=np.float32).tostring()
            num = len(frame_in_track)
            if prev_id in pos_id2lid:
              label = pos_id2lid[prev_id]
            else:
              label = 0
            example = tf.train.Example(features=tf.train.Features(feature={
              'id': _int64_feature(prev_id),
              'label': _int64_feature(label),
              'frame': _bytes_feature(_frames),
              'ft': _bytes_feature(_fts),
              'center': _bytes_feature(_centers),
              'num': _int64_feature(num),
              'dim_ft': _int64_feature(dim_ft),
              'dim_center': _int64_feature(dim_center),
              }))
            writer.write(example.SerializeToString())

            del ft_in_track
            del frame_in_track
            del center_in_track
            ft_in_track = []
            frame_in_track = []
            center_in_track = []
            prev_id = id

          frame = int(frames[i])
          ft = fts[i]
          center = centers[i]
          frame_in_track.append(frame)
          ft_in_track.append(ft)
          center_in_track.append(center)

        _fts = np.array(ft_in_track, dtype=np.float32).tostring()
        _frames = np.array(frame_in_track, dtype=np.float32).tostring()
        _centers = np.array(center_in_track, dtype=np.float32).tostring()
        num = len(ft_in_track)
        example = tf.train.Example(features=tf.train.Features(feature={
          'id': _int64_feature(id),
          'frame': _bytes_feature(_frames),
          'ft': _bytes_feature(_fts),
          'center': _bytes_feature(_centers),
          'num': _int64_feature(num),
          'dim_ft': _int64_feature(dim_ft),
          'dim_center': _int64_feature(dim_center),
          }))
        writer.write(example.SerializeToString())


def tst_load_tfrecords():
  root_dir = '/data/extDisk3/jiac/sed' # danny
  file = os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val', 'LGW_20071130_E2_CAM2.25.forward.backward.square.pos.0.75.tfrecords')

  options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) 
  record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    feature = example.features.feature
    id = int(feature['id'].int64_list.value[0])
    num = int(feature['num'].int64_list.value[0])
    dim_ft = int(feature['dim_ft'].int64_list.value[0])
    dim_center = int(feature['dim_center'].int64_list.value[0])
    fts = feature['ft'].bytes_list.value[0]
    fts = np.fromstring(fts, dtype=np.float32).reshape(num, dim_ft)
    centers = feature['center'].bytes_list.value[0]
    centers = np.fromstring(centers, dtype=np.float32).reshape(num, dim_center)

    print id, fts.shape, centers.shape


if __name__ == '__main__':
  transform_by_grouping()
  # tst_load_tfrecords()
