import os

import numpy as np
import tensorflow as tf


'''func
'''
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


'''expr
'''
def transform_by_grouping():
  root_dir = '/data/extDisk3/jiac/sed' # danny
  ft_dirs = [
    # os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split'),
    os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_val'),
    # os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_tst'),
  ]

  for ft_dir in ft_dirs:
    names = os.listdir(ft_dir)
    for name in names:
      _name, _ = os.path.splitext(name)
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
      writer = tf.python_io.TFRecordWriter(dst_file, options=options)
      prev_id = ids[0]
      ft_in_track = []
      frame_in_track = []
      center_in_track = []
      for i in range(num):
        id = int(ids[i])
        if id != prev_id:
          _fts = np.array(ft_in_track).tostring()
          _frames = np.array(frame_in_track).tostring()
          _centers = np.array(center_in_track).tostring()
          example = tf.train.Example(features=tf.train.Features(feature={
            'id': _int64_feature(prev_id),
            'frame': _bytes_feature(_frames),
            'ft': _bytes_feature(_fts),
            'center': _bytes_feature(_centers)
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

      _fts = np.array(ft_in_track).tostring()
      _frames = np.array(frame_in_track).tostring()
      _centers = np.array(center_in_track).tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
        'id': _int64_feature(prev_id),
        'frame': _bytes_feature(_frames),
        'ft': _bytes_feature(_fts),
        'center': _bytes_feature(_centers)
        }))
      writer.write(example.SerializeToString())

      writer.close()


if __name__ == '__main__':
  transform()
