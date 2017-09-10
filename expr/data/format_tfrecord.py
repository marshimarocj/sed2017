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
def transform():
  root_dir = '/data/extDisk3/jiac/sed' # danny
  ft_dirs = [
    os.path.join(root_dir, 'twostream', 'feat_anet_flow_6frame', 'track_group_trn_split'),
  ]

  for ft_dir in ft_dirs:
    names = os.listdir(ft_dir)
    for name in names:
      _name, _ = os.path.join(name)
      src_file = os.path.join(ft_dir, name)
      dst_file = os.path.join(ft_dir, _name + '.tfrecords')

      data = np.load(src_file)
      ids = data['ids']
      fts = data['fts']
      frames = data['frames']
      centers = data['centers']
      num = ids.shape[0]

      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) 
      writer = tf.python_io.TFRecordWriter(dst_file, options=)
      for i in range(num):
        id = int(ids[i])
        frame = int(frames[i])
        ft = fts[i].tostring()
        center = centers[i].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
          'id': _int64_feature(id),
          'frame': _int64_feature(frame),
          'ft': _bytes_feature(ft),
          'center': _bytes_feature(center)
          }))
        writer.write(example.SerializeToString())
      writer.close()


if __name__ == '__main__':
  transform()
