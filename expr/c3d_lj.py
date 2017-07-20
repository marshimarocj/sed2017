import os

# import cv2
import skvideo.io
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import backend as K
import scipy.sparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

global backend
backend = 'tf'


'''func
'''
def get_int_model(model, layer):
  input_shape=(16, 576, 720, 3) # l, h, w, c

  int_model = Sequential()

  int_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv1',
                          input_shape=input_shape,
                          weights=model.layers[0].get_weights()))
  if layer == 'conv1':
      return int_model
  int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         border_mode='valid', name='pool1'))
  if layer == 'pool1':
      return int_model

  # 2nd layer group
  int_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv2',
                          weights=model.layers[2].get_weights()))
  if layer == 'conv2':
      return int_model
  int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool2'))
  if layer == 'pool2':
      return int_model

  # 3rd layer group
  int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv3a',
                          weights=model.layers[4].get_weights()))
  if layer == 'conv3a':
      return int_model
  int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv3b',
                          weights=model.layers[5].get_weights()))
  if layer == 'conv3b':
      return int_model
  int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool3'))
  if layer == 'pool3':
      return int_model

  # 4th layer group
  int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv4a',
                          weights=model.layers[7].get_weights()))
  if layer == 'conv4a':
      return int_model
  int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv4b',
                          weights=model.layers[8].get_weights()))
  if layer == 'conv4b':
      return int_model
  int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool4'))
  if layer == 'pool4':
      return int_model

  # 5th layer group
  int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv5a',
                          weights=model.layers[10].get_weights()))
  if layer == 'conv5a':
      return int_model
  int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                          border_mode='same', name='conv5b',
                          weights=model.layers[11].get_weights()))
  if layer == 'conv5b':
      return int_model
  int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad'))
  int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         border_mode='valid', name='pool5'))

  return None


class C3dFeatureExtractor():
  #the initialization function
  def __init__(self, model_weight_filename, model_json_filename, mean_cube_filename):
    self.model_weight_filename = model_weight_filename
    self.model_json_filename = model_json_filename
    self.mean_cube_filename = mean_cube_filename
    self.backend = 'tf'
    self.__load_model__()

  #load C3D models
  def __load_model__(self,layer='conv5b'):
    #load model architecture
    print("[Info] Reading model architecture...")
    self.model = model_from_json(open(self.model_json_filename, 'r').read())
    self.model.load_weights(self.model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    # get activations for intermediate layers if needed
    self.int_model = get_int_model(model=self.model, layer=layer)
    print "int model type:",self.int_model
    #load the mean file
    self.mean_cube = np.load(self.mean_cube_filename)
    self.mean_cube = np.transpose(self.mean_cube, (1, 2, 3, 0))

  #extract feautres
  def extract_layer_feat(self,vid, start_frame, end_frame, layer='conv5b'):
    X = vid
    # X[:16] -= self.mean_cube
    # X[16:] -= self.mean_cube
    X -= self.mean_cube
    # feat = self.int_model.predict_on_batch(np.array([X[:16], X[16:]]))
    # feat = [np.expand_dims(feat[0][0], 0), np.expand_dims(feat[1][0], 0)]
    # feat = [np.expand_dims(feat[0][0], 0), np.expand_dims(feat[1][0], 0)]
    feat = self.int_model.predict_on_batch(np.array([X]))
    feat = feat[0][0] # the 2nd 0 is used to circumvent the tf issue
    return feat


'''expr
'''
def tst_c3d():
  # model_dir = '/home/jiac/models/tensorflow/sed' # uranus
  model_dir = '/data1/jiac/sed/code/c3d-keras/models'
  mean_file = os.path.join(model_dir, 'sed_16_576_720_mean.npy')
  net_weight_file = os.path.join(model_dir, 'sports1M_weights_tf.h5')
  net_json_file = os.path.join(model_dir, 'sed_sports1M_weights_tf.json')
  video_file = '/data1/jiac/sed/video/dev09/MCTTR0101a.mov.deint.avi'

  feat_extractor=C3dFeatureExtractor(net_weight_file, net_json_file, mean_file)
  #load video and run
  print("[Info] Loading a sample video...")
  cap = skvideo.io.VideoCapture(video_file)
  vid = []
  frame_count = 0
  while True:
      # print "Frame:", frame_count
      ret, img = cap.read()
      # print img.shape
      if not ret:
          break
      vid.append(img)

      frame_count += 1
      if frame_count == 32:
          break

  vid = np.array(vid, dtype=np.float32)
  start_frame = 0
  end_frame = start_frame+32
  layer = 'conv5b'
  vid_feats = feat_extractor.extract_layer_feat(vid,start_frame,end_frame,layer)

  vid_feats = np.concatenate(vid_feats)
  # print vid_feat.shape
  # print np.sum(vid_feat == 0)
  np.savez_compressed('/tmp/tmp.npz', vid_feat)

  vid_feat = vid_feat.reshape((vid_feat.shape[0], -1))
  dok = scipy.sparse.dok_matrix(vid_feat)
  keys = dok.keys()
  values = dok.values()
  np.savez_compressed('/tmp/tmp_sparse.npz', keys=keys, values=values)


def bat_c3d():
  root_dir = '/data1/jiac/sed' # uranus_c3d
  video_root_dir = os.path.join(root_dir, 'video', 'dev09')
  lst_file = os.path.join(root_dir, 'video', 'dev09', 'video2017.lst')
  out_root_dir = os.path.join(root_dir, 'c3d', 'sed_test_2017')
  model_dir = '/data1/jiac/sed/code/c3d-keras/models'
  mean_file = os.path.join(model_dir, 'sed_16_576_720_mean.npy')
  net_weight_file = os.path.join(model_dir, 'sports1M_weights_tf.h5')
  net_json_file = os.path.join(model_dir, 'sed_sports1M_weights_tf.json')

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  set_session(tf.Session(config=config))
  layer = 'conv5b'

  feat_extractor = C3dFeatureExtractor(
    net_weight_file, net_json_file, mean_file)

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      names.append(name)

  num_video = len(names)

  chunk_gap = 2000

  for i in range(0, num_video/4):
    name = names[i]
    video_file = os.path.join(video_root_dir, name + '.avi')
    print video_file

    cap = skvideo.io.VideoCapture(video_file)
    cnt = 0
    imgs = []
    fts = []
    while True:
      ret, img = cap.read()
      if not ret:
        break
      imgs.append(img)

      cnt += 1
      if cnt % 16 == 0:
        imgs = np.array(imgs, dtype=np.float32)
        ft = feat_extractor.extract_layer_feat(
          imgs, 0, 16,layer)
        fts.append(ft)
        if cnt % chunk_gap == 0:
          idx = cnt / chunk_gap -1
          out_file = os.path.join(out_root_dir, name, '%d.npz'%idx)
          np.savez_compressed(out_file, fts=fts)
          del fts
          fts = []
        del imgs
        imgs = []
        print name, cnt
    if len(fts) > 0:
      idx = cnt / chunk_gap
      out_file = os.path.join(out_root_dir, name, '%d.npz'%idx)
      np.savez_compressed(out_file, fts=fts)


if __name__ == '__main__':
  # tst_c3d()
  bat_c3d()
