import os

import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import backend as K

global backend
backend = 'tf'


'''func
'''
def get_int_model(model, layer):
  input_shape=(16, 112, 112, 3) # l, h, w, c

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
  # if layer == 'pool5':
  #     return int_model

  # int_model.add(Flatten())
  # # FC layers group
  # int_model.add(Dense(4096, activation='relu', name='fc6',
  #                         weights=model.layers[15].get_weights()))
  # if layer == 'fc6':
  #     return int_model
  # int_model.add(Dropout(.5))
  # int_model.add(Dense(4096, activation='relu', name='fc7',
  #                         weights=model.layers[17].get_weights()))
  # if layer == 'fc7':
  #     return int_model
  # int_model.add(Dropout(.5))
  # int_model.add(Dense(487, activation='softmax', name='fc8',
  #                         weights=model.layers[19].get_weights()))
  # if layer == 'fc8':
  #     return int_model

  return None


class C3dFeatureExtractor():
  #clear session in after 20 times calculation
  gpu_max_clear_limit=20;gpu_clear_count=0

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
    # self.model.compile(loss='mean_squared_error', optimizer='sgd')

    # get activations for intermediate layers if needed
    self.int_model = get_int_model(model=self.model, layer=layer)
    print "int model type:",self.int_model
    #load the mean file
    self.mean_cube = np.load(self.mean_cube_filename)
    self.mean_cube = np.transpose(self.mean_cube, (1, 2, 3, 0))

  #extract feautres
  def extract_layer_feat(self,vid, start_frame, end_frame, layer='conv5b'):
    self.gpu_clear_count+=1
    X = vid
    X -= self.mean_cube
    feat = self.int_model.predict_on_batch(np.array([X]))
    feat = feat[0, ...]
    if self.gpu_clear_count % self.gpu_max_clear_limit == 0:
        K.clear_session()
        self.__load_model__()
    return feat


'''expr
'''
def tst_c3d():
  model_dir = '/home/jiac/models/tensorflow/sed' # uranus
  mean_file = os.path.join(model_dir, 'sed_16_576_720_mean.npy')
  net_weight_file = os.path.join(model_dir, 'sports1M_weights_tf.h5')
  net_json_file = os.path.join(model_dir, 'sed_sports1M_weights_tf.json')
  video_file = '/data1/jiac/sed/video/dev09/MCTTR0101a.mov.deint.avi'

  feat_extractor=C3dFeatureExtractor(net_weight_file, net_json_file, mean_file)
  #load video and run
  print("[Info] Loading a sample video...")
  cap = cv2.VideoCapture(video_file)
  vid = []
  frame_count = 0
  while True:
      frame_count += 1
      print "Frame:", frame_count
      ret, img = cap.read()
      if not ret:
          break
      vid.append(img)
      if frame_count >= 32:
          break

  vid = np.array(vid, dtype=np.float32)
  print vid.shape
  start_frame = 0
  end_frame = start_frame+16
  layer = 'conv5b'
  vid_feat=feat_extractor.extract_layer_feat(vid,start_frame,end_frame,layer)
  print vid_feat.shape


if __name__ == '__main__':
  tst_c3d()
