import os
import random
import cPickle

import numpy as np
import tensorflow as tf

import framework.model.proto
import framework.model.trntst
import framework.model.data
import input_tool


class Config(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_center = 0
    self.dim_output = 0
    self.trn_neg2pos_in_batch = 10
    self.val_neg2pos_in_batch = 1

    self.l2_norm_input = False
    self.l2_norm_output = False
    self.dropin = False

    self.centers = np.empty((0,)) # (dim_ft, num_center)


class ModelCfg(framework.model.proto.FullModelConfig):
  def __init__(self):
    framework.model.proto.FullModelConfig.__init__(self)

    self.proto_cfg = Config()

    self.num_class = -1
    self.dropout = False

  def load(self, file):
    data = framework.model.proto.FullModelConfig.load(self, file)
    self.proto_cfg.load(data['proto'])


class ModelFocalLossCfg(ModelCfg):
  def __init__(self):
    ModelCfg.__init__(self)
    self.gamma = 2
    self.alphas = []


class NetVladEncoder(framework.model.proto.ModelProto):
  namespace = 'netvlad.NetVladEncoder'

  def __init__(self, config):
    framework.model.proto.ModelProto.__init__(self, config)

    # input
    self._fts = tf.no_op() # (None, num_ft, dim_ft)
    self._ft_masks = tf.no_op() # (None, num_ft)
    # output
    self._trn_feature_op = tf.no_op()
    self._tst_feature_op = tf.no_op()

  @property
  def fts(self):
    return self._fts
  
  @fts.setter
  def fts(self, val):
    self._fts = val

  @property
  def ft_masks(self):
    return self._ft_masks

  @ft_masks.setter
  def ft_masks(self, val):
    self._ft_masks = val

  @property
  def trn_feature_op(self):
    return self._trn_feature_op

  @property
  def tst_feature_op(self):
    return self._tst_feature_op

  def build_parameter_graph(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.centers = tf.Variable(
          np.array(self._config.centers, dtype=np.float32), name='centers')
        scale = 1.0 / (self._config.dim_ft ** 0.5)
        self.w = tf.get_variable('w',
          shape=(self._config.dim_ft, self._config.num_center), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-scale, scale))
        self.b = tf.get_variable('b',
          shape=(self._config.num_center,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

        dim_vlad = self._config.dim_ft * self._config.num_center
        scale = 1.0 / (dim_vlad ** 0.5)
        self.fc_W = tf.get_variable('fc_W',
          shape=(dim_vlad, self._config.dim_output), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-scale, scale))
        self.fc_B = tf.get_variable('fc_B',
          shape=(self._config.dim_output), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def _netvlad(self):
    fts = tf.reshape(self._fts, (-1, self._config.dim_ft)) # (None*num_ft, dim_ft)
    if self._config.l2_norm_input:
      fts = tf.nn.l2_normalize(fts, dim=1)
    logits = tf.nn.xw_plus_b(fts, self.w, self.b) # (None*num_ft, num_center)
    a = tf.nn.softmax(logits)

    a = tf.expand_dims(a, 1) # (None*num_ft, 1, num_center)
    fts = tf.expand_dims(fts, 2) # (None*num_ft, dim_ft, 1)
    centers = tf.expand_dims(self.centers, 0) # (1, dim_ft, num_center)
    diff = fts - centers # (None*num_ft, dim_ft, num_center)
    V_ijk = a * diff # (None*num_ft, dim_ft, num_center)
    mask = tf.reshape(self._ft_masks, (-1, 1, 1))
    V_ijk *= mask
    dim_vlad = self._config.dim_ft* self._config.num_center
    V_ijk = tf.reshape(V_ijk, (-1, self._config.num_ft, dim_vlad))
    V_jk = tf.reduce_sum(V_ijk, 1) # (None, dim_vlad)

    return V_jk

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        V_jk = self._netvlad()

        if self._config.l2_norm_output:
          V_jk = tf.nn.l2_normalize(V_jk, dim=1)

        self._tst_feature_op = tf.nn.xw_plus_b(V_jk, self.fc_W, self.fc_B)

  def build_inference_graph_in_trn_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        V_jk = self._netvlad()

        if self._config.l2_norm_output:
          V_jk = tf.nn.l2_normalize(V_jk, dim=1)
        trn_V_jk = V_jk
        if self._config.dropin:
          trn_V_jk = tf.nn.dropout(trn_V_jk, 0.5)

        self._trn_feature_op = tf.nn.xw_plus_b(trn_V_jk, self.fc_W, self.fc_B)
        self._tst_feature_op = tf.nn.xw_plus_b(V_jk, self.fc_W, self.fc_B)


class NetVladModel(framework.model.proto.FullModel):
  name_scope = 'netvlad.NetVladModel'

  def __init__(self, config):
    framework.model.proto.FullModel.__init__(self, config)
    self.logit_op = tf.no_op()
    self.predict_op = tf.no_op()

  def get_model_proto(self):
    nv = NetVladEncoder(self.config.proto_cfg)
    return nv

  def add_tst_input(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._fts = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft, self.config.proto_cfg.dim_ft), name='fts')
        self._ft_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft), name='ft_masks')

    self.model_proto._fts = self._fts
    self.model_proto._ft_masks = self._ft_masks

  def add_trn_tst_input(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._fts = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft, self.config.proto_cfg.dim_ft), name='fts')
        self._ft_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft), name='ft_masks')
        self._labels = tf.placeholder(
          tf.int32, shape=(None, self.config.num_class), name='labels')

    self.model_proto._fts = self._fts
    self.model_proto._ft_masks = self._ft_masks

  def _build_parameter_graph(self, basegraph):
    framework.model.proto.FullModel._build_parameter_graph(self, basegraph)
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        scale = 1.0 / (self._config.proto_cfg.dim_output ** 0.5)
        self.fc_class_W = tf.get_variable('fc_class_W',
          shape=(self._config.proto_cfg.dim_output, self._config.num_class),
          dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-scale, scale))
        self.fc_class_B = tf.get_variable('fc_class_B',
          shape=(self._config.num_class), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def _build_inference_graph_in_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        feature_op = tf.nn.relu(self.model_proto.tst_feature_op)
        self.logit_op = tf.nn.xw_plus_b(feature_op, self.fc_class_W, self.fc_class_B)
        self.predict_op = tf.nn.softmax(self.logit_op)

  def _build_inference_graph_in_trn_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_trn_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        trn_feature_op = self.model_proto.trn_feature_op
        if self._config.dropout:
          trn_feature_op = tf.nn.dropout(tf.nn.relu(trn_feature_op), 0.5)
        self.logit_op = tf.nn.xw_plus_b(trn_feature_op, self.fc_class_W, self.fc_class_B)
        self.predict_op = tf.nn.softmax(self.logit_op)

        tst_feature_op = tf.nn.relu(self.model_proto.tst_feature_op)
        self.val_logit_op = tf.nn.xw_plus_b(tst_feature_op, self.fc_class_W, self.fc_class_B)
        self.val_predict_op = tf.nn.softmax(self.val_logit_op)

  def add_loss(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels, logits=self.logit_op))
        self.val_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels, logits=self.val_logit_op))
        self.append_op2monitor('loss', loss_op)

    return loss_op

  def op_in_val(self):
    return {
      'loss_op': self.val_loss_op,
      'logit_op': self.val_logit_op,
      'predict_op': self.val_predict_op,
    }

  def op_in_tst(self):
    return {
      'logit_op': self.logit_op,
      'predict_op': self.predict_op,
    }


class NetVladFocalLossModel(NetVladModel):
  name_scope = 'netvlad.NetVladFocalLossModel'

  def add_loss(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        log_p = tf.nn.log_softmax(self.logit_op)
        # self.append_op2monitor('log_p_min', tf.reduce_min(log_p))
        # self.append_op2monitor('log_p_max', tf.reduce_max(log_p))
        p = tf.nn.softmax(self.logit_op)
        # self.append_op2monitor('p_min', tf.reduce_min(p))
        # self.append_op2monitor('p_max', tf.reduce_max(p))
        focus = tf.pow(tf.clip_by_value(1-p, 1e-7, 1.), self._config.gamma)
        # self.append_op2monitor('focus', focus)
        loss_op = - tf.to_float(self._labels) * focus * log_p # (None, num_class)
        alphas = tf.constant(self._config.alphas, dtype=tf.float32)
        loss_op *= tf.expand_dims(alphas, 0)
        loss_op = tf.reduce_sum(loss_op, axis=1)
        loss_op = tf.reduce_mean(loss_op)
        self.append_op2monitor('loss', loss_op)

        # log_p = tf.nn.log_softmax(self.logit_op)
        # loss_op = -tf.to_float(self._labels) * log_p
        # loss_op = tf.reduce_sum(loss_op, axis=1)
        # loss_op = tf.reduce_mean(loss_op)
        # self.append_op2monitor('loss', loss_op)

    return loss_op


class TrnTst(framework.model.trntst.TrnTst):
  def feed_data_and_trn(self, data, sess):
    op_dict = self.model.op_in_trn()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(
      [op_dict['loss_op'], op_dict['train_op']],
      feed_dict=feed_dict)

  def feed_data_and_monitor_in_trn(self, data, sess, step):
    op2monitor = self.model.op2monitor
    names = op2monitor.keys()
    ops = op2monitor.values()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(ops, feed_dict=feed_dict)
    for name, val in zip(names, out):
      print '(step %d) monitor "%s":'%(step, name)
      print val

  def feed_data_and_summary(self, data, sess):
    summary_op = self.model.summary_op

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(summary_op, feed_dict=feed_dict)

    return out

  def feed_data_and_run_loss_op_in_val(self, data, sess):
    op_dict = self.model.op_in_val()

    feed_dict = self._construct_feed_dict_in_trn(data)
    loss = sess.run(op_dict['loss_op'], feed_dict=feed_dict)

    return loss

  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model._fts: data[0],
      self.model._ft_masks: data[1],
      self.model._labels: data[2],
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    pass

  def predict_in_tst(self, sess, tst_reader, predict_file):
    op_dict = self.model.op_in_tst()
    tst_batch_size = self.model_cfg.tst_batch_size
    logits = []
    gt_labels = []
    for fts, masks, labels in tst_reader.yield_tst_batch(tst_batch_size):
      logit = sess.run(op_dict['logit_op'], feed_dict={
          self.model._fts: fts,
          self.model._ft_masks: masks,
        })
      logits.append(logit)
      gt_labels.append(labels)
    logits = np.concatenate(logits, axis=0)
    gt_labels = np.concatenate(gt_labels, axis=0)
    np.savez_compressed(predict_file, logits=logits, labels=gt_labels)


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)
    self.trn_video_lst_file = ''
    self.trn_neg_lst_file = ''
    self.val_video_lst_file = ''
    self.trn_ft_track_group_dir = ''
    self.val_ft_track_group_dir = ''
    self.tst_ft_track_group_dir = ''
    self.label_dir = ''
    self.label2lid_file = ''
    self.output_dir = ''
    self.track_lens = []
    self.init_weight_file = ''
    self.tst_video_name = ''


class TrnReader(framework.model.data.Reader):
  def __init__(self, video_lst_file, neg_lst_file, ft_track_group_dir, label_dir, 
      label2lid_file, model_cfg, track_lens=[25, 50]):
    self.ft_track_group_dir = ft_track_group_dir
    self.label_dir = label_dir
    self.cfg = model_cfg
    self.track_lens = track_lens

    self.capacity = 5000

    self.video_names = []
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        self.video_names.append(name)

    self.pos_files = []
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir, 
          '%s.%d.forward.backward.square.pos.0.75.tfrecords'%(video_name, track_len))
        self.pos_files.append(file)
    self.positive_generator = InstanceGenerator(self.pos_files, self.capacity, True,
      num_ft=model_cfg.proto_cfg.num_ft, num_class=model_cfg.num_class)
    self.pos_cnt = self.positive_generator.num_record()

    cam2neg_files = {}
    with open(neg_lst_file) as f:
      for line in f:
        line = line.strip()
        data = line.split('.')
        begin = data[0].rfind('_')
        cam = data[0][begin+1:]
        if cam not in cam2neg_files:
          cam2neg_files[cam] = []
        cam2neg_files[cam].append(os.path.join(self.ft_track_group_dir, line))
    self.negative_cam_generators = []
    for cam in cam2neg_files:
      neg_files = cam2neg_files[cam]
      negative_cam_generator = CircularInstanceGenerator(neg_files, self.capacity/5, True,
        num_ft=model_cfg.proto_cfg.num_ft, num_class=model_cfg.num_class)
      self.negative_cam_generators.append(negative_cam_generator)

  def num_record(self):
    return self.pos_cnt

  def yield_trn_batch(self, batch_size):
    self.positive_generator = InstanceGenerator(self.pos_files, self.capacity, True,
      num_ft=self.cfg.proto_cfg.num_ft, num_class=self.cfg.num_class)
    num_cam = len(self.negative_cam_generators)

    neg_batch_size = batch_size * self.cfg.proto_cfg.trn_neg2pos_in_batch / num_cam
    pos_iterator = self.positive_generator.iterator(batch_size)
    neg_iterators = [
      negative_cam_generator.iterator(neg_batch_size) \
      for negative_cam_generator in self.negative_cam_generators
    ]
    for pos_batch_data in pos_iterator:
      batch_data = [] + pos_batch_data
      # print len(batch_data)
      for neg_iterator in neg_iterators:
        neg_batch_data = neg_iterator.next()
        batch_data.extend(neg_batch_data)
        # print len(neg_batch_data)
      # print len(batch_data), len(pos_batch_data), neg_batch_size
      fts = np.array([d[0] for d in batch_data])
      masks = np.array([d[1] for d in batch_data])
      labels = np.array([d[2] for d in batch_data])
      yield fts, masks, labels

    del self.positive_generator


class ValReader(framework.model.data.Reader):
  def __init__(self, video_lst_file, ft_track_group_dir, label_dir, 
      label2lid_file, model_cfg, track_lens=[25, 50]):
    self.ft_track_group_dir = ft_track_group_dir
    self.label_dir = label_dir
    self.cfg = model_cfg
    self.track_lens = track_lens

    self.capacity = 128 

    self.video_names = []
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        self.video_names.append(name)

  def yield_val_batch(self, batch_size):
    pos_files = []
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir, 
          '%s.%d.forward.backward.square.pos.0.75.tfrecords'%(video_name, track_len))
        pos_files.append(file)
    positive_generator = InstanceGenerator(pos_files, self.capacity, False,
      num_ft=self.cfg.proto_cfg.num_ft, num_class=self.cfg.num_class)

    neg_files = []
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir,
          '%s.%d.forward.backward.square.neg.0.50.0.tfrecords'%(video_name, track_len))
        neg_files.append(file)
    negative_generator = InstanceGenerator(neg_files, self.capacity, False,
      num_ft=self.cfg.proto_cfg.num_ft, num_class=self.cfg.num_class)

    pos_iterator = positive_generator.iterator(batch_size)
    for batch_data in pos_iterator:
      fts = np.array([d[0] for d in batch_data])
      masks = np.array([d[1] for d in batch_data])
      labels = np.array([d[2] for d in batch_data])
      yield fts, masks, labels

    neg_iterator = negative_generator.iterator(batch_size)
    for batch_data in neg_iterator:
      fts = np.array([d[0] for d in batch_data])
      masks = np.array([d[1] for d in batch_data])
      labels = np.array([d[2] for d in batch_data])
      yield fts, masks, labels

    del positive_generator
    del negative_generator


class TstReader(framework.model.data.Reader):
  def __init__(self, tst_video_name, ft_track_group_dir, label_dir,
      label2lid_file, model_cfg, track_lens=[25, 50]):
    self.ft_track_group_dir = ft_track_group_dir
    self.label_dir = label_dir
    self.cfg = model_cfg
    self.track_lens = track_lens

    capacity = 128

    self.video_names = [tst_video_name]

    pos_files = []
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir, 
          '%s.%d.forward.backward.square.pos.0.75.tfrecords'%(video_name, track_len))
    pos_files.append(file)
    self.positive_generator = InstanceGenerator(pos_files, capacity, False,
      num_ft=model_cfg.proto_cfg.num_ft, num_class=model_cfg.num_class)

    neg_files = []
    for track_len in self.track_lens:
      file = os.path.join(self.ft_track_group_dir, 
        '%s.%d.forward.backward.square.neg.0.50.0.5.tfrecords'%(tst_video_name, track_len))
      neg_files.append(file)
    self.negative_generator = InstanceGenerator(neg_files, capacity, False,
      num_ft=model_cfg.proto_cfg.num_ft, num_class=model_cfg.num_class)

  def yield_tst_batch(self, batch_size):
    pos_iterator = self.positive_generator.iterator(batch_size)
    for batch_data in pos_iterator:
      fts = np.array([d[0] for d in batch_data])
      masks = np.array([d[1] for d in batch_data])
      labels = np.array([d[2] for d in batch_data])
      yield fts, masks, labels

    neg_iterator = self.negative_generator.iterator(batch_size)
    for batch_data in neg_iterator:
      fts = np.array([d[0] for d in batch_data])
      masks = np.array([d[1] for d in batch_data])
      labels = np.array([d[2] for d in batch_data])
      yield fts, masks, labels


class InstanceGenerator(input_tool.ShuffleBatchJoin):
  def __init__(self, files, capacity, shuffle_files, **kwargs):
    input_tool.ShuffleBatchJoin.__init__(self, files, capacity, shuffle_files)
    self.num_ft = kwargs['num_ft']
    self.num_class = kwargs['num_class']

  def generate_data_from_record(self, example):
    feature = example.features.feature
    num = int(feature['num'].int64_list.value[0])
    dim_ft = int(feature['dim_ft'].int64_list.value[0])

    fts = feature['ft'].bytes_list.value[0]
    fts = np.fromstring(fts, dtype=np.float32).reshape(num, dim_ft)
    ft, mask = _norm_ft_buffer(fts, self.num_ft, dim_ft)

    lid = int(feature['label'].int64_list.value[0])
    label = np.zeros((self.num_class,), dtype=np.int32)
    label[lid] = 1

    return ft, mask, label


class CircularInstanceGenerator(input_tool.CircularShuffleBatchJoin):
  def __init__(self, files, capacity, shuffle_files, **kwargs):
    input_tool.CircularShuffleBatchJoin.__init__(self, files, capacity, shuffle_files)
    self.num_ft = kwargs['num_ft']
    self.num_class = kwargs['num_class']

  def generate_data_from_record(self, example):
    feature = example.features.feature
    num = int(feature['num'].int64_list.value[0])
    dim_ft = int(feature['dim_ft'].int64_list.value[0])

    fts = feature['ft'].bytes_list.value[0]
    fts = np.fromstring(fts, dtype=np.float32).reshape(num, dim_ft)
    ft, mask = _norm_ft_buffer(fts, self.num_ft, dim_ft)

    lid = int(feature['label'].int64_list.value[0])
    label = np.zeros((self.num_class,), dtype=np.int32)
    label[lid] = 1

    return ft, mask, label


def _norm_ft_buffer(ft_buffer, num_ft, dim_ft):
  ft = np.zeros((num_ft, dim_ft))
  mask = np.zeros((num_ft,), dtype=np.int32)
  num_valid = min(len(ft_buffer), num_ft)
  idxs = range(len(ft_buffer))
  random.shuffle(idxs)
  ft[:num_valid] = np.array(ft_buffer)[idxs[:num_valid]]
  mask[:num_valid] = 1
  return ft, mask
