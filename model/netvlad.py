import os
import random
import cPickle

import numpy as np
import tensorflow as tf

import framework.model.proto
import framework.model.trntst
import framework.model.data


class Config(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_center = 0
    self.dim_output = 0
    self.trn_neg2pos_in_batch = 10
    self.val_neg2pos_in_batch = 1

    self.centers = np.empty((0,)) # (dim_ft, num_center)
    self.alpha = np.empty((0,))


class ConfigWB(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_center = 0
    self.dim_output = 0
    self.trn_neg2pos_in_batch = 10
    self.val_neg2pos_in_batch = 1

    self.centers = np.empty((0,)) # (dim_ft, num_center)


class ModelCfg(framework.model.proto.FullModelConfig):
  def __init__(self):
    framework.model.proto.FullModelConfig.__init__(self)

    self.proto_cfg = Config()

    self.num_class = -1

  def load(self, file):
    data = framework.model.proto.FullModelConfig.load(self, file)
    self.proto_cfg.load(data['proto'])


class ModelWBCfg(ModelCfg):
  def __init__(self):
    ModelCfg.__init__(self)

    self.proto_cfg = ConfigWB()


class ModelWBFocalLossCfg(ModelWBCfg):
  def __init__(self):
    ModelWBCfg.__init__(self)
    self.gamma = 2
    self.alpha = .25


class NetVladEncoder(framework.model.proto.ModelProto):
  namespace = 'netvlad.NetVladEncoder'

  def __init__(self, config):
    framework.model.proto.ModelProto.__init__(self, config)

    # input
    self._fts = tf.no_op() # (None, num_ft, dim_ft)
    self._ft_masks = tf.no_op() # (None, num_ft)
    # output
    self._feature_op = tf.no_op()

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
  def feature_op(self):
    return self._feature_op

  def build_parameter_graph(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.centers = tf.Variable(
          np.array(self._config.centers, dtype=np.float32), name='centers')
        scale = 1.0 / (self._config.dim_ft ** 0.5)
        self.alpha = tf.Variable(
          np.array(self._config.alpha, dtype=np.float32), name='alpha')

        dim_vlad = self._config.dim_ft * self._config.num_center
        scale = 1.0 / (dim_vlad ** 0.5)
        self.fc_W = tf.get_variable('fc_W',
          shape=(dim_vlad, self._config.dim_output), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(scale, scale))
        self.fc_B = tf.get_variable('fc_B',
          shape=(self._config.dim_output,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        w = 2.0 * self.alpha * self.centers # (dim_ft, num_center)
        b = - self.alpha * tf.reduce_sum(self.centers**2, axis=0)# (num_center)
        fts = tf.reshape(self._fts, (-1, self._config.dim_ft)) # (None*num_ft, dim_ft)
        logits = tf.nn.xw_plus_b(fts, w, b) # (None*num_ft, num_center)
        a = tf.nn.softmax(logits) 

        a = tf.expand_dims(a, 1) # (None*num_ft, 1, num_center)
        fts = tf.expand_dims(fts, 2) # (None*num_ft, dim_ft, num_center)
        centers = tf.expand_dims(self.centers, 0) # (1, dim_ft, num_center)
        V_ijk = a * (fts - centers) # (None*num_ft, dim_ft, num_center)
        mask = tf.reshape(self._ft_masks, (-1, 1, 1))
        V_ijk *= mask
        dim_vlad = self._config.dim_ft* self._config.num_center
        V_ijk = tf.reshape(V_ijk, (-1, self._config.num_ft, dim_vlad))
        V_jk = tf.reduce_sum(V_ijk, 1)

        self._feature_op = tf.nn.xw_plus_b(V_jk, self.fc_W, self.fc_B)

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)


class NetVladWBEncoder(NetVladEncoder):
  namespace = 'netvlad.NetVladWBEncoder'

  # @override
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

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        fts = tf.reshape(self._fts, (-1, self._config.dim_ft)) # (None*num_ft, dim_ft)
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
        V_jk = tf.reduce_sum(V_ijk, 1)

        self._feature_op = tf.nn.xw_plus_b(V_jk, self.fc_W, self.fc_B)

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)


class NetVladModel(framework.model.proto.FullModel):
  name_scope = 'netvlad.NetVladModel'

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
          initializer=tf.random_uniform_initializer(scale, scale))
        self.fc_class_B = tf.get_variable('fc_class_B',
          shape=(self._config.num_class), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def _add_predict_layer(self, feature_op):
    feature_op = tf.nn.relu(feature_op)
    logit_op = tf.nn.xw_plus_b(feature_op, self.fc_class_W, self.fc_class_B) # (None, num_class)
    return logit_op

  def _build_inference_graph_in_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.logit_op = self._add_predict_layer(self.model_proto.feature_op)

  def _build_inference_graph_in_trn_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_trn_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.logit_op = self._add_predict_layer(self.model_proto.feature_op)

  def add_loss(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels, logits=self.logit_op))
        self.append_op2monitor('loss', loss_op)

    return loss_op

  def op_in_val(self):
    return {
      'loss_op': self.loss_op,
      'logit_op': self.logit_op,
    }

  def op_in_tst(self):
    return {
      'logit_op': self.logit_op
    }


class NetVladWBModel(NetVladModel):
  name_scope = 'netvlad.NetVladWBModel'

  def get_model_proto(self):
    nv = NetVladWBEncoder(self.config.proto_cfg)
    return nv


class NetVladWBFocalLossModel(NetVladWBModel):
  name_scope = 'netvlad.NetVladWBFocalLossModel'

  def add_loss(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        log_p = tf.nn.log_softmax(self.logit_op)
        p = tf.nn.softmax(self.logit_op)
        loss_op = - tf.pow(1-p, self._config.gamma) * log_p
        # TODO
        loss_op = tf.reduce_mean(loss_op)
        self.append_op2monitor('loss', loss_op)

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

    self.video_names = []
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        self.video_names.append(name)

    self.label2lid = {}
    with open(label2lid_file) as f:
      self.label2lid = cPickle.load(f)

    self.pos_fts = []
    self.pos_masks = []
    self.pos_labels = []
    self.pos_idxs = []

    self.cam2neg_files = {}

    load_positive_ft_label(self)
    np.random.shuffle(self.pos_idxs)
    print 'positive load complete'

    self._prepare_neg_files(neg_lst_file)
    cam_neg_files = self.cam2neg_files.values()
    for neg_files in cam_neg_files:
      random.shuffle(neg_files)
    self.neg_instance_provider = NegInstanceProvider(cam_neg_files, self.cfg, shuffle=True)

  def _prepare_neg_files(self, neg_lst_file):
    self.cam2neg_files = {}
    with open(neg_lst_file) as f:
      for line in f:
        line = line.strip()
        begin = line.rfind('_')
        end = line.find('.')
        cam = line[begin+1:end]
        if cam not in self.cam2neg_files:
          self.cam2neg_files[cam] = []
        self.cam2neg_files[cam].append(os.path.join(self.ft_track_group_dir, line))

  def num_record(self):
    return self.pos_idxs.shape[0]

  def yield_trn_batch(self, batch_size):
    num_pos = self.pos_idxs.shape[0]
    for i in range(0, num_pos, batch_size):
      idxs = self.pos_idxs[i:i+batch_size]
      num = idxs.shape[0]
      pos_fts = self.pos_fts[idxs]
      pos_masks = self.pos_masks[idxs]
      pos_labels = self.pos_labels[idxs]
      neg_fts, neg_masks, neg_labels = self.neg_instance_provider.next_batch(
        batch_size * self.cfg.proto_cfg.trn_neg2pos_in_batch)
      fts = np.concatenate([pos_fts, neg_fts], axis=0)
      masks = np.concatenate([pos_masks, neg_masks], axis=0)
      labels = np.concatenate([pos_labels, neg_labels], axis=0)
      yield fts, masks, labels


class ValReader(framework.model.data.Reader):
  def __init__(self, video_lst_file, ft_track_group_dir, label_dir, 
      label2lid_file, model_cfg, track_lens=[25, 50]):
    self.ft_track_group_dir = ft_track_group_dir
    self.label_dir = label_dir
    self.cfg = model_cfg
    self.track_lens = track_lens

    self.video_names = []
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        self.video_names.append(name)

    self.label2lid = {}
    with open(label2lid_file) as f:
      self.label2lid = cPickle.load(f)

    self.pos_fts = []
    self.pos_masks = []
    self.pos_labels = []
    self.pos_idxs = []

    self.cam2neg_files = {}

    load_positive_ft_label(self)

    self._load_negative_ft_label()

  def _load_negative_ft_label(self):
    num = 0
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir,
          '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        _neg_fts, _neg_masks, _ = load_neg_chunk(file, self.cfg, False)
        num += len(_neg_fts)
        del _neg_fts, _neg_masks

    self.neg_fts = np.zeros((num, self.cfg.proto_cfg.num_ft, self.cfg.proto_cfg.dim_ft))
    self.neg_masks = np.zeros((num, self.cfg.proto_cfg.num_ft))
    self.neg_labels = np.zeros((num, self.cfg.num_class), dtype=np.int32)
    self.neg_labels[:, 0] = 1
    base = 0
    for video_name in self.video_names:
      for track_len in self.track_lens:
        file = os.path.join(self.ft_track_group_dir,
          '%s.%d.forward.backward.square.neg.0.50.0.npz'%(video_name, track_len))
        _neg_fts, _neg_masks, _ = load_neg_chunk(file, self.cfg, False)
        num = len(_neg_fts)
        self.neg_fts[base:base+num] = np.array(_neg_fts)
        self.neg_masks[base:base+num] = np.array(_neg_masks)
        del _neg_fts, _neg_masks
        base += num
    self.neg_idxs = np.arange(self.neg_fts.shape[0])

  def num_record(self):
    return self.pos_idxs.shape[0]

  def yield_val_batch(self, batch_size):
    num_pos = self.pos_idxs.shape[0]
    num_batch = (num_pos + batch_size -1) / batch_size
    neg_batch_size = self.neg_fts.shape[0] / num_batch
    idx_batch = 0
    for i in range(0, num_pos, batch_size):
      idxs = self.pos_idxs[i:i+batch_size]
      num = idxs.shape[0]
      pos_fts = self.pos_fts[idxs]
      pos_masks = self.pos_masks[idxs]
      pos_labels = self.pos_labels[idxs]

      neg_fts = self.neg_fts[idx_batch*neg_batch_size:(idx_batch+1)*neg_batch_size]
      neg_masks = self.neg_masks[idx_batch*neg_batch_size:(idx_batch+1)*neg_batch_size]
      neg_labels = self.neg_labels[idx_batch*neg_batch_size:(idx_batch+1)*neg_batch_size]

      fts = np.concatenate([pos_fts, neg_fts], axis=0)
      masks = np.concatenate([pos_masks, neg_masks], axis=0)
      labels = np.concatenate([pos_labels, neg_labels], axis=0)

      yield fts, masks, labels

      idx_batch += 1


class TstReader(framework.model.data.Reader):
  def __init__(self, tst_video_name, ft_track_group_dir, 
      label2lid_file, model_cfg, track_lens=[25, 50]):
    self.ft_track_group_dir = ft_track_group_dir
    self.cfg = model_cfg
    self.neg_lst = neg_lst
    self.track_lens = track_lens

    self.video_names = [tst_video_name]
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)

        self.video_names.append(name)

    self.label2lid = {}
    with open(label2lid_file) as f:
      self.label2lid = cPickle.load(f)

    self.pos_fts = []
    self.pos_masks = []
    self.pos_labels = []
    self.pos_idxs = []

    self.cam2neg_files = {}

    load_positive_ft_label(self)
    print 'positive load complete'
    self._prepare_neg_ft_label(tst_video_name)

  def _load_neg_ft_label(self, tst_video_name):
    self.neg_fts = []
    self.neg_masks = []
    self.neg_labels = []
    for track_len in self.track_lens:
      file = os.path.join(self.ft_track_group_dir, 
        '%s.%d.forward.backward.square.neg.0.50.0.5.npz'%(tst_video_name, track_len))
      _neg_fts, _neg_masks, _ = load_neg_chunk(file, self.cfg, False)
      neg_fts += _neg_fts
      neg_masks += _neg_masks
    num = len(self.neg_fts)
    self.neg_labels = np.zeros((num, self.cfg.num_class), dtype=np.int32)

  # assumption: will only process one video
  def yield_tst_batch(self, batch_size):
    cam_neg_files = self.cam2neg_files.values()

    # pos instances
    num_pos = len(self.pos_idxs)
    for i in range(0, num_pos, batch_size):
      idxs = self.pos_idxs[i:i+batch_size]
      fts = self.pos_fts[idxs]
      masks = self.pos_masks[idxs]
      labels = self.pos_labels[idxs]
      yield fts, masks, labels

    # neg instances
    num_neg = len(self.neg_fts)
    for i in range(0, num_neg, batch_size):
     fts = np.array(self.neg_fts[i:i+batch_size])
     masks = np.array(self.neg_masks[i:i+batch_size])
     labels = self.neg_labels[i:i+batch_size]
     yield fts, masks, labels


def norm_ft_buffer(ft_buffer, num_ft, dim_ft):
  ft = np.zeros((num_ft, dim_ft))
  mask = np.zeros((num_ft,), dtype=np.int32)
  num_valid = min(len(ft_buffer), num_ft)
  idxs = range(len(ft_buffer))
  random.shuffle(idxs)
  ft[:num_valid] = np.array(ft_buffer)[idxs[:num_valid]]
  mask[:num_valid] = 1
  return ft, mask


def load_positive_ft_label(reader):
  reader.pos_fts = []
  reader.pos_masks = []
  reader.pos_labels = []
  for video_name in reader.video_names:
    for track_len in reader.track_lens:
      label_file = os.path.join(reader.label_dir, 
        '%s.%d.forward.backward.square.0.75.pos'%(video_name, track_len))
      id2lid = {}
      with open(label_file) as f:
        for line in f:
          line = line.strip()
          data = line.split(' ')
          id = int(data[0])
          if data[1] in reader.label2lid:
            lid = reader.label2lid[data[1]]
            id2lid[id] = lid

      ft_file = os.path.join(reader.ft_track_group_dir, 
        '%s.%d.forward.backward.square.pos.0.75.npz'%(video_name, track_len))
      data = np.load(ft_file)
      ids = data['ids']
      fts = data['fts']
      num = ids.shape[0]
      prev_id = ids[0]
      ft_buffer = []
      for i in range(num):
        id = ids[i]
        if id != prev_id:
          if id in id2lid:
            pos_ft, pos_mask = norm_ft_buffer(
              ft_buffer, reader.cfg.proto_cfg.num_ft, reader.cfg.proto_cfg.dim_ft)
            reader.pos_fts.append(pos_ft)
            reader.pos_masks.append(pos_mask)
            label = np.zeros((reader.cfg.num_class,), dtype=np.int32)
            label[id2lid[prev_id]] = 1
            reader.pos_labels.append(label)

          ft_buffer = []
          prev_id = id
        ft_buffer.append(fts[i])

      if id in id2lid:
        pos_ft, pos_mask = norm_ft_buffer(
          ft_buffer, reader.cfg.proto_cfg.num_ft, reader.cfg.proto_cfg.dim_ft)
        reader.pos_fts.append(pos_ft)
        reader.pos_masks.append(pos_mask)
        label = np.zeros((reader.cfg.num_class,), dtype=np.int32)
        label[id2lid[prev_id]] = 1
        reader.pos_labels.append(label)
  reader.pos_fts = np.array(reader.pos_fts)
  reader.pos_masks = np.array(reader.pos_masks)
  reader.pos_labels = np.array(reader.pos_labels)
  reader.pos_idxs = np.arange(reader.pos_fts.shape[0])


# iterate files under each camera separately
class NegInstanceProvider(object):
  def __init__(self, cam_neg_files, cfg, shuffle=True):
    self.num_cam = len(cam_neg_files)
    self.cam_neg_files = cam_neg_files # loop queue
    self.cur_file_idxs = [0 for _ in range(self.num_cam)]
    self.num_files = [len(cam_neg_file) for cam_neg_file in cam_neg_files]
    self.cfg = cfg
    self.shuffle = shuffle

    self.cam_fts = []
    self.cam_masks = []
    self.cam_idxs = []
    self.cur_idxs = [0 for _ in range(self.num_cam)]
    for c in range(self.num_cam):
      neg_file = self.cam_neg_files[c][0]
      neg_fts, neg_masks, neg_idxs = load_neg_chunk(neg_file, self.cfg, self.shuffle)

      self.cam_fts.append(neg_fts)
      self.cam_masks.append(neg_masks)
      self.cam_idxs.append(neg_idxs)

      print 'load', neg_file

  def next_batch(self, batch_size):
    num = batch_size / self.num_cam
    fts = []
    masks = []
    for c in range(self.num_cam):
      _num = num
      if self.cur_idxs[c] + num > len(self.cam_fts[c]):
        if self.cur_idxs[c] == 0:
          _num = len(self.cam_fts[c])
        else:
          self.cur_file_idxs[c] = (self.cur_file_idxs[c] + 1) % self.num_files[c]
          self.cur_idxs[c] = 0
          neg_file = self.cam_neg_files[c][self.cur_file_idxs[c]]
          neg_fts, neg_masks, neg_idxs = load_neg_chunk(neg_file, self.cfg, self.shuffle)
          past_fts = self.cam_fts[c]
          past_masks = self.cam_masks[c]
          past_idxs = self.cam_idxs[c]
          self.cam_fts[c] = neg_fts
          self.cam_masks[c] = neg_masks
          self.cam_idxs[c] = neg_idxs
          del past_fts, past_masks, past_idxs
          _num = min(_num, len(self.cam_fts[c]))
          print 'load', neg_file
      # print len(self.cam_idxs[c]), self.cur_idxs[c], num
      for i in range(_num):
        idx = self.cam_idxs[c][self.cur_idxs[c] + i]
        ft = self.cam_fts[c][idx]
        mask = self.cam_masks[c][idx]
        fts.append(ft)
        masks.append(mask)
      self.cur_idxs[c] += _num
    fts = np.array(fts)
    masks = np.array(masks)
    labels = np.zeros((fts.shape[0], self.cfg.num_class), dtype=np.int32)
    labels[:, 0] = 1
    return fts, masks, labels


def load_neg_chunk(neg_file, cfg, shuffle):
  neg_fts = []
  neg_masks = []
  neg_idxs = []

  data = np.load(neg_file)
  ids = data['ids']
  fts = data['fts']
  previd = ids[0]
  num = ids.shape[0]
  ft_buffer = []
  for i in range(num):
    id = ids[i]
    if id != previd:
      neg_ft, neg_mask = norm_ft_buffer(
        ft_buffer, cfg.proto_cfg.num_ft, cfg.proto_cfg.dim_ft)
      neg_fts.append(neg_ft)
      neg_masks.append(neg_mask)

      ft_buffer = []
      previd = id
    ft_buffer.append(fts[i])
  neg_ft, neg_mask = norm_ft_buffer(
    ft_buffer, cfg.proto_cfg.num_ft, cfg.proto_cfg.dim_ft)
  neg_fts.append(neg_ft)
  neg_masks.append(neg_mask)
  neg_idxs = range(len(neg_fts))
  if shuffle:
    random.shuffle(neg_idxs)

  return neg_fts, neg_masks, neg_idxs
