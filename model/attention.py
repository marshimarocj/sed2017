import os
import random
import cPickle

import numpy as np
import tensorflow as tf

import framework.model.proto
import framework.model.trntst
import framework.model.data
import input_tool
import netvlad_clean


class Config(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_attention = 0
    self.trn_neg2pos_in_batch = 10
    self.val_neg2pos_in_batch = 1


class ModelCfg(framework.model.proto.FullModelConfig):
  def __init__(self):
    framework.model.proto.FullModelConfig.__init__(self)

    self.proto_cfg = Config()

    self.num_class = -1
    self.num_hidden = -1
    self.dropout = False

  def load(self, file):
    data = framework.model.proto.FullModelConfig.load(self, file)
    self.proto_cfg.load(data['proto'])

    assert self.proto_cfg.num_attention == self.num_class


class AttentionEncoder(framework.model.proto.ModelProto):
  namespace = 'attention.AttentionEncoder'

  def __init__(self, config):
    framework.model.proto.ModelProto.__init__(self, config)

    # input
    self._fts = tf.no_op() # (None, num_ft, dim_ft)
    self._ft_masks = tf.no_op() # (NOne, num_ft)
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
        scale = 1.0 / (self._config.dim_ft ** 0.5)
        self.A = tf.get_variable('A',
          shape=(self._config.dim_ft, self._config.num_attention), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-scale, scale))

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        fts = tf.reshape(self._fts, [-1, self._config.dim_ft]) # (None*num_ft, dim_ft)
        fts = tf.nn.l2_normalize(fts, dim=1)
        e = tf.matmul(fts, self.A) # (None*num_ft, num_attention)
        e = tf.reshape(e, [-1, self._config.num_ft, 1, self._config.num_attention])
        alpha = tf.nn.softmax(e, dim=1)
        ft_masks = tf.reshape(self._ft_masks, (-1, self._config.num_ft, 1, 1)) 
        alpha *= ft_masks
        alpha /= tf.reduce_sum(alpha, axis=1, keep_dims=True)
        fts = tf.expand_dims(self._fts, 3) # (None, num_ft, dim_ft, 1)
        self._feature_op = tf.reduce_sum(fts * alpha, axis=1) # (None, dim_ft, num_attention)

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)


class AttentionModel(framework.model.proto.FullModel):
  name_scope = 'attention.AttentionModel'

  def __init__(self, config):
    framework.model.proto.FullModel.__init__(self, config)
    self.logit_op = tf.no_op()
    self.predict_op = tf.no_op()

  def get_model_proto(self):
    ae = AttentionEncoder(self.config.proto_cfg)
    return ae

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
        scale = 1.0 / (self._config.proto_cfg.dim_ft**0.5)
        self.hidden_Ws = []
        self.hidden_Bs = []
        for i in range(self._config.num_class):
          hidden_W = tf.get_variable('hidden_W_%d'%i,
            shape=(self._config.proto_cfg.dim_ft, self._config.num_hidden),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-scale, scale))
          hidden_B = tf.get_variable('hidden_B_%d'%i,
            shape=(self._config.num_hidden), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
          self.hidden_Ws.append(hidden_W)
          self.hidden_Bs.append(hidden_B)

        self.fc_Ws = []
        self.fc_Bs = []
        scale = 1. / (self._config.num_hidden**0.5)
        for i in range(self._config.num_class):
          fc_W = tf.get_variable('fc_W_%d'%i,
            shape=(self._config.num_hidden, 1), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-scale, scale))
          fc_B = tf.get_variable('fc_B_%d'%i,
            shape=(1,), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
          self.fc_Ws.append(fc_W)
          self.fc_Bs.append(fc_B)

  def _add_predict_layer(self, feature_op):
    logits = []
    for i in range(self._config.num_class):
      hidden = tf.nn.xw_plus_b(feature_op[:, :, i], self.hidden_Ws[i], self.hidden_Bs[i])
      hidden = tf.nn.relu(hidden) # (None, num_hidden)
      if self._config.dropout:
        hidden = tf.nn.dropout(hidden, 0.5)
      logit = tf.nn.xw_plus_b(hidden, self.fc_Ws[i], self.fc_Bs[i]) # (None, 1)
      logits.append(logit)
    logits = tf.stack(logits, axis=1) # (None, num_class)
    print logits.get_shape()
    return logits

  def _build_inference_graph_in_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.logit_op = self._add_predict_layer(self.model_proto.feature_op)
        self.predict_op = tf.nn.softmax(self.logit_op)

  def _build_inference_graph_in_trn_tst(self, basegraph):
    framework.model.proto.FullModel._build_inference_graph_in_trn_tst(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.logit_op = self._add_predict_layer(self.model_proto.feature_op)
        self.predict_op = tf.nn.softmax(self.logit_op)

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
      'predict_op': self.predict_op,
    }

  def op_in_tst(self):
    return {
      'logit_op': self.logit_op,
      'predict_op': self.predict_op,
    }


class TrnTst(netvlad_clean.TrnTst):
  pass


class PathCfg(netvlad_clean.PathCfg):
  pass


class TrnReader(netvlad_clean.TrnReader):
  pass


class ValReader(netvlad_clean.ValReader):
  pass


class TstReader(netvlad_clean.TstReader):
  pass
