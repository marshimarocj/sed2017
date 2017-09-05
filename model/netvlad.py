import framework.model.proto


class Config(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_center = 0
    self.dim_output = 0

    self.centers = np.empty((0,)) # (dim_ft, num_center)
    self.alpha = np.empty((0,))


class ModelCfg(framework.model.proto.FullModelConfig):
  def __init__(self):
    framework.model.proto.FullModelConfig.__init__(self)

    self.proto_cfg = Config()

    self.num_class = -1

  def load(self, file):
    data = framework.model.proto.FullModelConfig.load(self, file)
    self.proto_cfg.load(data['proto'])


class NetVladEncoder(framework.proto.ModelProto):
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
    self._fts_masks = val

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
          shape=(self._config.dim_output), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        w = 2.0 * self.alpha * self.centers # (dim_ft, num_center)
        b = - self.alpha * tf.reduce_sum(self.centers**2, axis=0)# (num_center)
        b = tf.expand_dims(b, 0)
        fts = tf.reshape(self._fts, (-1, self._config.dim_ft)) # (None*num_ft, dim_ft)
        logits = tf.nn.xw_plus_b(fts, w, b) # (None*num_ft, num_center)
        a = tf.softmax(logits) 

        a = tf.expand_dims(a, 1) # (None*num_ft, 1, num_center)
        fts = tf.expand_dims(fts, 2) # (None*num_ft, dim_ft, num_center)
        centers = tf.expand_dims(centers, 0) # (1, dim_ft, num_center)
        V_ijk = a * (fts - centers) # (None*num_ft, dim_ft, num_center)
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

    self.model_proto.fts = self._fts
    self.model_proto.ft_masks = self._ft_masks

  def add_trn_tst_input(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._fts = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft, self.config.proto_cfg.dim_ft), name='fts')
        self._ft_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.proto_cfg.num_ft), name='ft_masks')
        self._labels = tf.placeholder(
          tf.int32, shape=(None, self.config.num_class), name='labels')

    self.model_proto.fts = self._fts
    self.model_proto.ft_masks = self._ft_masks

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
        loss_op = tf.nn.softmax_cross_entropy_with_logits(self._labels, self.logit_op)

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
    pass


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)
    self.trn_video_lst_file = ''
    self.val_video_lst_file = ''
    self.ft_track_group_dir = ''
    self.label_dir = ''
    self.label2lid_file = ''


class Reader(framework.model.data.Reader):
  def __init__(self, video_lst_file, ft_track_group_dir, label_dir, label2lid_file,
      neg_lst = [0]):
    self.ft_track_group_dir = ft_grack_group_dir
    self.label_dir = label_dir

    self.video_names = []
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' in name:
          continue

        video_names.append(name)

    self.label2lid = {}
    with open(label2lid_file) as f:
      self.label2lid = cPickle.load(f)

  def num_record(self):
    pass
