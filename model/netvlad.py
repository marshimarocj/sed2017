import framework.proto


class Config(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_ft = 0
    self.num_ft = 0
    self.num_center = 0
    self.dim_output = 0

    self.centers = np.empty((0,)) # (dim_ft, num_center)
    self.alpha = np.empty((0,))


class NetVlad(framework.proto.ModelProto):
  namespace = 'netvlad'

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
        logits = tf.xw_plus_b(fts, w, b) # (None*num_ft, num_center)
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

        self._feature_op = tf.xw_plus_b(V_jk, self.fc_W, self.fc_B)

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)