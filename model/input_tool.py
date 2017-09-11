import os
import random
from collections import deque

import numpy as np


class RandomShuffleQueue(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.q = deque([])

  def is_full(self):
    return len(self.q) == self.capacity

  def is_empty(self):
    return len(self.q) == 0

  def enqueue(self, ele):
    self.q.append(ele)

  def dequeue(self):
    r = random.randint(0, len(self.q)-1)
    tmp = self.q[0]
    self.q[0] = self.q[r]
    self.q[r] = tmp
    return self.q.popleft()


class ShuffleBatchJoin(object):
  def __init__(self, files, capacity, shuffle_files, shuffle, **kwargs):
    self.capacity = capacity
    if shuffle:
      self.files = random.shuffle(files)
    self.random_shuffle_queue = RandomShuffleQueue(capacity)

  def generate_data_from_record(self, example):
    raise NotImplementedError("""please customize generate_data_from_record""")

  def next(self, batch_size):
    assert batch_size < self.capacity

    for file in self.files:
      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
      for string_record in record_iterator:
        example = tf.train.Example()
        example.ParserFromString(string_record)
        data = self.generate_data_from_record(example)
        if self.random_shuffle_queue.is_full():
          batch_data = []
          for i in range(batch_size):
            batch_data.append(self.random_shuffle_queue.dequeue())
          yield batch_data
        self.random_shuffle_queue.enqueue(data)
    batch_data = []
    while not self.random_shuffle_queue.is_empty():
      batch_data.append(self.random_shuffle_queue.dequeue())
    yield batch_data


# Note: never ending circular queue
# don't call by for loop
# call next() instead
class CircularShuffleBatchJoin(ShuffleBatchJoin):
  def next(self, batch_size):
    assert batch_size < self.capacity

    while True:
      for file in self.files:
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
        for string_record in record_iterator:
          example = tf.train.Example()
          example.ParserFromString(string_record)
          data = self.generate_data_from_record(example)
          if self.random_shuffle_queue.is_full():
            batch_data = []
            for i in range(batch_size):
              batch_data.append(self.random_shuffle_queue.dequeue())
            yield batch_data
          self.random_shuffle_queue.enqueue(data)
      batch_data = []
      while not self.random_shuffle_queue.is_empty():
        batch_data.append(self.random_shuffle_queue.dequeue())
      yield batch_data
