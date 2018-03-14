# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model wrapper class for performing inference with a ShowAndTellModel."""

from im2txt import show_and_tell_model
from im2txt.inference_utils import inference_wrapper_base

from cg_profiler.cg_graph import CompGraph
import tensorflow as tf
import sys

class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with a ShowAndTellModel."""

  def __init__(self):
    super(InferenceWrapper, self).__init__()

  def build_model(self, model_config):
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="inference")
    model.build()
    return model

  def feed_image(self, sess, encoded_image):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    print('Model im2txt_feed_image start running')
    sys.stdout.flush()
    initial_state = sess.run(fetches="lstm/initial_state:0",
                             feed_dict={"image_feed:0": encoded_image},
                             options=options, run_metadata=run_metadata)
    print('Model im2txt_feed_image stop')
    sys.stdout.flush()
    #cg = CompGraph('im2txt_feed_image', run_metadata, tf.get_default_graph())

    #cg_tensor_dict = cg.get_tensors()
    #cg_sorted_keys = sorted(cg_tensor_dict.keys())
    #cg_sorted_items = []
    #for cg_key in cg_sorted_keys:
    #  cg_sorted_items.append(tf.shape(cg_tensor_dict[cg_key]))

    #cg_sorted_shape = sess.run(cg_sorted_items,
    #                           feed_dict={"image_feed:0": encoded_image})
    #cg.op_analysis(dict(zip(cg_sorted_keys, cg_sorted_shape)),
    #               'im2txt_feed_image.pickle')

    return initial_state

  def inference_step(self, sess, input_feed, state_feed):

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    print('Model im2txt_infer_step start running')
    sys.stdout.flush()
    softmax_output, state_output = sess.run(
        fetches=["softmax:0", "lstm/state:0"],
        feed_dict={
            "input_feed:0": input_feed,
            "lstm/state_feed:0": state_feed,
        }, options = options, run_metadata=run_metadata)
    print('Model im2txt_infer_step stop')
    sys.stdout.flush()
    #cg = CompGraph('im2txt_infer_step', run_metadata, tf.get_default_graph())

    #cg_tensor_dict = cg.get_tensors()
    #cg_sorted_keys = sorted(cg_tensor_dict.keys())
    #cg_sorted_items = []
    #for cg_key in cg_sorted_keys:
    #  cg_sorted_items.append(tf.shape(cg_tensor_dict[cg_key]))

    #cg_sorted_shape = sess.run(cg_sorted_items,
    #                           feed_dict={"input_feed:0": input_feed,
    #                                      "lstm/state_feed:0": state_feed})
    #cg.op_analysis(dict(zip(cg_sorted_keys, cg_sorted_shape)),
    #               'im2txt_infer_step.pickle')
    exit(0)

    return softmax_output, state_output, None
