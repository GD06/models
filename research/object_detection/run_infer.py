#!/usr/bin/env python3
#- coding: utf-8 -

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

import argparse
import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from cg_profiler.cg_graph import CompGraph

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def main():

    parser = argparse.ArgumentParser(
        description="run inference by using specified model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', help="specify the model name")
    parser.add_argument('work_dir', help="specify the work space directory")
    parser.add_argument('--model_dir', default=None,
                        help="specify the dir storing models.")

    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        assert os.getenv('MODEL_INPUT_DIR') is not None
        model_dir = os.path.join(os.getenv('MODEL_INPUT_DIR'),
                                 'object_detection')

    model_name = args.model_name
    model_file = model_name + '.tar.gz'
    tar_file = tarfile.open(os.path.join(model_dir, model_file))
    recorded_name = model_name
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            recorded_name = file.name
            tar_file.extract(file, args.work_dir)

    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    PATH_TO_CKPT = os.path.join(args.work_dir, recorded_name)
    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name=model_name)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
                    label_map, max_num_classes=NUM_CLASSES,
                    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,
                                     'image{}.jpg'.format(i))
                        for i in range(1, 2)]

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            image_tensor = detection_graph.get_tensor_by_name(
                '{}/image_tensor:0'.format(model_name))
            detection_boxes = detection_graph.get_tensor_by_name(
                '{}/detection_boxes:0'.format(model_name))
            detection_scores =  detection_graph.get_tensor_by_name(
                '{}/detection_scores:0'.format(model_name))
            detection_classes = detection_graph.get_tensor_by_name(
                '{}/detection_classes:0'.format(model_name))
            num_detections = detection_graph.get_tensor_by_name(
                '{}/num_detections:0'.format(model_name))

            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                results = sess.run([detection_boxes, detection_scores,
                                    detection_classes, num_detections],
                                   feed_dict={image_tensor: image_np_expanded},
                                   options=options, run_metadata=run_metadata)
                cg = CompGraph(model_name, run_metadata, detection_graph)

                cg_tensor_dict = cg.get_tensors()
                cg_sorted_keys = sorted(cg_tensor_dict.keys())
                #cg_sorted_shape = []
                #for cg_key in cg_sorted_keys:
                #    print(cg_key)
                #    t = tf.shape(cg_tensor_dict[cg_key])
                #    cg_sorted_shape.append(t.eval(feed_dict={image_tensor: image_np_expanded},
                #                                  session=sess))

                cg_sorted_items = []
                for cg_key in cg_sorted_keys:
                    cg_sorted_items.append(tf.shape(cg_tensor_dict[cg_key]))

                cg_sorted_shape = sess.run(cg_sorted_items,
                                            feed_dict={image_tensor: image_np_expanded})
                cg.op_analysis(dict(zip(cg_sorted_keys, cg_sorted_shape)),
                               '{}.pickle'.format(model_name))

                print('Image: {}, number of detected: {}'.format(
                    image_path, len(results[3])))

if __name__ == '__main__':
    main()
