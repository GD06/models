#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import pickle
import math
import numpy as np
import copy

def distance(list1, list2):
    assert len(list1) == len(list2)
    dis_sqr = 0
    for d1, d2 in zip(list1, list2):
        dis_sqr += (d1 - d2) * (d1 - d2)
    return math.sqrt(dis_sqr)

def extract_app_feature(op_list, op_centers, max_locality):
    total_time = 0.0
    each_center_time = []

    for i in range(len(op_centers)):
        each_center_time.append(0.0)

    for each_op in op_list:
        if (not each_op.is_aid_op) and (each_op.comp_instrs != 0):

            locality = (each_op.mem_trans / each_op.comp_instrs)
            norm_loc = locality / max_locality

            dis_array = [distance([each_op.parallelism, norm_loc], c)
                         for c in op_centers]
            min_i = np.argmin(dis_array)

            elapsed_time = float(each_op.elapsed_time)
            total_time += elapsed_time
            each_center_time[min_i] += elapsed_time

    app_feature = [(t / total_time) for t in each_center_time]
    return app_feature

def merge_two_clusters(label_a, label_b, app_label):
    assert label_a != label_b
    for app_name in app_label.keys():
        if app_label[app_name] == label_a:
            app_label[app_name] = label_b
    return

def update_label_center(app_dict, app_label):
    label_center = {}
    num_points = {}

    for app_name in app_label.keys():
        ndims = len(app_dict[app_name])
        label = app_label[app_name]
        label_center[label] = np.zeros(ndims).tolist()
        num_points[label] = 0

    for app_name in app_label.keys():
        label = app_label[app_name]
        ndims = len(app_dict[app_name])
        for i in range(ndims):
            label_center[label][i] += app_dict[app_name][i]
        num_points[label] += 1

    for label in label_center.keys():
        ndims = len(label_center[label])
        for i in range(ndims):
            label_center[label][i] /= num_points[label]

    return label_center

def comp_cohesion(app_dict, app_label, label_center):
    dis_array = [distance(app_dict[x], label_center[app_label[x]])
                 for x in app_dict.keys()]
    return np.max(dis_array)

def main():

    parser = argparse.ArgumentParser(
        description="cluster applications according to operator cluster results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "profiling results are saved")
    parser.add_argument('--op_centers', default="op_centers.txt",
                        help="specify the file of op centers")
    parser.add_argument('--suffix', default=".pickle", help="specify the suffix of "
                        "result files")
    parser.add_argument('--max_locality', default="max_locality.txt",
                        help="specify the value of max locality for normalization")
    parser.add_argument('--output_dir', default=None,
                        help="specify the directory where output results are saved")
    parser.add_argument('--output_file', default="app_cluster.pickle",
                        help="specify the filename of application cluster outputs")

    args = parser.parse_args()

    print('Loading Operator Centers...')
    op_centers = []
    with open(args.op_centers, 'r') as f:
        lines = f.readlines()
        for line in lines:
            center = []
            locs = line.split(',')
            for each_loc in locs:
                each_loc.strip('\n ')
                center.append(float(each_loc))
            op_centers.append(center)

    print('Loading Max Locality...')
    max_locality = 25.0
    with open(args.max_locality, 'r') as f:
        line = f.readline()
        line.strip('\n ')
        max_locality = float(line)

    print('Loading and Extracting Operator Features...')
    app_dict = {}
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                index = input_file.find(args.suffix)
                app_name = input_file[:index]
                app_dict[app_name] = extract_app_feature(op_list, op_centers,
                                                         max_locality)


    for app_name in app_dict.keys():
        print('App: {}, feature: {}'.format(app_name, app_dict[app_name]))

    print('Starting App Clustering...')
    app_label = {}
    label_center = {}
    for app_name in app_dict.keys():
        label = len(app_label) + 1
        app_label[app_name] = label
        label_center[label] = app_dict[app_name]

    result_list = []
    iter_num = 0
    tmp_app_label = copy.deepcopy(app_label)
    result_list.append([iter_num, 0.0, tmp_app_label])
    while len(label_center) > 1:
        iter_num += 1

        dis_dict = {}
        for i in label_center.keys():
            for j in label_center.keys():
                if not i == j:
                    dis_dict[(i, j)] = distance(label_center[i], label_center[j])

        label_a, label_b = min(dis_dict.items(), key=lambda x: x[1])[0]

        merge_two_clusters(label_a, label_b, app_label)
        label_center = update_label_center(app_dict, app_label)

        cohesion = comp_cohesion(app_dict, app_label, label_center)
        tmp_app_label = copy.deepcopy(app_label)
        result_list.append([iter_num, cohesion, tmp_app_label])
        print('Iteration: {}, Cohesion: {}'.format(iter_num, cohesion))

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'outputs')

    with open(os.path.join(args.output_dir, args.output_file), 'wb') as f:
        output_tuple = (app_dict, result_list)
        pickle.dump(output_tuple, f)

if __name__ == '__main__':
    main()
