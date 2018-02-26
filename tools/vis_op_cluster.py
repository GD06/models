#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

def distance(x_list, y_list):
    dis_sqr = 0
    for x, y in zip(x_list, y_list):
        dis_sqr += ((x - y) * (x - y))
    return math.sqrt(dis_sqr)

class OpPlot:
    def __init__(self, name='op_type'):
        self.name = name
        self.x_list = []
        self.y_list = []
        return

    def assign_attr(self, op_attr):
        self.marker = op_attr['marker']
        self.color = op_attr['color']
        if 'area' in op_attr:
            self.area = op_attr['area']
        else:
            self.area = np.pi * (3 ** 2)
        return

    def assign_set(self, op_set):
        self.op_set = op_set
        return

    def assign_center(self, x, y):
        self.op_center = [x, y]
        return

cluster_1 = OpPlot('Cluster 1')
cluster_1.assign_attr({'marker': 'o', 'color': 'r'})

cluster_2 = OpPlot('Cluster 2')
cluster_2.assign_attr({'marker': 'v', 'color': 'b'})

cluster_3 = OpPlot('Cluster 3')
cluster_3.assign_attr({'marker': '+', 'color': 'g'})

op_classes = [cluster_1, cluster_2, cluster_3]

def main():

    parser = argparse.ArgumentParser(
        description="draw the figure of clustering results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "profiling results are saved")
    parser.add_argument('op_center', help="specify the file in which cluster "
                        "results are saved")
    parser.add_argument('--suffix', default="pickle", help="specify the suffix of "
                        "result files")
    parser.add_argument('--output_dir', default=None, help="specify the output dir "
                        "to store output results")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'outputs')

    # Reading data from log files
    total_op_list = []
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                total_op_list.extend(op_list)

    with open(args.op_center, 'r') as f:
        lines = f.readlines()

    cnt = 0
    for line in lines:
        nums = line.split(',')
        op_x = float(nums[0].strip('\n '))
        op_y = float(nums[1].strip('\n '))
        op_classes[cnt].assign_center(op_x, op_y)
        cnt += 1

    max_locality = 0
    for each_op in total_op_list:
        if each_op.is_aid_op:
            continue
        if each_op.comp_instrs == 0:
            continue
        locality = (each_op.mem_trans / each_op.comp_instrs)
        max_locality = max(max_locality, locality)

    for each_op in total_op_list:
        if each_op.is_aid_op:
            continue
        if each_op.comp_instrs == 0:
            continue
        locality = (each_op.mem_trans / each_op.comp_instrs)
        if locality > 42:
            print('op_type: {}, locality: {}'.format(each_op.op_type,
                                                     locality))
            continue

        dis_array = [distance([each_op.parallelism, locality / max_locality],
                              each_op_class.op_center) for each_op_class in op_classes]
        index = np.argmin(dis_array)
        op_classes[index].x_list.append(each_op.parallelism)
        op_classes[index].y_list.append(locality)

    for each_op_class in op_classes:
        plt.scatter(each_op_class.x_list, each_op_class.y_list, s=each_op_class.area,
                    c=each_op_class.color, marker=each_op_class.marker,
                    label=each_op_class.name, alpha=0.5)

    plt.legend(loc='upper center')
    #plt.xlim(0, 1.0)
    #plt.ylim(0, 26.0)
    plt.xlabel('Parallelism')
    plt.ylabel('Locality')

    output_fig = os.path.join(args.output_dir, 'op_cluster_result.png')
    plt.savefig(output_fig, format='png')

if __name__ == '__main__':
    main()
