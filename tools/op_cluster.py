#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import pickle
import math

def distance(list1, list2):
    assert len(list1) == len(list2)
    dis_sqr = 0
    for d1, d2 in zip(list1, list2):
        dis_sqr += ((d1 - d2) * (d1 - d2))
    return math.sqrt(dis_sqr)

def main():

    parser = argparse.ArgumentParser(
        description="cluster operators according to initial cluster centers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "profiling results are saved")
    parser.add_argument('init_centers', help="specify the file of initial centers")
    parser.add_argument('--output_centers', default="op_centers.txt",
                        help="a path to save output centers")
    parser.add_argument('--suffix', default="pickle", help="specify the suffix of "
                        "result files")
    parser.add_argument('--threshold', default=0.01, type=float,
                        help="specify threshold of center distance to stop k-means")

    args = parser.parse_args()

    total_op_list = []
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                for each_op in op_list:
                    if each_op.parallelism is None:
                        print('op_type: {}, model: {}'.format(each_op.op_type,
                                                              input_file))
                total_op_list.extend(op_list)

    initial_centers = []
    with open(args.init_centers, 'r') as f:
        lines = f.readlines()
        for line in lines:
            center = []
            locs = line.split(',')
            for each_loc in locs:
                each_loc.strip('\n ')
                center.append(float(each_loc))

            initial_centers.append(center)

    print(initial_centers)

    data_points = []
    max_locality = 0.0
    for each_op in total_op_list:
        if (not each_op.is_aid_op) and (each_op.comp_instrs != 0):
            locality = (each_op.mem_trans / each_op.comp_instrs)
            if locality > 30:
                print('op_type: {}, locality: {}'.format(each_op.op_type,
                                                         locality))

            max_locality = max(max_locality, locality)
            data_points.append([each_op.parallelism, locality])

    print('Max locality: {}'.format(max_locality))
    for i in range(len(data_points)):
        data_points[i][1] = data_points[i][1] / max_locality

    print('Starting k-means cluster...')

    iter_num = 0
    while True:
        iter_num += 1
        new_centers = []
        new_count = []
        for i in range(len(initial_centers)):
            new_centers.append([0.0, 0.0])
            new_count.append(0.0)

        for point in data_points:
            cluster_id = 0
            min_dis = 3.0

            for i in range(len(initial_centers)):
                curr_dis = distance(initial_centers[i], point)
                if curr_dis < min_dis:
                    min_dis = curr_dis
                    min_i = i

            new_centers[min_i][0] += point[0]
            new_centers[min_i][1] += point[1]
            new_count[min_i] += 1

        for i in range(len(new_centers)):
            new_centers[i][0] /= new_count[i]
            new_centers[i][1] /= new_count[i]

        total_dis = 0
        for i in range(len(initial_centers)):
            total_dis += distance(initial_centers[i], new_centers[i])

        print('Iteration: {}, Total center moving distance: {}'.format(iter_num,
                                                                       total_dis))

        for i in range(len(initial_centers)):
            initial_centers[i][0] = new_centers[i][0]
            initial_centers[i][1] = new_centers[i][1]

        if total_dis < args.threshold:
            break

    print('Cluster results:')
    with open(args.output_centers, 'w') as f:
        for i in range(len(initial_centers)):
            output_str = '{}, {}'.format(initial_centers[i][0], initial_centers[i][1])
            f.write(output_str)
            f.write('\n')
            print(output_str)


if __name__ == '__main__':
    main()
