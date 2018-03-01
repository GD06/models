#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

line_style=['-', '--', '-.', ':']
marker_style=['o', 'v', '+']
color_style=['r', 'b', 'g']

def main():

    parser = argparse.ArgumentParser(
        description="draw the figure of application clustering results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "cluster results are saved")
    parser.add_argument('--suffix', default=".pickle", help="specify the suffix of "
                        "results are saved")
    parser.add_argument('--output_dir', default=None, help="specify the output dir "
                        "to store output results")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'outputs')

    cnt = 0
    cluster_dict = {}
    feature_dict = {}
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                index = input_file.find(args.suffix)
                platform_name = input_file[:index]

                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    app_dict, result_dict = pickle.load(f)

                x_list = [snapshot[0] for snapshot in result_dict]
                y_list = [snapshot[1] for snapshot in result_dict]
                delta = [(y_list[i+1] - y_list[i]) for i in range(0, len(y_list) - 1)]
                stop_index = np.argmax(delta)
                print('Platform: {}, Stop Index: {}'.format(platform_name, stop_index))
                cluster_dict[platform_name] = result_dict[stop_index][2]
                feature_dict[platform_name] = app_dict

                plt.plot(x_list, y_list, line_style[cnt], label=platform_name)
                cnt += 1

    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Cohesion')

    plt.savefig(os.path.join(args.output_dir, 'app_cohesion.png'), format='png')

    for platform_name in cluster_dict.keys():
        print('{} Application Cluster Result:'.format(platform_name))
        app_dict = feature_dict[platform_name]
        app_label = cluster_dict[platform_name]
        plt.figure()

        label_dict = {}
        for app_name in app_label.keys():
            label_dict[app_label[app_name]] = []
        for app_name in app_label.keys():
            label_dict[app_label[app_name]].append(app_name)

        cnt = 0
        for label in label_dict.keys():
            app_string = ', '.join(sorted(label_dict[label]))
            print('Cluster {}: {}'.format(cnt + 1, app_string))
            x_list = [app_dict[app_name][1] for app_name in label_dict[label]]
            y_list = [app_dict[app_name][2] for app_name in label_dict[label]]
            plt.scatter(x_list, y_list, s=(np.pi * (3 ** 2)), c=color_style[cnt],
                        marker=marker_style[cnt], label='Cluster {}'.format(cnt + 1),
                        alpha=0.8)
            cnt += 1

        plt.legend(loc='best')
        plt.xlabel('$R_2$')
        plt.ylabel('$R_3$')

        output_fig = os.path.join(args.output_dir,
                                  'app_cluster_{}.png'.format(platform_name.lower()))
        plt.savefig(output_fig, format='png')

if __name__ == '__main__':
    main()
