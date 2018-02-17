#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():

    parser = argparse.ArgumentParser(
        description="cluster all operators and save figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "profiling results are saved.")
    parser.add_argument('--suffix', default="pickle", help="specify the suffix of "
                        "result files.")
    parser.add_argument('--output_dir', default=None, help="specify the output dir "
                        "to store output results")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'outputs')

    total_op_list = []
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                total_op_list.extend(op_list)

    np.random.seed(19680801)
    color_dict = {}

    # Draw scatter figures for regular ops
    par_list = []
    locality_list = []
    color_list = []
    for each_op in total_op_list:
        if not each_op.is_aid_op and each_op.regular:
            if each_op.comp_instrs == 0:
                continue
            if each_op.op_type not in color_dict:
                color_dict[each_op.op_type] = np.random.rand()

            locality = (each_op.mem_trans / each_op.comp_instrs)
            if locality > 100:
                print('op_type: {}'.format(each_op.op_type))
                continue

            color_list.append(color_dict[each_op.op_type])
            par_list.append(each_op.parallelism)
            locality_list.append(each_op.mem_trans / each_op.comp_instrs)

    plt.scatter(par_list, locality_list, c=color_list)
    output_fig = os.path.join(args.output_dir, 'op_cluster_reg.pdf')
    plt.savefig(output_fig, format='pdf')

if __name__ == '__main__':
    main()
