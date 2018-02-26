#!/usr/bin/env python3
#--- coding: utf-8 ---

import argparse
import pickle
import os

def extract_app_topo(op_list):
    oprand_dict = {}
    for each_op in op_list:
        for oprand_name in each_op.input_tensor_name:
            oprand_dict[oprand_name] = {}
        for oprand_name in each_op.output_tensor_name:
            oprand_dict[oprand_name] = {}

    for each_op in op_list:
        for oprand_name in each_op.input_tensor_name:
            oprand_dict[oprand_name][each_op.op_name] = 1

    total_opr_num = 0
    par_opr_num = 0
    has_loop = 0
    for each_op in op_list:
        if each_op.is_aid_op:
            continue

        total_opr_num += 1
        par_opr_flag = False
        for oprand_name in each_op.input_tensor_name:
            if len(oprand_dict[oprand_name]) >= 2:
                par_opr_flag = True

        if par_opr_flag:
            par_opr_num += 1

        if each_op.op_type in {'Enter', 'Exit', 'LoopCond'}:
            has_loop = 1

    return [has_loop, par_opr_num, total_opr_num]

def main():

    parser = argparse.ArgumentParser(
        description="analyze application graph topology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="specify the directory under which "
                        "profiling results are saved")
    parser.add_argument('--suffix', default=".pickle", help="specify the suffix of "
                        "result files")
    parser.add_argument('--output_dir', default=None,
                        help="specify the directory where output results are saved")
    parser.add_argument('--output_file', default="app_topo.pickle",
                        help="specify the filename of application topology results")

    args = parser.parse_args()

    print('Loading and Extracting Application Features...')
    total_opr_num = 0
    app_dict = {}
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                index = input_file.find(args.suffix)
                app_name = input_file[:index]
                app_topo_f = extract_app_topo(op_list)
                total_opr_num += app_topo_f[2]
                print('App: {}, Has Loop: {}, #par_nodes: {}, #total_nodes: {}'.format(
                    app_name, app_topo_f[0], app_topo_f[1], app_topo_f[2]))
                app_dict[app_name] = app_topo_f

    print('Saving Application Topology Features...')
    with open(os.path.join(args.output_dir, args.output_file), 'wb') as f:
        pickle.dump(app_dict, f)
    print('Total number of operators: {}'.format(total_opr_num))

if __name__ == '__main__':
    main()
