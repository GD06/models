#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

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

matmul_op = OpPlot('MatMul')
matmul_op.assign_attr({'marker': 'o', 'color': 'r'})
matmul_op.assign_set({'MatMul', 'BatchMatMul'})

conv_op = OpPlot('Conv')
conv_op.assign_attr({'marker': '^', 'color': 'y'})
conv_op.assign_set({'Conv2D', 'Conv2DBackpropInput', 'Conv3DBackpropInputV2',
                    'DepthwiseConv2dNative'})

pooling_op = OpPlot('Pooling')
pooling_op.assign_attr({'marker': 'v', 'color': 'b'})
pooling_op.assign_set({'MaxPool', 'AvgPool'})

reduce_op = OpPlot('Reduce')
reduce_op.assign_attr({'marker': 's', 'color': 'c'})
reduce_op.assign_set({'Sum', 'ArgMin', 'ArgMax', 'Mean', 'All', 'Min', 'Max',
                      'SoftmaxCrossEntropyWithLogits', 'Softmax',
                      'SparseSoftmaxCrossEntropyWithLogits', 'AddN'})

elementwise_op = OpPlot('Element-wise')
elementwise_op.assign_attr({'marker': '+', 'color': 'g'})
elementwise_op.assign_set({'Mul', 'Sub', 'Cast', 'ConcatV2', 'BiasAdd',
                          'Sigmoid', 'Tanh', 'Add', 'GreaterEqual',
                          'LessEqual', 'Switch', 'LogicalNot',
                          'Greater', 'Where', 'Gather', 'Transpose',
                          'Pow', 'Sqrt', 'RealDiv', 'Unpack', 'Split',
                          'Select', 'Relu', 'Equal', 'AssignAdd', 'Sign',
                          'FusedBatchNorm', 'OneHot', 'Less', 'LoopCond',
                          'NextIteration', 'Minimum', 'Maximum', 'Range',
                          'Exp', 'Log', 'ReduceJoin', 'HashTableV2',
                          'LookupTableFindV2', 'StridedSlice', 'Pack',
                          'Pad', 'Neg', 'Sin', 'Cos', 'Floor', 'Fill',
                          'ResizeBilinear', 'DepthToSpace', 'SpaceToDepth',
                          'Round', 'Softplus', 'GatherV2', 'Square', 'Rsqrt',
                          'SquaredDifference', 'RefSwitch', 'Abs', 'Slice',
                          'ScatterUpdate', 'Concat', 'SparseToDense', 'Div',
                          'LogicalAnd', 'Tile', 'Relu6', 'CropAndResize',
                          'FloorMod', 'SpaceToBatchND', 'BatchToSpaceND',
                          'DynamicStitch', 'ReverseSequence', 'Multinomial',
                          'FloorDiv', 'TanhGrad', 'SigmoidGrad', 'Reciprocal',
                          'Lgamma', 'RsqrtGrad'})

others = OpPlot('Others')
others.assign_attr({'marker': '*', 'color': 'm'})
others.assign_set({})

op_classes = [matmul_op, conv_op, pooling_op, reduce_op, elementwise_op, others]

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

    # Reading data from log files
    total_op_list = []
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:
                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    op_list = pickle.load(f)
                total_op_list.extend(op_list)

    np.random.seed(19680801)
    color_dict = {}

    for each_op in total_op_list:
        if not each_op.is_aid_op:
            if each_op.comp_instrs == 0:
                continue

            locality = (each_op.mem_trans / each_op.comp_instrs)
            if locality > 42:
                print('op_type: {}, locality: {}'.format(each_op.op_type,
                                                         locality))
                continue

            found = False
            for each_op_class in op_classes:
                if each_op.op_type in each_op_class.op_set and each_op.regular:
                    found = True
                    each_op_class.x_list.append(each_op.parallelism)
                    each_op_class.y_list.append(locality)
                    break

            if not found:
                final_class = op_classes[-1]
                final_class.x_list.append(each_op.parallelism)
                final_class.y_list.append(locality)


    for each_op_class in op_classes:
        plt.scatter(each_op_class.x_list, each_op_class.y_list, s=each_op_class.area,
                    c=each_op_class.color, marker=each_op_class.marker,
                    label=each_op_class.name, alpha=0.5)

    plt.legend(loc='upper center')

    #plt.xlim(0, 1.0)
    #plt.ylim(0, 26.0)
    plt.xlabel('Parallelism')
    plt.ylabel('Locality')

    output_fig = os.path.join(args.output_dir, 'op_scatter.png')
    plt.savefig(output_fig, format='png')

if __name__ == '__main__':
    main()
