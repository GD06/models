import tensorflow as tf
import numpy as np

class Operator:

    def __init__(self, op_trace, tf_graph, model_name, shape_dict):

        self.elapsed_time = str(op_trace['dur'])

        op_args = op_trace['args']
        self.op_name = str(op_args['name'])
        self.op_type = str(op_args['op'])
        self.model_name = model_name

        tf_repr = tf_graph.get_operation_by_name(self.op_name)

        self.input_tensor_shape = []
        self.output_tensor_shape = []

        for input_tensor in tf_repr.inputs:
            tensor_name = input_tensor.name
            self.input_tensor_shape.append(shape_dict[tensor_name])

        for output_tensor in tf_repr.outputs:
            tensor_name = output_tensor.name
            self.output_tensor_shape.append(shape_dict[tensor_name])

        self.is_aid_op = self._is_framework_aid_op()

        self.mem_trans = self._calculate_mem_trans(tf_repr)
        self.comp_instrs = self._calculate_comp_instrs(tf_repr)
        self.parallelism = self._calculate_parallelism(tf_repr)

        return

    def _is_framework_aid_op(self):
        aid_op_set = {'VariableV2', 'Identity', 'Squeeze', 'Const',
                      'Reshape', 'StopGradient', 'Shape', 'Barrier',
                      'FIFOQueueV2', 'Assert', 'BarrierTakeMany',
                      'QueueDequeueManyV2', 'Merge', 'BarrierInsertMany',
                      'NoOp', 'ExpandDims', 'RandomUniformInt'}
        if self.op_type in aid_op_set:
            return True
        return False

    def _calculate_mem_trans(self, tf_opr):

        known_op_set = {'Mul', 'Sub', 'Cast', 'ConcatV2', 'MatMul',
                        'BiasAdd', 'Conv2D', 'Sigmoid', 'Tanh'}

        if self.is_aid_op:
            return 0

        if self.op_type in known_op_set:
            total_mem_trans = 0

            for input_tensor, input_tensor_shape in zip(
                    tf_opr.inputs, self.input_tensor_shape):

                tmp_list = [input_tensor.dtype.size] + input_tensor_shape
                total_mem_trans += np.prod(np.array(tmp_list))

            for output_tensor, output_tensor_shape in zip(
                    tf_opr.outputs, self.output_tensor_shape):

                tmp_list = [output_tensor.dtype.size] + output_tensor_shape
                total_mem_trans += np.prod(np.array(tmp_list))

            return total_mem_trans

        print('op_type: ', self.op_type)
        raise NotImplementedError

    def _calculate_comp_instrs(self, tf_opr):

        elementwise_op_set = {'Mul', 'Sub', 'Cast', 'ConcatV2', 'BiasAdd',
                              'Sigmoid', 'Tanh'}

        if self.is_aid_op:
            return 0

        if self.op_type == 'MatMul':
            return self._cal_comp_matmul(tf_opr)

        if self.op_type == 'Conv2D':
            return self._cal_comp_conv2d(tf_opr)

        if self.op_type in elementwise_op_set:
            return self._cal_comp_elementwise(tf_opr)

        print('op_type: ', self.op_type)
        raise NotImplementedError

    def _calculate_parallelism(self, tf_opr):
        return 1.0



    def _cal_comp_elementwise(self, tf_opr):
        tmp_list = self.output_tensor_shape[0]
        return np.prod(np.array(tmp_list))

    def _cal_comp_matmul(self, tf_opr):
        comp_ops = 1
        assert len(self.input_tensor_shape) == 2

        for input_tensor_shape in self.input_tensor_shape:
            comp_ops = comp_ops * np.prod(np.array(input_tensor_shape))

        for each_dim in self.input_tensor_shape[0]:
            if each_dim in set(self.input_tensor_shape[1]):
                comp_ops = comp_ops / each_dim
                break

        return comp_ops

    def _cal_comp_conv2d(self, tf_opr):

        comp_ops = 1

        comp_ops = comp_ops * np.prod(np.array(self.output_tensor_shape[0]))
        comp_ops = comp_ops * np.prod(np.array(self.input_tensor_shape[1]))

        if tf_opr.get_attr('data_format') == 'NHWC':
            oc = self.output_tensor_shape[0][3]
        else:
            oc = self.output_tensor_shape[0][1]

        comp_ops = comp_ops / oc

        return comp_ops
