import tensorflow as tf
import numpy as np
import math
import re

class Operator:

    def __init__(self, op_trace, tf_graph, model_name):

        self.elapsed_time = str(op_trace['dur'])

        op_args = op_trace['args']
        self.op_name = str(op_args['name'])
        self.op_type = str(op_args['op'])
        self.model_name = model_name

        self.is_aid_op = self._is_framework_aid_op()

        return

    def analysis(self, shape_dict, tf_graph):

        tf_repr = tf_graph.get_operation_by_name(self.op_name)

        self.input_tensor_name = []
        self.output_tensor_name = []

        self.input_tensor_shape = []
        self.output_tensor_shape = []

        for input_tensor in tf_repr.inputs:

            tensor_name = input_tensor.name
            self.input_tensor_name.append(tensor_name)
            if self.is_aid_op:
                continue

            t_shape = shape_dict[tensor_name]
            if isinstance(t_shape, tf.TensorShape):
                self.input_tensor_shape.append(self._tolist(t_shape))
            else:
                self.input_tensor_shape.append(t_shape)

        for output_tensor in tf_repr.outputs:

            tensor_name = output_tensor.name
            self.output_tensor_name.append(tensor_name)
            if self.is_aid_op:
                continue

            t_shape = shape_dict[tensor_name]
            if isinstance(t_shape, tf.TensorShape):
                self.output_tensor_shape.append(self._tolist(t_shape))
            else:
                self.output_tensor_shape.append(t_shape)

        self.mem_trans = self._calculate_mem_trans(tf_repr)
        self.comp_instrs = self._calculate_comp_instrs(tf_repr)
        self.parallelism = self._calculate_parallelism(tf_repr)

        return

    def _tolist(self, t_shape):
        rtval = []
        for i in t_shape.as_list():
            if i is None:
                rtval.append(int(1))
            else:
                rtval.append(i)
        return rtval

    def _is_framework_aid_op(self):
        aid_op_set = {'VariableV2', 'Identity', 'Squeeze', 'Const',
                      'Reshape', 'StopGradient', 'Shape', 'Barrier',
                      'FIFOQueueV2', 'Assert', 'BarrierTakeMany',
                      'QueueDequeueManyV2', 'Merge', 'BarrierInsertMany',
                      'NoOp', 'ExpandDims', 'RandomUniformInt',
                      'RandomStandardNormal', 'ShapeN', 'Enter', 'Exit',
                      'Size', 'NonMaxSuppressionV2'}

        tensor_array_matcher = re.compile('TensorArray')
        if tensor_array_matcher.match(self.op_type) is not None:
            return True

        if self.op_type in aid_op_set:
            return True

        return False

    def _calculate_mem_trans(self, tf_opr):

        known_op_set = {'Mul', 'Sub', 'Cast', 'ConcatV2', 'MatMul',
                        'BiasAdd', 'Conv2D', 'Sigmoid', 'Tanh', 'Add',
                        'Min', 'GreaterEqual', 'Max', 'LessEqual',
                        'LogicalNot', 'Greater', 'Gather', 'Sum', 'Transpose',
                        'Pow', 'Sqrt', 'RealDiv', 'Unpack', 'Split',
                        'Relu', 'Equal', 'AssignAdd', 'Sign', 'FusedBatchNorm',
                        'MaxPool', 'AvgPool', 'ArgMin', 'OneHot', 'Less',
                        'LoopCond', 'NextIteration', 'Minimum', 'Maximum',
                        'Range', 'ArgMax', 'Exp', 'Log', 'ReduceJoin',
                        'Pack', 'Pad', 'Neg', 'Sin', 'Cos', 'Floor', 'AddN',
                        'Fill', 'ResizeBilinear', 'Conv2DBackpropInput',
                        'DepthToSpace', 'SpaceToDepth', 'Mean', 'Round',
                        'Softplus', 'GatherV2', 'Square', 'Rsqrt'}

        softmax_op_set = {'SoftmaxCrossEntropyWithLogits',
                          'Softmax'}

        if self.is_aid_op:
            return 0

        if (self.op_type == 'Switch' or self.op_type == 'Select'):
            return self._cal_mem_switch()
        if self.op_type == 'Where':
            return self._cal_mem_where()

        if self.op_type == 'FusedBatchNormGrad':
            return self._cal_mem_fusedbatchnormgrad(tf_opr)

        if (self.op_type == 'HashTableV2' or
                self.op_type == 'LookupTableFindV2' or
                self.op_type == 'StridedSlice'):
            return self._cal_mem_tableindex(tf_opr)

        if self.op_type in softmax_op_set:
            return self._cal_mem_softmax()

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
                              'Sigmoid', 'Tanh', 'Add', 'Min', 'GreaterEqual',
                              'Max', 'LessEqual', 'Switch', 'LogicalNot',
                              'Greater', 'Where', 'Gather', 'Transpose',
                              'Pow', 'Sqrt', 'RealDiv', 'Unpack', 'Split',
                              'Select', 'Relu', 'Equal', 'AssignAdd', 'Sign',
                              'OneHot', 'Less', 'LoopCond', 'NextIteration',
                              'Minimum', 'Maximum', 'Range', 'Exp', 'Log',
                              'HashTableV2', 'LookupTableFindV2', 'StridedSlice',
                              'Pack', 'Pad', 'Neg', 'Sin', 'Cos', 'Floor', 'Fill',
                              'ResizeBilinear', 'DepthToSpace', 'SpaceToDepth',
                              'Round', 'Softplus', 'GatherV2', 'Square', 'Rsqrt'}

        reduce_op_set = {'Sum', 'ArgMin', 'ArgMax', 'ReduceJoin', 'Mean'}

        pooling_op_set = {'MaxPool', 'AvgPool'}

        softmax_op_set = {'SoftmaxCrossEntropyWithLogits',
                          'Softmax'}

        if self.is_aid_op:
            return 0

        if self.op_type == 'MatMul':
            return self._cal_comp_matmul(tf_opr)

        if (self.op_type == 'Conv2D'
                or self.op_type == 'Conv2DBackpropInput'):
            return self._cal_comp_conv2d(tf_opr)

        if self.op_type == 'FusedBatchNorm':
            return self._cal_comp_fusedbatchnorm(tf_opr)

        if self.op_type == 'AddN':
            return self._cal_comp_addn(tf_opr)

        if self.op_type in softmax_op_set:
            return self._cal_comp_softmax(tf_opr)

        if self.op_type in pooling_op_set:
            return self._cal_comp_pooling(tf_opr)

        if self.op_type in reduce_op_set:
            return self._cal_comp_reduce(tf_opr)

        if self.op_type in elementwise_op_set:
            return self._cal_comp_elementwise(tf_opr)

        print('op_type: ', self.op_type)
        raise NotImplementedError

    def _calculate_parallelism(self, tf_opr):

        elementwise_op_set = {'Mul', 'Sub', 'Cast', 'ConcatV2', 'BiasAdd',
                              'Sigmoid', 'Tanh', 'Add', 'Min', 'GreaterEqual',
                              'Max', 'LessEqual', 'Switch', 'LogicalNot',
                              'Greater', 'Where', 'Gather', 'Transpose',
                              'Pow', 'Sqrt', 'RealDiv', 'Unpack', 'Split',
                              'Select', 'Relu', 'Equal', 'AssignAdd', 'Sign',
                              'FusedBatchNorm', 'OneHot', 'Less', 'LoopCond',
                              'NextIteration', 'Minimum', 'Maximum', 'Range',
                              'Exp', 'Log', 'ReduceJoin', 'HashTableV2',
                              'LookupTableFindV2', 'StridedSlice', 'Pack',
                              'Pad', 'Neg', 'Sin', 'Cos', 'Floor', 'Fill',
                              'ResizeBilinear', 'DepthToSpace', 'SpaceToDepth',
                              'Round', 'Softplus', 'GatherV2', 'Square', 'Rsqrt'}

        reduce_op_set = {'Sum', 'ArgMin', 'ArgMax', 'Mean'}

        pooling_op_set = {'MaxPool', 'AvgPool'}

        softmax_op_set = {'SoftmaxCrossEntropyWithLogits',
                          'Softmax'}

        if self.is_aid_op:
            return 0.0

        if self.op_type == 'MatMul':
            return self._cal_par_matmul(tf_opr)

        if (self.op_type == 'Conv2D'
                or self.op_type == 'Conv2DBackpropInput'):
            return self._cal_par_conv2d(tf_opr)

        if self.op_type == 'AddN':
            return self._cal_par_addn(tf_opr)

        if self.op_type in softmax_op_set:
            return self._cal_par_softmax(tf_opr)

        if self.op_type in reduce_op_set:
            return self._cal_par_reduce(tf_opr)

        if self.op_type in pooling_op_set:
            return self._cal_par_pooling(tf_opr)

        if self.op_type in elementwise_op_set:
            return 1.0

        print('op_type: ', self.op_type)
        raise NotImplementedError



    def _cal_comp_elementwise(self, tf_opr):
        tmp_list = self.output_tensor_shape[0]
        comp_ops = 2 * np.prod(np.array(tmp_list))
        return comp_ops

    def _extract_m_n_k(self):
        k_sqr = 1
        assert len(self.input_tensor_shape) == 2
        assert len(self.output_tensor_shape) == 1
        assert len(self.output_tensor_shape[0]) == 2

        for input_tensor_shape in self.input_tensor_shape:
            k_sqr = k_sqr * np.prod(np.array(input_tensor_shape))

        for output_tensor_shape  in self.output_tensor_shape:
            k_sqr = k_sqr / np.prod(np.array(output_tensor_shape))
            m = output_tensor_shape[0]
            n = output_tensor_shape[1]

        return m, n, math.sqrt(k_sqr)

    def _cal_comp_matmul(self, tf_opr):
        m, n, k = self._extract_m_n_k()
        comp_ops = 2 * m * n * k
        return comp_ops

    def _extract_conv2d_params(self, tf_opr):
        conv_args = {}
        #assert len(self.input_tensor_shape) == 2
        #assert len(self.output_tensor_shape) == 1
        #assert len(self.input_tensor_shape[0]) == 4
        assert len(self.input_tensor_shape[1]) == 4
        assert len(self.output_tensor_shape[0]) == 4

        if (tf_opr.get_attr('data_format') == b'NHWC'
                or tf_opr.get_attr('data_format')  == 'NHWC'):
            conv_args['ON'] = self.output_tensor_shape[0][0]
            conv_args['OH'] = self.output_tensor_shape[0][1]
            conv_args['OW'] = self.output_tensor_shape[0][2]
            conv_args['OC'] = self.output_tensor_shape[0][3]
        else:
            conv_args['ON'] = self.output_tensor_shape[0][0]
            conv_args['OC'] = self.output_tensor_shape[0][1]
            conv_args['OH'] = self.output_tensor_shape[0][2]
            conv_args['OW'] = self.output_tensor_shape[0][3]

        conv_args['FH'] = self.input_tensor_shape[1][0]
        conv_args['FW'] = self.input_tensor_shape[1][1]
        #conv_args['IC'] = self.input_tensor_shape[1][2]
        #assert conv_args['OC'] == self.input_tensor_shape[1][3]
        if conv_args['OC'] == self.input_tensor_shape[1][3]:
            conv_args['IC'] = self.input_tensor_shape[1][2]
        elif conv_args['OC'] == self.input_tensor_shape[1][2]:
            conv_args['IC'] = self.input_tensor_shape[1][3]
        else:
            raise NotImplementedError

        conv_args['IN'] = conv_args['ON']

        return conv_args

    def _cal_comp_conv2d(self, tf_opr):
        comp_ops = 1

        conv_args = self._extract_conv2d_params(tf_opr)
        comp_ops = 2 * comp_ops * np.prod(np.array(self.output_tensor_shape[0]))
        comp_ops = comp_ops * conv_args['IC'] * conv_args['FH'] * conv_args['FW']

        return comp_ops

    def _cal_comp_reduce(self, tf_opr):
        tmp_list = self.input_tensor_shape[0]
        comp_ops = np.prod(np.array(tmp_list))
        return comp_ops

    def _cal_comp_fusedbatchnorm(self, tf_opr):
        tmp_list = self.output_tensor_shape[0]
        comp_ops =  5 * np.prod(np.array(tmp_list))
        return comp_ops

    def _cal_comp_pooling(self, tf_opr):
        ksize = tf_opr.get_attr('ksize')
        tmp_list = self.output_tensor_shape[0]
        comp_ops = np.prod(np.array(tmp_list))
        comp_ops = comp_ops * np.prod(np.array(ksize))
        return comp_ops

    def _cal_comp_softmax(self, tf_opr):
        tmp_list = self.input_tensor_shape[0]
        return 3 * np.prod(np.array(tmp_list))

    def _cal_comp_addn(self, tf_opr):
        tmp_list = self.input_tensor_shape[0]
        n = max(1, len(self.input_tensor_shape) - 1)
        return n * np.prod(np.array(tmp_list))

    def _cal_par_matmul(self, tf_opr):
        M, N, K = self._extract_m_n_k()
        K = max(K, 2.0)
        par_ratio = 0.5 + (1.0 / (2 * math.ceil(math.log2(K))))
        return par_ratio

    def _cal_par_conv2d(self, tf_opr):
        conv_args = self._extract_conv2d_params(tf_opr)
        K = conv_args['IC'] * conv_args['FH'] * conv_args['FW']

        K = max(K, 2.0)
        par_ratio = 0.5 + (1.0 / (2 * math.ceil(math.log2(K))))
        return par_ratio

    def _cal_par_reduce(self, tf_opr):
        prod_input = np.prod(np.array(self.input_tensor_shape[0]))
        prod_output = np.prod(np.array(self.output_tensor_shape[0]))
        K = prod_input / prod_output
        assert K >= 1.0

        K = max(K, 2.0)
        par_ratio = (1.0 / math.ceil(math.log2(K)))
        return par_ratio

    def _cal_par_pooling(self, tf_opr):
        ksize = tf_opr.get_attr('ksize')
        K = max(np.prod(np.array(ksize)), 2.0)
        par_ratio = (1.0 / math.ceil(math.log2(K)))
        return par_ratio

    def _cal_par_softmax(self, tf_opr):
        K = max(2.0, self.input_tensor_shape[0][1])
        par_ratio = 2.0 / 3.0 + (1.0 / (3.0 * math.ceil(math.log2(K))))
        return par_ratio

    def _cal_par_addn(self, tf_opr):
        K = max(2.0, len(self.input_tensor_shape))
        par_ratio = (1.0 / math.ceil(math.log2(K)))
        return par_ratio

    def _cal_mem_switch(self):
        tmp_list = self.input_tensor_shape[0]
        return 2 * np.prod(np.array(tmp_list))

    def _cal_mem_where(self):
        tmp_list = self.output_tensor_shape[0]
        return 3 * np.prod(np.array(tmp_list))

    def _cal_mem_softmax(self):
        tmp_list = self.input_tensor_shape[0]
        return 2 * np.prod(np.array(tmp_list))

    def _cal_mem_tableindex(self, tf_opr):
        tmp_list = self.output_tensor_shape[0]
        return 2 * np.prod(np.array(tmp_list))

