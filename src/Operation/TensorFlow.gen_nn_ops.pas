unit TensorFlow.gen_nn_ops;
(*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************)

interface
        uses System.SysUtils,
             Spring,
             TF4D.Core.CApi,
             TensorFlow.DApi,
             Numpy.Axis,

             TensorFlow.Context,
             TensorFlow.Variable,
             TensorFlow.NnOps ;

type
  gen_nn_ops = record
     private

     public
        /// <summary>
        /// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
        ///
        /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        /// and a filter / kernel tensor of shape
        /// `[filter_height, filter_width, in_channels, out_channels]`, this op
        /// performs the following:
        ///
        /// 1. Flattens the filter to a 2-D matrix with shape
        ///    `[filter_height * filter_width * in_channels, output_channels]`.
        /// 2. Extracts image patches from the input tensor to form a *virtual*
        ///    tensor of shape `[batch, out_height, out_width,
        ///    filter_height * filter_width * in_channels]`.
        /// 3. For each patch, right-multiplies the filter matrix and the image patch
        ///    vector.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        class function conv2d(parameters: Conv2dParams) : TFTensor; static;
        /// <summary>
        /// Computes the gradients of convolution with respect to the filter.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        class function conv2d_backprop_filter(input: TFTensor; filter_sizes: TFTensor; out_backprop: TFTensor; strides: TArray<Integer>; padding: string;
                                              use_cudnn_on_gpu : Boolean= true;  explicit_paddings : TArray<Integer>= []; data_format : string = 'NHWC';
                                              dilations: TArray<Integer> = []; name: string = ''): TFTensor; static;
        /// <summary>
        /// Computes the gradients of convolution with respect to the input.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        class function conv2d_backprop_input(input: TFTensor; filter_sizes: TFTensor; out_backprop: TFTensor; strides: TArray<Integer>; padding: string;
                                              use_cudnn_on_gpu : Boolean= true;  explicit_paddings : TArray<Integer>= []; data_format : string = 'NHWC';
                                              dilations: TArray<Integer> = []; name: string = ''): TFTensor; static;
        class function bias_add(value: TFTensor; bias: IVariableV1; data_format : string= ''; name : string= ''): TFTensor; static;
        class function bias_add_grad(out_backprop: TFTensor; data_format: string = 'NHWC'; name: string = ''): TFTensor; static;
        /// <summary>
        /// Computes exponential linear: <c>exp(features) - 1</c> if &amp;lt; 0, <c>features</c> otherwise.
        /// </summary>
        /// <param name="features">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Elu'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        ///    ](http://arxiv.org/abs/1511.07289)
        /// </remarks>
        class function elu(features: TFTensor; name: string = 'Elu'): TFTensor; static;
        /// <summary>
        /// Gradient for batch normalization.
        /// </summary>
        /// <param name="params"></param>
        /// <returns></returns>
        class function fused_batch_norm_grad(params: FusedBatchNormParams): TArray<TFTensor>; static;
        class function fused_batch_norm_grad_v3(params: FusedBatchNormParams): TArray<TFTensor>; static;
        class function fused_batch_norm(x:    TFTensor; scale: TFTensor; offset: TFTensor; mean: TFTensor; variance: TFTensor; epsilon: Single = 0.0001;                                       data_format: string = 'NHWC'; is_training: Boolean = true; name: string = '') : TArray<TFTensor>; static;
        class function fused_batch_norm_v3(x: TFTensor; scale: TFTensor; offset: TFTensor; mean: TFTensor; variance: TFTensor; epsilon: Single = 0.0001; exponential_avg_factor : Single= 1.0; data_format: string = 'NHWC'; is_training: Boolean = true; name: string = '') : TFTensors; static;
         /// <summary>
        /// Local Response Normalization.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="depth_radius"></param>
        /// <param name="bias"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function local_response_normalization(input: TfTensor; depth_radius: Integer = 5; bias: Integer = 1; alpha: Integer = 1; beta: Single = 0.5; name: string = ''): TFTensor; static;
        class function log_softmax(logits: TFTensor; name: string = ''): TFTensor; static;
        /// <summary>
        /// Says whether the targets are in the top `K` predictions.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="targets"></param>
        /// <param name="k"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` of type `bool`.</returns>
        class function in_top_kv2(predictions: TFTensor; targets: TFTensor; k: Integer; name: string = ''): TFTensor; static;
        class function leaky_relu(features: TFTensor; alpha: single = 0.2; name: string = ''): TFTensor; static;
        class function average_pool(input: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor; static;
        class function max_pool(input: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor; static;
        class function max_pool_grad(orig_input: TFTensor; orig_output: TFTensor; grad: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format : string = 'NHWC'; name: string = '') : TFTensor; static;
        class function top_kv2<T>(input: TFTensor; k: T; sorted: Boolean = true; name: string = ''): TArray<TFTensor>; static;
        class function relu_grad(gradients: TFTensor; features: TFTensor; name: string = ''): TFTensor; static;
        class function leaky_relu_grad(gradients: TFTensor; features: TFTensor; alpha: Single = 0.2; name: string = ''): TFTensor; static;
        class function softmax(logits: TFTensor; name: string = '') : TFTensor; static;
        /// <summary>
        /// Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="labels"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function softmax_cross_entropy_with_logits(features: TFTensor; labels: TFTensor; name: string = ''): Tuple<TFTensor, TFTensor>; static;
        /// <summary>
        ///    Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features">
        ///    batch_size x num_classes matrix
        /// </param>
        /// <param name="labels">
        ///    batch_size vector with values in [0, num_classes).
        ///    This is the label for the given minibatch entry.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SparseSoftmaxCrossEntropyWithLogits'.
        /// </param>
        /// <returns>
        ///    Returns a tuple with multiple values, as follows:
        ///    loss : Per example loss (batch_size vector).
        ///    backprop : backpropagated gradients (batch_size x num_classes matrix).
        ///    The Operation can be fetched from any of the Tensorreturned in the tuple values, by fetching the Operation property.
        /// </returns>
        /// <remarks>
        ///    Unlike <c>SoftmaxCrossEntropyWithLogits</c>, this operation does not accept
        ///    a matrix of label probabilities, but rather a single label per row
        ///    of features.  This label is considered to have probability 1.0 for the
        ///    given row.
        ///
        ///    Inputs are the logits, not probabilities.
        /// </remarks>
        class function sparse_softmax_cross_entropy_with_logits(features: TFTensor; labels: TFTensor; name: string = 'SparseSoftmaxCrossEntropyWithLogits'): Tuple<TFTensor, TFTensor>; static;
        /// <summary>
        /// Computes rectified linear: `max(features, 0)`.
        /// </summary>
        /// <param name="features">A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `features`.</returns>
        class function relu(features: TFTensor; name: string = ''): TFTensor; static;
        class function tanh(x: TFTensor; name: string = ''): TFTensor; static;
  end;

implementation
      uses Tensorflow,
           TensorFlow.Ops,
           Tensorflow.NameScope,
           Tensorflow.Utils;

{ gen_nn_ops }

class function gen_nn_ops.conv2d(parameters: Conv2dParams): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Conv2D', parameters.name, ExecuteOpArgs.Create([parameters.input, parameters.filter])
                                          .SetAttributes(['strides',          parameters.Strides,
                                                          'padding',          parameters.Padding,
                                                          'use_cudnn_on_gpu', parameters.UseCudnnOnGpu,
                                                          'explicit_paddings',parameters.ExplicitPaddings,
                                                          'data_format',      parameters.DataFormat,
                                                          'dilations',        parameters.Dilations]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.conv2d_backprop_filter(input, filter_sizes, out_backprop: TFTensor; strides: TArray<Integer>; padding: string; use_cudnn_on_gpu: Boolean;
  explicit_paddings: TArray<Integer>; data_format: string; dilations: TArray<Integer>; name: string): TFTensor;
begin
    if Length(dilations) = 0 then
      dilations := [ 1, 1, 1, 1 ];

    Result := tf.Context.ExecuteOp('Conv2DBackpropFilter', name, ExecuteOpArgs.Create([input, filter_sizes, out_backprop])
                                          .SetAttributes(['strides',          Strides,
                                                          'padding',          Padding,
                                                          'use_cudnn_on_gpu', use_cudnn_on_gpu,
                                                          'explicit_paddings',explicit_paddings,
                                                          'data_format',      data_format,
                                                          'dilations',        dilations]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.conv2d_backprop_input(input, filter_sizes, out_backprop: TFTensor; strides: TArray<Integer>; padding: string; use_cudnn_on_gpu: Boolean;
  explicit_paddings: TArray<Integer>; data_format: string; dilations: TArray<Integer>; name: string): TFTensor;
begin
    if Length(dilations) = 0 then
      dilations := [ 1, 1, 1, 1 ];

    Result := tf.Context.ExecuteOp('Conv2DBackpropInput', name, ExecuteOpArgs.Create([input, filter_sizes, out_backprop])
                                          .SetAttributes(['strides',          Strides,
                                                          'padding',          Padding,
                                                          'use_cudnn_on_gpu', use_cudnn_on_gpu,
                                                          'explicit_paddings',explicit_paddings,
                                                          'data_format',      data_format,
                                                          'dilations',        dilations]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.elu(features: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Elu', name,[ GetArg('features',features) ] );
    Result := _op.output;
end;

class function gen_nn_ops.fused_batch_norm(x, scale, offset, mean, variance: TFTensor; epsilon: Single; data_format: string; is_training: Boolean;
  name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('FusedBatchNorm', name,[ GetArg('x',x),
                                                                     GetArg('scale',scale),
                                                                     GetArg('offset',offset),
                                                                     GetArg('mean',mean),
                                                                     GetArg('variance',variance),
                                                                     GetArg('epsilon',epsilon),
                                                                     GetArg('data_format',data_format),
                                                                     GetArg('is_training',is_training) ] );
    Result := _op.outputs;
end;

class function gen_nn_ops.fused_batch_norm_grad(params: FusedBatchNormParams): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('FusedBatchNormGrad', params.name,[ GetArg('y_backprop',params.YBackprop),
                                                                     GetArg('x',params.X),
                                                                     GetArg('scale',params.Scale),
                                                                     GetArg('reserve_space_1',params.ReserveSpace1),
                                                                     GetArg('reserve_space_2',params.ReserveSpace2),
                                                                     GetArg('epsilon',params.Epsilon),
                                                                     GetArg('data_format',params.DataFormat),
                                                                     GetArg('is_training',params.IsTraining) ] );
    Result := _op.outputs;
end;

class function gen_nn_ops.fused_batch_norm_grad_v3(params: FusedBatchNormParams): TArray<TFTensor>;
begin
  Result := tf.Context.ExecuteOp('FusedBatchNormGradV3', params.name, ExecuteOpArgs.Create([params.YBackprop,
                                                                                            params.X,
                                                                                            params.Scale,
                                                                                            params.ReserveSpace1,
                                                                                            params.ReserveSpace2,
                                                                                            params.ReserveSpace3])
                                                                        .SetAttributes(['epsilon',     params.Epsilon,
                                                                                        'data_format', params.DataFormat,
                                                                                        'is_training', params.IsTraining]) ).ToArray;

end;

class function gen_nn_ops.fused_batch_norm_v3(x, scale, offset, mean, variance: TFTensor; epsilon, exponential_avg_factor: Single; data_format: string; is_training: Boolean;
  name: string): TFTensors;
begin
  Result := tf.Context.ExecuteOp('FusedBatchNormV3', name, ExecuteOpArgs.Create([x, scale, offset, mean, variance])
                                                                        .SetAttributes(['epsilon',     epsilon,
                                                                                        'data_format', data_format,
                                                                                        'exponential_avg_factor', exponential_avg_factor,
                                                                                        'is_training', is_training]) );
end;

class function gen_nn_ops.in_top_kv2(predictions, targets: TFTensor; k: Integer; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('InTopKV2', name,[ GetArg('predictions',predictions),
                                                               GetArg('targets',targets),
                                                               GetArg('k',k) ] );
    Result := _op.output;
end;

class function gen_nn_ops.leaky_relu(features: TFTensor; alpha: single; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LeakyRelu', name, ExecuteOpArgs.Create([features])
                                                                        .SetAttributes(['alpha', alpha]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.leaky_relu_grad(gradients, features: TFTensor; alpha: Single; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LeakyReluGrad', name, ExecuteOpArgs.Create([gradients,features])
                                                                        .SetAttributes(['alpha', alpha]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.local_response_normalization(input: TfTensor; depth_radius, bias, alpha: Integer; beta: Single; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('LRN', name,[ GetArg('input',input),
                                                               GetArg('depth_radius',depth_radius),
                                                               GetArg('bias',bias),
                                                               GetArg('alpha',alpha),
                                                               GetArg('beta',beta)] );
    Result := _op.output;
end;

class function gen_nn_ops.log_softmax(logits: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'LogSoftmax', name, ExecuteOpArgs.Create([logits]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.max_pool(input: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MaxPool', name, ExecuteOpArgs.Create([input])
                                                                        .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.max_pool_grad(orig_input, orig_output, grad: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MaxPoolGrad', name, ExecuteOpArgs.Create([orig_input, orig_output, grad])
                                                                        .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.relu(features: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Relu', name, ExecuteOpArgs.Create([features]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.relu_grad(gradients, features: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'ReluGrad', name, ExecuteOpArgs.Create([gradients, features]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.softmax(logits: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Softmax', name, ExecuteOpArgs.Create([logits]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.softmax_cross_entropy_with_logits(features, labels: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin
    var Res := tf.Context.ExecuteOp( 'SoftmaxCrossEntropyWithLogits', name, ExecuteOpArgs.Create([features, labels]) ).FirstOrDefault(nil);

    Result := Tuple<TFTensor, TFTensor>.Create(Res[0],Res[1]);
end;

class function gen_nn_ops.sparse_softmax_cross_entropy_with_logits(features, labels: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin
    var Res := tf.Context.ExecuteOp( 'SparseSoftmaxCrossEntropyWithLogits', name, ExecuteOpArgs.Create([features, labels]) ).FirstOrDefault(nil);

    Result := Tuple<TFTensor, TFTensor>.Create(Res[0],Res[1]);
end;

class function gen_nn_ops.tanh(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Tanh', name, ExecuteOpArgs.Create([x]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.top_kv2<T>(input: TFTensor; k: T; sorted: Boolean; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('TopKV2', name,[ GetArg('input',input), GetArg('k', TValue.From<T>(k)), GetArg('sorted',sorted) ] );
    Result := _op.outputs;
end;

class function gen_nn_ops.average_pool(input: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('AvgPool', name, ExecuteOpArgs.Create([input])
                                                                        .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.bias_add(value: TFTensor; bias: IVariableV1; data_format, name: string): TFTensor;
begin
    if data_format = '' then
      data_format := 'NHWC';

    Result := tf.Context.ExecuteOp('BiasAdd', name, ExecuteOpArgs.Create([value, TValue.From<IVariableV1>(bias)])
                                                                        .SetAttributes(['data_format', data_format ]) ).FirstOrDefault(nil);
end;

class function gen_nn_ops.bias_add_grad(out_backprop: TFTensor; data_format, name: string): TFTensor;
begin
    if data_format = '' then
      data_format := 'NHWC';

    Result := tf.Context.ExecuteOp('BiasAddGrad', name, ExecuteOpArgs.Create([out_backprop])
                                                                        .SetAttributes(['data_format', data_format ]) ).FirstOrDefault(nil);
end;

end.
