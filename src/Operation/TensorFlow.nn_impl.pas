unit TensorFlow.nn_impl;
{$REGION 'Licence'}
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
{$ENDREGION}

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
        uses System.SysUtils,

             Spring,

             TF4D.Core.CApi,
             TensorFlow.DApi,
             Numpy.Axis,

             TensorFlow.Context,
             TensorFlow.Variable;

type
  nn_impl = record
    private
      /// <summary>
      /// Same as math_ops.count_nonzero.
      /// The reduction is done in dtype, which can be faster for 32-bit dtypes.
      /// </summary>
      /// <param name="input_tensor">The numeric tensor.</param>
      /// <param name="dtype">The reduction dtype.</param>
      /// <returns>number of nonzero values with type dtype</returns>
      class function _count_nonzero(input_tensor: TFTensor; dtype : TF_DataType = TF_INT64): TFTensor; static;
    public
      class function conv2d_transpose(value        : TFTensor = nil;
                                      filter       : IVariableV1 = nil;
                                      output_shape : TFTensor = nil;
                                      strides      : PTFShape = nil;
                                      padding      : string = 'SAME';
                                      data_format  : string = 'NHWC';
                                      name         : string = '';
                                      dilations    : PTFShape= nil): TFTensor; static;
      /// <summary>
      /// Normalizes along dimension `axis` using an L2 norm.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="axis"></param>
      /// <param name="epsilon"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function l2_normalize(x: TFTensor; axis : Integer= 0; epsilon: TFTensor = nil; name: string = ''): TFTensor; static;
      /// <summary>
      /// Calculate the mean and variance of `x`
      /// </summary>
      /// <param name="x"> A `Tensor`.</param>
      /// <param name="axes"> Array of ints.  Axes along which to compute mean and variance.</param>
      /// <param name="name"> Name used to scope the operations that compute the moments.</param>
      /// <param name="keep_dims"> Produce moments with the same dimensionality as the input.</param>
      /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>
      class function moments(x: TFTensor; axes: TAxis; name: string = ''; keep_dims: Boolean = false) : Tuple<TFTensor, TFTensor>;static;
      class function normalize(tensor: TFTensor; _ord : string = 'euclidean'; axis: PAxis = nil; name: string = ''): TFTensor; static;
      class function batch_normalization(x, mean, variance, offset, scale: TFTensor; variance_epsilon: Single = 0.001; name: string = ''): TFTensor; static;
      /// <summary>
      /// Batch normalization.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="scale"></param>
      /// <param name="offset"></param>
      /// <param name="mean"></param>
      /// <param name="variance"></param>
      /// <param name="epsilon"></param>
      /// <param name="data_format"></param>
      /// <param name="is_training"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function fused_batch_norm(x          : TFTensor;
                                      scale      : TFTensor;
                                      offset     : TFTensor;
                                      mean       : TFTensor= nil;
                                      variance   : TFTensor= nil;
                                      epsilon    : Single = 0.001;
                                      data_format: string = 'NHWC';
                                      is_training: Boolean = true;
                                      name       : string = '';
                                      exponential_avg_factor : Single = 1.0): TArray<TFTensor>;  static;
      class function sigmoid_cross_entropy_with_logits(labels: TFTensor; logits: TFTensor; name: string = '') : TFTensor; static;
      /// <summary>
      /// Returns the fraction of zeros in value.
      /// </summary>
      /// <param name="value">A tensor of numeric type.</param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>The fraction of zeros in value, with type float32.</returns>
      class function zero_fraction(value: TFTensor; name: string = '') : TFTensor; static;
  end;

implementation
    uses TensorFlow,
         TensorFlow.Tensor,
         Tensorflow.Utils,
         TensorFlow.Ops,
         TensorFlow.Constant_op,
         Tensorflow.NameScope,
         Tensorflow.array_ops,
         TensorFlow.gen_nn_ops,
         TensorFlow.gen_math_ops,
         TensorFlow.control_flow_ops,
         Tensorflow.math_ops;

{ nn_impl }

class function nn_impl.conv2d_transpose(value: TFTensor; filter: IVariableV1; output_shape: TFTensor; strides: PTFShape; padding, data_format, name: string;
  dilations: PTFShape): TFTensor;
begin
    if dilations = nil then
    begin
        var _dilations := TFShape.Create([1, 1, 1, 1]);
        dilations := @_dilations;
    end;
    var vValues : TArray<TValue> := [value, TValue.From<IVariableV1>(filter), output_shape];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'conv2d_transpose', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    Result := gen_nn_ops.conv2d_backprop_input(output_shape, filter.AsTensor, value, strides^, padding, True, nil, data_format, dilations^, name);
                end);
end;

class function nn_impl.l2_normalize(x: TFTensor; axis: Integer; epsilon: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'l2_normalize', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    x      := Tops.convert_to_tensor(x, DtInvalid, 'x');
                    var sq := math_ops.square(x);
                    var square_sum := math_ops.reduce_sum(sq, constant_op.constant(axis), true);
                    var e : TFTensor;
                    if    epsilon = nil  then  e := tf.Variable(Single(1e-12)).ToTensor
                    else                       e := epsilon;

                    var x_inv_norm := math_ops.rsqrt(math_ops.maximum(square_sum, e));
                    Result := math_ops.multiply(x, x_inv_norm, name);
                end);
end;

class function nn_impl.moments(x: TFTensor; axes: TAxis; name: string; keep_dims: Boolean): Tuple<TFTensor, TFTensor>;
begin
    var vValues : TArray<TValue> := [x, axes];
    Result := TUtils.tf_with<TNameScope,Tuple<TFTensor, TFTensor>>( TOps.name_scope(name, 'moments', @vValues),
                function(v1: TNameScope): Tuple<TFTensor, TFTensor>
                begin
                    // The dynamic range of fp16 is too limited to support the collection of
                    // sufficient statistics. As a workaround we simply perform the operations
                    // on 32-bit floats before converting the mean and variance back to fp16
                    var y := math_ops.cast(x, TF_DataType.TF_FLOAT);
                    // Compute true mean while keeping the dims for proper broadcasting.
                    var mean := math_ops.reduce_mean(y, axes, true, 'mean');
                    // Sample variance, not unbiased variance
                    // Note: stop_gradient does not change the gradient that gets
                    // backpropagated to the mean from the variance calculation,
                    // because that gradient is zero
                    var variance := math_ops.reduce_mean(math_ops.square_difference(y, array_ops.stop_gradient(mean) ), axes, true, 'Variance');
                    if not keep_dims then
                    begin
                        mean     := array_ops.squeeze(mean, axes);
                        variance := array_ops.squeeze(variance, axes);
                    end;
                    // TODO: if x.dtype == dtypes.float16:
                    if x.dtype = TF_DataType.TF_HALF then
                    begin
                        Result := Tuple.Create( math_ops.cast(mean, x.dtype), math_ops.cast(variance, x.dtype) );
                    end else
                    begin
                         Result := Tuple.Create(mean, variance);
                    end;
                end);
end;

class function nn_impl.normalize(tensor: TFTensor; _ord: string; axis: PAxis; name: string): TFTensor;
begin
    var vValues : TValue := tensor;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'normalize', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    var norm       := tf.linalg.norm(tensor, _ord, axis, name);
                    var normalized := TTensor(tensor) / norm;
                    Result := normalized;
                end);
end;

class function nn_impl.batch_normalization(x, mean, variance, offset, scale: TFTensor; variance_epsilon: Single; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x, mean, variance, scale, offset];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'batchnorm', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    var inv := math_ops.rsqrt(TTensor(variance) + variance_epsilon);
                    inv := TTEnsor(inv) * scale;
                    var y : TFTensor;
                    if offset = nil then   y := (-TTensor(mean) * inv)
                    else                   y := (TTensor(offset) - mean * TTensor(inv));

                    Result :=  TTensor(x) * math_ops.cast(inv, x.dtype) + math_ops.cast(y, x.dtype);
                end);
end;

class function nn_impl.fused_batch_norm(x, scale, offset, mean, variance: TFTensor; epsilon: Single; data_format: string; is_training: Boolean; name: string;
  exponential_avg_factor: Single): TArray<TFTensor>;
begin
    var a : TArray<Single> := [];
    if mean = nil then mean := constant_op.constant(a);
    if variance = nil then variance := constant_op.constant(a);

    var min_epsilon := 1.001e-5;
    if epsilon > min_epsilon then  epsilon := epsilon
    else                           epsilon := min_epsilon;

    var res := gen_nn_ops.fused_batch_norm_v3(x, scale, offset, mean, variance,  epsilon, exponential_avg_factor, data_format, is_training, name);

    var y            := res[0];
    var running_mean := res[1];
    var running_var  := res[2];

    Result := [ y, running_mean, running_var ];
end;

class function nn_impl._count_nonzero(input_tensor: TFTensor; dtype: TF_DataType): TFTensor;
begin
    var vValues : TArray<TValue> := [input_tensor];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('count_nonzero', 'count_nonzero', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    var zero          := array_ops.zeros(TFShape.Null, input_tensor.dtype);
                    var nonzero_count := math_ops.reduce_sum( math_ops.cast(gen_math_ops.not_equal(input_tensor, zero), dtype), nil, False, 'nonzero_count' );
                    Result := nonzero_count;
                end);
end;

class function nn_impl.sigmoid_cross_entropy_with_logits(labels, logits: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [logits, labels];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'logistic_loss', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    name := v1.ToString;
                    logits := Tops.convert_to_tensor(logits, DtInvalid, 'logits');
                    labels := Tops.convert_to_tensor(labels, DtInvalid, 'labels');
                    labels.shape.merge_with(logits.shape);

                    var zeros         := array_ops.zeros_like(logits, logits.dtype);
                    var cond          := TTensor(logits) >= zeros;
                    var relu_logits   := array_ops.where(cond, logits, zeros);
                    var neg_abs_logits:= array_ops.where(cond, -TTensor(logits), logits);

                    Result := math_ops.add(
                        TTensor(relu_logits) - logits * TTensor(labels),
                        gen_math_ops.log1p(gen_math_ops.exp(neg_abs_logits)),
                        name);
                end);
end;

class function nn_impl.zero_fraction(value: TFTensor; name: string): TFTensor;
var
  truePred, falsePred : TFunc<TFTensor> ;
begin
    var vValues : TArray<TValue> := [value];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zero_fraction', @vValues),
                function(v1: TNameScope): TFTensor
                begin
                    value := Tops.convert_to_tensor(value, DtInvalid, 'value');
                    var size : TFTensor := array_ops.size(value, '',True, Tdtypes.cint64);
                    var zero_fraction_float32 : TFTensor := nil;

                    size := gen_math_ops.less_equal(size, Tdtypes.max(Tdtypes.cint32));
                    truePred  := function : TFTensor
                                         begin
                                             Result := math_ops.cast(_count_nonzero(value, Tdtypes.cint32), TF_DataType.TF_INT64)
                                         end;
                    falsePred := function : TFTensor
                                         begin
                                             Result := _count_nonzero(value, Tdtypes.cint64)
                                         end;

                    var num_nonzero : TFTensor := control_flow_ops.cond( size, truePred, falsePred );

                    TUtils.tf_with<TNameScope>( Tops.name_scope('counts_to_fraction'),
                        procedure(v1: TNameScope)
                        begin
                            var num_zero         := math_ops.subtract(math_ops.cast(size, TF_DataType.TF_INT64), num_nonzero);
                            var num_zero_float32 := math_ops.cast(num_zero, Tdtypes.cfloat32);
                            var size_float32     := math_ops.cast(size, Tdtypes.cfloat32);
                            zero_fraction_float32:= TTensor(num_zero_float32) / size_float32;
                        end);

                    Result := array_ops.identity(zero_fraction_float32, 'fraction');
                end);
end;

end.

