unit TensorFlow.gen_math_ops;
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

         TensorFlow.Context ;

type

  gen_math_ops = record
    private
     /// <summary>
     /// Subroutine for Min or Max functions. See _min and _max
     /// </summary>
     class function MinOrMax<Tx, Ty>(input: Tx; axis:Ty; methodName: string; keep_dims: Boolean = false; name : string = ''): TFTensor; static;
    public
     class function _all(input: TFTensor; axis: TFTensor; keep_dims: Boolean= false; name: string = ''): TFTensor; static;
     /// <summary>
     /// Add all input tensors element wise.
     /// </summary>
     /// <param name="inputs"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function add_n(inputs: TArray<TFTensor>; name: string = ''): TFTensor; static;
     /// <summary>
     /// Returns the index with the largest value across dimensions of a tensor.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="dimension"></param>
     /// <param name="output_type"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function arg_max(input: TFTensor; dimension: TAxis; output_type: TF_DataType = TF_DataType.TF_INT64; name: string = ''): TFTensor; static;
     /// <summary>
     /// Returns the index with the smallest value across dimensions of a tensor.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="dimension"></param>
     /// <param name="output_type"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function arg_min(input: TFTensor; dimension: Integer; output_type: TF_DataType = TF_DataType.TF_INT64; name: string = ''): TFTensor;static;
     /// <summary>
     /// Computes Psi, the derivative of Lgamma (the log of the absolute value of
     /// `Gamma(x)`), element-wise.
     /// </summary>
     /// <param name="x"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function digamma(x: TFTensor; name: string = ''): TFTensor; static;
     /// <summary>
     ///    Returns 0 if the denominator is zero.
     /// </summary>
     /// <param name="x">
     /// </param>
     /// <param name="y">
     /// </param>
     /// <param name="name">
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'DivNoNan'.
     /// </param>
     /// <returns>
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///
     ///    *NOTE*: <c>DivNoNan</c> supports broadcasting. More about broadcasting
     ///    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
     /// </remarks>
     class function div_no_nan(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; static;
     class function mean(input: TFTensor; axis: Integer; keep_dims: Boolean = false; name: string = ''): TFTensor; overload; static;
     /// <summary>
     /// Computes the mean of elements across dimensions of a tensor.
     /// Reduces `input` along the dimensions given in `axis`. Unless
     /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
     /// `axis`. If `keep_dims` is true, the reduced dimensions are retained with length 1.
     /// </summary>
     /// <param name="input">A `Tensor`. Must be one of the following types:
     /// `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
     /// The tensor to reduce.</param>
     /// <param name="axis">A `Tensor`. Must be one of the following types: `int32`, `int64`. The dimensions to reduce.</param>
     /// <param name="keep_dims"> An optional `bool`. Defaults to `False`. If true, retain reduced dimensions with length 1.</param>
     /// <param name="name"> A name for the operation (optional).</param>
     /// <returns> A `Tensor`. Has the same type as `input`.</returns>
     class function mean(input: TFTensor; axis: TFTensor; keep_dims: Boolean = false; name: string = ''): TFTensor; overload; static;
     class function mean(inputs: TArray<TFtensor>; axis: TFTensor; keep_dims: Boolean = false; name: string = ''): TFTensor; overload; static;
     class function mean_eager_fallback(inputs: TArray<TFTensor>; axis: TFTensor; keep_dims: Boolean = false; name: string = ''; ctx: TContext = nil): TFTensor; static;
     class function prod<T1, T2>(input: T1; axis: T2; keep_dims : Boolean= false; name: string = ''): TFTensor; static;
     class function acos(x: TFTensor; name: string = ''): TFTensor;   static;
     class function asin(x: TFTensor; name: string = ''): TFTensor;   static;
     class function add(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;  overload;  static;
     class function add<Tx, Ty>(x: Tx; y: Ty; name: string = ''): TFTensor; overload;   static;
     class function add_v2<Tx, Ty>(x: Tx; y: Ty; name: string = ''): TFTensor;  static;
     class function atan(x: TFTensor; name: string = ''): TFTensor;  static;
     class function ceil(x: TFTensor; name: string = ''): TFTensor;  static;
     class function sin(x: TFTensor; name: string = ''): TFTensor;  static;
     /// <summary>
     ///    Computes sigmoid of <c>x</c> element-wise.
     /// </summary>
     /// <param name="x">
     /// </param>
     /// <param name="name">
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Sigmoid'.
     /// </param>
     /// <returns>
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///    Specifically, <c>y = 1 / (1 + exp(-x))</c>.
     /// </remarks>
     class function sigmoid(x: TFTensor; name: string = 'Sigmoid'): TFTensor;  static;
     /// <summary>
     ///    Computes the gradient of the sigmoid of <c>x</c> wrt its input.
     /// </summary>
     /// <param name='y'>
     /// </param>
     /// <param name='dy'>
     /// </param>
     /// <param name='name'>
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SigmoidGrad'.
     /// </param>
     /// <returns>
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///    Specifically, <c>grad = dy * y * (1 - y)</c>, where <c>y = sigmoid(x)</c>, and
     ///    <c>dy</c> is the corresponding input gradient.
     /// </remarks>
     class function sigmoid_grad(y: TFTensor; dy: TFTensor; name: string = 'SigmoidGrad'): TFTensor; static;
     class function sign<T>(x: T; name: string = 'Sign'): TFTensor;  static;
     class function sinh(x: TFTensor; name : string = ''): TFTensor; static;
     class function cos<T>(x: T; name : string = ''): TFTensor; static;
     class function cosh(x: TFTensor; name : string = ''): TFTensor; static;
     /// <summary>
     /// Computes the sum along segments of a tensor.
     /// </summary>
     /// <param name='data'></param>
     /// <param name='segment_ids'></param>
     /// <param name='num_segments'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function unsorted_segment_sum(data: TFTensor; segment_ids: TFTensor; num_segments: TFTensor; name : string = ''): TFTensor; static;
     class function tan(x: TFTensor; name : string = ''): TFTensor; static;
     class function tanh(x: TFTensor; name : string = ''): TFTensor; static;
     /// <summary>
     /// Computes the gradient for the tanh of `x` wrt its input.
     /// </summary>
     /// <param name='y'></param>
     /// <param name='dy'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function tanh_grad(y: TFTensor; dy: TFTensor; name : string = '') : TFTensor; static;
     class function floor(x: TFTensor; name : string = ''): TFTensor; Static;
     class function _clip_by_value(t: TFTensor; clip_value_min: TFTensor; clip_value_max: TFTensor; name : string = ''): TFTensor; static;
     class function greater<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     /// <summary>
     /// Computes the log of the absolute value of `Gamma(x)` element-wise.
     /// </summary>
     /// <param name='x'>
     /// A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
     /// </param>
     /// <param name='name'>
     /// </param>
     /// <returns>
     /// The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     class function lgamma(x: TFTensor; name : string = ''): TFTensor;static;
     class function greater_equal<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     class function less<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     class function less_equal<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     class function log1p(x: TFTensor; name : string = ''): TFTensor; static;
     class function logical_and<T>(x: T; y: T; name : string = ''): TFTensor; static;
     class function logical_not(x: TFTensor; name : string = ''): TFTensor; static;
     class function logical_or(x: TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
     class function logical_xor(x: TFTensor; y: TFTensor; name: string = 'LogicalXor'): TFTensor; static;
     class function squared_difference(x: TFTensor; y : TFTensor; name : string = ''): TFTensor; static;
     /// <summary>
     /// Computes square of x element-wise.
     /// </summary>
     /// <param name='x'> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
     /// <param name='name'> A name for the operation (optional).</param>
     /// <returns> A `Tensor`. Has the same type as `x`.</returns>
     class function square(x: TFTensor; name : string = ''): TFTensor; static;
     /// <summary>
     /// Returns which elements of x are finite.
     /// </summary>
     /// <param name='x'> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
     /// <param name='name'> A name for the operation (optional).</param>
     /// <returns> A `Tensor` of type `bool`.</returns>
     class function is_finite(x: TFTensor; name : string = ''): TFTensor; static;
     class function is_nan(x: TFTensor; name : string = ''): TFTensor; static;
     /// <summary>
     /// Computes exponential of x element-wise.  \\(y = e^x\\).
     /// </summary>
     /// <param name='x'> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
     /// <param name='name'> A name for the operation (optional).</param>
     /// <returns> A `Tensor`. Has the same type as `x`.</returns>
     class function exp(x: TFTensor; name : string = '') : TFTensor; static;
     /// <summary>
     /// Computes natural logarithm of x element-wise.
     /// </summary>
     /// <param name='x'> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
     /// <param name='name'> name: A name for the operation (optional).</param>
     /// <returns> A `Tensor`. Has the same type as `x`.</returns>
     class function log(x: TFTensor; name : string = '') : TFTensor; static;
     class function softplus(features: TFTensor; name : string = ''): TFTensor; static;
     class function cast(x: TFTensor; DstT: TF_DataType; name: string = '';Truncate : Boolean = false): TFTensor;static;
     class function neg(x: TFTensor; name : string = ''): TFTensor; static;
     class function sqrt(x: TFTensor; name : string = '') : TFTensor; static;
     class function sub(x: TFTensor; y: TFTensor; name : string = '') : TFTensor; overload; static;
     class function sub<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; overload; static;
     /// <summary>
     /// Returns the truth value of (x == y) element-wise.
     /// </summary>
     /// <param name='x'></param>
     /// <param name='y'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function equal<Tx, Ty>(x: Tx; y: Ty; incompatible_shape_error: Boolean = true; name : string = ''): TFTensor; static;
     /// <summary>
     /// Returns the real part of a complex number.
     /// Given a tensor `input` of complex numbers, this operation returns a tensor of
     /// type `float` that is the real part of each element in `input`. All elements in
     /// `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
     /// part returned by this operation and *b* is the imaginary part.
     /// </summary>
     /// <param name='input'>A `Tensor`. Must be one of the following types: `complex64`, `complex128`.</param>
     /// <param name='Tout'>An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.</param>
     /// <param name='name'>A name for the operation (optional).</param>
     /// <returns></returns>
     class function real(input: TFTensor; Tout : TF_DataType= TF_FLOAT; name : string=''): TFTensor; static;
     /// <summary>
     /// Returns the truth value of (x != y) element-wise.
     /// </summary>
     /// <typeparam name='Tx'>The type of the x.</typeparam>
     /// <typeparam name='Ty'>The type of the y.</typeparam>
     /// <param name='x'>The x.</param>
     /// <param name='y'>The y.</param>
     /// <param name='name'>The name.</param>
     /// <returns></returns>
     class function not_equal<Tx, Ty>(x: Tx; y: Ty; name : string = '') : TFTensor; static;
     class function atan2(y: TFTensor; x: TFTensor; name : string = ''): TFTensor; static;
     class function mul<Tx, Ty>(x: Tx; y: Ty; name : string = '') : TFTensor; static;
     class function mul_no_nan<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     class function real_div(x: TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
     class function reciprocal(x: TFTensor; name : string = '') : TFTensor; static;
     class function floor_mod(x: TFTensor; y: TFTensor; name : string = '') : TFTensor; static;
     class function floor_div(x: TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
     /// Multiply the matrix 'a' by the matrix 'b'.
     /// </summary>
     /// <param name='a'></param>
     /// <param name='b'></param>
     /// <param name='transpose_a'></param>
     /// <param name='transpose_b'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function mat_mul(a: TFTensor; b: TFTensor; transpose_a : Boolean= false; transpose_b : Boolean = false; name : string = '') : TFTensor; static;
     /// <summary>
     /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
     /// </summary>
     /// <param name='x'></param>
     /// <param name='y'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function maximum<T1, T2>(x: T1; y: T2; name : string = ''): TFTensor; static;
     class function minimum<T1, T2>(x: T1; y: T2; name : string = ''): TFTensor; static;
     class function _abs(x: TFTensor; name : string = ''): TFTensor; static;
     class function _any<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name : string = '') : TFTensor; static;
     class function _max<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name : string = '') : TFTensor; static;
     class function _min<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name : string = '') : TFTensor; static;
     class function pow<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
     class function _sum<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name : string = '') : TFTensor; static;
     /// <summary>
     /// Creates a sequence of numbers.
     /// </summary>
     /// <param name='start'></param>
     /// <param name='limit'></param>
     /// <param name='delta'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function range(start: TFTensor; limit: TFTensor; delta: TFTensor; name : string = '') : TFTensor;static;
     /// <summary>
     ///    Rounds the values of a tensor to the nearest integer, element-wise.
     /// </summary>
     /// <param name='x'>
     /// </param>
     /// <param name='name'>
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Round'.
     /// </param>
     /// <returns>
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///    Rounds half to even.  Also known as bankers rounding. If you want to round
     ///    according to the current system rounding mode use std::cint.
     /// </remarks>
     class function round(x: TFTensor; name: string = 'Round') : TFTensor; static;
     /// <summary>
     /// Computes reciprocal of square root of x element-wise.
     /// </summary>
     /// <param name='x'></param>
     /// <param name='name'></param>
     /// <returns></returns>
     class function rsqrt(x: TFTensor; name : string = '') : TFTensor; static;
     /// <summary>
     /// Returns the fraction of zeros in value.
     /// </summary>
     /// <param name='value'>A tensor of numeric type.</param>
     /// <param name='name'>A name for the operation (optional).</param>
     /// <returns>The fraction of zeros in value, with type float32.</returns>
     class function zero_fraction(value: TFTensor; name : string = '') : TFTensor; static;
  end;

implementation
        uses Tensorflow,
             TensorFlow.Ops,
             Tensorflow.Utils;

{ gen_math_ops }

class function gen_math_ops._all(input: TFTensor; axis: TFTensor; keep_dims: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('All', name, [ GetArg('input',input),GetArg('reduction_indices',axis),GetArg('keep_dims', keep_dims)]);
    Result := _op.outputs[0];
end;

class function gen_math_ops.add_n(inputs: TArray<TFTensor>; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('AddN', name, ExecuteOpArgs.Create([inputs])).First;
end;

class function gen_math_ops.arg_max(input: TFTensor; dimension: TAxis; output_type: TF_DataType; name: string): TFTensor;
begin
    Result :=  tf.Context.ExecuteOp('ArgMax', name, ExecuteOpArgs.Create([input, dimension])
        .SetAttributes(['output_type', output_type ])).First;
end;

class function gen_math_ops.arg_min(input: TFTensor; dimension: Integer; output_type: TF_DataType; name: string): TFTensor;
begin
    Result := Tf.Context.ExecuteOp('ArgMin', name, ExecuteOpArgs.Create([input, dimension])
        .SetAttributes(['output_type', output_type ])).First;
end;

class function gen_math_ops.digamma(x: TFTensor; name: string): TFTensor;
begin
     Result := tf.OpDefLib._apply_op_helper('Digamma', name, [GetArg('x',x)]).output;
end;

class function gen_math_ops.div_no_nan(x: TFTensor; y: TFTensor; name :string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DivNoNan', name, ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.mean(input: TFTensor; axis: Integer; keep_dims: Boolean; name: string): TFTensor;
begin
    Result := mean(input, Tops.convert_to_tensor(axis), keep_dims, name);
end;

class function gen_math_ops.mean(input: TFTensor; axis: TFTensor; keep_dims: Boolean; name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([input, axis]);
    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;
                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr('T');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'Tidx' ;
                                   pParam.vValue:= op.get_attr('Tidx');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'keep_dims' ;
                                   pParam.vValue:= op.get_attr('keep_dims');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp('Mean', name, Args
        .SetAttributes(['keep_dims', keep_dims, 'reduction_indices', axis ])).First;
end;

class function gen_math_ops.mean(inputs: TArray<TFtensor>; axis: TFTensor; keep_dims: Boolean; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
       Exit ( mean_eager_fallback(inputs, axis, keep_dims, name, tf.Context) );

    var _op := tf.OpDefLib._apply_op_helper('Mean', name,[ GetArg('inputs',inputs), GetArg('reduction_indices', axis), GetArg('keep_dims',keep_dims) ] );
    Result := _op.output;
end;

class function gen_math_ops.mean_eager_fallback(inputs: TArray<TFTensor>; axis: TFTensor; keep_dims: Boolean; name: string; ctx: TContext): TFTensor;
begin
    var t1 :Tuple<TF_DataType, TArray<TFTensor>>;
    tf.Runner.ArgsToMatchingEager(ctx, t1, TF_Datatype.DtInvalid, [ inputs ]);
    var _attr_T := t1.Value1;
    var input   := t1.Value2;

    var t2 :Tuple<TF_DataType, TArray<TFTensor>>;
    tf.Runner.ArgsToMatchingEager(ctx, t2,tf.int32_t, [ axis ]);
    var _attr_Tidx := t2.Value1;
    var axis1      := t2.Value2;
    var _inputs_flat := input + axis1;
    var _attrs : TArray<TValue> := [ 'keep_dims', keep_dims, 'T', _attr_T, 'Tidx', _attr_Tidx ] ;
    Result := tf.Runner.Execute(ctx, 'Mean', 1, _inputs_flat, _attrs, name)[0];
end;

class function gen_math_ops.prod<T1, T2>(input: T1; axis: T2; keep_dims : Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Prod', name, ExecuteOpArgs.Create([TValue.From<T1>(input), TValue.From<T2>(axis)])
              .SetAttributes(['keep_dims', keep_dims, 'reduction_indices',  TValue.From<T2>(axis) ])).First;
end;

class function gen_math_ops.acos(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Acos', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.asin(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Asin', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.add(x: TFTensor; y: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Add', name, ExecuteOpArgs.Create([x, y])).First;;
end;

class function gen_math_ops.add<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Add', name, ExecuteOpArgs.Create([TValue.From<Tx>(x), TValue.From<Ty>(y)])).First;;
end;

class function gen_math_ops.add_v2<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('AddV2', name, ExecuteOpArgs.Create([TValue.From<Tx>(x), TValue.From<Ty>(y)])).First;;
end;

class function gen_math_ops.atan(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Atan', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.ceil(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Ceil', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.sin(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sin', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.sigmoid(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sigmoid', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.sigmoid_grad(y: TFTensor; dy: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('SigmoidGrad', name, ExecuteOpArgs.Create([y, dy])).First;
end;

class function gen_math_ops.sign<T>(x: T; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sign', name, ExecuteOpArgs.Create([ TValue.From<T>(x) ])).First;
end;

class function gen_math_ops.sinh(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sinh', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.cos<T>(x: T; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Cos', name, ExecuteOpArgs.Create([ TValue.From<T>(x) ])).First;
end;

class function gen_math_ops.cosh(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Cosh', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.unsorted_segment_sum(data: TFTensor; segment_ids: TFTensor; num_segments: TFTensor; name : string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('UnsortedSegmentSum', name, [ GetArg('data', data), GetArg('segment_ids',segment_ids), GetArg('num_segments',num_segments)]);
    Result := _op.outputs[0];
end;

class function gen_math_ops.tan(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Tan', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.tanh(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Tanh', name,  ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.tanh_grad(y: TFTensor; dy: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('TanhGrad', name, ExecuteOpArgs.Create([y, dy])).First;
end;

class function gen_math_ops.floor(x: TFTensor; name : string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Floor', name, [ GetArg('x', x) ]);
    Result := _op.outputs[0];
end;

class function gen_math_ops._clip_by_value(t: TFTensor; clip_value_min: TFTensor; clip_value_max: TFTensor; name : string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ClipByValue', name, [ GetArg('t',t), GetArg('clip_value_min',clip_value_min), GetArg('clip_value_max',clip_value_max) ]);
    Result := _op.outputs[0];
end;

class function gen_math_ops.greater<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Greater', name,  ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.lgamma(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Lgamma', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.greater_equal<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('GreaterEqual', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.less<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Less', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.less_equal<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LessEqual', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.log1p(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Log1p', name, ExecuteOpArgs.Create([x])).First;;
end;

class function gen_math_ops.logical_and<T>(x: T; y: T; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LogicalAnd', name, ExecuteOpArgs.Create([ TValue.From<T>(x), TValue.From<T>(y) ])).First;
end;

class function gen_math_ops.logical_not(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LogicalNot', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.logical_or(x: TFTensor; y: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LogicalOr', name, ExecuteOpArgs.Create([ x, y ])).First;
end;

class function gen_math_ops.logical_xor(x: TFTensor; y: TFTensor; name : string): TFTensor;
begin
    Result := logical_and( logical_or(x, y), logical_not(logical_and(x, y)), name);
end;

class function gen_math_ops.squared_difference(x: TFTensor; y : TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('SquaredDifference', name, ExecuteOpArgs.Create([ x, y ])).First;
end;

class function gen_math_ops.square(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Square', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.is_finite(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('IsFinite', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.is_nan(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('IsNan', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.exp(x: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Exp', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.log(x: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Log', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.softplus(features: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Softplus', name, ExecuteOpArgs.Create([features])).First;
end;

class function gen_math_ops.cast(x: TFTensor; DstT: TF_DataType; name: string;Truncate : Boolean): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Cast', name, ExecuteOpArgs.Create([x])
       .SetAttributes(['DstT',DstT, 'Truncate',Truncate]) ).First;
end;

class function gen_math_ops.neg(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Neg', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.sqrt(x: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sqrt', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.sub(x: TFTensor; y: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sub', name, ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.sub<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sub', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.equal<Tx, Ty>(x: Tx; y: Ty; incompatible_shape_error: Boolean; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Equal', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])
        .SetAttributes(['incompatible_shape_error',incompatible_shape_error])).First;
end;

class function gen_math_ops.not_equal<Tx, Ty>(x: Tx; y: Ty; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('NotEqual', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.atan2(y: TFTensor; x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Atan2', name, ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.mul<Tx, Ty>(x: Tx; y: Ty; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops.mul_no_nan<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('MulNoNan', name,  [ GetArg('x',TValue.From<Tx>(x)), GetArg('y', TValue.From<Ty>(y)) ]);
    Result := _op.outputs[0];
end;

class function gen_math_ops.real(input: TFTensor; Tout: TF_DataType; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Real', name, ExecuteOpArgs.Create([input])
                                          .SetAttributes(['Tout', Tout ]) ).First;
end;

class function gen_math_ops.real_div(x: TFTensor; y: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RealDiv', name, ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.reciprocal(x: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Reciprocal', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.floor_mod(x: TFTensor; y: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('FloorMod', name,  ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.floor_div(x: TFTensor; y: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('FloorDiv', name, ExecuteOpArgs.Create([x, y])).First;
end;

class function gen_math_ops.mat_mul(a: TFTensor; b: TFTensor; transpose_a : Boolean; transpose_b : Boolean; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatMul', name, ExecuteOpArgs.Create([a, b])
        .SetAttributes(['transpose_a',transpose_a,'transpose_b',transpose_b])).First;
end;

class function gen_math_ops.maximum<T1, T2>(x: T1; y: T2; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Maximum', name, ExecuteOpArgs.Create([ TValue.From<T1>(x), TValue.From<T2>(y) ])).First;
end;

class function gen_math_ops.minimum<T1, T2>(x: T1; y: T2; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Minimum', name, ExecuteOpArgs.Create([ TValue.From<T1>(x), TValue.From<T2>(y) ])).First;
end;

class function gen_math_ops._abs(x: TFTensor; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Abs', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops._any<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name : string) : TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Any', name, [ GetArg('input',TValue.From<Tx>(input)),
                                                          GetArg('reduction_indices',TValue.From<Ty>(axis)),
                                                          GetArg('keep_dims',keep_dims) ]);
    Result := _op.outputs[0];
end;

class function gen_math_ops.MinOrMax<Tx, Ty>(input: Tx; axis:Ty; methodName: string; keep_dims: Boolean; name : string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ TValue.From<Tx>(input), TValue.From<Ty>(axis) ]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;
                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr('T');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'align_corners' ;
                                   pParam.vValue:= op.get_attr('align_corners');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'half_pixel_centers' ;
                                   pParam.vValue:= op.get_attr('half_pixel_centers');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp(methodName, name, Args
        .SetAttributes(['keep_dims', keep_dims, 'reduction_indices',  TValue.From<Ty>(axis) ])).First;
end;

class function gen_math_ops._max<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name : string) : TFTensor;
begin
    Result := MinOrMax(input, axis, 'Max', keep_dims, name);
end;

class function gen_math_ops._min<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name : string) : TFTensor;
begin
    Result := MinOrMax(input, axis, 'Min', keep_dims, name);
end;

class function gen_math_ops.pow<Tx, Ty>(x: Tx; y: Ty; name : string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Pow', name, ExecuteOpArgs.Create([ TValue.From<Tx>(x), TValue.From<Ty>(y) ])).First;
end;

class function gen_math_ops._sum<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name : string = '') : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Sum', name,ExecuteOpArgs.Create([ TValue.From<Tx>(input), TValue.From<Ty>(axis) ])
        .SetAttributes(['keep_dims', keep_dims, 'reduction_indices',  TValue.From<Ty>(axis) ])).First;
end;

class function gen_math_ops.range(start: TFTensor; limit: TFTensor; delta: TFTensor; name : string = '') : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Range', name, ExecuteOpArgs.Create([start, limit,delta])).First;
end;

class function gen_math_ops.round(x: TFTensor; name: string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Round', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.rsqrt(x: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('Rsqrt', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_math_ops.zero_fraction(value: TFTensor; name : string) : TFTensor;
begin
    Result := tf.Context.ExecuteOp('zero_fraction', name, ExecuteOpArgs.Create([value])).First;
end;

end.
