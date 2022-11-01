unit Tensorflow.math_ops;
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
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils,
         Spring,
         Spring.Collections.Lists,
         Spring.Collections.Enumerable,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context,
         TensorFlow.Variable ;

type
    math_ops = record
     private
       {$HINTS OFF}
      class function _truediv_python3(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
      class function _ReductionDims(x, axis: TFTensor): TFTensor; overload; static;
      class function _ReductionDims(x: TFTensor; axis: PAxis) : TFTensor; overload; static;
      class function _may_reduce_to_scalar(keepdims: Boolean; axis: PAxis;    _output: TFTensor): TFTensor; overload; static;
      class function _may_reduce_to_scalar(keepdims: Boolean; axis: TFTensor; _output: TFTensor) : TFTensor;  overload; static;
      class function _may_reduce_to_scalar(keepdims: Boolean; axis: Integer;  _output: TFTensor): TFTensor;  overload; static;
      /// <summary>
      /// Casts a tensor to type `int32`.
      /// </summary>
      /// <param name="x">A `Tensor` or `SparseTensor` or `IndexedSlices`.</param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with type `int32`.</returns>
      class function to_int32(x : TFTensor; name: string = 'ToInt32') : TFTensor; static;
       {$HINTS ON}
     public
      class function cast(x: TFTensor;         dtype: TF_DataType = DtInvalid; name : string = ''): TFTensor;overload; static;
      class function cast(x: IVariableV1;      dtype: TF_DataType = DtInvalid; name : string = ''): TFTensor;overload; static;
      class function cast(x: ResourceVariable; dtype: TF_DataType = DtInvalid; name : string = ''): TFTensor;overload; static;
      class function add<Tx, Ty>(x: Tx; y: Ty; name: string = '') : TFTensor; static;
      class function add_v2(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;overload;static;
      class function add_v2<Tx, Ty>(x: Tx; y: Ty; name: string = '') :TFTensor; overload;static;
      /// <summary>
      /// Divide two values using Python 2 semantics. Used for Tensor.__div__.
      /// </summary>
      /// <param name="x">`Tensor` numerator of real numeric type.</param>
      /// <param name="y">`Tensor` denominator of real numeric type.</param>
      /// <param name="name">A name for the operation</param>
      /// <returns>`x / y` returns the quotient of x and y.</returns>
      class function &div(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; static;
      class function truediv(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
      class function multiply(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;overload; static;
      class function multiply<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; overload; static;
      /// <summary>
      /// Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
      /// </summary>
      /// <param name="a"></param>
      /// <param name="b"></param>
      /// <param name="transpose_a">If `True`, `a` is transposed before multiplication.</param>
      /// <param name="transpose_b">If `True`, `b` is transposed before multiplication.</param>
      /// <param name="adjoint_a">If `True`, `a` is conjugated and transposed before multiplication.</param>
      /// <param name="adjoint_b">If `True`, `b` is conjugated and transposed before multiplication.</param>
      /// <param name="a_is_sparse">If `True`, `a` is treated as a sparse matrix.</param>
      /// <param name="b_is_sparse">If `True`, `b` is treated as a sparse matrix.</param>
      /// <param name="name">Name for the operation (optional).</param>
      /// <returns>
      /// A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
      /// the product of the corresponding matrices in `a` and `b`, e.g. if all
      /// transpose or adjoint attributes are `False`:
      /// </returns>
      class function matmul(a: TFTensor; b: TFTensor;
                             transpose_a : Boolean = false; transpose_b : Boolean= false;
                             adjoint_a   : Boolean = false; adjoint_b   : Boolean= false;
                             a_is_sparse : Boolean = false; b_is_sparse : Boolean= false;
                             name: string = ''): TFTensor; overload;static;
      class function matmul(a: TFTensor; b: TFTensor; name: string): TFTensor; overload;static;
      /// <summary>
      /// Returns the complex conjugate of a complex number.
      /// </summary>
      /// <param name="x">`Tensor` to conjugate.  Must have numeric or variant type.</param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>A `Tensor` that is the conjugate of `x` (with the same type).</returns>
      class function conj(x: TFTensor; name: string = ''): TFTensor; static;
      class function equal<Tx, Ty>(x: Tx; y: Ty; name : string= ''): TFTensor; static;
      class function not_equal<Tx, Ty>(x: Tx; y: Ty; name : string= '') : TFTensor; static;
      class function range(start: TValue; limit: PValue= nil; delta: PValue= nil; dtype: TF_DataType= DtInvalid; name: string = 'range'): TFTensor; static;
      class function reduce_sum(input_tensor: TFTensor; axis : TFTensor = nil; keepdims: Boolean = false; name: string = ''): TFTensor; static;
      class function pow<Tx, Ty>(x: Tx; y: Ty; name: string = '') : TFTensor; static;
      class function logical_and(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; static;

      class function abs(x: TFTensor; name: string = ''): TFTensor; static;
      /// <summary>
      /// Adds all input tensors element-wise.
      /// </summary>
      /// <param name="inputs"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function add_n(inputs: TArray<TFTensor>; name : string = ''): TFTensor; static;
      class function argmax(input: TFTensor; dimension: TAxis; output_type: TF_DataType = TF_INT64; name : string = ''): TFTensor; static;
      class function Round(x: TFTensor; name : string = ''): TFTensor; static;
      class function cos(x : TFTensor; name : string = ''): TFTensor; static;
      class function saturate_cast(value: TFTensor; dtype: TF_DataType; name : string = ''): TFTensor; static;
      class function cumsum<T>(x : TFTensor; axis: T; exclusive: Boolean = false; reverse: Boolean = false; name : string = ''): TFTensor; static;
      /// <summary>
      /// Computes Psi, the derivative of Lgamma (the log of the absolute value of
      /// `Gamma(x)`), element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function digamma(x : TFTensor; name : string = ''): TFTensor; static;
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
      class function div_no_nan(x : TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
      class function einsum(equation: string; inputs: TFTensors; name : string = ''): TFTensor; static;
      class function greater_equal<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
      /// <summary>
      /// Computes the Gauss error function of `x` element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function erf(x : TFTensor; name : string = ''): TFTensor; static;
      class function sqrt(x : TFTensor; name : string = ''): TFTensor; static;
      class function mul_no_nan<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
      class function scalar_mul<Tscale, Tx>(scale: Tscale; x: Tx; name : string = ''): TFTensor; static;
      class function real(input: TFTensor; name : string = ''): TFTensor; static;
      /// <summary>
      /// Computes the mean of elements across dimensions of a tensor.
      /// Reduces `input_tensor` along the dimensions given in `axis`.
      /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
      /// entry in `axis`. If `keepdims` is true, the reduced dimensionsare retained with length 1.
      /// If `axis` is None, all dimensions are reduced, and a tensor with a single element is returned.
      /// </summary>
      /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
      /// <param name="axis">The dimensions to reduce. If `None` (the default), reduces all
      /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
      /// <param name="keepdims"> If true, retains reduced dimensions with length 1.</param>
      /// <param name="name"> A name for the operation (optional).</param>
      class function reduce_mean(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''; reduction_indices: PInteger = nil): TFTensor; static;
      /// <summary>
      /// Computes the product of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor"></param>
      /// <param name="axis"></param>
      /// <param name="keepdims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function reduce_prod    (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function reduce_std     (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function reduce_variance(input_tensor: TFTensor; axis: TAxis; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      /// <summary>
      /// Computes the "logical and" of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor"></param>
      /// <param name="axis"></param>
      /// <param name="keepdims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function reduce_all(input_tensor: TFTensor; axis: PAxis= nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      /// <summary>
      /// Computes log(sum(exp(elements across dimensions of a tensor))).
      /// Reduces `input_tensor` along the dimensions given in `axis`.
      /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
      /// entry in `axis`. If `keepdims` is true, the reduced dimensions
      /// are retained with length 1.
      ///
      /// If `axis` has no entries, all dimensions are reduced, and a
      /// tensor with a single element is returned.
      ///
      /// This function is more numerically stable than log(sum(exp(input))). It avoids
      /// overflows caused by taking the exp of large inputs and underflows caused by
      /// taking the log of small inputs.
      /// </summary>
      /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
      /// <param name="axis"> The dimensions to reduce. If `None` (the default), reduces all
      /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
      /// <param name="keepdims"></param>
      /// <returns> The reduced tensor.</returns>
      class function reduce_logsumexp(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function reduce_any(input_tensor:       TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function reduce_max(input_tensor:       TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function reduce_min(input_tensor:       TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name : string= ''): TFTensor; static;
      class function sigmoid<T>(x: T; name : string = ''): TFTensor; static;
      class function sign<T>(x: T; name : string = ''): TFTensor; static;
      class function sin(x : TFTensor; name : string = ''): TFTensor; static;
      /// <summary>
      /// Returns (x - y)(x - y) element-wise.
      /// </summary>
      /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
      /// <param name="y"> A `Tensor`. Must have the same type as `x`.</param>
      /// <param name="name"> A name for the operation (optional).</param>
      /// <returns>A `Tensor`. Has the same type as `x`.</returns>
      class function square_difference(x : TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
      class function square(x : TFTensor; name : string = ''): TFTensor; static;
      class function subtract<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
      class function log(x : TFTensor; name : string = ''): TFTensor; static;
      class function lgamma(x : TFTensor; name : string = ''): TFTensor; static;
      class function linspace(start: TFTensor; stop: TFTensor; num: Integer = 50; name: string = ''; axis: Integer = 0): TFTensor; static;
      /// <summary>
      /// Helper function for reduction ops.
      /// </summary>
      /// <param name="input_shape">1-D Tensor, the shape of the Tensor being reduced.</param>
      /// <param name="axes">1-D Tensor, the reduction axes.</param>
      /// <returns>A 1-D Tensor, the output shape as if keepdims were set to True.</returns>
      class function reduced_shape(input_shape: TFTensor; axes: TFTensor): TFTensor; static;
      /// <summary>
      /// Computes the reciprocal of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function reciprocal(x : TFTensor; name : string = ''): TFTensor; static;
      class function realdiv(x : TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
      /// <summary>
      /// Computes the sum along segments of a tensor.
      /// </summary>
      /// <param name="data"></param>
      /// <param name="segment_ids"></param>
      /// <param name="num_segments"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function unsorted_segment_sum(data: TFTensor; segment_ids: TFTensor; num_segments: TFTensor; name : string = ''): TFTensor; static;
      /// <summary>
      /// Computes reciprocal of square root of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function rsqrt(x : TFTensor; name : string = ''): TFTensor; static;
      class function floor(x : TFTensor; name : string = ''): TFTensor; static;
      class function floordiv(x : TFTensor; y: TFTensor; name : string = ''): TFTensor; static;
      class function minimum<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
      class function maximum<Tx, Ty>(x: Tx; y: Ty; name : string = ''): TFTensor; static;
      class function batch_matmul(x : TFTensor; y: TFTensor; adj_x: Boolean = false; adj_y: Boolean = false; name : string = ''): TFTensor; static;
      class function bincount(arr: TFTensor; weights : TFTensor = nil; minlength : TFTensor= nil; maxlength: TFTensor = nil; dtype: TF_DataType = TF_INT32; name: string = ''; axis: PTFShape = nil; binary_output: Boolean = false): TFTensor; static;
      class function tanh(x : TFTensor; name : string = ''): TFTensor; static;
      class function tensordot(a: TFTensor; b: TFTensor; axes: TNDArray; name : string = ''): TFTensor; static;
      class function _tensordot_axes(a: TFTensor; axes: TNDArray) : Tuple< TArray<Integer>,TArray<Integer> >; static;
      class function _tensordot_reshape(a: TFTensor; axes: TArray<Integer>; flipped: Boolean = false) : Tuple< TFTensor, TArray<Integer>,TArray<Integer> >; static;
  end;

implementation
         uses Tensorflow,
              TensorFlow.Tensor,
              TensorFlow.Constant_op,
              TensorFlow.Ops,
              Tensorflow.gen_array_ops,
              TensorFlow.gen_data_flow_ops,
              TensorFlow.gen_math_ops,
              Tensorflow.array_ops,
              Tensorflow.NameScope,
              Tensorflow.Utils,
              TensorFlow.Framework,
              NumPy.NDArray,
              Numpy;

{ math_ops }

class function math_ops.digamma(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.digamma(x, name);
end;

class function math_ops.&div(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x, y];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'div', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                                                y := Tops.convert_to_tensor(y, TdTypes.as_base_dtype(x.dtype), 'y');
                                                var x_dtype := TdTypes.as_base_dtype(x.dtype);
                                                var y_dtype := TdTypes.as_base_dtype(y.dtype);
                                                if x_dtype <> y_dtype then
                                                   raise Exception.Create('x and y must have the same dtype, got {x_dtype} != {y_dtype}');
                                                if ( TDtypes.is_floating(x_dtype) ) or ( TDtypes.is_complex(x_dtype) ) then  Result := gen_math_ops.real_div( x, y, name)
                                                else                                                                         Result := gen_math_ops.floor_div(x, y, name);
                                            end );
end;

class function math_ops.div_no_nan(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x, y];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'div_no_nan', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                                                y := Tops.convert_to_tensor(y, TdTypes.as_base_dtype(x.dtype), 'y');
                                                var x_dtype := TdTypes.as_base_dtype(x.dtype);
                                                var y_dtype := TdTypes.as_base_dtype(y.dtype);
                                                if x_dtype <> y_dtype then
                                                   raise Exception.Create('x and y must have the same dtype, got {x_dtype} != {y_dtype}');
                                                Result := gen_math_ops.div_no_nan(x, y, name);
                                            end );
end;

class function math_ops.einsum(equation: string; inputs: TFTensors; name: string): TFTensor;
begin
    var a := inputs.ToArray;
    var vValues : TArray<TValue> := [];
    for var i := 0 to Length(a) - 1 do
       vValues := vValues +[ a[i] ];

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'einsum', @inputs),
                                function(v1: TNameScope): TFTensor
                                  begin
                                      var Args := ExecuteOpArgs.Create(vValues);
                                      Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                             begin
                                                 Result := [];
                                                 var pParam : TParameter;
                                                 pParam.sNome := 'equation' ;
                                                 pParam.vValue:= op.get_attr('equation');
                                                 Result := Result + [ pParam ] ;

                                                 pParam.sNome := 'N' ;
                                                 pParam.vValue:= op.get_attr('N');
                                                 Result := Result + [ pParam ] ;

                                                 pParam.sNome := 'T' ;
                                                 pParam.vValue:= op.get_attr('T');
                                                 Result := Result + [ pParam ] ;
                                             end;
                                      Args.SetAttributes(['equation',equation]);
                                      Result := tf.Context.ExecuteOp('Einsum', name, Args).FirstOrDefault(nil);
                                  end );
end;

class function math_ops.equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.equal(x, y, True, name);
end;

class function math_ops.erf(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Erf', name, ExecuteOpArgs.Create([x])).FirstOrDefault(nil)
end;

class function math_ops.floor(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Floor', name, ExecuteOpArgs.Create([x])).FirstOrDefault(nil)
end;

class function math_ops.floordiv(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x, y];

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'floordiv', @vValues),
                                function(v1: TNameScope): TFTensor
                                  begin
                                      Result := gen_math_ops.floor_div(x, y, v1.ToString);
                                  end );
end;

class function math_ops.greater_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.greater_equal<Tx, Ty>(x, y, name);
end;

class function math_ops.lgamma(x: TFTensor; name: string): TFTensor;
begin
    Result :=  gen_math_ops.lgamma(x, name);
end;

class function math_ops.linspace(start, stop: TFTensor; num: Integer; name: string; axis: Integer): TFTensor;
begin
    var vValues : TArray<TValue> := [start, stop];

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'linspace', @vValues),
                                function(v1: TNameScope): TFTensor
                                  begin
                                      var num_int_tensor := array_ops.constant(num);
                                      {$HINTS OFF}
                                      var num_tensor     := array_ops.constant(num, start.dtype);
                                      var broadcast_shape := array_ops.broadcast_dynamic_shape(array_ops.shape(start), array_ops.shape(stop));
                                      {$HINTS ON}
                                      start               := gen_array_ops.broadcast_to(start, broadcast_shape);
                                      stop                := gen_array_ops.broadcast_to(stop,  broadcast_shape);
                                      var expanded_start := array_ops.expand_dims(start, axis);
                                      var expanded_stop  := array_ops.expand_dims(stop,  axis);
                                      var shape := array_ops.shape(expanded_start);
                                      var ndims := array_ops.shape(shape)[0];
                                      var axis_tensor := array_ops.where_v2(constant_op.constant(axis >= 0), TObject(axis), TTensor(ndims) + axis);
                                      // The purpose is to avoid having negative values when repeating.
                                      var num_fill := gen_math_ops.maximum( TTensor(num_int_tensor) - 2, 0);
                                      var n_steps  := gen_math_ops.maximum( TTensor(num_int_tensor) - 1, 1);
                                      var delta : TFTensor   := TTensor(TTensor(expanded_stop) - expanded_start) / cast(n_steps, expanded_stop.dtype);
                                      var range_end          := array_ops.where_v2( TTensor(num_int_tensor) >= 0, n_steps, TObject(-1));
                                      var range_end_value : TValue := range_end;
                                      var desired_range      := cast( range(1,@range_end_value, nil,Tdtypes.cint64), delta.dtype );
                                      var mask               := gen_math_ops.equal(axis_tensor, range(ndims));
                                      var desired_range_shape:= array_ops.where_v2(mask, num_fill, TObject(1));
                                      desired_range          := array_ops.reshape(desired_range, desired_range_shape);
                                      var res                := TTensor(TTensor(expanded_start) + delta) * desired_range;
                                      // Add the start and endpoints to the result, and slice out the desired
                                      // portion.
                                      var all_tensors : TArray<TFTensor> := [ expanded_start, res, expanded_stop ];
                                      var concatenated                   := array_ops.concat(all_tensors, axis);
                                      var _begin                         := array_ops.zeros_like(shape);
                                      var size                           := array_ops.where_v2(mask, num_int_tensor, shape);
                                      Result := array_ops.slice(concatenated, _begin, size);
                                  end );
end;

class function math_ops.log(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.log(x, name);
end;

class function math_ops.logical_and(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.logical_and(x, y, name);
end;

class function math_ops.not_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.not_equal(x, y, name)
end;

class function math_ops.pow<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    var vX := TValue.From<Tx>(x);
    var vY := TValue.From<Ty>(y);
    var newVal : TValue := TValue.From<TArray<TValue>>([vX,vY]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Pow', @newVal),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;

                        var x_tensor := Tops.convert_to_tensor(vX, DtInvalid, 'x');
                        var y_tensor := Tops.convert_to_tensor(vY, Tdtypes.as_base_dtype(x_tensor.dtype), 'y');

                        Result := tf.Context.ExecuteOp('Pow', name, ExecuteOpArgs.Create([x_tensor, y_tensor])).FirstOrDefault(nil)
                    end );
end;

class function math_ops.range(start: TValue; limit: PValue; delta: PValue; dtype: TF_DataType; name: string): TFTensor;
begin
    if limit = nil then
    begin
        limit := @start;
        start := 0;
    end;
    var dtype1 : TF_DataType;
    if not (dtype = dtinvalid) then  dtype1 := dtype
    else                             dtype1 := TUtils.GetdataType(limit^);

    var newVal : TValue := TValue.From<TArray<TValue>>([start, limit^]);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Range', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;

                                                var start1 := Tops.convert_to_tensor(start, dtype1, 'start');
                                                var limit1 := Tops.convert_to_tensor(limit^, dtype1, 'limit');
                                                var v : TValue;
                                                if delta = nil   then v := 1
                                                else                  v := delta^;
                                                var delta1 := Tops.convert_to_tensor(v, dtype1, 'delta');
                                                Result := gen_math_ops.range(start1, limit1, delta1, name);
                                            end );
end;

class function math_ops.real(input: TFTensor; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([input]);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Real', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 input := Tops.convert_to_tensor(input, DtInvalid, 'input');
                                 if TDtypes.is_complex(input.dtype) then
                                 begin
                                     var real_dtype := TDtypes.real_dtype(input.dtype);
                                     Result := gen_math_ops.real(input, real_dtype, v1.ToString);
                                 end else
                                 begin
                                     Result := input;
                                 end;
                            end );
end;

class function math_ops.realdiv(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.real_div(x, y, name);
end;

class function math_ops.reciprocal(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.reciprocal(x, name);
end;

class function math_ops.reduced_shape(input_shape, axes: TFTensor): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        var input_shape_val := input_shape.numpy;
        for  var axes_val in axes.ToArray<integer> do
            input_shape_val[axes_val] := NDArray(1);
        Result := tf.constant(input_shape_val);
        Exit;
    end;
    input_shape := to_int32(input_shape);
    axes        := to_int32(axes);
    var input_rank := array_ops.size(input_shape);
    axes           := TTensor(TTensor(axes) + input_rank) mod input_rank;
    var axes_shape := array_ops.shape(axes);
    var rng        := math_ops.range(input_rank);
    var a1 : TArray<TFTensor> := [ rng, axes ];
    var fill := gen_array_ops.fill(axes_shape, 1);
    var a2 : TArray<TFTensor> := [ input_shape, fill ];
    Result := gen_data_flow_ops.dynamic_stitch(a1, a2);
end;

class function math_ops.reduce_all(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var all := gen_math_ops._all(input_tensor, _ReductionDims(input_tensor, axis), keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis, all);
end;

class function math_ops.reduce_any(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var r   := _ReductionDims(input_tensor, axis);
    var max : TFTensor;
    if axis <> nil then  max := gen_math_ops._any(input_tensor, axis, keepdims, name)
    else                 max := gen_math_ops._any(input_tensor, r, keepdims, name);

    Result := _may_reduce_to_scalar(keepdims, axis, max);
end;

class function math_ops.reduce_logsumexp(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([input_tensor]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'ReduceLogSumExp', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                var raw_max := reduce_max(input_tensor, axis^, true);
                                var my_max  := array_ops.stop_gradient(array_ops.where(gen_math_ops.is_finite(raw_max), raw_max, array_ops.zeros_like(raw_max)));
                                var res     := gen_math_ops.log(
                                reduce_sum(
                                    gen_math_ops.exp(gen_math_ops.sub(input_tensor, my_max)),
                                    constant_op.constant(axis^[0]),
                                    keepdims));
                                if  not keepdims then
                                begin
                                    my_max := array_ops.reshape(my_max, array_ops.shape(res));
                                end;
                                res := gen_math_ops.add(res, my_max);
                                Result := _may_reduce_to_scalar(keepdims, axis, res);
                            end );
end;

class function math_ops.reduce_max(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var r := _ReductionDims(input_tensor, axis);
    var max : TFTensor;
    if axis <> nil  then max := gen_math_ops._max(input_tensor, axis, keepdims, name)
    else                 max := gen_math_ops._max(input_tensor, r, keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis, max);
end;

class function math_ops.reduce_mean(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string; reduction_indices: PInteger): TFTensor;
begin
    var r := _ReductionDims(input_tensor, axis);
    var axis_tensor : TFTensor;
    if axis = nil then axis_tensor := r
    else               axis_tensor := Tops.convert_to_tensor(axis^);

    var m := gen_math_ops.mean(input_tensor, axis_tensor, keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis_tensor, m);
end;

class function math_ops.reduce_min(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var r   := _ReductionDims(input_tensor, axis);
    var min := gen_math_ops._min(input_tensor, r, keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis, min);
end;

class function math_ops.reduce_prod(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    var r := _ReductionDims(input_tensor, axis);
    if axis = nil then
    begin
        var m  := gen_math_ops.prod(input_tensor, r, keepdims, name);
        Result := _may_reduce_to_scalar(keepdims, axis, m);
    end else
    begin
        var m  := gen_math_ops.prod(input_tensor, axis, keepdims, name);
        Result := _may_reduce_to_scalar(keepdims, axis, m);
    end
end;

class function math_ops.reduce_std(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    if name = '' then
      name := 'reduce_std';

    var newVal : TValue := TValue.From<TArray<TValue>>([input_tensor]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'reduce_std', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                var variance := reduce_variance(input_tensor, axis^, keepdims);
                                Result := gen_math_ops.sqrt(variance);
                            end );
end;

class function math_ops.reduce_sum(input_tensor, axis: TFTensor; keepdims: Boolean; name: string): TFTensor;
begin
    var r  := _ReductionDims(input_tensor, axis);
    var m  := gen_math_ops._sum(input_tensor, r, keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis, m);
end;

class function math_ops.reduce_variance(input_tensor: TFTensor; axis: TAxis; keepdims: Boolean; name: string): TFTensor;
begin
    if name = '' then
      name := 'reduce_variance';

    var newVal : TValue := TValue.From<TArray<TValue>>([input_tensor]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'reduce_variance', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 var means := reduce_mean(input_tensor, axis, true);
                                 if TDTypes.is_integer(means.dtype) then
                                    raise Exception.Create('Input must be either real or complex');
                                 var diff : TFTensor := TTensor(input_tensor) - means;
                                 var squared_deviations : TFTensor;
                                 if TDTypes.is_complex(diff.dtype) then
                                 begin
                                     var real_dtype := TDTypes.real_dtype(diff.dtype);
                                     squared_deviations := gen_math_ops.real( gen_math_ops.mul(conj(diff), diff), real_dtype );
                                 end else
                                 begin
                                     squared_deviations := gen_math_ops.square(diff);
                                 end;
                                 Result := reduce_mean(squared_deviations, axis, keepdims);
                            end );
end;

class function math_ops.Round(x: TFTensor; name: string): TFTensor;
begin
    x := Tops.convert_to_tensor(x, DtInvalid, 'x');

    if TDTypes.is_integer(x.dtype) then  Result := x
    else                                 Result := gen_math_ops.round(x, name);
end;

class function math_ops.rsqrt(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.rsqrt(x, name);
end;

class function math_ops.saturate_cast(value: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([value]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'saturate_cast', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 value := Tops.convert_to_tensor(value, DtInvalid, 'value');
                                 // dtype = dtypes.as_dtype(dtype).as_base_dtype();
                                 if TDtypes.min(value.dtype) < TDtypes.min(dtype) then
                                     value := gen_math_ops.maximum(
                                         value,
                                         Tops.convert_to_tensor(TDtypes.min(dtype), value.dtype, 'min'));
                                 if TDtypes.max(value.dtype) > TDtypes.max(dtype) then
                                     value := gen_math_ops.minimum(
                                         value,
                                         Tops.convert_to_tensor(TDtypes.max(dtype), value.dtype, 'max'));
                                 Result := cast(value, dtype, name);
                            end );
end;

class function math_ops.scalar_mul<Tscale, Tx>(scale: Tscale; x: Tx; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([TValue.From<Tscale>(scale), TValue.From<Tx>(x)])).FirstOrDefault(nil)
end;

class function math_ops.sigmoid<T>(x: T; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From<T>(x);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Sigmoid', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name         := v1.ToString;
                                var x_tensor := Tops.convert_to_tensor(TValue.From<T>(x),DtInvalid, 'x');
                                Result := gen_math_ops.sigmoid(x_tensor, name);
                            end );
end;

class function math_ops.sign<T>(x: T; name: string): TFTensor;
begin
   Result := gen_math_ops.sign(x, name);
end;

class function math_ops.sin(x: TFTensor; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('Sin', name, ExecuteOpArgs.Create([x])).FirstOrDefault(nil)
end;

class function math_ops.sqrt(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sqrt(x, name);
end;

class function math_ops.square(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.square(x, name);
end;

class function math_ops.square_difference(x, y: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.squared_difference(x, y);
end;

class function math_ops.subtract<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.sub(x, y, name);
end;

class function math_ops._may_reduce_to_scalar(keepdims: Boolean; axis: PAxis; _output: TFTensor) : TFTensor;
begin
    Result := nil;
    var dims: TArray<TF_int64_t> := [];
    if ( not common_shapes.has_fully_defined_shape(_output) ) and ( not keepdims) and (axis = nil)  then
    begin
         _output.shape := TFShape.Create(dims);
        Result := _output;
    end;
end;

class function math_ops._may_reduce_to_scalar(keepdims: Boolean; axis: TFTensor; _output: TFTensor) : TFTensor;
begin
    var dims: TArray<TF_int64_t> := [];
    if ( not common_shapes.has_fully_defined_shape(_output) ) and ( not keepdims) and (axis = nil)   then
        _output.shape := TFShape.Create(dims);
     Result := _output;
end;

class function math_ops._may_reduce_to_scalar(keepdims: Boolean; axis: Integer; _output: TFTensor): TFTensor;
begin
    Result := _output
end;

class function math_ops._ReductionDims(x, axis: TFTensor): TFTensor;
begin
    if axis <> nil then
    begin
        Result := axis;
    end else
    begin
        var rank := array_ops.rank(x);
        var rank_value : TValue := rank;
        var r1_value   : TValue := 1;
        Result := range(0, @rank_value, @r1_value);
    end;
end;


class function math_ops.matmul(a, b: TFTensor; transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse: Boolean;
  name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ a ];
    vValues := vValues + [ b ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MatMul', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;

                                                if (transpose_a) and (adjoint_a) then
                                                    raise Exception.Create('Only one of transpose_a and adjoint_a can be True.');
                                                if (transpose_b) and (adjoint_b) then
                                                    raise Exception.Create('Only one of transpose_b and adjoint_b can be True.');
                                                if adjoint_a then
                                                begin
                                                    a := conj(a);
                                                    transpose_a := true;
                                                end;
                                                if adjoint_b then
                                                begin
                                                    b := conj(b);
                                                    transpose_b := true;
                                                end;
                                                result := gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
                                            end );
end;

class function math_ops.matmul(a, b: TFTensor; name: string): TFTensor;
begin
    Result := matmul(a,b,False,False,False,False,False,False,name);
end;

class function math_ops.maximum<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
   Result := gen_math_ops.maximum(x, y, name);
end;

class function math_ops.minimum<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
   Result := gen_math_ops.minimum(x, y, name);
end;

class function math_ops.multiply(x, y: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([x, y])).FirstOrDefault(nil)
end;

class function math_ops.multiply<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
   Result := gen_math_ops.mul(x, y, name);
end;

class function math_ops.mul_no_nan<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.mul_no_nan(x, y, name);
end;

class function math_ops.tanh(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.tanh(x, name);
end;

class function math_ops.tensordot(a, b: TFTensor; axes: TNDArray; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [a,b,axes ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Tensordot', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var aAssi := _tensordot_axes(a, axes);
                                                var a_axes := aAssi.Value1;
                                                var b_axes := aAssi.Value2;

                                                var aReshape           := _tensordot_reshape(a, a_axes);
                                                var a_reshape          := aReshape.Value1;
                                                var a_free_dims        := aReshape.Value2;
                                                var a_free_dims_static := aReshape.Value3;
                                                var bReshape           := _tensordot_reshape(b, b_axes,True);
                                                var b_reshape          := aReshape.Value1;
                                                var b_free_dims        := aReshape.Value2;
                                                var b_free_dims_static := aReshape.Value3;
                                                var ab_matmul := matmul(a_reshape, b_reshape);
                                                var dims := TList<integer>.Create;
                                                try
                                                  dims.AddRange(a_free_dims);
                                                  dims.AddRange(b_free_dims);
                                                  if ab_matmul.shape.Equals(dims) then
                                                      Result := ab_matmul
                                                  else
                                                      Result := array_ops.reshape( ab_matmul, tf.constant( TValue.From< TArray<Integer> >(dims.ToArray) ), name);
                                                finally
                                                   dims.free;
                                                end;
                                            end );
end;

class function math_ops._tensordot_axes(a: TFTensor; axes: TNDArray): Tuple<TArray<Integer>, TArray<Integer>>;
begin
    if axes.rank = 0 then
    begin
        var axe : Integer := NDArray(axes);
        if axe > a.shape.ndim then
           raise Exception.Create('`axes` must not be larger than the number of dimensions of tensor {a}.  Received {axes}, vs tensor dimensions {a.ndim}.');
        Result := Tuple<TArray<Integer>, TArray<Integer>>.Create(TUtils.range(a.shape.ndim - axe, a.shape.ndim).ToArray,
                                                                 TUtils.range(0, axe).ToArray);
    end else
    begin
        var a_axe: Integer := NDArray(axes[0]);
        var b_axe: Integer := NDArray(axes[1]);
        Result := Tuple<TArray<Integer>, TArray<Integer>>.Create([ a_axe ], [ b_axe ]);
    end;
end;

class function math_ops._tensordot_reshape(a: TFTensor; axes: TArray<Integer>; flipped: Boolean): Tuple<TFTensor, TArray<Integer>, TArray<Integer>>;
var
  Selfun,Selfun1   : TFunc<Integer,Integer>;
  Wherefun         : TPredicate<Integer>;
  shape_a          : TArray<Integer>;
begin
    Selfun   := Function(x: Integer): Integer
                 begin
                      if x >= 0 then Result := x
                      else           Result := x +Length(shape_a);
                 end ;
    Selfun1   := Function(x: Integer): Integer
                 begin
                      Result := shape_a[x];
                 end ;
    Wherefun :=  function(const i: Integer): Boolean
                  begin
                      Result := not TArray.Contains<Integer>(axes, i);
                  end ;

    if (a.shape.IsFullyDefined) and (TUtils.IsInstance<TArray<Integer>,TFTensor, TArray<Integer>, TArray<Integer>>(axes, Result) ) then
    begin
        shape_a := a.shape.as_int_list;
        // axes
        axes := Enumerable<Integer>.Create(axes).Select(Selfun).ToArray;
        // free
        var free : TArray<Integer> := TUtils.range(a.shape.ndim).Where(Wherefun).ToArray;
        // free_dims
        var free_dims : TArray<Integer> := Enumerable<Integer>.Create(free).Select(Selfun1).ToArray;
        var prod_free : Integer := NDArray(np.prod<Integer>(free_dims));
        // prod_axes
        var prod_axes  : Integer :=  NDArray( np.prod<Integer>( Enumerable<Integer>.Create(axes).Select(Selfun1).ToArray ) );
        // perm
        var perm : TList<Integer> := TList<Integer>.Create;
        try
          if flipped then
          begin
              perm.AddRange(axes);
              perm.AddRange(free);
          end else
          begin
              perm.AddRange(free);
              perm.AddRange(axes);
          end;
          // new_shape
          var new_shape : TFShape;
          if flipped then  new_shape := TFShape.Create([ prod_axes, prod_free ])
          else             new_shape := TFShape.Create([ prod_free, prod_axes ]);
          var a_trans := a;
          var reshaped_a := array_ops.reshape(a_trans, new_shape);
          Result := Tuple<TFTensor, TArray<Integer>, TArray<Integer>>.Create(reshaped_a, free_dims, free_dims);
          Exit;
        finally
          perm.free;
        end;
    end;
    raise Exception.Create('Not Implemented "_tensordot_reshape');
end;

class function math_ops.to_int32(x: TFTensor; name: string): TFTensor;
begin
    Result := cast(x, Tdtypes.cint32, name)
end;

class function math_ops.truediv(x, y: TFTensor; name: string): TFTensor;
begin
    Result := _truediv_python3(x, y, name);
end;

class function math_ops.unsorted_segment_sum(data, segment_ids, num_segments: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments, name);
end;

class function math_ops._truediv_python3(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ x ];
    vValues := vValues + [ y ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'truediv', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var x_dtype := Tdtypes.as_base_dtype(x.Dtype);
                                                var y_dtype := Tdtypes.as_base_dtype(y.Dtype);
                                                if x_dtype <> y_dtype then
                                                   raise Exception.Create('x and y must have the same dtype, got {x_dtype} != {y_dtype}');
                                                var dtype : TF_DataType;
                                                case x_dtype of
                                                  TF_DataType.TF_UINT8  : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT8   : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT16  : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_UINT16 : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT32  : dtype := TF_DataType.TF_DOUBLE;
                                                  TF_DataType.TF_INT64  : dtype := TF_DataType.TF_DOUBLE;
                                                else
                                                  dtype := x_dtype;
                                                end;
                                                x := cast(x, dtype);
                                                y := cast(y, dtype);
                                                Result := gen_math_ops.real_div(x, y, name);
                                            end );
end;

class function math_ops.abs(x: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ x ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Abs', @newVal),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;
                        x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                        if TDtypes.is_complex(x.dtype) then
                           raise Exception.Create('math_ops.abs for dtype.is_complex');
                        //return gen_math_ops.complex_abs(x, Tout: x.dtype.real_dtype, name: name);
                        Result := gen_math_ops._abs(x, name);
                    end );
end;

class function math_ops.add<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.add(x, y, name);
end;

class function math_ops.add_n(inputs: TArray<TFTensor>; name: string): TFTensor;
begin
    inputs := Tops.convert_n_to_tensor_or_indexed_slices(inputs);

    if Length(inputs) = 1 then
    begin
        var values := inputs[0];
        if name <> '' then
            Exit( array_ops.identity(values,  name) );
        Exit( values );
    end;
    Result := gen_math_ops.add_n(inputs, name);
end;

class function math_ops.add_v2(x, y: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('AddV2', name, ExecuteOpArgs.Create([x, y])).FirstOrDefault(nil)
end;

class function math_ops.add_v2<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
   Result := gen_math_ops.add_v2(x, y, name);
end;

class function math_ops.argmax(input: TFTensor; dimension: TAxis; output_type: TF_DataType; name: string): TFTensor;
begin
   Result := gen_math_ops.arg_max(input, dimension, output_type, name);
end;

class function math_ops.batch_matmul(x, y: TFTensor; adj_x, adj_y: Boolean; name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := [ x, y ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MatMul', @newVal),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;
                        x := Tops.convert_to_tensor(x, DtInvalid, 'a');
                        y := Tops.convert_to_tensor(y, DtInvalid, 'b');

                        Result := tf.Context.ExecuteOp('BatchMatMul', name, ExecuteOpArgs.Create([x, y])
                                         .SetAttributes(['adj_x',adj_x,'adj_y',adj_y])).FirstOrDefault(nil);

                    end );
end;

class function math_ops.bincount(arr, weights, minlength, maxlength: TFTensor; dtype: TF_DataType; name: string; axis: PTFShape; binary_output: Boolean): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'bincount', nil),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;
                        var array_is_nonempty := TTEnsor( math_ops.reduce_prod(array_ops.shape(arr)) ) > 0;
                        var output_size := math_ops.cast(array_is_nonempty, Tdtypes.cint32) * (TTensor(math_ops.reduce_max(arr)) + 1);
                        if minlength <> nil then
                            output_size := math_ops.maximum(minlength, output_size);
                        if maxlength <> nil then
                            output_size := math_ops.minimum(maxlength, output_size);
                        var i := TArray<Int64>.Create();
                        var weights := constant_op.constant( TValue.From< TArray<Int64> >(i), dtype, 'Const' );

                        Result := tf.Context.ExecuteOp('Bincount', name, ExecuteOpArgs.Create([arr, output_size, weights])).FirstOrDefault(nil);

                    end );
end;

class function math_ops.cast(x: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var base_type := Tdtypes.as_base_dtype(dtype);
    if base_type = x.dtype then
        Exit(x);

    var vvalue := TValue.From<TFTensor>(x);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cast', @vvalue),
                      function(v1: TNameScope): TFTensor
                        begin
                            name := string(v1.ToString);
                            if Tdtypes.as_base_dtype(x.dtype) <> base_type then
                                x := gen_math_ops.cast(x, base_type, name);
                            Result := x;
                        end );
end;

class function math_ops.cast(x: IVariableV1; dtype: TF_DataType; name: string): TFTensor;
begin
    var base_type := Tdtypes.as_base_dtype(dtype);
    if base_type = x.dtype then
        Exit(x.AsTensor);

    var vValues : TArray<TValue>;
    vValues := [  TValue.From<IVariableV1>(x) ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cast', @newVal),
                      function(v1: TNameScope): TFTensor
                        begin
                            name := string(v1.ToString);
                            var t_x := Tops.convert_to_tensor( TValue.From<IVariableV1>(x),DtInvalid, 'x');
                            if TDTypes.as_base_dtype(t_x.dtype) <> base_type then
                                t_x := gen_math_ops.cast(t_x, base_type, name);
                            Result := t_x;
                        end );
end;

class function math_ops.cast(x: ResourceVariable; dtype: TF_DataType; name: string): TFTensor;
begin
    var base_type := Tdtypes.as_base_dtype(dtype);
    if base_type = x.dtype then
        Exit(TResourceVariable( x ));

    var vValues : TArray<TValue>;
    vValues := [  TValue.From<IVariableV1>(x) ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cast', @newVal),
                      function(v1: TNameScope): TFTensor
                        begin
                            name := string(v1.ToString);
                            var t_x := Tops.convert_to_tensor( TValue.From<IVariableV1>(x),DtInvalid, 'x');
                            if TDTypes.as_base_dtype(t_x.dtype) <> base_type then
                                t_x := gen_math_ops.cast(t_x, base_type, name);
                            Result := t_x;
                        end );
end;

class function math_ops.conj(x: TFTensor; name: string): TFTensor;
begin
    var dt := x.dtype;
    if (Tdtypes.is_floating(dt)) or (Tdtypes.is_integer(dt))then
        Exit( x );

    var vValues : TValue := x;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Conj', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                Result :=  v1._values.AsType<TFTensor>;
                                            end );
end;

class function math_ops.cos(x: TFTensor; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('Cos', name, ExecuteOpArgs.Create([x])).FirstOrDefault(nil)
end;

class function math_ops.cumsum<T>(x: TFTensor; axis: T; exclusive, reverse: Boolean; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [x];
    var newVal : TValue := TValue.From< TArray<TValue> >(vValues);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cumsum', @newVal),
                      function(v1: TNameScope): TFTensor
                        begin
                            name := string(v1.ToString);
                            Result := tf.Context.ExecuteOp('Cumsum', name, ExecuteOpArgs.Create([x, TValue.From<T>(axis)])
                                         .SetAttributes(['exclusive',exclusive,'reverse',reverse])).FirstOrDefault(nil);
                        end );
end;

class function math_ops._ReductionDims(x: TFTensor; axis: PAxis): TFTensor;
begin
    if axis <> nil then
    begin
        // should return axis. or check before.
        Result := Tops.convert_to_tensor(axis^, TF_DataType.TF_INT32);
    end else
    begin
        var rank := common_shapes.rank(x);
        // we rely on Range and Rank to do the right thing at run-time.
        if rank = -1 then
        begin
           var pA : TValue :=  array_ops.rank(x);
           Result := range( 0, @pA );
           Exit;
        end;
        var pA : TValue := rank;
        var i  : TValue := 1;
        Result := range(0, @pA, @i);
    end;
end;

end.



