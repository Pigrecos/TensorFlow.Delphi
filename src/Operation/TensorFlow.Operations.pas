unit TensorFlow.Operations;
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
         System.Generics.Collections,
         rtti,

         Spring,
         Spring.Collections,
         Spring.Collections.Enumerable,

         TF4D.Core.CApi,
         TensorFlow.DApiBase,
         TensorFlow.DApi,
         TensorFlow.Core,
         Tensorflow.NnOps,
         Tensorflow.Interfaces,
         Numpy.Axis,

         Keras.Core,

         Tensorflow.Proto;

type

  {$REGION 'tensor_array_ops'}
  tensor_array_ops = record
     public
        /// <summary>
        /// Builds a TensorArray with a new `flow` tensor.
        /// </summary>
        /// <param name="old_ta"></param>
        /// <param name="flow"></param>
        /// <returns></returns>
        class function build_ta_with_new_flow(old_ta: TTensorArray; flow: TFTensor) : TTensorArray; overload; static;
        class function build_ta_with_new_flow(old_ta: TGraphTensorArray; flow: TFTensor) : TTensorArray; overload; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_random_ops'}
  gen_random_ops = record
    private

    public
      /// <summary>
      /// Outputs random values from a normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_standard_normal(shape: TFTensor; dtype: TF_DataType = DtInvalid; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random integers from a uniform distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="minval"></param>
      /// <param name="maxval"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_uniform_int(shape: TFTensor; minval: TFTensor; maxval: TFTensor; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random values from a uniform distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_uniform(shape: TFTensor; dtype: TF_DataType; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      ///
      /// </summary>
      /// <param name="value"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_shuffle(value: TFTensor; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random values from a truncated normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function truncated_normal(shape: TFTensor; dtype: TF_DataType; seed: Integer = 0; seed2: Integer = 0; name: string = '') : TFTensor; static;
      Class function multinomial(logits: TFTensor; num_samples: Integer; seed: Integer = 0; seed2: Integer = 0; output_dtype: TF_DataType = TF_INT64; name: string = '') : TFTensor; static;
      Class function stateless_random_normal_v2(shape: TFTensor; key: TFTensor; counter: TFTensor; alg: Integer; dtype: TF_DataType; name: string = '') : TFTensor; static;
      Class function stateless_random_get_key_counter(seed: TArray<Integer>; name: string = '') : TFTensors; static;
  end;
  {$ENDREGION}

  {$REGION 'random_ops'}
  random_ops = record
    private
        class function _ShapeTensor(shape: TArray<Integer>) : TFTEnsor; static;
        /// <summary>
        /// Implementation for random.categorical (v1) and random.categorical (v2).
        /// </summary>
        /// <param name="logits"></param>
        /// <param name="num_samples"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        class function multinomial_categorical_impl(logits: TFTensor; num_samples: Integer; dtype: TF_DataType = DtInvalid; seed : pInteger= nil) : TFTEnsor; static;
    public
        /// <summary>
        ///
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function random_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        class function random_uniform(shape: TArray<Integer>; minval: Single = 0; maxval: Single = 1; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; overload; static;
        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        class function random_uniform_int(shape: TArray<Integer>; minval: Integer = 0; maxval: Integer = 1; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        class function random_uniform(shape: TFTensor; minval: Integer = 0; maxval: TFTensor = nil; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; overload; static;
        /// <summary>
        /// Randomly shuffles a tensor along its first dimension.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function random_shuffle(value: TFTensor; seed: Integer = 0; name: string = '') : TFTEnsor; static;
        class function truncated_normal(shape: TArray<Integer>; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        class function multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger = nil; name: string = ''; output_dtype: TF_DataType = DtInvalid) : TFTEnsor; static;
  end;
  {$ENDREGION}

  {$REGION 'clip_ops'}
  clip_ops = record
    private

    public
      class function clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor = nil; name: string = ''): Tuple<TFTensors,TFTensor> ; static;
      class function clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string = ''): TFTensor ; static;
      /// <summary>
      /// Computes the global norm of multiple tensors.
      /// </summary>
      /// <param name="t_list"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function global_norm(t_list:  TArray<TFTensor>; name: string = '') : TFTensor ; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_array_ops'}
  gen_array_ops = record
    private
     class function concat_v2_eager_fallback<T1, T2>(values: TArray<T1>; axis: T2; name: string; ctx: TContext) : TFTensor;static;
     class function pad_eager_fallback(inputs: TFTensor; padding: TFTensor; name: string = ''; ctx : TContext= nil): TFTensor; static;
     class function slice_eager_fallback(inputs: TFTensor; _begin: TArray<TFTensor>; size: TArray<TFTensor>; name: string; ctx : TContext) : TFTensor;static;
    public
     class function pack(values: TArray<TFTensor>; axis: Integer  = 0; name: string = ''): TFTensor;static;
     /// <summary>
     /// Creates a tensor filled with a scalar value.
     /// </summary>
     /// <param name="dims">A `Tensor`.</param>
     /// <param name="value">A `Tensor`. 0-D (scalar). Value to fill the returned tensor.</param>
     /// <param name="name">A name for the operation (optional).</param>
     /// <returns>A `Tensor`. Has the same type as `value`.</returns>
     class function fill<T>(dims: TFTensor; value: T; name: string = ''): TFTensor;static;
     class function size(input: TFTensor; out_type: TF_DataType = TF_DataType.TF_INT32; name : string = ''): TFTensor;static;
     class function reshape<T>(tensor: TFTensor; shape: T; name: string = ''): TFTensor;overload; static;
     class function reshape(tensor: TFTensor; shape: TArray<TValue>; name: string = ''): TFTensor;overload; static;
     class function strided_slice(input, tBegin, tEnd, tStrides : TFTensor; begin_mask : Int64 = 0; end_mask: Int64 = 0; ellipsis_mask: Int64 = 0; new_axis_mask: Int64 = 0; shrink_axis_mask: Int64 = 0;name: string = ''): TFTensor; overload ;static;
     class function strided_slice<T>(input: TFTensor; tBegin: TArray<T>; tEnd: TArray<T>; strides: TArray<T>; begin_mask : Integer= 0; end_mask: Integer = 0; ellipsis_mask: Integer = 0; new_axis_mask : Integer= 0; shrink_axis_mask : Integer= 0; name: string = ''): TFTensor;overload; static;
     class function resource_strided_slice_assign(input, tBegin, tEnd, tStrides, tVvalue: TFTensor; begin_mask: Integer = 0; end_mask: Integer= 0; ellipsis_mask: Integer=0;new_axis_mask: Integer=0; shrink_axis_mask: Integer=0; name: string=''): TFTensor;static;
     /// <summary>
     /// Return a tensor with the same shape and contents as the input tensor or value.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="name"></param>
     class function identity(input: TFTensor; name: string = ''): TFTensor; static;
     class function expand_dims(input: TFTensor; axis: integer; name: string = ''): TFTensor; static;
     class function batch_to_space_nd<T>(input: T; block_shape: TArray<Integer>; crops: TArray< TArray<Integer> >; name: string = ''): TFTensor; static;
     class function concat_v2(values: TArray<TFTensor>; axis: Integer; name: string = ''): TFTensor; overload; static;
     /// <summary>
     /// Concatenates tensors along one dimension.
     /// </summary>
     /// <param name="values"></param>
     /// <param name="axis"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function concat_v2<T, Ta>(values: TArray<T>; axis: Ta; name: string = ''): TFTensor; overload; static;
     class function concat_v2(values: TArray<TFTensor>; axis: TFTensor; name: string = ''): TFTensor; overload; static;
     class function shape(input: TFTensor; out_type: TF_DataType = TF_DataType.TF_INT32; name: string = '') : TFTensor; static;
     /// <summary>
     /// Returns shape of tensors.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="out_type"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function shape_n(input: TArray<TFTensor>; out_type: TF_DataType = TF_INT32; name: string = ''): TArray<TFTensor>; static;
     class function where(condition: TFTensor; name : string = ''): TFTensor; static;
     class function select<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string = ''): TFTensor; static;
     class function select_v2<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string = ''): TFTensor; static;
     /// <summary>
     /// Removes dimensions of size 1 from the shape of a tensor.
     /// Given a tensor `input`, this operation returns a tensor of the same type with
     /// all dimensions of size 1 removed.If you don't want to remove all size 1
     /// dimensions, you can remove specific size 1 dimensions by specifying
     /// `axis`.
     /// </summary>
     /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
     /// <param name="axis"> An optional list of `ints`. Defaults to `[]`. If specified, only squeezes the dimensions listed.</param>
     /// <param name="name"> A name for the operation (optional).</param>
     /// <returns> A `Tensor`. Has the same type as `input`.</returns>
     class function squeeze(input: TFTensor; axis: TArray<Integer> = []; name: string = '') : TFTensor; static;
     class function gather_v2<T1, T2>(params: T1; indices: T2; axis: Integer; batch_dims: Integer = 0; name: string = ''): TFTensor; static;
     class function rank(input: TFTensor; name: string = ''): TFTensor; static;
     class function slice<Tb, Ts>(input: TFTensor; _begin: Tb; size: Ts; name: string = ''): TFTensor; overload; static;
     class function slice(input: TFTensor; _begin: TArray<TFTensor>; size: TArray<TFTensor>; name: string = ''): TFTensor; overload; static;
     class function check_numerics(tensor: TFTensor; _message: string; name: string = ''): TFTensor; static;
     class function concat_offset(concat_dim: TFTensor; shape: TArray<TFTensor>; name: string = ''): TArray<TFTensor>; static;
     /// <summary>
     ///    Returns a diagonal tensor with a given diagonal values.
     /// </summary>
     /// <param name="diagonal">
     ///    Rank k tensor where k is at most 1.
     /// </param>
     /// <param name="name">
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Diag'.
     /// </param>
     /// <returns>
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///    Given a <c>diagonal</c>, this operation returns a tensor with the <c>diagonal</c> and
     ///    everything else padded with zeros. The diagonal is computed as follows:
     ///
     ///    Assume <c>diagonal</c> has dimensions [D1,..., Dk], then the output is a tensor of
     ///    rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
     ///
     ///    <c>output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]</c> and 0 everywhere else.
     ///
     ///    For example:
     ///
     ///   <code>
     ///    # 'diagonal' is [1, 2, 3, 4]
     ///    tf.diag(diagonal) ==&amp;gt; [[1, 0, 0, 0]
     ///    [0, 2, 0, 0]
     ///    [0, 0, 3, 0]
     ///    [0, 0, 0, 4]]
     ///   </code>
     /// </remarks>
     class function diag(diagonal: TFTensor; name: string = ''): TFTensor; static;
     class function diag_part(diagonal: TFTensor; name: string = '') : TFTensor; static;
     class function pad(input: TFTensor; paddings: TFTensor; name: string = ''): TFTensor; static;
     class function invert_permutation(x: TFTensor; name: string = ''): TFTensor; static;
     class function log(x: TFTensor; name: string = ''): TFTensor; static;
     /// <summary>
     /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
     /// </summary>
     /// <param name="s0">A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
     /// <param name="s1">A `Tensor`. Must have the same type as `s0`.</param>
     /// <param name="name">A name for the operation (optional).</param>
     /// <returns>A tuple of `Tensor` objects (r0, r1).</returns>
     class function broadcast_gradient_args(s0: TFTensor; s1: TFTensor; name: string = '') : Tuple<TFTensor,TFTensor>; static;
     class function reverse<T>(tensor: TFTensor; axis: T; name: string = ''): TFTensor; static;
     /// <summary>
     /// Finds unique elements in a 1-D tensor.
     /// </summary>
     /// <param name="x"></param>
     /// <param name="out_idx"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function unique(x: TFTensor; out_idx: TF_DataType = TF_INT32; name: string = '') : Tuple<TFTensor,TFTensor>; static;
     class function unpack(value: TFTensor; num: Integer; axis: Integer = 0; name: string = ''): TArray<TFTensor>; static;
     class function one_hot(indices: TFTensor; depth: TFTensor; on_value: TFTensor = nil; off_value: TFTensor = nil; dtype: TF_DataType = DtInvalid; axis: Integer = -1; name: string = ''): TFTensor; static;
     /// <summary>
     /// A placeholder op that passes through `input` when its output is not fed.
     /// </summary>
     /// <param name="input">The default value to produce when output is not fed.</param>
     /// <param name="shape"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function placeholder_with_default<T>(input: T; shape: TArray<Integer>; name: string = ''): TFTensor; static;
     class function scatter_nd(indices: TFTensor; updates: TFTensor; shape: TArray<TFTensor>; name: string = ''): TFTensor; static;
     class function split_v(value: TFTensor; size_splits: TFTensor; axis: Integer; num_split: Integer; name: string = ''): TArray<TFTensor>; static;
     class function tile(input: TFTensor; multiples: TFTensor; name: string = ''): TFTensor; overload;static;
     class function tile(input: TFTensor; multiples: TArray<TValue>; name: string = ''): TFTensor;overload; static;
     class function transpose<T1>(x: TFTensor; perm: T1; name: string = ''): TFTensor; static;
     class function ones_like(x: TFTensor; name: string = ''): TFTensor; static;
     class function zeros_like(x: TFTensor; name: string = ''): TFTensor; static;
     class function stop_gradient(x: TFTensor; name: string = ''): TFTensor; static;
     /// <summary>
     /// Return the shape of s0 op s1 with broadcast.
     /// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
     /// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
     /// </summary>
     /// <param name="s0"> A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
     /// <param name="s1"> A `Tensor`. Must have the same type as `s0`.</param>
     /// <param name="name"> A name for the operation (optional).</param>
     /// <returns> `Tensor`. Has the same type as `s0`.</returns>
     class function broadcast_args(s0: TFTensor; s1: TFTensor; name: string = ''): TFTensor; static;
     /// <summary>
     /// Broadcast an array for a compatible shape.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="shape"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function broadcast_to<T>(input: TFTensor; shape: T; name: string = ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'array_ops'}
  array_ops = record
   private
     class function _get_dtype_from_nested_lists(list_or_tuple: TArray<TValue>): TF_DataType; overload ;static;
     class function _get_dtype_from_nested_lists<T>(list_or_tuple: TArray<T>): TF_DataType; overload ;static;
     class function _constant_if_small<T>(value: T; shape: TFShape; dtype: TF_DataType; name: string): TFTensor; overload; static;
     class function size_internal<T>(input: T; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor;static;
     class function _apply_mask_1d(reshaped_tensor: TFTensor; mask: TFTensor; axis: Integer = 0): TFTensor;static;
     class function rank_internal(input: TFTensor; name: string = ''; optimize: Boolean = true) : TFTensor; static;
     class function split_eager_fallback<Ta, Tv>(axis: Ta; value: Tv; num_split: Integer; name: string; ctx: TContext= nil): TArray<TFTensor>; static;
   public
     class function constant(value: TValue; dtype : TF_DataType= DtInvalid; shape: TArray<Integer>= nil; name : AnsiString = 'Const'; verify_shape : Boolean = false):TFTensor; static;

     class function placeholder(_dtype: TF_DataType; _shape : PTFShape= nil; name: string = '') : TFTensor;static;
     /// <summary>
     /// Converts the given list or tuple to a tensor by packing.
     /// </summary>
     /// <param name="list_or_tuple">A (possibly nested) list or tuple containing a tensor.</param>
     /// <param name="dtype"></param>
     /// <param name="name"></param>
     /// <returns>A `tf.Tensor` with value equivalent to `list_or_tuple`.</returns>
     class function _autopacking_helper(list_or_tuple: TArray<TValue>; dtype: TF_DataType; name: string):TFTensor;static;
     class function _autopacking_conversion_function(v:TArray<TValue>; dtype : TF_DataType = DtInvalid; name: string=''; as_ref : Boolean= false): TFTensor;static;
     class function ones(shape: TFTensor; dtype: TF_DataType = TF_FLOAT; name : string= ''): TFTensor; overload; static;
     class function ones(shape: TArray<TFTensor>; dtype: TF_DataType = TF_FLOAT; name : string= ''): TFTensor; overload; static;
     class function ones(shape: TFShape; dtype: TF_DataType = TF_DataType.TF_FLOAT; name: string = ''): TFTensor; overload; static;
     class function zeros(shape: TFTensor; dtype: TF_DataType = TF_DataType.TF_FLOAT; name : string = '') : TFTensor; overload; static;
     class function zeros(shape: TFShape; dtype: TF_DataType = TF_DataType.TF_FLOAT; name : string = ''): TFTensor; overload; static;
     class function size<T>(input: T; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32):TFTensor;static;
     class function stack(values: TArray<TFTensor>; axis: Integer = 0; name: string = 'stack'):TFTensor;overload;static;
     class function stack(values: TValue; axis: Integer = 0; name: string = 'stack'):TFTensor;overload;static;
     class function unstack(value: TFTensor; num: PInteger = nil; axis: Integer = 0; name: string = 'unstack') : TArray<TFTensor>; static;
     class function identity(input: TFTensor; name: String = ''): TFTensor; static;
     class function expand_dims(input: TFTensor; axis: Integer = -1; name: string = ''): TFTensor; static;
     class function boolean_mask<T1, T2>(tensor: T1; mask: T2; name: string = 'boolean_mask'; axis: Integer = 0): TFTensor; static;
     class function shape_internal(input: TFTensor; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor; static;
     class function reshape(tensor: TFTensor; shape: TFTensor;       name: string = ''): TFTensor; overload; static;
     class function reshape(tensor: TFTensor; shape: TFShape;        name: string = ''): TFTensor; overload; static;
     class function reshape(tensor: TFTensor; shape: TArray<TValue>; name: string = ''): TFTensor; overload; static;
     /// <summary>
     /// Returns the shape of a tensor.
     /// </summary>
     /// <param name="input">A `Tensor` or `SparseTensor`.</param>
     /// <param name="name">A name for the operation (optional).</param>
     /// <param name="out_type">
     /// (Optional) The specified output type of the operation
     /// (`int32` or `int64`). Defaults to `tf.int32`.
     /// </param>
     /// <returns>A `Tensor` of type `out_type`.</returns>
     class function shape(input: TFTensor; name : string= ''; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor; static;
     class function shape_v2(input: TFTensor; name: string = ''; out_type: TF_DataType = TF_INT32): TFTensor; static;
     /// <summary>
     /// Concatenates tensors along one dimension.
     /// </summary>
     /// <param name="values"></param>
     /// <param name="axis"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function concat(values: TArray<TFTensor>; axis: Integer; name: string = 'concat'): TFTensor; overload ;static;
     class function concat(values: TArray<TFTensor>; axis: TFTensor; name: string = 'concat'): TFTensor; overload ;static;
     class function concat(values: TArray<TValue>; axis: Integer; name: string = 'concat'): TFTensor; overload ;static;
     class function where(condition: TFTensor; x: TObject = nil; y: TObject = nil; name: string = ''): TFTensor; static;
     class function where_v2(condition: TFTensor; x: TObject = nil; y: TObject= nil; name: string= ''): TFTensor; static;
     /// <summary>
     /// Removes dimensions of size 1 from the shape of a tensor.
     /// Given a tensor `input`, this operation returns a tensor of the same type with
     /// all dimensions of size 1 removed.If you don't want to remove all size 1
     /// dimensions, you can remove specific size 1 dimensions by specifying
     /// `axis`.
     /// </summary>
     /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
     /// <param name="axis"> An optional list of `ints`. Defaults to `[]`.
     /// If specified, only squeezes the dimensions listed.The dimension
     /// index starts at 0. It is an error to squeeze a dimension that is not 1.
     /// Must be in the range `[-rank(input), rank(input))`.</param>
     /// <param name="name"> A name for the operation (optional).</param>
     /// <param name="squeeze_dims" >Deprecated keyword argument that is now axis.</param>
     /// <returns>A `Tensor`. Has the same type as `input`.
     /// Contains the same data as `input`, but has one or more dimensions of
     /// size 1 removed.</returns>
     class function squeeze(input: TFTensor; axis: TArray<Integer> = []; name: string = ''): TFTensor; static;
     /// <summary>
     /// Gather slices from `params` according to `indices`. `indices` must be an integer tensor of any dimension(often 1-D).
     /// </summary>
     /// <typeparam name="T1">Element type of the indexed tensor.</typeparam>
     /// <typeparam name="T2">Element type of the index tensor.</typeparam>
     /// <param name="params">The `Tensor` from which to gather values. Must be at least rank `axis + 1`.</param>
     /// <param name="indices">The index `Tensor`.  Must be one of the following types: `int32`, `int64`. The values must be in range `[0, params.shape[axis])`.</param>
     /// <param name="name">A name for the operation (optional).</param>
     /// <param name="axis">
     /// A `Tensor`. Must be one of the following types: `int32`, `int64`.
     /// The `axis` in `params` to gather `indices` from.Must be greater than or equal to `batch_dims`.
     /// Defaults to the first non-batch dimension. Supports negative indexes.
     /// </param>
     /// <param name="batch_dims">An integer. The number of batch dimensions. Must be less than or equal to rank(indices).</param>
     /// <returns></returns>
     class function gather<T1, T2>(params:T1; indices: T2; name: string = ''; axis: Integer = 0; batch_dims: Integer = 0): TFTensor; static;
     /// <summary>
     /// Returns the rank of a tensor.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function rank(input: TFTensor; name: string = ''): TFTensor; static;
     class function slice<Tb, Ts>(input: TFTensor; _begin: Tb; size: Ts; name: string = ''): TFTensor; overload; static;
     class function slice(input: TFTensor; _begin: TArray<TFTensor>; size: TArray<TFTensor>; name: string = ''): TFTensor; overload; static;
     class function slice(input: TFTensor; _begin: TFTensor; size: TFTensor; name: string = ''): TFTensor; overload; static;

     /// <summary>
     /// Creates a tensor filled with a scalar value.
     /// This operation creates a tensor of shape `dims` and fills it with `value`.
     /// </summary>
     /// <param name="dims">A 1-D sequence of non-negative numbers.</param>
     /// <param name="value">A value to fill the returned `tf.Tensor`.</param>
     /// <param name="name">Optional string. The name of the output `tf.Tensor`.</param>
     /// <returns>A `tf.Tensor` with shape `dims` and the same dtype as `value`.</returns>
     class function fill<T>(dims: TFShape; value: T; name: string = ''): TFTensor; static;
     /// <summary>
     /// Creates a tensor with all elements set to 1.
     /// </summary>
     /// <param name="tensor"></param>
     /// <param name="dtype"></param>
     /// <param name="name"></param>
     /// <param name="optimize"></param>
     /// <returns></returns>
     class function ones_like(tensor: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''; optimize: Boolean = true): TFTensor; static;
     class function one_hot(indices: TFTensor; depth: TFTensor; on_value: TFTensor = nil; off_value: TFTensor = nil; dtype: TF_DataType = DtInvalid; axis: Integer = -1; name: string = ''): TFTensor; static;
     class function unique(x: TFTensor; out_idx: TF_DataType = TF_INT32; name: string = ''): Tuple<TFTensor, TFTensor>; static;
     class function tile(input: TFTensor; multiples: TFTensor; name: string = '') : TFTensor; static;
     class function zeros_like(tensor: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''; optimize: Boolean = true) : TFTensor; static;
     /// <summary>
     ///   When building ops to compute gradients, this op prevents the contribution of
     ///   its inputs to be taken into account.Normally, the gradient generator adds ops
     ///   to a graph to compute the derivatives of a specified 'loss' by recursively
     ///   finding out inputs that contributed to its computation.If you insert this op
     ///   in the graph it inputs are masked from the gradient generator.  They are not
     ///   taken into account for computing gradients.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="name"></param>
     /// <returns></returns>
    class function stop_gradient(input: TFTensor; name: string = ''): TFTensor; static;
     /// <summary>
     /// Extracts a strided slice of a tensor (generalized python array indexing).
     /// </summary>
     /// <param name="input_"></param>
     /// <param name="begin"></param>
     /// <param name="end"></param>
     /// <param name="strides"></param>
     /// <param name="begin_mask"></param>
     /// <param name="end_mask"></param>
     /// <param name="ellipsis_mask"></param>
     /// <param name="new_axis_mask"></param>
     /// <param name="shrink_axis_mask"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function strided_slice(input_: TFTensor; _begin: TFTensor; _end: TFTensor; strides: TFTensor = nil; begin_mask: Integer = 0; end_mask: Integer = 0; ellipsis_mask : Integer = 0; new_axis_mask: Integer = 0; shrink_axis_mask: Integer = 0; name: string = ''): TFTensor; static;
     /// <summary>
     /// Returns the gradient of `StridedSlice`.
     ///
     /// Since `StridedSlice` cuts out pieces of its `input` which is size
     /// `shape`, its gradient will have the same shape (which is passed here
     /// as `shape`). The gradient will be zero in any element that the slice
     /// does not select.
     /// </summary>
     /// <param name="shape">Must be one of the following types: `int32`, `int64`.</param>
     /// <param name="begin">Must have the same type as `shape`.</param>
     /// <param name="end">Must have the same type as `shape`.</param>
     /// <param name="strides">Must have the same type as `shape`.</param>
     /// <param name="dy">A `Tensor`.</param>
     /// <param name="begin_mask">An optional `int`. Defaults to `0`.</param>
     /// <param name="end_mask">An optional `int`. Defaults to `0`.</param>
     /// <param name="ellipsis_mask">An optional `int`. Defaults to `0`.</param>
     /// <param name="new_axis_mask">An optional `int`. Defaults to `0`.</param>
     /// <param name="shrink_axis_mask">An optional `int`. Defaults to `0`.</param>
     /// <param name="name">A name for the operation (optional).</param>
     /// <returns>A `Tensor`. Has the same type as `dy`.</returns>
     class function strided_slice_grad(shape: TFTensor; _begin: TFTensor; _end: TFTensor; strides: TFTensor; dy: TFTensor; begin_mask : Int64= 0; end_mask: Int64 = 0; ellipsis_mask: Int64 = 0; new_axis_mask: Int64 = 0; shrink_axis_mask: Int64 = 0; name: string = ''): TFTensor; static;
     class function invert_permutation(x: TFTensor; name: string = ''): TFTensor; static;
     class function matrix_diag(diagonal: TFTensor; name: string = 'diag'; k: Integer = 0; num_rows: Integer = -1; num_cols: Integer = -1; padding_value : Single = 0; align: string = 'RIGHT_LEFT'): TFTensor; static;
     class function matrix_set_diag(input: TFTensor; diagonal: TFTensor; name : string = 'set_diag'; k: Integer = 0; align: string = 'RIGHT_LEFT') : TFTensor; static;
     class function meshgrid<T>(_array: TArray<T>; copy: Boolean = true; sparse: Boolean = false; indexing: string = 'xy'): TArray<TFTensor>; static;
     class function moveaxis(_array: TNDArray; source: TAxis; destination: TAxis): TFTensor; static;
     /// <summary>
     /// Computes the shape of a broadcast given symbolic shapes.
     /// When shape_x and shape_y are Tensors representing shapes(i.e.the result of
     /// calling tf.shape on another Tensor) this computes a Tensor which is the shape
     /// of the result of a broadcasting op applied in tensors of shapes shape_x and
     /// shape_y.
     /// For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
     /// Tensor whose value is [5, 2, 3].
     /// This is useful when validating the result of a broadcasting operation when the
     /// tensors do not have statically known shapes.
     /// </summary>
     /// <param name="shape_x"> A rank 1 integer `Tensor`, representing the shape of x.</param>
     /// <param name="shape_y"> A rank 1 integer `Tensor`, representing the shape of y.</param>
     /// <returns> A rank 1 integer `Tensor` representing the broadcasted shape.</returns>
     class function broadcast_dynamic_shape(shape_x: TFTensor; shape_y: TFTensor): TFTensor; static;
     class function broadcast_static_shape(shape_x : TFTensor; shape_y: TFTensor): TFTensor; static;
     class function transpose<T1>(a: T1; perm: PAxis; name: string = 'transpose'; conjugate: Boolean = false): TFTensor; overload; static;
     class function transpose(a: TFTensor; perm: TFTensor; name: string = 'transpose'; conjugate : Boolean= false): TFTensor; overload; static;
     class function split(value: TFTensor; size_splits: TFTensor; axis: Integer; num: Integer = -1; name: string = 'split'): TArray<TFTensor>; overload; static;
     class function split<T>(value: TFTensor; num_split: Integer; axis: T; name: string = 'split') : TArray<TFTensor>; overload; static;
     class function pad(tensor: TFTensor; paddings: TFTensor; mode: string = 'CONSTANT'; name : string= ''; constant_values: Integer = 0): TFTensor; static;
     /// <summary>
     ///    An identity op that triggers an error if a gradient is requested.
     /// </summary>
     /// <param name="input">
     ///    any tensor.
     /// </param>
     /// <param name="name">
     /// If specified, the created operation in the graph will be this one, otherwise it will be named 'PreventGradient'.
     /// </param>
     /// <param name="message">
     ///    Will be printed in the error when anyone tries to differentiate
     ///    this operation.
     /// </param>
     /// <returns>
     ///    the same input tensor.
     ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
     /// </returns>
     /// <remarks>
     ///    When executed in a graph, this op outputs its input tensor as-is.
     ///
     ///    When building ops to compute gradients, the TensorFlow gradient system
     ///    will return an error when trying to lookup the gradient of this op,
     ///    because no gradient must ever be registered for this function.  This
     ///    op exists to prevent subtle bugs from silently returning unimplemented
     ///    gradients in some corner cases.
     /// </remarks>
     class function prevent_gradient(input: TFTensor; msg: string = ''; name: string = '') : TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_math_ops'}
  gen_math_ops = record
    private

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
  {$ENDREGION}

  {$REGION 'gen_data_flow_ops'}
  gen_data_flow_ops = record
    private

    public
      class function dynamic_stitch(indices: TArray<TFTensor>; data: TArray<TFTensor>; name: string = ''): TFTensor; static;
      class function tensor_array_size_v3(handle: TFTensor; flow_in: TFTensor; name: string = ''): TFTensor; static;
      class function tensor_array_gather_v3(handle: TFTensor; indices: TFTensor; flow_in: TFTensor; dtype: TF_DataType; element_shape : PTFShape= nil; name: string = ''): TFTensor; static;
      class function tensor_array_v3<T>(size: T; dtype: TF_DataType = DtInvalid; element_shape : PTFShape= nil; dynamic_size: Boolean = false; clear_after_read: Boolean = true;
                                        identical_element_shapes : Boolean= false; tensor_array_name: string = ''; name: string = '') : Tuple<TFTensor, TFTensor>; static;
      /// <summary>
      /// Read an element from the TensorArray into output `value`.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="index"></param>
      /// <param name="flow_in"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function tensor_array_read_v3(handle: TFTensor; index: TFTensor; flow_in: TFTensor; dtype: TF_DataType; name: string = ''): TFTensor; static;
      class function tensor_array_write_v3(handle: TFTensor; index: TFTensor; value: TFTensor; flow_in: TFTensor; name : string= ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'math_ops'}
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
      class function count_nonzero_v2(input: TFTensor; axis: TAxis; keepdims : Boolean= false; name: string = ''; dtype : TF_DataType = TF_INT64): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_nn_ops'}
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
  {$ENDREGION}

  {$REGION 'nn_ops'}
  nn_ops = record
    private
      class function _get_noise_shape(x: TFTensor; noise_shape: TFTensor): TFTensor; static;
      /// <summary>
      /// Flattens logits' outer dimensions and keep its last dimension.
      /// </summary>
      /// <param name="logits"></param>
      /// <returns></returns>
      class function _flatten_outer_dims(logits: TFTensor) : TFTensor; static;
    public
      class function top_kv2(input: TFTensor; k: Integer; sorted: Boolean = true; name: string = ''): TFTensors; static;
      class function convolution_internal(padding: string; strides: TArray<Integer>; dilation_rate: TArray<Integer>; rank: Integer; name: string = ''; data_format: string = ''):  ConvolutionInternal; static;
      class function l2_loss(t: TFTensor; name: string = ''): TFTensor; static;
      class function softplus(features: TFTensor; name: string = ''): TFTensor; static;
      /// <summary>
      /// Adds `bias` to `value`.
      /// </summary>
      /// <param name="value"></param>
      /// <param name="bias"></param>
      /// <param name="data_format"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function bias_add(value: TFTensor; bias: IVariableV1; data_format: string = ''; name: string = ''): TFTensor; static;
      /// <summary>
      /// Computes dropout.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="rate"></param>
      /// <param name="noise_shape"></param>
      /// <param name="seed"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function dropout_v2(x: TFTensor; rate: TFTensor; noise_shape: TFTensor = nil; seed: pInteger = nil; name: string = '') : TFTensor; static;
      class function in_top_k(predictions: TFTensor; targets: TFTensor; k: Integer; name: string = '') : TFTensor; static;
      class function log_softmax(logits: TFTensor; axis: Integer = -1; name:string = ''): TFTensor; static;
      class function softmax(logits: TFTensor; axis: Integer = -1; name: string = ''): TFTensor; static;
      class function leaky_relu(features: TFTensor; alpha: Single = 0.2; name: string = ''): TFTensor; static;
      class function max_pool(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor; static;
      class function _softmax(logits: TFTensor; compute_op: TFunc<TFTensor, string, TFTensor>; dim: Integer = -1; name: string = '') : TFTensor; static;
      /// <summary>
      /// Computes sparse softmax cross entropy between `logits` and `labels`.
      /// </summary>
      /// <param name="labels"></param>
      /// <param name="logits"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function sparse_softmax_cross_entropy_with_logits(labels: TFTensor = nil; logits: TFTensor = nil; name: string = '') : TFTensor; static;
      class function softmax_cross_entropy_with_logits_v2_helper(labels: TFTensor; logits: TFTensor; axis : Integer= -1; name : string= ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'nn_impl'}
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
      class function batch_normalization(const x: TFTensor; const mean: TFTensor; const variance: TFTensor; const offset: TFTensor; const scale: TFTensor; variance_epsilon: Single = 0.001; name: string = ''): TFTensor; static;
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
  {$ENDREGION}

  {$REGION 'bitwise_ops'}
  /// <summary>
  /// Operations for bitwise manipulation of integers.
  /// https://www.tensorflow.org/api_docs/python/tf/bitwise
  /// </summary>
  bitwise_ops = class
     private
        /// <summary>
        /// Helper method to invoke unary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function unary_op(x: TFTensor; opName: string; name: string) : TFTensor;
        /// <summary>
        /// Helper method to invoke binary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function binary_op(x: TFTensor; y: TFTensor; opName: string; name: string)  : TFTensor;
     public
        /// <summary>
        /// Elementwise computes the bitwise left-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/left_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function left_shift(x: TFTensor; y: TFTensor; name: string = '') : TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise right-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/right_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function right_shift(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise inversion of `x`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/invert
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function invert(x: TFTensor; name: string = '') : TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise AND of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_and
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_and(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise OR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_or
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_or(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise XOR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_xor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_xor(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
  end;
  {$ENDREGION}

  {$REGION 'string_ops'}
  string_ops = Class
     private

     public
        function lower(input: TFTensor; encoding : string = ''; name: String = ''): TFTensor;
        function regex_replace(input: TFTensor; pattern: string; rewrite: string; replace_global: Boolean = true; name: string = ''): TFTensor;
        /// <summary>
        /// Return substrings from `Tensor` of strings.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="pos"></param>
        /// <param name="len"></param>
        /// <param name="name"></param>
        /// <param name="uint"></param>
        /// <returns></returns>
        function substr<T>(input: T; pos: Integer; len: Integer; &uint: string = 'BYTE'; name: string = ''): TFTensor;
        /// <summary>
        /// Computes the length of each string given in the input tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <param name="unit"></param>
        /// <returns></returns>
        function string_length(input: TFTensor; name: string = ''; &unit: string = 'BYTE'): TFTensor;
        function string_format(inputs: TArray<TFTensor>; template: string = '%s'; placeholder: string = '%s'; summarize: Integer = 3; name: string = ''): TFTensor;
        function string_split_v2(input: TFTensor; sep: string = ' '; maxsplit: Integer = -1; name: string = ''):  RaggedTensor;
        function unicode_decode_with_offsets(input: TFTensor; input_encoding: string; errors: string; replacement_char: Integer = $FFFD; replace_control_characters: Boolean = false; name: string = ''): Tuple<RaggedTensor, RaggedTensor>;
        function _unicode_decode(input: TFTensor; input_encoding: string; errors: string; replacement_char: Integer; replace_control_characters: Boolean; with_offsets: Boolean; name: string = ''): Tuple<RaggedTensor, RaggedTensor>;
  end;
  {$ENDREGION}

  {$REGION 'ControlFlowState'}
  /// <summary>
  /// The state used for constructing the gradient graph for a while loop.
  /// </summary>
  GradLoopState_helper  = class helper for GradLoopState
     private

     public
  end;

  /// <summary>
  /// Maintain the mapping from the loops to their grad states.
  /// </summary>
  ControlFlowState  = class
     private
       Fmap : TDictionary<TControlFlowContext, GradLoopState>;
     public
       constructor Create;

       function  ProcessUnusedLoopExits(pending_count: TDictionary<string, Integer>; to_ops_set: TList<TFOperation>) : TArray<TFTensor>;
       procedure EnterGradWhileContext(op: TFOperation; before: Boolean);
       procedure ExitGradWhileContext(op: TFOperation; before: Boolean);
       //  def ZerosLikeForExit(self, val):
       //    """Create zeros_like gradient for a loop exit.
       //    If the result of a loop variable is not used but is involved in
       //    computing the result of some needed loop variable, we create a
       //    zero-valued tensor that is fed as gradient for the Exit node of that
       //    loop variable. Note that val.op is an Exit, and this method must be
       //    called in the control flow context where gradients() is called.
       //    Args:
       //      val: The output tensor of an Exit op.
       //    Returns:
       //      A zero tensor of the same shape of val.
       //    """
       //    val_shape = val.get_shape()
       //    forward_ctxt = val.op._get_control_flow_context()
       //    outer_forward_ctxt = forward_ctxt.outer_context
       //    if outer_forward_ctxt:
       //      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
       //    outer_grad_state = None
       //    if outer_forward_ctxt:
       //      outer_grad_state = self._map.get(outer_forward_ctxt)
       //    if outer_grad_state:
       //      # This is a nested loop.
       //      if val_shape.is_fully_defined():
       //        # If the shape is known statically, just create a zero tensor
       //        # with the right shape in the right context.
       //        outer_grad_state.grad_context.Enter()
       //        result = array_ops.zeros(val_shape.dims, val.dtype)
       //        outer_grad_state.grad_context.Exit()
       //      else:
       //        # Only the shape of value is needed for backprop.
       //        forward_ctxt.outer_context.Enter()
       //        shape = array_ops.shape_internal(val, optimize=False)
       //        forward_ctxt.outer_context.Exit()
       //        # Save the shape to a stack.
       //        history_shape = outer_grad_state.AddForwardAccumulator(shape)
       //        # Get the shape back from the stack.
       //        outer_grad_ctxt = outer_grad_state.grad_context
       //        outer_grad_ctxt.Enter()
       //        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
       //            history_shape, shape)
       //        result = array_ops.zeros(real_shape, val.dtype)
       //        outer_grad_ctxt.Exit()
       //    else:
       //      # This is not a nested loop.
       //      if val_shape.is_fully_defined():
       //        # If the shape is known statically, just create a zero tensor
       //        # with the right shape.
       //        result = array_ops.zeros(val_shape.dims, val.dtype)
       //      else:
       //        result = array_ops.zeros_like(val, optimize=False)
       //    return result
       function ZerosLike(op: TFOperation; index: Integer): TFTensor;
       /// <summary>
       /// Create zeros_like gradient for a loop exit.
       /// </summary>
       /// <param name="val"></param>
       /// <returns></returns>
       function ZerosLikeForExit(val: TFTensor): TFTensor;
       function ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
       procedure PostProcessing;
       //  def AddWhileContext(self, op, between_op_list, between_ops):
       //    """Add the grad state for the while loop that op belongs to.
       //    Note that op is an Exit, and this method must be called in
       //    the control flow context where gradients() is called.
       //    Note that this method modifies `between_op_list` and `between_ops`.
       //    """
       //    forward_ctxt = _GetWhileContext(op)
       //    grad_state = self._map.get(forward_ctxt)
       //    if grad_state is None:
       //      # This is a new while loop so create a grad state for it.
       //      outer_forward_ctxt = forward_ctxt.outer_context
       //      if outer_forward_ctxt:
       //        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
       //      outer_grad_state = None
       //      if outer_forward_ctxt:
       //        outer_grad_state = self._map.get(outer_forward_ctxt)
       //      grad_state = GradLoopState(forward_ctxt, outer_grad_state)
       //      self._map[forward_ctxt] = grad_state
       //      # We need to include all exits of a loop for backprop.
       //      for loop_exit in grad_state.forward_loop_exits:
       //        if loop_exit.op not in between_ops:
       //          between_ops.add(loop_exit.op)
       //          between_op_list.append(loop_exit.op)
       procedure AddWhileContext(op: TFOperation; between_op_list: TList<TFOperation>; between_ops: TList<TFOperation>);
       /// <summary>
       /// Return the grad state for this op if it's in a forward loop context.
       /// </summary>
       /// <param name="op"></param>
       /// <param name="before"></param>
       /// <returns></returns>
       function GetGradState(op: TFOperation; before: Boolean): GradLoopState;
  end;
  {$ENDREGION}

  {$REGION 'control_flow_ops'}
  MergeOutput = record
    private
       function GetItem(idx: Integer): TFTensor;
    public
       output      : TFTensor;
       value_index : TFTensor;

       constructor Create(values: TArray<TFTensor>);
       class operator implicit(merge: MergeOutput): TFTensor;

       property item[idx: Integer] : TFTensor read GetItem; default;
  end;

  control_flow_ops = record
     private
       class function _GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string = ''): TFOperation;static;
     public
       class function _convert_flows_to_tensorarrays(tensors_or_tensorarrays: TArray<ITensorOrTensorArray>; tensors_or_flows: TArray<TFTensor>): TArray<ITensorOrTensorArray>; static;
       class function tuple(tensors: TArray<TFTensor>; name: string = ''; control_inputs : TArray<TFOperation> = nil) : TArray<TFTensor>; static;
       class function group<T:  ITensorOrOperation>(inputs: TArray<T>; name : string= '') : TFOperation; static;
       /// <summary>
       /// Returns the value of an available element of `inputs`.
       ///
       /// This op tests each of the tensors in `inputs` in turn to determine if any of
       /// them is available.If it finds an available tensor, it returns it and its
       /// index in `inputs`.
       ///
       /// It is an error if more than one tensor in `inputs` is available.If no tensor
       /// in `inputs` is available, the returned tensor and index are not set.
       ///
       /// This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
       /// `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
       /// before merging.
       /// </summary>
       /// <param name="inputs">inputs: The input tensors, at most one of which is available.</param>
       /// <param name="name">A name for this operation (optional).</param>
       /// <returns></returns>
       class function merge(inputs: TArray<TFTensor>; name: string = ''): MergeOutput; static;
       ///  <summary>
       ///  Forwards `data` to an output determined by `pred`.
       ///  If `pred` is false, the `data` input is forwarded to the first output.
       ///  Otherwise, the data goes to the second output.
       ///
       ///  This op handles `Tensor`s and `IndexedSlices`.
       ///  </summary>
       ///  <param name="data">The tensor to be forwarded to the appropriate output.</param>
       ///  <param name="pred">A scalar that specifies which output port will receive data.</param>
       /// <param name="name"> A name for this operation (optional).</param>
       /// <returns>
       ///  `(output_false, output_true)`: If `pred` is true, data will be forwarded to
       /// `output_true`, otherwise it goes to `output_false`.
       /// </returns>
       class function _SwitchRefOrTensor(data: TFTensor; pred: TFTensor; name: string = 'Switch'): TArray<TFTensor>; static;
       /// <summary>
       /// Produces the content of `output_tensor` only after `dependencies`.
       ///
       /// In some cases, a user may want the output of an operation to be
       /// consumed externally only after some other dependencies have run
       /// first.This function ensures returns `output_tensor`, but only after all
       /// operations in `dependencies` have run.Note that this means that there is
       /// no guarantee that `output_tensor` will be evaluated after any `dependencies`
       /// have run.
       ///
       /// See also `tf.tuple` and `tf.group`.
       /// </summary>
       /// <param name="dependencies">Iterable of operations to run before this op finishes.</param>
       /// <param name="output_tensor">A `Tensor` or `IndexedSlices` that will be returned.</param>
       /// <param name="name">(Optional) A name for this operation.</param>
       /// <returns>Same as `output_tensor`.</returns>
       class function with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Does nothing. Only useful as a placeholder for control edges.
       /// </summary>
       /// <param name="name"></param>
       /// <returns></returns>
       class function no_op(name : string= ''): TFOperation; static;
       class function _Identity(data: TFTensor;  name : string = ''): TFTensor; static;
       class function ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor ; static;
       /// <summary>
       /// Forwards `data` to an output determined by `pred`.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="pred"></param>
       /// <param name="dtype"></param>
       /// <param name="name"></param>
       class function switch(data: TFTensor; pred: TFTensor; dtype : TF_DataType = DtInvalid; name: string = ''): TArray<TFTensor>; static;
       /// <summary>
       /// Create the state for all the while loops involved in one gradients().
       /// </summary>
       /// <param name="between_op_list"></param>
       /// <param name="between_ops"></param>
       /// <param name="colocate_gradients_with_ops"></param>
       class function MaybeCreateControlFlowState(between_op_list: TList<TFOperation>; between_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean) : ControlFlowState;Static;
       class function IsLoopExit(op: TFOperation): Boolean; static;
       class function  _NextIteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
       ///
       /// `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
       /// `false_fn` must have the same non-zero number and type of outputs.
       ///
       /// **WARNING**: Any Tensors or Operations created outside of `true_fn` and
       /// `false_fn` will be executed regardless of which branch is selected at runtime.
       ///
       /// Although this behavior is consistent with the dataflow model of TensorFlow,
       /// it has frequently surprised users who expected a lazier semantics.
       /// Consider the following simple program:
       ///
       /// z = tf.multiply(a, b)
       /// result = tf.cond(x &lt; y, ()=> tf.add(x, z), ()=> tf.square(y))
       ///
       /// If `x&lt;y`, the `tf.add` operation will be executed and `tf.square`
       /// operation will not be executed.Since `z` is needed for at least one
       /// branch of the `cond`, the `tf.multiply` operation is always executed,
       /// unconditionally.
       ///
       /// Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
       /// call to `cond`, and not at all during `Session.run()`). `cond`
       /// stitches together the graph fragments created during the `true_fn` and
       /// `false_fn` calls with some additional graph nodes to ensure that the right
       /// branch gets executed depending on the value of `pred`.
       ///
       /// `tf.cond` supports nested structures as implemented in
       /// `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
       /// same(possibly nested) value structure of lists, tuples, and/or named tuples.
       /// Singleton lists and tuples form the only exceptions to this: when returned by
       /// `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.
       /// This behavior is disabled by passing `strict= True`.
       /// </summary>
       /// <param name="pred"> A scalar determining whether to return the result of `true_fn` or
       /// `false_fn`.</param>
       /// <param name="true_fn">The callable to be performed if pred is true.</param>
       /// <param name="false_fn">The callable to be performed if pred is false.</param>
       /// <param name="strict"> A boolean that enables/disables 'strict' mode; see above.</param>
       /// <param name="name">Optional name prefix for the returned tensors.</param>
       /// <returns>Tensors returned by the call to either `true_fn` or `false_fn`. If the
       /// callables return a singleton list, the element is extracted from the list.</returns>
       class function cond(pred: TFTensor; true_fn : TFunc<TFTensor>= nil; false_fn: TFunc<TFTensor>= nil; name: string = ''): TFTensor; overload ;static;
       class function cond<T>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<T>>; name: string): TArray<TFTensor>; overload ;static;
        /// <summary>
        /// Repeat `body` while the condition `cond` is true.
        /// </summary>
        /// <param name="cond"></param>
        /// <param name="body"></param>
        /// <param name="loop_vars"></param>
        /// <param name="shape_invariants"></param>
        class function while_loop<TItem: IFromMergeVars<TItem>>(_cond: TFunc<TItem, TFTensor>; body: TFunc<TItem, TItem>; loop_vars: TItem;
                                          shape_invariants   : TArray<TFShape>= [];
                                          parallel_iterations: Integer = 10;
                                          back_prop          : Boolean= true;
                                          swap_memory        : Boolean= false;
                                          name               : string= '';
                                          maximum_iterations : TFTensor= nil;
                                          return_same_structure : Boolean= false): TItem; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_control_flow_ops'}
  gen_control_flow_ops = record
    private

    public
       class function merge(inputs: TArray<TFTensor>; name: string = ''): MergeOutput; static;
       class function control_trigger(name: string = ''): TFOperation;static;
       class function no_op(name: string = ''): TFOperation; static;
       /// <summary>
       /// Creates or finds a child frame, and makes `data` available to the child frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="frame_name"></param>
       /// <param name="is_constant"></param>
       /// <param name="parallel_iterations"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function enter(data: TFTensor; frame_name: string = 'frame_name'; is_constant:Boolean = false; parallel_iterations : Integer= 10; name: string = ''): TFTensor;static;
       /// <summary>
       /// Forwards the input to the output.
       /// </summary>
       /// <param name="input"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function loop_cond(input: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Makes its input available to the next iteration.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function ref_next_iteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Makes its input available to the next iteration.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function next_iteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Exits the current frame to its parent frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function ref_exit(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Exits the current frame to its parent frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function _exit(data: TFTensor; name: string = ''): TFTensor; static;
       class function ref_switch(data, pred: TFTensor; name: string = '') : TArray<TFTensor>; static;
       /// <summary>
       /// Forwards `data` to the output port determined by `pred`.
       ///
       /// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
       /// the data goes to `output_false`.
       ///
       /// See also `RefSwitch` and `Merge`.
       /// </summary>
       /// <param name="data">A `Tensor`. The tensor to be forwarded to the appropriate output.</param>
       /// <param name="pred">A `Tensor` of type `bool`.
       /// A scalar that specifies which output port will receive data.
       /// </param>
       /// <param name="name"> A name for the operation (optional).</param>
       /// <returns>A tuple of `Tensor` objects (output_false, output_true).
       ///
       /// output_false: A `Tensor`. Has the same type as `data`.
       /// output_true: A `Tensor`. Has the same type as `data`.
       /// </returns>
       class function switch(data, pred: TFTensor; name: string = '') : TArray<TFTensor>; static;
  end;
  {$ENDREGION}

  {$REGION 'CondContext'}
  CondContext = class(TControlFlowContext)
   private
      Fexternal_values : TDictionary<string,TFTensor>;

      procedure _init_from_proto(context_def: TCondContextDef; import_scope: string = '');
      function  _BuildCondTensor(v: ITensorOrOperation): TFTensor;
      /// <summary>
      /// Process an output tensor of a conditional branch.
      /// </summary>
      function _ProcessOutputTensor(_val: TFTensor): TFTensor;
   protected
      procedure _AddOpInternal(op: TFOperation); override;
   public
      function AddValue(val: TFTensor): TFTensor; override;
      /// <summary>
      ///
      /// </summary>
      /// <param name="pred">The `boolean` tensor for the conditional predicate.</param>
      /// <param name="pivot">The predicate tensor in this branch.</param>
      /// <param name="branch">0 or 1 representing this branch.</param>
      /// <param name="name">Name of the `CondContext` python object.</param>
      /// <param name="context_def"></param>
      /// <param name="import_scope"></param>
      constructor Create(pred: TFTensor = nil; pivot: TFTensor = nil; branch: Integer = 0; name: string = 'cond_text'; context_def : TCondContextDef= nil; import_scope: string = '');
      /// <summary>
      /// Add the subgraph defined by fn() to the graph.
      /// </summary>
      function BuildCondBranch<T>(fn: TFunc<T>): tuple<T, TFTensor>;  overload;
      function BuildCondBranch<T>(fn: TFunc<TArray<T>>): tuple<TArray<T>, TArray<TFTensor>>; overload;
      destructor Destroy; override;
  end;
  {$ENDREGION}

  {$REGION 'control_flow_util'}
  control_flow_util = record
    private

    public
      /// <summary>
      /// Return true if `op` is a Switch.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsSwitch(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return true if `op` is an Exit.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopExit(op: TFOperation): Boolean; static;
      /// <summary>
      /// Returns true if `op` is an Enter.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopEnter(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return true iff op is a loop invariant.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopConstantEnter(op: TFOperation): Boolean; static;
      class function GetWhileContext(op: TFOperation): WhileContext; static;
      class function IsCondSwitch(op: TFOperation): Boolean; static;
      class function IsLoopSwitch(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return the control flow context for the output of an op.
      /// </summary>
      class function  GetOutputContext(op: TFOperation): TControlFlowContext; static;
      class procedure CheckInputFromValidContext(op: TFOperation; input_op: TFOperation); static;
      class function  GetLoopConstantEnter(value: TFTEnsor): TFOperation; static;
      class function  IsContainingContext(ctxt: WhileContext; maybe_containing_ctxt: WhileContext): Boolean; static;
      class function  GetContainingWhileContext(ctxt: TControlFlowContext; stop_ctxt : TControlFlowContext= nil): WhileContext; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_ops'}
  gen_ops = record
    private

    public
      /// <summary>
      ///    Computes exponential linear: <c>exp(features) - 1</c> if &amp;lt; 0, <c>features</c> otherwise.
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
      ///    Computes softplus: <c>log(exp(features) + 1)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Softplus'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function softplus(features: TFTensor; name: string = 'Softplus'): TFTensor; static;
      /// <summary>
      ///    Computes softsign: <c>features / (abs(features) + 1)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Softsign'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function softsign(features: TFTensor; name: string = 'Softsign'): TFTensor; static;
      /// <summary>
      ///    Computes rectified linear: <c>max(features, 0)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Relu'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function relu(features: TFTensor; name: string = 'Relu') : TFTensor; static;
      /// <summary>
      ///    Computes rectified linear 6: <c>min(max(features, 0), 6)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Relu6'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function relu6(features: TFTensor; name: string = 'Relu6') : TFTensor; static;
      /// <summary>
      ///    Clips tensor values to a specified min and max.
      /// </summary>
      /// <param name="t">
      ///    A <c>Tensor</c>.
      /// </param>
      /// <param name="clip_value_min">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The minimum value to clip by.
      /// </param>
      /// <param name="clip_value_max">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The maximum value to clip by.
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ClipByValue'.
      /// </param>
      /// <returns>
      ///    A clipped <c>Tensor</c> with the same shape as input 't'.
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      /// <remarks>
      ///    Given a tensor <c>t</c>, this operation returns a tensor of the same type and
      ///    shape as <c>t</c> with its values clipped to <c>clip_value_min</c> and <c>clip_value_max</c>.
      ///    Any values less than <c>clip_value_min</c> are set to <c>clip_value_min</c>. Any values
      ///    greater than <c>clip_value_max</c> are set to <c>clip_value_max</c>.
      /// </remarks>
      class function clip_by_value(t: TFTensor; clip_value_min: TFTensor; clip_value_max: TFTensor; name: string = 'ClipByValue') : TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'gen_image_ops'}
  gen_image_ops = record
    class function resize_nearest_neighbor<Tsize>(images: TFTensor; size: Tsize; align_corners: Boolean = false; half_pixel_centers : Boolean= false; name: string = ''): TFTensor; static;
    class function resize_bilinear(images: TFTensor; size: TFTensor; align_corners : Boolean= false; half_pixel_centers : Boolean = false; name : string= ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'image_ops_impl'}
  ResizeMethod= record
    const
      BILINEAR        : string = 'bilinear';
      NEAREST_NEIGHBOR: string = 'nearest';
      BICUBIC         : string = 'bicubic';
      AREA            : string = 'area';
      LANCZOS3        : string = 'lanczos3';
      LANCZOS5        : string = 'lanczos5';
      GAUSSIAN        : string = 'gaussian';
      MITCHELLCUBIC   : string = 'mitchellcubic';
  end;

  image_ops_impl = record
    private
       class function _resize_images_common(images: TFTensor; resizer_fn : TFunc<TFTensor, TFTensor, TFTensor>; size: TFTensor; preserve_aspect_ratio: Boolean; name: string; skip_resize_if_same: Boolean): TFTensor; static;
       class function _ImageDimensions(image: TFTensor; rank: Integer): TArray<Int64>; static;
    public
       /// <summary>
       /// Resize `images` to `size` using the specified `method`.
       /// </summary>
       /// <param name="images"></param>
       /// <param name="size"></param>
       /// <param name="method"></param>
       /// <param name="preserve_aspect_ratio"></param>
       /// <param name="antialias"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function resize_images_v2<T>(images: TFTensor; size: T; method: string = 'bilinear'; preserve_aspect_ratio : Boolean= false; antialias : Boolean= false; name : string= ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'dataset_ops'}
  dataset_ops = record
    private
      function anonymous_iterator_v3_eager_fallback(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string ; ctx: TContext): TFTensor;
    public
      function tensor_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string = ''): TFTensor;
      /// <summary>
      /// Creates a dataset that emits each dim-0 slice of `components` once.
      /// </summary>
      /// <param name="components"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function tensor_slice_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string = ''): TFTensor;
      function range_dataset(start: TFTensor; stop: TFTensor; step: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function repeat_dataset(input_dataset: TFTensor; count: TFTensor; output_types : TArray<TF_DataType>; output_shapes: TArray<TFShape>; name : string= '') : TFTensor;
      function shard_dataset(input_dataset: TFTensor; num_shards: TFTensor; index: TFTensor; output_types: TArray<TF_DataType> ; output_shapes: TArray<TFShape>; require_non_empty: Boolean = false; name: string = '') : TFTensor;
      function zip_dataset(input_datasets: TArray<TFTEnsor>; output_types: TArray<TF_DataType>; output_shapes:TArray<TFShape>; name: string = '') : TFTensor;
      function shuffle_dataset_v3(input_dataset: TFTensor; buffer_size: TFTensor; seed: TFTensor; seed2: TFTensor; seed_generator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; reshuffle_each_iteration: Boolean = true; name: string = ''): TFTEnsor;
      function skip_dataset(input_dataset : TFTensor; count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function dummy_seed_generator(name: string = '') : TFTensor;
      function concatenate_dataset(input_dataset: TFTensor; another_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function cache_dataset_v2(input_dataset: TFTensor; filename: TFTensor; cache: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that batches `batch_size` elements from `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="buffer_size"></param>
      /// <param name="drop_remainder"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="parallel_copy"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function batch_dataset_v2(input_dataset: TFTensor; buffer_size: TFTensor; drop_remainder: TFTensor; output_types: TArray<TF_DataType>; output_shapes:TArray<TFShape>; parallel_copy: Boolean = false; name: string = '') : TFTensor;
      /// <summary>
      ///
      /// </summary>
      /// <param name="name"></param>
      /// <returns></returns>
      function dummy_memory_cache(name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that asynchronously prefetches elements from `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="buffer_size"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="slack_period"></param>
      /// <param name="legacy_autotune"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function prefetch_dataset(input_dataset: TFTensor; buffer_size: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; slack_period: Integer = 0; legacy_autotune: Boolean= true; name: string = ''): TFTensor;
      /// <summary>
      /// Creates a dataset that contains `count` elements from the `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="count"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function take_dataset(input_dataset: TFTensor; count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset by applying optimizations to `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="optimizations"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="optimization_configs"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function optimize_dataset(input_dataset: TFTensor; optimizations: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; optimization_configs : TArray<string>= []; name: string = '') : TFTensor;
      function optimize_dataset_v2(input_dataset: TFTensor; optimizations_enabled: TFTensor; optimizations_disabled: TFTensor; optimizations_default: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; optimization_configs: TArray<string> = []; name: string = '') : TFTensor;
      /// <summary>
      /// Identity transformation that models performance.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="algorithm"></param>
      /// <param name="cpu_budget"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function model_dataset(input_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64; name: string = '') : TFTensor;
      /// <summary>
      /// A container for an iterator resource.
      /// </summary>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns>A tuple of `Tensor` objects (handle, deleter).</returns>
      function anonymous_iterator_v2(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : Tuple<TFTensor, TFTensor>;
      function anonymous_iterator_v3(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Makes a new iterator from the given `dataset` and stores it in `iterator`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="iterator"></param>
      /// <param name="name"></param>
      /// <returns>The created Operation.</returns>
      procedure make_iterator(dataset: TFTensor; iterator: TFTensor; name: string = '') ;
      /// <summary>
      ///
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="iterator"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism : Boolean= true; preserve_cardinality: Boolean = false; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset containing elements of `input_dataset` matching `predicate`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="predicate"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function filter_dataset(dataset: TFTensor; predicate: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '')  : TFTensor;
      /// <summary>
      /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="f"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function flat_map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="num_parallel_calls"></param>
      /// <param name="f"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="use_inter_op_parallelism"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function parallel_map_dataset_v2(dataset: TFTensor; num_parallel_calls: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism: Boolean = true; deterministic: string = 'default'; preserve_cardinality : Boolean= false; name : string= '') : TFTensor;
      /// <summary>
      /// A container for an iterator resource.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="deleter"></param>
      /// <param name="name"></param>
      /// <returns>The created Operation.</returns>
      procedure delete_iterator(handle: TFTensor; deleter: TFTensor; name: string = '') ;
      /// <summary>
      /// Gets the next output from the given iterator .
      /// </summary>
      /// <param name="iterator"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function iterator_get_next(iterator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = ''): TArray<TFTensor>;
  end;
  {$ENDREGION}

  {$REGION 'embedding_ops'}
  embedding_ops = record
    private

    public
        /// <summary>
        /// Helper function for embedding_lookup and _compute_sampled_logits.
        /// </summary>
        /// <param name="params"></param>
        /// <param name="ids"></param>
        /// <param name="partition_strategy"></param>
        /// <param name="name"></param>
        /// <param name="max_norm"></param>
        /// <returns></returns>
        class function _embedding_lookup_and_transform(params: IVariableV1;      ids: TFTensor; partition_strategy : string= 'mod'; name : string= ''; max_norm: string = ''): TFTensor; overload; static;
        class function _embedding_lookup_and_transform(params: TArray<TFTensor>; ids: TFTensor; partition_strategy : string= 'mod'; name : string= ''; max_norm: string = ''): TFTensor; overload; static;
        class function _clip(params: TFTensor; ids: TFTensor; max_norm : string= '') : TFTensor; static;
        class function embedding_lookup(params: TArray<TFTensor>; ids: TFTensor; partition_strategy: string = 'mod'; name: string = ''; validate_indices : Boolean= true; max_norm: string = ''): TFTensor; overload; static;
        class function embedding_lookup(params: IVariableV1;      ids: TFTensor; partition_strategy: string = 'mod'; name: string = ''; validate_indices : Boolean= true; max_norm : string= ''): TFTensor; overload; static;
  end;
  {$ENDREGION}

  {$REGION 'linalg_ops'}
  linalg_ops = class
    private

    public
      function eye(num_rows: Integer; num_columns: Integer = -1; batch_shape : PTFShape= nil; dtype: TF_DataType = TF_DOUBLE; name: string = ''): TFTensor;
      function matrix_inverse(input: TFTensor; adjoint: Boolean = false; name : string= ''): TFTensor;
      function matrix_solve_ls(matrix: TFTensor; rhs: TFTensor; l2_regularizer: TFTensor = nil; fast: Boolean = true; name: string = ''): TFTensor;
      function norm(tensor: TFTensor; _ord: string = 'euclidean'; axis: PAxis = nil; name: string = ''; keepdims: Boolean = true): TFTensor;
      function _composite_impl(matrix: TFTensor; rhs: TfTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _overdetermined(matrix: TFTensor; rhs: TFTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _underdetermined(matrix: TFTensor; rhs: TFTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _RegularizedGramianCholesky(matrix: TFTensor; l2_regularizer: TFTensor; first_kind: Boolean): TFTensor;
      function cholesky(input: TFTensor; name: string = '') : TFTensor;
      function cholesky_solve(chol: TFTensor; rhs: TFTensor; name: string = '') : TFTensor;
      function matrix_triangular_solve(matrix: TFTensor; rhs: TFTensor; lower: Boolean = true; adjoint: Boolean = false; name: string = ''): TFTensor;
      function qr(input: TFTensor; full_matrices: Boolean = false; name: string = ''): TFTensors;
  end;
  {$ENDREGION}

  {$REGION 'gen_resource_variable_ops'}
  gen_resource_variable_ops = record

    public
      class function assign_sub_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      /// <summary>
      /// Adds a value to the current value of a variable.
      /// </summary>
      /// <param name="resource"></param>
      /// <param name="value"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function assign_add_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      class function assign_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      class function var_is_initialized_op(resource: TFTensor; name: string = ''): TFTensor; static;
      /// <summary>
      /// Creates a handle to a Variable resource.
      /// </summary>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function var_handle_op(dtype: TF_DataType; shape: TFShape; container: string = ''; shared_name: string = ''; name: string = ''): TFTensor; static;
      class function destroy_resource_op(resource: TFTensor; ignore_lookup_error: Boolean = true; name: string = ''): TFTensor; static;
      /// <summary>
      /// Reads the value of a variable.
      /// </summary>
      /// <param name="resource"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function read_variable_op(resource: TFTensor; dtype: TF_DataType; name: string = ''): TFTensor; static;
      class function resource_gather(resource: TFTensor; indices: TFTensor; dtype: TF_DataType; batch_dims: Integer = 0; validate_indices: Boolean = true; name: string = ''): TFTensor; static;
  end;
  {$ENDREGION}

  {$REGION 'resource_variable_ops'}
  resource_variable_ops = record
    private
      /// <summary>
      /// Concats HandleData from tensors `handle` and `initial_value`.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="initial_value"></param>
      /// <returns></returns>
      class function  _combine_handle_data(handle: TFTensor; initial_value: TFTensor) : THandleData; static;
      class function  get_eager_safe_handle_data(handle: TFTensor) : THandleData; static;
      /// <summary>
      /// Sets the shape inference result HandleData on tensor.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="handle_data"></param>
      /// <param name="graph_mode"></param>
      class procedure _set_handle_shapes_and_types(tensor: TFTensor; handle_data: THandleData; graph_mode: Boolean);static;
    public
      /// <summary>
      /// Creates a variable handle with information to do shape inference.
      /// </summary>
      /// <param name="initial_value"></param>
      /// <param name="shape"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <param name="graph_mode"></param>
      /// <returns></returns>
      class function eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name: string; name: string; graph_mode: Boolean): TFTensor; Static;
      class function shape_safe_assign_variable_handle(tHandle: TFTensor;  shape: TArray<Integer>; value: TFTensor; name: string = '') : TFOperation; static;
      class function is_resource_variable(vVar :IVariableV1): Boolean; static;
      /// <summary>
      /// Create a new variable handle, optionally copying in `extra_handle_data`
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <param name="graph_mode"></param>
      /// <param name="initial_value"></param>
      /// <returns></returns>
      class function variable_handle_from_shape_and_dtype(shape: TFShape; dtype: TF_DataType; shared_name: string; name: string; graph_mode: Boolean; initial_value : TFTensor= nil) : TFTensor; Static;
  end;
  {$ENDREGION}

  {$REGION 'stateless_random_ops'}
   stateless_random_ops  = record
      private
        class function _ShapeTensor(shape: TArray<Integer>): TFTensor; static;
        class function _get_key_counter(seed: TArray<Integer>; alg: Integer) : Tuple<TFTensor, TFTensor>; static;
      public
        class function  stateless_random_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed : TArray<Integer> = []; name: string = '') : TFTensor; static;
   end;
  {$ENDREGION}

  {$REGION 'gen_sparse_ops'}
  gen_sparse_ops = Record
    private

    public
      /// <summary>
      /// Converts a sparse representation into a dense tensor.
      /// </summary>
      /// <param name="sparse_indices"></param>
      /// <param name="output_shape"></param>
      /// <param name="sparse_values"></param>
      /// <param name="default_value"></param>
      /// <param name="validate_indices"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TArray<Integer>;sparse_values: T; default_value: T; validate_indices: boolean = true; name: string = ''): TFTensor; overload; static;
      Class function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFTensor; sparse_values: TFTensor; default_value: T; validate_indices: boolean = true; name: string = ''): TFTensor; overload; static;
  End;
  {$ENDREGION}

  {$REGION 'gen_state_ops'}
  gen_state_ops = record
    private

    public
      /// <summary>
      /// Holds state in the form of a tensor that persists across steps.
      /// Outputs a ref to the tensor state so it may be read or modified.
      /// </summary>
      /// <param name="shape">The shape of the variable tensor.</param>
      /// <param name="dtype">The type of elements in the variable tensor.</param>
      /// <param name="name"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <returns></returns>
      class function variable_v2(shape: TArray<Integer>; dtype: TF_DataType; name: string = ''; container : string= ''; shared_name: string = '') : TFTensor; static;
      /// <summary>
      /// Update 'ref' by assigning 'value' to it
      /// </summary>
      /// <param name="ref"></param>
      /// <param name="value"></param>
      /// <param name="validate_shape"></param>
      /// <param name="use_locking"></param>
      /// <param name="name"></param>
      class function assign<T>(ref: T; value: TValue; validate_shape : Boolean= true; use_locking : Boolean= true;  name: string = '') : TFTensor; static;
      class function assign_add<T>(ref: IVariableV1; value: T; use_locking : Boolean= false; name: string = '') : TFTensor; static;
      class function assign_sub(ref: IVariableV1; value: TFTensor; use_locking : Boolean= false; name : string = '')  : TFTensor; static;
      /// <summary>
      /// Adds sparse updates to a variable reference.
      /// </summary>
      /// <param name="ref"></param>
      /// <param name="indices"></param>
      /// <param name="updates"></param>
      /// <param name="use_locking"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function scatter_add(ref: IVariableV1; indices: TFTensor; updates: TFTensor; use_locking: Boolean = false; name : string= '') : TFTensor; static;
      class function is_variable_initialized(ref: RefVariable; name: string = '') : TFTensor; static;
  end;
  {$ENDREGION}

implementation
       uses System.Math,

            Oz.Pb.Classes,

            Tensorflow,
            Tensorflow.Tensor,
            TensorFlow.Ops,
            Tensorflow.Utils,
            TensorFlow.Slice,
            Tensorflow.Variable,

            Numpy,
            NumPy.NDArray,

            ProtoGen.Main;

{$REGION 'tensor_array_ops'}
{ tensor_array_ops }

class function tensor_array_ops.build_ta_with_new_flow(old_ta: TTensorArray; flow: TFTensor): TTensorArray;
begin
    var new_ta := tf.TensorArray(old_ta.dtype, 0, false, True, nil, old_ta.colocate_with_first_write_call, old_ta.infer_shape);
    Result := new_ta;
end;

class function tensor_array_ops.build_ta_with_new_flow(old_ta: TGraphTensorArray; flow: TFTensor): TTensorArray;
begin
    var new_ta := tf.TensorArray(old_ta.dtype, 0, false, True, nil, old_ta.colocate_with_first_write_call, old_ta.infer_shape);
    Result := new_ta;
end;
{$ENDREGION}

{$REGION 'gen_random_ops'}
{ gen_random_ops }

class function gen_random_ops.random_standard_normal(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomStandardNormal', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.random_uniform_int(shape, minval, maxval: TFTensor; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomUniformInt', name, ExecuteOpArgs.Create([ shape, minval, maxval ])
                                                 .SetAttributes(['seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.stateless_random_get_key_counter(seed: TArray<Integer>; name: string): TFTensors;
begin
     Result := tf.Context.ExecuteOp('StatelessRandomGetKeyCounter', name, ExecuteOpArgs.Create([ seed ]))
end;

class function gen_random_ops.stateless_random_normal_v2(shape, key, counter: TFTensor; alg: Integer; dtype: TF_DataType; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('StatelessRandomNormalV2', name, ExecuteOpArgs.Create([ shape, key, counter, alg ])
                                                              .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)) ]) ).First;
end;

class function gen_random_ops.random_uniform(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomUniform', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.random_shuffle(value: TFTensor; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomShuffle', name, ExecuteOpArgs.Create([ value ])
                                                 .SetAttributes(['seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.truncated_normal(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('TruncatedNormal', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue(dtype),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.multinomial(logits: TFTensor; num_samples, seed, seed2: Integer; output_dtype: TF_DataType; name: string): TFTensor;
begin
   var _op := tf.OpDefLib._apply_op_helper('Multinomial', name, [ GetArg('logits',logits), GetArg('seed',seed), GetArg('seed2',seed2), GetArg('output_dtype',TValue(output_dtype)) ]);
   Result := _op.Output;
end;
{$ENDREGION}

{$REGION 'random_ops'}
{ random_ops }

class function random_ops.random_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From<TFShape>(shape), mean, stddev]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_normal', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            rnd   : TFTensor;
                            begin
                                name := string(v1.ToString);
                                var shape_tensor  := _ShapeTensor(shape);
                                var mean_tensor   := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor := Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed := random_seed.get_seed(nSeed);
                                if tSeed.Value1.HasValue then rnd := gen_random_ops.random_standard_normal(shape_tensor, dtype, tSeed.Value1, tSeed.Value2)
                                else                          rnd := gen_random_ops.random_standard_normal(shape_tensor, dtype);

                                var mul           := TTensor(rnd) * TTensor(stddev_tensor);
                                var value         := math_ops.add(mul, mean_tensor, name);
                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function random_ops.random_uniform(shape: TArray<Integer>; minval, maxval: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            rnd   : TTensor;
                            minTensor, maxTensor: TFTensor;
                            begin
                                var name := string(v1.ToString);

                                var nSeed : Nullable<Integer> := nil;
                                if Assigned(seed) then nSeed := seed^;

                                tSeed:= random_seed.get_seed(nSeed);

                                var tensorShape   := TUtils.shape_tensor(shape);
                                minTensor := Tops.convert_to_tensor(minval, dtype, 'min');
                                maxTensor := Tops.convert_to_tensor(maxval, dtype, 'max');
                                if tSeed.Value1.HasValue then rnd := gen_random_ops.random_uniform(tensorShape, dtype, tSeed.Value1, tSeed.Value2)
                                else                          rnd := gen_random_ops.random_uniform(tensorShape, dtype);

                                Result := math_ops.add(rnd * (TTensor(maxTensor) - minTensor), minTensor, name);
                            end );
end;

class function random_ops.random_uniform_int(shape: TArray<Integer>; minval: Integer; maxval: Integer; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform_int', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                var name := string(v1.ToString);

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed := random_seed.get_seed(nSeed);

                                var tensorShape   := TUtils.shape_tensor(shape);
                                var minTensor := Tops.convert_to_tensor(minval, DtInvalid, 'min');
                                var maxTensor := Tops.convert_to_tensor(maxval, DtInvalid, 'max');
                                if tSeed.Value1.HasValue then Result := gen_random_ops.random_uniform_int(tensorShape, minTensor, maxTensor, tSeed.Value1, tSeed.Value2)
                                else                          Result := gen_random_ops.random_uniform_int(tensorShape, minTensor, maxTensor);
                            end );
end;

class function random_ops.random_uniform(shape: TFTensor; minval: Integer; maxval: TFTensor; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([shape, minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                var name      := string(v1.ToString);
                                var minTensor : TTensor := Tops.convert_to_tensor(minval, dtype, 'min');
                                var maxTensor : TTensor;
                                if maxval = nil then maxTensor := Tops.convert_to_tensor(1 , dtype, 'max')
                                else                 maxTensor := Tops.convert_to_tensor(integer(TTensor(maxval)) , dtype, 'max');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed := random_seed.get_seed(nSeed);

                                if Tdtypes.is_integer(dType) then
                                begin
                                    if tSeed.Value1.HasValue then Result := gen_random_ops.random_uniform_int(shape, minTensor, maxTensor, tSeed.Value1, tSeed.Value2, name)
                                    else                          Result := gen_random_ops.random_uniform_int(shape, minTensor, maxTensor, 0, 0, name)
                                end else
                                begin
                                    var rnd := gen_random_ops.random_uniform(shape, dtype);
                                    Result  := math_ops.add(rnd * (maxTensor - minTensor), minTensor, name);
                                end;
                            end );
end;

class function random_ops.random_shuffle(value: TFTensor; seed: Integer; name: string): TFTEnsor;
var
  tSeed : Tuple<TNullableInteger, TNullableInteger>;
  seed1,
  seed2 : Integer ;
begin
    tSeed      := random_seed.get_seed(seed);
    seed1  := tSeed.Value1;
    seed2  := tSeed.Value1;

    Result := gen_random_ops.random_shuffle(value, seed1, seed2, name);
end;

class function random_ops.truncated_normal(shape: TArray<Integer>; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), mean, stddev]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'truncated_normal', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            rnd   : TFTensor;
                            begin
                                name              := string(v1.ToString);
                                var shape_tensor  := _ShapeTensor(shape);
                                var mean_tensor   := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor := Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);
                                if tSeed.Value1.HasValue then rnd := gen_random_ops.truncated_normal(shape_tensor, dtype, tSeed.Value1, tSeed.Value2)
                                else                          rnd := gen_random_ops.truncated_normal(shape_tensor, dtype);

                                var mul           := TTensor(rnd) * TTensor(stddev_tensor);
                                var value         := math_ops.add(mul, mean_tensor, name);
                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function random_ops._ShapeTensor(shape: TArray<Integer>): TFTEnsor;
begin
   Result := Tops.convert_to_tensor(TValue.From< TArray<Integer> >(shape), DtInvalid, 'shape');
end;

class function random_ops.multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger; name: string; output_dtype: TF_DataType): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([ logits ]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'multinomial', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                Result := multinomial_categorical_impl(logits, num_samples, output_dtype, seed);
                            end );
end;

class function random_ops.multinomial_categorical_impl(logits: TFTensor; num_samples: Integer; dtype: TF_DataType; seed: pInteger): TFTEnsor;
var
  tSeed : Tuple<TNullableInteger, TNullableInteger>;
  seed1,
  seed2 : Integer ;
begin
    logits := Tops.convert_to_tensor(logits, DtInvalid, 'logits');

    var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
    if Assigned(seed) then nSeed := seed^;

    tSeed             := random_seed.get_seed(nSeed);
    seed1             := tSeed.Value1;
    seed2             := tSeed.Value1;
    Result :=  gen_random_ops.multinomial(logits, num_samples, seed1, seed2, dtype);
end;
{$ENDREGION}

{$REGION 'clip_ops'}
{ clip_ops }

class function clip_ops.clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor; name: string): Tuple<TFTensors, TFTensor>;
begin
    var vValues : TArray<TValue> := [];
    for var i := 0 to Length(t_list)-1 do
       vValues := vValues + [ t_list[i] ];

    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    use_norm := global_norm(t_list, name);
    Result := TUtils.tf_with<TNameScope,Tuple<TFTensors, TFTensor>>( TOps.name_scope(name, 'clip_by_global_norm', @newVal),
                                          function(v1: TNameScope): Tuple<TFTensors, TFTensor>
                                            begin
                                                // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
                                                var scale_for_finite := clip_norm * TTensor(math_ops.minimum(
                                                    Single(1.0) / TTensor(use_norm),
                                                    TTensor(constant_op.constant(1.0, use_norm.dtype, 'Const')) / clip_norm));
                                                // If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
                                                // this will make scale NaN.
                                                var scale := scale_for_finite + (TTensor(use_norm) - use_norm);
                                                var values_clipped : TFTensors := TFTensors.Create;
                                                var i : Integer := 0;
                                                for var v in t_list do
                                                begin
                                                    values_clipped.Add(array_ops.identity(v * scale, 'name_'+ IntToStr(i)) );
                                                    Inc(i);
                                                end;
                                                Result := Tuple<TFTensors, TFTensor>.Create(values_clipped, use_norm);
                                            end );
end;

class function clip_ops.clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [t, TValue.From<T1>(clip_value_min), TValue.From<T2>(clip_value_max)];
    var newVal  : TValue         := TValue.From<TArray<TValue>>(vValues);;

    Result := TUtils.tf_with<TNameScope, TFTensor>( TOps.name_scope(name, 'clip_by_value', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var values := Tops.convert_to_tensor(t, DtInvalid, 't');
                                                // Go through list of tensors, for each value in each tensor clip
                                                var t_min := math_ops.minimum(values, clip_value_max);
                                                // Assert that the shape is compatible with the initial shape,
                                                // to prevent unintentional broadcasting.
                                                var _ := values.shape.merge_with(t_min.shape);
                                                var t_max := math_ops.maximum<TFTensor, T1>(t_min, clip_value_min, name);
                                                _ := values.shape.merge_with(t_max.shape);
                                                Result := t_max;
                                            end );
end;

class function clip_ops.global_norm(t_list: TArray<TFTensor>; name: string): TFTensor;
var
  Selfun   : TFunc<TFTensor,TFTensor>;
begin
    Selfun   := Function(x: TFTensor): TFTensor
                 begin
                     Result := nn_ops.l2_loss(x);
                 end ;
    var vValues : TArray<TValue> := [];
    for var i := 0 to Length(t_list)-1 do
       vValues := vValues + [ t_list[i] ];

    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'global_norm', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var half_squared_norms := Enumerable<TFTensor>.Create(t_list).Select(Selfun).ToArray;
                                                var half_squared_norm := math_ops.reduce_sum(array_ops.stack(half_squared_norms));
                                                var norm := math_ops.sqrt(TTensor(half_squared_norm) *
                                                    constant_op.constant(2.0, half_squared_norm.dtype,'Const'),
                                                    'global_norm');
                                                Result := norm;
                                            end );
end;
{$ENDREGION}

{$REGION 'gen_array_ops'}
{ gen_array_ops }

class function gen_array_ops.batch_to_space_nd<T>(input: T; block_shape: TArray<Integer>; crops: TArray<TArray<Integer>>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('BatchToSpaceND', name,[ GetArg('input',TValue.From<T>(input)),
                                                                     GetArg('block_shape', TValue.From<TArray<Integer>>(block_shape)),
                                                                     GetArg('crops',TValue.From< TArray<TArray<Integer>> >(crops)) ] );
    Result := _op.output;
end;

class function gen_array_ops.expand_dims(input: TFTensor; axis: integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ExpandDims', name, ExecuteOpArgs.Create([ input, axis ])
                                                 .SetAttributes(['axis', TValue.From<Integer>(axis) ]) ).First;
end;

class function gen_array_ops.fill<T>(dims: TFTensor; value: T; name: string): TFTensor;
begin
    var v := TValue.From<T>(value);
    Result := tf.Context.ExecuteOp('Fill', name, ExecuteOpArgs.Create([dims, v])).First;
end;

class function gen_array_ops.gather_v2<T1, T2>(params: T1; indices: T2; axis, batch_dims: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('GatherV2', name, ExecuteOpArgs.Create([ TValue.From<T1>(params), TValue.From<T2>(indices), axis ])
                                                      .SetAttributes(['batch_dims', batch_dims ]) ).First;
end;

class function gen_array_ops.identity(input: TFTensor; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('Identity', name, ExecuteOpArgs.Create([input])).First;
end;

class function gen_array_ops.invert_permutation(x: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('InvertPermutation', name,  [ GetArg('x',x) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.log(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Log', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_array_ops.ones_like(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('OnesLike', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_array_ops.one_hot(indices, depth, on_value, off_value: TFTensor; dtype: TF_DataType; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('OneHot', name, ExecuteOpArgs.Create([indices, depth, on_value, off_value])
                                          .SetAttributes(['axis', axis ]) ).First;
end;

class function gen_array_ops.pack(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Pack', name, ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(values) ])
                                                 .SetAttributes(['axis', TValue.From<Integer>(axis) ]) ).First;
end;

class function gen_array_ops.pad(input, paddings: TFTensor; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        (*var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
            "Pad", name,
            null,
            input, paddings);
        return results[0];*)
        Result := pad_eager_fallback(input, paddings, name, tf.Context);
        Exit;
    end;
    var _op := tf.OpDefLib._apply_op_helper('Pad', name,  [ GetArg('input',input), GetArg('paddings',paddings) ] );
    Result := _op.output;
end;

class function gen_array_ops.pad_eager_fallback(inputs, padding: TFTensor; name: string; ctx: TContext): TFTensor;
begin
    var t1 :Tuple<TF_DataType, TArray<TFTensor>>;
    t1 := tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, [ inputs ]);
    var _attr_T := t1.Value1;
    var input   := t1.Value2;

    var t2 :Tuple<TF_DataType, TArray<TFTensor>>;
    t2 := tf.Runner.ArgsToMatchingEager(ctx, tf.int32_t, [ padding ]);
    var _attr_Tpaddings := t2.Value1;
    var paddings        := t2.Value2;

    var _inputs_flat := input + paddings;

    var _attrs : TArray<TValue> := [ 'T', _attr_T, 'Tpaddings',_attr_Tpaddings ] ;

    var Res := tf.Runner.Execute(ctx, 'Pad', 1, _inputs_flat, _attrs, name);

    if tf.Runner.MustRecordGradient then
        tf.Runner.RecordGradient('Pad', _inputs_flat, _attrs, res);
    Result := res[0];
end;

class function gen_array_ops.placeholder_with_default<T>(input: T; shape: TArray<Integer>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('PlaceholderWithDefault', name,  [ GetArg('input', TValue.From<T>(input)), GetArg('shape',TValue.From<TArray<Integer>>(shape)),  GetArg('name',name) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.rank(input: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Rank', name, ExecuteOpArgs.Create([input])).First;
end;

class function gen_array_ops.reshape(tensor: TFTensor; shape: TArray<TValue>; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Reshape', name, ExecuteOpArgs.Create([tensor, TValue.From< TArray<TValue> >(shape)]) ).First;
end;

class function gen_array_ops.reshape<T>(tensor: TFTensor; shape: T; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Reshape', name, ExecuteOpArgs.Create([tensor, TValue.From<T>(shape)])).First;
end;

class function gen_array_ops.resource_strided_slice_assign(input, tBegin, tEnd, tStrides, tVvalue: TFTensor; begin_mask: Integer; end_mask: Integer; ellipsis_mask: Integer;new_axis_mask: Integer; shrink_axis_mask: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ResourceStridedSliceAssign', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides, tVvalue ])
                                                 .SetAttributes(['begin_mask',      begin_mask,
                                                                 'end_mask',        end_mask,
                                                                 'ellipsis_mask',   ellipsis_mask,
                                                                 'new_axis_mask',   new_axis_mask,
                                                                 'shrink_axis_mask',shrink_axis_mask ]) ).First;
end;

class function gen_array_ops.reverse<T>(tensor: TFTensor; axis: T; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ReverseV2', name, [ GetArg('tensor',tensor), GetArg('axis', TValue.From<T>(axis) ) ]);
    Result := _op.output;
end;

class function gen_array_ops.shape(input: TFTensor; out_type: TF_DataType; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Shape', name, ExecuteOpArgs.Create([ input ]).SetAttributes(['out_type', out_type ]) ).First;
end;

class function gen_array_ops.shape_n(input: TArray<TFTensor>; out_type: TF_DataType; name: string): TArray<TFTensor>;
begin
    Result := tf.Context.ExecuteOp('ShapeN', name, ExecuteOpArgs.Create([ input ])
                                                 .SetAttributes(['out_type', TValue.From<Integer>(Ord(out_type)) ]) ).ToArray;
end;

class function gen_array_ops.size(input: TFTensor; out_type: TF_DataType; name: string): TFTensor;
begin
     var _op := tf.OpDefLib._apply_op_helper('Size', name, [ GetArg('input',input), GetArg('out_type',out_type) ]);
     Result := _op.outputs[0];
end;

class function gen_array_ops.slice(input: TFTensor; _begin, size: TArray<TFTensor>; name: string): TFTensor;
begin
    if tf.executing_eagerly then
    begin
        var res := slice_eager_fallback(input, _begin, size, name, tf.Context);
        Result := res;
        Exit;
    end;
    var _op := tf.OpDefLib._apply_op_helper('Slice', name, [ GetArg('input',input), GetArg('begin',_begin ), GetArg('size',size ) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.slice<Tb, Ts>(input: TFTensor; _begin: Tb; size: Ts; name: string): TFTensor;
begin
    if tf.executing_eagerly then
    begin
        var outputs := tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('Slice', name, [input, TValue.From<Tb>(_begin), TValue.From<Ts>(size) ]));
        Result := outputs[0];
        Exit;
    end;
    var _op := tf.OpDefLib._apply_op_helper('Slice', name, [ GetArg('input',input), GetArg('begin',TValue.From<Tb>(_begin)),GetArg('size',TValue.From<Ts>(size)) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.slice_eager_fallback(inputs: TFTensor; _begin, size: TArray<TFTensor>; name: string; ctx: TContext): TFTensor;
begin
    var t1 :Tuple<TF_DataType, TArray<TFTensor>>;
    t1 := tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, [ inputs ]);
    var _attr_T := t1.Value1;
    var input   := t1.Value2;

    var t2 :Tuple<TF_DataType, TArray<TFTensor>>;
    t2 := tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, [  _begin, size ]);
    var _attr_Tidx    := t2.Value1;
    var _inputs_Index := t2.Value2;

    var _inputs_flat := input + _inputs_Index;
    var _attrs : TArray<TValue> := [ 'T', _attr_T, 'Index', _attr_Tidx ] ;

    var Res := tf.Runner.Execute(ctx, 'Slice', 1, _inputs_flat, _attrs, name);

    if tf.Runner.MustRecordGradient then
        tf.Runner.RecordGradient('Slice', _inputs_flat, _attrs, res);
    Result := res[0];
end;

class function gen_array_ops.split_v(value, size_splits: TFTensor; axis, num_split: Integer; name: string): TArray<TFTensor>;
begin
    Result := tf.Context.ExecuteOp('SplitV', name, ExecuteOpArgs.Create([ value, size_splits, axis ])
                                                 .SetAttributes(['num_split', num_split ]) ).ToArray;
end;

class function gen_array_ops.squeeze(input: TFTensor; axis: TArray<Integer>; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Squeeze', name, ExecuteOpArgs.Create([ input ]).SetAttributes([ 'squeeze_dims',TValue.From< TArray<Integer> >(axis) ] ) ).First;
end;

class function gen_array_ops.stop_gradient(x: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('StopGradient', name, [ GetArg('input',x), GetArg('name',name) ]);
    Result := _op.output;
end;

class function gen_array_ops.strided_slice(input, tBegin, tEnd, tStrides: TFTensor; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Int64;
  name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StridedSlice', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides ])
                                                 .SetAttributes(['begin_mask',      begin_mask,
                                                                 'end_mask',        end_mask,
                                                                 'ellipsis_mask',   ellipsis_mask,
                                                                 'new_axis_mask',   new_axis_mask,
                                                                 'shrink_axis_mask',shrink_axis_mask]) ).First;
end;

class function gen_array_ops.strided_slice<T>(input: TFTensor; tBegin, tEnd, strides: TArray<T>; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Integer;
  name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('StridedSlice', name,
              [ GetArg('input',            input),
                GetArg('Begin',            TValue.From< TArray<T> >(tBegin)),
                GetArg('End',              TValue.From< TArray<T> >(tEnd)),
                GetArg('strides',          TValue.From< TArray<T> >(strides)),
                GetArg('begin_mask',       begin_mask),
                GetArg('end_mask',         end_mask),
                GetArg('ellipsis_mask',    ellipsis_mask),
                GetArg('new_axis_mask',    new_axis_mask),
                GetArg('shrink_axis_mask', shrink_axis_mask) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.tile(input, multiples: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Tile', name, ExecuteOpArgs.Create([input, multiples])).First;
end;

class function gen_array_ops.tile(input: TFTensor; multiples: TArray<TValue>; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Tile', name, ExecuteOpArgs.Create([ input, TValue.From< TArray<TValue> >(multiples) ])).First;
end;

class function gen_array_ops.transpose<T1>(x: TFTensor; perm: T1; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Transpose', name, ExecuteOpArgs.Create([x, TValue.From<T1>(perm)])).First;
end;

class function gen_array_ops.unique(x: TFTensor; out_idx: TF_DataType; name: string): Tuple<TFTensor, TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('Unique', name, [ GetArg('x',x), GetArg('out_idx', TValue.From<Integer>(Ord(out_idx)) ) ] );
    // TODO
    //var _result = _UniqueOutput._make(_op.outputs);
    Result := Tuple.Create(_op.outputs[0], _op.outputs[1]);
end;

class function gen_array_ops.unpack(value: TFTensor; num, axis: Integer; name: string): TArray<TFTensor>;
begin
    Result := tf.Context.ExecuteOp('Unpack', name, ExecuteOpArgs.Create([value, num])
                    .SetAttributes(['axis', axis, num, 'num'])).ToArray
end;

class function gen_array_ops.where(condition: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Where', name, [ GetArg('input',condition) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.scatter_nd(indices, updates: TFTensor; shape: TArray<TFTensor>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ScatterNd', name, [ GetArg('indices',indices),GetArg('updates',updates), GetArg('shape',shape) ]);
    Result := _op.outputs[0];
end;

class function gen_array_ops.zeros_like(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ZerosLike', name, ExecuteOpArgs.Create([x])).First;
end;

class function gen_array_ops.select<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Select', name, ExecuteOpArgs.Create([ condition, TValue.From<Tx>(x), TValue.From<Ty>(y) ]) ).First;
end;

class function gen_array_ops.select_v2<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('SelectV2', name, ExecuteOpArgs.Create([ condition, TValue.From<Tx>(x), TValue.From<Ty>(y) ]) ).First;
end;

class function gen_array_ops.broadcast_args(s0, s1: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('BroadcastArgs', name, ExecuteOpArgs.Create([s0, s1])).First;
end;

class function gen_array_ops.broadcast_gradient_args(s0, s1: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin
    var res := tf.Context.ExecuteOp('BroadcastGradientArgs', name, ExecuteOpArgs.Create([s0, s1]));
    Result := Tuple<TFTensor, TFTensor>.Create(res[0], res[1]);
end;

class function gen_array_ops.broadcast_to<T>(input: TFTensor; shape: T; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('BroadcastTo', name, ExecuteOpArgs.Create([input, TValue.From<T>(shape)])).First;
end;

class function gen_array_ops.check_numerics(tensor: TFTensor; _message, name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('CheckNumerics', name, [ GetArg('tensor',tensor), GetArg('message',_message) ]);
    Result := _op.output;
end;

class function gen_array_ops.concat_offset(concat_dim: TFTensor; shape: TArray<TFTensor>; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('ConcatOffset', name, [ GetArg('concat_dim',concat_dim), GetArg('shape',shape) ] );
    Result := _op.outputs;
end;

class function gen_array_ops.concat_v2<T, Ta>(values: TArray<T>; axis: Ta; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ConcatV2', name, ExecuteOpArgs.Create([ TValue.From< TArray<T> >(values), TValue.From<Ta>(axis)])).First;
end;

class function gen_array_ops.concat_v2(values: TArray<TFTensor>; axis: TFTensor; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        Result := concat_v2_eager_fallback<TFTensor,TFTensor>(values, axis, name, tf.Context);
        Exit;
    end;
    Result := tf.Context.ExecuteOp('ConcatV2', name, ExecuteOpArgs.Create([values, axis])).First;
end;

class function gen_array_ops.concat_v2(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ConcatV2', name, ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(values), axis])).First;
end;

class function gen_array_ops.concat_v2_eager_fallback<T1, T2>(values: TArray<T1>; axis: T2; name: string; ctx: TContext): TFTensor;
begin
    var _attr_N := Length(values);
    var aValue : TArray<TValue> := [];
    for var i := 0 to Length(values)-1 do
        aValue := aValue + [ TValue.From<T1>(values[i]) ];

    var tup1 :Tuple<TF_DataType, TArray<TFTensor>>;
    tup1 := tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, aValue);
    var _attr_T := tup1.Value1;
    var input   := tup1.Value2;
    var tup2 :Tuple<TF_DataType, TArray<TFTensor>>;
    tup2 := tf.Runner.ArgsToMatchingEager(ctx, tf.int32_t, [ TValue.From<T2>(axis) ]);
    var _attr_Tidx := tup2.Value1;
    var axis1      := tup2.Value2;

    var _inputs_flat := input + axis1;
    var _attrs : TArray<TValue> := [ 'N', _attr_N, 'T', _attr_T, 'Tidx', _attr_Tidx ] ;
    Result := tf.Runner.Execute(ctx, 'ConcatV2', 1, _inputs_flat, _attrs, name)[0];
end;

class function gen_array_ops.diag(diagonal: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Diag', name, ExecuteOpArgs.Create([diagonal])).First;
end;

class function gen_array_ops.diag_part(diagonal: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DiagPart', name, ExecuteOpArgs.Create([diagonal])).First;
end;
{$ENDREGION}

{$REGION 'array_ops'}
{ array_ops }

class function array_ops.concat(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    if Length(values) = 1 then // Degenerate case of one tensor.
    begin
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Shape'),
                    function(v1: TNameScope): TFTensor
                      begin
                          // Make a throwaway call to convert_to_tensor to make sure
                          // that axis is of the correct type, and make sure that
                          // the returned tensor is a scalar.
                          Tops.convert_to_tensor(axis, TF_DataType.TF_INT32, 'concat_dim' );
                          Result := identity(values[0], v1.ToString);;
                      end );
        Exit;
    end;
    Result := gen_array_ops.concat_v2(values, axis, name);
end;

class function array_ops.concat(values: TArray<TFTensor>; axis: TFTensor; name: string): TFTensor;
begin
    Result := gen_array_ops.concat_v2(values, axis, name);
end;

class function array_ops.concat(values: TArray<TValue>; axis: Integer; name: string): TFTensor;
begin
    Result := gen_array_ops.concat_v2<TValue,Integer>(values, axis, name);
end;

class function array_ops.constant(value: TValue; dtype : TF_DataType= DtInvalid; shape: TArray<Integer>= nil; name : AnsiString = 'Const'; verify_shape : Boolean = false):TFTensor;
begin
   if shape  = nil then
        Result := constant_op.constant(value, dtype, nil, verify_shape, false, name)
   else begin
    var s : TFShape := shape;
    Result := constant_op.constant(value, dtype, @s, verify_shape, false, name);
   end;
end;

class function array_ops.pad(tensor, paddings: TFTensor; mode, name: string; constant_values: Integer): TFTensor;
begin
    Result := nil;;
    mode := mode.ToUpper;
    if mode = 'CONSTANT' then
    begin
        if constant_values <> 0 then
           raise TFException.Create('Not Implemented gen_array_ops.pad_v2')
        else
            Result := gen_array_ops.pad(tensor, paddings, name);
    end;

    // Restore shape information where possible.
    if  not tf.Context.executing_eagerly then
    begin
        var paddings_constant := TUtils.constant_value(paddings);
        var input_shape := result.op.inputs[0].shape;

        var ePaddings_constant := Enumerable<TNDArray>.Create([paddings_constant]);
        var eInput_shape       := TList<Integer>.Create(input_shape.as_int_list);

        if (input_shape.ndim > -1) and (not result.shape.IsFullyDefined) and (not(paddings_constant = nil)) then
        begin
            var new_shape := TList<Integer>.Create;
            try
              {TODO -o Max-cGeneral : Zip function verify!!}
              for var item in TUtils.zip<TNDArray,Integer>(ePaddings_constant, eInput_shape) do
              begin
                  var padding := item.Value1;
                  var dim     := Item.Value2;
                  if (padding = nil) or (dim = -1) or (TArray.Contains<Integer>(padding.ToArray<Integer>, -1) ) then
                      new_shape.Add(-1)
                  else
                      new_shape.Add( NDarray(np.sum(padding)) + dim );
              end;
              result.shape := new_shape.ToArray;
            finally
              new_shape.free;
            end;
        end;
    end;

    //return result;
end;

class function array_ops.placeholder(_dtype: TF_DataType; _shape: PTFShape; name: string): TFTensor;
begin
    var sShape  : TFShape := System.default(TFShape);
    if Assigned(_shape) then sShape := _shape^;

    if tf.Context.executing_eagerly then
       raise Exception.Create('tf.placeholder is not compatible with eager execution.');

   var _op := tf.OpDefLib._apply_op_helper('Placeholder', name, [ GetArg('dtype',TValue.From<Integer>(Ord(_dtype))), GetArg('shape',TValue.From<TFShape>(sShape)) ]);
   Result := _op.Output;
end;

class function array_ops.prevent_gradient(input: TFTensor; msg, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('PreventGradient', name, ExecuteOpArgs.Create([ input ])
                           .SetAttributes(['message', msg ])).First;
end;

class function array_ops.rank(input: TFTensor; name: string ): TFTensor;
begin
    Result := rank_internal(input, name, true);
end;

class function array_ops.rank_internal(input: TFTensor; name: string; optimize: Boolean) : TFTensor;
begin
    var lst := TList<TFTensor>.Create([input]);
    var vvalue := TValue.From< TList<TFTensor> >(lst);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Rank', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                name := string(v1.ToString);
                                var input_shape := input.shape;
                                if (optimize) and (input_shape.ndim > 0) then
                                    Result := constant_op.constant(input_shape.ndim, tf.int32_t, name)
                                else
                                    Result := gen_array_ops.rank(input, name);
                            end );
end;

class function array_ops.reshape(tensor: TFTensor; shape: TFShape; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

class function array_ops.reshape(tensor, shape: TFTensor; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

class function array_ops.reshape(tensor: TFTensor; shape: TArray<TValue>; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

class function array_ops.shape(input: TFTensor; name: string; out_type: TF_DataType): TFTensor;
begin
    Result := shape_internal(input, name,  true, out_type);
end;

class function array_ops.shape_internal(input: TFTensor; name: string; optimize: Boolean; out_type: TF_DataType): TFTensor;
begin
    var vvalue := TValue.From<TFTensor>(input);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Shape', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              name := string(v1.ToString);
                              if not tf.Context.executing_eagerly then
                              begin
                                  var input_shape := input.shape;
                                  if (optimize) and (input.ndim > -1) and (input_shape.IsFullyDefined) then
                                  begin
                                      if(out_type = TF_DataType.TF_INT32) then
                                      begin
                                          var v : TValue := TValue.From< TArray<Integer> >(input.shape.as_int_list);
                                          Result := constant_op.constant(v, DtInvalid, name);
                                          Exit;
                                      end else
                                      begin
                                          var v : TValue := TValue.From< TArray<Int64> >(input.shape.dims);
                                          Result := constant_op.constant(v, DtInvalid, name);
                                          Exit;
                                      end;
                                  end;
                              end ;

                              Result := gen_array_ops.shape(input,out_type, name );
                          end );
end;

class function array_ops.shape_v2(input: TFTensor; name: string; out_type: TF_DataType): TFTensor;
begin
    Result := shape_internal(input, name, true, out_type);
end;

class function array_ops.boolean_mask<T1, T2>(tensor: T1; mask: T2; name: string; axis: Integer): TFTensor;
begin
    var tensor_tensor := Tops.convert_to_tensor( TValue.From<T1>(tensor), dtInvalid,'tensor');
    var mask_tensor   := Tops.convert_to_tensor( TValue.From<T2>(mask), dtInvalid,'mask');

    var shape_mask   := mask_tensor.shape;
    var ndims_mask   := shape_mask.ndim;
    var shape_tensor := tensor_tensor.shape;

    if ndims_mask < 1 then
        raise Exception.Create('mask cannot be scalar.');

    var aAxis : TArray<Integer> := [0];
    var _leading_size := gen_math_ops.prod( Shape(tensor_tensor)[ [Format('%d:%d',[axis,axis + ndims_mask])] ], aAxis );

    (*            leading_size = gen_math_ops.prod(shape(tensor)[axis:axis + ndims_mask], [0])
                  tensor = reshape(
                      tensor,
                      concat([
                          shape(tensor)[:axis], [leading_size],
                          shape(tensor)[axis + ndims_mask:]
                      ], 0))
      FIX: Leading_size is list.
    *)
    var l1 : TArray<TValue> := [_leading_size];
    var leading_size := array_ops._autopacking_conversion_function(l1, _leading_size.dtype, '');

    var shape1 := concat([shape(tensor_tensor)[ [ Format(':%d',[axis]) ] ],
                          leading_size,
                          shape(tensor_tensor)[ [ Format('%d:',[axis + ndims_mask]) ] ] ],0);

    tensor_tensor := reshape(tensor_tensor, shape1);
    var eDim := Enumerable<Int64>.Create(shape_tensor.dims) ;

    var first_dim := eDim.Skip(axis).Take(ndims_mask).First;
    var s1        := TFShape.Create( eDim.Take(axis).ToArray );
    var s2        := s1.concatenate([first_dim ]).concatenate(eDim.Skip(axis + ndims_mask).ToArray);
    tensor_tensor.shape := s2;

    mask_tensor := reshape( mask_tensor, TFShape.Create([-1]) ) ;
    Result := _apply_mask_1d(tensor_tensor, mask_tensor, axis);
end;

class function array_ops._apply_mask_1d(reshaped_tensor, mask: TFTensor; axis: Integer): TFTensor;
begin
     var indices := squeeze( where(mask), [ 1 ]);
     Result      := gather(reshaped_tensor, indices,'', axis);
end;

class function array_ops.size<T>(input: T; name: string; optimize: Boolean; out_type: TF_DataType): TFTensor;
begin
   Result := size_internal(input, name, optimize, out_type);
end;

class function array_ops.size_internal<T>(input: T; name: string; optimize: Boolean; out_type: TF_DataType): TFTensor;
begin
    var vvalue := TValue.From<T>(input);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Size', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              name := string(v1.ToString);
                              var input_tensor := Tops.convert_to_tensor(TValue.From<T>(input));
                              var input_shape  := input_tensor.shape;
                              if optimize then
                              begin
                                  if input_shape.IsFullyDefined then
                                  begin
                                      Result := constant_op.constant(input_shape.size, out_type, name);
                                      Exit;
                                  end;
                              end;

                              Result := gen_array_ops.size(input_tensor, out_type, name);
                          end );
end;

class function array_ops.slice<Tb, Ts>(input: TFTensor; _begin: Tb; size: Ts; name: string): TFTensor;
begin
    Result := gen_array_ops.slice(input, _begin, size, name);
end;

class function array_ops.split(value, size_splits: TFTensor; axis, num: Integer; name: string): TArray<TFTensor>;
begin
    if num = -1 then
        num := size_splits.shape[0];

    Result := gen_array_ops.split_v(value, size_splits, axis, num, name);
end;

class function array_ops.split<T>(value: TFTensor; num_split: Integer; axis: T; name: string): TArray<TFTensor>;
begin
    var size_splits := Tops.convert_to_tensor(num_split);

    if tf.Context.executing_eagerly then
    begin
        Result := split_eager_fallback(axis, value, num_split, name, tf.Context);
        Exit;
    end;

    var _op := tf.OpDefLib._apply_op_helper('Split', name, [ GetArg('split_dim', TValue.From<T>(axis)), GetArg('value',value), GetArg('num_split',num_split) ]);
    Result := _op.Outputs;
end;

class function array_ops.split_eager_fallback<Ta, Tv>(axis: Ta; value: Tv; num_split: Integer; name: string; ctx: TContext ) : TArray<TFTensor>;
begin
    var aValue : TArray<TValue> := [TValue.From<Tv>(value)];

    var tup1 :Tuple<TF_DataType, TArray<TFTensor>>;
    tup1 :=tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, aValue);
    var _attr_T := tup1.Value1;
    var input   := tup1.Value2;

    var axis_tensor := Tops.convert_to_tensor( TValue.From<Ta>(axis), TF_INT32 );
    var _inputs_flat := TList<TFTensor>.Create([axis_tensor]);
    _inputs_flat.AddRange(input);

    var _attrs : TArray<TValue> := [ 'num_split', num_split, 'T', _attr_T ] ;
    Result := tf.Runner.Execute(ctx, 'Split', 1, _inputs_flat.ToArray, _attrs, name);
end;

class function array_ops.slice(input: TFTensor; _begin, size: TArray<TFTensor>; name: string): TFTensor;
begin
    Result := gen_array_ops.slice(input, _begin, size, name);
end;

class function array_ops.slice(input, _begin, size: TFTensor; name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([input, _begin, size]);
    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                   begin
                       Result := [];
                       var pParam : TParameter;
                       pParam.sNome := 'T' ;
                       pParam.vValue:= op.get_attr('T');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'Index' ;
                       pParam.vValue:= op.get_attr('Index');
                       Result := Result + [ pParam ] ;
                   end;

    Result := tf.Context.ExecuteOp('Mean', name, Args).First;
end;

class function array_ops.squeeze(input: TFTensor; axis: TArray<Integer>; name: string): TFTensor;
begin
    Result := gen_array_ops.squeeze(input, axis, name)
end;

class function array_ops.stack(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    if axis = 0 then
     Result :=  Tops.convert_to_tensor(values, DtInvalid,name)
    else
     Result := gen_array_ops.pack(values, axis, name);
end;

class function array_ops.stack(values: TValue; axis: Integer; name: string): TFTensor;
begin
    if axis = 0 then
      // If the input is a constant list, it can be converted to a constant op
     Exit( Tops.convert_to_tensor(values, DtInvalid,name) );

    raise Exception.Create('array_ops.stack');
end;

class function array_ops.stop_gradient(input: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StopGradient', name, ExecuteOpArgs.Create([ input ])).First;
end;

class function array_ops.strided_slice(input_, _begin, _end, strides: TFTensor; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Integer;
  name: string): TFTensor;
begin
    var op := gen_array_ops.strided_slice(input_, _begin, _end, strides,  begin_mask, end_mask, ellipsis_mask,  new_axis_mask, shrink_axis_mask, name);
    Result := op;
end;

class function array_ops.strided_slice_grad(shape, _begin, _end, strides, dy: TFTensor; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Int64;
  name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([shape, _begin, _end, strides, dy]);
    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                   begin
                       Result := [];
                       var pParam : TParameter;
                       pParam.sNome := 'T' ;
                       pParam.vValue:= op.get_attr('T');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'Index' ;
                       pParam.vValue:= op.get_attr('Index');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'begin_mask' ;
                       pParam.vValue:= op.get_attr('begin_mask');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'end_mask' ;
                       pParam.vValue:= op.get_attr('end_mask');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'ellipsis_mask' ;
                       pParam.vValue:= op.get_attr('ellipsis_mask');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'new_axis_mask' ;
                       pParam.vValue:= op.get_attr('new_axis_mask');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'shrink_axis_mask' ;
                       pParam.vValue:= op.get_attr('shrink_axis_mask');
                       Result := Result + [ pParam ] ;
                   end;
    Args.SetAttributes(['begin_mask',begin_mask,'end_mask',end_mask,'ellipsis_mask',ellipsis_mask,'new_axis_mask',new_axis_mask,'shrink_axis_mask',shrink_axis_mask]) ;
    Result := tf.Context.ExecuteOp('StridedSliceGrad', name, Args).First;
end;

class function array_ops.tile(input, multiples: TFTensor; name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([input, multiples]);
    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                   begin
                       Result := [];
                       var pParam : TParameter;
                       pParam.sNome := 'T' ;
                       pParam.vValue:= op.get_attr('T');
                       Result := Result + [ pParam ] ;

                       pParam.sNome := 'Tmultiples' ;
                       pParam.vValue:= op.get_attr('Tmultiples');
                       Result := Result + [ pParam ] ;
                   end;

    Result := tf.Context.ExecuteOp('Tile', name, Args).First;
end;

class function array_ops.transpose(a, perm: TFTensor; name: string; conjugate: Boolean): TFTensor;
begin
    var vvalue := TValue.From<TFTensor>(a);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'transpose', @vvalue),
                function(v1: TNameScope): TFTensor
                  begin
                      name   := string(v1.ToString);
                      Result := gen_array_ops.transpose(a, perm, name);
                  end );
end;

class function array_ops.transpose<T1>(a: T1; perm: PAxis; name: string; conjugate: Boolean): TFTensor;
begin
    var vvalue := TValue.From<TFTensor>(TValue.From<T1>(a));
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'transpose', @vvalue),
                function(v1: TNameScope): TFTensor
                var
                  e1 :  TAxis;
                  begin
                      var a_tensor := Tops.convert_to_tensor(TValue.From<T1>(a));
                      if perm = nil then
                      begin
                          var rank := a_tensor.rank;
                          e1 := TUtils.range(0, rank).OrderByDescending<integer>(function(x: Integer): Integer
                                                                                   begin
                                                                                       Result := x;
                                                                                   end).ToArray();
                          perm := @e1;
                      end;

                      name   := string(v1.ToString);
                      Result := gen_array_ops.transpose(a_tensor, perm^, name);
                  end );
end;

class function array_ops.unique(x: TFTensor; out_idx: TF_DataType; name: string): Tuple<TFTensor, TFTensor>;
begin
   Result := gen_array_ops.unique(x, out_idx, name);
end;

class function array_ops.unstack(value: TFTensor; num: PInteger; axis: Integer; name: string): TArray<TFTensor>;
begin
     if num = nil then
     begin
         var numero := value.shape.as_int_list[axis];
         num := @numero ;
     end;
     Result := gen_array_ops.unpack(value, num^, axis, name);
end;

class function array_ops.where(condition: TFTensor; x: TObject; y: TObject; name: string): TFTensor;
begin
    if (x = nil) and (y = nil) then
    begin
        var vvalue := TValue.From<TFTensor>(condition);
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Where', @vvalue),
                    function(v1: TNameScope): TFTensor
                      begin
                          name      := string(v1.ToString);
                          condition := Tops.convert_to_tensor(condition, DtInvalid, 'condition',False, TDtypes.cbool);
                          Result    := gen_array_ops.where(condition, name);
                      end );
    end
    else if (x <> nil) and (y <> nil) then
    begin
        Result := gen_array_ops.select(condition, x, y, name);
    end else
    begin
        raise TFException.Create('x and y must both be non-None or both be None.');
    end;
end;

class function array_ops.where_v2(condition: TFTensor; x, y: TObject; name: string): TFTensor;
begin
    if (x = nil) and (y = nil) then
    begin
        var vvalue := TValue.From<TFTensor>(condition);
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Where', @vvalue),
                    function(v1: TNameScope): TFTensor
                      begin
                          name      := string(v1.ToString);
                          condition := Tops.convert_to_tensor(condition, DtInvalid, 'condition',False, TDtypes.cbool);
                          Result    := gen_array_ops.where(condition, name);
                      end );
    end
    else if (x <> nil) and (y <> nil) then
    begin
        Result := gen_array_ops.select_v2(condition, x, y, name);
    end else
    begin
        raise TFException.Create('x and y must both be non-None or both be None.');
    end;
end;

class function array_ops._get_dtype_from_nested_lists<T>(list_or_tuple: TArray<T>): TF_DataType;
var
  v : TArray<TValue>;
begin
    v := [];

    for var i := 0 to Length(list_or_tuple)-1 do
      v := v + [ TValue.From<T>(list_or_tuple[i]) ]  ;

    Result := _get_dtype_from_nested_lists(v)
end;

class function array_ops._get_dtype_from_nested_lists(list_or_tuple: TArray<TValue>): TF_DataType;
begin
    var dtype : TF_DataType  := TF_DataType.DtInvalid;

    for var v in list_or_tuple do
    begin
        if v.IsType<TFTensor> then
        begin
           var tten := v.AsType<TFTensor>;
           dtype := Tdtypes.as_base_dtype( tten.dtype ) ;
           Break;
        end
        else if v.IsType<Integer> then
        begin
           dtype := TF_INT32;
           Break;
        end;
        if (dtype <> TF_DataType.DtInvalid) then
            break;
    end;
    Result :=  dtype;
end;

class function array_ops._autopacking_conversion_function(v: TArray<TValue>; dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    var inferred_dtype := _get_dtype_from_nested_lists(v);
    if dtype = TF_DataType.DtInvalid then
        dtype := inferred_dtype;

    if name = '' then Result := _autopacking_helper(v, dtype, 'packed')
    else              Result := _autopacking_helper(v, dtype, name);
end;

class function array_ops._autopacking_helper(list_or_tuple: TArray<TValue>; dtype: TF_DataType; name: string): TFTensor;
begin
    var must_pack := false;
    var converted_elems := TList<TValue>.Create;

    var switch_to_graph: Boolean := tf.Context.switched_to_graph(list_or_tuple);

    var res := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name),
                          function(v1: TNameScope): TFTensor
                            begin
                                var i := 0;
                                for var elem in list_or_tuple do
                                begin
                                    converted_elems.Add(elem);
                                    must_pack := true;
                                end ;

                                if must_pack then
                                begin
                                    var elems_as_tensors := TList<TFTensor>.Create;
                                    i := 0;
                                    for var elem in converted_elems do
                                    begin
                                        if elem.IsType<TEagerTensor> then
                                        begin
                                            var eager_tensor := elem.AsType<TEagerTensor>;
                                            if switch_to_graph then
                                                elems_as_tensors.Add( constant_op.constant(eager_tensor.numpy, dtype, i.ToString) )
                                            else
                                                elems_as_tensors.Add(eager_tensor);
                                        end
                                        else if elem.IsType<TFTensor> then
                                        begin
                                            var tensor := elem.AsType<TFTensor>;
                                            elems_as_tensors.Add(tensor);
                                        end else
                                        begin
                                            var elem_tensor := constant_op.constant(elem, dtype, i.ToString);
                                            elems_as_tensors.Add(elem_tensor);
                                        end;
                                        Inc(i);
                                    end;

                                    Result := gen_array_ops.pack(elems_as_tensors.ToArray(), 0, v1.ToString);
                                end else
                                begin
                                    Result := tf.constant(np.np_array<Single>( [] ));
                                end;
                            end );
    if switch_to_graph then
      tf.Context.restore_mode;

    Result := res;
end;

class function array_ops._constant_if_small<T>(value: T; shape: TFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    if shape.size < 1000 then
    begin
        Result := constant_op.constant(TValue.From<T>(value), dtype, @shape, False, True, name);
    end else
    begin
        var shape_t := constant_op._tensor_shape_tensor_conversion_function(shape);
        var c       := constant_op.constant(0, dtype,'Const');
        Result      := gen_array_ops.fill(shape_t, c, name);
    end;
end;

class function array_ops.ones(shape: TFTensor; dtype: TF_DataType; name : string): TFTensor;
begin
    var vvalue := TValue.From<TFTensor>(shape);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'ones', @vvalue),
                    function(v1: TNameScope): TFTensor
                      begin
                          name := string(v1.ToString);
                          var output : TFTensor ;
                          if dtype = TF_DOUBLE then
                             output := gen_array_ops.fill(shape, constant_op.constant(Double(1), dtype, 'Const'), name)
                          else
                             output := gen_array_ops.fill(shape, constant_op.constant(Single(1), dtype, 'Const'), name);
                          Result := output;
                      end );
end;

class function array_ops.ones(shape: TArray<TFTensor>; dtype: TF_DataType; name : string): TFTensor;
begin
    dtype := Tdtypes.as_base_dtype(dtype);
    var vvalue := TValue.From< TArray<TFTensor> >(shape);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'ones', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                name := string(v1.ToString);
                                var shape1 := Tops.convert_to_tensor(shape, TF_DataType.TF_INT32);
                                var output := gen_array_ops.fill(shape1, constant_op.constant(1, dtype, 'Const'), name);
                                Result := output;
                            end );
end;

class function array_ops.expand_dims(input: TFTensor; axis: Integer; name: string): TFTensor;
begin
    Result := gen_array_ops.expand_dims(input, axis, name);
end;

class function array_ops.fill<T>(dims: TFShape; value: T; name: string): TFTensor;
begin

end;

class function array_ops.gather<T1, T2>(params: T1; indices: T2; name: string; axis, batch_dims: Integer): TFTensor;
begin
     if axis <> 0 then
        Exit ( gen_array_ops.gather_v2(params, indices, axis, 0, name) );

    if ( TypeInfo(T1) = TypeInfo(ResourceVariable) ) and ( TypeInfo(T2) = TypeInfo(TFTensor) ) then
    begin
        var v1 := Tvalue.From<T1>(params);
        var v2 := Tvalue.From<T2>(indices);
        var variable      : ResourceVariable := v1.AsType<ResourceVariable>;
        var indices_tensor: TFTensor         := v2.AsType<TFTensor>;
        Result := variable.sparse_read(indices_tensor, name);
        Exit;
    end;

    Result := gen_array_ops.gather_v2(params, indices, axis, 0, name);
end;

class function array_ops.identity(input: TFTensor; name: String): TFTensor;
begin
   Result := gen_array_ops.identity(input, name)
end;

class function array_ops.invert_permutation(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_array_ops.invert_permutation(x, name);
end;

class function array_ops.matrix_diag(diagonal: TFTensor; name: string; k, num_rows, num_cols: Integer; padding_value: Single; align: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('MatrixDiagV3', name, ExecuteOpArgs.Create([ diagonal, k, num_rows, num_cols, Tops.convert_to_tensor(padding_value, diagonal.dtype) ])
                           .SetAttributes(['align', align ])).First;
end;

class function array_ops.matrix_set_diag(input, diagonal: TFTensor; name: string; k: Integer; align: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatrixSetDiagV3', name, ExecuteOpArgs.Create([ input, diagonal, k ])
                           .SetAttributes(['align', align ])).First;
end;

class function array_ops.meshgrid<T>(_array: TArray<T>; copy, sparse: Boolean; indexing: string): TArray<TFTensor>;
begin
    var vvalue := TValue.From< TArray<T> >(_array);
    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope('', 'meshgrid', @vvalue),
                      function(v1: TNameScope): TArray<TFTensor>
                      var
                       Selfun : TFunc<Integer,Integer>;
                        begin
                            Selfun := Function(x: Integer): Integer
                                       begin
                                           Result := 1;
                                       end ;

                            var ndim := Length(_array);
                            var s0   := TUtils.range(ndim).Select( Selfun );

                            var output := TList<TFTensor>.Create;
                            try
                              for var i:= 0 to Length(_array) do
                              begin
                                  var x := _array[i];
                                  var shape := s0.Take(i).concat( TCollections.CreateList<Integer>([-1]) ).concat(s0.Skip(i + 1)).ToArray;
                                  output.add(reshape(stack(TValue.From<T>(x)), shape));
                              end;

                              // Create parameters for broadcasting each tensor to the full size
                              var shapes := Enumerable<T>.Create(_array).Select<TFTensor>( function(arg: T): TFTensor
                                                                                  begin
                                                                                     Result := size(arg);
                                                                                  end ).ToArray;

                              var output_dtype := TDTypes.as_base_dtype(_get_dtype_from_nested_lists<T>(_array) );
                              if (indexing = 'xy') and ( ndim > 1) then
                              begin
                                  output[0] := reshape(output[0], TCollections.CreateList<Integer>( [  1, -1]).concat(TUtils.range(ndim - 2).Select(Selfun) ).ToArray );
                                  output[1] := reshape(output[1], TCollections.CreateList<Integer>( [ -1,  1]).concat(TUtils.range(ndim - 2).Select(Selfun) ).ToArray );
                                  var sw := shapes[0];
                                  shapes[0] := shapes[1];
                                  shapes[1] := sw;
                              end;

                              if sparse then
                                  Result := output.ToArray
                              else begin

                                  var mult_fact := ones(shapes, output_dtype);
                                  Result := Enumerable<TFTensor>.Create(output.ToArray).Select( function(arg: TFTensor): TFTensor
                                                                                                 begin
                                                                                                     Result := TTensor(arg) * mult_fact;
                                                                                                 end).ToArray;
                              end
                            finally
                              output.free;
                            end;

                        end );
end;

class function array_ops.moveaxis(_array: TNDArray; source, destination: TAxis): TFTensor;
var
 Selfun : TFunc<Integer,Integer>;
 OrdFun : TFunc<Tuple<integer,integer>,Integer>;
begin
    Selfun := Function(x: Integer): Integer
               begin
                    if x < 0 then Result := _array.rank + x
                    else          Result := x
               end ;
    OrdFun := Function(x: Tuple<integer,integer>): Integer
               begin
                    Result := x.Value1;
               end ;

    var perm : IList<Integer> := nil;
    source      := Enumerable<Integer>.Create(source.axis.Value).Select( Selfun ).ToArray;
    destination := Enumerable<Integer>.Create(destination.axis.Value).Select( Selfun ).ToArray;

    if _array.shape.rank > -1 then
    begin
        var x : Enumerable<Integer> := TUtils.range(0, _array.rank).Where(  function(const i: Integer): Boolean
                                                                                begin
                                                                                    Result := not Enumerable<Integer>.Create(source.axis.Value).Contains(i)
                                                                                end );
        perm := x.ToList;

        var eSrc     := Tlist<Integer>.Create(source.axis.Value);
        var eDst     := Enumerable<Integer>.Create(destination.axis.Value);
        var eZipEnum : Enumerable< Tuple<integer,integer> > :=TUtils.zip<integer,integer>(eDst, eSrc);
        eZipEnum.OrderBy<integer>(OrdFun);
        for  var tt  in eZipEnum do
        begin
            var dst := tt.Value1;
            var src := tt.Value2;
            perm.Insert(dst, src);
        end;
    end
    else
        raise TFException.Create('Not Implemented');

    var A : TAxis := perm.ToArray;
    Result := array_ops.transpose(_array, @A );
end;

class function array_ops.ones(shape: TFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    var vvalue := TValue.From<TFShape>(shape);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'ones', @vvalue),
                      function(v1: TNameScope): TFTensor
                        begin
                            dtype := Tdtypes.as_base_dtype(dtype);
                            name := string(v1.ToString);
                            var ones : TFTensor;
                            case dtype of
                               TF_DOUBLE : ones := constant(Double(1));
                               TF_FLOAT  : ones := constant(Single(1));
                            else
                               ones := constant(1)
                            end;

                            if shape.ndim = 0 then
                               Exit(ones);

                            Result := gen_array_ops.fill(shape, ones,  name);

                        end );
end;

class function array_ops.ones_like(tensor: TFTensor; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
    var vVAlue := TValue.From< TArray<TFTensor> >([tensor]) ;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'ones_like', @vVAlue),
                        function(v1: TNameScope): TFTensor
                          begin
                              name := v1.ToString;
                              tensor := Tops.convert_to_tensor(tensor, DtInvalid, 'tensor');

                              // is_fully_defined return unexpected value.
                              if (optimize) and (tensor.shape.IsFullyDefined) and (dtype <> TF_DataType.TF_VARIANT) then
                              begin

                              end;

                              if (dtype <> TF_DataType.DtInvalid) and (dtype <> tensor.dtype) and (dtype <> TF_DataType.TF_VARIANT)  then
                              begin
                                  raise Exception.Create('Not Implemented("ones_like"');
                                  // return ones(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                              end else
                              begin
                                  Result := gen_array_ops.ones_like(tensor, name);
                              end;
                          end );
end;

class function array_ops.one_hot(indices: TFTensor; depth, on_value, off_value: TFTensor; dtype: TF_DataType; axis: Integer; name: string): TFTensor;
begin
    var vVAlue := TValue.From< TArray<TValue> >([indices,depth, dtype]) ;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'one_hot', @vVAlue),
                        function(v1: TNameScope): TFTensor
                          begin
                              name := v1.ToString;
                              var on_exists  := false;
                              var off_exists := false;
                              var on_dtype   := TF_DataType.DtInvalid;
                              var off_dtype  := TF_DataType.DtInvalid;

                              if (dtype = TF_DataType.DtInvalid) then
                                  dtype := TF_DataType.TF_FLOAT;

                              if not on_exists then
                              begin
                                  on_value := Tops.convert_to_tensor(1, dtype, 'on_value');
                                  on_dtype := dtype;
                              end;

                              if not off_exists then
                              begin
                                  off_value := Tops.convert_to_tensor(0, dtype, 'off_value');
                                  off_dtype := dtype;
                              end;

                                  if on_dtype <> off_dtype then
                                     raise Exception.Create('dtype {0} of on_value does not match dtype {1} of off_value');

                              Result := gen_array_ops.one_hot(indices, depth, on_value, off_value, DtInvalid, axis,  name);
                          end );
end;

class function array_ops.zeros(shape: TFTensor; dtype: TF_DataType; name: string) : TFTensor;
begin
    var ddtype := Tdtypes.as_base_dtype(dtype);
    var vShape := TValue.From<TFTensor>(shape) ;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros', @vShape),
                        function(v1: TNameScope): TFTensor
                          begin

                              name := v1.ToString;
                              case ddtype of
                                TF_DataType.TF_BOOL:
                                  Result := gen_array_ops.fill(shape, tf.constant(false, dtype), name);
                                TF_DataType.TF_DOUBLE:
                                  Result := gen_array_ops.fill(shape, tf.constant(Double(0), dtype),  name);
                                TF_DataType.TF_FLOAT:
                                  Result := gen_array_ops.fill(shape, tf.constant(Single(0), dtype),  name);
                                TF_DataType.TF_INT32:
                                  Result := gen_array_ops.fill(shape, tf.constant(0, dtype),  name);
                                else
                                  raise Exception.Create('can''t find type for zeros');
                              end;

                          end );
end;

class function array_ops.zeros(shape: TFShape; dtype: TF_DataType = TF_DataType.TF_FLOAT; name : string = ''): TFTensor;
begin
    dtype := Tdtypes.as_base_dtype(dtype);
    var value_Shape := TValue.From<TFShape>(shape);
    if tf.executing_eagerly then
    begin
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros', @value_Shape),
                            function(v1: TNameScope): TFTensor
                              begin
                                  name := string(v1.ToString);
                                  var zeros : TFTensor;
                                  case dtype of
                                     TF_DOUBLE : zeros := constant(Double(0));
                                     TF_FLOAT  : zeros := constant(Single(0));
                                     TF_INT8   : zeros := constant(Int8(0));
                                     TF_UINT8  : zeros := constant(UInt8(0));
                                  else
                                     zeros := constant(0)
                                  end;
                                  Result := gen_array_ops.fill(shape, zeros,  name);
                              end );
    end else
    begin
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros', @value_Shape),
                          function(v1: TNameScope): TFTensor
                            begin
                                name := string(v1.ToString);
                                case dtype of
                                   TF_BOOL    : Result := _constant_if_small(false,     shape, dtype, name);
                                   TF_DOUBLE  : Result := _constant_if_small(Double(0), shape, dtype, name);
                                   TF_FLOAT   : Result := _constant_if_small(Single(0), shape, dtype, name);
                                   TF_INT64   : Result := _constant_if_small(Int64(0),  shape, dtype, name);
                                   TF_INT32   : Result := _constant_if_small(Int32(0),  shape, dtype, name);
                                   TF_INT8    : Result := _constant_if_small(Int8(0),   shape, dtype, name);
                                else
                                   raise Exception.Create('can''t find type for zeros');
                                end;
                            end );
    end;

end;
class function array_ops.broadcast_dynamic_shape(shape_x, shape_y: TFTensor): TFTensor;
begin
    Result :=  gen_array_ops.broadcast_args(shape_x, shape_y);
end;

class function array_ops.broadcast_static_shape(shape_x, shape_y: TFTensor): TFTensor;
begin
    Result := common_shapes.broadcast_shape(shape_x, shape_y);
end;

class function array_ops.zeros_like(tensor: TFTensor; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
    var vVAlue := TValue.From< TArray<TFTensor> >([tensor]) ;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros_like', @vVAlue),
                        function(v1: TNameScope): TFTensor
                          begin
                              name := v1.ToString;
                              tensor := Tops.convert_to_tensor(tensor, DtInvalid, 'tensor');

                              // is_fully_defined return unexpected value.
                              if (optimize) and (tensor.shape.IsFullyDefined) and (dtype <> TF_DataType.TF_VARIANT) then
                              begin

                              end;

                              if (dtype <> TF_DataType.DtInvalid) and (dtype <> tensor.dtype) and (dtype <> TF_DataType.TF_VARIANT)  then
                              begin
                                  raise Exception.Create('Not Implemented("zeros_like"');
                                  // return zeros(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                              end else
                              begin
                                  Result := gen_array_ops.zeros_like(tensor, name);
                              end;
                          end );
end;
{$ENDREGION}

{$REGION 'gen_math_ops'}
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
    t1 :=tf.Runner.ArgsToMatchingEager(ctx, TF_Datatype.DtInvalid, [ inputs ]);
    var _attr_T := t1.Value1;
    var input   := t1.Value2;

    var t2 :Tuple<TF_DataType, TArray<TFTensor>>;
    t2 := tf.Runner.ArgsToMatchingEager(ctx, tf.int32_t, [ axis ]);
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

class function gen_math_ops._max<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name : string) : TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ TValue.From<Tx>(input), TValue.From<Ty>(axis) ]) ;

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

    Result := tf.Context.ExecuteOp('Max', name, Args
        .SetAttributes(['keep_dims', keep_dims, 'reduction_indices',  TValue.From<Ty>(axis) ])).First;
end;

class function gen_math_ops._min<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name : string) : TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ TValue.From<Tx>(input), TValue.From<Ty>(axis) ]) ;

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

    Result := tf.Context.ExecuteOp('Min', name, Args
        .SetAttributes(['keep_dims', keep_dims, 'reduction_indices',  TValue.From<Ty>(axis) ])).First;
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
{$ENDREGION}

{$REGION 'gen_data_flow_ops'}
{ gen_data_flow_ops }

class function gen_data_flow_ops.dynamic_stitch(indices, data: TArray<TFTensor>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('DynamicStitch', name, [GetArg('indices',indices), GetArg('data',data)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_gather_v3(handle, indices, flow_in: TFTensor; dtype: TF_DataType; element_shape: PTFShape; name: string): TFTensor;
begin
    var sShape : TFShape := System.default(TFShape);
    if Assigned(element_shape) then
       sShape := element_shape^;

    var _op := tf.OpDefLib._apply_op_helper('TensorArrayGatherV3', name, [GetArg('handle',handle),
                                                                          GetArg('indices',indices),
                                                                          GetArg('dtype',dtype),
                                                                          GetArg('element_shape', TValue.From<TFShape>(sShape)),
                                                                          GetArg('flow_in',flow_in)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_read_v3(handle, index, flow_in: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArrayReadV3', name, [GetArg('handle',handle),
                                                                        GetArg('index',index),
                                                                        GetArg('flow_in',flow_in),
                                                                        GetArg('dtype', dtype)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_size_v3(handle, flow_in: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArraySizeV3', name, [GetArg('handle',handle), GetArg('flow_in',flow_in)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_v3<T>(size: T; dtype: TF_DataType; element_shape: PTFShape; dynamic_size, clear_after_read, identical_element_shapes: Boolean;
  tensor_array_name, name: string): Tuple<TFTensor, TFTensor>;
begin
    var sShape : TFShape := System.default(TFShape);
    if Assigned(element_shape) then
       sShape := element_shape^;

     var _op := tf.OpDefLib._apply_op_helper('TensorArrayV3', name, [GetArg('size', TValue.From<T>(size)),
                                                                     GetArg('dtype',dtype),
                                                                     GetArg('element_shape',TValue.From<TFShape>(sShape)),
                                                                     GetArg('dynamic_size', dynamic_size),
                                                                     GetArg('clear_after_read', clear_after_read),
                                                                     GetArg('identical_element_shapes', identical_element_shapes),
                                                                     GetArg('tensor_array_name', tensor_array_name)]);
    Result  :=  Tuple.Create(_op.outputs[0], _op.outputs[1]);
end;

class function gen_data_flow_ops.tensor_array_write_v3(handle, index, value, flow_in: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArrayWriteV3', name, [GetArg('handle',handle),
                                                                        GetArg('index',index),
                                                                        GetArg('value',value),
                                                                        GetArg('flow_in', flow_in)]);
    Result  := _op.output;
end;

{$ENDREGION}

{$REGION 'math_ops'}
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

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'einsum', @vValues),
                                function(v1: TNameScope): TFTensor
                                  begin
                                      var Args := ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(inputs.ToArray)]) ;
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
                                      Result := tf.Context.ExecuteOp('Einsum', name, Args).First;
                                  end );
end;

class function math_ops.equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.equal(x, y, True, name);
end;

class function math_ops.erf(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Erf', name, ExecuteOpArgs.Create([x])).First
end;

class function math_ops.floor(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Floor', name, ExecuteOpArgs.Create([x])).First
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

                                      start               := gen_array_ops.broadcast_to(start, broadcast_shape);
                                      stop                := gen_array_ops.broadcast_to(stop,  broadcast_shape);
                                      var expanded_start := array_ops.expand_dims(start, axis);
                                      var expanded_stop  := array_ops.expand_dims(stop,  axis);
                                      var shape := array_ops.shape(expanded_start);
                                      var ndims := array_ops.shape(shape)[0];
                                      var axis_tensor := array_ops.where_v2(constant_op.constant(axis >= 0), TObject(axis), TFtensor(TTensor(ndims) + axis));
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

                        Result := tf.Context.ExecuteOp('Pow', name, ExecuteOpArgs.Create([x_tensor, y_tensor])).First
                    end );
end;

class function math_ops.range(start: TValue; limit: PValue; delta: PValue; dtype: TF_DataType; name: string): TFTensor;
var
  tmp,tmp1 : TValue ;
begin
    if limit = nil then
    begin
        tmp := TValue.From<TValue>(start) ;
        tmp1 := TValue.From<Integer>(0);
        start := tmp1;

        tmp := tmp.AsType<TValue>;
        limit := @tmp;
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
                                                if delta = nil   then v := Integer(1)
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
            input_shape_val[axes_val] := NDArray(Integer(1));

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
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([TValue.From<Tscale>(scale), TValue.From<Tx>(x)])).First
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
   Result := tf.Context.ExecuteOp('Sin', name, ExecuteOpArgs.Create([x])).First
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
    var dims: TArray<TF_int64_t> := [];
    if ( not common_shapes.has_fully_defined_shape(_output) ) and ( not keepdims) and (axis = nil)  then
      _output.shape := TFShape.Create(dims);
    Result := _output;
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
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([x, y])).First
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
                                                var b_reshape          := bReshape.Value1;
                                                var b_free_dims        := bReshape.Value2;
                                                var b_free_dims_static := bReshape.Value3;

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
    Result := tf.Context.ExecuteOp('AddV2', name, ExecuteOpArgs.Create([x, y])).First
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
                                         .SetAttributes(['adj_x',adj_x,'adj_y',adj_y])).First;

                    end );
end;

class function math_ops.bincount(arr, weights, minlength, maxlength: TFTensor; dtype: TF_DataType; name: string; axis: PTFShape; binary_output: Boolean): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'bincount', nil),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;
                        if ( not binary_output) and (axis = nil) then
                        begin
                            var array_is_nonempty := TTEnsor( math_ops.reduce_prod(array_ops.shape(arr)) ) > int32(0);
                            var output_size := math_ops.cast(array_is_nonempty, Tdtypes.cint32) * (TTensor(math_ops.reduce_max(arr)) + 1);
                            if minlength <> nil then
                                output_size := math_ops.maximum(minlength, output_size);
                            if maxlength <> nil then
                                output_size := math_ops.minimum(maxlength, output_size);
                            var i := TArray<Int64>.Create();
                            var w := weights;
                            if w = nil then
                               w := constant_op.constant( TValue.From< TArray<Int64> >(i), dtype, 'Const' );

                            Result := tf.Context.ExecuteOp('Bincount', name, ExecuteOpArgs.Create([arr, output_size, w])).First;
                        end else
                        begin
                            var array_is_nonempty := TTEnsor( math_ops.reduce_prod(array_ops.shape(arr)) ) > int32(0);
                            var output_size := math_ops.cast(array_is_nonempty, Tdtypes.cint32) * (TTensor(math_ops.reduce_max(arr)) + 1);
                            if minlength <> nil then
                                output_size := math_ops.maximum(minlength, output_size);
                            if maxlength <> nil then
                                output_size := math_ops.minimum(maxlength, output_size);
                            var i := TArray<Int64>.Create();
                            var w := weights;
                            if w = nil then
                               w := constant_op.constant( TValue.From< TArray<Int64> >(i), dtype, 'Const' );

                            Result := tf.Context.ExecuteOp('DenseBincount', name, ExecuteOpArgs.Create([arr, output_size, w, binary_output])
                                                   .SetAttributes(['binary_output',binary_output])).First;
                        end;

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
   Result := tf.Context.ExecuteOp('Cos', name, ExecuteOpArgs.Create([x])).First
end;

class function math_ops.count_nonzero_v2(input: TFTensor; axis: TAxis; keepdims: Boolean; name: string; dtype: TF_DataType): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'count_nonzero', @input),
                      function(v1: TNameScope): TFTensor
                        begin
                            name := string(v1.ToString);
                            var zero := array_ops.zeros(TFShape.Scalar, input.dtype);
                            Result   := reduce_sum(cast(gen_math_ops.not_equal(input, zero), dtype), axis, keepdims);
                        end );
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
                                         .SetAttributes(['exclusive',exclusive,'reverse',reverse])).First;
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
{$ENDREGION}

{$REGION 'gen_nn_ops'}
{ gen_nn_ops }

class function gen_nn_ops.conv2d(parameters: Conv2dParams): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Conv2D', parameters.name, ExecuteOpArgs.Create([parameters.input, parameters.filter])
                                          .SetAttributes(['strides',          parameters.Strides,
                                                          'padding',          parameters.Padding,
                                                          'use_cudnn_on_gpu', parameters.UseCudnnOnGpu,
                                                          'explicit_paddings',parameters.ExplicitPaddings,
                                                          'data_format',      parameters.DataFormat,
                                                          'dilations',        parameters.Dilations]) ).First;
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
                                                          'dilations',        dilations]) ).First;
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
                                                          'dilations',        dilations]) ).First;
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
    Result := tf.Context.ExecuteOp( 'InTopKV2', name, ExecuteOpArgs.Create([predictions, targets, k]) ).First;
end;

class function gen_nn_ops.leaky_relu(features: TFTensor; alpha: single; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LeakyRelu', name, ExecuteOpArgs.Create([features])
                                                                        .SetAttributes(['alpha', alpha]) ).First;
end;

class function gen_nn_ops.leaky_relu_grad(gradients, features: TFTensor; alpha: Single; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('LeakyReluGrad', name, ExecuteOpArgs.Create([gradients,features])
                                                                        .SetAttributes(['alpha', alpha]) ).First;
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
    Result := tf.Context.ExecuteOp( 'LogSoftmax', name, ExecuteOpArgs.Create([logits]) ).First;
end;

class function gen_nn_ops.max_pool(input: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MaxPool', name, ExecuteOpArgs.Create([input])
                                                                        .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).First;
end;

class function gen_nn_ops.max_pool_grad(orig_input, orig_output, grad: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MaxPoolGrad', name, ExecuteOpArgs.Create([orig_input, orig_output, grad])
                                                                        .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).First;
end;

class function gen_nn_ops.relu(features: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Relu', name, ExecuteOpArgs.Create([features]) ).First;
end;

class function gen_nn_ops.relu_grad(gradients, features: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'ReluGrad', name, ExecuteOpArgs.Create([gradients, features]) ).First;
end;

class function gen_nn_ops.softmax(logits: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Softmax', name, ExecuteOpArgs.Create([logits]) ).First;
end;

class function gen_nn_ops.softmax_cross_entropy_with_logits(features, labels: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin
    var Res := tf.Context.ExecuteOp( 'SoftmaxCrossEntropyWithLogits', name, ExecuteOpArgs.Create([features, labels]) );

    Result := Tuple<TFTensor, TFTensor>.Create(Res[0],Res[1]);
end;

class function gen_nn_ops.sparse_softmax_cross_entropy_with_logits(features, labels: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin
    var Res := tf.Context.ExecuteOp( 'SparseSoftmaxCrossEntropyWithLogits', name, ExecuteOpArgs.Create([features, labels]) );

    Result := Tuple<TFTensor, TFTensor>.Create(Res[0],Res[1]);
end;

class function gen_nn_ops.tanh(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Tanh', name, ExecuteOpArgs.Create([x]) ).First;
end;

class function gen_nn_ops.top_kv2<T>(input: TFTensor; k: T; sorted: Boolean; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('TopKV2', name,[ GetArg('input',input), GetArg('k', TValue.From<T>(k)), GetArg('sorted',sorted) ] );
    Result := _op.outputs;
end;

class function gen_nn_ops.average_pool(input: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('AvgPool', name, ExecuteOpArgs.Create([input])
                                          .SetAttributes(['ksize', ksize,'strides', strides,'padding', padding,'data_format', data_format ]) ).First;
end;

class function gen_nn_ops.bias_add(value: TFTensor; bias: IVariableV1; data_format, name: string): TFTensor;
begin
    if data_format = '' then
      data_format := 'NHWC';

    Result := tf.Context.ExecuteOp('BiasAdd', name, ExecuteOpArgs.Create([value, TValue.From<IVariableV1>(bias)])
                                          .SetAttributes(['data_format', data_format ]) ).First;
end;

class function gen_nn_ops.bias_add_grad(out_backprop: TFTensor; data_format, name: string): TFTensor;
begin
    if data_format = '' then
      data_format := 'NHWC';

    Result := tf.Context.ExecuteOp('BiasAddGrad', name, ExecuteOpArgs.Create([out_backprop])
                                                                        .SetAttributes(['data_format', data_format ]) ).First;
end;
{$ENDREGION}

{$REGION 'nn_ops'}
{ nn_ops }

class function nn_ops.convolution_internal(padding: string; strides, dilation_rate: TArray<Integer>; rank: Integer; name, data_format: string): ConvolutionInternal;
var
  args: ConvolutionalArgs;
begin
    args := ConvolutionalArgs.Create;

    args.Rank         := rank;
    args.Padding      := padding;
    args.Strides      := strides;
    args.DilationRate := dilation_rate;
    args.DataFormat   := data_format;
    args.Name         := name;

    Result := ConvolutionInternal.Create(args)
end;

class function nn_ops.bias_add(value: TFTensor; bias: IVariableV1; data_format, name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([value, TValue.From<IVariableV1>(bias)]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'BiasAdd', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 name   := v1.ToString;
                                 Result := gen_nn_ops.bias_add(value, bias, data_format, name);
                            end );
end;

class function nn_ops.dropout_v2(x, rate, noise_shape: TFTensor; seed: pInteger; name: string): TFTensor;
begin
    var newVal : TValue := x;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'dropout', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                                if not Tdtypes.is_floating(x.dtype) then
                                   raise TFException.Create('Not Implemented, x has to be a floating point tensor since it''s going to be scaled. Got a {x.dtype} tensor instead.');
                                var keep_prob := 1 - TTEnsor(rate);
                                var scale     := 1 / keep_prob;
                                var scale_tensor := Tops.convert_to_tensor(scale, x.dtype);
                                gen_math_ops.mul(x, scale_tensor);
                                noise_shape := _get_noise_shape(x, noise_shape);
                                // Sample a uniform distribution on [0.0, 1.0) and select values larger than
                                // rate.
                                //
                                // NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
                                // and subtract 1.0.
                                var random_tensor := random_ops.random_uniform(noise_shape, 0,nil, x.dtype ,seed);
                                // NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
                                // float to be selected, hence we use a >= comparison.
                                var keep_mask := TTEnsor(random_tensor) >= rate;
                                var ret : TFTensor := x * scale * math_ops.cast(keep_mask, x.dtype);
                                if  not tf.executing_eagerly then
                                    ret.shape := x.shape;
                                Result := ret;
                            end );
end;

class function nn_ops.in_top_k(predictions, targets: TFTensor; k: Integer; name: string): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'in_top_k', nil),
                          function(v1: TNameScope): TFTensor
                            begin
                                Result := gen_nn_ops.in_top_kv2(predictions, targets, k, name);
                            end );
end;

class function nn_ops.l2_loss(t: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('L2Loss', name, ExecuteOpArgs.Create([ t ])).First;
end;

class function nn_ops.softplus(features: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Softplus', name, ExecuteOpArgs.Create([ features ])).First;
end;

class function nn_ops.leaky_relu(features: TFTensor; alpha: Single; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([features, alpha]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'LeakyRelu', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                features := Tops.convert_to_tensor(features, DtInvalid, 'features');
                                if TDTypes.is_integer(features.dtype) then
                                    features := math_ops.cast(features, Tdtypes.cfloat32);
                                Result := gen_nn_ops.leaky_relu(features, alpha, name);
                                //return math_ops.maximum(alpha * features, features, name: name);
                            end );
end;

class function nn_ops.log_softmax(logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
     Result :=  _softmax(logits, gen_nn_ops.log_softmax, axis, name);
end;

class function nn_ops.softmax(logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
    Result := _softmax(logits, gen_nn_ops.softmax, axis, name);
end;


class function nn_ops.max_pool(value: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    var newVal : TValue := value;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MaxPool', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                value := Tops.convert_to_tensor(value, DtInvalid, 'input');
                                Result := gen_nn_ops.max_pool(value,
                                                              ksize,
                                                              strides,
                                                              padding,
                                                              data_format,
                                                              name);
                            end );
end;

class function nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
    {$HINTS OFF}
    var newVal : TValue := TValue.From< TArray<TValue> >([logits, labels]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'softmax_cross_entropy_with_logits', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;

                                var precise_logits := logits;
                                var input_rank     := array_ops.rank(precise_logits);
                                var shape          := logits.shape;
                                if axis <> -1 then
                                   raise TFException.Create('Not Implemented softmax_cross_entropy_with_logits_v2_helper axis <> -1');
                                var input_shape := array_ops.shape(precise_logits);
                                // Make precise_logits and labels into matrices.
                                precise_logits := _flatten_outer_dims(precise_logits);
                                labels         := _flatten_outer_dims(labels);
                                // Do the actual op computation.
                                // The second output tensor contains the gradients.  We use it in
                                // _CrossEntropyGrad() in nn_grad but not here.
                                var tt := gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name);
                                var cost            := tt.Value1;
                                var unused_backprop := tt.Value2;
                                // The output cost shape should be the input minus axis.
                                var output_shape := array_ops.slice(input_shape, TArray<TFTensor>.Create(constant_op.constant(0)),
                                                                                 TArray<TFTensor>.Create( math_ops.subtract(input_rank, 1) ) );
                                cost := array_ops.reshape(cost, output_shape);
                                Result := cost;
                            end );
end;

class function nn_ops.sparse_softmax_cross_entropy_with_logits(labels, logits: TFTensor; name: string): TFTensor;
begin
    {$HINTS OFF}
    var newVal : TValue := TValue.From< TArray<TValue> >([labels, logits]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'SparseSoftmaxCrossEntropyWithLogits', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                labels := tops.convert_to_tensor(labels);
                                logits := tops.convert_to_tensor(logits);

                                var precise_logits : TFTensor;
                                if logits.dtype = TF_HALF then precise_logits := math_ops.cast(logits, tdtypes.Cfloat32)
                                else                           precise_logits := logits;
                                // Store label shape for result later.
                                var labels_static_shape := labels.shape;
                                var labels_shape        := array_ops.shape(labels);
                                (*bool static_shapes_fully_defined = (
                                    labels_static_shape.is_fully_defined() &&
                                        logits.get_shape()[:-1].is_fully_defined());*)
                                // Check if no reshapes are required.
                                if logits.shape.ndim = 2 THEN
                                begin
                                    var tCost := gen_nn_ops.sparse_softmax_cross_entropy_with_logits(precise_logits, labels, name);
                                    var cost := tCost.Value1;
                                    var _    := tCost.Value2;
                                    if logits.dtype = Tdtypes.cfloat16 then
                                        Result := math_ops.cast(cost, Tdtypes.cfloat32)
                                    else
                                        Result := cost;
                                    Exit;
                                end;
                                // Perform a check of the dynamic shapes if the static shapes are not fully
                                // defined.
                                raise TFException.Create('Not Implemented sparse_softmax_cross_entropy_with_logits');
                            end );
end;

class function nn_ops.top_kv2(input: TFTensor; k: Integer; sorted: Boolean; name: string): TFTensors;
begin
     Result :=  tf.Context.ExecuteOp('TopKV2', name, ExecuteOpArgs.Create([input, k])
                                .SetAttributes(['sorted', sorted ]));
end;

class function nn_ops._flatten_outer_dims(logits: TFTensor): TFTensor;
begin
    var rank := array_ops.rank(logits);

    var last_dim_size := array_ops.slice(array_ops.shape(logits), TArray<TFTensor>.Create(math_ops.subtract(rank, 1)),
                                                                  TArray<TFTensor>.Create(constant_op.constant(1) ) );

    var ops    := array_ops.concat([ [-1] , last_dim_size ], 0);
    var output := array_ops.reshape(logits, ops);
    // Set output shape if known.
    if not tf.Context.executing_eagerly then
    begin
        var shape := logits.shape;
        if (not shape.IsNil) and (shape.ndim > 0)  then
        begin
            var product: Int64 := 1;
            var product_valid  := true;
            var eDim := Enumerable<Int64>.create(shape.dims) ;
            for var d in eDim.Take(shape.ndim - 1) do
            begin
                if d = -1 then
                begin
                    product_valid := false;
                    break;
                end else
                begin
                    product := product * d;
                end
            end;
            if product_valid then
            begin
                var output_shape := [ product ];
                raise TFException.Create('Not Implemented _flatten_outer_dims product_valid');
            end;
        end;
    end;
    Result := output;
end;

class function nn_ops._get_noise_shape(x, noise_shape: TFTensor): TFTensor;
begin
    if noise_shape = nil then  Result := array_ops.shape(x)
    else                       Result := noise_shape;
end;

class function nn_ops._softmax(logits: TFTensor; compute_op: TFunc<TFTensor, string, TFTensor>; dim: Integer; name: string): TFTensor;
begin
    logits := Tops.convert_to_tensor(logits);

    var shape := logits.shape;
    var is_last_dim : Boolean := (dim = -1) or (dim = shape.ndim - 1);
    if is_last_dim then
        Exit( compute_op(logits, name) );
    raise TFException.Create('Not Implemented _softmax helper');
end;

{$ENDREGION}

{$REGION 'nn_impl'}
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

class function nn_impl.batch_normalization(const x, mean, variance, offset, scale: TFTensor; variance_epsilon: Single; name: string): TFTensor;
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
                    var neg_abs_logits:= array_ops.where(cond, TFTensor(-TTensor(logits)), logits);

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
{$ENDREGION}

{$REGION 'bitwise_ops'}
{ bitwise_ops }

function bitwise_ops.binary_op(x, y: TFTensor; opName, name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp(opName, name, ExecuteOpArgs.Create([x,y])).First;
end;

function bitwise_ops.unary_op(x: TFTensor; opName, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp(opName, name, ExecuteOpArgs.Create([x])).First;
end;

function bitwise_ops.bitwise_and(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseAnd', name);
end;

function bitwise_ops.bitwise_or(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseOr', name);
end;

function bitwise_ops.bitwise_xor(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseXor', name);
end;

function bitwise_ops.invert(x: TFTensor; name: string): TFTensor;
begin
   Result := unary_op(x, 'Invert', name);
end;

function bitwise_ops.left_shift(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'LeftShift', name);
end;

function bitwise_ops.right_shift(x, y: TFTensor; name: string): TFTensor;
begin
     Result := binary_op(x, y, 'RightShift', name);
end;
{$ENDREGION}

{$REGION 'string_ops'}
{ string_ops }

function string_ops.lower(input: TFTensor; encoding, name: String): TFTensor;
begin
     Result := tf.Context.ExecuteOp('StringLower', name, ExecuteOpArgs.Create([ input, encoding])).First;
end;

function string_ops.regex_replace(input: TFTensor; pattern, rewrite: string; replace_global: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StaticRegexReplace', name, ExecuteOpArgs.Create([ input ])
                         .SetAttributes(['pattern',pattern,'rewrite',rewrite,'replace_global',replace_global])).First;
end;

function string_ops.string_length(input: TFTensor; name, &unit: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ input ]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;
                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr<string>('unit');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp('StringLength', name, Args
                           .SetAttributes(['unit', &unit ])).First;
end;

function string_ops.substr<T>(input: T; pos, len: Integer; uint, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Substr', name, ExecuteOpArgs.Create([ TValue.From<T>(input), Pos, len]).SetAttributes(['unit',uint])).First;
end;

function string_ops.string_format(inputs: TArray<TFTensor>; template, placeholder: string; summarize: Integer; name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;

                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr('T');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'template' ;
                                   pParam.vValue:= op.get_attr<string>('template');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'placeholder' ;
                                   pParam.vValue:= op.get_attr<string>('placeholder');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'summarize' ;
                                   pParam.vValue:= op.get_attr<Integer>('summarize');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp('StringFormat', name, Args
                           .SetAttributes(['template',template, 'placeholder',placeholder, 'summarize', summarize ])).First;
end;

function string_ops.string_split_v2(input: TFTensor; sep: string; maxsplit: Integer; name: string): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'StringSplit'),
                          function(v1: TNameScope): RaggedTensor
                            begin
                                Tops.convert_to_tensor(sep, TF_DataType.TF_STRING);
                                if input.rank = 0 then
                                begin
                                    var parts := string_split_v2(array_ops.stack([ input ]), sep, maxsplit, name);
                                    Result := parts;
                                    exit;
                                end;

                                var Args := ExecuteOpArgs.Create([input, sep]) ;

                                Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                         begin
                                             Result := [];
                                             var pParam : TParameter;

                                             pParam.sNome := 'maxsplit' ;
                                             pParam.vValue:= op.get_attr<Integer>('maxsplit');
                                             Result := Result + [ pParam ] ;
                                         end;

                                var Res := tf.Context.ExecuteOp('StringSplitV2', name, Args.SetAttributes(['maxsplit',maxsplit ]));
                                var indices := res[0];
                                var values  := res[1];
                                var shape   := res[2];

                                indices.shape := TFShape.Create([-1, 2]);
                                values.shape  := TFShape.Create([-1]);
                                shape.shape   := TFShape.Create([2]);
                                var sparse_res := TSparseTensor.Create(indices, values, shape);
                                Result := RaggedTensor.from_value_rowids(sparse_res.values, sparse_res.indices[[Slice.All, 0]], sparse_res.dense_shape[0], '', false);
                            end );
end;

function string_ops.unicode_decode_with_offsets(input: TFTensor; input_encoding, errors: string; replacement_char: Integer; replace_control_characters: Boolean;
  name: string): Tuple<RaggedTensor, RaggedTensor>;
begin
     Result := TUtils.tf_with<TNameScope,Tuple<RaggedTensor, RaggedTensor>>( TOps.name_scope(name, 'UnicodeDecodeWithOffsets'),
                          function(v1: TNameScope): Tuple<RaggedTensor, RaggedTensor>
                            begin
                                var t := _unicode_decode(input, input_encoding, errors, replacement_char, replace_control_characters, true, name);
                                var codepoints := t.value1;
                                var byte_start_offsets := t.value2;
                                Result := Tuple<RaggedTensor, RaggedTensor>.Create(codepoints, byte_start_offsets);
                            end);
end;

function string_ops._unicode_decode(input: TFTensor; input_encoding, errors: string; replacement_char: Integer; replace_control_characters, with_offsets: Boolean;
  name: string): Tuple<RaggedTensor, RaggedTensor>;
begin
    if with_offsets then
    begin
        var Args := ExecuteOpArgs.Create([input]);
        Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                     begin
                                         Result := [];
                                         var pParam : TParameter;

                                         pParam.sNome := 'input_encoding' ;
                                         pParam.vValue:= op.get_attr<String>('input_encoding');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'errors' ;
                                         pParam.vValue:= op.get_attr<String>('errors');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'replacement_char' ;
                                         pParam.vValue:= op.get_attr<Integer>('replacement_char');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'replace_control_characters' ;
                                         pParam.vValue:= op.get_attr<Boolean>('replace_control_characters');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'Tsplits' ;
                                         pParam.vValue:= op.get_attr('Tsplits');
                                         Result := Result + [ pParam ] ;
                                     end;

        var flat_result := tf.Context.ExecuteOp('RaggedTensorToVariant', name, Args
                      .SetAttributes(['input_encoding',input_encoding,'errors',errors,'replacement_char',replacement_char,'replace_control_characters',replace_control_characters]));

        var codepoints := RaggedTensor.from_row_splits(flat_result[1], flat_result[0],'', false);
        var offsets    := RaggedTensor.from_row_splits(flat_result[2], flat_result[0], '',false);
        Result         :=  Tuple<RaggedTensor, RaggedTensor>.Create(codepoints, offsets);
    end;
    Result :=  Tuple<RaggedTensor, RaggedTensor>.Create(nil,nil);
end;
{$ENDREGION}

{$REGION 'control_flow_ops'}
{ control_flow_ops }

class function control_flow_ops._SwitchRefOrTensor(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    data := Tops.convert_to_tensor_or_composite(data, DtInvalid, 'data');
    // NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
    // addresses the following scenario.
    //
    // Assume you execute Optimizer.apply_gradients() in a branch of a cond().
    //
    // 1. The update op is created inside a `with ops.colocate(var):` block
    //
    // 2. Some tensor `data` is captured and a switch is created in a
    //    `with ops.colocate_with(data):` block.
    //
    // with ops.colocate_with(var):
    //  with ops.colocate_with(data):
    //    op = ...
    //
    // var and data may be pinned to different devices, so we want to ops
    // created within ops.colocate_with(data) to ignore the existing stack.
    Tops.colocate_with(data, true);
    begin
        if data is TFTensor then
        begin
            if TDtypes.is_ref_dtype(data.dtype) then
            begin
                Result := gen_control_flow_ops.ref_switch(data, pred, name);
                Exit;
            end;
        end;
        Result := switch(data, pred, DtInvalid, name);
    end
end;

class function control_flow_ops.while_loop<TItem>(_cond: TFunc<TItem, TFTensor>; body: TFunc<TItem, TItem>; loop_vars: TItem; shape_invariants: TArray<TFShape>;
  parallel_iterations: Integer; back_prop, swap_memory: Boolean; name: string; maximum_iterations: TFTensor; return_same_structure: Boolean): TItem;
var
  orig_cond : TFunc<TItem, TFTensor>;
  orig_body : TFunc<TItem, TItem>;
begin
    var vValue : TValue := TValue.From<TItem>(loop_vars);
    TUtils.tf_with<TNameScope,TItem>(Tops.name_scope(name, 'while', @vValue), function(Scope: TNameScope): TItem
            Begin
                if loop_vars = nil then
                   raise Exception.Create('No loop variables provided');
                if not Assigned(_cond) then
                   raise Exception.Create('cond must be callable.');
                if not Assigned(body) then
                   raise Exception.Create('body must be callable.');
                if parallel_iterations < 1 then
                   raise Exception.Create('parallel_iterations must be a positive integer.');

                var try_to_pack := (loop_vars is TFTensor) and (not return_same_structure);
                var counter := constant_op.constant(0, maximum_iterations.dtype, 'iteration_counter');
                orig_cond := _cond;
                orig_body := body;

                var loop_vars_1 : LoopVar<TItem>  := nil;
                var body_buildloop : TFunc<LoopVar<TItem>, LoopVar<TItem>> := nil;
                var cond_buildloop : TFunc<LoopVar<TItem>, TFTensor>       := nil;

                if try_to_pack then
                begin

                end else
                begin
                    loop_vars_1 := LoopVar<TItem>.Create(counter, loop_vars);
                    cond_buildloop := function(item: LoopVar<TItem>) : TFTensor
                        begin
                            var i := item.Counter;
                            var lv:= item.Item;
                            var oc := orig_cond(lv);
                            Result := math_ops.logical_and(i < TTensor(maximum_iterations), oc);
                            Exit;
                        end;

                    body_buildloop := function(item:  LoopVar<TItem>):  LoopVar<TItem>
                        begin
                            var i := item.Counter;
                            var lv:= item.Item;
                            var ob := orig_body(lv);
                            Result := LoopVar<TItem>.Create(TTensor(i) + 1, ob);
                            Exit;
                        end;
                end;
                try_to_pack := false;

                var loop_context := WhileContext.Create(maximum_iterations, parallel_iterations, back_prop, swap_memory);

                if loop_context.outer_context = nil then
                    Tops.add_to_collection(tf.GraphKeys.WHILE_CONTEXT, loop_context);

                var res : LoopVar<TItem> := loop_context.BuildLoop<TItem>(cond_buildloop, body_buildloop, loop_vars_1, shape_invariants, return_same_structure);

                //if (maximum_iterations != null)
                Result := res.Item;
                //else
                //return results;
            end);
end;

class function control_flow_ops.merge(inputs: TArray<TFTensor>; name: string): MergeOutput;
begin
    for var i := 0 to Length(inputs) - 1 do
    begin
        if inputs[i] = nil then
          raise Exception.Create('At least one of the merge inputs is null: {inputs}');
    end;

    var vvalue : TArray<TValue>:= [ TValue.From<  TArray<TFTensor> >(inputs) ];
    Result := TUtils.tf_with<TNameScope,MergeOutput>(Tops.name_scope(name, 'Merge', @vvalue), function(scope: TNameScope): MergeOutput
                  begin
                       name := scope.ToString;
                       var a : TArray<TFTensor> := [];
                       for var i := 0 to Length(inputs) - 1 do
                       begin
                           var inp : TFTensor := inputs[i];
                           inp := Tops.internal_convert_to_tensor_or_indexed_slices(inp, DtInvalid, '', true);
                           a := a + [ inp ];
                       end;
                       Result := gen_control_flow_ops.merge(a, name);
                  end);
end;

class function control_flow_ops.cond(pred: TFTensor; true_fn, false_fn: TFunc<TFTensor>; name: string): TFTensor;
begin
    var vvalue : TArray<TValue>:= [ TValue.From< TFTensor >(pred) ];
    result := TUtils.tf_with<TNameScope,TFTensor>(Tops.name_scope(name, 'cond',@vvalue), function(scope: TNameScope): TFTensor
                  begin
                      if tf.Context.executing_eagerly then
                      begin
                          var bTensor : TTensor := pred;
                          var flag : Boolean := Boolean(bTensor);
                          if flag then
                          begin
                              Result := true_fn;
                              Exit;
                          end else
                          begin
                             Result := false_fn;
                             Exit;
                          end;
                      end;

                      // Add the Switch to the graph.
                      var switch_result := switch(pred, pred);
                      var p_2 := switch_result[0];
                      var p_1 := switch_result[1];
                      var pivot_1 := array_ops.identity(p_1, 'switch_t');
                      var pivot_2 := array_ops.identity(p_2, 'switch_f');
                      pred := array_ops.identity(pred, 'pred_id');

                      // Disable the fetching of tensors that are only on one branch of cond.
                      var aTensor : TArray<TFTensor>:=  [ p_1, p_2, pivot_1, pivot_2, pred ];
                      for var tensor in aTensor do
                          tensor.op.graph.prevent_fetching(tensor.op);

                      // Build the graph for the true branch in a new context.
                      var context_t := CondContext.Create(pred, pivot_1, 1);
                      var orig_res_t : ITensorOrOperation ;
                      var res_t      : TFTensor;
                      try
                          context_t.Enter_;
                          var tTupleBranch := context_t.BuildCondBranch<TFTensor>(true_fn);
                          orig_res_t := tTupleBranch.Value1;
                          res_t      := tTupleBranch.Value2;
                          context_t.ExitResult([ res_t ]);
                      finally
                          context_t.Exit_;
                      end ;
                      // Build the graph for the false branch in a new context.
                      var context_f  := CondContext.Create(pred, pivot_2, 0);
                      var orig_res_f : ITensorOrOperation ;
                      var res_f      : TFTensor ;
                      try
                          context_f.Enter_;
                          var fTupleBranch := context_f.BuildCondBranch<TFTensor>(false_fn);
                          orig_res_f := fTupleBranch.Value1;
                          res_f      := fTupleBranch.Value2;
                          context_f.ExitResult([ res_f ]);
                      finally
                          context_f.Exit_;
                      end;

                      var res_t_flat := [ res_t ];
                      var res_f_flat := [ res_f ];

                      var tMerge := merge([ res_t_flat[0], res_f_flat[0] ] )[0];
                      var merges : TArray<TFTensor> := [ tMerge ];

                      if orig_res_t is TFTensor then
                      begin
                         {var orig_res_tensor := orig_res_t as TFTensor;
                         var aRes := _convert_flows_to_tensorarrays([ orig_res_tensor ], merges);
                         merges := [];
                         for var i := 0 to Length(aRes) - 1 do
                         begin
                             merges := merges + [ aRes[i] as TFTensor ];
                         end; }
                      end else
                      begin

                      end;

                      if context_t.outer_context = nil then
                      begin
                          Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                          Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                      end;

                      Result := merges[0];
                  end);
end;

class function control_flow_ops.cond<T>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<T>>; name: string): TArray<TFTensor>;
var
  aRes : TArray<ITensorOrTensorArray>;
begin
    var vvalue : TArray<TValue>:= [ TValue.From< TFTensor >(pred) ];
    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>(Tops.name_scope(name, 'cond', @vvalue), function(scope: TNameScope): TArray<TFTensor>
                begin
                    if tf.Context.executing_eagerly then
                    begin
                        var bTensor : TTensor := pred;
                        var flag : Boolean := Boolean(bTensor);
                        if flag then
                        begin
                            var a : TArray<T> := true_fn;
                            var v := TValue.From<TArray<T>>(a);
                            Result := v.astype< TArray<TFTensor> >;
                            Exit;
                        end else
                        begin
                            var a : TArray<T> := false_fn;
                            var v := TValue.From<TArray<T>>(a);
                            Result := v.astype< TArray<TFTensor> >;
                            Exit;
                        end;
                    end;

                    // Add the Switch to the graph.
                    var switch_result := switch(pred, pred);
                    var p_2 := switch_result[0];
                    var p_1 := switch_result[1];
                    var pivot_1 := array_ops.identity(p_1, 'switch_t');
                    var pivot_2 := array_ops.identity(p_2, 'switch_f');
                    pred := array_ops.identity(pred, 'pred_id');

                    // Disable the fetching of tensors that are only on one branch of cond.
                    var aTensor : TArray<TFTensor>:=  [ p_1, p_2, pivot_1, pivot_2, pred ];
                    for var tensor in aTensor do
                        tensor.op.graph.prevent_fetching(tensor.op);

                    // Build the graph for the true branch in a new context.
                    var context_t := CondContext.Create(pred, pivot_1, 1);
                    var orig_res_t : TArray<T> ;
                    var res_t      : TArray<TFTensor>;
                    try
                        context_t.Enter_;
                        var tTupleBranch := context_t.BuildCondBranch<T>(true_fn);
                        orig_res_t := tTupleBranch.Value1;
                        res_t      := tTupleBranch.Value2;
                        context_t.ExitResult(res_t );
                    finally
                        context_t.Exit_;
                    end ;
                    // Build the graph for the false branch in a new context.
                    var context_f  := CondContext.Create(pred, pivot_2, 0);
                    var orig_res_f : TArray<T> ;
                    var res_f      : TArray<TFTensor> ;
                    try
                        context_f.Enter_;
                        var fTupleBranch := context_f.BuildCondBranch<T>(false_fn);
                        orig_res_f := fTupleBranch.Value1;
                        res_f      := fTupleBranch.Value2;
                        context_f.ExitResult(res_f );
                    finally
                        context_f.Exit_;
                    end;

                    var res_t_flat :=  res_t ;
                    var res_f_flat :=  res_f ;

                    var merges : TArray<TFTensor> := [];
                    for var i := 0 to Length(res_t_flat) - 1 do
                    begin
                        var tMerge := merge([ res_t_flat[i], res_f_flat[i] ] )[0];
                        merges :=  merges + [ tMerge ];
                    end;

                    if TypeInfo(T) = TypeInfo(TFTensor) then
                    begin
                       var orig_res_tensor : TArray<ITensorOrTensorArray>;
                       for var i := 0 to Length(orig_res_t) - 1 do
                       begin
                           var v := TValue.From<T>(orig_res_t[i]);
                           orig_res_tensor := orig_res_tensor + [ v.AsType<TFTensor> ] ;
                       end;

                       {aRes := _convert_flows_to_tensorarrays(orig_res_tensor, merges);
                       merges := [];
                       for var i := 0 to Length(aRes) - 1 do
                           merges := merges + [ aRes[i] as TFTensor ]; }
                    end
                    else if TypeInfo(T) = TypeInfo(Single) then
                    begin

                    end else
                    begin

                    end;

                    if context_t.outer_context = nil then
                    begin
                        Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                        Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                    end;

                    Result := merges;

               end);
end;

class function control_flow_ops.group<T>(inputs: TArray<T>; name: string): TFOperation;
begin
    var vInputs := TValue.From< TArray<T> >(inputs) ;

    Result := TUtils.tf_with<TNameScope,TFOperation>( TOps.name_scope(name, 'group_deps', @vInputs),
                function(v1: TNameScope): TFOperation
                  begin
                      name := v1.ToString;

                      // Sorts *inputs according to their devices.
                      var ops_on_device := TDictionary< string, TList<T> >.Create;
                      for var inp in inputs do
                      begin
                          if ops_on_device.ContainsKey(inp.Device) then
                              ops_on_device[inp.Device].Add(inp)
                          else
                              ops_on_device.add( inp.Device, TList<T>.Create([ inp ]) );
                      end;
                      // 1-level tree. The root node is the returned NoOp node.
                      if ops_on_device.Count = 1 then
                      begin
                          var dev  := ops_on_device.Keys.ToArray[0];
                          var deps := ops_on_device.Values.ToArray[0];
                          var aOp : TArray<TFOperation> := [];
                          for var i := 0 to deps.Count - 1 do
                             aOp := aOp + [ deps[i].op ];
                          Result := _GroupControlDeps(dev, aOp, name);
                          Exit;
                      end;
                      // 2-level tree. The root node is the returned NoOp node.
                      // deps contains 1 NoOp node for each device.
                      raise TFException.Create('control_flow_ops.group');

                  end );

end;

class function control_flow_ops.IsLoopExit(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Exit') or (op.Tipo = 'RefExit');
end;

class function control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean): ControlFlowState;
begin
    var loop_state : ControlFlowState := nil;
    var pos : Integer := 0;
    while pos < between_op_list.Count do
    begin
        var op := between_op_list[pos];
        if IsLoopExit(op) then
        begin
            if loop_state = nil then
            begin
                loop_state := ControlFlowState.Create;
            end;
            if colocate_gradients_with_ops then
                Tops.colocate_with(op);
            loop_state.AddWhileContext(op, between_op_list, between_ops);
        end;
        Inc(pos);
    end;
    Result := loop_state;
end;

class function control_flow_ops.no_op(name: string): TFOperation;
begin
    Result := gen_control_flow_ops.no_op(name)
end;

class function control_flow_ops.switch(data, pred: TFTensor; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin
    var vInputs := TValue.From< TArray<TFTensor> >([data, pred]) ;

    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope(name, 'Switch', @vInputs),
                function(v1: TNameScope): TArray<TFTensor>
                  begin
                      name := v1.ToString;
                      data := Tops.internal_convert_to_tensor_or_indexed_slices(data, dtype, 'data', true);

                      pred := Tops.convert_to_tensor(pred, DtInvalid, 'pred');
                      Result := gen_control_flow_ops.switch(data, pred, name);
                  end );
end;

class function control_flow_ops.tuple(tensors: TArray<TFTensor>; name: string; control_inputs: TArray<TFOperation>): TArray<TFTensor>;
begin
    var vInputs := TValue.From< TArray<TFTensor> >(tensors) ;

    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope(name, 'tuple', @vInputs),
                function(v1: TNameScope): TArray<TFTensor>
                  begin
                      name := v1.ToString;


                      var gating_ops : TArray<TFOperation>:= [];
                      for var i := 0 to Length(tensors)-1 do
                      begin
                          if tensors[i] <> nil then
                             gating_ops := gating_ops + [ tensors[i].Op ];
                      end;

                      if control_inputs <> nil  then
                      begin
                          for var c in control_inputs do
                             gating_ops := gating_ops + [ c ];
                      end;
                      // Note that in order to ensure ordering in the pbtxt, we must take care to
                      // ensure the order here.
                      var l_gating_ops := Enumerable<TFOperation>.create(gating_ops);
                      l_gating_ops := l_gating_ops.OrderBy<Integer>(function (o: TFOperation): Integer
                                                                      begin
                                                                        Result := o.id;
                                                                      end);
                      var gate := group<TFOperation>(l_gating_ops.ToArray);
                      var tpl := TList<TFTensor>.Create ;
                      try
                        for var t in tensors do
                        begin
                            if t <> nil then tpl.Add( with_dependencies([ gate ], t) )
                            else             tpl.Add(nil);
                        end;
                        Result := tpl.ToArray;
                      finally
                        tpl.Free;
                      end;

                  end );
end;

class function control_flow_ops.with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string): TFTensor;
begin
   var adeps : TArray<TValue> := [];
   var aValue : TArray<TValue> := [];

   for var i := 0 to Length(dependencies) -1  do
     adeps := adeps + [ TValue.From<TFOperation>(dependencies[i]) ] ;

   aValue := adeps + [ output_tensor ];

   //TODO: missing original code
   //if context.executing_eagerly():
   //    return output_tensor
   Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'control_dependency', @aValue),
                function(v1: TNameScope): TFTensor
                  begin
                      name := v1.ToString;

                      Tops.colocate_with(output_tensor);
                      Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(adeps),
                                  function(v1: TControlDependenciesController): TFTensor
                                    begin
                                        output_tensor := Tops.convert_to_tensor_or_composite(output_tensor);
                                        Result := _Identity(output_tensor, name);
                                    end );
                  end );
end;

class function control_flow_ops.ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
begin
    var val := op.outputs[index];
    if not control_flow_util.IsSwitch(op) then
    begin
        if val.dtype = TF_DataType.TF_RESOURCE then
           raise TFException.Create('Not Implemented - ("ZerosLikeOutsideLoop")');
        Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end else
    begin
        var op_ctxt := op._get_control_flow_context;
        if op_ctxt <> nil then
        begin
            // We are in a cond context. Use a switch to create zeros only when needed.
            var pred   := op_ctxt.pred;
            var branch := op_ctxt.branch;
            var switch_val := switch(op.inputs[0], pred)[1 - branch];
            var pivot      := array_ops.identity(switch_val);
            if val.dtype = TDtypes.cresource then
               raise TFException.Create('Not Implemented');
            var zeros_shape := array_ops.shape_internal(switch_val,'', false);
            // Ensure ops created within array_ops.zeros are dominated by switch in
            // cond context.
            var aValue : TArray<TValue> := [pivot];
            Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(aValue),
                          function(v1: TControlDependenciesController): TFTensor
                            begin
                                Result := array_ops.zeros(zeros_shape, val.dtype);
                            end );
        end else
        begin
            Result := array_ops.zeros_like(val, DtInvalid,'', false);
        end;
    end;
end;

class function control_flow_ops._convert_flows_to_tensorarrays(tensors_or_tensorarrays: TArray<ITensorOrTensorArray>; tensors_or_flows: TArray<TFTensor>): TArray<ITensorOrTensorArray>;
begin
    Result := [];
    for var i := 0 to Length(tensors_or_tensorarrays)- 1 do
    begin
         var ta       := tensors_or_tensorarrays[i];
         var t_or_flow : TFTensor := tensors_or_flows[i];

         if ta is TTensorArray  then
         begin
            var ta_1 := ta as TTensorArray;
            var res := tensor_array_ops.build_ta_with_new_flow(ta_1, t_or_flow)  ;
            Result := Result + [ res ] ;
         end else
         begin
             var res := t_or_flow ;
             Result := Result + [ res ] ;
         end;
    end;
end;

class function control_flow_ops._GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string): TFOperation;
begin
   var aValue : TArray<TValue> := [];
   for var i := 0 to Length(deps) -1  do
     aValue := aValue + [ TValue.From<TFOperation>(deps[i]) ] ;

   Result := TUtils.tf_with<TControlDependenciesController,TFOperation>( Tops.control_dependencies(aValue),
                                          function(v1: TControlDependenciesController): TFOperation
                                            begin
                                                if dev = '' then
                                                  Result := gen_control_flow_ops.no_op(name)
                                                else
                                                   Result := gen_control_flow_ops.no_op(name);
                                            end );
end;

class function control_flow_ops._Identity(data: TFTensor; name: string): TFTensor;
begin
    data := Tops.internal_convert_to_tensor_or_composite(data, DtInvalid, '', true);
    if Ord(data.dtype) > 100 then
       raise TFException.Create('Not Implemented "_Identity"')
    else
        Result := gen_array_ops.identity(data, name);
end;

class function control_flow_ops._NextIteration(data: TFTensor; name: string): TFTensor;
begin
    data := Tops.internal_convert_to_tensor_or_indexed_slices(data, DtInvalid, '', true);

    if TDTypes.is_ref_dtype(data.dtype) then  Result := gen_control_flow_ops.ref_next_iteration(data, name)
    else                                      Result := gen_control_flow_ops.next_iteration(data, name);
end;

{ MergeOutput }

constructor MergeOutput.Create(values: TArray<TFTensor>);
begin
    output      := values[0];
    value_index := values[1];
end;

function MergeOutput.GetItem(idx: Integer): TFTensor;
begin
    case idx of
      0: Result := output;
      1: Result := value_index;
    else
      Result := nil;
    end;
end;

class operator MergeOutput.implicit(merge: MergeOutput): TFTensor;
begin
   Result := merge.output;
end;
{$ENDREGION}

{$REGION 'ControlFlowState'}
{ ControlFlowState }

constructor ControlFlowState.Create;
begin
   FMap := TDictionary<TControlFlowContext, GradLoopState>.Create;
end;

procedure ControlFlowState.AddWhileContext(op: TFOperation; between_op_list, between_ops: TList<TFOperation>);
begin
    var forward_ctxt := op.GetWhileContext;
    var grad_state : GradLoopState := nil;
    if Fmap.ContainsKey(forward_ctxt) then grad_state := Fmap[forward_ctxt];

    if grad_state = nil then
    begin
        var outer_grad_state : GradLoopState := nil;
        var outer_forward_ctxt := forward_ctxt.outer_context;
        if outer_forward_ctxt <> nil then
            outer_forward_ctxt := outer_forward_ctxt.GetWhileContext;
        if outer_forward_ctxt <> nil then
            outer_grad_state := Fmap[outer_forward_ctxt];
        grad_state := GradLoopState.Create(forward_ctxt, outer_grad_state);
        Fmap[forward_ctxt] := grad_state;
        // We need to include all exits of a loop for backprop.
        for var loop_exit in grad_state.forward_loop_exits do
        begin
            if not between_ops.Contains(loop_exit.op) then
            begin
                between_ops.add(loop_exit.op);
                between_op_list.add(loop_exit.op);
            end;
        end;
    end;
end;

procedure ControlFlowState.EnterGradWhileContext(op: TFOperation; before: Boolean);
begin
    var grad_state := GetGradState(op, before);
    if grad_state <> nil then
        grad_state.grad_context.Enter_;
end;

procedure ControlFlowState.ExitGradWhileContext(op: TFOperation; before: Boolean);
begin
    var grad_state := GetGradState(op, before);
     if grad_state <> nil then
        grad_state.grad_context.Exit_
end;

function ControlFlowState.GetGradState(op: TFOperation; before: Boolean): GradLoopState;
begin
    var forward_ctxt : TControlFlowContext ;
    if (before) and (control_flow_util.IsLoopExit(op)) then
    begin
        forward_ctxt := op._get_control_flow_context;
        forward_ctxt := forward_ctxt.outer_context;
        if forward_ctxt <> nil then
            forward_ctxt := forward_ctxt.GetWhileContext;
    end else
        forward_ctxt := control_flow_util.GetWhileContext(op);

    if forward_ctxt <> nil then
    begin
        Result := nil;
        if Fmap.ContainsKey(forward_ctxt) then
          Result := Fmap[forward_ctxt] ;
        Exit;
    end;
    Result := nil;
end;

procedure ControlFlowState.PostProcessing;
begin
    for var grad_state in Fmap.Values do
    begin
        for var b_merge in grad_state.switch_map.Values do
        begin
            if b_merge.op.inputs[0] = b_merge.op.inputs[1] then
            begin
                var next_grad_val : TFTensor ;
                // The value of this loop variable at iteration i+1 doesn't
                // depend on its value at iteration i. So use zeros as the
                // gradients for all iterations > 0.
                var dtype := b_merge.op.inputs[0].dtype;
                var shape := b_merge.op.inputs[0].shape;
                if shape.IsFullyDefined then
                begin
                    grad_state.grad_context.Enter_;
                    // Create a zeros and use it for iterations > 0.
                    var grad_val  := constant_op.constant(0, dtype,  @shape);
                    next_grad_val := control_flow_ops._NextIteration(grad_val);
                    grad_state.grad_context.Exit_;
                end else
                begin
                    raise TFException.Create('PostProcessing shape is not fully defined.');
                end;
                b_merge.op._update_input(1, next_grad_val);
            end;
        end;
    end;
end;

function ControlFlowState.ProcessUnusedLoopExits(pending_count: TDictionary<string, Integer>; to_ops_set: TList<TFOperation>): TArray<TFTensor>;
begin
    var loop_exits := TList<TFTensor>.Create;
    for var grad_state in Fmap.Values do
    begin
        for var y in grad_state.forward_loop_exits do
        begin
            if not pending_count.ContainsKey(y.op.name) then
            begin
                grad_state.pending_exits_count := grad_state.pending_exits_count - 1;
                if  not to_ops_set.Contains(y.op) then
                    grad_state.unused_exits.add(y);
                if grad_state.pending_exits_count = 0 then
                    loop_exits.AddRange(grad_state.unused_exits);
            end;
        end;
        for var y in grad_state.forward_context.loop_enters do
        begin
            if not pending_count.ContainsKey(y.op.name) then
                pending_count.AddOrSetValue(y.op.name, 1);
        end;
    end;
    Result := loop_exits.ToArray
end;

function ControlFlowState.ZerosLike(op: TFOperation; index: Integer): TFTensor;
begin
    if control_flow_util.IsLoopSwitch(op) then
        Exit(nil);
    if op.graph.building_function then
        Exit( array_ops.zeros_like(op.outputs[index]) );

    control_flow_util.IsSwitch(op);
    var forward_ctxt := control_flow_util.GetWhileContext(op);

    var grad_state := nil;
    if Fmap.ContainsKey(forward_ctxt) then
      grad_state := Fmap[forward_ctxt] ;

    // op is not in a while loop that is part of gradients().
    if grad_state = nil then
        Exit( ZerosLikeOutsideLoop(op, index) );

    raise TFException.Create('ZerosLike');
end;

function ControlFlowState.ZerosLikeForExit(val: TFTensor): TFTensor;
begin

    var val_shape          := val.shape;
    var forward_ctxt       := val.op._get_control_flow_context;
    var outer_forward_ctxt := forward_ctxt.outer_context;
    if outer_forward_ctxt <> nil then
        outer_forward_ctxt := outer_forward_ctxt.GetWhileContext;
    var outer_grad_state : GradLoopState := nil;
    if outer_forward_ctxt <> nil then
    begin
        if Fmap.ContainsKey(outer_forward_ctxt) then
           outer_grad_state := Fmap[outer_forward_ctxt] ;
    end;
    // This is a nested loop.
    if outer_grad_state <> nil then
    begin
        raise TFException.Create('ZerosLikeForExit');
    end else
    begin
        // If the shape is known statically, just create a zero tensor
        // with the right shape.
        if val_shape.IsFullyDefined then
            Result := array_ops.zeros(val_shape.dims, val.dtype)
        else
            Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end;
end;

function ControlFlowState.ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
begin
    var val := op.outputs[index];
    if not control_flow_util.IsSwitch(op) then
    begin
        if val.dtype = Tdtypes.cresource then
           raise TFException.Create('ZerosLikeOutsideLoop');
        (*return array_ops.zeros(
          gen_resource_variable_ops.variable_shape(val),
          dtype: default_gradient.get_zeros_dtype(val));*)
        Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end else
        raise TFException.Create('ZerosLikeOutsideLoop');
end;
{$ENDREGION}

{$REGION 'gen_control_flow_ops'}
{ gen_control_flow_ops }

class function gen_control_flow_ops.control_trigger(name: string): TFOperation;
begin
    var _op := tf.OpDefLib._apply_op_helper('ControlTrigger', name, []);
    Result := _op;
end;

class function gen_control_flow_ops.enter(data: TFTensor; frame_name: string; is_constant: Boolean; parallel_iterations: Integer; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Enter', name, [GetArg('data',data),GetArg('frame_name',frame_name),GetArg('is_constant',is_constant),GetArg('parallel_iterations',parallel_iterations)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.loop_cond(input: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('NoOp', name, [GetArg('input',input)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.merge(inputs: TArray<TFTensor>; name: string): MergeOutput;
begin
    var _op := tf.OpDefLib._apply_op_helper('Merge', name, [GetArg('inputs',inputs)]);
    Result := MergeOutput.Create(_op.outputs);
end;

class function gen_control_flow_ops.next_iteration(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('NextIteration', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.no_op(name: string): TFOperation;
begin
    var _op := tf.OpDefLib._apply_op_helper('NoOp', name, []);
    Result := _op;
end;

class function gen_control_flow_ops.ref_next_iteration(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefNextIteration', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.ref_exit(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefExit', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.ref_switch(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefSwitch', name, [GetArg('data',data)]);
    Result := _op.outputs;
end;

class function gen_control_flow_ops.switch(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('Switch', name, [GetArg('data',data),GetArg('pred',pred)]);
    Result := [ _op.outputs[0], _op.outputs[1] ];
end;

class function gen_control_flow_ops._exit(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Exit', name, [GetArg('data',data)]);
    Result := _op.output;
end;
{$ENDREGION}

{$REGION 'CondContext'}
{ CondContext }

constructor CondContext.Create(pred, pivot: TFTensor; branch: Integer; name: string; context_def: TCondContextDef; import_scope: string);
begin
    inherited Create;

    Fexternal_values := TDictionary<string,TFTensor>.Create;

    if (pred = nil) and (context_def = nil) then Exit;

    Fname := Tops.get_default_graph.unique_name(name);
    if context_def <> nil then
    begin
        _init_from_proto(context_def, import_scope);
    end else
    begin
        // Initializes the default fields.
        __init__;
        Fpred  := pred;
        Fpivot := pivot;
        Fbranch:= branch; // 0 or 1 representing this branch
        // Values considered to have been already seen in this context. pred is not
        // included in this context.
        Fvalues.Add(pred.name);
        Fexternal_values.AddOrSetValue(pred.name, pred);
        Fvalues.Add(pivot.name);
        pivot.op._set_control_flow_context(self);
    end;
end;

destructor CondContext.Destroy;
begin
    Fexternal_values.Free;

    inherited Destroy;
end;

procedure CondContext._init_from_proto(context_def: TCondContextDef; import_scope: string);
begin
    var g := Tops.get_default_graph;
    Fname := Tops.prepend_name_scope(context_def.ContextName, import_scope);
    var p1 := Tops.prepend_name_scope(context_def.PredName, import_scope);
    Fpred  := g.as_graph_element(p1) as TFTensor;
    var p2 := Tops.prepend_name_scope(context_def.PivotName, import_scope);
    Fpivot := g.as_graph_element(p2) as TFTensor;
    Fbranch:= context_def.Branch;
    __init__(context_def.ValuesDef, import_scope);
end;

function CondContext.AddValue(val: TFTensor): TFTensor;
var
  rResult : TFTensor;
begin
    rResult := nil;
    if Fvalues.Contains(val.name) then
    begin
        // Use the real value if it comes from outer context. This is needed in
        // particular for nested conds.
        if Fexternal_values.ContainsKey(val.name) then
            rResult := Fexternal_values[val.name];

        if rResult = nil then  rResult := val
    end else
    begin
        rResult := val;
        Fvalues.Add(val.name);
        // TODO: _outer_context
        if Fouter_context <> nil then
        begin
            rResult := Fouter_context.AddValue(val);
            Fvalues.Add(rResult.name);
            Fexternal_values.AddOrSetValue(rResult.name,rResult);
        end ;

        TUtils.tf_with<TControlDependenciesController>(Tops.control_dependencies(nil), procedure(ctrl: TControlDependenciesController)
                    begin
                        var res := control_flow_ops._SwitchRefOrTensor(rResult, Fpred);
                        rResult := res[Fbranch];
                        if Fouter_context <> nil then
                            Fouter_context.AddInnerOp(rResult.op);
                    end);

        rResult.op.graph.prevent_fetching(rResult.op);
        rResult.op._set_control_flow_context(Self);

        // Mark Switch output as seen by this context and any outer contexts,
        // just like what we do for normal op outputs in _AddOpInternal() below.
        var ctxt : TControlFlowContext := Self;
        while ctxt <> nil do
        begin
            ctxt.values.Add(rResult.name);
            ctxt := ctxt.outer_context;
        end;
        Fexternal_values.AddOrSetValue(val.name, rResult);
    end;
    Result := rResult;
end;

procedure CondContext._AddOpInternal(op: TFOperation);
var
  LRemResult: Tuple<TArray<TFOperation>, TArray<TFOperation>>;
begin
    if op.inputs.Count = 0 then
    begin
        //If we're in a while loop, remove any control inputs from outside the
        // loop.
        LRemResult := _RemoveExternalControlEdges(op);
        for var i := 0 to Length(op.control_inputs) -1 do
        begin
            var input_op := op.control_inputs[i];
            if not OpInContext(input_op) then
            begin
               op._add_control_input(Fpivot.op);
               Break;
            end;
        end;
    end else
    begin
        var real_x : TFTensor;
        // Make each input to 'op' available in this CondContext. If an input is
        // already part of this context there's nothing to do, but if it's
        // external, AddValue() will handle adding the appropriate Switch node and
        // other bookkeeping.
        for var index : Integer := 0 to  op.inputs.Count -1 do
        begin
            var x := op.inputs[index];
            if (op.Tipo = 'Merge') and (x.op.tipo = 'NextIteration') then
            begin
                //# Edge case: if we're importing a while loop inside this CondContext,
                //# AddValue() will not correctly handle the NextIteration inputs to
                //# Merge node. The problem is that the NextIteration should also be
                //# part of this context, but if we're importing it won't have been
                //# processed and added to the context yet, so AddValue() will try to
                //# add a Switch which results in an invalid graph. Instead, we use the
                //# NextIteration input as-is here, and it will eventually be added to
                //# the context via AddOp().
                real_x := x;
            end else
            begin
                real_x := AddValue(x);
            end;
            if real_x <> x then
                op._update_input(index, real_x);
        end;
        // Remove any external control dependency on this op.
        LRemResult := _RemoveExternalControlEdges(op);
        // TODO: implement below code dependencies
        //if (op.graph._is_function(op.type) || op.type == "SymbolicGradient")
        //    op._add_control_input(_pivot.op);
    end;

    // Mark op's outputs as seen by this context and any outer contexts.
    var output_names : TArray<String>;
    for var i := 0 to Length(op.outputs) -1 do
       output_names := output_names + [ op.outputs[i].Name ];

    var ctxt : TControlFlowContext := self;
    while ctxt <> nil do
    begin
        for var name in output_names do
            ctxt.values.Add(name);
        ctxt := ctxt.outer_context;
    end;

    if (Fouter_context <> nil) or (not control_flow_ops.IsLoopExit(op)) then
        op.graph.prevent_fetching(op);

    if Fouter_context <> nil then
        Fouter_context.AddInnerOp(op);
end;

function CondContext.BuildCondBranch<T>(fn: TFunc<TArray<T>>): tuple<TArray<T>, TArray<TFTensor>>;
begin
    // Add the subgraph defined by fn() to the graph.
    var pre_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
    var original_result : TArray<T> := fn;
    var post_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

    var v := TValue.From<TArray<T>>(original_result);
    if      v.TypeInfo = TypeInfo(TArray<TFTensor>)    then
    begin
        var Res := v.AsType<TArray<TFTensor>>;
        var aTens : TArray<TFTensor> := [];
        for var i := 0 to Length(Res) - 1 do
          aTens := aTens + [ _BuildCondTensor(Res[i]) ] ;

        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,aTens)
    end
    else if v.TypeInfo = TypeInfo(TArray<TFOperation>) then
    begin
        var Res := v.AsType<TArray<TFOperation>>;
        var aTens : TArray<TFTensor> := [];
        for var i := 0 to Length(Res) - 1 do
          aTens := aTens + [ _BuildCondTensor(Res[i]) ] ;

        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,aTens)
    end
    else if v.TypeInfo = TypeInfo(TArray<Single>) then
    begin
        var fv : TArray<Single> :=  v.AsType< TArray<Single> >;
        var res := Tops.convert_to_tensor(fv[0]);
        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,[ _BuildCondTensor(res) ])
    end else
    begin
        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result, nil)
    end;

end;

function CondContext.BuildCondBranch<T>(fn: TFunc<T>): Tuple<T, TFTensor>;
begin
    // Add the subgraph defined by fn() to the graph.
    var pre_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
    var original_result : T := fn;
    var post_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

    //TODO: port this chunck of missing code:
    (*
    if len(post_summaries) > len(pre_summaries):
        new_summaries = post_summaries[len(pre_summaries):]
        summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
        summary_ref[:] = pre_summaries
        with ops.control_dependencies(new_summaries):
        if original_result is None:
            return no_op(), None
        else:
            original_result = nest.map_structure(array_ops.identity,
                                                original_result)
    *)

    var v := TValue.From<T>(original_result);
    if      v.TypeInfo = TypeInfo(TFTensor)    then
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(v.AsType<TFTensor>))
    end
    else if v.TypeInfo = TypeInfo(TFOperation) then
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(v.AsType<TFOperation>))
    end
    else if v.TypeInfo = TypeInfo(TArray<Single>) then
    begin
        var fv : TArray<Single> :=  v.AsType< TArray<Single> >;
        var res := Tops.convert_to_tensor(fv[0]);
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(res))
    end else
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, nil)
    end;

end;

function CondContext._ProcessOutputTensor(_val: TFTensor): TFTensor;
begin
    var real_val := _val;
    if not Fvalues.Contains(_val.name) then
    begin
        // Handle the special case of lambda: x
        Fvalues.Add(_val.name);
        if Fouter_context <> nil then
        begin
            real_val := Fouter_context.AddValue(_val);
            Fvalues.Add(real_val.name);
            Fexternal_values.AddOrSetValue(real_val.name, real_val);
        end;
        var res := control_flow_ops._SwitchRefOrTensor(real_val, Fpred);
        real_val := res[Fbranch];
        Fexternal_values.AddOrSetValue(_val.name, real_val);
    end else
    begin
        var external_val : TFTensor := nil;
        if Fexternal_values.ContainsKey(_val.name) then
            external_val := Fexternal_values[_val.name];
        if external_val <> nil then
            real_val := external_val;
    end;
    Result := real_val;
end;

function CondContext._BuildCondTensor(v: ITensorOrOperation): TFTensor;
begin
    if v is TFOperation then
    begin
        var op : TFOperation := v as TFOperation;
        Result := control_flow_ops.with_dependencies([ op ], Fpivot);
    end
    else if v is TFTensor then
    begin
        var t : TFTensor := v as TFTensor;
        Result := _ProcessOutputTensor(t);
    end else
    begin
        Result := _ProcessOutputTensor( Tops.convert_to_tensor(v) );
    end;
end;
{$ENDREGION}

{$REGION 'control_flow_util'}
{ control_flow_util }

class procedure control_flow_util.CheckInputFromValidContext(op, input_op: TFOperation);
begin
    var op_ctxt    := op._get_control_flow_context;
    var input_ctxt := GetOutputContext(input_op);
    var valid := false;
    if input_ctxt = nil then
        valid := true
    else if op_ctxt = input_ctxt then
        valid := true
    else
    begin
        var while_ctxt       := GetContainingWhileContext(op_ctxt);
        var input_while_ctxt := GetContainingWhileContext(input_ctxt);
        if while_ctxt = nil then
        begin
            // Neither op nor input_op is in a while loop, but one or both are in
            // conds. We allow this, although execution will fail if the branch
            // corresponding to input_op's cond context isn't taken.
            if input_while_ctxt = nil then
                valid := true;
            // Invalid if op isn't in a while loop and input_op is. Unless...
            if IsLoopEnter(op) then
                // WhileContext._BuildLoop clears context for Enter nodes.
                valid := true;
            if IsSwitch(op) then
                // CondContext.AddValue clears context for Switch nodes.
                valid := true;
        end
        else if IsContainingContext(while_ctxt, input_while_ctxt) then
        begin
            // input_op is in a while loop which contains op's while loop (or not in a
            // while loop at all).
            valid := true;
        end
        else if (while_ctxt.grad_state <> nil) and (IsContainingContext(while_ctxt.grad_state.forward_context, input_while_ctxt) ) then
        begin
            valid := true;
        end
        else
           raise TFException.Create('CheckInputFromValidContext');
    end;
    if not valid then
       raise TFException.Create('CheckInputFromValidContext');

end;

class function control_flow_util.GetContainingWhileContext(ctxt, stop_ctxt: TControlFlowContext): WhileContext;
begin
    while ctxt <> nil do
    begin
        if (ctxt.IsWhileContext) or (ctxt = stop_ctxt) then
            Exit ( ctxt as WhileContext );
        ctxt := ctxt.outer_context;
    end;
    Result := nil;
end;

class function control_flow_util.GetLoopConstantEnter(value: TFTEnsor): TFOperation;
begin
    var id_ops : TArray<String> := [ 'Switch', 'RefSwitch', 'Identity', 'RefIdentity' ];
    var op := value.op;
    while TArray.Contains<String>(id_ops, op.tipo) do
        op := op.inputs[0].op;

    Result := nil;
    if IsLoopConstantEnter(op) then
       Result := op;
end;

class function control_flow_util.GetOutputContext(op: TFOperation): TControlFlowContext;
begin
    var ctxt := op._get_control_flow_context;
    // Exit nodes usually have a control flow context, except in the case where the
    // exit node was imported via import_graph_def (in which case no nodes have
    // control flow contexts).
    if (ctxt <> nil) and (IsLoopExit(op)) then
        ctxt := ctxt.outer_context;
    Result := ctxt;
end;

class function control_flow_util.GetWhileContext(op: TFOperation): WhileContext;
begin
    Result := op.GetWhileContext
end;

class function control_flow_util.IsCondSwitch(op: TFOperation): Boolean;
begin
    if  not IsSwitch(op) then
        Exit( false );
    if (op.outputs = nil) or (Length(op.outputs) = 0) then
        Exit( false );
    // Switch nodes are not part of the cond control flow context that they
    // represent, so consider the consumers of its outputs to determine if it is
    // cond switch or not. A switch is a cond switch iff all its consumers are in
    // cond contexts.
    var is_cond_switch := true;
    for var o in op.outputs do
    begin
        for var c in o.consumers do
        begin
            var ctxt := c._get_control_flow_context;
            if IsLoopEnter(c) then
                ctxt := ctxt.outer_context;
            is_cond_switch := (is_cond_switch) and ( (ctxt <> nil) and (ctxt.IsCondContext) );
        end;
    end;
    Result := is_cond_switch;
end;

class function control_flow_util.IsContainingContext(ctxt, maybe_containing_ctxt: WhileContext): Boolean;
begin
   while ctxt <> maybe_containing_ctxt do
   begin
      if ctxt = nil then
          Exit( false );
      ctxt := ctxt.outer_context as WhileContext;
   end;
   Result := true;
end;

class function control_flow_util.IsLoopConstantEnter(op: TFOperation): Boolean;
begin
    Result := (IsLoopEnter(op)) and (op.get_attr<boolean>('is_constant'));
end;

class function control_flow_util.IsLoopEnter(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Enter') or (op.tipo = 'RefEnter');
end;

class function control_flow_util.IsLoopExit(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Exit') or (op.tipo = 'RefExit');
end;

class function control_flow_util.IsLoopSwitch(op: TFOperation): Boolean;
begin
    if IsSwitch(op) then
    begin
        var ctxt := op._get_control_flow_context;
        Result := (ctxt <> nil) and (ctxt.IsWhileContext) and (not IsCondSwitch(op) );
        Exit;
    end;
    Result := false;
end;

class function control_flow_util.IsSwitch(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Switch') or (op.tipo = 'RefSwitch');
end;
{$ENDREGION}

{$REGION 'gen_ops'}
{ gen_ops }

class function gen_ops.clip_by_value(t, clip_value_min, clip_value_max: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('t', t);
      dict.add('clip_value_min', clip_value_min);
      dict.add('clip_value_max', clip_value_max);
      var op := tf.OpDefLib._apply_op_helperDict('ClipByValue', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.elu(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Elu', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.relu(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Relu', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.relu6(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Relu6', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.softplus(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Softplus', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.softsign(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Softsign', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;
{$ENDREGION}

{$REGION 'gen_image_ops'}
{ gen_image_ops }

class function gen_image_ops.resize_bilinear(images, size: TFTensor; align_corners, half_pixel_centers: Boolean; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('ResizeBilinear', name, ExecuteOpArgs.Create([images, size])
              .SetAttributes(['align_corners', align_corners, 'half_pixel_centers',  half_pixel_centers ])).First;
end;

class function gen_image_ops.resize_nearest_neighbor<Tsize>(images: TFTensor; size: Tsize; align_corners, half_pixel_centers: Boolean; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('ResizeNearestNeighbor', name, ExecuteOpArgs.Create([images, TValue.From<Tsize>(size)])
              .SetAttributes(['align_corners', align_corners, 'half_pixel_centers',  half_pixel_centers ])).First;
end;
{$ENDREGION}

{$REGION 'image_ops_impl'}
{ image_ops_impl }

class function image_ops_impl.resize_images_v2<T>(images: TFTensor; size: T; method: string; preserve_aspect_ratio, antialias: Boolean; name: string): TFTensor;
begin
    var resize_fn : TFunc<TFTensor, TFTensor, TfTensor> := function(Img : TFTensor; _size: TFTensor): TFTensor
                                        begin
                                            if method = ResizeMethod.BILINEAR then
                                                Exit( gen_image_ops.resize_bilinear(Img, _size, false, true) )
                                            else if method = ResizeMethod.NEAREST_NEIGHBOR then
                                                Exit(  gen_image_ops.resize_nearest_neighbor(Img, _size, false, true) );
                                            raise Exception.Create('resize_images_v2');
                                        end;
    var size_tensor := Tops.convert_to_tensor(TValue.From<T>(size), tf.int32_t);
    Result          := _resize_images_common(images, resize_fn, size_tensor, preserve_aspect_ratio, name, false);
end;

class function image_ops_impl._ImageDimensions(image: TFTensor; rank: Integer): TArray<Int64>;
begin
    if image.shape.IsFullyDefined then
        Exit( image.shape.dims)
    else begin
        var static_shape  := image.shape.with_rank(rank).dims;
        var dynamic_shape := array_ops.unstack(array_ops.shape(image), @rank);

        var ss_storage : TArray<Int64> := [];
        var ds_storage : TArray<Int64> := [];
        // var sd = static_shape.Zip(dynamic_shape, (first, second) => storage[storage.Length] = first;
        for var i := 0 to Length(static_shape)-1 do
        begin
            var ss := static_shape[i];
            var ds := dynamic_shape[i];

            ss_storage := ss_storage + [ ss ] ;
            ds_storage := ds_storage + [ Int64(TTensor(ds)) ] ;
        end;
        var sd := Enumerable<Int64>.Create(static_shape);
        var sdd:= TCollections.CreateList<TFTensor>(dynamic_shape);
        sd.Zip<TFTensor,Boolean>(sdd, function(ss: Int64; ds: TFTensor): Boolean
                                       begin
                                           ss_storage := ss_storage + [ ss ] ;
                                           ds_storage := ds_storage + [ Int64(TTensor(ds)) ] ;
                                           Result := true;
                                       end);


        if Length(ss_storage) > 0 then Result := ss_storage
        else                           Result := ds_storage;
    end;
end;

class function image_ops_impl._resize_images_common(images: TFTensor; resizer_fn: TFunc<TFTensor, TFTensor, TFTensor>; size: TFTensor; preserve_aspect_ratio: Boolean;
  name: string; skip_resize_if_same: Boolean): TFTensor;
begin
     var vValues : TArray<TValue> := [images, size];
     Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'resize', @vValues),
                        function(v1: TNameScope): TFTensor
                          begin
                              if images.shape.ndim = -1 then
                                 raise Exception.Create('"images" contains no shape.');
                              var is_batch : Boolean := true;
                              if images.shape.ndim = 3 then
                              begin
                                  is_batch := false;
                                  images := array_ops.expand_dims(images, 0);
                              end
                              else if images.shape.ndim <> 4 then
                                 raise Exception.Create('"images" must have either 3 or 4 dimensions.');

                              var height := images.dims[1];
                              var width  := images.dims[2];

                              if not size.shape.is_compatible_with(TFShape.Create([ 2 ])) then
                                 raise Exception.Create('"size" must be a 1-D Tensor of 2 elements: new_height, new_width');

                              if preserve_aspect_ratio then
                              begin
                                  var _chcw_ := _ImageDimensions(images, 4);

                                  var scale_factor_height := TTensor(math_ops.cast(size[0], Tdtypes.cfloat32)) / _chcw_[1];
                                  var scale_factor_width  := TTensor(math_ops.cast(size[1], Tdtypes.cfloat32)) /  _chcw_[2];

                                  var scale_factor        := math_ops.minimum(scale_factor_height, scale_factor_width);

                                  var scaled_height_const := math_ops.cast(math_ops.round(TTEnsor(scale_factor) * _chcw_[1]), Tdtypes.cint32);
                                  var scaled_width_const  := math_ops.cast(math_ops.round(TTEnsor(scale_factor) * _chcw_[2]), Tdtypes.cint32);

                                  var v : TValue := [scaled_height_const, scaled_width_const];
                                  size := Tops.convert_to_tensor( v, Tdtypes.cint32, 'size');
                              end;

                              var size_const_as_shape := TUtils.constant_value_as_shape(size);
                              var new_height_const    := tensor_shape.dimension_at_index(size_const_as_shape, 0).value;
                              var new_width_const     := tensor_shape.dimension_at_index(size_const_as_shape, 1).value;

                              var x_null : Boolean := true;
                              if skip_resize_if_same then
                              begin
                                  for var x: Integer in [ new_width_const, width, new_height_const, height ] do
                                  begin
                                      if (width <> new_width_const) and (height = new_height_const) then
                                          break;

                                      if x <> 0 then
                                        x_null := false;

                                  end;
                                  if  not x_null then
                                      images := array_ops.squeeze(images, [ 0 ]);
                                  Result := images;
                                  Exit;
                              end;

                              images := resizer_fn(images, size);

                              images.shape := TFShape.Create([-1, new_height_const, new_width_const, -1]);

                              if not is_batch then
                                  images := array_ops.squeeze(images, [ 0 ]);
                              Result := images;
                          end);

end;
{$ENDREGION}

{$REGION 'dataset_ops'}
{ dataset_ops }

function dataset_ops.anonymous_iterator_v3(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
var
 iInfo : TFastPathOpExecInfo;
begin
    var ctx := tf.Context;
    var attrs := TDictionary<string, TValue>.Create;
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])];

    attrs.Add('output_types', TValue.From<TArray<Integer>>(typeIntArray));
    attrs.Add('output_shapes', TValue.From<TArray<TFShape>>(output_shapes));
    if ctx.executing_eagerly then
    begin
        try
            iInfo := TFastPathOpExecInfo.Create('AnonymousIteratorV3', name,[]);
            iInfo.attrs := attrs;
            var Res := tf.Runner.TFE_FastPathExecute(iInfo);
            Result :=  Res[0];
            Exit;
        except
            Result := anonymous_iterator_v3_eager_fallback(output_types, output_shapes, name, ctx);
            Exit;
        end
    end;
    Result := tf.OpDefLib._apply_op_helper('AnonymousIteratorV3', name, [ GetArg('output_types',  attrs['output_types']),
                                                                          GetArg('output_shapes', attrs['output_shapes']) ]).outputs[0];
end;

function dataset_ops.anonymous_iterator_v3_eager_fallback(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string ; ctx: TContext): TFTensor;
begin
    var dictAttrs := TDictionary<string, TValue>.Create;
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])];

    dictAttrs.Add('output_types', TValue.From<TArray<Integer>>(typeIntArray));
    dictAttrs.Add('output_shapes', TValue.From<TArray<TFShape>>(output_shapes));

    var attrs : TArray<TValue> := [ dictAttrs['output_types'], dictAttrs['output_shapes'] ];

    var Res := TExecute.quick_execute('AnonymousIteratorV3', 1, [], attrs, ctx, name);
    Result  := Res[0];
end;

function dataset_ops.anonymous_iterator_v2(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): Tuple<TFTensor, TFTensor>;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var Res := tf.Context.ExecuteOp('AnonymousIteratorV2', name, ExecuteOpArgs.Create([])
          .SetAttributes(['output_types', TValue.From<TArray<Integer>>(typeIntArray), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )]));

    Result := Tuple.Create(Res[0],Res[0])
end;

function dataset_ops.batch_dataset_v2(input_dataset, buffer_size, drop_remainder: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  parallel_copy: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('BatchDatasetV2', name, ExecuteOpArgs.Create([input_dataset, buffer_size, drop_remainder])
                 .SetAttributes(['parallel_copy', parallel_copy, 'output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.cache_dataset_v2(input_dataset, filename, cache: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('CacheDatasetV2', name, ExecuteOpArgs.Create([input_dataset, filename, cache])
                 .SetAttributes(['output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.concatenate_dataset(input_dataset, another_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('ConcatenateDataset', name, ExecuteOpArgs.Create([input_dataset, another_dataset])
                 .SetAttributes(['output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

procedure dataset_ops.delete_iterator(handle, deleter: TFTensor; name: string);
begin
    tf.Context.ExecuteOp('DeleteIterator', name, ExecuteOpArgs.Create([handle, deleter]));
end;

function dataset_ops.dummy_memory_cache(name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DummyMemoryCache', name, ExecuteOpArgs.Create([])).First;
end;

function dataset_ops.dummy_seed_generator(name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DummySeedGenerator', name, ExecuteOpArgs.Create([])).First;
end;

function dataset_ops.filter_dataset(dataset: TFTensor; predicate: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('FilterDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['predicate',     TValue.From<ConcreteFunction>(predicate),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.flat_map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('FlatMapDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['f',     TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.iterator_get_next(iterator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TArray<TFTensor>;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    try
    Result := tf.Context.ExecuteOp('IteratorGetNext', name, ExecuteOpArgs.Create([iterator])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).toArray;
    except
      result := [];
      Exit;
    end;
end;

procedure dataset_ops.make_iterator(dataset, iterator: TFTensor; name: string);
begin
    tf.Context.ExecuteOp('MakeIterator', name, ExecuteOpArgs.Create([dataset, iterator]));
end;

function dataset_ops.map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism,
  preserve_cardinality: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('MapDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['f',             TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'use_inter_op_parallelism',  use_inter_op_parallelism,
                                       'preserve_cardinality', preserve_cardinality])).First;
end;

function dataset_ops.model_dataset(input_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; algorithm: AutotuneAlgorithm; cpu_budget,
  ram_budget: Int64; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('ModelDataset', name, ExecuteOpArgs.Create([input_dataset])
                       .SetAttributes(['algorithm',     TValue.From<Integer>(Ord(algorithm)),
                                       'cpu_budget',    cpu_budget,
                                       'ram_budget',    ram_budget,
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.optimize_dataset(input_dataset, optimizations: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  optimization_configs: TArray<string>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('OptimizeDataset', name, ExecuteOpArgs.Create([input_dataset, optimizations])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'optimization_configs', TValue.From<TArray<string>>( optimization_configs )])).First;
end;

function dataset_ops.optimize_dataset_v2(input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default: TFTensor; output_types: TArray<TF_DataType>;
  output_shapes: TArray<TFShape>; optimization_configs: TArray<string>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('OptimizeDatasetV2', name, ExecuteOpArgs.Create([input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'optimization_configs', TValue.From<TArray<string>>( optimization_configs )])).First;
end;

function dataset_ops.parallel_map_dataset_v2(dataset, num_parallel_calls: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  use_inter_op_parallelism: Boolean; deterministic: string; preserve_cardinality: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('ParallelMapDatasetV2', name, ExecuteOpArgs.Create([dataset, a, num_parallel_calls])
                       .SetAttributes(['f',             TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'use_inter_op_parallelism',  use_inter_op_parallelism,
                                       'deterministic',             deterministic,
                                       'preserve_cardinality', preserve_cardinality])).First;
end;

function dataset_ops.prefetch_dataset(input_dataset, buffer_size: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; slack_period: Integer;
  legacy_autotune: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('PrefetchDataset', name, ExecuteOpArgs.Create([input_dataset, buffer_size])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes ),
                                       'slack_period',   slack_period,
                                       'legacy_autotune',legacy_autotune])).First;
end;

function dataset_ops.range_dataset(start, stop, step: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   var typeIntArray : TArray<Integer> := [];
   for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('RangeDataset', name, ExecuteOpArgs.Create([start, stop, step])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.repeat_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('RepeatDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.shard_dataset(input_dataset, num_shards, index: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; require_non_empty: Boolean;
  name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ShardDataset', name, ExecuteOpArgs.Create([input_dataset, num_shards, index])
                       .SetAttributes(['require_non_empty', require_non_empty,
                                       'output_types',      TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',     TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.shuffle_dataset_v3(input_dataset, buffer_size, seed, seed2, seed_generator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  reshuffle_each_iteration: Boolean; name: string): TFTEnsor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ShuffleDatasetV3', name, ExecuteOpArgs.Create([input_dataset, buffer_size, seed, seed2, seed_generator])
                       .SetAttributes(['reshuffle_each_iteration', reshuffle_each_iteration,
                                       'output_types',             TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',            TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.skip_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('SkipDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.take_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('TakeDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.tensor_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('TensorDataset', name, ExecuteOpArgs.Create([components])
                       .SetAttributes(['output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.tensor_slice_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('TensorSliceDataset', name, ExecuteOpArgs.Create([components])
                       .SetAttributes(['output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.zip_dataset(input_datasets: TArray<TFTEnsor>; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ZipDataset', name, ExecuteOpArgs.Create([input_datasets])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;
{$ENDREGION}

{$REGION 'embedding_ops'}
{ embedding_ops }

class function embedding_ops._embedding_lookup_and_transform(params: IVariableV1; ids: TFTensor; partition_strategy, name, max_norm: string): TFTensor;
begin
    var vValues : TArray<TValue> := [TValue.From<IVariableV1>(params), ids];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'embedding_lookup', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var np : Integer := 1;
                                                ids := Tops.convert_to_tensor(ids, DtInvalid, 'ids');
                                                if np = 1 then
                                                begin
                                                    var gather := array_ops.gather(params.AsTensor, ids, name);
                                                    var res    := _clip(gather, ids, max_norm);

                                                    Result := array_ops.identity(res);
                                                    Exit;
                                                end;

                                                raise Exception.Create('_embedding_lookup_and_transform');
                                            end );
end;

class function embedding_ops._embedding_lookup_and_transform(params: TArray<TFTensor>; ids: TFTensor; partition_strategy, name, max_norm: string): TFTensor;
begin
    var vValues : TArray<TValue> := [TValue.From<TArray<TFTensor>>(params), ids];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'embedding_lookup', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var np : Integer := Length(params);
                                                params := Tops.convert_n_to_tensor_or_indexed_slices(params, DtInvalid, 'params');
                                                ids := Tops.convert_to_tensor(ids, DtInvalid, 'ids');
                                                if np = 1 then
                                                begin
                                                    Tops.colocate_with(params[0]);
                                                    var res := _clip(array_ops.gather(@params[0], ids, name), ids, max_norm);
                                                    Result := array_ops.identity(res);
                                                end else
                                                begin
                                                    // Flatten the ids. There are two cases where we need to do this.
                                                    raise Exception.Create('_embedding_lookup_and_transform');
                                                end;
                                            end );
end;

class function embedding_ops.embedding_lookup(params: TArray<TFTensor>; ids: TFTensor; partition_strategy, name: string; validate_indices: Boolean; max_norm: string): TFTensor;
begin
    Result := _embedding_lookup_and_transform(params, ids, partition_strategy, name, max_norm);
end;

class function embedding_ops.embedding_lookup(params: IVariableV1; ids: TFTensor; partition_strategy, name: string; validate_indices: Boolean; max_norm: string): TFTensor;
begin
    Result := _embedding_lookup_and_transform(params, ids, partition_strategy, name, max_norm)
end;

class function embedding_ops._clip(params, ids: TFTensor; max_norm: string): TFTensor;
begin
    if max_norm = '' then
       Exit(params);
    raise Exception.Create('_clip');
end;
{$ENDREGION}

{$REGION 'linalg_ops'}
{ linalg_ops }

function linalg_ops.cholesky(input: TFTensor; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('Cholesky', name, ExecuteOpArgs.Create([ input])).First;
end;

function linalg_ops.cholesky_solve(chol, rhs: TFTensor; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([chol, rhs]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'cholesky_solve', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                var y := matrix_triangular_solve(chol, rhs, true, false);
                                var x := matrix_triangular_solve(chol, y,  true,  true);
                                Result := x;
                            end );
end;

function linalg_ops.eye(num_rows, num_columns: Integer; batch_shape: PTFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([num_rows, num_columns, TValue.From<PTFShape>(batch_shape)]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'eye', @vvalue),
                  function(v1: TNameScope): TFTensor
                    begin
                        if num_columns = -1 then
                            num_columns := num_rows;
                        var is_square: Boolean := num_columns = num_rows;
                        var diag_size := Min(num_rows, num_columns);
                        var sShape : TFShape;
                        if batch_shape = nil  then
                            sShape := TFShape.Create(TArray<Integer>.Create())
                        else
                            sShape := batch_shape^;
                        var batch_shape_tensor := Tops.convert_to_tensor(TValue.From<TFShape>(sShape), tf.int32_t, 'shape');
                        var diag_shape := array_ops.concat([ batch_shape_tensor, tf.constant(TArray<Integer>.Create(diag_size )) ], 0);
                        var shape : TFTensor := nil;
                        if not is_square then
                            shape := array_ops.concat([ batch_shape_tensor, tf.constant(TArray<Integer>.Create( num_rows, num_columns )) ], 0);
                        var diag_ones := array_ops.ones(diag_shape, dtype);
                        if is_square then
                            Exit ( array_ops.matrix_diag(diag_ones) )
                        else begin
                            var zero_matrix := array_ops.zeros(shape, dtype);
                            Result := array_ops.matrix_set_diag(zero_matrix, diag_ones);
                        end;
                    end );
end;

function linalg_ops.matrix_inverse(input: TFTensor; adjoint: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatrixInverse', name, ExecuteOpArgs.Create([ input ])
                                                 .SetAttributes(['adjoint', adjoint ]) ).First;
end;

function linalg_ops.matrix_solve_ls(matrix, rhs, l2_regularizer: TFTensor; fast: Boolean; name: string): TFTensor;
begin
   Result := _composite_impl(matrix, rhs, l2_regularizer);
end;

function linalg_ops.matrix_triangular_solve(matrix, rhs: TFTensor; lower, adjoint: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatrixTriangularSolve', name, ExecuteOpArgs.Create([ matrix, rhs ])
                                                 .SetAttributes(['lower',lower,'adjoint', adjoint ]) ).First;
end;

function linalg_ops.norm(tensor: TFTensor; _ord: string; axis: PAxis; name: string; keepdims: Boolean): TFTensor;
begin
    var vvalue : TValue := tensor;
    var is_matrix_norm := (axis <> nil) and (Length(axis^.axis.Value) = 2);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'eye', @vvalue),
                  function(v1: TNameScope): TFTensor
                    begin
                        if is_matrix_norm  then
                           raise TFException.Create('Not Implemented');
                        var res := math_ops.sqrt( math_ops.reduce_sum(TTensor(tensor) * math_ops.conj(tensor), axis^, true) );
                        if not keepdims then
                            res := array_ops.squeeze(res, axis^.axis);
                        Result := res;
                    end );
end;

function linalg_ops.qr(input: TFTensor; full_matrices: Boolean; name: string): TFTensors;
begin
    Result := tf.Context.ExecuteOp('Qr', name, ExecuteOpArgs.Create([ input ])
                                                 .SetAttributes(['full_matrices',full_matrices ]) );
end;

function linalg_ops._composite_impl(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var matrix_shape : TFShape := Enumerable<int64>.Create(matrix.shape.dims).Skip(matrix.shape.ndim - 2).ToArray;
    if matrix_shape.IsFullyDefined then
    begin
        if matrix_shape[-2] >= matrix_shape[-1] then
            Exit( _overdetermined(matrix, rhs, l2_regularizer) )
        else
            Exit( _underdetermined(matrix, rhs, l2_regularizer) );
    end;
    raise TFException.Create('Not Implemented');
end;

function linalg_ops._overdetermined(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var chol := _RegularizedGramianCholesky(matrix, l2_regularizer, true);
    Result   := cholesky_solve(chol, math_ops.matmul(matrix, rhs, False, False, true));
end;

function linalg_ops._RegularizedGramianCholesky(matrix, l2_regularizer: TFTensor; first_kind: Boolean): TFTensor;
begin
    var gramian := math_ops.matmul(matrix, matrix, False, False, first_kind, not first_kind);

    if l2_regularizer <> nil then
    begin
        var matrix_shape := array_ops.shape(matrix);
        var batch_shape  := matrix_shape[':-2'];
        var sShape := batch_shape.shape;
        var small_dim : TFTensor;
        if first_kind then small_dim := matrix_shape[-1]
        else               small_dim := matrix_shape[-2];
        var npy : NDArray := small_dim.numpy;
        var identity      := eye(npy, -1, @sShape, matrix.dtype);
        var small_dim_static : Int64;
        if first_kind then small_dim_static := matrix.shape[-1]
        else               small_dim_static := matrix.shape[-2];
        var a := Enumerable<int64>.Create(matrix.shape.dims).Take(matrix.shape.ndim - 2).ToArray;
        TArray.Concat<Int64>([a, TArray<Int64>.Create(small_dim_static, small_dim_static)] );
        identity.shape := TArray.Concat<Int64>([a, TArray<Int64>.Create(small_dim_static, small_dim_static)] );
        gramian := gramian + (TTensor(l2_regularizer) * identity);
    end;
    Result := cholesky(gramian);
end;

function linalg_ops._underdetermined(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var chol := _RegularizedGramianCholesky(matrix, l2_regularizer, false);
    Result := math_ops.matmul(matrix, cholesky_solve(chol, rhs), False, False, true);
end;
{$ENDREGION}

{$REGION 'gen_resource_variable_ops'}
{ gen_resource_variable_ops }

class function gen_resource_variable_ops.assign_sub_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignSubVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    Result := nil;
end;

class function gen_resource_variable_ops.assign_add_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignAddVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    var _op := tf.OpDefLib._apply_op_helper('AssignAddVariableOp', name, [ GetArg('resource',resource), GetArg('value',value) ]);
    Result := _op;
end;

class function gen_resource_variable_ops.assign_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    var _op := tf.OpDefLib._apply_op_helper('AssignVariableOp', name, [ GetArg('resource',resource), GetArg('value',value) ]);
    Result := _op;
end;

class function gen_resource_variable_ops.var_is_initialized_op(resource: TFTensor; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        var res := tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('VarIsInitializedOp', name, [resource]));
        Exit( res[0] );
    end;
    var _op := tf.OpDefLib._apply_op_helper('VarIsInitializedOp', name, [ GetArg('resource',resource) ]);
    Result := _op.Output;
end;

class function gen_resource_variable_ops.var_handle_op(dtype: TF_DataType; shape: TFShape; container, shared_name, name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        var pdtype : TParameter;
        pdtype.sNome := 'dtype';
        pdtype.vValue:= TValue.From<Integer>( Ord(dtype) );

        var pshape : TParameter;
        pshape.sNome := 'shape';
        pshape.vValue:= TValue.From< TArray<Int64> >(shape.Dims);

        var pcontainer : TParameter;
        pcontainer.sNome := 'container';
        pcontainer.vValue:= container;

        var pshared_name : TParameter;
        pshared_name.sNome := 'shared_name';
        pshared_name.vValue:= shared_name;

        var pallowed_devices : TParameter;
        pallowed_devices.sNome := 'allowed_devices';
        var v : TArray<AnsiString> :=[];
        pallowed_devices.vValue :=   TValue.From<  TArray<AnsiString> >(v);

        var dAtrr := TUtils.ConvertToDict([pdtype,pshape,pcontainer,pshared_name,pallowed_devices]) ;
        var OpExecInfo := TFastPathOpExecInfo.Create('VarHandleOp', name,[]);
        OpExecInfo.attrs := dAtrr;

        var res := tf.Runner.TFE_FastPathExecute( OpExecInfo );
        Exit( res[0] );
    end;

    var _op := tf.OpDefLib._apply_op_helper('VarHandleOp', name, [ GetArg('dtype',dtype), GetArg('shape',TValue.From< TFShape >(shape)), GetArg('container',container), GetArg('shared_name',shared_name) ]);
    Result := _op.Output;
end;

class function gen_resource_variable_ops.destroy_resource_op(resource: TFTensor; ignore_lookup_error: Boolean; name: string): TFTensor;
begin
    Result :=  tf.Context.ExecuteOp('DestroyResourceOp', name, ExecuteOpArgs.Create([resource])
                                                      .SetAttributes(['ignore_lookup_error', ignore_lookup_error ])).First;
end;

class function gen_resource_variable_ops.read_variable_op(resource: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result :=  tf.Context.ExecuteOp('ReadVariableOp', name, ExecuteOpArgs.Create([resource])
                                                      .SetAttributes(['dtype', dtype ])).First;
end;

class function gen_resource_variable_ops.resource_gather(resource, indices: TFTensor; dtype: TF_DataType; batch_dims: Integer; validate_indices: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('VarHandleOp', name, [ GetArg('resource',resource),
                                                                   GetArg('indices',indices),
                                                                   GetArg('dtype',dtype),
                                                                   GetArg('batch_dims',batch_dims),
                                                                   GetArg('validate_indices',validate_indices) ]);
    Result := _op.Output;
end;
{$ENDREGION}

{$REGION 'resource_variable_ops'}
{ resorce_variable_ops }

class function resource_variable_ops.eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name, name: string; graph_mode: Boolean): TFTensor;
begin
    var dtype := TDtypes.as_base_dtype(initial_value.dtype);
    Result :=  variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name, graph_mode, initial_value);
end;

class function resource_variable_ops.is_resource_variable(vVar: IVariableV1): Boolean;
begin
    Result := vVar is ResourceVariable;
end;

class function resource_variable_ops.shape_safe_assign_variable_handle(tHandle: TFTensor; shape: TArray<Integer>; value: TFTensor; name: string): TFOperation;
begin
    var value_tensor := Tops.convert_to_tensor(value);
    Result           := gen_resource_variable_ops.assign_variable_op(tHandle, value_tensor, name)
end;

class function resource_variable_ops.variable_handle_from_shape_and_dtype(shape: TFShape; dtype: TF_DataType; shared_name, name: string; graph_mode: Boolean;
                                                 initial_value: TFTensor): TFTensor;
begin
    var container := Tops.get_default_graph.Container;
    var handle    := gen_resource_variable_ops.var_handle_op(dtype,shape, container, shared_name, name );
    if initial_value = nil then
        initial_value := handle;
    if graph_mode then
    begin
        var full_handle_data := _combine_handle_data(handle, initial_value);
        _set_handle_shapes_and_types(handle, full_handle_data, graph_mode);
        Result := handle;
        Exit;
    end else
    begin
        // We do not want two distinct ResourceVariable objects for the same
        // underlying resource in the runtime.
        // When in eager mode, explicitly ensure so here. When in graph mode, it's
        // ensured by always generating different variable names.
        {$HINTS OFF}
        var exists := gen_resource_variable_ops.var_is_initialized_op(handle);

        // We create an assert Op instead of checking right away in order to be
        // compatible with ASYNC execution mode. Further, since not all devices
        // support string tensors, we encode the assertion string in the Op name
        (*gen_logging_ops.assert(gen_math_ops.logical_not(exists),
            new[] { exists },
            name: "EagerVariableNameReuse");*)
        var handle_data : THandleData         := THandleData.Create;
        var item        : THandleShapeAndType :=  THandleShapeAndType.Create;
        item.Shape := TUtils.as_shape_proto(shape);
        item.Dtype := TDtypes.as_datatype_enum(dtype);
        handle_data.ShapeAndTypes.Add(item);
        _set_handle_shapes_and_types(handle, handle_data, graph_mode);
        Result := handle;
    end;
end;

class function resource_variable_ops._combine_handle_data(handle, initial_value: TFTensor): THandleData;
begin
    var variable_handle_data := get_eager_safe_handle_data(initial_value);

    if initial_value.dtype <> Tdtypes.cvariant then
        Exit(variable_handle_data);
    raise TFException.Create('Not Implemented') ;
end;

class procedure resource_variable_ops._set_handle_shapes_and_types(tensor: TFTensor; handle_data: THandleData; graph_mode: Boolean);
begin
    if not graph_mode then
        Exit;
    var size := handle_data.ShapeAndTypes.Count;
    var types : TArray<TDataType> ; SetLength(types,size);
    var ranks : TArray<Integer>;   SetLength(ranks,size);
    for var i := 0 to size -1 do
    begin
        var shapeAndType := handle_data.ShapeAndTypes[i];
        types[i] := shapeAndType.Dtype;
        if shapeAndType.Shape.UnknownRank then ranks[i] := -1
        else                                   ranks[i] := shapeAndType.Shape.Dims.Count;
    end;
end;

class function resource_variable_ops.get_eager_safe_handle_data(handle: TFTensor): THandleData;
begin
    if handle.Handle = nil then
    begin
        var data : THandleData         := THandleData.Create ;
        var item : THandleShapeAndType := THandleShapeAndType.Create;
        item.Shape := TUtils.as_shape_proto(handle.shape);
        item.Dtype := TDtypes.as_datatype_enum(handle.dtype);
        data.ShapeAndTypes.Add(item);
        Result := data;
    end else
    begin
        var protoByte := handle.BufferToArray;
        var Loader: TpbLoader; Loader.Init;
        Loader.Pb.Init(@protoByte[0],Length(protoByte),false);
        var protoHandle : THandleData;
        Loader.LoadHandleData(protoHandle);
        Result := protoHandle;
    end;
end;
{$ENDREGION}

{$REGION 'stateless_random_ops'}
{ stateless_random_ops }

class function stateless_random_ops.stateless_random_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: TArray<Integer>; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([TValue.From<TFShape>(shape), TValue.From<TArray<Integer>>(seed), mean, stddev]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'stateless_random_normal', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                name := v1.ToString;
                                var shape_tensor := _ShapeTensor(shape);
                                var mean_tensor  := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor:= Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                if seed = nil then
                                begin
                                     var f_rnd : Integer := Random(Integer.MaxValue);
                                     seed := [ f_rnd, 0 ];
                                end;

                                var key_count := _get_key_counter(seed, 3);
                                var key     := key_count.Value1;
                                var counter := key_count.Value2;

                                var rnd := gen_random_ops.stateless_random_normal_v2(shape_tensor, key, counter, 3, dtype);
                                var value := math_ops.add(TTensor(rnd) * stddev, mean_tensor, name);

                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function stateless_random_ops._get_key_counter(seed: TArray<Integer>; alg: Integer): Tuple<TFTensor, TFTensor>;
begin
    var results := gen_random_ops.stateless_random_get_key_counter(seed);
    Result := Tuple.Create(results[0], results[1]);
end;

class function stateless_random_ops._ShapeTensor(shape: TArray<Integer>): TFTensor;
begin
    Result := Tops.convert_to_tensor(shape, dtInvalid, 'shape');
end;
{$ENDREGION}

{$REGION 'gen_sparse_ops'}
{ gen_sperse_ops }

class function gen_sparse_ops.sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TArray<Integer>; sparse_values, default_value: T; validate_indices: boolean;
  name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('SparseToDense', name,[ GetArg('sparse_indices',sparse_indices),
                                                                    GetArg('output_shape', TValue.From< TArray<Integer> >(output_shape)),
                                                                    GetArg('sparse_values',TValue.From<T>(sparse_values)),
                                                                    GetArg('default_value',TValue.From<T>(default_value)),
                                                                    GetArg('validate_indices',validate_indices ) ] );
    Result := _op.output;
end;


class function gen_sparse_ops.sparse_to_dense<T>(sparse_indices, output_shape, sparse_values: TFTensor; default_value: T; validate_indices: boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('SparseToDense', name,[ GetArg('sparse_indices',sparse_indices),
                                                                    GetArg('output_shape', output_shape),
                                                                    GetArg('sparse_values',sparse_values),
                                                                    GetArg('default_value',TValue.From<T>(default_value)),
                                                                    GetArg('validate_indices',validate_indices ) ] );
    Result := _op.output;
end;

{$ENDREGION}

{$REGION 'gen_state_ops'}
{ gen_state_ops }

class function gen_state_ops.variable_v2(shape: TArray<Integer>; dtype: TF_DataType; name, container, shared_name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('VariableV2', name, [ GetArg('dtype',dtype),GetArg('shape',TValue.From< TArray<Integer> >(shape)),GetArg('container', container),GetArg('shared_name', shared_name)]);
    Result := _op.outputs[0];

    // test
    //
    var _attrs := TDictionary<string, TValue>.Create;
    try
      _attrs.Add('dtype', _op.get_attr('dtype') );
      _attrs.Add('shape', _op.get_attr('shape') );
      _attrs.Add('container', _op.get_attr('container') );
       _attrs.Add('shared_name', _op.get_attr('shared_name') );
    finally
      _attrs.free;
    end;
end;

class function gen_state_ops.assign<T>(ref: T; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('Assign', name, ExecuteOpArgs.Create([TValue.From<T>(ref),value])
                            .SetAttributes(['validate_shape', validate_shape,'use_locking',use_locking ])).First;
end;

class function gen_state_ops.assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('AssignAdd', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('value',TValue.From<T>(value)),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('AssignSub', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('value',value),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.scatter_add(ref: IVariableV1; indices, updates: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ScatterAdd', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('indices',indices),GetArg('updates',updates),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.is_variable_initialized(ref: RefVariable; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('IsVariableInitialized', name, [ GetArg('ref',TValue.From<RefVariable>(ref))]);
    Result := _op.outputs[0];
end;
{$ENDREGION}


end.
