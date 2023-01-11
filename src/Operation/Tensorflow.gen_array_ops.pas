unit Tensorflow.gen_array_ops;
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

implementation
          uses Tensorflow,
               TensorFlow.EagareRunner,
               TensorFlow.Ops,
               Tensorflow.Utils;
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
                    .SetAttributes(['axis', axis ])).ToArray
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

end.
