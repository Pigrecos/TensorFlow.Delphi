unit TensorFlow.array_grad;
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
         Generics.Defaults,
         System.Math,

         Spring,
         Spring.Collections.Enumerable,

         TF4D.Core.CApi,
         TensorFlow.Core,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         Tensorflow.Gradient;

type
    array_grad = class
      private
        FGradFunction     : TArray<TGradFunc>;
      public
        constructor Create;
        destructor Destroy;  override;

        property GradFunction  : TArray<TGradFunc> read FGradFunction;
    end;

implementation
      uses Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops,
           TensorFlow.Tensor,
           TensorFlow.Operations,
           TensorFlow.Slice,

           NumPy.NDArray;

// [RegisterGradient("BroadcastTo")]
function _BroadcastToGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad            := grads[0];
    var input_value     := op.inputs[0];
    var broadcast_shape := op.inputs[1];
    var input_value_shape := array_ops.shape(input_value);
    var _reduction_axes   := gen_array_ops.broadcast_gradient_args(broadcast_shape, input_value_shape);
    var reduction_axes    := _reduction_axes.Value2;
    var updates_grad_reshaped := math_ops.reduce_sum(grad, reduction_axes, true);
    var updates_grad          := array_ops.reshape(updates_grad_reshaped, input_value_shape);
    Result := [updates_grad, nil];
end;
/// <summary>
/// Extract the shapes of a set of input tensors.
/// </summary>
/// <param name="inputs"></param>
/// <returns></returns>
function _ExtractInputShapes(inputs: TArray<TFTensor>): TArray<TFTensor>;
begin
    var sizes : TArray<TFTensor>; SetLength(sizes, Length(inputs));
    var fully_known : Boolean := true;
    for var i := 0 to Length(inputs)-1 do
    begin
        var x := inputs[i];
        var input_shape := array_ops.shape(x);
        if (not (input_shape is TFTensor)) or (input_shape.op.tipo <> 'Const') then
        begin
            fully_known := false;
            break;
        end;
        sizes[i] := input_shape;
    end;
    if fully_known then Result := sizes
    else                Result := gen_array_ops.shape_n(inputs);
end;
/// <summary>
/// Gradient for concat op.
/// </summary>
/// <param name="op">An operation.</param>
/// <param name="grad">
/// `Tensor` or `IndexedSlices` representing the gradients with respect
/// to each output of the op.
/// </param>
/// <param name="start_value_index">An integer index of the first value in the op.inputs.</param>
/// <param name="end_value_index">An integer index of the last value in the op.inputs.</param>
/// <param name="dim_index">An interger index of concat_dim or axis parameter in op.inputs.</param>
/// <returns>
/// Tensors representing the partial gradients with respect to each input
/// of the op.
/// </returns>
function _ConcatGradHelper(op: TFOperation; grad: TFTensor; start_value_index, end_value_index, dim_index: Integer): TArray<TFTensor>;
begin
    // Degenerate concatenation, just return grad.
    if op.inputs.Count = 2 then
    begin
        if end_value_index <= dim_index then Result :=  [ grad, nil ]
        else                                 Result :=  [ nil, grad ];
        Exit;
    end;
    var concat_dim   := op.inputs[dim_index];
    var takeCount : Integer;
    if end_value_index = -1 then  takeCount := op.inputs.Count - 1
    else                          takeCount := end_value_index - start_value_index;
    var input_values := Enumerable<TFTensor>.Create(op.inputs.inputs).Skip(start_value_index)
        .Take(takeCount)
        .ToArray;
    var out_grads := TList<TFTensor>.Create;
    if concat_dim is TEagerTensor then
    begin
        var dim_int : Integer := Integer(TTensor(concat_dim) );
        var non_neg_concat_dim : Integer;
        if dim_int < 0 then non_neg_concat_dim :=  input_values[0].rank + dim_int
        else                non_neg_concat_dim :=  dim_int mod input_values[0].rank;
        var sizes : TArray<Int64> := [];
        for var i := 0 to Length(input_values) -1 do
           sizes := sizes + [ input_values[i].shape[non_neg_concat_dim] ] ;
        var sizes_tensor := constant_op.constant( TValue.From< TArray<Int64> >(sizes));
        out_grads.AddRange( array_ops.split(grad, sizes_tensor, non_neg_concat_dim) );
    end
    else if constant_op.is_constant(concat_dim) then
    begin
        (*If concat_dim is a constant defined in a different context,
        then we duplicate it in the current context to avoid passing it
        through an Enter node.
        This is a small optimization in general, but it is required when
        compiling with XLA, as XLA needs the concat input to be folded into a
        constant.*)
        var grad_context := control_flow_util.GetOutputContext(grad.op);
        var dim_context  := control_flow_util.GetOutputContext(concat_dim.op);
        if dim_context <> grad_context then
        begin
            var value  := TUtils.constant_value(concat_dim);
            concat_dim := constant_op.constant(value, concat_dim.dtype,'Const');
        end;
        // Using mod here for convenience since concat_dim is already verified
        // in concat implementation to be within the allowed [-rank, rank) range.
        var non_neg_concat_dim := TTensor(concat_dim) mod array_ops.rank(input_values[0]);
        // Get the inputs' tensor shapes
        var sizes := _ExtractInputShapes(input_values);
        (* The magic number of 16 was found through benchmarking a range of sizes
         on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
         cases when switching implementations at N=16, but it is possible that
         there will be a small number of performance regressions.*)
        if Length(sizes) > 16 then
        begin
            // extract the size of each input along the concat dimension
            var slice         := array_ops.slice(array_ops.stack(sizes,  1), [ non_neg_concat_dim, tf.constant(0) ], [ tf.constant(1), tf.constant(-1) ]);
            var squeeze_sizes := array_ops.squeeze(slice);
            // class function array_ops.split<T>(value: TFTensor; num_split: Integer; axis: T; name: string): TArray<TFTensor>;
            out_grads.AddRange( array_ops.split(squeeze_sizes, Integer(TTensor(non_neg_concat_dim)), grad) );
        end else
        begin
            var offset := gen_array_ops.concat_offset(non_neg_concat_dim, sizes);
            for var begin_size in TUtils.zip<TFTensor>( TList<TFTensor>.create(offset), TList<TFTensor>.create(sizes) ) do
                out_grads.Add(gen_array_ops.slice(grad, begin_size.Value1, begin_size.Value2));
        end;
    end;
    if end_value_index <= dim_index then  Result := out_grads.ToArray + [nil]
    else                                  Result :=  [nil] + out_grads.ToArray ;
end;
// [RegisterGradient("ConcatV2")]
function _ConcatV2Grad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    Result := _ConcatGradHelper(op, grad, 0, -1, -1);
end;
function _ReshapeToInput(op: TFOperation; grad: TFTensor): TFTensor;
begin
    Result := array_ops.reshape(grad, array_ops.shape(op.inputs[0]) );
end;
// [RegisterGradient("ExpandDims")]
function _ExpandDimsGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ _ReshapeToInput(op, grads[0]), nil ];
end;
/// <summary>
/// Gradient for GatherV2 op.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
// [RegisterGradient("GatherV2")]
function _GatherV2Grad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad    := grads[0];
    var params  := op.inputs[0];
    Tops.colocate_with(params);
    var params_shape := array_ops.shape(params, '', tf.int64_t);
    params_shape     := math_ops.cast(params_shape, tf.int32_t);
    var indices      := op.inputs[1];
    var indices_size := array_ops.expand_dims(array_ops.size(indices), 0);
    var axis         := op.inputs[2];
    var axis_static  := TUtils.constant_value(axis);
    // For axis 0 gathers, build an appropriately shaped IndexedSlices.
    if Integer(NDarray(axis_static)) = 0 then
    begin
        var params_tail_shape := params_shape._slice( Slice.Create(1,nil) );
        var values_shape      := array_ops.concat( [indices_size, params_tail_shape], 0);
        var values            := array_ops.reshape(grad, values_shape);
        indices               := array_ops.reshape(indices, indices_size);
        Result := [ IndexedSlices.create(values, indices, params_shape), nil, nil ];
        Exit;
    end;
    Result := [ nil, nil ];
end;
// [RegisterGradient("Reshape")]
function _ReshapeGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ array_ops.reshape(grads[0], array_ops.shape(op.inputs[0])), nil ];
end;
// [RegisterGradient("Pack")]
function _PackGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var num  := op.get_attr<Integer>('N');
    var axis := op.get_attr<integer>('axis');
    Result := array_ops.unstack(grad, @num, axis);
end;
// [RegisterGradient("Unpack")]
function _UnpackGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var axis := op.get_attr<Integer>('axis');
    Result :=[ array_ops.stack(grads, axis) ];
end;
// [RegisterGradient("Pad")]
function _PadGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var a    := op.inputs[1];
    var aValue : TArray<TFTensor> := [ array_ops.rank(x), constant_op.constant(1) ];
    var size := array_ops.stack(aValue);
    var aB : TArray<Integer> := [0, 0];
    var _begin := constant_op.constant( aB );
    var pad_before := array_ops.slice(a, _begin, size);
    // Make it a 1-D tensor.
    _begin     := array_ops.reshape(pad_before, [ -1 ]);
    size       := array_ops.shape(x);
    var x_grad := array_ops.slice(grad, _begin, size);
    if op.inputs.Count = 3 then
       Result := [ x_grad, nil, nil ]
    else
       Result := [ x_grad, nil ];
end;
// [RegisterGradient("Split")]
function _SplitGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ nil, array_ops.concat(grads, op.inputs[0]) ];
end;
// [RegisterGradient("Slice")]
function _SliceGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad      := grads[0];
    var input_vec := op.inputs[0];
    var begin_vec := op.inputs[1];
    var input_rank:= array_ops.rank(input_vec);
    var slice_size:= array_ops.shape(op.outputs[0]);
    var aValue : TArray<TFTensor> := [ input_rank, Tops.convert_to_tensor(Integer(1)) ];
    var shape      := array_ops.stack( aValue );
    var before_pad := array_ops.reshape(begin_vec, shape);
    var after_pad  := array_ops.reshape( TTensor(array_ops.shape(input_vec)) - slice_size - begin_vec, shape);
    var paddings   := array_ops.concat([ before_pad, after_pad ], 1);
    Result := [ array_ops.pad(grad, paddings), nil, nil ];
end;
// [RegisterGradient("Squeeze")]
function _SqueezeGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ _ReshapeToInput(op, grads[0]) ];
end;
// [RegisterGradient("StopGradient")]
function _NoGradient(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ nil ];
end;
/// <summary>
/// Gradient for StridedSlice op.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
// [RegisterGradient("StridedSlice")]
function _StridedSliceGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad    := grads[0];
    var _begin  := op.inputs[1];
    var _end    := op.inputs[2];
    var strides := op.inputs[3];
    var x             := array_ops.shape(op.inputs[0], '', _begin.dtype);
    var x_static      := TUtils.constant_value(x);
    var begin_static  := TUtils.constant_value(_begin);
    var end_static    := TUtils.constant_value(_end);
    var strides_static:= TUtils.constant_value(strides);
    Result :=
    [
        array_ops.strided_slice_grad(x_static,
                                     begin_static,
                                     end_static,
                                     strides_static,
                                     grad,
                                     op.get_attr<Int64>('begin_mask'),
                                     op.get_attr<Int64>('end_mask'),
                                     op.get_attr<Int64>('ellipsis_mask'),
                                     op.get_attr<Int64>('new_axis_mask'),
                                     op.get_attr<Int64>('shrink_axis_mask')),
        nil,
        nil,
        nil
    ];
end;
// [RegisterGradient("StridedSliceGrad")]
function _StridedSliceGradGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad    := grads[0];
    var _begin  := op.inputs[1];
    var _end    := op.inputs[2];
    var strides := op.inputs[3];
    Result :=
    [
        nil,
        nil,
        nil,
        array_ops.strided_slice_grad(grad,
                                     _begin,
                                     _end,
                                     strides,
                                     grad,
                                     op.get_attr<Int64>('begin_mask'),
                                     op.get_attr<Int64>('end_mask'),
                                     op.get_attr<Int64>('ellipsis_mask'),
                                     op.get_attr<Int64>('new_axis_mask'),
                                     op.get_attr<Int64>('shrink_axis_mask'))];
end;
// [RegisterGradient("Transpose")]
function _TransposeGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var p := op.inputs[1];
    Result := [ array_ops.transpose(grads[0], array_ops.invert_permutation(p)), nil ];
end;

{ array_grad }

constructor array_grad.Create;
begin
    FGradFunction := [ TGradFunc.Create('BroadcastTo',     _BroadcastToGrad),
                       TGradFunc.Create('ConcatV2',        _ConcatV2Grad),
                       TGradFunc.Create('ExpandDims',      _ExpandDimsGrad),
                       TGradFunc.Create('GatherV2',        _GatherV2Grad),
                       TGradFunc.Create('Reshape',         _ReshapeGrad),
                       TGradFunc.Create('Pack',            _PackGrad),
                       TGradFunc.Create('Unpack',          _UnpackGrad),
                       TGradFunc.Create('Pad',             _PadGrad),
                       TGradFunc.Create('Split',           _SplitGrad),
                       TGradFunc.Create('Slice',           _SliceGrad),
                       TGradFunc.Create('Squeeze',         _SqueezeGrad),
                       TGradFunc.Create('StridedSlice',    _StridedSliceGrad),
                       TGradFunc.Create('StridedSliceGrad',_StridedSliceGradGrad),
                       TGradFunc.Create('Transpose',       _TransposeGrad)
                     ] ;
end;

destructor array_grad.Destroy;
begin
  FGradFunction := [];
  inherited;
end;

end.
