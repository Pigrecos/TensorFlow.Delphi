unit Tensorflow.gen_array_ops;

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
     class function reshape<T>(tensor: TFTensor; shape: T; name: string = ''): TFTensor;static;
     class function strided_slice(input, tBegin, tEnd, tStrides : TFTensor; begin_mask : Int64 = 0; end_mask: Int64 = 0; ellipsis_mask: Int64 = 0; new_axis_mask: Int64 = 0; shrink_axis_mask: Int64 = 0;name: string = ''): TFTensor;static;
     class function resource_strided_slice_assign(input, tBegin, tEnd, tStrides, tVvalue: TFTensor; begin_mask: Integer = 0; end_mask: Integer= 0; ellipsis_mask: Integer=0;new_axis_mask: Integer=0; shrink_axis_mask: Integer=0; name: string=''): TFTensor;static;
     /// <summary>
     /// Return a tensor with the same shape and contents as the input tensor or value.
     /// </summary>
     /// <param name="input"></param>
     /// <param name="name"></param>
     class function identity(input: TFTensor; name: string = ''): TFTensor; static;
     class function expand_dims(input: TFTensor; axis: integer; name: string = ''): TFTensor; static;
     class function batch_to_space_nd<T>(input: T; block_shape: TArray<Integer>; crops: TArray< TArray<Integer> >; name: string = ''): TFTensor; static;
     class function concat_v2(values: TArray<TFTensor>; axis: Integer; name: string = ''): TFTensor; static;
     class function shape(input: TFTensor; out_type: TF_DataType = TF_DataType.TF_INT32; name: string = '') : TFTensor; static;
     class function where(condition: TFTensor; name : string = ''): TFTensor; static;
     class function select<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string = ''): TFTensor; static;
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
  end;

implementation
          uses Tensorflow,
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

class function gen_array_ops.concat_v2(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ConcatV2', name, ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(values), axis])).FirstOrDefault(nil);
end;

class function gen_array_ops.expand_dims(input: TFTensor; axis: integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ExpandDims', name, ExecuteOpArgs.Create([ input, axis ])
                                                 .SetAttributes(['axis', TValue.From<Integer>(axis) ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.fill<T>(dims: TFTensor; value: T; name: string): TFTensor;
begin
    var v := TValue.From<T>(value);
    Result := tf.Context.ExecuteOp('Fill', name, ExecuteOpArgs.Create([dims, v])).FirstOrDefault(nil);
end;

class function gen_array_ops.gather_v2<T1, T2>(params: T1; indices: T2; axis, batch_dims: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('GatherV2', name, ExecuteOpArgs.Create([ TValue.From<T1>(params), TValue.From<T2>(indices), axis ])
                                                      .SetAttributes(['batch_dims', batch_dims ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.identity(input: TFTensor; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('Identity', name, ExecuteOpArgs.Create([input])).FirstOrDefault(nil);
end;

class function gen_array_ops.pack(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Pack', name, ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(values) ])
                                                 .SetAttributes(['axis', TValue.From<Integer>(axis) ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.reshape<T>(tensor: TFTensor; shape: T; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Reshape', name, ExecuteOpArgs.Create([tensor, TValue.From<T>(shape)])).FirstOrDefault(nil);
end;

class function gen_array_ops.resource_strided_slice_assign(input, tBegin, tEnd, tStrides, tVvalue: TFTensor; begin_mask: Integer; end_mask: Integer; ellipsis_mask: Integer;new_axis_mask: Integer; shrink_axis_mask: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ResourceStridedSliceAssign', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides, tVvalue ])
                                                 .SetAttributes(['begin_mask',      begin_mask,
                                                                 'end_mask',        end_mask,
                                                                 'ellipsis_mask',   ellipsis_mask,
                                                                 'new_axis_mask',   new_axis_mask,
                                                                 'shrink_axis_mask',shrink_axis_mask ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.shape(input: TFTensor; out_type: TF_DataType; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Shape', name, ExecuteOpArgs.Create([ input ]).SetAttributes(['out_type', out_type ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.size(input: TFTensor; out_type: TF_DataType; name: string): TFTensor;
begin
     var _op := tf.OpDefLib._apply_op_helper('Size', name, [ GetArg('input',input), GetArg('out_type',out_type) ]);
     Result := _op.outputs[0];
end;

class function gen_array_ops.squeeze(input: TFTensor; axis: TArray<Integer>; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Squeeze', name, ExecuteOpArgs.Create([ input ]).SetAttributes([ 'squeeze_dims',TValue.From< TArray<Integer> >(axis) ] ) ).FirstOrDefault(nil);
end;

class function gen_array_ops.strided_slice(input, tBegin, tEnd, tStrides: TFTensor; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Int64;
  name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StridedSlice', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides ])
                                                 .SetAttributes(['begin_mask',      begin_mask,
                                                                 'end_mask',        end_mask,
                                                                 'ellipsis_mask',   ellipsis_mask,
                                                                 'new_axis_mask',   new_axis_mask,
                                                                 'shrink_axis_mask',shrink_axis_mask]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.where(condition: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Where', name, [ GetArg('input',condition) ]);
     Result := _op.outputs[0];
end;

class function gen_array_ops.select<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Select', name, ExecuteOpArgs.Create([ condition, TValue.From<Tx>(x), TValue.From<Ty>(y) ]) ).FirstOrDefault(nil);
end;

end.
