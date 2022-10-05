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
  end;

implementation
          uses Tensorflow, TensorFlow.Ops, Tensorflow.Utils;
{ gen_array_ops }

class function gen_array_ops.fill<T>(dims: TFTensor; value: T; name: string): TFTensor;
begin
    var v := TValue.From<T>(value);
    Result := tf.Context.ExecuteOp('Fill', name, ExecuteOpArgs.Create([dims, v])).FirstOrDefault(nil);
end;

class function gen_array_ops.identity(input: TFTensor; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('Identity', name, ExecuteOpArgs.Create([input])).FirstOrDefault(nil);
end;

class function gen_array_ops.pack(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Pack', name, ExecuteOpArgs.Create([ TValue.From< TArray<TFTensor> >(values) ])
                                                 .SetAttributes([ TValue.From<Integer>(axis) ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.reshape<T>(tensor: TFTensor; shape: T; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Reshape', name, ExecuteOpArgs.Create([tensor, TValue.From<T>(shape)])).FirstOrDefault(nil);
end;

class function gen_array_ops.resource_strided_slice_assign(input, tBegin, tEnd, tStrides, tVvalue: TFTensor; begin_mask: Integer; end_mask: Integer; ellipsis_mask: Integer;new_axis_mask: Integer; shrink_axis_mask: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ResourceStridedSliceAssign', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides, tVvalue ])
                                                 .SetAttributes([ begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask ]) ).FirstOrDefault(nil);
end;

class function gen_array_ops.size(input: TFTensor; out_type: TF_DataType; name: string): TFTensor;
begin
     var _op := tf.OpDefLib._apply_op_helper('Size', name, [ GetArg('input',input), GetArg('out_type',out_type) ]);
     Result := _op.outputs[0];
end;

class function gen_array_ops.strided_slice(input, tBegin, tEnd, tStrides: TFTensor; begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask: Int64;
  name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StridedSlice', name, ExecuteOpArgs.Create([ input, tBegin, tEnd, tStrides ])
                                                 .SetAttributes([ begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask ]) ).FirstOrDefault(nil);
end;

end.
