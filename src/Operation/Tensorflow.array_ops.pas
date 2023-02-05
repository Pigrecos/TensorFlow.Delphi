unit Tensorflow.array_ops;
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
         Numpy.Axis,

         TensorFlow.Context ;

type

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

implementation
        uses Tensorflow,
             Numpy,
             NumPy.NDArray,
             TensorFlow.Ops,
             Tensorflow.gen_array_ops,
             TensorFlow.gen_math_ops,
             Tensorflow.NameScope,
             Tensorflow.Utils,
             TensorFlow.Constant_op,
             TensorFlow.EagerTensor,
             TensorFlow.Variable,
             TensorFlow.Framework,
             TensorFlow.Tensor;

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
    var sShape  : TFShape := default(TFShape);
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

end.




