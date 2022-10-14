unit Tensorflow.array_ops;
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils,
         rtti,
         Spring,
         Spring.Collections.Enumerable,
         Spring.Collections.Lists,
         TF4D.Core.CApi,
         TensorFlow.DApiBase,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context ;

type

  array_ops = record
   private
     class function constant(value: TValue; dtype : TF_DataType= DtInvalid; shape: TArray<Integer>= nil; name : AnsiString = 'Const'; verify_shape : Boolean = false):TFTensor; static;
     class function _get_dtype_from_nested_lists(list_or_tuple: TArray<TValue>): TF_DataType; static;
     class function _constant_if_small<T>(value: T; shape: TFShape; dtype: TF_DataType; name: string): TFTensor; overload; static;
     class function size_internal<T>(input: T; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor;static;
     class function _apply_mask_1d(reshaped_tensor: TFTensor; mask: TFTensor; axis: Integer = 0): TFTensor;static;
   public
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
     class function zeros(shape: TFTensor; dtype: TF_DataType; name: AnsiString) : TFTensor; overload; static;
     class function zeros(shape: TFShape; dtype: TF_DataType = TF_DataType.TF_FLOAT; name : string = ''): TFTensor; overload; static;
     class function size<T>(input: T; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32):TFTensor;static;
     class function stack(values: TValue; axis: Integer = 0; name: string = 'stack'):TFTensor;static;
     class function identity(input: TFTensor; name: String = ''): TFTensor; static;
     class function expand_dims(input: TFTensor; axis: Integer = -1; name: string = ''): TFTensor; static;
     class function boolean_mask<T1, T2>(tensor: T1; mask: T2; name: string = 'boolean_mask'; axis: Integer = 0): TFTensor; static;
     class function shape_internal(input: TFTensor; name: string = ''; optimize: Boolean = true; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor; static;
     class function reshape(tensor: TFTensor; shape: TFTensor; name: string = ''): TFTensor; overload; static;
     class function reshape(tensor: TFTensor; shape: TFShape; name: string = ''): TFTensor; overload; static;
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
     /// <summary>
     /// Concatenates tensors along one dimension.
     /// </summary>
     /// <param name="values"></param>
     /// <param name="axis"></param>
     /// <param name="name"></param>
     /// <returns></returns>
     class function concat(values: TArray<TFTensor>; axis: Integer; name: string = 'concat'): TFTensor; static;
     class function where(condition: TFTensor; x : PValue= nil ; y : PValue= nil; name: string = ''): TFTensor; static;
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
  end;

implementation
        uses Tensorflow,
             Numpy,
             TensorFlow.Ops,
             Tensorflow.gen_array_ops,
             TensorFlow.gen_math_ops,
             Tensorflow.NameScope,
             Tensorflow.Utils,
             TensorFlow.Constant_op,
             TensorFlow.EagerTensor,
             TensorFlow.Variable;

{ array_ops }

class function array_ops.concat(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    if Length(values) = 1 then // Degenerate case of one tensor.
    begin
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Shape'),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var t := Tops.convert_to_tensor(axis, TF_DataType.TF_INT32, 'concat_dim' );
                                                Result := identity(values[0], v1.ToString);;
                                            end );
        Exit;
    end;
    Result := gen_array_ops.concat_v2(values, axis, name);
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

class function array_ops.placeholder(_dtype: TF_DataType; _shape: PTFShape; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
       raise Exception.Create('tf.placeholder is not compatible with eager execution.');

   var _op := tf.OpDefLib._apply_op_helper('Placeholder', name, [ GetArg('dtype',TValue(_dtype)), GetArg('shape',TValue.From<TFShape>(_shape^)) ]);
   Result := _op.Output;

end;

class function array_ops.reshape(tensor: TFTensor; shape: TFShape; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

class function array_ops.reshape(tensor, shape: TFTensor; name: string): TFTensor;
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
                                                    end;
                                                end;

                                                Result := gen_array_ops.size(input_tensor, out_type, name);
                                            end );
end;

class function array_ops.squeeze(input: TFTensor; axis: TArray<Integer>; name: string): TFTensor;
begin
    Result := gen_array_ops.squeeze(input, axis, name)
end;

class function array_ops.stack(values: TValue; axis: Integer; name: string): TFTensor;
begin
    if axis = 0 then
     // If the input is a constant list, it can be converted to a constant op
     Exit( Tops.convert_to_tensor(values, DtInvalid,name) );

     raise TFException.Create('Not Implemented ("array_ops.stack")');
end;

class function array_ops.where(condition: TFTensor; x, y: PValue; name: string): TFTensor;
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
                                                var output := gen_array_ops.fill(shape, constant_op.constant(Single(1), dtype, 'Const'), name);
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

class function array_ops.zeros(shape: TFTensor; dtype: TF_DataType; name: AnsiString) : TFTensor;
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
end.

