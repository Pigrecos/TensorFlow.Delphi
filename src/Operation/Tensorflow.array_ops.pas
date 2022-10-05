unit Tensorflow.array_ops;
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils,
         Spring,
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
             TensorFlow.EagerTensor;

{ array_ops }

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

   var _op := tf.OpDefLib._apply_op_helper('Placeholder', name, [ GetArg('dtype',TValue(_dtype)), GetArg('shape',TValue(_shape^)) ]);
   Result := _op.Output;

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

class function array_ops.stack(values: TValue; axis: Integer; name: string): TFTensor;
begin
    if axis = 0 then
     // If the input is a constant list, it can be converted to a constant op
     Exit( Tops.convert_to_tensor(values, DtInvalid,name) );

     raise TFException.Create('Not Implemented ("array_ops.stack")');
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

