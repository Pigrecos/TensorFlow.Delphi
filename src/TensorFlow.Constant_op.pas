unit TensorFlow.Constant_op;
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
     uses System.SysUtils, System.Rtti,  System.TypInfo,
          Spring,
          Spring.Collections.Dictionaries,
          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.Context,

          ProtoGen.Tensor,
          Protogen.tensorShape,
          ProtoGen.attrValue;

type

 constant_op = class
    private
      class function convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType=DtInvalid): TFTensor; overload;
      class function _eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
      class function _eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
    public
      class function convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;
      class function convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;overload;
      /// <summary>
      /// Creates a constant tensor.
      ///
      /// The resulting tensor is populated with values of type `dtype`, as
      /// specified by arguments `value` and (optionally) `shape`
      /// </summary>
      /// <param name="value">A constant value (or list) of output type `dtype`.</param>
      /// <param name="dtype">The type of the elements of the resulting tensor.</param>
      /// <param name="shape">Optional dimensions of resulting tensor.</param>
      /// <param name="name">Optional name for the tensor.</param>
      /// <returns></returns>
      class function constant(value: TValue; dtype : TF_DataType= DtInvalid; shape : PTFShape = nil; verify_shape : Boolean = false; allow_broadcast : Boolean = true; name : AnsiString = 'Const'): TFTensor; overload;
      class function constant(value: TValue; dtype : TF_DataType; name : AnsiString = 'Const'): TFTensor; overload;
      /// <summary>
      /// Function to convert Shape to Tensor.
      /// </summary>
      /// <param name="s"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="as_ref"></param>
      /// <returns></returns>
      class function _tensor_shape_tensor_conversion_function(s: TFShape; dtype: TF_DataType = TF_DataType.DtInvalid; name: string = ''; as_ref : Boolean = false) : TFTensor;
 end;

implementation
       uses System.Math, Tensorflow, TensorFlow.Ops,Tensorflow.Utils, Tensorflow.math_ops, TensorFlow.EagerTensor, Numpy.Axis, NumPy.NDArray, Oz.Pb.Classes;

{ constant_op }

class function constant_op.constant(value: TValue; dtype: TF_DataType; name: AnsiString): TFTensor;
begin
    Result := constant(value, dtype, nil, false, True, name);
end;

class function constant_op.constant(value: TValue; dtype: TF_DataType; shape: PTFShape; verify_shape, allow_broadcast: Boolean; name: AnsiString): TFTensor;
begin
    if value.IsEmpty then
        Exit(nil);

    if tf.executing_eagerly then
        Result := convert_to_eager_tensor(value, dtype, shape, name, verify_shape, allow_broadcast)
    else
        Result := convert_to_graph_tensor(value, dtype, shape, name, verify_shape, allow_broadcast);
end;

class function constant_op.convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape,
  allow_broadcast: Boolean): TFTensor;
var
  v : TpbOneof;
begin
    var g : TFGraph := TOps.get_default_graph;

    var tp := TUtils.make_tensor_proto(value, dtype,@shape, verify_shape, allow_broadcast);

    var tensor_value : TAttrValue;
    tensor_value.Init;
    v.tag   := TAttrValue.ftTensor;
    v.value := TValue.From<TTensorProto>(tp);
    tensor_value.Value := v;

    var dtype_value : TAttrValue;
    dtype_value.Init;
    v.tag   := TAttrValue.ftType;
    v.value := TValue.From<Integer>( Ord(dtype)  );
    dtype_value.Value := v;

    var attrs := TDictionary<string, TAttrValue>.Create;

    attrs.Add('value',tensor_value);
    attrs.Add('dtype',dtype_value);

    var oper := g.create_op(
        'Const',
        [],
        [TF_DataType(dtype_value.Value.value.AsType<Integer>)],
        [],
        name,
        attrs);

    Result := oper.outputs[0];
end;

class function constant_op._eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(tensor.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(shape), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ tensor, dims_t ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'Tshape', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Reshape', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function constant_op._tensor_shape_tensor_conversion_function(s: TFShape; dtype: TF_DataType; name: string;
  as_ref: Boolean): TFTensor;
begin
    var s_list := s.dims;
    var int64_value : Int64 := 0;
    for var dim in s_list do
    begin
        if dim > Power(2, 31) then
        begin
            int64_value := dim;
            break;
        end;
    end;
    if  int64_value > 0 then dtype := TF_DataType.TF_INT64
    else                     dtype := TF_DataType.TF_INT32;
    if string.IsNullOrEmpty(name) then
        name := 'shape_as_tensor';
    Result := constant_op.constant(TValue.From< TArray<Int64> >(s_list), dtype, name);
end;

class function constant_op._eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(value.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(dims), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ dims_t, value ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'index_type', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Fill', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function constant_op.convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape, allow_broadcast: Boolean): TFTensor;
begin
    var t := convert_to_eager_tensor(value, tf.Context, dtype);

    if ( PTFShape(shape) = nil) or (shape.IsNull) then
        Exit(t);

    if t.shape.Equals( TValue.From<TFShape>(shape)) then
        Exit(t);

    if verify_shape then
        raise Exception.Create( Format('Expected Tensor''s shape: %s, got %s.',[shape.ToString,t.Shape.ToString]));

    var num_t := t.shape.size;
    if num_t = shape.size then
        Exit(_eager_reshape(t, shape^, tf.Context) );
    if num_t = 1 then
    begin
        if t.dtype = Tdtypes.cbool then
            raise Exception.Create('Not Implemented')
        else
            Exit( _eager_fill(shape^, t, tf.Context) );
    end;

    raise Exception.Create('Not Implemented')
end;

class function constant_op.convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType): TFTensor;
begin
    ctx.ensure_initialized;
    var tipo : PTypeInfo;
    tipo:= value.TypeInfo;
    // convert data type
    if (dtype <> TF_DataType.DtInvalid) and
       (string.LowerCase(tipo.Name) <> 'tndarray') and
       (value.IsArray = False) and
       (dtype <> TUtils.GetDataType(value))  then
    begin
        case dtype of
            TF_DataType.TF_DOUBLE: value := value.AsType<Double>;
            TF_DataType.TF_FLOAT:  value := value.AsType<Single>;
            TF_DataType.TF_INT64:  value := value.AsType<Int64>;
            TF_DataType.TF_INT32:  value := value.AsType<Int32>;
        end;
    end
    else if (dtype <> TF_DataType.DtInvalid) and (value.IsType<TNDArray>) and  ( value.AsType<TNDArray>.Dtype = dtype ) then
    begin
        var nd := value.AsType<TNDArray>;
        value := math_ops.cast(nd, dtype);
    end;
    // non ascii char
    if (dtype = TF_DataType.TF_STRING) and (value.IsArray) and (value.GetArrayElement(0).IsType<Byte> ) then
    begin
        Result := TEagerTensor.Create(Value.AsType< TArray<Byte> >, TFShape.Scalar, TF_DataType.TF_STRING);
        Exit;
    end;
    if value.IsType<TEagerTensor>      then  Result := value.AsType<TEagerTensor>
    else if value.IsType<TNDArray>     then  Result := value.AsType<TNDArray>
    else if value.IsType<TFShape>      then
    begin
         var vval := Value.AsType<TFShape>;
         Result := TEagerTensor.Create(vval.dims, TFShape.Create([vval.ndim]),TUtils.GetDataType(Value) );
    end
    else if value.IsType<TAxis>      then
    begin
         var vval := Value.AsType<TAxis>;
         var shape : TFShape;
         if vval.IsScalar then shape := TFShape.Scalar
         else                  shape := TFShape.Create([vval.size]);
         Result := TEagerTensor.Create(vval.axis, shape,TUtils.GetDataType(Value) );
    end
    else if (value.IsType<string>) or (value.IsType<AnsiString>) then
    begin
        var vval := Value.AsType<string>;
        Result := TEagerTensor.Create([vval], TFShape.scalar );
    end
    else if (value.IsType<TArray<String>>) or (value.IsType<TArray<AnsiString>>) then
    begin
        var vval : TArray<TF_TString> := Value.AsType<TArray<TF_TString>>;
        Result := TEagerTensor.Create(vval, TFShape.Create( [ Length(vval) ] ) );
    end
    else if value.IsType<Boolean> then
    begin
        var vval := Value.AsType<Boolean>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_BOOL);
    end
    else if value.IsType<Boolean> then
    begin
        var vval := Value.AsType<Byte>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT8);
    end
    else if value.IsType<Integer> then
    begin
        var vval := Value.AsType<Integer>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT32);
    end
    else if value.IsType<Int64> then
    begin
        var vval := Value.AsType<Int64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT64);
    end
    else if value.IsType<UInt64> then
    begin
        var vval := Value.AsType<UInt64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT64);
    end
    else if (value.IsType<Single>) and (Value.TypeInfo.Name ='Single') then
    begin
        var vval : Single := Value.AsType<Single>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_FLOAT);
    end
    else if value.IsType<Single> and (Value.TypeInfo.Name ='Double') then
    begin
        var vval : Double := Value.AsType<Double>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_DOUBLE);
    end
    else if value.isArray then
    begin
        var sShape : TFShape := TUtils.GetShape(value);
        Result := TEagerTensor.Create(value, @sShape);
    end else
    begin
       raise Exception.Create('NotImplemented convert_to_eager_tensor Type: '+ Value.TypeInfo.Name);
    end;
end;

end.
