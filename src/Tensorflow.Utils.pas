unit Tensorflow.Utils;

interface
    uses System.SysUtils, System.Rtti, System.TypInfo, Spring, System.Variants,
         Spring.Collections,Spring.Collections.Dictionaries, Spring.Collections.Lists,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         TensorFlow.LowLevelAPI,

         ProtoGen.tensorShape,
         ProtoGen.types,
         ProtoGen.Tensor;

type
 Tdtypes = class
   public
     const
       cbool       : TF_DataType  = TF_DataType.TF_BOOL;
       cint8       : TF_DataType  = TF_DataType.TF_INT8;
       cint32      : TF_DataType  = TF_DataType.TF_INT32;
       cint64      : TF_DataType  = TF_DataType.TF_INT64;
       cuint8      : TF_DataType  = TF_DataType.TF_UINT8;
       cuint32     : TF_DataType  = TF_DataType.TF_UINT32;
       cuint64     : TF_DataType  = TF_DataType.TF_UINT64;
       cfloat32    : TF_DataType  = TF_DataType.TF_FLOAT; // is that float32?
       cfloat16    : TF_DataType  = TF_DataType.TF_HALF;
       cfloat64    : TF_DataType  = TF_DataType.TF_DOUBLE;
       ccomplex    : TF_DataType  = TF_DataType.TF_COMPLEX;
       ccomplex64  : TF_DataType  = TF_DataType.TF_COMPLEX64;
       ccomplex128 : TF_DataType  = TF_DataType.TF_COMPLEX128;
       cvariant    : TF_DataType  = TF_DataType.TF_VARIANT;
       cresource   : TF_DataType  = TF_DataType.TF_RESOURCE;

       class function as_numpy_name(value: TF_DataType): string;
       class function as_base_dtype(value: TF_DataType): TF_DataType; overload;
       class function as_base_dtype(value: TDataType): TDataType;  overload;
       class function as_ref(value: TF_DataType): TF_DataType;
       class function as_tf_dtype(value: TValue): TF_DataType; overload;
       class function as_tf_dtype(value: PTypeInfo): TF_DataType; overload;
       class function as_tf_dtype(value: TDataType): TF_DataType; overload;
       class function get_datatype_size(tipo: TF_DataType): Integer; static;
       class function as_datatype_enum(value: TF_DataType): TDataType; static;
       class function ToIntArray(value: TArray<TF_DataType>): TArray<Integer>;
       /// <summary>
       ///
       /// </summary>
       /// <param name="type"></param>
       /// <returns><see cref="System.Type"/> equivalent to <paramref name="type"/>, if none exists, returns null.</returns>
       class function as_system_dtype(tipo: TF_DataType): PTypeInfo; static;
 end;

 TSorted = class
   public
      class function Sort<T,T1>(dict_: IDictionary<T, T1> ): IEnumerable<T>;
 end;

 TUtils = class
  private
    class function ChangeType(x: TValue; new_system_dtype: PTypeInfo): TValue;
    class function ArrayToArrayTipo<T>(a: Tarray<T>; toTipo: PTypeInfo): TArray<Integer>;

   public
      class function flatten<T>(obj : TArray<T>                        ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<T>>                ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<TArray<T>>>        ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<TArray<TArray<T>>>>): TList<T> ;  overload;

      class function tf_with<TIn, TOut>(py: TIn; action: TFunc<TIn, TOut>): TOut;
      class function GetDataType(value: TValue): TF_DataType;
      class function GetShape(value: TValue): TFShape;overload;
      class function GetShape<T>(Tval: TArray<TArray<TArray<TArray<T>>>>): TFShape;  overload;
      class function ConvertToDict(dny: TArray<TParameter>): TDictionary<string,TValue> ;
      class function isinstance(v: TValue; t : PTypeInfo):Boolean;
      class function as_shape_proto(tshape: TFShape): TTensorShapeProto; static;
      class function as_shape<T>(dims: TArray<T>): TTensorShapeProto;
      /// <summary>
      /// Create a TensorProto, invoked in graph mode
      /// </summary>
      /// <param name="values"></param>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="verify_shape"></param>
      /// <param name="allow_broadcast"></param>
      /// <returns></returns>
      class function make_tensor_proto(values: TValue; var dtype : TF_DataType; shape : PTFShape; verify_shape : Boolean= false; allow_broadcast : Boolean= false) : TTensorProto;

 end;


implementation
        uses Tensorflow, TensorFlow.Ops, NDArray,Numpy.Axis,Complex,TensorFlow.Variable;

{ TSorted }

class function TSorted.Sort<T,T1>(dict_: IDictionary<T, T1> ): IEnumerable<T>;
begin
     Result := TDictionary<T, T1>(dict_).Keys.Ordered;
end;

{ Tdtypes }

class function Tdtypes.as_datatype_enum(value: TF_DataType): TDataType;
begin
    Result := TDataType(Ord(value)) ;
end;

class function Tdtypes.ToIntArray(value: TArray<TF_DataType>): TArray<Integer>;
begin
    Result := [];
    for var i:= 0 to Length(value)-1 do
       Result := Result + [Ord(value[i]) ] ;
end;

class function Tdtypes.as_base_dtype(value: TF_DataType): TF_DataType;
begin
    if Ord(value) > 100 then Result := TF_DataType(Ord(value) - 100 )
    else                     Result := value;
end;

class function Tdtypes.as_base_dtype(value: TDataType): TDataType;
begin
    if Ord(value) > 100 then Result := TDataType(Ord(value) - 100 )
    else                     Result := value;
end;

class function Tdtypes.as_ref(value: TF_DataType): TF_DataType;
begin
    if Ord(value) < 100 then Result := TF_DataType(Ord(value) + 100 )
    else                     Result := value;
end;

class function Tdtypes.as_numpy_name(value: TF_DataType): string;
begin
    case value of
        TF_DataType.TF_STRING   : Result :='string';
        TF_DataType.TF_UINT8    : Result :='uint8';
        TF_DataType.TF_INT8     : Result :='int8';
        TF_DataType.TF_UINT32   : Result :='uint32';
        TF_DataType.TF_INT32    : Result :='int32';
        TF_DataType.TF_UINT64   : Result :='uint64';
        TF_DataType.TF_INT64    : Result :='int64';
        TF_DataType.TF_FLOAT    : Result :='float32';
        TF_DataType.TF_DOUBLE   : Result :='float64';
        TF_DataType.TF_BOOL     : Result :='bool';
        TF_DataType.TF_RESOURCE : Result :='resource';
        TF_DataType.TF_VARIANT  : Result :='variant';
    else
        Result := TEnum.GetName<TF_DataType>(value);
    end;
end;

class function Tdtypes.as_tf_dtype(value: TDataType): TF_DataType;
begin
    Result := TF_DataType(value);
end;

class function Tdtypes.as_tf_dtype(value: TValue): TF_DataType;
var
  tTipo : PTypeInfo;
  dType : TF_DataType;
begin
     dType := TF_DataType.TF_DATATYPE_UNKNOWN;

     while value.IsArray do
       value := value.GetArrayElement(0);

     tTipo:= value.TypeInfo;

     if      String.LowerCase(tTipo.TypeName) = 'integer'   then dType := TF_DataType.TF_INT32
     else if String.LowerCase(tTipo.TypeName) = 'cardinal'  then dType := TF_DataType.TF_UINT32
     else if String.LowerCase(tTipo.TypeName) = 'int64'     then dType := TF_DataType.TF_INT64
     else if String.LowerCase(tTipo.TypeName) = 'uint64'    then dType := TF_DataType.TF_UINT64
     else if String.LowerCase(tTipo.TypeName) = 'word'      then dType := TF_DataType.TF_UINT16
     else if String.LowerCase(tTipo.TypeName) = 'smallint'  then dType := TF_DataType.TF_INT16
     else if String.LowerCase(tTipo.TypeName) = 'byte'      then dType := TF_DataType.TF_UINT8
     else if String.LowerCase(tTipo.TypeName) = 'char'      then dType := TF_DataType.TF_UINT8
     else if String.LowerCase(tTipo.TypeName) = 'shortint'  then dType := TF_DataType.TF_INT8
     else if String.LowerCase(tTipo.TypeName) = 'boolean'   then dType := TF_DataType.TF_BOOL
     else if String.LowerCase(tTipo.TypeName) = 'single'    then dType := TF_DataType.TF_FLOAT
     else if String.LowerCase(tTipo.TypeName) = 'double'    then dType := TF_DataType.TF_DOUBLE
     else if String.LowerCase(tTipo.TypeName) = 'string'    then dType := TF_DataType.TF_STRING
     else if String.LowerCase(tTipo.TypeName) = 'ansistring'then dType := TF_DataType.TF_STRING;

     Result := dType;

end;

class function Tdtypes.as_tf_dtype(value: PTypeInfo): TF_DataType;
var
  tTipo : PTypeInfo;
  dType : TF_DataType;
begin
     dType := TF_DataType.TF_DATATYPE_UNKNOWN;


     while (value.Kind = tkDynArray) or (value.Kind = tkArray) do
       value := value^.TypeData^.DynArrElType^;

     tTipo:= value;

     if      String.LowerCase(tTipo.TypeName) = 'integer'   then dType := TF_DataType.TF_INT32
     else if String.LowerCase(tTipo.TypeName) = 'cardinal'  then dType := TF_DataType.TF_UINT32
     else if String.LowerCase(tTipo.TypeName) = 'int64'     then dType := TF_DataType.TF_INT64
     else if String.LowerCase(tTipo.TypeName) = 'uint64'    then dType := TF_DataType.TF_UINT64
     else if String.LowerCase(tTipo.TypeName) = 'word'      then dType := TF_DataType.TF_UINT16
     else if String.LowerCase(tTipo.TypeName) = 'smallint'  then dType := TF_DataType.TF_INT16
     else if String.LowerCase(tTipo.TypeName) = 'byte'      then dType := TF_DataType.TF_UINT8
     else if String.LowerCase(tTipo.TypeName) = 'char'      then dType := TF_DataType.TF_UINT8
     else if String.LowerCase(tTipo.TypeName) = 'shortint'  then dType := TF_DataType.TF_INT8
     else if String.LowerCase(tTipo.TypeName) = 'boolean'   then dType := TF_DataType.TF_BOOL
     else if String.LowerCase(tTipo.TypeName) = 'single'    then dType := TF_DataType.TF_FLOAT
     else if String.LowerCase(tTipo.TypeName) = 'double'    then dType := TF_DataType.TF_DOUBLE
     else if String.LowerCase(tTipo.TypeName) = 'string'    then dType := TF_DataType.TF_STRING
     else if String.LowerCase(tTipo.TypeName) = 'ansistring'then dType := TF_DataType.TF_STRING;

     Result := dType;

end;

class function Tdtypes.as_system_dtype(tipo: TF_DataType): PTypeInfo ;
begin
    case as_base_dtype(tipo) of
        TF_DataType.TF_BOOL:   Result := TypeInfo(Boolean) ;
        TF_DataType.TF_UINT8:  Result := TypeInfo(UInt8) ;
        TF_DataType.TF_INT8:   Result := TypeInfo(Int8) ;
        TF_DataType.TF_INT64:  Result := TypeInfo(Int64) ;
        TF_DataType.TF_UINT64: Result := TypeInfo(UInt64) ;
        TF_DataType.TF_INT32:  Result := TypeInfo(Int32) ;
        TF_DataType.TF_UINT32: Result := TypeInfo(UInt32) ;
        TF_DataType.TF_INT16:  Result := TypeInfo(Int16) ;
        TF_DataType.TF_UINT16: Result := TypeInfo(UInt16) ;
        TF_DataType.TF_FLOAT:  Result := TypeInfo(Single) ;
        TF_DataType.TF_DOUBLE: Result := TypeInfo(Double) ;
        TF_DataType.TF_STRING: Result := TypeInfo(String) ;
        TF_DataType.TF_COMPLEX128,
        TF_DataType.TF_COMPLEX64:  Result := TypeInfo(TComplex) ;
        else
            raise Exception.Create('Unable to convert {type} to a system data type.');
    end;
end;

class function Tdtypes.get_datatype_size(tipo: TF_DataType) : Integer;
begin
     case as_base_dtype(tipo) of
        TF_DataType.TF_BOOL     : Result := SizeOf(Boolean);
        TF_DataType.TF_UINT8    : Result := SizeOf(UInt8);
        TF_DataType.TF_INT8     : Result := SizeOf(Int8);
        TF_DataType.TF_UINT16   : Result := SizeOf(UInt16);
        TF_DataType.TF_INT16    : Result := SizeOf(Int16);
        TF_DataType.TF_UINT32   : Result := SizeOf(UInt32);
        TF_DataType.TF_INT32    : Result := SizeOf(Int32);
        TF_DataType.TF_UINT64   : Result := SizeOf(UInt64);
        TF_DataType.TF_INT64    : Result := SizeOf(Int64);
        TF_DataType.TF_FLOAT    : Result := SizeOf(Single);
        TF_DataType.TF_DOUBLE   : Result := SizeOf(Double);
        TF_DataType.TF_STRING   : Result := 1;
    else
        raise Exception.Create('TUtils.get_datatype_size - NotImplemented');
    end;
end;

{ TUtils }

class function TUtils.GetDataType(value: TValue): TF_DataType;
var
  tTipo : PTypeInfo;
begin
   tTipo:= value.TypeInfo;
   Result := TF_DATATYPE_UNKNOWN;

   case ttipo.Kind of
     tkClass,tkRecord,tkMRecord : begin
          if      string.LowerCase(string(tTipo.Name)) = 'tfshape'  then   Exit(TF_DataType.TF_INT64)
          else if string.LowerCase(string(tTipo.Name)) = 'taxis'    then   Exit(TF_DataType.TF_INT32)
          else if string.LowerCase(string(tTipo.Name)) = 'tndarray' then
          begin
             var v : TNDArray := value.AsType<TNDArray>;
             Exit(v.TensorDataType);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tftensor' then
          begin
             var v : TFTensor := value.AsType<TFTensor>;
             Exit(v.TensorDataType);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'teagertensor' then
          begin
             var v : TEagerTensor := value.AsType<TEagerTensor>;
             Exit(v.TensorDataType);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tftensors' then
          begin
             var v : TFTensors := value.AsType<TFTensors>;
             Exit(v.First.TensorDataType );
          end
          else if string.LowerCase(string(tTipo.Name)) = 'refvariable' then
          begin
             var v : RefVariable := value.AsType<RefVariable>;
             Exit(v.dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'resourcevariable' then
          begin
             var v : ResourceVariable := value.AsType<ResourceVariable>;
             Exit(v.dtype);
          end;
     end;
     tkArray,tkDynArray: begin
          Result := GetDataType( value.GetArrayElement(0) )
     end
   else
     Result := Tdtypes.as_tf_dtype(value);
   end;
end;

class function TUtils.GetShape(value: TValue): TFShape;
var
  tTipo : PTypeInfo;
begin
   tTipo:= value.TypeInfo;

   case ttipo.Kind of
     tkClass,tkRecord,tkMRecord : begin
          if string.LowerCase(string(tTipo.Name)) = 'taxis'     then
          begin
              var v : TAxis:= value.AsType<TAxis>;

              if v.isScalar then Exit( TFShape.scalar );

              var vAx : TArray<Int64>; SetLength(vAx,Length(v.axis.Value));
              Result := TFShape.Create(vAx);
              Exit;
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tndarray' then
          begin
             var v : TNDArray := value.AsType<TNDArray>;
             Result := v.Shape;
             Exit;
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tftensor' then
          begin
             var v : TFTensor := value.AsType<TFTensor>;
             Result := v.Shape;
             Exit;
          end
     end;
   end;

   if not value.IsArray then
       Exit( TFShape.scalar );

   if value.IsArray then
   begin
       var aDim : TArray<Int64>;
       while Value.IsArray do
       begin
            aDim := aDim + [ Value.GetArrayLength ];
            Value := Value.GetArrayElement(0);
       end;
       Result := TFShape.Create(aDim);
   end else
   begin
       raise Exception.Create('NotImplementedException');
   end;

end;

class function TUtils.GetShape<T>(Tval: TArray<TArray<TArray<TArray<T>>>>): TFShape;
var
  aDim : TArray<Int64>;
begin
    SetLength(aDim,4);
    aDim[0] := Length(Tval);
    aDim[1] := Length(Tval[0]);
    aDim[2] := Length(Tval[0][0]);
    aDim[3] := Length(Tval[0][0][0]);

    Result := TFShape.Create(aDim)

end;

class function TUtils.tf_with<TIn, TOut>(py: TIn; action: TFunc<TIn, TOut>): TOut;
var
  vVal : TValue;

begin
    var tTipo : PTypeInfo:= TypeInfo(TIn);

    if tTipo <> nil then
    begin
        vVal := TValue.From<TIn>(py) ;

        if vVal.IsType<TNameScope>  then
        begin
           var ns := vVal.AsType<TNameScope>;
           ns._Enter_;
        end;
    end;

    Result := action(py);

    if tTipo <> nil then
    begin
        vVal := TValue.From<TIn>(py) ;

        if vVal.IsType<TNameScope>  then
        begin
           var ns := vVal.AsType<TNameScope>;
           ns._Exit_;
        end;
    end;

end;

class function TUtils.flatten<T>(obj :TArray<T>): TList<T> ;
var
  list : TList<T>;
begin
    list:= TList<T>.create;

    for var i := 0 to Length(obj) -1  do
       list.Add(obj[i]);

    Result := list;
end;

class function TUtils.flatten<T>(obj: TArray<TArray<T>>): TList<T> ;
var
  list : TList<T>;
begin
    list:= TList<T>.create;

    for var i := 0 to Length(obj) -1  do
      for var j := 0 to Length(obj[i]) -1  do
       list.Add(obj[i][j]);

    Result := list;
end;

class function TUtils.flatten<T>(obj: TArray<TArray<TArray<T>>>): TList<T> ;
var
  list : TList<T>;
begin
    list:= TList<T>.create;

    for var i := 0 to Length(obj) -1  do
      for var j := 0 to Length(obj[i]) -1  do
        for var y := 0 to Length(obj[i][j]) -1  do
          list.Add(obj[i][j][y]);

    Result := list;
end;

class function TUtils.flatten<T>(obj: TArray<TArray<TArray<TArray<T>>>>): TList<T> ;
var
  list : TList<T>;
begin
    list:= TList<T>.create;

    for var i := 0 to Length(obj) -1  do
      for var j := 0 to Length(obj[i]) -1  do
        for var y := 0 to Length(obj[i][j]) -1  do
          for var z := 0 to Length(obj[i][j][y]) -1  do
             list.Add(obj[i][j][y][z]);

    Result := list;
end;

class function TUtils.ConvertToDict(dny: TArray<TParameter>): TDictionary<string,TValue> ;
var
  i : Integer;

begin
     var dictionary := TDictionary<string,TValue>.Create;

     for i := 0 to Length(dny)-1 do
     begin
         var Typ  := dny[i].vValue;
         var name : string := dny[i].sNome;


         dictionary.Add(name,Typ);
     end;


     

     Result := dictionary;

end;

class function TUtils.isinstance(v: TValue; t : PTypeInfo):Boolean;
begin
    Result := v.TypeInfo = t
end;

class function TUtils.ArrayToArrayTipo<T>(a : Tarray<T>; toTipo: PTypeInfo): TArray<Integer>;
var
  i : Integer;
  res : TArray<Integer>;
begin
    res := [];
    for i := 0 to Length(a) - 1 do
       res := res + [ ChangeType( TValue.From<T>(a[i]) , toTipo).AsInteger ];
end;

class function TUtils.ChangeType(x: TValue; new_system_dtype: PTypeInfo): TValue;
begin
    Result := x.Cast(new_system_dtype) ;

    {if new_system_dtype = TypeInfo(Boolean) then
    begin
        Result := x.Cast(new_system_dtype) ;
    end
    if new_system_dtype = TypeInfo(Char) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varByte);
    end
    else if new_system_dtype = TypeInfo(UInt8) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varByte);
    end
    else if new_system_dtype = TypeInfo(Int8) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varShortInt);
    end
    else if new_system_dtype = TypeInfo(Int16) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varSmallint);
    end
    else if new_system_dtype = TypeInfo(UInt16) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varWord);
    end
    else if new_system_dtype = TypeInfo(Int32) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varInteger);
    end
    else if new_system_dtype = TypeInfo(UInt32) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varLongWord);
    end
    else if new_system_dtype = TypeInfo(Int64) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varInt64);
    end
    else if new_system_dtype = TypeInfo(UInt64) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varInt64);
    end
    else if new_system_dtype = TypeInfo(Single) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varSingle);
    end
    else if new_system_dtype = TypeInfo(Double) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varDouble);
    end
    else if new_system_dtype = TypeInfo(TDateTime) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varDate);
    end
    else if new_system_dtype = TypeInfo(String) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varString);
    end
    else if new_system_dtype = TypeInfo(AnsiString) then
    begin
        var b := TValue.From<T>(x).AsVariant ;
        Result := VarAsType(b,varString);
    end
    else
      raise Exception.Create('type code UnknownTypeCode');  }

end;

class function TUtils.as_shape<T>(dims: TArray<T>): TTensorShapeProto;
var
  shape : TTensorShapeProto;
  i     : Integer;
begin
    shape.Init;
    var v := TValue.From< TArray<T> >(dims) ;

    for i := 0 to Length(dims) - 1 do
    begin
        var dim : TDim ;
        dim.Init;
        if TypeInfo(T) = TypeInfo(Integer) then
          dim.Size := v.AsType< TArray<Integer> >[i]
        else if TypeInfo(T) = TypeInfo(Int64) then
          dim.Size := v.AsType< TArray<Int64> >[i]
        else
          raise Exception.Create('as_shape Not Implemented');

        shape.Dims.Add(@dim);
    end;
    Result := shape;

end;

class function TUtils.as_shape_proto(tshape : TFShape): TTensorShapeProto;
var
  shape : TTensorShapeProto;
  i     : Integer;
begin
    shape.Init;

    for i := 0 to tshape.ndim - 1 do
    begin
        var dim : TDim ;
        dim.Init;
        dim.Size := tshape.dims[i];
        //dim.Name = $"dim_{i}";
        shape.Dims.Add(@dim);
    end;
    Result := shape;
end;

class function TUtils.make_tensor_proto(values: TValue; var dtype: TF_DataType; shape: PTFShape; verify_shape,
                       allow_broadcast: Boolean): TTensorProto;
begin

    if allow_broadcast and verify_shape then
       raise Exception.Create('allow_broadcast and verify_shape are not both allowed.');

    if values.IsType<TTensorProto> then  Exit( values.AsType<TTensorProto> );

    var origin_dtype := GetDataType(values);

    if dtype = TF_DataType.TF_DATATYPE_UNKNOWN then
        dtype := origin_dtype
    else if origin_dtype <> dtype then
    begin
        var new_system_dtype := Tdtypes.as_system_dtype(dtype);
        if values.IsType< TArray<Int64> > then
        begin
            if dtype = TF_DataType.TF_INT32 then
            begin
                var a := ArrayToArrayTipo<Int64>( values.AsType< TArray<Int64> >, new_system_dtype);
                values := TValue.From< TArray<Integer> >(a);
            end;
        end else
        begin
            values := ChangeType(values, new_system_dtype);
        end;

        dtype := GetDataType(values);
    end;

    var sShape : TFShape;
    if shape = nil then
    begin
        sShape := GetShape(values);
        shape :=  @sShape;
    end;

    var tensor_proto : TTensorProto;
    tensor_proto.Init;

    tensor_proto.Dtype       := Tdtypes.as_datatype_enum(dtype);
    tensor_proto.TensorShape := TUtils.as_shape_proto(shape);
    
    if values.IsType<TNDArray> then
    begin
        var nd := values.AsType<TNDArray>;

        // scalar
        if nd.shape.IsScalar then
        begin
            case nd.dtype of
                TF_DataType.TF_BOOL: tensor_proto.BoolVals.AddRange(nd.ToArray<Boolean>);
                TF_DataType.TF_UINT8:
                    begin
                       var a : TArray<Integer>;
                       var b := nd.ToArray<byte>;
                       for var i := 0 to Length(b) - 1 do
                         a := a + [ b[i] ];

                       tensor_proto.IntVals.AddRange(a);
                    end;
                TF_DataType.TF_INT32: tensor_proto.IntVals.AddRange   (nd.ToArray<Integer>);
                TF_DataType.TF_INT64: tensor_proto.Int64Vals.AddRange (nd.ToArray<Int64>);
                TF_DataType.TF_FLOAT: tensor_proto.FloatVals.AddRange (nd.ToArray<Single>);
                TF_DataType.TF_DOUBLE:tensor_proto.DoubleVals.AddRange(nd.ToArray<double>);
                else
                    raise Exception.Create('make_tensor_proto Not Implemented');
            end;
        end else
        begin
            var len := nd.dtypesize * nd.size;
            var bytes := nd.ToByteArray;
            tensor_proto.TensorContent := bytes;
        end;
    end
    else if (dtype = TF_DataType.TF_STRING) and  (not (values.IsType<TNDArray>)) then
    begin
        if (values.IsType<string>) or (values.IsType<AnsiString>) then
        begin
            var str :=  values.AsType<string> ;
            var bytes := TEncoding.UTF8.GetBytes(str);
            tensor_proto.StringVals.Add(@bytes);
        end
        else if (values.IsType<TArray<string>>) or (values.IsType<TArray<AnsiString>>) then
        begin
            var a : TArray<TBytes>;
            var b := values.AsType< TArray<string> >;
            for var i := 0 to Length(b) - 1 do
              a := a + [ TEncoding.UTF8.GetBytes( b[i] ) ];
            tensor_proto.StringVals.AddRange( a );
        end
        else if (values.IsType< TArray<Byte> >) then
        begin
            var byte_values := values.AsType< TArray<Byte> >;
            tensor_proto.TensorContent := byte_values;
        end;
    end
    else if values.IsArray then
    begin
        // array
        var len := Tdtypes.get_datatype_size(dtype) * shape.size;
        var bytes : TArray<Byte>;
        var src := values.GetReferenceToRawData;
        var dst := @bytes[0];
        SetLength(bytes,len);
        Move(src^,dst^,len);
        tensor_proto.TensorContent := bytes;
    end else
    begin
        if values.IsType<TAxis> then
        begin
            var vval := values.AsType<TAxis>;
             tensor_proto.IntVals.AddRange(vval.axis);
        end
        else if values.IsType<TFShape> then
        begin
            var vval := values.AsType<TFShape>;
            tensor_proto.Int64Vals.AddRange(vval.dims);
        end
        else if values.IsType<Boolean> then
        begin
            var vval := values.AsType<Boolean>;
            tensor_proto.BoolVals.AddRange([ vval ]);
        end
        else if values.IsType<Int8> then
        begin
            var vval := values.AsType<Int8>;
            tensor_proto.IntVals.AddRange([ vval ]);
        end
        else if values.IsType<Integer> then
        begin
            var vval := values.AsType<Integer>;
            tensor_proto.IntVals.AddRange([ vval ]);
        end
        else if values.IsType<Int64> then
        begin
            var vval := values.AsType<Int64>;
            tensor_proto.Int64Vals.AddRange([ vval ]);
        end
        else if values.IsType<Single> then
        begin
            var vval := values.AsType<Single>;
            tensor_proto.FloatVals.AddRange([ vval ]);
        end
        else if values.IsType<Double> then
        begin
            var vval := values.AsType<Double>;
            tensor_proto.DoubleVals.AddRange([ vval ]);
        end
    end;
    Result := tensor_proto;

end;

end.
