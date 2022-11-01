unit Tensorflow.Utils;
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

interface
    uses System.SysUtils,
         System.TypInfo,
         System.Variants,
         System.Rtti,

         Spring, Spring.Collections,
         Spring.Collections.Dictionaries,
         Spring.Collections.Lists,
         Spring.Collections.Enumerable,

         TensorFlow.DApi,
         TensorFlow.DApiBase,
         TF4D.Core.CApi,
         TensorFlow.Slice,
         Tensorflow.Tensor,
         NumPy.NDArray,

         ProtoGen.tensorShape,
         ProtoGen.types,
         ProtoGen.Tensor;

type

  TValueHelp = record Helper for TValue

    public
      class operator Implicit(const Value: TNDArray): TValue;
      class operator Implicit(const Value: TValue): TNDArray;
      class operator Implicit(const Value: NDArray): TValue;
      class operator Implicit(const Value: TValue): NDArray;
      //
      class operator Implicit(const Value: TTensor): TValue;
      class operator Implicit(const Value: TValue): TTensor;
      class operator Implicit(const Value: TFTensor): TValue;
      class operator Implicit(const Value: TValue): TFTensor;
      //
      class operator Implicit(const Value: TArray<TFTensor>): TValue;
      class operator Implicit(const Value: TArray<Integer>): TValue;
      class operator Implicit(const Value: TArray<Single>): TValue;
      class operator Implicit(const Value: TF_DataType): TValue;
      class operator Implicit(const Value: TArray< TArray<Integer> >): TValue;

  end;

 Tdtypes = record
  private

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

       class function as_numpy_name(value: TF_DataType): string; static;
       class function as_base_dtype(value: TF_DataType): TF_DataType; overload; static;
       class function as_base_dtype(value: TDataType): TDataType;  overload; static;
       class function as_ref(value: TF_DataType): TF_DataType; static;
       class function as_tf_dtype(value: TValue): TF_DataType; overload; static;
       class function as_tf_dtype(value: PTypeInfo): TF_DataType; overload;static;
       class function as_tf_dtype(value: TDataType): TF_DataType; overload;static;
       class function get_datatype_size(tipo: TF_DataType): Integer; static;
       class function as_datatype_enum(value: TF_DataType): TDataType; static;
       class function ToIntArray(value: TArray<TF_DataType>): TArray<Integer>; static;
       class function is_integer(tipo: TF_DataType ): Boolean; static;
       class function is_floating(tipo: TF_DataType ): Boolean;static;
       class function is_complex(tipo: TF_DataType): Boolean; static;
       class function real_dtype(tipo: TF_DataType): TF_DataType; static;
       class function is_ref_dtype(tipo: TF_DataType ): Boolean; static;
       class function min(tipo: TF_DataType ): Int64; static;
       class function max(tipo: TF_DataType ): Int64; static;

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
    class function _ConstantValue(tensor: TFTensor; partial: Boolean): TNDArray;

   public
      class function MakeNdarray(tensor: TTensorProto): TNDArray; static;

      class function SequenceEqual<T>(const v1,v2: TArray<T>): boolean;
      class function IsInstance(v: TValue; t : PTypeInfo):Boolean; overload;
      class function IsInstance<T>(tipo1 : T; Tipo2: PTypeInfo): boolean; overload;
      class function IsInstance<T,T1,T2>(tipo1 : T; Tipo2: Tuple<T1,T2>): boolean;  overload;
      class function IsInstance<T,T1,T2,T3>(tipo1 : T; Tipo2: Tuple<T1,T2,T3>): boolean; overload;

      class function flatten<T>(obj : TArray<T>                        ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<T>>                ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<TArray<T>>>        ): TList<T> ;  overload;
      class function flatten<T>(obj : TArray<TArray<TArray<TArray<T>>>>): TList<T> ;  overload;

      class procedure tf_with<T>(py: T; action: TProc<T>); overload;
      class function tf_with<TIn, TOut>(py: TIn; action: TFunc<TIn, TOut>): TOut;overload;
      class function GetDataType(value: TValue): TF_DataType;
      class function GetShape(value: TValue): TFShape;overload;
      class function GetShape<T>(Tval: TArray<TArray<TArray<TArray<T>>>>): TFShape;  overload;
      class function ConvertToDict(dny: TArray<TParameter>): TDictionary<string,TValue> ;

      class function as_shape_proto(tshape: TFShape): TTensorShapeProto; static;
      class function as_shape<T>(dims: TArray<T>): TTensorShapeProto;
      class function shape_tensor(shape: TArray<Integer>): TFTensor; static;
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
      /// <summary>
      /// Returns the constant value of the given tensor, if efficiently calculable.
      /// </summary>
      /// <param name="tensor"></param>
      /// <param name="partial"></param>
      /// <returns></returns>
      class function constant_value(tensor: TFTensor; partial: Boolean = false): TNDArray;
      class function ParseSlices(slices: TArray<Slice>): ParsedSliceArgs;
      class function zip<T1, T2>(e1 : Enumerable<T1>; e2 : IEnumerable<T2>): Enumerable<Tuple<T1,T2>> ;
      class function range(start, _end: Integer): Enumerable<integer>;  overload; static;
      class function range(_end: Integer): Enumerable<integer> ;  overload; static;
 end;

 function GetArg(sNome: string; vVal : TValue):  TParameter;

implementation
        uses system.Generics.Defaults,
             Winapi.Windows,
             Tensorflow,
             TensorFlow.EagerTensor,
             Tensorflow.NameScope,
             TensorFlow.Ops,
             Numpy,
             Numpy.Axis,Complex,
             TensorFlow.Variable;


function GetArg(sNome: string; vVal : TValue):  TParameter;
begin
     Result.sNome := sNome;
     Result.vValue:= vVal;
end;

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
        TF_DataType.TF_INT16    : Result :='int16';
        TF_DataType.TF_UINT16   : Result :='uint16';
        TF_DataType.TF_UINT64   : Result :='uint64';
        TF_DataType.TF_INT64    : Result :='int64';
        TF_DataType.TF_FLOAT    : Result :='float32';
        TF_DataType.TF_DOUBLE   : Result :='float64';
        TF_DataType.TF_BOOL     : Result :='bool';
        TF_DataType.TF_RESOURCE : Result :='resource';
        TF_DataType.TF_VARIANT  : Result :='variant';
    else
        Result := TEnum.GetName<TF_DataType>( value);
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

     while value.IsArray do
       value := value.GetArrayElement(0);

     tTipo:= value.TypeInfo;

     if      tTipo = TypeInfo(integer)   then dType := TF_DataType.TF_INT32
     else if tTipo = TypeInfo(cardinal)  then dType := TF_DataType.TF_UINT32
     else if tTipo = TypeInfo(int64)     then dType := TF_DataType.TF_INT64
     else if tTipo = TypeInfo(uint64)    then dType := TF_DataType.TF_UINT64
     else if tTipo = TypeInfo(word)      then dType := TF_DataType.TF_UINT16
     else if tTipo = TypeInfo(smallint)  then dType := TF_DataType.TF_INT16
     else if tTipo = TypeInfo(byte)      then dType := TF_DataType.TF_UINT8
     else if tTipo = TypeInfo(char)      then dType := TF_DataType.TF_UINT8
     else if tTipo = TypeInfo(shortint)  then dType := TF_DataType.TF_INT8
     else if tTipo = TypeInfo(boolean)   then dType := TF_DataType.TF_BOOL
     else if tTipo = TypeInfo(single)    then dType := TF_DataType.TF_FLOAT
     else if tTipo = TypeInfo(double)    then dType := TF_DataType.TF_DOUBLE
     else if tTipo = TypeInfo(Extended)  then dType := TF_DataType.TF_DOUBLE
     else if tTipo = TypeInfo(string)    then dType := TF_DataType.TF_STRING
     else if tTipo = TypeInfo(ansistring)then dType := TF_DataType.TF_STRING

     else if tTipo.Kind = tkInteger      then dType := TF_DataType.TF_INT32
     else if tTipo.Kind = tkInt64        then dType := TF_DataType.TF_INT64
     else if tTipo.Kind = tkfloat        then dType := TF_DataType.TF_FLOAT
     else
        raise TFException.Create('Type not found');

     Result := dType;

end;

class function Tdtypes.as_tf_dtype(value: PTypeInfo): TF_DataType;
var
  tTipo : PTypeInfo;
  dType : TF_DataType;
begin
     dType := TF_DataType.DtInvalid;


     while (value.Kind = tkDynArray) or (value.Kind = tkArray) do
       value := value^.TypeData^.DynArrElType^;

     tTipo:= value;

     if      tTipo = TypeInfo(integer)   then dType := TF_DataType.TF_INT32
     else if tTipo = TypeInfo(cardinal)  then dType := TF_DataType.TF_UINT32
     else if tTipo = TypeInfo(int64)     then dType := TF_DataType.TF_INT64
     else if tTipo = TypeInfo(uint64)    then dType := TF_DataType.TF_UINT64
     else if tTipo = TypeInfo(word)      then dType := TF_DataType.TF_UINT16
     else if tTipo = TypeInfo(smallint)  then dType := TF_DataType.TF_INT16
     else if tTipo = TypeInfo(byte)      then dType := TF_DataType.TF_UINT8
     else if tTipo = TypeInfo(char)      then dType := TF_DataType.TF_UINT8
     else if tTipo = TypeInfo(shortint)  then dType := TF_DataType.TF_INT8
     else if tTipo = TypeInfo(boolean)   then dType := TF_DataType.TF_BOOL
     else if tTipo = TypeInfo(single)    then dType := TF_DataType.TF_FLOAT
     else if tTipo = TypeInfo(double)    then dType := TF_DataType.TF_DOUBLE
     else if tTipo = TypeInfo(Extended)  then dType := TF_DataType.TF_DOUBLE
     else if tTipo = TypeInfo(string)    then dType := TF_DataType.TF_STRING
     else if tTipo = TypeInfo(ansistring)then dType := TF_DataType.TF_STRING;

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
            raise TFException.Create('Unable to convert {type} to a system data type.');
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
        raise TFException.Create('TUtils.get_datatype_size - NotImplemented');
    end;
end;

class function Tdtypes.is_complex(tipo: TF_DataType): Boolean;
begin
     Result := (tipo = TF_DataType.TF_COMPLEX) or (tipo = TF_DataType.TF_COMPLEX64) or  (tipo = TF_DataType.TF_COMPLEX128);
end;

class function Tdtypes.is_floating(tipo: TF_DataType): Boolean;
begin
     Result := (tipo = TF_DataType.TF_HALF) or (tipo = TF_DataType.TF_FLOAT) or  (tipo = TF_DataType.TF_DOUBLE);
end;

class function Tdtypes.is_integer(tipo: TF_DataType): Boolean;
begin
    Result := (tipo = TF_DataType.TF_INT8) or (tipo = TF_DataType.TF_INT16) or (tipo = TF_DataType.TF_INT32) or (tipo = TF_DataType.TF_INT64) or
              (tipo = TF_DataType.TF_UINT8) or (tipo = TF_DataType.TF_UINT16) or (tipo = TF_DataType.TF_UINT32) or (tipo = TF_DataType.TF_UINT64)
end;

class function Tdtypes.is_ref_dtype(tipo: TF_DataType): Boolean;
begin
     Result := Ord(tipo) > 100;
end;

class function Tdtypes.max(tipo: TF_DataType): Int64;
begin
    case tipo of
      TF_INT8:   Result := Int8.MaxValue;
      TF_INT16:  Result := Int16.MaxValue;
      TF_INT32:  Result := Int32.MaxValue;
      TF_INT64:  Result := Int64.MaxValue;
      TF_UINT8:  Result := UInt8.MaxValue;
      TF_UINT16: Result := UInt16.MaxValue;
      TF_UINT32: Result := UInt32.MaxValue;
      TF_UINT64: Result := UInt64.MaxValue;
    else
      raise Exception.Create(' Not Implemented - Tdtypes.max');
    end;
end;

class function Tdtypes.min(tipo: TF_DataType): Int64;
begin
    case tipo of
      TF_INT8:   Result := Int8.MinValue;
      TF_INT16:  Result := Int16.MinValue;
      TF_INT32:  Result := Int32.MinValue;
      TF_INT64:  Result := Int64.MinValue;
      TF_UINT8:  Result := UInt8.MinValue;
      TF_UINT16: Result := UInt16.MinValue;
      TF_UINT32: Result := UInt32.MinValue;
      TF_UINT64: Result := UInt64.MinValue;
    else
      raise Exception.Create(' Not Implemented - Tdtypes.min');
    end;
end;

class function Tdtypes.real_dtype(tipo: TF_DataType): TF_DataType;
begin
    var base_ : TF_DataType := as_base_dtype(tipo);
    if base_ = ccomplex64 then
        Exit( cfloat32)
    else if base_ = ccomplex128 then
        Exit(cfloat64)
    else
        Result := tipo;
end;

{ TUtils }

class function TUtils.GetDataType(value: TValue): TF_DataType;
var
  tTipo : PTypeInfo;
begin
   tTipo:= value.TypeInfo;
   Result := DtInvalid;

   case ttipo.Kind of
     tkClass,tkRecord,tkMRecord : begin
          if      string.LowerCase(string(tTipo.Name)) = 'tfshape'  then   Exit(TF_DataType.TF_INT64)
          else if string.LowerCase(string(tTipo.Name)) = 'taxis'    then   Exit(TF_DataType.TF_INT32)
          else if string.LowerCase(string(tTipo.Name)) = 'tndarray' then
          begin
             var v : TNDArray := value.AsType<TNDArray>;
             Exit(v.Dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'ndarray' then
          begin
             var v : NDArray := value.AsType<NDArray>;
             Exit(TNDArray(v).Dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tftensor' then
          begin
             var v : TFTensor := value.AsType<TFTensor>;
             Exit(v.Dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'ttensor' then
          begin
             var v : TTensor := value.AsType<TTensor>;
             Exit(TFTensor(v).Dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'teagertensor' then
          begin
             var v : TEagerTensor := value.AsType<TEagerTensor>;
             Exit(v.Dtype);
          end
          else if string.LowerCase(string(tTipo.Name)) = 'tftensors' then
          begin
             var v : TFTensors := value.AsType<TFTensors>;
             Exit(v.First.Dtype );
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
          var cnt := value.GetArrayLength;
          if cnt < 1 then
             raise TFException.Create(' Array Length Error');
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
          if value.IsType<TAxis>  then
          begin
              var v : TAxis:= value.AsType<TAxis>;

              if v.isScalar then Exit( TFShape.scalar );

              var vAx : TArray<Int64>; SetLength(vAx,Length(v.axis.Value));
              Result := TFShape.Create(vAx);
              Exit;
          end
          else if value.IsType<TNDArray> then
          begin
             var v : TNDArray := value.AsType<TNDArray>;
             Result := v.Shape;
             Exit;
          end
          else if value.IsType<TFTensor> then
          begin
             var v : TFTensor := value.AsType<TFTensor>;
             Result := v.Shape;
             Exit;
          end
          else if value.IsType<TFShape> then
          begin
             var v : TFShape := value.AsType<TFShape>;
             Result := TFShape.Create([v.rank]);
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
       raise TFException.Create('NotImplementedException');
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

class procedure TUtils.tf_with<T>(py: T; action: TProc<T>);
var
  vVal : TValue;

begin
    var tTipo : PTypeInfo:= TypeInfo(T);

    if tTipo <> nil then
    begin
        vVal := TValue.From<T>(py) ;

        if vVal.IsType<TNameScope>  then
        begin
            var ns := vVal.AsType<TNameScope>;
            ns._Enter_;
        end
        else if vVal.IsType<TControlDependenciesController>  then
        begin
            var ns := vVal.AsType<TControlDependenciesController>;
            ns._Enter_;
        end
        else if vVal.IsType<TControlFlowContext>  then
        begin
            var ns := vVal.AsType<TControlFlowContext>;
            ns._Enter_;
        end;
    end;

    action(py);

    if tTipo <> nil then
    begin
        vVal := TValue.From<T>(py) ;

        if vVal.IsType<TNameScope>  then
        begin
            var ns := vVal.AsType<TNameScope>;
            ns._Exit_;
        end
        else if vVal.IsType<TControlDependenciesController>  then
        begin
            var ns := vVal.AsType<TControlDependenciesController>;
            ns._Exit_;
        end
        else if vVal.IsType<TControlFlowContext>  then
        begin
            var ns := vVal.AsType<TControlFlowContext>;
            ns._Exit_;
        end;

    end;

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
        end
        else if vVal.IsType<TControlDependenciesController>  then
        begin
            var ns := vVal.AsType<TControlDependenciesController>;
            ns._Enter_;
        end
        else if vVal.IsType<TControlFlowContext>  then
        begin
            var ns := vVal.AsType<TControlFlowContext>;
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
        end
        else if vVal.IsType<TControlDependenciesController>  then
        begin
            var ns := vVal.AsType<TControlDependenciesController>;
            ns._Exit_;
        end
        else if vVal.IsType<TControlFlowContext>  then
        begin
            var ns := vVal.AsType<TControlFlowContext>;
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

class function TUtils.ArrayToArrayTipo<T>(a : Tarray<T>; toTipo: PTypeInfo): TArray<Integer>;
var
  i : Integer;
  res : TArray<Integer>;
begin
    res := [];
    for i := 0 to Length(a) - 1 do
       res := res + [ ChangeType( TValue.From<T>(a[i]) , toTipo).AsInteger ];

    Result := res;
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

class function TUtils.range(_end: Integer): Enumerable<integer> ;
begin
     Result := TEnumerable.range(0, _end);
end;

class function TUtils.range(start: Integer; _end: Integer): Enumerable<integer> ;
begin
    Result := TEnumerable.range(start, _end - start);
end;

class function TUtils.zip<T1, T2>(e1 : Enumerable<T1>; e2 : IEnumerable<T2>): Enumerable<Tuple<T1,T2>> ;
begin
    var r := e1.Zip<T2, Tuple<T1,T2> >( e2,function(first:  T1; second : T2 ): Tuple<T1,T2>
                                                begin
                                                    Result := Tuple<T1,T2>.Create(first,second)
                                                end );
    Result := r;
end;

class function TUtils.constant_value(tensor: TFTensor; partial: Boolean): TNDArray;
begin
    if tensor is TNDArray then Exit(TNDArray(tensor))
    else if tensor is TEagerTensor then Exit( tensor.numpy) ;
    var ret: TNDArray := _ConstantValue(tensor, partial);
    if not (ret = nil) then
        tensor.graph.prevent_feeding(tensor);
    Result := ret;
end;

class function TUtils._ConstantValue(tensor: TFTensor; partial: Boolean): TNDArray;
begin
    if tensor.op.tipo = 'Const' then
    begin
        var v  := tensor.op.get_attr('value');
        Result := MakeNdarray( v.Astype<TTensorProto> );
    end else
    begin
       Result := nil;
    end;
end;

class function TUtils.MakeNdarray(tensor: TTensorProto) : TNDArray;
begin

    var aSize : TArray<Int64> := [];
    for var i := 0 to tensor.TensorShape.Dims.Count - 1 do
     aSize := aSize + [ tensor.TensorShape.Dims[i].Size  ] ;
    var shape        := TFShape.Create(aSize);
    {$HINTS OFF}
    var num_elements := shape.size;
    {$HINTS ON}
    var tensor_dtype := TDTypes.as_tf_dtype(tensor.Dtype);
    if (shape.ndim > 0) and (Length(tensor.TensorContent) > 0) then
    begin
        Result := np.frombuffer(tensor.TensorContent, shape, tensor_dtype);
    end
    else if (tensor.Dtype = TDataType.DT_HALF) or (tensor.Dtype = TDataType.DT_BFLOAT16) then
    begin
        Result := np.np_array(tensor.HalfVals.ToArray).reshape(shape);
    end
    else if tensor.Dtype = TDataType.DT_FLOAT then
    begin
        Result := np.np_array(tensor.FloatVals.ToArray).reshape(shape);
    end
    else if (tensor.Dtype = TDataType.DT_INT32) or (tensor.Dtype = TDataType.DT_UINT8) then
    begin
        Result := np.np_array(tensor.IntVals.ToArray).reshape(shape);
    end
    else if tensor.Dtype = TDataType.DT_BOOL then
    begin
        Result := np.np_array(tensor.BoolVals.ToArray).reshape(shape);
    end else
        raise TFException.Create('Not Implemented ("MakeNdarray")');
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
          raise TFException.Create('as_shape Not Implemented');

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

class function TUtils.shape_tensor(shape : TArray<Integer>): TFTensor;
begin
    Result := Tops.convert_to_tensor( TValue.From< TArray<Integer> >(shape), TF_DataType.TF_INT32, 'shape');
end;

class function TUtils.make_tensor_proto(values: TValue; var dtype: TF_DataType; shape: PTFShape; verify_shape,
                       allow_broadcast: Boolean): TTensorProto;
 var
   bytes  : TArray<Byte>;
begin

    if allow_broadcast and verify_shape then
       raise TFException.Create('allow_broadcast and verify_shape are not both allowed.');

    if values.IsType<TTensorProto> then  Exit( values.AsType<TTensorProto> );

    var origin_dtype := GetDataType(values);

    if dtype = TF_DataType.DtInvalid then
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
    if (shape = nil) or (shape.IsNil) then
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
                    raise TFException.Create('make_tensor_proto Not Implemented');
            end;
        end else
        begin
            bytes := nd.ToByteArray;
            tensor_proto.TensorContent := bytes;
        end;
    end
    else if (dtype = TF_DataType.TF_STRING) and  (not (values.IsType<TNDArray>)) then
    begin
        if (values.IsType<string>) or (values.IsType<AnsiString>) then
        begin
            var str :=  values.AsType<AnsiString> ;
            bytes := TEncoding.UTF8.GetBytes(string(str));
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
        if shape.ndim = 2 then
        begin
            var lenBytes := Tdtypes.get_datatype_size(dtype) * shape.size;
            SetLength(bytes,lenBytes);

            var len0 := values.GetArrayLength;
            var BytesIdx: Integer := 0;
            for var i := 0 to len0-1 do
            begin
                var v1  := values.GetArrayElement(i);
                var len := v1.GetArrayLength;

                var src := v1.GetReferenceToRawArrayElement(0);
                var dst := @bytes[BytesIdx];
                CopyMemory(dst,src, len * Tdtypes.get_datatype_size(dtype) );
                Inc(BytesIdx,len * Tdtypes.get_datatype_size(dtype));
            end;
            tensor_proto.TensorContent := bytes;
        end
        else if shape.ndim = 3 then
        begin
            var lenBytes := Tdtypes.get_datatype_size(dtype) * shape.size;
            SetLength(bytes,lenBytes);

            var len0 := values.GetArrayLength;
            var BytesIdx: Integer := 0;
            for var i := 0 to len0-1 do
            begin
                var v1  := values.GetArrayElement(i);
                var len := v1.GetArrayLength;

                var src := v1.GetReferenceToRawArrayElement(0);
                var dst := @bytes[BytesIdx];
                CopyMemory(dst,src, len * Tdtypes.get_datatype_size(dtype) );
                Inc(BytesIdx,len * Tdtypes.get_datatype_size(dtype));
            end;
            tensor_proto.TensorContent := bytes;
        end else
        begin
            // array
            var len := Tdtypes.get_datatype_size(dtype) * shape.size;
            var src := values.GetReferenceToRawArrayElement(0);
            SetLength(bytes,len);
            var dst := @bytes[0];

            CopyMemory(dst,src,len);
            tensor_proto.TensorContent := bytes;
        end;
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
        else if (values.IsType<Single>) and (Values.TypeInfo.Name ='Single') then
        begin
            var vval := values.AsType<Single>;
            tensor_proto.FloatVals.AddRange([ vval ]);
        end
        else if (values.IsType<Single>) and (Values.TypeInfo.Name ='Double') then
        begin
            var vval := values.AsType<Double>;
            tensor_proto.DoubleVals.AddRange([ vval ]);
        end
    end;
    Result := tensor_proto;

end;

class function TUtils.ParseSlices(slices: TArray<Slice>): ParsedSliceArgs;
begin
    var abegin := TList<Integer>.Create;
    var aend   := TList<Integer>.Create;
    var strides:= TList<Integer>.Create;
    try
      var index            : Integer := 0;
      var new_axis_mask    : Integer := 0;
      var shrink_axis_mask : Integer := 0;
      var begin_mask       : Integer := 0;
      var end_mask         : Integer := 0;
      var ellipsis_mask    : Integer := 0;
      for var s in slices do
      begin
          if s.IsNewAxis then
          begin
              abegin.Add(0);
              aend.Add(0);
              strides.Add(1);
              new_axis_mask := new_axis_mask or (1 shl index);
          end
          else if s.IsEllipsis then
          begin
              abegin.Add(0);
              aend.Add(0);
              strides.Add(1);
              ellipsis_mask := ellipsis_mask or (1 shl index);
          end else
          begin
              if s.Start.HasValue then
              begin
                  abegin.Add(s.Start.Value);
              end else
              begin
                  abegin.Add(0);
                  begin_mask := begin_mask or (1 shl index);
              end;
              if s.Stop.HasValue then
              begin
                  aend.Add(s.Stop.Value);
              end else
              begin
                  aend.Add(0);
                  end_mask := end_mask or (1 shl index);
              end;
              strides.Add(s.Step);
              if s.IsIndex then
                  shrink_axis_mask := shrink_axis_mask or (1 shl index);
          end;
          Inc(index);
      end;
      Result := default(ParsedSliceArgs);
      Result.aBegin         := abegin.ToArray;
      Result.aEnd           := aend.ToArray;
      Result.aStrides       := strides.ToArray;
      Result.iBeginMask     := begin_mask;
      Result.iEndMask       := end_mask;
      Result.iEllipsisMask  := ellipsis_mask;
      Result.iShrinkAxisMask:= shrink_axis_mask;
      Result.iNewAxisMask   := new_axis_mask ;
    finally
      abegin.free;
      aend.free;
      strides.free;
    end;

end;

class function TUtils.SequenceEqual<T>(const v1, v2: TArray<T>): boolean;
var
  comparer: IEqualityComparer<T>;
  i: Integer;
begin
  comparer := TEqualityComparer<T>.Default;
  for i := Low(v1) to High(v1) do
    if not comparer.Equals(v1[i], v2[i]) then
      Exit(false);
  Result := true;

end;

class function TUtils.isinstance(v: TValue; t : PTypeInfo):Boolean;
begin
    Result := v.TypeInfo = t
end;

class function TUtils.IsInstance<T>(tipo1 : T; Tipo2: PTypeInfo): boolean;
begin
    Result := PTypeInfo(TypeInfo(T)) = Tipo2;
end;

class function TUtils.IsInstance<T,T1,T2>(tipo1 : T; Tipo2: Tuple<T1,T2>): boolean;
begin
    Result := False;
    if PTypeInfo(TypeInfo(T)) = PTypeInfo(TypeInfo(T1)) then
      Exit(True);
   if PTypeInfo(TypeInfo(T)) = PTypeInfo(TypeInfo(T2)) then
      Exit(True);
end;

class function TUtils.IsInstance<T,T1,T2,T3>(tipo1 : T; Tipo2: Tuple<T1,T2,T3>): boolean;
begin
    Result := False;
    if PTypeInfo(TypeInfo(T)) = PTypeInfo(TypeInfo(T1)) then
      Exit(True);
   if PTypeInfo(TypeInfo(T)) = PTypeInfo(TypeInfo(T2)) then
      Exit(True);
   if PTypeInfo(TypeInfo(T)) = PTypeInfo(TypeInfo(T3)) then
      Exit(True);
end;

{ TValueHelper }

class operator TValueHelp.Implicit(const Value: TValue): TFTensor;
begin
    Result := nil;

    if Value.IsType<TFTensor> then
      Result := Value.AsType<TFTensor>
end;

class operator TValueHelp.Implicit(const Value: TArray<TFTensor>): TValue;
begin
    Result := TValue.From< TArray<TFTensor> >(Value);
end;

class operator TValueHelp.Implicit(const Value: TFTensor): TValue;
begin
    Result := TValue.From<TFTensor>(Value);
end;

class operator TValueHelp.Implicit(const Value: TF_DataType): TValue;
begin
    Result := TValue.From<Integer>(Ord(Value));
end;

class operator TValueHelp.Implicit(const Value: TValue): TNDArray;
begin
    Result := nil;

    if Value.IsType<TNDArray> then
      Result := Value.AsType<TNDArray>
end;

class operator TValueHelp.Implicit(const Value: TNDArray): TValue;
begin
   Result := TValue.From<TNDArray>(Value);
end;

class operator TValueHelp.Implicit(const Value: TValue): NDArray;
begin
    Result := Value.AsType<NDArray>
end;

class operator TValueHelp.Implicit(const Value: NDArray): TValue;
begin
   Result := TValue.From<TNDArray>(Value.HandleNDArray);
end;

class operator TValueHelp.Implicit(const Value: TValue): TTensor;
begin
    Result := Value.AsType<TTensor>
end;

class operator TValueHelp.Implicit(const Value: TTensor): TValue;
begin
     Result := TValue.From<TFTensor>(Value.HTensor);
end;

class operator TValueHelp.Implicit(const Value: TArray<TArray<Integer>>): TValue;
begin
   Result := TValue.From< TArray<TArray<Integer>> >(Value);
end;

class operator TValueHelp.Implicit(const Value: TArray<Integer>): TValue;
begin
    Result := TValue.From< TArray<Integer> >(Value);
end;

class operator TValueHelp.Implicit(const Value: TArray<Single>): TValue;
begin
    Result := TValue.From< TArray<Single> >(Value);
end;

end.
