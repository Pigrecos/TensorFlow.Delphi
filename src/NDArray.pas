unit NDArray;

interface
   uses System.SysUtils,
        Spring,
        Spring.Collections,
        rtti,
        TensorFlow.LowLevelAPI,
        TensorFlow.DApi;

type


  TNDArray = class(TFTensor)
   private
     procedure NewEagerTensorHandle;
   public
     constructor Create(value: Boolean); overload; override;
     constructor Create(value: Byte);    overload; override;
     constructor Create(value: Word);    overload;
     constructor Create(value: Integer); overload; override;
     constructor Create(value: Int64);   overload; override;
     constructor Create(value: Single);  overload; override;
     constructor Create(value: Double);  overload; override;
     //
     constructor Create(bytes: TArray<TFString>;shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Boolean>;                        shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Boolean>>;                shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Boolean>>>;        shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>;shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Byte>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Byte>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Byte>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int16>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int16>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int16>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int32>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int32>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int32>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int64>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int64>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int64>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>;  shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Single>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Single>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Single>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Double>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Double>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Double>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(value: TValue;shape: PTFShape= nil); overload;
     //
     constructor Create(shape: TFShape;                      dtype: TF_DataType); overload;
     constructor Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType); overload;
     constructor Create(value: TArray<Int64>;                dtype: TF_DataType); overload;
     constructor Create(address: Pointer;                    dtype: TF_DataType); overload;

     class function Scalar<T>(value: T):TNDArray;

     function ToByteArray: TArray<Byte>;
  end;

implementation
        uses Tensorflow, Tensorflow.Utils;

{ TNDArray }

constructor TNDArray.Create(value: Integer);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Int64);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Single);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Boolean);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Byte);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Word);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: Double);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Boolean>; shape: PTFShape);
begin
     Create(TValue.From(Bytes),shape );
     NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Boolean>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Boolean>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_BOOL;

    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Byte>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Byte>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Byte>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_UINT8;

    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int16>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int16>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int16>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT16;

    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int32>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int32>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int32>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_INT32;

    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int64>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int64>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int64>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT64;

    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Single>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Single>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Single>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Single>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_FLOAT;

    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Double>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Double>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Double>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Double>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_DOUBLE;

    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: TArray<Int64>; dtype: TF_DataType);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(address: Pointer; dtype: TF_DataType);
begin
    inherited Create(address,dtype,address);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<TFString>; shape: PTFShape);
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TFShape.Create([Length(bytes)]);
        shape := @v ;
    end;

    inherited Create( StringTensor(bytes,shape).Handle );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: TValue; shape: PTFShape);
var
  aDim  : Integer;
  dtype : TF_DataType;
  vValue: TValue;
begin
    aDim := 0;
    vValue := value;
    while vValue.IsArray do
    begin
        vValue := vValue.GetArrayElement(0);
        inc(aDim)
    end;

    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape(value);
        shape := @v;
    end;

    dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
        Exit;
    end;

    case aDim of
       0 : begin
         case dtype of
           TF_FLOAT:  inherited Create( Single(value.AsExtended) );
           TF_DOUBLE: inherited Create( double(value.AsExtended) );
           TF_INT32:  inherited Create( value.AsInteger );
           TF_UINT8:  inherited Create( Byte(value.AsOrdinal) );
           TF_INT16:  inherited Create( int16(value.AsOrdinal) );
           TF_INT8:   inherited Create( int8(value.AsOrdinal) ) ;
           TF_STRING: inherited Create( value.AsString );
           TF_INT64:  inherited Create( value.AsInt64 );
           TF_BOOL:   inherited Create( value.AsBoolean );
           TF_UINT16: inherited Create( word(value.AsOrdinal) );
           TF_UINT32: inherited Create( Cardinal(value.AsOrdinal) );
           TF_UINT64: inherited Create( value.AsUInt64 );
         end;
       end;
       1 : begin
         case dtype of
           TF_FLOAT:  inherited Create( value.AsType< TArray<Single> >);
           TF_DOUBLE: inherited Create( value.AsType< TArray<Double> >);
           TF_INT32:  inherited Create( value.AsType< TArray<Int32> >);
           TF_UINT8:  inherited Create( value.AsType< TArray<UInt8> >);
           TF_INT16:  inherited Create( value.AsType< TArray<Int16> >);
           TF_INT8:   inherited Create( value.AsType< TArray<Int8> >);
           TF_STRING: inherited Create( value.AsType< TArray<string> >);
           TF_INT64:  inherited Create( value.AsType< TArray<Int64> >);
           TF_BOOL:   inherited Create( value.AsType< TArray<Boolean> >);
           TF_UINT16: inherited Create( value.AsType< TArray<UInt16> >);
           TF_UINT32: inherited Create( value.AsType< TArray<UInt32> >);
           TF_UINT64: inherited Create( value.AsType< TArray<UInt64> >);
         end;
       end;
       2 : begin
         case dtype of
           TF_FLOAT:  inherited Create( TFTensor.InitTensor<Single>(value.AsType< TArray<TArray<Single>> >,  shape,dtype).Handle );
           TF_DOUBLE: inherited Create( TFTensor.InitTensor<Double>(value.AsType< TArray<TArray<Double>> >,  shape,dtype).Handle );
           TF_INT32:  inherited Create( TFTensor.InitTensor<Int32>(value.AsType< TArray<TArray<Int32>> >,    shape,dtype).Handle );
           TF_UINT8:  inherited Create( TFTensor.InitTensor<UInt8>(value.AsType< TArray<TArray<UInt8>> >,    shape,dtype).Handle );
           TF_INT16:  inherited Create( TFTensor.InitTensor<Int16>(value.AsType< TArray<TArray<Int16>> >,    shape,dtype).Handle );
           TF_INT8:   inherited Create( TFTensor.InitTensor<Int8>(value.AsType< TArray<TArray<Int8>> >,      shape,dtype).Handle );
           TF_STRING: inherited Create( TFTensor.InitTensor<string>(value.AsType< TArray<TArray<string>> >,  shape,dtype).Handle );
           TF_INT64:  inherited Create( TFTensor.InitTensor<Int64>(value.AsType< TArray<TArray<Int64>> >,    shape,dtype).Handle );
           TF_BOOL:   inherited Create( TFTensor.InitTensor<Boolean>(value.AsType< TArray<TArray<Boolean>> >,shape,dtype).Handle );
           TF_UINT16: inherited Create( TFTensor.InitTensor<UInt16>(value.AsType< TArray<TArray<UInt16>> >,  shape,dtype).Handle );
           TF_UINT32: inherited Create( TFTensor.InitTensor<UInt32>(value.AsType< TArray<TArray<UInt32>> >,  shape,dtype).Handle );
           TF_UINT64: inherited Create( TFTensor.InitTensor<UInt64>(value.AsType< TArray<TArray<UInt64>> >,  shape,dtype).Handle );
         end;
       end;
       3 : begin
         case dtype of
           TF_FLOAT:  inherited Create( TFTensor.InitTensor<Single>(value.AsType< TArray<TArray<TArray<Single>>> >,  shape,dtype).Handle );
           TF_DOUBLE: inherited Create( TFTensor.InitTensor<Double>(value.AsType< TArray<TArray<TArray<Double>>> >,  shape,dtype).Handle );
           TF_INT32:  inherited Create( TFTensor.InitTensor<Int32>(value.AsType< TArray<TArray<TArray<Int32>>> >,    shape,dtype).Handle );
           TF_UINT8:  inherited Create( TFTensor.InitTensor<UInt8>(value.AsType< TArray<TArray<TArray<UInt8>>> >,    shape,dtype).Handle );
           TF_INT16:  inherited Create( TFTensor.InitTensor<Int16>(value.AsType< TArray<TArray<TArray<Int16>>> >,    shape,dtype).Handle );
           TF_INT8:   inherited Create( TFTensor.InitTensor<Int8>(value.AsType< TArray<TArray<TArray<Int8>>> >,      shape,dtype).Handle );
           TF_STRING: inherited Create( TFTensor.InitTensor<string>(value.AsType< TArray<TArray<TArray<string>>> >,  shape,dtype).Handle );
           TF_INT64:  inherited Create( TFTensor.InitTensor<Int64>(value.AsType< TArray<TArray<TArray<Int64>>> >,    shape,dtype).Handle );
           TF_BOOL:   inherited Create( TFTensor.InitTensor<Boolean>(value.AsType< TArray<TArray<TArray<Boolean>>> >,shape,dtype).Handle );
           TF_UINT16: inherited Create( TFTensor.InitTensor<UInt16>(value.AsType< TArray<TArray<TArray<UInt16>>> >,  shape,dtype).Handle );
           TF_UINT32: inherited Create( TFTensor.InitTensor<UInt32>(value.AsType< TArray<TArray<TArray<UInt32>>> >,  shape,dtype).Handle );
           TF_UINT64: inherited Create( TFTensor.InitTensor<UInt64>(value.AsType< TArray<TArray<TArray<UInt64>>> >,  shape,dtype).Handle );
         end;

       end;

    end;

end;

constructor TNDArray.Create(shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(shape,dtype);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TNDArray.InitTensor(bytes,shape ,dtype) );
    NewEagerTensorHandle ;
end;

procedure TNDArray.NewEagerTensorHandle;
begin
    if Assigned(Handle) then
      eagerTensorHandle := TEagerTensor.Create(Handle,true).EagerTensorHandle;
end;


class function TNDArray.Scalar<T>(value: T): TNDArray;
begin
   var ttipo: pTypeInfo := TypeInfo(T);
   var x := TValue.From<T>(value);

   if      ttipo = TypeInfo(Boolean) then  Result := TNDArray.Create(x.AsBoolean)
   else if ttipo = TypeInfo(byte)    then  Result := TNDArray.Create(Byte(x.AsInteger))
   else if ttipo = TypeInfo(Integer) then  Result := TNDArray.Create(x.AsInteger)
   else if ttipo = TypeInfo(Int64)   then  Result := TNDArray.Create(x.AsInt64)
   else if ttipo = TypeInfo(single)  then  Result := TNDArray.Create(Single(x.AsExtended))
   else if ttipo = TypeInfo(Double)  then  Result := TNDArray.Create(Double(x.AsExtended))
   else
     raise Exception.Create('NotImplementedException');
end;

function TNDArray.ToByteArray: TArray<Byte>;
begin
    Result :=  BufferToArray;
end;

end.
