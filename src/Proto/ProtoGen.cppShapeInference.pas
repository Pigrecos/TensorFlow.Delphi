unit ProtoGen.CppShapeInference;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,Oz.SGL.Heap,
  ProtoGen.fulltype,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PHandleShapeAndType = ^THandleShapeAndType;
  THandleShapeAndType = record
  const
    ftShape = 1;
    ftDtype = 2;
    ftType = 4;
  private
    FShape: TTensorShapeProto;
    FDtype: TDataType;
    FType: TFullTypeDef;
  public
    procedure Init;
    procedure Free;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property Dtype: TDataType read FDtype write FDtype;
    property &Type: TFullTypeDef read FType write FType;
  end;

  PHandleData = ^THandleData;
  THandleData = record
  const
    ftIsSet = 1;
    ftShapeAndTypes = 2;
  private
    FIsSet: Boolean;
    FShapeAndTypes: TsgRecordList<THandleShapeAndType>;
  public
    procedure Init;
    procedure Free;
    // properties
    property IsSet: Boolean read FIsSet write FIsSet;
    property ShapeAndTypes: TsgRecordList<THandleShapeAndType> read FShapeAndTypes;
  end;

  PCppShapeInferenceResult = ^TCppShapeInferenceResult;
  TCppShapeInferenceResult = record
  const
    ftShape = 1;
    ftHandleData = 4;
  private
    FShape: TTensorShapeProto;
    FHandleData: THandleData;
  public
    procedure Init;
    procedure Free;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property HandleData: THandleData read FHandleData write FHandleData;
  end;

  PCppShapeInferenceInputsNeeded = ^TCppShapeInferenceInputsNeeded;
  TCppShapeInferenceInputsNeeded = record
  const
    ftInputTensorsNeededs = 1;
    ftInputTensorsAsShapesNeededs = 2;
  private
    FInputTensorsNeededs: TsgRecordList<Integer>;
    FInputTensorsAsShapesNeededs: TsgRecordList<Integer>;
  public
    procedure Init;
    procedure Free;
    // properties
    property InputTensorsNeededs: TsgRecordList<Integer> read FInputTensorsNeededs;
    property InputTensorsAsShapesNeededs: TsgRecordList<Integer> read FInputTensorsAsShapesNeededs;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadCppShapeInferenceResult(var Value: TCppShapeInferenceResult);
    procedure LoadHandleShapeAndType(var Value: THandleShapeAndType);
    procedure LoadHandleData(var Value: THandleData);
    procedure LoadCppShapeInferenceInputsNeeded(var Value: TCppShapeInferenceInputsNeeded);
    procedure LoadFullTypeDef(var Value: TFullTypeDef);
    procedure LoadTensorShapeProto(var Value: TTensorShapeProto);
    procedure LoadDim(var Value: TDim);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);

  public
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveCppShapeInferenceResult(const S: TpbSaver; const Value: TCppShapeInferenceResult); static;
    class procedure SaveHandleShapeAndType(const S: TpbSaver; const Value: THandleShapeAndType); static;
    class procedure SaveHandleData(const S: TpbSaver; const Value: THandleData); static;
    class procedure SaveCppShapeInferenceInputsNeeded(const S: TpbSaver; const Value: TCppShapeInferenceInputsNeeded); static;
    class procedure SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
  end;

implementation

{ THandleShapeAndType }

procedure THandleShapeAndType.Init;
begin
  Self := Default(THandleShapeAndType);

end;

procedure THandleShapeAndType.Free;
begin
end;

{ THandleData }

procedure THandleData.Init;
begin
  Self := Default(THandleData);
  var m : TsgItemMeta;
  
  m := Default(TsgItemMeta);
  m.Init<THandleShapeAndType>;
  FShapeAndTypes := TsgRecordList<THandleShapeAndType>.From(@m);
end;

procedure THandleData.Free;
begin
  FShapeAndTypes.Free;
end;

{ TCppShapeInferenceResult }

procedure TCppShapeInferenceResult.Init;
begin
  Self := Default(TCppShapeInferenceResult);

end;

procedure TCppShapeInferenceResult.Free;
begin
end;

{ TCppShapeInferenceInputsNeeded }

procedure TCppShapeInferenceInputsNeeded.Init;
begin
  Self := Default(TCppShapeInferenceInputsNeeded);
  var m : TsgItemMeta;
  
  m := Default(TsgItemMeta);
  m.Init<Integer>;
  FInputTensorsNeededs := TsgRecordList<Integer>.From(@m);
  
  m := Default(TsgItemMeta);
  m.Init<Integer>;
  FInputTensorsAsShapesNeededs := TsgRecordList<Integer>.From(@m);
end;

procedure TCppShapeInferenceInputsNeeded.Free;
begin
  FInputTensorsNeededs.Free;
  FInputTensorsAsShapesNeededs.Free;
end;

procedure TLoadHelper.LoadHandleShapeAndType(var Value: THandleShapeAndType);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      THandleShapeAndType.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.FShape;
            LoadTensorShapeProto(v);
            Value.FShape := v;
          finally
            Pb.Pop;
          end;
        end;
      THandleShapeAndType.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      THandleShapeAndType.ftType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.FType;
            LoadFullTypeDef(v);
            Value.FType := v;
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadHandleData(var Value: THandleData);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      THandleData.ftIsSet:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsSet := Pb.readBoolean;
        end;
      THandleData.ftShapeAndTypes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : THandleShapeAndType;
            LoadHandleShapeAndType(v);
            Value.FShapeAndTypes.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadCppShapeInferenceResult(var Value: TCppShapeInferenceResult);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCppShapeInferenceResult.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.FShape;
            LoadTensorShapeProto(v);
            Value.FShape := v;
          finally
            Pb.Pop;
          end;
        end;
      TCppShapeInferenceResult.ftHandleData:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : THandleData := Value.FHandleData;
            LoadHandleData(v);
            Value.FHandleData := v;
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadCppShapeInferenceInputsNeeded(var Value: TCppShapeInferenceInputsNeeded);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCppShapeInferenceInputsNeeded.ftInputTensorsNeededs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.FInputTensorsNeededs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TCppShapeInferenceInputsNeeded.ftInputTensorsAsShapesNeededs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.FInputTensorsAsShapesNeededs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadFullTypeDef(var Value: TFullTypeDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TFullTypeDef.ftTypeId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TypeId := TFullTypeId(Pb.readInt32);
        end;
      TFullTypeDef.ftArgss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef;
            LoadFullTypeDef(v);
            Value.Argss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFullTypeDef.ftS:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TFullTypeDef.ftS;
          v.value := Pb.readString;
          Value.attr := v;
        end;
      TFullTypeDef.ftI:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TFullTypeDef.ftI;
          v.value := Pb.readInt64;
          Value.attr := v;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDim(var Value: TDim);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TDim.ftSize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Size := Pb.readInt64;
        end;
      TDim.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadTensorShapeProto(var Value: TTensorShapeProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TTensorShapeProto.ftDims:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDim;
            LoadDim(v);
            Value.Dims.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorShapeProto.ftUnknownRank:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UnknownRank := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

{ TSaveHelper }

procedure TSaveHelper.SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
var
  h: TpbSaver;
begin
  h.Init;
  try
    Save(h, obj);
    Pb.writeMessage(tag, h.Pb^);
  finally
    h.Free;
  end;
end;

procedure TSaveHelper.SaveList<T>(const List: TsgRecordList<T>;
  Save: TSave<T>; Tag: Integer);
var
  i: Integer;
  h: TpbSaver;
begin
  h.Init;
  try
    for i := 0 to List.Count - 1 do
    begin
      h.Clear;
      Save(h, List[i]^);
      Pb.writeMessage(tag, h.Pb^);
    end;
  finally
    h.Free;
  end;
end;

procedure TSaveHelper.SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>;
  Save: TSavePair<Key, Value>; Tag: Integer);
var
  h: TpbSaver;
  Pair: TsgHashMapIterator<Key, Value>.PPair;
  it: TsgHashMapIterator<Key, Value>;
begin
  h.Init;
  try
    it := Map.Begins;
    while it <> Map.Ends do
    begin
      h.Clear;
      Save(h, it.GetPair^);
      Pb.writeMessage(tag, h.Pb^);
      it.Next;
    end;
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveHandleShapeAndType(const S: TpbSaver; const Value: THandleShapeAndType);
begin
  S.SaveObj<TTensorShapeProto>(Value.FShape, SaveTensorShapeProto, THandleShapeAndType.ftShape);
  S.Pb.writeInt32(THandleShapeAndType.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TFullTypeDef>(Value.FType, SaveFullTypeDef, THandleShapeAndType.ftType);
end;

class procedure TSaveHelper.SaveHandleData(const S: TpbSaver; const Value: THandleData);
begin
  S.Pb.writeBoolean(THandleData.ftIsSet, Value.IsSet);
  if Value.FShapeAndTypes.Count > 0 then
    S.SaveList<THandleShapeAndType>(Value.FShapeAndTypes, SaveHandleShapeAndType, THandleData.ftShapeAndTypes);
end;

class procedure TSaveHelper.SaveCppShapeInferenceResult(const S: TpbSaver; const Value: TCppShapeInferenceResult);
begin
  S.SaveObj<TTensorShapeProto>(Value.FShape, SaveTensorShapeProto, TCppShapeInferenceResult.ftShape);
  S.SaveObj<THandleData>(Value.FHandleData, SaveHandleData, TCppShapeInferenceResult.ftHandleData);
end;

class procedure TSaveHelper.SaveCppShapeInferenceInputsNeeded(const S: TpbSaver; const Value: TCppShapeInferenceInputsNeeded);
var
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.InputTensorsNeededs.Count - 1 do
      h.Pb.writeRawVarint32(Value.FInputTensorsNeededs[i]^);
    S.Pb.writeMessage(TCppShapeInferenceInputsNeeded.ftInputTensorsNeededs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.InputTensorsAsShapesNeededs.Count - 1 do
      h.Pb.writeRawVarint32(Value.FInputTensorsAsShapesNeededs[i]^);
    S.Pb.writeMessage(TCppShapeInferenceInputsNeeded.ftInputTensorsAsShapesNeededs, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef);
begin
  S.Pb.writeInt32(TFullTypeDef.ftTypeId, Ord(Value.TypeId));
  if Value.Argss.Count > 0 then
    S.SaveList<TFullTypeDef>(Value.Argss, SaveFullTypeDef, TFullTypeDef.ftArgss);
  case Value.attr.tag of
    TFullTypeDef.ftS:
      begin
        S.Pb.writeString(Value.ftS, Value.Attr.value.AsType<string>);
      end;
    TFullTypeDef.ftI:
      begin
        S.Pb.writeInt64(Value.ftI, Value.Attr.value.AsType<Int64>);
      end;
  end;
end;

class procedure TSaveHelper.SaveDim(const S: TpbSaver; const Value: TDim);
begin
  S.Pb.writeInt64(TDim.ftSize, Value.Size);
  S.Pb.writeString(TDim.ftName, Value.Name);
end;

class procedure TSaveHelper.SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto);
begin
  if Value.Dims.Count > 0 then
    S.SaveList<TDim>(Value.Dims, SaveDim, TTensorShapeProto.ftDims);
  S.Pb.writeBoolean(TTensorShapeProto.ftUnknownRank, Value.UnknownRank);
end;

end.
