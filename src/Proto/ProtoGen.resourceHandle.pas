unit ProtoGen.ResourceHandle;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PDtypeAndShape = ^TDtypeAndShape;
  TDtypeAndShape = record
  const
    ftDtype = 1;
    ftShape = 2;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
  public
    procedure Init;
    procedure Free;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
  end;

  PResourceHandleProto = ^TResourceHandleProto;
  TResourceHandleProto = record
  const
    ftDevice = 1;
    ftContainer = 2;
    ftName = 3;
    ftHashCode = 4;
    ftMaybeTypeName = 5;
    ftDtypesAndShapess = 6;
  private
    FDevice: string;
    FContainer: string;
    FName: string;
    FHashCode: Int64;
    FMaybeTypeName: string;
    FDtypesAndShapess: TsgRecordList<TDtypeAndShape>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Device: string read FDevice write FDevice;
    property Container: string read FContainer write FContainer;
    property Name: string read FName write FName;
    property HashCode: Int64 read FHashCode write FHashCode;
    property MaybeTypeName: string read FMaybeTypeName write FMaybeTypeName;
    property DtypesAndShapess: TsgRecordList<TDtypeAndShape> read FDtypesAndShapess;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadResourceHandleProto(var Value: TResourceHandleProto);
    procedure LoadDtypeAndShape(var Value: TDtypeAndShape);
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

    class procedure SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto); static;
    class procedure SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
  end;

implementation

{ TDtypeAndShape }

procedure TDtypeAndShape.Init;
begin
  Self := Default(TDtypeAndShape);
end;

procedure TDtypeAndShape.Free;
begin
end;

{ TResourceHandleProto }

procedure TResourceHandleProto.Init;
begin
  Self := Default(TResourceHandleProto);
  FDtypesAndShapess := TsgRecordList<TDtypeAndShape>.From(nil);
end;

procedure TResourceHandleProto.Free;
begin
  FDtypesAndShapess.Free;
end;

procedure TLoadHelper.LoadDtypeAndShape(var Value: TDtypeAndShape);
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
      TDtypeAndShape.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TDtypeAndShape.ftShape:
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
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadResourceHandleProto(var Value: TResourceHandleProto);
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
      TResourceHandleProto.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TResourceHandleProto.ftContainer:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Container := Pb.readString;
        end;
      TResourceHandleProto.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TResourceHandleProto.ftHashCode:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HashCode := Pb.readInt64;
        end;
      TResourceHandleProto.ftMaybeTypeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.MaybeTypeName := Pb.readString;
        end;
      TResourceHandleProto.ftDtypesAndShapess:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDtypeAndShape;
            LoadDtypeAndShape(v);
            Value.FDtypesAndShapess.Add(@v);
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

class procedure TSaveHelper.SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape);
begin
  S.Pb.writeInt32(TDtypeAndShape.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.FShape, SaveTensorShapeProto, TDtypeAndShape.ftShape);
end;

class procedure TSaveHelper.SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto);
begin
  S.Pb.writeString(TResourceHandleProto.ftDevice, Value.Device);
  S.Pb.writeString(TResourceHandleProto.ftContainer, Value.Container);
  S.Pb.writeString(TResourceHandleProto.ftName, Value.Name);
  S.Pb.writeInt64(TResourceHandleProto.ftHashCode, Value.HashCode);
  S.Pb.writeString(TResourceHandleProto.ftMaybeTypeName, Value.MaybeTypeName);
  if Value.FDtypesAndShapess.Count > 0 then
    S.SaveList<TDtypeAndShape>(Value.FDtypesAndShapess, SaveDtypeAndShape, TResourceHandleProto.ftDtypesAndShapess);
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
