unit ProtoGen.TensorDescription;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  ProtoGen.allocationdescription,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PTensorDescription = ^TTensorDescription;
  TTensorDescription = record
  const
    ftDtype = 1;
    ftShape = 2;
    ftAllocationDescription = 4;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
    FAllocationDescription: TAllocationDescription;
  public
    procedure Init;
    procedure Free;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
    property AllocationDescription: TAllocationDescription read FAllocationDescription write FAllocationDescription;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadTensorDescription(var Value: TTensorDescription);
    procedure LoadAllocationDescription(var Value: TAllocationDescription);
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
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>;
      Save: TSavePair<Key, Value>; Tag: Integer);
  public
    class procedure SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription); static;
    class procedure SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
  end;

implementation

{ TTensorDescription }

procedure TTensorDescription.Init;
begin
  Self := Default(TTensorDescription);
end;

procedure TTensorDescription.Free;
begin
end;

procedure TLoadHelper.LoadTensorDescription(var Value: TTensorDescription);
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
      TTensorDescription.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TTensorDescription.ftShape:
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
      TTensorDescription.ftAllocationDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationDescription := Value.FAllocationDescription;
            LoadAllocationDescription(v);
            Value.FAllocationDescription := v;
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

procedure TLoadHelper.LoadAllocationDescription(var Value: TAllocationDescription);
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
      TAllocationDescription.ftRequestedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.RequestedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocatedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatorName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorName := Pb.readString;
        end;
      TAllocationDescription.ftAllocationId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocationId := Pb.readInt64;
        end;
      TAllocationDescription.ftHasSingleReference:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HasSingleReference := Pb.readBoolean;
        end;
      TAllocationDescription.ftPtr:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Ptr := Pb.readInt64;
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

class procedure TSaveHelper.SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TTensorDescription.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.FShape, SaveTensorShapeProto, TTensorDescription.ftShape);
  S.SaveObj<TAllocationDescription>(Value.FAllocationDescription, SaveAllocationDescription, TTensorDescription.ftAllocationDescription);
end;

class procedure TSaveHelper.SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt64(TAllocationDescription.ftRequestedBytes, Value.RequestedBytes);
  S.Pb.writeInt64(TAllocationDescription.ftAllocatedBytes, Value.AllocatedBytes);
  S.Pb.writeString(TAllocationDescription.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocationDescription.ftAllocationId, Value.AllocationId);
  S.Pb.writeBoolean(TAllocationDescription.ftHasSingleReference, Value.HasSingleReference);
  S.Pb.writeInt64(TAllocationDescription.ftPtr, Value.Ptr);
end;

class procedure TSaveHelper.SaveDim(const S: TpbSaver; const Value: TDim);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt64(TDim.ftSize, Value.Size);
  S.Pb.writeString(TDim.ftName, Value.Name);
end;

class procedure TSaveHelper.SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto);
var 
  i : Integer;
  h : TpbSaver;

begin
  if Value.Dims.Count > 0 then
    S.SaveList<TDim>(Value.Dims, SaveDim, TTensorShapeProto.ftDims);
  S.Pb.writeBoolean(TTensorShapeProto.ftUnknownRank, Value.UnknownRank);
end;

end.
