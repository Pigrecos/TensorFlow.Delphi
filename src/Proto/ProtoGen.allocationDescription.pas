unit ProtoGen.AllocationDescription;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  PAllocationDescription = ^TAllocationDescription;
  TAllocationDescription = record
  const
    ftRequestedBytes = 1;
    ftAllocatedBytes = 2;
    ftAllocatorName = 3;
    ftAllocationId = 4;
    ftHasSingleReference = 5;
    ftPtr = 6;
  private
    FRequestedBytes: Int64;
    FAllocatedBytes: Int64;
    FAllocatorName: string;
    FAllocationId: Int64;
    FHasSingleReference: Boolean;
    FPtr: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property RequestedBytes: Int64 read FRequestedBytes write FRequestedBytes;
    property AllocatedBytes: Int64 read FAllocatedBytes write FAllocatedBytes;
    property AllocatorName: string read FAllocatorName write FAllocatorName;
    property AllocationId: Int64 read FAllocationId write FAllocationId;
    property HasSingleReference: Boolean read FHasSingleReference write FHasSingleReference;
    property Ptr: Int64 read FPtr write FPtr;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadAllocationDescription(var Value: TAllocationDescription);
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
    class procedure SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription); static;
  end;

implementation

{ TAllocationDescription }

procedure TAllocationDescription.Init;
begin
  Self := Default(TAllocationDescription);
end;

procedure TAllocationDescription.Free;
begin
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

end.
