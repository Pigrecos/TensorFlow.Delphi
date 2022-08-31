unit ProtoGen.FullType;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,Oz.SGL.Heap;

{$T+}

type


  TFullTypeId = (
    TFT_UNSET = 0,
    TFT_VAR = 1,
    TFT_ANY = 2,
    TFT_PRODUCT = 3,
    TFT_NAMED = 4,
    TFT_FOR_EACH = 20,
    TFT_CALLABLE = 100,
    TFT_TENSOR = 1000,
    TFT_ARRAY = 1001,
    TFT_OPTIONAL = 1002,
    TFT_LITERAL = 1003,
    TFT_BOOL = 200,
    TFT_UINT8 = 201,
    TFT_UINT16 = 202,
    TFT_UINT32 = 203,
    TFT_UINT64 = 204,
    TFT_INT8 = 205,
    TFT_INT16 = 206,
    TFT_INT32 = 207,
    TFT_INT64 = 208,
    TFT_HALF = 209,
    TFT_FLOAT = 210,
    TFT_DOUBLE = 211,
    TFT_BFLOAT16 = 215,
    TFT_COMPLEX64 = 212,
    TFT_COMPLEX128 = 213,
    TFT_STRING = 214,
    TFT_DATASET = 10102,
    TFT_RAGGED = 10103,
    TFT_MUTEX_LOCK = 10202,
    TFT_LEGACY_VARIANT = 10203);

  PFullTypeDef = ^TFullTypeDef;
  TFullTypeDef = record
  const
    ftTypeId = 1;
    ftArgss = 2;
    ftS = 3;
    ftI = 4;
  private
    FTypeId: TFullTypeId;
    FArgss: TsgRecordList<TFullTypeDef>;
    FAttr: TpbOneof;
  public
    procedure Init;
    procedure Free;
    // properties
    property TypeId: TFullTypeId read FTypeId write FTypeId;
    property Argss: TsgRecordList<TFullTypeDef> read FArgss;
    property Attr: TpbOneof read FAttr write FAttr;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadFullTypeDef(var Value: TFullTypeDef);
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
    class procedure SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef); static;
  end;

implementation

{ TFullTypeDef }

procedure TFullTypeDef.Init;
begin
  Self := Default(TFullTypeDef);

  var m : TsgItemMeta;
  m.Init<TFullTypeDef>;
  FArgss := TsgRecordList<TFullTypeDef>.From(@m);
end;

procedure TFullTypeDef.Free;
begin
  FArgss.Free;
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
            Value.FArgss.Add(@v);
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

class procedure TSaveHelper.SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TFullTypeDef.ftTypeId, Ord(Value.TypeId));
  if Value.FArgss.Count > 0 then
    S.SaveList<TFullTypeDef>(Value.FArgss, SaveFullTypeDef, TFullTypeDef.ftArgss);
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

end.
