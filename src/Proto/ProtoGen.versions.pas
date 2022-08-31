unit ProtoGen.Versions;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  PVersionDef = ^TVersionDef;
  TVersionDef = record
  const
    ftProducer = 1;
    ftMinConsumer = 2;
    ftBadConsumerss = 3;
  private
    FProducer: Integer;
    FMinConsumer: Integer;
    FBadConsumerss: TsgRecordList<Integer>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Producer: Integer read FProducer write FProducer;
    property MinConsumer: Integer read FMinConsumer write FMinConsumer;
    property BadConsumerss: TsgRecordList<Integer> read FBadConsumerss;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadVersionDef(var Value: TVersionDef);
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
    class procedure SaveVersionDef(const S: TpbSaver; const Value: TVersionDef); static;
  end;

implementation

{ TVersionDef }

procedure TVersionDef.Init;
begin
  Self := Default(TVersionDef);
  FBadConsumerss := TsgRecordList<Integer>.From(nil);
end;

procedure TVersionDef.Free;
begin
  FBadConsumerss.Free;
end;

procedure TLoadHelper.LoadVersionDef(var Value: TVersionDef);
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
      TVersionDef.ftProducer:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Producer := Pb.readInt32;
        end;
      TVersionDef.ftMinConsumer:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MinConsumer := Pb.readInt32;
        end;
      TVersionDef.ftBadConsumerss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.FBadConsumerss.Add(@v);
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

class procedure TSaveHelper.SaveVersionDef(const S: TpbSaver; const Value: TVersionDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TVersionDef.ftProducer, Value.Producer);
  S.Pb.writeInt32(TVersionDef.ftMinConsumer, Value.MinConsumer);
  h.Init;
  try
    for i := 0 to Value.BadConsumerss.Count - 1 do
      h.Pb.writeRawVarint32(Value.FBadConsumerss[i]^);
    S.Pb.writeMessage(TVersionDef.ftBadConsumerss, h.Pb^);
  finally
    h.Free;
  end;
end;

end.
