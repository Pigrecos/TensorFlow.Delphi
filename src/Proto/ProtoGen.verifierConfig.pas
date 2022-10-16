unit ProtoGen.VerifierConfig;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  TToggle = (
    DEFAULT = 0,
    ON = 1,
    OFF = 2);

  PVerifierConfig = ^TVerifierConfig;
  TVerifierConfig = record
  const
    ftVerificationTimeoutInMs = 1;
    ftStructureVerifier = 2;
  private
    FVerificationTimeoutInMs: Int64;
    FStructureVerifier: TToggle;
  public
    procedure Init;
    procedure Free;
    // properties
    property VerificationTimeoutInMs: Int64 read FVerificationTimeoutInMs write FVerificationTimeoutInMs;
    property StructureVerifier: TToggle read FStructureVerifier write FStructureVerifier;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadVerifierConfig(var Value: TVerifierConfig);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private

  public
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig); static;
  end;

implementation

{ TVerifierConfig }

procedure TVerifierConfig.Init;
begin
  Self := System.Default(TVerifierConfig);
end;

procedure TVerifierConfig.Free;
begin
end;

procedure TLoadHelper.LoadVerifierConfig(var Value: TVerifierConfig);
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
      TVerifierConfig.ftVerificationTimeoutInMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.VerificationTimeoutInMs := Pb.readInt64;
        end;
      TVerifierConfig.ftStructureVerifier:
        begin
          Assert(wireType = TWire.VARINT);
          Value.StructureVerifier := TToggle(Pb.readInt32);
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

class procedure TSaveHelper.SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig);
begin
  S.Pb.writeInt64(TVerifierConfig.ftVerificationTimeoutInMs, Value.VerificationTimeoutInMs);
  S.Pb.writeInt32(TVerifierConfig.ftStructureVerifier, Ord(Value.StructureVerifier));
end;

end.
