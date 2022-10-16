unit ProtoGen.CoordinationConfig;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  PCoordinationServiceConfig = ^TCoordinationServiceConfig;
  TCoordinationServiceConfig = record
  const
    ftServiceType = 1;
    ftServiceLeader = 2;
    ftEnableHealthCheck = 3;
    ftClusterRegisterTimeoutInMs = 4;
    ftHeartbeatTimeoutInMs = 5;
    ftCoordinatedJobss = 6;
  private
    FServiceType: string;
    FServiceLeader: string;
    FEnableHealthCheck: Boolean;
    FClusterRegisterTimeoutInMs: Int64;
    FHeartbeatTimeoutInMs: Int64;
    FCoordinatedJobss: TsgRecordList<string>;
  public
    procedure Init;
    procedure Free;
    // properties
    property ServiceType: string read FServiceType write FServiceType;
    property ServiceLeader: string read FServiceLeader write FServiceLeader;
    property EnableHealthCheck: Boolean read FEnableHealthCheck write FEnableHealthCheck;
    property ClusterRegisterTimeoutInMs: Int64 read FClusterRegisterTimeoutInMs write FClusterRegisterTimeoutInMs;
    property HeartbeatTimeoutInMs: Int64 read FHeartbeatTimeoutInMs write FHeartbeatTimeoutInMs;
    property CoordinatedJobss: TsgRecordList<string> read FCoordinatedJobss;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadCoordinationServiceConfig(var Value: TCoordinationServiceConfig);
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

    class procedure SaveCoordinationServiceConfig(const S: TpbSaver; const Value: TCoordinationServiceConfig); static;
  end;

implementation

{ TCoordinationServiceConfig }

procedure TCoordinationServiceConfig.Init;
begin
  Self := Default(TCoordinationServiceConfig);
  FCoordinatedJobss := TsgRecordList<string>.From(nil);
end;

procedure TCoordinationServiceConfig.Free;
begin
  FCoordinatedJobss.Free;
end;

procedure TLoadHelper.LoadCoordinationServiceConfig(var Value: TCoordinationServiceConfig);
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
      TCoordinationServiceConfig.ftServiceType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ServiceType := Pb.readString;
        end;
      TCoordinationServiceConfig.ftServiceLeader:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ServiceLeader := Pb.readString;
        end;
      TCoordinationServiceConfig.ftEnableHealthCheck:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableHealthCheck := Pb.readBoolean;
        end;
      TCoordinationServiceConfig.ftClusterRegisterTimeoutInMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ClusterRegisterTimeoutInMs := Pb.readInt64;
        end;
      TCoordinationServiceConfig.ftHeartbeatTimeoutInMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HeartbeatTimeoutInMs := Pb.readInt64;
        end;
      TCoordinationServiceConfig.ftCoordinatedJobss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FCoordinatedJobss.Add(@v);
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

class procedure TSaveHelper.SaveCoordinationServiceConfig(const S: TpbSaver; const Value: TCoordinationServiceConfig);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TCoordinationServiceConfig.ftServiceType, Value.ServiceType);
  S.Pb.writeString(TCoordinationServiceConfig.ftServiceLeader, Value.ServiceLeader);
  S.Pb.writeBoolean(TCoordinationServiceConfig.ftEnableHealthCheck, Value.EnableHealthCheck);
  S.Pb.writeInt64(TCoordinationServiceConfig.ftClusterRegisterTimeoutInMs, Value.ClusterRegisterTimeoutInMs);
  S.Pb.writeInt64(TCoordinationServiceConfig.ftHeartbeatTimeoutInMs, Value.HeartbeatTimeoutInMs);
  h.Init;
  try
    for i := 0 to Value.CoordinatedJobss.Count - 1 do
      h.Pb.writeRawString(Value.FCoordinatedJobss[i]^);
    S.Pb.writeMessage(TCoordinationServiceConfig.ftCoordinatedJobss, h.Pb^);
  finally
    h.Free;
  end;
end;

end.
