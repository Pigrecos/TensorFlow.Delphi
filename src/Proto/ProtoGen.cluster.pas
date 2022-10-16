unit ProtoGen.Cluster;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  TInt32String = TsgHashMap<Integer, string>;

  PJobDef = ^TJobDef;
  TJobDef = record
  const
    ftName = 1;
    ftTasks = 2;
  private
    FName: string;
    FTasks: TInt32String;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property Tasks: TInt32String read FTasks write FTasks;
  end;

  PClusterDef = ^TClusterDef;
  TClusterDef = record
  const
    ftJobs = 1;
  private
    FJobs: TsgRecordList<TJobDef>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Jobs: TsgRecordList<TJobDef> read FJobs;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadJobDef(var Value: TJobDef);
    procedure LoadClusterDef(var Value: TClusterDef);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private

    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);
  public
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveJobDef(const S: TpbSaver; const Value: TJobDef); static;
    class procedure SaveClusterDef(const S: TpbSaver; const Value: TClusterDef); static;
    procedure SaveInt32String(Item: TsgPair<Integer, string>);
  end;

implementation

{ TJobDef }

procedure TJobDef.Init;
begin
  Self := Default(TJobDef);
  FTasks := TsgHashMap<Integer, string>.From(0,nil);
end;

procedure TJobDef.Free;
begin
  FTasks.Free;
end;

{ TClusterDef }

procedure TClusterDef.Init;
begin
  Self := Default(TClusterDef);
  FJobs := TsgRecordList<TJobDef>.From(nil);
end;

procedure TClusterDef.Free;
begin
  FJobs.Free;
end;

procedure TLoadHelper.LoadJobDef(var Value: TJobDef);
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
      TJobDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TJobDef.ftTasks:
        begin
          Value.Tasks.InsertOrAssign(TsgPair<Integer, string>.From(Pb.readInt32, Pb.readString));
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadClusterDef(var Value: TClusterDef);
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
      TClusterDef.ftJobs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TJobDef;
            LoadJobDef(v);
            Value.FJobs.Add(@v);
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

class procedure TSaveHelper.SaveJobDef(const S: TpbSaver; const Value: TJobDef);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TJobDef.ftName, Value.Name);
  if Value.FTasks.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.FTasks.Begins;
      while it <> Value.FTasks.Ends do
      begin
          h.clear;
          h.SaveInt32String(it.GetPair^);
          S.Pb.writeMessage(TJobDef.ftTasks, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveClusterDef(const S: TpbSaver; const Value: TClusterDef);
begin
  if Value.FJobs.Count > 0 then
    S.SaveList<TJobDef>(Value.FJobs, SaveJobDef, TClusterDef.ftJobs);
end;

procedure TSaveHelper.SaveInt32String(Item: TsgPair<Integer, string>);
begin
  Pb.writeInt32(1, Item.Key);
  Pb.writeString(2, Item.Value);
end;

end.
