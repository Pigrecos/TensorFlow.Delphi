unit ProtoGen.Variable;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type

  TSaveSliceInfoDef = class;

  TVariableSynchronization = (
    VARIABLE_SYNCHRONIZATION_AUTO = 0,
    VARIABLE_SYNCHRONIZATION_NONE = 1,
    VARIABLE_SYNCHRONIZATION_ON_WRITE = 2,
    VARIABLE_SYNCHRONIZATION_ON_READ = 3);

  TVariableAggregation = (
    VARIABLE_AGGREGATION_NONE = 0,
    VARIABLE_AGGREGATION_SUM = 1,
    VARIABLE_AGGREGATION_MEAN = 2,
    VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA = 3);

  PVariableDef = ^TVariableDef;
  TVariableDef = record
  const
    ftVariableName = 1;
    ftInitialValueName = 6;
    ftInitializerName = 2;
    ftSnapshotName = 3;
    ftSaveSliceInfoDef = 4;
    ftIsResource = 5;
    ftTrainable = 7;
    ftSynchronization = 8;
    ftAggregation = 9;
  private
    FVariableName: string;
    FInitialValueName: string;
    FInitializerName: string;
    FSnapshotName: string;
    FSaveSliceInfoDef: TSaveSliceInfoDef;
    FIsResource: Boolean;
    FTrainable: Boolean;
    FSynchronization: TVariableSynchronization;
    FAggregation: TVariableAggregation;
  public
    procedure Init;
    procedure Free;
    // properties
    property VariableName: string read FVariableName write FVariableName;
    property InitialValueName: string read FInitialValueName write FInitialValueName;
    property InitializerName: string read FInitializerName write FInitializerName;
    property SnapshotName: string read FSnapshotName write FSnapshotName;
    property SaveSliceInfoDef: TSaveSliceInfoDef read FSaveSliceInfoDef write FSaveSliceInfoDef;
    property IsResource: Boolean read FIsResource write FIsResource;
    property Trainable: Boolean read FTrainable write FTrainable;
    property Synchronization: TVariableSynchronization read FSynchronization write FSynchronization;
    property Aggregation: TVariableAggregation read FAggregation write FAggregation;
  end;

  TSaveSliceInfoDef = Class
  const
    ftFullName = 1;
    ftFullShapes = 2;
    ftVarOffsets = 3;
    ftVarShapes = 4;
  private
    FFullName: string;
    FFullShapes: TsgRecordList<Int64>;
    FVarOffsets: TsgRecordList<Int64>;
    FVarShapes: TsgRecordList<Int64>;
  public
    Constructor Init;
    destructor Free;
    // properties
    property FullName: string read FFullName write FFullName;
    property FullShapes: TsgRecordList<Int64> read FFullShapes;
    property VarOffsets: TsgRecordList<Int64> read FVarOffsets;
    property VarShapes: TsgRecordList<Int64> read FVarShapes;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadVariableDef(var Value: TVariableDef);
    procedure LoadSaveSliceInfoDef(var Value: TSaveSliceInfoDef);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);
  public
    class procedure SaveVariableDef(const S: TpbSaver; const Value: TVariableDef); static;
    class procedure SaveSaveSliceInfoDef(const S: TpbSaver; const Value: TSaveSliceInfoDef); static;
  end;

implementation

{ TVariableDef }

procedure TVariableDef.Init;
begin
  Self := Default(TVariableDef);
end;

procedure TVariableDef.Free;
begin
end;

{ TSaveSliceInfoDef }

Constructor TSaveSliceInfoDef.Init;
begin
  inherited Create;
  FFullShapes := TsgRecordList<Int64>.From(nil);
  FVarOffsets := TsgRecordList<Int64>.From(nil);
  FVarShapes := TsgRecordList<Int64>.From(nil);
end;

destructor TSaveSliceInfoDef.Free;
begin
  FFullShapes.Free;
  FVarOffsets.Free;
  FVarShapes.Free;
  inherited Destroy;
end;

procedure TLoadHelper.LoadVariableDef(var Value: TVariableDef);
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
      TVariableDef.ftVariableName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.VariableName := Pb.readString;
        end;
      TVariableDef.ftInitialValueName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.InitialValueName := Pb.readString;
        end;
      TVariableDef.ftInitializerName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.InitializerName := Pb.readString;
        end;
      TVariableDef.ftSnapshotName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.SnapshotName := Pb.readString;
        end;
      TVariableDef.ftSaveSliceInfoDef:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TSaveSliceInfoDef := Value.FSaveSliceInfoDef;
            LoadSaveSliceInfoDef(v);
            Value.FSaveSliceInfoDef := v;
          finally
            Pb.Pop;
          end;
        end;
      TVariableDef.ftIsResource:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsResource := Pb.readBoolean;
        end;
      TVariableDef.ftTrainable:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Trainable := Pb.readBoolean;
        end;
      TVariableDef.ftSynchronization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Synchronization := TVariableSynchronization(Pb.readInt32);
        end;
      TVariableDef.ftAggregation:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Aggregation := TVariableAggregation(Pb.readInt32);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadSaveSliceInfoDef(var Value: TSaveSliceInfoDef);
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
      TSaveSliceInfoDef.ftFullName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.FullName := Pb.readString;
        end;
      TSaveSliceInfoDef.ftFullShapes:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.FFullShapes.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TSaveSliceInfoDef.ftVarOffsets:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.FVarOffsets.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TSaveSliceInfoDef.ftVarShapes:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.FVarShapes.Add(@v);
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

class procedure TSaveHelper.SaveVariableDef(const S: TpbSaver; const Value: TVariableDef);
begin
  S.Pb.writeString(TVariableDef.ftVariableName, Value.VariableName);
  S.Pb.writeString(TVariableDef.ftInitialValueName, Value.InitialValueName);
  S.Pb.writeString(TVariableDef.ftInitializerName, Value.InitializerName);
  S.Pb.writeString(TVariableDef.ftSnapshotName, Value.SnapshotName);
  S.SaveObj<TSaveSliceInfoDef>(Value.FSaveSliceInfoDef, SaveSaveSliceInfoDef, TVariableDef.ftSaveSliceInfoDef);
  S.Pb.writeBoolean(TVariableDef.ftIsResource, Value.IsResource);
  S.Pb.writeBoolean(TVariableDef.ftTrainable, Value.Trainable);
  S.Pb.writeInt32(TVariableDef.ftSynchronization, Ord(Value.Synchronization));
  S.Pb.writeInt32(TVariableDef.ftAggregation, Ord(Value.Aggregation));
end;

class procedure TSaveHelper.SaveSaveSliceInfoDef(const S: TpbSaver; const Value: TSaveSliceInfoDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TSaveSliceInfoDef.ftFullName, Value.FullName);
  h.Init;
  try
    for i := 0 to Value.FullShapes.Count - 1 do
      h.Pb.writeRawVarint64(Value.FFullShapes[i]^);
    S.Pb.writeMessage(TSaveSliceInfoDef.ftFullShapes, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.VarOffsets.Count - 1 do
      h.Pb.writeRawVarint64(Value.FVarOffsets[i]^);
    S.Pb.writeMessage(TSaveSliceInfoDef.ftVarOffsets, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.VarShapes.Count - 1 do
      h.Pb.writeRawVarint64(Value.FVarShapes[i]^);
    S.Pb.writeMessage(TSaveSliceInfoDef.ftVarShapes, h.Pb^);
  finally
    h.Free;
  end;
end;

end.
