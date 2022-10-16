unit ProtoGen.CostGraph;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PInputInfo = ^TInputInfo;
  TInputInfo = record
  const
    ftPrecedingNode = 1;
    ftPrecedingPort = 2;
  private
    FPrecedingNode: Integer;
    FPrecedingPort: Integer;
  public
    procedure Init;
    procedure Free;
    // properties
    property PrecedingNode: Integer read FPrecedingNode write FPrecedingNode;
    property PrecedingPort: Integer read FPrecedingPort write FPrecedingPort;
  end;

  POutputInfo = ^TOutputInfo;
  TOutputInfo = record
  const
    ftSize = 1;
    ftAliasInputPort = 2;
    ftShape = 3;
    ftDtype = 4;
  private
    FSize: Int64;
    FAliasInputPort: Int64;
    FShape: TTensorShapeProto;
    FDtype: TDataType;
  public
    procedure Init;
    procedure Free;
    // properties
    property Size: Int64 read FSize write FSize;
    property AliasInputPort: Int64 read FAliasInputPort write FAliasInputPort;
    property Shape: TTensorShapeProto read FShape write FShape;
    property Dtype: TDataType read FDtype write FDtype;
  end;

  PNode = ^TNode;
  TNode = record
  const
    ftName = 1;
    ftDevice = 2;
    ftId = 3;
    ftInputInfos = 4;
    ftOutputInfos = 5;
    ftTemporaryMemorySize = 6;
    ftPersistentMemorySize = 12;
    ftHostTempMemorySize = 10;
    ftDeviceTempMemorySize = 11;
    ftDevicePersistentMemorySize = 16;
    ftComputeCost = 9;
    ftComputeTime = 14;
    ftMemoryTime = 15;
    ftIsFinal = 7;
    ftControlInputs = 8;
    ftInaccurate = 17;
  private
    FName: string;
    FDevice: string;
    FId: Integer;
    FInputInfos: TsgRecordList<TInputInfo>;
    FOutputInfos: TsgRecordList<TOutputInfo>;
    FTemporaryMemorySize: Int64;
    FPersistentMemorySize: Int64;
    FHostTempMemorySize: Int64;
    FDeviceTempMemorySize: Int64;
    FDevicePersistentMemorySize: Int64;
    FComputeCost: Int64;
    FComputeTime: Int64;
    FMemoryTime: Int64;
    FIsFinal: Boolean;
    FControlInputs: TsgRecordList<Integer>;
    FInaccurate: Boolean;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property Device: string read FDevice write FDevice;
    property Id: Integer read FId write FId;
    property InputInfos: TsgRecordList<TInputInfo> read FInputInfos;
    property OutputInfos: TsgRecordList<TOutputInfo> read FOutputInfos;
    property TemporaryMemorySize: Int64 read FTemporaryMemorySize write FTemporaryMemorySize;
    property PersistentMemorySize: Int64 read FPersistentMemorySize write FPersistentMemorySize;
    property HostTempMemorySize: Int64 read FHostTempMemorySize write FHostTempMemorySize;
    property DeviceTempMemorySize: Int64 read FDeviceTempMemorySize write FDeviceTempMemorySize;
    property DevicePersistentMemorySize: Int64 read FDevicePersistentMemorySize write FDevicePersistentMemorySize;
    property ComputeCost: Int64 read FComputeCost write FComputeCost;
    property ComputeTime: Int64 read FComputeTime write FComputeTime;
    property MemoryTime: Int64 read FMemoryTime write FMemoryTime;
    property IsFinal: Boolean read FIsFinal write FIsFinal;
    property ControlInputs: TsgRecordList<Integer> read FControlInputs;
    property Inaccurate: Boolean read FInaccurate write FInaccurate;
  end;

  PAggregatedCost = ^TAggregatedCost;
  TAggregatedCost = record
  const
    ftCost = 1;
    ftDimension = 2;
  private
    FCost: Single;
    FDimension: string;
  public
    procedure Init;
    procedure Free;
    // properties
    property Cost: Single read FCost write FCost;
    property Dimension: string read FDimension write FDimension;
  end;

  PCostGraphDef = ^TCostGraphDef;
  TCostGraphDef = record
  const
    ftNodes = 1;
    ftCosts = 2;
  private
    FNodes: TsgRecordList<TNode>;
    FCosts: TsgRecordList<TAggregatedCost>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Nodes: TsgRecordList<TNode> read FNodes;
    property Costs: TsgRecordList<TAggregatedCost> read FCosts;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadCostGraphDef(var Value: TCostGraphDef);
    procedure LoadNode(var Value: TNode);
    procedure LoadInputInfo(var Value: TInputInfo);
    procedure LoadOutputInfo(var Value: TOutputInfo);
    procedure LoadAggregatedCost(var Value: TAggregatedCost);
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

    class procedure SaveCostGraphDef(const S: TpbSaver; const Value: TCostGraphDef); static;
    class procedure SaveNode(const S: TpbSaver; const Value: TNode); static;
    class procedure SaveInputInfo(const S: TpbSaver; const Value: TInputInfo); static;
    class procedure SaveOutputInfo(const S: TpbSaver; const Value: TOutputInfo); static;
    class procedure SaveAggregatedCost(const S: TpbSaver; const Value: TAggregatedCost); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
  end;

implementation

{ TInputInfo }

procedure TInputInfo.Init;
begin
  Self := Default(TInputInfo);
end;

procedure TInputInfo.Free;
begin
end;

{ TOutputInfo }

procedure TOutputInfo.Init;
begin
  Self := Default(TOutputInfo);
end;

procedure TOutputInfo.Free;
begin
end;

{ TNode }

procedure TNode.Init;
begin
  Self := Default(TNode);
  FInputInfos := TsgRecordList<TInputInfo>.From(nil);
  FOutputInfos := TsgRecordList<TOutputInfo>.From(nil);
  FControlInputs := TsgRecordList<Integer>.From(nil);
end;

procedure TNode.Free;
begin
  FInputInfos.Free;
  FOutputInfos.Free;
  FControlInputs.Free;
end;

{ TAggregatedCost }

procedure TAggregatedCost.Init;
begin
  Self := Default(TAggregatedCost);
end;

procedure TAggregatedCost.Free;
begin
end;

{ TCostGraphDef }

procedure TCostGraphDef.Init;
begin
  Self := Default(TCostGraphDef);
  FNodes := TsgRecordList<TNode>.From(nil);
  FCosts := TsgRecordList<TAggregatedCost>.From(nil);
end;

procedure TCostGraphDef.Free;
begin
  FNodes.Free;
  FCosts.Free;
end;

procedure TLoadHelper.LoadInputInfo(var Value: TInputInfo);
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
      TInputInfo.ftPrecedingNode:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PrecedingNode := Pb.readInt32;
        end;
      TInputInfo.ftPrecedingPort:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PrecedingPort := Pb.readInt32;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOutputInfo(var Value: TOutputInfo);
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
      TOutputInfo.ftSize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Size := Pb.readInt64;
        end;
      TOutputInfo.ftAliasInputPort:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AliasInputPort := Pb.readInt64;
        end;
      TOutputInfo.ftShape:
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
      TOutputInfo.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadNode(var Value: TNode);
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
      TNode.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TNode.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TNode.ftId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Id := Pb.readInt32;
        end;
      TNode.ftInputInfos:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TInputInfo;
            LoadInputInfo(v);
            Value.FInputInfos.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNode.ftOutputInfos:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOutputInfo;
            LoadOutputInfo(v);
            Value.FOutputInfos.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNode.ftTemporaryMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TemporaryMemorySize := Pb.readInt64;
        end;
      TNode.ftPersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PersistentMemorySize := Pb.readInt64;
        end;
      TNode.ftHostTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HostTempMemorySize := Pb.readInt64;
        end;
      TNode.ftDeviceTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DeviceTempMemorySize := Pb.readInt64;
        end;
      TNode.ftDevicePersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DevicePersistentMemorySize := Pb.readInt64;
        end;
      TNode.ftComputeCost:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ComputeCost := Pb.readInt64;
        end;
      TNode.ftComputeTime:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ComputeTime := Pb.readInt64;
        end;
      TNode.ftMemoryTime:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MemoryTime := Pb.readInt64;
        end;
      TNode.ftIsFinal:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsFinal := Pb.readBoolean;
        end;
      TNode.ftControlInputs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.FControlInputs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TNode.ftInaccurate:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Inaccurate := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadAggregatedCost(var Value: TAggregatedCost);
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
      TAggregatedCost.ftCost:
        begin
          Assert(wireType = TWire.FIXED32);
          Value.Cost := Pb.readFloat;
        end;
      TAggregatedCost.ftDimension:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Dimension := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadCostGraphDef(var Value: TCostGraphDef);
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
      TCostGraphDef.ftNodes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNode;
            LoadNode(v);
            Value.FNodes.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TCostGraphDef.ftCosts:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAggregatedCost;
            LoadAggregatedCost(v);
            Value.FCosts.Add(@v);
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

class procedure TSaveHelper.SaveInputInfo(const S: TpbSaver; const Value: TInputInfo);
begin
  S.Pb.writeInt32(TInputInfo.ftPrecedingNode, Value.PrecedingNode);
  S.Pb.writeInt32(TInputInfo.ftPrecedingPort, Value.PrecedingPort);
end;

class procedure TSaveHelper.SaveOutputInfo(const S: TpbSaver; const Value: TOutputInfo);
begin
  S.Pb.writeInt64(TOutputInfo.ftSize, Value.Size);
  S.Pb.writeInt64(TOutputInfo.ftAliasInputPort, Value.AliasInputPort);
  S.SaveObj<TTensorShapeProto>(Value.FShape, SaveTensorShapeProto, TOutputInfo.ftShape);
  S.Pb.writeInt32(TOutputInfo.ftDtype, Ord(Value.Dtype));
end;

class procedure TSaveHelper.SaveNode(const S: TpbSaver; const Value: TNode);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TNode.ftName, Value.Name);
  S.Pb.writeString(TNode.ftDevice, Value.Device);
  S.Pb.writeInt32(TNode.ftId, Value.Id);
  if Value.FInputInfos.Count > 0 then
    S.SaveList<TInputInfo>(Value.FInputInfos, SaveInputInfo, TNode.ftInputInfos);
  if Value.FOutputInfos.Count > 0 then
    S.SaveList<TOutputInfo>(Value.FOutputInfos, SaveOutputInfo, TNode.ftOutputInfos);
  S.Pb.writeInt64(TNode.ftTemporaryMemorySize, Value.TemporaryMemorySize);
  S.Pb.writeInt64(TNode.ftPersistentMemorySize, Value.PersistentMemorySize);
  S.Pb.writeInt64(TNode.ftHostTempMemorySize, Value.HostTempMemorySize);
  S.Pb.writeInt64(TNode.ftDeviceTempMemorySize, Value.DeviceTempMemorySize);
  S.Pb.writeInt64(TNode.ftDevicePersistentMemorySize, Value.DevicePersistentMemorySize);
  S.Pb.writeInt64(TNode.ftComputeCost, Value.ComputeCost);
  S.Pb.writeInt64(TNode.ftComputeTime, Value.ComputeTime);
  S.Pb.writeInt64(TNode.ftMemoryTime, Value.MemoryTime);
  S.Pb.writeBoolean(TNode.ftIsFinal, Value.IsFinal);
  h.Init;
  try
    for i := 0 to Value.ControlInputs.Count - 1 do
      h.Pb.writeRawVarint32(Value.FControlInputs[i]^);
    S.Pb.writeMessage(TNode.ftControlInputs, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeBoolean(TNode.ftInaccurate, Value.Inaccurate);
end;

class procedure TSaveHelper.SaveAggregatedCost(const S: TpbSaver; const Value: TAggregatedCost);
begin
  S.Pb.writeFloat(TAggregatedCost.ftCost, Value.Cost);
  S.Pb.writeString(TAggregatedCost.ftDimension, Value.Dimension);
end;

class procedure TSaveHelper.SaveCostGraphDef(const S: TpbSaver; const Value: TCostGraphDef);
begin
  if Value.FNodes.Count > 0 then
    S.SaveList<TNode>(Value.FNodes, SaveNode, TCostGraphDef.ftNodes);
  if Value.FCosts.Count > 0 then
    S.SaveList<TAggregatedCost>(Value.FCosts, SaveAggregatedCost, TCostGraphDef.ftCosts);
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
