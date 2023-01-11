unit ProtoGen.CostGraph;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  TInputInfo = Class
  const
    ftPrecedingNode = 1;
    ftPrecedingPort = 2;
  private
    FPrecedingNode: Integer;
    FPrecedingPort: Integer;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property PrecedingNode: Integer read FPrecedingNode write FPrecedingNode;
    property PrecedingPort: Integer read FPrecedingPort write FPrecedingPort;
  end;

  TOutputInfo = Class
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
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Size: Int64 read FSize write FSize;
    property AliasInputPort: Int64 read FAliasInputPort write FAliasInputPort;
    property Shape: TTensorShapeProto read FShape write FShape;
    property Dtype: TDataType read FDtype write FDtype;
  end;

  TNode = Class
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
    FInputInfos: TList<TInputInfo>;
    FOutputInfos: TList<TOutputInfo>;
    FTemporaryMemorySize: Int64;
    FPersistentMemorySize: Int64;
    FHostTempMemorySize: Int64;
    FDeviceTempMemorySize: Int64;
    FDevicePersistentMemorySize: Int64;
    FComputeCost: Int64;
    FComputeTime: Int64;
    FMemoryTime: Int64;
    FIsFinal: Boolean;
    FControlInputs: TList<Integer>;
    FInaccurate: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Device: string read FDevice write FDevice;
    property Id: Integer read FId write FId;
    property InputInfos: TList<TInputInfo> read FInputInfos;
    property OutputInfos: TList<TOutputInfo> read FOutputInfos;
    property TemporaryMemorySize: Int64 read FTemporaryMemorySize write FTemporaryMemorySize;
    property PersistentMemorySize: Int64 read FPersistentMemorySize write FPersistentMemorySize;
    property HostTempMemorySize: Int64 read FHostTempMemorySize write FHostTempMemorySize;
    property DeviceTempMemorySize: Int64 read FDeviceTempMemorySize write FDeviceTempMemorySize;
    property DevicePersistentMemorySize: Int64 read FDevicePersistentMemorySize write FDevicePersistentMemorySize;
    property ComputeCost: Int64 read FComputeCost write FComputeCost;
    property ComputeTime: Int64 read FComputeTime write FComputeTime;
    property MemoryTime: Int64 read FMemoryTime write FMemoryTime;
    property IsFinal: Boolean read FIsFinal write FIsFinal;
    property ControlInputs: TList<Integer> read FControlInputs;
    property Inaccurate: Boolean read FInaccurate write FInaccurate;
  end;

  TAggregatedCost = Class
  const
    ftCost = 1;
    ftDimension = 2;
  private
    FCost: Single;
    FDimension: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Cost: Single read FCost write FCost;
    property Dimension: string read FDimension write FDimension;
  end;

  TCostGraphDef = Class
  const
    ftNodes = 1;
    ftCosts = 2;
  private
    FNodes: TList<TNode>;
    FCosts: TList<TAggregatedCost>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Nodes: TList<TNode> read FNodes;
    property Costs: TList<TAggregatedCost> read FCosts;
  end;

implementation

{ TInputInfo }

Constructor TInputInfo.Create;
begin
  inherited Create;
end;

destructor TInputInfo.Destroy;
begin
  inherited Destroy;
end;

{ TOutputInfo }

Constructor TOutputInfo.Create;
begin
  inherited Create;
end;

destructor TOutputInfo.Destroy;
begin
  inherited Destroy;
end;

{ TNode }

Constructor TNode.Create;
begin
  inherited Create;
  
  FInputInfos := TList<TInputInfo>.Create;
  
  FOutputInfos := TList<TOutputInfo>.Create;
  
  FControlInputs := TList<Integer>.Create;
end;

destructor TNode.Destroy;
begin
  FInputInfos.Free;
  FOutputInfos.Free;
  FControlInputs.Free;
  inherited Destroy;
end;

{ TAggregatedCost }

Constructor TAggregatedCost.Create;
begin
  inherited Create;
end;

destructor TAggregatedCost.Destroy;
begin
  inherited Destroy;
end;

{ TCostGraphDef }

Constructor TCostGraphDef.Create;
begin
  inherited Create;
  
  FNodes := TList<TNode>.Create;
  
  FCosts := TList<TAggregatedCost>.Create;
end;

destructor TCostGraphDef.Destroy;
begin
  FNodes.Free;
  FCosts.Free;
  inherited Destroy;
end;

end.
