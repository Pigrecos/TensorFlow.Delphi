unit ProtoGen.StepStats;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.allocationdescription,
  ProtoGen.tensordescription,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  TAllocationRecord = Class
  const
    ftAllocMicros = 1;
    ftAllocBytes = 2;
  private
    FAllocMicros: Int64;
    FAllocBytes: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property AllocMicros: Int64 read FAllocMicros write FAllocMicros;
    property AllocBytes: Int64 read FAllocBytes write FAllocBytes;
  end;

  TAllocatorMemoryUsed = Class
  const
    ftAllocatorName = 1;
    ftTotalBytes = 2;
    ftPeakBytes = 3;
    ftLiveBytes = 4;
    ftAllocationRecordss = 6;
    ftAllocatorBytesInUse = 5;
  private
    FAllocatorName: string;
    FTotalBytes: Int64;
    FPeakBytes: Int64;
    FLiveBytes: Int64;
    FAllocationRecordss: TList<TAllocationRecord>;
    FAllocatorBytesInUse: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property AllocatorName: string read FAllocatorName write FAllocatorName;
    property TotalBytes: Int64 read FTotalBytes write FTotalBytes;
    property PeakBytes: Int64 read FPeakBytes write FPeakBytes;
    property LiveBytes: Int64 read FLiveBytes write FLiveBytes;
    property AllocationRecordss: TList<TAllocationRecord> read FAllocationRecordss;
    property AllocatorBytesInUse: Int64 read FAllocatorBytesInUse write FAllocatorBytesInUse;
  end;

  TNodeOutput = Class
  const
    ftSlot = 1;
    ftTensorDescription = 3;
  private
    FSlot: Integer;
    FTensorDescription: TTensorDescription;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Slot: Integer read FSlot write FSlot;
    property TensorDescription: TTensorDescription read FTensorDescription write FTensorDescription;
  end;

  TMemoryStats = Class
  const
    ftTempMemorySize = 1;
    ftPersistentMemorySize = 3;
    ftPersistentTensorAllocIdss = 5;
    ftDeviceTempMemorySize = 2;
    ftDevicePersistentMemorySize = 4;
    ftDevicePersistentTensorAllocIdss = 6;
  private
    FTempMemorySize: Int64;
    FPersistentMemorySize: Int64;
    FPersistentTensorAllocIdss: TList<Int64>;
    FDeviceTempMemorySize: Int64;
    FDevicePersistentMemorySize: Int64;
    FDevicePersistentTensorAllocIdss: TList<Int64>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TempMemorySize: Int64 read FTempMemorySize write FTempMemorySize;
    property PersistentMemorySize: Int64 read FPersistentMemorySize write FPersistentMemorySize;
    property PersistentTensorAllocIdss: TList<Int64> read FPersistentTensorAllocIdss;
    property DeviceTempMemorySize: Int64 read FDeviceTempMemorySize write FDeviceTempMemorySize;
    property DevicePersistentMemorySize: Int64 read FDevicePersistentMemorySize write FDevicePersistentMemorySize;
    property DevicePersistentTensorAllocIdss: TList<Int64> read FDevicePersistentTensorAllocIdss;
  end;

  TNodeExecStats = Class
  const
    ftNodeName = 1;
    ftAllStartMicros = 2;
    ftOpStartRelMicros = 3;
    ftOpEndRelMicros = 4;
    ftAllEndRelMicros = 5;
    ftMemorys = 6;
    ftOutputs = 7;
    ftTimelineLabel = 8;
    ftScheduledMicros = 9;
    ftThreadId = 10;
    ftReferencedTensors = 11;
    ftMemoryStats = 12;
    ftAllStartNanos = 13;
    ftOpStartRelNanos = 14;
    ftOpEndRelNanos = 15;
    ftAllEndRelNanos = 16;
    ftScheduledNanos = 17;
  private
    FNodeName: string;
    FAllStartMicros: Int64;
    FOpStartRelMicros: Int64;
    FOpEndRelMicros: Int64;
    FAllEndRelMicros: Int64;
    FMemorys: TList<TAllocatorMemoryUsed>;
    FOutputs: TList<TNodeOutput>;
    FTimelineLabel: string;
    FScheduledMicros: Int64;
    FThreadId: UInt32;
    FReferencedTensors: TList<TAllocationDescription>;
    FMemoryStats: TMemoryStats;
    FAllStartNanos: Int64;
    FOpStartRelNanos: Int64;
    FOpEndRelNanos: Int64;
    FAllEndRelNanos: Int64;
    FScheduledNanos: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property NodeName: string read FNodeName write FNodeName;
    property AllStartMicros: Int64 read FAllStartMicros write FAllStartMicros;
    property OpStartRelMicros: Int64 read FOpStartRelMicros write FOpStartRelMicros;
    property OpEndRelMicros: Int64 read FOpEndRelMicros write FOpEndRelMicros;
    property AllEndRelMicros: Int64 read FAllEndRelMicros write FAllEndRelMicros;
    property Memorys: TList<TAllocatorMemoryUsed> read FMemorys;
    property Outputs: TList<TNodeOutput> read FOutputs;
    property TimelineLabel: string read FTimelineLabel write FTimelineLabel;
    property ScheduledMicros: Int64 read FScheduledMicros write FScheduledMicros;
    property ThreadId: UInt32 read FThreadId write FThreadId;
    property ReferencedTensors: TList<TAllocationDescription> read FReferencedTensors;
    property MemoryStats: TMemoryStats read FMemoryStats write FMemoryStats;
    property AllStartNanos: Int64 read FAllStartNanos write FAllStartNanos;
    property OpStartRelNanos: Int64 read FOpStartRelNanos write FOpStartRelNanos;
    property OpEndRelNanos: Int64 read FOpEndRelNanos write FOpEndRelNanos;
    property AllEndRelNanos: Int64 read FAllEndRelNanos write FAllEndRelNanos;
    property ScheduledNanos: Int64 read FScheduledNanos write FScheduledNanos;
  end;

  TDeviceStepStats = Class
  const
    ftDevice = 1;
    ftNodeStatss = 2;
    ftThreadNames = 3;
  private
    FDevice: string;
    FNodeStatss: TList<TNodeExecStats>;
    FThreadNames: TUint32String;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Device: string read FDevice write FDevice;
    property NodeStatss: TList<TNodeExecStats> read FNodeStatss;
    property ThreadNames: TUint32String read FThreadNames write FThreadNames;
  end;

  TStepStats = Class
  const
    ftDevStatss = 1;
  private
    FDevStatss: TList<TDeviceStepStats>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property DevStatss: TList<TDeviceStepStats> read FDevStatss;
  end;

implementation

{ TAllocationRecord }

Constructor TAllocationRecord.Create;
begin
  inherited Create;
end;

destructor TAllocationRecord.Destroy;
begin
  inherited Destroy;
end;

{ TAllocatorMemoryUsed }

Constructor TAllocatorMemoryUsed.Create;
begin
  inherited Create;
  
  FAllocationRecordss := TList<TAllocationRecord>.Create;
end;

destructor TAllocatorMemoryUsed.Destroy;
begin
  FAllocationRecordss.Free;
  inherited Destroy;
end;

{ TNodeOutput }

Constructor TNodeOutput.Create;
begin
  inherited Create;
end;

destructor TNodeOutput.Destroy;
begin
  inherited Destroy;
end;

{ TMemoryStats }

Constructor TMemoryStats.Create;
begin
  inherited Create;
  
  FPersistentTensorAllocIdss := TList<Int64>.Create;
  
  FDevicePersistentTensorAllocIdss := TList<Int64>.Create;
end;

destructor TMemoryStats.Destroy;
begin
  FPersistentTensorAllocIdss.Free;
  FDevicePersistentTensorAllocIdss.Free;
  inherited Destroy;
end;

{ TNodeExecStats }

Constructor TNodeExecStats.Create;
begin
  inherited Create;
  
  FMemorys := TList<TAllocatorMemoryUsed>.Create;
  
  FOutputs := TList<TNodeOutput>.Create;
  
  FReferencedTensors := TList<TAllocationDescription>.Create;
end;

destructor TNodeExecStats.Destroy;
begin
  FMemorys.Free;
  FOutputs.Free;
  FReferencedTensors.Free;
  inherited Destroy;
end;

{ TDeviceStepStats }

Constructor TDeviceStepStats.Create;
begin
  inherited Create;
  
  FNodeStatss := TList<TNodeExecStats>.Create;
  FThreadNames := TDictionary<UInt32, string>.Create;
end;

destructor TDeviceStepStats.Destroy;
begin
  FNodeStatss.Free;
  FThreadNames.Free;
  inherited Destroy;
end;

{ TStepStats }

Constructor TStepStats.Create;
begin
  inherited Create;
  
  FDevStatss := TList<TDeviceStepStats>.Create;
end;

destructor TStepStats.Destroy;
begin
  FDevStatss.Free;
  inherited Destroy;
end;

end.
