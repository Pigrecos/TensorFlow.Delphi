unit TensorFlow.Proto;
{$REGION 'Licence'}
(*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************)
{$ENDREGION}

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}
{$T+}

interface
uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes ;

type
  TVariantTensorDataProto = class;
  TAttrValue              = Class;
  TNameAttrList           = class;
  TOpDeprecation          = class;
  TFunctionDef            = class;
  TRegisteredGradient     = class;
  TGradientDef            = class;
  TSaveSliceInfoDef       = class;
  TCondContextDef         = class;
  TWhileContextDef        = class;

  TStringAttrValue = TDictionary<string, TAttrValue>;

  TToggle = (
    DEFAULT = 0,
    ON = 1,
    OFF = 2,
    AGGRESSIVE = 3);

  TDataType = (
    DT_INVALID = 0,
    DT_FLOAT = 1,
    DT_DOUBLE = 2,
    DT_INT32 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_INT8 = 6,
    DT_STRING = 7,
    DT_COMPLEX64 = 8,
    DT_INT64 = 9,
    DT_BOOL = 10,
    DT_QINT8 = 11,
    DT_QUINT8 = 12,
    DT_QINT32 = 13,
    DT_BFLOAT16 = 14,
    DT_QINT16 = 15,
    DT_QUINT16 = 16,
    DT_UINT16 = 17,
    DT_COMPLEX128 = 18,
    DT_HALF = 19,
    DT_RESOURCE = 20,
    DT_VARIANT = 21,
    DT_UINT32 = 22,
    DT_UINT64 = 23,
    DT_FLOAT_REF = 101,
    DT_DOUBLE_REF = 102,
    DT_INT32_REF = 103,
    DT_UINT8_REF = 104,
    DT_INT16_REF = 105,
    DT_INT8_REF = 106,
    DT_STRING_REF = 107,
    DT_COMPLEX64_REF = 108,
    DT_INT64_REF = 109,
    DT_BOOL_REF = 110,
    DT_QINT8_REF = 111,
    DT_QUINT8_REF = 112,
    DT_QINT32_REF = 113,
    DT_BFLOAT16_REF = 114,
    DT_QINT16_REF = 115,
    DT_QUINT16_REF = 116,
    DT_UINT16_REF = 117,
    DT_COMPLEX128_REF = 118,
    DT_HALF_REF = 119,
    DT_RESOURCE_REF = 120,
    DT_VARIANT_REF = 121,
    DT_UINT32_REF = 122,
    DT_UINT64_REF = 123);


	TInt32String     = TDictionary<Integer, string>;
  TUint32Uint32    = TDictionary<UInt32, UInt32>;
  TStringString    = TDictionary<string, string>;
	TStringInt32     = TDictionary<string, Integer>;
	TUint32String    = TDictionary<UInt32, string>;

  {$REGION 'TensorShape'}
  TDim = Class
  const
    ftSize = 1;
    ftName = 2;
  private
    FSize: Int64;
    FName: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Size: Int64 read FSize write FSize;
    property Name: string read FName write FName;
  end;

  TTensorShapeProto = Class
  const
    ftDims = 2;
    ftUnknownRank = 3;
  private
    FDims: TObjectList<TDim>;
    FUnknownRank: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dims: TObjectList<TDim> read FDims;
    property UnknownRank: Boolean read FUnknownRank write FUnknownRank;
  end;
  {$ENDREGION}

  {$REGION 'CostGraph'}
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
  {$ENDREGION}

  {$REGION 'AllocationDescription'}
  TAllocationDescription = Class
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
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property RequestedBytes: Int64 read FRequestedBytes write FRequestedBytes;
    property AllocatedBytes: Int64 read FAllocatedBytes write FAllocatedBytes;
    property AllocatorName: string read FAllocatorName write FAllocatorName;
    property AllocationId: Int64 read FAllocationId write FAllocationId;
    property HasSingleReference: Boolean read FHasSingleReference write FHasSingleReference;
    property Ptr: Int64 read FPtr write FPtr;
  end;
  {$ENDREGION}

  {$REGION 'TensorDescription'}
  TTensorDescription = Class
  const
    ftDtype = 1;
    ftShape = 2;
    ftAllocationDescription = 4;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
    FAllocationDescription: TAllocationDescription;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
    property AllocationDescription: TAllocationDescription read FAllocationDescription write FAllocationDescription;
  end;
  {$ENDREGION}

  {$REGION 'StepStats'}
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
  {$ENDREGION}

  {$REGION 'Cluster'}
  TJobDef = Class
  const
    ftName = 1;
    ftTasks = 2;
  private
    FName: string;
    FTasks: TInt32String;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Tasks: TInt32String read FTasks write FTasks;
  end;

  TClusterDef = Class
  const
    ftJobs = 1;
  private
    FJobs: TList<TJobDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Jobs: TList<TJobDef> read FJobs;
  end;
  {$ENDREGION}

  {$REGION 'CoordinationConfig'}
  TCoordinationServiceConfig = Class
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
    FCoordinatedJobss: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ServiceType: string read FServiceType write FServiceType;
    property ServiceLeader: string read FServiceLeader write FServiceLeader;
    property EnableHealthCheck: Boolean read FEnableHealthCheck write FEnableHealthCheck;
    property ClusterRegisterTimeoutInMs: Int64 read FClusterRegisterTimeoutInMs write FClusterRegisterTimeoutInMs;
    property HeartbeatTimeoutInMs: Int64 read FHeartbeatTimeoutInMs write FHeartbeatTimeoutInMs;
    property CoordinatedJobss: TList<string> read FCoordinatedJobss;
  end;
  {$ENDREGION}

  {$REGION 'Debug'}
  TDebugTensorWatch = Class
  const
    ftNodeName = 1;
    ftOutputSlot = 2;
    ftDebugOpss = 3;
    ftDebugUrlss = 4;
    ftTolerateDebugOpCreationFailures = 5;
  private
    FNodeName: string;
    FOutputSlot: Integer;
    FDebugOpss: TList<string>;
    FDebugUrlss: TList<string>;
    FTolerateDebugOpCreationFailures: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property NodeName: string read FNodeName write FNodeName;
    property OutputSlot: Integer read FOutputSlot write FOutputSlot;
    property DebugOpss: TList<string> read FDebugOpss;
    property DebugUrlss: TList<string> read FDebugUrlss;
    property TolerateDebugOpCreationFailures: Boolean read FTolerateDebugOpCreationFailures write FTolerateDebugOpCreationFailures;
  end;

  TDebugOptions = Class
  const
    ftDebugTensorWatchOptss = 4;
    ftGlobalStep = 10;
    ftResetDiskByteUsage = 11;
  private
    FDebugTensorWatchOptss: TList<TDebugTensorWatch>;
    FGlobalStep: Int64;
    FResetDiskByteUsage: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property DebugTensorWatchOptss: TList<TDebugTensorWatch> read FDebugTensorWatchOptss;
    property GlobalStep: Int64 read FGlobalStep write FGlobalStep;
    property ResetDiskByteUsage: Boolean read FResetDiskByteUsage write FResetDiskByteUsage;
  end;

  TDebuggedSourceFile = Class
  const
    ftHost = 1;
    ftFilePath = 2;
    ftLastModified = 3;
    ftBytes = 4;
    ftLiness = 5;
  private
    FHost: string;
    FFilePath: string;
    FLastModified: Int64;
    FBytes: Int64;
    FLiness: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Host: string read FHost write FHost;
    property FilePath: string read FFilePath write FFilePath;
    property LastModified: Int64 read FLastModified write FLastModified;
    property Bytes: Int64 read FBytes write FBytes;
    property Liness: TList<string> read FLiness;
  end;

  TDebuggedSourceFiles = Class
  const
    ftSourceFiless = 1;
  private
    FSourceFiless: TList<TDebuggedSourceFile>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property SourceFiless: TList<TDebuggedSourceFile> read FSourceFiless;
  end;
  {$ENDREGION}

  {$REGION 'ResourceHandle'}
  TDtypeAndShape = Class
  const
    ftDtype = 1;
    ftShape = 2;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
  end;

  TResourceHandleProto = Class
  const
    ftDevice = 1;
    ftContainer = 2;
    ftName = 3;
    ftHashCode = 4;
    ftMaybeTypeName = 5;
    ftDtypesAndShapess = 6;
  private
    FDevice: string;
    FContainer: string;
    FName: string;
    FHashCode: Int64;
    FMaybeTypeName: string;
    FDtypesAndShapess: TList<TDtypeAndShape>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Device: string read FDevice write FDevice;
    property Container: string read FContainer write FContainer;
    property Name: string read FName write FName;
    property HashCode: Int64 read FHashCode write FHashCode;
    property MaybeTypeName: string read FMaybeTypeName write FMaybeTypeName;
    property DtypesAndShapess: TList<TDtypeAndShape> read FDtypesAndShapess;
  end;
  {$ENDREGION}

  {$REGION 'Tensor'}
  TTensorProto = Class
  const
    ftDtype = 1;
    ftTensorShape = 2;
    ftVersionNumber = 3;
    ftTensorContent = 4;
    ftHalfVals = 13;
    ftFloatVals = 5;
    ftDoubleVals = 6;
    ftIntVals = 7;
    ftStringVals = 8;
    ftScomplexVals = 9;
    ftInt64Vals = 10;
    ftBoolVals = 11;
    ftDcomplexVals = 12;
    ftResourceHandleVals = 14;
    ftVariantVals = 15;
    ftUint32Vals = 16;
    ftUint64Vals = 17;
  private
    FDtype: TDataType;
    FTensorShape: TTensorShapeProto;
    FVersionNumber: Integer;
    FTensorContent: TBytes;
    FHalfVals: TList<Integer>;
    FFloatVals: TList<Single>;
    FDoubleVals: TList<Double>;
    FIntVals: TList<Integer>;
    FStringVals: TList<TBytes>;
    FScomplexVals: TList<Single>;
    FInt64Vals: TList<Int64>;
    FBoolVals: TList<Boolean>;
    FDcomplexVals: TList<Double>;
    FResourceHandleVals: TObjectList<TResourceHandleProto>;
    FVariantVals: TObjectList<TVariantTensorDataProto>;
    FUint32Vals: TList<UInt32>;
    FUint64Vals: TList<Int64>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property TensorShape: TTensorShapeProto read FTensorShape write FTensorShape;
    property VersionNumber: Integer read FVersionNumber write FVersionNumber;
    property TensorContent: TBytes read FTensorContent write FTensorContent;
    property HalfVals: TList<Integer> read FHalfVals;
    property FloatVals: TList<Single> read FFloatVals;
    property DoubleVals: TList<Double> read FDoubleVals;
    property IntVals: TList<Integer> read FIntVals;
    property StringVals: TList<TBytes> read FStringVals;
    property ScomplexVals: TList<Single> read FScomplexVals;
    property Int64Vals: TList<Int64> read FInt64Vals;
    property BoolVals: TList<Boolean> read FBoolVals;
    property DcomplexVals: TList<Double> read FDcomplexVals;
    property ResourceHandleVals: TObjectList<TResourceHandleProto> read FResourceHandleVals;
    property VariantVals: TObjectList<TVariantTensorDataProto> read FVariantVals;
    property Uint32Vals: TList<UInt32> read FUint32Vals;
    property Uint64Vals: TList<Int64> read FUint64Vals;
  end;

  TVariantTensorDataProto = Class
  const
    ftTypeName = 1;
    ftMetadata = 2;
    ftTensorss = 3;
  private
    FTypeName: string;
    FMetadata: TBytes;
    FTensorss: TObjectList<TTensorProto>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TypeName: string read FTypeName write FTypeName;
    property Metadata: TBytes read FMetadata write FMetadata;
    property Tensorss: TObjectList<TTensorProto> read FTensorss;
  end;
  {$ENDREGION}

  {$REGION 'AttrValue'}
  TListValue = Class
  const
    ftSs = 2;
    ftIs = 3;
    ftFs = 4;
    ftBs = 5;
    ftTypes = 6;
    ftShapes = 7;
    ftTensors = 8;
    ftFuncs = 9;
  private
    FSs     : TList<TBytes>;
    FIs     : TList<Int64>;
    FFs     : TList<Single>;
    FBs     : TList<Boolean>;
    FTypes  : TList<TDataType>;
    FShapes : TList<TTensorShapeProto>;
    FTensors: TList<TTensorProto>;
    FFuncs  : TList<TNameAttrList>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ss: TList<TBytes> read FSs;
    property &Is: TList<Int64> read FIs;
    property Fs: TList<Single> read FFs;
    property Bs: TList<Boolean> read FBs;
    property Types: TList<TDataType> read FTypes;
    property Shapes: TList<TTensorShapeProto> read FShapes;
    property Tensors: TList<TTensorProto> read FTensors;
    property Funcs: TList<TNameAttrList> read FFuncs;
  end;

  TAttrValue = Class
  const
    ftS = 2;
    ftI = 3;
    ftF = 4;
    ftB = 5;
    ftType = 6;
    ftShape = 7;
    ftTensor = 8;
    ftList = 1;
    ftFunc = 10;
    ftPlaceholder = 9;
  private
    FValue: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Value: TpbOneof read FValue write FValue;
  end;

  TNameAttrList = Class
  const
    ftName = 1;
    ftAttr = 2;
  private
    FName: string;
    FAttr: TStringAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Attr: TStringAttrValue read FAttr write FAttr;
  end;
  {$ENDREGION}

  {$REGION 'VerifierConfig'}
  TVerifierConfig = Class
  const
    ftVerificationTimeoutInMs = 1;
    ftStructureVerifier = 2;
  private
    FVerificationTimeoutInMs: Int64;
    FStructureVerifier: TToggle;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property VerificationTimeoutInMs: Int64 read FVerificationTimeoutInMs write FVerificationTimeoutInMs;
    property StructureVerifier: TToggle read FStructureVerifier write FStructureVerifier;
  end;
  {$ENDREGION}

  {$REGION 'RewriterConfig'}
  TAutoParallelOptions = Class
  const
    ftEnable = 1;
    ftNumReplicas = 2;
  private
    FEnable: Boolean;
    FNumReplicas: Integer;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Enable: Boolean read FEnable write FEnable;
    property NumReplicas: Integer read FNumReplicas write FNumReplicas;
  end;

  TScopedAllocatorOptions = Class
  const
    ftEnableOps = 1;
  private
    FEnableOps: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property EnableOps: TList<string> read FEnableOps;
  end;

  TCpuLayout = (
    NO_CONVERSION_ON_CPU = 0,
    NCHW_TO_NHWC = 1,
    NHWC_TO_NCHW = 2);

  TNumIterationsType = (
    DEFAULT_NUM_ITERS = 0,
    ONE = 1,
    TWO = 2);

  TMemOptType = (
    DEFAULT_MEM_OPT = 0,
    NO_MEM_OPT = 1,
    MANUAL = 2,
    SWAPPING_HEURISTICS = 4,
    RECOMPUTATION_HEURISTICS = 5,
    SCHEDULING_HEURISTICS = 6,
    HEURISTICS = 3);

  TCustomGraphOptimizer = Class
  const
    ftName = 1;
    ftParameterMap = 2;
  private
    FName: string;
    FParameterMap: TStringAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property ParameterMap: TStringAttrValue read FParameterMap write FParameterMap;
  end;

  TRewriterConfig = Class
  const
    ftCpuLayoutConversion = 50;
    ftLayoutOptimizer = 1;
    ftConstantFolding = 3;
    ftShapeOptimization = 13;
    ftRemapping = 14;
    ftCommonSubgraphElimination = 24;
    ftArithmeticOptimization = 7;
    ftDependencyOptimization = 8;
    ftLoopOptimization = 9;
    ftFunctionOptimization = 10;
    ftDebugStripper = 11;
    ftDisableModelPruning = 2;
    ftScopedAllocatorOptimization = 15;
    ftPinToHostOptimization = 18;
    ftImplementationSelector = 22;
    ftAutoMixedPrecision = 23;
    ftAutoMixedPrecisionMkl = 25;
    ftAutoMixedPrecisionCpu = 29;
    ftDisableMetaOptimizer = 19;
    ftUsePluginOptimizers = 28;
    ftMetaOptimizerIterations = 12;
    ftMinGraphNodes = 17;
    ftExperimentalDisableCompressedTensorOptimization = 26;
    ftExperimentalDisableFoldingQuantizationEmulation = 27;
    ftMemoryOptimization = 4;
    ftMemoryOptimizerTargetNodeNameScope = 6;
    ftMetaOptimizerTimeoutMs = 20;
    ftAutoParallel = 5;
    ftFailOnOptimizerErrors = 21;
    ftScopedAllocatorOpts = 16;
    ftOptimizerss = 100;
    ftCustomOptimizerss = 200;
    ftInterOptimizerVerifierConfig = 300;
    ftPostOptimizationVerifierConfig = 301;
  private
    FCpuLayoutConversion: TCpuLayout;
    FLayoutOptimizer: TToggle;
    FConstantFolding: TToggle;
    FShapeOptimization: TToggle;
    FRemapping: TToggle;
    FCommonSubgraphElimination: TToggle;
    FArithmeticOptimization: TToggle;
    FDependencyOptimization: TToggle;
    FLoopOptimization: TToggle;
    FFunctionOptimization: TToggle;
    FDebugStripper: TToggle;
    FDisableModelPruning: Boolean;
    FScopedAllocatorOptimization: TToggle;
    FPinToHostOptimization: TToggle;
    FImplementationSelector: TToggle;
    FAutoMixedPrecision: TToggle;
    FAutoMixedPrecisionMkl: TToggle;
    FAutoMixedPrecisionCpu: TToggle;
    FDisableMetaOptimizer: Boolean;
    FUsePluginOptimizers: TToggle;
    FMetaOptimizerIterations: TNumIterationsType;
    FMinGraphNodes: Integer;
    FExperimentalDisableCompressedTensorOptimization: Boolean;
    FExperimentalDisableFoldingQuantizationEmulation: Boolean;
    FMemoryOptimization: TMemOptType;
    FMemoryOptimizerTargetNodeNameScope: string;
    FMetaOptimizerTimeoutMs: Int64;
    FAutoParallel: TAutoParallelOptions;
    FFailOnOptimizerErrors: Boolean;
    FScopedAllocatorOpts: TScopedAllocatorOptions;
    FOptimizerss: TList<string>;
    FCustomOptimizerss: TList<TCustomGraphOptimizer>;
    FInterOptimizerVerifierConfig: TVerifierConfig;
    FPostOptimizationVerifierConfig: TVerifierConfig;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property CpuLayoutConversion: TCpuLayout read FCpuLayoutConversion write FCpuLayoutConversion;
    property LayoutOptimizer: TToggle read FLayoutOptimizer write FLayoutOptimizer;
    property ConstantFolding: TToggle read FConstantFolding write FConstantFolding;
    property ShapeOptimization: TToggle read FShapeOptimization write FShapeOptimization;
    property Remapping: TToggle read FRemapping write FRemapping;
    property CommonSubgraphElimination: TToggle read FCommonSubgraphElimination write FCommonSubgraphElimination;
    property ArithmeticOptimization: TToggle read FArithmeticOptimization write FArithmeticOptimization;
    property DependencyOptimization: TToggle read FDependencyOptimization write FDependencyOptimization;
    property LoopOptimization: TToggle read FLoopOptimization write FLoopOptimization;
    property FunctionOptimization: TToggle read FFunctionOptimization write FFunctionOptimization;
    property DebugStripper: TToggle read FDebugStripper write FDebugStripper;
    property DisableModelPruning: Boolean read FDisableModelPruning write FDisableModelPruning;
    property ScopedAllocatorOptimization: TToggle read FScopedAllocatorOptimization write FScopedAllocatorOptimization;
    property PinToHostOptimization: TToggle read FPinToHostOptimization write FPinToHostOptimization;
    property ImplementationSelector: TToggle read FImplementationSelector write FImplementationSelector;
    property AutoMixedPrecision: TToggle read FAutoMixedPrecision write FAutoMixedPrecision;
    property AutoMixedPrecisionMkl: TToggle read FAutoMixedPrecisionMkl write FAutoMixedPrecisionMkl;
    property AutoMixedPrecisionCpu: TToggle read FAutoMixedPrecisionCpu write FAutoMixedPrecisionCpu;
    property DisableMetaOptimizer: Boolean read FDisableMetaOptimizer write FDisableMetaOptimizer;
    property UsePluginOptimizers: TToggle read FUsePluginOptimizers write FUsePluginOptimizers;
    property MetaOptimizerIterations: TNumIterationsType read FMetaOptimizerIterations write FMetaOptimizerIterations;
    property MinGraphNodes: Integer read FMinGraphNodes write FMinGraphNodes;
    property ExperimentalDisableCompressedTensorOptimization: Boolean read FExperimentalDisableCompressedTensorOptimization write FExperimentalDisableCompressedTensorOptimization;
    property ExperimentalDisableFoldingQuantizationEmulation: Boolean read FExperimentalDisableFoldingQuantizationEmulation write FExperimentalDisableFoldingQuantizationEmulation;
    property MemoryOptimization: TMemOptType read FMemoryOptimization write FMemoryOptimization;
    property MemoryOptimizerTargetNodeNameScope: string read FMemoryOptimizerTargetNodeNameScope write FMemoryOptimizerTargetNodeNameScope;
    property MetaOptimizerTimeoutMs: Int64 read FMetaOptimizerTimeoutMs write FMetaOptimizerTimeoutMs;
    property AutoParallel: TAutoParallelOptions read FAutoParallel write FAutoParallel;
    property FailOnOptimizerErrors: Boolean read FFailOnOptimizerErrors write FFailOnOptimizerErrors;
    property ScopedAllocatorOpts: TScopedAllocatorOptions read FScopedAllocatorOpts write FScopedAllocatorOpts;
    property Optimizerss: TList<string> read FOptimizerss;
    property CustomOptimizerss: TList<TCustomGraphOptimizer> read FCustomOptimizerss;
    property InterOptimizerVerifierConfig: TVerifierConfig read FInterOptimizerVerifierConfig write FInterOptimizerVerifierConfig;
    property PostOptimizationVerifierConfig: TVerifierConfig read FPostOptimizationVerifierConfig write FPostOptimizationVerifierConfig;
  end;
  {$ENDREGION}

  {$REGION 'FullType'}
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

  TFullTypeDef = Class
  const
    ftTypeId = 1;
    ftArgss = 2;
    ftS = 3;
    ftI = 4;
  private
    FTypeId: TFullTypeId;
    FArgss: TObjectList<TFullTypeDef>;
    FAttr: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TypeId: TFullTypeId read FTypeId write FTypeId;
    property Argss: TObjectList<TFullTypeDef> read FArgss;
    property Attr: TpbOneof read FAttr write FAttr;
  end;
  {$ENDREGION}

  {$REGION 'NodeDef'}
  TExperimentalDebugInfo = Class
  const
    ftOriginalNodeNamess = 1;
    ftOriginalFuncNamess = 2;
  private
    FOriginalNodeNamess: TList<string>;
    FOriginalFuncNamess: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property OriginalNodeNamess: TList<string> read FOriginalNodeNamess;
    property OriginalFuncNamess: TList<string> read FOriginalFuncNamess;
  end;

  TNodeDef = Class
  const
    ftName = 1;
    ftOp = 2;
    ftInputs = 3;
    ftDevice = 4;
    ftAttr = 5;
    ftExperimentalDebugInfo = 6;
    ftExperimentalType = 7;
  private
    FName: string;
    FOp: string;
    FInputs: TList<string>;
    FDevice: string;
    FAttr: TStringAttrValue;
    FExperimentalDebugInfo: TExperimentalDebugInfo;
    FExperimentalType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Op: string read FOp write FOp;
    property Inputs: TList<string> read FInputs;
    property Device: string read FDevice write FDevice;
    property Attr: TStringAttrValue read FAttr write FAttr;
    property ExperimentalDebugInfo: TExperimentalDebugInfo read FExperimentalDebugInfo write FExperimentalDebugInfo;
    property ExperimentalType: TFullTypeDef read FExperimentalType write FExperimentalType;
  end;
  {$ENDREGION}

  {$REGION 'OpDef'}
  TArgDef = Class
  const
    ftName = 1;
    ftDescription = 2;
    ftType = 3;
    ftTypeAttr = 4;
    ftNumberAttr = 5;
    ftTypeListAttr = 6;
    ftHandleDatas = 7;
    ftIsRef = 16;
    ftExperimentalFullType = 17;
  private
    FName: string;
    FDescription: string;
    FType: TDataType;
    FTypeAttr: string;
    FNumberAttr: string;
    FTypeListAttr: string;
    FHandleDatas: TObjectList<TResourceHandleProto>;
    FIsRef: Boolean;
    FExperimentalFullType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Description: string read FDescription write FDescription;
    property &Type: TDataType read FType write FType;
    property TypeAttr: string read FTypeAttr write FTypeAttr;
    property NumberAttr: string read FNumberAttr write FNumberAttr;
    property TypeListAttr: string read FTypeListAttr write FTypeListAttr;
    property HandleDatas: TObjectList<TResourceHandleProto> read FHandleDatas;
    property IsRef: Boolean read FIsRef write FIsRef;
    property ExperimentalFullType: TFullTypeDef read FExperimentalFullType write FExperimentalFullType;
  end;

  TAttrDef = Class
  const
    ftName = 1;
    ftType = 2;
    ftDefaultValue = 3;
    ftDescription = 4;
    ftHasMinimum = 5;
    ftMinimum = 6;
    ftAllowedValues = 7;
  private
    FName: string;
    FType: string;
    FDefaultValue: TAttrValue;
    FDescription: string;
    FHasMinimum: Boolean;
    FMinimum: Int64;
    FAllowedValues: TAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property &Type: string read FType write FType;
    property DefaultValue: TAttrValue read FDefaultValue write FDefaultValue;
    property Description: string read FDescription write FDescription;
    property HasMinimum: Boolean read FHasMinimum write FHasMinimum;
    property Minimum: Int64 read FMinimum write FMinimum;
    property AllowedValues: TAttrValue read FAllowedValues write FAllowedValues;
  end;

  TOpDef = Class
  const
    ftName = 1;
    ftInputArgs = 2;
    ftOutputArgs = 3;
    ftControlOutputs = 20;
    ftAttrs = 4;
    ftDeprecation = 8;
    ftSummary = 5;
    ftDescription = 6;
    ftIsCommutative = 18;
    ftIsAggregate = 16;
    ftIsStateful = 17;
    ftAllowsUninitializedInput = 19;
    ftIsDistributedCommunication = 21;
  private
    FName: string;
    FInputArgs: TObjectList<TArgDef>;
    FOutputArgs: TObjectList<TArgDef>;
    FControlOutputs: TList<string>;
    FAttrs: TObjectList<TAttrDef>;
    FDeprecation: TOpDeprecation;
    FSummary: string;
    FDescription: string;
    FIsCommutative: Boolean;
    FIsAggregate: Boolean;
    FIsStateful: Boolean;
    FAllowsUninitializedInput: Boolean;
    FIsDistributedCommunication: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property InputArgs: TObjectList<TArgDef> read FInputArgs;
    property OutputArgs: TObjectList<TArgDef> read FOutputArgs;
    property ControlOutputs: TList<string> read FControlOutputs;
    property Attrs: TObjectList<TAttrDef> read FAttrs;
    property Deprecation: TOpDeprecation read FDeprecation write FDeprecation;
    property Summary: string read FSummary write FSummary;
    property Description: string read FDescription write FDescription;
    property IsCommutative: Boolean read FIsCommutative write FIsCommutative;
    property IsAggregate: Boolean read FIsAggregate write FIsAggregate;
    property IsStateful: Boolean read FIsStateful write FIsStateful;
    property AllowsUninitializedInput: Boolean read FAllowsUninitializedInput write FAllowsUninitializedInput;
    property IsDistributedCommunication: Boolean read FIsDistributedCommunication write FIsDistributedCommunication;
  end;

  TOpDeprecation = Class
  const
    ftVersion = 1;
    ftExplanation = 2;
  private
    FVersion: Integer;
    FExplanation: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Version: Integer read FVersion write FVersion;
    property Explanation: string read FExplanation write FExplanation;
  end;

  TOpList = Class
  const
    ftOps = 1;
  private
    FOps: TObjectList<TOpDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ops: TObjectList<TOpDef> read FOps;
  end;
  {$ENDREGION}

  {$REGION 'Function'}
  TFunctionDefLibrary = Class
  const
    ftFunctions = 1;
    ftGradients = 2;
    ftRegisteredGradientss = 3;
  private
    FFunctions: TList<TFunctionDef>;
    FGradients: TList<TGradientDef>;
    FRegisteredGradientss: TList<TRegisteredGradient>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Functions: TList<TFunctionDef> read FFunctions;
    property Gradients: TList<TGradientDef> read FGradients;
    property RegisteredGradientss: TList<TRegisteredGradient> read FRegisteredGradientss;
  end;

  TArgAttrs = Class
  const
    ftAttr = 1;
  private
    FAttr: TStringAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Attr: TStringAttrValue read FAttr write FAttr;
  end;

  TUint32ArgAttrs  = TDictionary<UInt32, TArgAttrs>;

  TFunctionDef = Class
  const
    ftSignature = 1;
    ftAttr = 5;
    ftArgAttr = 7;
    ftResourceArgUniqueId = 8;
    ftNodeDefs = 3;
    ftRet = 4;
    ftControlRet = 6;
  private
    FSignature: TOpDef;
    FAttr: TStringAttrValue;
    FArgAttr: TUint32ArgAttrs;
    FResourceArgUniqueId: TUint32Uint32;
    FNodeDefs: TList<TNodeDef>;
    FRet: TStringString;
    FControlRet: TStringString;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Signature: TOpDef read FSignature write FSignature;
    property Attr: TStringAttrValue read FAttr write FAttr;
    property ArgAttr: TUint32ArgAttrs read FArgAttr write FArgAttr;
    property ResourceArgUniqueId: TUint32Uint32 read FResourceArgUniqueId write FResourceArgUniqueId;
    property NodeDefs: TList<TNodeDef> read FNodeDefs;
    property Ret: TStringString read FRet write FRet;
    property ControlRet: TStringString read FControlRet write FControlRet;
  end;

  TGradientDef = Class
  const
    ftFunctionName = 1;
    ftGradientFunc = 2;
  private
    FFunctionName: string;
    FGradientFunc: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property FunctionName: string read FFunctionName write FFunctionName;
    property GradientFunc: string read FGradientFunc write FGradientFunc;
  end;

  TRegisteredGradient = Class
  const
    ftGradientFunc = 1;
    ftRegisteredOpType = 2;
  private
    FGradientFunc: string;
    FRegisteredOpType: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property GradientFunc: string read FGradientFunc write FGradientFunc;
    property RegisteredOpType: string read FRegisteredOpType write FRegisteredOpType;
  end;
  {$ENDREGION}

  {$REGION 'Versions'}
  TVersionDef = Class
  const
    ftProducer = 1;
    ftMinConsumer = 2;
    ftBadConsumerss = 3;
  private
    FProducer: Integer;
    FMinConsumer: Integer;
    FBadConsumerss: TList<Integer>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Producer: Integer read FProducer write FProducer;
    property MinConsumer: Integer read FMinConsumer write FMinConsumer;
    property BadConsumerss: TList<Integer> read FBadConsumerss;
  end;
  {$ENDREGION}

  {$REGION 'Graph'}
  TGraphDef = Class
  const
    ftNodes = 1;
    ftVersions = 4;
    ftVersion = 3;
    ftLibrary = 2;
  private
    FNodes: TList<TNodeDef>;
    FVersions: TVersionDef;
    FVersion: Integer;
    FLibrary: TFunctionDefLibrary;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Nodes: TList<TNodeDef> read FNodes;
    property Versions: TVersionDef read FVersions write FVersions;
    property Version: Integer read FVersion write FVersion;
    property &Library: TFunctionDefLibrary read FLibrary write FLibrary;
  end;
  {$ENDREGION}

  {$REGION 'Config'}
  TVirtualDevices = Class
  const
    ftMemoryLimitMbs = 1;
    ftPrioritys = 2;
  private
    FMemoryLimitMbs: TList<Single>;
    FPrioritys: TList<Integer>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property MemoryLimitMbs: TList<Single> read FMemoryLimitMbs;
    property Prioritys: TList<Integer> read FPrioritys;
  end;

  TGPUOptions = Class
  const
    ftPerProcessGpuMemoryFraction = 1;
    ftAllowGrowth = 4;
    ftAllocatorType = 2;
    ftDeferredDeletionBytes = 3;
    ftVisibleDeviceList = 5;
    ftPollingActiveDelayUsecs = 6;
    ftPollingInactiveDelayMsecs = 7;
    ftForceGpuCompatible = 8;
    ftExperimental = 9;
  public

    type
      TExperimental = Class
        const
          ftVirtualDevicess = 1;
          ftUseUnifiedMemory = 2;
          ftNumDevToDevCopyStreams = 3;
          ftCollectiveRingOrder = 4;
          ftTimestampedAllocator = 5;
          ftKernelTrackerMaxInterval = 7;
          ftKernelTrackerMaxBytes = 8;
          ftKernelTrackerMaxPending = 9;
          ftInternalFragmentationFraction = 10;
          ftUseCudaMallocAsync = 11;
        private
          FVirtualDevicess: TList<TVirtualDevices>;
          FUseUnifiedMemory: Boolean;
          FNumDevToDevCopyStreams: Integer;
          FCollectiveRingOrder: string;
          FTimestampedAllocator: Boolean;
          FKernelTrackerMaxInterval: Integer;
          FKernelTrackerMaxBytes: Integer;
          FKernelTrackerMaxPending: Integer;
          FInternalFragmentationFraction: Double;
          FUseCudaMallocAsync: Boolean;
        public
          Constructor Create;
          destructor  Destroy; Override;
          // properties
          property VirtualDevicess: TList<TVirtualDevices> read FVirtualDevicess;
          property UseUnifiedMemory: Boolean read FUseUnifiedMemory write FUseUnifiedMemory;
          property NumDevToDevCopyStreams: Integer read FNumDevToDevCopyStreams write FNumDevToDevCopyStreams;
          property CollectiveRingOrder: string read FCollectiveRingOrder write FCollectiveRingOrder;
          property TimestampedAllocator: Boolean read FTimestampedAllocator write FTimestampedAllocator;
          property KernelTrackerMaxInterval: Integer read FKernelTrackerMaxInterval write FKernelTrackerMaxInterval;
          property KernelTrackerMaxBytes: Integer read FKernelTrackerMaxBytes write FKernelTrackerMaxBytes;
          property KernelTrackerMaxPending: Integer read FKernelTrackerMaxPending write FKernelTrackerMaxPending;
          property InternalFragmentationFraction: Double read FInternalFragmentationFraction write FInternalFragmentationFraction;
          property UseCudaMallocAsync: Boolean read FUseCudaMallocAsync write FUseCudaMallocAsync;
      end;

  private
    FPerProcessGpuMemoryFraction: Double;
    FAllowGrowth: Boolean;
    FAllocatorType: string;
    FDeferredDeletionBytes: Int64;
    FVisibleDeviceList: string;
    FPollingActiveDelayUsecs: Integer;
    FPollingInactiveDelayMsecs: Integer;
    FForceGpuCompatible: Boolean;
    FExperimental: TExperimental;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property PerProcessGpuMemoryFraction: Double read FPerProcessGpuMemoryFraction write FPerProcessGpuMemoryFraction;
    property AllowGrowth: Boolean read FAllowGrowth write FAllowGrowth;
    property AllocatorType: string read FAllocatorType write FAllocatorType;
    property DeferredDeletionBytes: Int64 read FDeferredDeletionBytes write FDeferredDeletionBytes;
    property VisibleDeviceList: string read FVisibleDeviceList write FVisibleDeviceList;
    property PollingActiveDelayUsecs: Integer read FPollingActiveDelayUsecs write FPollingActiveDelayUsecs;
    property PollingInactiveDelayMsecs: Integer read FPollingInactiveDelayMsecs write FPollingInactiveDelayMsecs;
    property ForceGpuCompatible: Boolean read FForceGpuCompatible write FForceGpuCompatible;
    property &Experimental: TExperimental read FExperimental write FExperimental;
  end;

  TLevel = (
    L1 = 0,
    L0 = 1);

  TGlobalJitLevel = (
    GJL_DEFAULT = 0,
    GJL_OFF = 1,
    ON_1 = 1,
    ON_2 = 2);

  TOptimizerOptions = Class
  const
    ftDoCommonSubexpressionElimination = 1;
    ftDoConstantFolding = 2;
    ftMaxFoldedConstantInBytes = 6;
    ftDoFunctionInlining = 4;
    ftOptLevel = 3;
    ftGlobalJitLevel = 5;
    ftCpuGlobalJit = 7;
  private
    FDoCommonSubexpressionElimination: Boolean;
    FDoConstantFolding: Boolean;
    FMaxFoldedConstantInBytes: Int64;
    FDoFunctionInlining: Boolean;
    FOptLevel: TLevel;
    FGlobalJitLevel: TGlobalJitLevel;
    FCpuGlobalJit: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property DoCommonSubexpressionElimination: Boolean read FDoCommonSubexpressionElimination write FDoCommonSubexpressionElimination;
    property DoConstantFolding: Boolean read FDoConstantFolding write FDoConstantFolding;
    property MaxFoldedConstantInBytes: Int64 read FMaxFoldedConstantInBytes write FMaxFoldedConstantInBytes;
    property DoFunctionInlining: Boolean read FDoFunctionInlining write FDoFunctionInlining;
    property OptLevel: TLevel read FOptLevel write FOptLevel;
    property GlobalJitLevel: TGlobalJitLevel read FGlobalJitLevel write FGlobalJitLevel;
    property CpuGlobalJit: Boolean read FCpuGlobalJit write FCpuGlobalJit;
  end;

  TGraphOptions = Class
  const
    ftEnableRecvScheduling = 2;
    ftOptimizerOptions = 3;
    ftBuildCostModel = 4;
    ftBuildCostModelAfter = 9;
    ftInferShapes = 5;
    ftPlacePrunedGraph = 6;
    ftEnableBfloat16Sendrecv = 7;
    ftTimelineStep = 8;
    ftRewriteOptions = 10;
  private
    FEnableRecvScheduling: Boolean;
    FOptimizerOptions: TOptimizerOptions;
    FBuildCostModel: Int64;
    FBuildCostModelAfter: Int64;
    FInferShapes: Boolean;
    FPlacePrunedGraph: Boolean;
    FEnableBfloat16Sendrecv: Boolean;
    FTimelineStep: Integer;
    FRewriteOptions: TRewriterConfig;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property EnableRecvScheduling: Boolean read FEnableRecvScheduling write FEnableRecvScheduling;
    property OptimizerOptions: TOptimizerOptions read FOptimizerOptions write FOptimizerOptions;
    property BuildCostModel: Int64 read FBuildCostModel write FBuildCostModel;
    property BuildCostModelAfter: Int64 read FBuildCostModelAfter write FBuildCostModelAfter;
    property InferShapes: Boolean read FInferShapes write FInferShapes;
    property PlacePrunedGraph: Boolean read FPlacePrunedGraph write FPlacePrunedGraph;
    property EnableBfloat16Sendrecv: Boolean read FEnableBfloat16Sendrecv write FEnableBfloat16Sendrecv;
    property TimelineStep: Integer read FTimelineStep write FTimelineStep;
    property RewriteOptions: TRewriterConfig read FRewriteOptions write FRewriteOptions;
  end;

  TThreadPoolOptionProto = Class
  const
    ftNumThreads = 1;
    ftGlobalName = 2;
  private
    FNumThreads: Integer;
    FGlobalName: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property NumThreads: Integer read FNumThreads write FNumThreads;
    property GlobalName: string read FGlobalName write FGlobalName;
  end;

  TRPCOptions = Class
  const
    ftUseRpcForInprocessMaster = 1;
    ftCompressionAlgorithm = 2;
    ftCompressionLevel = 3;
    ftCacheRpcResponse = 4;
    ftDisableSessionConnectionSharing = 5;
    ftNumChannelsPerTarget = 6;
  private
    FUseRpcForInprocessMaster: Boolean;
    FCompressionAlgorithm: string;
    FCompressionLevel: Integer;
    FCacheRpcResponse: Boolean;
    FDisableSessionConnectionSharing: Boolean;
    FNumChannelsPerTarget: Integer;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property UseRpcForInprocessMaster: Boolean read FUseRpcForInprocessMaster write FUseRpcForInprocessMaster;
    property CompressionAlgorithm: string read FCompressionAlgorithm write FCompressionAlgorithm;
    property CompressionLevel: Integer read FCompressionLevel write FCompressionLevel;
    property CacheRpcResponse: Boolean read FCacheRpcResponse write FCacheRpcResponse;
    property DisableSessionConnectionSharing: Boolean read FDisableSessionConnectionSharing write FDisableSessionConnectionSharing;
    property NumChannelsPerTarget: Integer read FNumChannelsPerTarget write FNumChannelsPerTarget;
  end;

  TSessionMetadata = Class
  const
    ftName = 1;
    ftVersion = 2;
  private
    FName: string;
    FVersion: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Version: Int64 read FVersion write FVersion;
  end;

  TMlirBridgeRollout = (
    MLIR_BRIDGE_ROLLOUT_UNSPECIFIED = 0,
    MLIR_BRIDGE_ROLLOUT_ENABLED = 1,
    MLIR_BRIDGE_ROLLOUT_DISABLED = 2,
    MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED = 3,
    MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED = 4);

  TConfigProto = Class
  const
    ftDeviceCount = 1;
    ftIntraOpParallelismThreads = 2;
    ftInterOpParallelismThreads = 5;
    ftUsePerSessionThreads = 9;
    ftSessionInterOpThreadPools = 12;
    ftPlacementPeriod = 3;
    ftDeviceFilterss = 4;
    ftGpuOptions = 6;
    ftAllowSoftPlacement = 7;
    ftLogDevicePlacement = 8;
    ftGraphOptions = 10;
    ftOperationTimeoutInMs = 11;
    ftRpcOptions = 13;
    ftClusterDef = 14;
    ftIsolateSessionState = 15;
    ftShareClusterDevicesInSession = 17;
    ftExperimental = 16;
  public
    type
      TExperimental = Class
        const
          ftCollectiveGroupLeader = 1;
          ftExecutorType = 3;
          ftRecvBufMaxChunk = 4;
          ftUseNumaAffinity = 5;
          ftCollectiveDeterministicSequentialExecution = 6;
          ftCollectiveNccl = 7;
          ftShareSessionStateInClusterspecPropagation = 8;
          ftDisableThreadSpinning = 9;
          ftShareClusterDevicesInSession = 10;
          ftSessionMetadata = 11;
          ftOptimizeForStaticGraph = 12;
          ftEnableMlirBridge = 13;
          ftMlirBridgeRollout = 17;
          ftEnableMlirGraphOptimization = 16;
          ftDisableOutputPartitionGraphs = 14;
          ftXlaFusionAutotunerThresh = 15;
          ftUseTfrt = 18;
          ftDisableFunctionalOpsLowering = 21;
          ftXlaPreferSingleGraphCluster = 22;
          ftCoordinationConfig = 23;
        private
          FCollectiveGroupLeader: string;
          FExecutorType: string;
          FRecvBufMaxChunk: Integer;
          FUseNumaAffinity: Boolean;
          FCollectiveDeterministicSequentialExecution: Boolean;
          FCollectiveNccl: Boolean;
          FShareSessionStateInClusterspecPropagation: Boolean;
          FDisableThreadSpinning: Boolean;
          FShareClusterDevicesInSession: Boolean;
          FSessionMetadata: TSessionMetadata;
          FOptimizeForStaticGraph: Boolean;
          FEnableMlirBridge: Boolean;
          FMlirBridgeRollout: TMlirBridgeRollout;
          FEnableMlirGraphOptimization: Boolean;
          FDisableOutputPartitionGraphs: Boolean;
          FXlaFusionAutotunerThresh: Int64;
          FUseTfrt: Boolean;
          FDisableFunctionalOpsLowering: Boolean;
          FXlaPreferSingleGraphCluster: Boolean;
          FCoordinationConfig: TCoordinationServiceConfig;
        public
          Constructor Create;
          destructor  Destroy; Override;
          // properties
          property CollectiveGroupLeader: string read FCollectiveGroupLeader write FCollectiveGroupLeader;
          property ExecutorType: string read FExecutorType write FExecutorType;
          property RecvBufMaxChunk: Integer read FRecvBufMaxChunk write FRecvBufMaxChunk;
          property UseNumaAffinity: Boolean read FUseNumaAffinity write FUseNumaAffinity;
          property CollectiveDeterministicSequentialExecution: Boolean read FCollectiveDeterministicSequentialExecution write FCollectiveDeterministicSequentialExecution;
          property CollectiveNccl: Boolean read FCollectiveNccl write FCollectiveNccl;
          property ShareSessionStateInClusterspecPropagation: Boolean read FShareSessionStateInClusterspecPropagation write FShareSessionStateInClusterspecPropagation;
          property DisableThreadSpinning: Boolean read FDisableThreadSpinning write FDisableThreadSpinning;
          property ShareClusterDevicesInSession: Boolean read FShareClusterDevicesInSession write FShareClusterDevicesInSession;
          property SessionMetadata: TSessionMetadata read FSessionMetadata write FSessionMetadata;
          property OptimizeForStaticGraph: Boolean read FOptimizeForStaticGraph write FOptimizeForStaticGraph;
          property EnableMlirBridge: Boolean read FEnableMlirBridge write FEnableMlirBridge;
          property MlirBridgeRollout: TMlirBridgeRollout read FMlirBridgeRollout write FMlirBridgeRollout;
          property EnableMlirGraphOptimization: Boolean read FEnableMlirGraphOptimization write FEnableMlirGraphOptimization;
          property DisableOutputPartitionGraphs: Boolean read FDisableOutputPartitionGraphs write FDisableOutputPartitionGraphs;
          property XlaFusionAutotunerThresh: Int64 read FXlaFusionAutotunerThresh write FXlaFusionAutotunerThresh;
          property UseTfrt: Boolean read FUseTfrt write FUseTfrt;
          property DisableFunctionalOpsLowering: Boolean read FDisableFunctionalOpsLowering write FDisableFunctionalOpsLowering;
          property XlaPreferSingleGraphCluster: Boolean read FXlaPreferSingleGraphCluster write FXlaPreferSingleGraphCluster;
          property CoordinationConfig: TCoordinationServiceConfig read FCoordinationConfig write FCoordinationConfig;
      end;
  private
    FDeviceCount: TStringInt32;
    FIntraOpParallelismThreads: Integer;
    FInterOpParallelismThreads: Integer;
    FUsePerSessionThreads: Boolean;
    FSessionInterOpThreadPools: TList<TThreadPoolOptionProto>;
    FPlacementPeriod: Integer;
    FDeviceFilterss: TList<string>;
    FGpuOptions: TGPUOptions;
    FAllowSoftPlacement: Boolean;
    FLogDevicePlacement: Boolean;
    FGraphOptions: TGraphOptions;
    FOperationTimeoutInMs: Int64;
    FRpcOptions: TRPCOptions;
    FClusterDef: TClusterDef;
    FIsolateSessionState: Boolean;
    FShareClusterDevicesInSession: Boolean;
    FExperimental: TExperimental;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property DeviceCount: TStringInt32 read FDeviceCount write FDeviceCount;
    property IntraOpParallelismThreads: Integer read FIntraOpParallelismThreads write FIntraOpParallelismThreads;
    property InterOpParallelismThreads: Integer read FInterOpParallelismThreads write FInterOpParallelismThreads;
    property UsePerSessionThreads: Boolean read FUsePerSessionThreads write FUsePerSessionThreads;
    property SessionInterOpThreadPools: TList<TThreadPoolOptionProto> read FSessionInterOpThreadPools;
    property PlacementPeriod: Integer read FPlacementPeriod write FPlacementPeriod;
    property DeviceFilterss: TList<string> read FDeviceFilterss;
    property GpuOptions: TGPUOptions read FGpuOptions write FGpuOptions;
    property AllowSoftPlacement: Boolean read FAllowSoftPlacement write FAllowSoftPlacement;
    property LogDevicePlacement: Boolean read FLogDevicePlacement write FLogDevicePlacement;
    property GraphOptions: TGraphOptions read FGraphOptions write FGraphOptions;
    property OperationTimeoutInMs: Int64 read FOperationTimeoutInMs write FOperationTimeoutInMs;
    property RpcOptions: TRPCOptions read FRpcOptions write FRpcOptions;
    property ClusterDef: TClusterDef read FClusterDef write FClusterDef;
    property IsolateSessionState: Boolean read FIsolateSessionState write FIsolateSessionState;
    property ShareClusterDevicesInSession: Boolean read FShareClusterDevicesInSession write FShareClusterDevicesInSession;
    property &Experimental: TExperimental read FExperimental write FExperimental;
  end;

  TTraceLevel = (
    NO_TRACE = 0,
    SOFTWARE_TRACE = 1,
    HARDWARE_TRACE = 2,
    FULL_TRACE = 3);

  TRunHandlerPoolOptions = Class
  const
    ftPriority = 1;
  private
    FPriority: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Priority: Int64 read FPriority write FPriority;
  end;

  TRunOptions = Class
  const
    ftTraceLevel = 1;
    ftTimeoutInMs = 2;
    ftInterOpThreadPool = 3;
    ftOutputPartitionGraphs = 5;
    ftDebugOptions = 6;
    ftReportTensorAllocationsUponOom = 7;
    ftExperimental = 8;
  public
    type
      TExperimental = Class
        const
          ftCollectiveGraphKey = 1;
          ftUseRunHandlerPool = 2;
          ftRunHandlerPoolOptions = 3;
        private
          FCollectiveGraphKey: Int64;
          FUseRunHandlerPool: Boolean;
          FRunHandlerPoolOptions: TRunHandlerPoolOptions;
        public
          Constructor Create;
          destructor  Destroy; Override;
          // properties
          property CollectiveGraphKey: Int64 read FCollectiveGraphKey write FCollectiveGraphKey;
          property UseRunHandlerPool: Boolean read FUseRunHandlerPool write FUseRunHandlerPool;
          property RunHandlerPoolOptions: TRunHandlerPoolOptions read FRunHandlerPoolOptions write FRunHandlerPoolOptions;
      end;
  private
    FTraceLevel: TTraceLevel;
    FTimeoutInMs: Int64;
    FInterOpThreadPool: Integer;
    FOutputPartitionGraphs: Boolean;
    FDebugOptions: TDebugOptions;
    FReportTensorAllocationsUponOom: Boolean;
    FExperimental: TExperimental;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TraceLevel: TTraceLevel read FTraceLevel write FTraceLevel;
    property TimeoutInMs: Int64 read FTimeoutInMs write FTimeoutInMs;
    property InterOpThreadPool: Integer read FInterOpThreadPool write FInterOpThreadPool;
    property OutputPartitionGraphs: Boolean read FOutputPartitionGraphs write FOutputPartitionGraphs;
    property DebugOptions: TDebugOptions read FDebugOptions write FDebugOptions;
    property ReportTensorAllocationsUponOom: Boolean read FReportTensorAllocationsUponOom write FReportTensorAllocationsUponOom;
    property &Experimental: TExperimental read FExperimental write FExperimental;
  end;

  TFunctionGraphs = Class
  const
    ftPartitionGraphss = 1;
    ftPreOptimizationGraph = 2;
    ftPostOptimizationGraph = 3;
  private
    FPartitionGraphss: TList<TGraphDef>;
    FPreOptimizationGraph: TGraphDef;
    FPostOptimizationGraph: TGraphDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property PartitionGraphss: TList<TGraphDef> read FPartitionGraphss;
    property PreOptimizationGraph: TGraphDef read FPreOptimizationGraph write FPreOptimizationGraph;
    property PostOptimizationGraph: TGraphDef read FPostOptimizationGraph write FPostOptimizationGraph;
  end;

  TRunMetadata = Class
  const
    ftStepStats = 1;
    ftCostGraph = 2;
    ftPartitionGraphss = 3;
    ftFunctionGraphss = 4;
  private
    FStepStats: TStepStats;
    FCostGraph: TCostGraphDef;
    FPartitionGraphss: TList<TGraphDef>;
    FFunctionGraphss: TList<TFunctionGraphs>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property StepStats: TStepStats read FStepStats write FStepStats;
    property CostGraph: TCostGraphDef read FCostGraph write FCostGraph;
    property PartitionGraphss: TList<TGraphDef> read FPartitionGraphss;
    property FunctionGraphss: TList<TFunctionGraphs> read FFunctionGraphss;
  end;

  TTensorConnection = Class
  const
    ftFromTensor = 1;
    ftToTensor = 2;
  private
    FFromTensor: string;
    FToTensor: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property FromTensor: string read FFromTensor write FFromTensor;
    property ToTensor: string read FToTensor write FToTensor;
  end;

  TCallableOptions = Class
  const
    ftFeeds = 1;
    ftFetchs = 2;
    ftTargets = 3;
    ftRunOptions = 4;
    ftTensorConnections = 5;
    ftFeedDevices = 6;
    ftFetchDevices = 7;
    ftFetchSkipSync = 8;
  private
    FFeeds: TList<string>;
    FFetchs: TList<string>;
    FTargets: TList<string>;
    FRunOptions: TRunOptions;
    FTensorConnections: TList<TTensorConnection>;
    FFeedDevices: TStringString;
    FFetchDevices: TStringString;
    FFetchSkipSync: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Feeds: TList<string> read FFeeds;
    property Fetchs: TList<string> read FFetchs;
    property Targets: TList<string> read FTargets;
    property RunOptions: TRunOptions read FRunOptions write FRunOptions;
    property TensorConnections: TList<TTensorConnection> read FTensorConnections;
    property FeedDevices: TStringString read FFeedDevices write FFeedDevices;
    property FetchDevices: TStringString read FFetchDevices write FFetchDevices;
    property FetchSkipSync: Boolean read FFetchSkipSync write FFetchSkipSync;
  end;
  {$ENDREGION}

  {$REGION 'Variable'}
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

  TVariableDef = Class
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
    Constructor Create;
    destructor  Destroy; Override;
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
    FFullShapes: TList<Int64>;
    FVarOffsets: TList<Int64>;
    FVarShapes: TList<Int64>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property FullName: string read FFullName write FFullName;
    property FullShapes: TList<Int64> read FFullShapes;
    property VarOffsets: TList<Int64> read FVarOffsets;
    property VarShapes: TList<Int64> read FVarShapes;
  end;
  {$ENDREGION}

  {$REGION 'CppShapeInference'}
  THandleShapeAndType = Class
  const
    ftShape = 1;
    ftDtype = 2;
    ftType = 4;
  private
    FShape: TTensorShapeProto;
    FDtype: TDataType;
    FType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property Dtype: TDataType read FDtype write FDtype;
    property &Type: TFullTypeDef read FType write FType;
  end;

  THandleData = Class
  const
    ftIsSet = 1;
    ftShapeAndTypes = 2;
  private
    FIsSet: Boolean;
    FShapeAndTypes: TList<THandleShapeAndType>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property IsSet: Boolean read FIsSet write FIsSet;
    property ShapeAndTypes: TList<THandleShapeAndType> read FShapeAndTypes;
  end;

  TCppShapeInferenceResult = Class
  const
    ftShape = 1;
    ftHandleData = 4;
  private
    FShape: TTensorShapeProto;
    FHandleData: THandleData;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property HandleData: THandleData read FHandleData write FHandleData;
  end;

  TCppShapeInferenceInputsNeeded = Class
  const
    ftInputTensorsNeededs = 1;
    ftInputTensorsAsShapesNeededs = 2;
  private
    FInputTensorsNeededs: TList<Integer>;
    FInputTensorsAsShapesNeededs: TList<Integer>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property InputTensorsNeededs: TList<Integer> read FInputTensorsNeededs;
    property InputTensorsAsShapesNeededs: TList<Integer> read FInputTensorsAsShapesNeededs;
  end;
  {$ENDREGION}

  {$REGION 'ControlFlow'}
  TValuesDef = Class
  const
    ftValuess = 1;
    ftExternalValues = 2;
  private
    FValuess: TList<string>;
    FExternalValues: TStringString;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Valuess: TList<string> read FValuess;
    property ExternalValues: TStringString read FExternalValues write FExternalValues;
  end;

  TControlFlowContextDef = Class
  const
    ftCondCtxt = 1;
    ftWhileCtxt = 2;
  private
    FCtxt: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ctxt: TpbOneof read FCtxt write FCtxt;
  end;

  TCondContextDef = Class
  const
    ftContextName = 1;
    ftPredName = 2;
    ftPivotName = 3;
    ftBranch = 4;
    ftValuesDef = 5;
    ftNestedContextss = 6;
  private
    FContextName: string;
    FPredName: string;
    FPivotName: string;
    FBranch: Integer;
    FValuesDef: TValuesDef;
    FNestedContextss: TList<TControlFlowContextDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ContextName: string read FContextName write FContextName;
    property PredName: string read FPredName write FPredName;
    property PivotName: string read FPivotName write FPivotName;
    property Branch: Integer read FBranch write FBranch;
    property ValuesDef: TValuesDef read FValuesDef write FValuesDef;
    property NestedContextss: TList<TControlFlowContextDef> read FNestedContextss;
  end;

  TWhileContextDef = Class
  const
    ftContextName = 1;
    ftParallelIterations = 2;
    ftBackProp = 3;
    ftSwapMemory = 4;
    ftPivotName = 5;
    ftPivotForPredName = 6;
    ftPivotForBodyName = 7;
    ftLoopExitNamess = 8;
    ftLoopEnterNamess = 10;
    ftValuesDef = 9;
    ftMaximumIterationsName = 11;
    ftNestedContextss = 12;
  private
    FContextName: string;
    FParallelIterations: Integer;
    FBackProp: Boolean;
    FSwapMemory: Boolean;
    FPivotName: string;
    FPivotForPredName: string;
    FPivotForBodyName: string;
    FLoopExitNamess: TList<string>;
    FLoopEnterNamess: TList<string>;
    FValuesDef: TValuesDef;
    FMaximumIterationsName: string;
    FNestedContextss: TList<TControlFlowContextDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ContextName: string read FContextName write FContextName;
    property ParallelIterations: Integer read FParallelIterations write FParallelIterations;
    property BackProp: Boolean read FBackProp write FBackProp;
    property SwapMemory: Boolean read FSwapMemory write FSwapMemory;
    property PivotName: string read FPivotName write FPivotName;
    property PivotForPredName: string read FPivotForPredName write FPivotForPredName;
    property PivotForBodyName: string read FPivotForBodyName write FPivotForBodyName;
    property LoopExitNamess: TList<string> read FLoopExitNamess;
    property LoopEnterNamess: TList<string> read FLoopEnterNamess;
    property ValuesDef: TValuesDef read FValuesDef write FValuesDef;
    property MaximumIterationsName: string read FMaximumIterationsName write FMaximumIterationsName;
    property NestedContextss: TList<TControlFlowContextDef> read FNestedContextss;
  end;
  {$ENDREGION}

implementation

{$REGION 'TensorShape'}
{ TDim }

Constructor TDim.Create;
begin
  inherited Create;
  FSize := 0;
  FName:= '' ;
end;

destructor TDim.Destroy;
begin
  inherited Destroy;
end;

{ TTensorShapeProto }

Constructor TTensorShapeProto.Create;
begin
  inherited Create;

  FDims := TObjectList<TDim>.Create;
end;

destructor TTensorShapeProto.Destroy;
begin
  FDims.Clear;
  FDims.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'CostGraph'}
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
{$ENDREGION}

{$REGION 'AllocationDescription'}
{ TAllocationDescription }

Constructor TAllocationDescription.Create;
begin
  inherited Create;
end;

destructor TAllocationDescription.Destroy;
begin
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'TensorDescription'}
{ TTensorDescription }

Constructor TTensorDescription.Create;
begin
  inherited Create;
end;

destructor TTensorDescription.Destroy;
begin
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'StepStats'}
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
{$ENDREGION}

{$REGION 'Cluster'}
{ TJobDef }

Constructor TJobDef.Create;
begin
  inherited Create;
  FTasks := TDictionary<Integer, string>.Create;
end;

destructor TJobDef.Destroy;
begin
  FTasks.Free;
  inherited Destroy;
end;

{ TClusterDef }

Constructor TClusterDef.Create;
begin
  inherited Create;

  FJobs := TList<TJobDef>.Create;
end;

destructor TClusterDef.Destroy;
begin
  FJobs.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'CoordinationConfig'}
{ TCoordinationServiceConfig }

Constructor TCoordinationServiceConfig.Create;
begin
  inherited Create;

  FCoordinatedJobss := TList<string>.Create;
end;

destructor TCoordinationServiceConfig.Destroy;
begin
  FCoordinatedJobss.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Debug'}
{ TDebugTensorWatch }

Constructor TDebugTensorWatch.Create;
begin
  inherited Create;

  FDebugOpss := TList<string>.Create;

  FDebugUrlss := TList<string>.Create;
end;

destructor TDebugTensorWatch.Destroy;
begin
  FDebugOpss.Free;
  FDebugUrlss.Free;
  inherited Destroy;
end;

{ TDebugOptions }

Constructor TDebugOptions.Create;
begin
  inherited Create;

  FDebugTensorWatchOptss := TList<TDebugTensorWatch>.Create;
end;

destructor TDebugOptions.Destroy;
begin
  FDebugTensorWatchOptss.Free;
  inherited Destroy;
end;

{ TDebuggedSourceFile }

Constructor TDebuggedSourceFile.Create;
begin
  inherited Create;

  FLiness := TList<string>.Create;
end;

destructor TDebuggedSourceFile.Destroy;
begin
  FLiness.Free;
  inherited Destroy;
end;

{ TDebuggedSourceFiles }

Constructor TDebuggedSourceFiles.Create;
begin
  inherited Create;

  FSourceFiless := TList<TDebuggedSourceFile>.Create;
end;

destructor TDebuggedSourceFiles.Destroy;
begin
  FSourceFiless.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'ResourceHandle'}
{ TDtypeAndShape }

Constructor TDtypeAndShape.Create;
begin
  inherited Create;
end;

destructor TDtypeAndShape.Destroy;
begin
  inherited Destroy;
end;

{ TResourceHandleProto }

Constructor TResourceHandleProto.Create;
begin
  inherited Create;

  FDtypesAndShapess := TList<TDtypeAndShape>.Create;
end;

destructor TResourceHandleProto.Destroy;
begin
  FDtypesAndShapess.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Tensor'}
{ TTensorProto }

Constructor TTensorProto.Create;
begin
  inherited Create;

  FHalfVals     := TList<Integer>.Create;
  FFloatVals    := TList<Single>.Create;
  FDoubleVals   := TList<Double>.Create;
  FIntVals      := TList<Integer>.Create;
  FStringVals   := TList<TBytes>.Create;
  FScomplexVals := TList<Single>.Create;
  FInt64Vals    := TList<Int64>.Create;
  FBoolVals     := TList<Boolean>.Create;
  FDcomplexVals := TList<Double>.Create;

  FResourceHandleVals := TObjectList<TResourceHandleProto>.Create;
  FVariantVals        := TObjectList<TVariantTensorDataProto>.Create;

  FUint32Vals := TList<UInt32>.Create;
  FUint64Vals := TList<Int64>.Create;
end;

destructor TTensorProto.Destroy;
begin
  FHalfVals.Free;
  FFloatVals.Free;
  FDoubleVals.Free;
  FIntVals.Free;
  FStringVals.Free;
  FScomplexVals.Free;
  FInt64Vals.Free;
  FBoolVals.Free;
  FDcomplexVals.Free;
  FResourceHandleVals.Free;
  FVariantVals.Free;
  FUint32Vals.Free;
  FUint64Vals.Free;
  inherited Destroy;
end;

{ TVariantTensorDataProto }

Constructor TVariantTensorDataProto.Create;
begin
  inherited Create;

  FTensorss := TObjectList<TTensorProto>.Create;
end;

destructor TVariantTensorDataProto.Destroy;
begin
  FTensorss.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'AttrValue'}
{ TListValue }

Constructor TListValue.Create;
begin
  inherited Create;

  FSs      := TList<TBytes>.Create;
  FIs      := TList<Int64>.Create;
  FFs      := TList<Single>.Create;
  FBs      := TList<Boolean>.Create;
  FTypes   := TList<TDataType>.Create;
  FShapes  := TList<TTensorShapeProto>.Create;
  FTensors := TList<TTensorProto>.Create;
  FFuncs   := TList<TNameAttrList>.Create;
end;

destructor TListValue.Destroy;
begin
  FSs.Free;
  FIs.Free;
  FFs.Free;
  FBs.Free;
  FTypes.Free;
  FShapes.Free;
  FTensors.Free;
  FFuncs.Free;
  inherited Destroy;
end;

{ TAttrValue }

Constructor TAttrValue.Create;
begin
  inherited Create;
end;

destructor TAttrValue.Destroy;
begin
  inherited Destroy;
end;

{ TNameAttrList }

Constructor TNameAttrList.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TNameAttrList.Destroy;
begin
  FAttr.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'VerifierConfig'}
{ TVerifierConfig }

Constructor TVerifierConfig.Create;
begin
  inherited Create;
end;

destructor TVerifierConfig.Destroy;
begin
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'RewriterConfig'}
{ TAutoParallelOptions }

Constructor TAutoParallelOptions.Create;
begin
  inherited Create;
end;

destructor TAutoParallelOptions.Destroy;
begin
  inherited Destroy;
end;

{ TScopedAllocatorOptions }

Constructor TScopedAllocatorOptions.Create;
begin
  inherited Create;

  FEnableOps := TList<string>.Create;
end;

destructor TScopedAllocatorOptions.Destroy;
begin
  FEnableOps.Free;
  inherited Destroy;
end;

{ TCustomGraphOptimizer }

Constructor TCustomGraphOptimizer.Create;
begin
  inherited Create;
  FParameterMap := TDictionary<string, TAttrValue>.Create;
end;

destructor TCustomGraphOptimizer.Destroy;
begin
  FParameterMap.Free;
  inherited Destroy;
end;

{ TRewriterConfig }

Constructor TRewriterConfig.Create;
begin
  inherited Create;

  FOptimizerss := TList<string>.Create;

  FCustomOptimizerss := TList<TCustomGraphOptimizer>.Create;
end;

destructor TRewriterConfig.Destroy;
begin
  FOptimizerss.Free;
  FCustomOptimizerss.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'FullType'}
{ TFullTypeDef }

Constructor TFullTypeDef.Create;
begin
  inherited Create;

  FArgss := TObjectList<TFullTypeDef>.Create;
end;

destructor TFullTypeDef.Destroy;
begin
  FArgss.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'NodeDef'}
{ TExperimentalDebugInfo }

Constructor TExperimentalDebugInfo.Create;
begin
  inherited Create;

  FOriginalNodeNamess := TList<string>.Create;

  FOriginalFuncNamess := TList<string>.Create;
end;

destructor TExperimentalDebugInfo.Destroy;
begin
  FOriginalNodeNamess.Free;
  FOriginalFuncNamess.Free;
  inherited Destroy;
end;

{ TNodeDef }

Constructor TNodeDef.Create;
begin
  inherited Create;

  FInputs := TList<string>.Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TNodeDef.Destroy;
begin
  FInputs.Free;
  FAttr.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'OpDef'}
{ TArgDef }

Constructor TArgDef.Create;
begin
  inherited Create;

  FHandleDatas := TObjectList<TResourceHandleProto>.Create;
end;

destructor TArgDef.Destroy;
begin
  FHandleDatas.Clear;
  FHandleDatas.Free;
  inherited Destroy;
end;

{ TAttrDef }

Constructor TAttrDef.Create;
begin
  inherited Create;
end;

destructor TAttrDef.Destroy;
begin
  inherited Destroy;
end;

{ TOpDef }

Constructor TOpDef.Create;
begin
  inherited Create;

  FInputArgs      := TObjectList<TArgDef>.Create;
  FOutputArgs     := TObjectList<TArgDef>.Create;
  FControlOutputs := TList<string>.Create;
  FAttrs          := TObjectList<TAttrDef>.Create;
end;

destructor TOpDef.Destroy;
begin
  FInputArgs.Clear;
  FInputArgs.Free;
  FOutputArgs.Clear;
  FOutputArgs.Free;
  FControlOutputs.Clear;
  FControlOutputs.Free;
  FAttrs.Clear;
  FAttrs.Free;
  inherited Destroy;
end;

{ TOpDeprecation }

Constructor TOpDeprecation.Create;
begin
  inherited Create;
end;

destructor TOpDeprecation.Destroy;
begin
  inherited Destroy;
end;

{ TOpList }

Constructor TOpList.Create;
begin
  inherited Create;

  FOps := TObjectList<TOpDef>.Create;
end;

destructor TOpList.Destroy;
begin
  FOps.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Function'}
{ TFunctionDefLibrary }

Constructor TFunctionDefLibrary.Create;
begin
  inherited Create;

  FFunctions := TList<TFunctionDef>.Create;

  FGradients := TList<TGradientDef>.Create;

  FRegisteredGradientss := TList<TRegisteredGradient>.Create;
end;

destructor TFunctionDefLibrary.Destroy;
begin
  FFunctions.Free;
  FGradients.Free;
  FRegisteredGradientss.Free;
  inherited Destroy;
end;

{ TArgAttrs }

Constructor TArgAttrs.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TArgAttrs.Destroy;
begin
  FAttr.Free;
  inherited Destroy;
end;

{ TFunctionDef }

Constructor TFunctionDef.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
  FArgAttr := TDictionary<UInt32, TArgAttrs>.Create;
  FResourceArgUniqueId := TDictionary<UInt32, UInt32>.Create;

  FNodeDefs := TList<TNodeDef>.Create;
  FRet := TDictionary<string, string>.Create;
  FControlRet := TDictionary<string, string>.Create;
end;

destructor TFunctionDef.Destroy;
begin
  FAttr.Free;
  FArgAttr.Free;
  FResourceArgUniqueId.Free;
  FNodeDefs.Free;
  FRet.Free;
  FControlRet.Free;
  inherited Destroy;
end;

{ TGradientDef }

Constructor TGradientDef.Create;
begin
  inherited Create;
end;

destructor TGradientDef.Destroy;
begin
  inherited Destroy;
end;

{ TRegisteredGradient }

Constructor TRegisteredGradient.Create;
begin
  inherited Create;
end;

destructor TRegisteredGradient.Destroy;
begin
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Versions'}
{ TVersionDef }

Constructor TVersionDef.Create;
begin
  inherited Create;

  FBadConsumerss := TList<Integer>.Create;
end;

destructor TVersionDef.Destroy;
begin
  FBadConsumerss.Free;
  inherited Destroy;
end;

{$ENDREGION}

{$REGION 'Graph'}
{ TGraphDef }

Constructor TGraphDef.Create;
begin
  inherited Create;

  FNodes := TList<TNodeDef>.Create;
end;

destructor TGraphDef.Destroy;
begin
  FNodes.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Config'}
{ TVirtualDevices }

Constructor TVirtualDevices.Create;
begin
  inherited Create;

  FMemoryLimitMbs := TList<Single>.Create;

  FPrioritys := TList<Integer>.Create;
end;

destructor TVirtualDevices.Destroy;
begin
  FMemoryLimitMbs.Free;
  FPrioritys.Free;
  inherited Destroy;
end;


{ TGPUOptions }

Constructor TGPUOptions.Create;
begin
  inherited Create;
  FExperimental := TExperimental.Create;
end;

destructor TGPUOptions.Destroy;
begin
  inherited Destroy;
  FExperimental.Free;
end;

{ TOptimizerOptions }

Constructor TOptimizerOptions.Create;
begin
  inherited Create;
end;

destructor TOptimizerOptions.Destroy;
begin
  inherited Destroy;
end;

{ TGraphOptions }

Constructor TGraphOptions.Create;
begin
  inherited Create;
end;

destructor TGraphOptions.Destroy;
begin
  inherited Destroy;
end;

{ TThreadPoolOptionProto }

Constructor TThreadPoolOptionProto.Create;
begin
  inherited Create;
end;

destructor TThreadPoolOptionProto.Destroy;
begin
  inherited Destroy;
end;

{ TRPCOptions }

Constructor TRPCOptions.Create;
begin
  inherited Create;
end;

destructor TRPCOptions.Destroy;
begin
  inherited Destroy;
end;

{ TSessionMetadata }

Constructor TSessionMetadata.Create;
begin
  inherited Create;
end;

destructor TSessionMetadata.Destroy;
begin
  inherited Destroy;
end;

{ TConfigProto }

Constructor TConfigProto.Create;
begin
  inherited Create;
  FDeviceCount               := TDictionary<string, Integer>.Create;
  FSessionInterOpThreadPools := TList<TThreadPoolOptionProto>.Create;
  FDeviceFilterss            := TList<string>.Create;
  FExperimental              := TExperimental.Create;
end;

destructor TConfigProto.Destroy;
begin
  FDeviceCount.Free;
  FSessionInterOpThreadPools.Free;
  FDeviceFilterss.Free;
  FExperimental.Free;
  inherited Destroy;
end;

{ TRunHandlerPoolOptions }

Constructor TRunHandlerPoolOptions.Create;
begin
  inherited Create;
end;

destructor TRunHandlerPoolOptions.Destroy;
begin
  inherited Destroy;
end;

{ TRunOptions }

Constructor TRunOptions.Create;
begin
  inherited Create;
  FExperimental := TExperimental.Create;
end;

destructor TRunOptions.Destroy;
begin
  inherited Destroy;
  FExperimental.Free;
end;

{ TFunctionGraphs }

Constructor TFunctionGraphs.Create;
begin
  inherited Create;

  FPartitionGraphss := TList<TGraphDef>.Create;
end;

destructor TFunctionGraphs.Destroy;
begin
  FPartitionGraphss.Free;
  inherited Destroy;
end;

{ TRunMetadata }

Constructor TRunMetadata.Create;
begin
  inherited Create;

  FPartitionGraphss := TList<TGraphDef>.Create;

  FFunctionGraphss := TList<TFunctionGraphs>.Create;
end;

destructor TRunMetadata.Destroy;
begin
  FPartitionGraphss.Free;
  FFunctionGraphss.Free;
  inherited Destroy;
end;

{ TTensorConnection }

Constructor TTensorConnection.Create;
begin
  inherited Create;
end;

destructor TTensorConnection.Destroy;
begin
  inherited Destroy;
end;

{ TCallableOptions }

Constructor TCallableOptions.Create;
begin
  inherited Create;

  FFeeds := TList<string>.Create;

  FFetchs := TList<string>.Create;

  FTargets := TList<string>.Create;

  FTensorConnections := TList<TTensorConnection>.Create;
  FFeedDevices := TDictionary<string, string>.Create;
  FFetchDevices := TDictionary<string, string>.Create;
end;

destructor TCallableOptions.Destroy;
begin
  FFeeds.Free;
  FFetchs.Free;
  FTargets.Free;
  FTensorConnections.Free;
  FFeedDevices.Free;
  FFetchDevices.Free;
  inherited Destroy;
end;

{ TGPUOptions.TExperimental }

constructor TGPUOptions.TExperimental.Create;
begin
  inherited Create;
  FVirtualDevicess := TList<TVirtualDevices>.Create;
end;

destructor TGPUOptions.TExperimental.Destroy;
begin
  FVirtualDevicess.Free;
  inherited Destroy;
end;

{ TConfigProto.TExperimental }

constructor TConfigProto.TExperimental.Create;
begin
   inherited Create;
end;

destructor TConfigProto.TExperimental.Destroy;
begin
   inherited Destroy;
end;

{ TRunOptions.TExperimental }

constructor TRunOptions.TExperimental.Create;
begin
     inherited Create;
end;

destructor TRunOptions.TExperimental.Destroy;
begin
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'Variable'}
{ TVariableDef }

Constructor TVariableDef.Create;
begin
  inherited Create;
end;

destructor TVariableDef.Destroy;
begin
  inherited Destroy;
end;

{ TSaveSliceInfoDef }

Constructor TSaveSliceInfoDef.Create;
begin
  inherited Create;

  FFullShapes := TList<Int64>.Create;
  FVarOffsets := TList<Int64>.Create;
  FVarShapes  := TList<Int64>.Create;
end;

destructor TSaveSliceInfoDef.Destroy;
begin
  FFullShapes.Free;
  FVarOffsets.Free;
  FVarShapes.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'CppShapeInference'}
{ THandleShapeAndType }

Constructor THandleShapeAndType.Create;
begin
  inherited Create;
end;

destructor THandleShapeAndType.Destroy;
begin
  inherited Destroy;
end;

{ THandleData }

Constructor THandleData.Create;
begin
  inherited Create;

  FShapeAndTypes := TList<THandleShapeAndType>.Create;
end;

destructor THandleData.Destroy;
begin
  FShapeAndTypes.Free;
  inherited Destroy;
end;

{ TCppShapeInferenceResult }

Constructor TCppShapeInferenceResult.Create;
begin
  inherited Create;
end;

destructor TCppShapeInferenceResult.Destroy;
begin
  inherited Destroy;
end;

{ TCppShapeInferenceInputsNeeded }

Constructor TCppShapeInferenceInputsNeeded.Create;
begin
  inherited Create;

  FInputTensorsNeededs := TList<Integer>.Create;

  FInputTensorsAsShapesNeededs := TList<Integer>.Create;
end;

destructor TCppShapeInferenceInputsNeeded.Destroy;
begin
  FInputTensorsNeededs.Free;
  FInputTensorsAsShapesNeededs.Free;
  inherited Destroy;
end;
{$ENDREGION}

{$REGION 'ControlFlow'}
{ TValuesDef }

Constructor TValuesDef.Create;
begin
  inherited Create;

  FValuess := TList<string>.Create;
  FExternalValues := TDictionary<string, string>.Create;
end;

destructor TValuesDef.Destroy;
begin
  FValuess.Free;
  FExternalValues.Free;
  inherited Destroy;
end;

{ TControlFlowContextDef }

Constructor TControlFlowContextDef.Create;
begin
  inherited Create;
end;

destructor TControlFlowContextDef.Destroy;
begin
  inherited Destroy;
end;

{ TCondContextDef }

Constructor TCondContextDef.Create;
begin
  inherited Create;

  FNestedContextss := TList<TControlFlowContextDef>.Create;
end;

destructor TCondContextDef.Destroy;
begin
  FNestedContextss.Free;
  inherited Destroy;
end;

{ TWhileContextDef }

Constructor TWhileContextDef.Create;
begin
  inherited Create;

  FLoopExitNamess  := TList<string>.Create;
  FLoopEnterNamess := TList<string>.Create;
  FNestedContextss := TList<TControlFlowContextDef>.Create;
end;

destructor TWhileContextDef.Destroy;
begin
  FLoopExitNamess.Free;
  FLoopEnterNamess.Free;
  FNestedContextss.Free;
  inherited Destroy;
end;
{$ENDREGION}

end.
