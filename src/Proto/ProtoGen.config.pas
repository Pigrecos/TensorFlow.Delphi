unit ProtoGen.Config;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  ProtoGen.costgraph,
  ProtoGen.graph,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.opdef,
  ProtoGen.stepstats,
  ProtoGen.tensorshape,
  ProtoGen.types,
  ProtoGen.cluster,
  ProtoGen.coordinationconfig,
  ProtoGen.debug,
  ProtoGen.rewriterconfig,
  ProtoGen.tensor,
  ProtoGen.&function,
  ProtoGen.fulltype,
  ProtoGen.nodedef,
  ProtoGen.versions,
  ProtoGen.verifierconfig,
  ProtoGen.allocationdescription,
  ProtoGen.tensordescription;

{$T+}

type


  PVirtualDevices = ^TVirtualDevices;
  TVirtualDevices = record
  const
    ftMemoryLimitMbs = 1;
    ftPrioritys = 2;
  private
    FMemoryLimitMbs: TsgRecordList<Single>;
    FPrioritys: TsgRecordList<Integer>;
  public
    procedure Init;
    procedure Free;
    // properties
    property MemoryLimitMbs: TsgRecordList<Single> read FMemoryLimitMbs;
    property Prioritys: TsgRecordList<Integer> read FPrioritys;
  end;

  PExperimental = ^TExperimental;
  TExperimental = record
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
    FVirtualDevicess: TsgRecordList<TVirtualDevices>;
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
    procedure Init;
    procedure Free;
    // properties
    property VirtualDevicess: TsgRecordList<TVirtualDevices> read FVirtualDevicess;
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

  PGPUOptions = ^TGPUOptions;
  TGPUOptions = record
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
    procedure Init;
    procedure Free;
    // properties
    property PerProcessGpuMemoryFraction: Double read FPerProcessGpuMemoryFraction write FPerProcessGpuMemoryFraction;
    property AllowGrowth: Boolean read FAllowGrowth write FAllowGrowth;
    property AllocatorType: string read FAllocatorType write FAllocatorType;
    property DeferredDeletionBytes: Int64 read FDeferredDeletionBytes write FDeferredDeletionBytes;
    property VisibleDeviceList: string read FVisibleDeviceList write FVisibleDeviceList;
    property PollingActiveDelayUsecs: Integer read FPollingActiveDelayUsecs write FPollingActiveDelayUsecs;
    property PollingInactiveDelayMsecs: Integer read FPollingInactiveDelayMsecs write FPollingInactiveDelayMsecs;
    property ForceGpuCompatible: Boolean read FForceGpuCompatible write FForceGpuCompatible;
    property Experimental: TExperimental read FExperimental write FExperimental;
  end;

  TLevel = (
    L1 = 0,
    L0 = 1);

  TGlobalJitLevel = (
    DEFAULT = 0,
    OFF = 1,
    ON_1 = 1,
    ON_2 = 2);

  POptimizerOptions = ^TOptimizerOptions;
  TOptimizerOptions = record
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
    procedure Init;
    procedure Free;
    // properties
    property DoCommonSubexpressionElimination: Boolean read FDoCommonSubexpressionElimination write FDoCommonSubexpressionElimination;
    property DoConstantFolding: Boolean read FDoConstantFolding write FDoConstantFolding;
    property MaxFoldedConstantInBytes: Int64 read FMaxFoldedConstantInBytes write FMaxFoldedConstantInBytes;
    property DoFunctionInlining: Boolean read FDoFunctionInlining write FDoFunctionInlining;
    property OptLevel: TLevel read FOptLevel write FOptLevel;
    property GlobalJitLevel: TGlobalJitLevel read FGlobalJitLevel write FGlobalJitLevel;
    property CpuGlobalJit: Boolean read FCpuGlobalJit write FCpuGlobalJit;
  end;

  PGraphOptions = ^TGraphOptions;
  TGraphOptions = record
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
    procedure Init;
    procedure Free;
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

  PThreadPoolOptionProto = ^TThreadPoolOptionProto;
  TThreadPoolOptionProto = record
  const
    ftNumThreads = 1;
    ftGlobalName = 2;
  private
    FNumThreads: Integer;
    FGlobalName: string;
  public
    procedure Init;
    procedure Free;
    // properties
    property NumThreads: Integer read FNumThreads write FNumThreads;
    property GlobalName: string read FGlobalName write FGlobalName;
  end;

  PRPCOptions = ^TRPCOptions;
  TRPCOptions = record
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
    procedure Init;
    procedure Free;
    // properties
    property UseRpcForInprocessMaster: Boolean read FUseRpcForInprocessMaster write FUseRpcForInprocessMaster;
    property CompressionAlgorithm: string read FCompressionAlgorithm write FCompressionAlgorithm;
    property CompressionLevel: Integer read FCompressionLevel write FCompressionLevel;
    property CacheRpcResponse: Boolean read FCacheRpcResponse write FCacheRpcResponse;
    property DisableSessionConnectionSharing: Boolean read FDisableSessionConnectionSharing write FDisableSessionConnectionSharing;
    property NumChannelsPerTarget: Integer read FNumChannelsPerTarget write FNumChannelsPerTarget;
  end;

  PSessionMetadata = ^TSessionMetadata;
  TSessionMetadata = record
  const
    ftName = 1;
    ftVersion = 2;
  private
    FName: string;
    FVersion: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property Version: Int64 read FVersion write FVersion;
  end;

  TStringInt32 = TsgHashMap<string, Integer>;

  TMlirBridgeRollout = (
    MLIR_BRIDGE_ROLLOUT_UNSPECIFIED = 0,
    MLIR_BRIDGE_ROLLOUT_ENABLED = 1,
    MLIR_BRIDGE_ROLLOUT_DISABLED = 2,
    MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED = 3,
    MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED = 4);

  PExperimental_Config = ^TExperimental_Config;
  TExperimental_Config = record
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
    procedure Init;
    procedure Free;
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

  PConfigProto = ^TConfigProto;
  TConfigProto = record
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
  private
    FDeviceCount: TStringInt32;
    FIntraOpParallelismThreads: Integer;
    FInterOpParallelismThreads: Integer;
    FUsePerSessionThreads: Boolean;
    FSessionInterOpThreadPools: TsgRecordList<TThreadPoolOptionProto>;
    FPlacementPeriod: Integer;
    FDeviceFilterss: TsgRecordList<string>;
    FGpuOptions: TGPUOptions;
    FAllowSoftPlacement: Boolean;
    FLogDevicePlacement: Boolean;
    FGraphOptions: TGraphOptions;
    FOperationTimeoutInMs: Int64;
    FRpcOptions: TRPCOptions;
    FClusterDef: TClusterDef;
    FIsolateSessionState: Boolean;
    FShareClusterDevicesInSession: Boolean;
    FExperimental: TExperimental_Config;
  public
    procedure Init;
    procedure Free;
    // properties
    property DeviceCount: TStringInt32 read FDeviceCount write FDeviceCount;
    property IntraOpParallelismThreads: Integer read FIntraOpParallelismThreads write FIntraOpParallelismThreads;
    property InterOpParallelismThreads: Integer read FInterOpParallelismThreads write FInterOpParallelismThreads;
    property UsePerSessionThreads: Boolean read FUsePerSessionThreads write FUsePerSessionThreads;
    property SessionInterOpThreadPools: TsgRecordList<TThreadPoolOptionProto> read FSessionInterOpThreadPools;
    property PlacementPeriod: Integer read FPlacementPeriod write FPlacementPeriod;
    property DeviceFilterss: TsgRecordList<string> read FDeviceFilterss;
    property GpuOptions: TGPUOptions read FGpuOptions write FGpuOptions;
    property AllowSoftPlacement: Boolean read FAllowSoftPlacement write FAllowSoftPlacement;
    property LogDevicePlacement: Boolean read FLogDevicePlacement write FLogDevicePlacement;
    property GraphOptions: TGraphOptions read FGraphOptions write FGraphOptions;
    property OperationTimeoutInMs: Int64 read FOperationTimeoutInMs write FOperationTimeoutInMs;
    property RpcOptions: TRPCOptions read FRpcOptions write FRpcOptions;
    property ClusterDef: TClusterDef read FClusterDef write FClusterDef;
    property IsolateSessionState: Boolean read FIsolateSessionState write FIsolateSessionState;
    property ShareClusterDevicesInSession: Boolean read FShareClusterDevicesInSession write FShareClusterDevicesInSession;
    property Experimental: TExperimental_Config read FExperimental write FExperimental;
  end;

  TTraceLevel = (
    NO_TRACE = 0,
    SOFTWARE_TRACE = 1,
    HARDWARE_TRACE = 2,
    FULL_TRACE = 3);

  PRunHandlerPoolOptions = ^TRunHandlerPoolOptions;
  TRunHandlerPoolOptions = record
  const
    ftPriority = 1;
  private
    FPriority: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property Priority: Int64 read FPriority write FPriority;
  end;

  PExperimental_Option = ^TExperimental_Option;
  TExperimental_Option = record
  const
    ftCollectiveGraphKey = 1;
    ftUseRunHandlerPool = 2;
    ftRunHandlerPoolOptions = 3;
  private
    FCollectiveGraphKey: Int64;
    FUseRunHandlerPool: Boolean;
    FRunHandlerPoolOptions: TRunHandlerPoolOptions;
  public
    procedure Init;
    procedure Free;
    // properties
    property CollectiveGraphKey: Int64 read FCollectiveGraphKey write FCollectiveGraphKey;
    property UseRunHandlerPool: Boolean read FUseRunHandlerPool write FUseRunHandlerPool;
    property RunHandlerPoolOptions: TRunHandlerPoolOptions read FRunHandlerPoolOptions write FRunHandlerPoolOptions;
  end;

  PRunOptions = ^TRunOptions;
  TRunOptions = record
  const
    ftTraceLevel = 1;
    ftTimeoutInMs = 2;
    ftInterOpThreadPool = 3;
    ftOutputPartitionGraphs = 5;
    ftDebugOptions = 6;
    ftReportTensorAllocationsUponOom = 7;
    ftExperimental = 8;
  private
    FTraceLevel: TTraceLevel;
    FTimeoutInMs: Int64;
    FInterOpThreadPool: Integer;
    FOutputPartitionGraphs: Boolean;
    FDebugOptions: TDebugOptions;
    FReportTensorAllocationsUponOom: Boolean;
    FExperimental: TExperimental_Option;
  public
    procedure Init;
    procedure Free;
    // properties
    property TraceLevel: TTraceLevel read FTraceLevel write FTraceLevel;
    property TimeoutInMs: Int64 read FTimeoutInMs write FTimeoutInMs;
    property InterOpThreadPool: Integer read FInterOpThreadPool write FInterOpThreadPool;
    property OutputPartitionGraphs: Boolean read FOutputPartitionGraphs write FOutputPartitionGraphs;
    property DebugOptions: TDebugOptions read FDebugOptions write FDebugOptions;
    property ReportTensorAllocationsUponOom: Boolean read FReportTensorAllocationsUponOom write FReportTensorAllocationsUponOom;
    property Experimental: TExperimental_Option read FExperimental write FExperimental;
  end;

  PFunctionGraphs = ^TFunctionGraphs;
  TFunctionGraphs = record
  const
    ftPartitionGraphss = 1;
    ftPreOptimizationGraph = 2;
    ftPostOptimizationGraph = 3;
  private
    FPartitionGraphss: TsgRecordList<TGraphDef>;
    FPreOptimizationGraph: TGraphDef;
    FPostOptimizationGraph: TGraphDef;
  public
    procedure Init;
    procedure Free;
    // properties
    property PartitionGraphss: TsgRecordList<TGraphDef> read FPartitionGraphss;
    property PreOptimizationGraph: TGraphDef read FPreOptimizationGraph write FPreOptimizationGraph;
    property PostOptimizationGraph: TGraphDef read FPostOptimizationGraph write FPostOptimizationGraph;
  end;

  PRunMetadata = ^TRunMetadata;
  TRunMetadata = record
  const
    ftStepStats = 1;
    ftCostGraph = 2;
    ftPartitionGraphss = 3;
    ftFunctionGraphss = 4;
  private
    FStepStats: TStepStats;
    FCostGraph: TCostGraphDef;
    FPartitionGraphss: TsgRecordList<TGraphDef>;
    FFunctionGraphss: TsgRecordList<TFunctionGraphs>;
  public
    procedure Init;
    procedure Free;
    // properties
    property StepStats: TStepStats read FStepStats write FStepStats;
    property CostGraph: TCostGraphDef read FCostGraph write FCostGraph;
    property PartitionGraphss: TsgRecordList<TGraphDef> read FPartitionGraphss;
    property FunctionGraphss: TsgRecordList<TFunctionGraphs> read FFunctionGraphss;
  end;

  PTensorConnection = ^TTensorConnection;
  TTensorConnection = record
  const
    ftFromTensor = 1;
    ftToTensor = 2;
  private
    FFromTensor: string;
    FToTensor: string;
  public
    procedure Init;
    procedure Free;
    // properties
    property FromTensor: string read FFromTensor write FFromTensor;
    property ToTensor: string read FToTensor write FToTensor;
  end;

  TStringString = TsgHashMap<string, string>;

  PCallableOptions = ^TCallableOptions;
  TCallableOptions = record
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
    FFeeds: TsgRecordList<string>;
    FFetchs: TsgRecordList<string>;
    FTargets: TsgRecordList<string>;
    FRunOptions: TRunOptions;
    FTensorConnections: TsgRecordList<TTensorConnection>;
    FFeedDevices: TStringString;
    FFetchDevices: TStringString;
    FFetchSkipSync: Boolean;
  public
    procedure Init;
    procedure Free;
    // properties
    property Feeds: TsgRecordList<string> read FFeeds;
    property Fetchs: TsgRecordList<string> read FFetchs;
    property Targets: TsgRecordList<string> read FTargets;
    property RunOptions: TRunOptions read FRunOptions write FRunOptions;
    property TensorConnections: TsgRecordList<TTensorConnection> read FTensorConnections;
    property FeedDevices: TStringString read FFeedDevices write FFeedDevices;
    property FetchDevices: TStringString read FFetchDevices write FFetchDevices;
    property FetchSkipSync: Boolean read FFetchSkipSync write FFetchSkipSync;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadGPUOptions(var Value: TGPUOptions);
    procedure LoadExperimental(var Value: TExperimental);
    procedure LoadVirtualDevices(var Value: TVirtualDevices);
    procedure LoadOptimizerOptions(var Value: TOptimizerOptions);
    procedure LoadGraphOptions(var Value: TGraphOptions);
    procedure LoadThreadPoolOptionProto(var Value: TThreadPoolOptionProto);
    procedure LoadRPCOptions(var Value: TRPCOptions);
    procedure LoadSessionMetadata(var Value: TSessionMetadata);
    procedure LoadConfigProto(var Value: TConfigProto);
    procedure LoadExperimental_Config(var Value: TExperimental_Config);
    procedure LoadRunOptions(var Value: TRunOptions);
    procedure LoadExperimental_Option(var Value: TExperimental_Option);
    procedure LoadRunHandlerPoolOptions(var Value: TRunHandlerPoolOptions);
    procedure LoadRunMetadata(var Value: TRunMetadata);
    procedure LoadFunctionGraphs(var Value: TFunctionGraphs);
    procedure LoadTensorConnection(var Value: TTensorConnection);
    procedure LoadCallableOptions(var Value: TCallableOptions);
    procedure LoadCostGraphDef(var Value: TCostGraphDef);
    procedure LoadNode(var Value: TNode);
    procedure LoadInputInfo(var Value: TInputInfo);
    procedure LoadOutputInfo(var Value: TOutputInfo);
    procedure LoadAggregatedCost(var Value: TAggregatedCost);
    procedure LoadGraphDef(var Value: TGraphDef);
    procedure LoadAttrValue(var Value: TAttrValue);
    procedure LoadListValue(var Value: TListValue);
    procedure LoadNameAttrList(var Value: TNameAttrList);
    procedure LoadResourceHandleProto(var Value: TResourceHandleProto);
    procedure LoadDtypeAndShape(var Value: TDtypeAndShape);
    procedure LoadOpDef(var Value: TOpDef);
    procedure LoadArgDef(var Value: TArgDef);
    procedure LoadAttrDef(var Value: TAttrDef);
    procedure LoadOpDeprecation(var Value: TOpDeprecation);
    procedure LoadOpList(var Value: TOpList);
    procedure LoadAllocationRecord(var Value: TAllocationRecord);
    procedure LoadAllocatorMemoryUsed(var Value: TAllocatorMemoryUsed);
    procedure LoadNodeOutput(var Value: TNodeOutput);
    procedure LoadMemoryStats(var Value: TMemoryStats);
    procedure LoadNodeExecStats(var Value: TNodeExecStats);
    procedure LoadDeviceStepStats(var Value: TDeviceStepStats);
    procedure LoadStepStats(var Value: TStepStats);
    procedure LoadTensorShapeProto(var Value: TTensorShapeProto);
    procedure LoadDim(var Value: TDim);
    procedure LoadJobDef(var Value: TJobDef);
    procedure LoadClusterDef(var Value: TClusterDef);
    procedure LoadCoordinationServiceConfig(var Value: TCoordinationServiceConfig);
    procedure LoadDebugTensorWatch(var Value: TDebugTensorWatch);
    procedure LoadDebugOptions(var Value: TDebugOptions);
    procedure LoadDebuggedSourceFile(var Value: TDebuggedSourceFile);
    procedure LoadDebuggedSourceFiles(var Value: TDebuggedSourceFiles);
    procedure LoadAutoParallelOptions(var Value: TAutoParallelOptions);
    procedure LoadScopedAllocatorOptions(var Value: TScopedAllocatorOptions);
    procedure LoadRewriterConfig(var Value: TRewriterConfig);
    procedure LoadCustomGraphOptimizer(var Value: TCustomGraphOptimizer);
    procedure LoadTensorProto(var Value: TTensorProto);
    procedure LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
    procedure LoadFunctionDefLibrary(var Value: TFunctionDefLibrary);
    procedure LoadFunctionDef(var Value: TFunctionDef);
    procedure LoadArgAttrs(var Value: TArgAttrs);
    procedure LoadGradientDef(var Value: TGradientDef);
    procedure LoadRegisteredGradient(var Value: TRegisteredGradient);
    procedure LoadFullTypeDef(var Value: TFullTypeDef);
    procedure LoadNodeDef(var Value: TNodeDef);
    procedure LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
    procedure LoadVersionDef(var Value: TVersionDef);
    procedure LoadVerifierConfig(var Value: TVerifierConfig);
    procedure LoadAllocationDescription(var Value: TAllocationDescription);
    procedure LoadTensorDescription(var Value: TTensorDescription);
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
    class procedure SaveGPUOptions(const S: TpbSaver; const Value: TGPUOptions); static;
    class procedure SaveExperimental(const S: TpbSaver; const Value: TExperimental); static;
    class procedure SaveVirtualDevices(const S: TpbSaver; const Value: TVirtualDevices); static;
    class procedure SaveOptimizerOptions(const S: TpbSaver; const Value: TOptimizerOptions); static;
    class procedure SaveGraphOptions(const S: TpbSaver; const Value: TGraphOptions); static;
    class procedure SaveThreadPoolOptionProto(const S: TpbSaver; const Value: TThreadPoolOptionProto); static;
    class procedure SaveRPCOptions(const S: TpbSaver; const Value: TRPCOptions); static;
    class procedure SaveSessionMetadata(const S: TpbSaver; const Value: TSessionMetadata); static;
    class procedure SaveConfigProto(const S: TpbSaver; const Value: TConfigProto); static;
    class procedure SaveExperimental_Config(const S: TpbSaver; const Value: TExperimental_Config); static;
    class procedure SaveRunOptions(const S: TpbSaver; const Value: TRunOptions); static;
    class procedure SaveExperimental_Option(const S: TpbSaver; const Value: TExperimental_Option); static;
    class procedure SaveRunHandlerPoolOptions(const S: TpbSaver; const Value: TRunHandlerPoolOptions); static;
    class procedure SaveRunMetadata(const S: TpbSaver; const Value: TRunMetadata); static;
    class procedure SaveFunctionGraphs(const S: TpbSaver; const Value: TFunctionGraphs); static;
    class procedure SaveTensorConnection(const S: TpbSaver; const Value: TTensorConnection); static;
    class procedure SaveCallableOptions(const S: TpbSaver; const Value: TCallableOptions); static;
    class procedure SaveCostGraphDef(const S: TpbSaver; const Value: TCostGraphDef); static;
    class procedure SaveNode(const S: TpbSaver; const Value: TNode); static;
    class procedure SaveInputInfo(const S: TpbSaver; const Value: TInputInfo); static;
    class procedure SaveOutputInfo(const S: TpbSaver; const Value: TOutputInfo); static;
    class procedure SaveAggregatedCost(const S: TpbSaver; const Value: TAggregatedCost); static;
    class procedure SaveGraphDef(const S: TpbSaver; const Value: TGraphDef); static;
    class procedure SaveAttrValue(const S: TpbSaver; const Value: TAttrValue); static;
    class procedure SaveListValue(const S: TpbSaver; const Value: TListValue); static;
    class procedure SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList); static;
    class procedure SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto); static;
    class procedure SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape); static;
    class procedure SaveOpDef(const S: TpbSaver; const Value: TOpDef); static;
    class procedure SaveArgDef(const S: TpbSaver; const Value: TArgDef); static;
    class procedure SaveAttrDef(const S: TpbSaver; const Value: TAttrDef); static;
    class procedure SaveOpDeprecation(const S: TpbSaver; const Value: TOpDeprecation); static;
    class procedure SaveOpList(const S: TpbSaver; const Value: TOpList); static;
    class procedure SaveAllocationRecord(const S: TpbSaver; const Value: TAllocationRecord); static;
    class procedure SaveAllocatorMemoryUsed(const S: TpbSaver; const Value: TAllocatorMemoryUsed); static;
    class procedure SaveNodeOutput(const S: TpbSaver; const Value: TNodeOutput); static;
    class procedure SaveMemoryStats(const S: TpbSaver; const Value: TMemoryStats); static;
    class procedure SaveNodeExecStats(const S: TpbSaver; const Value: TNodeExecStats); static;
    class procedure SaveDeviceStepStats(const S: TpbSaver; const Value: TDeviceStepStats); static;
    class procedure SaveStepStats(const S: TpbSaver; const Value: TStepStats); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
    class procedure SaveJobDef(const S: TpbSaver; const Value: TJobDef); static;
    class procedure SaveClusterDef(const S: TpbSaver; const Value: TClusterDef); static;
    class procedure SaveCoordinationServiceConfig(const S: TpbSaver; const Value: TCoordinationServiceConfig); static;
    class procedure SaveDebugTensorWatch(const S: TpbSaver; const Value: TDebugTensorWatch); static;
    class procedure SaveDebugOptions(const S: TpbSaver; const Value: TDebugOptions); static;
    class procedure SaveDebuggedSourceFile(const S: TpbSaver; const Value: TDebuggedSourceFile); static;
    class procedure SaveDebuggedSourceFiles(const S: TpbSaver; const Value: TDebuggedSourceFiles); static;
    class procedure SaveAutoParallelOptions(const S: TpbSaver; const Value: TAutoParallelOptions); static;
    class procedure SaveScopedAllocatorOptions(const S: TpbSaver; const Value: TScopedAllocatorOptions); static;
    class procedure SaveRewriterConfig(const S: TpbSaver; const Value: TRewriterConfig); static;
    class procedure SaveCustomGraphOptimizer(const S: TpbSaver; const Value: TCustomGraphOptimizer); static;
    class procedure SaveTensorProto(const S: TpbSaver; const Value: TTensorProto); static;
    class procedure SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto); static;
    class procedure SaveFunctionDefLibrary(const S: TpbSaver; const Value: TFunctionDefLibrary); static;
    class procedure SaveFunctionDef(const S: TpbSaver; const Value: TFunctionDef); static;
    class procedure SaveArgAttrs(const S: TpbSaver; const Value: TArgAttrs); static;
    class procedure SaveGradientDef(const S: TpbSaver; const Value: TGradientDef); static;
    class procedure SaveRegisteredGradient(const S: TpbSaver; const Value: TRegisteredGradient); static;
    class procedure SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef); static;
    class procedure SaveNodeDef(const S: TpbSaver; const Value: TNodeDef); static;
    class procedure SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo); static;
    class procedure SaveVersionDef(const S: TpbSaver; const Value: TVersionDef); static;
    class procedure SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig); static;
    class procedure SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription); static;
    class procedure SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription); static;
    procedure SaveStringInt32(Item: TsgPair<string, Integer>);
    procedure SaveStringString(Item: TsgPair<string, string>);
    procedure SaveStringAttrValue(Item: TPair<string, TAttrValue>);
    procedure SaveUint32String(Item: TsgPair<UInt32, string>);
    procedure SaveInt32String(Item: TsgPair<Integer, string>);
    procedure SaveUint32ArgAttrs(Item: TsgPair<UInt32, TArgAttrs>);
    procedure SaveUint32Uint32(Item: TsgPair<UInt32, UInt32>);

  end;

implementation

{ TVirtualDevices }

procedure TVirtualDevices.Init;
begin
  Self := System.Default(TVirtualDevices);
  FMemoryLimitMbs := TsgRecordList<Single>.From(nil);
  FPrioritys := TsgRecordList<Integer>.From(nil);
end;

procedure TVirtualDevices.Free;
begin
  FMemoryLimitMbs.Free;
  FPrioritys.Free;
end;

{ TExperimental }

procedure TExperimental.Init;
begin
  Self := System.Default(TExperimental);
  FVirtualDevicess := TsgRecordList<TVirtualDevices>.From(nil);
end;

procedure TExperimental.Free;
begin
  FVirtualDevicess.Free;
end;

{ TGPUOptions }

procedure TGPUOptions.Init;
begin
  Self := System.Default(TGPUOptions);
end;

procedure TGPUOptions.Free;
begin
end;

{ TOptimizerOptions }

procedure TOptimizerOptions.Init;
begin
  Self := System.Default(TOptimizerOptions);
end;

procedure TOptimizerOptions.Free;
begin
end;

{ TGraphOptions }

procedure TGraphOptions.Init;
begin
  Self := System.Default(TGraphOptions);
end;

procedure TGraphOptions.Free;
begin
end;

{ TThreadPoolOptionProto }

procedure TThreadPoolOptionProto.Init;
begin
  Self := System.Default(TThreadPoolOptionProto);
end;

procedure TThreadPoolOptionProto.Free;
begin
end;

{ TRPCOptions }

procedure TRPCOptions.Init;
begin
  Self := System.Default(TRPCOptions);
end;

procedure TRPCOptions.Free;
begin
end;

{ TSessionMetadata }

procedure TSessionMetadata.Init;
begin
  Self := System.Default(TSessionMetadata);
end;

procedure TSessionMetadata.Free;
begin
end;

{ TExperimental }

procedure TExperimental_Config.Init;
begin
  Self := System.Default(TExperimental_Config);
end;

procedure TExperimental_Config.Free;
begin
end;

{ TConfigProto }

procedure TConfigProto.Init;
begin
  Self := System.Default(TConfigProto);
  FDeviceCount := TsgHashMap<string, Integer>.From(0,nil);
  FSessionInterOpThreadPools := TsgRecordList<TThreadPoolOptionProto>.From(nil);
  FDeviceFilterss := TsgRecordList<string>.From(nil);
end;

procedure TConfigProto.Free;
begin
  FDeviceCount.Free;
  FSessionInterOpThreadPools.Free;
  FDeviceFilterss.Free;
end;

{ TRunHandlerPoolOptions }

procedure TRunHandlerPoolOptions.Init;
begin
  Self := System.Default(TRunHandlerPoolOptions);
end;

procedure TRunHandlerPoolOptions.Free;
begin
end;

{ TExperimental }

procedure TExperimental_Option.Init;
begin
  Self := System.Default(TExperimental_Option);
end;

procedure TExperimental_Option.Free;
begin
end;

{ TRunOptions }

procedure TRunOptions.Init;
begin
  Self := System.Default(TRunOptions);
end;

procedure TRunOptions.Free;
begin
end;

{ TFunctionGraphs }

procedure TFunctionGraphs.Init;
begin
  Self := System.Default(TFunctionGraphs);
  FPartitionGraphss := TsgRecordList<TGraphDef>.From(nil);
end;

procedure TFunctionGraphs.Free;
begin
  FPartitionGraphss.Free;
end;

{ TRunMetadata }

procedure TRunMetadata.Init;
begin
  Self := System.Default(TRunMetadata);
  FPartitionGraphss := TsgRecordList<TGraphDef>.From(nil);
  FFunctionGraphss := TsgRecordList<TFunctionGraphs>.From(nil);
end;

procedure TRunMetadata.Free;
begin
  FPartitionGraphss.Free;
  FFunctionGraphss.Free;
end;

{ TTensorConnection }

procedure TTensorConnection.Init;
begin
  Self := System.Default(TTensorConnection);
end;

procedure TTensorConnection.Free;
begin
end;

{ TCallableOptions }

procedure TCallableOptions.Init;
begin
  Self := System.Default(TCallableOptions);
  FFeeds := TsgRecordList<string>.From(nil);
  FFetchs := TsgRecordList<string>.From(nil);
  FTargets := TsgRecordList<string>.From(nil);
  FTensorConnections := TsgRecordList<TTensorConnection>.From(nil);
  FFeedDevices := TsgHashMap<string, string>.From(0,nil);
  FFetchDevices := TsgHashMap<string, string>.From(0,nil);
end;

procedure TCallableOptions.Free;
begin
  FFeeds.Free;
  FFetchs.Free;
  FTargets.Free;
  FTensorConnections.Free;
  FFeedDevices.Free;
  FFetchDevices.Free;
end;

procedure TLoadHelper.LoadVirtualDevices(var Value: TVirtualDevices);
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
      TVirtualDevices.ftMemoryLimitMbs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.FMemoryLimitMbs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TVirtualDevices.ftPrioritys:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.FPrioritys.Add(@v);
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

procedure TLoadHelper.LoadExperimental(var Value: TExperimental);
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
      TExperimental.ftVirtualDevicess:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVirtualDevices;
            LoadVirtualDevices(v);
            Value.FVirtualDevicess.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TExperimental.ftUseUnifiedMemory:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseUnifiedMemory := Pb.readBoolean;
        end;
      TExperimental.ftNumDevToDevCopyStreams:
        begin
          Assert(wireType = TWire.VARINT);
          Value.NumDevToDevCopyStreams := Pb.readInt32;
        end;
      TExperimental.ftCollectiveRingOrder:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.CollectiveRingOrder := Pb.readString;
        end;
      TExperimental.ftTimestampedAllocator:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TimestampedAllocator := Pb.readBoolean;
        end;
      TExperimental.ftKernelTrackerMaxInterval:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxInterval := Pb.readInt32;
        end;
      TExperimental.ftKernelTrackerMaxBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxBytes := Pb.readInt32;
        end;
      TExperimental.ftKernelTrackerMaxPending:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxPending := Pb.readInt32;
        end;
      TExperimental.ftInternalFragmentationFraction:
        begin
          Assert(wireType = TWire.FIXED64);
          Value.InternalFragmentationFraction := Pb.readDouble;
        end;
      TExperimental.ftUseCudaMallocAsync:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseCudaMallocAsync := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadGPUOptions(var Value: TGPUOptions);
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
      TGPUOptions.ftPerProcessGpuMemoryFraction:
        begin
          Assert(wireType = TWire.FIXED64);
          Value.PerProcessGpuMemoryFraction := Pb.readDouble;
        end;
      TGPUOptions.ftAllowGrowth:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllowGrowth := Pb.readBoolean;
        end;
      TGPUOptions.ftAllocatorType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorType := Pb.readString;
        end;
      TGPUOptions.ftDeferredDeletionBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DeferredDeletionBytes := Pb.readInt64;
        end;
      TGPUOptions.ftVisibleDeviceList:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.VisibleDeviceList := Pb.readString;
        end;
      TGPUOptions.ftPollingActiveDelayUsecs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PollingActiveDelayUsecs := Pb.readInt32;
        end;
      TGPUOptions.ftPollingInactiveDelayMsecs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PollingInactiveDelayMsecs := Pb.readInt32;
        end;
      TGPUOptions.ftForceGpuCompatible:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ForceGpuCompatible := Pb.readBoolean;
        end;
      TGPUOptions.ftExperimental:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TExperimental := Value.FExperimental;
            LoadExperimental(v);
            Value.FExperimental := v;
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

procedure TLoadHelper.LoadOptimizerOptions(var Value: TOptimizerOptions);
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
      TOptimizerOptions.ftDoCommonSubexpressionElimination:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DoCommonSubexpressionElimination := Pb.readBoolean;
        end;
      TOptimizerOptions.ftDoConstantFolding:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DoConstantFolding := Pb.readBoolean;
        end;
      TOptimizerOptions.ftMaxFoldedConstantInBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MaxFoldedConstantInBytes := Pb.readInt64;
        end;
      TOptimizerOptions.ftDoFunctionInlining:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DoFunctionInlining := Pb.readBoolean;
        end;
      TOptimizerOptions.ftOptLevel:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OptLevel := TLevel(Pb.readInt32);
        end;
      TOptimizerOptions.ftGlobalJitLevel:
        begin
          Assert(wireType = TWire.VARINT);
          Value.GlobalJitLevel := TGlobalJitLevel(Pb.readInt32);
        end;
      TOptimizerOptions.ftCpuGlobalJit:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CpuGlobalJit := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadGraphOptions(var Value: TGraphOptions);
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
      TGraphOptions.ftEnableRecvScheduling:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableRecvScheduling := Pb.readBoolean;
        end;
      TGraphOptions.ftOptimizerOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOptimizerOptions := Value.FOptimizerOptions;
            LoadOptimizerOptions(v);
            Value.FOptimizerOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TGraphOptions.ftBuildCostModel:
        begin
          Assert(wireType = TWire.VARINT);
          Value.BuildCostModel := Pb.readInt64;
        end;
      TGraphOptions.ftBuildCostModelAfter:
        begin
          Assert(wireType = TWire.VARINT);
          Value.BuildCostModelAfter := Pb.readInt64;
        end;
      TGraphOptions.ftInferShapes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.InferShapes := Pb.readBoolean;
        end;
      TGraphOptions.ftPlacePrunedGraph:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PlacePrunedGraph := Pb.readBoolean;
        end;
      TGraphOptions.ftEnableBfloat16Sendrecv:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableBfloat16Sendrecv := Pb.readBoolean;
        end;
      TGraphOptions.ftTimelineStep:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TimelineStep := Pb.readInt32;
        end;
      TGraphOptions.ftRewriteOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRewriterConfig := Value.FRewriteOptions;
            LoadRewriterConfig(v);
            Value.FRewriteOptions := v;
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

procedure TLoadHelper.LoadThreadPoolOptionProto(var Value: TThreadPoolOptionProto);
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
      TThreadPoolOptionProto.ftNumThreads:
        begin
          Assert(wireType = TWire.VARINT);
          Value.NumThreads := Pb.readInt32;
        end;
      TThreadPoolOptionProto.ftGlobalName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.GlobalName := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadRPCOptions(var Value: TRPCOptions);
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
      TRPCOptions.ftUseRpcForInprocessMaster:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseRpcForInprocessMaster := Pb.readBoolean;
        end;
      TRPCOptions.ftCompressionAlgorithm:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.CompressionAlgorithm := Pb.readString;
        end;
      TRPCOptions.ftCompressionLevel:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CompressionLevel := Pb.readInt32;
        end;
      TRPCOptions.ftCacheRpcResponse:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CacheRpcResponse := Pb.readBoolean;
        end;
      TRPCOptions.ftDisableSessionConnectionSharing:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableSessionConnectionSharing := Pb.readBoolean;
        end;
      TRPCOptions.ftNumChannelsPerTarget:
        begin
          Assert(wireType = TWire.VARINT);
          Value.NumChannelsPerTarget := Pb.readInt32;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadSessionMetadata(var Value: TSessionMetadata);
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
      TSessionMetadata.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TSessionMetadata.ftVersion:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Version := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadExperimental_Config(var Value: TExperimental_Config);
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
      TExperimental_Config.ftCollectiveGroupLeader:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.CollectiveGroupLeader := Pb.readString;
        end;
      TExperimental_Config.ftExecutorType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ExecutorType := Pb.readString;
        end;
      TExperimental_Config.ftRecvBufMaxChunk:
        begin
          Assert(wireType = TWire.VARINT);
          Value.RecvBufMaxChunk := Pb.readInt32;
        end;
      TExperimental_Config.ftUseNumaAffinity:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseNumaAffinity := Pb.readBoolean;
        end;
      TExperimental_Config.ftCollectiveDeterministicSequentialExecution:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveDeterministicSequentialExecution := Pb.readBoolean;
        end;
      TExperimental_Config.ftCollectiveNccl:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveNccl := Pb.readBoolean;
        end;
      TExperimental_Config.ftShareSessionStateInClusterspecPropagation:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShareSessionStateInClusterspecPropagation := Pb.readBoolean;
        end;
      TExperimental_Config.ftDisableThreadSpinning:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableThreadSpinning := Pb.readBoolean;
        end;
      TExperimental_Config.ftShareClusterDevicesInSession:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShareClusterDevicesInSession := Pb.readBoolean;
        end;
      TExperimental_Config.ftSessionMetadata:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TSessionMetadata := Value.FSessionMetadata;
            LoadSessionMetadata(v);
            Value.FSessionMetadata := v;
          finally
            Pb.Pop;
          end;
        end;
      TExperimental_Config.ftOptimizeForStaticGraph:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OptimizeForStaticGraph := Pb.readBoolean;
        end;
      TExperimental_Config.ftEnableMlirBridge:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableMlirBridge := Pb.readBoolean;
        end;
      TExperimental_Config.ftMlirBridgeRollout:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MlirBridgeRollout := TMlirBridgeRollout(Pb.readInt32);
        end;
      TExperimental_Config.ftEnableMlirGraphOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableMlirGraphOptimization := Pb.readBoolean;
        end;
      TExperimental_Config.ftDisableOutputPartitionGraphs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableOutputPartitionGraphs := Pb.readBoolean;
        end;
      TExperimental_Config.ftXlaFusionAutotunerThresh:
        begin
          Assert(wireType = TWire.VARINT);
          Value.XlaFusionAutotunerThresh := Pb.readInt64;
        end;
      TExperimental_Config.ftUseTfrt:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseTfrt := Pb.readBoolean;
        end;
      TExperimental_Config.ftDisableFunctionalOpsLowering:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableFunctionalOpsLowering := Pb.readBoolean;
        end;
      TExperimental_Config.ftXlaPreferSingleGraphCluster:
        begin
          Assert(wireType = TWire.VARINT);
          Value.XlaPreferSingleGraphCluster := Pb.readBoolean;
        end;
      TExperimental_Config.ftCoordinationConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCoordinationServiceConfig := Value.FCoordinationConfig;
            LoadCoordinationServiceConfig(v);
            Value.FCoordinationConfig := v;
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

procedure TLoadHelper.LoadConfigProto(var Value: TConfigProto);
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
      TConfigProto.ftDeviceCount:
        begin
          Value.DeviceCount.InsertOrAssign(TsgPair<string, Integer>.From(Pb.readString, Pb.readInt32));
        end;
      TConfigProto.ftIntraOpParallelismThreads:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IntraOpParallelismThreads := Pb.readInt32;
        end;
      TConfigProto.ftInterOpParallelismThreads:
        begin
          Assert(wireType = TWire.VARINT);
          Value.InterOpParallelismThreads := Pb.readInt32;
        end;
      TConfigProto.ftUsePerSessionThreads:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UsePerSessionThreads := Pb.readBoolean;
        end;
      TConfigProto.ftSessionInterOpThreadPools:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TThreadPoolOptionProto;
            LoadThreadPoolOptionProto(v);
            Value.FSessionInterOpThreadPools.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftPlacementPeriod:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PlacementPeriod := Pb.readInt32;
        end;
      TConfigProto.ftDeviceFilterss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FDeviceFilterss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftGpuOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGPUOptions := Value.FGpuOptions;
            LoadGPUOptions(v);
            Value.FGpuOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftAllowSoftPlacement:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllowSoftPlacement := Pb.readBoolean;
        end;
      TConfigProto.ftLogDevicePlacement:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LogDevicePlacement := Pb.readBoolean;
        end;
      TConfigProto.ftGraphOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphOptions := Value.FGraphOptions;
            LoadGraphOptions(v);
            Value.FGraphOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftOperationTimeoutInMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OperationTimeoutInMs := Pb.readInt64;
        end;
      TConfigProto.ftRpcOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRPCOptions := Value.FRpcOptions;
            LoadRPCOptions(v);
            Value.FRpcOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftClusterDef:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TClusterDef := Value.FClusterDef;
            LoadClusterDef(v);
            Value.FClusterDef := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftIsolateSessionState:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsolateSessionState := Pb.readBoolean;
        end;
      TConfigProto.ftShareClusterDevicesInSession:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShareClusterDevicesInSession := Pb.readBoolean;
        end;
      TConfigProto.ftExperimental:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TExperimental_Config := Value.FExperimental;
            LoadExperimental_Config(v);
            Value.FExperimental := v;
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

procedure TLoadHelper.LoadRunHandlerPoolOptions(var Value: TRunHandlerPoolOptions);
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
      TRunHandlerPoolOptions.ftPriority:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Priority := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadExperimental_Option(var Value: TExperimental_Option);
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
      TExperimental_Option.ftCollectiveGraphKey:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveGraphKey := Pb.readInt64;
        end;
      TExperimental_Option.ftUseRunHandlerPool:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseRunHandlerPool := Pb.readBoolean;
        end;
      TExperimental_Option.ftRunHandlerPoolOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRunHandlerPoolOptions := Value.FRunHandlerPoolOptions;
            LoadRunHandlerPoolOptions(v);
            Value.FRunHandlerPoolOptions := v;
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

procedure TLoadHelper.LoadRunOptions(var Value: TRunOptions);
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
      TRunOptions.ftTraceLevel:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TraceLevel := TTraceLevel(Pb.readInt32);
        end;
      TRunOptions.ftTimeoutInMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TimeoutInMs := Pb.readInt64;
        end;
      TRunOptions.ftInterOpThreadPool:
        begin
          Assert(wireType = TWire.VARINT);
          Value.InterOpThreadPool := Pb.readInt32;
        end;
      TRunOptions.ftOutputPartitionGraphs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OutputPartitionGraphs := Pb.readBoolean;
        end;
      TRunOptions.ftDebugOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDebugOptions := Value.FDebugOptions;
            LoadDebugOptions(v);
            Value.FDebugOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TRunOptions.ftReportTensorAllocationsUponOom:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ReportTensorAllocationsUponOom := Pb.readBoolean;
        end;
      TRunOptions.ftExperimental:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TExperimental_Option := Value.FExperimental;
            LoadExperimental_Option(v);
            Value.FExperimental := v;
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

procedure TLoadHelper.LoadFunctionGraphs(var Value: TFunctionGraphs);
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
      TFunctionGraphs.ftPartitionGraphss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef;
            LoadGraphDef(v);
            Value.FPartitionGraphss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionGraphs.ftPreOptimizationGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef := Value.FPreOptimizationGraph;
            LoadGraphDef(v);
            Value.FPreOptimizationGraph := v;
          finally
            Pb.Pop;
          end;
        end;
      TFunctionGraphs.ftPostOptimizationGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef := Value.FPostOptimizationGraph;
            LoadGraphDef(v);
            Value.FPostOptimizationGraph := v;
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

procedure TLoadHelper.LoadRunMetadata(var Value: TRunMetadata);
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
      TRunMetadata.ftStepStats:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TStepStats := Value.FStepStats;
            LoadStepStats(v);
            Value.FStepStats := v;
          finally
            Pb.Pop;
          end;
        end;
      TRunMetadata.ftCostGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCostGraphDef := Value.FCostGraph;
            LoadCostGraphDef(v);
            Value.FCostGraph := v;
          finally
            Pb.Pop;
          end;
        end;
      TRunMetadata.ftPartitionGraphss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef;
            LoadGraphDef(v);
            Value.FPartitionGraphss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TRunMetadata.ftFunctionGraphss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFunctionGraphs;
            LoadFunctionGraphs(v);
            Value.FFunctionGraphss.Add(@v);
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

procedure TLoadHelper.LoadTensorConnection(var Value: TTensorConnection);
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
      TTensorConnection.ftFromTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.FromTensor := Pb.readString;
        end;
      TTensorConnection.ftToTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ToTensor := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadCallableOptions(var Value: TCallableOptions);
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
      TCallableOptions.ftFeeds:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FFeeds.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftFetchs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FFetchs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftTargets:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FTargets.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftRunOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRunOptions := Value.FRunOptions;
            LoadRunOptions(v);
            Value.FRunOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftTensorConnections:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorConnection;
            LoadTensorConnection(v);
            Value.FTensorConnections.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftFeedDevices:
        begin
          Value.FeedDevices.InsertOrAssign(TsgPair<string, string>.From(Pb.readString, Pb.readString));
        end;
      TCallableOptions.ftFetchDevices:
        begin
          Value.FetchDevices.InsertOrAssign(TsgPair<string, string>.From(Pb.readString, Pb.readString));
        end;
      TCallableOptions.ftFetchSkipSync:
        begin
          Assert(wireType = TWire.VARINT);
          Value.FetchSkipSync := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
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
            var v : TTensorShapeProto := Value.Shape;
            LoadTensorShapeProto(v);
            Value.Shape := v;
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
            Value.InputInfos.Add(@v);
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
            Value.OutputInfos.Add(@v);
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
              Value.ControlInputs.Add(@v);
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
            Value.Nodes.Add(@v);
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
            Value.Costs.Add(@v);
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

procedure TLoadHelper.LoadGraphDef(var Value: TGraphDef);
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
      TGraphDef.ftNodes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeDef;
            LoadNodeDef(v);
            Value.Nodes.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TGraphDef.ftVersions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVersionDef := Value.Versions;
            LoadVersionDef(v);
            Value.Versions := v;
          finally
            Pb.Pop;
          end;
        end;
      TGraphDef.ftVersion:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Version := Pb.readInt32;
        end;
      TGraphDef.ftLibrary:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFunctionDefLibrary := Value.&Library;
            LoadFunctionDefLibrary(v);
            Value.&Library := v;
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

procedure TLoadHelper.LoadListValue(var Value: TListValue);
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
      TListValue.ftSs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TBytes := Pb.readBytes;
              Value.Ss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftIs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.&Is.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftFs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.Fs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftBs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : boolean := Pb.readBoolean;
              Value.Bs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftTypes:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TDataType := TDataType(Pb.readInt32);
              Value.Types.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftShapes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto;
            LoadTensorShapeProto(v);
            Value.Shapes.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftTensors:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorProto;
            LoadTensorProto(v);
            Value.Tensors.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftFuncs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNameAttrList;
            LoadNameAttrList(v);
            Value.Funcs.Add(@v);
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

procedure TLoadHelper.LoadAttrValue(var Value: TAttrValue);
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
      TAttrValue.ftS:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftS;
          v.value := TValue.From<TBytes>(Pb.readBytes);
          Value.value := v;
        end;
      TAttrValue.ftI:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftI;
          v.value := Pb.readInt64;
          Value.value := v;
        end;
      TAttrValue.ftF:
        begin
          Assert(wireType = TWire.FIXED32);
          var v : TpbOneof;
          v.tag := TAttrValue.ftF;
          v.value := Pb.readFloat;
          Value.value := v;
        end;
      TAttrValue.ftB:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftB;
          v.value := Pb.readBoolean;
          Value.value := v;
        end;
      TAttrValue.ftType:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftType;
          v.value := TValue.From<TDataType>(TDataType(Pb.readInt32));
          Value.value := v;
        end;
      TAttrValue.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftShape;
        
          var v1 : TTensorShapeProto;
          LoadTensorShapeProto(v1);
          v.value := TValue.From<TTensorShapeProto>(v1);
          Value.value := v;
        end;
      TAttrValue.ftTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftTensor;
        
          var v1 : TTensorProto;
          LoadTensorProto(v1);
          v.value := TValue.From<TTensorProto>(v1);
          Value.value := v;
        end;
      TAttrValue.ftList:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftList;
        
          var v1 : TListValue;
          LoadListValue(v1);
          v.value := TValue.From<TListValue>(v1);
          Value.value := v;
        end;
      TAttrValue.ftFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftFunc;
        
          var v1 : TNameAttrList;
          LoadNameAttrList(v1);
          v.value := TValue.From<TNameAttrList>(v1);
          Value.value := v;
        end;
      TAttrValue.ftPlaceholder:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftPlaceholder;
          v.value := Pb.readString;
          Value.value := v;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadNameAttrList(var Value: TNameAttrList);
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
      TNameAttrList.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TNameAttrList.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDtypeAndShape(var Value: TDtypeAndShape);
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
      TDtypeAndShape.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TDtypeAndShape.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.Shape;
            LoadTensorShapeProto(v);
            Value.Shape := v;
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

procedure TLoadHelper.LoadResourceHandleProto(var Value: TResourceHandleProto);
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
      TResourceHandleProto.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TResourceHandleProto.ftContainer:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Container := Pb.readString;
        end;
      TResourceHandleProto.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TResourceHandleProto.ftHashCode:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HashCode := Pb.readInt64;
        end;
      TResourceHandleProto.ftMaybeTypeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.MaybeTypeName := Pb.readString;
        end;
      TResourceHandleProto.ftDtypesAndShapess:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDtypeAndShape;
            LoadDtypeAndShape(v);
            Value.DtypesAndShapess.Add(@v);
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

procedure TLoadHelper.LoadArgDef(var Value: TArgDef);
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
      TArgDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TArgDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TArgDef.ftType:
        begin
          Assert(wireType = TWire.VARINT);
          Value.&Type := TDataType(Pb.readInt32);
        end;
      TArgDef.ftTypeAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeAttr := Pb.readString;
        end;
      TArgDef.ftNumberAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NumberAttr := Pb.readString;
        end;
      TArgDef.ftTypeListAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeListAttr := Pb.readString;
        end;
      TArgDef.ftHandleDatas:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TResourceHandleProto;
            LoadResourceHandleProto(v);
            Value.HandleDatas.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TArgDef.ftIsRef:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsRef := Pb.readBoolean;
        end;
      TArgDef.ftExperimentalFullType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.ExperimentalFullType;
            LoadFullTypeDef(v);
            Value.ExperimentalFullType := v;
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

procedure TLoadHelper.LoadAttrDef(var Value: TAttrDef);
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
      TAttrDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TAttrDef.ftType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.&Type := Pb.readString;
        end;
      TAttrDef.ftDefaultValue:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrValue := Value.DefaultValue;
            LoadAttrValue(v);
            Value.DefaultValue := v;
          finally
            Pb.Pop;
          end;
        end;
      TAttrDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TAttrDef.ftHasMinimum:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HasMinimum := Pb.readBoolean;
        end;
      TAttrDef.ftMinimum:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Minimum := Pb.readInt64;
        end;
      TAttrDef.ftAllowedValues:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrValue := Value.AllowedValues;
            LoadAttrValue(v);
            Value.AllowedValues := v;
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

procedure TLoadHelper.LoadOpDef(var Value: TOpDef);
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
      TOpDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TOpDef.ftInputArgs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TArgDef;
            LoadArgDef(v);
            Value.InputArgs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftOutputArgs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TArgDef;
            LoadArgDef(v);
            Value.OutputArgs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftControlOutputs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.ControlOutputs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftAttrs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrDef;
            LoadAttrDef(v);
            Value.Attrs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftDeprecation:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOpDeprecation := Value.Deprecation;
            LoadOpDeprecation(v);
            Value.Deprecation := v;
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftSummary:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Summary := Pb.readString;
        end;
      TOpDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TOpDef.ftIsCommutative:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsCommutative := Pb.readBoolean;
        end;
      TOpDef.ftIsAggregate:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsAggregate := Pb.readBoolean;
        end;
      TOpDef.ftIsStateful:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsStateful := Pb.readBoolean;
        end;
      TOpDef.ftAllowsUninitializedInput:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllowsUninitializedInput := Pb.readBoolean;
        end;
      TOpDef.ftIsDistributedCommunication:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsDistributedCommunication := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOpDeprecation(var Value: TOpDeprecation);
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
      TOpDeprecation.ftVersion:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Version := Pb.readInt32;
        end;
      TOpDeprecation.ftExplanation:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Explanation := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOpList(var Value: TOpList);
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
      TOpList.ftOps:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOpDef;
            LoadOpDef(v);
            Value.Ops.Add(@v);
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

procedure TLoadHelper.LoadAllocationRecord(var Value: TAllocationRecord);
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
      TAllocationRecord.ftAllocMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocMicros := Pb.readInt64;
        end;
      TAllocationRecord.ftAllocBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocBytes := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadAllocatorMemoryUsed(var Value: TAllocatorMemoryUsed);
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
      TAllocatorMemoryUsed.ftAllocatorName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorName := Pb.readString;
        end;
      TAllocatorMemoryUsed.ftTotalBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TotalBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftPeakBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PeakBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftLiveBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LiveBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftAllocationRecordss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationRecord;
            LoadAllocationRecord(v);
            Value.AllocationRecordss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TAllocatorMemoryUsed.ftAllocatorBytesInUse:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocatorBytesInUse := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadNodeOutput(var Value: TNodeOutput);
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
      TNodeOutput.ftSlot:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Slot := Pb.readInt32;
        end;
      TNodeOutput.ftTensorDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorDescription := Value.TensorDescription;
            LoadTensorDescription(v);
            Value.TensorDescription := v;
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

procedure TLoadHelper.LoadMemoryStats(var Value: TMemoryStats);
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
      TMemoryStats.ftTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TempMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftPersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PersistentMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftPersistentTensorAllocIdss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.PersistentTensorAllocIdss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TMemoryStats.ftDeviceTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DeviceTempMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftDevicePersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DevicePersistentMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftDevicePersistentTensorAllocIdss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.DevicePersistentTensorAllocIdss.Add(@v);
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

procedure TLoadHelper.LoadNodeExecStats(var Value: TNodeExecStats);
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
      TNodeExecStats.ftNodeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NodeName := Pb.readString;
        end;
      TNodeExecStats.ftAllStartMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllStartMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftOpStartRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpStartRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftOpEndRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpEndRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftAllEndRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllEndRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftMemorys:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocatorMemoryUsed;
            LoadAllocatorMemoryUsed(v);
            Value.Memorys.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftOutputs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeOutput;
            LoadNodeOutput(v);
            Value.Outputs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftTimelineLabel:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TimelineLabel := Pb.readString;
        end;
      TNodeExecStats.ftScheduledMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScheduledMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftThreadId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ThreadId := Pb.readUint32;
        end;
      TNodeExecStats.ftReferencedTensors:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationDescription;
            LoadAllocationDescription(v);
            Value.ReferencedTensors.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftMemoryStats:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TMemoryStats := Value.MemoryStats;
            LoadMemoryStats(v);
            Value.MemoryStats := v;
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftAllStartNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllStartNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftOpStartRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpStartRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftOpEndRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpEndRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftAllEndRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllEndRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftScheduledNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScheduledNanos := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDeviceStepStats(var Value: TDeviceStepStats);
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
      TDeviceStepStats.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TDeviceStepStats.ftNodeStatss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeExecStats;
            LoadNodeExecStats(v);
            Value.NodeStatss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TDeviceStepStats.ftThreadNames:
        begin
          Value.ThreadNames.InsertOrAssign(TsgPair<UInt32, string>.From(Pb.readUint32, Pb.readString));
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadStepStats(var Value: TStepStats);
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
      TStepStats.ftDevStatss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDeviceStepStats;
            LoadDeviceStepStats(v);
            Value.DevStatss.Add(@v);
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
            Value.Jobs.Add(@v);
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
              Value.CoordinatedJobss.Add(@v);
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

procedure TLoadHelper.LoadDebugTensorWatch(var Value: TDebugTensorWatch);
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
      TDebugTensorWatch.ftNodeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NodeName := Pb.readString;
        end;
      TDebugTensorWatch.ftOutputSlot:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OutputSlot := Pb.readInt32;
        end;
      TDebugTensorWatch.ftDebugOpss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.DebugOpss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TDebugTensorWatch.ftDebugUrlss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.DebugUrlss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TDebugTensorWatch.ftTolerateDebugOpCreationFailures:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TolerateDebugOpCreationFailures := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDebugOptions(var Value: TDebugOptions);
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
      TDebugOptions.ftDebugTensorWatchOptss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDebugTensorWatch;
            LoadDebugTensorWatch(v);
            Value.DebugTensorWatchOptss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TDebugOptions.ftGlobalStep:
        begin
          Assert(wireType = TWire.VARINT);
          Value.GlobalStep := Pb.readInt64;
        end;
      TDebugOptions.ftResetDiskByteUsage:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ResetDiskByteUsage := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDebuggedSourceFile(var Value: TDebuggedSourceFile);
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
      TDebuggedSourceFile.ftHost:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Host := Pb.readString;
        end;
      TDebuggedSourceFile.ftFilePath:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.FilePath := Pb.readString;
        end;
      TDebuggedSourceFile.ftLastModified:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LastModified := Pb.readInt64;
        end;
      TDebuggedSourceFile.ftBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Bytes := Pb.readInt64;
        end;
      TDebuggedSourceFile.ftLiness:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.Liness.Add(@v);
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

procedure TLoadHelper.LoadDebuggedSourceFiles(var Value: TDebuggedSourceFiles);
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
      TDebuggedSourceFiles.ftSourceFiless:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDebuggedSourceFile;
            LoadDebuggedSourceFile(v);
            Value.SourceFiless.Add(@v);
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

procedure TLoadHelper.LoadAutoParallelOptions(var Value: TAutoParallelOptions);
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
      TAutoParallelOptions.ftEnable:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Enable := Pb.readBoolean;
        end;
      TAutoParallelOptions.ftNumReplicas:
        begin
          Assert(wireType = TWire.VARINT);
          Value.NumReplicas := Pb.readInt32;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadScopedAllocatorOptions(var Value: TScopedAllocatorOptions);
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
      TScopedAllocatorOptions.ftEnableOps:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.EnableOps.Add(@v);
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

procedure TLoadHelper.LoadCustomGraphOptimizer(var Value: TCustomGraphOptimizer);
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
      TCustomGraphOptimizer.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TCustomGraphOptimizer.ftParameterMap:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.ParameterMap.Add(Pb.readString, v1);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadRewriterConfig(var Value: TRewriterConfig);
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
      TRewriterConfig.ftCpuLayoutConversion:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CpuLayoutConversion := TCpuLayout(Pb.readInt32);
        end;
      TRewriterConfig.ftLayoutOptimizer:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LayoutOptimizer := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftConstantFolding:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ConstantFolding := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftShapeOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShapeOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftRemapping:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Remapping := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftCommonSubgraphElimination:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CommonSubgraphElimination := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftArithmeticOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ArithmeticOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDependencyOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DependencyOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftLoopOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LoopOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftFunctionOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.FunctionOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDebugStripper:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DebugStripper := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDisableModelPruning:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableModelPruning := Pb.readBoolean;
        end;
      TRewriterConfig.ftScopedAllocatorOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScopedAllocatorOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftPinToHostOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PinToHostOptimization := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftImplementationSelector:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ImplementationSelector := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecision:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecision := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecisionMkl:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecisionMkl := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecisionCpu:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecisionCpu := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDisableMetaOptimizer:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableMetaOptimizer := Pb.readBoolean;
        end;
      TRewriterConfig.ftUsePluginOptimizers:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UsePluginOptimizers := ProtoGen.RewriterConfig.TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftMetaOptimizerIterations:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MetaOptimizerIterations := TNumIterationsType(Pb.readInt32);
        end;
      TRewriterConfig.ftMinGraphNodes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MinGraphNodes := Pb.readInt32;
        end;
      TRewriterConfig.ftExperimentalDisableCompressedTensorOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ExperimentalDisableCompressedTensorOptimization := Pb.readBoolean;
        end;
      TRewriterConfig.ftExperimentalDisableFoldingQuantizationEmulation:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ExperimentalDisableFoldingQuantizationEmulation := Pb.readBoolean;
        end;
      TRewriterConfig.ftMemoryOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MemoryOptimization := TMemOptType(Pb.readInt32);
        end;
      TRewriterConfig.ftMemoryOptimizerTargetNodeNameScope:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.MemoryOptimizerTargetNodeNameScope := Pb.readString;
        end;
      TRewriterConfig.ftMetaOptimizerTimeoutMs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MetaOptimizerTimeoutMs := Pb.readInt64;
        end;
      TRewriterConfig.ftAutoParallel:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAutoParallelOptions := Value.AutoParallel;
            LoadAutoParallelOptions(v);
            Value.AutoParallel := v;
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftFailOnOptimizerErrors:
        begin
          Assert(wireType = TWire.VARINT);
          Value.FailOnOptimizerErrors := Pb.readBoolean;
        end;
      TRewriterConfig.ftScopedAllocatorOpts:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TScopedAllocatorOptions := Value.ScopedAllocatorOpts;
            LoadScopedAllocatorOptions(v);
            Value.ScopedAllocatorOpts := v;
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftOptimizerss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.Optimizerss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftCustomOptimizerss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCustomGraphOptimizer;
            LoadCustomGraphOptimizer(v);
            Value.CustomOptimizerss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftInterOptimizerVerifierConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVerifierConfig := Value.InterOptimizerVerifierConfig;
            LoadVerifierConfig(v);
            Value.InterOptimizerVerifierConfig := v;
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftPostOptimizationVerifierConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVerifierConfig := Value.PostOptimizationVerifierConfig;
            LoadVerifierConfig(v);
            Value.PostOptimizationVerifierConfig := v;
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

procedure TLoadHelper.LoadTensorProto(var Value: TTensorProto);
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
      TTensorProto.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TTensorProto.ftTensorShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.TensorShape;
            LoadTensorShapeProto(v);
            Value.TensorShape := v;
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftVersionNumber:
        begin
          Assert(wireType = TWire.VARINT);
          Value.VersionNumber := Pb.readInt32;
        end;
      TTensorProto.ftTensorContent:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TensorContent := Pb.readBytes;
        end;
      TTensorProto.ftHalfVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.HalfVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftFloatVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.FloatVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftDoubleVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : double := Pb.readDouble;
              Value.DoubleVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftIntVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.IntVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftStringVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TBytes := Pb.readBytes;
              Value.StringVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftScomplexVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.ScomplexVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftInt64Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.Int64Vals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftBoolVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : boolean := Pb.readBoolean;
              Value.BoolVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftDcomplexVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : double := Pb.readDouble;
              Value.DcomplexVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftResourceHandleVals:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TResourceHandleProto;
            LoadResourceHandleProto(v);
            Value.ResourceHandleVals.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftVariantVals:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVariantTensorDataProto;
            LoadVariantTensorDataProto(v);
            Value.VariantVals.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftUint32Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : uint32 := Pb.readUint32;
              Value.Uint32Vals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftUint64Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.Uint64Vals.Add(@v);
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

procedure TLoadHelper.LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
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
      TVariantTensorDataProto.ftTypeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeName := Pb.readString;
        end;
      TVariantTensorDataProto.ftMetadata:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Metadata := Pb.readBytes;
        end;
      TVariantTensorDataProto.ftTensorss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorProto;
            LoadTensorProto(v);
            Value.Tensorss.Add(@v);
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

procedure TLoadHelper.LoadFunctionDefLibrary(var Value: TFunctionDefLibrary);
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
      TFunctionDefLibrary.ftFunctions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFunctionDef;
            LoadFunctionDef(v);
            Value.Functions.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionDefLibrary.ftGradients:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGradientDef;
            LoadGradientDef(v);
            Value.Gradients.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionDefLibrary.ftRegisteredGradientss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRegisteredGradient;
            LoadRegisteredGradient(v);
            Value.RegisteredGradientss.Add(@v);
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

procedure TLoadHelper.LoadArgAttrs(var Value: TArgAttrs);
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
      TArgAttrs.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadFunctionDef(var Value: TFunctionDef);
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
      TFunctionDef.ftSignature:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOpDef := Value.Signature;
            LoadOpDef(v);
            Value.Signature := v;
          finally
            Pb.Pop;
          end;
        end;
      TFunctionDef.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      TFunctionDef.ftArgAttr:
        begin
          var v1 : TArgAttrs;
          LoadArgAttrs(v1);
          Value.ArgAttr.InsertOrAssign(TsgPair<UInt32, TArgAttrs>.From(Pb.readUint32, v1));
        end;
      TFunctionDef.ftResourceArgUniqueId:
        begin
          Value.ResourceArgUniqueId.InsertOrAssign(TsgPair<UInt32, UInt32>.From(Pb.readUint32, Pb.readUint32));
        end;
      TFunctionDef.ftNodeDefs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeDef;
            LoadNodeDef(v);
            Value.NodeDefs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionDef.ftRet:
        begin
          Value.Ret.InsertOrAssign(TsgPair<string, string>.From(Pb.readString, Pb.readString));
        end;
      TFunctionDef.ftControlRet:
        begin
          Value.ControlRet.InsertOrAssign(TsgPair<string, string>.From(Pb.readString, Pb.readString));
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadGradientDef(var Value: TGradientDef);
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
      TGradientDef.ftFunctionName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.FunctionName := Pb.readString;
        end;
      TGradientDef.ftGradientFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.GradientFunc := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadRegisteredGradient(var Value: TRegisteredGradient);
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
      TRegisteredGradient.ftGradientFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.GradientFunc := Pb.readString;
        end;
      TRegisteredGradient.ftRegisteredOpType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.RegisteredOpType := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
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
            Value.Argss.Add(@v);
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

procedure TLoadHelper.LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
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
      TExperimentalDebugInfo.ftOriginalNodeNamess:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.OriginalNodeNamess.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TExperimentalDebugInfo.ftOriginalFuncNamess:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.OriginalFuncNamess.Add(@v);
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

procedure TLoadHelper.LoadNodeDef(var Value: TNodeDef);
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
      TNodeDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TNodeDef.ftOp:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Op := Pb.readString;
        end;
      TNodeDef.ftInputs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.Inputs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TNodeDef.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TNodeDef.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      TNodeDef.ftExperimentalDebugInfo:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TExperimentalDebugInfo := Value.ExperimentalDebugInfo;
            LoadExperimentalDebugInfo(v);
            Value.ExperimentalDebugInfo := v;
          finally
            Pb.Pop;
          end;
        end;
      TNodeDef.ftExperimentalType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.ExperimentalType;
            LoadFullTypeDef(v);
            Value.ExperimentalType := v;
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
              Value.BadConsumerss.Add(@v);
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

procedure TLoadHelper.LoadAllocationDescription(var Value: TAllocationDescription);
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
      TAllocationDescription.ftRequestedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.RequestedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocatedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatorName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorName := Pb.readString;
        end;
      TAllocationDescription.ftAllocationId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocationId := Pb.readInt64;
        end;
      TAllocationDescription.ftHasSingleReference:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HasSingleReference := Pb.readBoolean;
        end;
      TAllocationDescription.ftPtr:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Ptr := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadTensorDescription(var Value: TTensorDescription);
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
      TTensorDescription.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TTensorDescription.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.Shape;
            LoadTensorShapeProto(v);
            Value.Shape := v;
          finally
            Pb.Pop;
          end;
        end;
      TTensorDescription.ftAllocationDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationDescription := Value.AllocationDescription;
            LoadAllocationDescription(v);
            Value.AllocationDescription := v;
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

class procedure TSaveHelper.SaveVirtualDevices(const S: TpbSaver; const Value: TVirtualDevices);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.MemoryLimitMbs.Count - 1 do
      h.Pb.writeRawData(Value.FMemoryLimitMbs[i], sizeof(Single));
    S.Pb.writeMessage(TVirtualDevices.ftMemoryLimitMbs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Prioritys.Count - 1 do
      h.Pb.writeRawVarint32(Value.FPrioritys[i]^);
    S.Pb.writeMessage(TVirtualDevices.ftPrioritys, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveExperimental(const S: TpbSaver; const Value: TExperimental);
begin
  if Value.FVirtualDevicess.Count > 0 then
    S.SaveList<TVirtualDevices>(Value.FVirtualDevicess, SaveVirtualDevices, TExperimental.ftVirtualDevicess);
  S.Pb.writeBoolean(TExperimental.ftUseUnifiedMemory, Value.UseUnifiedMemory);
  S.Pb.writeInt32(TExperimental.ftNumDevToDevCopyStreams, Value.NumDevToDevCopyStreams);
  S.Pb.writeString(TExperimental.ftCollectiveRingOrder, Value.CollectiveRingOrder);
  S.Pb.writeBoolean(TExperimental.ftTimestampedAllocator, Value.TimestampedAllocator);
  S.Pb.writeInt32(TExperimental.ftKernelTrackerMaxInterval, Value.KernelTrackerMaxInterval);
  S.Pb.writeInt32(TExperimental.ftKernelTrackerMaxBytes, Value.KernelTrackerMaxBytes);
  S.Pb.writeInt32(TExperimental.ftKernelTrackerMaxPending, Value.KernelTrackerMaxPending);
  S.Pb.writeDouble(TExperimental.ftInternalFragmentationFraction, Value.InternalFragmentationFraction);
  S.Pb.writeBoolean(TExperimental.ftUseCudaMallocAsync, Value.UseCudaMallocAsync);
end;

class procedure TSaveHelper.SaveGPUOptions(const S: TpbSaver; const Value: TGPUOptions);
begin
  S.Pb.writeDouble(TGPUOptions.ftPerProcessGpuMemoryFraction, Value.PerProcessGpuMemoryFraction);
  S.Pb.writeBoolean(TGPUOptions.ftAllowGrowth, Value.AllowGrowth);
  S.Pb.writeString(TGPUOptions.ftAllocatorType, Value.AllocatorType);
  S.Pb.writeInt64(TGPUOptions.ftDeferredDeletionBytes, Value.DeferredDeletionBytes);
  S.Pb.writeString(TGPUOptions.ftVisibleDeviceList, Value.VisibleDeviceList);
  S.Pb.writeInt32(TGPUOptions.ftPollingActiveDelayUsecs, Value.PollingActiveDelayUsecs);
  S.Pb.writeInt32(TGPUOptions.ftPollingInactiveDelayMsecs, Value.PollingInactiveDelayMsecs);
  S.Pb.writeBoolean(TGPUOptions.ftForceGpuCompatible, Value.ForceGpuCompatible);
  S.SaveObj<TExperimental>(Value.FExperimental, SaveExperimental, TGPUOptions.ftExperimental);
end;

class procedure TSaveHelper.SaveOptimizerOptions(const S: TpbSaver; const Value: TOptimizerOptions);

begin
  S.Pb.writeBoolean(TOptimizerOptions.ftDoCommonSubexpressionElimination, Value.DoCommonSubexpressionElimination);
  S.Pb.writeBoolean(TOptimizerOptions.ftDoConstantFolding, Value.DoConstantFolding);
  S.Pb.writeInt64(TOptimizerOptions.ftMaxFoldedConstantInBytes, Value.MaxFoldedConstantInBytes);
  S.Pb.writeBoolean(TOptimizerOptions.ftDoFunctionInlining, Value.DoFunctionInlining);
  S.Pb.writeInt32(TOptimizerOptions.ftOptLevel, Ord(Value.OptLevel));
  S.Pb.writeInt32(TOptimizerOptions.ftGlobalJitLevel, Ord(Value.GlobalJitLevel));
  S.Pb.writeBoolean(TOptimizerOptions.ftCpuGlobalJit, Value.CpuGlobalJit);
end;

class procedure TSaveHelper.SaveGraphOptions(const S: TpbSaver; const Value: TGraphOptions);

begin
  S.Pb.writeBoolean(TGraphOptions.ftEnableRecvScheduling, Value.EnableRecvScheduling);
  S.SaveObj<TOptimizerOptions>(Value.FOptimizerOptions, SaveOptimizerOptions, TGraphOptions.ftOptimizerOptions);
  S.Pb.writeInt64(TGraphOptions.ftBuildCostModel, Value.BuildCostModel);
  S.Pb.writeInt64(TGraphOptions.ftBuildCostModelAfter, Value.BuildCostModelAfter);
  S.Pb.writeBoolean(TGraphOptions.ftInferShapes, Value.InferShapes);
  S.Pb.writeBoolean(TGraphOptions.ftPlacePrunedGraph, Value.PlacePrunedGraph);
  S.Pb.writeBoolean(TGraphOptions.ftEnableBfloat16Sendrecv, Value.EnableBfloat16Sendrecv);
  S.Pb.writeInt32(TGraphOptions.ftTimelineStep, Value.TimelineStep);
  S.SaveObj<TRewriterConfig>(Value.FRewriteOptions, SaveRewriterConfig, TGraphOptions.ftRewriteOptions);
end;

class procedure TSaveHelper.SaveThreadPoolOptionProto(const S: TpbSaver; const Value: TThreadPoolOptionProto);

begin
  S.Pb.writeInt32(TThreadPoolOptionProto.ftNumThreads, Value.NumThreads);
  S.Pb.writeString(TThreadPoolOptionProto.ftGlobalName, Value.GlobalName);
end;

class procedure TSaveHelper.SaveRPCOptions(const S: TpbSaver; const Value: TRPCOptions);

begin
  S.Pb.writeBoolean(TRPCOptions.ftUseRpcForInprocessMaster, Value.UseRpcForInprocessMaster);
  S.Pb.writeString(TRPCOptions.ftCompressionAlgorithm, Value.CompressionAlgorithm);
  S.Pb.writeInt32(TRPCOptions.ftCompressionLevel, Value.CompressionLevel);
  S.Pb.writeBoolean(TRPCOptions.ftCacheRpcResponse, Value.CacheRpcResponse);
  S.Pb.writeBoolean(TRPCOptions.ftDisableSessionConnectionSharing, Value.DisableSessionConnectionSharing);
  S.Pb.writeInt32(TRPCOptions.ftNumChannelsPerTarget, Value.NumChannelsPerTarget);
end;

class procedure TSaveHelper.SaveSessionMetadata(const S: TpbSaver; const Value: TSessionMetadata);

begin
  S.Pb.writeString(TSessionMetadata.ftName, Value.Name);
  S.Pb.writeInt64(TSessionMetadata.ftVersion, Value.Version);
end;

class procedure TSaveHelper.SaveExperimental_Config(const S: TpbSaver; const Value: TExperimental_Config);

begin
  S.Pb.writeString(TExperimental_Config.ftCollectiveGroupLeader, Value.CollectiveGroupLeader);
  S.Pb.writeString(TExperimental_Config.ftExecutorType, Value.ExecutorType);
  S.Pb.writeInt32(TExperimental_Config.ftRecvBufMaxChunk, Value.RecvBufMaxChunk);
  S.Pb.writeBoolean(TExperimental_Config.ftUseNumaAffinity, Value.UseNumaAffinity);
  S.Pb.writeBoolean(TExperimental_Config.ftCollectiveDeterministicSequentialExecution, Value.CollectiveDeterministicSequentialExecution);
  S.Pb.writeBoolean(TExperimental_Config.ftCollectiveNccl, Value.CollectiveNccl);
  S.Pb.writeBoolean(TExperimental_Config.ftShareSessionStateInClusterspecPropagation, Value.ShareSessionStateInClusterspecPropagation);
  S.Pb.writeBoolean(TExperimental_Config.ftDisableThreadSpinning, Value.DisableThreadSpinning);
  S.Pb.writeBoolean(TExperimental_Config.ftShareClusterDevicesInSession, Value.ShareClusterDevicesInSession);
  S.SaveObj<TSessionMetadata>(Value.FSessionMetadata, SaveSessionMetadata, TExperimental_Config.ftSessionMetadata);
  S.Pb.writeBoolean(TExperimental_Config.ftOptimizeForStaticGraph, Value.OptimizeForStaticGraph);
  S.Pb.writeBoolean(TExperimental_Config.ftEnableMlirBridge, Value.EnableMlirBridge);
  S.Pb.writeInt32(TExperimental_Config.ftMlirBridgeRollout, Ord(Value.MlirBridgeRollout));
  S.Pb.writeBoolean(TExperimental_Config.ftEnableMlirGraphOptimization, Value.EnableMlirGraphOptimization);
  S.Pb.writeBoolean(TExperimental_Config.ftDisableOutputPartitionGraphs, Value.DisableOutputPartitionGraphs);
  S.Pb.writeInt64(TExperimental_Config.ftXlaFusionAutotunerThresh, Value.XlaFusionAutotunerThresh);
  S.Pb.writeBoolean(TExperimental_Config.ftUseTfrt, Value.UseTfrt);
  S.Pb.writeBoolean(TExperimental_Config.ftDisableFunctionalOpsLowering, Value.DisableFunctionalOpsLowering);
  S.Pb.writeBoolean(TExperimental_Config.ftXlaPreferSingleGraphCluster, Value.XlaPreferSingleGraphCluster);
  S.SaveObj<TCoordinationServiceConfig>(Value.FCoordinationConfig, SaveCoordinationServiceConfig, TExperimental_Config.ftCoordinationConfig);
end;

class procedure TSaveHelper.SaveConfigProto(const S: TpbSaver; const Value: TConfigProto);
var
  i : Integer;
  h : TpbSaver;

begin
  if Value.FDeviceCount.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.FDeviceCount.Begins;
      while it <> Value.FDeviceCount.Ends do
      begin
          h.clear;
          h.SaveStringInt32(it.GetPair^);
          S.Pb.writeMessage(TConfigProto.ftDeviceCount, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  S.Pb.writeInt32(TConfigProto.ftIntraOpParallelismThreads, Value.IntraOpParallelismThreads);
  S.Pb.writeInt32(TConfigProto.ftInterOpParallelismThreads, Value.InterOpParallelismThreads);
  S.Pb.writeBoolean(TConfigProto.ftUsePerSessionThreads, Value.UsePerSessionThreads);
  if Value.FSessionInterOpThreadPools.Count > 0 then
    S.SaveList<TThreadPoolOptionProto>(Value.FSessionInterOpThreadPools, SaveThreadPoolOptionProto, TConfigProto.ftSessionInterOpThreadPools);
  S.Pb.writeInt32(TConfigProto.ftPlacementPeriod, Value.PlacementPeriod);
  h.Init;
  try
    for i := 0 to Value.DeviceFilterss.Count - 1 do
      h.Pb.writeRawString(Value.FDeviceFilterss[i]^);
    S.Pb.writeMessage(TConfigProto.ftDeviceFilterss, h.Pb^);
  finally
    h.Free;
  end;
  S.SaveObj<TGPUOptions>(Value.FGpuOptions, SaveGPUOptions, TConfigProto.ftGpuOptions);
  S.Pb.writeBoolean(TConfigProto.ftAllowSoftPlacement, Value.AllowSoftPlacement);
  S.Pb.writeBoolean(TConfigProto.ftLogDevicePlacement, Value.LogDevicePlacement);
  S.SaveObj<TGraphOptions>(Value.FGraphOptions, SaveGraphOptions, TConfigProto.ftGraphOptions);
  S.Pb.writeInt64(TConfigProto.ftOperationTimeoutInMs, Value.OperationTimeoutInMs);
  S.SaveObj<TRPCOptions>(Value.FRpcOptions, SaveRPCOptions, TConfigProto.ftRpcOptions);
  S.SaveObj<TClusterDef>(Value.FClusterDef, SaveClusterDef, TConfigProto.ftClusterDef);
  S.Pb.writeBoolean(TConfigProto.ftIsolateSessionState, Value.IsolateSessionState);
  S.Pb.writeBoolean(TConfigProto.ftShareClusterDevicesInSession, Value.ShareClusterDevicesInSession);
  S.SaveObj<TExperimental_Config>(Value.FExperimental, SaveExperimental_Config, TConfigProto.ftExperimental);
end;

class procedure TSaveHelper.SaveRunHandlerPoolOptions(const S: TpbSaver; const Value: TRunHandlerPoolOptions);
begin
  S.Pb.writeInt64(TRunHandlerPoolOptions.ftPriority, Value.Priority);
end;

class procedure TSaveHelper.SaveExperimental_Option(const S: TpbSaver; const Value: TExperimental_Option);
begin
  S.Pb.writeInt64(TExperimental_Option.ftCollectiveGraphKey, Value.CollectiveGraphKey);
  S.Pb.writeBoolean(TExperimental_Option.ftUseRunHandlerPool, Value.UseRunHandlerPool);
  S.SaveObj<TRunHandlerPoolOptions>(Value.FRunHandlerPoolOptions, SaveRunHandlerPoolOptions, TExperimental_Option.ftRunHandlerPoolOptions);
end;

class procedure TSaveHelper.SaveRunOptions(const S: TpbSaver; const Value: TRunOptions);
begin
  S.Pb.writeInt32(TRunOptions.ftTraceLevel, Ord(Value.TraceLevel));
  S.Pb.writeInt64(TRunOptions.ftTimeoutInMs, Value.TimeoutInMs);
  S.Pb.writeInt32(TRunOptions.ftInterOpThreadPool, Value.InterOpThreadPool);
  S.Pb.writeBoolean(TRunOptions.ftOutputPartitionGraphs, Value.OutputPartitionGraphs);
  S.SaveObj<TDebugOptions>(Value.FDebugOptions, SaveDebugOptions, TRunOptions.ftDebugOptions);
  S.Pb.writeBoolean(TRunOptions.ftReportTensorAllocationsUponOom, Value.ReportTensorAllocationsUponOom);
  S.SaveObj<TExperimental_Option>(Value.FExperimental, SaveExperimental_Option, TRunOptions.ftExperimental);
end;

class procedure TSaveHelper.SaveFunctionGraphs(const S: TpbSaver; const Value: TFunctionGraphs);
begin
  if Value.FPartitionGraphss.Count > 0 then
    S.SaveList<TGraphDef>(Value.FPartitionGraphss, SaveGraphDef, TFunctionGraphs.ftPartitionGraphss);
  S.SaveObj<TGraphDef>(Value.FPreOptimizationGraph, SaveGraphDef, TFunctionGraphs.ftPreOptimizationGraph);
  S.SaveObj<TGraphDef>(Value.FPostOptimizationGraph, SaveGraphDef, TFunctionGraphs.ftPostOptimizationGraph);
end;

class procedure TSaveHelper.SaveRunMetadata(const S: TpbSaver; const Value: TRunMetadata);
begin
  S.SaveObj<TStepStats>(Value.FStepStats, SaveStepStats, TRunMetadata.ftStepStats);
  S.SaveObj<TCostGraphDef>(Value.FCostGraph, SaveCostGraphDef, TRunMetadata.ftCostGraph);
  if Value.FPartitionGraphss.Count > 0 then
    S.SaveList<TGraphDef>(Value.FPartitionGraphss, SaveGraphDef, TRunMetadata.ftPartitionGraphss);
  if Value.FFunctionGraphss.Count > 0 then
    S.SaveList<TFunctionGraphs>(Value.FFunctionGraphss, SaveFunctionGraphs, TRunMetadata.ftFunctionGraphss);
end;

class procedure TSaveHelper.SaveTensorConnection(const S: TpbSaver; const Value: TTensorConnection);
begin
  S.Pb.writeString(TTensorConnection.ftFromTensor, Value.FromTensor);
  S.Pb.writeString(TTensorConnection.ftToTensor, Value.ToTensor);
end;

class procedure TSaveHelper.SaveCallableOptions(const S: TpbSaver; const Value: TCallableOptions);
var
  i : Integer;
  h : TpbSaver;
begin
  h.Init;
  try
    for i := 0 to Value.Feeds.Count - 1 do
      h.Pb.writeRawString(Value.FFeeds[i]^);
    S.Pb.writeMessage(TCallableOptions.ftFeeds, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Fetchs.Count - 1 do
      h.Pb.writeRawString(Value.FFetchs[i]^);
    S.Pb.writeMessage(TCallableOptions.ftFetchs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Targets.Count - 1 do
      h.Pb.writeRawString(Value.FTargets[i]^);
    S.Pb.writeMessage(TCallableOptions.ftTargets, h.Pb^);
  finally
    h.Free;
  end;
  S.SaveObj<TRunOptions>(Value.FRunOptions, SaveRunOptions, TCallableOptions.ftRunOptions);
  if Value.FTensorConnections.Count > 0 then
    S.SaveList<TTensorConnection>(Value.FTensorConnections, SaveTensorConnection, TCallableOptions.ftTensorConnections);
  if Value.FFeedDevices.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.FFeedDevices.Begins;
      while it <> Value.FFeedDevices.Ends do
      begin
          h.clear;
          h.SaveStringString(it.GetPair^);
          S.Pb.writeMessage(TCallableOptions.ftFeedDevices, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  if Value.FFetchDevices.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.FFetchDevices.Begins;
      while it <> Value.FFetchDevices.Ends do
      begin
          h.clear;
          h.SaveStringString(it.GetPair^);
          S.Pb.writeMessage(TCallableOptions.ftFetchDevices, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  S.Pb.writeBoolean(TCallableOptions.ftFetchSkipSync, Value.FetchSkipSync);
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
  S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TOutputInfo.ftShape);
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
  if Value.InputInfos.Count > 0 then
    S.SaveList<TInputInfo>(Value.InputInfos, SaveInputInfo, TNode.ftInputInfos);
  if Value.OutputInfos.Count > 0 then
    S.SaveList<TOutputInfo>(Value.OutputInfos, SaveOutputInfo, TNode.ftOutputInfos);
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
      h.Pb.writeRawVarint32(Value.ControlInputs[i]^);
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
  if Value.Nodes.Count > 0 then
    S.SaveList<TNode>(Value.Nodes, SaveNode, TCostGraphDef.ftNodes);
  if Value.Costs.Count > 0 then
    S.SaveList<TAggregatedCost>(Value.Costs, SaveAggregatedCost, TCostGraphDef.ftCosts);
end;

class procedure TSaveHelper.SaveGraphDef(const S: TpbSaver; const Value: TGraphDef);
begin
  if Value.Nodes.Count > 0 then
    S.SaveList<TNodeDef>(Value.Nodes, SaveNodeDef, TGraphDef.ftNodes);
  S.SaveObj<TVersionDef>(Value.Versions, SaveVersionDef, TGraphDef.ftVersions);
  S.Pb.writeInt32(TGraphDef.ftVersion, Value.Version);
  S.SaveObj<TFunctionDefLibrary>(Value.&Library, SaveFunctionDefLibrary, TGraphDef.ftLibrary);
end;

class procedure TSaveHelper.SaveListValue(const S: TpbSaver; const Value: TListValue);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.Ss.Count - 1 do
      h.Pb.writeRawBytes(Value.Ss[i]^);
    S.Pb.writeMessage(TListValue.ftSs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.&Is.Count - 1 do
      h.Pb.writeRawVarint64(Value.&Is[i]^);
    S.Pb.writeMessage(TListValue.ftIs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Fs.Count - 1 do
      h.Pb.writeRawData(Value.Fs[i], sizeof(Single));
    S.Pb.writeMessage(TListValue.ftFs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Bs.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.Bs[i]^));
    S.Pb.writeMessage(TListValue.ftBs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.&Types.Count - 1 do
      h.Pb.writeRawVarint32(Ord(Value.&Types[i]^));
    S.Pb.writeMessage(TListValue.ftTypes, h.Pb^);
  finally
    h.Free;
  end;
  if Value.Shapes.Count > 0 then
    S.SaveList<TTensorShapeProto>(Value.Shapes, SaveTensorShapeProto, TListValue.ftShapes);
  if Value.Tensors.Count > 0 then
    S.SaveList<TTensorProto>(Value.Tensors, SaveTensorProto, TListValue.ftTensors);
  if Value.Funcs.Count > 0 then
    S.SaveList<TNameAttrList>(Value.Funcs, SaveNameAttrList, TListValue.ftFuncs);
end;

class procedure TSaveHelper.SaveAttrValue(const S: TpbSaver; const Value: TAttrValue);

begin
  case Value.value.tag of
    TAttrValue.ftS:
      begin
        S.Pb.writeBytes(Value.ftS, Value.Value.value.AsType<TBytes>);
      end;
    TAttrValue.ftI:
      begin
        S.Pb.writeInt64(Value.ftI, Value.Value.value.AsType<Int64>);
      end;
    TAttrValue.ftF:
      begin
        S.Pb.writeFloat(Value.ftF, Value.Value.value.AsType<Single>);
      end;
    TAttrValue.ftB:
      begin
        S.Pb.writeBoolean(Value.ftB, Value.Value.value.AsType<Boolean>);
      end;
    TAttrValue.ftType:
      begin
        S.Pb.writeInt32(Value.ftType, Ord(Value.Value.value.AsType<TDataType>));
      end;
    TAttrValue.ftShape:
      begin
        S.SaveObj<TTensorShapeProto>(Value.Value.value.AsType<TTensorShapeProto>, SaveTensorShapeProto, Value.ftShape);
      end;
    TAttrValue.ftTensor:
      begin
        S.SaveObj<TTensorProto>(Value.Value.value.AsType<TTensorProto>, SaveTensorProto, Value.ftTensor);
      end;
    TAttrValue.ftList:
      begin
        S.SaveObj<TListValue>(Value.Value.value.AsType<TListValue>, SaveListValue, Value.ftList);
      end;
    TAttrValue.ftFunc:
      begin
        S.SaveObj<TNameAttrList>(Value.Value.value.AsType<TNameAttrList>, SaveNameAttrList, Value.ftFunc);
      end;
    TAttrValue.ftPlaceholder:
      begin
        S.Pb.writeString(Value.ftPlaceholder, Value.Value.value.AsType<string>);
      end;
  end;
end;

class procedure TSaveHelper.SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TNameAttrList.ftName, Value.Name);
  if Value.Attr.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TNameAttrList.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape);
begin
  S.Pb.writeInt32(TDtypeAndShape.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TDtypeAndShape.ftShape);
end;

class procedure TSaveHelper.SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto);
begin
  S.Pb.writeString(TResourceHandleProto.ftDevice, Value.Device);
  S.Pb.writeString(TResourceHandleProto.ftContainer, Value.Container);
  S.Pb.writeString(TResourceHandleProto.ftName, Value.Name);
  S.Pb.writeInt64(TResourceHandleProto.ftHashCode, Value.HashCode);
  S.Pb.writeString(TResourceHandleProto.ftMaybeTypeName, Value.MaybeTypeName);
  if Value.DtypesAndShapess.Count > 0 then
    S.SaveList<TDtypeAndShape>(Value.DtypesAndShapess, SaveDtypeAndShape, TResourceHandleProto.ftDtypesAndShapess);
end;

class procedure TSaveHelper.SaveArgDef(const S: TpbSaver; const Value: TArgDef);
begin
  S.Pb.writeString(TArgDef.ftName, Value.Name);
  S.Pb.writeString(TArgDef.ftDescription, Value.Description);
  S.Pb.writeInt32(TArgDef.ftType, Ord(Value.&Type));
  S.Pb.writeString(TArgDef.ftTypeAttr, Value.TypeAttr);
  S.Pb.writeString(TArgDef.ftNumberAttr, Value.NumberAttr);
  S.Pb.writeString(TArgDef.ftTypeListAttr, Value.TypeListAttr);
  if Value.HandleDatas.Count > 0 then
    S.SaveList<TResourceHandleProto>(Value.HandleDatas, SaveResourceHandleProto, TArgDef.ftHandleDatas);
  S.Pb.writeBoolean(TArgDef.ftIsRef, Value.IsRef);
  S.SaveObj<TFullTypeDef>(Value.ExperimentalFullType, SaveFullTypeDef, TArgDef.ftExperimentalFullType);
end;

class procedure TSaveHelper.SaveAttrDef(const S: TpbSaver; const Value: TAttrDef);
begin
  S.Pb.writeString(TAttrDef.ftName, Value.Name);
  S.Pb.writeString(TAttrDef.ftType, Value.&Type);
  S.SaveObj<TAttrValue>(Value.DefaultValue, SaveAttrValue, TAttrDef.ftDefaultValue);
  S.Pb.writeString(TAttrDef.ftDescription, Value.Description);
  S.Pb.writeBoolean(TAttrDef.ftHasMinimum, Value.HasMinimum);
  S.Pb.writeInt64(TAttrDef.ftMinimum, Value.Minimum);
  S.SaveObj<TAttrValue>(Value.AllowedValues, SaveAttrValue, TAttrDef.ftAllowedValues);
end;

class procedure TSaveHelper.SaveOpDef(const S: TpbSaver; const Value: TOpDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TOpDef.ftName, Value.Name);
  if Value.InputArgs.Count > 0 then
    S.SaveList<TArgDef>(Value.InputArgs, SaveArgDef, TOpDef.ftInputArgs);
  if Value.OutputArgs.Count > 0 then
    S.SaveList<TArgDef>(Value.OutputArgs, SaveArgDef, TOpDef.ftOutputArgs);
  h.Init;
  try
    for i := 0 to Value.ControlOutputs.Count - 1 do
      h.Pb.writeRawString(Value.ControlOutputs[i]^);
    S.Pb.writeMessage(TOpDef.ftControlOutputs, h.Pb^);
  finally
    h.Free;
  end;
  if Value.Attrs.Count > 0 then
    S.SaveList<TAttrDef>(Value.Attrs, SaveAttrDef, TOpDef.ftAttrs);
  S.SaveObj<TOpDeprecation>(Value.Deprecation, SaveOpDeprecation, TOpDef.ftDeprecation);
  S.Pb.writeString(TOpDef.ftSummary, Value.Summary);
  S.Pb.writeString(TOpDef.ftDescription, Value.Description);
  S.Pb.writeBoolean(TOpDef.ftIsCommutative, Value.IsCommutative);
  S.Pb.writeBoolean(TOpDef.ftIsAggregate, Value.IsAggregate);
  S.Pb.writeBoolean(TOpDef.ftIsStateful, Value.IsStateful);
  S.Pb.writeBoolean(TOpDef.ftAllowsUninitializedInput, Value.AllowsUninitializedInput);
  S.Pb.writeBoolean(TOpDef.ftIsDistributedCommunication, Value.IsDistributedCommunication);
end;

class procedure TSaveHelper.SaveOpDeprecation(const S: TpbSaver; const Value: TOpDeprecation);
begin
  S.Pb.writeInt32(TOpDeprecation.ftVersion, Value.Version);
  S.Pb.writeString(TOpDeprecation.ftExplanation, Value.Explanation);
end;

class procedure TSaveHelper.SaveOpList(const S: TpbSaver; const Value: TOpList);
begin
  if Value.Ops.Count > 0 then
    S.SaveList<TOpDef>(Value.Ops, SaveOpDef, TOpList.ftOps);
end;

class procedure TSaveHelper.SaveAllocationRecord(const S: TpbSaver; const Value: TAllocationRecord);
begin
  S.Pb.writeInt64(TAllocationRecord.ftAllocMicros, Value.AllocMicros);
  S.Pb.writeInt64(TAllocationRecord.ftAllocBytes, Value.AllocBytes);
end;

class procedure TSaveHelper.SaveAllocatorMemoryUsed(const S: TpbSaver; const Value: TAllocatorMemoryUsed);
begin
  S.Pb.writeString(TAllocatorMemoryUsed.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftTotalBytes, Value.TotalBytes);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftPeakBytes, Value.PeakBytes);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftLiveBytes, Value.LiveBytes);
  if Value.AllocationRecordss.Count > 0 then
    S.SaveList<TAllocationRecord>(Value.AllocationRecordss, SaveAllocationRecord, TAllocatorMemoryUsed.ftAllocationRecordss);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftAllocatorBytesInUse, Value.AllocatorBytesInUse);
end;

class procedure TSaveHelper.SaveNodeOutput(const S: TpbSaver; const Value: TNodeOutput);
begin
  S.Pb.writeInt32(TNodeOutput.ftSlot, Value.Slot);
  S.SaveObj<TTensorDescription>(Value.TensorDescription, SaveTensorDescription, TNodeOutput.ftTensorDescription);
end;

class procedure TSaveHelper.SaveMemoryStats(const S: TpbSaver; const Value: TMemoryStats);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt64(TMemoryStats.ftTempMemorySize, Value.TempMemorySize);
  S.Pb.writeInt64(TMemoryStats.ftPersistentMemorySize, Value.PersistentMemorySize);
  h.Init;
  try
    for i := 0 to Value.PersistentTensorAllocIdss.Count - 1 do
      h.Pb.writeRawVarint64(Value.PersistentTensorAllocIdss[i]^);
    S.Pb.writeMessage(TMemoryStats.ftPersistentTensorAllocIdss, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeInt64(TMemoryStats.ftDeviceTempMemorySize, Value.DeviceTempMemorySize);
  S.Pb.writeInt64(TMemoryStats.ftDevicePersistentMemorySize, Value.DevicePersistentMemorySize);
  h.Init;
  try
    for i := 0 to Value.DevicePersistentTensorAllocIdss.Count - 1 do
      h.Pb.writeRawVarint64(Value.DevicePersistentTensorAllocIdss[i]^);
    S.Pb.writeMessage(TMemoryStats.ftDevicePersistentTensorAllocIdss, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveNodeExecStats(const S: TpbSaver; const Value: TNodeExecStats);
begin
  S.Pb.writeString(TNodeExecStats.ftNodeName, Value.NodeName);
  S.Pb.writeInt64(TNodeExecStats.ftAllStartMicros, Value.AllStartMicros);
  S.Pb.writeInt64(TNodeExecStats.ftOpStartRelMicros, Value.OpStartRelMicros);
  S.Pb.writeInt64(TNodeExecStats.ftOpEndRelMicros, Value.OpEndRelMicros);
  S.Pb.writeInt64(TNodeExecStats.ftAllEndRelMicros, Value.AllEndRelMicros);
  if Value.Memorys.Count > 0 then
    S.SaveList<TAllocatorMemoryUsed>(Value.Memorys, SaveAllocatorMemoryUsed, TNodeExecStats.ftMemorys);
  if Value.Outputs.Count > 0 then
    S.SaveList<TNodeOutput>(Value.Outputs, SaveNodeOutput, TNodeExecStats.ftOutputs);
  S.Pb.writeString(TNodeExecStats.ftTimelineLabel, Value.TimelineLabel);
  S.Pb.writeInt64(TNodeExecStats.ftScheduledMicros, Value.ScheduledMicros);
  S.Pb.writeInt32(TNodeExecStats.ftThreadId, Value.ThreadId);
  if Value.ReferencedTensors.Count > 0 then
    S.SaveList<TAllocationDescription>(Value.ReferencedTensors, SaveAllocationDescription, TNodeExecStats.ftReferencedTensors);
  S.SaveObj<TMemoryStats>(Value.MemoryStats, SaveMemoryStats, TNodeExecStats.ftMemoryStats);
  S.Pb.writeInt64(TNodeExecStats.ftAllStartNanos, Value.AllStartNanos);
  S.Pb.writeInt64(TNodeExecStats.ftOpStartRelNanos, Value.OpStartRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftOpEndRelNanos, Value.OpEndRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftAllEndRelNanos, Value.AllEndRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftScheduledNanos, Value.ScheduledNanos);
end;

class procedure TSaveHelper.SaveDeviceStepStats(const S: TpbSaver; const Value: TDeviceStepStats);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TDeviceStepStats.ftDevice, Value.Device);
  if Value.NodeStatss.Count > 0 then
    S.SaveList<TNodeExecStats>(Value.NodeStatss, SaveNodeExecStats, TDeviceStepStats.ftNodeStatss);
  if Value.ThreadNames.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.ThreadNames.Begins;
      while it <> Value.ThreadNames.Ends do
      begin
          h.clear;
          h.SaveUint32String(it.GetPair^);
          S.Pb.writeMessage(TDeviceStepStats.ftThreadNames, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveStepStats(const S: TpbSaver; const Value: TStepStats);

begin
  if Value.DevStatss.Count > 0 then
    S.SaveList<TDeviceStepStats>(Value.DevStatss, SaveDeviceStepStats, TStepStats.ftDevStatss);
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

class procedure TSaveHelper.SaveJobDef(const S: TpbSaver; const Value: TJobDef);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TJobDef.ftName, Value.Name);
  if Value.Tasks.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.Tasks.Begins;
      while it <> Value.Tasks.Ends do
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
  if Value.Jobs.Count > 0 then
    S.SaveList<TJobDef>(Value.Jobs, SaveJobDef, TClusterDef.ftJobs);
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
      h.Pb.writeRawString(Value.CoordinatedJobss[i]^);
    S.Pb.writeMessage(TCoordinationServiceConfig.ftCoordinatedJobss, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveDebugTensorWatch(const S: TpbSaver; const Value: TDebugTensorWatch);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TDebugTensorWatch.ftNodeName, Value.NodeName);
  S.Pb.writeInt32(TDebugTensorWatch.ftOutputSlot, Value.OutputSlot);
  h.Init;
  try
    for i := 0 to Value.DebugOpss.Count - 1 do
      h.Pb.writeRawString(Value.DebugOpss[i]^);
    S.Pb.writeMessage(TDebugTensorWatch.ftDebugOpss, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DebugUrlss.Count - 1 do
      h.Pb.writeRawString(Value.DebugUrlss[i]^);
    S.Pb.writeMessage(TDebugTensorWatch.ftDebugUrlss, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeBoolean(TDebugTensorWatch.ftTolerateDebugOpCreationFailures, Value.TolerateDebugOpCreationFailures);
end;

class procedure TSaveHelper.SaveDebugOptions(const S: TpbSaver; const Value: TDebugOptions);
begin
  if Value.DebugTensorWatchOptss.Count > 0 then
    S.SaveList<TDebugTensorWatch>(Value.DebugTensorWatchOptss, SaveDebugTensorWatch, TDebugOptions.ftDebugTensorWatchOptss);
  S.Pb.writeInt64(TDebugOptions.ftGlobalStep, Value.GlobalStep);
  S.Pb.writeBoolean(TDebugOptions.ftResetDiskByteUsage, Value.ResetDiskByteUsage);
end;

class procedure TSaveHelper.SaveDebuggedSourceFile(const S: TpbSaver; const Value: TDebuggedSourceFile);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TDebuggedSourceFile.ftHost, Value.Host);
  S.Pb.writeString(TDebuggedSourceFile.ftFilePath, Value.FilePath);
  S.Pb.writeInt64(TDebuggedSourceFile.ftLastModified, Value.LastModified);
  S.Pb.writeInt64(TDebuggedSourceFile.ftBytes, Value.Bytes);
  h.Init;
  try
    for i := 0 to Value.Liness.Count - 1 do
      h.Pb.writeRawString(Value.Liness[i]^);
    S.Pb.writeMessage(TDebuggedSourceFile.ftLiness, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveDebuggedSourceFiles(const S: TpbSaver; const Value: TDebuggedSourceFiles);

begin
  if Value.SourceFiless.Count > 0 then
    S.SaveList<TDebuggedSourceFile>(Value.SourceFiless, SaveDebuggedSourceFile, TDebuggedSourceFiles.ftSourceFiless);
end;

class procedure TSaveHelper.SaveAutoParallelOptions(const S: TpbSaver; const Value: TAutoParallelOptions);
begin
  S.Pb.writeBoolean(TAutoParallelOptions.ftEnable, Value.Enable);
  S.Pb.writeInt32(TAutoParallelOptions.ftNumReplicas, Value.NumReplicas);
end;

class procedure TSaveHelper.SaveScopedAllocatorOptions(const S: TpbSaver; const Value: TScopedAllocatorOptions);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.EnableOps.Count - 1 do
      h.Pb.writeRawString(Value.EnableOps[i]^);
    S.Pb.writeMessage(TScopedAllocatorOptions.ftEnableOps, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveCustomGraphOptimizer(const S: TpbSaver; const Value: TCustomGraphOptimizer);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TCustomGraphOptimizer.ftName, Value.Name);
  if Value.ParameterMap.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.ParameterMap do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TCustomGraphOptimizer.ftParameterMap, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveRewriterConfig(const S: TpbSaver; const Value: TRewriterConfig);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TRewriterConfig.ftCpuLayoutConversion, Ord(Value.CpuLayoutConversion));
  S.Pb.writeInt32(TRewriterConfig.ftLayoutOptimizer, Ord(Value.LayoutOptimizer));
  S.Pb.writeInt32(TRewriterConfig.ftConstantFolding, Ord(Value.ConstantFolding));
  S.Pb.writeInt32(TRewriterConfig.ftShapeOptimization, Ord(Value.ShapeOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftRemapping, Ord(Value.Remapping));
  S.Pb.writeInt32(TRewriterConfig.ftCommonSubgraphElimination, Ord(Value.CommonSubgraphElimination));
  S.Pb.writeInt32(TRewriterConfig.ftArithmeticOptimization, Ord(Value.ArithmeticOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftDependencyOptimization, Ord(Value.DependencyOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftLoopOptimization, Ord(Value.LoopOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftFunctionOptimization, Ord(Value.FunctionOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftDebugStripper, Ord(Value.DebugStripper));
  S.Pb.writeBoolean(TRewriterConfig.ftDisableModelPruning, Value.DisableModelPruning);
  S.Pb.writeInt32(TRewriterConfig.ftScopedAllocatorOptimization, Ord(Value.ScopedAllocatorOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftPinToHostOptimization, Ord(Value.PinToHostOptimization));
  S.Pb.writeInt32(TRewriterConfig.ftImplementationSelector, Ord(Value.ImplementationSelector));
  S.Pb.writeInt32(TRewriterConfig.ftAutoMixedPrecision, Ord(Value.AutoMixedPrecision));
  S.Pb.writeInt32(TRewriterConfig.ftAutoMixedPrecisionMkl, Ord(Value.AutoMixedPrecisionMkl));
  S.Pb.writeInt32(TRewriterConfig.ftAutoMixedPrecisionCpu, Ord(Value.AutoMixedPrecisionCpu));
  S.Pb.writeBoolean(TRewriterConfig.ftDisableMetaOptimizer, Value.DisableMetaOptimizer);
  S.Pb.writeInt32(TRewriterConfig.ftUsePluginOptimizers, Ord(Value.UsePluginOptimizers));
  S.Pb.writeInt32(TRewriterConfig.ftMetaOptimizerIterations, Ord(Value.MetaOptimizerIterations));
  S.Pb.writeInt32(TRewriterConfig.ftMinGraphNodes, Value.MinGraphNodes);
  S.Pb.writeBoolean(TRewriterConfig.ftExperimentalDisableCompressedTensorOptimization, Value.ExperimentalDisableCompressedTensorOptimization);
  S.Pb.writeBoolean(TRewriterConfig.ftExperimentalDisableFoldingQuantizationEmulation, Value.ExperimentalDisableFoldingQuantizationEmulation);
  S.Pb.writeInt32(TRewriterConfig.ftMemoryOptimization, Ord(Value.MemoryOptimization));
  S.Pb.writeString(TRewriterConfig.ftMemoryOptimizerTargetNodeNameScope, Value.MemoryOptimizerTargetNodeNameScope);
  S.Pb.writeInt64(TRewriterConfig.ftMetaOptimizerTimeoutMs, Value.MetaOptimizerTimeoutMs);
  S.SaveObj<TAutoParallelOptions>(Value.AutoParallel, SaveAutoParallelOptions, TRewriterConfig.ftAutoParallel);
  S.Pb.writeBoolean(TRewriterConfig.ftFailOnOptimizerErrors, Value.FailOnOptimizerErrors);
  S.SaveObj<TScopedAllocatorOptions>(Value.ScopedAllocatorOpts, SaveScopedAllocatorOptions, TRewriterConfig.ftScopedAllocatorOpts);
  h.Init;
  try
    for i := 0 to Value.Optimizerss.Count - 1 do
      h.Pb.writeRawString(Value.Optimizerss[i]^);
    S.Pb.writeMessage(TRewriterConfig.ftOptimizerss, h.Pb^);
  finally
    h.Free;
  end;
  if Value.CustomOptimizerss.Count > 0 then
    S.SaveList<TCustomGraphOptimizer>(Value.CustomOptimizerss, SaveCustomGraphOptimizer, TRewriterConfig.ftCustomOptimizerss);
  S.SaveObj<TVerifierConfig>(Value.InterOptimizerVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftInterOptimizerVerifierConfig);
  S.SaveObj<TVerifierConfig>(Value.PostOptimizationVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftPostOptimizationVerifierConfig);
end;

class procedure TSaveHelper.SaveTensorProto(const S: TpbSaver; const Value: TTensorProto);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TTensorProto.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.TensorShape, SaveTensorShapeProto, TTensorProto.ftTensorShape);
  S.Pb.writeInt32(TTensorProto.ftVersionNumber, Value.VersionNumber);
  S.Pb.writeBytes(TTensorProto.ftTensorContent, Value.TensorContent);
  h.Init;
  try
    for i := 0 to Value.HalfVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.HalfVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftHalfVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.FloatVals.Count - 1 do
      h.Pb.writeRawData(Value.FloatVals[i], sizeof(Single));
    S.Pb.writeMessage(TTensorProto.ftFloatVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DoubleVals.Count - 1 do
      h.Pb.writeRawData(Value.DoubleVals[i], sizeof(Double));
    S.Pb.writeMessage(TTensorProto.ftDoubleVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.IntVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.IntVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftIntVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.StringVals.Count - 1 do
      h.Pb.writeRawBytes(Value.StringVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftStringVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.ScomplexVals.Count - 1 do
      h.Pb.writeRawData(Value.ScomplexVals[i], sizeof(Single));
    S.Pb.writeMessage(TTensorProto.ftScomplexVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Int64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Int64Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftInt64Vals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.BoolVals.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.BoolVals[i]^));
    S.Pb.writeMessage(TTensorProto.ftBoolVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DcomplexVals.Count - 1 do
      h.Pb.writeRawData(Value.DcomplexVals[i], sizeof(Double));
    S.Pb.writeMessage(TTensorProto.ftDcomplexVals, h.Pb^);
  finally
    h.Free;
  end;
  if Value.ResourceHandleVals.Count > 0 then
    S.SaveList<TResourceHandleProto>(Value.ResourceHandleVals, SaveResourceHandleProto, TTensorProto.ftResourceHandleVals);
  if Value.VariantVals.Count > 0 then
    S.SaveList<TVariantTensorDataProto>(Value.VariantVals, SaveVariantTensorDataProto, TTensorProto.ftVariantVals);
  h.Init;
  try
    for i := 0 to Value.Uint32Vals.Count - 1 do
      h.Pb.writeRawVarint32(Value.Uint32Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftUint32Vals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Uint64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Uint64Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftUint64Vals, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto);

begin
  S.Pb.writeString(TVariantTensorDataProto.ftTypeName, Value.TypeName);
  S.Pb.writeBytes(TVariantTensorDataProto.ftMetadata, Value.Metadata);
  if Value.Tensorss.Count > 0 then
    S.SaveList<TTensorProto>(Value.Tensorss, SaveTensorProto, TVariantTensorDataProto.ftTensorss);
end;

class procedure TSaveHelper.SaveFunctionDefLibrary(const S: TpbSaver; const Value: TFunctionDefLibrary);
begin
  if Value.&Functions.Count > 0 then
    S.SaveList<TFunctionDef>(Value.&Functions, SaveFunctionDef, TFunctionDefLibrary.ftFunctions);
  if Value.Gradients.Count > 0 then
    S.SaveList<TGradientDef>(Value.Gradients, SaveGradientDef, TFunctionDefLibrary.ftGradients);
  if Value.RegisteredGradientss.Count > 0 then
    S.SaveList<TRegisteredGradient>(Value.RegisteredGradientss, SaveRegisteredGradient, TFunctionDefLibrary.ftRegisteredGradientss);
end;

class procedure TSaveHelper.SaveArgAttrs(const S: TpbSaver; const Value: TArgAttrs);
var 
 h : TpbSaver;

begin
  if Value.Attr.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TArgAttrs.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveFunctionDef(const S: TpbSaver; const Value: TFunctionDef);
var 
  h : TpbSaver;

begin
  S.SaveObj<TOpDef>(Value.Signature, SaveOpDef, TFunctionDef.ftSignature);
  if Value.Attr.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TFunctionDef.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ArgAttr.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.ArgAttr.Begins;
      while it <> Value.ArgAttr.Ends do
      begin
          h.clear;
          h.SaveUint32ArgAttrs(it.GetPair^);
          S.Pb.writeMessage(TFunctionDef.ftArgAttr, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ResourceArgUniqueId.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.ResourceArgUniqueId.Begins;
      while it <> Value.ResourceArgUniqueId.Ends do
      begin
          h.clear;
          h.SaveUint32Uint32(it.GetPair^);
          S.Pb.writeMessage(TFunctionDef.ftResourceArgUniqueId, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  if Value.NodeDefs.Count > 0 then
    S.SaveList<TNodeDef>(Value.NodeDefs, SaveNodeDef, TFunctionDef.ftNodeDefs);
  if Value.Ret.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.Ret.Begins;
      while it <> Value.Ret.Ends do
      begin
          h.clear;
          h.SaveStringString(it.GetPair^);
          S.Pb.writeMessage(TFunctionDef.ftRet, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ControlRet.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.ControlRet.Begins;
      while it <> Value.ControlRet.Ends do
      begin
          h.clear;
          h.SaveStringString(it.GetPair^);
          S.Pb.writeMessage(TFunctionDef.ftControlRet, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveGradientDef(const S: TpbSaver; const Value: TGradientDef);

begin
  S.Pb.writeString(TGradientDef.ftFunctionName, Value.FunctionName);
  S.Pb.writeString(TGradientDef.ftGradientFunc, Value.GradientFunc);
end;

class procedure TSaveHelper.SaveRegisteredGradient(const S: TpbSaver; const Value: TRegisteredGradient);

begin
  S.Pb.writeString(TRegisteredGradient.ftGradientFunc, Value.GradientFunc);
  S.Pb.writeString(TRegisteredGradient.ftRegisteredOpType, Value.RegisteredOpType);
end;

class procedure TSaveHelper.SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef);

begin
  S.Pb.writeInt32(TFullTypeDef.ftTypeId, Ord(Value.TypeId));
  if Value.Argss.Count > 0 then
    S.SaveList<TFullTypeDef>(Value.Argss, SaveFullTypeDef, TFullTypeDef.ftArgss);
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

class procedure TSaveHelper.SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.OriginalNodeNamess.Count - 1 do
      h.Pb.writeRawString(Value.OriginalNodeNamess[i]^);
    S.Pb.writeMessage(TExperimentalDebugInfo.ftOriginalNodeNamess, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.OriginalFuncNamess.Count - 1 do
      h.Pb.writeRawString(Value.OriginalFuncNamess[i]^);
    S.Pb.writeMessage(TExperimentalDebugInfo.ftOriginalFuncNamess, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveNodeDef(const S: TpbSaver; const Value: TNodeDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TNodeDef.ftName, Value.Name);
  S.Pb.writeString(TNodeDef.ftOp, Value.Op);
  h.Init;
  try
    for i := 0 to Value.Inputs.Count - 1 do
      h.Pb.writeRawString(Value.Inputs[i]^);
    S.Pb.writeMessage(TNodeDef.ftInputs, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeString(TNodeDef.ftDevice, Value.Device);
  if Value.Attr.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TNodeDef.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  S.SaveObj<TExperimentalDebugInfo>(Value.ExperimentalDebugInfo, SaveExperimentalDebugInfo, TNodeDef.ftExperimentalDebugInfo);
  S.SaveObj<TFullTypeDef>(Value.ExperimentalType, SaveFullTypeDef, TNodeDef.ftExperimentalType);
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
      h.Pb.writeRawVarint32(Value.BadConsumerss[i]^);
    S.Pb.writeMessage(TVersionDef.ftBadConsumerss, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig);

begin
  S.Pb.writeInt64(TVerifierConfig.ftVerificationTimeoutInMs, Value.VerificationTimeoutInMs);
  S.Pb.writeInt32(TVerifierConfig.ftStructureVerifier, Ord(Value.StructureVerifier));
end;

class procedure TSaveHelper.SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription);

begin
  S.Pb.writeInt64(TAllocationDescription.ftRequestedBytes, Value.RequestedBytes);
  S.Pb.writeInt64(TAllocationDescription.ftAllocatedBytes, Value.AllocatedBytes);
  S.Pb.writeString(TAllocationDescription.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocationDescription.ftAllocationId, Value.AllocationId);
  S.Pb.writeBoolean(TAllocationDescription.ftHasSingleReference, Value.HasSingleReference);
  S.Pb.writeInt64(TAllocationDescription.ftPtr, Value.Ptr);
end;

class procedure TSaveHelper.SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription);

begin
  S.Pb.writeInt32(TTensorDescription.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TTensorDescription.ftShape);
  S.SaveObj<TAllocationDescription>(Value.AllocationDescription, SaveAllocationDescription, TTensorDescription.ftAllocationDescription);
end;

procedure TSaveHelper.SaveStringInt32(Item: TsgPair<string, Integer>);

begin
  Pb.writeString(1, Item.Key);
  Pb.writeInt32(2, Item.Value);
end;

procedure TSaveHelper.SaveStringString(Item: TsgPair<string, string>);

begin
  Pb.writeString(1, Item.Key);
  Pb.writeString(2, Item.Value);
end;

procedure TSaveHelper.SaveStringAttrValue(Item: TPair<string, TAttrValue>);

begin
  Pb.writeString(1, Item.Key);
  SaveObj<TAttrValue>(Item.Value, SaveAttrValue, 2);
end;

procedure TSaveHelper.SaveUint32String(Item: TsgPair<UInt32, string>);

begin
  Pb.writeInt32(1, Item.Key);
  Pb.writeString(2, Item.Value);
end;

procedure TSaveHelper.SaveInt32String(Item: TsgPair<Integer, string>);

begin
  Pb.writeInt32(1, Item.Key);
  Pb.writeString(2, Item.Value);
end;

procedure TSaveHelper.SaveUint32ArgAttrs(Item: TsgPair<UInt32, TArgAttrs>);

begin
  Pb.writeInt32(1, Item.Key);
  SaveObj<TArgAttrs>(Item.Value, SaveArgAttrs, 2);
end;

procedure TSaveHelper.SaveUint32Uint32(Item: TsgPair<UInt32, UInt32>);

begin
  Pb.writeInt32(1, Item.Key);
  Pb.writeInt32(2, Item.Value);
end;

end.
