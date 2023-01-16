unit ProtoGen.Config;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
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
    DEFAULT = 0,
    OFF = 1,
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

implementation

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

end.
