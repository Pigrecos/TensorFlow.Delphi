unit ProtoGen.RewriterConfig;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.verifierconfig,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


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

implementation

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

end.
