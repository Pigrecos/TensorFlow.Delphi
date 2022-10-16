unit ProtoGen.RewriterConfig;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  Spring.Collections, spring.Collections.MultiMaps,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.verifierconfig,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PAutoParallelOptions = ^TAutoParallelOptions;
  TAutoParallelOptions = record
  const
    ftEnable = 1;
    ftNumReplicas = 2;
  private
    FEnable: Boolean;
    FNumReplicas: Integer;
  public
    procedure Init;
    procedure Free;
    // properties
    property Enable: Boolean read FEnable write FEnable;
    property NumReplicas: Integer read FNumReplicas write FNumReplicas;
  end;

  PScopedAllocatorOptions = ^TScopedAllocatorOptions;
  TScopedAllocatorOptions = record
  const
    ftEnableOps = 1;
  private
    FEnableOps: TsgRecordList<string>;
  public
    procedure Init;
    procedure Free;
    // properties
    property EnableOps: TsgRecordList<string> read FEnableOps;
  end;

  TToggle = (
    DEFAULT = 0,
    ON = 1,
    OFF = 2,
    AGGRESSIVE = 3);

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

  TStringAttrValue = TMultiMap<string, TAttrValue>;

  PCustomGraphOptimizer = ^TCustomGraphOptimizer;
  TCustomGraphOptimizer = record
  const
    ftName = 1;
    ftParameterMap = 2;
  private
    FName: string;
    FParameterMap: TStringAttrValue;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property ParameterMap: TStringAttrValue read FParameterMap write FParameterMap;
  end;

  PRewriterConfig = ^TRewriterConfig;
  TRewriterConfig = record
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
    FOptimizerss: TsgRecordList<string>;
    FCustomOptimizerss: TsgRecordList<TCustomGraphOptimizer>;
    FInterOptimizerVerifierConfig: TVerifierConfig;
    FPostOptimizationVerifierConfig: TVerifierConfig;
  public
    procedure Init;
    procedure Free;
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
    property Optimizerss: TsgRecordList<string> read FOptimizerss;
    property CustomOptimizerss: TsgRecordList<TCustomGraphOptimizer> read FCustomOptimizerss;
    property InterOptimizerVerifierConfig: TVerifierConfig read FInterOptimizerVerifierConfig write FInterOptimizerVerifierConfig;
    property PostOptimizationVerifierConfig: TVerifierConfig read FPostOptimizationVerifierConfig write FPostOptimizationVerifierConfig;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadAutoParallelOptions(var Value: TAutoParallelOptions);
    procedure LoadScopedAllocatorOptions(var Value: TScopedAllocatorOptions);
    procedure LoadRewriterConfig(var Value: TRewriterConfig);
    procedure LoadCustomGraphOptimizer(var Value: TCustomGraphOptimizer);
    procedure LoadAttrValue(var Value: TAttrValue);
    procedure LoadListValue(var Value: TListValue);
    procedure LoadNameAttrList(var Value: TNameAttrList);
    procedure LoadResourceHandleProto(var Value: TResourceHandleProto);
    procedure LoadDtypeAndShape(var Value: TDtypeAndShape);
    procedure LoadVerifierConfig(var Value: TVerifierConfig);
    procedure LoadTensorProto(var Value: TTensorProto);
    procedure LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
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

    class procedure SaveAutoParallelOptions(const S: TpbSaver; const Value: TAutoParallelOptions); static;
    class procedure SaveScopedAllocatorOptions(const S: TpbSaver; const Value: TScopedAllocatorOptions); static;
    class procedure SaveRewriterConfig(const S: TpbSaver; const Value: TRewriterConfig); static;
    class procedure SaveCustomGraphOptimizer(const S: TpbSaver; const Value: TCustomGraphOptimizer); static;
    class procedure SaveAttrValue(const S: TpbSaver; const Value: TAttrValue); static;
    class procedure SaveListValue(const S: TpbSaver; const Value: TListValue); static;
    class procedure SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList); static;
    class procedure SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto); static;
    class procedure SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape); static;
    class procedure SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig); static;
    class procedure SaveTensorProto(const S: TpbSaver; const Value: TTensorProto); static;
    class procedure SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
    procedure SaveStringAttrValue(Item: TPair<string, TAttrValue>);

  end;

implementation
           uses Oz.Pb.StrBuffer;

{ TAutoParallelOptions }

procedure TAutoParallelOptions.Init;
begin
  Self := System.Default(TAutoParallelOptions);
end;

procedure TAutoParallelOptions.Free;
begin
end;

{ TScopedAllocatorOptions }

procedure TScopedAllocatorOptions.Init;
begin
  Self := System.Default(TScopedAllocatorOptions);
  FEnableOps := TsgRecordList<string>.From(nil);
end;

procedure TScopedAllocatorOptions.Free;
begin
  FEnableOps.Free;
end;

{ TCustomGraphOptimizer }

procedure TCustomGraphOptimizer.Init;
begin
  Self := System.Default(TCustomGraphOptimizer);
  FParameterMap := TMultiMap<string, TAttrValue>.Create;
end;

procedure TCustomGraphOptimizer.Free;
begin
  FParameterMap.Free;
end;

{ TRewriterConfig }

procedure TRewriterConfig.Init;
begin
  Self := System.Default(TRewriterConfig);
  FOptimizerss := TsgRecordList<string>.From(nil);
  FCustomOptimizerss := TsgRecordList<TCustomGraphOptimizer>.From(nil);
end;

procedure TRewriterConfig.Free;
begin
  FOptimizerss.Free;
  FCustomOptimizerss.Free;
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
              Value.FEnableOps.Add(@v);
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
          Value.LayoutOptimizer := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftConstantFolding:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ConstantFolding := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftShapeOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShapeOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftRemapping:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Remapping := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftCommonSubgraphElimination:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CommonSubgraphElimination := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftArithmeticOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ArithmeticOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDependencyOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DependencyOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftLoopOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LoopOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftFunctionOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.FunctionOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDebugStripper:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DebugStripper := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDisableModelPruning:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableModelPruning := Pb.readBoolean;
        end;
      TRewriterConfig.ftScopedAllocatorOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScopedAllocatorOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftPinToHostOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PinToHostOptimization := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftImplementationSelector:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ImplementationSelector := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecision:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecision := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecisionMkl:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecisionMkl := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftAutoMixedPrecisionCpu:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AutoMixedPrecisionCpu := TToggle(Pb.readInt32);
        end;
      TRewriterConfig.ftDisableMetaOptimizer:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableMetaOptimizer := Pb.readBoolean;
        end;
      TRewriterConfig.ftUsePluginOptimizers:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UsePluginOptimizers := TToggle(Pb.readInt32);
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
            var v : TAutoParallelOptions := Value.FAutoParallel;
            LoadAutoParallelOptions(v);
            Value.FAutoParallel := v;
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
            var v : TScopedAllocatorOptions := Value.FScopedAllocatorOpts;
            LoadScopedAllocatorOptions(v);
            Value.FScopedAllocatorOpts := v;
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
              Value.FOptimizerss.Add(@v);
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
            Value.FCustomOptimizerss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftInterOptimizerVerifierConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVerifierConfig := Value.FInterOptimizerVerifierConfig;
            LoadVerifierConfig(v);
            Value.FInterOptimizerVerifierConfig := v;
          finally
            Pb.Pop;
          end;
        end;
      TRewriterConfig.ftPostOptimizationVerifierConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVerifierConfig := Value.FPostOptimizationVerifierConfig;
            LoadVerifierConfig(v);
            Value.FPostOptimizationVerifierConfig := v;
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
          Value.StructureVerifier := ProtoGen.VerifierConfig.TToggle(Pb.readInt32);
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
      h.Pb.writeRawString(Value.FEnableOps[i]^);
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
  if Value.FParameterMap.Count > 0 then
  begin
    h.Init;
    try
      For var p in Value.FParameterMap do
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
  S.SaveObj<TAutoParallelOptions>(Value.FAutoParallel, SaveAutoParallelOptions, TRewriterConfig.ftAutoParallel);
  S.Pb.writeBoolean(TRewriterConfig.ftFailOnOptimizerErrors, Value.FailOnOptimizerErrors);
  S.SaveObj<TScopedAllocatorOptions>(Value.FScopedAllocatorOpts, SaveScopedAllocatorOptions, TRewriterConfig.ftScopedAllocatorOpts);
  h.Init;
  try
    for i := 0 to Value.Optimizerss.Count - 1 do
      h.Pb.writeRawString(Value.FOptimizerss[i]^);
    S.Pb.writeMessage(TRewriterConfig.ftOptimizerss, h.Pb^);
  finally
    h.Free;
  end;
  if Value.FCustomOptimizerss.Count > 0 then
    S.SaveList<TCustomGraphOptimizer>(Value.FCustomOptimizerss, SaveCustomGraphOptimizer, TRewriterConfig.ftCustomOptimizerss);
  S.SaveObj<TVerifierConfig>(Value.FInterOptimizerVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftInterOptimizerVerifierConfig);
  S.SaveObj<TVerifierConfig>(Value.FPostOptimizationVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftPostOptimizationVerifierConfig);
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

class procedure TSaveHelper.SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig);
begin
  S.Pb.writeInt64(TVerifierConfig.ftVerificationTimeoutInMs, Value.VerificationTimeoutInMs);
  S.Pb.writeInt32(TVerifierConfig.ftStructureVerifier, Ord(Value.StructureVerifier));
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

procedure TSaveHelper.SaveStringAttrValue(Item: TPair<string, TAttrValue>);
begin
  Pb.writeString(1, Item.Key);
  SaveObj<TAttrValue>(Item.Value, SaveAttrValue, 2);
end;

end.
