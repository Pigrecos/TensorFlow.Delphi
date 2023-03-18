unit ProtoGen.Main;

interface
      uses
          System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
          TensorFlow.Proto;

type
  TLoadHelper = record helper for TpbLoader
      type
        TLoad<T: constructor> = procedure(var Value: T) of object;
        TLoadPair<Key, Value> = procedure(var Pair: TPair<Key, Value>) of object;
      protected
        procedure LoadObj<T: constructor>(var obj: T; Load: TLoad<T>);
        procedure LoadList<T: constructor>(const List: TList<T>; Load: TLoad<T>);
        procedure LoadMap<Key, Value>(var map: TDictionary<Key, Value>; Load: TLoadPair<Key, Value>; Tag: Integer);
      public
        procedure LoadStringAttrValue(var Value: TPair<string, TAttrValue>);
	      // ProtoGen.TensorShape
		    procedure LoadTensorShapeProto(var Value: TTensorShapeProto);
        procedure LoadDim(var Value: TDim);
        // ProtoGen.ResourceHandle
        procedure LoadResourceHandleProto(var Value: TResourceHandleProto);
        procedure LoadDtypeAndShape(var Value: TDtypeAndShape);
        // ProtoGen.Tensor
        procedure LoadTensorProto(var Value: TTensorProto);
        procedure LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
        // ProtoGen.AttrValue
        procedure LoadAttrValue(var Value: TAttrValue);
        procedure LoadListValue(var Value: TListValue);
        procedure LoadNameAttrList(var Value: TNameAttrList);
        // ProtoGen.CostGraph
        procedure LoadCostGraphDef(var Value: TCostGraphDef);
        procedure LoadNode(var Value: TNode);
        procedure LoadInputInfo(var Value: TInputInfo);
        procedure LoadOutputInfo(var Value: TOutputInfo);
        procedure LoadAggregatedCost(var Value: TAggregatedCost);
        // ProtoGen.AllocationDescription
        procedure LoadAllocationDescription(var Value: TAllocationDescription);
        // ProtoGen.Cluster
        procedure LoadJobDef(var Value: TJobDef);
        procedure LoadClusterDef(var Value: TClusterDef);
        // ProtoGen.CoordinationConfig
        procedure LoadCoordinationServiceConfig(var Value: TCoordinationServiceConfig);
        // ProtoGen.OpDef
        procedure LoadOpDef(var Value: TOpDef);
        procedure LoadArgDef(var Value: TArgDef);
        procedure LoadAttrDef(var Value: TAttrDef);
        procedure LoadOpDeprecation(var Value: TOpDeprecation);
        procedure LoadOpList(var Value: TOpList);
        // ProtoGen.FullType
        procedure LoadFullTypeDef(var Value: TFullTypeDef);
        // ProtoGen.Debug
        procedure LoadDebugTensorWatch(var Value: TDebugTensorWatch);
        procedure LoadDebugOptions(var Value: TDebugOptions);
        procedure LoadDebuggedSourceFile(var Value: TDebuggedSourceFile);
        procedure LoadDebuggedSourceFiles(var Value: TDebuggedSourceFiles);
        // ProtoGen.Versions
        procedure LoadVersionDef(var Value: TVersionDef);
        // ProtoGen.VerifierConfig
        procedure LoadVerifierConfig(var Value: TVerifierConfig);
        // ProtoGen.Variable
        procedure LoadVariableDef(var Value: TVariableDef);
        procedure LoadSaveSliceInfoDef(var Value: TSaveSliceInfoDef);
        // ProtoGen.Function
        procedure LoadFunctionDefLibrary(var Value: TFunctionDefLibrary);
        procedure LoadFunctionDef(var Value: TFunctionDef);
        procedure LoadArgAttrs(var Value: TArgAttrs);
        procedure LoadGradientDef(var Value: TGradientDef);
        procedure LoadRegisteredGradient(var Value: TRegisteredGradient);
        // ProtoGen.CppShapeInference
        procedure LoadCppShapeInferenceResult(var Value: TCppShapeInferenceResult);
        procedure LoadHandleShapeAndType(var Value: THandleShapeAndType);
        procedure LoadHandleData(var Value: THandleData);
        procedure LoadCppShapeInferenceInputsNeeded(var Value: TCppShapeInferenceInputsNeeded);
        // ProtoGen.RewriterConfig
        procedure LoadAutoParallelOptions(var Value: TAutoParallelOptions);
        procedure LoadScopedAllocatorOptions(var Value: TScopedAllocatorOptions);
        procedure LoadRewriterConfig(var Value: TRewriterConfig);
        procedure LoadCustomGraphOptimizer(var Value: TCustomGraphOptimizer);
        // ProtoGen.Graph
        procedure LoadGraphDef(var Value: TGraphDef);
        // ProtoGen.TensorDescription
        procedure LoadTensorDescription(var Value: TTensorDescription);
        // ProtoGen.NodeDef
        procedure LoadNodeDef(var Value: TNodeDef);
        procedure LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
        // ProtoGen.Config
        procedure LoadGPUOptions(var Value: TGPUOptions);
        procedure LoadExperimental(var Value: TGPUOptions.TExperimental); overload;
        procedure LoadVirtualDevices(var Value: TVirtualDevices);
        procedure LoadOptimizerOptions(var Value: TOptimizerOptions);
        procedure LoadGraphOptions(var Value: TGraphOptions);
        procedure LoadThreadPoolOptionProto(var Value: TThreadPoolOptionProto);
        procedure LoadRPCOptions(var Value: TRPCOptions);
        procedure LoadSessionMetadata(var Value: TSessionMetadata);
        procedure LoadConfigProto(var Value: TConfigProto);
        procedure LoadExperimental(var Value: TConfigProto.TExperimental); overload;
        procedure LoadRunOptions(var Value: TRunOptions);
        procedure LoadExperimental(var Value: TRunOptions.TExperimental);  overload;
        procedure LoadRunHandlerPoolOptions(var Value: TRunHandlerPoolOptions);
        procedure LoadRunMetadata(var Value: TRunMetadata);
        procedure LoadFunctionGraphs(var Value: TFunctionGraphs);
        procedure LoadTensorConnection(var Value: TTensorConnection);
        procedure LoadCallableOptions(var Value: TCallableOptions);
        // ProtoGen.StepStats
        procedure LoadAllocationRecord(var Value: TAllocationRecord);
        procedure LoadAllocatorMemoryUsed(var Value: TAllocatorMemoryUsed);
        procedure LoadNodeOutput(var Value: TNodeOutput);
        procedure LoadMemoryStats(var Value: TMemoryStats);
        procedure LoadNodeExecStats(var Value: TNodeExecStats);
        procedure LoadDeviceStepStats(var Value: TDeviceStepStats);
        procedure LoadStepStats(var Value: TStepStats);
        // ProtoGen.ControlFlow
        procedure LoadValuesDef(var Value: TValuesDef);
        procedure LoadControlFlowContextDef(var Value: TControlFlowContextDef);
        procedure LoadCondContextDef(var Value: TCondContextDef);
        procedure LoadWhileContextDef(var Value: TWhileContextDef);
  end;  

  TSaveHelper = record helper for TpbSaver
      type
        TSave<T>              = procedure(const S: TpbSaver; const Value: T);
        TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TPair<Key, Value>);
      protected
        procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
        procedure SaveList<T>(const List: TList<T>; Save: TSave<T>; Tag: Integer);
        procedure SaveMap<Key, Value>(const Map: TDictionary<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);
      public
        procedure SaveInt32String(Value: TPair<Integer, string>);
        procedure SaveStringAttrValue(Value: TPair<string, TAttrValue>);
        procedure SaveUint32ArgAttrs(Value: TPair<UInt32, TArgAttrs>);
        procedure SaveUint32Uint32(Value: TPair<UInt32, UInt32>);
        procedure SaveStringString(Value: TPair<string, string>);
        procedure SaveStringInt32(Value: TPair<string, Integer>);
        procedure SaveUint32String(Value: TPair<UInt32, string>);
        // ProtoGen.TensorShape
	    	class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
        class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
        // ProtoGen.ResourceHandle
        class procedure SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto); static;
        class procedure SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape); static;
        // ProtoGen.Tensor
        class procedure SaveTensorProto(const S: TpbSaver; const Value: TTensorProto); static;
        class procedure SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto); static;
        // ProtoGen.AttrValue
        class procedure SaveAttrValue(const S: TpbSaver; const Value: TAttrValue); static;
        class procedure SaveListValue(const S: TpbSaver; const Value: TListValue); static;
        class procedure SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList); static;
        // ProtoGen.CostGraph
        class procedure SaveCostGraphDef(const S: TpbSaver; const Value: TCostGraphDef); static;
        class procedure SaveNode(const S: TpbSaver; const Value: TNode); static;
        class procedure SaveInputInfo(const S: TpbSaver; const Value: TInputInfo); static;
        class procedure SaveOutputInfo(const S: TpbSaver; const Value: TOutputInfo); static;
        class procedure SaveAggregatedCost(const S: TpbSaver; const Value: TAggregatedCost); static;
        // ProtoGen.AllocationDescription
        class procedure SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription); static;
        // ProtoGen.Cluster
        class procedure SaveJobDef(const S: TpbSaver; const Value: TJobDef); static;
        class procedure SaveClusterDef(const S: TpbSaver; const Value: TClusterDef); static;
        // ProtoGen.CoordinationConfig
        class procedure SaveCoordinationServiceConfig(const S: TpbSaver; const Value: TCoordinationServiceConfig); static;
	      // ProtoGen.OpDef
		    class procedure SaveOpDef(const S: TpbSaver; const Value: TOpDef); static;
        class procedure SaveArgDef(const S: TpbSaver; const Value: TArgDef); static;
        class procedure SaveAttrDef(const S: TpbSaver; const Value: TAttrDef); static;
        class procedure SaveOpDeprecation(const S: TpbSaver; const Value: TOpDeprecation); static;
        class procedure SaveOpList(const S: TpbSaver; const Value: TOpList); static;
        // ProtoGen.FullType
        class procedure SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef); static;
        // ProtoGen.Debug
        class procedure SaveDebugTensorWatch(const S: TpbSaver; const Value: TDebugTensorWatch); static;
        class procedure SaveDebugOptions(const S: TpbSaver; const Value: TDebugOptions); static;
        class procedure SaveDebuggedSourceFile(const S: TpbSaver; const Value: TDebuggedSourceFile); static;
        class procedure SaveDebuggedSourceFiles(const S: TpbSaver; const Value: TDebuggedSourceFiles); static;
        // ProtoGen.Versions
        class procedure SaveVersionDef(const S: TpbSaver; const Value: TVersionDef); static;
        // ProtoGen.VerifierConfig
        class procedure SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig); static;
        // ProtoGen.Variable
        class procedure SaveVariableDef(const S: TpbSaver; const Value: TVariableDef); static;
        class procedure SaveSaveSliceInfoDef(const S: TpbSaver; const Value: TSaveSliceInfoDef); static;
        // ProtoGen.Function
        class procedure SaveFunctionDefLibrary(const S: TpbSaver; const Value: TFunctionDefLibrary); static;
        class procedure SaveFunctionDef(const S: TpbSaver; const Value: TFunctionDef); static;
        class procedure SaveArgAttrs(const S: TpbSaver; const Value: TArgAttrs); static;
        class procedure SaveGradientDef(const S: TpbSaver; const Value: TGradientDef); static;
        class procedure SaveRegisteredGradient(const S: TpbSaver; const Value: TRegisteredGradient); static;
        // ProtoGen.CppShapeInference
        class procedure SaveCppShapeInferenceResult(const S: TpbSaver; const Value: TCppShapeInferenceResult); static;
        class procedure SaveHandleShapeAndType(const S: TpbSaver; const Value: THandleShapeAndType); static;
        class procedure SaveHandleData(const S: TpbSaver; const Value: THandleData); static;
        class procedure SaveCppShapeInferenceInputsNeeded(const S: TpbSaver; const Value: TCppShapeInferenceInputsNeeded); static;
        // ProtoGen.RewriterConfig
        class procedure SaveAutoParallelOptions(const S: TpbSaver; const Value: TAutoParallelOptions); static;
        class procedure SaveScopedAllocatorOptions(const S: TpbSaver; const Value: TScopedAllocatorOptions); static;
        class procedure SaveRewriterConfig(const S: TpbSaver; const Value: TRewriterConfig); static;
        class procedure SaveCustomGraphOptimizer(const S: TpbSaver; const Value: TCustomGraphOptimizer); static;
        // ProtoGen.Graph
        class procedure SaveGraphDef(const S: TpbSaver; const Value: TGraphDef); static;
        // ProtoGen.TensorDescription
        class procedure SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription); static;
        // ProtoGen.NodeDef
	    	class procedure SaveNodeDef(const S: TpbSaver; const Value: TNodeDef); static;
        class procedure SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo); static;
        // ProtoGen.Config
        class procedure SaveGPUOptions(const S: TpbSaver; const Value: TGPUOptions); static;
        class procedure SaveExperimental(const S: TpbSaver; const Value: TGPUOptions.TExperimental); overload; static;
        class procedure SaveVirtualDevices(const S: TpbSaver; const Value: TVirtualDevices); static;
        class procedure SaveOptimizerOptions(const S: TpbSaver; const Value: TOptimizerOptions); static;
        class procedure SaveGraphOptions(const S: TpbSaver; const Value: TGraphOptions); static;
        class procedure SaveThreadPoolOptionProto(const S: TpbSaver; const Value: TThreadPoolOptionProto); static;
        class procedure SaveRPCOptions(const S: TpbSaver; const Value: TRPCOptions); static;
        class procedure SaveSessionMetadata(const S: TpbSaver; const Value: TSessionMetadata); static;
        class procedure SaveConfigProto(const S: TpbSaver; const Value: TConfigProto); static;
        class procedure SaveExperimental(const S: TpbSaver; const Value: TConfigProto.TExperimental); overload; static;
        class procedure SaveRunOptions(const S: TpbSaver; const Value: TRunOptions); static;
        class procedure SaveExperimental(const S: TpbSaver; const Value: TRunOptions.TExperimental); overload; static;
        class procedure SaveRunHandlerPoolOptions(const S: TpbSaver; const Value: TRunHandlerPoolOptions); static;
        class procedure SaveRunMetadata(const S: TpbSaver; const Value: TRunMetadata); static;
        class procedure SaveFunctionGraphs(const S: TpbSaver; const Value: TFunctionGraphs); static;
        class procedure SaveTensorConnection(const S: TpbSaver; const Value: TTensorConnection); static;
        class procedure SaveCallableOptions(const S: TpbSaver; const Value: TCallableOptions); static;
        // ProtoGen.StepStats
        class procedure SaveAllocationRecord(const S: TpbSaver; const Value: TAllocationRecord); static;
        class procedure SaveAllocatorMemoryUsed(const S: TpbSaver; const Value: TAllocatorMemoryUsed); static;
        class procedure SaveNodeOutput(const S: TpbSaver; const Value: TNodeOutput); static;
        class procedure SaveMemoryStats(const S: TpbSaver; const Value: TMemoryStats); static;
        class procedure SaveNodeExecStats(const S: TpbSaver; const Value: TNodeExecStats); static;
        class procedure SaveDeviceStepStats(const S: TpbSaver; const Value: TDeviceStepStats); static;
        class procedure SaveStepStats(const S: TpbSaver; const Value: TStepStats); static;
        // ProtoGen.ControlFlow
        class procedure SaveValuesDef(const S: TpbSaver; const Value: TValuesDef); static;
        class procedure SaveControlFlowContextDef(const S: TpbSaver; const Value: TControlFlowContextDef); static;
        class procedure SaveCondContextDef(const S: TpbSaver; const Value: TCondContextDef); static;
        class procedure SaveWhileContextDef(const S: TpbSaver; const Value: TWhileContextDef); static;
  end;

implementation
        uses Oz.Pb.StrBuffer;

procedure TSaveHelper.SaveUint32String(Value: TPair<UInt32, string>);
var
  S: TpbSaver;
begin
  S.Pb.writeInt32(1, Value.Key);
  S.Pb.writeString(2, Value.Value);
end;

procedure TSaveHelper.SaveStringInt32(Value: TPair<string, Integer>);
var
  S: TpbSaver;
begin
  S.Pb.writeString(1, Value.Key);
  S.Pb.writeInt32(2, Value.Value);
end;

procedure TSaveHelper.SaveUint32ArgAttrs(Value: TPair<UInt32, TArgAttrs>);
var
  S: TpbSaver;
begin
  S.Pb.writeInt32(1, Value.Key);
  S.SaveObj<TArgAttrs>(Value.Value, SaveArgAttrs, 2);
end;

procedure TSaveHelper.SaveUint32Uint32(Value: TPair<UInt32, UInt32>);
var
  S: TpbSaver;
begin
  S.Pb.writeInt32(1, Value.Key);
  S.Pb.writeInt32(2, Value.Value);
end;

procedure TSaveHelper.SaveStringString(Value: TPair<string, string>);
var
  S: TpbSaver;
begin
  S.Pb.writeString(1, Value.Key);
  S.Pb.writeString(2, Value.Value);
end;

procedure TLoadHelper.LoadStringAttrValue(var Value: TPair<string, TAttrValue>);
var
  tTag        : TpbTag;
  fieldNumber : Integer;
  key         : string;
  vAttr       : TAttrValue;
begin
    key   := '';
    vAttr := nil;
    
    tTag := pb.readTag;
    while tTag.v <> 0 do
    begin
        fieldNumber := tTag.FieldNumber;
        case fieldNumber of
           1:  key  := Pb.readString;
           2:  begin
                 Pb.push; 
                 try 
                   LoadAttrValue(vAttr)
                 finally
                   pb.pop;
                 end;
           end
        else
          pb.skipField(tTag);  
        end;
        tTag := pb.readTag;
    end;
    Value.Key   := key;
    Value.Value := vAttr;
end;

procedure TSaveHelper.SaveStringAttrValue(Value: TPair<string, TAttrValue>);
begin
  Pb.writeString(1, Value.Key);
  SaveObj<TAttrValue>(Value.Value, SaveAttrValue, 2);
end;

procedure TSaveHelper.SaveInt32String(Value: TPair<Integer, string>);
var
  S: TpbSaver;
begin
  S.Pb.writeInt32(1, Value.Key);
  S.Pb.writeString(2, Value.Value);
end;

{ TLoadHelper }

procedure TLoadHelper.LoadObj<T>(var obj: T; Load: TLoad<T>);
begin
  Pb.Push;
  try
    var v : TValue := TValue.From<T>(obj);
    var vObj       := v.AsObject;
    obj := vObj.Create;
    Load(obj);
  finally
    Pb.Pop;
  end;
end;

procedure TLoadHelper.LoadList<T>(const List: TList<T>; Load: TLoad<T>);
var
  obj: T;
begin
  Pb.Push;
  try
    var v : TValue := TValue.From<T>(obj);
    var vObj       := v.AsObject;
    obj := vObj.Create;
    Load(obj);
    List.Add(obj);
  finally
    Pb.Pop;
  end;
end;

procedure TLoadHelper.LoadMap<Key, Value>(var map: TDictionary<Key, Value>; Load: TLoadPair<Key, Value>; Tag: Integer);
var
  Pair        : TPair<Key, Value>;
begin
   repeat
      Pb.Push;
      try
        Load(pair);
        if map = nil then
           map := TDictionary<Key, Value>.Create;

        map.Add(Pair.Key,Pair.Value);
      finally
        Pb.pop
      end;
   until not Pb.ConsumeTag(Tag);
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

procedure TSaveHelper.SaveList<T>(const List: TList<T>; Save: TSave<T>; Tag: Integer);
var
  i: Integer;
  h: TpbSaver;
  Item: T;
begin
  h.Init;
  try
    for i := 0 to List.Count - 1 do
    begin
      h.Clear;
      Item := List[i];
      Save(h, Item);
      Pb.writeMessage(tag, h.Pb^);
    end;
  finally
    h.Free;
  end;
end;

procedure TSaveHelper.SaveMap<Key, Value>(const Map: TDictionary<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);
var
  h: TpbSaver;
  Pair: TPair<Key, Value>;
begin
  h.Init;
  try
    for Pair in Map do
    begin
      h.Clear;
      Save(h, Pair);
      Pb.writeMessage(tag, h.Pb^);
    end;
  finally
    h.Free;
  end;
end;

{$Region 'ProtoGen.TensorShape'}
procedure TLoadHelper.LoadDim(var Value: TDim);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TDim.Create;
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
  Value := TTensorShapeProto.Create;
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
            Value.Dims.Add(v);
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

{$EndRegion}

{$Region 'ProtoGen.ResourceHandle'}
procedure TLoadHelper.LoadDtypeAndShape(var Value: TDtypeAndShape);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TDtypeAndShape.Create;
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
  Value := TResourceHandleProto.Create;
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
            Value.DtypesAndShapess.Add(v);
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

class procedure TSaveHelper.SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape);
begin
  S.Pb.writeInt32(TDtypeAndShape.ftDtype, Ord(Value.Dtype));
  if Value.Shape <> nil then
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

{$EndRegion}

{$Region 'ProtoGen.Tensor'}
procedure TLoadHelper.LoadTensorProto(var Value: TTensorProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TTensorProto.Create;
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
            var vTipo : int32;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.HalfVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.HalfVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftFloatVals:
        begin
            var vTipo : single;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : single := Pb.readFloat;
                  Value.FloatVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : single := Pb.readFloat;
                Value.FloatVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftDoubleVals:
        begin
            var vTipo : double;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : double := Pb.readDouble;
                  Value.DoubleVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : double := Pb.readDouble;
                Value.DoubleVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftIntVals:
        begin
            var vTipo : int32;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.IntVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.IntVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftStringVals:
        begin
            var vTipo : TBytes;
            if IsPackedRepeatedField(tag, TValue.From<String>('')) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : TBytes := Pb.readBytes;
                  if Length(v) > 0   then
                      Value.StringVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : TBytes := Pb.readBytes;
                if Length(v) > 0   then
                    Value.StringVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftScomplexVals:
        begin
            var vTipo : single;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : single := Pb.readFloat;
                  Value.ScomplexVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : single := Pb.readFloat;
                Value.ScomplexVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftInt64Vals:
        begin
            var vTipo : int64;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.Int64Vals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.Int64Vals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftBoolVals:
        begin
            var vTipo : boolean;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : boolean := Pb.readBoolean;
                  Value.BoolVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : boolean := Pb.readBoolean;
                Value.BoolVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftDcomplexVals:
        begin
            var vTipo : double;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : double := Pb.readDouble;
                  Value.DcomplexVals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : double := Pb.readDouble;
                Value.DcomplexVals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftResourceHandleVals:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TResourceHandleProto;
            LoadResourceHandleProto(v);
            Value.ResourceHandleVals.Add(v);
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
            Value.VariantVals.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftUint32Vals:
        begin
            var vTipo : uint32;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : uint32 := Pb.readUint32;
                  Value.Uint32Vals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : uint32 := Pb.readUint32;
                Value.Uint32Vals.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TTensorProto.ftUint64Vals:
        begin
            var vTipo : uint64;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : uint64 := Pb.readInt64;
                  Value.Uint64Vals.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : uint64 := Pb.readInt64;
                Value.Uint64Vals.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TVariantTensorDataProto.Create;
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
            Value.Tensorss.Add(v);
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

class procedure TSaveHelper.SaveTensorProto(const S: TpbSaver; const Value: TTensorProto);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TTensorProto.ftDtype, Ord(Value.Dtype));
  if Value.TensorShape <> nil then
    S.SaveObj<TTensorShapeProto>(Value.TensorShape, SaveTensorShapeProto, TTensorProto.ftTensorShape);

  S.Pb.writeInt32(TTensorProto.ftVersionNumber, Value.VersionNumber);
  S.Pb.writeBytes(TTensorProto.ftTensorContent, Value.TensorContent);

  h.Init;
  try
    for i := 0 to Value.HalfVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.HalfVals[i]);
    if Value.HalfVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftHalfVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.FloatVals.Count - 1 do
    begin
      var vVar : Single := Value.FloatVals[i];
      h.Pb.writeRawData(@vVar, sizeof(Single));
    end;
    if Value.FloatVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftFloatVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.DoubleVals.Count - 1 do
    begin
      var vVar : Double := Value.DoubleVals[i];
      h.Pb.writeRawData(@vVar, sizeof(Double));
    end;
    if Value.DoubleVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftDoubleVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.IntVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.IntVals[i]);
    if Value.IntVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftIntVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.StringVals.Count - 1 do
      h.Pb.writeRawData(Value.StringVals[i], Length(Value.StringVals[i]));
    if Value.StringVals.Count > 0 then  
      S.Pb.writeMessage(TTensorProto.ftStringVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.ScomplexVals.Count - 1 do
    begin
      var vVar : Single := Value.ScomplexVals[i];
      h.Pb.writeRawData(@vVar, sizeof(Single));
    end;
    if Value.ScomplexVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftScomplexVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.Int64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Int64Vals[i]);
    if Value.Int64Vals.Count > 0then
      S.Pb.writeMessage(TTensorProto.ftInt64Vals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.BoolVals.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.BoolVals[i]));
    if Value.BoolVals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftBoolVals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.DcomplexVals.Count - 1 do
    begin
      var vVar : Double := Value.DcomplexVals[i];
      h.Pb.writeRawData(@vVar, sizeof(Double));
    end;
    if Value.DcomplexVals.Count > 0 then
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
      h.Pb.writeRawVarint32(Value.Uint32Vals[i]);
    if Value.Uint32Vals.Count > 0 then
      S.Pb.writeMessage(TTensorProto.ftUint32Vals, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.Uint64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Uint64Vals[i]);
    if Value.Uint64Vals.Count > 0 then
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

{$EndRegion}

{$Region 'ProtoGen.AttrValue'}
procedure TLoadHelper.LoadListValue(var Value: TListValue);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TListValue.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TListValue.ftSs:
        begin
            var vTipo : TBytes;
            if IsPackedRepeatedField(tag, TValue.From<string>('')) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : TBytes := Pb.readBytes;
                  if Length(v) > 0   then
                     Value.Ss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : TBytes := Pb.readBytes;
                if Length(v) > 0   then
                    Value.Ss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TListValue.ftIs:
        begin
            var vTipo : int64;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.&Is.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.&Is.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TListValue.ftFs:
        begin
            var vTipo : single;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : single := Pb.readFloat;
                  Value.Fs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : single := Pb.readFloat;
                Value.Fs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TListValue.ftBs:
        begin
            var vTipo : boolean;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : boolean := Pb.readBoolean;
                  Value.Bs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : boolean := Pb.readBoolean;
                Value.Bs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TListValue.ftTypes:
        begin
            var vTipo : Integer;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : TDataType := TDataType(Pb.readInt32);
                  Value.Types.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : TDataType := TDataType(Pb.readInt32);
                Value.Types.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TListValue.ftShapes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto;
            LoadTensorShapeProto(v);
            Value.Shapes.Add(v);
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
            Value.Tensors.Add(v);
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
            Value.Funcs.Add(v);
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
  Value := TAttrValue.Create;
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
          v.value := TValue.From<Integer>(Pb.readInt32);
          Value.value := v;
        end;
      TAttrValue.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftShape;
          Pb.Push;
          try
            
              var v1 : TTensorShapeProto;
              LoadTensorShapeProto(v1);
              v.value := TValue.From<TTensorShapeProto>(v1);
              Value.value := v;
          finally
           Pb.Pop
          end;
        end;
      TAttrValue.ftTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftTensor;
          Pb.Push;
          try
            
              var v1 : TTensorProto;
              LoadTensorProto(v1);
              v.value := TValue.From<TTensorProto>(v1);
              Value.value := v;
          finally
           Pb.Pop
          end;
        end;
      TAttrValue.ftList:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftList;
          Pb.Push;
          try
            
              var v1 : TListValue;
              LoadListValue(v1);
              v.value := TValue.From<TListValue>(v1);
              Value.value := v;
          finally
           Pb.Pop
          end;
        end;
      TAttrValue.ftFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftFunc;
          Pb.Push;
          try
            
              var v1 : TNameAttrList;
              LoadNameAttrList(v1);
              v.value := TValue.From<TNameAttrList>(v1);
              Value.value := v;
          finally
           Pb.Pop
          end;
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
  Value := TNameAttrList.Create;
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
          Assert(wireType = TWire.LENGTH_DELIMITED);

          var map : TDictionary<string, TAttrValue>  := nil ;
          LoadMap<string, TAttrValue>(map, LoadStringAttrValue,tag.v);
          Value.Attr := map;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

class procedure TSaveHelper.SaveListValue(const S: TpbSaver; const Value: TListValue);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.Ss.Count - 1 do
      h.Pb.writeRawBytes(Value.Ss[i]);
    if Value.Ss.Count > 0 then
      S.Pb.writeMessage(TListValue.ftSs, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.&Is.Count - 1 do
      h.Pb.writeRawVarint64(Value.&Is[i]);
    if Value.&Is.Count > 0 then
      S.Pb.writeMessage(TListValue.ftIs, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.Fs.Count - 1 do
    begin
      var vVar : Single := Value.Fs[i];
      h.Pb.writeRawData(@vVar, sizeof(Single));
    end;
    if  Value.Fs.Count > 0 then
      S.Pb.writeMessage(TListValue.ftFs, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.Bs.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.Bs[i]));
    if Value.Bs.Count > 0 then
      S.Pb.writeMessage(TListValue.ftBs, h.Pb^);
  finally
    h.Free;
  end;

  h.Init;
  try
    for i := 0 to Value.&Types.Count - 1 do
      h.Pb.writeRawVarint32(Ord(Value.&Types[i]));
    if Value.&Types.Count >0  then
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
        S.Pb.writeInt32(Value.ftType, Ord(Value.Value.value.AsType<Integer>));
      end;
    TAttrValue.ftShape:
      begin
        if Value.Value.value.AsType<TTensorShapeProto> <> nil then
          S.SaveObj<TTensorShapeProto>(Value.Value.value.AsType<TTensorShapeProto>, SaveTensorShapeProto, Value.ftShape);
      end;
    TAttrValue.ftTensor:
      begin
        if Value.Value.value.AsType<TTensorProto> <> nil then
          S.SaveObj<TTensorProto>(Value.Value.value.AsType<TTensorProto>, SaveTensorProto, Value.ftTensor);
      end;
    TAttrValue.ftList:
      begin
        if Value.Value.value.AsType<TListValue> <> nil then
          S.SaveObj<TListValue>(Value.Value.value.AsType<TListValue>, SaveListValue, Value.ftList);
      end;
    TAttrValue.ftFunc:
      begin
        if Value.Value.value.AsType<TNameAttrList> <> nil then
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
  if Value.Attr <> nil then
  begin
    h.Init;
    try
      for var it in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(it);
          S.Pb.writeMessage(TNameAttrList.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

{$EndRegion}

{$Region 'ProtoGen.CostGraph'}
procedure TLoadHelper.LoadInputInfo(var Value: TInputInfo);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TInputInfo.Create;
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
  Value := TOutputInfo.Create;
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
  Value := TNode.Create;
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
            Value.InputInfos.Add(v);
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
            Value.OutputInfos.Add(v);
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
            var vTipo : int32 :=0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.ControlInputs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.ControlInputs.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TAggregatedCost.Create;
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
  Value := TCostGraphDef.Create;
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
            Value.Nodes.Add(v);
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
            Value.Costs.Add(v);
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

class procedure TSaveHelper.SaveInputInfo(const S: TpbSaver; const Value: TInputInfo);
begin
  S.Pb.writeInt32(TInputInfo.ftPrecedingNode, Value.PrecedingNode);
  S.Pb.writeInt32(TInputInfo.ftPrecedingPort, Value.PrecedingPort);
end;

class procedure TSaveHelper.SaveOutputInfo(const S: TpbSaver; const Value: TOutputInfo);
begin
  S.Pb.writeInt64(TOutputInfo.ftSize, Value.Size);
  S.Pb.writeInt64(TOutputInfo.ftAliasInputPort, Value.AliasInputPort);
  if Value.Shape <> nil then
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
      h.Pb.writeRawVarint32(Value.ControlInputs[i]);
    if Value.ControlInputs.Count > 0 then
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

{$EndRegion}

{$Region 'ProtoGen.AllocationDescription'}
procedure TLoadHelper.LoadAllocationDescription(var Value: TAllocationDescription);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TAllocationDescription.Create;
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

class procedure TSaveHelper.SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription);
begin
  S.Pb.writeInt64(TAllocationDescription.ftRequestedBytes, Value.RequestedBytes);
  S.Pb.writeInt64(TAllocationDescription.ftAllocatedBytes, Value.AllocatedBytes);
  S.Pb.writeString(TAllocationDescription.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocationDescription.ftAllocationId, Value.AllocationId);
  S.Pb.writeBoolean(TAllocationDescription.ftHasSingleReference, Value.HasSingleReference);
  S.Pb.writeInt64(TAllocationDescription.ftPtr, Value.Ptr);
end;
{$EndRegion}

{$Region 'ProtoGen.Cluster'}
procedure TLoadHelper.LoadJobDef(var Value: TJobDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TJobDef.Create;
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
          Value.Tasks.AddOrSetValue(Pb.readInt32, Pb.readString);
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
  Value := TClusterDef.Create;
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
            Value.Jobs.Add(v);
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

class procedure TSaveHelper.SaveJobDef(const S: TpbSaver; const Value: TJobDef);
var
  h : TpbSaver;

begin
  S.Pb.writeString(TJobDef.ftName, Value.Name);
  if Value.Tasks <> nil then
  begin
    h.Init;
    try
      for var it in Value.Tasks do
      begin
          h.clear;
          h.SaveInt32String(it);
          S.Pb.writeMessage(TJobDef.ftTasks, h.Pb^);
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
{$EndRegion}

{$Region 'ProtoGen.CoordinationConfig'}
procedure TLoadHelper.LoadCoordinationServiceConfig(var Value: TCoordinationServiceConfig);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TCoordinationServiceConfig.Create;
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.CoordinatedJobss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.CoordinatedJobss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
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
      h.Pb.writeRawString(Value.CoordinatedJobss[i]);
    if Value.CoordinatedJobss.Count > 0 then
      S.Pb.writeMessage(TCoordinationServiceConfig.ftCoordinatedJobss, h.Pb^);
  finally
    h.Free;
  end;
end;

{$EndRegion}

{$Region 'ProtoGen.OpDef'}
procedure TLoadHelper.LoadArgDef(var Value: TArgDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TArgDef.Create;
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
            Value.HandleDatas.Add(v);
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
  Value := TAttrDef.Create;
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
  Value := TOpDef.Create;
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
            Value.InputArgs.Add(v);
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
            Value.OutputArgs.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftControlOutputs:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.ControlOutputs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.ControlOutputs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TOpDef.ftAttrs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrDef;
            LoadAttrDef(v);
            Value.Attrs.Add(v);
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
  Value := TOpDeprecation.Create;
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
  Value := TOpList.Create;
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
            Value.Ops.Add(v);
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
  if Value.ExperimentalFullType <> nil then
    S.SaveObj<TFullTypeDef>(Value.ExperimentalFullType, SaveFullTypeDef, TArgDef.ftExperimentalFullType);
end;

class procedure TSaveHelper.SaveAttrDef(const S: TpbSaver; const Value: TAttrDef);
begin
  S.Pb.writeString(TAttrDef.ftName, Value.Name);
  S.Pb.writeString(TAttrDef.ftType, Value.&Type);
  if Value.DefaultValue <> nil then
    S.SaveObj<TAttrValue>(Value.DefaultValue, SaveAttrValue, TAttrDef.ftDefaultValue);
  S.Pb.writeString(TAttrDef.ftDescription, Value.Description);
  S.Pb.writeBoolean(TAttrDef.ftHasMinimum, Value.HasMinimum);
  S.Pb.writeInt64(TAttrDef.ftMinimum, Value.Minimum);
  if Value.AllowedValues <> nil then
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
      h.Pb.writeRawString(Value.ControlOutputs[i]);
    if Value.ControlOutputs.Count > 0 then
      S.Pb.writeMessage(TOpDef.ftControlOutputs, h.Pb^);
  finally
    h.Free;
  end;
  if Value.Attrs.Count > 0 then
    S.SaveList<TAttrDef>(Value.Attrs, SaveAttrDef, TOpDef.ftAttrs);
  if Value.Deprecation <> nil then
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

{$EndRegion}

{$Region 'ProtoGen.FullType'}
procedure TLoadHelper.LoadFullTypeDef(var Value: TFullTypeDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TFullTypeDef.Create;
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
            Value.Argss.Add(v);
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
{$EndRegion}

{$Region 'ProtoGen.Debug'}
procedure TLoadHelper.LoadDebugTensorWatch(var Value: TDebugTensorWatch);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TDebugTensorWatch.Create;
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.DebugOpss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.DebugOpss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TDebugTensorWatch.ftDebugUrlss:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.DebugUrlss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.DebugUrlss.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TDebugOptions.Create;
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
            Value.DebugTensorWatchOptss.Add(v);
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
  Value := TDebuggedSourceFile.Create;
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Liness.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Liness.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TDebuggedSourceFiles.Create;
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
            Value.SourceFiless.Add(v);
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
      h.Pb.writeRawString(Value.DebugOpss[i]);
    if Value.DebugOpss.Count >0 then
      S.Pb.writeMessage(TDebugTensorWatch.ftDebugOpss, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DebugUrlss.Count - 1 do
      h.Pb.writeRawString(Value.DebugUrlss[i]);
    if Value.DebugUrlss.Count > 0 then
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
      h.Pb.writeRawString(Value.Liness[i]);
    if Value.Liness.Count > 0 then
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
{$EndRegion}

{$Region 'ProtoGen.Versions'}
procedure TLoadHelper.LoadVersionDef(var Value: TVersionDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TVersionDef.Create;
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
            var vTipo : int32 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.BadConsumerss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.BadConsumerss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
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
      h.Pb.writeRawVarint32(Value.BadConsumerss[i]);
    if Value.BadConsumerss.Count > 0 then
      S.Pb.writeMessage(TVersionDef.ftBadConsumerss, h.Pb^);
  finally
    h.Free;
  end;
end;
{$EndRegion}

{$Region 'ProtoGen.VerifierConfig'}
procedure TLoadHelper.LoadVerifierConfig(var Value: TVerifierConfig);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TVerifierConfig.Create;
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

class procedure TSaveHelper.SaveVerifierConfig(const S: TpbSaver; const Value: TVerifierConfig);
begin
  S.Pb.writeInt64(TVerifierConfig.ftVerificationTimeoutInMs, Value.VerificationTimeoutInMs);
  S.Pb.writeInt32(TVerifierConfig.ftStructureVerifier, Ord(Value.StructureVerifier));
end;
{$EndRegion}

{$Region 'ProtoGen.Variable'}
procedure TLoadHelper.LoadVariableDef(var Value: TVariableDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TVariableDef.Create;
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
            var v : TSaveSliceInfoDef := Value.SaveSliceInfoDef;
            LoadSaveSliceInfoDef(v);
            Value.SaveSliceInfoDef := v;
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
  Value := TSaveSliceInfoDef.Create;
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
            var vTipo : int64 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.FullShapes.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.FullShapes.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TSaveSliceInfoDef.ftVarOffsets:
        begin
            var vTipo : int64 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.VarOffsets.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.VarOffsets.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TSaveSliceInfoDef.ftVarShapes:
        begin
            var vTipo : int64 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.VarShapes.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.VarShapes.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

class procedure TSaveHelper.SaveVariableDef(const S: TpbSaver; const Value: TVariableDef);
begin
  S.Pb.writeString(TVariableDef.ftVariableName, Value.VariableName);
  S.Pb.writeString(TVariableDef.ftInitialValueName, Value.InitialValueName);
  S.Pb.writeString(TVariableDef.ftInitializerName, Value.InitializerName);
  S.Pb.writeString(TVariableDef.ftSnapshotName, Value.SnapshotName);
  if Value.SaveSliceInfoDef <> nil then
    S.SaveObj<TSaveSliceInfoDef>(Value.SaveSliceInfoDef, SaveSaveSliceInfoDef, TVariableDef.ftSaveSliceInfoDef);
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
      h.Pb.writeRawVarint64(Value.FullShapes[i]);
    if  Value.FullShapes.Count > 0 then
      S.Pb.writeMessage(TSaveSliceInfoDef.ftFullShapes, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.VarOffsets.Count - 1 do
      h.Pb.writeRawVarint64(Value.VarOffsets[i]);
    if  Value.VarOffsets.Count > 0 then
      S.Pb.writeMessage(TSaveSliceInfoDef.ftVarOffsets, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.VarShapes.Count - 1 do
      h.Pb.writeRawVarint64(Value.VarShapes[i]);
    if Value.VarShapes.Count > 0 then
      S.Pb.writeMessage(TSaveSliceInfoDef.ftVarShapes, h.Pb^);
  finally
    h.Free;
  end;
end;

{$EndRegion}

{$Region 'ProtoGen.Function'}
procedure TLoadHelper.LoadFunctionDefLibrary(var Value: TFunctionDefLibrary);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TFunctionDefLibrary.Create;
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
            Value.Functions.Add(v);
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
            Value.Gradients.Add(v);
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
            Value.RegisteredGradientss.Add(v);
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
  Value := TArgAttrs.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TArgAttrs.ftAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);

          var map : TDictionary<string, TAttrValue>  := nil ;
          LoadMap<string, TAttrValue>(map, LoadStringAttrValue,tag.v);
          Value.Attr := map;
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
  Value := TFunctionDef.Create;
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
          Assert(wireType = TWire.LENGTH_DELIMITED);

          var map : TDictionary<string, TAttrValue>  := nil ;
          LoadMap<string, TAttrValue>(map, LoadStringAttrValue,tag.v);
          Value.Attr := map;
        end;
      TFunctionDef.ftArgAttr:
        begin
          var v1 : TArgAttrs;
          LoadArgAttrs(v1);
          Value.ArgAttr.AddOrSetValue(Pb.readUint32, v1);
        end;
      TFunctionDef.ftResourceArgUniqueId:
        begin
          Value.ResourceArgUniqueId.AddOrSetValue(Pb.readUint32, Pb.readUint32);
        end;
      TFunctionDef.ftNodeDefs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeDef;
            LoadNodeDef(v);
            Value.NodeDefs.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionDef.ftRet:
        begin
          Value.Ret.AddOrSetValue(Pb.readString, Pb.readString);
        end;
      TFunctionDef.ftControlRet:
        begin
          Value.ControlRet.AddOrSetValue(Pb.readString, Pb.readString);
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
  Value := TGradientDef.Create;
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
  Value := TRegisteredGradient.Create;
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
  if Value.Attr <> nil then
  begin
    h.Init;
    try
      for var it in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(it);
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
  if Value.Signature <> nil then
    S.SaveObj<TOpDef>(Value.Signature, SaveOpDef, TFunctionDef.ftSignature);
  if Value.Attr <> nil then
  begin
    h.Init;
    try
      for var it in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(it);
          S.Pb.writeMessage(TFunctionDef.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ArgAttr <> nil then
  begin
    h.Init;
    try
      for var it in Value.ArgAttr do
      begin
          h.clear;
          h.SaveUint32ArgAttrs(it);
          S.Pb.writeMessage(TFunctionDef.ftArgAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ResourceArgUniqueId <> nil then
  begin
    h.Init;
    try
      for var it in Value.ResourceArgUniqueId do
      begin
          h.clear;
          h.SaveUint32Uint32(it);
          S.Pb.writeMessage(TFunctionDef.ftResourceArgUniqueId, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.NodeDefs.Count > 0 then
    S.SaveList<TNodeDef>(Value.NodeDefs, SaveNodeDef, TFunctionDef.ftNodeDefs);
  if Value.Ret <> nil then
  begin
    h.Init;
    try
      for var it in Value.Ret do
      begin
          h.clear;
          h.SaveStringString(it);
          S.Pb.writeMessage(TFunctionDef.ftRet, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ControlRet <> nil then
  begin
    h.Init;
    try
      for var it in Value.ControlRet do
      begin
          h.clear;
          h.SaveStringString(it);
          S.Pb.writeMessage(TFunctionDef.ftControlRet, h.Pb^);
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
{$EndRegion}

{$Region 'ProtoGen.CppShapeInference'}
procedure TLoadHelper.LoadHandleShapeAndType(var Value: THandleShapeAndType);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := THandleShapeAndType.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      THandleShapeAndType.ftShape:
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
      THandleShapeAndType.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      THandleShapeAndType.ftType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.&Type;
            LoadFullTypeDef(v);
            Value.&Type := v;
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

procedure TLoadHelper.LoadHandleData(var Value: THandleData);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := THandleData.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      THandleData.ftIsSet:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsSet := Pb.readBoolean;
        end;
      THandleData.ftShapeAndTypes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : THandleShapeAndType;
            LoadHandleShapeAndType(v);
            Value.ShapeAndTypes.Add(v);
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

procedure TLoadHelper.LoadCppShapeInferenceResult(var Value: TCppShapeInferenceResult);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TCppShapeInferenceResult.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCppShapeInferenceResult.ftShape:
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
      TCppShapeInferenceResult.ftHandleData:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : THandleData := Value.HandleData;
            LoadHandleData(v);
            Value.HandleData := v;
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

procedure TLoadHelper.LoadCppShapeInferenceInputsNeeded(var Value: TCppShapeInferenceInputsNeeded);
var
  fieldNumber : integer;
  tag: TpbTag;
begin
  Value := TCppShapeInferenceInputsNeeded.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCppShapeInferenceInputsNeeded.ftInputTensorsNeededs:
        begin
            var vTipo : int32 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.InputTensorsNeededs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.InputTensorsNeededs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TCppShapeInferenceInputsNeeded.ftInputTensorsAsShapesNeededs:
        begin
            var vTipo : int32 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.InputTensorsAsShapesNeededs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.InputTensorsAsShapesNeededs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

class procedure TSaveHelper.SaveHandleShapeAndType(const S: TpbSaver; const Value: THandleShapeAndType);
begin
  if Value.Shape <> nil then
    S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, THandleShapeAndType.ftShape);
  S.Pb.writeInt32(THandleShapeAndType.ftDtype, Ord(Value.Dtype));
  if Value.&Type <> nil then
    S.SaveObj<TFullTypeDef>(Value.&Type, SaveFullTypeDef, THandleShapeAndType.ftType);
end;

class procedure TSaveHelper.SaveHandleData(const S: TpbSaver; const Value: THandleData);
begin
  S.Pb.writeBoolean(THandleData.ftIsSet, Value.IsSet);
  if Value.ShapeAndTypes.Count > 0 then
    S.SaveList<THandleShapeAndType>(Value.ShapeAndTypes, SaveHandleShapeAndType, THandleData.ftShapeAndTypes);
end;

class procedure TSaveHelper.SaveCppShapeInferenceResult(const S: TpbSaver; const Value: TCppShapeInferenceResult);
begin
  if Value.Shape <> nil then
    S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TCppShapeInferenceResult.ftShape);
  if Value.HandleData <> nil then
    S.SaveObj<THandleData>(Value.HandleData, SaveHandleData, TCppShapeInferenceResult.ftHandleData);
end;

class procedure TSaveHelper.SaveCppShapeInferenceInputsNeeded(const S: TpbSaver; const Value: TCppShapeInferenceInputsNeeded);
var
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.InputTensorsNeededs.Count - 1 do
      h.Pb.writeRawVarint32(Value.InputTensorsNeededs[i]);
    if Value.InputTensorsNeededs.Count > 0 then
      S.Pb.writeMessage(TCppShapeInferenceInputsNeeded.ftInputTensorsNeededs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.InputTensorsAsShapesNeededs.Count - 1 do
      h.Pb.writeRawVarint32(Value.InputTensorsAsShapesNeededs[i]);
    if Value.InputTensorsAsShapesNeededs.Count > 0 then
      S.Pb.writeMessage(TCppShapeInferenceInputsNeeded.ftInputTensorsAsShapesNeededs, h.Pb^);
  finally
    h.Free;
  end;
end;
{$EndRegion}

{$Region 'ProtoGen.RewriterConfig'}
procedure TLoadHelper.LoadAutoParallelOptions(var Value: TAutoParallelOptions);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TAutoParallelOptions.Create;
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
  fieldNumber: integer;
  tag: TpbTag;
begin
  Value := TScopedAllocatorOptions.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TScopedAllocatorOptions.ftEnableOps:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.EnableOps.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.EnableOps.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TCustomGraphOptimizer.Create;
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
         Assert(wireType = TWire.LENGTH_DELIMITED);

          var map : TDictionary<string, TAttrValue>  := nil ;
          LoadMap<string, TAttrValue>(map, LoadStringAttrValue,tag.v);
          Value.ParameterMap := map;
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
  Value := TRewriterConfig.Create;
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Optimizerss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Optimizerss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TRewriterConfig.ftCustomOptimizerss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCustomGraphOptimizer;
            LoadCustomGraphOptimizer(v);
            Value.CustomOptimizerss.Add(v);
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
      h.Pb.writeRawString(Value.EnableOps[i]);
    if Value.EnableOps.Count > 0 then
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
  if Value.ParameterMap <> nil then
  begin
    h.Init;
    try
      for var it in Value.ParameterMap do
      begin
          h.clear;
          h.SaveStringAttrValue(it);
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
  if Value.AutoParallel <> nil then
    S.SaveObj<TAutoParallelOptions>(Value.AutoParallel, SaveAutoParallelOptions, TRewriterConfig.ftAutoParallel);
  S.Pb.writeBoolean(TRewriterConfig.ftFailOnOptimizerErrors, Value.FailOnOptimizerErrors);
  if Value.ScopedAllocatorOpts <> nil then
    S.SaveObj<TScopedAllocatorOptions>(Value.ScopedAllocatorOpts, SaveScopedAllocatorOptions, TRewriterConfig.ftScopedAllocatorOpts);
  h.Init;
  try
    for i := 0 to Value.Optimizerss.Count - 1 do
      h.Pb.writeRawString(Value.Optimizerss[i]);
    if  Value.Optimizerss.Count > 0 then
      S.Pb.writeMessage(TRewriterConfig.ftOptimizerss, h.Pb^);
  finally
    h.Free;
  end;
  if Value.CustomOptimizerss.Count > 0 then
    S.SaveList<TCustomGraphOptimizer>(Value.CustomOptimizerss, SaveCustomGraphOptimizer, TRewriterConfig.ftCustomOptimizerss);
  if Value.InterOptimizerVerifierConfig <> nil then
    S.SaveObj<TVerifierConfig>(Value.InterOptimizerVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftInterOptimizerVerifierConfig);
  if Value.PostOptimizationVerifierConfig <> nil then
    S.SaveObj<TVerifierConfig>(Value.PostOptimizationVerifierConfig, SaveVerifierConfig, TRewriterConfig.ftPostOptimizationVerifierConfig);
end;

{$EndRegion}

{$Region 'ProtoGen.Graph'}
procedure TLoadHelper.LoadGraphDef(var Value: TGraphDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TGraphDef.Create;
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
            Value.Nodes.Add(v);
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

class procedure TSaveHelper.SaveGraphDef(const S: TpbSaver; const Value: TGraphDef);
begin
  if Value.Nodes.Count > 0 then
    S.SaveList<TNodeDef>(Value.Nodes, SaveNodeDef, TGraphDef.ftNodes);
  if Value.Versions <> nil then
    S.SaveObj<TVersionDef>(Value.Versions, SaveVersionDef, TGraphDef.ftVersions);
  S.Pb.writeInt32(TGraphDef.ftVersion, Value.Version);
  if Value.&Library <> nil then
    S.SaveObj<TFunctionDefLibrary>(Value.&Library, SaveFunctionDefLibrary, TGraphDef.ftLibrary);
end;

{$EndRegion}

{$Region 'ProtoGen.TensorDescription'}
procedure TLoadHelper.LoadTensorDescription(var Value: TTensorDescription);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TTensorDescription.Create;
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

class procedure TSaveHelper.SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription);
begin
  S.Pb.writeInt32(TTensorDescription.ftDtype, Ord(Value.Dtype));
  if Value.Shape <> nil then
    S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TTensorDescription.ftShape);
  if Value.AllocationDescription <> nil then
    S.SaveObj<TAllocationDescription>(Value.AllocationDescription, SaveAllocationDescription, TTensorDescription.ftAllocationDescription);
end;
{$EndRegion}

{$Region 'ProtoGen.NodeDef'}
procedure TLoadHelper.LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
var
  fieldNumber: integer;
  tag: TpbTag;
begin
  Value := TExperimentalDebugInfo.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TExperimentalDebugInfo.ftOriginalNodeNamess:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.OriginalNodeNamess.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.OriginalNodeNamess.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TExperimentalDebugInfo.ftOriginalFuncNamess:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.OriginalFuncNamess.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.OriginalFuncNamess.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TNodeDef.Create;
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Inputs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Inputs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TNodeDef.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TNodeDef.ftAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);

          var map : TDictionary<string, TAttrValue>  := nil ;
          LoadMap<string, TAttrValue>(map, LoadStringAttrValue,tag.v);
          Value.Attr := map;
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

class procedure TSaveHelper.SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.OriginalNodeNamess.Count - 1 do
      h.Pb.writeRawString(Value.OriginalNodeNamess[i]);
    if Value.OriginalNodeNamess.Count > 0 then
      S.Pb.writeMessage(TExperimentalDebugInfo.ftOriginalNodeNamess, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.OriginalFuncNamess.Count - 1 do
      h.Pb.writeRawString(Value.OriginalFuncNamess[i]);
    if Value.OriginalFuncNamess.Count > 0 then
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
      h.Pb.writeRawString(Value.Inputs[i]);
    if Value.Inputs.Count > 0 then
      S.Pb.writeMessage(TNodeDef.ftInputs, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeString(TNodeDef.ftDevice, Value.Device);
  if Value.Attr <> nil then
  begin
    h.Init;
    try
      for var it in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(it);
          S.Pb.writeMessage(TNodeDef.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.ExperimentalDebugInfo <> nil then
    S.SaveObj<TExperimentalDebugInfo>(Value.ExperimentalDebugInfo, SaveExperimentalDebugInfo, TNodeDef.ftExperimentalDebugInfo);
  if Value.ExperimentalType <> nil then
    S.SaveObj<TFullTypeDef>(Value.ExperimentalType, SaveFullTypeDef, TNodeDef.ftExperimentalType);
end;

{$EndRegion}

{$Region 'ProtoGen.Config'}
procedure TLoadHelper.LoadVirtualDevices(var Value: TVirtualDevices);
var
  fieldNumber : integer;
  tag: TpbTag;
begin
  Value := TVirtualDevices.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TVirtualDevices.ftMemoryLimitMbs:
        begin
            var vTipo : single := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : single := Pb.readFloat;
                  Value.MemoryLimitMbs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : single := Pb.readFloat;
                Value.MemoryLimitMbs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TVirtualDevices.ftPrioritys:
        begin
            var vTipo : int32 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int32 := Pb.readInt32;
                  Value.Prioritys.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int32 := Pb.readInt32;
                Value.Prioritys.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadExperimental(var Value: TGPUOptions.TExperimental);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TGPUOptions.TExperimental.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TGPUOptions.TExperimental.ftVirtualDevicess:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVirtualDevices;
            LoadVirtualDevices(v);
            Value.VirtualDevicess.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TGPUOptions.TExperimental.ftUseUnifiedMemory:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseUnifiedMemory := Pb.readBoolean;
        end;
      TGPUOptions.TExperimental.ftNumDevToDevCopyStreams:
        begin
          Assert(wireType = TWire.VARINT);
          Value.NumDevToDevCopyStreams := Pb.readInt32;
        end;
      TGPUOptions.TExperimental.ftCollectiveRingOrder:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.CollectiveRingOrder := Pb.readString;
        end;
      TGPUOptions.TExperimental.ftTimestampedAllocator:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TimestampedAllocator := Pb.readBoolean;
        end;
      TGPUOptions.TExperimental.ftKernelTrackerMaxInterval:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxInterval := Pb.readInt32;
        end;
      TGPUOptions.TExperimental.ftKernelTrackerMaxBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxBytes := Pb.readInt32;
        end;
      TGPUOptions.TExperimental.ftKernelTrackerMaxPending:
        begin
          Assert(wireType = TWire.VARINT);
          Value.KernelTrackerMaxPending := Pb.readInt32;
        end;
      TGPUOptions.TExperimental.ftInternalFragmentationFraction:
        begin
          Assert(wireType = TWire.FIXED64);
          Value.InternalFragmentationFraction := Pb.readDouble;
        end;
      TGPUOptions.TExperimental.ftUseCudaMallocAsync:
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

procedure TLoadHelper.LoadExperimental(var Value: TConfigProto.TExperimental);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TConfigProto.TExperimental.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TConfigProto.TExperimental.ftCollectiveGroupLeader:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.CollectiveGroupLeader := Pb.readString;
        end;
      TConfigProto.TExperimental.ftExecutorType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ExecutorType := Pb.readString;
        end;
      TConfigProto.TExperimental.ftRecvBufMaxChunk:
        begin
          Assert(wireType = TWire.VARINT);
          Value.RecvBufMaxChunk := Pb.readInt32;
        end;
      TConfigProto.TExperimental.ftUseNumaAffinity:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseNumaAffinity := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftCollectiveDeterministicSequentialExecution:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveDeterministicSequentialExecution := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftCollectiveNccl:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveNccl := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftShareSessionStateInClusterspecPropagation:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShareSessionStateInClusterspecPropagation := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftDisableThreadSpinning:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableThreadSpinning := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftShareClusterDevicesInSession:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ShareClusterDevicesInSession := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftSessionMetadata:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TSessionMetadata := Value.SessionMetadata;
            LoadSessionMetadata(v);
            Value.SessionMetadata := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.TExperimental.ftOptimizeForStaticGraph:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OptimizeForStaticGraph := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftEnableMlirBridge:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableMlirBridge := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftMlirBridgeRollout:
        begin
          Assert(wireType = TWire.VARINT);
          Value.MlirBridgeRollout := TMlirBridgeRollout(Pb.readInt32);
        end;
      TConfigProto.TExperimental.ftEnableMlirGraphOptimization:
        begin
          Assert(wireType = TWire.VARINT);
          Value.EnableMlirGraphOptimization := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftDisableOutputPartitionGraphs:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableOutputPartitionGraphs := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftXlaFusionAutotunerThresh:
        begin
          Assert(wireType = TWire.VARINT);
          Value.XlaFusionAutotunerThresh := Pb.readInt64;
        end;
      TConfigProto.TExperimental.ftUseTfrt:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseTfrt := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftDisableFunctionalOpsLowering:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DisableFunctionalOpsLowering := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftXlaPreferSingleGraphCluster:
        begin
          Assert(wireType = TWire.VARINT);
          Value.XlaPreferSingleGraphCluster := Pb.readBoolean;
        end;
      TConfigProto.TExperimental.ftCoordinationConfig:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCoordinationServiceConfig := Value.CoordinationConfig;
            LoadCoordinationServiceConfig(v);
            Value.CoordinationConfig := v;
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

procedure TLoadHelper.LoadExperimental(var Value: TRunOptions.TExperimental);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TRunOptions.TExperimental.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TRunOptions.TExperimental.ftCollectiveGraphKey:
        begin
          Assert(wireType = TWire.VARINT);
          Value.CollectiveGraphKey := Pb.readInt64;
        end;
      TRunOptions.TExperimental.ftUseRunHandlerPool:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UseRunHandlerPool := Pb.readBoolean;
        end;
      TRunOptions.TExperimental.ftRunHandlerPoolOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRunHandlerPoolOptions := Value.RunHandlerPoolOptions;
            LoadRunHandlerPoolOptions(v);
            Value.RunHandlerPoolOptions := v;
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

procedure TLoadHelper.LoadGPUOptions(var Value: TGPUOptions);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TGPUOptions.Create;
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
            var v : TGPUOptions.TExperimental := Value.Experimental;
            LoadExperimental(v);
            Value.Experimental := v;
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
  Value := TOptimizerOptions.Create;
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
  Value := TGraphOptions.Create;
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
            var v : TOptimizerOptions := Value.OptimizerOptions;
            LoadOptimizerOptions(v);
            Value.OptimizerOptions := v;
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
            var v : TRewriterConfig := Value.RewriteOptions;
            LoadRewriterConfig(v);
            Value.RewriteOptions := v;
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
  Value := TThreadPoolOptionProto.Create;
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
  Value := TRPCOptions.Create;
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
  Value := TSessionMetadata.Create;
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

procedure TLoadHelper.LoadConfigProto(var Value: TConfigProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TConfigProto.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TConfigProto.ftDeviceCount:
        begin
          Value.DeviceCount.AddOrSetValue(Pb.readString, Pb.readInt32);
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
            Value.SessionInterOpThreadPools.Add(v);
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
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.DeviceFilterss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.DeviceFilterss.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TConfigProto.ftGpuOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGPUOptions := Value.GpuOptions;
            LoadGPUOptions(v);
            Value.GpuOptions := v;
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
            var v : TGraphOptions := Value.GraphOptions;
            LoadGraphOptions(v);
            Value.GraphOptions := v;
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
            var v : TRPCOptions := Value.RpcOptions;
            LoadRPCOptions(v);
            Value.RpcOptions := v;
          finally
            Pb.Pop;
          end;
        end;
      TConfigProto.ftClusterDef:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TClusterDef := Value.ClusterDef;
            LoadClusterDef(v);
            Value.ClusterDef := v;
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
            var v : TConfigProto.TExperimental := Value.Experimental;
            LoadExperimental(v);
            Value.Experimental := v;
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
  Value := TRunHandlerPoolOptions.Create;
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

procedure TLoadHelper.LoadRunOptions(var Value: TRunOptions);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TRunOptions.Create;
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
            var v : TDebugOptions := Value.DebugOptions;
            LoadDebugOptions(v);
            Value.DebugOptions := v;
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
            var v : TRunOptions.TExperimental := Value.Experimental;
            LoadExperimental(v);
            Value.Experimental := v;
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
  Value := TFunctionGraphs.Create;
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
            Value.PartitionGraphss.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TFunctionGraphs.ftPreOptimizationGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef := Value.PreOptimizationGraph;
            LoadGraphDef(v);
            Value.PreOptimizationGraph := v;
          finally
            Pb.Pop;
          end;
        end;
      TFunctionGraphs.ftPostOptimizationGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TGraphDef := Value.PostOptimizationGraph;
            LoadGraphDef(v);
            Value.PostOptimizationGraph := v;
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
  Value := TRunMetadata.Create;
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
            var v : TStepStats := Value.StepStats;
            LoadStepStats(v);
            Value.StepStats := v;
          finally
            Pb.Pop;
          end;
        end;
      TRunMetadata.ftCostGraph:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TCostGraphDef := Value.CostGraph;
            LoadCostGraphDef(v);
            Value.CostGraph := v;
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
            Value.PartitionGraphss.Add(v);
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
            Value.FunctionGraphss.Add(v);
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
  Value := TTensorConnection.Create;
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
  Value := TCallableOptions.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCallableOptions.ftFeeds:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Feeds.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Feeds.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TCallableOptions.ftFetchs:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Fetchs.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Fetchs.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TCallableOptions.ftTargets:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Targets.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Targets.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TCallableOptions.ftRunOptions:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TRunOptions := Value.RunOptions;
            LoadRunOptions(v);
            Value.RunOptions := v;
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
            Value.TensorConnections.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TCallableOptions.ftFeedDevices:
        begin
          Value.FeedDevices.AddOrSetValue(Pb.readString, Pb.readString);
        end;
      TCallableOptions.ftFetchDevices:
        begin
          Value.FetchDevices.AddOrSetValue(Pb.readString, Pb.readString);
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

class procedure TSaveHelper.SaveVirtualDevices(const S: TpbSaver; const Value: TVirtualDevices);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.MemoryLimitMbs.Count - 1 do
    begin
      var vVar : Single := Value.MemoryLimitMbs[i];
      h.Pb.writeRawData(@vVar, sizeof(Single));
    end;
    if Value.MemoryLimitMbs.Count > 0 then
      S.Pb.writeMessage(TVirtualDevices.ftMemoryLimitMbs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Prioritys.Count - 1 do
      h.Pb.writeRawVarint32(Value.Prioritys[i]);
    if Value.Prioritys.Count > 0 then
      S.Pb.writeMessage(TVirtualDevices.ftPrioritys, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveExperimental(const S: TpbSaver; const Value: TGPUOptions.TExperimental);
begin
  if Value.VirtualDevicess.Count > 0 then
    S.SaveList<TVirtualDevices>(Value.VirtualDevicess, SaveVirtualDevices, TGPUOptions.TExperimental.ftVirtualDevicess);
  S.Pb.writeBoolean(TGPUOptions.TExperimental.ftUseUnifiedMemory, Value.UseUnifiedMemory);
  S.Pb.writeInt32(TGPUOptions.TExperimental.ftNumDevToDevCopyStreams, Value.NumDevToDevCopyStreams);
  S.Pb.writeString(TGPUOptions.TExperimental.ftCollectiveRingOrder, Value.CollectiveRingOrder);
  S.Pb.writeBoolean(TGPUOptions.TExperimental.ftTimestampedAllocator, Value.TimestampedAllocator);
  S.Pb.writeInt32(TGPUOptions.TExperimental.ftKernelTrackerMaxInterval, Value.KernelTrackerMaxInterval);
  S.Pb.writeInt32(TGPUOptions.TExperimental.ftKernelTrackerMaxBytes, Value.KernelTrackerMaxBytes);
  S.Pb.writeInt32(TGPUOptions.TExperimental.ftKernelTrackerMaxPending, Value.KernelTrackerMaxPending);
  S.Pb.writeDouble(TGPUOptions.TExperimental.ftInternalFragmentationFraction, Value.InternalFragmentationFraction);
  S.Pb.writeBoolean(TGPUOptions.TExperimental.ftUseCudaMallocAsync, Value.UseCudaMallocAsync);
end;

class procedure TSaveHelper.SaveExperimental(const S: TpbSaver; const Value: TConfigProto.TExperimental);
begin
  S.Pb.writeString(TConfigProto.TExperimental.ftCollectiveGroupLeader, Value.CollectiveGroupLeader);
  S.Pb.writeString(TConfigProto.TExperimental.ftExecutorType, Value.ExecutorType);
  S.Pb.writeInt32(TConfigProto.TExperimental.ftRecvBufMaxChunk, Value.RecvBufMaxChunk);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftUseNumaAffinity, Value.UseNumaAffinity);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftCollectiveDeterministicSequentialExecution, Value.CollectiveDeterministicSequentialExecution);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftCollectiveNccl, Value.CollectiveNccl);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftShareSessionStateInClusterspecPropagation, Value.ShareSessionStateInClusterspecPropagation);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftDisableThreadSpinning, Value.DisableThreadSpinning);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftShareClusterDevicesInSession, Value.ShareClusterDevicesInSession);
  if Value.SessionMetadata <> nil then
    S.SaveObj<TSessionMetadata>(Value.SessionMetadata, SaveSessionMetadata, TConfigProto.TExperimental.ftSessionMetadata);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftOptimizeForStaticGraph, Value.OptimizeForStaticGraph);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftEnableMlirBridge, Value.EnableMlirBridge);
  S.Pb.writeInt32(TConfigProto.TExperimental.ftMlirBridgeRollout, Ord(Value.MlirBridgeRollout));
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftEnableMlirGraphOptimization, Value.EnableMlirGraphOptimization);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftDisableOutputPartitionGraphs, Value.DisableOutputPartitionGraphs);
  S.Pb.writeInt64(TConfigProto.TExperimental.ftXlaFusionAutotunerThresh, Value.XlaFusionAutotunerThresh);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftUseTfrt, Value.UseTfrt);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftDisableFunctionalOpsLowering, Value.DisableFunctionalOpsLowering);
  S.Pb.writeBoolean(TConfigProto.TExperimental.ftXlaPreferSingleGraphCluster, Value.XlaPreferSingleGraphCluster);
  if Value.CoordinationConfig <> nil then
    S.SaveObj<TCoordinationServiceConfig>(Value.CoordinationConfig, SaveCoordinationServiceConfig, TConfigProto.TExperimental.ftCoordinationConfig);
end;

class procedure TSaveHelper.SaveExperimental(const S: TpbSaver; const Value: TRunOptions.TExperimental);
begin
  S.Pb.writeInt64(TRunOptions.TExperimental.ftCollectiveGraphKey, Value.CollectiveGraphKey);
  S.Pb.writeBoolean(TRunOptions.TExperimental.ftUseRunHandlerPool, Value.UseRunHandlerPool);
  if Value.RunHandlerPoolOptions <> nil then
    S.SaveObj<TRunHandlerPoolOptions>(Value.RunHandlerPoolOptions, SaveRunHandlerPoolOptions, TRunOptions.TExperimental.ftRunHandlerPoolOptions);
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
  if Value.Experimental <> nil then
    S.SaveObj<TGPUOptions.TExperimental>(Value.Experimental, SaveExperimental, TGPUOptions.ftExperimental);
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
  if Value.OptimizerOptions <> nil then
    S.SaveObj<TOptimizerOptions>(Value.OptimizerOptions, SaveOptimizerOptions, TGraphOptions.ftOptimizerOptions);
  S.Pb.writeInt64(TGraphOptions.ftBuildCostModel, Value.BuildCostModel);
  S.Pb.writeInt64(TGraphOptions.ftBuildCostModelAfter, Value.BuildCostModelAfter);
  S.Pb.writeBoolean(TGraphOptions.ftInferShapes, Value.InferShapes);
  S.Pb.writeBoolean(TGraphOptions.ftPlacePrunedGraph, Value.PlacePrunedGraph);
  S.Pb.writeBoolean(TGraphOptions.ftEnableBfloat16Sendrecv, Value.EnableBfloat16Sendrecv);
  S.Pb.writeInt32(TGraphOptions.ftTimelineStep, Value.TimelineStep);
  if Value.RewriteOptions <> nil then
    S.SaveObj<TRewriterConfig>(Value.RewriteOptions, SaveRewriterConfig, TGraphOptions.ftRewriteOptions);
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

class procedure TSaveHelper.SaveConfigProto(const S: TpbSaver; const Value: TConfigProto);
var 
  i : Integer;
  h : TpbSaver;

begin
  if Value.DeviceCount <> nil then
  begin
    h.Init;
    try
      for var it in Value.DeviceCount do
      begin
          h.clear;
          h.SaveStringInt32(it);
          S.Pb.writeMessage(TConfigProto.ftDeviceCount, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;

  S.Pb.writeInt32(TConfigProto.ftIntraOpParallelismThreads, Value.IntraOpParallelismThreads);
  S.Pb.writeInt32(TConfigProto.ftInterOpParallelismThreads, Value.InterOpParallelismThreads);
  S.Pb.writeBoolean(TConfigProto.ftUsePerSessionThreads, Value.UsePerSessionThreads);

  if (Value.SessionInterOpThreadPools <> nil) and (Value.SessionInterOpThreadPools.Count > 0) then
    S.SaveList<TThreadPoolOptionProto>(Value.SessionInterOpThreadPools, SaveThreadPoolOptionProto, TConfigProto.ftSessionInterOpThreadPools);
  S.Pb.writeInt32(TConfigProto.ftPlacementPeriod, Value.PlacementPeriod);

  h.Init;
  try
    if Assigned(Value.DeviceFilterss) then
    begin
        for i := 0 to Value.DeviceFilterss.Count - 1 do
          h.Pb.writeRawString(Value.DeviceFilterss[i]);
        if Value.DeviceFilterss.Count > 0 then
         S.Pb.writeMessage(TConfigProto.ftDeviceFilterss, h.Pb^);
    end;
  finally
    h.Free;
  end;

  if Value.GpuOptions <> nil then
    S.SaveObj<TGPUOptions>(Value.GpuOptions, SaveGPUOptions, TConfigProto.ftGpuOptions);

  S.Pb.writeBoolean(TConfigProto.ftAllowSoftPlacement, Value.AllowSoftPlacement);
  S.Pb.writeBoolean(TConfigProto.ftLogDevicePlacement, Value.LogDevicePlacement);

  if Value.GraphOptions <> nil then
    S.SaveObj<TGraphOptions>(Value.GraphOptions, SaveGraphOptions, TConfigProto.ftGraphOptions);

  S.Pb.writeInt64(TConfigProto.ftOperationTimeoutInMs, Value.OperationTimeoutInMs);

  if Value.RpcOptions <> nil then
    S.SaveObj<TRPCOptions>(Value.RpcOptions, SaveRPCOptions, TConfigProto.ftRpcOptions);

  if Value.ClusterDef <> nil then
    S.SaveObj<TClusterDef>(Value.ClusterDef, SaveClusterDef, TConfigProto.ftClusterDef);

  S.Pb.writeBoolean(TConfigProto.ftIsolateSessionState, Value.IsolateSessionState);
  S.Pb.writeBoolean(TConfigProto.ftShareClusterDevicesInSession, Value.ShareClusterDevicesInSession);

  if Value.Experimental <> nil then
     S.SaveObj<TConfigProto.TExperimental>(Value.Experimental, SaveExperimental, TConfigProto.ftExperimental);
end;

class procedure TSaveHelper.SaveRunHandlerPoolOptions(const S: TpbSaver; const Value: TRunHandlerPoolOptions);
begin
  S.Pb.writeInt64(TRunHandlerPoolOptions.ftPriority, Value.Priority);
end;

class procedure TSaveHelper.SaveRunOptions(const S: TpbSaver; const Value: TRunOptions);
begin
  S.Pb.writeInt32(TRunOptions.ftTraceLevel, Ord(Value.TraceLevel));
  S.Pb.writeInt64(TRunOptions.ftTimeoutInMs, Value.TimeoutInMs);
  S.Pb.writeInt32(TRunOptions.ftInterOpThreadPool, Value.InterOpThreadPool);
  S.Pb.writeBoolean(TRunOptions.ftOutputPartitionGraphs, Value.OutputPartitionGraphs);
  if Value.DebugOptions <> nil then
    S.SaveObj<TDebugOptions>(Value.DebugOptions, SaveDebugOptions, TRunOptions.ftDebugOptions);
  S.Pb.writeBoolean(TRunOptions.ftReportTensorAllocationsUponOom, Value.ReportTensorAllocationsUponOom);
  if Value.Experimental <> nil then
    S.SaveObj<TRunOptions.TExperimental>(Value.Experimental, SaveExperimental, TRunOptions.ftExperimental);
end;

class procedure TSaveHelper.SaveFunctionGraphs(const S: TpbSaver; const Value: TFunctionGraphs);
begin
  if Value.PartitionGraphss.Count > 0 then
    S.SaveList<TGraphDef>(Value.PartitionGraphss, SaveGraphDef, TFunctionGraphs.ftPartitionGraphss);
  if Value.PreOptimizationGraph <> nil then
    S.SaveObj<TGraphDef>(Value.PreOptimizationGraph, SaveGraphDef, TFunctionGraphs.ftPreOptimizationGraph);
  if Value.PostOptimizationGraph <> nil then
    S.SaveObj<TGraphDef>(Value.PostOptimizationGraph, SaveGraphDef, TFunctionGraphs.ftPostOptimizationGraph);
end;

class procedure TSaveHelper.SaveRunMetadata(const S: TpbSaver; const Value: TRunMetadata);
begin
  if Value.StepStats <> nil then
    S.SaveObj<TStepStats>(Value.StepStats, SaveStepStats, TRunMetadata.ftStepStats);
  if Value.CostGraph <> nil then
    S.SaveObj<TCostGraphDef>(Value.CostGraph, SaveCostGraphDef, TRunMetadata.ftCostGraph);
  if Value.PartitionGraphss.Count > 0 then
    S.SaveList<TGraphDef>(Value.PartitionGraphss, SaveGraphDef, TRunMetadata.ftPartitionGraphss);
  if Value.FunctionGraphss.Count > 0 then
    S.SaveList<TFunctionGraphs>(Value.FunctionGraphss, SaveFunctionGraphs, TRunMetadata.ftFunctionGraphss);
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
      h.Pb.writeRawString(Value.Feeds[i]);
    if Value.Feeds.Count > 0  then
      S.Pb.writeMessage(TCallableOptions.ftFeeds, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Fetchs.Count - 1 do
      h.Pb.writeRawString(Value.Fetchs[i]);
    if Value.Fetchs.Count > 0 then
      S.Pb.writeMessage(TCallableOptions.ftFetchs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Targets.Count - 1 do
      h.Pb.writeRawString(Value.Targets[i]);
    if Value.Targets.Count > 0 then
      S.Pb.writeMessage(TCallableOptions.ftTargets, h.Pb^);
  finally
    h.Free;
  end;
  if Value.RunOptions <> nil then
    S.SaveObj<TRunOptions>(Value.RunOptions, SaveRunOptions, TCallableOptions.ftRunOptions);
  if Value.TensorConnections.Count > 0 then
    S.SaveList<TTensorConnection>(Value.TensorConnections, SaveTensorConnection, TCallableOptions.ftTensorConnections);
  if Value.FeedDevices <> nil then
  begin
    h.Init;
    try
      for var it in Value.FeedDevices do
      begin
          h.clear;
          h.SaveStringString(it);
          S.Pb.writeMessage(TCallableOptions.ftFeedDevices, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  if Value.FetchDevices <> nil then
  begin
    h.Init;
    try
      for var it in Value.FetchDevices do
      begin
          h.clear;
          h.SaveStringString(it);
          S.Pb.writeMessage(TCallableOptions.ftFetchDevices, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  S.Pb.writeBoolean(TCallableOptions.ftFetchSkipSync, Value.FetchSkipSync);
end;
{$EndRegion}

{$Region 'ProtoGen.StepStats'}
procedure TLoadHelper.LoadAllocationRecord(var Value: TAllocationRecord);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TAllocationRecord.Create;
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
  Value := TAllocatorMemoryUsed.Create;
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
            Value.AllocationRecordss.Add(v);
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
  Value := TNodeOutput.Create;
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
  Value := TMemoryStats.Create;
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
            var vTipo : int64 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.PersistentTensorAllocIdss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.PersistentTensorAllocIdss.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
            var vTipo : int64 := 0;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : int64 := Pb.readInt64;
                  Value.DevicePersistentTensorAllocIdss.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : int64 := Pb.readInt64;
                Value.DevicePersistentTensorAllocIdss.Add(v);
              until not Pb.ConsumeTag(tag.v);
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
  Value := TNodeExecStats.Create;
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
            Value.Memorys.Add(v);
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
            Value.Outputs.Add(v);
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
            Value.ReferencedTensors.Add(v);
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
  Value := TDeviceStepStats.Create;
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
            Value.NodeStatss.Add(v);
          finally
            Pb.Pop;
          end;
        end;
      TDeviceStepStats.ftThreadNames:
        begin
          Value.ThreadNames.AddOrSetValue(Pb.readUint32, Pb.readString);
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
  Value := TStepStats.Create;
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
            Value.DevStatss.Add(v);
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
  if Value.TensorDescription <> nil then
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
      h.Pb.writeRawVarint64(Value.PersistentTensorAllocIdss[i]);
    if Value.PersistentTensorAllocIdss.Count > 0 then
      S.Pb.writeMessage(TMemoryStats.ftPersistentTensorAllocIdss, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeInt64(TMemoryStats.ftDeviceTempMemorySize, Value.DeviceTempMemorySize);
  S.Pb.writeInt64(TMemoryStats.ftDevicePersistentMemorySize, Value.DevicePersistentMemorySize);
  h.Init;
  try
    for i := 0 to Value.DevicePersistentTensorAllocIdss.Count - 1 do
      h.Pb.writeRawVarint64(Value.DevicePersistentTensorAllocIdss[i]);
    if Value.DevicePersistentTensorAllocIdss.Count > 0 then
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
  if Value.MemoryStats <> nil then
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
  if Value.ThreadNames <> nil then
  begin
    h.Init;
    try
      for var it in Value.ThreadNames do
      begin
          h.clear;
          h.SaveUint32String(it);
          S.Pb.writeMessage(TDeviceStepStats.ftThreadNames, h.Pb^);
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
{$EndRegion}

{$Region 'ProtoGen.ControlFlow'}
procedure TLoadHelper.LoadValuesDef(var Value: TValuesDef);
var
  fieldNumber: integer;
  tag: TpbTag;
begin
  Value := TValuesDef.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TValuesDef.ftValuess:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.Valuess.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.Valuess.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TValuesDef.ftExternalValues:
        begin
          Value.ExternalValues.AddOrSetValue(Pb.readString, Pb.readString);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadControlFlowContextDef(var Value: TControlFlowContextDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TControlFlowContextDef.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TControlFlowContextDef.ftCondCtxt:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TControlFlowContextDef.ftCondCtxt;
          Pb.Push;
          try
              var v1 : TCondContextDef;
              LoadCondContextDef(v1);
              v.value := TValue.From<TCondContextDef>(v1);
              Value.ctxt := v;
          finally
           Pb.Pop
          end;
        end;
      TControlFlowContextDef.ftWhileCtxt:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TControlFlowContextDef.ftWhileCtxt;
          Pb.Push;
          try

              var v1 : TWhileContextDef;
              LoadWhileContextDef(v1);
              v.value := TValue.From<TWhileContextDef>(v1);
              Value.ctxt := v;
          finally
           Pb.Pop
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadCondContextDef(var Value: TCondContextDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TCondContextDef.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TCondContextDef.ftContextName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ContextName := Pb.readString;
        end;
      TCondContextDef.ftPredName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.PredName := Pb.readString;
        end;
      TCondContextDef.ftPivotName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.PivotName := Pb.readString;
        end;
      TCondContextDef.ftBranch:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Branch := Pb.readInt32;
        end;
      TCondContextDef.ftValuesDef:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TValuesDef := Value.ValuesDef;
            LoadValuesDef(v);
            Value.ValuesDef := v;
          finally
            Pb.Pop;
          end;
        end;
      TCondContextDef.ftNestedContextss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TControlFlowContextDef;
            LoadControlFlowContextDef(v);
            Value.NestedContextss.Add(v);
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

procedure TLoadHelper.LoadWhileContextDef(var Value: TWhileContextDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value := TWhileContextDef.Create;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TWhileContextDef.ftContextName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.ContextName := Pb.readString;
        end;
      TWhileContextDef.ftParallelIterations:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ParallelIterations := Pb.readInt32;
        end;
      TWhileContextDef.ftBackProp:
        begin
          Assert(wireType = TWire.VARINT);
          Value.BackProp := Pb.readBoolean;
        end;
      TWhileContextDef.ftSwapMemory:
        begin
          Assert(wireType = TWire.VARINT);
          Value.SwapMemory := Pb.readBoolean;
        end;
      TWhileContextDef.ftPivotName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.PivotName := Pb.readString;
        end;
      TWhileContextDef.ftPivotForPredName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.PivotForPredName := Pb.readString;
        end;
      TWhileContextDef.ftPivotForBodyName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.PivotForBodyName := Pb.readString;
        end;
      TWhileContextDef.ftLoopExitNamess:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.LoopExitNamess.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.LoopExitNamess.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TWhileContextDef.ftLoopEnterNamess:
        begin
            var vTipo : string;
            if IsPackedRepeatedField(tag, vTipo) then
            begin
              Pb.Push;
              try
                while not Pb.Eof do
                begin
                  var v : string := Pb.readString;
                  Value.LoopEnterNamess.Add(v);
                end
              finally
                Pb.Pop;
              end;
            end
            else begin
              repeat
                var v : string := Pb.readString;
                Value.LoopEnterNamess.Add(v);
              until not Pb.ConsumeTag(tag.v);
            end;
        end;
      TWhileContextDef.ftValuesDef:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TValuesDef := Value.ValuesDef;
            LoadValuesDef(v);
            Value.ValuesDef := v;
          finally
            Pb.Pop;
          end;
        end;
      TWhileContextDef.ftMaximumIterationsName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.MaximumIterationsName := Pb.readString;
        end;
      TWhileContextDef.ftNestedContextss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TControlFlowContextDef;
            LoadControlFlowContextDef(v);
            Value.NestedContextss.Add(v);
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

class procedure TSaveHelper.SaveValuesDef(const S: TpbSaver; const Value: TValuesDef);
var
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.Valuess.Count - 1 do
      h.Pb.writeRawString(Value.Valuess[i]);
    if Value.Valuess.Count > 0 then
      S.Pb.writeMessage(TValuesDef.ftValuess, h.Pb^);
  finally
    h.Free;
  end;
  if Value.ExternalValues <> nil then
  begin
    h.Init;
    try
      for var it in Value.ExternalValues do
      begin
          h.clear;
          h.SaveStringString(it);
          S.Pb.writeMessage(TValuesDef.ftExternalValues, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveControlFlowContextDef(const S: TpbSaver; const Value: TControlFlowContextDef);
begin
  case Value.ctxt.tag of
    TControlFlowContextDef.ftCondCtxt:
      begin
        if Value.Ctxt.value.AsType<TCondContextDef> <> nil then
          S.SaveObj<TCondContextDef>(Value.Ctxt.value.AsType<TCondContextDef>, SaveCondContextDef, TControlFlowContextDef.ftCondCtxt);
      end;
    TControlFlowContextDef.ftWhileCtxt:
      begin
        if Value.Ctxt.value.AsType<TWhileContextDef> <> nil then
          S.SaveObj<TWhileContextDef>(Value.Ctxt.value.AsType<TWhileContextDef>, SaveWhileContextDef, TControlFlowContextDef.ftWhileCtxt);
      end;
  end;
end;

class procedure TSaveHelper.SaveCondContextDef(const S: TpbSaver; const Value: TCondContextDef);
begin
  S.Pb.writeString(TCondContextDef.ftContextName, Value.ContextName);
  S.Pb.writeString(TCondContextDef.ftPredName, Value.PredName);
  S.Pb.writeString(TCondContextDef.ftPivotName, Value.PivotName);
  S.Pb.writeInt32(TCondContextDef.ftBranch, Value.Branch);
  if Value.ValuesDef <> nil then
    S.SaveObj<TValuesDef>(Value.ValuesDef, SaveValuesDef, TCondContextDef.ftValuesDef);
  if Value.NestedContextss.Count > 0 then
    S.SaveList<TControlFlowContextDef>(Value.NestedContextss, SaveControlFlowContextDef, TCondContextDef.ftNestedContextss);
end;

class procedure TSaveHelper.SaveWhileContextDef(const S: TpbSaver; const Value: TWhileContextDef);
var
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TWhileContextDef.ftContextName, Value.ContextName);
  S.Pb.writeInt32(TWhileContextDef.ftParallelIterations, Value.ParallelIterations);
  S.Pb.writeBoolean(TWhileContextDef.ftBackProp, Value.BackProp);
  S.Pb.writeBoolean(TWhileContextDef.ftSwapMemory, Value.SwapMemory);
  S.Pb.writeString(TWhileContextDef.ftPivotName, Value.PivotName);
  S.Pb.writeString(TWhileContextDef.ftPivotForPredName, Value.PivotForPredName);
  S.Pb.writeString(TWhileContextDef.ftPivotForBodyName, Value.PivotForBodyName);
  h.Init;
  try
    for i := 0 to Value.LoopExitNamess.Count - 1 do
      h.Pb.writeRawString(Value.LoopExitNamess[i]);
    if Value.LoopExitNamess.Count > 0 then
      S.Pb.writeMessage(TWhileContextDef.ftLoopExitNamess, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.LoopEnterNamess.Count - 1 do
      h.Pb.writeRawString(Value.LoopEnterNamess[i]);
    if Value.LoopEnterNamess.Count > 0 then
      S.Pb.writeMessage(TWhileContextDef.ftLoopEnterNamess, h.Pb^);
  finally
    h.Free;
  end;
  if Value.ValuesDef <> nil then
    S.SaveObj<TValuesDef>(Value.ValuesDef, SaveValuesDef, TWhileContextDef.ftValuesDef);
  S.Pb.writeString(TWhileContextDef.ftMaximumIterationsName, Value.MaximumIterationsName);
  if Value.NestedContextss.Count > 0 then
    S.SaveList<TControlFlowContextDef>(Value.NestedContextss, SaveControlFlowContextDef, TWhileContextDef.ftNestedContextss);
end;

{$EndRegion}

end.
