unit Tensorflow;

interface
  uses System.SysUtils, System.Rtti,  System.TypInfo,
       quick.Logger,
       System.Generics.Collections,
       Spring.Collections.Dictionaries,
       Spring.Collections.Extensions,
       Spring.Collections.Stacks,
       Spring,
       Quick.Logger.Provider.Files,
       TensorFlow.LowLevelAPI,
       TensorFlow.DApiBase,
       TensorFlow.DApi,
       TensorFlow.DApiOperations,
       TensorFlow.Eager,
       Tensorflow.Utils,

       ProtoGen.Tensor,
       Protogen.tensorShape,
       ProtoGen.attrValue,
       ProtoGen.types,
       ProtoGen.opDef,
       protogen.config;



const
  C_GRAPH_MODE : Integer = 0;
  C_EAGER_MODE : Integer = 1;

type
  TTensors = class (TEmptyEnumerable<TFTensor>)
  private
    Fitems : TList<TFTensor> ;
    Fdtype : TF_DataType;
    Fshape : TFShape;
    Frank  : Integer;
    Fgraph : TFGraph;
    FIsList: Boolean;
    FLength: Integer;
    FIsCreatedInGraphMode : Boolean;
    function  GetItem(index: Integer): TFTensor;
    procedure SetItem(index: Integer; const Value: TFTensor);
    function Getdtype: TF_DataType;
    function Getshape: TFShape;
    function GetRank: Integer;
    function GetGraph: TFGraph;
    function GetLen: Integer;

  public
     constructor Create(tensors: TArray<TFTensor>);
     procedure   Add(tensor: TFTensor);
     procedure   AddRange(tensors: TArray<TFTensor>);
     procedure   Insert(index: Integer; tensor: TFTensor);

     property IsCreatedInGraphMode: Boolean  read FIsCreatedInGraphMode write FIsCreatedInGraphMode;
     property IsList: Boolean      read FIsList write FIsList;
     property Length: Integer      read GetLen;
     property graph : TFGraph      read GetGraph;
     property rank  : Integer      read GetRank;
     property shape : TFShape      read Getshape;
     property dtype : TF_DataType  read Getdtype;
     property item[index: Integer]: TFTensor  read GetItem write SetItem; default;
  end;


  TEagerTensor = class(TFTensor)
    protected
       procedure NewEagerTensorHandle(h:Pointer);
    private
       m_Device : string;
       procedure Resolve;
    function GetDeviceName: string;
    public
       constructor Create(h:Pointer);overload;
       constructor Create(h: Pointer;NewEagerTensor: Boolean);overload;
       constructor Create(shape: TFShape;dType: TF_DataType);overload;

       constructor Create(bytes: TArray<TFString>;shape: TFShape);overload;

       constructor Create(bytes: TArray<Boolean>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Boolean>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Boolean>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Byte>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Byte>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Byte>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int16>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int16>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int16>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int32>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int32>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int32>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int64>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int64>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int64>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Single>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Single>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Single>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Single>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Double>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Double>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Double>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Double>>>>;shape: TFShape; dtype: TF_DataType);overload;

       class function GetRank(eTensor:TEagerTensor): Integer;
       class function GetDims(eTensor:TEagerTensor): TArray<Integer>;

       property Device : string read GetDeviceName;

  end;


  ContextSwitch = class
    private
       FEagerMode          : Boolean;
       FIsBuildingFunction : Boolean;
       FDeviceStack        : string;
    public
       constructor Create; overload;
       constructor Create(isEager, isFunc: Boolean); overload;
       function ToString: string;override;

       property EagerMode          : Boolean read FEagerMode;
       property IsBuildingFunction : Boolean read FIsBuildingFunction;
       property DeviceStack        : string  read FDeviceStack;
  end;

  ContextSwitchStack = class
    private
       FStack : TStack<ContextSwitch>;
    public
      constructor Create(isEager, isFunc: Boolean);
      destructor  Destroy; override;
      procedure   Push(isEager, isFunc: Boolean);
      procedure   Clear;
      procedure   Pop;
      function    Count: Integer;
      function    Current: ContextSwitch;
  end;

  TPhysicalDevice = record
      DeviceName : AnsiString;
      DeviceType : AnsiString;
      function ToString: string;
  end;

  TContextOptions = class(TFDisposable)
    protected
       procedure NativeDispose(hnd: Pointer); override;
    public
       constructor Create;
       /// <summary>
       /// Delete the instance.
       /// </summary>
       destructor  Destroy; override;
  end;

  ExecuteOpArgs = class
    private
      FGetGradientAttrs : TFunc<TFOperation,TObject>;
      FOpAttrs          : TDictionary<string,TValue>;
      FOpInputArgs      : TArray<TValue>;
    public
      constructor Create(inputArgs: TArray<TValue>);
      function    SetAttributes(attrs: TArray<TValue>): ExecuteOpArgs;

    property GetGradientAttrs : TFunc<TFOperation,TObject> read FGetGradientAttrs write FGetGradientAttrs;
    property OpAttrs          : TDictionary<string,TValue> read FOpAttrs          write FOpAttrs;
    property OpInputArgs      : TArray<TValue>             read FOpInputArgs      write FOpInputArgs ;
  end;

  /// <summary>
  /// Environment in which eager operations execute.
  /// </summary>
  TContext = class(TFDisposable)
    private

       function  GetHandle: Pointer;  protected
       procedure NativeDispose(hnd: Pointer); override;
     public
        _device_policy        : TFE_ContextDevicePlacementPolicy;
        _log_device_placement : Boolean;
        _memory_growth_map    : TDictionary<TPhysicalDevice,Boolean>;
        defaultExecutionMode  : Integer;
        DeviceName            : string;
        ScopeName             : string;
        initialized           : Boolean;
        context_switches      : ContextSwitchStack;
        _seed                 : Nullable<Integer>;
        _rng                  : Nullable<Integer>;
        FConfig               : TConfigProto;
        /// <summary>
        /// Initializes a new instance of the <see cref="T:TensorFlow.TFE_NewContext"/> class.
        /// </summary>
        constructor Create;
        /// <summary>
        /// Delete the instance.
        /// </summary>
        destructor  Destroy; override;
        /// <summary>
        /// Initialize handle and devices if not already done so.
        /// </summary>
        procedure ensure_initialized;
        function  internal_operation_seed: Nullable<Integer>;
        procedure set_global_seed(seed:Nullable<Integer>);
        function  global_seed: Nullable<Integer>;
        procedure start_step;
        procedure end_step;
        function  executing_eagerly: Boolean;
        function  is_build_function: Boolean;
        function  shared_name(name: AnsiString = ''): AnsiString;
        procedure graph_mode(isFunc: Boolean = False);
        procedure eager_mode(isFunc: Boolean = False);
        function  has_graph_arg(args: TArray<TValue>): Boolean;
        function  switched_to_graph(args: TArray<TValue>): Boolean;
        procedure restore_mode;
        procedure reset_context;
        procedure log_device_placement(enable: Boolean);
        function  get_memory_growth(device_type: AnsiString):Boolean;
        procedure Set_memory_growth(device: TPhysicalDevice; enable: Boolean);
        function  list_physical_devices(device_type: AnsiString = ''): TArray<TPhysicalDevice>;
        function  MergeConfig: TConfigProto;
        /// <summary>
        /// Environment in which eager operations execute.
        /// </summary>
        function ExecGraphAction(OpType: string; Name: string; args: ExecuteOpArgs): TFTensors;
        function ExecEagerAction(OpType: string; Name: string; args: ExecuteOpArgs): TFTensors;
        function ExecuteOp(OpType: string; Name: string; args: ExecuteOpArgs): TFTensors;

        property Handle_ : Pointer      read GetHandle;
        property Config  : TConfigProto read FConfig write FConfig;
  end;

  OpDefLibrary = class
    private
      class function SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
      class function _IsListParameter(arg: TArgDef): Boolean;
      class function _IsListValue(v: TValue): Boolean;
      class procedure SetAttrs(op_type_name : string;
                               input_arg    : TArgDef;
                               op_def       : TOpDef;
                               attrs        : TDictionary<string, TValue>;
                               inferred_from: TDictionary<string, TValue>;
                               types        : TList<TF_DataType>;
                               base_types   : TList<TF_DataType>;
                               input_types  : TList<TF_DataType>;
                               values       : TValue);
    public
      class function _apply_op_helper(op_type_name: string;name: string = ''; args : TArray<TParameter> = nil): TFOperation; overload;
      class function _apply_op_helperDict(op_type_name: string; name: string = ''; keywords: TDictionary<string, TValue> = nil): TFOperation;overload;

  end;


  TTensorflow = class(TFDisposable)
    private
      function GetVersion: string;
    protected
		  procedure NativeDispose(hnd: Pointer); override;

    public
      byte8_t   : TF_DataType;
      int8_t    : TF_DataType;
      int16_t   : TF_DataType;
      int32_t   : TF_DataType;
      int64_t   : TF_DataType;
      float16_t : TF_DataType;
      float32_  : TF_DataType;
      float64_t : TF_DataType;
      bool_t    : TF_DataType;
      chars_t   : TF_DataType;
      string_t  : TF_DataType;

      Status  : TFStatus;
      Context : TContext;
      OpDefLib: OpDefLibrary;

      constructor Create;
      procedure   enable_eager_execution;
      function    executing_eagerly:Boolean;
      function    get_default_graph: TFgraph;
      procedure   reset_default_graph;
      function    peak_default_graph: TFgraph;
      /// <summary>
      ///
      /// </summary>
      /// <param name="value"></param>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function constant(value: TValue; dtype : TF_DataType = TF_DATATYPE_UNKNOWN; shape : TFShape= nil; name : AnsiString = 'Const'): TFTensor;
      /// <summary>
      ///     Creates a new graph.
      /// </summary>
      ///<remarks>Has no interaction with graph defaulting. Equivalent to new Graph();</remarks>
      function Graph: TFGraph;

      property Version : string read GetVersion;
  end;

  TConstant_op = class
    private
      class function convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType=TF_DATATYPE_UNKNOWN): TFTensor; overload;
    public
      class function convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;
      class function convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;overload;
      /// <summary>
      /// Creates a constant tensor.
      ///
      /// The resulting tensor is populated with values of type `dtype`, as
      /// specified by arguments `value` and (optionally) `shape`
      /// </summary>
      /// <param name="value">A constant value (or list) of output type `dtype`.</param>
      /// <param name="dtype">The type of the elements of the resulting tensor.</param>
      /// <param name="shape">Optional dimensions of resulting tensor.</param>
      /// <param name="name">Optional name for the tensor.</param>
      /// <returns></returns>
      class function constant(value: PValue; dtype : TF_DataType= TF_DATATYPE_UNKNOWN;
                              shape : TFShape = nil; verify_shape : Boolean = false;
                              allow_broadcast : Boolean = true; name : AnsiString = 'Const'): TFTensor;

  end;

  var
   tf : TTensorflow;

implementation
   uses Oz.Pb.Classes, Oz.SGL.Collections,oz.Pb.StrBuffer, pbPublic, pbInput, pbOutput,
        NDArray,
        TensorFlow.Ops;

// Utils
function As_Proto(tshape:TFShape): TTensorShapeProto;
begin
    var shape : TTensorShapeProto;
    shape.Init;

    for var i := 0 to tshape.ndim - 1 do
    begin
        var dim : TDim;
        dim.Init;
        dim.Size := tshape.dims[i];
        //dim.Name = $"dim_{i}";
        shape.Dims.Add(@dim);
    end;
    Result := shape;
end;

//

{ TTensorflow }


function TTensorflow.constant(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString): TFTensor;
begin
    TConstant_op.constant(@value,
                          dtype,
                          shape,
                          False,
                          True,
                          name);
end;

constructor TTensorflow.Create;
begin
    byte8_t   := TF_DataType.TF_UINT8;
    int8_t    := TF_DataType.TF_INT8;
    int16_t   := TF_DataType.TF_INT16;
    int32_t   := TF_DataType.TF_INT32;
    int64_t   := TF_DataType.TF_INT64;
    float16_t := TF_DataType.TF_HALF;
    float32_  := TF_DataType.TF_FLOAT;
    float64_t := TF_DataType.TF_DOUBLE;
    bool_t    := TF_DataType.TF_BOOL;
    chars_t   := TF_DataType.TF_STRING;
    string_t  := TF_DataType.TF_STRING;

    Context   := TContext.Create;
    Status    := TFStatus.Create;
    OpDefLib  := OpDefLibrary.Create;

    Logger.Providers.Add(GlobalLogFileProvider);
    with GlobalLogFileProvider do
    begin
      FileName := '.\Logs.log';
      LogLevel := LOG_ALL;
      TimePrecission := True;
      MaxRotateFiles := 3;
      MaxFileSizeInMB := 5;
      RotatedFilesPath := '.\RotatedLogs';
      CompressRotatedFiles := False;
      Enabled := True;
    end;
end;

procedure TTensorflow.enable_eager_execution;
begin
    Context.eager_mode;
end;

function TTensorflow.executing_eagerly: Boolean;
begin
    Result := Context.executing_eagerly;
end;

function TTensorflow.GetVersion: string;
begin
     Result := string(AnsiString(TF_Version));
end;

procedure TTensorflow.NativeDispose(hnd: Pointer);
begin
  inherited;

  Context.Free;
  Status.Free;
  OpDefLib.Free;

end;

function TTensorflow.get_default_graph: TFgraph;
begin
    Result := TOps.get_default_graph;
end;

function TTensorflow.Graph: TFGraph;
begin
    Result := TFGraph.Create;
end;

function TTensorflow.peak_default_graph: TFgraph;
begin
    Result := TOps.peak_default_graph;
end;

procedure TTensorflow.reset_default_graph;
begin
    TOps.reset_default_graph
end;

{ ContextSwitch }

constructor ContextSwitch.Create;
begin
     inherited create;
     FEagerMode          := False;
     FIsBuildingFunction := False;
end;

constructor ContextSwitch.Create(isEager, isFunc: Boolean);
begin
    FEagerMode          := isEager;
    FIsBuildingFunction := isFunc;
end;

function ContextSwitch.ToString: string;
begin
    Result := Format('EagerMode: %s, IsBuildingFunction: %s',[BoolToStr(FEagerMode,True),BoolToStr(FIsBuildingFunction,True)]);
end;

{ ContextSwitchStack }

constructor ContextSwitchStack.Create(isEager, isFunc: Boolean);
begin
    FStack := TStack<ContextSwitch>.Create;
    Push(isEager, isFunc)
end;

destructor ContextSwitchStack.Destroy;
begin
     FStack.Free;
     inherited Destroy;
end;

procedure ContextSwitchStack.Push(isEager, isFunc: Boolean);
begin
     FStack.Push(ContextSwitch.Create(isEager, isFunc))
end;

procedure ContextSwitchStack.Pop;
begin
     FStack.Pop;
end;

procedure ContextSwitchStack.Clear;
begin
    FStack.Clear;
end;

function ContextSwitchStack.Count: Integer;
begin
    Result := FStack.Count;
end;

function ContextSwitchStack.Current: ContextSwitch;
begin
    Result := FStack.Peek;
end;

{ TPhysicalDevice }

function TPhysicalDevice.ToString: string;
begin
     Result := Format('%s: %s',[DeviceType,DeviceName])
end;

{ TContext }

constructor TContext.Create;
begin
    _device_policy   := TFE_DEVICE_PLACEMENT_SILENT;
    context_switches := ContextSwitchStack.Create(defaultExecutionMode = C_EAGER_MODE, false);
    initialized      := false;
    FConfig.Init;
end;

destructor TContext.Destroy;
begin
   inherited Destroy;
end;

procedure TContext.ensure_initialized;
var
   Saver  : TpbSaver;
begin
    if initialized then  Exit;


    var opts   := TContextOptions.Create;
    var status := TFStatus.Create;

    FConfig := MergeConfig;

    Saver.Init;
    TpbSaver.SaveConfigProto(Saver,FConfig);
    var config_str := Saver.Pb.GetBytes;

    TFE_ContextOptionsSetConfig(opts.Handle ,@config_str[0],Length(config_str),status.Handle);
    status.RaiseEx;

    TFE_ContextOptionsSetDevicePlacementPolicy(opts.Handle, _device_policy);
    inherited Create( TFE_NewContext(opts.Handle, status.Handle) ) ;
    status.RaiseEx;

    initialized := True;
end;

function TContext.ExecGraphAction(OpType, Name: string; args: ExecuteOpArgs): TFTensors;
begin
    var keywords := TDictionary<string, TValue>.Create;
    if Length(args.OpInputArgs) > 0 then
    begin
        for var i := 0 to Length(args.OpInputArgs) - 1 do
            keywords.Add(Format('input_%d',[i]), args.OpInputArgs[i]);
    end;
    if args.OpAttrs <> nil then
    begin
        for var attr in args.OpAttrs do
            keywords[attr.Key] := attr.Value;
    end;
    var Operation := tf.OpDefLib._apply_op_helperDict(OpType, Name, keywords);

    Result :=  TFTensors.Create( tf.OpDefLib._apply_op_helperDict(OpType, Name, keywords).Outputs );
end;

function TContext.ExecEagerAction(OpType, Name: string; args: ExecuteOpArgs): TFTensors;
begin

end;

function TContext.ExecuteOp(OpType, Name: string; args: ExecuteOpArgs): TFTensors;
begin
    if tf.Context.has_graph_arg(args.OpInputArgs) then
    begin
        if executing_eagerly then
        begin
            graph_mode;
            var res := ExecGraphAction(opType, name, args);
            restore_mode;
            exit(res);
        end else
        begin
            var res := ExecGraphAction(opType, name, args);
            (*if (tf.Runner.MustRecordGradient())
            {
                var op = result[0].op;
                Dictionary<string, object> attrs;
                if (args.GetGradientAttrs == null)
                {
                    attrs = new Dictionary<string, object>();
                    attrs["T"] = op.get_attr<TF_DataType>("T");
                }
                else
                {
                    attrs = ConvertToDict(args.GetGradientAttrs(op));
                }
                var args1 = new object[attrs.Count() * 2];
                int i = 0;
                foreach (var arg in attrs)
                {
                    args1[i] = arg.Key;
                    args1[i + 1] = arg.Value;
                    i += 2;
                }
                tf.Runner.RecordGradient(opType, op.inputs, args1, op.outputs);
            }
            *)
            Exit(res);
        end;
    end else
    begin
        Result := ExecEagerAction(opType, name, args);
        Exit;
    end;
end;

procedure TContext.NativeDispose(hnd: Pointer);
begin
  if Assigned(hnd) then
    TFE_DeleteContext(hnd);
end;

procedure TContext.set_global_seed(seed: Nullable<Integer>);
begin
    _seed := seed;
    if seed.HasValue then
    begin
         RandSeed := seed;
         Randomize;
        _rng := Random(Integer.MaxValue);
    end else
    begin
        Randomize;
        _rng := Random(Integer.MaxValue);
    end;
    // Also clear the kernel cache, to reset any existing seeds
    if Handle <> nil then
        TFE_ContextClearCaches(handle);
end;

function TContext.global_seed: Nullable<Integer>;
begin
    Result := _seed;
end;

function TContext.shared_name(name: AnsiString): AnsiString;
begin
    if (not string.IsNullOrEmpty(string(name))) or (executing_eagerly) then
      Result := name
    else
      Result := 'cd2c89b7-88b7-44c8-ad83-06c2a9158347';
end;

procedure TContext.graph_mode(isFunc: Boolean);
begin
    context_switches.Push(False,isFunc)
end;

procedure TContext.eager_mode(isFunc: Boolean);
begin
    context_switches.Push(True,isFunc)
end;

function TContext.is_build_function: Boolean;
begin
    Result := context_switches.Current.IsBuildingFunction;
end;

procedure TContext.log_device_placement(enable: Boolean);
begin
    if Assigned(Handle)  then
       TFE_ContextSetLogDevicePlacement(Handle,Byte(enable),tf.Status.Handle);
    _log_device_placement := enable;
end;

function TContext.MergeConfig: TConfigProto;
begin
    FConfig.LogDevicePlacement := _log_device_placement;
    // var gpu_options = _compute_gpu_options();
    // Config.GpuOptions.AllowGrowth = gpu_options.AllowGrowth;
    Result := FConfig;
end;

function TContext.get_memory_growth(device_type: AnsiString): Boolean;
begin
    for var map in _memory_growth_map do
    begin
        if map.Key.DeviceType = device_type then
            Exit(map.Value);
    end;
    Result := False;
end;

procedure TContext.Set_memory_growth(device: TPhysicalDevice; enable: Boolean);
begin
    _memory_growth_map.AddOrSetValue(device,enable);
end;

function TContext.list_physical_devices(device_type: AnsiString): TArray<TPhysicalDevice>;
 var
   //opts    : TContextOptions ;
   //ctx     : TContext;
   devices : PTF_DeviceList;
   i       : Integer;
begin
    //opts := TContextOptions.Create;
    //ctx  := TContext.Create();
    devices := TFE_ContextListDevices(Handle, tf.Status.Handle);
    tf.Status.RaiseEx;

    var num_devices := TF_DeviceListCount(devices);
    result := [];
    for i := 0 to  num_devices - 1 do
    begin
        var dev_type := AnsiString(TF_DeviceListType(devices, i, tf.Status.Handle));
        tf.Status.RaiseEx;

        if string(dev_type).StartsWith('XLA') then
          Continue;

        if (device_type = '') or (dev_type = device_type) then
        begin
            var dev_name := AnsiString(TF_DeviceListName(devices, i, tf.Status.Handle));
            tf.Status.RaiseEx;

            var item : TPhysicalDevice;
            item.DeviceName := dev_name;
            item.DeviceType := dev_type;

            Result := Result + [ item ];
        end;

    end;
end;

function TContext.internal_operation_seed: Nullable<Integer>;
begin
    Result := Random(Integer.MaxValue);
end;

procedure TContext.start_step;
begin
    TFE_ContextStartStep(Handle);
end;

procedure TContext.reset_context;
begin
    Tops.reset_uid();
    // tf.defaultSession = null;
    Tops.reset_default_graph;
    context_switches.Clear;
    tf.Context.ensure_initialized;
    if Handle <> nil then
        TFE_ContextClearCaches(Handle);
end;

procedure TContext.restore_mode;
begin
     context_switches.Pop;
     tf.get_default_graph;
end;

function TContext.switched_to_graph(args: TArray<TValue>): Boolean;
var
  switching_to_graph: Boolean;
begin
    switching_to_graph := has_graph_arg(args) and tf.Context.executing_eagerly;
    if switching_to_graph then
        tf.Context.graph_mode(tf.Context.is_build_function);
    Result := switching_to_graph;
end;

function TContext.has_graph_arg(args: TArray<TValue>): Boolean;
var
  el : TValue;

begin
    var flatten_args := TUtils.flatten<TValue>(args);

    (*if (flatten_args.Count(x => x.GetType().IsValueType) == flatten_args.Count())
        return tf.Context.executing_eagerly() == false *)
    { DONE -oMax -c :  completare poi 11/12/2021 10:02:13 }
    var  has_graph_arg := not tf.Context.executing_eagerly;
    for var i := 0 to flatten_args.Count -1 do
    begin
        el := flatten_args[i];
        if string.LowerCase(el.TypeInfo.Name) = 'tndarray' then
            continue
        else if string.LowerCase(el.TypeInfo.Name) = 'teagertensor' then
            continue
        else if string.LowerCase(el.TypeInfo.Name) = 'tftensor' then
        begin
            has_graph_arg := true;
            break;
        end;
    end;
    Result := has_graph_arg;

end;

procedure TContext.end_step;
begin
    TFE_ContextEndStep(Handle);
end;

function TContext.GetHandle: Pointer;
begin
    if Handle = nil then
        ensure_initialized ;

    Result := Handle;
end;

function TContext.executing_eagerly: Boolean;
begin
    if context_switches.Count = 0 then
        tf.enable_eager_execution();
    Result := context_switches.Current.EagerMode;
end;

{ TContextOptions }

constructor TContextOptions.Create;

begin
    inherited Create( TFE_NewContextOptions ) ;
end;

destructor TContextOptions.Destroy;

begin
   inherited Destroy;
end;

procedure TContextOptions.NativeDispose(hnd: Pointer);

begin
  if Assigned(hnd) then
    TFE_DeleteContextOptions(hnd);
end;

{ TEagerTensor }

constructor TEagerTensor.Create(h: Pointer);
begin
    EagerTensorHandle := h;
    Resolve;

    self.MMLck();
end;

constructor TEagerTensor.Create(h: Pointer;NewEagerTensor: Boolean);
begin
     NewEagerTensorHandle(h);
end;

constructor TEagerTensor.Create(shape: TFShape;dType: TF_DataType);
begin
    inherited Create(shape,dType);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Boolean>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Boolean>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Byte>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int16>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int16>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int32>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int32>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int64>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int64>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Single>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Single>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Single>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Double>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Double>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Double>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

function TEagerTensor.GetDeviceName: string;
begin
    m_Device :=  string(AnsiString( TFE_TensorHandleDeviceName(EagerTensorHandle, tf.Status.Handle)));
    Result := m_Device;
end;

class function TEagerTensor.GetDims(eTensor: TEagerTensor): TArray<Integer>;
var
  dims : TArray<Integer>;

begin
    var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
    SetLength(dims, TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle));
    for var i := 0 to Length(dims)-1 do
        dims[i] := TFE_TensorHandleDim(tfe_tensor_handle, i, tf.Status.Handle);
    Result := dims;
end;

class function TEagerTensor.GetRank(eTensor: TEagerTensor): Integer;
begin
     var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
     Result := TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TFString>; shape: TFShape);
begin
    inherited Create( StringTensor(bytes,shape).Handle );
    NewEagerTensorHandle(Handle);
end;

procedure TEagerTensor.NewEagerTensorHandle(h: Pointer);
begin
    EagerTensorHandle := TFE_NewTensorHandle(h,tf.Status.Handle) ;
    tf.Status.RaiseEx;
end;

procedure TEagerTensor.Resolve;
begin
    Handle := TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

{ ExecuteOpArgs }

constructor ExecuteOpArgs.Create(inputArgs: TArray<TValue>);
begin
    FOpInputArgs := inputArgs;
end;

function ExecuteOpArgs.SetAttributes(attrs: TArray<TValue>): ExecuteOpArgs;
begin
     var att : TArray<TParameter> ;
     for var i := 0 to Length(attrs)-1 do
     begin
         SetLength(att,Length(att)+1);
         att[i].vValue := attrs[i];
         att[i].sNome  := attrs[i].TypeInfo.Name;
     end;
     FOpAttrs := TUtils.ConvertToDict(att);
     Result := Self;
end;

{ OpDefLibrary }

class function OpDefLibrary._IsListParameter(arg: TArgDef): Boolean;
begin
     if not string.IsNullOrEmpty(arg.NumberAttr)  then
       Exit(True)
     else if not string.IsNullOrEmpty(arg.TypeListAttr) then
       Exit(True)
     else
       Result := False;
end;

class function OpDefLibrary._IsListValue(v: TValue): Boolean;
begin
    Result := v.IsArray;
end;

class procedure OpDefLibrary.SetAttrs(op_type_name : string;input_arg: TArgDef; op_def: TOpDef; attrs, inferred_from: TDictionary<string, TValue>; types,
                                      base_types, input_types: TList<TF_DataType>; values: TValue);
begin
    var input_name := input_arg.Name;

    if  not string.IsNullOrEmpty(input_arg.NumberAttr) then
    begin
        if attrs.ContainsKey(input_arg.NumberAttr) then
        begin

        end else
        begin
            if(values.IsArray) and (values.GetArrayElement(0).TypeInfo = TypeInfo(TFTensor)) then
            begin
                var num_attr : TAttrDef;
                for var i := 0 to op_def.Attrs.Count -1 do
                begin
                    if op_def.Attrs.Items[i]^.Name = input_arg.NumberAttr then
                    begin
                        num_attr := op_def.Attrs.Items[i]^;
                        Break;
                    end;
                end;
                if (num_attr.HasMinimum) and (values.GetArrayLength < num_attr.Minimum) then
                    raise Exception.Create(Format('"%s" to "%s" Op with length %d shorter than minimum length %d',[input_name,op_type_name,values.GetArrayLength,num_attr.Minimum]));

                attrs.AddOrSetValue(input_arg.NumberAttr,TObject(values.GetArrayLength));
                inferred_from.AddOrSetValue(input_arg.NumberAttr, TObject(input_name));
            end;
        end;
        // All tensors must have the same base type.
        if input_arg.&Type <> TDataType.DT_INVALID then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr, TObject(base_types[0]));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TObject(input_name));
            //var type_attr = op_def.Attr.First(x => x.Name == input_arg.TypeAttr);
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeAttr) then
    begin
        var attr_value := base_types[0];
        if attrs.ContainsKey(input_arg.TypeAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr,TObject(attr_value));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TObject(input_name));
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
    begin
        var attr_value := base_types;
        if attrs.ContainsKey(input_arg.TypeListAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeListAttr, attr_value);
            inferred_from.AddOrSetValue(input_arg.TypeListAttr,TObject(input_name));
        end;
    end;
    if input_arg.IsRef then
        input_types.AddRange(types)
    else
        input_types.AddRange(base_types);
end;

class function OpDefLibrary._apply_op_helper(op_type_name: string; name: string; args: TArray<TParameter>): TFOperation;
begin
    if args = nil then
       Result := _apply_op_helperDict(op_type_name, name)
    else
      Result := _apply_op_helperDict(op_type_name, name, TUtils.ConvertToDict(args));
end;

class function OpDefLibrary._apply_op_helperDict(op_type_name: string; name: string; keywords: TDictionary<string, TValue>): TFOperation;
var
  aObj        : TArray<TValue>;
  g           : TFGraph;
  attrs       : TDictionary<string, TValue>;
  attr_protos : System.Generics.Collections.TDictionary<string, TAttrValue>;
  inputs      : TList<TFTensor>;
  input_types : TList<TF_DataType>;
  values      : TValue;

  op_def      : TOpDef;
begin
    if keywords = nil then  aObj := aObj + [nil]
    else                    aObj := keywords.Values.ToArray;

    g := TOps._get_graph_from_inputs(aObj);
    op_def := g.GetOpDef(op_type_name);
    // Default name if not specified.
    if String.IsNullOrEmpty(name) then
        name := op_type_name;
    (*// Check for deprecation
    if (op_def.Deprecation != null && op_def.Deprecation.Version > 0)
    {
    }*)
    var default_type_attr_map := TDictionary<string, TValue>.Create;
    for var attr_def in op_def.Attrs do
    begin
        if attr_def.&Type <> 'type' then continue;
        var key := attr_def.Name;
        if attr_def.DefaultValue.Value.tag =  attr_def.DefaultValue.ftType then
        begin
            default_type_attr_map.AddOrSetValue(key, attr_def.DefaultValue.Value.value);
        end;
    end;
    attrs       := TDictionary<string, TValue>.Create;
    inputs      := TList<TFTensor>.Create;
    input_types := TList<TF_DataType>.Create;
    values      := nil;
    g.as_default;

    var ret_op  := TUtils.tf_with<TFScope,TFOperation>(g.WithScope(name),
      function(arg1: TFScope):TFOperation
      begin
          var inferred_from := TDictionary<string, TValue>.Create;
          var base_types    := TList<TF_DataType>.Create;
          var types         := TList<TF_DataType>.CReate;
          var _scope_name   := g.WithScope(name);
          // Perform input type inference
          for var i := 0 to op_def.InputArgs.Count - 1 do
          begin
              var input_arg : TArgDef := op_def.InputArgs[i]^;
              var input_name:= input_arg.Name;
              if keywords.ContainsKey(input_name) then
                  values := keywords[input_name]
              else if keywords.ContainsKey(input_name + '_') then
              begin
                  input_name := input_name + '_';
                  values     := keywords[input_name];
              end
              else if keywords.ContainsKey('input_'+ IntTostr(i)) then
              begin
                  values := keywords['input_'+ IntTostr(i)];
              end
              else
                  raise Exception.Create('No argument for input ' + input_name);
              // Goals:
              // * Convert values to Tensors if it contains constants.
              // * Verify that values is a list if that matches the input_arg's
              // type.
              // * If the input_arg's type is determined by attrs, either set
              // those attrs and validate those attr values are legal (if
              // they have not yet been set) or validate the input matches
              // the type indicated by the attrs (if they have already been
              // inferred via an earlier input).
              // * If the input_arg has an explicit type, make sure the input
              // conforms.
              var dtype        : TF_DataType := TF_DataType.TF_DATATYPE_UNKNOWN;
              var default_dtype: TF_DataType := TF_DataType.TF_DATATYPE_UNKNOWN;
              if _IsListParameter(input_arg) then
              begin
                  if not _IsListValue(values) then
                      raise Exception.Create('Expected list for {input_name} argument to {op_type_name} Op, not {values}.');

                  if input_arg.&Type <> TDataType.DT_INVALID then
                      dtype := TF_DataType(input_arg.&Type)
                  else if not String.IsNullOrEmpty(input_arg.NumberAttr) then
                  begin
                      if attrs.ContainsKey(input_arg.TypeAttr) then
                          dtype := TF_DataType( attrs[input_arg.TypeAttr].AsInteger )
                      else begin
                         var aEle := values.GetArrayElement(0);
                         if string.LowerCase(aEle.TypeInfo.Name) = 'tftensor' then
                             dtype := (values.GetArrayElement(0).asType<TFTensor>).TensorDataType
                         else if aEle.IsObject then
                         begin
                             for var t := 0 to values.GetArrayLength - 1 do
                             begin
                                var item := values.GetArrayElement(t);
                                if item.AsType<TObject> is TFTensor then
                                begin
                                    dtype := (item.AsType<TObject> as TFTensor).TensorDataType;
                                end;
                             end;
                         end else
                             raise Exception.Create('can''t infer the dtype for {values.GetType()}');
                      end;
                      if (dtype = TF_DataType.TF_DATATYPE_UNKNOWN) and (default_type_attr_map.ContainsKey(input_arg.TypeAttr)) then
                          default_dtype := TF_DataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);
                  end;

                  if ( not input_arg.IsRef) and (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) then
                      dtype := TUtils.as_base_dtype(dtype);

                  var RetVal := TOps.internal_convert_n_to_tensor(values.AsType< TArray<TObject> >,dtype,input_arg.Name,default_dtype,input_arg.IsRef);
                  values := TValue.From< TArray<TFTensor> >(RetVal);
              end else
              begin
                  if input_arg.&Type <> TDataType.DT_INVALID then
                      dtype := TF_DataType(input_arg.&Type)
                  else if attrs.ContainsKey(input_arg.TypeAttr) then
                      dtype := TF_DataType(attrs[input_arg.TypeAttr].AsInteger)
                  else if (TUtils.isinstance(values, TypeInfo(string))) and (dtype = TF_DataType.TF_DATATYPE_UNKNOWN) then
                      dtype := TF_DataType.TF_STRING
                  else if default_type_attr_map.ContainsKey(input_arg.TypeAttr) then
                      default_dtype := TF_DataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);

                  var value := TOps.convert_to_tensor(values.AsType<TObject>,dtype,input_arg.Name,input_arg.IsRef,default_dtype);

                  values := TValue.From< TArray<TFTensor> >([ value ] );
              end;

              if (values.IsArray) and ( values.GetArrayElement(0).TypeInfo = TypeInfo(TFTensor) ) then
              begin
                  var values2 : TArray<TFTensor> := values.AsType< TArray<TFTensor> >;
                  inputs.AddRange(values2);
                  for var j := 0 to Length(values2) -1 do
                  begin
                      types.Add(values2[j].TensorDataType);
                      base_types.Add( TUtils.as_base_dtype(values2[j].TensorDataType) ) ;
                  end;
              end
              else
                 raise Exception.Create('NotImplementedException("_IsListParameter")');

              SetAttrs(op_type_name, input_arg, op_def, attrs, inferred_from, types, base_types, input_types, values);
          end;

          // Process remaining attrs
          for var attr in op_def.Attrs do
          begin
              if keywords.ContainsKey(attr.Name) then
              begin
                  attrs.AddOrSetValue(attr.Name, keywords[attr.Name] );
              end
          end;
          // Convert attr values to AttrValue protos.
          attr_protos := System.Generics.Collections.TDictionary<string, TAttrValue>.Create;
          for var  attr_def in op_def.Attrs do
          begin
              var key := attr_def.Name;
              if attrs.ContainsKey(key) then
              begin
                  attr_protos.AddOrSetValue(key, SetAttrValue(op_def, attr_def^, attrs[key] ) );
              end else
              begin
                  if attr_def.DefaultValue.Value.value.AsObject = nil then
                  begin
                      raise Exception.Create('Missing required positional argument ' + key);
                  end;
              end;
          end;
          attrs.Clear();

          // Determine output types (possibly using attrs)
          var output_types := TList<TF_DataType>.Create;
          for var arg in op_def.OutputArgs do
          begin
              types := TList<TF_DataType>.Create;
              if not string.IsNullOrEmpty(arg.NumberAttr) then
              begin
              end
              else if not string.IsNullOrEmpty(arg.TypeAttr) then
              begin
                  types := TList<TF_DataType>.Create;
                  types.Add( TF_DataType(attr_protos[arg.TypeAttr].Value.value.AsInteger) );
              end;
              if arg.IsRef then
              begin
                  var aTemp : TArray<TF_DataType> := [];
                  for var i := 0 to types.Count - 1 do
                  begin
                      aTemp := aTemp + [ TUtils.as_ref(types[i]) ];
                  end;
                  types.Clear;
                  types.Free;
                  types := TList<TF_DataType>.Create(aTemp);
              end;
              output_types.AddRange(types);
          end;

          // We add an explicit colocation constraint between
          // the newly created op and any of its reference-typed inputs.
         (* var must_colocate_inputs = zip(op_def.InputArg, inputs)
              .Where(x => x.Item1.IsRef)
              .Select(x => x.Item2)
              .ToArray();
          _MaybeColocateWith(must_colocate_inputs);
          *)

          // Add Op to graph
          g  := TFGraph.Create;
          var op := g.create_op(
              op_type_name,
              inputs.ToArray,
              output_types.ToArray,
              input_types.ToArray,
              name{_scope_name},
              attr_protos,
              @op_def);

          Exit(op);

      end);
      g.gExit;
      Result := ret_op;
end;

class function OpDefLibrary.SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
var
   v          : TpbOneof;
   attr_value : TAttrValue;

begin
    attr_value.Init;

    if attr_def.&Type.StartsWith('list(') then
    begin
        if attr_def.HasMinimum then
        begin
            v.tag := TAttrValue.ftList;
            var v1 : TListValue; v1.Init;
            v.value := TValue.From<TListValue>(v1);

            attr_value.Value := v;
        end;
    end;

    if attr_def.&Type = 'string' then
    begin
         v.tag   := TAttrValue.ftS;
         var b   := TEncoding.UTF8.GetBytes( value.AsString );
         v.value := TValue.From< TBytes >(b);

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'type' then
    begin
        v.tag   := TAttrValue.ftType;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(type)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TsgRecordList<Integer> >;
        for var i := 0 to l.Count - 1 do
        begin
            var d : TDataType := TDataType(l.Items[i]^);
            v1.Types.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(int)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TsgRecordList<Int64> >;
        for var i := 0 to l.Count - 1 do
        begin
            var d : Int64 := l.Items[i]^;
            v1.Types.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'bool' then
    begin
        v.tag   := TAttrValue.ftB;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'float' then
    begin
        v.tag   := TAttrValue.ftF;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'int' then
    begin
        v.tag   := TAttrValue.ftI;
        v.value := value;

        attr_value.Value := v;
        if (attr_def.HasMinimum)and ( v.value.AsInt64 < attr_def.Minimum) then
           raise Exception.Create(Format('Attr %s of %s Op passed %d less than minimum $d.',[attr_def.Name,op_def.Name,v.value.AsInt64,attr_def.Minimum]));
    end
    else if attr_def.&Type = 'shape' then
    begin
         if (value.IsEmpty) and ( not attr_def.DefaultValue.Value.value.IsEmpty) then
             attr_value.Value := attr_def.DefaultValue.Value;

         if (value.IsEmpty=False) and (value.IsClass) then
         begin
             if string(value.TypeInfo.Name).Contains('Shape') then
             begin
                 var v1 := As_Proto(value.AsType<TFShape>) ;
                 v.tag  := TTensorShapeProto.ftDims;
                 v.value:= TValue.From<TTensorShapeProto>(v1) ;
                 attr_value.Value := v;
             end;

         end;

    end
    else if attr_def.&Type = 'list(shape)' then
    begin

    end;

    Result := attr_value;
end;

{ TTensors }

procedure TTensors.Add(tensor: TFTensor);
begin
    Fitems.Add(tensor)
end;

procedure TTensors.AddRange(tensors: TArray<TFTensor>);
begin
    Fitems.AddRange(tensors)
end;

constructor TTensors.Create(tensors: TArray<TFTensor>);
begin
    Fitems.AddRange(tensors);
end;

function TTensors.Getdtype: TF_DataType;
begin
   Result := Fitems.First.dtype
end;

function TTensors.GetGraph: TFGraph;
begin
    Result := Fitems.First.graph;
end;

function TTensors.GetItem(index: Integer): TFTensor;
begin
    Result := Fitems[index]
end;

function TTensors.GetLen: Integer;
begin
    Result := Fitems.Count;
end;

function TTensors.GetRank: Integer;
begin
    Result := Fitems.First.rank;
end;

function TTensors.Getshape: TFShape;
begin
    Result := Fitems.First.Shape;
end;

procedure TTensors.Insert(index: Integer; tensor: TFTensor);
begin
    Fitems.Insert(index,tensor)
end;

procedure TTensors.SetItem(index: Integer; const Value: TFTensor);
begin
    Fitems[index] :=  Value;
end;

{ TConstant_op }

class function TConstant_op.constant(value: PValue; dtype: TF_DataType; shape: TFShape; verify_shape, allow_broadcast: Boolean;
  name: AnsiString): TFTensor;
begin
    if value = nil then
        Exit(nil);
    if tf.executing_eagerly then
        Result := convert_to_eager_tensor(value^, dtype, shape, name, verify_shape, allow_broadcast)
    else
        Result := convert_to_graph_tensor(value^, dtype, shape, name, verify_shape, allow_broadcast);
end;

class function TConstant_op.convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape,
  allow_broadcast: Boolean): TFTensor;
var
  v : TpbOneof;
begin
    var g : TFGraph := TOps.get_default_graph;

    var tp := TUtils.make_tensor_proto(value, dtype, shape, verify_shape, allow_broadcast);

    var tensor_value : TAttrValue;
    tensor_value.Init;
    v.tag   := TAttrValue.ftTensor;
    v.value := TValue.From<TTensorProto>(tp);
    tensor_value.Value := v;

    var dtype_value : TAttrValue;
    dtype_value.Init;
    v.tag   := TAttrValue.ftType;
    v.value := TValue.From<Integer>( Ord(dtype)  );
    dtype_value.Value := v;

    var attrs := System.Generics.Collections.TDictionary<string, TAttrValue>.Create;

    attrs.Add('value',tensor_value);
    attrs.Add('dtype',dtype_value);

    var op := g.create_op(
        'Const',
        [],
        [TF_DataType(Ord(dtype_value.Value.value.AsType<Integer>))],
        [],
        name,
        attrs);

    Result := op.outputs[0];
end;

class function TConstant_op.convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape, allow_broadcast: Boolean): TFTensor;
begin
    var t := convert_to_eager_tensor(value, tf.Context, dtype);
end;

class function TConstant_op.convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType): TFTensor;
begin
    ctx.ensure_initialized;
    var tipo : PTypeInfo;
    tipo:= value.TypeInfo;
    // convert data type
    if (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) and
       (string.LowerCase(tipo.Name) <> 'tndarray') and
       (value.IsArray = False) and
       (dtype <> TUtils.GetDataType(value))  then
    begin
        case dtype of
            TF_DataType.TF_DOUBLE: value := value.AsType<Double>;
            TF_DataType.TF_FLOAT:  value := value.AsType<Single>;
            TF_DataType.TF_INT64:  value := value.AsType<Int64>;
            TF_DataType.TF_INT32:  value := value.AsType<Int32>;

        end;
    end
    else if (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) and (value.IsType<TNDArray>) and  ( value.AsType<TNDArray>.Dtype = dtype ) then
    begin
        value = math_ops.cast(nd, dtype);
    end;

    // non ascii char
    if (dtype == TF_DataType.TF_STRING && value is byte[] bytes)
        return new EagerTensor(bytes, Shape.Scalar, TF_DataType.TF_STRING);

    switch (value)
    {
        case EagerTensor val:
            return val;
        case NDArray val:
            return val;
        case Shape val:
            return new EagerTensor(val.dims, new Shape(val.ndim));
        case Axis val:
            return new EagerTensor(val.axis, val.IsScalar ? Shape.Scalar : new Shape(val.size));
        case string val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case string[] val:
            return new EagerTensor(val, new Shape(val.Length));
        case bool val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case byte val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case int val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case long val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case ulong val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case float val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case double val:
            return new EagerTensor(new[] { val }, Shape.Scalar);
        case IEnumerable<Tensor> val:
            return ops.convert_to_tensor(val);
        case Array val:
            return new EagerTensor(val, val.GetShape());
        default:
            throw new NotImplementedException($"convert_to_eager_tensor {value.GetType()}");
    }
end;

initialization
begin
    tf := TTensorflow.Create;
end;

end.


