unit TensorFlow.Context;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
       uses System.SysUtils, System.Rtti,  System.TypInfo,
       System.Generics.Collections,
       Spring.Collections.Dictionaries,
       Spring.Collections.Extensions,
       Spring.Collections.Stacks,
       Spring,
       Quick.Logger.Provider.Files,

       TF4D.Core.CApi,
       TensorFlow.DApiBase,
       TensorFlow.DApi,
       TensorFlow.DApiEager,

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
    TPhysicalDevice = record
      DeviceName : AnsiString;
      DeviceType : AnsiString;
      function ToString: string;
  end;

    ExecuteOpArgs = class
    private
      FGetGradientAttrs : TFunc< TFOperation,TArray<TParameter> >;
      FOpAttrs          : TDictionary<string,TValue>;
      FOpInputArgs      : TArray<TValue>;
    public
      constructor Create(inputArgs: TArray<TValue>);
      function    SetAttributes(attrs: TArray<TValue>): ExecuteOpArgs;

    property GetGradientAttrs : TFunc< TFOperation,TArray<TParameter> >  read FGetGradientAttrs write FGetGradientAttrs;
    property OpAttrs          : TDictionary<string,TValue> read FOpAttrs          write FOpAttrs;
    property OpInputArgs      : TArray<TValue>             read FOpInputArgs      write FOpInputArgs ;
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

implementation
          uses Tensorflow,
               TensorFlow.Ops,
               TensorFlow.EagareRunner,
               Tensorflow.Utils,

               Oz.Pb.Classes,
               Oz.SGL.Collections,
               Oz.Pb.StrBuffer,
               pbPublic,
               pbInput,
               pbOutput;

{ ExecuteOpArgs }

constructor ExecuteOpArgs.Create(inputArgs: TArray<TValue>);
begin
    FOpInputArgs := inputArgs;
end;

function ExecuteOpArgs.SetAttributes(attrs: TArray<TValue>): ExecuteOpArgs;
begin
     var att : TArray<TParameter> ;
     var j : Integer := 0;
     for var i := 0 to (Length(attrs) div 2)-1 do
     begin
         SetLength(att,Length(att)+1);
         att[i].sNome  := attrs[j].AsString;
         att[i].vValue := attrs[j+1];
         Inc(j,2)
     end;
     FOpAttrs := TUtils.ConvertToDict(att);
     Result := Self;
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
    _device_policy       := TFE_DEVICE_PLACEMENT_SILENT;
    defaultExecutionMode := C_GRAPH_MODE;
    context_switches := ContextSwitchStack.Create(defaultExecutionMode = C_EAGER_MODE, false);
    initialized      := false;
    FConfig.Init;
end;

destructor TContext.Destroy;
begin
   inherited Destroy;
   context_switches.Free;
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

    var res := tf.OpDefLib._apply_op_helperDict(OpType, Name, keywords).Outputs;
    Result :=  TFTensors.Create( res );
end;

function TContext.ExecEagerAction(OpType, Name: string; args: ExecuteOpArgs): TFTensors;
begin
    var opExecInfo := TFastPathOpExecInfo.Create(OpType, Name, args.OpInputArgs);
    opExecInfo.attrs := args.OpAttrs;

    var ts := tf.Runner.TFE_FastPathExecute(opExecInfo);
    Result := TFTensors.Create(ts);
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
            if tf.Runner.MustRecordGradient then
            begin
                var op : TFOperation := res[0].op;
                var attrs :=  TDictionary<string, TValue>.Create;
                if args.GetGradientAttrs = nil then
                begin
                    attrs.Add('T', op.DType );
                end else
                begin
                    attrs := TUtils.ConvertToDict( args.GetGradientAttrs(op) );
                end;
                var args1 : TArray<TValue>;
                SetLength(args1,attrs.Count * 2);
                var i : Integer := 0;
                for var arg in attrs do
                begin
                    args1[i]     := arg.Key;
                    args1[i + 1] := arg.Value;
                    i := i + 2;
                end;
                tf.Runner.RecordGradient(opType, op.inputs.ToArray, args1, op.outputs);
            end;

            Exit(res);
        end;
    end else
    begin
        Result := ExecEagerAction(opType, name, args);
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
    if (not string.IsNullOrEmpty(string(name))) or (not executing_eagerly) then
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

end.
