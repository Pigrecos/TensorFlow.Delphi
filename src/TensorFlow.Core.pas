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
unit TensorFlow.Core;
{$POINTERMATH ON}
{$WARN DUPLICATE_CTOR_DTOR OFF}
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
uses
  System.SysUtils,
  System.Types,
  Winapi.Windows,
  System.Rtti,
  system.TypInfo,
  System.Generics.Collections,

  Spring,
  Spring.Collections,
  Spring.Collections.Base,
  Spring.Collections.Enumerable,

  TF4D.Core.CApi,
  Tensorflow.DApi,
  TF4D.Core.CApiEager,
  TensorFlow.DApiBase,
  TensorFlow.Slice,
  Tensorflow.Interfaces,
  TensorFlow.Initializer,

  TensorFlow.Proto;

  const
  C_GRAPH_MODE : Integer = 0;
  C_EAGER_MODE : Integer = 1;

  EAGER_CONST_THRESHOLD : Integer = 128;


type
 TCallBack        = Reference to procedure;
 BackwardFunction = Reference to function(grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>;

 ConcreteFunction     = class;
 EagerDefinedFunction = class ;
 ResourceVariable     = class;

 IVariableV1 = interface;

{$REGION 'Context'}
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
       destructor Destroy;override;
       function ToString: string;override;

       property EagerMode          : Boolean read FEagerMode;
       property IsBuildingFunction : Boolean read FIsBuildingFunction;
       property DeviceStack        : string  read FDeviceStack;
  end;

  ContextSwitchStack = class
    private
       FStack : TObjectStack<ContextSwitch>;
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

  TFunctionCallOptions = class
    public
      Config : TConfigProto;
      function config_proto_serialized : string;
      constructor Create;
      destructor Destroy; override;
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
        FFunctionCallOptions  : TFunctionCallOptions;
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

        property Handle_              : Pointer              read GetHandle;
        property Config               : TConfigProto         read FConfig write FConfig;
        property FunctionCallOptions  : TFunctionCallOptions read FFunctionCallOptions;
  end;
{$ENDREGION}

{$REGION 'constant_op'}
 constant_op = class
    private
      class function convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType=DtInvalid): TFTensor; overload;
      class function _eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
      class function _eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
    public
      class function convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;
      class function convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;overload;
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
      class function constant(value: TValue; dtype : TF_DataType= DtInvalid; shape : PTFShape = nil; verify_shape : Boolean = false; allow_broadcast : Boolean = true; name : AnsiString = 'Const'): TFTensor; overload;
      class function constant(value: TValue; dtype : TF_DataType; name : AnsiString = 'Const'): TFTensor; overload;
      /// <summary>
      /// Function to convert Shape to Tensor.
      /// </summary>
      /// <param name="s"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="as_ref"></param>
      /// <returns></returns>
      class function _tensor_shape_tensor_conversion_function(s: TFShape; dtype: TF_DataType = TF_DataType.DtInvalid; name: string = ''; as_ref : Boolean = false) : TFTensor;
      class function is_constant(tensor_or_op: ITensorOrOperation) : Boolean;
 end;
{$ENDREGION}

{$REGION 'Gradient'}
  TGradFunc = record
    Name : string;
    func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>;

    constructor Create(_name: string; _func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>);
  end;

  TapeTensor = record
    private
       Ftensor  : TFTensor;

       function GetShape: TFShape;
       function GetType: TF_DataType;

    public
       constructor Create(t: TFTensor);
       function GetID : Int64;
       function GetTensor : TFTensor;
       function ZerosLike: TFTensor;
       function OnesLike: TFTensor;
       function ToString: string;

       property tensor : TFTensor    read Ftensor;
       property Id     : Int64       read GetId;
       property dtype  : TF_DataType read GetType;
       property shape  : TFShape     read GetShape;
  end;

  ITape = class abstract
     private
     public
        F_persistent : Boolean;

        procedure SetTapeId(id: Integer); virtual; abstract;
        function  ShouldRecord(tensors: TArray<TFTensor>): Boolean;virtual; abstract;
        procedure StartRecord;virtual; abstract;
        procedure StopRecord;virtual; abstract;
        procedure RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);virtual; abstract;
        procedure VariableAccessed(variable: ResourceVariable);virtual; abstract;
        procedure Watch(x: TFTensor);virtual; abstract;
        function  WatchedVariables: TArray<ResourceVariable>;virtual; abstract;
        function  ComputeGradient(target_tensor_ids: TArray<TFTensor>; source_tensor_ids: TArray<TFTensor>;  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;virtual; abstract;

        property Persistente : Boolean read F_persistent;
  end;

  /// <summary>
  /// Represents an entry in the tape.
  /// </summary>
  OpTapeEntry = record
      op_type            : string;
      output_tensor_info : TArray<TapeTensor> ;
      input_tensor_id    : TArray<TFTensor>;
      backward_function  : BackwardFunction;
      ToString           : string;
  end;
  /// <summary>
  /// Map from operation-id to tape entry.
  /// </summary>
  OpTape = TDictionary<Int64, OpTapeEntry> ;
  /// <summary>
  /// Map from tensor to internally-defined operation-id of the operation which
  /// produced this tensor. A value of -1 means that the tensor was directly
  /// watched and not the result of any operation in the tape.
  /// </summary>
  TensorTape = TDictionary<TFTensor, Int64>;


   BackpropInitialState = class
      private

      public
        op_tape : OpTape;
        /// <summary>
        /// Map from tensor to how many references still exist for this tensor in
        /// the tape.
        /// </summary>
        tensor_usage_counts : TDictionary<TFTensor, Int64>;
        /// <summary>
        /// Maps from op ID to how many output tensors of this op still need to have
        /// their gradients computed.
        /// </summary>
        op_missing_tensor : TDictionary<Int64, Int64>;

        constructor Create;
        destructor Destroy; override;
   end;

  /// <summary>
  /// Gradient Tape Set
  /// Record operations for automatic differentiation.
  ///
  /// Operations are recorded if they are executed within this context manager and
  /// at least one of their inputs is being "watched".
  ///
  /// Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
  /// where `trainable=True` is default in both cases) are automatically watched.
  /// Tensors can be manually watched by invoking the `watch` method on this context
  /// manager.
  /// </summary>
  TGradientTape = class(TFDisposable)
    private
       FnextTapeId : Integer;
       FtapeSet    : TStack<ITape>;
       function GetTape: ITape;
    protected
      procedure NativeDispose(hnd: Pointer); override;
    public
      constructor Create;
      destructor  Destroy; override;
      /// <summary>
      /// New tape onto the tape stack.
      /// </summary>
      function PushTape(persistent: Boolean = false; watch_accessed_variables: Boolean = true): ITape;
      function PopTape: ITape;
      /// <summary>
      /// Marks this tensor to be watched by the given tape.
      /// </summary>
      /// <param name="x"></param>
      procedure watch(x: TFTensor);
      /// <summary>
      /// Computes the gradient using operations recorded in context of this tape.
      /// </summary>
      /// <param name="target"></param>
      /// <param name="source"></param>
      /// <returns></returns>
      function gradient(target: TFTensor; const source: TFTensor): TFTensor;overload;
      function gradient(target: TFTensor; const source: ResourceVariable): TFTensor;overload;
      function gradient(target: TFTensor; const sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor,TFTensor> overload;
      function gradient(target: TFTensor; const sources: TArray<IVariableV1>): TArray<TFTensor>;overload;
      /// <summary>
      /// Temporarily stops recording operations on this tape.
      /// </summary>
      function stop_recording: ITape;
      function GetTapeSet: TStack<ITape>;

      property Ftape: ITape read GetTape;
  end;

  TTape = class(ITape)
     private
        Fid              : Integer;
        // static int tape_nesting_id_counter = 0;
        F_recording      : Boolean;
        F_created_eagerly: Boolean;
        Ftensor_tape_    : TensorTape;
        Fop_tape_        : OpTape;
        Ftensor_usage_   : TDictionary<TFTensor, Int64>;
     public
        next_op_id_ : Integer;
        /// <summary>
        /// A deque-backed stack, whose element references are not invalidated by
        /// pushes and pops at the back.
        /// </summary>
        // Stack<AccumulatorCallState> call_state_;
        constructor Create(persistent: Boolean; watch_accessed_variables: Boolean);
        destructor Destroy;override;

        function  InitialStack(op_tape: OpTape; op_missing_tensor: TDictionary<Int64, Int64>): IQueue<Int64>;
        function  InitialGradients(target_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients : TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape): TDictionary<TFTensor, TList<TFTensor>>;
        function  FunctionsAcceptingNoneForIndicesMap : TDictionary<string, ISet<Integer>> ;
        function  PrepareBackprop(target: TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape; sources_set: ISet<TFTensor>; persistent_tape: Boolean) : BackpropInitialState;
        function  ComputeGradient(target_tensor_ids: TArray<TFTensor>; source_tensor_ids: TArray<TFTensor>;  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;override;
        procedure RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);override;
        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        procedure Watch(x: TFTensor);override;
        function  ShouldRecord(tensors: TArray<TFTensor>): Boolean; override;
        procedure VariableAccessed(variable: ResourceVariable); override;
        function  WatchedVariables: TArray<ResourceVariable>;  override;
        function  IsDtypeTrainable(dtype: TF_DataType): Boolean;
        procedure StartRecord; override;
        procedure StopRecord; override;
        procedure SetTapeId(id: Integer); override;
        function  ToString: string; reintroduce;

        property  Persistente : Boolean read F_persistent;

  end;
{$ENDREGION}

{$REGION 'EagareRunner'}
TFastPathOpExecInfo  = class
  public
    ctx                     : TContext;
    device_name             : string;
    op_name                 : string;
    name                    : string;
    args                    : TArray<TValue> ;
    attrs                   : TDictionary<string, TValue>;
    run_gradient_callback   : Boolean;
    run_post_exec_callbacks : Boolean;
    run_callbacks           : Boolean;
    callbacks               : TCallBack;

    constructor Create(opName:string; name: string; inputArgs: TArray<TValue>);
end;

TExecute  = record
  private

  public
    class function convert_to_mixed_eager_tensors(values:  TArray<TFTensor>; ctx: TContext) : Tuple< TArray<TDataType>, TArray<TFTensor> >; static;
    class function quick_execute(op_name: string; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>; ctx: TContext; name: string = ''): TArray<TFTensor>; static;
    class function must_record_gradient: Boolean; static;
end;

TEagerRunner = class(TFDisposable)
  private
    function SetOpAttrList  (ctx: TContext; op: PTFE_Op; key: string; values: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
    function SetOpAttrScalar(ctx: TContext; op: PTFE_Op; key: string; value:  TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
    /// <summary>
    /// This function will set the op attrs required. If an attr has the value of
    /// None, then it will read the AttrDef to get the default value and set that
    /// instead. Any failure in this function will simply fall back to the slow
    /// path.
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="op"></param>
    /// <param name="attr"></param>
    /// <param name="attr_name"></param>
    /// <param name="attr_value"></param>
    /// <param name="attr_list_sizes"></param>
    /// <param name="status"></param>
    procedure SetOpAttrWithDefaults(ctx: TContext; op: PTFE_Op; attr: TAttrDef; attr_name: string; attr_value: TValue; attr_list_sizes: TDictionary<string, Int64>;status: TFStatus) ;
    /// <summary>
    /// Adds input and type attr to the op, and to the list of flattened
    /// inputs/attrs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="add_type_attr"></param>
    /// <param name="input_arg"></param>
    /// <param name="op"></param>
    /// <param name="status"></param>
    /// <returns></returns>
    function AddInputToOp(inputs: TValue; add_type_attr: Boolean; input_arg: TArgDef;flattened_attrs: TList<TValue>; flattened_inputs: TList<TFTensor>; op: PTFE_Op;status: TFStatus): Boolean;
    function HasAccumulator: Boolean;
    function HasAccumulatorOrTape: Boolean;
    function HasGradientTape: Boolean;
    function MakeTensorList(tensors: TArray<TFTensor>): TArray<TFTensor>;
    function ShouldRecord(inputs: TArray<TFTensor>): Boolean;

  protected
    procedure NativeDispose(hnd: Pointer); override;
  public
    thread_local_eager_operation_map : TDictionary<string, PTFE_Op>;

    constructor Create; overload;
    constructor Create(hHandle: Pointer); overload;
    destructor  Destroy; override;

    function  GetOp(ctx: TContext; op_or_function_name: AnsiString; status: TFStatus): PTFE_Op;
    procedure SetOpAttrs(op: PTFE_Op; attrs: TArray<TValue>);

    function CouldForwardprop : Boolean;
    function CouldBackprop : Boolean;
    function TapeSetRecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TFTensor>; backward_function: BackwardFunction): Boolean;
    function TapeSetRecordForwardprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function_getter: BackwardFunction): Boolean;
    procedure TapeSetRecordBackprop(   op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);

    function GetGradientFunction(op_name: string; op_inputs: TArray<TFTensor>; attrs: TArray<TValue>; op_outputs: TArray<TFTensor>) : BackwardFunction;
    function RunCallbacks(op_exec_info: TFastPathOpExecInfo; num_inferred_attrs: Integer; inputs: TArray<TFTensor>;attrs: TArray<TValue>; flattened_result: TArray<TFTensor>): Boolean;
    function RecordGradient(op_name: string; inputs: TArray<TFTensor>; attrs: TArray<TValue>; results: TArray<TFTensor>; backward_Function: BackwardFunction = nil) : Boolean;
    function MustRecordGradient: Boolean;
    function TFE_TapeGradient(tape: ITape; target, sources, output_gradients: TArray<TFTensor>): TArray<TFTensor>;
    function TFE_Execute(ctx: TContext; device_name: AnsiString; op_name: AnsiString; inputs: TArray<TFTensor>;attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
    function TFE_ExecuteCancelable(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>; attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
    function TFE_FastPathExecute(op_exec_info: TFastPathOpExecInfo): TArray<TFTensor>;
    procedure ClearEagerOperationMap;
    /// <summary>
    /// Execute a TensorFlow operation.
    /// </summary>
    /// <param name="op_name">
    /// Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
    /// execute.
    /// </param>
    /// <param name="num_outputs">
    /// The number of outputs of the operation to fetch.
    /// </param>
    /// <param name="inputs">
    /// A list of inputs to the operation. Each entry should be a Tensor, or
    /// a value which can be passed to the Tensor constructor to create one.
    /// </param>
    /// <param name="attrs">
    /// A tuple with alternating string attr names and attr values for this
    /// operation.
    /// </param>
    /// <param name="ctx">The value of context.context().</param>
    /// <param name="name">Customized name for the operation.</param>
    /// <returns>List of output Tensor objects. The list is empty if there are no outputs</returns>
    function Execute(ctx: TContext; op_name: AnsiString; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>;name: AnsiString = '') : TArray<TFTensor>;
    function ArgsToMatchingEager(ctx: TContext; default_dtype: TF_DataType = TF_DataType.DtInvalid; args: TArray<TValue> = nil):  Tuple<TF_DataType, TArray<TFTensor>> ;
end;
{$ENDREGION}

{$REGION 'Framework'}
  /// <summary>
  /// Specifies a TensorFlow value type.
  /// </summary>
  TypeSpec = class

  end;

  /// <summary>
  /// Describes a dense object with shape, dtype, and name.
  /// </summary>
  DenseSpec = class(TypeSpec)
     private

     protected
        Fshape : TFShape;
        Fdtype : TF_DataType;
        Fname  : string;
     public

       constructor Create(_shape: TFShape; _dtype: TF_DataType = TF_FLOAT; _name: string = '');
       function ToString: string; override;

       property shape : TFShape     read Fshape;
       property dtype : TF_DataType read Fdtype;
       property name  : string      read Fname;
  end;

  TensorSpec  = class(DenseSpec)
     public
       constructor Create(shape: TFShape; dtype: TF_DataType = TF_FLOAT; name: string = '');
       function _unbatch: TensorSpec;
       function _batch(dim: Integer = -1):  TensorSpec;
  end;

  smart_module = class
    class function smart_cond(_pred: TFTensor; true_fn : TFunc< TArray<TFTensor> > = nil; false_fn : TFunc< TArray<TFTensor> > = nil; name: string = ''): TArray<TFTensor>; overload;
    class function smart_cond(_pred: Boolean; true_fn : TFunc<TFTensor> = nil; false_fn : TFunc<TFTensor> = nil; name: string = ''): TFTensor; overload;

    class function smart_constant_value(_pred: TFTensor) : Nullable<Boolean>;
  end;

  common_shapes = class
     public
       class function has_fully_defined_shape(tensor: TFTensor): Boolean;
       class function rank(tensor: TFTensor): Integer;
       /// <summary>
       /// Returns the broadcasted shape between `shape_x` and `shape_y
       /// </summary>
       /// <param name="shape_x"></param>
       /// <param name="shape_y"></param>
       class function broadcast_shape(shape_x: TFTensor; shape_y: TFTEnsor): TFtensor;
       /// <summary>
       /// Helper functions for is_broadcast_compatible and broadcast_shape.
       /// </summary>
       /// <param name="shape_x"> A `Shape`</param>
       /// <param name="shape_y"> A `Shape`</param>
       /// <return> Returns None if the shapes are not broadcast compatible,
       /// a list of the broadcast dimensions otherwise.
       /// </return>
       class function _broadcast_shape_helper(shape_x: TFTensor; shape_y: TFTEnsor): TFtensor;
  end;

  // tensor_shape move to Tensorflow.tensor.Ragged  for Circular reference

  /// <summary>
  /// Abstract base class for Tensor-like objects that are composed from Tensors.
  /// </summary>
  CompositeTensor = class abstract
  end;

  /// <summary>
  /// A sparse representation of a set of tensor slices at given indices.
  /// </summary>
  IndexedSlices = {class(CompositeTensor)} record
    private
       Fvalues     : TFTensor;
       Findices    : TFTensor;
       Fdense_shape: TFTensor;

       function GetDevice: string;
       function GetDtype: TF_DataType;
       function GetGraph: TFGraph;
       function GetName: string;
       function GetOp: TFOperation;
    public
        constructor Create(_values: TFTensor; _indices: TFTensor; _dense_shape: TFTensor = nil);
        class operator implicit(iSlices: IndexedSlices): TFTensor;
        class operator implicit(tTEnsor: TFTensor): IndexedSlices;

        property values      : TFTensor    read Fvalues;
        property indices     : TFTensor    read Findices;
        property dense_shape : TFTensor    read Fdense_shape;
        property name        : string      read GetName;
        property device      : string      read GetDevice;
        property op          : TFOperation read GetOp ;
        property dtype       : TF_DataType read GetDtype;
        property graph       : TFGraph     read GetGraph;
  end;

  op_def_registry = class
     private
       class var registered_ops  : TDictionary<string,TOpDef>;
     public
       class procedure FreeDictionary;
       class function get_registered_ops: TDictionary<string,TOpDef> ;
       class function GetOpDef(tipo : string): TOpDef;
   end;

   random_seed = record
     private
        const DEFAULT_GRAPH_SEED = 87654321;
        class var Fgraph_to_seed_dict : TDictionary<string,Integer> ;
     public
       class function get_seed(op_seed: TNullableInteger) : Tuple<TNullableInteger,TNullableInteger>; static;
       class function get_seed_tensor(op_seed: TNullableInteger) : Tuple<TFTensor,TFTensor>; static;
   end;
{$ENDREGION}

{$REGION 'Graph'}
  /// <summary>
  ///     Serves as a stack for determining current default graph.
  /// </summary>
  DefaultGraphStack = class
     private
      F_stack : TStack<TFGraph>;
      F_global_default_graph : TFGraph;
     public
      constructor Create;
      destructor Destroy; override;
      function  get_default: TFGraph;
      function  get_controller(g: TFGraph): TFGraph;
      function  peak_controller: TFGraph;
      procedure pop;
      procedure reset;

      property global_default_graph : TFGraph         read F_global_default_graph;
      property stack                : TStack<TFGraph> read F_stack;
  end;

  /// <summary>
  /// Graph representing a function body.
  /// </summary>
  TFuncGraph = class(TFGraph)
     private
       F_func_graph_handle : Pointer;
       F_captures          : TDictionary<Int64, Tuple<TFTensor, TFTensor> > ;

       function getFuncName: string;
       function getCapture_Inputs: TArray<TFTensor>;
       function getCaptures: TArray<Tuple<TFTensor, TFTensor>>;
       function getExCapture: TArray<TFTensor>;
       function InterCaptures: TArray<TFTensor>;

     protected
       procedure NativeDispose(hnd: Pointer); override;
     public
       Inputs           : TFTensors;
       Outputs          : TFTensors;
       Attrs            : TDictionary<string, string>;

       // <summary>
       /// Construct a new FuncGraph.
       /// </summary>
       constructor Create(name: string) ; overload;
       constructor Create(_handle: Pointer; name: string; _attrs: TDictionary<string, string>) ; overload;
       function    capture(tensor: TFTensor; name: string = ''; shape: PTFShape = nil): TFTensor;
       function    capture_eager_tensor(tensor: TFTensor; name: string): TFTensor;
       function    _capture_helper(tensor: TFTensor; name: string; shape: PTFShape = nil): TFTensor;
       procedure   ToGraph(opers: TArray<TFOperation>; _inputs : TArray<TFTensor>; _outputs : TArray<TFTEnsor>; output_names: TArray<string>) ;
       function    _create_substitute_placeholder(value: TFTensor; name : string= ''; dtype: TF_DataType = DtInvalid; shape: PTFShape = nil): TFTensor;
       procedure   SetAttrs;
       function    as_default: TFGraph; override;
       procedure   gExit; override;
       procedure  add_capture(tensor: TFTensor; placeholder: TFTensor);
       function    Create_op(op_type    : TF_TString ;
                        inputs         : TArray<TFTensor>;
                        dtypes         : TArray<TF_DataType>;
                        input_types    : TArray<TF_DataType> = [];
                        Name           : TF_TString= '';
                        attrs          : TDictionary<string, TAttrValue> = nil;
                        op_def         : TOpDef= nil;
                        compute_device : Boolean = True) : TFOperation; override;

       property FuncName         : string read getFuncName;
       property external_captures: TArray<TFTensor>                    read getExCapture;
       property captures         : TArray< Tuple<TFTensor, TFTensor> > read getCaptures;
       property internal_captures: TArray<TFTensor>                    read InterCaptures;
       property captured_inputs  : TArray<TFTensor>                    read getCapture_Inputs;
  end;

  SubGraphUtility = record
     private

     public

        /// <summary>
        /// Copies the tensor and all its inputs recursively to the outer graph.
        /// </summary>
        /// <param name="tensors"></param>
        /// <param name="graph"></param>
        /// <param name="add_sources"></param>
        /// <param name="handle_captures"></param>
        /// <param name="base_graph"></param>
        /// <returns></returns>
        class function lift_to_graph(init_tensors: TFTensors;
                                graph           : TFuncGraph;
                                sources         : TList<TFTensor>;
                                add_sources     : Boolean = false;
                                handle_captures : Boolean = false;
                                base_graph      : TFGraph = nil;
                                op_map : TDictionary<ITensorOrOperation, TFOperation> = nil): TDictionary<ITensorOrOperation, TFOperation> ;static;

        class Procedure _copy_source(s : TFTensor;
                                graph           : TFuncGraph;
                                op_map          : TDictionary<ITensorOrOperation, TFOperation>;
                                handle_captures : Boolean;
                                inverse_captures: TDictionary<TFTensor, TFTensor>;
                                base_graph      : TFGraph); static;

        class Procedure _copy_non_source(op: TFOperation; graph: TFuncGraph; op_map: TDictionary<ITensorOrOperation, TFOperation>; base_graph: TFGraph); static;

        /// <summary>
        /// Walk a Graph and capture the subgraph between init_tensor and sources.
        /// </summary>
        /// <param name="init_tensor"></param>
        /// <param name="add_sources"></param>
        class function  map_subgraph(init_tensor: TFTensor; sources: TList<TFTensor>; visited_ops: TList<TFOperation>; add_sources: Boolean): TList<TFTensor>; static;
  end;
{$ENDREGION}

{$REGION 'Functions'}
   /// <summary>
   /// Caches forward and backward functions compatible with eager gradients.
   /// </summary>
   TapeGradientFunctions = class
     private

     protected
       Ffunc_graph                 : TFuncGraph;
       Fforward                    : EagerDefinedFunction;
       Fforward_graph              : TFuncGraph;
       Fforwardprop_output_indices : TList<Integer>;
       Fnum_forwardprop_outputs    : Integer;
       Fbackward                   : ConcreteFunction;
       Fbackward_function_wrapper  : BackwardFunction;

       function BuildFunctionsForOutputs(outputs: TFTensors; inference_args: TFTensors): Tuple<EagerDefinedFunction,TFuncGraph,ConcreteFunction> ;
     public
      const
        FORWARD_FUNCTION_ATTRIBUTE_NAME  : string = 'forward_function_name';
        BACKWARD_FUNCTION_ATTRIBUTE_NAME : string = 'backward_function_name';
        _FORWARD_PREFIX                  : string = '__forward_';
        _BACKWARD_PREFIX                 : string = '__backward_';
        _INFERENCE_PREFIX                : string = '__inference_';
     public

       constructor Create(f_func_graph: TFuncGraph; need_gradients_for_jvps : Boolean);
       function    &Forward(inference_args: TFTensors): EagerDefinedFunction;
       /// <summary>
       /// Record the function call operation.
       /// </summary>
       /// <param name="flat_outputs"></param>
       /// <param name="inference_args"></param>
       procedure &Record(flat_outputs: TFTensors; inference_args: TFTensors);
       /// <summary>
       /// Create a backward function given `outputs` from the forward function.
       /// </summary>
       /// <param name="forward_graph"></param>
       /// <param name="backward"></param>
       /// <param name="outputs"></param>
       /// <returns></returns>
       function _wrap_backward_function(forward_graph: TFuncGraph; backward: ConcreteFunction; outputs: TFTensors): Tuple<BackwardFunction, TFTensors>;

       function ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction; virtual;
   end;

   FirstOrderTapeGradientFunctions = class(TapeGradientFunctions)
     public

       constructor Create(func_graph: TFuncGraph; need_gradients_for_jvps: Boolean);
       function    ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction; override;
   end;

   EagerDefinedFunction = class
     private

       function Get_Name: string;
     protected
       func_graph      : TFuncGraph;
       fnum_outputs    : Integer;

     public
       constructor Create(name: string; graph: TFuncGraph; inputs: TFTensors; outputs: TFTensors; attrs: TDictionary<string, string> );
       function Call(args: TFTensors): TFTensors;

       property Name : string  read Get_Name;
   end;

   /// <summary>
   /// Holds the state of a function call between execution and recording.
   /// </summary>
   ForwardBackwardCall = class
     private
       Ffunctions       : TapeGradientFunctions;
       Finference_args  : TFTensors;
       Finput_tangents  : TFTensors;
       Ftape_watching   : Boolean;
       Fforward_function: EagerDefinedFunction ;
     public

       constructor Create(functions: TapeGradientFunctions; inference_args: TFTensors; tape_watching: Boolean);
       function  &Forward:  Tuple<EagerDefinedFunction, TFTensors>;
       procedure &Record(flat_outputs: TFTensors);
   end;

   ConcreteFunction  = class
     private
        function Get_CaptInput: TArray<TFTensor>;
        function Get_Inputs: TArray<TFTensor>;
        function Get_Name: string;
     protected
        func_graph      : TFuncGraph;
        forward_backward: ForwardBackwardCall;
        Outputs         : TArray<TFTensor>;
        ReturnType      : PTypeInfo;
     public
        OutputStructure : TArray<TensorSpec>;
        ArgKeywords     : TArray<String>;
        NumPositionArgs : Int64;

        constructor Create(_name: string); overload;
        constructor Create(graph: TFuncGraph; attrs: TDictionary<string, string> = nil); overload;
        constructor Create(func: TFunc<TFTensor, TFTensor>; dtype: TF_DataType; func_name: string = ''); overload;
        procedure   ToGraph(inputs: TFTensors; outputs: TFTensors);
        procedure   AddTograph(g : TFGraph = nil);
        function    FilteredCall(inputs: TFTensors): TFTensors;
        function    SelectForwardAndBackwardFunctions(args: TFTensors; possible_gradient_type: Integer; executing_eagerly: Boolean) : ForwardBackwardCall;
        /// <summary>
        /// Executes the wrapped function.
        /// </summary>
        /// <param name="args"></param>
        /// <param name="captured_inputs"></param>
        /// <returns></returns>
        function CallFlat(args: TArray<TFTensor>; captured_inputs: TArray<TFTensor>) : TFTensors;
        function ToString: string; override;
        procedure   Enter;
        procedure   _Exit;

        property Inputs         : TArray<TFTensor> read Get_Inputs;
        property CapturedInputs : TArray<TFTensor> read Get_CaptInput;
        property Name           : string           read Get_Name;

  end;

  TFTensor_helper  = class Helper for TFTensor
    public
       function ToTensorSpec: TensorSpec;
  end;
{$ENDREGION}

{$REGION 'NameScope'}
  /// <summary>
  /// Returns a context manager that creates hierarchical names for operations.
  /// </summary>
  TNameScope = class(TInterfacedObject,ITensorFlowObject)
    private
      function enter_eager_name_scope(ctx: TContext; name:TF_TString): Tuple<TF_TString,TF_TString>;
    public
      _name          : TF_TString;
      _default_name  : TF_TString;
      _values        : TValue;
      scope_name     : TF_TString;
      old_scope_name : TF_TString;
      _skip_on_eager : boolean;

      constructor Create(name: TF_TString; default_name : TF_TString = ''; values : PValue = nil; skip_on_eager : Boolean = True);
      function ToString: TF_TString; reintroduce;
      procedure _Enter_;
      procedure _Exit_;
  end;
{$ENDREGION}

{$REGION 'Variable'}
   VariableArgs = record
      public
        InitialValue    : TValue;
        Getter          : TFunc<VariableArgs, IVariableV1>;
        Name            : string;
        Shape           : TFShape;
        DType           : TF_DataType;
        Initializer     : IInitializer;
        Trainable       : Boolean;
        ValidateShape   : Boolean;
        UseResource     : Boolean;
        Overwrite       : Boolean;
        Collections     : TList<string>;
        CachingDevice   : string;
        VariableDef     : TVariableDef;
        ImportScope     : string;
        Synchronization : TVariableSynchronization;
        Aggregation     : TVariableAggregation;
        class operator Initialize (out Dest: VariableArgs);
   end;

   /// <summary>
   /// Mode for variable access within a variable scope.
   /// </summary>
   _ReuseMode = (
        NOT_REUSE = 0,
        // Indicates that variables are to be fetched if they already exist or
        // otherwise created.
        AUTO_REUSE = 1);

   /// <summary>
   /// Variable store that carries a number of named Variables.
   /// </summary>
   _VariableStore = class
      private
         Fvars                 : TDictionary<string, TObject>;
         Fpartitioned_vars     : TDictionary<string, TObject>;
         Fstore_eager_variables: Boolean ;
         function  _true_getter(name           : string;
                                shape          : PTFShape = nil;
                                dtype          : TF_DataType = TF_DataType.TF_FLOAT;
                                initializer    : TObject = nil;
                                trainable      : PBoolean = nil;
                                collections    : TList<string> = nil;
                                validate_shape : Boolean = true;
                                synchronization: TVariableSynchronization = VARIABLE_SYNCHRONIZATION_AUTO;
                                aggregation    : TVariableAggregation     = VARIABLE_AGGREGATION_NONE): IVariableV1;
         function _get_single_variable(name           : string;
                                       shape          : PTFShape = nil;
                                       dtype          : TF_DataType = TF_DataType.DtInvalid;
                                       initializer    : IInitializer = nil;
                                       init_value     : TFTensor = nil;
                                       reuse          : Boolean = false;
                                       trainable      : PBoolean = nil;
                                       collections    : TList<string> = nil ;
                                       validate_shape : Boolean = false;
                                       use_resource   : PBoolean = nil;
                                       synchronization: TVariableSynchronization = VARIABLE_SYNCHRONIZATION_AUTO;
                                       aggregation    : TVariableAggregation = VARIABLE_AGGREGATION_NONE) : IVariableV1;
      public
         constructor Create;
         destructor Destroy; override;
         function get_variable(name           : string;
                               shape          : PTFShape = nil;
                               dtype          : TF_DataType = TF_DataType.TF_FLOAT;
                               initializer    : TObject = nil; // IInitializer or Tensor
                               reuse          : PBoolean = nil;
                               trainable      : PBoolean = nil;
                               collections    : TList<string> = nil;
                               validate_shape : Boolean = true;
                               synchronization: TVariableSynchronization = VARIABLE_SYNCHRONIZATION_AUTO;
                               aggregation    : TVariableAggregation= VARIABLE_AGGREGATION_NONE): IVariableV1;
   end;

   /// <summary>
   /// Variable scope object to carry defaults to provide to `get_variable`
   /// </summary>
   VariableScope = class
      private
        Freuse     : _ReuseMode;
        Fdtype     : TF_DataType;
        Fname      : string;
        Fname_scope: string;
      public
        use_resource : Boolean;
        resue        : Boolean;
      public
        constructor Create(reuse: Boolean; _name: string = ''; name_scope: string = ''; dtype: TF_DataType = TF_FLOAT) ;
        procedure reuse_variables;
        function get_variable(var_store: _VariableStore;
                              name           : string;
                              shape          : PTFShape = nil;
                              dtype          : TF_DataType = TF_DataType.DtInvalid;
                              initializer    : TObject= nil; // IInitializer or Tensor
                              trainable      : PBoolean = nil;
                              collections    : TList<string> = nil;
                              use_resource   : PBoolean = nil;
                              validate_shape : Boolean = true;
                              synchronization: TVariableSynchronization =VARIABLE_SYNCHRONIZATION_AUTO;
                              aggregation    : TVariableAggregation =VARIABLE_AGGREGATION_NONE): IVariableV1;
        property name               : string  read Fname;
        property original_name_scope: string  read Fname_scope ;

   end;

   _VariableScopeStore  = class
     private

     public
        current_scope         : VariableScope;
        variable_scopes_count : TDictionary<string, Integer>;
     public
        constructor Create;
        destructor Destroy; override;
        procedure open_variable_scope(scope_name: string);
        procedure close_variable_subscopes(scope_name: string);
        function variable_scope_count(scope_name: string) : Integer;
   end;

   PureVariableScope = class(TInterfacedObject, ITensorFlowObject)
     private
        Fname                        : string;
        Fscope                       : VariableScope;
        Fnew_name                    : string;
        Fold_name_scope              : string;
        Freuse                       : Boolean;
        Fvar_store                   : _VariableStore ;
        Fold                         : VariableScope;
        Fvar_scope_store             : _VariableScopeStore;
        Fcached_variable_scope_object: VariableScope;
        Fvariable_scope_object         : VariableScope;
        Flast_variable_scope_object  : VariableScope;
        Fold_subscopes               : TDictionary<string, Integer>;
     public
       constructor Create(name:  string;        old_name_scope : string = ''; dtype : TF_DataType = DtInvalid); overload;
       constructor Create(scope: VariableScope; old_name_scope : string = ''; dtype : TF_DataType = DtInvalid); overload;
       destructor Destroy; override;
       function ToVarScope: VariableScope;
       procedure _Enter_ ;
       procedure _Exit_ ;
   end;

   variable_scope = class(TInterfacedObject, ITensorFlowObject)
     private
        Fuse_resource              : Boolean;
        Fname                      : string;
        Fscope                     : VariableScope;
        Fdefault_name              : string;
        Fvalues                    : TArray<TFTensor>;
        Fcurrent_name_scope        : TNameScope;
        Fauxiliary_name_scope      : Boolean;
        Fcached_pure_variable_scope: PureVariableScope;
        Freuse                     : Nullable<Boolean>;
        Fin_graph_mode             : Boolean;
        Fgraph                     : TFGraph;
        F_building_function        : Boolean;
        function _enter_scope_uncached: VariableScope;
     public
        const _VARSTORE_KEY         : string  = '__variable_store';
        const _VARSCOPESTORE_KEY    : string  = '__varscope';
        const _DEFAULT_USE_RESOURCE : Boolean = true;
     public
        constructor Create(name: string;         default_name: string = ''; values: TArray<TFTensor> = nil; reuse: PBoolean = nil; auxiliary_name_scope : Boolean = true);overload;
        constructor Create(scope: VariableScope; default_name: string = ''; values: TArray<TFTensor> = nil; reuse: PBoolean = nil; auxiliary_name_scope : Boolean = true);overload;
        class function default_variable_creator(initial_value: TValue;
                                                name           : string   = '';
                                                trainable      : PBoolean = nil;
                                                collections    : TList<string>= nil;
                                                dtype          : TF_DataType = DtInvalid;
                                                shape          : TArray<Integer> = nil;
                                                validate_shape : Boolean = false;
                                                use_resource   : pBoolean = nil;
                                                synchronization: TVariableSynchronization = TVariableSynchronization. VARIABLE_SYNCHRONIZATION_AUTO;
                                                aggregation    : TVariableAggregation     = TVariableAggregation.VARIABLE_AGGREGATION_NONE) :  IVariableV1;
        procedure _Enter_ ;
        procedure _Exit_ ;
        /// <summary>
        /// Get a name with the given prefix unique in the current variable scope.
        /// </summary>
        /// <param name="prefix"></param>
        /// <returns></returns>
        class function  _get_unique_variable_scope(prefix: string): string;
        class function  _get_default_variable_store: _VariableStore;
        class function  get_variable_scope: VariableScope;
        class function  get_variable_scope_store : _VariableScopeStore;
        class function  _get_trainable_value(synchronization: TVariableSynchronization; trainable: Boolean = true): Boolean;

        property UseResource : Boolean       read Fuse_resource;
        property scope       : VariableScope read  Fscope;
   end;

  /// <summary>
  /// A variable maintains state in the graph across calls to `run()`. You add a
  /// variable to the graph by constructing an instance of the class `Variable`.
  ///
  /// The `Variable()` constructor requires an initial value for the variable,
  /// which can be a `Tensor` of any type and shape. The initial value defines the
  /// type and shape of the variable. After construction, the type and shape of
  /// the variable are fixed. The value can be changed using one of the assign methods.
  /// https://tensorflow.org/guide/variables
  /// </summary>
  IVariableV1 = interface

      function GetTipo: TF_DataType;
      function GetShape: TFShape;
      function GetGraph: TFGraph;
      function GetGraphEle:TFTensor;
      function GetOP: TFOperation;
      function GetInitializer: TFOperation;
      function GetDevice: string;
      function GetUniqueId: string;
      function GetName: String;
      function GetHandle:TFTensor;

      function assign_lazy_load(value: TFTensor; name: string = ''): IVariableV1;
      function assign_sub_lazy_load(delta: TFTensor; name: string = ''): IVariableV1;
      function _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
      function AsTensor(dtype: TF_DataType = TF_DataType.DtInvalid; name : string= ''; as_ref : Boolean= false): TFTensor;
      function numpy: TNDArray;

      property UniqueId    : string      read GetUniqueId;
      property Name        : string      read GetName;
      property tHandle     : TFTensor    read GetHandle;
      property Device      : String      read GetDevice;
      property Initializer : TFOperation read GetInitializer;
      property Op          : TFOperation read GetOP;
      property GraphElement: TFTEnsor    read GetGraphEle;
      property Graph       : TFGraph     read GetGraph;
      property Shape       : TFShape     read GetShape;
      property dtype       : TF_DataType read GetTipo;
  end;

  RefVariable = class(TInterfacedObject,IVariableV1)
     private
        Fgraph_element    : TFTEnsor;
        Fis_initialized_op: TFTensor;
        F_in_graph_mode   : Boolean;
        F_initial_value   : TFTensor;
        Ftrainable        : Boolean;
        Fsnapshot         : TFTensor;
        Fsave_slice_info  : Boolean;
        Finitializer_op   : TFOperation;
        FDevice           : string;

        function GetTipo: TF_DataType;
        function GetGraph: TFGraph;
        function GetOp: TFOperation;
        function GetShape: TFShape;
        function GetName: String;
        function GetHandle:TFTensor;
        function GetUniqueId: string;
        function GetDevice: String;
        function GetInitializer: TFOperation;
        function GetGraphEle: TFTEnsor;
     protected
        Fdtype    : TF_DataType;
        FName     : string;
        Fgraph_key: string;

     public
        _Variable : TFTensor;

        constructor Create(initial_value  : PValue = nil;
                           trainable      : Boolean = true;
                           collections    : TList<string> = nil;
                           validate_shape : Boolean = true;
                           caching_device : string = '';
                           name           : string = '';
                           variable_def   : TVariableDef= nil;
                           dtype          : TF_DataType = DtInvalid;
                           import_scope   : string = '');
        function _as_graph_element: TFTEnsor;
        function Eval: TFTensor;
        function AsTensor(dtype: TF_DataType = TF_DataType.DtInvalid; name : string= ''; as_ref : Boolean= false): TFTensor;
        function numpy: TNDArray;
        function _ref : TFTensor;
        function value: TFTensor;
        function  _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
        //  Update 'ref' by adding 'value' to it.
        //  This operation outputs "ref" after the update is done.
        //  This makes it easier to chain operations that need to use the reset value.
        //  Args:
        //    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
        //      Should be from a `Variable` node.
        //    value: A `Tensor`. Must have the same type as `ref`.
        //      The value to be added to the variable.
        //    use_locking: An optional `bool`. Defaults to `False`.
        //      If True, the addition will be protected by a lock;
        //        otherwise the behavior is undefined, but may exhibit less contention.
        //      name: A name for the operation(optional).
        //  Returns:
        //    A mutable `Tensor`. Has the same type as `ref`.
        function assign_add<T>(value : T ; use_locking : Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
        function assign_sub<T>(delta: T; use_locking: Boolean = false; name: string = ''; read_value : Boolean= true):TFTensor;
        function assign_sub_lazy_load(delta: TFTensor; name: string = ''): IVariableV1;
        /// <summary>
        /// Assigns a new value to the variable.
        /// </summary>
        /// <param name="value">The new value for this variable.</param>
        /// <param name="use_locking">If `True`, use locking during the assignment.</param>
        /// <param name="name">The name of the operation to be created</param>
        /// <param name="read_value">
        /// if True, will return something which evaluates to the
        /// new value of the variable; if False will return the assign op.
        /// </param>
        /// <returns>
        /// A `Tensor` that will hold the new value of this variable after
        /// the assignment has completed.
        /// </returns>
        function assign<T>(value: T; use_locking: Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
        function assign_lazy_load(value: TFTensor; name: string = ''): IVariableV1;
        function ToString: string; override;
        function To_VarScopeStore: _VariableScopeStore;
        function To_Tensor: TFTensor;

        property dtype            : TF_DataType read GetTipo;
        property UniqueId         : string      read GetUniqueId;
        property GraphElement     : TFTEnsor    read GetGraphEle;
        property Graph            : TFGraph     read GetGraph;
        property in_graph_mode    : Boolean     read F_in_graph_mode;
        property initial_value    : TFTensor    read F_initial_value;
        property trainable        : Boolean     read Ftrainable;
        property snapshot         : TFTensor    read Fsnapshot;
        property save_slice_info  : Boolean     read Fsave_slice_info;
        property Initializer      : TFOperation read GetInitializer;
        property Op               : TFOperation read GetOp;
        property Shape            : TFShape     read GetShape;
        property Device           : String      read GetDevice;
        property Name             : String      read GetName;
        property is_initialized_op: TFTensor    read Fis_initialized_op write Fis_initialized_op;
  end;

  BaseResourceVariable = class(TFDisposable)
     private
        Fdtype         : TF_DataType;
        Fname          : string;
        Fhandle_name   : string;
        Funique_id     : string;
        Fin_graph_mode : Boolean;
        Ftrainable     : Boolean;
        Finitializer_op: TFOperation;
        Fparent_op     : TFTensor;
        /// <summary>
        /// Tensor handle
        /// </summary>
        Fhandle        : TFTensor;
        Fgraph_element : TFTensor;

        function GetGraph: TFGraph;
        function GetDevice: string;
      protected
        Fshape         : TFShape;

        procedure NativeDispose(hnd: Pointer); override;
        function _read_variable_op: TFTensor;
     public
        constructor Create;
        destructor  Destroy; override;
        procedure __init__(ttrainable : Boolean= true; hHandle: TFTensor = nil; sName: string = ''; unique_id: string = ''; handle_name: string = '');
        function AsTensor(dtype: TF_DataType = TF_DataType.DtInvalid; name : string= ''; as_ref : Boolean= false): TFTensor;
        function numpy: TNDArray; virtual;
        function assign_add<T>(delta :T ; use_locking : Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
        function assign_sub<T>(delta: T; use_locking: Boolean = false; name: string = ''; read_value : Boolean= true):TFTensor;
        function assign_sub_lazy_load(delta: TFTensor; name: string = ''): IVariableV1;
        function assign<T>(value: T; use_locking: Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
        function assign_lazy_load(value: TFTensor; name: string = ''): IVariableV1;
        function _lazy_read(op: TFOperation; value: TFTensor ): IVariableV1;
        procedure StridedSliceAssign(value: TFTensor; slice: ParsedSliceArgs);
        procedure _strided_slice_assign(tBegin: TFTensor; tEnd: TFTensor; strides: TFTensor; value: TFTensor; name: string = '';
                                        begin_mask : Integer = 0; end_mask : Integer= 0; ellipsis_mask: Integer = 0; new_axis_mask: Integer = 0; shrink_axis_mask : Integer= 0);
        /// <summary>
        /// Records that `variable` was accessed for the tape and FuncGraph.
        /// </summary>
        procedure variable_accessed(variable: BaseResourceVariable) ;
        /// <summary>
        /// Constructs an op which reads the value of this variable.
        ///
        /// Should be used when there are multiple reads, or when it is desirable to
        /// read the value only after some condition is true.
        /// </summary>
        /// <returns></returns>
        function read_value: TFTensor;
        function value: TFTensor;
        function ToString: string; override;

        property dtype        : TF_DataType read Fdtype;
        property Name         : string      read Fhandle_name;
        property UniqueId     : string      read Funique_id;
        property trainable    : Boolean     read Ftrainable;
        property initializer  : TFOperation read Finitializer_op;
        property parent_op    : TFTensor    read Fparent_op;
        property GraphElement : TFTensor    read Fgraph_element;
        property shape        : TFShape     read Fshape;
        property Graph        : TFGraph     read GetGraph;
        property Device       : string      read GetDevice;
        property tHandle      : TFTensor    read Fhandle;
  end;

  /// <summary>
  /// Represents a future for a read of a variable.
  /// Pretends to be the tensor if anyone looks.
  /// </summary>
  _UnreadVariable = class(BaseResourceVariable, IVariableV1 )
     private
        function GetOP: TFOperation;
        function GetTipo: TF_DataType;
        function GetShape: TFShape;
        function GetGraphEle: TFTEnsor;
        function GetInitializer: TFOperation;
        function GetUniqueId: string;
        function GetName: string;
        function GetHandle:TFTensor;
     public
        constructor Create(hHandle: TFTensor; dDtype: TF_DataType; sShape: TFShape; in_graph_mode: Boolean; unique_id: string);
        function  _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;

        property Name  : string  read GetName;
  end;

  /// <summary>
  /// Variable based on resource handles.
  /// </summary>
  ResourceVariable = class(BaseResourceVariable,IVariableV1)
     private
        function GetOP: TFOperation;
        function GetGraph: TFGraph;
        function GetDevice: String;
        function GetHandle:TFTensor;
        function GetName: string;
        function GetUniqueId: string;
        function GetInitializer: TFOperation;
        function GetGraphEle: TFTEnsor;
        function GetShape: TFShape;
        function GetTipo: TF_DataType;

        procedure _init_from_proto(variable_def : TVariableDef; import_scope : string= '');
        procedure _init_from_args(_initial_value : PValue = nil;
                                  _trainable     : Boolean = true;
                                  collections   : TList<string> = nil;
                                  caching_device: string = '';
                                  name          : string = '';
                                  dtype         : TF_DataType = TF_DataType.DtInvalid;
                                  aggregation   : TVariableAggregation = TVariableAggregation.VARIABLE_AGGREGATION_NONE;
                                  shape         : PTFShape= nil);
       function GetTrainable: Boolean;
       function GetParent_op: TFTEnsor;
       function GetItem(slices: TArray<Slice>): TFTensor;overload;
       function GetItem(slices: TArray<string>): TFTensor;overload;

     protected
        Finitial_value : TFTensor;

     public
        constructor Create(_initial_value    : PValue;
                           _trainable        : Boolean= true;
                           collections      : TList<string>= nil;
                           validate_shape   : Boolean = true;
                           caching_device   : string = '';
                           name             : string= '';
                           variable_def     : TVariableDef= nil;
                           dtype            : TF_DataType = TF_DataType.DtInvalid;
                           import_scope     : string = '';
                           aggregation      : TVariableAggregation = TVariableAggregation.VARIABLE_AGGREGATION_NONE;
                           shape            : PTFShape= nil);
        function _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
        function _dense_var_to_tensor(dtype: TF_DataType = TF_DataType.DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
        function  sparse_read(indices: TFTensor; name : string= 'Gather') : TFTensor;
        function  to_proto(export_scope: string): TVariableDef;
        function  eval(session: TFSession = nil): TNDArray;
        function  ToTensor: TFTensor;
        function  numpy: TNDArray; override;

        property Name        : string       read GetName;
        property dtype       : TF_DataType  read GetTipo;
        property UniqueId    : string       read GetUniqueId;
        property trainable   : Boolean      read GetTrainable;
        property Initializer : TFOperation  read GetInitializer;
        property parent_op   : TFTEnsor     read GetParent_op;
        property GraphElement: TFTEnsor     read GetGraphEle;
        property Shape       : TFShape      read GetShape;
        property Op          : TFOperation  read GetOP;
        property Graph       : TFGraph      read GetGraph;
        property Device      : String       read GetDevice;
        property Item[slices: TArray<string>]: TFTensor read GetItem ; default;
        property Item[slices: TArray<Slice> ]: TFTensor read GetItem ; default;
  end;
{$ENDREGION}

implementation
          uses System.Math,
               Tensorflow,
               Tensorflow.Gradient,
               Tensorflow.Variable,
               TensorFlow.Ops,
               Tensorflow.Utils,
               Tensorflow.math_ops,
               Tensorflow.array_ops,
               Tensorflow.gen_array_ops,
               Tensorflow.gen_math_ops,
               Tensorflow.gen_state_ops,
               Tensorflow.control_flow_ops,
               TensorFlow.resource_variable_ops,
               Tensorflow.gen_resource_variable_ops,

               Oz.Pb.Classes,
               Oz.SGL.Collections,
               Oz.Pb.StrBuffer,

               Numpy.Axis,
               NumPy.NDArray,

               ProtoGen.Main;

{$REGION 'Context'}
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

destructor ContextSwitch.Destroy;
begin

  inherited Destroy;
end;

function ContextSwitch.ToString: string;
begin
    Result := Format('EagerMode: %s, IsBuildingFunction: %s',[BoolToStr(FEagerMode,True),BoolToStr(FIsBuildingFunction,True)]);
end;

{ ContextSwitchStack }

constructor ContextSwitchStack.Create(isEager, isFunc: Boolean);
begin
    FStack := TObjectStack<ContextSwitch>.Create;
    Push(isEager, isFunc)
end;

destructor ContextSwitchStack.Destroy;
begin
     FStack.Clear;
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
    inherited Create;
    _device_policy       := TFE_DEVICE_PLACEMENT_SILENT;
    defaultExecutionMode := C_EAGER_MODE;
    context_switches := ContextSwitchStack.Create(defaultExecutionMode = C_EAGER_MODE, false);
    initialized      := false;
    FConfig          := TConfigProto.Create;

    FFunctionCallOptions  := TFunctionCallOptions.Create;
end;

destructor TContext.Destroy;
begin
   context_switches.Free;
   FFunctionCallOptions.Free;
   FConfig.Free;
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
    FFunctionCallOptions.Config := FConfig;

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
            keywords.AddOrSetValue(attr.Key, attr.Value);
    end;

    var res := tf.OpDefLib._apply_op_helperDict(OpType, Name, keywords).Outputs;
    Result :=  TFTensors.Create( res );
end;

function TContext.ExecEagerAction(OpType, Name: string; args: ExecuteOpArgs): TFTensors;
begin
    var opExecInfo := TFastPathOpExecInfo.Create(OpType, Name, args.OpInputArgs);
    opExecInfo.attrs := args.OpAttrs;

    var ts := tf.Runner.TFE_FastPathExecute(opExecInfo);
    if ts = nil then  Exit(TFTensors.Create(TFTensor(nil)));

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
         //Randomize;
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
    var flatten_args := nest.flatten<TValue>(  TValue.From<TArray<TValue>>(args) );

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

{ TFunctionCallOptions }

constructor TFunctionCallOptions.Create;
begin
   // Config := TConfigProto.Create;
end;

destructor TFunctionCallOptions.Destroy;
begin
   inherited Destroy;
end;

function TFunctionCallOptions.config_proto_serialized: string;
var
   Saver  : TpbSaver;
begin
    Saver.Init;
    TpbSaver.SaveConfigProto(Saver,Config);
    Result := Saver.Pb.ToStringUtf8;
end;
{$ENDREGION}

{$REGION 'constant_op'}
{ constant_op }

class function constant_op.constant(value: TValue; dtype: TF_DataType; name: AnsiString): TFTensor;
begin
    Result := constant(value, dtype, nil, false, True, name);
end;

class function constant_op.constant(value: TValue; dtype: TF_DataType; shape: PTFShape; verify_shape, allow_broadcast: Boolean; name: AnsiString): TFTensor;
begin
    { TODO -oMax -c : verificare 06/11/2022 14:15:17 }
    if value.typeInfo = nil then
        Exit(nil);

    if tf.executing_eagerly then
        Result := convert_to_eager_tensor(value, dtype, shape, name, verify_shape, allow_broadcast)
    else
        Result := convert_to_graph_tensor(value, dtype, shape, name, verify_shape, allow_broadcast);
end;

class function constant_op.convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape,
  allow_broadcast: Boolean): TFTensor;
var
  v            : TpbOneof;
  tp           : TTensorProto;
  tensor_value,
  dtype_value  : TAttrValue;
   attrs       : TDictionary<string, TAttrValue>;
begin
    var g : TFGraph := TOps.get_default_graph;

    tp := TUtils.make_tensor_proto(value, dtype,@shape, verify_shape, allow_broadcast);

    v.tag        := TAttrValue.ftTensor;
    v.value      := TValue.From<TTensorProto>(tp);
    tensor_value       := TAttrValue.Create;
    tensor_value.Value := v;
    //
    v.tag   := TAttrValue.ftType;
    v.value := TValue.From<Integer>( Ord(dtype)  );
    dtype_value       := TAttrValue.Create;
    dtype_value.Value := v;

    attrs := TDictionary<string, TAttrValue>.Create;

    attrs.Add('value',tensor_value);
    attrs.Add('dtype',dtype_value);

    var oper := g.create_op(
        'Const',
        [],
        [TF_DataType(dtype_value.Value.value.AsType<Integer>)],
        [],
        name,
        attrs);

    Result := oper.outputs[0];
end;

class function constant_op.is_constant(tensor_or_op: ITensorOrOperation): Boolean;
begin
    if (tensor_or_op is TFTensor ) then
    begin
        var tensor : TFTensor := tensor_or_op as TFTensor;
        Result := tensor.op.Tipo = 'Const';
    end
    else if (tensor_or_op is TFOperation) then
    begin
        var op : TFOperation := tensor_or_op as TFOperation;
        Result := op.Tipo = 'Const';
    end
    else
       raise Exception.Create('is_constant');
end;

class function constant_op._eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(tensor.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(shape), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ tensor, dims_t ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'Tshape', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Reshape', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function constant_op._tensor_shape_tensor_conversion_function(s: TFShape; dtype: TF_DataType; name: string;
  as_ref: Boolean): TFTensor;
begin
    var s_list := s.dims;
    var int64_value : Int64 := 0;
    for var dim in s_list do
    begin
        if dim > Power(2, 31) then
        begin
            int64_value := dim;
            break;
        end;
    end;
    if  int64_value > 0 then dtype := TF_DataType.TF_INT64
    else                     dtype := TF_DataType.TF_INT32;
    if string.IsNullOrEmpty(name) then
        name := 'shape_as_tensor';
    Result := constant_op.constant(TValue.From< TArray<Int64> >(s_list), dtype, name);
end;

class function constant_op._eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(value.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(dims), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ dims_t, value ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'index_type', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Fill', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function constant_op.convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape, allow_broadcast: Boolean): TFTensor;
begin
    var t := convert_to_eager_tensor(value, tf.Context, dtype);
    if (dtype <> TF_DataType.DtInvalid) and (dtype <> t.dtype) then
       t := math_ops.cast(t, dtype);

    if ( PTFShape(shape) = nil) or (shape.IsNull) then
        Exit(t);

    if t.shape.Equals( TValue.From<TFShape>(shape)) then
        Exit(t);

    if verify_shape then
        raise Exception.Create( Format('Expected Tensor''s shape: %s, got %s.',[shape.ToString,t.Shape.ToString]));

    var num_t := t.shape.size;
    if num_t = shape.size then
        Exit(_eager_reshape(t, shape^, tf.Context) );
    if num_t = 1 then
    begin
        if t.dtype = Tdtypes.cbool then
            raise Exception.Create('Not Implemented')
        else
            Exit( _eager_fill(shape^, t, tf.Context) );
    end;

    raise Exception.Create('Not Implemented')
end;

class function constant_op.convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType): TFTensor;
begin
    ctx.ensure_initialized;
    var tipo : PTypeInfo;
    tipo:= value.TypeInfo;
    var tipoName : string := string.LowerCase(tipo.Name);
    // convert data type
    if (dtype <> TF_DataType.DtInvalid) and
       (tipoName <> 'tndarray') and (tipoName <> 'teagertensor') and
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
    else if (dtype <> TF_DataType.DtInvalid) and (value.TypeInfo = TypeInfo(TNDArray)) then
    begin
       if value.AsType<TNDArray>.Dtype <> dtype then
       begin
           var nd := value.AsType<TNDArray>;
           value := math_ops.cast(nd, dtype);
       end;
    end
    else if (dtype <> TF_DataType.DtInvalid) and  (value.TypeInfo = TypeInfo(TEagerTensor)) then
    begin
       if value.AsType<TEagerTensor>.Dtype <> dtype then
       begin
           var nd := value.AsType<TEagerTensor>;
           value := math_ops.cast(nd, dtype);
       end;
    end;

    // non ascii char
    if (dtype = TF_DataType.TF_STRING) and (value.IsArray) and (tipoName.Contains('byte') ) then
    begin
        Result := TEagerTensor.Create(Value.AsType< TArray<Byte> >, TFShape.Scalar, TF_DataType.TF_STRING);
        Exit;
    end
    else if (dtype = TF_DataType.TF_STRING) and (value.IsArray) and (tipoName.contains('uint8') ) then
    begin
        Result := TEagerTensor.Create(Value.AsType< TArray<Byte> >, TFShape.Scalar, TF_DataType.TF_STRING);
        Exit;
    end;
    if      value.TypeInfo = TypeInfo(TEagerTensor)  then  Result := value.AsType<TEagerTensor>
    else if value.TypeInfo = TypeInfo(TNDArray)      then  Result := value.AsType<TNDArray>
    else if value.TypeInfo = TypeInfo(TFShape)  then
    begin
         var vval := Value.AsType<TFShape>;
         Result := TEagerTensor.Create(vval.dims, TFShape.Create([vval.ndim]),TUtils.GetDataType(Value) );
    end
    else if tipoName = 'ptfshape' then
    begin
         var v := Value.AsType<PTFShape>;
         var vval := v^;
         Result := TEagerTensor.Create(vval.dims, TFShape.Create([vval.ndim]),TUtils.GetDataType(Value) );
    end
    else if value.TypeInfo = TypeInfo(TAxis)      then
    begin
         var vval := Value.AsType<TAxis>;
         var shape : TFShape;
         if vval.IsScalar then shape := TFShape.Scalar
         else                  shape := TFShape.Create([vval.size]);
         Result := TEagerTensor.Create(vval.axis, shape,TUtils.GetDataType(Value) );
    end
    else if tipoName = 'paxis'      then
    begin
         var v := Value.AsType<PAxis>;
         var vval := v^;
         var shape : TFShape;
         if vval.IsScalar then shape := TFShape.Scalar
         else                  shape := TFShape.Create([vval.size]);
         Result := TEagerTensor.Create(vval.axis, shape,TUtils.GetDataType(Value) );
    end
    else if (value.TypeInfo = TypeInfo(string)) or (value.TypeInfo = TypeInfo(AnsiString)) then
    begin
        var vval := Value.AsType<string>;
        Result := TEagerTensor.Create([vval], TFShape.scalar );
    end
    else if value.TypeInfo = TypeInfo(Boolean) then
    begin
        var vval := Value.AsType<Boolean>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_BOOL);
    end
    else if value.TypeInfo = TypeInfo(Boolean) then
    begin
        var vval := Value.AsType<Byte>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT8);
    end
    else if value.TypeInfo = TypeInfo(Integer) then
    begin
        var vval := Value.AsType<Integer>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT32);
    end
    else if value.TypeInfo = TypeInfo(Int64) then
    begin
        var vval := Value.AsType<Int64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT64);
    end
    else if value.TypeInfo = TypeInfo(UInt64) then
    begin
        var vval := Value.AsType<UInt64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT64);
    end
    else if (value.IsType<Single>) and (tipoName ='single') then
    begin
        var vval : Single := Value.AsType<Single>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_FLOAT);
    end
    else if value.IsType<Double> and (tipoName ='double') then
    begin
        var vval : Double := Value.AsType<Double>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_DOUBLE);
    end
    else if ((value.IsType<TArray<String>>) or (value.IsType<TArray<AnsiString>>))  and (tipoName.Contains('string')) then
    begin
        var vval : TArray<TF_TString> := [];
        if value.IsType<TArray<String>> then
        begin
            var vval0 : TArray<String> := Value.AsType<TArray<String>>;
            for var i := 0 to Length(vval0)-1 do
              vval := vval + [ vval0[i] ];
        end else
        begin
            vval := Value.AsType<TArray<TF_TString>>;
        end;
        Result := TEagerTensor.Create(vval, TFShape.Create( [ Length(vval) ] ) );
    end
    else if value.isArray then
    begin
        var sShape : TFShape := TUtils.GetShape(value);
        Result := TEagerTensor.Create(value, @sShape);
    end else
    begin
       var tData := GetTypeData(Value.TypeInfo)^;
       case tData.floatType of
         ftSingle: begin
           var vval : Single := Value.AsType<Single>;
           Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_FLOAT);
         end;
         ftDouble: begin
           var vval : Double := Value.AsType<Double>;
           Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_DOUBLE);
         end;
         ftExtended: begin
           var vval : Double := Value.AsType<Double>;
           Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_DOUBLE);
         end;
         (*ftComp:  begin
         end;
         ftCurr:  begin
         end;
         *)
       else
         raise Exception.Create('NotImplemented convert_to_eager_tensor Type: '+ Value.TypeInfo.Name);
       end;
    end;
end;
{$ENDREGION}

{$REGION 'EagareRunner'}
{ TEagerRunner }

constructor TEagerRunner.Create;
begin
    inherited Create(Nil);
    thread_local_eager_operation_map := TDictionary<string, PTFE_Op>.Create;
end;

constructor TEagerRunner.Create(hHandle: Pointer);
begin
    inherited Create(hHandle);
    thread_local_eager_operation_map := TDictionary<string, PTFE_Op>.Create;
end;

destructor TEagerRunner.Destroy;
begin
    thread_local_eager_operation_map.Free;
    inherited Destroy;
end;

function TEagerRunner.Execute(ctx: TContext; op_name: AnsiString; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>;
  name: AnsiString): TArray<TFTensor>;
begin
    ctx.ensure_initialized();

    Result := tf.Runner.TFE_Execute(ctx, ctx.DeviceName,op_name, inputs, attrs, num_outputs);
end;

procedure TEagerRunner.NativeDispose(hnd: Pointer);
begin
   if Assigned(hnd) then
     TFE_DeleteTensorHandle(hnd);
end;

function TEagerRunner.RunCallbacks(op_exec_info: TFastPathOpExecInfo; num_inferred_attrs: Integer; inputs: TArray<TFTensor>;
  attrs: TArray<TValue>; flattened_result: TArray<TFTensor>): Boolean;
begin
    if op_exec_info.run_gradient_callback then
    begin
        if not RecordGradient(op_exec_info.op_name, inputs, attrs, flattened_result) then
          Exit(false);
    end;
    if op_exec_info.run_post_exec_callbacks then
    begin

    end;
    Result := true;
end;

function TEagerRunner.RecordGradient(op_name: string; inputs: TArray<TFTensor>; attrs: TArray<TValue>; results: TArray<TFTensor>;
  backward_Function: BackwardFunction): Boolean;
begin
    var should_record := ShouldRecord(inputs);

    if  not should_record then
    begin
        (*for (TFE_Py_ForwardAccumulator* accumulator : SafeAccumulatorSet())
        {
            if (accumulator->accumulator->ShouldRecord(input_ids, input_dtypes))
            {
                should_record = true;
                break;
            }
        }*)
    end;
    if  not should_record  then Exit(should_record);
    // tf.Logger.Debug($"RecordGradient: op_name={op_name}");
    (*Tensor[] op_outputs = null;
    var unused_output_indices = gradient_exclustions.OpGradientUnusedOutputIndices(op_name);
    if (unused_output_indices != null)
    {
        if (unused_output_indices.Length == 0)
            op_outputs = new Tensor[0];
        else
        {
            // op_outputs = CopySequenceSettingIndicesToNull(results, *unused_output_indices);
        }
    }
    else
        op_outputs = results;
    Tensor[] op_inputs = null;
    var unused_input_indices = gradient_exclustions.OpGradientUnusedInputIndices(op_name);
    if (unused_input_indices != null)
    {
        if (unused_input_indices.Length == 0)
            op_inputs = new Tensor[0];
        else
        {
            // op_inputs = CopySequenceSettingIndicesToNull(inputs, *unused_input_indices);
        }
    }
    else
        op_inputs = inputs;*)

    if not Assigned(backward_Function) then
      backward_Function := GetGradientFunction(op_name, inputs, attrs, results);

    TapeSetRecordOperation(op_name, inputs, results, backward_Function);
    Result := true;
end;

function TEagerRunner.GetGradientFunction(op_name: string; op_inputs: TArray<TFTensor>; attrs: TArray<TValue>; op_outputs: TArray<TFTensor>): BackwardFunction;
begin
    Result := function(out_grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
                const
                 nonRegGradFuncs : Array[0..2] of string = ('GreaterEqual','OnesLike','ZerosLike');
               begin
                   if (not gradientFunctions.ContainsKey(op_name)) or (TArray.Contains<string>(nonRegGradFuncs, op_name)) then
                   begin
                       SetLength( Result,Length(op_inputs) );
                       Exit;
                   end;
                   var oper := EagerOperation.Create;
                   oper.Name            := op_name;
                   oper.NumInputs       := Length(op_inputs);
                   oper.Inputs          := op_inputs;
                   oper.NumOutputs      := Length(op_outputs);
                   oper.Outputs         := op_outputs;
                   oper.SkipInputIndices:= unneeded_gradients;
                   oper.Attrs           := attrs;

                   Result := gradientFunctions[op_name](oper, out_grads);
               end;
end;

function TEagerRunner.TapeSetRecordOperation(op_type: string; input_tensors, output_tensors: TArray<TFTensor>; backward_function: BackwardFunction): Boolean;
begin
    var output_info : TArray<TapeTensor> := [];
    for var i := 0 to Length(output_tensors)-1  do
    begin
        output_info :=  output_info + [ TapeTensor.Create(output_tensors[i]) ]
    end;

    if not TapeSetRecordForwardprop(op_type, input_tensors, output_info, backward_function) then
        Exit(false);
    TapeSetRecordBackprop(op_type, input_tensors, output_info, backward_function);
    Result := true;
end;

procedure TEagerRunner.TapeSetRecordBackprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);
begin
    if not CouldBackprop then
       Exit;
    for var tape in tf.GetTapeSet do
    begin
        tape.RecordOperation(op_type, input_tensors, output_tensors, backward_function);
    end
end;

function TEagerRunner.TapeSetRecordForwardprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function_getter: BackwardFunction): Boolean;
begin
    if not CouldForwardprop then
       Exit(true);
    raise TFException.Create('Not Implemented');
end;

function TEagerRunner.CouldForwardprop : Boolean;
begin
    Result := HasAccumulator;
end;
procedure TEagerRunner.ClearEagerOperationMap;
begin
   thread_local_eager_operation_map.Clear;
end;

function TEagerRunner.CouldBackprop : Boolean;
begin
     Result := HasGradientTape;
end;

procedure TEagerRunner.SetOpAttrs(op: PTFE_Op; attrs: TArray<TValue>);
begin
    var status := tf.Status;
    var len := Length(attrs);
    var i : Integer := 0;
    while i < len do
    begin
        var key   := attrs[i].AsString;
        var value := attrs[i + 1];
        var  is_list : Byte := 0;
        var tipo := TFE_OpGetAttrType(op, PAnsiChar(AnsiString(key)), @is_list, status.Handle);
        if not status.Ok then Exit;
        if is_list <> 0 then
            SetOpAttrList(tf.Context, op, key, value , tipo, nil, status)
        else
            SetOpAttrScalar(tf.Context, op, key, value, tipo, nil, status);
        status.RaiseEx;

       Inc(i,2);
    end;
end;
function TEagerRunner.SetOpAttrList(ctx: TContext; op: PTFE_Op; key: string; values: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, Int64>; status: TFStatus): Boolean;
var
   StrArrray : TArray<string>;
   pStrArray : TArray<AnsiString>;
   vlen      : TArray<UInt64>;
begin
    if (tipo = TF_AttrType.TF_ATTR_STRING) and (values.IsType< TArray<string> > ) then
    begin
        StrArrray := values.AsType< TArray<string> >;
        pStrArray := [];
        vlen     := [];
        for var i := 0 to Length(StrArrray) - 1 do
        begin
            pStrArray := pStrArray + [ AnsiString(StrArrray[i]) ];
            vlen      := vlen + [ Length( pStrArray[i] ) ];
        end;
        var pStrValue : PAnsiChar := nil;
        var pLen      : PUInt64   := nil;
        if Length(pStrArray) > 0 then
        begin
            pStrValue := @pStrArray[0];
            pLen      := @vlen[0];
        end;

        TFE_OpSetAttrStringList(op, PAnsiChar(AnsiString(key)), pStrValue,pLen, Length(pStrArray));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(pStrArray) );
    end
    else if (tipo = TF_AttrType.TF_ATTR_SHAPE) and (values.IsType< TArray<TFShape> >) then
    begin
        var values1 := values.AsType< TArray<TFShape> >;
        // Make one pass through the input counting the total number of
        // dims across all the input lists.
        var num_values := Length(values1);
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key,num_values);
        var dims : TArray<Pointer>;
        SetLength(dims,num_values);
        var num_dims : TArray<Integer>;
        for var i := 0 to Length(values1) - 1 do
            num_dims := num_dims + [ values1[i].ndim ] ;

        for var i := 0 to num_values - 1 do
        begin
            dims[i] := AllocMem(SizeOf(Int64) * values1[i].ndim);
            if values1[i].ndim > 0 then
               CopyMemory(dims[i], @values1[i].dims[0], values1[i].ndim * sizeof(Int64));
        end;
        var pIntValues : PInteger := nil;
        var pLen       : PInteger := nil;
        if Length(dims) > 0 then
        begin
            pIntValues := @dims[0];
            pLen       := @num_dims[0];
        end;

        TFE_OpSetAttrShapeList(op, PAnsiChar(AnsiString(key)), pIntValues, pLen, num_values, status.Handle);
        for var i := 0 to num_values - 1 do
           FreeMem(dims[i]);
    end
    else if (tipo = TF_AttrType.TF_ATTR_TYPE) and (values.IsType< TArray<Integer> >) then
    begin
        var values2 := values.AsType< TArray<Integer> >;
        TFE_OpSetAttrTypeList(op, PAnsiChar(AnsiString(key)), @values2[0], Length(values2));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(values2));
    end
    else if (tipo = TF_AttrType.TF_ATTR_INT) and (values.IsType< TArray<Integer> >) then
    begin

        var values4 := values.AsType< TArray<Integer> >;
        var vvalues4 : TArray<Int64>;
        for var i := 0 to Length(values4) - 1 do
            vvalues4 := vvalues4 + [ values4[i] ] ;

        var pInt64Value : PInt64 := nil;
        if Length(vvalues4) > 0 then pInt64Value := @vvalues4[0];

        TFE_OpSetAttrIntList(op, PAnsiChar(AnsiString(key)), pInt64Value, Length(values4));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(values4));
    end else
    begin
        raise Exception.Create('Not Implemented.');
    end;
    Result :=true;
end;

function TEagerRunner.SetOpAttrScalar(ctx: TContext; op: PTFE_Op; key: string; value: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
begin
    case tipo of
      TF_AttrType.TF_ATTR_STRING:
         begin
             var vValue := value.AsType< string >;
             TFE_OpSetAttrString(op, PAnsiChar(AnsiString(key)), PAnsiChar(AnsiString(vValue)), UInt64(Length(AnsiString( vValue ))));
         end;
      TF_AttrType.TF_ATTR_TYPE:
         begin
             var vValue := value.AsType< Integer >;
             TFE_OpSetAttrType(op, PAnsiChar(AnsiString(key)), TF_DataType(vValue));
         end;
      TF_AttrType.TF_ATTR_BOOL: TFE_OpSetAttrBool(op, PAnsiChar(AnsiString(key)), value.AsType< Boolean >);
      TF_AttrType.TF_ATTR_INT:
         begin
             var size := Int64(value.AsType<Integer>);
             TFE_OpSetAttrInt(op, PAnsiChar(AnsiString(key)), size);
             if attr_list_sizes <> nil then
                attr_list_sizes.Add(key,size);
         end;
      TF_AttrType.TF_ATTR_FLOAT:
         begin
             var size := value.AsType<Single>;
             TFE_OpSetAttrFloat(op, PAnsiChar(AnsiString(key)), size);
         end;
      TF_AttrType.TF_ATTR_SHAPE:
         begin
            var dims := value.AsType<  TArray<Int64>>;

            var pInt64Value : PInt64 := nil;
            if Length(dims) > 0 then pInt64Value := @dims[0];

            TFE_OpSetAttrShape(op, PAnsiChar(AnsiString(key)), pInt64Value, Length(dims), status.Handle);
            status.RaiseEx;
         end;
      TF_AttrType.TF_ATTR_FUNC:
         begin
             if value.IsType<ConcreteFunction> then
             begin
                 var func := value.AsType<ConcreteFunction>;
                 TFE_OpSetAttrFunctionName(op, PAnsiChar(AnsiString(key)), PAnsiChar(AnsiString(func.Name)), func.Name.Length);
             end else
                 raise Exception.Create('Not Implemented, SetOpAttrScalar for'+ TEnum.GetName(tipo));

         end;
      else
         raise Exception.Create('Not Implemented, SetOpAttrScalar for'+ TEnum.GetName(tipo));
    end;
    Result := true;
end;

function TEagerRunner.GetOp(ctx: TContext; op_or_function_name: AnsiString; status: TFStatus): PTFE_Op;
var
  op : PTFE_Op;
begin
    if thread_local_eager_operation_map.ContainsKey( string(op_or_function_name)) then
    begin
        op := thread_local_eager_operation_map[ op_or_function_name ];
        TFE_OpReset(op, PAnsiChar(op_or_function_name), PAnsiChar(AnsiString(ctx.DeviceName)), status.Handle);
    end else
    begin
        op := TFE_NewOp(ctx.Handle_, PAnsiChar(op_or_function_name), status.Handle);
        thread_local_eager_operation_map.Add(op_or_function_name,op);
    end;
    status.RaiseEx;
    Result := op;
end;

function TEagerRunner.TFE_Execute(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>; attrs: TArray<TValue>;
  num_outputs: Integer): TArray<TFTensor>;
begin
    Result := TFE_ExecuteCancelable(ctx, device_name, op_name, inputs, attrs, num_outputs);
end;

function TEagerRunner.TFE_ExecuteCancelable(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>;
  attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
begin
    var status := tf.Status;

    var op := GetOp(ctx, op_name, status);
    TFE_OpSetDevice(op, PAnsiChar(device_name), status.Handle);
    if status.Ok then
    begin
        for var i := 0 to Length(inputs) - 1 do
        begin
            var tensor_handle : PTFE_TensorHandle := inputs[i].EagerTensorHandle ;
            if tensor_handle = nil then
              raise Exception.Create('Eager tensor handle has not been allocated.');
            TFE_OpAddInput(op, tensor_handle, status.Handle);
            status.RaiseEx;
        end;
    end;
    if (status.ok) and (Length(attrs) > 0 ) then
        SetOpAttrs(op, attrs);
    var outputs : TArray<PTFE_Op>;
    SetLength( outputs,num_outputs);
    var pOutputs : PTFE_TensorHandle := nil;
    if num_outputs > 0 then
        pOutputs := @outputs[0];
    if status.ok then
    begin
        TF4D.Core.CApiEager.TFE_Execute(op, pOutputs, num_outputs, status.Handle);
        status.RaiseEx;
    end;
    Result := [];
    for var i := 0 to num_outputs - 1 do
       Result := Result + [ TEagerTensor.Create( outputs[i]) ]
end;

procedure TEagerRunner.SetOpAttrWithDefaults(ctx: TContext; op: PTFE_Op; attr: TAttrDef; attr_name: string; attr_value: TValue; attr_list_sizes: TDictionary<string, Int64>;status: TFStatus) ;
begin
    var  is_list : Byte := 0;
    var tipo := TFE_OpGetAttrType(op, PAnsiChar(AnsiString(attr_name)), @is_list, status.Handle);
    if not status.Ok then Exit;

    if attr_value.Kind = tkUnknown then
    begin

    end else
    begin
        if is_list <> 0 then
            SetOpAttrList  (ctx, op, PAnsiChar(AnsiString(attr_name)), attr_value, tipo, attr_list_sizes, status)
        else
            SetOpAttrScalar(ctx, op, PAnsiChar(AnsiString(attr_name)), attr_value, tipo, attr_list_sizes, status);
    end;
end;

function TEagerRunner.ShouldRecord(inputs: TArray<TFTensor>): Boolean;
begin
    var should_record : Boolean := false;
    for var tape in tf.GetTapeSet do
    begin
        if tape.ShouldRecord(inputs) then
        begin
            should_record := true;
            break;
        end;
    end;
    Result := should_record;
end;

function  TEagerRunner.AddInputToOp(inputs: TValue; add_type_attr: Boolean; input_arg: TArgDef;flattened_attrs: TList<TValue>; flattened_inputs: TList<TFTensor>; op: PTFE_Op;status: TFStatus): Boolean;
begin
    var tensor := tf.convert_to_tensor(inputs);
    flattened_inputs.Add(tensor);
    if (add_type_attr) and ( not string.IsNullOrEmpty(input_arg.TypeAttr )) then
    begin
        var dtype := tensor.dtype;
        TFE_OpSetAttrType(op, PAnsiChar(AnsiString(input_arg.TypeAttr)), dtype);
        flattened_attrs.Add(input_arg.TypeAttr);
        flattened_attrs.Add( TValue.From<Integer>( Ord(dtype)) );
    end;
    TFE_OpAddInput(op, tensor.EagerTensorHandle, status.Handle);
    status.RaiseEx;
    Result := true;
end;

function TEagerRunner.TFE_FastPathExecute(op_exec_info: TFastPathOpExecInfo): TArray<TFTensor>;

   function FindAttr(a: TList<TAttrDef>; sName : AnsiString): TAttrDef;
   var
     i : Integer;
   begin
       Result := nil;
       for i := 0 to a.Count-1 do
       begin
           if AnsiString(a[i].Name) = sName then
             Exit( a[i] );
       end;
   end;

begin
    Result := [];
    if op_exec_info.ctx = nil then
        op_exec_info.ctx := tf.Context;

    if string.IsNullOrEmpty(op_exec_info.device_name) then
        op_exec_info.device_name := tf.Context.DeviceName;

    var attr_list_sizes := TDictionary<string, Int64>.Create;

    op_exec_info.run_gradient_callback   := HasAccumulatorOrTape;
    op_exec_info.run_post_exec_callbacks := op_exec_info.callbacks <> nil;
    op_exec_info.run_callbacks           := op_exec_info.run_gradient_callback or op_exec_info.run_post_exec_callbacks;

    var status := tf.Status;
    var op     := GetOp(op_exec_info.ctx, op_exec_info.op_name, status);
    var op_def := tf.get_default_graph.GetOpDef(op_exec_info.op_name);

    var flattened_attrs      := TList<TValue>.Create;
    flattened_attrs.Capacity := op_def.Attrs.Count * 2;

    var flattened_inputs      := TList<TFTensor>.Create;
    flattened_inputs.Capacity := op_def.InputArgs.Count;

    // Set non-inferred attrs, including setting defaults if the attr is passed in
    // as None.
    if op_exec_info.attrs <> nil then
    begin
        for var attr1 in op_exec_info.attrs do
        begin                     ;
            var attr := FindAttr(op_def.Attrs, attr1.Key);
            if attr <> nil then
            begin
                flattened_attrs.Add(attr.Name);
                flattened_attrs.Add(attr1.Value);
                SetOpAttrWithDefaults(op_exec_info.ctx, op, attr, attr.Name, attr1.Value, attr_list_sizes, status);
                status.RaiseEx;
            end
        end;
    end;
    // c_api.TFE_OpSetDevice(op, op_exec_info.device_name, status.Handle);
    // status.Check(true);
    // Add inferred attrs and inputs.
    for var i := 0 to op_def.InputArgs.Count - 1 do
    begin
        var input     := op_exec_info.args[i];
        var input_arg := op_def.InputArgs[i];
        if not string.IsNullOrEmpty(input_arg.NumberAttr) then
        begin
            var len : Int64 := input.GetArrayLength ;
            TFE_OpSetAttrInt(op, PAnsiChar(AnsiString( input_arg.NumberAttr )), len);
            if op_exec_info.run_callbacks then
            begin
                flattened_attrs.Add(input_arg.NumberAttr);
                flattened_attrs.Add(len);
            end;
            attr_list_sizes.Add(input_arg.NumberAttr,len);
            if len > 0 then
            begin
                var fast_input_array := op_exec_info.args[i] ;
                var rr := fast_input_array.GetArrayLength;
                //var x : TArray<TFTensor> := fast_input_array.AsType<TArray<TFTensor>>;
                // First item adds the type attr.
                if not AddInputToOp(fast_input_array.GetArrayElement(i), true, input_arg, flattened_attrs, flattened_inputs, op, status) then
                    Exit;
                for var j := 1 to len-1 do
                begin
                    // Since the list is homogeneous, we don't need to re-add the attr.
                    if not AddInputToOp(fast_input_array.GetArrayElement(j), false, input_arg, flattened_attrs, flattened_inputs, op, status) then
                        Exit;
                end;
            end;
        end
        else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
        begin
            var attr_name        := input_arg.TypeListAttr;
            var fast_input_array := input;
            var len              := fast_input_array.GetArrayLength;
            var attr_values : TArray<TF_DataType>; SetLength(attr_values,len);
            for var j := 0 to len-1 do
            begin
                var eager_tensor := TOps.convert_to_tensor(fast_input_array.GetArrayElement(j));
                attr_values[j] := eager_tensor.dtype;
                TFE_OpAddInput(op, eager_tensor.EagerTensorHandle, status.Handle);
                if op_exec_info.run_callbacks then
                   flattened_inputs.Add(eager_tensor);
            end;
            if op_exec_info.run_callbacks then
            begin
                flattened_attrs.Add(attr_name);
                flattened_attrs.Add( TValue.From< TArray<Integer> >( Tdtypes.ToIntArray(attr_values) ));
            end;
            var pDatatypes : PTF_DataType := nil;
            if Length(attr_values) > 0  then  pDatatypes := @attr_values[0];

            TFE_OpSetAttrTypeList(op, PAnsiChar(AnsiString(attr_name)), pDatatypes, Length(attr_values));
            attr_list_sizes.Add(attr_name, len);
        end else
        begin
            // The item is a single item.
            AddInputToOp(op_exec_info.args[i], true, input_arg, flattened_attrs, flattened_inputs, op, status);
        end;
    end;
    var num_retvals : Integer := 0;
    for var i := 0 to op_def.OutputArgs.Count - 1 do
    begin
        var output_arg := op_def.OutputArgs[i];
        var delta : Int64 := 1;
        if not string.IsNullOrEmpty(output_arg.NumberAttr) then
            delta := attr_list_sizes[output_arg.NumberAttr]
        else if not string.IsNullOrEmpty(output_arg.TypeListAttr) then
            delta := attr_list_sizes[output_arg.TypeListAttr];
        if delta < 0  then
          raise Exception.Create('Attributes suggest that the size of an output list is less than 0');
        num_retvals := num_retvals + Integer(delta);
    end;

    var retVals : TArray<PTFE_Op> ;
    SetLength(retVals,num_retvals);

    var pRetVals : Pointer := nil;
    if Length(retVals) > 0 then pRetVals := @retVals[0];

    TF4D.Core.CApiEager.TFE_Execute(op, pRetVals, num_retvals, status.Handle);
    status.RaiseEx;

    var flat_result : TArray<TFTensor> ;
    for var i := 0 to num_retvals - 1 do
       flat_result := flat_result + [ TEagerTensor.Create( retVals[i] ) ];
    if op_exec_info.run_callbacks then
    begin
        RunCallbacks(op_exec_info, op_def.InputArgs.Count,flattened_inputs.ToArray, flattened_attrs.ToArray, flat_result);
    end;
    Result := flat_result;
end;

function TEagerRunner.ArgsToMatchingEager(ctx: TContext; default_dtype: TF_DataType; args: TArray<TValue>):  Tuple<TF_DataType, TArray<TFTensor>>;
begin
    var res : Tuple<TF_DataType, TArray<TFTensor>>;

    var eArgs := Enumerable<TValue>.Create(args) ;
    var predicateCount :  TPredicate<TValue> := function(const x: TValue): Boolean
                             begin
                                 Result := x.IsType<TFTensor>;
                             end;

    if (Length(Args) = 0) and (default_dtype <> TF_DataType.DtInvalid) then
    begin
        Res := Tuple<TF_DataType, TArray<TFTensor>>.create(default_dtype, nil);
    end;

    if (eArgs.Count( predicateCount ) = Length(args)) then
    begin
        var sel := eArgs.Select<TFTensor>(function(x: TValue): TFTensor
                                 begin
                                     Result := x.AsType<TFTensor>;
                                 end).ToArray;

        res := Tuple<TF_DataType, TArray<TFTensor>>.create(sel[0].Dtype, sel);
        Result := res;
        exit;
    end;

    var dtype := TF_DataType.DtInvalid;
    for var x in args do
    begin
        if x.IsType<TFTensor> then
            dtype := x.AsType<TFTensor>.dtype;
    end;
    if dtype = TF_DataType.DtInvalid then
    begin
        var ret := TList<TFTensor>.Create;
        for var t in args do
        begin
            ret.Add(Tops.convert_to_tensor(t, dtype, '',False,default_dtype, ctx));
            if dtype = TF_DataType.DtInvalid then
                dtype := ret.Last.dtype;
        end;
        res := Tuple<TF_DataType, TArray<TFTensor>>.create(dtype, ret.ToArray) ;
        Result := res;
        exit;
    end else
    begin
        raise Exception.Create('Not Implemented');
    end;
end;

function TEagerRunner.HasAccumulator: Boolean;
begin
    Result := false;
end;

function TEagerRunner.HasGradientTape: Boolean;
begin
   Result := tf.GetTapeSet.Count > 0;
end;

function TEagerRunner.HasAccumulatorOrTape: Boolean;
begin
    Result := HasGradientTape or HasAccumulator;
end;

function TEagerRunner.MakeTensorList(tensors: TArray<TFTensor>): TArray<TFTensor>;
Begin
    Result := tensors;
end;

function TEagerRunner.MustRecordGradient: Boolean;
begin
    Result := HasGradientTape;
end;

function TEagerRunner.TFE_TapeGradient(tape: ITape; target: TArray<TFTensor>; sources: TArray<TFTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;
begin
    var target_vec  := target;
    var sources_vec := sources;
    var sources_set := sources_vec;
    var seq_array := target;
    var source_tensors_that_are_targets := TDictionary<TFTensor, TapeTensor>.Create;
    for var i := 0 to Length(target) - 1 do
    begin
        source_tensors_that_are_targets.Add(target_vec[i], TapeTensor.Create( seq_array[i]) );
    end;
    if output_gradients <> nil then
        raise Exception.Create('Not Implemented')
    else
        output_gradients :=  [];
    var outgrad_vec := MakeTensorList(output_gradients);
    Result := tape.ComputeGradient(target_vec, sources_vec, source_tensors_that_are_targets, outgrad_vec);
end;
{ TFastPathOpExecInfo }

constructor TFastPathOpExecInfo.Create(opName, name: string; inputArgs: TArray<TValue>);
begin
    op_name := opName;
    name    := name;
    args    := inputArgs;
end;

{ TExecute }

class function TExecute.convert_to_mixed_eager_tensors(values: TArray<TFTensor>; ctx: TContext): Tuple<TArray<TDataType>, TArray<TFTensor>>;
begin            //(value: TValue; dtype : TF_DataType = DtInvalid; name: string= ''; as_ref: Boolean = False; preferred_dtype : TF_DataType = DtInvalid; ctx: TContext= nil)
    var v : TArray<TFTensor> ;
    for var i := 0 to  Length(values) - 1 do
      v := v + [ Tops.convert_to_tensor(values[i], DtInvalid, '', False, DtInvalid, ctx) ];

    var types : TArray<TDataType>;
    for var i := 0 to  Length(v) - 1 do
      types := types + [ TDTypes.as_datatype_enum(v[i].Dtype) ];

    Result := Tuple.Create(types, v);
end;

class function TExecute.quick_execute(op_name: string; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>; ctx: TContext;
  name: string): TArray<TFTensor>;
begin
    var device_name : string := ctx.DeviceName;

    ctx.ensure_initialized;
    var tensors := tf.Runner.TFE_Execute(ctx, device_name, op_name, inputs, attrs, num_outputs);

    Result := tensors;
end;

class function TExecute.must_record_gradient: Boolean;
begin
    Result := false;
end;
{$ENDREGION}

{$REGION 'Framework'}
{ op_def_registry }

class procedure op_def_registry.FreeDictionary;
var
  LItem: TPair<string, TOpDef>;
begin
    if Assigned(registered_ops)  then
    begin
       for LItem in registered_ops do
       begin
         LItem.Value.Free;
       end;
       registered_ops.Clear;
       registered_ops.Free;
       registered_ops := nil;
    end;

end;

class function op_def_registry.GetOpDef(tipo: string): TOpDef;
begin
    var ops := get_registered_ops;
    Result  := ops[tipo];
end;

class function op_def_registry.get_registered_ops: TDictionary<string, TOpDef>;
var
  Loader: TpbLoader;

begin
    if not Assigned(registered_ops)  then
       registered_ops := TDictionary<string, TOpDef>.Create;

    // double validation to avoid multi-thread executing
    if registered_ops.Count > 0 then
        Exit(registered_ops);

    var buffer := TFBuffer.Create( TF_GetAllOpList );
    var op_list : TOpList;

    var aBuf := buffer.toArray;
    Loader.Init;
    Loader.Pb.Init(@aBuf[0],Length(aBuf),false);

    Loader.LoadOpList(op_list);

    for var i := 0 to op_list.Ops.Count - 1 do
    begin
       var op_def : TOpDef := op_list.Ops[i];
       registered_ops.AddOrSetValue(op_def.Name,op_def);
    end;

    Result := registered_ops
end;

{ common_shapes }

class function common_shapes.broadcast_shape(shape_x, shape_y: TFTEnsor): TFtensor;
begin

    var return_dims := _broadcast_shape_helper(shape_x, shape_y);
    // return tensor_shape(return_dims);
    raise TFException.Create('Not Finite NumberException');
end;

class function common_shapes._broadcast_shape_helper(shape_x, shape_y: TFTEnsor): TFtensor;
begin
    raise TFException.Create('Not Finite NumberException');
end;

class function common_shapes.has_fully_defined_shape(tensor: TFTensor): Boolean;
begin
   Result := tensor.shape.IsFullyDefined;
end;

class function common_shapes.rank(tensor: TFTensor): Integer;
begin
   Result := tensor.rank;
end;

{ random_seed }

class function random_seed.get_seed(op_seed: TNullableInteger): Tuple<TNullableInteger, TNullableInteger>;
var
 seed: Integer;
begin
    var global_seed: Nullable<Integer>;

    if tf.executing_eagerly then
        global_seed := tf.Context.global_seed
    else
        global_seed := Tops.get_default_graph.seed;
    if global_seed.HasValue then
    begin
        if  not op_seed.HasValue then
        begin
            if tf.executing_eagerly then
            begin
                op_seed := tf.Context.internal_operation_seed;
            end
            else begin
                 if  not Fgraph_to_seed_dict.TryGetValue(Tops.get_default_graph.graph_key, seed) then
                    seed := 0;
                 Fgraph_to_seed_dict.AddOrSetValue(Tops.get_default_graph.graph_key, seed + 1);
                 op_seed := seed;
            end;
        end;
        Result := Tuple<TNullableInteger, TNullableInteger>.Create(global_seed, op_seed);
        Exit;
    end;
    if op_seed <> nil then  Result :=  Tuple<TNullableInteger, TNullableInteger>.Create(DEFAULT_GRAPH_SEED, op_seed)
    else                    Result :=  Tuple<TNullableInteger, TNullableInteger>.Create(nil, nil)
end;

class function random_seed.get_seed_tensor(op_seed: TNullableInteger): Tuple<TFTensor, TFTensor>;
begin
    var tseed := get_seed(op_seed);
    var seed := tseed.Value1;
    var seed2:= tseed.Value2;

    var _seed, _seed2 : TFTensor;
    if seed = nil then  _seed := constant_op.constant(Int64(0), DtInvalid, 'seed')
    else                _seed := constant_op.constant(Int64(seed.Value), DtInvalid, 'seed');
    if seed2 = nil then
        _seed2 := constant_op.constant(Int64(0), DtInvalid, 'seed2')
    else begin
        _seed2 := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('seed2'),
                          function(v1: TNameScope): TFTensor
                            begin
                                _seed2 := constant_op.constant(Int64(seed2.Value));
                                Result :=  array_ops.where_v2(
                                  math_ops.logical_and(
                                      math_ops.equal(_seed, Int64(0)),
                                      math_ops.equal(_seed2, Int64(0))),
                                  constant_op.constant( Power(2,31) - 1),
                                  _seed2,
                                  v1.ToString);
                            end );
    end;
    Result := Tuple<TFTensor, TFTensor>.Create(_seed, _seed2);
end;

{ IndexedSlices }

constructor IndexedSlices.Create(_values, _indices, _dense_shape: TFTensor);
begin
    Fvalues     := _values;
    Findices    := _indices;
    Fdense_shape:= _dense_shape;
    Fvalues.Tag := TValue.From<IndexedSlices>(Self);
end;

function IndexedSlices.GetDevice: string;
begin
    Result := Fvalues.Device;
end;

function IndexedSlices.GetDtype: TF_DataType;
begin
    Result := Fvalues.Dtype;
end;

function IndexedSlices.GetGraph: TFGraph;
begin
    Result := Fvalues.graph;
end;

function IndexedSlices.GetName: string;
begin
    Result := Fvalues.Name;
end;

function IndexedSlices.GetOp: TFOperation;
begin
    Result := Fvalues.Op;
end;

class operator IndexedSlices.implicit(iSlices: IndexedSlices): TFTensor;
begin
     Result := iSlices.values;
end;

class operator IndexedSlices.implicit(tTEnsor: TFTensor): IndexedSlices;
begin
    Result := tTEnsor.Tag.AsType<IndexedSlices>;
end;

{ smart_module }

class function smart_module.smart_cond(_pred: Boolean; true_fn, false_fn: TFunc<TFTensor>; name: string): TFTensor;
begin
    if _pred then Result := true_fn
    else          Result := false_fn;
end;

class function smart_module.smart_cond(_pred: TFTensor; true_fn, false_fn: TFunc<TArray<TFTensor>>; name: string): TArray<TFTensor>;
var
  pred_value :Nullable<Boolean>;
begin
    pred_value := smart_module.smart_constant_value(_pred);
    if pred_value.HasValue then
    begin
        var res : TArray<TFTensor>;
        if pred_value.Value then res := true_fn
        else                     res := false_fn;
        Result := res;
    end else
    begin
        Result := control_flow_ops.cond<TFTensor>(_pred, true_fn, false_fn, name);
    end;
end;

class function smart_module.smart_constant_value(_pred: TFTensor): Nullable<Boolean>;
begin
    var pred_value := TUtils.constant_value(_pred);
    if pred_value = nil then
    begin
        var res : TArray<Pointer> ;
        SetLength(res, _pred.op.NumOutputs) ;

        var evaluated := TF_TryEvaluateConstant(_pred.graph.handle, _pred._as_tf_output, @res[0], tf.Status.Handle);
        if (evaluated= 0) or (TF_GetCode(tf.Status.Handle) <> TF_Code.TF_OK)  then
            Exit(nil)
        else
            raise TFException.Create('Not Implemented');
    end;
    var res : NDArray := pred_value;
    var b   : Boolean := res;
    Result := b;
end;

{ DenseSpec }

constructor DenseSpec.Create(_shape: TFShape; _dtype: TF_DataType; _name: string);
begin
    Fshape := _shape;
    Fdtype := _dtype;
    Fname  := _name;
end;

function DenseSpec.ToString: string;
begin
   Result := Format('shape=%s, dtype=%s, name=%s',[Fshape.ToString, Tdtypes.ToString(Fdtype), Fname] )
end;

{ TensorSpec }

constructor TensorSpec.Create(shape: TFShape; dtype: TF_DataType; name: string);
begin
    inherited Create(shape, dtype, name);
end;

function TensorSpec._batch(dim: Integer): TensorSpec;
begin
    var shapes := shape.dims;
    shapes := [dim] + shapes ;
    Result := TensorSpec.Create(shapes, Fdtype);
end;

function TensorSpec._unbatch: TensorSpec;
begin
    if Fshape.ndim = 0 then
       raise Exception.Create('Unbatching a tensor is only supported for rank >= 1');
    var a  := Fshape.dims;
    Delete(a,0,1) ;
    Result := TensorSpec.Create(a, Fdtype);
end;
{$ENDREGION}

{$REGION 'Graph'}
{ DefaultGraphStack }

constructor DefaultGraphStack.Create;
begin
    inherited Create;
    F_stack := TStack<TFGraph>.Create;
end;

destructor DefaultGraphStack.Destroy;
begin
  F_stack.Clear;
  F_stack.Free;
  inherited Destroy;
end;

function DefaultGraphStack.get_controller(g: TFGraph): TFGraph;
begin
    F_stack.Push(g);
    Result := g;
end;

function DefaultGraphStack.get_default: TFGraph;
begin
    if      F_stack.Count > 0             then  Exit(F_stack.Peek)
    else if F_global_default_graph <> nil then  Exit(F_global_default_graph)
    else                                        F_global_default_graph := TFGraph.Create;
    Result := F_global_default_graph;
end;

function DefaultGraphStack.peak_controller: TFGraph;
begin
    if F_stack.Count = 0 then
        Exit(nil);
    Result := F_stack.Peek;
end;

procedure DefaultGraphStack.pop;
begin
    F_stack.Pop
end;

procedure DefaultGraphStack.reset;
begin
    F_stack.Clear;
    F_global_default_graph.Destroy;
end;

{ TFuncGraph }

constructor TFuncGraph.Create(name: string);
begin
   inherited Create;

   Inputs     := TFTensors.Create;
   Outputs    := TFTensors.Create;
   F_captures := TDictionary<Int64, Tuple<TFTensor, TFTensor> >.Create;

   Fouter_graph := Tops.get_default_graph;

   while Fouter_graph.building_function do
     Fouter_graph := Fouter_graph.OuterGraph;

   Fgraph_key := name;
   Fbuilding_function := true;
end;

constructor TFuncGraph.Create(_handle: Pointer; name: string; _attrs: TDictionary<string, string>);
begin
   inherited Create;

   Fouter_graph := Tops.get_default_graph;

   while Fouter_graph.building_function do
       Fouter_graph := Fouter_graph.OuterGraph;

   Fgraph_key := name;
   Fbuilding_function := true;
   Attrs := attrs;
   // Will to test if FuncGraph has memory leak
   TF_DeleteGraph(Handle);
   Handle := _handle
end;

procedure TFuncGraph.NativeDispose(hnd: Pointer);
begin
  TFE_ContextRemoveFunction(tf.Context.Handle_,PAnsiChar(AnsiString (Fgraph_key) ), tf.Status.Handle);
  TF_DeleteFunction(F_func_graph_handle);

  inherited NativeDispose(hnd);
end;

procedure TFuncGraph.add_capture(tensor, placeholder: TFTensor);
begin
    F_captures.AddOrSetValue(tensor.Id, Tuple.Create(tensor, placeholder));
    Inputs.Add(placeholder);
end;

function TFuncGraph.as_default: TFGraph;
begin
    tf.Context.graph_mode(True);
    Tops.set_default_graph(self);
    Result := self;
end;

function TFuncGraph.capture(tensor: TFTensor; name: string; shape: PTFShape): TFTensor;
begin
    if tensor is TEagerTensor then
    begin
        if name = '' then
            name := Tops.uid.ToString;

        // Small EagerTensors are captured with Const ops
        if (TDtypes.is_value_dtype(tensor.dtype)) and ((tensor.rank = 0) or (tensor.size < EAGER_CONST_THRESHOLD)) then
            Exit(capture_eager_tensor(tensor, name) );

        // Large EagerTensors and resources are captured with Placeholder ops
        Result := _capture_helper(tensor, name, shape);
        Exit;
    end;

    if tensor.graph <> Self then
    begin
        if name = '' then
            name := tensor.op.name;
        var inner_graph := tensor.graph;
        while(inner_graph <> nil) and (inner_graph is TFuncGraph ) do
        begin
            var inner_func_graph := inner_graph as TFuncGraph;
            if inner_graph = Self then
               raise Exception.Create('The tensor '+tensor.name+' cannot be accessed here: it is defined' +
                    ' in another function or code block. Use return values,' +
                    ' explicit Python locals or TensorFlow collections to access' +
                    ' it. Defined in: '+tensor.graph.graph_key+'; accessed from: '+graph_key+'.');
            inner_graph := inner_func_graph.Fouter_graph;
        end;
        Result := _capture_helper(tensor, name);
        Exit;
    end;

    Result := tensor;
end;

function TFuncGraph.capture_eager_tensor(tensor: TFTensor; name: string): TFTensor;
var
  graph_const : TFTensor;
begin
    if not F_captures.ContainsKey(tensor.Id) then
    begin
        graph_const := TUtils.tf_with<TControlDependenciesController, TFTensor>(Tops.control_dependencies([]),
                           function(ctl : TControlDependenciesController) : TFTensor
                            begin
                               var sShape := tensor.shape;
                               Result := constant_op.constant(tensor.numpy, tensor.dtype, @sShape, False, True, name);
                            end);

        add_capture(tensor, graph_const);
    end else
    begin
        graph_const := F_captures[tensor.Id].Value2;
    end;

    var _backward_function_wrapper : BackwardFunction := function(output_grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
                                                            begin
                                                                Result := output_grads;
                                                            end;

    tf.Runner.RecordGradient('captured_value', [ graph_const ], nil, [ tensor ], _backward_function_wrapper (*getForwardFunction: forward_function*));
    Result := graph_const;
end;

function TFuncGraph.Create_op(op_type: TF_TString; inputs: TArray<TFTensor>; dtypes, input_types: TArray<TF_DataType>; Name: TF_TString; attrs: TDictionary<string, TAttrValue>;
  op_def: TOpDef; compute_device: Boolean): TFOperation;
begin
    for var i:= 0 to Length(inputs)-1 do
    begin
        inputs[i] := capture(inputs[i]);
    end;

    Result := inherited create_op(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device);
end;

function TFuncGraph.getCaptures: TArray<Tuple<TFTensor, TFTensor>>;
begin
    Result := [];
    for var item in F_captures.Values do
      Result := Result + [ Item ]
end;

function TFuncGraph.getCapture_Inputs: TArray<TFTensor>;
begin
    Result := external_captures
end;

function TFuncGraph.getExCapture: TArray<TFTensor>;
begin
    Result := [];
    for var item in F_captures.Values do
      Result := Result + [ Item.Value1 ]
end;

function TFuncGraph.InterCaptures: TArray<TFTensor>;
begin
    Result := [];
    for var item in F_captures.Values do
      Result := Result + [ Item.Value2 ]
end;

function TFuncGraph.getFuncName: string;
begin
    Result := Fgraph_key;
end;

procedure TFuncGraph.gExit;
begin
  tf.Context.restore_mode;
  Tops.pop_graph;
end;

function TFuncGraph._capture_helper(tensor: TFTensor; name: string; shape: PTFShape): TFTensor;
var
  placeholder : TFTensor;
begin
    if not F_captures.ContainsKey(tensor.Id) then
    begin
        placeholder := _create_substitute_placeholder(tensor, name, tensor.dtype, shape);
        add_capture(tensor, placeholder);
    end else
    begin
        placeholder := F_captures[tensor.Id].Value2;
    end;

    var _backward_function_wrapper : BackwardFunction := function(output_grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
                                                            begin
                                                                Result := output_grads;
                                                            end;

    tf.Runner.RecordGradient('captured_value', [ placeholder ], nil, [ tensor ], _backward_function_wrapper (*getForwardFunction: forward_function*));
    Result := placeholder;
end;

procedure TFuncGraph.SetAttrs;
var
  serialized : TAttrValue;
  v          : TpbOneof;
  S          : TpbSaver;
  bytes      : TBytes ;
begin
    if Attrs = nil then
        Exit;

    for var item in Attrs do
    begin
        var _name      := Item.Key;
        var attr_value := Item.Value;
        serialized := TAttrValue.Create;

        v.tag := TAttrValue.ftS;
        v.value := TValue.From<TBytes>( TEncoding.UTF8.GetBytes(attr_value) );
        serialized.value := v;

        S.Init;
        bytes := [];
        TpbSaver.SaveAttrValue(S,serialized);
        bytes:= s.Pb.GetBytes;
        var Len : NativeInt := Length(bytes);

        TF_FunctionSetAttrValueProto(F_func_graph_handle, PAnsiChar(AnsiString(_name)), @bytes[0], Len, tf.Status.Handle);
        tf.Status.RaiseEx;
    end;
end;

procedure TFuncGraph.ToGraph(opers: TArray<TFOperation>; _inputs, _outputs: TArray<TFTEnsor>; output_names: TArray<string>);
var
  Status : TFStatus;
  pOper  : PPTF_Operation ;
  aOperz : TArray<PTF_Operation>;
  pInput : TArray<TF_Output>;
  pOutput: TArray<TF_Output>;
  aNames : TArray<PAnsiChar>;
  pNames : PPAnsiChar;
begin
    Status := TFStatus.Create;

    pOper  := nil;
    aOperz := [];
    for var i := 0 to Length(opers) - 1 do
      aOperz := aOperz + [ opers[i].Handle ];
    if Length(aOperz) > 0 then  pOper := @aOperz[0];

    for var i := 0 to Length(_inputs) - 1 do
      pInput := pInput + [ TF_Output.Create(_inputs[i].Op.Handle,0) ];

    for var i := 0 to Length(_outputs) - 1 do
      pOutput := pOutput + [ TF_Output.Create(_outputs[i].Op.Handle,0) ];

    pNames := nil;
    for var i := 0 to Length(output_names) - 1 do
      aNames := aNames + [ PAnsiChar(AnsiString(output_names[i])) ];
    if Length(aNames) > 0  then pNames := @aNames[0];

    F_func_graph_handle := TF_GraphToFunction(Handle,
        PAnsiChar(AnsiString(Fgraph_key)),
        0,
        Length(opers),pOper,
        Length(pInput), @(pInput[0]),
        Length(pOutput), @(pOutput[0]),
        pNames,
        nil,
        nil,
        status.Handle);

    status.RaiseEx;

    SetAttrs;

    TFE_ContextAddFunction(tf.Context.Handle_, F_func_graph_handle, status.Handle);
    status.RaiseEx;

    Fgraph_key := string(AnsiString(TF_FunctionName(F_func_graph_handle)));

    Inputs := TFTensors.Create(_inputs);
    // mark_as_return
    Outputs := TFTensors.Create(_outputs);// .Select(x => array_ops.identity(x)).ToArray();
end;

function TFuncGraph._create_substitute_placeholder(value: TFTensor; name: string; dtype: TF_DataType; shape: PTFShape): TFTensor;
var
  sShape : TFShape;
begin
    if shape = nil then sShape := value.shape
    else                sShape := shape^;

    if dtype = TF_DataType.DtInvalid then
        dtype := value.dtype;

    var placeholder := TUtils.tf_with<TControlDependenciesController, TFTensor>(Tops.control_dependencies([]),
                           function(ctl : TControlDependenciesController) : TFTensor
                            begin
                                Result := array_ops.placeholder(dtype, @sShape, name);
                            end);
    // custom_gradient.copy_handle_data(value, placeholder)
    Result := placeholder;
end;

{ SubGraphUtility }

class function SubGraphUtility.lift_to_graph(init_tensors: TFTensors; graph: TFuncGraph; sources: TList<TFTensor>; add_sources, handle_captures: Boolean; base_graph: TFGraph;
  op_map: TDictionary<ITensorOrOperation, TFOperation>): TDictionary<ITensorOrOperation, TFOperation>;
var
  visited_ops, ops_to_copy,
  marked_ops,unvisited_ops  : TList<TFOperation>;
  ops_to_visit              : TStack<TFOperation>;
  src                       : TList<TFTensor>;
begin
    if base_graph = nil then
      base_graph := init_tensors[0].graph;

    if op_map = nil then
      op_map  := TDictionary<ITensorOrOperation, TFOperation>.Create;

    visited_ops := TList<TFOperation>.Create;
    for var i := 0 to sources.Count - 1 do
      visited_ops.Add( sources[i].Op );

    var aOper := TList<TFOperation>.Create ;
    for var init_tensor in init_tensors do
    begin
        src := map_subgraph(init_tensor, sources, visited_ops, add_sources);
        sources.AddRange(src);

        aOper.Add(init_tensor.Op);
    end;

    ops_to_copy := TList<TFOperation>.Create;
    marked_ops  := TList<TFOperation>.Create;

    ops_to_visit  := TStack<TFOperation>.Create(aOper);
    unvisited_ops := TList<TFOperation>(ops_to_visit);
    while unvisited_ops.Count > 0 do
    begin
        while ops_to_visit.Count > 0 do
        begin
            var op := ops_to_visit.Pop;
            if marked_ops.Contains(op) then
                continue;
            marked_ops.Add(op);
            ops_to_copy.Add(op);
            for var inp in op.inputs do
            begin

            end;
        end;
        // difference_update
        TUtils.difference_update<TFOperation>(unvisited_ops,marked_ops);
        if unvisited_ops.Count > 0 then
            ops_to_visit.Push(unvisited_ops.Last);
    end;

    // When lifting from one FuncGraph to another, we will need to capture the
    // relevant tensors as well.
    var inverse_captures := TDictionary<TFTensor, TFTensor>.Create;
    var internal_captures : TArray<TFTensor> := nil;
    if base_graph is TFuncGraph then
    begin
        var base_func_graph := base_graph as TFuncGraph;
        var captures := base_func_graph.captures;

        for var item in captures  do
        begin
            var external_capture := item.Value1;
            var internal_capture := item.Value2;
            inverse_captures.AddOrSetValue(internal_capture, external_capture);
        end;

        internal_captures := base_func_graph.internal_captures;
    end;

    graph.as_default;
    var source_ops := TList<TFOperation>.Create;
    // Add the sources in the same order as the original graph.
    for var s in internal_captures do
    begin
        if sources.Contains(s) then
        begin
            sources.Remove(s);
            source_ops.Add(s.op);
            _copy_source(s, graph, op_map, handle_captures, inverse_captures, base_graph);
        end;
    end;

    for var op in TUtils.reversed<TFOperation>(ops_to_copy) do
    begin
        if (source_ops.Contains(op)) or (op_map.ContainsKey(op)) then
            continue;
        _copy_non_source(op, graph, op_map, base_graph);
    end;

    graph.gExit;

    Result := op_map;
end;

class function SubGraphUtility.map_subgraph(init_tensor: TFTensor; sources: TList<TFTensor>; visited_ops: TList<TFOperation>; add_sources: Boolean): TList<TFTensor>;
var
  ops_to_visit : TStack<TFOperation>;
  extra_sources: TList<TFTensor>;
begin
    ops_to_visit := TStack<TFOperation>.Create;
    ops_to_visit.Push(init_tensor.op);
    extra_sources := TList<TFTensor>.Create;
    while ops_to_visit.Count > 0 do
    begin
        var op := ops_to_visit.Pop;
        if visited_ops.Contains(op) then
            continue;
        visited_ops.Add(op);
        var should_raise : Boolean := false;
        if should_raise then
           raise TFException.Create('Unable to lift tensor '+init_tensor.name+'.');
        if op.tipo = 'Placeholder' then
        begin
            extra_sources.AddRange(op.outputs);
        end;
        for var inp in op.inputs do
        begin

        end;
    end;
    Result := extra_sources;
end;

class procedure SubGraphUtility._copy_non_source(op: TFOperation; graph: TFuncGraph; op_map: TDictionary<ITensorOrOperation, TFOperation>; base_graph: TFGraph);
var
  copied_op    : TFOperation;
  copied_inputs: TFTensors;
  dtypes       : TArray<TF_DataType>;
begin
    copied_op := nil;
    copied_inputs := TFTensors.Create;
    TUtils.tf_with<TControlDependenciesController>(Tops.control_dependencies([op]),
         procedure(ctl : TControlDependenciesController)
          begin
              // Create a new op in the destination graph if it doesn't exist before.
              var attrs := TDictionary<string, TAttrValue>.Create;
              for var attr_def in op.NodeDef.Attr do
                  attrs.AddOrSetValue(attr_def.Key, attr_def.Value);
              for var i := 0 to Length(op.outputs) -1 do
                  dtypes := dtypes + [ op.outputs[i].Dtype ];

              copied_op := graph.create_op(op.tipo, copied_inputs.ToArray, dtypes, [], op.name, attrs);
          end);

    op_map.AddOrSetValue(op, copied_op);
    for var i := 0 to Length(op.outputs) - 1 do
        op_map.AddOrSetValue(op.outputs[i], copied_op.outputs[i].Op);
end;

class procedure SubGraphUtility._copy_source(s: TFTensor; graph: TFuncGraph; op_map: TDictionary<ITensorOrOperation, TFOperation>; handle_captures: Boolean;
  inverse_captures: TDictionary<TFTensor, TFTensor>; base_graph: TFGraph);
var
  copied_placeholder : TFTensor;
begin
    if (handle_captures) and  (inverse_captures.ContainsKey(s)) then
        copied_placeholder := graph.capture(inverse_captures[s], s.op.name)
    else
        raise TFException.Create('Not Implemented');

    op_map.AddOrSetValue(s, copied_placeholder.op);
    // Add an entry for the op of the source tensor so that if there are any nodes
    // depending on that op via control dependencies it can work correctly.
    op_map.AddOrSetValue(s.op, copied_placeholder.op);
end;
{$ENDREGION}

{$REGION 'Functions'}

{ TFTensor_helper }

function TFTensor_helper.ToTensorSpec: TensorSpec;
begin
    Result := TensorSpec.Create(shape, dtype, name)
end;

{ ConcreteFunction }

constructor ConcreteFunction.Create(func: TFunc<TFTensor, TFTensor>; dtype: TF_DataType; func_name: string);
var
  opers : TArray<TFOperation>;
begin
    func_name := func_name+'_'+ Tops.uid_function.ToString;

    func_graph := TFuncGraph.Create(func_name);

    func_graph.as_default;
    var input  := tf.placeholder(dtype);
    var output := func(input);

    for var it in func_graph.nodes_by_name.Values  do
        opers :=  opers + [ it as TFOperation] ;

    func_graph.ToGraph(opers, [ input ], [ output ], nil);
    func_graph.gExit;
end;

constructor ConcreteFunction.Create(graph: TFuncGraph; attrs: TDictionary<string, string>);
var
 tensorArray : TArray<TFTensor>;
begin
    func_graph := graph;

    for var i := 0 to graph.Outputs.Count -1  do
    begin
       if graph.Outputs[i] <> nil then
         tensorArray := tensorArray + [ graph.Outputs[i] ];
    end;
    ToGraph(graph.Inputs, TFTensors.Create(tensorArray));
end;

constructor ConcreteFunction.Create(_name: string);
begin
    func_graph := TFuncGraph.Create(_name);
end;

procedure ConcreteFunction.ToGraph(inputs, outputs: TFTensors);
var
  opers : TArray<TFOperation>;
begin
    for var it in func_graph.nodes_by_name.Values  do
        opers :=  opers + [ it as TFOperation] ;

    func_graph.ToGraph(opers, inputs.ToArray, outputs.ToArray, nil);

    OutputStructure := [];
    for var i := 0 to  outputs.Count -1  do
        OutputStructure :=  OutputStructure + [ outputs[i].ToTensorSpec ] ;
end;

procedure ConcreteFunction.AddTograph(g: TFGraph);
begin
    if ( not tf.Context.executing_eagerly) and ( g = nil) then
    begin
        g := Tops.get_default_graph;
    end;
end;

function ConcreteFunction.CallFlat(args, captured_inputs: TArray<TFTensor>): TFTensors;
begin
    var executing_eagerly := tf.Context.executing_eagerly;
    var tensor_inputs     := TFTensors.Create;
    for var i := 0 to Length(args) - 1 do
    begin
        var arg := args[i];
        tensor_inputs.Add(arg);
        // If we're graph building, shape inference is on.
        if not executing_eagerly then
        begin
        end;
    end;
    tensor_inputs.AddRange(captured_inputs);

    args := tensor_inputs.ToArray;
    var possible_gradient_type : Integer := 0;
    if tf.Runner.MustRecordGradient then
      possible_gradient_type := 1;

    // No tape is watching; skip to running the function.
    if (possible_gradient_type = 0) and (executing_eagerly) then
    begin
        var attrs : TArray<TValue> := ['executor_type', '', 'config_proto', tf.Context.FunctionCallOptions.config_proto_serialized];
        var Res := tf.Runner.Execute(tf.Context, func_graph.FuncName, func_graph.Outputs.count, args, attrs);
        Result := TFTensors.Create(Res) ;
        Exit;
    end;
    if forward_backward = nil then
        forward_backward := SelectForwardAndBackwardFunctions(TFTensors.Create(args), possible_gradient_type, executing_eagerly);
    var tFunc := forward_backward.&Forward;
    var forward_function   := tFunc.Value1;
    var args_with_tangents := tFunc.Value2;

    var flat_outputs : TFTensors := nil;
    if executing_eagerly then
        flat_outputs := forward_function.Call(args_with_tangents);

    forward_backward.&Record(flat_outputs);
    Result := flat_outputs;
end;

procedure ConcreteFunction.Enter;
begin
    func_graph.as_default;
end;

procedure ConcreteFunction._Exit;
begin
   func_graph.gExit;
end;

function ConcreteFunction.FilteredCall(inputs: TFTensors): TFTensors;
begin
    Result := CallFlat(inputs.ToArray, CapturedInputs);
end;

function ConcreteFunction.Get_CaptInput: TArray<TFTensor>;
begin
    Result := func_graph.external_captures
end;

function ConcreteFunction.Get_Inputs: TArray<TFTensor>;
begin
    Result := func_graph.Inputs.ToArray;
end;

function ConcreteFunction.Get_Name: string;
begin
    Result := func_graph.FuncName
end;

function ConcreteFunction.SelectForwardAndBackwardFunctions(args: TFTensors; possible_gradient_type: Integer; executing_eagerly: Boolean): ForwardBackwardCall;
begin
    var functions := FirstOrderTapeGradientFunctions.Create(func_graph, false);
    Result        := ForwardBackwardCall.Create(functions, args, true);
end;

function ConcreteFunction.ToString: string;
begin
    Result := Name
end;

{ ForwardBackwardCall }

constructor ForwardBackwardCall.Create(functions: TapeGradientFunctions; inference_args: TFTensors; tape_watching: Boolean);
begin
    Ffunctions      := functions;
    Finference_args := inference_args;
    Ftape_watching  := tape_watching;
end;

function ForwardBackwardCall.Forward: Tuple<EagerDefinedFunction, TFTensors>;
begin
    if Fforward_function = nil then
      Fforward_function := Ffunctions.Forward(Finference_args);
    Result := Tuple.Create(Fforward_function, Finference_args);
end;

procedure ForwardBackwardCall.&Record(flat_outputs: TFTensors);
begin
   if (Ftape_watching) and (flat_outputs <> nil) then
     Ffunctions.&Record(flat_outputs, Finference_args);
end;

{ EagerDefinedFunction }

constructor EagerDefinedFunction.Create(name: string; graph: TFuncGraph; inputs, outputs: TFTensors; attrs: TDictionary<string, string>);
var
  operations  : TArray<TFOperation>;
  input_ops   : TArray<TFOperation>;
  output_names: TArray<string>;
  i           : integer;
begin
    fnum_outputs := outputs.Count;
    for i:= 0 to inputs.Count - 1 do
      input_ops := input_ops + [ inputs[i].op ];

    var IOperation := graph.get_operations;
    for i := 0 to Length(IOperation) - 1 do
    begin
        var operation := IOperation[i];
        if not Tarray.Contains<TFOperation>(input_ops, operation.op) then
          operations := operations + [ operation as TFOperation ];
    end;
    func_graph := TFuncGraph.Create(graph.Handle, name, attrs);
    func_graph.ToGraph(operations, inputs.ToArray, outputs.ToArray, output_names);
end;

function EagerDefinedFunction.Call(args: TFTensors): TFTensors;
begin
    var attrs : TArray<TValue> := ['executor_type', '', 'config_proto', tf.Context.FunctionCallOptions.config_proto_serialized];

    var results := tf.Runner.TFE_Execute(tf.Context, tf.Context.DeviceName, func_graph.FuncName, args.ToArray, attrs, Fnum_outputs);

    Result := TFTensors.Create(results);
end;

function EagerDefinedFunction.Get_Name: string;
begin
    Result := func_graph.FuncName
end;

{ TapeGradientFunctions }

constructor TapeGradientFunctions.Create(f_func_graph: TFuncGraph; need_gradients_for_jvps : Boolean);
begin
    Ffunc_graph := f_func_graph;
end;

function TapeGradientFunctions.BuildFunctionsForOutputs(outputs, inference_args: TFTensors): Tuple<EagerDefinedFunction, TFuncGraph, ConcreteFunction>;
begin
    var trainable_outputs := TList<TFTensor>.Create;
    var trainable_indices := TList<Integer>.Create;
    for var i := 0 to outputs.Count -1 do
    begin
        var output := outputs[i];
        if gradients_util.IsTrainable(output) then
        begin
            trainable_outputs.Add(output);
            trainable_indices.Add(i);
        end;
    end;

    var gradients_wrt_outputs := TList<TFTensor>.Create;
    var backwards_graph       := TFuncGraph.Create(_BACKWARD_PREFIX+'_'+Ffunc_graph.FuncName+'_'+ Tops.uid.ToString);
    backwards_graph.as_default;
    for var output in trainable_outputs do
        gradients_wrt_outputs.Add(tf.placeholder(output.dtype, output.shape));

    var gradients_wrt_inputs := gradients_util._GradientsHelper(trainable_outputs.ToArray, Ffunc_graph.Inputs.ToArray, gradients_wrt_outputs.ToArray,'gradients', False, False, 0, nil, Ffunc_graph);

    var captures_from_forward : TArray<TFTensor> := [];
    for var i := 0 to Length(backwards_graph.external_captures) - 1 do
    begin
         var it := backwards_graph.external_captures[i];
         if ((it is TEagerTensor) = False) and ((it is TNDArray) = False) and (it.graph = Ffunc_graph) then
            captures_from_forward := captures_from_forward + [ it ];
    end;
    for var capture in captures_from_forward do
    begin
        if not Ffunc_graph.Outputs.Contains(capture) then
            Ffunc_graph.Outputs.Add(capture);
    end;
    backwards_graph.gExit;

    var forward_function_name  := _FORWARD_PREFIX +'_'+Ffunc_graph.FuncName+'_'+ Tops.uid.ToString;
    var backward_function_attr := TDictionary<string, string>.Create;

    backward_function_attr.AddOrSetValue(FORWARD_FUNCTION_ATTRIBUTE_NAME, forward_function_name);
    gradients_wrt_outputs.AddRange(backwards_graph.internal_captures);
    backwards_graph.Inputs  := TFTensors.Create(gradients_wrt_outputs.ToArray);
    backwards_graph.Outputs := TFTensors.Create(gradients_wrt_inputs);

    var backward_function     := ConcreteFunction.Create(backwards_graph, backward_function_attr);
    var forward_function_attr := TDictionary<string, string>.Create;
    forward_function_attr.AddOrSetValue(BACKWARD_FUNCTION_ATTRIBUTE_NAME, backward_function.Name);

    var forward_function := EagerDefinedFunction.Create(forward_function_name, Ffunc_graph, Ffunc_graph.Inputs, Ffunc_graph.Outputs, forward_function_attr);

    Result := Tuple.Create(forward_function, Ffunc_graph, backward_function);
end;

function TapeGradientFunctions.Forward(inference_args: TFTensors): EagerDefinedFunction;
begin
    Result := ForwardAndBackwardFunctions(inference_args);
end;

function TapeGradientFunctions.ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction;
begin
    raise Exception.Create('ForwardAndBackwardFunctions .  Virtual method');
end;

procedure TapeGradientFunctions.&Record(flat_outputs, inference_args: TFTensors);
begin
   var tWrapFun          := _wrap_backward_function(Fforward_graph, Fbackward, flat_outputs);
   var backward_function := tWrapFun.Value1;
   var to_record         := tWrapFun.Value2;

   tf.Runner.RecordGradient(Fforward.Name, inference_args.ToArray, [], to_record.ToArray, backward_function);
end;

function TapeGradientFunctions._wrap_backward_function(forward_graph: TFuncGraph; backward: ConcreteFunction; outputs: TFTensors): Tuple<BackwardFunction, TFTensors>;
begin
    var backward_function_inputs := Length(backward.Inputs) - Length(backward.CapturedInputs);
    var recorded_outputs := TFTensors.Create;
    var trainable_recorded_outputs := 0;
    for  var i := 0 to outputs.Count - 1 do
    begin
        var output       :=  outputs[i];
        if trainable_recorded_outputs < backward_function_inputs then
            recorded_outputs.Add(output);
        if gradients_util.IsTrainable(output) then
            trainable_recorded_outputs := trainable_recorded_outputs + 1;
    end;

    if not Assigned(Fbackward_function_wrapper) then
    begin
        var capture_mapping := TDictionary<Int64, TFTensor>.Create;
        for var i := 0 to outputs.Count -1 do
            capture_mapping.AddOrSetValue(forward_graph.Outputs[i].Id, outputs[i]);

        var remapped_captures := TFTensors.Create;
        for var capture in backward.CapturedInputs do
        begin
            if capture_mapping.ContainsKey(capture.Id) then
                remapped_captures.Add( capture_mapping[capture.Id] );
        end;

        var skip_positions := TList<Integer>.Create;
        for  var i := 0 to outputs.Count - 1 do
        begin
            var output_index :=  i;
            var output       :=  outputs[i];
            if not gradients_util.IsTrainable(output) then
               skip_positions.Add(output_index);
        end;

        Fbackward_function_wrapper := function (args : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
                begin
                    var processed_args := TFTensors.Create;
                    var input_index    := 0;
                    for var output_index := 0 to Length(args) - 1 do
                    begin
                        var arg := args[output_index];
                        if skip_positions.Contains(output_index) then
                            continue;
                        if arg = nil then
                            raise Exception.Create('Not Implemented');
                        processed_args.Add(arg);
                        input_index := input_index + 1;
                        if input_index >= backward_function_inputs then
                            break;
                    end;

                    tf.LogMsg('Invoke backward function: '+backward.Name);
                    var gradients := backward.CallFlat(processed_args.ToArray, remapped_captures.ToArray);

                    for var unneeded_gradient_index in unneeded_gradients do
                    begin
                        var index : Integer := unneeded_gradient_index;
                        if gradients.Count <= index then
                            gradients.Insert(index, nil);
                    end;

                    Result := gradients.ToArray;
                end;
    end;

    Result := Tuple<BackwardFunction, TFTensors>.Create(Fbackward_function_wrapper, recorded_outputs);
end;

{ FirstOrderTapeGradientFunctions }

constructor FirstOrderTapeGradientFunctions.Create(func_graph: TFuncGraph; need_gradients_for_jvps: Boolean);
begin
    inherited Create(func_graph, need_gradients_for_jvps);
end;

function FirstOrderTapeGradientFunctions.ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction;
begin
    var outputs := Ffunc_graph.Outputs;
    var tRes     := BuildFunctionsForOutputs(outputs, inference_args);

    Fforward       := tRes.Value1;
    Fforward_graph := tRes.Value2;
    Fbackward      := tRes.Value3;
    Fforwardprop_output_indices := nil;
    Fnum_forwardprop_outputs    := 0;

    Result := Fforward;
end;
{$ENDREGION}

{$REGION 'NameScope'}
{ TNameScope }

constructor TNameScope.Create(name, default_name: TF_TString; values: PValue; skip_on_eager: Boolean);
begin
    _name := name;
    _default_name := default_name;
    if values <> nil then
      _values := values^
    else
      _values := System.default(TValue);
    _skip_on_eager := skip_on_eager;
end;

function TNameScope.enter_eager_name_scope(ctx: TContext; name: TF_TString): Tuple<TF_TString, TF_TString>;
begin
    if _skip_on_eager then
        Exit(Tuple<TF_TString, TF_TString>.Create('',''));
    if name = '' then
        name := _default_name;
    var scope_name := name;
    var old_name   := ctx.ScopeName;
    // A trailing slash breaks out of nested name scopes, indicating a
    // fully specified scope name, for compatibility with Graph.name_scope.
    if not string(name).EndsWith('/') then
    begin
        scope_name := name + '/';
        if not string.IsNullOrEmpty(old_name) then
            scope_name := AnsiString(old_name) + scope_name;
    end;
    ctx.ScopeName := string(scope_name);

    Result  := Tuple<TF_TString, TF_TString>.Create(scope_name, old_name);
end;

function TNameScope.ToString: TF_TString;
begin
    Result := scope_name;
end;

procedure TNameScope._Enter_;
var
  tRes : Tuple<TF_TString,TF_TString>;
begin
    if tf.Context.executing_eagerly then
    begin
        tRes := enter_eager_name_scope(tf.Context, _name);
        scope_name     := tRes.Value1;
        old_scope_name := tRes.Value2;
    end else
    begin
        if _name = '' then
          _name := _default_name;
        var g : TFGraph := nil;
        if (not _values.IsEmpty) and (_values.IsType< TList<TFTensor> >) then
        begin
            var vList : TList<TFTensor> := _values.AsType< TList<TFTensor> > ;
            g := TOps._get_graph_from_inputs(vList.ToArray);
        end
        else if (not _values.IsEmpty) and (_values.IsType< TArray<TFTensor> >) then
        begin
            var vArray : TArray<TFTensor> := _values.AsType<  TArray<TFTensor> >;
            g := TOps._get_graph_from_inputs(vArray);
        end;
        if g = nil then
            g := TOps.get_default_graph;
        old_scope_name := g.name_stack;
        scope_name     := g.name_scope(_name);
    end;
end;

procedure TNameScope._Exit_;
begin
    if tf.Context.executing_eagerly then
        tf.Context.ScopeName := string(old_scope_name)
    else
        TOps.get_default_graph.name_stack := old_scope_name;
end;
{$ENDREGION}

{$REGION 'Gradient'}
{ TapeTensor }

constructor TapeTensor.Create(t: TFTensor);
begin
    Ftensor := t;
end;

function TapeTensor.GetID: Int64;
begin
    Result := tensor.Id;
end;

function TapeTensor.GetShape: TFShape;
begin
    Result := Ftensor.Shape;
end;

function TapeTensor.GetTensor: TFTensor;
begin
    Result := Ftensor;
end;

function TapeTensor.GetType: TF_DataType;
begin
    Result := Ftensor.Dtype
end;

function TapeTensor.OnesLike: TFTensor;
begin
    Result := tf.ones(shape, dtype);
end;

function TapeTensor.ZerosLike: TFTensor;
begin
    Result := tf.zeros(shape, dtype);
end;

function TapeTensor.ToString: string;
begin
   Result := Format('%d, %s, %s',[Id, shape.ToString, Tdtypes.as_numpy_name(dtype)])
end;

{ GradientTape }

constructor TGradientTape.Create;
begin
    FtapeSet := TStack<ITape>.Create;
end;

destructor TGradientTape.Destroy;
begin
  if Assigned(Ftape) then
      Ftape.Free;
  FtapeSet.Clear;
  FtapeSet.Free;

  inherited;
end;

function TGradientTape.GetTape: ITape;
begin
    Result := nil;
    if Assigned(FtapeSet) and (FtapeSet.Count > 0) then
      Result :=  FtapeSet.Peek;
end;

function TGradientTape.GetTapeSet: TStack<ITape>;
begin
    Result :=  FtapeSet;
end;

function TGradientTape.gradient(target: TFTensor; const source: ResourceVariable): TFTensor;
begin
     var res := gradient(target, [ source ]);
     Result := res[0];
end;

function TGradientTape.gradient(target: TFTensor; const source: TFTensor): TFTensor;
begin
    var tape : ITape := stop_recording;

    var res := tf.Runner.TFE_TapeGradient(tape, [ target ],[ source ], nil);
    Result := res[0];
end;

function TGradientTape.gradient(target: TFTensor; const sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor, TFTensor>;
begin
    var res := gradient(target, [ sources.Value1, sources.Value2 ]);
    Result := Tuple<TFTensor, TFTensor>.Create(res[0], res[1]);
end;

function TGradientTape.gradient(target: TFTensor; const sources: TArray<IVariableV1>): TArray<TFTensor>;
begin
    var tape := stop_recording;

    var aSource: TArray<TFTensor> := [];
    for var i := 0 to Length(sources)-1  do
      aSource := aSource + [ sources[i].tHandle ];

    var res := tf.Runner.TFE_TapeGradient(tape,[ target ], aSource, nil);
    if not tape.Persistente then
    begin
        // Keep track of watched variables before setting tape to None
        // _watched_variables = _tape.WatchedVariables();
    end;
    Result := res;
end;

procedure TGradientTape.NativeDispose(hnd: Pointer);
begin
  inherited;
  FtapeSet.Clear;
  FtapeSet.Free;
end;

function TGradientTape.PopTape: ITape;
begin
     Ftape.StopRecord;
     Result := FtapeSet.Pop;
end;

function TGradientTape.PushTape(persistent, watch_accessed_variables: Boolean): ITape;
begin
    // Enters a context inside which operations are recorded on this tape.
    if tf.Context.executing_eagerly then
        tf.Context.ensure_initialized;
    var tape := TTape.Create(persistent, watch_accessed_variables);
    tape.SetTapeId(FnextTapeId);
    Inc(FnextTapeId);
    FtapeSet.Push(tape);
    Result := tape;
end;

function TGradientTape.stop_recording: ITape;
begin
    var tape := Ftape;
    if not tape.Persistente then
        tape := PopTape;
    Result := tape;
end;

procedure TGradientTape.watch(x: TFTensor);
begin
    if FtapeSet.count  < 1 then
        Exit;
    Ftape.Watch(x);
end;

{ TTape }

constructor TTape.Create(persistent, watch_accessed_variables: Boolean);
begin
    inherited Create;

    next_op_id_       := 0;
    F_persistent      := persistent;
    F_created_eagerly := tf.Context.executing_eagerly;
    Ftensor_tape_     := TDictionary<TFTensor, Int64>.Create;
    Fop_tape_         := TDictionary<Int64, OpTapeEntry>.Create;
    Ftensor_usage_    := TDictionary<TFTensor, Int64>.Create;
    if F_created_eagerly then
        tf.Context.start_step;

end;

destructor TTape.Destroy;
begin
    Ftensor_tape_.free;
    Fop_tape_.Free;
    Ftensor_usage_.free;

    inherited Destroy
end;

function TTape.IsDtypeTrainable(dtype: TF_DataType): Boolean;
begin
    case dtype of
      TF_DataType.TF_HALF,
      TF_DataType.TF_BFLOAT16,
      TF_DataType.TF_FLOAT,
      TF_DataType.TF_DOUBLE,
      TF_DataType.TF_COMPLEX64,
      TF_DataType.TF_COMPLEX128,
      TF_DataType.TF_RESOURCE,
      TF_DataType.TF_VARIANT: Result := True;
    else
      Result := False;
    end;
end;

function TTape.InitialGradients(target_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>;
  tensor_tape: TensorTape; op_tape: OpTape): TDictionary<TFTensor, TList<TFTensor>>;
begin
    var res := TDictionary<TFTensor, TList<TFTensor>>.Create;
    for var i : Integer := 0 to Length(target_tensor_ids) - 1 do
    begin
        var id := target_tensor_ids[i];
        if (Length(output_gradients) = 0) or (output_gradients[i] = nil) then
        begin
            if (tensor_tape.ContainsKey(id)) and (id <> nil) then
            begin
                if not op_tape.ContainsKey(tensor_tape[id]) then
                    raise TFException.Create('Iternal state of the gradient tape is invalid: failed to find operation producing a tensor');
                var op_it := op_tape[ tensor_tape[id] ];
                var found : Boolean := false;
                for var j := 0 to Length(op_it.output_tensor_info) - 1 do
                begin
                    if op_it.output_tensor_info[j].GetTensor = id then
                    begin
                        found    := true;
                        var ones := op_it.output_tensor_info[j].OnesLike;
                        if res.ContainsKey(id) then res[id].Add(ones)
                        else                        res.AddOrSetValue(id, TList<TFTensor>.Create([ones]));
                        break;
                    end;
                end;
                if not found then
                begin
                    raise TFException.Create('Internal state of the gradient tape is invalid: none of operations outputs match expected tensor');
                end;
            end else
            begin
                if sources_that_are_targets.ContainsKey(id) then
                begin
                    var source_tensor := sources_that_are_targets[id];
                    if res.ContainsKey(id) then res[id].Add(source_tensor.OnesLike)
                    else                        res.AddOrSetValue(id, TList<TFTensor>.Create([source_tensor.OnesLike]));
                end;
            end;
        end else
        begin
            if res.ContainsKey(id) then res[id].Add(output_gradients[i])
            else                        res.AddOrSetValue(id, TList<TFTensor>.Create([ output_gradients[i] ]));
        end;
    end;
    result := res;
end;

function TTape.InitialStack(op_tape: OpTape; op_missing_tensor: TDictionary<Int64, Int64>): IQueue<Int64>;
begin
    var res := TCollections.CreateQueue<Int64>;
    for var op_entry in op_tape do
    begin
        if not op_missing_tensor.ContainsKey(op_entry.Key) then
            res.Enqueue(op_entry.Key);
    end;
    Result := res;
end;

function TTape.FunctionsAcceptingNoneForIndicesMap: TDictionary<string, ISet<Integer>>;
begin
    var m := TDictionary<string, ISet<integer>>.Create;
    m.Add('SoftmaxCrossEntropyWithLogits',       TCollections.CreateSet<integer>([ 1 ]));
    m.Add('SparseSoftmaxCrossEntropyWithLogits', TCollections.CreateSet<integer>([ 1 ]));
    m.Add('FusedBatchNorm',                      TCollections.CreateSet<integer>([ 1, 2, 3, 4 ]));
    Result := m;
end;

function TTape.PrepareBackprop(target: TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape; sources_set: ISet<TFTensor>; persistent_tape: Boolean): BackpropInitialState;
begin
    var res : BackpropInitialState :=  BackpropInitialState.Create;
    var tensor_stack := TCollections.CreateQueue<TFTensor>(target);
    while tensor_stack.Count > 0 do
    begin
        var tensor_id := tensor_stack.Dequeue;
        if not tensor_tape.ContainsKey(tensor_id) then Continue;
        var op_id := tensor_tape[tensor_id] ;
        if (op_id = -1) or ( not op_tape.ContainsKey(op_id)) or (res.op_tape.ContainsKey(op_id) ) then Continue;
        var op_it        := op_tape[op_id] ;
        //var result_op_it := result.op_tape[op_id] ;
        res.op_tape.AddOrSetValue(op_id, op_it);
        for var it in op_it.input_tensor_id do
        begin
            if res.tensor_usage_counts.ContainsKey(it) then
               res.tensor_usage_counts[it] := res.tensor_usage_counts[it] + 1
            else begin
               res.tensor_usage_counts.AddOrSetValue(it, 1);
               if tensor_tape.ContainsKey(it) then
                    tensor_stack.Enqueue(it);
            end;
        end;
        if not persistent_tape then
            op_tape.Remove(op_id);
    end;
    for var pair in res.tensor_usage_counts do
    begin
        if (tensor_tape.ContainsKey(pair.Key)) and (tensor_tape[pair.Key] <> -1) then
        begin
           var it := tensor_tape[pair.Key];
           if res.op_missing_tensor.ContainsKey(it) then  res.op_missing_tensor[it] := res.op_missing_tensor[it] + 1
           else                                           res.op_missing_tensor.Add(it,1);
        end;
    end;
    if not persistent_tape then
    begin
        // Call destructors for all unneeded gradient functions and
        // clear the op_tape. We can clear the tape because ownership of
        // backward functions that will be used for gradient computation
        // has been transferred to `result`.
        (*for (const auto&op_pair : *op_tape) {
            op_pair.second.backward_function_deleter(
                op_pair.second.backward_function);
        } *)
        op_tape.Clear;
    end;
    Result := res;
end;


function TTape.ComputeGradient(target_tensor_ids, source_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>;
  output_gradients: TArray<TFTensor>): TArray<TFTensor>;
var
  sources_set                     : ISet<TFTensor>;
  func_AcceptingNoneForIndicesMap : TDictionary<string, ISet<Integer>> ;
  state                           : BackpropInitialState;
  op_stack                        : IQueue<Int64>;
  gradients                       : TDictionary<TFTensor, TList<TFTensor>>;
  trace                           : OpTapeEntry;
  out_gradients                   : TList<TFTensor>;
  unneeded_gradients              : TList<Int64>;
  zero_indices                    : TList<Integer>;
  in_gradients                    : TArray<TFTensor>;
begin
    sources_set := TCollections.CreateSet<TFTensor>(source_tensor_ids);
    // var gradients_size = new UnorderedMap<Tensor, long>();
    func_AcceptingNoneForIndicesMap := FunctionsAcceptingNoneForIndicesMap;
    state    := PrepareBackprop(target_tensor_ids, Ftensor_tape_, Fop_tape_, sources_set, F_persistent);
    op_stack := InitialStack(state.op_tape, state.op_missing_tensor);
    gradients:= InitialGradients(target_tensor_ids, sources_that_are_targets, output_gradients, Ftensor_tape_, state.op_tape);
    while op_stack.Count > 0 do
    begin
        var op := op_stack.Dequeue;
        if not state.op_tape.ContainsKey(op) then
            continue;
        trace := state.op_tape[op];
        // Console.WriteLine($"ComputeGradient: {state.op_tape[op].op_type}");
        state.op_tape.Remove(op);
        out_gradients          := TList<TFTensor>.Create;
        out_gradients.Capacity := Length(trace.output_tensor_info);
        unneeded_gradients := TList<Int64>.Create;
        for var i := 0 to Length(trace.input_tensor_id)- 1 do
        begin
            var in_tensor_id := trace.input_tensor_id[i];
            if (not Ftensor_tape_.ContainsKey(in_tensor_id)) and (not sources_set.Contains(in_tensor_id)) then
                unneeded_gradients.Add(i);
        end;
        var any_gradient_nonzero : boolean := false;
        zero_indices := TList<Integer>.Create;
        for var i := 0 to Length(trace.output_tensor_info)-1 do
        begin
            var id := trace.output_tensor_info[i].GetTensor;
            if  not gradients.ContainsKey(id) then
            begin
                if (func_AcceptingNoneForIndicesMap.ContainsKey(trace.op_type)) and  (func_AcceptingNoneForIndicesMap[trace.op_type].Contains(i)) then
                begin
                    out_gradients.Add(nil);
                end else
                begin
                    out_gradients.Add(nil);
                    zero_indices.Add(i);
                end;
            end else
            begin
                any_gradient_nonzero := true;
                var grad_it := gradients[id];
                var new_gradients : TFTensor ;
                if grad_it.Count = 1 then new_gradients := grad_it[0]
                else                      new_gradients := gen_math_ops.add_n(grad_it.ToArray);  // vspace.AggregateGradients
                if not sources_set.Contains(id) then
                    gradients.Remove(id)
                else begin
                    // grad_it.Clear();
                    // grad_it.Add(new_gradients);
                    // vspace.MarkAsResult(new_gradients);
                end;
                out_gradients.Add(new_gradients);
            end;
        end;
        in_gradients := [];
        if any_gradient_nonzero then
        begin
            // foreach (var i in zero_indices)
            //     out_gradients[i] = trace.output_tensor_info[i].ZerosLike();
            in_gradients := trace.backward_function(out_gradients.ToArray, unneeded_gradients.ToArray);
            if (Length(in_gradients) <> Length(trace.input_tensor_id)) and ((Length(in_gradients) + unneeded_gradients.Count) <> Length(trace.input_tensor_id))then
                raise TFException.Create( Format('Recorded operation "%s" returned too few gradients. Expected %d but received %d',[trace.op_type, Length(trace.input_tensor_id), Length(in_gradients)]) );
            if not F_persistent then
            begin
                // trace.backward_function_deleter(trace.backward_function);
                trace.backward_function := nil;
            end;
        end else
        begin
            SetLength(in_gradients, Length(trace.input_tensor_id));
        end;

        var k : Integer := 0;
        var skip_unneeded_id : Boolean := Length(trace.input_tensor_id) > Length(in_gradients);
        for var i := 0 to Length(in_gradients) - 1 do
        begin
            if k >= Length(trace.input_tensor_id) then Break;

            if (skip_unneeded_id) and (unneeded_gradients.Contains(k)) then Inc(k);
            var id := trace.input_tensor_id[k];

            Inc(k);

            if in_gradients[i] <> nil then
            begin
                if not gradients.ContainsKey(id) then
                      gradients.Add(id,TList<TFTensor>.Create );

                var unaggregated_grads := gradients[id];
                unaggregated_grads.Add(in_gradients[i]);
                (*if (unaggregated_grads.Count > kMinAggregateCount)
                {
                    if (!gradients_size.find(id, out var size))
                    {
                        size = (long)unaggregated_grads[0].size;
                        gradients_size.emplace(id, size);
                    }
                    if (unaggregated_grads.Count * size * 4 > kMinAggregateBytes)
                    {
                        throw new NotImplementedException("");
                    }
                }*)
            end;
            if not state.tensor_usage_counts.ContainsKey(id) then
                continue;
            state.tensor_usage_counts[id] := state.tensor_usage_counts[id] - 1;
            if state.tensor_usage_counts[id] > 0 then
                continue;
            if not Ftensor_tape_.ContainsKey(id) then
            begin
                if gradients.ContainsKey(id) then
                begin
                    // foreach (var g in grad_it)
                    // DeleteGradient(g);
                    gradients.Remove(id);
                end;
                continue;
            end;
            var tape_it := Ftensor_tape_[id] ;
            var op_id   := tape_it;
            if op_id = -1 then
                continue;
            if state.op_missing_tensor.ContainsKey(op_id) then
            begin
                state.op_missing_tensor[op_id] := state.op_missing_tensor[op_id] - 1;
                if state.op_missing_tensor[op_id] = 0 then
                    op_stack.Enqueue(op_id);
            end;
        end;
    end;
    if state.op_tape.Count > 0 then
       raise Exception.Create('Invalid tape state.');
    var res : TArray<TFTensor>; SetLength(res, Length(source_tensor_ids) );
    var j : Integer := 0;
    for var id in source_tensor_ids do
    begin
        if gradients.ContainsKey(id) then
        begin
            var grad_it := gradients[id];
            if grad_it.Count > 1 then  res[j] := gen_math_ops.add_n(grad_it.ToArray)
            else                       res[j] := grad_it[0];
        end;
        Inc(j);
    end;
    Result := res;
end;

procedure TTape.RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);
begin
    if not ShouldRecord(input_tensors) then
        Exit;
    var op_id := next_op_id_;
    Inc(next_op_id_);
    for var i in input_tensors do
    begin
        if Ftensor_usage_.ContainsKey(i) then  Ftensor_usage_[i] := Ftensor_usage_[i] + 1
        else                                   Ftensor_usage_.AddOrSetValue(i,0);
    end;
    for var o in output_tensors do
    begin
        //tf.Logger.Debug($"RecordOperation: tensor_tape_[{o.GetID()}] = {op_id}");
        Ftensor_tape_.AddOrSetValue(o.GetTensor, op_id);
        Ftensor_usage_.AddOrSetValue(o.GetTensor, 1);
    end;
    var opT : OpTapeEntry;
    opT.op_type            := op_type;
    opT.output_tensor_info := output_tensors;
    opT.input_tensor_id    := input_tensors;
    opT.backward_function  := backward_function;
    Fop_tape_.AddOrSetValue(op_id,opT);
end;

procedure TTape.SetTapeId(id: Integer);
begin
    Fid := id;
end;

function TTape.ShouldRecord(tensors: TArray<TFTensor>): Boolean;
begin

    for var i := 0 to Length(tensors) - 1 do
    begin
        if Ftensor_tape_.Containskey(tensors[i]) then
        begin
            if IsDtypeTrainable(tensors[i].Dtype) then
                Exit( true );
        end;
    end;
    Result := false;
end;

procedure TTape.StartRecord;
begin
    if F_recording then
       raise TFException.Create('Tape is still recording, This can happen if you try to re-enter an already-active tape.');
    F_recording := true;
end;

procedure TTape.StopRecord;
begin
    if not F_recording then
       raise TFException.Create('Tape is not recording.');
    if F_created_eagerly then
        tf.Context.end_step;
    F_recording := false;
end;

function TTape.ToString: string;
begin
     if F_recording then Result := Format('Tape % Recording',[Fid])
     else                Result := Format('Tape % Stopped',[Fid])

end;

procedure TTape.VariableAccessed(variable: ResourceVariable);
begin
    Watch(variable.tHandle);
end;

procedure TTape.Watch(x: TFTensor);
begin
    //tf.Logger.Debug($"Watch tensor id={x.Id}, name={x.name}");
    Ftensor_tape_.AddOrSetValue(x, -1);
end;

function TTape.WatchedVariables: TArray<ResourceVariable>;
begin
    Result := nil;
end;

{ TGradFunc }

constructor TGradFunc.Create(_name: string; _func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>);
begin
    Self.Name := _name;
    Self.func := _func;
end;


{ BackpropInitialState }

constructor BackpropInitialState.Create;
begin
   op_tape             := OpTape.Create;
   tensor_usage_counts := TDictionary<TFTensor, Int64>.Create;
   op_missing_tensor   := TDictionary<Int64, Int64>.Create;
end;

destructor BackpropInitialState.Destroy;
begin
   op_tape.Free;
   tensor_usage_counts.Free;
   op_missing_tensor.Free;
end;
{$ENDREGION}

{$REGION 'Variable'}
{ RefVariable }

function RefVariable.assign<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var vValue : TValue := TValue.From<T>(value);
    var assign := gen_state_ops.assign(_variable, vValue, True, use_locking, name);
    if read_value then
        Exit( assign );
    Result := assign.op.Output;
end;

function RefVariable.assign_add<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
   var variable := self;
   var _op := tf.OpDefLib._apply_op_helper('AssignAdd', name, [ GetArg('variable',TValue.From<RefVariable>(variable)),GetArg('value',TValue.From<T>(value)),GetArg('use_locking', use_locking)]);
   Result := _op.outputs[0];
end;

function RefVariable.assign_sub<T>(delta: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    raise  TFException.Create('Not Implemented');
end;

function RefVariable.assign_lazy_load(value: TFTensor; name: string): IVariableV1;
begin
    raise  TFException.Create('Not Implemented');
end;

function RefVariable.assign_sub_lazy_load(delta: TFTensor; name: string): IVariableV1;
begin
    raise  TFException.Create('Not Implemented');
end;

function RefVariable.AsTensor(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    Result := Fsnapshot;
end;

constructor RefVariable.Create(initial_value: PValue; trainable: Boolean; collections: TList<string>; validate_shape: Boolean; caching_device, name: string;
  variable_def: TVariableDef; dtype: TF_DataType; import_scope: string);
begin

end;

function  RefVariable.GetHandle:TFTensor;
begin
    Result := _Variable;
end;

function RefVariable.GetInitializer: TFOperation;
begin
   Result := Finitializer_op
end;

function RefVariable.numpy: TNDArray;
begin
   raise Exception.Create('Graph mode can''t use numpy().');
end;

function RefVariable.ToString: string;
begin
    Result := Format('tf.RefVariable %s  shape=%s  dtype=%d',[Name, Shape.ToString, Ord(dtype)]);
end;

function RefVariable.To_Tensor: TFTensor;
begin
    Result := self.AsTensor;
end;

function RefVariable.To_VarScopeStore: _VariableScopeStore;
begin
    Result := nil;
end;

function RefVariable.value: TFTensor;
begin
    Result := Fsnapshot;
end;

function RefVariable._ref: TFTensor;
begin
    Result := _Variable;
end;

function RefVariable._TensorConversionFunction(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    if as_ref then Result := _ref()
    else           Result := value;
end;

function RefVariable.Eval: TFTensor;
begin
   Result := _Variable;
end;

function RefVariable.GetDevice: String;
begin
   Result := FDevice
end;

function RefVariable.GetGraph: TFGraph;
begin
    Result := _Variable.graph;
end;

function RefVariable.GetGraphEle: TFTEnsor;
begin
    Result := Fgraph_element;
end;

function RefVariable.GetName: String;
begin
    Result := _Variable.Name;
end;

function RefVariable.GetOp: TFOperation;
begin
   Result := _Variable.Op;
end;

function RefVariable.GetShape: TFShape;
begin
   Result := _Variable.Shape;
end;

function RefVariable.GetTipo: TF_DataType;
begin
    Fdtype := _variable.dtype;
    Result := Fdtype;

end;

function RefVariable.GetUniqueId: string;
begin
    Result := FName
end;

function RefVariable._as_graph_element: TFTEnsor;
begin
    Result := _Variable;
end;

{ _UnreadVariable }

constructor _UnreadVariable.Create(hHandle: TFTensor; dDtype: TF_DataType; sShape: TFShape; in_graph_mode: Boolean; unique_id: string);
begin
    Fdtype         := dDtype;
    Fshape         := sShape;
    Handle         := hHandle;
    Funique_id     := unique_id;
    Fin_graph_mode := in_graph_mode;
    if hHandle is TEagerTensor then  Fhandle_name := ''
    else                             Fhandle_name := Fhandle.name;
end;

function _UnreadVariable.GetGraphEle: TFTEnsor;
begin
    Result := Fgraph_element;
end;

function _UnreadVariable.GetHandle: TFTensor;
begin
    Result := Fhandle;
end;

function _UnreadVariable.GetInitializer: TFOperation;
begin
   Result := Finitializer_op
end;

function _UnreadVariable.GetName: string;
begin
    if Fin_graph_mode then  Result := Fparent_op.name
    else                    Result := 'UnreadVariable';
end;

function _UnreadVariable.GetOP: TFOperation;
begin
    Result := nil;
end;

function _UnreadVariable.GetShape: TFShape;
begin
    Result := Fshape;
end;

function _UnreadVariable.GetTipo: TF_DataType;
begin
    Result := Fdtype ;
end;

function _UnreadVariable.GetUniqueId: string;
begin
    Result := Funique_id;
end;

function _UnreadVariable._TensorConversionFunction(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    result := nil;
end;

{ ResourceVariable }

constructor ResourceVariable.Create(_initial_value: PValue; _trainable: Boolean; collections: TList<string>; validate_shape: Boolean; caching_device, name: string;
                                       variable_def: TVariableDef; dtype: TF_DataType; import_scope: string; aggregation: TVariableAggregation; shape: PTFShape);
begin
    if Assigned(variable_def) then
    begin
        if Assigned(_initial_value) then
           raise  TFException.Create('variable_def and initial_value are mutually exclusive.');
        _init_from_proto(variable_def, import_scope);
    end else
    begin
        _init_from_args(_initial_value, _trainable, collections, caching_device, name, dtype, aggregation, shape);
    end;
end;

function ResourceVariable._dense_var_to_tensor(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    Result := value;
end;

procedure ResourceVariable._init_from_args(_initial_value: PValue; _trainable: Boolean; collections: TList<string>; caching_device, name: string; dtype: TF_DataType;
                                             aggregation: TVariableAggregation; shape: PTFShape);
begin
    var initial_value := _initial_value^;

    var t      := GetTypeData( initial_value.TypeInfo)^;
    var parent : PTypeInfo := nil;
    if (initial_value.IsClass) and (t.ParentInfo <> nil) then
        parent := t.parentInfo^ ;

    var init_from_fn : boolean := False;
    if (string(initial_value.TypeInfo^.Name).Contains('Func')) or (initial_value.IsType<IInitializer>) or ( (parent <> nil) and (parent.Name = 'IInitializer') ) then
       init_from_fn  := true;

    if collections = nil then
        collections := TList<string>.Create( tf.GraphKeys.GLOBAL_VARIABLES );

    Ftrainable := _trainable;

    if (_trainable) and ( not collections.Contains(tf.GraphKeys.TRAINABLE_VARIABLES) ) then
        collections.Add(tf.GraphKeys.TRAINABLE_VARIABLES);

    TUtils.tf_with<TNameScope>( TOps.init_scope,
        procedure(v1: TNameScope)
          begin
              Fin_graph_mode := not tf.Context.executing_eagerly;

              TUtils.tf_with<TNameScope>( TOps.name_scope(name, 'Variable', _initial_value, false),
                procedure(v1: TNameScope)
                  begin
                      name := v1.ToString;
                      var handle_name := Tops.name_from_scope_name(name);
                      var unique_id  : string := '';
                      var shared_name: string := '';
                      if Fin_graph_mode then
                      begin
                          shared_name := handle_name;
                          unique_id   := shared_name;
                      end else
                      begin
                          unique_id   := handle_name+'_'+IntTostr(Tops.uid);
                          shared_name := tf.Context.shared_name;
                      end;

                      TUtils.tf_with<TNameScope>( TOps.name_scope('Initializer'),
                        procedure(v1: TNameScope)
                          begin
                              if (initial_value.IsType<IInitializer>) or ( (parent <> nil) and (parent.Name = 'IInitializer') ) then
                              begin
                                 Finitial_value := Tops.convert_to_tensor((initial_value.AsType<IInitializer>).Apply( InitializerArgs.Create(shape, dtype)))
                              end else
                              begin
                                  var value : TValue;
                                  if init_from_fn then
                                  begin
                                       var ffunc :=  initial_value.AsType<TFunc<TFTensor>>;
                                       var t : TFTEnsor := ffunc();
                                       value := t;
                                  end else
                                  begin
                                      value := initial_value;
                                  end;
                                  Finitial_value := Tops.convert_to_tensor(value, dtype, 'initial_value' );
                              end;
                          end );
                      if shape <> nil then Fshape  := shape^
                      else                 Fshape  := Finitial_value.Shape;

                      if Fin_graph_mode then
                      begin
                          Fhandle         := state_ops.variable_op_v2(Finitial_value.shape, Tdtypes.as_base_dtype(Finitial_value.dtype), name);
                          Finitializer_op := gen_state_ops.assign(Fhandle, Finitial_value, true).op;
                          Tops.colocate_with(Finitializer_op);
                          Fgraph_element := gen_array_ops.identity(Fhandle, 'read');
                          Tops.Add_to_collection<IVariableV1>(collections, Self);
                          Fdtype :=  Fhandle.dtype;
                      end else
                      begin
                          Fhandle := resource_variable_ops.eager_safe_variable_handle(Finitial_value,Fshape, shared_name,name, Fin_graph_mode);
                          gen_resource_variable_ops.assign_variable_op(Fhandle, Finitial_value);
                          Finitializer_op := nil;
                          Fgraph_element  := nil;
                          Fdtype          := Tdtypes.as_base_dtype(Finitial_value.dtype);
                          {TODO -oMax -c Test for free variable resource : Variable}
                          var collections_ := TList<string>.Create( tf.GraphKeys.GLOBAL_STEP );
                          Tops.Add_to_collection<IVariableV1>(collections_, Self);
                          // initial_value = _in_graph_mode ? initial_value : null;
                      end;

                      __init__(_trainable,Fhandle, name, unique_id, handle_name);

                  end );
          end );
end;

procedure ResourceVariable._init_from_proto(variable_def: TVariableDef; import_scope: string);
begin
    Fin_graph_mode := true;
    if not variable_def.IsResource then
       raise TFException.Create('Trying to restore Variable as ResourceVariable.');
    // Create from variable_def.
    var g := Tops.get_default_graph;
    var prepend_name_scope := Tops.prepend_name_scope(variable_def.VariableName, import_scope);
    Fhandle                := g.as_graph_element(prepend_name_scope) as TFTensor;
    Fhandle_name           := Fhandle.name;
    Fname                  := Fhandle.name;
    Fshape                 := TFShape.Create( Fhandle.op.get_attr('shape').AsType<TTensorShapeProto> );
    prepend_name_scope     := Tops.prepend_name_scope(variable_def.InitializerName, import_scope);
    Finitializer_op        := g.as_graph_element(prepend_name_scope) as TFOperation;
    if  not string.IsNullOrEmpty(variable_def.InitialValueName) then
    begin
        prepend_name_scope := Tops.prepend_name_scope(variable_def.InitialValueName, import_scope);
        Finitial_value     := g.as_graph_element(prepend_name_scope) as TFTensor;
    end;
    Ftrainable := variable_def.Trainable;
    (*var (synchronization, aggregation, trainable) =
                       variables.validate_synchronization_aggregation_trainable(
    variable_def.Synchronization,
    variable_def.Aggregation,
    variable_def.Trainable,
    variable_def.VariableName);*)
    if not string.IsNullOrEmpty(variable_def.SnapshotName) then
    begin
        prepend_name_scope := Tops.prepend_name_scope(variable_def.SnapshotName, import_scope);
        var snapshot       := g.as_graph_element(prepend_name_scope) as TFTensor;
        while (snapshot.op.tipo <> 'ReadVariableOp') do
            snapshot := snapshot.op.inputs[0];
        Fgraph_element := snapshot;
    end else
    begin
        raise TFException.Create('Not Implemented SnapshotName _init_from_proto');
    end;
    if variable_def.SaveSliceInfoDef <> nil then
    begin
        raise TFException.Create('Not Implemented SaveSliceInfoDef _init_from_proto');
    end;
    Fdtype := Tdtypes.as_tf_dtype( Fhandle.op.get_attr('dtype') );
end;

function ResourceVariable.to_proto(export_scope: string): TVariableDef;
begin
    if (string.IsNullOrEmpty(export_scope)) or (FHandle.name.StartsWith(export_scope)) then
    begin
        var var_def : TVariableDef := TVariableDef.Create;
        var_def.VariableName := Tops.strip_name_scope(FHandle.name, export_scope);

        if Finitial_value <> nil then
            var_def.InitialValueName := Tops.strip_name_scope(Finitial_value.name, export_scope);

        var_def.Trainable       := Ftrainable;
        var_def.InitializerName := Tops.strip_name_scope(initializer.name, export_scope);
        var_def.SnapshotName    := Tops.strip_name_scope(Fgraph_element.name, export_scope);
        Result := var_def;
        Exit;
    end;
    raise TFException.Create('Not Implemented to_proto RefVariable');
end;

function ResourceVariable.eval(session: TFSession): TNDArray;
begin
    Result := Fgraph_element.eval(session);
end;

function ResourceVariable.GetHandle:TFTensor;
 begin
     Result := FHandle;
 end;

function ResourceVariable.GetInitializer: TFOperation;
begin
   Result := Finitializer_op;
end;

function ResourceVariable.GetItem(slices: TArray<string>): TFTensor;
begin
    var sl: TArray<Slice> := [];
    for var i := 0 to Length(slices) -1 do
    begin
        sl := sl + [ Slice.Create( slices[i] ) ]
    end;
    Result := item[sl];
end;

function ResourceVariable.GetItem(slices: TArray<Slice>): TFTensor;
begin
    var args := TUtils.ParseSlices(slices);
    var newVal : TValue := TValue.From< ParsedSliceArgs >(args);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'strided_slice', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var name : string := v1.ToString;
                                                if args.aBegin <> nil then
                                                begin
                                                    var packed_begin  := array_ops.stack( TValue.from< TArray<Integer> >(args.aBegin) );
                                                    var packed_end    := array_ops.stack( TValue.from< TArray<Integer> >(args.aEnd) );
                                                    var packed_strides:= array_ops.stack( TValue.from< TArray<Integer> >(args.aStrides) );
                                                    Result := gen_array_ops.strided_slice(self._dense_var_to_tensor,
                                                                                          packed_begin,
                                                                                          packed_end,
                                                                                          packed_strides,
                                                                                          args.iBeginMask,
                                                                                          args.iEndMask,
                                                                                          args.iEllipsisMask,
                                                                                          args.iNewAxisMask,
                                                                                          args.iShrinkAxisMask,
                                                                                          name);
                                                    Exit;
                                                end;
                                                raise  TFException.Create('Not Implemented');
                                            end );
end;

function ResourceVariable.GetName: string;
begin
   Result := Fhandle_name
end;

function ResourceVariable.GetDevice: String;
begin
     Result  := FHandle.Device;
end;

function ResourceVariable.GetGraph: TFGraph;
begin
    Result := FHandle.graph;
end;

function ResourceVariable.GetGraphEle: TFTEnsor;
begin
    Result := Fgraph_element
end;

function ResourceVariable.GetOP: TFOperation;
begin
    Result := FHandle.Op;
end;

function ResourceVariable.GetParent_op: TFTEnsor;
begin
    Result := Fparent_op;
end;

function ResourceVariable.GetShape: TFShape;
begin
    Result := FShape
end;

function ResourceVariable.GetTipo: TF_DataType;
begin
    Result := Fdtype
end;

function ResourceVariable.GetTrainable: Boolean;
begin
    Result := Ftrainable;
end;

function ResourceVariable.GetUniqueId: string;
begin
    Result := Funique_id
end;

function ResourceVariable.numpy: TNDArray;
begin
    if tf.context.executing_eagerly then
      Result := inherited numpy
    else
      raise TFException.Create('numpy() is only available when eager execution is enabled.')
end;

function ResourceVariable.sparse_read(indices: TFTensor; name: string): TFTensor;
begin
     Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('Read'),
                    function(v1: TNameScope): TFTensor
                      begin
                          var sName := v1.ToString;
                          var value := gen_resource_variable_ops.resource_gather(Fhandle, indices, Fdtype, 0, True, sName);
                          Result := array_ops.identity(value);
                       end );

end;

function ResourceVariable.ToTensor: TFTensor;
begin
   Result :=  _dense_var_to_tensor
end;

function ResourceVariable._TensorConversionFunction(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    if as_ref then
       Result := Fhandle
    else begin
             if GraphElement <> nil then Result  := GraphElement
             else                        Result  := read_value;
    end;
end;

{ BaseResourceVariable }

constructor BaseResourceVariable.Create;
begin

end;

destructor BaseResourceVariable.Destroy;
begin
   inherited Destroy;
end;

procedure BaseResourceVariable.NativeDispose(hnd: Pointer);
begin
    if Fhandle is TEagerTensor then
      tf.Runner.TFE_Execute(tf.Context, AnsiString(Fhandle.Device), 'DestroyResourceOp',[ FHandle ], ['ignore_lookup_error', true ], 0);
end;


function BaseResourceVariable.assign<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var vValue := TValue.From<T>(value) ;
    if String.lowercase(vValue.typeinfo.name) = 'tvalue' then
          vValue := vValue.AsType<TValue>;
    if string.LowerCase(string(vValue.TypeInfo.Name)) = 'tftensor' then
    begin
        var assign := gen_state_ops.assign(FHandle, TValue.From<T>(value), True,use_locking, name);
        if read_value then
            Exit(assign);
        Exit( assign.op.Output );
    end;
    var value_tensor := Tops.convert_to_tensor(vValue, dtype);
    var assign_op    := gen_resource_variable_ops.assign_variable_op(FHandle, value_tensor, name);
    if read_value then
    begin
        Result := gen_resource_variable_ops.read_variable_op(FHandle, dtype);
        Exit;
    end;
    if assign_op = nil then
        Exit(nil);
    Result := assign_op.output;
end;

function BaseResourceVariable.assign_add<T>(delta: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var assign_add_op := gen_resource_variable_ops.assign_add_variable_op(FHandle, Tops.convert_to_tensor(TValue.From<T>(delta), dtype),  name);

    if read_value then
        Exit( gen_resource_variable_ops.read_variable_op(FHandle, dtype) );
    // return _lazy_read(assign_add_op);
    Result := assign_add_op.output;
end;

function BaseResourceVariable.assign_sub<T>(delta: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var assign_sub_op := gen_resource_variable_ops.assign_sub_variable_op(FHandle, Tops.convert_to_tensor(TValue.From<T>(delta), dtype),  name);

    if read_value then
        Exit( gen_resource_variable_ops.read_variable_op(FHandle, dtype) );
    // return _lazy_read(assign_add_op);
    Result := assign_sub_op.output;
end;

function BaseResourceVariable.assign_sub_lazy_load(delta: TFTensor; name: string): IVariableV1;
begin
    var assign_sub_op := gen_resource_variable_ops.assign_sub_variable_op(FHandle, Tops.convert_to_tensor(delta, dtype), name);

    Result := _lazy_read(assign_sub_op, delta);
end;

function BaseResourceVariable.assign_lazy_load(value: TFTensor; name: string): IVariableV1;
begin
    var value_tensor := Tops.convert_to_tensor(value, dtype);
    var assign_op    := gen_resource_variable_ops.assign_variable_op(Fhandle, value_tensor, name);
    var variable     := _lazy_read(assign_op, value_tensor);
    Result := variable;
end;

function BaseResourceVariable._read_variable_op: TFTensor;
begin
    variable_accessed(Self);
    var res := gen_resource_variable_ops.read_variable_op(Fhandle, Fdtype);
    // _maybe_set_handle_data(_dtype, _handle, result);
    // have to set shape when converting to substituent placeholder
    if (res.shape.ndim = -1) or( (res.shape.ndim = 0) and (res.shape.isscalar = false) ) then
    begin
        var p : PInt64 := nil;
        if Length(shape.dims) > 0 then
           p := @shape.dims[0];
        TF_GraphSetTensorShape(res.graph.Handle, res._as_tf_output, p, shape.ndim, tf.Status.Handle);
        tf.Status.RaiseEx;
    end;
    Result := res;
end;

function BaseResourceVariable.read_value: TFTensor;
begin
    var value := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('Read'),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                Result := _read_variable_op();
                                            end );
    Result := array_ops.identity(value);
end;

procedure BaseResourceVariable.StridedSliceAssign(value: TFTensor; slice: ParsedSliceArgs);
begin
    _strided_slice_assign(slice.tPackedBegin, slice.tPackedEnd, slice.tPackedStrides, value);
end;

procedure BaseResourceVariable._strided_slice_assign(tBegin, tEnd, strides, value: TFTensor; name: string; begin_mask, end_mask, ellipsis_mask, new_axis_mask,
  shrink_axis_mask: Integer);
begin
    gen_array_ops.resource_strided_slice_assign(Fhandle, tBegin, tEnd, strides, value,
                begin_mask,
                end_mask,
                ellipsis_mask,
                new_axis_mask,
                shrink_axis_mask)
end;

function BaseResourceVariable.GetDevice: string;
begin
   Result  :=  Fhandle.Device;
end;

function BaseResourceVariable.GetGraph: TFGraph;
begin
    Result := Fhandle.graph;
end;

function BaseResourceVariable.value: TFTensor;
begin
    if Assigned(GraphElement) then Result := GraphElement
    else                           Result := _read_variable_op;

end;

procedure BaseResourceVariable.variable_accessed(variable: BaseResourceVariable);
begin
    if variable.trainable then
    begin
        var st : TStack<ITape> := tf.GetTapeSet;
        for var i:= 0 to st.Count - 1 do
        begin
            var tape := st.List[i] ;
            tape.VariableAccessed(variable as ResourceVariable);
        end;
    end;
end;

function BaseResourceVariable._lazy_read(op: TFOperation; value: TFTensor): IVariableV1;
begin
    variable_accessed(Self);
    Result := _UnreadVariable.Create(Fhandle, Fdtype, Fshape, Fin_graph_mode, Funique_id);
end;

procedure BaseResourceVariable.__init__(ttrainable: Boolean; hHandle: TFTensor; sName, unique_id, handle_name: string);
begin
    Ftrainable   := ttrainable;
    Fhandle_name := handle_name + ':0';
    Funique_id   := unique_id;
    FHandle      := hHandle;
    Fname        := sName;
    // After the handle has been created, set up a way to clean it up when
    // executing eagerly. We'll hold the only reference to the deleter, so that
    // when this object is garbage collected the deleter will be too. This
    // means ResourceVariables can be part of reference cycles without those
    // cycles being uncollectable.
    if Fhandle is TEagerTensor then
    begin
        Handle := Fhandle.EagerTensorHandle
    end else
    begin
        Handle := Fhandle.Handle;
    end;
end;

function BaseResourceVariable.ToString: string;
begin
   if tf.Context.executing_eagerly then
     Result := Format('tf.Variable: %s shape= %s, dtype=%d, numpy=%s',[Name,shape.ToString,Ord(dtype),read_value.numpy.ToString])
   else
      Result := Format('tf.Variable: %s shape= %s, dtype=%d',[Name,shape.ToString,Ord(dtype)]);
end;

function BaseResourceVariable.AsTensor(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    if as_ref then Result := read_value.op.inputs[0]
    else           Result := value;
end;

function BaseResourceVariable.numpy: TNDArray;
begin
    Result := read_value.numpy
end;

{ VariableArgs }

class operator VariableArgs.Initialize(out Dest: VariableArgs);
begin
    Dest.DType          := TF_DataType.DtInvalid;
    Dest.ValidateShape  := true;
    Dest.UseResource    := true;
    Dest.CachingDevice  := '';
    Dest.ImportScope    := '';
    Dest.Synchronization:= TVariableSynchronization.VARIABLE_SYNCHRONIZATION_AUTO;
    Dest.Aggregation    := TVariableAggregation.VARIABLE_AGGREGATION_NONE;
end;

{ PureVariableScope }

constructor PureVariableScope.Create(scope: VariableScope; old_name_scope: string; dtype: TF_DataType);
begin
    Fscope          := scope;
    Fold_name_scope := old_name_scope;
    Fvar_store      := variable_scope._get_default_variable_store();
    Fvar_scope_store:= variable_scope.get_variable_scope_store();
    Fnew_name       := Fscope.name;
    var name_scope : string := Fscope.Fname_scope;
    Fvariable_scope_object :=  VariableScope.Create(Freuse, Fnew_name, name_scope);
    Fcached_variable_scope_object := Fvariable_scope_object;
end;

constructor PureVariableScope.Create(name, old_name_scope: string; dtype: TF_DataType);
begin
    Fname           := name;
    Fold_name_scope := old_name_scope;
    Fvar_store      := variable_scope._get_default_variable_store;
    Fvar_scope_store:= variable_scope.get_variable_scope_store;
end;

destructor PureVariableScope.Destroy;
begin
  if Assigned(Fscope) then Fscope.Free;
  if Assigned(Fvar_store) then Fvar_store.Free;
  if Assigned(Fvar_scope_store) then Fvar_scope_store.Free;
  if Assigned(Fvariable_scope_object) then Fvariable_scope_object.Free;
  if Assigned(Fcached_variable_scope_object) then Fcached_variable_scope_object.Free;

  inherited;
end;

function PureVariableScope.ToVarScope: VariableScope;
begin
    Result := self.Fvariable_scope_object;
end;

procedure PureVariableScope._Enter_;
begin
    Fold := Fvar_scope_store.current_scope;
    if Fscope <> nil   then
    begin
        Fvar_scope_store.open_variable_scope(Fnew_name);
        Fold_subscopes.Create( Fvar_scope_store.variable_scopes_count );
        Fvariable_scope_object := Fcached_variable_scope_object;
    end else
    begin
        if string.IsNullOrEmpty(Fold.name)  then  Fnew_name := Fname
        else                                      Fnew_name := Fold.name + '/' + Fname;
        Freuse := Freuse or Fold.resue;
        var name_scope : string ;
        if Fold_name_scope = '' then name_scope := Fname
        else                         name_scope := Fold_name_scope;
        Fvariable_scope_object := VariableScope.Create(Freuse, Fnew_name, name_scope);
        Fvar_scope_store.open_variable_scope(Fnew_name);
    end;
    Fvar_scope_store.current_scope := Fvariable_scope_object;
    Flast_variable_scope_object    := Fvariable_scope_object;
end;

procedure PureVariableScope._Exit_;
begin
    // If jumping out from a non-prolonged scope, restore counts.
    if Fscope <> nil then  Fvar_scope_store.variable_scopes_count := Fold_subscopes
    else                   Fvar_scope_store.close_variable_subscopes(Fnew_name);
    Fvar_scope_store.current_scope := Fold;
end;

{ variable_scope }

constructor variable_scope.Create(name, default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean);
begin
    Fname               := name;
    Fdefault_name       := default_name;
    Fvalues             := values;
    Fcurrent_name_scope := nil;
    if reuse <> nil then
       Freuse  := reuse^;
    Fuse_resource       := false;
    if (Fdefault_name = '') and (Fname = '') then
       raise TFException.Create('If default_name is None then name is required');
    Fauxiliary_name_scope := auxiliary_name_scope;
end;

constructor variable_scope.Create(scope: VariableScope; default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean);
begin
    Fscope              := scope;
    Fdefault_name       := default_name;
    Fvalues             := values;
    Fcurrent_name_scope := nil;
    if reuse <> nil then
       Freuse  := reuse^;
    Fuse_resource       := false;
    if (Fdefault_name = '') and (Fscope = nil) then
        raise TFException.Create('If default_name is None then scope is required');
    if Fvalues = nil then
        FValues := [];
    Fin_graph_mode := true;
    if Fin_graph_mode then
        Fgraph := Tops._get_graph_from_inputs(Fvalues);

    Fauxiliary_name_scope := auxiliary_name_scope;
end;

class function variable_scope.default_variable_creator(initial_value: TValue; name: string; trainable: PBoolean; collections: TList<string>; dtype: TF_DataType;
  shape: TArray<Integer>; validate_shape: Boolean; use_resource: pBoolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin
    trainable^ := _get_trainable_value(synchronization, trainable^);
    if use_resource = nil then
    begin
        var bUseRes  := get_variable_scope.use_resource;
        use_resource := @bUseRes;
    end;
    if use_resource = nil then
        use_resource := @_DEFAULT_USE_RESOURCE;
    if use_resource^ then
    begin
        var sShape  : TFShape := shape;
        Result := ResourceVariable.Create(@initial_value, trainable^, collections, validate_shape, '', name, nil, dtype, '', VARIABLE_AGGREGATION_NONE, @sShape);
    end else
    begin
        Result := RefVariable.Create(@initial_value, trainable^, collections, validate_shape, '', name, nil, dtype);
    end;
end;

class function variable_scope.get_variable_scope: VariableScope;
begin
    Result := get_variable_scope_store.current_scope;
end;

class function variable_scope.get_variable_scope_store: _VariableScopeStore;
begin
    var ret : _VariableScopeStore ;
    var scope_store: TValue := Tops.get_collection(_VARSCOPESTORE_KEY);
    if (scope_store.TypeInfo = nil) or (scope_store.IsEmpty) then
    begin
        ret := _VariableScopeStore.Create;
        Tops.add_to_collection(_VARSCOPESTORE_KEY, ret);
    end else
    begin
        if scope_store.IsType< TList<RefVariable> > then
        begin
            var values := scope_store.AsType< TList<RefVariable> >;
            ret := values[0].To_VarScopeStore;
        end
        else if scope_store.IsType< TList<_VariableScopeStore> > then
        begin
            var values := scope_store.AsType< TList<_VariableScopeStore> >;
            ret := values[0];
        end else
        begin
           raise TFException.Create('Error! get_variable_scope_store');
        end;
    end;
    Result := ret;
end;

procedure variable_scope._Enter_;
begin
    // If the default graph is building a function, then we should not replace it
    // with the cached graph.
    if Tops.get_default_graph.building_function then
        F_building_function := true
    else
        F_building_function := false;
    if (Fin_graph_mode) and (not F_building_function) then
       Fgraph.as_default;
    Fscope := _enter_scope_uncached;
end;

function variable_scope._enter_scope_uncached: VariableScope;
begin
    var current_name_scope          : TNameScope;
    var pure_variable_scope         : PureVariableScope ;
    var entered_pure_variable_scope : VariableScope;
    if Fauxiliary_name_scope then
        // Create a new name scope later
        current_name_scope := nil
    else begin
        // Reenter the current name scope
        var name_scope : string := Tops.get_name_scope;
        if not string.IsNullOrEmpty(name_scope) then
            // Hack to reenter
            name_scope := name_scope + '/';
        current_name_scope := Tops.name_scope(name_scope);
    end;
    if (not string.IsNullOrEmpty(Fname)) or (Fscope <> nil) then
    begin
        var name_scope : string;
        if Fscope = nil then  name_scope := Fname
        else                  name_scope := Enumerable<String>.Create( Fscope.name.Split(['/']) ).Last;
        if current_name_scope = nil then
            current_name_scope := Tops.name_scope(name_scope);
        current_name_scope._enter_;
        var current_name_scope_name : string := current_name_scope.ToString;
        Fcurrent_name_scope := current_name_scope;
        var old_name_scope : string;
        if Fscope = nil then  old_name_scope := current_name_scope_name
        else                  old_name_scope := Fscope.original_name_scope;
        if Fscope = nil then pure_variable_scope := PureVariableScope.Create(Fname, old_name_scope)
        else                 pure_variable_scope := PureVariableScope.Create(Fscope, old_name_scope);
        pure_variable_scope._enter_;
        entered_pure_variable_scope := pure_variable_scope.ToVarScope;
        Fcached_pure_variable_scope := pure_variable_scope;
        Result := entered_pure_variable_scope;
    end else
    begin
        current_name_scope := Tops.name_scope(Fdefault_name);
        current_name_scope._enter_;
        var current_name_scope_name : string := current_name_scope.ToString;
        Fcurrent_name_scope := current_name_scope;
        var unique_default_name : string := _get_unique_variable_scope(Fdefault_name);
        pure_variable_scope := PureVariableScope.Create(unique_default_name, current_name_scope_name);
        pure_variable_scope._enter_;
        entered_pure_variable_scope := pure_variable_scope.ToVarScope;
        Fcached_pure_variable_scope := pure_variable_scope;
        Result := entered_pure_variable_scope;
    end;
end;

procedure variable_scope._Exit_;
begin
    Fcached_pure_variable_scope._exit_;
    if Fcurrent_name_scope <> nil then
        Fcurrent_name_scope._exit_;
end;

class function variable_scope._get_default_variable_store: _VariableStore;
begin
   var store := Tops.get_collection(_VARSTORE_KEY);
   if (store.TypeInfo <> nil) and (not store.IsEmpty) then
   begin
      Result := (store.AsType< TList<_VariableStore> >)[0];
      Exit;
   end;
   var store1 := _VariableStore.Create;
   Tops.add_to_collection(_VARSTORE_KEY, store1);
   Result := store1;
end;

class function variable_scope._get_trainable_value(synchronization: TVariableSynchronization; trainable: Boolean): Boolean;
begin
    if synchronization = VARIABLE_SYNCHRONIZATION_ON_READ then
    begin
        if trainable then
           raise TFException.Create('Synchronization value can be set to ' +
                                    'VariableSynchronization.ON_READ only for non-trainable variables. ' +
                                    'You have specified trainable=True and ' +
                                    'synchronization=VariableSynchronization.ON_READ.');
    end ;
    { TODO -oMax -c : Verificare "Nullable(record!!) del Ca...o" 02/11/2022 16:46:22 }
    (*else if (!trainable.HasValue)
    {
        trainable = true;
    }*)
    Result :=  trainable;
end;

class function variable_scope._get_unique_variable_scope(prefix: string): string;
begin
    var var_scope_store := get_variable_scope_store;
    var current_scope   := get_variable_scope;
    var name : string;
    if not string.IsNullOrEmpty(current_scope.name) then name := current_scope.name + '/' + prefix
    else                                                 name := prefix;
    if var_scope_store.variable_scope_count(name) = 0 then
    begin
        Result := prefix;
        Exit;
    end;
    var idx : Integer := 1;
    while var_scope_store.variable_scope_count( name+'_'+ IntToStr(idx) ) > 0 do
        idx := idx + 1;
    Result := prefix+'_'+ IntToStr(idx);
end;

{ VariableScope }

constructor VariableScope.Create(reuse: Boolean; _name, name_scope: string; dtype: TF_DataType);
begin
    Fname       := _name;
    Fname_scope := name_scope;
    Freuse      := _ReuseMode.AUTO_REUSE;
    Fdtype      := dtype;
end;

function VariableScope.get_variable(var_store: _VariableStore; name: string; shape: PTFShape; dtype: TF_DataType; initializer: TObject; trainable: PBoolean;
  collections: TList<string>; use_resource: PBoolean; validate_shape: Boolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin
    var full_name : string;
    if not string.IsNullOrEmpty(Self.name) then  full_name := Self.name +'/' + name
    else                                         full_name := name;

     Result := TUtils.tf_with<TNameScope,IVariableV1>( TOps.name_scope(''),
                  function(v1: TNameScope): IVariableV1
                    begin
                        if dtype = TF_DataType.DtInvalid then
                            dtype := Fdtype;
                        Result := var_store.get_variable(full_name, shape, dtype, initializer, @resue, trainable, collections, True, synchronization, aggregation);
                    end );
end;

procedure VariableScope.reuse_variables;
begin
    Freuse := _ReuseMode.AUTO_REUSE;
end;

{ _VariableStore }

constructor _VariableStore.Create;
begin
    Fvars                  := TDictionary<string, TObject>.Create ;
    Fpartitioned_vars      := TDictionary<string, TObject>.Create ;
    Fstore_eager_variables := false;
end;

destructor _VariableStore.Destroy;
begin
  Fvars.Free;
  Fpartitioned_vars.Free;

  inherited;
end;

function _VariableStore.get_variable(name: string; shape: PTFShape; dtype: TF_DataType; initializer: TObject; reuse, trainable: PBoolean; collections: TList<string>;
  validate_shape: Boolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin
    dtype     := TDtypes.as_base_dtype(dtype);
    trainable^:= variable_scope._get_trainable_value(synchronization, trainable^);
    Result := _true_getter(name, shape, dtype, initializer, trainable, collections, validate_shape, synchronization, aggregation);
end;

function _VariableStore._get_single_variable(name: string; shape: PTFShape; dtype: TF_DataType; initializer: IInitializer; init_value: TFTensor; reuse: Boolean;
  trainable: PBoolean; collections: TList<string>; validate_shape: Boolean; use_resource: PBoolean; synchronization: TVariableSynchronization;
  aggregation: TVariableAggregation): IVariableV1;
begin
    {$HINTS OFF}
    var initializing_from_value : Boolean := init_value <> nil;
    if use_resource = nil then
        use_resource := @variable_scope._DEFAULT_USE_RESOURCE;
    if Fvars.ContainsKey(name) then
    begin
        if not reuse then
        begin
            var _var := Fvars[name];
        end;
        raise Exception.Create('Not Implemented _get_single_variable');
    end;
    var v : IVariableV1 := nil;
    // Create the tensor to initialize the variable with default value.
    if (initializer = nil) and (init_value = nil) then
    begin
        if TDTypes.is_floating(dtype) then
        begin
            initializer := tf.glorot_uniform_initializer;
            initializing_from_value := false;
        end;
    end;
    // Create the variable.
    Tops.init_scope;
    if initializing_from_value then
    begin
        var pinit_value : TValue  := init_value;
        v := ResourceVariable.Create(@pinit_value, trainable^,nil,validate_shape,'',name);
    end else
    begin
        var init_val := initializer.Apply( InitializerArgs.Create(shape, dtype) );
        var variable_dtype := TDTypes.as_base_dtype(dtype);

        v := variable_scope.default_variable_creator(init_val, name, trainable, collections, variable_dtype, nil, validate_shape, use_resource, synchronization, aggregation);
    end;

    Fvars.AddOrSetValue(name, TObject(v));
    Result := v;
end;

function _VariableStore._true_getter(name: string; shape: PTFShape; dtype: TF_DataType; initializer: TObject; trainable: PBoolean; collections: TList<string>;
  validate_shape: Boolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin
    if initializer is IInitializer then
    begin
        var init := initializer as IInitializer;
        Result := _get_single_variable (name, shape, dtype, init, nil, False, trainable, collections, validate_shape, nil, synchronization, aggregation);
    end
    else if initializer is TFTensor  then
    begin
        var tensor : TFTensor := TFTensor(initializer);
        Result := _get_single_variable(name, shape, dtype, nil, tensor, False, trainable, nil, validate_shape, nil, synchronization, aggregation);
    end else
    begin
        var init1 : IInitializer := nil;
        Result := _get_single_variable(name, shape, dtype, init1, nil, False, trainable, nil, validate_shape, nil, synchronization, aggregation);
    end;
end;

{ _VariableScopeStore }

constructor _VariableScopeStore.Create;
begin
    current_scope         := VariableScope.Create(false);
    variable_scopes_count := TDictionary<string, Integer>.Create;
end;

destructor _VariableScopeStore.Destroy;
begin
    current_scope.Free;
    variable_scopes_count.Free;
end;

procedure _VariableScopeStore.close_variable_subscopes(scope_name: string);
begin
    var variable_scopes_count_tmp := TDictionary<string, Integer>.Create;
    try
      for  var k in variable_scopes_count.Keys do
          variable_scopes_count_tmp.Add(k, variable_scopes_count[k]);
      for var k in variable_scopes_count_tmp.Keys do
          if (scope_name = '') or ( k.StartsWith(scope_name + '/') ) then
              variable_scopes_count[k] := 0;
    finally
      variable_scopes_count_tmp.Free;
    end;
end;

procedure _VariableScopeStore.open_variable_scope(scope_name: string);
begin
    if variable_scopes_count.ContainsKey(scope_name) then
        variable_scopes_count[scope_name] := variable_scopes_count[scope_name] + 1
    else
        variable_scopes_count[scope_name] := 1;
end;

function _VariableScopeStore.variable_scope_count(scope_name: string): Integer;
begin
    if variable_scopes_count.ContainsKey(scope_name) then
        Result := variable_scopes_count[scope_name]
    else
        Result := 0;
end;
{$ENDREGION}

initialization
begin
    random_seed.Fgraph_to_seed_dict := TDictionary<string,Integer>.Create;
end;

finalization
begin
    random_seed.Fgraph_to_seed_dict.Free;
end;


end.
