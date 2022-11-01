unit TensorFlow.Variable;
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
     uses System.SysUtils,
          System.Rtti,
          System.TypInfo,

          Spring,
          Spring.Collections.Lists,

          TensorFlow.Slice,
          TensorFlow.Initializer,
          TF4D.Core.CApi,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          Tensorflow.NameScope,
          TensorFlow.EagerTensor,

          ProtoGen.tensorShape,
          ProtoGen.variable,
          ProtoGen.attrValue;

type

   IVariableV1 = interface;

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
        constructor Create(reuse: Boolean; name: string = ''; name_scope: string = ''; dtype: TF_DataType = TF_FLOAT) ;
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

   end;

   PureVariableScope = class(TInterfacedObject, ITensorFlowObject)
     private

     public
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
        F_current_name_scope       : TNameScope;
        Fauxiliary_name_scope      : Boolean;
        Fcached_pure_variable_scope: PureVariableScope;
        F_reuse                    : Nullable<Boolean>;
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

        property UseResource : Boolean read Fuse_resource;
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
  ['{DEBD12E5-E613-4F9A-AEDC-99579EFA9798}']

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

      function AsTensor(dtype: TF_DataType = TF_DataType.DtInvalid; name : string= ''; as_ref : Boolean= false): TFTensor;
      function numpy: TNDArray;

      property UniqueId    : string      read GetUniqueId;
      property Name        : string      read GetName;
      property Handle      : TFTensor    read GetHandle;
      property Device      : String      read GetDevice;
      property Initializer : TFOperation read GetInitializer;
      property Op          : TFOperation read GetOP;
      property GraphElement: TFTEnsor    read GetGraphEle;
      property Graph       : TFGraph     read GetGraph;
      property Shape       : TFShape     read GetShape;
      property dtype       : TF_DataType read GetTipo;
  end;

  RefVariable = class(TInterfacedObject, IVariableV1)
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
        Fshape         : TFShape;
        function GetGraph: TFGraph;
        function GetDevice: string;
      protected
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
  _UnreadVariable = class( BaseResourceVariable, IVariableV1 )
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
       function _dense_var_to_tensor(dtype: TF_DataType = TF_DataType.DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
     protected
        Finitial_value : TFTensor;
        FShape         : TFShape;

     public
        constructor Create(_initial_value    : PValue;
                           _trainable        : Boolean= true;
                           collections      : TList<string>= nil;
                           validate_shape   : Boolean = true;
                           caching_device   : string = '';
                           name             : string= '';
                           variable_def     : PVariableDef= nil;
                           dtype            : TF_DataType = TF_DataType.DtInvalid;
                           import_scope     : string = '';
                           aggregation      : TVariableAggregation = TVariableAggregation.VARIABLE_AGGREGATION_NONE;
                           shape            : PTFShape= nil);
        function  _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
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

  variables = class
    private

    public
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        class  function trainable_variables : TValue;
        /// <summary>
        /// Returns all variables and `SaveableObject`s that must be checkpointed.
        /// </summary>
        /// <param name="scope"></param>
        /// <returns></returns>
        class  function _all_saveable_objects(scope: string = ''): TArray<IVariableV1>;
        /// <summary>
        /// Returns global variables.
        /// </summary>
        /// <param name="scope">
        /// (Optional.) A string. If supplied, the resulting list is filtered
        /// to include only items whose `name` attribute matches `scope` using
        /// `re.match`. Items without a `name` attribute are never returned if a
        /// scope is supplied. The choice of `re.match` means that a `scope` without
        /// special tokens filters by prefix.
        /// </param>
        /// <returns>A list of `Variable` objects.</returns>
        class  function  global_variables(scope: string = ''): TList<IVariableV1>;
        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        class  function  variables_initializer(var_list: TArray<IVariableV1>; name: string = 'init'): TFOperation;
  end;

  state_ops = class
    private

    public
      /// <summary>
      /// Create a variable Operation.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <returns></returns>
      class function variable_op_v2(shape: TArray<Integer>; dtype: TF_DataType; name: string = 'Variable'; container : string= ''; shared_name: string = ''): TFTensor;
      class function assign<T>(ref: T; value: TValue; validate_shape: Boolean = true; use_locking: Boolean = true; name: string = ''): TFTensor; overload;
      class function assign(ref: IVariableV1; value: TValue; validate_shape: Boolean = true; use_locking: Boolean = true; name: string = '') : TFTensor; overload;
      class function assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean = false; name : string= ''): TFTensor;
      //"""Update 'ref' by adding 'value' to it.
      //
      //  This operation outputs "ref" after the update is done.
      //  This makes it easier to chain operations that need to use the reset value.
      //
      //  Args:
      //    ref: A mutable `Tensor`. Must be one of the following types:
      //      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
      //      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      //      Should be from a `Variable` node.
      //    value: A `Tensor`. Must have the same type as `ref`.
      //      The value to be added to the variable.
      //    use_locking: An optional `bool`. Defaults to `False`.
      //      If True, the addition will be protected by a lock;
      //      otherwise the behavior is undefined, but may exhibit less contention.
      //    name: A name for the operation (optional).
      //
      //  Returns:
      //    Same as "ref".  Returned as a convenience for operations that want
      //    to use the new value after the variable has been updated.
      class function assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean = false; name: string = ''): TFTensor;
      class function scatter_add(ref: IVariableV1; indices: TFTensor; updates: TFTensor; use_locking: Boolean = false; name: string = '')  : TFTensor;
      class function is_variable_initialized(ref: RefVariable; name: string = '') : TFTensor;
  end;

/// <summary>
/// Wrapper record for ResourceVariable for Operator Overloading
/// </summary>
TResourceVariable = record
  private
      FResourceHandle : ResourceVariable;

      function value: TFTensor;
  public
      function assign<T>(value: T; use_locking: Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
      function numpy: TNDArray;

      class operator Implicit(t : ResourceVariable): TResourceVariable;
      class operator Implicit(t : TResourceVariable): ResourceVariable;
      class operator Implicit(t : TResourceVariable): TFTensor;
      class operator Implicit(t : TResourceVariable): TEagerTensor;

      class operator Add(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Add(x: TResourceVariable; y: Single): TFTensor;
      class operator Add(x: TResourceVariable; y: Double) : TFTensor;
      class operator Add(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Add(x: TResourceVariable; y: TFTensor) : TFTensor;
      //
      class operator Subtract(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: Single): TFTensor;
      class operator Subtract(x: TResourceVariable; y: Double) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: TFTensor) : TFTensor;
      //
      class operator Multiply(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: Single): TFTensor;
      class operator Multiply(x: TResourceVariable; y: Double) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TFTensor) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TNDArray) : TFTensor;
      //
      class operator LessThan(x: TResourceVariable; y: TFTensor) : TFTensor;
      class operator GreaterThan(x: TResourceVariable; y: TFTensor) : TFTensor;
end;


implementation
     uses Oz.Pb.Classes,
          Spring.Collections.Stacks,

          Tensorflow.Gradient,
          TensorFlow.Tensor,

          Tensorflow,
          Tensorflow.Utils,
          TensorFlow.Ops,
          TensorFlow.EagareRunner,
          Tensorflow.array_ops,
          Tensorflow.gen_array_ops,
          TensorFlow.gen_state_ops,
          TensorFlow.control_flow_ops,
          TensorFlow.gen_control_flow_ops,
          TensorFlow.gen_resource_variable_ops,
          TensorFlow.resource_variable_ops;


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

{ ResourceVariable }

constructor ResourceVariable.Create(_initial_value: PValue; _trainable: Boolean; collections: TList<string>; validate_shape: Boolean; caching_device, name: string;
                                       variable_def: PVariableDef; dtype: TF_DataType; import_scope: string; aggregation: TVariableAggregation; shape: PTFShape);
begin
    if Assigned(variable_def) then
    begin
        if Assigned(_initial_value) then
           raise  TFException.Create('variable_def and initial_value are mutually exclusive.');
        _init_from_proto(variable_def^, import_scope);
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
                      var attr : TAttrValue; attr.Init;
                      var lst : TListValue; lst.Init;
                      var b := TEncoding.UTF8.GetBytes('loc:@'+handle_name);
                      lst.Ss.Add(@b);
                      var v : TpbOneof;
                      v.tag := TAttrValue.ftList;
                      v.value := TValue.From<TListValue>(lst);
                      attr.Value := v;

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
                                       var func :=  initial_value.AsType<TFunc<TFTensor>>;
                                       value := TValue.From<TFunc<TFTensor>>(func)
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
        var var_def : TVariableDef; var_def.Init;
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
     Result := FHandle.Device;
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
      tf.Runner.TFE_Execute(tf.Context, AnsiString(Fhandle.Device), '"DestroyResourceOp',[ FHandle ], ['ignore_lookup_error', true ], 0);
end;


function BaseResourceVariable.assign<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var vValue := TValue.From<T>(value) ;
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
    if res.shape.ndim = -1 then
    begin
        var p : PInt64 := nil;
        if Length(shape.dims) > 0 then
           p := PInt64(shape.dims[0]);
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
   Result := Fhandle.Device;
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
            var tape := st.ElementAt(i) ;
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

{ variables }

class function variables.global_variables(scope: string): TList<IVariableV1>;
begin
    Result := Tops.get_collection<IVariableV1>(tf.GraphKeys.GLOBAL_VARIABLES, scope);
end;

class function variables.trainable_variables: TValue;
begin
    Result := Tops.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES);
end;

class function variables.variables_initializer(var_list: TArray<IVariableV1>; name: string): TFOperation;
begin
    if Length(var_list) > 0 then
    begin
        var opList : TArray<TFOperation> := [];
        for var i := 0 to Length(var_list)- 1 do
          opList := opList + [ var_list[i].Initializer ] ;
        Result :=  control_flow_ops.group<TFOperation>(opList, name) ;
    end else
    begin
        Result := gen_control_flow_ops.no_op(name);
    end;
end;

class function variables._all_saveable_objects(scope: string): TArray<IVariableV1>;
begin
    var all := TList<IVariableV1>.Create;
    try

      all.AddRange(Tops.get_collection<IVariableV1>(tf.GraphKeys.GLOBAL_VARIABLES, scope));
      all.AddRange(Tops.get_collection<IVariableV1>(tf.GraphKeys.SAVEABLE_OBJECTS, scope));
      Result :=  all.ToArray;
    finally
      all.free;
    end;
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

{ state_ops }

class function state_ops.assign(ref: IVariableV1; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
        Result := gen_state_ops.assign(ref, value,  validate_shape, use_locking, name)
    else begin
        if      ref is RefVariable          then Result := (ref as RefVariable).assign(value, False , name)
        else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign(value, False ,name)
        else
           raise Exception.Create('state_ops.assign Error!');
    end;
end;

class function state_ops.assign<T>(ref: T; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
    Result := gen_state_ops.assign(ref, value, validate_shape, use_locking, name)
end;

class function state_ops.assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean; name: string): TFTensor;
begin
    if tf.executing_eagerly then
    begin
        if      ref is RefVariable          then Result := (ref as RefVariable).assign_add(value, use_locking , name)
        else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign_add(value, use_locking ,name)
    end
    else
        Result := gen_state_ops.assign_add(ref, value, use_locking, name);
end;

class function state_ops.assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
      Result := gen_state_ops.assign_sub(ref, value, use_locking, name)
    else begin
      if      ref is RefVariable          then Result := (ref as RefVariable).assign_sub(value, use_locking , name)
      else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign_sub(value, use_locking ,name)
      else
           raise Exception.Create('state_ops.assign_sub Error!');
    end;
end;

class function state_ops.is_variable_initialized(ref: RefVariable; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
    begin
        Result := gen_state_ops.is_variable_initialized(ref, name);
        Exit;
    end;
    raise TFException.Create('Not Implemented - is_variable_initialized');
end;

class function state_ops.scatter_add(ref: IVariableV1; indices, updates: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
    begin
        Result := gen_state_ops.scatter_add(ref, indices, updates, use_locking, name);
        Exit;
    end;
    raise TFException.Create('Not Implemented - scatter_add');
end;

class function state_ops.variable_op_v2(shape: TArray<Integer>; dtype: TF_DataType; name, container, shared_name: string): TFTensor;
begin
    Result := gen_state_ops.variable_v2(shape, dtype, name, container,shared_name)
end;

{ TResourceVariable }

class operator TResourceVariable.Implicit(t: TResourceVariable): ResourceVariable;
begin
    Result := t.FResourceHandle;
end;

function TResourceVariable.numpy: TNDArray;
begin
    Result := FResourceHandle.numpy;
end;

function TResourceVariable.value: TFTensor;
begin
    Result := FResourceHandle.value;
end;

class operator TResourceVariable.Implicit(t: ResourceVariable): TResourceVariable;
begin
    Result.FResourceHandle := t;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t + t1;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t - t1;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t * t1;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: TNDArray): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.LessThan(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t < y;
end;

class operator TResourceVariable.GreaterThan(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t > y;
end;

function TResourceVariable.assign<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    Result := FResourceHandle.assign<T>(value,use_locking, name, read_value)
end;

class operator TResourceVariable.Implicit(t: TResourceVariable): TEagerTensor;
begin
    Result := t.FResourceHandle._dense_var_to_tensor as TEagerTensor;
end;

class operator TResourceVariable.Implicit(t: TResourceVariable): TFTensor;
begin
    Result := t.FResourceHandle._dense_var_to_tensor;
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

procedure PureVariableScope._Enter_;
begin

end;

procedure PureVariableScope._Exit_;
begin

end;

{ variable_scope }

constructor variable_scope.Create(name, default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean);
begin

end;

constructor variable_scope.Create(scope: VariableScope; default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean);
begin

end;

class function variable_scope.default_variable_creator(initial_value: TValue; name: string; trainable: PBoolean; collections: TList<string>; dtype: TF_DataType;
  shape: TArray<Integer>; validate_shape: Boolean; use_resource: pBoolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin

end;

class function variable_scope.get_variable_scope: VariableScope;
begin

end;

class function variable_scope.get_variable_scope_store: _VariableScopeStore;
begin

end;

procedure variable_scope._Enter_;
begin

end;

function variable_scope._enter_scope_uncached: VariableScope;
begin

end;

procedure variable_scope._Exit_;
begin

end;

class function variable_scope._get_default_variable_store: _VariableStore;
begin

end;

class function variable_scope._get_trainable_value(synchronization: TVariableSynchronization; trainable: Boolean): Boolean;
begin

end;

class function variable_scope._get_unique_variable_scope(prefix: string): string;
begin

end;

{ VariableScope }

constructor VariableScope.Create(reuse: Boolean; name, name_scope: string; dtype: TF_DataType);
begin

end;

function VariableScope.get_variable(var_store: _VariableStore; name: string; shape: PTFShape; dtype: TF_DataType; initializer: TObject; trainable: PBoolean;
  collections: TList<string>; use_resource: PBoolean; validate_shape: Boolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin

end;

procedure VariableScope.reuse_variables;
begin

end;

end.


