unit TensorFlow.Variable;

interface
     uses System.SysUtils,
          System.Rtti,

          spring.Collections.Lists,

          TF4D.Core.CApi,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          NumPy.NDArray;

type
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
        function assign(value: TValue; use_locking: Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
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
        function numpy: TNDArray;
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
     protected
        Fname          : string;
        Fhandle_name   : string;
        Fdtype         : TF_DataType ;
        Funique_id     : string;
        Fin_graph_mode : Boolean;
        Ftrainable     : Boolean;
        Finitial_value : TFTensor;
        Finitializer_op: TFOperation;
        Fparent_op     : TFTEnsor;
        FHandle        : TFTEnsor;
        Fgraph_element : TFTEnsor;
        FShape         : TFShape;
     public
        function  _TensorConversionFunction(dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;

        property Name        : string       read GetName;
        property dtype       : TF_DataType  read GetTipo;
        property UniqueId    : string       read GetUniqueId;
        property trainable   : Boolean      read Ftrainable;
        property Initializer : TFOperation  read GetInitializer;
        property parent_op   : TFTEnsor     read Fparent_op;
        property GraphElement: TFTEnsor     read GetGraphEle;
        property Shape       : TFShape      read GetShape;
        property Op          : TFOperation  read GetOP;
        property Graph       : TFGraph      read GetGraph;
        property Device      : String       read GetDevice;
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


implementation
     uses Tensorflow,
          Tensorflow.Utils,
          TensorFlow.Ops,
          Tensorflow.NameScope,
          TensorFlow.EagerTensor,
          TensorFlow.EagareRunner,
          Tensorflow.array_ops,
          Tensorflow.gen_array_ops,
          TensorFlow.gen_state_ops,
          TensorFlow.control_flow_ops,
          TensorFlow.gen_control_flow_ops,
          TensorFlow.gen_resource_variable_ops;


{ RefVariable }

function RefVariable.assign(value: TValue; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var assign := gen_state_ops.assign(_variable, value, True, use_locking, name);
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

 function ResourceVariable.GetHandle:TFTensor;
 begin
     Result := FHandle;
 end;

function ResourceVariable.GetInitializer: TFOperation;
begin
   Result := Finitializer_op;
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

function ResourceVariable.GetShape: TFShape;
begin
    Result := FShape
end;

function ResourceVariable.GetTipo: TF_DataType;
begin
    Result := Fdtype
end;

function ResourceVariable.GetUniqueId: string;
begin
    Result := Funique_id
end;

function ResourceVariable._TensorConversionFunction(dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin

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
    if TypeInfo(T) = TypeInfo(TFTensor) then
    begin
        var assign := gen_state_ops.assign(handle, TValue.From<T>(value), True,use_locking, name);
        if read_value then
            Exit(assign);
        Exit( assign.op.Output );
    end;
    var value_tensor := Tops.convert_to_tensor(TValue.From<T>(value), dtype);
    var assign_op    := gen_resource_variable_ops.assign_variable_op(handle, value_tensor, name);
    if read_value then
    begin
        Result := gen_resource_variable_ops.read_variable_op(handle, dtype);
        Exit;
    end;
    if assign_op = nil then
        Exit(nil);
    Result := assign_op.output;
end;

function BaseResourceVariable.assign_add<T>(delta: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var assign_add_op := gen_resource_variable_ops.assign_add_variable_op(Handle, Tops.convert_to_tensor(TValue.From<T>(delta), dtype),  name);

    if read_value then
        Exit( gen_resource_variable_ops.read_variable_op(handle, dtype) );
    // return _lazy_read(assign_add_op);
    Result := assign_add_op.output;
end;

function BaseResourceVariable.assign_sub<T>(delta: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    var assign_sub_op := gen_resource_variable_ops.assign_sub_variable_op(Handle, Tops.convert_to_tensor(TValue.From<T>(delta), dtype),  name);

    if read_value then
        Exit( gen_resource_variable_ops.read_variable_op(handle, dtype) );
    // return _lazy_read(assign_add_op);
    Result := assign_sub_op.output;
end;

function BaseResourceVariable.assign_sub_lazy_load(delta: TFTensor; name: string): IVariableV1;
begin
    var assign_sub_op := gen_resource_variable_ops.assign_sub_variable_op(Handle, Tops.convert_to_tensor(delta, dtype), name);

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
    var op := gen_array_ops.resource_strided_slice_assign(Fhandle, tBegin, tEnd, strides, value,
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
        for var tape in tf.GetTapeSet do
            tape.VariableAccessed(variable as ResourceVariable);
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

end.
