unit TensorFlow.Ops;

interface
     uses System.SysUtils, System.SyncObjs, Winapi.Windows, System.Rtti,
          Spring,
          Spring.Collections, spring.Collections.MultiMaps, Spring.Collections.Enumerable, Spring.Collections.Extensions,
          system.Generics.Collections,
          TensorFlow.LowLevelAPI,
          NDArray,
          TensorFlow.Context,
          Tensorflow.Graph,
          TensorFlow.Variable,
          TensorFlow.DApi,
          TensorFlow.DApiBase,
          Tensorflow,

          ProtoGen.attrValue,
          ProtoGen.opDef,
          ProtoGen.nodeDef;

type
  /// <summary>
  /// Feed dictionary item
  /// </summary>
  FeedItem = class

  end;

  /// <summary>
  /// Returns a context manager that creates hierarchical names for operations.
  /// </summary>
  TNameScope = class(TInterfacedObject,ITensorFlowObject)
    private
      function enter_eager_name_scope(ctx: TContext; name:TFString): Tuple<TFString,TFString>;
    public
      _name          : TFString;
      _default_name  : TFString;
      _values        : TValue;
      scope_name     : TFString;
      old_scope_name : TFString;
      _skip_on_eager : boolean;

      constructor Create(name: TFString; default_name : TFString = ''; values : PValue = nil; skip_on_eager : Boolean = True);
      function ToString: TFString; reintroduce;
      procedure _Enter_;
      procedure _Exit_;
  end;

  TOps = class
    private
      FisSingleThreaded   : Boolean;
      F_singleSesson      : TFSession;
      F_singleGraphStack  : DefaultGraphStack;
      FLock               : TCriticalSection;

      class var Fdefault_graph_stack: DefaultGraphStack;
      class var Fuid_number      : Integer;
      class var Fgraph_uid_number: Integer;
      class var Fuid_number_for_function : Integer;

      procedure SetSingleThread(const Value: Boolean);
      function  Get_default_graph_stack: DefaultGraphStack;

    public

      function  Tensor_Id(tensor: TFTensor): Int64;
      procedure Add_to_collection<T>(name: string; value: T);overload;
      procedure Add_to_collection<T>(names: TList<string>; value: T);overload;
      /// <summary>
      /// Wrapper for `Graph.get_collection()` using the default graph.
      /// contains many standard names for collections.
      /// </summary>
      /// <param name="key">
      /// The key for the collection. For example, the `GraphKeys` class
      /// </param>
      /// <param name="scope"></param>
      /// <returns>
      /// The list of values in the collection with the given `name`, or
      /// an empty list if no value has been added to that collection. The
      /// list contains the values in the order under which they were
      /// collected.
      /// </returns>
      function get_collection(key: string; scope : string = ''): TObject;overload;
      function get_collection<T>(key: string; scope : string = ''): TList<T>;overload;
      function get_collection_ref<T>(key: string): TList<T>;overload;
      class function _get_graph_from_inputs(op_input_list: TArray<TFTensor>): TFGraph; overload;
      class function _get_graph_from_inputs(op_input_list: TArray<TValue>): TFGraph; overload;
      class function _get_graph_from_inputs(op_input_list: TFTensor): TFGraph ;  overload;
      class function _get_graph_from_inputs(op_input_list: TFTensor; graph: TFGraph = nil): TFGraph ;  overload;
      class function name_scope(name: TFString; default_name: TFString = ''; values : PValue= nil; skip_on_eager: Boolean = true): TNameScope; static;

      /// <summary>
      /// Converts the given `value` to a `Tensor`.
      /// </summary>
      /// <param name="value"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function convert_to_tensor(value: TValue; dtype : TF_DataType = TF_DATATYPE_UNKNOWN; name: string= ''; as_ref: Boolean = False; preferred_dtype : TF_DataType = TF_DATATYPE_UNKNOWN; ctx: TContext= nil): TFTensor;
      function convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''): TFTensor;
      function internal_convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''; as_ref: Boolean = false): TFTensor;
      /// <summary>
      /// Wrapper for `Graph.control_dependencies()` using the default graph.
      ///
      /// See `tf.Graph.control_dependencies` for more details.
      ///
      /// When eager execution is enabled, any callable object in the `control_inputs`
      /// list will be called.
      /// </summary>
      /// <param name="control_inputs">
      /// A list of `Operation` or `Tensor` objects which
      /// must be executed or computed before running the operations
      /// defined in the context.Can also be `None` to clear the control
      /// dependencies.If eager execution is enabled, any callable object in the
      /// `control_inputs` list will be called.
      /// </param>
      /// <returns>
      /// A context manager that specifies control dependencies for all
      /// operations constructed within the context.
      /// </returns>
      function control_dependencies(control_inputs: TArray<TObject>): _ControlDependenciesController;
      /// <summary>
      /// Creates a TF_Operation.
      /// </summary>
      /// <param name="graph">a `Graph`.</param>
      /// <param name="node_def">`node_def_pb2.NodeDef` for the operation to create.</param>
      /// <param name="inputs">
      /// A list of `Tensor`s (corresponding to scalar inputs) and lists of
      /// `Tensor`s (corresponding to sequence inputs, e.g. "int64 * N",
      /// "list(int64)"). The length of the list should be equal to the number of
      /// inputs specified by this operation's op def.
      /// </param>
      /// <param name="control_inputs">A list of `Operation`s to set as control dependencies.</param>
      /// <returns>A wrapped TF_Operation*.</returns>
      class function _create_c_op(graph: TFGraph; node_def: TNodeDef; inputs: TArray<TFTensor>; control_inputs: TArray<TFOperation>; op_def : POpDef ) : Tuple<Pointer, TFOperationDesc>;
      class function _reconstruct_sequence_inputs(op_def: TOpDef; inputs: TArray<TFTensor>; attrs: TMultiMap<string, TAttrValue>): TArray<TTensors>;
      function _get_op_def(graph: TFGraph; tipo: string): TOpDef;
      class function _NodeDef(op_type: string; name: string; attrs : TDictionary<string, TAttrValue> = nil): TNodeDef;
      class function name_from_scope_name(name: string): string;
      /// <summary>
      /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
      /// </summary>
      /// <returns></returns>
      function init_scope: TNameScope;
      /// <summary>
      /// A unique (within this program execution) integer.
      /// Not thread safe
      /// </summary>
      /// <returns></returns>
      class function uid: Integer;
      class function GraphUniqueId: Integer;
      class function uid_function: Integer;
      class procedure reset_uid;
      //
      procedure colocate_with(ignore_existing : Boolean = false); overload;
      procedure colocate_with(op: TFOperation; ignore_existing : Boolean = false); overload;
      procedure colocate_with(tensor: TFTensor; ignore_existing : Boolean= false); overload;
      procedure colocate_with(variable: IVariableV1; ignore_existing : Boolean = false); overload;
      procedure _colocate_with_for_gradient(op: TFOperation; gradient_uid: string; ignore_existing : Boolean = false);
      /// <summary>
      /// Uses the default session to evaluate one or more tensors.
      /// </summary>
      /// <param name="tensor">A single Tensor, or a list of Tensor objects.</param>
      /// <param name="feed_dict">
      /// A dictionary that maps Tensor objects (or tensor names) to lists,
      /// numpy ndarrays, TensorProtos, or strings.
      /// </param>
      /// <param name="graph">The graph in which the tensors are defined.</param>
      /// <param name="session">A different session to use to evaluate "tensors".</param>
      /// <returns>
      /// Either a single numpy ndarray if "tensors" is a single tensor; or a list
      /// of numpy ndarrays that each correspond to the respective element in
      /// "tensors".
      /// </returns>
      function  _eval_using_default_session(tensor: TFTensor; feed_dict : TArray<FeedItem>; graph: TFGraph; session : TFSession = nil): TNDArray;
      function  prepend_name_scope(name: string; import_scope: string): string;
      procedure _run_using_default_session(operation: TFOperation; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession);
      function  convert_n_to_tensor(values: TObject; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name : string = ''): TArray<TFTensor>;
      function  convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''): TArray<TFTensor>;
      function  convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''): TFTensor;
      function  internal_convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''; as_ref : Boolean = false): TFTensor;
      function  internal_convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name : string= ''; as_ref : Boolean= false): TArray<TFTensor>;
      class function  internal_convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''; preferred_dtype : TF_DataType = TF_DATATYPE_UNKNOWN; as_ref: Boolean = false):TArray<TFTensor>;
      function  strip_name_scope(name: string; export_scope: string = ''): string;
      function  get_name_scope: string;
      function  executing_eagerly_outside_functions: Boolean;

      // array_ops Class
      class function constant(value: TValue; dtype: TF_DataType= TF_DATATYPE_UNKNOWN; shape: TArray<TF_int64_t> =[]; name: AnsiString= 'Const'; verify_shape: Boolean=False): TFTensor; static;


      (*// threading*)

      /// <summary>
      ///     Clears the default graph stack and resets the global default graph.
      ///
      ///     NOTE: The default graph is a property of the current thread.This
      ///     function applies only to the current thread.Calling this function while
      ///     a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
      ///     behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
      ///     after calling this function will result in undefined behavior.
      /// </summary>
      /// <returns></returns>
      class procedure reset_default_graph;
      /// <summary>
      ///     Returns the default graph for the current thread.
      ///
      ///     The returned graph will be the innermost graph on which a
      ///     `Graph.as_default()` context has been entered, or a global default
      ///     graph if none has been explicitly created.
      ///
      ///     NOTE: The default graph is a property of the current thread.If you
      ///     create a new thread, and wish to use the default graph in that
      ///     thread, you must explicitly add a `with g.as_default():` in that
      ///     thread's function.
      /// </summary>
      /// <returns></returns>
      class function get_default_graph: TFGraph;
      /// <summary>
      /// Returns the default session for the current thread.
      /// </summary>
      /// <returns>The default `Session` being used in the current thread.</returns>
      function get_default_session: TFSession;
      /// <summary>
      /// Returns the default session for the current thread.
      /// </summary>
      /// <returns>The default `Session` being used in the current thread.</returns>
      function set_default_session(sess: TFSession): TFSession;
      /// <summary>
      ///     Returns the default graph for the current thread.
      ///
      ///     The returned graph will be the innermost graph on which a
      ///     `Graph.as_default()` context has been entered, or a global default
      ///     graph if none has been explicitly created.
      ///
      ///     NOTE: The default graph is a property of the current thread.If you
      ///     create a new thread, and wish to use the default graph in that
      ///     thread, you must explicitly add a `with g.as_default():` in that
      ///     thread's function.
      /// </summary>
      /// <returns></returns>
      class function set_default_graph(g: TFGraph): TFGraph;
      class function peak_default_graph: TFGraph;
      class procedure pop_graph;

      constructor Create;
      destructor Destroy;override;

      property isSingleThreaded    : Boolean           read FisSingleThreaded   write SetSingleThread;
      property default_graph_stack : DefaultGraphStack read Get_default_graph_stack;
  end;

  TGen_Math_Ops = record
     class function cast(x: TFTensor; DstT: TF_DataType; name: string = '';Truncate : Boolean = false): TFTensor;static;
  end;

   T_Math_Ops = record
     class function cast(x: TFTensor; dtype: TF_DataType = TF_DATATYPE_UNKNOWN; name: string = ''): TFTensor;static;
  end;

implementation
           uses Tensorflow.Utils, oz.Pb.Classes, Oz.SGL.Collections;

{ TOps }

function TOps.Tensor_Id(tensor: TFTensor): Int64;
begin
     Result := tensor.id
end;

class function TOps.uid: Integer;
begin
    Result := TInterlocked.Increment(Fuid_number)
end;

procedure TOps.Add_to_collection<T>(name: string; value: T);
begin
     var graph := tf.get_default_graph;
     graph.add_to_collection<T>(name, value);
end;

procedure TOps.Add_to_collection<T>(names: TList<string>; value: T);
begin
    var graph := tf.get_default_graph;
    graph.add_to_collection<T>(names, value);
end;

function TOps.get_collection(key, scope: string): TObject;
begin
    Result := get_default_graph.get_collection(key, scope);
end;

function TOps.get_collection<T>(key, scope: string): TList<T>;
begin
     Result := get_default_graph.get_collection<T>(key, scope);
end;

function TOps.get_collection_ref<T>(key: string): TList<T>;
begin
    Result :=  get_default_graph.get_collection_ref<T>(key);
end;

class function TOps._get_graph_from_inputs(op_input_list: TArray<TFTensor>): TFGraph;
var
  AValue : TArray<TValue>;
begin
    AValue := [];
    for var i := 0 to Length(op_input_list) - 1 do
        AValue := AValue + [ TValue.From<TFTensor>(op_input_list[i]) ] ;

     Result :=  _get_graph_from_inputs(AValue);
end;

class function TOps._get_graph_from_inputs(op_input_list: TArray<TValue>): TFGraph;
begin
    var current_default_graph := get_default_graph;
    if current_default_graph.building_function then
        Exit(current_default_graph);
    var graph : TFGraph := nil;
    for var op_input in op_input_list do
    begin
        if string.LowerCase(string(op_input.TypeInfo^.Name)) = 'tftensor' then
        begin
            if graph = nil then
               graph := op_input.AsType<TFTensor>.graph
            else
               graph := graph;
        end;
    end;
    if graph <> nil then
        Result := graph
    else
        Result := current_default_graph;
end;

class function TOps._get_graph_from_inputs(op_input_list: TFTensor): TFGraph;
begin
    Result := _get_graph_from_inputs(op_input_list, nil);
end;

class function TOps._get_graph_from_inputs(op_input_list: TFTensor; graph: TFGraph): TFGraph;
begin
    //for var op_input in op_input_list do
    begin
        // Determine if this is a valid graph_element.
        // var graph_element = op_input;
    end;
    Result := get_default_graph;
end;

class function TOps.convert_to_tensor(value: TValue; dtype: TF_DataType; name: string; as_ref: Boolean; preferred_dtype: TF_DataType; ctx: TContext): TFTensor;
begin
    if dtype = TF_DataType.TF_DATATYPE_UNKNOWN then
        dtype := preferred_dtype;
    if dtype = TF_DataType.TF_DATATYPE_UNKNOWN then
        dtype :=  TUtils.GetDataType( value ) ;

    if value.IsType<TEagerTensor> then
    begin
        var eager_tensor := value.AsType<TEagerTensor>;
        if tf.executing_eagerly then
        begin
            if (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) and (dtype <> eager_tensor.TensorDataType) then
                Exit(TGen_Math_Ops.cast(eager_tensor, Tdtypes.as_base_dtype(dtype), name));
            Exit(eager_tensor);
        end else
        begin
            var graph := get_default_graph;
            if not graph.building_function then
               raise Exception.Create('Attempting to capture an EagerTensor without building a function.');
//            return (graph as FuncGraph).capture(eager_tensor, name: name);
        end;
    end;
    // graph mode
    (*Tensor ret = value switch
    {
        NDArray nd => constant_op.constant(nd, dtype: dtype, name: name),
        EagerTensor tensor => tensor.dtype == TF_DataType.TF_RESOURCE
                    ? tensor.AsPlaceholder(name: name)
                    : tensor.AsConstant(name: name),
        Tensor tensor => tensor,
        IEnumerable<Tensor> tensors => array_ops._autopacking_helper(tensors, dtype, name == null ? "packed" : name),
        RefVariable varVal => varVal._TensorConversionFunction(dtype: dtype, name: name, as_ref: as_ref),
        ResourceVariable varVal => varVal._TensorConversionFunction(dtype: dtype, name: name, as_ref: as_ref),
        Axis ts => constant_op.constant(ts, dtype: dtype, name: name),
        Shape ts => constant_op.constant(ts.dims, dtype: dtype, name: name),
        string str => constant_op.constant(str, dtype: tf.@string, name: name),
        string[] str => constant_op.constant(str, dtype: tf.@string, name: name),
        IEnumerable<object> objects => array_ops._autopacking_conversion_function(objects, dtype: dtype, name: name),
        _ => constant_op.constant(value, dtype: dtype, name: name)
    };
    if (dtype == TF_DataType.TF_STRING)
        return ret;
    if (dtype != TF_DataType.DtInvalid && dtype.as_base_dtype() != ret.dtype.as_base_dtype())
        ret = gen_math_ops.cast(ret, dtype, name: name);
    return ret;
    *)
end;

class function TOps.constant(value: TValue; dtype: TF_DataType; shape: TArray<TF_int64_t>; name: AnsiString; verify_shape: Boolean): TFTensor;
begin
    var sh : TFShape;
    if Length(shape) > 0  then
      sh :=  TFShape.Create(shape)
    else
      sh := nil;


    Result := TConstant_op.constant(@value,
                                    dtype,
                                    @sh,
                                    verify_shape,
                                    false,
                                    name);
end;

function TOps.convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin

end;

procedure TOps.colocate_with(variable: IVariableV1; ignore_existing: Boolean);
begin

end;

procedure TOps.colocate_with(tensor: TFTensor; ignore_existing: Boolean);
begin

end;

procedure TOps.colocate_with(ignore_existing: Boolean);
begin

end;

procedure TOps.colocate_with(op: TFOperation; ignore_existing: Boolean);
begin

end;

function TOps.control_dependencies(control_inputs: TArray<TObject>): _ControlDependenciesController;
begin

end;

function TOps.convert_n_to_tensor(values: TObject; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin

end;

function TOps.convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin

end;

function TOps.convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin

end;

constructor TOps.Create;
begin
    FLock := TCriticalSection.Create;
    FisSingleThreaded := False;
end;

destructor TOps.Destroy;
begin
    FreeAndNil(FLock);
end;

function TOps.executing_eagerly_outside_functions: Boolean;
begin

end;

class function TOps.get_default_graph: TFGraph;
begin
    if Fdefault_graph_stack = nil then
           Fdefault_graph_stack  := DefaultGraphStack.Create;

    Result := Fdefault_graph_stack.get_default
end;

class function TOps.set_default_graph(g: TFGraph): TFGraph;
begin
    Fdefault_graph_stack.get_controller(g);
    Result := g;
end;

function TOps.Get_default_graph_stack: DefaultGraphStack;
begin
    //if (!isSingleThreaded)
    //    return _defaultGraphFactory.Value;
    if F_singleGraphStack = nil then
    begin
        FLock.Acquire;
        if F_singleGraphStack = nil then
          F_singleGraphStack := DefaultGraphStack.Create;
    end;
    Result := F_singleGraphStack;
    FLock.Release;
end;

function TOps.get_default_session: TFSession;
begin
    //if (!isSingleThreaded)
    //    return tf.defaultSession;
    if F_singleSesson = nil then
    begin
        FLock.Acquire;
        if F_singleSesson = nil then
            F_singleSesson := TFSession.Create;
    end;
    Result := F_singleSesson;
end;

function TOps.set_default_session(sess: TFSession): TFSession;
begin
    //if (!isSingleThreaded)
    //    return tf.defaultSession = sess;
    FLock.Acquire;
    F_singleSesson := sess;

    Result := F_singleSesson;
end;

class procedure TOps.reset_default_graph;
begin
    //if (!_default_graph_stack.is_cleared())
    //    throw new InvalidOperationException("Do not use tf.reset_default_graph() to clear " +
    //                                    "nested graphs. If you need a cleared graph, " +
    //                                    "exit the nesting and create a new graph.");
    Fdefault_graph_stack.reset;
end;

function TOps.get_name_scope: string;
begin
    var g := get_default_graph;
    Result := g.get_name_scope;
end;

class function TOps.GraphUniqueId: Integer;
begin
    TInterlocked.Increment(Fgraph_uid_number);
    Result := Fgraph_uid_number
end;

class function TOps.uid_function: Integer;
begin
    TInterlocked.Increment(Fuid_number_for_function);
    Result := Fuid_number_for_function
end;

function TOps.init_scope: TNameScope;
begin

end;

class function TOps.internal_convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType; name: string; preferred_dtype: TF_DataType; as_ref: Boolean): TArray<TFTensor>;
begin

end;

function TOps.internal_convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType; name: string; as_ref: Boolean): TArray<TFTensor>;
begin

end;

function TOps.internal_convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin

end;

function TOps.internal_convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin

end;

class function TOps.name_from_scope_name(name: string): string;
begin
    if name = '' then  Exit('');

    if name.EndsWith('/') then  Exit( name.Substring(0,name.Length - 1) );

    Result := name;
end;

class function TOps.peak_default_graph: TFGraph;
begin
    Fdefault_graph_stack.peak_controller;
end;

class procedure TOps.pop_graph;
begin
    Fdefault_graph_stack.pop;
end;

function TOps.prepend_name_scope(name, import_scope: string): string;
begin

end;

procedure TOps._colocate_with_for_gradient(op: TFOperation; gradient_uid: string; ignore_existing: Boolean);
begin

end;

class function TOps._create_c_op(graph: TFGraph; node_def: TNodeDef; inputs: TArray<TFTensor>; control_inputs: TArray<TFOperation>; op_def: POpDef): Tuple<Pointer, TFOperationDesc>;
var
 status: TFStatus;
begin
   // This will be set by self.inputs.
   if op_def = nil then
   begin
        var op := graph.GetOpDef(node_def.Op);
        op_def := @op;
   end;

   var input_tensors : TArray<TTensors> := _reconstruct_sequence_inputs(op_def^, inputs, node_def.Attr);

   var op_desc := graph.NewOperation(node_def.Op, node_def.Name);

   if not string.IsNullOrEmpty(node_def.Device) then
       TF_SetDevice( op_desc.Handle, PTFChar( TFString(node_def.Device) ) );

   // Add inputs
   for var op_input : TTensors in input_tensors do
   begin
      if op_input.IsList then
      begin
          var aListO : TArray<TF_Output> := [];
          for var i := 0 to op_input.Count do
            aListO := aListO + [ op_input[0]._as_tf_output ];

          TF_AddInputList(op_desc.Handle, PTF_Output(@aListO[0]), Length(aListO))  
      end else
      begin
          TF_AddInput(op_desc.Handle,op_input[0]._as_tf_output);
      end;
   end;

   Status := TFStatus.Create;

   // Add control inputs
   for var control_input in control_inputs do
      TF_AddControlInput(op_desc.Handle, control_input.Handle);

   var S     : TpbSaver;
   var bytes : TBytes;
   // Add attrs
   for var attr in node_def.Attr do
   begin
       bytes := [] ;
       S.Init;
       TpbSaver.SaveAttrValue(S,attr.Value);
       bytes:= s.Pb.GetBytes;
       TF_SetAttrValueProto(op_desc.Handle, PTFChar( TFString(attr.Key)), @bytes[0], Length(bytes), status.Handle);
       status.CheckMaybeRaise(status,True);
   end;

   var c_op := TF_FinishOperation(op_desc.Handle, status.Handle);
   if c_op = nil then
      MessageBoxA(0,PAnsiChar(AnsiString(status.ToString)),'Status Message',MB_OK);

   status.CheckMaybeRaise(status,True);
   Result.Create(c_op, op_desc);

end;

function TOps._eval_using_default_session(tensor: TFTensor; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession): TNDArray;
begin

end;

function TOps._get_op_def(graph: TFGraph; tipo: string): TOpDef;
begin

end;

class function TOps._NodeDef(op_type, name: string; attrs: TDictionary<string, TAttrValue>): TNodeDef;
var
  node_def : TNodeDef;
begin
    node_def.Init;
    node_def.Op   := op_type;
    node_def.Name := name;

    if Assigned(attrs) then
    begin
        for var attr in attrs do
        begin

            node_def.Attr.Add( attr.Key, attr.Value );
        end;
    end;
    Result := node_def;
end;

class function TOps._reconstruct_sequence_inputs(op_def: TOpDef; inputs: TArray<TFTensor>; attrs: TMultiMap<string, TAttrValue>): TArray<TTensors>;
var
  grouped_inputs : TList<TTensors>;
  i : Integer;
begin
    grouped_inputs := TList<TTensors>.Create;
    i := 0;

    for var input_arg in op_def.InputArgs do
    begin
        var input_len  : Integer := 1;
        var is_sequence: Boolean  := false;

        if not string.IsNullOrEmpty(input_arg.NumberAttr) then
        begin
            input_len := attrs[input_arg.NumberAttr][0].Value.value.AsInteger;
            is_sequence := true;
        end
        else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
        begin
            var v1 := attrs[input_arg.TypeListAttr][0].Value.Value.AsType<TListValue>;
            input_len := v1.Types.Count;
            is_sequence := true;
        end;

        if is_sequence then
        begin
            var r := Enumerable<TFTensor>.Create(inputs);
            var rr := r.Skip(i).Take(input_len).ToArray; 
            var input_tensors := TTensors.Create(rr);
            input_tensors.IsList := true;
            grouped_inputs.Add(input_tensors);
        end else
        begin
            grouped_inputs.Add(TTensors.Create ([ inputs[i] ]));
        end;
        i := i + input_len;
    end;
end;

procedure TOps._run_using_default_session(operation: TFOperation; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession);
begin

end;

class procedure TOps.reset_uid;
begin
    Fuid_number := -1;
end;

procedure TOps.SetSingleThread(const Value: Boolean);
begin
  FisSingleThreaded := Value;
end;

function TOps.strip_name_scope(name, export_scope: string): string;
begin

end;

class function TOps.name_scope(name: TFString; default_name: TFString; values : PValue; skip_on_eager: Boolean): TNameScope;
begin
    Result := TNameScope.Create(name, default_name, values, skip_on_eager)
end;

{ TNameScope }

constructor TNameScope.Create(name, default_name: TFString; values: PValue; skip_on_eager: Boolean);
begin
    _name := name;
    _default_name := default_name;
    if values <> nil then
      _values := values^
    else
      _values := default(TValue);
    _skip_on_eager := skip_on_eager;
end;

function TNameScope.enter_eager_name_scope(ctx: TContext; name: TFString): Tuple<TFString, TFString>;
begin
    if _skip_on_eager then
        Exit(Tuple<TFString, TFString>.Create('',''));
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

    Result  := Tuple<TFString, TFString>.Create(scope_name, old_name);
end;

function TNameScope.ToString: TFString;
begin
    Result := scope_name;
end;

procedure TNameScope._Enter_;
var
  tRes : Tuple<TFString,TFString>;
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
        old_scope_name := g._name_stack;
        scope_name     := g.name_scope(_name);
    end;
end;

procedure TNameScope._Exit_;
begin
    if tf.Context.executing_eagerly then
        tf.Context.ScopeName := string(old_scope_name)
    else
        TOps.get_default_graph._name_stack := old_scope_name;
end;

{ TGen_Math_Ops }

class function TGen_Math_Ops.cast(x: TFTensor; DstT: TF_DataType; name: string = '';Truncate : Boolean = false): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Cast', name, ExecuteOpArgs.Create([TValue.From<TFTensor>(x)]).SetAttributes([TValue.From<Integer>(Ord(DstT)), Truncate]) ).FirstOrDefault;
end;

{ T_Math_Ops }

class function T_Math_Ops.cast(x: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var base_type := Tdtypes.as_base_dtype(dtype);
    if base_type = x.dtype then
        Exit(x);

    var vvalue := TValue.From<TFTensor>(x);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cast', @vvalue),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := string(v1.ToString);
                                                if Tdtypes.as_base_dtype(x.dtype) <> base_type then
                                                    x := TGen_Math_Ops.cast(x, base_type, name);
                                                Result := x;
                                            end );
end;

end.
