unit TensorFlow.Ops;
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
     uses System.SysUtils, System.SyncObjs, Winapi.Windows, System.Rtti,
          Spring,
          Spring.Collections, Spring.Collections.MultiMaps, Spring.Collections.Enumerable,
          Spring.Collections.Extensions, Spring.Collections.Lists, Spring.Collections.Dictionaries,

          TF4D.Core.CApi,
          NumPy.NDArray,
          Tensorflow.NameScope,
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
  TOps = class
    private
      FisSingleThreaded   : Boolean;
      F_singleGraphStack  : DefaultGraphStack;
      FLock               : TCriticalSection;

      class var Fdefault_graph_stack: DefaultGraphStack;
      class var Fuid_number      : Integer;
      class var Fgraph_uid_number: Integer;
      class var Fuid_number_for_function : Integer;

      procedure SetSingleThread(const Value: Boolean);
      function  Get_default_graph_stack: DefaultGraphStack;

    public
      class var FdefaultSession     : TFSession;
      function  Tensor_Id(tensor: TFTensor): Int64;
      class procedure Add_to_collection<T>(name: string; value: T);overload;
      class procedure Add_to_collection<T>(names: TList<string>; value: T);overload;
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
      class function get_collection(key: string; scope : string = ''): TValue;overload;
      class function get_collection<T>(key: string; scope : string = ''): TList<T>;overload;
      class function get_collection_ref<T>(key: string): TList<T>;overload;
      class function _get_graph_from_inputs(op_input_list: TArray<TFTensor>): TFGraph; overload;
      class function _get_graph_from_inputs(op_input_list: TArray<TValue>): TFGraph; overload;
      class function _get_graph_from_inputs(op_input_list: TFTensor): TFGraph ;  overload;
      class function _get_graph_from_inputs(op_input_list: TFTensor; graph: TFGraph = nil): TFGraph ;  overload;
      class function name_scope(name: TF_TString; default_name: TF_TString = ''; values : PValue= nil; skip_on_eager: Boolean = true): TNameScope; static;

      /// <summary>
      /// Converts the given `value` to a `Tensor`.
      /// </summary>
      /// <param name="value"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function convert_to_tensor(value: TValue; dtype : TF_DataType = DtInvalid; name: string= ''; as_ref: Boolean = False; preferred_dtype : TF_DataType = DtInvalid; ctx: TContext= nil): TFTensor;
      class function convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''): TFTensor;
      class function internal_convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''; as_ref: Boolean = false): TFTensor;
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
      class function _reconstruct_sequence_inputs(op_def: TOpDef; inputs: TArray<TFTensor>; attrs: TMultiMap<string, TAttrValue>): TArray<TFTensors>;
      function _get_op_def(graph: TFGraph; tipo: string): TOpDef;
      class function _NodeDef(op_type: string; name: string; attrs : TDictionary<string, TAttrValue> = nil): TNodeDef;
      class function name_from_scope_name(name: string): string;
      /// <summary>
      /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
      /// </summary>
      /// <returns></returns>
      class function init_scope: TNameScope;
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
      class procedure colocate_with(ignore_existing : Boolean = false); overload;
      class procedure colocate_with(op: TFOperation; ignore_existing : Boolean = false); overload;
      class procedure colocate_with(tensor: TFTensor; ignore_existing : Boolean= false); overload;
      class procedure colocate_with(variable: IVariableV1; ignore_existing : Boolean = false); overload;
      class procedure _colocate_with_for_gradient(op: TFOperation; gradient_uid: string; ignore_existing : Boolean = false);
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
      class function  _eval_using_default_session(tensor: TFTensor; feed_dict : TArray<FeedItem>; graph: TFGraph; session : TFSession = nil): TNDArray;
      /// <summary>
      /// Prepends name scope to a name.
      /// </summary>
      /// <param name="name"></param>
      /// <param name="import_scope"></param>
      /// <returns></returns>
      class function  prepend_name_scope(name: string; import_scope: string): string;
      class procedure _run_using_default_session(operation: TFOperation; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession);
      class function  convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType = DtInvalid; name : string = ''): TArray<TFTensor>;
      class function  convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType = DtInvalid; name: string = ''): TArray<TFTensor>;
      class function  convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''): TFTensor;
      class function  internal_convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''; as_ref : Boolean = false): TFTensor;
      class function  internal_convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType = DtInvalid; name : string= ''; as_ref : Boolean= false): TArray<TFTensor>;
      class function  internal_convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType = DtInvalid; name: string = ''; preferred_dtype : TF_DataType = DtInvalid; as_ref: Boolean = false):TArray<TFTensor>;overload;
      class function  internal_convert_n_to_tensor(values: TArray<TFTensor>; dtype: TF_DataType = DtInvalid; name: string = ''; preferred_dtype : TF_DataType = DtInvalid; as_ref: Boolean = false):TArray<TFTensor>;overload;
      class function  strip_name_scope(name: string; export_scope: string = ''): string;
      class function  get_name_scope: string;
      class function  executing_eagerly_outside_functions: Boolean;

      // array_ops Class
      class function constant(value: TValue; dtype: TF_DataType= DtInvalid; shape: TArray<TF_int64_t> =[]; name: AnsiString= 'Const'; verify_shape: Boolean=False): TFTensor; static;


      (*// threading*)

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
      class function  control_dependencies(control_inputs: TArray<TValue>): TControlDependenciesController;
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
      class function get_default_session: TFSession;
      /// <summary>
      /// Returns the default session for the current thread.
      /// </summary>
      /// <returns>The default `Session` being used in the current thread.</returns>
      class function set_default_session(sess: TFSession): TFSession;
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

implementation
           uses Tensorflow.Utils,
                TensorFlow.EagerTensor,
                TensorFlow.Constant_op,
                Tensorflow.gen_array_ops,
                TensorFlow.gen_math_ops,
                Tensorflow.array_ops,
                Numpy.Axis, Numpy,
                TensorFlow.Tensor,

                oz.Pb.Classes,
                Oz.SGL.Collections,
                oz.Pb.StrBuffer,

                System.TypInfo;

{ TOps }

function TOps.Tensor_Id(tensor: TFTensor): Int64;
begin
     Result := tensor.id
end;

class function TOps.uid: Integer;
begin
    Result := TInterlocked.Increment(Fuid_number)
end;

class procedure TOps.Add_to_collection<T>(name: string; value: T);
begin
     var graph := tf.get_default_graph;
     graph.add_to_collection<T>(name, value);
end;

class procedure TOps.Add_to_collection<T>(names: TList<string>; value: T);
begin
    var graph := tf.get_default_graph;
    graph.add_to_collection<T>(names, value);
end;

class function TOps.get_collection(key, scope: string): TValue;
begin
    Result := get_default_graph.get_collection(key, scope);
end;

class function TOps.get_collection<T>(key, scope: string): TList<T>;
begin
     Result := get_default_graph.get_collection<T>(key, scope);
end;

class function TOps.get_collection_ref<T>(key: string): TList<T>;
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
        if (op_input.TypeInfo<> nil) and (string.LowerCase(string(op_input.TypeInfo^.Name))  = 'tftensor') then
        begin
            if graph = nil then
            begin
               var t := op_input.AsType<TFTensor>;
               graph := t.graph;
            end
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
    if dtype = TF_DataType.DtInvalid then
        dtype := preferred_dtype;

    if dtype = TF_DataType.DtInvalid then
        dtype :=  TUtils.GetDataType( value ) ;

    if value.IsType<TEagerTensor> then
    begin
        var eager_tensor := value.AsType<TEagerTensor>;
        if tf.executing_eagerly then
        begin
            if (dtype <> TF_DataType.DtInvalid) and (dtype <> eager_tensor.Dtype) then
                Exit(gen_math_ops.cast(eager_tensor, Tdtypes.as_base_dtype(dtype), name));
            Exit(eager_tensor);
        end else
        begin
            var graph := get_default_graph;
            if not graph.building_function then
               raise Exception.Create('Attempting to capture an EagerTensor without building a function.');
            Result := (graph as TFuncGraph).capture(eager_tensor, name);
            Exit;
        end;
    end;

    // graph mode
    var ret : TFTensor;
    if value.IsType<TNDArray> then
    begin
        var nd := value.AsType<TNDArray>;
        ret := constant_op.constant(nd, dtype, name);
    end
    else if value.IsType<NDArray> then
    begin
        var nd := TNDArray(value.AsType<NDArray>);
        ret := constant_op.constant(nd, dtype, name);
    end
    else if value.IsType<TEagerTensor> then
    begin
        var tensor := value.AsType<TEagerTensor>;
        if tensor.Dtype = TF_RESOURCE then  ret := tensor.AsPlaceholder(name)
        else                                ret := tensor.AsConstant(name)
    end
    else if value.IsType<TFTensor> then
    begin
        var tensor := value.AsType<TFTensor>;
        ret := tensor;
    end
    else if value.IsType<TTensor> then
    begin
        var tensor := TFTensor(value.AsType<TTensor>);
        ret := tensor;
    end
    else if value.IsType<TFTensors> then
    begin
        var tensors  := value.AsType<TFTensors>;
        var vArray : TArray<TValue>;
        for var i := 0 to tensors.Count - 1 do
            vArray := vArray + [ TValue.From<TFTensor>(tensors[i]) ];

        if name = ''  then ret := array_ops._autopacking_helper(vArray, dtype, 'packed')
        else               ret := array_ops._autopacking_helper(vArray, dtype, name)
    end
    else if value.IsType<RefVariable> then
    begin
        var varVal  := value.AsType<RefVariable>;
        ret := varVal._TensorConversionFunction(dtype, name, as_ref)
    end
    else if value.IsType<ResourceVariable> then
    begin
        var varVal  := value.AsType<ResourceVariable>;
        ret := varVal._TensorConversionFunction(dtype, name, as_ref)
    end
    else if value.IsType<TAxis> then
    begin
        ret := constant_op.constant(value, dtype, name)
    end
    else if value.IsType<TFShape> then
    begin
        var ts  := value.AsType<TFShape>;
        var d   := TValue.From< TArray<Int64> >(ts.dims);
        ret := constant_op.constant(d, dtype, name);
    end
    else if value.IsType<string> then
    begin
        ret := constant_op.constant(value, tf.string_t, name);
    end
    else if value.IsType<AnsiString> then
    begin
        ret := constant_op.constant(value, tf.string_t, name);
    end
    else if value.IsType< TArray<string> > then
    begin
        ret := constant_op.constant(value, tf.string_t, name);
    end
    else if value.IsType< IEnumerable<TValue> > then
    begin
        var obj  := value.AsType< IEnumerable<TValue> >;
        var vArray : TArray<TValue> := obj.ToArray;

         ret := array_ops._autopacking_conversion_function(vArray, dtype, name)
    end
    else if value.IsType< TArray<TValue> > then
    begin
        var obj  := value.AsType< TArray<TValue> >;
        ret := array_ops._autopacking_conversion_function(obj, dtype, name)
    end
    else if value.IsType< TArray<TFTensor> > then
    begin
        var obj  := value.AsType< TArray<TFTensor> >;
        var vArray : TArray<TValue> ;
        for var i := 0 to Length(obj)-1 do
            vArray := vArray + [ obj[i] ];

         ret := array_ops._autopacking_conversion_function(vArray, dtype, name)
    end else
    begin
        ret := constant_op.constant(value, dtype, name)
    end;

    if dtype = TF_STRING then
        Exit(ret);

    if (dtype <> DtInvalid) and ( Tdtypes.as_base_dtype(dtype) <> Tdtypes.as_base_dtype(ret.Dtype)) then
        ret := gen_math_ops.cast(ret, dtype, name);

    Result := ret;

end;

class function TOps.constant(value: TValue; dtype: TF_DataType; shape: TArray<TF_int64_t>; name: AnsiString; verify_shape: Boolean): TFTensor;
begin
    var sh : TFShape;
    if Length(shape) > 0  then
      sh :=  TFShape.Create(shape)
    else
      sh := nil;


    Result := constant_op.constant(@value,
                                    dtype,
                                    @sh,
                                    verify_shape,
                                    false,
                                    name);
end;

class function TOps.control_dependencies(control_inputs: TArray<TValue>): TControlDependenciesController;
begin
    Result := get_default_graph.control_dependencies(control_inputs);
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

class function TOps.executing_eagerly_outside_functions: Boolean;
begin
    if tf.Context.executing_eagerly then
       Result := true
    else
       raise Exception.Create('Not Implemented');
end;

class function TOps.get_default_graph: TFGraph;
begin
    if Fdefault_graph_stack = nil then
           Fdefault_graph_stack  := DefaultGraphStack.Create;

    Result := Fdefault_graph_stack.get_default
end;

class function TOps.set_default_graph(g: TFGraph): TFGraph;
begin
    if Fdefault_graph_stack = nil then
           Fdefault_graph_stack  := DefaultGraphStack.Create;

    Fdefault_graph_stack.get_controller(g);
    Result := g;
end;

function TOps.Get_default_graph_stack: DefaultGraphStack;
begin
   if F_singleGraphStack = nil then
    begin
        FLock.Acquire;
        if F_singleGraphStack = nil then
          F_singleGraphStack := DefaultGraphStack.Create;
    end;
    Result := F_singleGraphStack;
    FLock.Release;
end;

class function TOps.get_default_session: TFSession;
begin
     if FdefaultSession = nil then
        FdefaultSession := TFSession.Create(tf.get_default_graph);

    Result := FdefaultSession;
end;

class function TOps.set_default_session(sess: TFSession): TFSession;
begin
   FdefaultSession := sess;

    Result := sess;
end;

class procedure TOps.reset_default_graph;
begin
    if Fdefault_graph_stack = nil then
       Exit;
    //if (!_default_graph_stack.is_cleared())
    //    throw new InvalidOperationException("Do not use tf.reset_default_graph() to clear " +
    //                                    "nested graphs. If you need a cleared graph, " +
    //                                    "exit the nesting and create a new graph.");
    Fdefault_graph_stack.reset;
end;

class function TOps.get_name_scope: string;
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

class function TOps.init_scope: TNameScope;
begin
    // Retrieve the active name scope: entering an `init_scope` preserves
    // the name scope of the current context.
    var default_graph := get_default_graph();
    var scope := default_graph.get_name_scope;
    if ( not String.IsNullOrEmpty(scope)) and ( not string(scope).EndsWith('/')) then
        // Names that end with trailing slashes are treated by `name_scope` as
        // absolute.
        scope := scope + '/';
    // inner_device_stack = default_graph._device_function_stack
    // var outer_context = default_graph.as_default;
    TUtils.tf_with<TControlDependenciesController>(Tops.control_dependencies(nil),
                                   procedure(v : TControlDependenciesController)
                                                    begin
                                                        // var outer_graph = get_default_graph();
                                                        // outer_device_stack = None
                                                    end );

    tf.Context.ScopeName := scope;
    Result := Tops.name_scope(scope);
end;

class function TOps.convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := internal_convert_to_tensor_or_composite(value, dtype, name, false);
end;

class function TOps.internal_convert_to_tensor_or_composite(value: TFTensor; dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    Result := convert_to_tensor(value, dtype, name, as_ref);
end;

class function TOps.internal_convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType; name: string; as_ref: Boolean): TArray<TFTensor>;
begin
    var ret := TList<TFTensor>.Create;

    var i : Integer := 0;
    for var value in values do
    begin
        if value = nil  then
        begin
            ret.Add(value);
        end else
        begin
            var n : string;
            if string.IsNullOrEmpty(name) then  n := ''
            else                                n := name+'_'+IntToStr(i);
            ret.Add(internal_convert_to_tensor_or_indexed_slices(value, dtype, n, as_ref));
            inc(i);
        end;
    end;
    Result := ret.ToArray;
end;

class function TOps.internal_convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType; name: string; preferred_dtype: TF_DataType; as_ref: Boolean): TArray<TFTensor>;
begin
    var ret := TList<TFTensor>.Create;

    var i : Integer := 0;
    for var value in values do
    begin
        var n : string;
        if string.IsNullOrEmpty(name) then  n := ''
        else                                n := name+'_'+IntToStr(i);
        ret.Add( convert_to_tensor(value, dtype, n, as_ref, preferred_dtype) );
        inc(i);
    end;
    Result := ret.ToArray;
end;

class function TOps.internal_convert_n_to_tensor(values: TArray<TFTensor>; dtype: TF_DataType; name: string; preferred_dtype: TF_DataType; as_ref: Boolean): TArray<TFTensor>;
begin
    var aValue : TArray<TValue> := [];
    for var i := 0 to Length(values)-1 do
         aValue := aValue + [ TValue.From<TFTensor>( values[i] ) ] ;

    Result := internal_convert_n_to_tensor(aValue, dtype, name,preferred_dtype, as_ref)
end;

class function TOps.convert_n_to_tensor(values: TArray<TValue>; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin
    Result := internal_convert_n_to_tensor(values, dtype, name, DtInvalid, false);
end;

class function TOps.convert_n_to_tensor_or_indexed_slices(values: TArray<TFTensor>; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin
    Result := internal_convert_n_to_tensor_or_indexed_slices(values, dtype, name);
end;

class function TOps.convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := internal_convert_to_tensor_or_indexed_slices(value, dtype, name, false);
end;

class function TOps.internal_convert_to_tensor_or_indexed_slices(value: TFTensor; dtype: TF_DataType; name: string; as_ref: Boolean): TFTensor;
begin
    Result := value;
end;

class function TOps.name_from_scope_name(name: string): string;
begin
    if name = '' then  Exit('');

    if name.EndsWith('/') then  Exit( name.Substring(0,name.Length - 1) );

    Result := name;
end;

class function TOps.peak_default_graph: TFGraph;
begin
    Result := Fdefault_graph_stack.peak_controller;
end;

class procedure TOps.pop_graph;
begin
    Fdefault_graph_stack.pop;
end;

class function TOps.prepend_name_scope(name, import_scope: string): string;
begin
    if not string.IsNullOrEmpty(import_scope) then
    begin
        if import_scope.EndsWith('/') then
            import_scope := import_scope.Substring(0, import_scope.Length - 1);
        Result := import_scope +'/'+ name;
    end
    else
        Result := name;
end;

class procedure TOps.colocate_with(variable: IVariableV1; ignore_existing: Boolean);
begin
     _colocate_with_for_gradient(variable.AsTensor.Op, '', ignore_existing);
end;

class procedure TOps.colocate_with(tensor: TFTensor; ignore_existing: Boolean);
begin
    _colocate_with_for_gradient(tensor.op, '', ignore_existing);
end;

class procedure TOps.colocate_with(ignore_existing: Boolean);
begin
    _colocate_with_for_gradient(nil, '', ignore_existing);
end;

class procedure TOps.colocate_with(op: TFOperation; ignore_existing: Boolean);
begin
     _colocate_with_for_gradient(op, '', ignore_existing);
end;

class procedure TOps._colocate_with_for_gradient(op: TFOperation; gradient_uid: string; ignore_existing: Boolean);
begin
    var default_graph := get_default_graph;
    default_graph.colocate_with_for_gradient(op, gradient_uid, ignore_existing);
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

   var input_tensors : TArray<TFTensors> := _reconstruct_sequence_inputs(op_def^, inputs, node_def.Attr);

   var op_desc := graph.NewOperation(node_def.Op, node_def.Name);

   if not string.IsNullOrEmpty(node_def.Device) then
       TF_SetDevice( op_desc.Handle, PTFChar( TF_TString(node_def.Device) ) );

   // Add inputs
   for var op_input : TFTensors in input_tensors do
   begin
      if op_input.IsList then
      begin
          var aListO : TArray<TF_Output> := [];
          for var i := 0 to op_input.Count -1 do
            aListO := aListO + [ op_input[i]._as_tf_output ];

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
       var Len : NativeInt := Length(bytes);
       TF_SetAttrValueProto(op_desc.Handle, PTFChar( TF_TString(attr.Key)), @bytes[0], Len, status.Handle);
       status.CheckMaybeRaise(status,True);
   end;

   var c_op := TF_FinishOperation(op_desc.Handle, status.Handle);
   if c_op = nil then
      MessageBoxA(0,PAnsiChar(AnsiString(status.ToString)),'Status Message',MB_OK);

   status.CheckMaybeRaise(status,True);
   Result.Create(c_op, op_desc);

end;

function TOps._get_op_def(graph: TFGraph; tipo: string): TOpDef;
begin
    Result := graph.GetOpDef(tipo);
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

class function TOps._reconstruct_sequence_inputs(op_def: TOpDef; inputs: TArray<TFTensor>; attrs: TMultiMap<string, TAttrValue>): TArray<TFTensors>;
var
  grouped_inputs : TList<TFTensors>;
  i : Integer;
begin
    grouped_inputs := TList<TFTensors>.Create;
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
            var input_tensors := TFTensors.Create(rr);
            input_tensors.IsList := true;
            grouped_inputs.Add(input_tensors);
        end else
        begin
            grouped_inputs.Add(TFTensors.Create ([ inputs[i] ]));
        end;
        i := i + input_len;
    end;
    Result := grouped_inputs.ToArray;
end;

class function TOps._eval_using_default_session(tensor: TFTensor; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession): TNDArray;
begin
    if session = nil then
    begin
        session := get_default_session;
        if session = nil then
           raise Exception.Create( 'Cannot evaluate tensor using `eval()`: No default ' + sLineBreak +
                                   'session is registered. Use `with '                  + sLineBreak +
                                   'sess.as_default()` or pass an explicit session to ' + sLineBreak +
                                   '`eval(session=sess)`');
        if session.graph <> graph then
           raise Exception.Create( 'Cannot use the default session to evaluate tensor: '   + sLineBreak +
                                   'the tensor''s graph is different from the session''s ' + sLineBreak +
                                   'graph. Pass an explicit session to '                   + sLineBreak +
                                   '`eval(session=sess)`.');
    end else
    begin
        if session.graph <> graph then
           raise Exception.Create( 'Cannot use the default session to evaluate tensor: '   + sLineBreak +
                                   'the tensor''s graph is different from the session''s ' + sLineBreak +
                                   'graph. Pass an explicit session to '                   + sLineBreak +
                                   '`eval(session=sess)`.');
    end;
    Result := session.run(tensor, feed_dict);
end;

class procedure TOps._run_using_default_session(operation: TFOperation; feed_dict: TArray<FeedItem>; graph: TFGraph; session: TFSession);
begin
    if session = nil then
    begin
        session := get_default_session;
        if session = nil then
           raise Exception.Create( 'Cannot execute operation using `run()`: No default ' + sLineBreak +
                                   'session is registered. Use `with '                   + sLineBreak +
                                   'sess.as_default():` or pass an explicit session to ' + sLineBreak +
                                   '`run(session=sess)`');
    end ;
    if session.graph <> graph then
       raise Exception.Create( 'Cannot use the default session to execute operation: ' + sLineBreak +
                               'the operation''s graph is different from the '         + sLineBreak +
                               'session''s graph. Pass an explicit session to '        + sLineBreak +
                               'run(session=sess).');
    //session.run(operation, feed_dict);
end;

class procedure TOps.reset_uid;
begin
    Fuid_number := -1;
end;

procedure TOps.SetSingleThread(const Value: Boolean);
begin
  FisSingleThreaded := Value;
end;

class function TOps.strip_name_scope(name, export_scope: string): string;
begin
    if not string.IsNullOrEmpty(export_scope) then
       raise Exception.Create('NotImplemented - ops.strip_name_scope')
    else
      Result := name;
end;

class function TOps.name_scope(name: TF_TString; default_name: TF_TString; values : PValue; skip_on_eager: Boolean): TNameScope;
begin
    Result := TNameScope.Create(name, default_name, values, skip_on_eager)
end;

end.
