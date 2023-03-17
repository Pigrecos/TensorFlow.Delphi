unit Tensorflow.Graph;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
     uses System.SysUtils,
          System.Generics.Collections,

          Spring,

          TF4D.Core.CApi,
          TensorFlow.DApiEager,
          TensorFlow.DApi,
          TensorFlow.DApiBase,
          TensorFlow.Framework,

          ProtoGen.attrValue,
          ProtoGen.OpDef;

         // Keras.Data;

const
  EAGER_CONST_THRESHOLD : Integer = 128;

type

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

implementation
          uses Tensorflow,
               Tensorflow.Utils,
               TensorFlow.Constant_op,
               Tensorflow.Gradient,
               TensorFlow.Ops,
               Tensorflow.array_ops,
               TensorFlow.EagerTensor,
               TensorFlow.EagareRunner,
               Oz.Pb.Classes,
               Oz.Pb.StrBuffer,

               ProtoGen.Main;

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

end.
