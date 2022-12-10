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

          ProtoGen.attrValue,
          ProtoGen.OpDef;

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

     protected
       procedure NativeDispose(hnd: Pointer); override;
     public
       FuncName         : string;
       Inputs           : TFTensors;
       Outputs          : TFTensors;
       Attrs            : TDictionary<string, string>;
       external_captures: TArray<TFTensor>;
       captures         : TArray< Tuple<TFTensor, TFTensor> >;
       internal_captures: TArray<TFTensor>;
       captured_inputs  : TArray<TFTensor>;

       // <summary>
       /// Construct a new FuncGraph.
       /// </summary>
       constructor Create(name: string) ; overload;
       constructor Create(_handle: Pointer; name: string; attrs: TDictionary<string, string>) ; overload;
       function    capture(tensor: TFTensor; name: string = ''; shape: PTFShape = nil): TFTensor;
       function    capture_eager_tensor(tensor: TFTensor; name: string): TFTensor;
       function    _capture_helper(tensor: TFTensor; name: string; shape: PTFShape = nil): TFTensor;
       procedure   ToGraph(opers: TArray<TFOperation>; inputs : TArray<TFTensor>; outputs : TArray<TFTEnsor>; output_names: TArray<string>) ;
       function    _create_substitute_placeholder(value: TFTensor; name : string= ''; dtype: TF_DataType = DtInvalid; shape: PTFShape = nil): TFTensor;
       procedure   SetAttrs;
       function    as_default: TFGraph; override;
       procedure   gExit; override;
       function    Create_op(op_type    : TF_TString ;
                        inputs         : TArray<TFTensor>;
                        dtypes         : TArray<TF_DataType>;
                        input_types    : TArray<TF_DataType> = [];
                        Name           : TF_TString= '';
                        attrs          : TDictionary<string, TAttrValue> = nil;
                        op_def         : POpDef= nil;
                        compute_device : Boolean = True) : TFOperation; override;
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
               TensorFlow.Ops;

{ DefaultGraphStack }

constructor DefaultGraphStack.Create;
begin
    inherited Create;
    F_stack := TStack<TFGraph>.Create;
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

   Fouter_graph := Tops.get_default_graph;

   while Fouter_graph.building_function do
     Fouter_graph := Fouter_graph.OuterGraph;

   Fgraph_key := name;
   Fbuilding_function := true;
end;

constructor TFuncGraph.Create(_handle: Pointer; name: string; attrs: TDictionary<string, string>);
begin

end;

procedure TFuncGraph.NativeDispose(hnd: Pointer);
begin
  TFE_ContextRemoveFunction(tf.Context.Handle,PAnsiChar(AnsiString (Fgraph_key) ), tf.Status.Handle);
  TF_DeleteFunction(F_func_graph_handle);

  inherited NativeDispose(hnd);
end;

function TFuncGraph.as_default: TFGraph;
begin

end;

function TFuncGraph.capture(tensor: TFTensor; name: string; shape: PTFShape): TFTensor;
begin

end;

function TFuncGraph.capture_eager_tensor(tensor: TFTensor; name: string): TFTensor;
begin

end;

function TFuncGraph.Create_op(op_type: TF_TString; inputs: TArray<TFTensor>; dtypes, input_types: TArray<TF_DataType>; Name: TF_TString; attrs: TDictionary<string, TAttrValue>;
  op_def: POpDef; compute_device: Boolean): TFOperation;
begin

end;

procedure TFuncGraph.gExit;
begin
  inherited;

end;

procedure TFuncGraph.SetAttrs;
begin

end;

procedure TFuncGraph.ToGraph(opers: TArray<TFOperation>; inputs, outputs: TArray<TFTEnsor>; output_names: TArray<string>);
begin

end;

function TFuncGraph._capture_helper(tensor: TFTensor; name: string; shape: PTFShape): TFTensor;
begin

end;

function TFuncGraph._create_substitute_placeholder(value: TFTensor; name: string; dtype: TF_DataType; shape: PTFShape): TFTensor;
begin

end;

{ SubGraphUtility }

class function SubGraphUtility.lift_to_graph(init_tensors: TFTensors; graph: TFuncGraph; sources: TList<TFTensor>; add_sources, handle_captures: Boolean; base_graph: TFGraph;
  op_map: TDictionary<ITensorOrOperation, TFOperation>): TDictionary<ITensorOrOperation, TFOperation>;
begin

end;

class function SubGraphUtility.map_subgraph(init_tensor: TFTensor; sources: TList<TFTensor>; visited_ops: TList<TFOperation>; add_sources: Boolean): TList<TFTensor>;
begin

end;

class procedure SubGraphUtility._copy_non_source(op: TFOperation; graph: TFuncGraph; op_map: TDictionary<ITensorOrOperation, TFOperation>; base_graph: TFGraph);
begin

end;

class procedure SubGraphUtility._copy_source(s: TFTensor; graph: TFuncGraph; op_map: TDictionary<ITensorOrOperation, TFOperation>; handle_captures: Boolean;
  inverse_captures: TDictionary<TFTensor, TFTensor>; base_graph: TFGraph);
begin

end;

end.

