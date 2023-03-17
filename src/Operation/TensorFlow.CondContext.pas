unit TensorFlow.CondContext;
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
         System.Rtti,

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,

         TensorFlow.Ops,

         ProtoGen.controlFlow;
type

 CondContext = class(TControlFlowContext)
   private
      Fexternal_values : TDictionary<string,TFTensor>;

      procedure _init_from_proto(context_def: TCondContextDef; import_scope: string = '');
      function  _BuildCondTensor(v: ITensorOrOperation): TFTensor;
      /// <summary>
      /// Process an output tensor of a conditional branch.
      /// </summary>
      function _ProcessOutputTensor(_val: TFTensor): TFTensor;
   protected
      procedure _AddOpInternal(op: TFOperation); override;
   public
      function AddValue(val: TFTensor): TFTensor; override;
      /// <summary>
      ///
      /// </summary>
      /// <param name="pred">The `boolean` tensor for the conditional predicate.</param>
      /// <param name="pivot">The predicate tensor in this branch.</param>
      /// <param name="branch">0 or 1 representing this branch.</param>
      /// <param name="name">Name of the `CondContext` python object.</param>
      /// <param name="context_def"></param>
      /// <param name="import_scope"></param>
      constructor Create(pred: TFTensor = nil; pivot: TFTensor = nil; branch: Integer = 0; name: string = 'cond_text'; context_def : TCondContextDef= nil; import_scope: string = '');
      /// <summary>
      /// Add the subgraph defined by fn() to the graph.
      /// </summary>
      function BuildCondBranch<T>(fn: TFunc<T>): tuple<T, TFTensor>;  overload;
      function BuildCondBranch<T>(fn: TFunc<TArray<T>>): tuple<TArray<T>, TArray<TFTensor>>; overload;
      destructor Destroy; override;
 end;

implementation
        uses System.TypInfo,
             Tensorflow,
             Tensorflow.Utils,
             Tensorflow.control_flow_ops;

{ CondContext }

constructor CondContext.Create(pred, pivot: TFTensor; branch: Integer; name: string; context_def: TCondContextDef; import_scope: string);
begin
    inherited Create;

    Fexternal_values := TDictionary<string,TFTensor>.Create;

    if (pred = nil) and (context_def = nil) then Exit;

    Fname := Tops.get_default_graph.unique_name(name);
    if context_def <> nil then
    begin
        _init_from_proto(context_def, import_scope);
    end else
    begin
        // Initializes the default fields.
        __init__;
        Fpred  := pred;
        Fpivot := pivot;
        Fbranch:= branch; // 0 or 1 representing this branch
        // Values considered to have been already seen in this context. pred is not
        // included in this context.
        Fvalues.Add(pred.name);
        Fexternal_values.AddOrSetValue(pred.name, pred);
        Fvalues.Add(pivot.name);
        pivot.op._set_control_flow_context(self);
    end;
end;

destructor CondContext.Destroy;
begin
    Fexternal_values.Free;

    inherited Destroy;
end;

procedure CondContext._init_from_proto(context_def: TCondContextDef; import_scope: string);
begin
    var g := Tops.get_default_graph;
    Fname := Tops.prepend_name_scope(context_def.ContextName, import_scope);
    var p1 := Tops.prepend_name_scope(context_def.PredName, import_scope);
    Fpred  := g.as_graph_element(p1) as TFTensor;
    var p2 := Tops.prepend_name_scope(context_def.PivotName, import_scope);
    Fpivot := g.as_graph_element(p2) as TFTensor;
    Fbranch:= context_def.Branch;
    __init__(context_def.ValuesDef, import_scope);
end;

function CondContext.AddValue(val: TFTensor): TFTensor;
var
  rResult : TFTensor;
begin
    rResult := nil;
    if Fvalues.Contains(val.name) then
    begin
        // Use the real value if it comes from outer context. This is needed in
        // particular for nested conds.
        if Fexternal_values.ContainsKey(val.name) then
            rResult := Fexternal_values[val.name];

        if rResult = nil then  rResult := val
    end else
    begin
        rResult := val;
        Fvalues.Add(val.name);
        // TODO: _outer_context
        if Fouter_context <> nil then
        begin
            rResult := Fouter_context.AddValue(val);
            Fvalues.Add(rResult.name);
            Fexternal_values.AddOrSetValue(rResult.name,rResult);
        end ;

        TUtils.tf_with<TControlDependenciesController>(Tops.control_dependencies(nil), procedure(ctrl: TControlDependenciesController)
                    begin
                        var res := control_flow_ops._SwitchRefOrTensor(rResult, Fpred);
                        rResult := res[Fbranch];
                        if Fouter_context <> nil then
                            Fouter_context.AddInnerOp(rResult.op);
                    end);

        rResult.op.graph.prevent_fetching(rResult.op);
        rResult.op._set_control_flow_context(Self);

        // Mark Switch output as seen by this context and any outer contexts,
        // just like what we do for normal op outputs in _AddOpInternal() below.
        var ctxt : TControlFlowContext := Self;
        while ctxt <> nil do
        begin
            ctxt.values.Add(rResult.name);
            ctxt := ctxt.outer_context;
        end;
        Fexternal_values.AddOrSetValue(val.name, rResult);
    end;
    Result := rResult;
end;

procedure CondContext._AddOpInternal(op: TFOperation);
var
  LRemResult: Tuple<TArray<TFOperation>, TArray<TFOperation>>;
begin
    if op.inputs.Count = 0 then
    begin
        //If we're in a while loop, remove any control inputs from outside the
        // loop.
        LRemResult := _RemoveExternalControlEdges(op);
        for var i := 0 to Length(op.control_inputs) -1 do
        begin
            var input_op := op.control_inputs[i];
            if not OpInContext(input_op) then
            begin
               op._add_control_input(Fpivot.op);
               Break;
            end;
        end;
    end else
    begin
        var real_x : TFTensor;
        // Make each input to 'op' available in this CondContext. If an input is
        // already part of this context there's nothing to do, but if it's
        // external, AddValue() will handle adding the appropriate Switch node and
        // other bookkeeping.
        for var index : Integer := 0 to  op.inputs.Count -1 do
        begin
            var x := op.inputs[index];
            if (op.Tipo = 'Merge') and (x.op.tipo = 'NextIteration') then
            begin
                //# Edge case: if we're importing a while loop inside this CondContext,
                //# AddValue() will not correctly handle the NextIteration inputs to
                //# Merge node. The problem is that the NextIteration should also be
                //# part of this context, but if we're importing it won't have been
                //# processed and added to the context yet, so AddValue() will try to
                //# add a Switch which results in an invalid graph. Instead, we use the
                //# NextIteration input as-is here, and it will eventually be added to
                //# the context via AddOp().
                real_x := x;
            end else
            begin
                real_x := AddValue(x);
            end;
            if real_x <> x then
                op._update_input(index, real_x);
        end;
        // Remove any external control dependency on this op.
        LRemResult := _RemoveExternalControlEdges(op);
        // TODO: implement below code dependencies
        //if (op.graph._is_function(op.type) || op.type == "SymbolicGradient")
        //    op._add_control_input(_pivot.op);
    end;

    // Mark op's outputs as seen by this context and any outer contexts.
    var output_names : TArray<String>;
    for var i := 0 to Length(op.outputs) -1 do
       output_names := output_names + [ op.outputs[i].Name ];

    var ctxt : TControlFlowContext := self;
    while ctxt <> nil do
    begin
        for var name in output_names do
            ctxt.values.Add(name);
        ctxt := ctxt.outer_context;
    end;

    if (Fouter_context <> nil) or (not control_flow_ops.IsLoopExit(op)) then
        op.graph.prevent_fetching(op);

    if Fouter_context <> nil then
        Fouter_context.AddInnerOp(op);
end;

function CondContext.BuildCondBranch<T>(fn: TFunc<TArray<T>>): tuple<TArray<T>, TArray<TFTensor>>;
begin
    // Add the subgraph defined by fn() to the graph.
    var pre_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
    var original_result : TArray<T> := fn;
    var post_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

    var v := TValue.From<TArray<T>>(original_result);
    if      v.TypeInfo = TypeInfo(TArray<TFTensor>)    then
    begin
        var Res := v.AsType<TArray<TFTensor>>;
        var aTens : TArray<TFTensor> := [];
        for var i := 0 to Length(Res) - 1 do
          aTens := aTens + [ _BuildCondTensor(Res[i]) ] ;

        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,aTens)
    end
    else if v.TypeInfo = TypeInfo(TArray<TFOperation>) then
    begin
        var Res := v.AsType<TArray<TFOperation>>;
        var aTens : TArray<TFTensor> := [];
        for var i := 0 to Length(Res) - 1 do
          aTens := aTens + [ _BuildCondTensor(Res[i]) ] ;

        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,aTens)
    end
    else if v.TypeInfo = TypeInfo(TArray<Single>) then
    begin
        var fv : TArray<Single> :=  v.AsType< TArray<Single> >;
        var res := Tops.convert_to_tensor(fv[0]);
        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result,[ _BuildCondTensor(res) ])
    end else
    begin
        Result := Tuple<TArray<T>, TArray<TFTensor>>.Create(original_result, nil)
    end;

end;

function CondContext.BuildCondBranch<T>(fn: TFunc<T>): Tuple<T, TFTensor>;
begin
    // Add the subgraph defined by fn() to the graph.
    var pre_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
    var original_result : T := fn;
    var post_summaries := Tops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

    //TODO: port this chunck of missing code:
    (*
    if len(post_summaries) > len(pre_summaries):
        new_summaries = post_summaries[len(pre_summaries):]
        summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
        summary_ref[:] = pre_summaries
        with ops.control_dependencies(new_summaries):
        if original_result is None:
            return no_op(), None
        else:
            original_result = nest.map_structure(array_ops.identity,
                                                original_result)
    *)

    var v := TValue.From<T>(original_result);
    if      v.TypeInfo = TypeInfo(TFTensor)    then
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(v.AsType<TFTensor>))
    end
    else if v.TypeInfo = TypeInfo(TFOperation) then
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(v.AsType<TFOperation>))
    end
    else if v.TypeInfo = TypeInfo(TArray<Single>) then
    begin
        var fv : TArray<Single> :=  v.AsType< TArray<Single> >;
        var res := Tops.convert_to_tensor(fv[0]);
        Result := Tuple<T, TFTensor>.Create(original_result, _BuildCondTensor(res))
    end else
    begin
        Result := Tuple<T, TFTensor>.Create(original_result, nil)
    end;

end;

function CondContext._ProcessOutputTensor(_val: TFTensor): TFTensor;
begin
    var real_val := _val;
    if not Fvalues.Contains(_val.name) then
    begin
        // Handle the special case of lambda: x
        Fvalues.Add(_val.name);
        if Fouter_context <> nil then
        begin
            real_val := Fouter_context.AddValue(_val);
            Fvalues.Add(real_val.name);
            Fexternal_values.AddOrSetValue(real_val.name, real_val);
        end;
        var res := control_flow_ops._SwitchRefOrTensor(real_val, Fpred);
        real_val := res[Fbranch];
        Fexternal_values.AddOrSetValue(_val.name, real_val);
    end else
    begin
        var external_val : TFTensor := nil;
        if Fexternal_values.ContainsKey(_val.name) then
            external_val := Fexternal_values[_val.name];
        if external_val <> nil then
            real_val := external_val;
    end;
    Result := real_val;
end;

function CondContext._BuildCondTensor(v: ITensorOrOperation): TFTensor;
begin
    if v is TFOperation then
    begin
        var op : TFOperation := v as TFOperation;
        Result := control_flow_ops.with_dependencies([ op ], Fpivot);
    end
    else if v is TFTensor then
    begin
        var t : TFTensor := v as TFTensor;
        Result := _ProcessOutputTensor(t);
    end else
    begin
        Result := _ProcessOutputTensor( Tops.convert_to_tensor(v) );
    end;
end;


end.
