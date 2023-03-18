unit TensorFlow.control_flow_ops;
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
   uses  System.SysUtils,
         System.Generics.Collections,

         Spring,
         Spring.Collections.Enumerable,

         TensorFlow.DApiBase,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.ControlFlowState,
         TensorFlow.Interfaces,
         TensorFlow.Core ;

type
  MergeOutput = record
    private
       function GetItem(idx: Integer): TFTensor;
    public
       output      : TFTensor;
       value_index : TFTensor;

       constructor Create(values: TArray<TFTensor>);
       class operator implicit(merge: MergeOutput): TFTensor;

       property item[idx: Integer] : TFTensor read GetItem; default;
  end;

  control_flow_ops = record
     private
       class function _GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string = ''): TFOperation;static;
     public
       class function _convert_flows_to_tensorarrays(tensors_or_tensorarrays: TArray<ITensorOrTensorArray>; tensors_or_flows: TArray<TFTensor>): TArray<ITensorOrTensorArray>; static;
       class function tuple(tensors: TArray<TFTensor>; name: string = ''; control_inputs : TArray<TFOperation> = nil) : TArray<TFTensor>; static;
       class function group<T:  ITensorOrOperation>(inputs: TArray<T>; name : string= '') : TFOperation; static;
       /// <summary>
       /// Returns the value of an available element of `inputs`.
       ///
       /// This op tests each of the tensors in `inputs` in turn to determine if any of
       /// them is available.If it finds an available tensor, it returns it and its
       /// index in `inputs`.
       ///
       /// It is an error if more than one tensor in `inputs` is available.If no tensor
       /// in `inputs` is available, the returned tensor and index are not set.
       ///
       /// This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
       /// `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
       /// before merging.
       /// </summary>
       /// <param name="inputs">inputs: The input tensors, at most one of which is available.</param>
       /// <param name="name">A name for this operation (optional).</param>
       /// <returns></returns>
       class function merge(inputs: TArray<TFTensor>; name: string = ''): MergeOutput; static;
       ///  <summary>
       ///  Forwards `data` to an output determined by `pred`.
       ///  If `pred` is false, the `data` input is forwarded to the first output.
       ///  Otherwise, the data goes to the second output.
       ///
       ///  This op handles `Tensor`s and `IndexedSlices`.
       ///  </summary>
       ///  <param name="data">The tensor to be forwarded to the appropriate output.</param>
       ///  <param name="pred">A scalar that specifies which output port will receive data.</param>
       /// <param name="name"> A name for this operation (optional).</param>
       /// <returns>
       ///  `(output_false, output_true)`: If `pred` is true, data will be forwarded to
       /// `output_true`, otherwise it goes to `output_false`.
       /// </returns>
       class function _SwitchRefOrTensor(data: TFTensor; pred: TFTensor; name: string = 'Switch'): TArray<TFTensor>; static;
       /// <summary>
       /// Produces the content of `output_tensor` only after `dependencies`.
       ///
       /// In some cases, a user may want the output of an operation to be
       /// consumed externally only after some other dependencies have run
       /// first.This function ensures returns `output_tensor`, but only after all
       /// operations in `dependencies` have run.Note that this means that there is
       /// no guarantee that `output_tensor` will be evaluated after any `dependencies`
       /// have run.
       ///
       /// See also `tf.tuple` and `tf.group`.
       /// </summary>
       /// <param name="dependencies">Iterable of operations to run before this op finishes.</param>
       /// <param name="output_tensor">A `Tensor` or `IndexedSlices` that will be returned.</param>
       /// <param name="name">(Optional) A name for this operation.</param>
       /// <returns>Same as `output_tensor`.</returns>
       class function with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Does nothing. Only useful as a placeholder for control edges.
       /// </summary>
       /// <param name="name"></param>
       /// <returns></returns>
       class function no_op(name : string= ''): TFOperation; static;
       class function _Identity(data: TFTensor;  name : string = ''): TFTensor; static;
       class function ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor ; static;
       /// <summary>
       /// Forwards `data` to an output determined by `pred`.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="pred"></param>
       /// <param name="dtype"></param>
       /// <param name="name"></param>
       class function switch(data: TFTensor; pred: TFTensor; dtype : TF_DataType = DtInvalid; name: string = ''): TArray<TFTensor>; static;
       /// <summary>
       /// Create the state for all the while loops involved in one gradients().
       /// </summary>
       /// <param name="between_op_list"></param>
       /// <param name="between_ops"></param>
       /// <param name="colocate_gradients_with_ops"></param>
       class function MaybeCreateControlFlowState(between_op_list: TList<TFOperation>; between_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean) : ControlFlowState;Static;
       class function IsLoopExit(op: TFOperation): Boolean; static;
       class function  _NextIteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
       ///
       /// `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
       /// `false_fn` must have the same non-zero number and type of outputs.
       ///
       /// **WARNING**: Any Tensors or Operations created outside of `true_fn` and
       /// `false_fn` will be executed regardless of which branch is selected at runtime.
       ///
       /// Although this behavior is consistent with the dataflow model of TensorFlow,
       /// it has frequently surprised users who expected a lazier semantics.
       /// Consider the following simple program:
       ///
       /// z = tf.multiply(a, b)
       /// result = tf.cond(x &lt; y, ()=> tf.add(x, z), ()=> tf.square(y))
       ///
       /// If `x&lt;y`, the `tf.add` operation will be executed and `tf.square`
       /// operation will not be executed.Since `z` is needed for at least one
       /// branch of the `cond`, the `tf.multiply` operation is always executed,
       /// unconditionally.
       ///
       /// Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
       /// call to `cond`, and not at all during `Session.run()`). `cond`
       /// stitches together the graph fragments created during the `true_fn` and
       /// `false_fn` calls with some additional graph nodes to ensure that the right
       /// branch gets executed depending on the value of `pred`.
       ///
       /// `tf.cond` supports nested structures as implemented in
       /// `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
       /// same(possibly nested) value structure of lists, tuples, and/or named tuples.
       /// Singleton lists and tuples form the only exceptions to this: when returned by
       /// `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.
       /// This behavior is disabled by passing `strict= True`.
       /// </summary>
       /// <param name="pred"> A scalar determining whether to return the result of `true_fn` or
       /// `false_fn`.</param>
       /// <param name="true_fn">The callable to be performed if pred is true.</param>
       /// <param name="false_fn">The callable to be performed if pred is false.</param>
       /// <param name="strict"> A boolean that enables/disables 'strict' mode; see above.</param>
       /// <param name="name">Optional name prefix for the returned tensors.</param>
       /// <returns>Tensors returned by the call to either `true_fn` or `false_fn`. If the
       /// callables return a singleton list, the element is extracted from the list.</returns>
       class function cond(pred: TFTensor; true_fn : TFunc<TFTensor>= nil; false_fn: TFunc<TFTensor>= nil; name: string = ''): TFTensor; overload ;static;
       class function cond<T>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<T>>; name: string): TArray<TFTensor>; overload ;static;
        /// <summary>
        /// Repeat `body` while the condition `cond` is true.
        /// </summary>
        /// <param name="cond"></param>
        /// <param name="body"></param>
        /// <param name="loop_vars"></param>
        /// <param name="shape_invariants"></param>
        class function while_loop<TItem: IFromMergeVars<TItem>>(_cond: TFunc<TItem, TFTensor>; body: TFunc<TItem, TItem>; loop_vars: TItem;
                                          shape_invariants   : TArray<TFShape>= [];
                                          parallel_iterations: Integer = 10;
                                          back_prop          : Boolean= true;
                                          swap_memory        : Boolean= false;
                                          name               : string= '';
                                          maximum_iterations : TFTensor= nil;
                                          return_same_structure : Boolean= false): TItem; static;
  end;

implementation
      uses Tensorflow,
           TensorFlow.Tensor,
           Tensorflow.Utils,
           TensorFlow.Ops,
           Tensorflow.math_ops,
           Tensorflow.gen_array_ops,
           Tensorflow.array_ops,
           TensorFlow.gen_control_flow_ops,
           Tensorflow.tensor_array_ops,
           TensorFlow.control_flow_util,
           Tensorflow.CondContext ;

{ control_flow_ops }


class function control_flow_ops._SwitchRefOrTensor(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    data := Tops.convert_to_tensor_or_composite(data, DtInvalid, 'data');
    // NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
    // addresses the following scenario.
    //
    // Assume you execute Optimizer.apply_gradients() in a branch of a cond().
    //
    // 1. The update op is created inside a `with ops.colocate(var):` block
    //
    // 2. Some tensor `data` is captured and a switch is created in a
    //    `with ops.colocate_with(data):` block.
    //
    // with ops.colocate_with(var):
    //  with ops.colocate_with(data):
    //    op = ...
    //
    // var and data may be pinned to different devices, so we want to ops
    // created within ops.colocate_with(data) to ignore the existing stack.
    Tops.colocate_with(data, true);
    begin
        if data is TFTensor then
        begin
            if TDtypes.is_ref_dtype(data.dtype) then
            begin
                Result := gen_control_flow_ops.ref_switch(data, pred, name);
                Exit;
            end;
        end;
        Result := switch(data, pred, DtInvalid, name);
    end
end;

class function control_flow_ops.while_loop<TItem>(_cond: TFunc<TItem, TFTensor>; body: TFunc<TItem, TItem>; loop_vars: TItem; shape_invariants: TArray<TFShape>;
  parallel_iterations: Integer; back_prop, swap_memory: Boolean; name: string; maximum_iterations: TFTensor; return_same_structure: Boolean): TItem;
var
  orig_cond : TFunc<TItem, TFTensor>;
  orig_body : TFunc<TItem, TItem>;
begin
    var vValue : TValue := TValue.From<TItem>(loop_vars);
    TUtils.tf_with<TNameScope,TItem>(Tops.name_scope(name, 'while', @vValue), function(Scope: TNameScope): TItem
            Begin
                if loop_vars = nil then
                   raise Exception.Create('No loop variables provided');
                if not Assigned(_cond) then
                   raise Exception.Create('cond must be callable.');
                if not Assigned(body) then
                   raise Exception.Create('body must be callable.');
                if parallel_iterations < 1 then
                   raise Exception.Create('parallel_iterations must be a positive integer.');

                var try_to_pack := (loop_vars is TFTensor) and (not return_same_structure);
                var counter := constant_op.constant(0, maximum_iterations.dtype, 'iteration_counter');
                orig_cond := _cond;
                orig_body := body;

                var loop_vars_1 : LoopVar<TItem>  := nil;
                var body_buildloop : TFunc<LoopVar<TItem>, LoopVar<TItem>> := nil;
                var cond_buildloop : TFunc<LoopVar<TItem>, TFTensor>       := nil;

                if try_to_pack then
                begin

                end else
                begin
                    loop_vars_1 := LoopVar<TItem>.Create(counter, loop_vars);
                    cond_buildloop := function(item: LoopVar<TItem>) : TFTensor
                        begin
                            var i := item.Counter;
                            var lv:= item.Item;
                            var oc := orig_cond(lv);
                            Result := math_ops.logical_and(i < TTensor(maximum_iterations), oc);
                            Exit;
                        end;

                    body_buildloop := function(item:  LoopVar<TItem>):  LoopVar<TItem>
                        begin
                            var i := item.Counter;
                            var lv:= item.Item;
                            var ob := orig_body(lv);
                            Result := LoopVar<TItem>.Create(TTensor(i) + 1, ob);
                            Exit;
                        end;
                end;
                try_to_pack := false;

                var loop_context := WhileContext.Create(maximum_iterations, parallel_iterations, back_prop, swap_memory);

                if loop_context.outer_context = nil then
                    Tops.add_to_collection(tf.GraphKeys.WHILE_CONTEXT, loop_context);

                var res : LoopVar<TItem> := loop_context.BuildLoop<TItem>(cond_buildloop, body_buildloop, loop_vars_1, shape_invariants, return_same_structure);

                //if (maximum_iterations != null)
                Result := res.Item;
                //else
                //return results;
            end);
end;

class function control_flow_ops.merge(inputs: TArray<TFTensor>; name: string): MergeOutput;
begin
    for var i := 0 to Length(inputs) - 1 do
    begin
        if inputs[i] = nil then
          raise Exception.Create('At least one of the merge inputs is null: {inputs}');
    end;

    var vvalue : TArray<TValue>:= [ TValue.From<  TArray<TFTensor> >(inputs) ];
    Result := TUtils.tf_with<TNameScope,MergeOutput>(Tops.name_scope(name, 'Merge', @vvalue), function(scope: TNameScope): MergeOutput
                  begin
                       name := scope.ToString;
                       var a : TArray<TFTensor> := [];
                       for var i := 0 to Length(inputs) - 1 do
                       begin
                           var inp : TFTensor := inputs[i];
                           inp := Tops.internal_convert_to_tensor_or_indexed_slices(inp, DtInvalid, '', true);
                           a := a + [ inp ];
                       end;
                       Result := gen_control_flow_ops.merge(a, name);
                  end);
end;

class function control_flow_ops.cond(pred: TFTensor; true_fn, false_fn: TFunc<TFTensor>; name: string): TFTensor;
begin
    var vvalue : TArray<TValue>:= [ TValue.From< TFTensor >(pred) ];
    result := TUtils.tf_with<TNameScope,TFTensor>(Tops.name_scope(name, 'cond',@vvalue), function(scope: TNameScope): TFTensor
                  begin
                      if tf.Context.executing_eagerly then
                      begin
                          var bTensor : TTensor := pred;
                          var flag : Boolean := Boolean(bTensor);
                          if flag then
                          begin
                              Result := true_fn;
                              Exit;
                          end else
                          begin
                             Result := false_fn;
                             Exit;
                          end;
                      end;

                      // Add the Switch to the graph.
                      var switch_result := switch(pred, pred);
                      var p_2 := switch_result[0];
                      var p_1 := switch_result[1];
                      var pivot_1 := array_ops.identity(p_1, 'switch_t');
                      var pivot_2 := array_ops.identity(p_2, 'switch_f');
                      pred := array_ops.identity(pred, 'pred_id');

                      // Disable the fetching of tensors that are only on one branch of cond.
                      var aTensor : TArray<TFTensor>:=  [ p_1, p_2, pivot_1, pivot_2, pred ];
                      for var tensor in aTensor do
                          tensor.op.graph.prevent_fetching(tensor.op);

                      // Build the graph for the true branch in a new context.
                      var context_t := CondContext.Create(pred, pivot_1, 1);
                      var orig_res_t : ITensorOrOperation ;
                      var res_t      : TFTensor;
                      try
                          context_t.Enter_;
                          var tTupleBranch := context_t.BuildCondBranch<TFTensor>(true_fn);
                          orig_res_t := tTupleBranch.Value1;
                          res_t      := tTupleBranch.Value2;
                          context_t.ExitResult([ res_t ]);
                      finally
                          context_t.Exit_;
                      end ;
                      // Build the graph for the false branch in a new context.
                      var context_f  := CondContext.Create(pred, pivot_2, 0);
                      var orig_res_f : ITensorOrOperation ;
                      var res_f      : TFTensor ;
                      try
                          context_f.Enter_;
                          var fTupleBranch := context_f.BuildCondBranch<TFTensor>(false_fn);
                          orig_res_f := fTupleBranch.Value1;
                          res_f      := fTupleBranch.Value2;
                          context_f.ExitResult([ res_f ]);
                      finally
                          context_f.Exit_;
                      end;

                      var res_t_flat := [ res_t ];
                      var res_f_flat := [ res_f ];

                      var tMerge := merge([ res_t_flat[0], res_f_flat[0] ] )[0];
                      var merges : TArray<TFTensor> := [ tMerge ];

                      if orig_res_t is TFTensor then
                      begin
                         {var orig_res_tensor := orig_res_t as TFTensor;
                         var aRes := _convert_flows_to_tensorarrays([ orig_res_tensor ], merges);
                         merges := [];
                         for var i := 0 to Length(aRes) - 1 do
                         begin
                             merges := merges + [ aRes[i] as TFTensor ];
                         end; }
                      end else
                      begin

                      end;

                      if context_t.outer_context = nil then
                      begin
                          Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                          Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                      end;

                      Result := merges[0];
                  end);
end;

class function control_flow_ops.cond<T>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<T>>; name: string): TArray<TFTensor>;
var
  aRes : TArray<ITensorOrTensorArray>;
begin
    var vvalue : TArray<TValue>:= [ TValue.From< TFTensor >(pred) ];
    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>(Tops.name_scope(name, 'cond', @vvalue), function(scope: TNameScope): TArray<TFTensor>
                begin
                    if tf.Context.executing_eagerly then
                    begin
                        var bTensor : TTensor := pred;
                        var flag : Boolean := Boolean(bTensor);
                        if flag then
                        begin
                            var a : TArray<T> := true_fn;
                            var v := TValue.From<TArray<T>>(a);
                            Result := v.astype< TArray<TFTensor> >;
                            Exit;
                        end else
                        begin
                            var a : TArray<T> := false_fn;
                            var v := TValue.From<TArray<T>>(a);
                            Result := v.astype< TArray<TFTensor> >;
                            Exit;
                        end;
                    end;

                    // Add the Switch to the graph.
                    var switch_result := switch(pred, pred);
                    var p_2 := switch_result[0];
                    var p_1 := switch_result[1];
                    var pivot_1 := array_ops.identity(p_1, 'switch_t');
                    var pivot_2 := array_ops.identity(p_2, 'switch_f');
                    pred := array_ops.identity(pred, 'pred_id');

                    // Disable the fetching of tensors that are only on one branch of cond.
                    var aTensor : TArray<TFTensor>:=  [ p_1, p_2, pivot_1, pivot_2, pred ];
                    for var tensor in aTensor do
                        tensor.op.graph.prevent_fetching(tensor.op);

                    // Build the graph for the true branch in a new context.
                    var context_t := CondContext.Create(pred, pivot_1, 1);
                    var orig_res_t : TArray<T> ;
                    var res_t      : TArray<TFTensor>;
                    try
                        context_t.Enter_;
                        var tTupleBranch := context_t.BuildCondBranch<T>(true_fn);
                        orig_res_t := tTupleBranch.Value1;
                        res_t      := tTupleBranch.Value2;
                        context_t.ExitResult(res_t );
                    finally
                        context_t.Exit_;
                    end ;
                    // Build the graph for the false branch in a new context.
                    var context_f  := CondContext.Create(pred, pivot_2, 0);
                    var orig_res_f : TArray<T> ;
                    var res_f      : TArray<TFTensor> ;
                    try
                        context_f.Enter_;
                        var fTupleBranch := context_f.BuildCondBranch<T>(false_fn);
                        orig_res_f := fTupleBranch.Value1;
                        res_f      := fTupleBranch.Value2;
                        context_f.ExitResult(res_f );
                    finally
                        context_f.Exit_;
                    end;

                    var res_t_flat :=  res_t ;
                    var res_f_flat :=  res_f ;

                    var merges : TArray<TFTensor> := [];
                    for var i := 0 to Length(res_t_flat) - 1 do
                    begin
                        var tMerge := merge([ res_t_flat[i], res_f_flat[i] ] )[0];
                        merges :=  merges + [ tMerge ];
                    end;

                    if TypeInfo(T) = TypeInfo(TFTensor) then
                    begin
                       var orig_res_tensor : TArray<ITensorOrTensorArray>;
                       for var i := 0 to Length(orig_res_t) - 1 do
                       begin
                           var v := TValue.From<T>(orig_res_t[i]);
                           orig_res_tensor := orig_res_tensor + [ v.AsType<TFTensor> ] ;
                       end;

                       {aRes := _convert_flows_to_tensorarrays(orig_res_tensor, merges);
                       merges := [];
                       for var i := 0 to Length(aRes) - 1 do
                           merges := merges + [ aRes[i] as TFTensor ]; }
                    end
                    else if TypeInfo(T) = TypeInfo(Single) then
                    begin

                    end else
                    begin

                    end;

                    if context_t.outer_context = nil then
                    begin
                        Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_t);
                        Tops.add_to_collection(tf.GraphKeys.COND_CONTEXT, context_f);
                    end;

                    Result := merges;

               end);
end;

class function control_flow_ops.group<T>(inputs: TArray<T>; name: string): TFOperation;
begin
    var vInputs := TValue.From< TArray<T> >(inputs) ;

    Result := TUtils.tf_with<TNameScope,TFOperation>( TOps.name_scope(name, 'group_deps', @vInputs),
                function(v1: TNameScope): TFOperation
                  begin
                      name := v1.ToString;

                      // Sorts *inputs according to their devices.
                      var ops_on_device := TDictionary< string, TList<T> >.Create;
                      for var inp in inputs do
                      begin
                          if ops_on_device.ContainsKey(inp.Device) then
                              ops_on_device[inp.Device].Add(inp)
                          else
                              ops_on_device.add( inp.Device, TList<T>.Create([ inp ]) );
                      end;
                      // 1-level tree. The root node is the returned NoOp node.
                      if ops_on_device.Count = 1 then
                      begin
                          var dev  := ops_on_device.Keys.ToArray[0];
                          var deps := ops_on_device.Values.ToArray[0];
                          var aOp : TArray<TFOperation> := [];
                          for var i := 0 to deps.Count - 1 do
                             aOp := aOp + [ deps[i].op ];
                          Result := _GroupControlDeps(dev, aOp, name);
                          Exit;
                      end;
                      // 2-level tree. The root node is the returned NoOp node.
                      // deps contains 1 NoOp node for each device.
                      raise TFException.Create('control_flow_ops.group');

                  end );

end;

class function control_flow_ops.IsLoopExit(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Exit') or (op.Tipo = 'RefExit');
end;

class function control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean): ControlFlowState;
begin
    var loop_state : ControlFlowState := nil;
    var pos : Integer := 0;
    while pos < between_op_list.Count do
    begin
        var op := between_op_list[pos];
        if IsLoopExit(op) then
        begin
            if loop_state = nil then
            begin
                loop_state := ControlFlowState.Create;
            end;
            if colocate_gradients_with_ops then
                Tops.colocate_with(op);
            loop_state.AddWhileContext(op, between_op_list, between_ops);
        end;
        Inc(pos);
    end;
    Result := loop_state;
end;

class function control_flow_ops.no_op(name: string): TFOperation;
begin
    Result := gen_control_flow_ops.no_op(name)
end;

class function control_flow_ops.switch(data, pred: TFTensor; dtype: TF_DataType; name: string): TArray<TFTensor>;
begin
    var vInputs := TValue.From< TArray<TFTensor> >([data, pred]) ;

    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope(name, 'Switch', @vInputs),
                function(v1: TNameScope): TArray<TFTensor>
                  begin
                      name := v1.ToString;
                      data := Tops.internal_convert_to_tensor_or_indexed_slices(data, dtype, 'data', true);

                      pred := Tops.convert_to_tensor(pred, DtInvalid, 'pred');
                      Result := gen_control_flow_ops.switch(data, pred, name);
                  end );
end;

class function control_flow_ops.tuple(tensors: TArray<TFTensor>; name: string; control_inputs: TArray<TFOperation>): TArray<TFTensor>;
begin
    var vInputs := TValue.From< TArray<TFTensor> >(tensors) ;

    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope(name, 'tuple', @vInputs),
                function(v1: TNameScope): TArray<TFTensor>
                  begin
                      name := v1.ToString;


                      var gating_ops : TArray<TFOperation>:= [];
                      for var i := 0 to Length(tensors)-1 do
                      begin
                          if tensors[i] <> nil then
                             gating_ops := gating_ops + [ tensors[i].Op ];
                      end;

                      if control_inputs <> nil  then
                      begin
                          for var c in control_inputs do
                             gating_ops := gating_ops + [ c ];
                      end;
                      // Note that in order to ensure ordering in the pbtxt, we must take care to
                      // ensure the order here.
                      var l_gating_ops := Enumerable<TFOperation>.create(gating_ops);
                      l_gating_ops := l_gating_ops.OrderBy<Integer>(function (o: TFOperation): Integer
                                                                      begin
                                                                        Result := o.id;
                                                                      end);
                      var gate := group<TFOperation>(l_gating_ops.ToArray);
                      var tpl := TList<TFTensor>.Create ;
                      try
                        for var t in tensors do
                        begin
                            if t <> nil then tpl.Add( with_dependencies([ gate ], t) )
                            else             tpl.Add(nil);
                        end;
                        Result := tpl.ToArray;
                      finally
                        tpl.Free;
                      end;

                  end );
end;

class function control_flow_ops.with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string): TFTensor;
begin
   var adeps : TArray<TValue> := [];
   var aValue : TArray<TValue> := [];

   for var i := 0 to Length(dependencies) -1  do
     adeps := adeps + [ TValue.From<TFOperation>(dependencies[i]) ] ;

   aValue := adeps + [ output_tensor ];

   //TODO: missing original code
   //if context.executing_eagerly():
   //    return output_tensor
   Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'control_dependency', @aValue),
                function(v1: TNameScope): TFTensor
                  begin
                      name := v1.ToString;

                      Tops.colocate_with(output_tensor);
                      Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(adeps),
                                  function(v1: TControlDependenciesController): TFTensor
                                    begin
                                        output_tensor := Tops.convert_to_tensor_or_composite(output_tensor);
                                        Result := _Identity(output_tensor, name);
                                    end );
                  end );
end;

class function control_flow_ops.ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
begin
    var val := op.outputs[index];
    if not control_flow_util.IsSwitch(op) then
    begin
        if val.dtype = TF_DataType.TF_RESOURCE then
           raise TFException.Create('Not Implemented - ("ZerosLikeOutsideLoop")');
        Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end else
    begin
        var op_ctxt := op._get_control_flow_context;
        if op_ctxt <> nil then
        begin
            // We are in a cond context. Use a switch to create zeros only when needed.
            var pred   := op_ctxt.pred;
            var branch := op_ctxt.branch;
            var switch_val := switch(op.inputs[0], pred)[1 - branch];
            var pivot      := array_ops.identity(switch_val);
            if val.dtype = TDtypes.cresource then
               raise TFException.Create('Not Implemented');
            var zeros_shape := array_ops.shape_internal(switch_val,'', false);
            // Ensure ops created within array_ops.zeros are dominated by switch in
            // cond context.
            var aValue : TArray<TValue> := [pivot];
            Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(aValue),
                          function(v1: TControlDependenciesController): TFTensor
                            begin
                                Result := array_ops.zeros(zeros_shape, val.dtype);
                            end );
        end else
        begin
            Result := array_ops.zeros_like(val, DtInvalid,'', false);
        end;
    end;
end;

class function control_flow_ops._convert_flows_to_tensorarrays(tensors_or_tensorarrays: TArray<ITensorOrTensorArray>; tensors_or_flows: TArray<TFTensor>): TArray<ITensorOrTensorArray>;
begin
    Result := [];
    for var i := 0 to Length(tensors_or_tensorarrays)- 1 do
    begin
         var ta       := tensors_or_tensorarrays[i];
         var t_or_flow : TFTensor := tensors_or_flows[i];

         if ta is TTensorArray  then
         begin
            var ta_1 := ta as TTensorArray;
            var res := tensor_array_ops.build_ta_with_new_flow(ta_1, t_or_flow)  ;
            Result := Result + [ res ] ;
         end else
         begin
             var res := t_or_flow ;
             Result := Result + [ res ] ;
         end;
    end;
end;

class function control_flow_ops._GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string): TFOperation;
begin
   var aValue : TArray<TValue> := [];
   for var i := 0 to Length(deps) -1  do
     aValue := aValue + [ TValue.From<TFOperation>(deps[i]) ] ;

   Result := TUtils.tf_with<TControlDependenciesController,TFOperation>( Tops.control_dependencies(aValue),
                                          function(v1: TControlDependenciesController): TFOperation
                                            begin
                                                if dev = '' then
                                                  Result := gen_control_flow_ops.no_op(name)
                                                else
                                                   Result := gen_control_flow_ops.no_op(name);
                                            end );
end;

class function control_flow_ops._Identity(data: TFTensor; name: string): TFTensor;
begin
    data := Tops.internal_convert_to_tensor_or_composite(data, DtInvalid, '', true);
    if Ord(data.dtype) > 100 then
       raise TFException.Create('Not Implemented "_Identity"')
    else
        Result := gen_array_ops.identity(data, name);
end;

class function control_flow_ops._NextIteration(data: TFTensor; name: string): TFTensor;
begin
    data := Tops.internal_convert_to_tensor_or_indexed_slices(data, DtInvalid, '', true);

    if TDTypes.is_ref_dtype(data.dtype) then  Result := gen_control_flow_ops.ref_next_iteration(data, name)
    else                                      Result := gen_control_flow_ops.next_iteration(data, name);
end;



{ MergeOutput }

constructor MergeOutput.Create(values: TArray<TFTensor>);
begin
    output      := values[0];
    value_index := values[1];
end;

function MergeOutput.GetItem(idx: Integer): TFTensor;
begin
    case idx of
      0: Result := output;
      1: Result := value_index;
    else
      Result := nil;
    end;
end;

class operator MergeOutput.implicit(merge: MergeOutput): TFTensor;
begin
   Result := merge.output;
end;

end.


