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
         Numpy.Axis,
         TensorFlow.ControlFlowState,
         TensorFlow.Context ;

type
  control_flow_ops = record
     private
       class function _GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string = ''): TFOperation;static;
     public
       class function tuple(tensors: TArray<TFTensor>; name: string = ''; control_inputs : TArray<TFOperation> = nil) : TArray<TFTensor>; static;
       class function group<T:  ITensorOrOperation>(inputs: TArray<T>; name : string= '') : TFOperation; static;
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
       class function cond<TFTensor>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<TFTensor>>; name: string): TArray<TFTensor>; overload ;static;
  end;

implementation
      uses Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops,
           Tensorflow.NameScope,
           Tensorflow.gen_array_ops,
           Tensorflow.array_ops,
           TensorFlow.gen_control_flow_ops,
           TensorFlow.control_flow_util ;

{ control_flow_ops }

class function control_flow_ops.cond(pred: TFTensor; true_fn, false_fn: TFunc<TFTensor>; name: string): TFTensor;
begin
    { TODO -oMax -c : Implementare 03/12/2022 10:57:15 }
    raise Exception.Create('control_flow_ops.cond');
end;

class function control_flow_ops.cond<TFTensor>(pred: TFTensor; true_fn, false_fn: TFunc<TArray<TFTensor>>; name: string): TArray<TFTensor>;
begin
   { TODO -oMax -c : Implementare 03/12/2022 10:57:15 }
    raise Exception.Create('control_flow_ops.cond');
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

end.
