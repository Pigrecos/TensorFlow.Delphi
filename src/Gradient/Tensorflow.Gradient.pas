unit Tensorflow.Gradient;
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
         System.TypInfo,
         Generics.Defaults,

         Spring,
         Spring.Collections,
         Spring.Collections.Enumerable,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         Tensorflow.Core,
         TensorFlow.DApiBase,
         TensorFlow.Variable,
         TensorFlow.ControlFlowState;

type
  gradients_impl  = class

      public
      class function gradients(ys: TArray<TFTensor>; xs: TArray<TFTensor>; grad_ys: TArray<TFTensor> = nil; name: string = 'gradients'; colocate_gradients_with_ops: Boolean = false; gate_gradients : Boolean= false; aggregation_method : PInteger= nil) : TArray<TFTensor>;
  end;

  gradients_util = record
     private
        /// <summary>
        /// Fill in default values for grad_ys.
        /// </summary>
        /// <param name="grad_ys">List of gradients, can contain None.</param>
        /// <param name="ys">List of tensors.</param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="gradient_uid"></param>
        class function _DefaultGradYs(grad_ys: TArray<TFTensor>; ys: TArray<TFTensor>; colocate_gradients_with_ops: Boolean; gradient_uid: string = '__unsupported__'): TArray<TFTensor>; Static;
        /// <summary>
        /// Initialize the pending count for ops between two lists of Operations.
        /// 'pending_count[op]' indicates the number of backprop inputs
        /// to this operation.
        /// </summary>
        /// <param name="to_ops"></param>
        /// <param name="from_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="func_graphs"></param>
        /// <param name="xs"></param>
        class function _PendingCount(to_ops: TList<TFOperation>; from_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean; func_graphs: TList<TFuncGraph>; xs: TArray<TFTensor>): Tuple<TArray<TFOperation>, TDictionary<string, integer>, ControlFlowState>; static;
        class function _GetGrad(grads: TDictionary<string, TList<TList<TFTensor>>>; t: TFTensor): TFTensor; static;
        class function _GetGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation): TList<TList<TFTensor>>; static;
        /// <summary>
        /// Sets gradient "grad" in "grads" for tensor "t".
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="t"></param>
        /// <param name="grad"></param>
        class procedure _SetGrad(grads: TDictionary<string, TList<TList<TFTensor>> >;t: TFTensor; grad: TFTensor); static;
        /// <summary>
        /// The set of ops that terminate the gradient computation.
        /// </summary>
        /// <param name="from_ops">list of Operations.</param>
        /// <param name="stop_gradient_ops">list of Operations never to backprop through.</param>
        /// <param name="pending_count">mapping from operation to number of backprop inputs.</param>
        /// <param name="xs">list of Tensors.</param>
        /// <returns>The set of operations.</returns>
        class function _StopOps(from_ops: TList<TFOperation>; stop_gradient_ops: TList<TFOperation>; pending_count: TDictionary<string, integer>; xs: TArray<TFTensor>): TArray<TFOperation>; static;
        class Procedure _maybe_colocate_with(op: TFOperation; gradient_uid: string; colocate_gradients_with_ops: Boolean); static;
        class function _AggregatedGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation; gradient_uid: string; loop_state: ControlFlowState; aggregation_method : Integer = 0) : TList<TList<TFTensor>>; Static;
        /// <summary>
        /// Adds tensors from potentially multiple devices.
        /// </summary>
        /// <param name="tensor_list"></param>
        /// <param name="gradient_uid"></param>
        /// <returns></returns>
        class function _MultiDeviceAddN(tensor_list: TArray<TFTensor>; gradient_uid: string): TFTensor; static;
        class function _IsPartitionedCall(op: TFOperation) : Boolean; static;
        class function _IsTrainable(tensor: TFTensor): Boolean; static;
        class function _MaybeCompile(scope: string; op: TFOperation; out_grads: TArray<TFTensor>; grad_fn: TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>): Enumerable<TFTensor>; static;
        class procedure _VerifyGeneratedGradients(grads: TArray<TFTensor>; op: TFOperation); static;
        class function _NonEagerInputs(op: TFOperation; xs: TArray<TFTensor>): Enumerable<TFTensor>; static;
        /// <summary>
        /// Mark all ops reached from "from_ops"
        /// </summary>
        /// <param name="from_ops"></param>
        /// <param name="reached_ops"></param>
        /// <param name="func_graphs"></param>
        class procedure _MarkReachedOps(from_ops: TList<TFOperation>; var reached_ops: TList<TFOperation>; func_graphs: TList<TFuncGraph>); static;
        class function  _IsBackpropagatable(tensor: TFTensor): Boolean; static;
        /// <summary>
        /// Return true if op has real gradient.
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="op"></param>
        /// <returns></returns>
        class function _HasAnyNotNoneGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation): Boolean; static;
        /// <summary>
        /// Update pending count for the inputs of op and enqueue ready ops.
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="op"></param>
        /// <param name="queue"></param>
        /// <param name="pending_count"></param>
        /// <param name="loop_state"></param>
        /// <param name="xs"></param>
        class procedure _UpdatePendingAndEnqueueReady(grads        : TDictionary<string, TList<TList<TFTensor>>>;
                                                      op           : TFOperation;
                                                      queue        : TQueue<TFOperation>;
                                                      pending_count: TDictionary<string, Integer>;
                                                      loop_state   : ControlFlowState;
                                                      xs           : TArray<TFTensor>); static;
     public
        class function IsTrainable(tensor: TFTensor): Boolean; static;

        class function  _GradientsHelper(ys                          : TArray<TFTensor>;
                                         xs                          : TArray<TFTensor>;
                                         grad_ys                     : TArray<TFTensor> = nil;
                                         name                        : string = 'gradients';
                                         colocate_gradients_with_ops : Boolean = false;
                                         gate_gradients              : Boolean = false;
                                         aggregation_method          : Integer = 0;
                                         stop_gradients              : TArray<TFTensor> = nil;
                                         src_graph                   : TFGraph = nil): TArray<TFTensor>; Static;
  end;

var
  gradientFunctions : TDictionary<string, TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>> >;

implementation
       uses Tensorflow,
            TensorFlow.Ops,
            Tensorflow.Utils,
            Tensorflow.math_ops,
            TensorFlow.gen_math_ops,
            Tensorflow.array_ops,
            Tensorflow.gen_array_ops,
            TensorFlow.control_flow_ops,
            TensorFlow.control_flow_util;

{ gradients_util }

class function gradients_util._AggregatedGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation; gradient_uid: string; loop_state: ControlFlowState;
  aggregation_method: Integer): TList< TList<TFTensor> >;
begin
    var out_grads := _GetGrads(grads, op);

    for var i := 0 to out_grads.Count - 1 do
    begin
        var out_grad := out_grads[i];
        if loop_state <> nil  then
        begin
            if (out_grads.Count > 1) and
               (out_grads[1].Count > 0) and
               (control_flow_util.IsLoopSwitch(op)) then
                continue;
        end;
        // Aggregate multiple gradients, and convert [] to None.
        if out_grad.Count > 0 then
        begin
            var used : string := '';
            if out_grad.Count < 2 then
            begin
                used := 'nop';
                if out_grad.Count = 0 then
                  raise TFException.Create('_AggregatedGrads out_grad.Length = 0');
                out_grads[i] := TList<TFTensor>.Create([ out_grad[0] ]);
            end else
            begin
                used := 'add_n';
                out_grads[i] := TList<TFTensor>.Create( [ _MultiDeviceAddN(out_grad.ToArray, gradient_uid) ] );
            end;
        end else
        begin
            out_grads[i] := nil;
        end
    end;
    Result := out_grads;
end;

class function gradients_util._MultiDeviceAddN(tensor_list: TArray<TFTensor>; gradient_uid: string): TFTensor;
begin
    // Basic function structure comes from control_flow_ops.group().
    // Sort tensors according to their devices.
    var tensors_on_device := TDictionary<string, TList<TFTensor>>.Create;
    for var tensor in tensor_list do
    begin
        if not tensors_on_device.ContainsKey(tensor.Device) then
            tensors_on_device.AddOrSetValue(tensor.Device, TList<TFTensor>.Create) ;
        tensors_on_device[tensor.Device].Add(tensor);
    end;
    // For each device, add the tensors on that device first.
    var summands := TList<TFTensor>.Create;
    for var dev in tensors_on_device.Keys do
    begin
        var tensors := tensors_on_device[dev];
        Tops._colocate_with_for_gradient(tensors[0].op, gradient_uid, true);
        summands.Add( math_ops.add_n(tensors.ToArray) );
    end;
    Result := math_ops.add_n(summands.ToArray);
end;

class function gradients_util._DefaultGradYs(grad_ys, ys: TArray<TFTensor>; colocate_gradients_with_ops: Boolean; gradient_uid: string): TArray<TFTensor>;
begin
    var new_grad_ys := TList<TFTensor>.Create ;

    var i : Integer := 0;
    for var item in TUtils.zip<TFTEnsor>( TList<TFTensor>.Create(ys), TList<TFTEnsor>.Create(grad_ys) ) do
    begin
        var y      := item.Value1;
        var grad_y := item.Value2;
        _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops);
        if grad_y = nil then
        begin
            if TDTypes.is_complex(y.dtype) then
               raise Exception.Create('Gradients of complex tensors must set grad_ys (y.dtype = {y.dtype})');

            var shape    := array_ops.shape(y);
            var constant := constant_op.constant(1, y.dtype, 'grad_ys_'+ IntToStr(i));
            var fill     := gen_array_ops.fill(shape, constant);
            new_grad_ys.Add(fill);
            continue;
        end;
        if TDTypes.is_floating(y.dtype) or TDTypes.is_integer(y.dtype)  then
        begin
        end;
        // Create a grad_y tensor in the name scope of the gradient.
        new_grad_ys.Add(array_ops.identity(grad_y, 'grad_ys_' + IntToStr(i)));
    end;
    Result := new_grad_ys.ToArray;
end;

class function gradients_util._PendingCount(to_ops, from_ops: TList<TFOperation>; colocate_gradients_with_ops: Boolean; func_graphs: TList<TFuncGraph>;
  xs: TArray<TFTensor>): Tuple<TArray<TFOperation>, TDictionary<string, integer>, ControlFlowState>;
var
  reached_ops      : TList<TFOperation>;
  reachable_to_ops : Tarray<TFOperation>;
  i                : Integer;
  op               : TFOperation;
begin
    var Comparer := TDelegatedComparer<TFOperation>.Construct(
          function (const L, R: TFOperation): Integer
          begin
            Result := NativeInt(L.Handle) - NativeInt(R.Handle);
          end);

    // Mark reachable ops from from_ops.
    reached_ops := TList<TFOperation>.Create;
    try
      _MarkReachedOps(from_ops, reached_ops, func_graphs);
      // X in reached_ops iff X is reachable from from_ops by a path of zero or more
      // backpropagatable tensors.
      reachable_to_ops := Enumerable<TFOperation>.Create(to_ops.ToArray).Where(function(const x: TFOperation) : Boolean
                                            begin
                                                var iFound : Integer;
                                                reached_ops.sort(Comparer);
                                                Result := reached_ops.BinarySearch(x,iFound,Comparer);
                                            end).ToArray;
      var between_ops     := TList<TFOperation>.Create;
      var between_op_list := TList<TFOperation>.Create;
      var queue : TQueue<TFOperation> := TQueue<TFOperation>.Create(to_ops);
      reached_ops.sort(Comparer);
      var iFound : Integer;
      try
        while queue.Count > 0 do
        begin
            op := queue.Dequeue;
            if reached_ops.BinarySearch(op,iFound,Comparer) then
            begin
                between_ops.Add(op);
                between_op_list.Insert(between_op_list.Count, op);
                // Clear the boolean so we won't add the inputs again.
                reached_ops.Remove(op);
                for var inp in _NonEagerInputs(op, xs) do
                    queue.Enqueue(inp.op);
            end;
        end;
        // X in between_ops iff X is on a path of zero or more backpropagatable tensors
        // between from_ops and to_ops
        // 'loop_state' is None if there are no while loops.
        var loop_state := control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops);
        // Initialize pending count for between ops.
        var pending_count := TDictionary<string, integer>.Create;
        for i := 0 to between_op_list.Count-1 do
        begin
            op := between_op_list[i];
            for var x: TFTensor in _NonEagerInputs(op, xs) do
            begin
                if between_ops.Contains(x.op) then
                begin
                    if not pending_count.ContainsKey(x.op.name) then
                        pending_count.AddOrSetValue(x.op.name, 0);
                    pending_count[x.op.name] := pending_count[x.op.name] + 1;
                end;
            end;
        end;
        Result := Tuple.Create(reachable_to_ops, pending_count, loop_state);
      finally
        between_ops.free;
        between_op_list.free;
        queue.free;
      end;
    finally
      reached_ops.free;
    end;
end;

class procedure gradients_util._SetGrad(grads: TDictionary<string, TList<TList<TFTensor>> >; t, grad: TFTensor);
begin
    var op := t.op;
    var op_grads : TList<TList<TFTensor>> := nil ;
    if  grads.ContainsKey(op.name)  then
       op_grads := grads[op.name];
    if op_grads = nil then
    begin
        op_grads := TList<TList<TFTensor>>.Create( Enumerable<TFTensor>.Create(op.outputs)
                                                           .Select< TList<TFTensor> >( function(x: TFTensor) : TList<TFTensor>
                                                                      begin
                                                                          Result := TList<TFTensor>.Create;
                                                                      end).ToArray);
        grads.AddOrSetValue(op.name, op_grads);
    end;
    var t_grads := op_grads[t.value_index];
    if (t_grads.Count > 0) and (control_flow_util.IsLoopSwitch(op)) then  op_grads[t.value_index][0] := grad
    else                                                                  t_grads.Add(grad);
end;

class function gradients_util._StopOps(from_ops, stop_gradient_ops: TList<TFOperation>; pending_count: TDictionary<string, integer>; xs: TArray<TFTensor>): TArray<TFOperation>;
begin
    var stop_ops := TList<TFOperation>.Create;

    for var i: Integer := 0 to from_ops.Count -1 do
    begin
        var op := from_ops[i];
        var is_stop_op : Boolean := true;
        for var inp in _NonEagerInputs(op, xs) do
        begin
            if not pending_count.ContainsKey(inp.op.name) then
                pending_count.AddOrSetValue(inp.op.name, 0);
            if pending_count[inp.op.name] > 0 then
            begin
                is_stop_op := false;
                break;
            end;
        end;
        if is_stop_op then
            stop_ops.Insert(0, op);
    end;
    stop_ops.AddRange(Enumerable<TFOperation>.Create(stop_gradient_ops.ToArray)
                                .Where( function(const x: TFOperation): Boolean
                                         begin
                                             Result := not stop_ops.Contains(x);
                                         end).ToArray );
    Result := stop_ops.ToArray;
end;

class function gradients_util._HasAnyNotNoneGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation): Boolean;
begin
    var out_grads := _GetGrads(grads, op);
    for var out_grad in out_grads do
    begin
        for var i := 0 to out_grad.Count -1 do
        begin
           var t : TFTensor := out_grad[i];
           if t <> nil then
             Exit(True)
        end;
    end;
    Result := false;
end;

class procedure gradients_util._UpdatePendingAndEnqueueReady(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation; queue: TQueue<TFOperation>;
  pending_count: TDictionary<string, Integer>; loop_state: ControlFlowState; xs: TArray<TFTensor>);
begin
    for var x in _NonEagerInputs(op, xs) do
    begin
        if not pending_count.ContainsKey(x.op.name) then
            pending_count.AddOrSetValue(x.op.name, 0);
        pending_count[x.op.name] := pending_count[x.op.name] - 1;

        var ready := pending_count[x.op.name] = 0;
        if (loop_state <> nil) and ( not ready) then
            ready := (pending_count[x.op.name] > 0) and (control_flow_util.IsLoopSwitch(x.op) );

        if ready then
        begin
            // if x is an exit without real gradient, defer processing them.
            if control_flow_util.IsLoopExit(x.op) then
            begin
                var grad_state := loop_state.GetGradState(x.op, false);
                grad_state.deferred_exits.add(x);
                grad_state.pending_exits_count :=  grad_state.pending_exits_count - 1;
                // We now have all the exits so process them.
                if grad_state.pending_exits_count = 0 then
                begin
                    var has_not_none_grad := false;
                    for var i : Integer := 0 to grad_state.deferred_exits.Count - 1 do
                    begin
                        var y := grad_state.deferred_exits[i];
                        if _HasAnyNotNoneGrads(grads, y.op) then
                        begin
                            has_not_none_grad := true;
                            queue.Enqueue(y.op);
                        end
                        else
                            grad_state.unused_exits.add(y);
                    end;
                    if has_not_none_grad then
                    begin
                        // For an unused exit, if it has trainable outputs, backprop
                        // a zero gradient. Otherwise, just ignore it.
                        for var i:= 0 to grad_state.unused_exits.Count-1 do
                        begin
                            var y := grad_state.unused_exits[i];
                            if IsTrainable(y) then
                                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y));
                            queue.Enqueue(y.op);
                        end;
                    end else
                    begin
                        // All exits are "unused" so use None as gradient.
                        for var i:= 0 to grad_state.unused_exits.Count-1 do
                        begin
                            var y := grad_state.unused_exits[i];
                            queue.Enqueue(y.op);
                        end;
                    end;
                end;
            end else
            begin
                queue.Enqueue(x.op);
            end;
        end;
    end;
end;

class procedure gradients_util._MarkReachedOps(from_ops: TList<TFOperation>; var reached_ops: TList<TFOperation>; func_graphs: TList<TFuncGraph>);
var
  iFound : Integer;
begin
    var Comparer := TDelegatedComparer<TFOperation>.Construct(
    function (const L, R: TFOperation): Integer
    begin
       Result := NativeInt(L.Handle) - NativeInt(R.Handle);
    end);

    var queue : TQueue<TFOperation> := TQueue<TFOperation>.Create(from_ops);
    try
      while queue.Count > 0 do
      begin
          var op := queue.Dequeue;
          queue.TrimExcess ;
          reached_ops.Sort(Comparer);
          if not reached_ops.BinarySearch(Op,iFound,Comparer) then
          begin
              reached_ops.Add(op);

              for var output in op.outputs do
              begin
                  if _IsBackpropagatable(output) then
                  begin
                      var c := output.consumers;
                      for var operazione in c  do
                         queue.Enqueue(operazione)
                  end;
              end;
          end;
      end;
    finally
      queue.Free;
    end;
end;

class function gradients_util._IsBackpropagatable(tensor: TFTensor): Boolean;
begin
    if _IsTrainable(tensor) then
    begin
        Result := true;
    end else
    begin
        var dtype := TDTypes.as_base_dtype(tensor.dtype);
        Result    := TArray.contains<TF_DataType>([ TF_DataType.TF_BFLOAT16, TF_DataType.TF_VARIANT ],dtype );
    end
end;

class procedure gradients_util._VerifyGeneratedGradients(grads: TArray<TFTensor>; op: TFOperation);
begin
    if (op.tipo = 'While') or (op.tipo = 'StatelessWhile') then
        Exit;
    if Length(grads) <> Length(op.inputs.inputs) then
       raise Exception.Create( Format('Num gradients %d generated for op do not match num inputs %d',[Length(grads), Length(op.inputs.inputs)]));
end;

class procedure gradients_util._maybe_colocate_with(op: TFOperation; gradient_uid: string; colocate_gradients_with_ops: Boolean);
begin

end;

class function gradients_util._NonEagerInputs(op: TFOperation; xs: TArray<TFTensor>): Enumerable<TFTensor>;
begin
    var a : TArray<TFTensor> := [];
    for var i := 0 to op.inputs.count - 1 do
        a := a + [ op.inputs[i] ];
    Result := Enumerable<TFTensor>.Create(a);
end;

class function gradients_util._IsPartitionedCall(op: TFOperation): Boolean;
begin
    Result := (op.Tipo = 'PartitionedCall') or (op.Tipo = 'StatefulPartitionedCall');
end;

class function gradients_util.IsTrainable(tensor: TFTensor): Boolean;
const
  cc_Trainable : array[0..5] of TF_DataType = (TF_HALF, TF_FLOAT, TF_DOUBLE, TF_COMPLEX64, TF_COMPLEX128, TF_RESOURCE);
begin
     var dtype := TDTypes.as_base_dtype(tensor.dtype);
     Result    := TArray.Contains<TF_DataType>(cc_Trainable,dtype )
end;

class function gradients_util._IsTrainable(tensor: TFTensor): Boolean;
const
  c_Trainable : array[0..5] of TF_DataType = (TF_HALF, TF_FLOAT, TF_DOUBLE, TF_COMPLEX64, TF_COMPLEX128, TF_RESOURCE);
begin
     var dtype := TDTypes.as_base_dtype(tensor.dtype);
     Result := TArray.Contains<TF_DataType>(c_Trainable,dtype )
end;

class function gradients_util._MaybeCompile(scope: string; op: TFOperation; out_grads: TArray<TFTensor>;
  grad_fn: TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>): Enumerable<TFTensor>;
begin
   if scope.EndsWith('/') then
      scope := scope.Substring(0, scope.Length - 1);

   Result := Enumerable<TFTensor>.Create( grad_fn(op, out_grads) );
end;

class function gradients_util._GetGrad(grads: TDictionary<string, TList<TList<TFTensor>>>; t: TFTensor): TFTensor;
begin
    var op := t.op;
    if not grads.ContainsKey(op.name) then
        Exit(nil);
    var op_grads := grads[op.name];
    var t_grad   := op_grads[t.value_index];
    Result       := t_grad[0];
end;

class function gradients_util._GetGrads(grads: TDictionary<string, TList<TList<TFTensor>>>; op: TFOperation): TList<TList<TFTensor>>;
begin
    if grads.ContainsKey(op.name) then
        Result := grads[op.name]
    else begin
        var aList := TList<TList<TFTensor>>.Create;
        for var i := 0 to Length(op.outputs)-1 do
          aList.add( TList<TFTensor>.Create  );
        Result := aList;
    end;
end;

class function gradients_util._GradientsHelper(ys, xs, grad_ys: TArray<TFTensor>; name: string; colocate_gradients_with_ops, gate_gradients: Boolean; aggregation_method: Integer;
                                                stop_gradients: TArray<TFTensor>; src_graph: TFGraph): TArray<TFTensor>;
var
  to_ops,
  from_ops,
  stop_gradient_ops : TList<TFOperation>;
  grads             : TDictionary<string, TList<TList<TFTensor>> >;
  reachable_to_ops  : TArray<TFOperation> ;
  loop_state        : ControlFlowState;
  pending_count     : TDictionary<string, Integer> ;
  aconcat           : TArray<TFTensor>;
  vValues           : TArray<TValue>;
begin
    if src_graph = nil then
        src_graph := Tops.get_default_graph;
    // If src_graph is a _FuncGraph (i.e. a function body), gather it and all
    // ancestor graphs. This is necessary for correctly handling captured values.
    var func_graphs := TList<TFuncGraph>.Create;
    var curr_graph := src_graph;
    if src_graph is TFuncGraph then
    begin
        var func_graph := src_graph as TFuncGraph;
        func_graphs.Add(func_graph);
        curr_graph := func_graph.OuterGraph;
    end;
    if stop_gradients = nil then
        stop_gradients := [];
    if grad_ys = nil then
        SetLength(grad_ys,Length(ys));
    // Iterate over the collected ops.
    (*
     * grads: op => list of gradients received on each output endpoint of the
     * op.  The gradients for each endpoint are initially collected as a list.
     * When it is time to call the op's gradient function, for each endpoint we
     * aggregate the list of received gradients into a Add() Operation if there
     * is more than one.
     *)
    grads             := TDictionary<string, TList<TList<TFTensor>> >.Create;
    reachable_to_ops  := nil;
    loop_state        := nil;
    pending_count     := nil;
    aconcat := ys + xs + stop_gradients+ grad_ys;
    vValues := [];
    for var i := 0 to Length(aconcat)-1 do
       vValues := vValues + [ aconcat[i] ];

    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;

    TUtils.tf_with<TNameScope>( TOps.name_scope(name, 'gradients', @newVal),
        procedure(v1: TNameScope)
        var
          grad_scope : string;
          begin
              grad_scope := v1.ToString;
              // Get a uid for this call to gradients that can be used to help
              // cluster ops for compilation.
              var gradient_uid := curr_graph.unique_name('uid');
              ys := Tops.convert_n_to_tensor_or_indexed_slices(ys, DtInvalid, 'y');
              xs := Tops.internal_convert_n_to_tensor_or_indexed_slices(xs, DtInvalid, 'x', true);
              grad_ys := _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid);
              (*
               * The approach we take here is as follows: Create a list of all ops in the
               * subgraph between the ys and xs.  Visit these ops in reverse order of ids
               * to ensure that when we visit an op the gradients w.r.t its outputs have
               * been collected.  Then aggregate these gradients if needed, call the op's
               * gradient function, and add the generated gradients to the gradients for
               * its input.
               *)
              // Initialize the pending count for ops in the connected subgraph from ys
              // to the xs.
              to_ops :=  TList<TFOperation>.Create( Enumerable<TFTensor>.create(ys).Select<TFOperation>( function(x : TFTensor): TFOperation
                                                                                                             begin
                                                                                                                 Result := x.op;
                                                                                                             end).ToArray);
              from_ops :=TList<TFOperation>.Create( Enumerable<TFTensor>.create(xs).Select<TFOperation>( function(x : TFTensor): TFOperation
                                                                                                             begin
                                                                                                                 Result := x.op;
                                                                                                             end).ToArray);
              stop_gradient_ops := TList<TFOperation>.Create( Enumerable<TFTensor>.create(stop_gradients).Select<TFOperation>( function(x : TFTensor): TFOperation
                                                                                                             begin
                                                                                                                 Result := x.op;
                                                                                                             end).ToArray);

              var tPCount := _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs);
              reachable_to_ops := tpCount.Value1;
              pending_count    := tpCount.Value2;
              loop_state       := tpCount.Value3;
              // Add the initial gradients for the ys.
              for var i := 0 to Length(ys) - 1 do
              begin
                  var y      := ys[i];
                  var grad_y := grad_ys[i] ;
                  _SetGrad(grads, y, grad_y);
              end;
              // Initialize queue with to_ops.
              var queue := TQueue<TFOperation>.Create;
              // Add the ops in 'to_ops' into the queue.
              var to_ops_set := TList<TFOperation>.Create;
              for var op in to_ops do
              begin
                  // 'ready' handles the case where one output gradient relies on
                  // another output's gradient.
                  if not pending_count.ContainsKey(op.name) then
                      pending_count.AddOrSetValue(op.name, 0);
                  var ready : Boolean := pending_count[op.name] = 0;
                  if (ready) and (not to_ops_set.Contains(op)) and (TArray.Contains<TFOperation>(reachable_to_ops,op) )  then
                  begin
                      to_ops_set.Add(op);
                      queue.Enqueue(op);
                  end;
              end;
              if loop_state <> nil  then
              begin
                  var loop_exits := loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set);
                  for var y in loop_exits do
                  begin
                      //if(IsTrainable(y))
                      raise TFException.Create('Not Implemented' );
                  end;
              end;
              var stop_ops := _StopOps(from_ops, stop_gradient_ops, pending_count, xs);
              while queue.Count > 0 do
              begin
                  // generate gradient subgraph for op.
                  var op := queue.Dequeue;
                  _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops);

                  if loop_state <> nil then
                      loop_state.EnterGradWhileContext(op, true);

                  var out_grads : TList< TList<TFTensor> > := _AggregatedGrads(grads, op, gradient_uid, loop_state, aggregation_method);
                  if loop_state <> nil then
                      loop_state.ExitGradWhileContext(op, true);

                  var in_grads : Enumerable<TFTensor> := nil;
                  var grad_fn  : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>> := nil;
                  var is_partitioned_call := _IsPartitionedCall(op);
                  var is_func_call := false;
                  var has_out_grads : Boolean := False ;
                  for var i := 0 to out_grads.Count- 1 do
                  begin
                      if out_grads[i] <> nil then
                      begin
                          has_out_grads := True;
                          break
                      end;
                  end;
                  if (has_out_grads) and (not TArray.Contains<TFOperation>(stop_ops,op)) then
                  begin
                      // A grad_fn must be defined, either as a function or as None
                      // for ops that do not have gradients.
                      try
                        grad_fn := Tops.get_gradient_function(op);

                      Except
                          if is_func_call then
                          begin
                              if is_partitioned_call then
                              begin

                              end else
                              begin

                              end;
                          end else
                          begin
                             Raise TFException.Create('No gradient defined for operation '+ op.name +' (op type: '+op.tipo+')');
                          end;
                      end;
                  end;
                  if loop_state <> nil then
                      loop_state.EnterGradWhileContext(op, false);
                  if ((is_func_call) or (grad_fn <> nil)) and (has_out_grads) then
                  begin
                      // NOTE: If _AggregatedGrads didn't compute a value for the i'th
                      // output, it means that the cost does not depend on output[i],
                      // therefore dC/doutput[i] is 0.
                      for var i := 0 to out_grads.Count - 1 do
                      begin
                          var out_grad : TList<TFTensor> := out_grads[i];
                          if (out_grad = nil) and ((grad_fn = nil) or (_IsTrainable(op.outputs[i]))) then
                          begin
                              // Only trainable outputs or outputs for a function call that
                              // will use SymbolicGradient get a zero gradient. Gradient
                              // functions should ignore the gradient for other outputs.
                              if loop_state <> nil then
                                  out_grads[i] := TList<TFTensor>.Create([ loop_state.ZerosLike(op, i) ])
                              else
                                  out_grads[i] := TList<TFTensor>.Create([ control_flow_ops.ZerosLikeOutsideLoop(op, i) ]);
                          end;
                      end;
                      TUtils.tf_with<TNameScope>( TOps.name_scope(op.name, '_grad'),
                         procedure(v1: TNameScope)
                          begin
                              if grad_fn <> nil then
                              begin
                                  var eout_grads := Enumerable< TList<TFTensor> >.Create(out_grads.ToArray);

                                  var array_out_grads := Enumerable<TList<TFTensor>>( eout_grads
                                       .Where(function(const x:TList<TFTensor>): Boolean
                                                begin
                                                    Result := x <> nil;
                                                end))
                                       .Select<TFTensor>(function(x: TList<TFTensor>) : TFTensor
                                                begin
                                                    Result := x[0];
                                                end).ToArray;

                                  in_grads := _MaybeCompile(grad_scope, op, array_out_grads, grad_fn);
                              end else
                              begin
                                  raise Exception.Create('Not Implemented "lambda: _SymGrad(op, out_grads)"');
                              end;
                              _VerifyGeneratedGradients(in_grads.toArray, op);
                              if (gate_gradients) and (in_grads.Count( function(const x: TFTensor): Boolean
                                                                        begin
                                                                            Result := x <> nil
                                                                        end) > 1)  then
                              begin
                                  Tops._colocate_with_for_gradient(nil, gradient_uid, true);
                                  in_grads := Enumerable<TFTensor>.Create( control_flow_ops.tuple(in_grads.ToArray) );
                              end;
                          end);
                  end else
                  begin
                      // If no grad_fn is defined or none of out_grads is available,
                      // just propagate a list of None backwards.
                      var a : TArray<TFTensor>; SetLength(a, _NonEagerInputs(op, xs).Count );
                      in_grads := Enumerable<TFTensor>.Create(a);
                  end;
                  var inputs := _NonEagerInputs(op, xs).ToList;
                  for  var i := 0 to inputs.Count -1 do
                  begin
                      var t_in    := inputs[i];
                      var in_grad := in_grads.ToArray[i];
                      if in_grad <> nil then
                      begin
                          if ( not(in_grad = nil) ) and
                             (in_grad.Tag.TypeInfo = nil) and // maybe a IndexedSlice
                             (t_in.dtype <> TF_DataType.TF_RESOURCE) then
                          begin
                              in_grad.shape := t_in.shape;
                          end ;
                          _SetGrad(grads, t_in, in_grad);
                      end;
                  end ;
                  if loop_state <> nil then
                      loop_state.ExitGradWhileContext(op, false);

                  // Update pending count for the inputs of op and enqueue ready ops.
                  _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs);
              end;
          end );
    if loop_state <> nil then
        loop_state.PostProcessing;
    Result := [];
    for var i := 0 to Length(xs) -1 do
       Result := Result + [ _GetGrad(grads, xs[i]) ];
end;

{ gradients_impl }

class function gradients_impl.gradients(ys, xs, grad_ys: TArray<TFTensor>; name: string; colocate_gradients_with_ops, gate_gradients: Boolean;
  aggregation_method: PInteger): TArray<TFTensor>;
begin
    var aggrMethod : Integer := 0;
    if aggregation_method <> nil then
        aggrMethod := aggregation_method^;

    Result := gradients_util._GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops, gate_gradients,aggrMethod);
end;

end.



