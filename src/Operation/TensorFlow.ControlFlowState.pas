unit TensorFlow.ControlFlowState;
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

interface
    uses System.Generics.Collections,
         System.Rtti,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase;
type

    /// <summary>
    /// The state used for constructing the gradient graph for a while loop.
    /// </summary>
    GradLoopState_helper  = class helper for GradLoopState
       private

       public
    end;

    /// <summary>
    /// Maintain the mapping from the loops to their grad states.
    /// </summary>
    ControlFlowState  = class
       private
         Fmap : TDictionary<TControlFlowContext, GradLoopState>;
       public
         constructor Create;

         function  ProcessUnusedLoopExits(pending_count: TDictionary<string, Integer>; to_ops_set: TList<TFOperation>) : TArray<TFTensor>;
         procedure EnterGradWhileContext(op: TFOperation; before: Boolean);
         procedure ExitGradWhileContext(op: TFOperation; before: Boolean);
         //  def ZerosLikeForExit(self, val):
         //    """Create zeros_like gradient for a loop exit.
         //    If the result of a loop variable is not used but is involved in
         //    computing the result of some needed loop variable, we create a
         //    zero-valued tensor that is fed as gradient for the Exit node of that
         //    loop variable. Note that val.op is an Exit, and this method must be
         //    called in the control flow context where gradients() is called.
         //    Args:
         //      val: The output tensor of an Exit op.
         //    Returns:
         //      A zero tensor of the same shape of val.
         //    """
         //    val_shape = val.get_shape()
         //    forward_ctxt = val.op._get_control_flow_context()
         //    outer_forward_ctxt = forward_ctxt.outer_context
         //    if outer_forward_ctxt:
         //      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
         //    outer_grad_state = None
         //    if outer_forward_ctxt:
         //      outer_grad_state = self._map.get(outer_forward_ctxt)
         //    if outer_grad_state:
         //      # This is a nested loop.
         //      if val_shape.is_fully_defined():
         //        # If the shape is known statically, just create a zero tensor
         //        # with the right shape in the right context.
         //        outer_grad_state.grad_context.Enter()
         //        result = array_ops.zeros(val_shape.dims, val.dtype)
         //        outer_grad_state.grad_context.Exit()
         //      else:
         //        # Only the shape of value is needed for backprop.
         //        forward_ctxt.outer_context.Enter()
         //        shape = array_ops.shape_internal(val, optimize=False)
         //        forward_ctxt.outer_context.Exit()
         //        # Save the shape to a stack.
         //        history_shape = outer_grad_state.AddForwardAccumulator(shape)
         //        # Get the shape back from the stack.
         //        outer_grad_ctxt = outer_grad_state.grad_context
         //        outer_grad_ctxt.Enter()
         //        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
         //            history_shape, shape)
         //        result = array_ops.zeros(real_shape, val.dtype)
         //        outer_grad_ctxt.Exit()
         //    else:
         //      # This is not a nested loop.
         //      if val_shape.is_fully_defined():
         //        # If the shape is known statically, just create a zero tensor
         //        # with the right shape.
         //        result = array_ops.zeros(val_shape.dims, val.dtype)
         //      else:
         //        result = array_ops.zeros_like(val, optimize=False)
         //    return result
         function ZerosLike(op: TFOperation; index: Integer): TFTensor;
         /// <summary>
         /// Create zeros_like gradient for a loop exit.
         /// </summary>
         /// <param name="val"></param>
         /// <returns></returns>
         function ZerosLikeForExit(val: TFTensor): TFTensor;
         function ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
         procedure PostProcessing;
         //  def AddWhileContext(self, op, between_op_list, between_ops):
         //    """Add the grad state for the while loop that op belongs to.
         //    Note that op is an Exit, and this method must be called in
         //    the control flow context where gradients() is called.
         //    Note that this method modifies `between_op_list` and `between_ops`.
         //    """
         //    forward_ctxt = _GetWhileContext(op)
         //    grad_state = self._map.get(forward_ctxt)
         //    if grad_state is None:
         //      # This is a new while loop so create a grad state for it.
         //      outer_forward_ctxt = forward_ctxt.outer_context
         //      if outer_forward_ctxt:
         //        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
         //      outer_grad_state = None
         //      if outer_forward_ctxt:
         //        outer_grad_state = self._map.get(outer_forward_ctxt)
         //      grad_state = GradLoopState(forward_ctxt, outer_grad_state)
         //      self._map[forward_ctxt] = grad_state
         //      # We need to include all exits of a loop for backprop.
         //      for loop_exit in grad_state.forward_loop_exits:
         //        if loop_exit.op not in between_ops:
         //          between_ops.add(loop_exit.op)
         //          between_op_list.append(loop_exit.op)
         procedure AddWhileContext(op: TFOperation; between_op_list: TList<TFOperation>; between_ops: TList<TFOperation>);
         /// <summary>
         /// Return the grad state for this op if it's in a forward loop context.
         /// </summary>
         /// <param name="op"></param>
         /// <param name="before"></param>
         /// <returns></returns>
         function GetGradState(op: TFOperation; before: Boolean): GradLoopState;

    end;

implementation
       uses Tensorflow.Utils,
            TensorFlow.control_flow_util,
            TensorFlow.Constant_op,
            TensorFlow.control_flow_ops,
            Tensorflow.array_ops;

{ ControlFlowState }

constructor ControlFlowState.Create;
begin
   FMap := TDictionary<TControlFlowContext, GradLoopState>.Create;
end;

procedure ControlFlowState.AddWhileContext(op: TFOperation; between_op_list, between_ops: TList<TFOperation>);
begin
    var forward_ctxt := op.GetWhileContext;
    var grad_state : GradLoopState := nil;
    if Fmap.ContainsKey(forward_ctxt) then grad_state := Fmap[forward_ctxt];

    if grad_state = nil then
    begin
        var outer_grad_state : GradLoopState := nil;
        var outer_forward_ctxt := forward_ctxt.outer_context;
        if outer_forward_ctxt <> nil then
            outer_forward_ctxt := outer_forward_ctxt.GetWhileContext;
        if outer_forward_ctxt <> nil then
            outer_grad_state := Fmap[outer_forward_ctxt];
        grad_state := GradLoopState.Create(forward_ctxt, outer_grad_state);
        Fmap[forward_ctxt] := grad_state;
        // We need to include all exits of a loop for backprop.
        for var loop_exit in grad_state.forward_loop_exits do
        begin
            if not between_ops.Contains(loop_exit.op) then
            begin
                between_ops.add(loop_exit.op);
                between_op_list.add(loop_exit.op);
            end;
        end;
    end;
end;

procedure ControlFlowState.EnterGradWhileContext(op: TFOperation; before: Boolean);
begin
    var grad_state := GetGradState(op, before);
    if grad_state <> nil then
        grad_state.grad_context.Enter_;
end;

procedure ControlFlowState.ExitGradWhileContext(op: TFOperation; before: Boolean);
begin
    var grad_state := GetGradState(op, before);
     if grad_state <> nil then
        grad_state.grad_context.Exit_
end;

function ControlFlowState.GetGradState(op: TFOperation; before: Boolean): GradLoopState;
begin
    var forward_ctxt : TControlFlowContext ;
    if (before) and (control_flow_util.IsLoopExit(op)) then
    begin
        forward_ctxt := op._get_control_flow_context;
        forward_ctxt := forward_ctxt.outer_context;
        if forward_ctxt <> nil then
            forward_ctxt := forward_ctxt.GetWhileContext;
    end else
        forward_ctxt := control_flow_util.GetWhileContext(op);

    if forward_ctxt <> nil then
    begin
        Result := nil;
        if Fmap.ContainsKey(forward_ctxt) then
          Result := Fmap[forward_ctxt] ;
        Exit;
    end;
    Result := nil;
end;

procedure ControlFlowState.PostProcessing;
begin
    for var grad_state in Fmap.Values do
    begin
        for var b_merge in grad_state.switch_map.Values do
        begin
            if b_merge.op.inputs[0] = b_merge.op.inputs[1] then
            begin
                var next_grad_val : TFTensor ;
                // The value of this loop variable at iteration i+1 doesn't
                // depend on its value at iteration i. So use zeros as the
                // gradients for all iterations > 0.
                var dtype := b_merge.op.inputs[0].dtype;
                var shape := b_merge.op.inputs[0].shape;
                if shape.IsFullyDefined then
                begin
                    grad_state.grad_context.Enter_;
                    // Create a zeros and use it for iterations > 0.
                    var grad_val  := constant_op.constant(0, dtype,  @shape);
                    next_grad_val := control_flow_ops._NextIteration(grad_val);
                    grad_state.grad_context.Exit_;
                end else
                begin
                    raise TFException.Create('PostProcessing shape is not fully defined.');
                end;
                b_merge.op._update_input(1, next_grad_val);
            end;
        end;
    end;
end;

function ControlFlowState.ProcessUnusedLoopExits(pending_count: TDictionary<string, Integer>; to_ops_set: TList<TFOperation>): TArray<TFTensor>;
begin
    var loop_exits := TList<TFTensor>.Create;
    for var grad_state in Fmap.Values do
    begin
        for var y in grad_state.forward_loop_exits do
        begin
            if not pending_count.ContainsKey(y.op.name) then
            begin
                grad_state.pending_exits_count := grad_state.pending_exits_count - 1;
                if  not to_ops_set.Contains(y.op) then
                    grad_state.unused_exits.add(y);
                if grad_state.pending_exits_count = 0 then
                    loop_exits.AddRange(grad_state.unused_exits);
            end;
        end;
        for var y in grad_state.forward_context.loop_enters do
        begin
            if not pending_count.ContainsKey(y.op.name) then
                pending_count.AddOrSetValue(y.op.name, 1);
        end;
    end;
    Result := loop_exits.ToArray
end;

function ControlFlowState.ZerosLike(op: TFOperation; index: Integer): TFTensor;
begin
    if control_flow_util.IsLoopSwitch(op) then
        Exit(nil);
    if op.graph.building_function then
        Exit( array_ops.zeros_like(op.outputs[index]) );

    control_flow_util.IsSwitch(op);
    var forward_ctxt := control_flow_util.GetWhileContext(op);

    var grad_state := nil;
    if Fmap.ContainsKey(forward_ctxt) then
      grad_state := Fmap[forward_ctxt] ;

    // op is not in a while loop that is part of gradients().
    if grad_state = nil then
        Exit( ZerosLikeOutsideLoop(op, index) );

    raise TFException.Create('ZerosLike');
end;

function ControlFlowState.ZerosLikeForExit(val: TFTensor): TFTensor;
begin

    var val_shape          := val.shape;
    var forward_ctxt       := val.op._get_control_flow_context;
    var outer_forward_ctxt := forward_ctxt.outer_context;
    if outer_forward_ctxt <> nil then
        outer_forward_ctxt := outer_forward_ctxt.GetWhileContext;
    var outer_grad_state : GradLoopState := nil;
    if outer_forward_ctxt <> nil then
    begin
        if Fmap.ContainsKey(outer_forward_ctxt) then
           outer_grad_state := Fmap[outer_forward_ctxt] ;
    end;
    // This is a nested loop.
    if outer_grad_state <> nil then
    begin
        raise TFException.Create('ZerosLikeForExit');
    end else
    begin
        // If the shape is known statically, just create a zero tensor
        // with the right shape.
        if val_shape.IsFullyDefined then
            Result := array_ops.zeros(val_shape.dims, val.dtype)
        else
            Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end;
end;

function ControlFlowState.ZerosLikeOutsideLoop(op: TFOperation; index: Integer): TFTensor;
begin
    var val := op.outputs[index];
    if not control_flow_util.IsSwitch(op) then
    begin
        if val.dtype = Tdtypes.cresource then
           raise TFException.Create('ZerosLikeOutsideLoop');
        (*return array_ops.zeros(
          gen_resource_variable_ops.variable_shape(val),
          dtype: default_gradient.get_zeros_dtype(val));*)
        Result := array_ops.zeros_like(val, DtInvalid,'', false);
    end else
        raise TFException.Create('ZerosLikeOutsideLoop');
end;

end.
