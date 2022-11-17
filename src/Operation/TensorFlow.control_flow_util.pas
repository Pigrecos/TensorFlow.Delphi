unit TensorFlow.control_flow_util;

interface
    uses System.SysUtils,

         Spring,

         TensorFlow.DApiBase,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.Context,
         TensorFlow.ControlFlowState ;

type

  control_flow_util = record
    private

    public
      /// <summary>
      /// Return true if `op` is a Switch.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsSwitch(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return true if `op` is an Exit.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopExit(op: TFOperation): Boolean; static;
      /// <summary>
      /// Returns true if `op` is an Enter.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopEnter(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return true iff op is a loop invariant.
      /// </summary>
      /// <param name="op"></param>
      /// <returns></returns>
      class function IsLoopConstantEnter(op: TFOperation): Boolean; static;
      class function GetWhileContext(op: TFOperation): WhileContext; static;
      class function IsCondSwitch(op: TFOperation): Boolean; static;
      class function IsLoopSwitch(op: TFOperation): Boolean; static;
      /// <summary>
      /// Return the control flow context for the output of an op.
      /// </summary>
      class function  GetOutputContext(op: TFOperation): TControlFlowContext; static;
      class procedure CheckInputFromValidContext(op: TFOperation; input_op: TFOperation); static;
      class function  GetLoopConstantEnter(value: TFTEnsor): TFOperation; static;
      class function  IsContainingContext(ctxt: WhileContext; maybe_containing_ctxt: WhileContext): Boolean; static;
      class function  GetContainingWhileContext(ctxt: TControlFlowContext; stop_ctxt : TControlFlowContext= nil): WhileContext; static;
  end;

implementation
        uses Tensorflow,
             Tensorflow.Utils,
             TensorFlow.Ops,
             Tensorflow.NameScope;

{ control_flow_util }

class procedure control_flow_util.CheckInputFromValidContext(op, input_op: TFOperation);
begin
    var op_ctxt    := op._get_control_flow_context;
    var input_ctxt := GetOutputContext(input_op);
    var valid := false;
    if input_ctxt = nil then
        valid := true
    else if op_ctxt = input_ctxt then
        valid := true
    else
    begin
        var while_ctxt       := GetContainingWhileContext(op_ctxt);
        var input_while_ctxt := GetContainingWhileContext(input_ctxt);
        if while_ctxt = nil then
        begin
            // Neither op nor input_op is in a while loop, but one or both are in
            // conds. We allow this, although execution will fail if the branch
            // corresponding to input_op's cond context isn't taken.
            if input_while_ctxt = nil then
                valid := true;
            // Invalid if op isn't in a while loop and input_op is. Unless...
            if IsLoopEnter(op) then
                // WhileContext._BuildLoop clears context for Enter nodes.
                valid := true;
            if IsSwitch(op) then
                // CondContext.AddValue clears context for Switch nodes.
                valid := true;
        end
        else if IsContainingContext(while_ctxt, input_while_ctxt) then
        begin
            // input_op is in a while loop which contains op's while loop (or not in a
            // while loop at all).
            valid := true;
        end
        else if (while_ctxt.grad_state <> nil) and (IsContainingContext(while_ctxt.grad_state.forward_context, input_while_ctxt) ) then
        begin
            valid := true;
        end
        else
           raise TFException.Create('CheckInputFromValidContext');
    end;
    if not valid then
       raise TFException.Create('CheckInputFromValidContext');

end;

class function control_flow_util.GetContainingWhileContext(ctxt, stop_ctxt: TControlFlowContext): WhileContext;
begin
    while ctxt <> nil do
    begin
        if (ctxt.IsWhileContext) or (ctxt = stop_ctxt) then
            Exit ( ctxt as WhileContext );
        ctxt := ctxt.outer_context;
    end;
    Result := nil;
end;

class function control_flow_util.GetLoopConstantEnter(value: TFTEnsor): TFOperation;
begin
    var id_ops : TArray<String> := [ 'Switch', 'RefSwitch', 'Identity', 'RefIdentity' ];
    var op := value.op;
    while TArray.Contains<String>(id_ops, op.tipo) do
        op := op.inputs[0].op;

    Result := nil;
    if IsLoopConstantEnter(op) then
       Result := op;
end;

class function control_flow_util.GetOutputContext(op: TFOperation): TControlFlowContext;
begin
    var ctxt := op._get_control_flow_context;
    // Exit nodes usually have a control flow context, except in the case where the
    // exit node was imported via import_graph_def (in which case no nodes have
    // control flow contexts).
    if (ctxt <> nil) and (IsLoopExit(op)) then
        ctxt := ctxt.outer_context;
    Result := ctxt;
end;

class function control_flow_util.GetWhileContext(op: TFOperation): WhileContext;
begin
    Result := op.GetWhileContext
end;

class function control_flow_util.IsCondSwitch(op: TFOperation): Boolean;
begin
    if  not IsSwitch(op) then
        Exit( false );
    if (op.outputs = nil) or (Length(op.outputs) = 0) then
        Exit( false );
    // Switch nodes are not part of the cond control flow context that they
    // represent, so consider the consumers of its outputs to determine if it is
    // cond switch or not. A switch is a cond switch iff all its consumers are in
    // cond contexts.
    var is_cond_switch := true;
    for var o in op.outputs do
    begin
        for var c in o.consumers do
        begin
            var ctxt := c._get_control_flow_context;
            if IsLoopEnter(c) then
                ctxt := ctxt.outer_context;
            is_cond_switch := (is_cond_switch) and ( (ctxt <> nil) and (ctxt.IsCondContext) );
        end;
    end;
    Result := is_cond_switch;
end;

class function control_flow_util.IsContainingContext(ctxt, maybe_containing_ctxt: WhileContext): Boolean;
begin
   while ctxt <> maybe_containing_ctxt do
   begin
      if ctxt = nil then
          Exit( false );
      ctxt := ctxt.outer_context as WhileContext;
   end;
   Result := true;
end;

class function control_flow_util.IsLoopConstantEnter(op: TFOperation): Boolean;
begin
    Result := (IsLoopEnter(op)) and (op.get_attr<boolean>('is_constant'));
end;

class function control_flow_util.IsLoopEnter(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Enter') or (op.tipo = 'RefEnter');
end;

class function control_flow_util.IsLoopExit(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Exit') or (op.tipo = 'RefExit');
end;

class function control_flow_util.IsLoopSwitch(op: TFOperation): Boolean;
begin
    if IsSwitch(op) then
    begin
        var ctxt := op._get_control_flow_context;
        Result := (ctxt <> nil) and (ctxt.IsWhileContext) and (not IsCondSwitch(op) );
        Exit;
    end;
    Result := false;
end;

class function control_flow_util.IsSwitch(op: TFOperation): Boolean;
begin
    Result := (op.tipo = 'Switch') or (op.tipo = 'RefSwitch');
end;

end.
