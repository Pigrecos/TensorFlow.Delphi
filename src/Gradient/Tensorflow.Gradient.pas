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
         Tensorflow.Graph,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         TensorFlow.Variable,
         TensorFlow.ControlFlowState,
         TensorFlow.Context ;

type

  BackwardFunction = Reference to function(grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>;

  TGradFunc = record
    Name : string;
    func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>;

    constructor Create(_name: string; _func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>);
  end;

  TapeTensor = record
    private
       Ftensor  : TFTensor;
       
       function GetShape: TFShape;
       function GetType: TF_DataType;

    public
       constructor Create(t: TFTensor);
       function GetID : Int64;
       function GetTensor : TFTensor;
       function ZerosLike: TFTensor;
       function OnesLike: TFTensor;
       function ToString: string;

       property tensor : TFTensor    read Ftensor;
       property Id     : Int64       read GetId;
       property dtype  : TF_DataType read GetType;
       property shape  : TFShape     read GetShape;
  end;

  ITape = class abstract
     private
     public
        F_persistent : Boolean;

        procedure SetTapeId(id: Integer); virtual; abstract;
        function  ShouldRecord(tensors: TArray<TFTensor>): Boolean;virtual; abstract;
        procedure StartRecord;virtual; abstract;
        procedure StopRecord;virtual; abstract;
        procedure RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);virtual; abstract;
        procedure VariableAccessed(variable: ResourceVariable);virtual; abstract;
        procedure Watch(x: TFTensor);virtual; abstract;
        function  WatchedVariables: TArray<ResourceVariable>;virtual; abstract;
        function  ComputeGradient(target_tensor_ids: TArray<TFTensor>; source_tensor_ids: TArray<TFTensor>;  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;virtual; abstract;

        property Persistente : Boolean read F_persistent;
  end;

  /// <summary>
  /// Represents an entry in the tape.
  /// </summary>
  OpTapeEntry = record
      op_type            : string;
      output_tensor_info : TArray<TapeTensor> ;
      input_tensor_id    : TArray<TFTensor>;
      backward_function  : BackwardFunction;
      ToString           : string;
  end;
  /// <summary>
  /// Map from operation-id to tape entry.
  /// </summary>
  OpTape = TDictionary<Int64, OpTapeEntry> ;
  /// <summary>
  /// Map from tensor to internally-defined operation-id of the operation which
  /// produced this tensor. A value of -1 means that the tensor was directly
  /// watched and not the result of any operation in the tape.
  /// </summary>
  TensorTape = TDictionary<TFTensor, Int64>;


   BackpropInitialState = class
      private

      public
        op_tape : OpTape;
        /// <summary>
        /// Map from tensor to how many references still exist for this tensor in
        /// the tape.
        /// </summary>
        tensor_usage_counts : TDictionary<TFTensor, Int64>;
        /// <summary>
        /// Maps from op ID to how many output tensors of this op still need to have
        /// their gradients computed.
        /// </summary>
        op_missing_tensor : TDictionary<Int64, Int64>;

        constructor Create;
        destructor Destroy; override;
   end;

  /// <summary>
  /// Gradient Tape Set
  /// Record operations for automatic differentiation.
  ///
  /// Operations are recorded if they are executed within this context manager and
  /// at least one of their inputs is being "watched".
  ///
  /// Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
  /// where `trainable=True` is default in both cases) are automatically watched.
  /// Tensors can be manually watched by invoking the `watch` method on this context
  /// manager.
  /// </summary>
  TGradientTape = class(TFDisposable)
    private
       FnextTapeId : Integer;
       FtapeSet    : TStack<ITape>;
       function GetTape: ITape;
    protected
      procedure NativeDispose(hnd: Pointer); override;
    public
      constructor Create;
      destructor  Destroy; override;
      /// <summary>
      /// New tape onto the tape stack.
      /// </summary>
      function PushTape(persistent: Boolean = false; watch_accessed_variables: Boolean = true): ITape;
      function PopTape: ITape;
      /// <summary>
      /// Marks this tensor to be watched by the given tape.
      /// </summary>
      /// <param name="x"></param>
      procedure watch(x: TFTensor);
      /// <summary>
      /// Computes the gradient using operations recorded in context of this tape.
      /// </summary>
      /// <param name="target"></param>
      /// <param name="source"></param>
      /// <returns></returns>
      function gradient(target: TFTensor; const source: TFTensor): TFTensor;overload;
      function gradient(target: TFTensor; const source: ResourceVariable): TFTensor;overload;
      function gradient(target: TFTensor; const sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor,TFTensor> overload;
      function gradient(target: TFTensor; const sources: TArray<IVariableV1>): TArray<TFTensor>;overload;
      /// <summary>
      /// Temporarily stops recording operations on this tape.
      /// </summary>
      function stop_recording: ITape;
      function GetTapeSet: TStack<ITape>;

      property Ftape: ITape read GetTape;
  end;

  TTape = class(ITape)
     private
        Fid              : Integer;
        // static int tape_nesting_id_counter = 0;
        F_recording      : Boolean;
        F_created_eagerly: Boolean;
        Ftensor_tape_    : TensorTape;
        Fop_tape_        : OpTape;
        Ftensor_usage_   : TDictionary<TFTensor, Int64>;
     public
        next_op_id_ : Integer;
        /// <summary>
        /// A deque-backed stack, whose element references are not invalidated by
        /// pushes and pops at the back.
        /// </summary>
        // Stack<AccumulatorCallState> call_state_;
        constructor Create(persistent: Boolean; watch_accessed_variables: Boolean);
        destructor Destroy;override;

        function  InitialStack(op_tape: OpTape; op_missing_tensor: TDictionary<Int64, Int64>): IQueue<Int64>;
        function  InitialGradients(target_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients : TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape): TDictionary<TFTensor, TList<TFTensor>>;
        function  FunctionsAcceptingNoneForIndicesMap : TDictionary<string, ISet<Integer>> ;
        function  PrepareBackprop(target: TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape; sources_set: ISet<TFTensor>; persistent_tape: Boolean) : BackpropInitialState;
        function  ComputeGradient(target_tensor_ids: TArray<TFTensor>; source_tensor_ids: TArray<TFTensor>;  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;override;
        procedure RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);override;
        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        procedure Watch(x: TFTensor);override;
        function  ShouldRecord(tensors: TArray<TFTensor>): Boolean; override;
        procedure VariableAccessed(variable: ResourceVariable); override;
        function  WatchedVariables: TArray<ResourceVariable>;  override;
        function  IsDtypeTrainable(dtype: TF_DataType): Boolean;
        procedure StartRecord; override;
        procedure StopRecord; override;
        procedure SetTapeId(id: Integer); override;
        function  ToString: string; reintroduce;

        property  Persistente : Boolean read F_persistent;

  end;

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
            TensorFlow.Constant_op,
            Tensorflow.NameScope,
            Tensorflow.math_ops,
            TensorFlow.gen_math_ops,
            Tensorflow.array_ops,
            Tensorflow.gen_array_ops,
            TensorFlow.control_flow_ops,
            TensorFlow.control_flow_util;
{ TapeTensor }

constructor TapeTensor.Create(t: TFTensor);
begin
    Ftensor := t;
end;

function TapeTensor.GetID: Int64;
begin
    Result := tensor.Id;
end;

function TapeTensor.GetShape: TFShape;
begin
    Result := Ftensor.Shape;
end;

function TapeTensor.GetTensor: TFTensor;
begin
    Result := Ftensor;
end;

function TapeTensor.GetType: TF_DataType;
begin
    Result := Ftensor.Dtype
end;

function TapeTensor.OnesLike: TFTensor;
begin
    Result := tf.ones(shape, dtype);
end;

function TapeTensor.ZerosLike: TFTensor;
begin
    Result := tf.zeros(shape, dtype);
end;

function TapeTensor.ToString: string;
begin
   Result := Format('%d, %s, %s',[Id, shape.ToString, Tdtypes.as_numpy_name(dtype)])
end;

{ GradientTape }

constructor TGradientTape.Create;
begin
    FtapeSet := TStack<ITape>.Create;
end;

destructor TGradientTape.Destroy;
begin
  if Assigned(Ftape) then
      Ftape.Free;
  FtapeSet.Clear;
  FtapeSet.Free;

  inherited;
end;

function TGradientTape.GetTape: ITape;
begin
    Result := nil;
    if Assigned(FtapeSet) and (FtapeSet.Count > 0) then
      Result :=  FtapeSet.Peek;
end;

function TGradientTape.GetTapeSet: TStack<ITape>;
begin
    Result :=  FtapeSet;
end;

function TGradientTape.gradient(target: TFTensor; const source: ResourceVariable): TFTensor;
begin
     var res := gradient(target, [ source ]);
     Result := res[0];
end;

function TGradientTape.gradient(target: TFTensor; const source: TFTensor): TFTensor;
begin
    var tape : ITape := stop_recording;

    var res := tf.Runner.TFE_TapeGradient(tape, [ target ],[ source ], nil);
    Result := res[0];
end;

function TGradientTape.gradient(target: TFTensor; const sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor, TFTensor>;
begin
    var res := gradient(target, [ sources.Value1, sources.Value2 ]);
    Result := Tuple<TFTensor, TFTensor>.Create(res[0], res[1]);
end;

function TGradientTape.gradient(target: TFTensor; const sources: TArray<IVariableV1>): TArray<TFTensor>;
begin
    var tape := stop_recording;

    var aSource: TArray<TFTensor> := [];
    for var i := 0 to Length(sources)-1  do
      aSource := aSource + [ sources[i].tHandle ];

    var res := tf.Runner.TFE_TapeGradient(tape,[ target ], aSource, nil);
    if not tape.Persistente then
    begin
        // Keep track of watched variables before setting tape to None
        // _watched_variables = _tape.WatchedVariables();
    end;
    Result := res;
end;

procedure TGradientTape.NativeDispose(hnd: Pointer);
begin
  inherited;
  FtapeSet.Clear;
  FtapeSet.Free;
end;

function TGradientTape.PopTape: ITape;
begin
     Ftape.StopRecord;
     Result := FtapeSet.Pop;
end;

function TGradientTape.PushTape(persistent, watch_accessed_variables: Boolean): ITape;
begin
    // Enters a context inside which operations are recorded on this tape.
    if tf.Context.executing_eagerly then
        tf.Context.ensure_initialized;
    var tape := TTape.Create(persistent, watch_accessed_variables);
    tape.SetTapeId(FnextTapeId);
    Inc(FnextTapeId);
    FtapeSet.Push(tape);
    Result := tape;
end;

function TGradientTape.stop_recording: ITape;
begin
    var tape := Ftape;
    if not tape.Persistente then
        tape := PopTape;
    Result := tape;
end;

procedure TGradientTape.watch(x: TFTensor);
begin
    if FtapeSet.count  < 1 then
        Exit;
    Ftape.Watch(x);
end;

{ TTape }

constructor TTape.Create(persistent, watch_accessed_variables: Boolean);
begin
    inherited Create;

    next_op_id_       := 0;
    F_persistent      := persistent;
    F_created_eagerly := tf.Context.executing_eagerly;
    Ftensor_tape_     := TDictionary<TFTensor, Int64>.Create;
    Fop_tape_         := TDictionary<Int64, OpTapeEntry>.Create;
    Ftensor_usage_    := TDictionary<TFTensor, Int64>.Create;
    if F_created_eagerly then
        tf.Context.start_step;

end;

destructor TTape.Destroy;
begin
    Ftensor_tape_.free;
    Fop_tape_.Free;
    Ftensor_usage_.free;

    inherited Destroy
end;

function TTape.IsDtypeTrainable(dtype: TF_DataType): Boolean;
begin
    case dtype of
      TF_DataType.TF_HALF,
      TF_DataType.TF_BFLOAT16,
      TF_DataType.TF_FLOAT,
      TF_DataType.TF_DOUBLE,
      TF_DataType.TF_COMPLEX64,
      TF_DataType.TF_COMPLEX128,
      TF_DataType.TF_RESOURCE,
      TF_DataType.TF_VARIANT: Result := True;
    else
      Result := False;
    end;
end;

function TTape.InitialGradients(target_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>;
  tensor_tape: TensorTape; op_tape: OpTape): TDictionary<TFTensor, TList<TFTensor>>;
begin
    var res := TDictionary<TFTensor, TList<TFTensor>>.Create;
    for var i : Integer := 0 to Length(target_tensor_ids) - 1 do
    begin
        var id := target_tensor_ids[i];
        if (Length(output_gradients) = 0) or (output_gradients[i] = nil) then
        begin
            if (tensor_tape.ContainsKey(id)) and (id <> nil) then
            begin
                if not op_tape.ContainsKey(tensor_tape[id]) then
                    raise TFException.Create('Iternal state of the gradient tape is invalid: failed to find operation producing a tensor');
                var op_it := op_tape[ tensor_tape[id] ];
                var found : Boolean := false;
                for var j := 0 to Length(op_it.output_tensor_info) - 1 do
                begin
                    if op_it.output_tensor_info[j].GetTensor = id then
                    begin
                        found    := true;
                        var ones := op_it.output_tensor_info[j].OnesLike;
                        if res.ContainsKey(id) then res[id].Add(ones)
                        else                        res.AddOrSetValue(id, TList<TFTensor>.Create([ones]));
                        break;
                    end;
                end;
                if not found then
                begin
                    raise TFException.Create('Internal state of the gradient tape is invalid: none of operations outputs match expected tensor');
                end;
            end else
            begin
                if sources_that_are_targets.ContainsKey(id) then
                begin
                    var source_tensor := sources_that_are_targets[id];
                    if res.ContainsKey(id) then res[id].Add(source_tensor.OnesLike)
                    else                        res.AddOrSetValue(id, TList<TFTensor>.Create([source_tensor.OnesLike]));
                end;
            end;
        end else
        begin
            if res.ContainsKey(id) then res[id].Add(output_gradients[i])
            else                        res.AddOrSetValue(id, TList<TFTensor>.Create([ output_gradients[i] ]));
        end;
    end;
    result := res;
end;

function TTape.InitialStack(op_tape: OpTape; op_missing_tensor: TDictionary<Int64, Int64>): IQueue<Int64>;
begin
    var res := TCollections.CreateQueue<Int64>;
    for var op_entry in op_tape do
    begin
        if not op_missing_tensor.ContainsKey(op_entry.Key) then
            res.Enqueue(op_entry.Key);
    end;
    Result := res;
end;

function TTape.FunctionsAcceptingNoneForIndicesMap: TDictionary<string, ISet<Integer>>;
begin
    var m := TDictionary<string, ISet<integer>>.Create;
    m.Add('SoftmaxCrossEntropyWithLogits',       TCollections.CreateSet<integer>([ 1 ]));
    m.Add('SparseSoftmaxCrossEntropyWithLogits', TCollections.CreateSet<integer>([ 1 ]));
    m.Add('FusedBatchNorm',                      TCollections.CreateSet<integer>([ 1, 2, 3, 4 ]));
    Result := m;
end;

function TTape.PrepareBackprop(target: TArray<TFTensor>; tensor_tape: TensorTape; op_tape: OpTape; sources_set: ISet<TFTensor>; persistent_tape: Boolean): BackpropInitialState;
begin
    var res : BackpropInitialState :=  BackpropInitialState.Create;
    var tensor_stack := TCollections.CreateQueue<TFTensor>(target);
    while tensor_stack.Count > 0 do
    begin
        var tensor_id := tensor_stack.Dequeue;
        if not tensor_tape.ContainsKey(tensor_id) then Continue;
        var op_id := tensor_tape[tensor_id] ;
        if (op_id = -1) or ( not op_tape.ContainsKey(op_id)) or (res.op_tape.ContainsKey(op_id) ) then Continue;
        var op_it        := op_tape[op_id] ;
        //var result_op_it := result.op_tape[op_id] ;
        res.op_tape.AddOrSetValue(op_id, op_it);
        for var it in op_it.input_tensor_id do
        begin
            if res.tensor_usage_counts.ContainsKey(it) then
               res.tensor_usage_counts[it] := res.tensor_usage_counts[it] + 1
            else begin
               res.tensor_usage_counts.AddOrSetValue(it, 1);
               if tensor_tape.ContainsKey(it) then
                    tensor_stack.Enqueue(it);
            end;
        end;
        if not persistent_tape then
            op_tape.Remove(op_id);
    end;
    for var pair in res.tensor_usage_counts do
    begin
        if (tensor_tape.ContainsKey(pair.Key)) and (tensor_tape[pair.Key] <> -1) then
        begin
           var it := tensor_tape[pair.Key];
           if res.op_missing_tensor.ContainsKey(it) then  res.op_missing_tensor[it] := res.op_missing_tensor[it] + 1
           else                                           res.op_missing_tensor.Add(it,1);
        end;
    end;
    if not persistent_tape then
    begin
        // Call destructors for all unneeded gradient functions and
        // clear the op_tape. We can clear the tape because ownership of
        // backward functions that will be used for gradient computation
        // has been transferred to `result`.
        (*for (const auto&op_pair : *op_tape) {
            op_pair.second.backward_function_deleter(
                op_pair.second.backward_function);
        } *)
        op_tape.Clear;
    end;
    Result := res;
end;


function TTape.ComputeGradient(target_tensor_ids, source_tensor_ids: TArray<TFTensor>; sources_that_are_targets: TDictionary<TFTensor, TapeTensor>;
  output_gradients: TArray<TFTensor>): TArray<TFTensor>;
var
  sources_set                     : ISet<TFTensor>;
  func_AcceptingNoneForIndicesMap : TDictionary<string, ISet<Integer>> ;
  state                           : BackpropInitialState;
  op_stack                        : IQueue<Int64>;
  gradients                       : TDictionary<TFTensor, TList<TFTensor>>;
  trace                           : OpTapeEntry;
  out_gradients                   : TList<TFTensor>;
  unneeded_gradients              : TList<Int64>;
  zero_indices                    : TList<Integer>;
  in_gradients                    : TArray<TFTensor>;
begin
    sources_set := TCollections.CreateSet<TFTensor>(source_tensor_ids);
    // var gradients_size = new UnorderedMap<Tensor, long>();
    func_AcceptingNoneForIndicesMap := FunctionsAcceptingNoneForIndicesMap;
    state    := PrepareBackprop(target_tensor_ids, Ftensor_tape_, Fop_tape_, sources_set, F_persistent);
    op_stack := InitialStack(state.op_tape, state.op_missing_tensor);
    gradients:= InitialGradients(target_tensor_ids, sources_that_are_targets, output_gradients, Ftensor_tape_, state.op_tape);
    while op_stack.Count > 0 do
    begin
        var op := op_stack.Dequeue;
        if not state.op_tape.ContainsKey(op) then
            continue;
        trace := state.op_tape[op];
        // Console.WriteLine($"ComputeGradient: {state.op_tape[op].op_type}");
        state.op_tape.Remove(op);
        out_gradients          := TList<TFTensor>.Create;
        out_gradients.Capacity := Length(trace.output_tensor_info);
        unneeded_gradients := TList<Int64>.Create;
        for var i := 0 to Length(trace.input_tensor_id)- 1 do
        begin
            var in_tensor_id := trace.input_tensor_id[i];
            if (not Ftensor_tape_.ContainsKey(in_tensor_id)) and (not sources_set.Contains(in_tensor_id)) then
                unneeded_gradients.Add(i);
        end;
        var any_gradient_nonzero : boolean := false;
        zero_indices := TList<Integer>.Create;
        for var i := 0 to Length(trace.output_tensor_info)-1 do
        begin
            var id := trace.output_tensor_info[i].GetTensor;
            if  not gradients.ContainsKey(id) then
            begin
                if (func_AcceptingNoneForIndicesMap.ContainsKey(trace.op_type)) and  (func_AcceptingNoneForIndicesMap[trace.op_type].Contains(i)) then
                begin
                    out_gradients.Add(nil);
                end else
                begin
                    out_gradients.Add(nil);
                    zero_indices.Add(i);
                end;
            end else
            begin
                any_gradient_nonzero := true;
                var grad_it := gradients[id];
                var new_gradients : TFTensor ;
                if grad_it.Count = 1 then new_gradients := grad_it[0]
                else                      new_gradients := gen_math_ops.add_n(grad_it.ToArray);  // vspace.AggregateGradients
                if not sources_set.Contains(id) then
                    gradients.Remove(id)
                else begin
                    // grad_it.Clear();
                    // grad_it.Add(new_gradients);
                    // vspace.MarkAsResult(new_gradients);
                end;
                out_gradients.Add(new_gradients);
            end;
        end;
        in_gradients := [];
        if any_gradient_nonzero then
        begin
            // foreach (var i in zero_indices)
            //     out_gradients[i] = trace.output_tensor_info[i].ZerosLike();
            in_gradients := trace.backward_function(out_gradients.ToArray, unneeded_gradients.ToArray);
            if (Length(in_gradients) <> Length(trace.input_tensor_id)) and ((Length(in_gradients) + unneeded_gradients.Count) <> Length(trace.input_tensor_id))then
                raise TFException.Create( Format('Recorded operation "%s" returned too few gradients. Expected %d but received %d',[trace.op_type, Length(trace.input_tensor_id), Length(in_gradients)]) );
            if not F_persistent then
            begin
                // trace.backward_function_deleter(trace.backward_function);
                trace.backward_function := nil;
            end;
        end else
        begin
            SetLength(in_gradients, Length(trace.input_tensor_id));
        end;

        var k : Integer := 0;
        var skip_unneeded_id : Boolean := Length(trace.input_tensor_id) > Length(in_gradients);
        for var i := 0 to Length(in_gradients) - 1 do
        begin
            if k >= Length(trace.input_tensor_id) then Break;

            if (skip_unneeded_id) and (unneeded_gradients.Contains(k)) then Inc(k);
            var id := trace.input_tensor_id[k];

            Inc(k);

            if in_gradients[i] <> nil then
            begin
                if not gradients.ContainsKey(id) then
                      gradients.Add(id,TList<TFTensor>.Create );

                var unaggregated_grads := gradients[id];
                unaggregated_grads.Add(in_gradients[i]);
                (*if (unaggregated_grads.Count > kMinAggregateCount)
                {
                    if (!gradients_size.find(id, out var size))
                    {
                        size = (long)unaggregated_grads[0].size;
                        gradients_size.emplace(id, size);
                    }
                    if (unaggregated_grads.Count * size * 4 > kMinAggregateBytes)
                    {
                        throw new NotImplementedException("");
                    }
                }*)
            end;
            if not state.tensor_usage_counts.ContainsKey(id) then
                continue;
            state.tensor_usage_counts[id] := state.tensor_usage_counts[id] - 1;
            if state.tensor_usage_counts[id] > 0 then
                continue;
            if not Ftensor_tape_.ContainsKey(id) then
            begin
                if gradients.ContainsKey(id) then
                begin
                    // foreach (var g in grad_it)
                    // DeleteGradient(g);
                    gradients.Remove(id);
                end;
                continue;
            end;
            var tape_it := Ftensor_tape_[id] ;
            var op_id   := tape_it;
            if op_id = -1 then
                continue;
            if state.op_missing_tensor.ContainsKey(op_id) then
            begin
                state.op_missing_tensor[op_id] := state.op_missing_tensor[op_id] - 1;
                if state.op_missing_tensor[op_id] = 0 then
                    op_stack.Enqueue(op_id);
            end;
        end;
    end;
    if state.op_tape.Count > 0 then
       raise Exception.Create('Invalid tape state.');
    var res : TArray<TFTensor>; SetLength(res, Length(source_tensor_ids) );
    var j : Integer := 0;
    for var id in source_tensor_ids do
    begin
        if gradients.ContainsKey(id) then
        begin
            var grad_it := gradients[id];
            if grad_it.Count > 1 then  res[j] := gen_math_ops.add_n(grad_it.ToArray)
            else                       res[j] := grad_it[0];
        end;
        Inc(j);
    end;
    Result := res;
end;

procedure TTape.RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);
begin
    if not ShouldRecord(input_tensors) then
        Exit;
    var op_id := next_op_id_;
    Inc(next_op_id_);
    for var i in input_tensors do
    begin
        if Ftensor_usage_.ContainsKey(i) then  Ftensor_usage_[i] := Ftensor_usage_[i] + 1
        else                                   Ftensor_usage_.AddOrSetValue(i,0);
    end;
    for var o in output_tensors do
    begin
        //tf.Logger.Debug($"RecordOperation: tensor_tape_[{o.GetID()}] = {op_id}");
        Ftensor_tape_.AddOrSetValue(o.GetTensor, op_id);
        Ftensor_usage_.AddOrSetValue(o.GetTensor, 1);
    end;
    var opT : OpTapeEntry;
    opT.op_type            := op_type;
    opT.output_tensor_info := output_tensors;
    opT.input_tensor_id    := input_tensors;
    opT.backward_function  := backward_function;
    Fop_tape_.AddOrSetValue(op_id,opT);
end;

procedure TTape.SetTapeId(id: Integer);
begin
    Fid := id;
end;

function TTape.ShouldRecord(tensors: TArray<TFTensor>): Boolean;
begin

    for var i := 0 to Length(tensors) - 1 do
    begin
        if Ftensor_tape_.Containskey(tensors[i]) then
        begin
            if IsDtypeTrainable(tensors[i].Dtype) then
                Exit( true );
        end;
    end;
    Result := false;
end;

procedure TTape.StartRecord;
begin
    if F_recording then
       raise TFException.Create('Tape is still recording, This can happen if you try to re-enter an already-active tape.');
    F_recording := true;
end;

procedure TTape.StopRecord;
begin
    if not F_recording then
       raise TFException.Create('Tape is not recording.');
    if F_created_eagerly then
        tf.Context.end_step;
    F_recording := false;
end;

function TTape.ToString: string;
begin
     if F_recording then Result := Format('Tape % Recording',[Fid])
     else                Result := Format('Tape % Stopped',[Fid])

end;

procedure TTape.VariableAccessed(variable: ResourceVariable);
begin
    Watch(variable.tHandle);
end;

procedure TTape.Watch(x: TFTensor);
begin
    //tf.Logger.Debug($"Watch tensor id={x.Id}, name={x.name}");
    Ftensor_tape_.AddOrSetValue(x, -1);
end;

function TTape.WatchedVariables: TArray<ResourceVariable>;
begin
    Result := nil;
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

{ TGradFunc }

constructor TGradFunc.Create(_name: string; _func : TFunc<TFOperation, TArray<TFTensor>, TArray<TFTensor>>);
begin
    Self.Name := _name;
    Self.func := _func;
end;


{ BackpropInitialState }

constructor BackpropInitialState.Create;
begin
   op_tape             := OpTape.Create;
   tensor_usage_counts := TDictionary<TFTensor, Int64>.Create;
   op_missing_tensor   := TDictionary<Int64, Int64>.Create;
end;

destructor BackpropInitialState.Destroy;
begin
   op_tape.Free;
   tensor_usage_counts.Free;
   op_missing_tensor.Free;
end;

end.

