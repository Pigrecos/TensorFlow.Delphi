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
         Spring,
         Spring.Collections.Lists,
         Spring.Collections.Stacks,
         Spring.Collections.Dictionaries,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         TensorFlow.Variable,

         TensorFlow.Context ;

type

   BackwardFunction = Reference to function(grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>;

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

  ITape = class
     private
     public
        F_persistent : Boolean;

        procedure SetTapeId(id: Integer); virtual; abstract;
        function  ShouldRecord(tensors: TArray<TFTensor>): Boolean;virtual; abstract;
        procedure StartRecord;virtual; abstract;
        procedure StopRecord;virtual; abstract;
        procedure RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);virtual;
        procedure VariableAccessed(variable: ResourceVariable);virtual; abstract;
        procedure Watch(x: TFTensor);virtual; abstract;
        function  WatchedVariables: TArray<ResourceVariable>;virtual; abstract;
        function  ComputeGradient(target_tensor_ids: TArray<TFTensor>; source_tensor_ids: TArray<TFTensor>;  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;virtual;

        property Persistente : Boolean read F_persistent;
  end;

  /// <summary>
  /// Represents an entry in the tape.
  /// </summary>
  OpTapeEntry = record
      op_type : string;
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
       Ftape       : ITape;
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
      function gradient(target: TFTensor; source: TFTensor): TFTensor;overload;
      function gradient(target: TFTensor; source: ResourceVariable): TFTensor;overload;
      function gradient(target: TFTensor; sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor,TFTensor> overload;
      function gradient(target: TFTensor; sources: TArray<IVariableV1>): TArray<TFTensor>;overload;
      /// <summary>
      /// Temporarily stops recording operations on this tape.
      /// </summary>
      function stop_recording: ITape;
      function GetTapeSet: TStack<ITape>;

      property tape: ITape read GetTape;
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
        /// <summary>
        /// A deque-backed stack, whose element references are not invalidated by
        /// pushes and pops at the back.
        /// </summary>
        // Stack<AccumulatorCallState> call_state_;
        constructor Create(persistent: Boolean; watch_accessed_variables: Boolean);
        destructor Destroy;override;
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

implementation
       uses Tensorflow,
            Tensorflow.Utils;
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
  FtapeSet.Clear;
  FtapeSet.Free;
  Ftape.Free;
  inherited;
end;

function TGradientTape.GetTape: ITape;
begin
    Result :=  FtapeSet.Peek;
end;

function TGradientTape.GetTapeSet: TStack<ITape>;
begin
    Result :=  FtapeSet;
end;

function TGradientTape.gradient(target: TFTensor; source: ResourceVariable): TFTensor;
begin
     var res := gradient(target, [ source ]);
     Result := res[0];
end;

function TGradientTape.gradient(target, source: TFTensor): TFTensor;
begin
    var tape : ITape := stop_recording;

    var res := tf.Runner.TFE_TapeGradient(tape, [ target ],[ source ], nil);
    Result := res[0];
end;

function TGradientTape.gradient(target: TFTensor; sources: Tuple<ResourceVariable, ResourceVariable>): Tuple<TFTensor, TFTensor>;
begin
    var res := gradient(target, [ sources.Value1, sources.Value2 ]);

    Result := Tuple<TFTensor, TFTensor>.Create(res[0], res[1]);
end;

function TGradientTape.gradient(target: TFTensor; sources: TArray<IVariableV1>): TArray<TFTensor>;
begin
    var tape := stop_recording;

    var aSource: TArray<TFTensor> := [];
    for var i := 0 to Length(sources)-1  do
      aSource := aSource + [ sources[i].Handle ];
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
    if not FtapeSet.Any then
        Exit;
    Ftape.Watch(x);
end;

{ TTape }

constructor TTape.Create(persistent, watch_accessed_variables: Boolean);
begin
    inherited Create;

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
    Watch(variable.Handle);
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

{ ITape }

function ITape.ComputeGradient(target_tensor_ids, source_tensor_ids: TArray<TFTensor>;
  sources_that_are_targets: TDictionary<TFTensor, TapeTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;
begin

end;

procedure ITape.RecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>;
  backward_function: BackwardFunction);
begin

end;

{ gradients_impl }

class function gradients_impl.gradients(ys, xs, grad_ys: TArray<TFTensor>; name: string; colocate_gradients_with_ops, gate_gradients: Boolean;
  aggregation_method: PInteger): TArray<TFTensor>;
begin

end;

end.
