unit TensorFlow.EagareRunner;
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
          System.Rtti,
          System.TypInfo,
          Winapi.Windows,

          System.Generics.Collections,

          Spring,
          Spring.Collections.Enumerable,

          TensorFlow.Context,
          TF4D.Core.CApi,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          TensorFlow.DApiEager,
          Tensorflow.Gradient,

          ProtoGen.OpDef;

type

TCallBack = Reference to procedure;

TFastPathOpExecInfo  = class
  public
    ctx                     : TContext;
    device_name             : string;
    op_name                 : string;
    name                    : string;
    args                    : TArray<TValue> ;
    attrs                   : TDictionary<string, TValue>;
    run_gradient_callback   : Boolean;
    run_post_exec_callbacks : Boolean;
    run_callbacks           : Boolean;
    callbacks               : TCallBack;

    constructor Create(opName:string; name: string; inputArgs: TArray<TValue>);
end;

TEagerRunner = class(TFDisposable)
  private
    function SetOpAttrList  (ctx: TContext; op: PTFE_Op; key: string; values: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
    function SetOpAttrScalar(ctx: TContext; op: PTFE_Op; key: string; value:  TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
    /// <summary>
    /// This function will set the op attrs required. If an attr has the value of
    /// None, then it will read the AttrDef to get the default value and set that
    /// instead. Any failure in this function will simply fall back to the slow
    /// path.
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="op"></param>
    /// <param name="attr"></param>
    /// <param name="attr_name"></param>
    /// <param name="attr_value"></param>
    /// <param name="attr_list_sizes"></param>
    /// <param name="status"></param>
    procedure SetOpAttrWithDefaults(ctx: TContext; op: PTFE_Op; attr: TAttrDef; attr_name: string; attr_value: TValue; attr_list_sizes: TDictionary<string, Int64>;status: TFStatus) ;
    /// <summary>
    /// Adds input and type attr to the op, and to the list of flattened
    /// inputs/attrs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="add_type_attr"></param>
    /// <param name="input_arg"></param>
    /// <param name="op"></param>
    /// <param name="status"></param>
    /// <returns></returns>
    function AddInputToOp(inputs: TValue; add_type_attr: Boolean; input_arg: TArgDef;flattened_attrs: TList<TValue>; flattened_inputs: TList<TFTensor>; op: PTFE_Op;status: TFStatus): Boolean;
    function HasAccumulator: Boolean;
    function HasAccumulatorOrTape: Boolean;
    function HasGradientTape: Boolean;
    function MakeTensorList(tensors: TArray<TFTensor>): TArray<TFTensor>;
    function ShouldRecord(inputs: TArray<TFTensor>): Boolean;

  protected
    procedure NativeDispose(hnd: Pointer); override;
  public
    thread_local_eager_operation_map : TDictionary<string, PTFE_Op>;

    constructor Create; overload;
    constructor Create(hHandle: Pointer); overload;
    destructor  Destroy; override;

    function  GetOp(ctx: TContext; op_or_function_name: AnsiString; status: TFStatus): PTFE_Op;
    procedure SetOpAttrs(op: PTFE_Op; attrs: TArray<TValue>);

    function CouldForwardprop : Boolean;
    function CouldBackprop : Boolean;
    function TapeSetRecordOperation(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TFTensor>; backward_function: BackwardFunction): Boolean;
    function TapeSetRecordForwardprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function_getter: BackwardFunction): Boolean;
    procedure TapeSetRecordBackprop(   op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);

    function GetGradientFunction(op_name: string; op_inputs: TArray<TFTensor>; attrs: TArray<TValue>; op_outputs: TArray<TFTensor>) : BackwardFunction;
    function RunCallbacks(op_exec_info: TFastPathOpExecInfo; num_inferred_attrs: Integer; inputs: TArray<TFTensor>;attrs: TArray<TValue>; flattened_result: TArray<TFTensor>): Boolean;
    function RecordGradient(op_name: string; inputs: TArray<TFTensor>; attrs: TArray<TValue>; results: TArray<TFTensor>; backward_Function: BackwardFunction = nil) : Boolean;
    function MustRecordGradient: Boolean;
    function TFE_TapeGradient(tape: ITape; target, sources, output_gradients: TArray<TFTensor>): TArray<TFTensor>;
    function TFE_Execute(ctx: TContext; device_name: AnsiString; op_name: AnsiString; inputs: TArray<TFTensor>;attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
    function TFE_ExecuteCancelable(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>; attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
    function TFE_FastPathExecute(op_exec_info: TFastPathOpExecInfo): TArray<TFTensor>;
    procedure ClearEagerOperationMap;
    /// <summary>
    /// Execute a TensorFlow operation.
    /// </summary>
    /// <param name="op_name">
    /// Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
    /// execute.
    /// </param>
    /// <param name="num_outputs">
    /// The number of outputs of the operation to fetch.
    /// </param>
    /// <param name="inputs">
    /// A list of inputs to the operation. Each entry should be a Tensor, or
    /// a value which can be passed to the Tensor constructor to create one.
    /// </param>
    /// <param name="attrs">
    /// A tuple with alternating string attr names and attr values for this
    /// operation.
    /// </param>
    /// <param name="ctx">The value of context.context().</param>
    /// <param name="name">Customized name for the operation.</param>
    /// <returns>List of output Tensor objects. The list is empty if there are no outputs</returns>
    function Execute(ctx: TContext; op_name: AnsiString; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>;name: AnsiString = '') : TArray<TFTensor>;
    function ArgsToMatchingEager(ctx: TContext; default_dtype: TF_DataType = TF_DataType.DtInvalid; args: TArray<TValue> = nil):  Tuple<TF_DataType, TArray<TFTensor>> ;
end;

implementation
      uses Tensorflow,
           TensorFlow.EagerTensor,
           TensorFlow.Functions,
           TensorFlow.Ops,
           Tensorflow.Utils,
           Oz.SGL.Collections;

{ TEagerRunner }

constructor TEagerRunner.Create;
begin
    inherited Create(Nil);
    thread_local_eager_operation_map := TDictionary<string, PTFE_Op>.Create;
end;

constructor TEagerRunner.Create(hHandle: Pointer);
begin
    inherited Create(hHandle);
    thread_local_eager_operation_map := TDictionary<string, PTFE_Op>.Create;
end;

destructor TEagerRunner.Destroy;
begin
    thread_local_eager_operation_map.Free;
    inherited Destroy;
end;

function TEagerRunner.Execute(ctx: TContext; op_name: AnsiString; num_outputs: Integer; inputs: TArray<TFTensor>; attrs: TArray<TValue>;
  name: AnsiString): TArray<TFTensor>;
begin
    ctx.ensure_initialized();

    Result := tf.Runner.TFE_Execute(ctx, ctx.DeviceName,op_name, inputs, attrs, num_outputs);
end;

procedure TEagerRunner.NativeDispose(hnd: Pointer);
begin
   if Assigned(hnd) then
     TFE_DeleteTensorHandle(hnd);
end;

function TEagerRunner.RunCallbacks(op_exec_info: TFastPathOpExecInfo; num_inferred_attrs: Integer; inputs: TArray<TFTensor>;
  attrs: TArray<TValue>; flattened_result: TArray<TFTensor>): Boolean;
begin
    if op_exec_info.run_gradient_callback then
    begin
        if not RecordGradient(op_exec_info.op_name, inputs, attrs, flattened_result) then
          Exit(false);
    end;
    if op_exec_info.run_post_exec_callbacks then
    begin

    end;
    Result := true;
end;

function TEagerRunner.RecordGradient(op_name: string; inputs: TArray<TFTensor>; attrs: TArray<TValue>; results: TArray<TFTensor>;
  backward_Function: BackwardFunction): Boolean;
begin
    var should_record := ShouldRecord(inputs);

    if  not should_record then
    begin
        (*for (TFE_Py_ForwardAccumulator* accumulator : SafeAccumulatorSet())
        {
            if (accumulator->accumulator->ShouldRecord(input_ids, input_dtypes))
            {
                should_record = true;
                break;
            }
        }*)
    end;
    if  not should_record  then Exit(should_record);
    // tf.Logger.Debug($"RecordGradient: op_name={op_name}");
    (*Tensor[] op_outputs = null;
    var unused_output_indices = gradient_exclustions.OpGradientUnusedOutputIndices(op_name);
    if (unused_output_indices != null)
    {
        if (unused_output_indices.Length == 0)
            op_outputs = new Tensor[0];
        else
        {
            // op_outputs = CopySequenceSettingIndicesToNull(results, *unused_output_indices);
        }
    }
    else
        op_outputs = results;
    Tensor[] op_inputs = null;
    var unused_input_indices = gradient_exclustions.OpGradientUnusedInputIndices(op_name);
    if (unused_input_indices != null)
    {
        if (unused_input_indices.Length == 0)
            op_inputs = new Tensor[0];
        else
        {
            // op_inputs = CopySequenceSettingIndicesToNull(inputs, *unused_input_indices);
        }
    }
    else
        op_inputs = inputs;*)

    if not Assigned(backward_Function) then
      backward_Function := GetGradientFunction(op_name, inputs, attrs, results);

    TapeSetRecordOperation(op_name, inputs, results, backward_Function);
    Result := true;
end;

function TEagerRunner.GetGradientFunction(op_name: string; op_inputs: TArray<TFTensor>; attrs: TArray<TValue>; op_outputs: TArray<TFTensor>): BackwardFunction;
begin
    Result := function(out_grads : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
               begin
                   if not gradientFunctions.ContainsKey(op_name) then
                   begin
                       SetLength( Result,Length(op_inputs) );
                   end;
                   var oper := EagerOperation.Create;
                   oper.Name            := op_name;
                   oper.NumInputs       := Length(op_inputs);
                   oper.Inputs          := op_inputs;
                   oper.NumOutputs      := Length(op_outputs);
                   oper.Outputs         := op_outputs;
                   oper.SkipInputIndices:= unneeded_gradients;
                   oper.Attrs           := attrs;

                   Result := gradientFunctions[op_name](oper, out_grads);
               end;
end;

function TEagerRunner.TapeSetRecordOperation(op_type: string; input_tensors, output_tensors: TArray<TFTensor>; backward_function: BackwardFunction): Boolean;
begin
    var output_info : TArray<TapeTensor> := [];
    for var i := 0 to Length(output_tensors)-1  do
    begin
        output_info :=  output_info + [ TapeTensor.Create(output_tensors[i]) ]
    end;

    if not TapeSetRecordForwardprop(op_type, input_tensors, output_info, backward_function) then
        Exit(false);
    TapeSetRecordBackprop(op_type, input_tensors, output_info, backward_function);
    Result := true;
end;

procedure TEagerRunner.TapeSetRecordBackprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function: BackwardFunction);
begin
    if not CouldBackprop then
       Exit;
    for var tape in tf.GetTapeSet do
    begin
        tape.RecordOperation(op_type, input_tensors, output_tensors, backward_function);
    end
end;

function TEagerRunner.TapeSetRecordForwardprop(op_type: string; input_tensors: TArray<TFTensor>; output_tensors: TArray<TapeTensor>; backward_function_getter: BackwardFunction): Boolean;
begin
    if not CouldForwardprop then
       Exit(true);
    raise TFException.Create('Not Implemented');
end;

function TEagerRunner.CouldForwardprop : Boolean;
begin
    Result := HasAccumulator;
end;
procedure TEagerRunner.ClearEagerOperationMap;
begin
   thread_local_eager_operation_map.Clear;
end;

function TEagerRunner.CouldBackprop : Boolean;
begin
     Result := HasGradientTape;
end;

procedure TEagerRunner.SetOpAttrs(op: PTFE_Op; attrs: TArray<TValue>);
begin
    var status := tf.Status;
    var len := Length(attrs);
    var i : Integer := 0;
    while i < len do
    begin
        var key   := attrs[i].AsString;
        var value := attrs[i + 1];
        var  is_list : Byte := 0;
        var tipo := TFE_OpGetAttrType(op, PAnsiChar(AnsiString(key)), @is_list, status.Handle);
        if not status.Ok then Exit;
        if is_list <> 0 then
            SetOpAttrList(tf.Context, op, key, value , tipo, nil, status)
        else
            SetOpAttrScalar(tf.Context, op, key, value, tipo, nil, status);
        status.RaiseEx;

       Inc(i,2);
    end;
end;
function TEagerRunner.SetOpAttrList(ctx: TContext; op: PTFE_Op; key: string; values: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, Int64>; status: TFStatus): Boolean;
var
   StrArrray : TArray<string>;
   pStrArray : TArray<AnsiString>;
   vlen      : TArray<UInt64>;
begin
    if (tipo = TF_AttrType.TF_ATTR_STRING) and (values.IsType< TArray<string> > ) then
    begin
        StrArrray := values.AsType< TArray<string> >;
        pStrArray := [];
        vlen     := [];
        for var i := 0 to Length(StrArrray) - 1 do
        begin
            pStrArray := pStrArray + [ AnsiString(StrArrray[i]) ];
            vlen      := vlen + [ Length( pStrArray[i] ) ];
        end;
        TFE_OpSetAttrStringList(op, PAnsiChar(AnsiString(key)), @pStrArray[0],@vlen[0], Length(pStrArray));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(pStrArray) );
    end
    else if (tipo = TF_AttrType.TF_ATTR_SHAPE) and (values.IsType< TArray<TFShape> >) then
    begin
        var values1 := values.AsType< TArray<TFShape> >;
        // Make one pass through the input counting the total number of
        // dims across all the input lists.
        var num_values := Length(values1);
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key,num_values);
        var dims : TArray<Pointer>;
        SetLength(dims,num_values);
        var num_dims : TArray<Integer>;
        for var i := 0 to Length(values1) - 1 do
            num_dims := num_dims + [ values1[i].ndim ] ;

        for var i := 0 to num_values - 1 do
        begin
            dims[i] := AllocMem(SizeOf(Int64) * values1[i].ndim);
            CopyMemory(dims[i], @values1[i].dims[0], values1[i].ndim * sizeof(Int64));
        end;
        TFE_OpSetAttrShapeList(op, PAnsiChar(AnsiString(key)), @dims[0], @num_dims[0], num_values, status.Handle);
        for var i := 0 to num_values - 1 do
           FreeMem(dims[i]);
    end
    else if (tipo = TF_AttrType.TF_ATTR_TYPE) and (values.IsType< TArray<Integer> >) then
    begin
        var values2 := values.AsType< TArray<Integer> >;
        TFE_OpSetAttrTypeList(op, PAnsiChar(AnsiString(key)), @values2[0], Length(values2));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(values2));
    end
    else if (tipo = TF_AttrType.TF_ATTR_INT) and (values.IsType< TArray<Integer> >) then
    begin
        var values4 := values.AsType< TArray<Integer> >;
        var vvalues4 : TArray<Int64>;
        for var i := 0 to Length(values4) - 1 do
            vvalues4 := vvalues4 + [ values4[i] ] ;
        TFE_OpSetAttrIntList(op, PAnsiChar(AnsiString(key)), @vvalues4[0], Length(values4));
        if attr_list_sizes <> nil then
          attr_list_sizes.Add(key, Length(values4));
    end else
    begin
        raise Exception.Create('Not Implemented.');
    end;
    Result :=true;
end;

function TEagerRunner.SetOpAttrScalar(ctx: TContext; op: PTFE_Op; key: string; value: TValue; Tipo: TF_AttrType; attr_list_sizes: TDictionary<string, INt64>; status: TFStatus): Boolean;
begin
    case tipo of
      TF_AttrType.TF_ATTR_STRING:
         begin
             var vValue := value.AsType< string >;
             TFE_OpSetAttrString(op, PAnsiChar(AnsiString(key)), PAnsiChar(AnsiString(vValue)), UInt64(Length(AnsiString( vValue ))));
         end;
      TF_AttrType.TF_ATTR_TYPE:
         begin
             var vValue := value.AsType< Integer >;
             TFE_OpSetAttrType(op, PAnsiChar(AnsiString(key)), TF_DataType(vValue));
         end;
      TF_AttrType.TF_ATTR_BOOL: TFE_OpSetAttrBool(op, PAnsiChar(AnsiString(key)), value.AsType< Boolean >);
      TF_AttrType.TF_ATTR_INT:
         begin
             var size := Int64(value.AsType<Integer>);
             TFE_OpSetAttrInt(op, PAnsiChar(AnsiString(key)), size);
             if attr_list_sizes <> nil then
                attr_list_sizes.Add(key,size);
         end;
      TF_AttrType.TF_ATTR_FLOAT:
         begin
             var size := value.AsType<Single>;
             TFE_OpSetAttrFloat(op, PAnsiChar(AnsiString(key)), size);
         end;
      TF_AttrType.TF_ATTR_SHAPE:
         begin
            var value1 := value.AsType<  TArray<Int64>>;
            var dims := value1;
            TFE_OpSetAttrShape(op, PAnsiChar(AnsiString(key)), @dims[0], Length(dims), status.Handle);
            status.RaiseEx;
         end;
      TF_AttrType.TF_ATTR_FUNC:
         begin
             if value.IsType<ConcreteFunction> then
             begin
                 var func := value.AsType<ConcreteFunction>;
                 TFE_OpSetAttrFunctionName(op, PAnsiChar(AnsiString(key)), PAnsiChar(AnsiString(func.Name)), func.Name.Length);
             end else
                 raise Exception.Create('Not Implemented, SetOpAttrScalar for'+ TEnum.GetName(tipo));

         end;
      else
         raise Exception.Create('Not Implemented, SetOpAttrScalar for'+ TEnum.GetName(tipo));
    end;
    Result := true;
end;

function TEagerRunner.GetOp(ctx: TContext; op_or_function_name: AnsiString; status: TFStatus): PTFE_Op;
var
  op : PTFE_Op;
begin
    if thread_local_eager_operation_map.ContainsKey( string(op_or_function_name)) then
    begin
        op := thread_local_eager_operation_map[ op_or_function_name ];
        TFE_OpReset(op, PAnsiChar(op_or_function_name), PAnsiChar(AnsiString(ctx.DeviceName)), status.Handle);
    end else
    begin
        op := TFE_NewOp(ctx.Handle_, PAnsiChar(op_or_function_name), status.Handle);
        thread_local_eager_operation_map.Add(op_or_function_name,op);
    end;
    status.RaiseEx;
    Result := op;
end;

function TEagerRunner.TFE_Execute(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>; attrs: TArray<TValue>;
  num_outputs: Integer): TArray<TFTensor>;
begin
    Result := TFE_ExecuteCancelable(ctx, device_name, op_name, inputs, attrs, num_outputs);
end;

function TEagerRunner.TFE_ExecuteCancelable(ctx: TContext; device_name, op_name: AnsiString; inputs: TArray<TFTensor>;
  attrs: TArray<TValue>; num_outputs: Integer): TArray<TFTensor>;
begin
    var status := tf.Status;

    var op := GetOp(ctx, op_name, status);
    TFE_OpSetDevice(op, PAnsiChar(device_name), status.Handle);
    if status.Ok then
    begin
        for var i := 0 to Length(inputs) - 1 do
        begin
            var tensor_handle : PTFE_TensorHandle := inputs[i].EagerTensorHandle ;
            if tensor_handle = nil then
              raise Exception.Create('Eager tensor handle has not been allocated.');
            TFE_OpAddInput(op, tensor_handle, status.Handle);
            status.RaiseEx;
        end;
    end;
    if (status.ok) and (Length(attrs) > 0 ) then
        SetOpAttrs(op, attrs);
    var outputs : TArray<PTFE_Op>;
    SetLength( outputs,num_outputs);
    if status.ok then
    begin
        TensorFlow.DApiEager.TFE_Execute(op, @outputs[0], num_outputs, status.Handle);
        status.RaiseEx;
    end;
    Result := [];
    for var i := 0 to num_outputs - 1 do
       Result := Result + [ TEagerTensor.Create( outputs[i]) ]
end;

procedure TEagerRunner.SetOpAttrWithDefaults(ctx: TContext; op: PTFE_Op; attr: TAttrDef; attr_name: string; attr_value: TValue; attr_list_sizes: TDictionary<string, Int64>;status: TFStatus) ;
begin
    var  is_list : Byte := 0;
    var tipo := TFE_OpGetAttrType(op, PAnsiChar(AnsiString(attr_name)), @is_list, status.Handle);
    if not status.Ok then Exit;

    if attr_value.Kind = tkUnknown then
    begin

    end else
    begin
        if is_list <> 0 then
            SetOpAttrList  (ctx, op, PAnsiChar(AnsiString(attr_name)), attr_value, tipo, attr_list_sizes, status)
        else
            SetOpAttrScalar(ctx, op, PAnsiChar(AnsiString(attr_name)), attr_value, tipo, attr_list_sizes, status);
    end;
end;

function TEagerRunner.ShouldRecord(inputs: TArray<TFTensor>): Boolean;
begin
    var should_record : Boolean := false;
    for var tape in tf.GetTapeSet do
    begin
        if tape.ShouldRecord(inputs) then
        begin
            should_record := true;
            break;
        end;
    end;
    Result := should_record;
end;

function  TEagerRunner.AddInputToOp(inputs: TValue; add_type_attr: Boolean; input_arg: TArgDef;flattened_attrs: TList<TValue>; flattened_inputs: TList<TFTensor>; op: PTFE_Op;status: TFStatus): Boolean;
begin
    var tensor := tf.convert_to_tensor(inputs);
    flattened_inputs.Add(tensor);
    if (add_type_attr) and ( not string.IsNullOrEmpty(input_arg.TypeAttr )) then
    begin
        var dtype := tensor.dtype;
        TFE_OpSetAttrType(op, PAnsiChar(AnsiString(input_arg.TypeAttr)), dtype);
        flattened_attrs.Add(input_arg.TypeAttr);
        flattened_attrs.Add( TValue.From<Integer>( Ord(dtype)) );
    end;
    TFE_OpAddInput(op, tensor.EagerTensorHandle, status.Handle);
    status.RaiseEx;
    Result := true;
end;

function TEagerRunner.TFE_FastPathExecute(op_exec_info: TFastPathOpExecInfo): TArray<TFTensor>;

   function FindAttr(a: TList<TAttrDef>; sName : AnsiString): TAttrDef;
   var
     i : Integer;
   begin
       Result := nil;
       for i := 0 to a.Count-1 do
       begin
           if AnsiString(a[i].Name) = sName then
             Exit( a[i] );
       end;
   end;

begin
    Result := [];
    if op_exec_info.ctx = nil then
        op_exec_info.ctx := tf.Context;

    if string.IsNullOrEmpty(op_exec_info.device_name) then
        op_exec_info.device_name := tf.Context.DeviceName;

    var attr_list_sizes := TDictionary<string, Int64>.Create;

    op_exec_info.run_gradient_callback   := HasAccumulatorOrTape;
    op_exec_info.run_post_exec_callbacks := op_exec_info.callbacks <> nil;
    op_exec_info.run_callbacks           := op_exec_info.run_gradient_callback or op_exec_info.run_post_exec_callbacks;

    var status := tf.Status;
    var op     := GetOp(op_exec_info.ctx, op_exec_info.op_name, status);
    var op_def := tf.get_default_graph.GetOpDef(op_exec_info.op_name);

    var flattened_attrs      := TList<TValue>.Create;
    flattened_attrs.Capacity := op_def.Attrs.Count * 2;

    var flattened_inputs      := TList<TFTensor>.Create;
    flattened_inputs.Capacity := op_def.InputArgs.Count;

    // Set non-inferred attrs, including setting defaults if the attr is passed in
    // as None.
    if op_exec_info.attrs <> nil then
    begin
        for var attr1 in op_exec_info.attrs do
        begin                     ;
            var attr := FindAttr(op_def.Attrs, attr1.Key);
            if attr <> nil then
            begin
                flattened_attrs.Add(attr.Name);
                flattened_attrs.Add(attr1.Value);
                SetOpAttrWithDefaults(op_exec_info.ctx, op, attr, attr.Name, attr1.Value, attr_list_sizes, status);
                status.RaiseEx;
            end
        end;
    end;
    // c_api.TFE_OpSetDevice(op, op_exec_info.device_name, status.Handle);
    // status.Check(true);
    // Add inferred attrs and inputs.
    for var i := 0 to op_def.InputArgs.Count - 1 do
    begin
        var input     := op_exec_info.args[i];
        var input_arg := op_def.InputArgs[i];
        if not string.IsNullOrEmpty(input_arg.NumberAttr) then
        begin
            var len : Int64 := input.GetArrayLength ;
            TFE_OpSetAttrInt(op, PAnsiChar(AnsiString( input_arg.NumberAttr )), len);
            if op_exec_info.run_callbacks then
            begin
                flattened_attrs.Add(input_arg.NumberAttr);
                flattened_attrs.Add(len);
            end;
            attr_list_sizes.Add(input_arg.NumberAttr,len);
            if len > 0 then
            begin
                var fast_input_array := op_exec_info.args[i];
                // First item adds the type attr.
                if not AddInputToOp(fast_input_array.GetArrayElement(i), true, input_arg, flattened_attrs, flattened_inputs, op, status) then
                    Exit;
                for var j := 1 to len-1 do
                begin
                    // Since the list is homogeneous, we don't need to re-add the attr.
                    if not AddInputToOp(fast_input_array.GetArrayElement(j), false, input_arg, flattened_attrs, flattened_inputs, op, status) then
                        Exit;
                end;
            end;
        end
        else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
        begin
            var attr_name        := input_arg.TypeListAttr;
            var fast_input_array := input;
            var len              := fast_input_array.GetArrayLength;
            var attr_values : TArray<TF_DataType>; SetLength(attr_values,len);
            for var j := 0 to len-1 do
            begin
                var eager_tensor := TOps.convert_to_tensor(fast_input_array.GetArrayElement(j));
                attr_values[j] := eager_tensor.dtype;
                TFE_OpAddInput(op, eager_tensor.EagerTensorHandle, status.Handle);
                if op_exec_info.run_callbacks then
                   flattened_inputs.Add(eager_tensor);
            end;
            if op_exec_info.run_callbacks then
            begin
                flattened_attrs.Add(attr_name);
                flattened_attrs.Add( TValue.From< TArray<Integer> >( Tdtypes.ToIntArray(attr_values) ));
            end;
            TFE_OpSetAttrTypeList(op, PAnsiChar(AnsiString(attr_name)), @attr_values[0], Length(attr_values));
            attr_list_sizes.Add(attr_name, len);
        end else
        begin
            // The item is a single item.
            AddInputToOp(op_exec_info.args[i], true, input_arg, flattened_attrs, flattened_inputs, op, status);
        end;
    end;
    var num_retvals : Integer := 0;
    for var i := 0 to op_def.OutputArgs.Count - 1 do
    begin
        var output_arg := op_def.OutputArgs[i];
        var delta : Int64 := 1;
        if not string.IsNullOrEmpty(output_arg.NumberAttr) then
            delta := attr_list_sizes[output_arg.NumberAttr]
        else if not string.IsNullOrEmpty(output_arg.TypeListAttr) then
            delta := attr_list_sizes[output_arg.TypeListAttr];
        if delta < 0  then
          raise Exception.Create('Attributes suggest that the size of an output list is less than 0');
        num_retvals := num_retvals + Integer(delta);
    end;

    var retVals : TArray<PTFE_Op> ;
    SetLength(retVals,num_retvals);
    TensorFlow.DApiEager.TFE_Execute(op, @retVals[0], num_retvals, status.Handle);
    status.RaiseEx;

    var flat_result : TArray<TFTensor> ;
    for var i := 0 to num_retvals - 1 do
       flat_result := flat_result + [ TEagerTensor.Create( retVals[i] ) ];
    if op_exec_info.run_callbacks then
    begin
        RunCallbacks(op_exec_info, op_def.InputArgs.Count,flattened_inputs.ToArray, flattened_attrs.ToArray, flat_result);
    end;
    Result := flat_result;
end;

function TEagerRunner.ArgsToMatchingEager(ctx: TContext; default_dtype: TF_DataType; args: TArray<TValue>):  Tuple<TF_DataType, TArray<TFTensor>>;
begin
    var res : Tuple<TF_DataType, TArray<TFTensor>>;

    var eArgs := Enumerable<TValue>.Create(args) ;
    var predicateCount :  TPredicate<TValue> := function(const x: TValue): Boolean
                             begin
                                 Result := x.IsType<TFTensor>;
                             end;

    if (Length(Args) = 0) and (default_dtype <> TF_DataType.DtInvalid) then
    begin
        Res := Tuple<TF_DataType, TArray<TFTensor>>.create(default_dtype, nil);
    end;

    if (eArgs.Count( predicateCount ) = Length(args)) then
    begin
        var sel := eArgs.Select<TFTensor>(function(x: TValue): TFTensor
                                 begin
                                     Result := x.AsType<TFTensor>;
                                 end).ToArray;

        res := Tuple<TF_DataType, TArray<TFTensor>>.create(sel[0].Dtype, sel);
        Result := res;
        exit;
    end;

    var dtype := TF_DataType.DtInvalid;
    for var x in args do
    begin
        if x.IsType<TFTensor> then
            dtype := x.AsType<TFTensor>.dtype;
    end;
    if dtype = TF_DataType.DtInvalid then
    begin
        var ret := TList<TFTensor>.Create;
        for var t in args do
        begin
            ret.Add(Tops.convert_to_tensor(t, dtype, '',False,default_dtype, ctx));
            if dtype = TF_DataType.DtInvalid then
                dtype := ret.Last.dtype;
        end;
        res := Tuple<TF_DataType, TArray<TFTensor>>.create(dtype, ret.ToArray) ;
        Result := res;
        exit;
    end else
    begin
        raise Exception.Create('Not Implemented');
    end;
end;

function TEagerRunner.HasAccumulator: Boolean;
begin
    Result := false;
end;

function TEagerRunner.HasGradientTape: Boolean;
begin
   Result := tf.GetTapeSet.Count > 0;
end;

function TEagerRunner.HasAccumulatorOrTape: Boolean;
begin
    Result := HasGradientTape or HasAccumulator;
end;

function TEagerRunner.MakeTensorList(tensors: TArray<TFTensor>): TArray<TFTensor>;
Begin
    Result := tensors;
end;

function TEagerRunner.MustRecordGradient: Boolean;
begin
    Result := HasGradientTape;
end;

function TEagerRunner.TFE_TapeGradient(tape: ITape; target: TArray<TFTensor>; sources: TArray<TFTensor>; output_gradients: TArray<TFTensor>): TArray<TFTensor>;
begin
    var target_vec  := target;
    var sources_vec := sources;
    var sources_set := sources_vec;
    var seq_array := target;
    var source_tensors_that_are_targets := TDictionary<TFTensor, TapeTensor>.Create;
    for var i := 0 to Length(target) - 1 do
    begin
        source_tensors_that_are_targets.Add(target_vec[i], TapeTensor.Create( seq_array[i]) );
    end;
    if output_gradients <> nil then
        raise Exception.Create('Not Implemented')
    else
        output_gradients :=  [];
    var outgrad_vec := MakeTensorList(output_gradients);
    Result := tape.ComputeGradient(target_vec, sources_vec, source_tensors_that_are_targets, outgrad_vec);
end;
{ TFastPathOpExecInfo }

constructor TFastPathOpExecInfo.Create(opName, name: string; inputArgs: TArray<TValue>);
begin
    op_name := opName;
    name    := name;
    args    := inputArgs;
end;

end.
