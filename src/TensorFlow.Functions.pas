unit TensorFlow.Functions;
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
     uses System.SysUtils, Winapi.Windows,
          System.Generics.Collections,

          Spring,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.DApiBase,
          Tensorflow.Graph,
          TensorFlow.Framework,
          Tensorflow.Gradient;
type
    ConcreteFunction     = class;
    EagerDefinedFunction = class ;

   /// <summary>
   /// Caches forward and backward functions compatible with eager gradients.
   /// </summary>
   TapeGradientFunctions = class
     private

     protected
       Ffunc_graph                 : TFuncGraph;
       Fforward                    : EagerDefinedFunction;
       Fforward_graph              : TFuncGraph;
       Fforwardprop_output_indices : TList<Integer>;
       Fnum_forwardprop_outputs    : Integer;
       Fbackward                   : ConcreteFunction;
       Fbackward_function_wrapper  : BackwardFunction;

       function BuildFunctionsForOutputs(outputs: TFTensors; inference_args: TFTensors): Tuple<EagerDefinedFunction,TFuncGraph,ConcreteFunction> ;
     public
      const
        FORWARD_FUNCTION_ATTRIBUTE_NAME  : string = 'forward_function_name';
        BACKWARD_FUNCTION_ATTRIBUTE_NAME : string = 'backward_function_name';
        _FORWARD_PREFIX                  : string = '__forward_';
        _BACKWARD_PREFIX                 : string = '__backward_';
        _INFERENCE_PREFIX                : string = '__inference_';
     public

       constructor Create(f_func_graph: TFuncGraph; need_gradients_for_jvps : Boolean);
       function    &Forward(inference_args: TFTensors): EagerDefinedFunction;
       /// <summary>
       /// Record the function call operation.
       /// </summary>
       /// <param name="flat_outputs"></param>
       /// <param name="inference_args"></param>
       procedure &Record(flat_outputs: TFTensors; inference_args: TFTensors);
       /// <summary>
       /// Create a backward function given `outputs` from the forward function.
       /// </summary>
       /// <param name="forward_graph"></param>
       /// <param name="backward"></param>
       /// <param name="outputs"></param>
       /// <returns></returns>
       function _wrap_backward_function(forward_graph: TFuncGraph; backward: ConcreteFunction; outputs: TFTensors): Tuple<BackwardFunction, TFTensors>;

       function ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction; virtual;
   end;

   FirstOrderTapeGradientFunctions = class(TapeGradientFunctions)
     public

       constructor Create(func_graph: TFuncGraph; need_gradients_for_jvps: Boolean);
       function    ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction; override;
   end;

   EagerDefinedFunction = class
     private

       function Get_Name: string;
     protected
       func_graph      : TFuncGraph;
       fnum_outputs    : Integer;

     public
       constructor Create(name: string; graph: TFuncGraph; inputs: TFTensors; outputs: TFTensors; attrs: TDictionary<string, string> );
       function Call(args: TFTensors): TFTensors;

       property Name : string  read Get_Name;
   end;

   /// <summary>
   /// Holds the state of a function call between execution and recording.
   /// </summary>
   ForwardBackwardCall = class
     private
       Ffunctions       : TapeGradientFunctions;
       Finference_args  : TFTensors;
       Finput_tangents  : TFTensors;
       Ftape_watching   : Boolean;
       Fforward_function: EagerDefinedFunction ;
     public

       constructor Create(functions: TapeGradientFunctions; inference_args: TFTensors; tape_watching: Boolean);
       function  &Forward:  Tuple<EagerDefinedFunction, TFTensors>;
       procedure &Record(flat_outputs: TFTensors);
   end;

   ConcreteFunction  = class
     private
        function Get_CaptInput: TArray<TFTensor>;
        function Get_Inputs: TArray<TFTensor>;
        function Get_Name: string;
     protected
        func_graph      : TFuncGraph;
        forward_backward: ForwardBackwardCall;
        Outputs         : TArray<TFTensor>;
        ReturnType      : PTypeInfo;
     public
        OutputStructure : TArray<TensorSpec>;
        ArgKeywords     : TArray<String>;
        NumPositionArgs : Int64;

        constructor Create(_name: string); overload;
        constructor Create(graph: TFuncGraph; attrs: TDictionary<string, string> = nil); overload;
        constructor Create(func: TFunc<TFTensor, TFTensor>; dtype: TF_DataType; func_name: string = ''); overload;
        procedure   ToGraph(inputs: TFTensors; outputs: TFTensors);
        procedure   AddTograph(g : TFGraph = nil);
        function    FilteredCall(inputs: TFTensors): TFTensors;
        function    SelectForwardAndBackwardFunctions(args: TFTensors; possible_gradient_type: Integer; executing_eagerly: Boolean) : ForwardBackwardCall;
        /// <summary>
        /// Executes the wrapped function.
        /// </summary>
        /// <param name="args"></param>
        /// <param name="captured_inputs"></param>
        /// <returns></returns>
        function CallFlat(args: TArray<TFTensor>; captured_inputs: TArray<TFTensor>) : TFTensors;
        function ToString: string; override;
        procedure   Enter;
        procedure   _Exit;

        property Inputs         : TArray<TFTensor> read Get_Inputs;
        property CapturedInputs : TArray<TFTensor> read Get_CaptInput;
        property Name           : string           read Get_Name;

  end;

  TFTensor_helper  = class Helper for TFTensor
    public
       function ToTensorSpec: TensorSpec;
  end;

implementation
          uses Tensorflow,
               Tensorflow.EagerTensor,
               TensorFlow.Ops;

{ TFTensor_helper }

function TFTensor_helper.ToTensorSpec: TensorSpec;
begin
    Result := TensorSpec.Create(shape, dtype, name)
end;

{ ConcreteFunction }

constructor ConcreteFunction.Create(func: TFunc<TFTensor, TFTensor>; dtype: TF_DataType; func_name: string);
var
  opers : TArray<TFOperation>;
begin
    func_name := func_name+'_'+ Tops.uid_function.ToString;

    func_graph := TFuncGraph.Create(func_name);

    func_graph.as_default;
    var input  := tf.placeholder(dtype);
    var output := func(input);

    for var it in func_graph.nodes_by_name.Values  do
        opers :=  opers + [ it as TFOperation] ;

    func_graph.ToGraph(opers, [ input ], [ output ], nil);
    func_graph.gExit;
end;

constructor ConcreteFunction.Create(graph: TFuncGraph; attrs: TDictionary<string, string>);
var
 tensorArray : TArray<TFTensor>;
begin
    func_graph := graph;

    for var i := 0 to graph.Outputs.Count -1  do
    begin
       if graph.Outputs[i] <> nil then
         tensorArray := tensorArray + [ graph.Outputs[i] ];
    end;
    ToGraph(graph.Inputs, TFTensors.Create(tensorArray));
end;

constructor ConcreteFunction.Create(_name: string);
begin
    func_graph := TFuncGraph.Create(_name);
end;

procedure ConcreteFunction.ToGraph(inputs, outputs: TFTensors);
var
  opers : TArray<TFOperation>;
begin
    for var it in func_graph.nodes_by_name.Values  do
        opers :=  opers + [ it as TFOperation] ;

    func_graph.ToGraph(opers, inputs.ToArray, outputs.ToArray, nil);

    OutputStructure := [];
    for var i := 0 to  outputs.Count -1  do
        OutputStructure :=  OutputStructure + [ outputs[i].ToTensorSpec ] ;
end;

procedure ConcreteFunction.AddTograph(g: TFGraph);
begin
    if ( not tf.Context.executing_eagerly) and ( g = nil) then
    begin
        g := Tops.get_default_graph;
    end;
end;

function ConcreteFunction.CallFlat(args, captured_inputs: TArray<TFTensor>): TFTensors;
begin
    var executing_eagerly := tf.Context.executing_eagerly;
    var tensor_inputs     := TFTensors.Create;
    for var i := 0 to Length(args) - 1 do
    begin
        var arg := args[i];
        tensor_inputs.Add(arg);
        // If we're graph building, shape inference is on.
        if not executing_eagerly then
        begin
        end;
    end;
    tensor_inputs.AddRange(captured_inputs);

    args := tensor_inputs.ToArray;
    var possible_gradient_type : Integer := 0;
    if tf.Runner.MustRecordGradient then
      possible_gradient_type := 1;

    // No tape is watching; skip to running the function.
    if (possible_gradient_type = 0) and (executing_eagerly) then
    begin
        var attrs : TArray<TValue> := ['executor_type', '', 'config_proto', tf.Context.FunctionCallOptions.config_proto_serialized];
        var Res := tf.Runner.Execute(tf.Context, func_graph.FuncName, func_graph.Outputs.count, args, attrs);
        Result := TFTensors.Create(Res) ;
        Exit;
    end;
    if forward_backward = nil then
        forward_backward := SelectForwardAndBackwardFunctions(TFTensors.Create(args), possible_gradient_type, executing_eagerly);
    var tFunc := forward_backward.&Forward;
    var forward_function   := tFunc.Value1;
    var args_with_tangents := tFunc.Value2;

    var flat_outputs : TFTensors := nil;
    if executing_eagerly then
        flat_outputs := forward_function.Call(args_with_tangents);

    forward_backward.&Record(flat_outputs);
    Result := flat_outputs;
end;

procedure ConcreteFunction.Enter;
begin
    func_graph.as_default;
end;

procedure ConcreteFunction._Exit;
begin
   func_graph.gExit;
end;

function ConcreteFunction.FilteredCall(inputs: TFTensors): TFTensors;
begin
    Result := CallFlat(inputs.ToArray, CapturedInputs);
end;

function ConcreteFunction.Get_CaptInput: TArray<TFTensor>;
begin
    Result := func_graph.external_captures
end;

function ConcreteFunction.Get_Inputs: TArray<TFTensor>;
begin
    Result := func_graph.Inputs.ToArray;
end;

function ConcreteFunction.Get_Name: string;
begin
    Result := func_graph.FuncName
end;

function ConcreteFunction.SelectForwardAndBackwardFunctions(args: TFTensors; possible_gradient_type: Integer; executing_eagerly: Boolean): ForwardBackwardCall;
begin
    var functions := FirstOrderTapeGradientFunctions.Create(func_graph, false);
    Result        := ForwardBackwardCall.Create(functions, args, true);
end;

function ConcreteFunction.ToString: string;
begin
    Result := Name
end;

{ ForwardBackwardCall }

constructor ForwardBackwardCall.Create(functions: TapeGradientFunctions; inference_args: TFTensors; tape_watching: Boolean);
begin
    Ffunctions      := functions;
    Finference_args := inference_args;
    Ftape_watching  := tape_watching;
end;

function ForwardBackwardCall.Forward: Tuple<EagerDefinedFunction, TFTensors>;
begin
    if Fforward_function = nil then
      Fforward_function := Ffunctions.Forward(Finference_args);
    Result := Tuple.Create(Fforward_function, Finference_args);
end;

procedure ForwardBackwardCall.&Record(flat_outputs: TFTensors);
begin
   if (Ftape_watching) and (flat_outputs <> nil) then
     Ffunctions.&Record(flat_outputs, Finference_args);
end;

{ EagerDefinedFunction }

constructor EagerDefinedFunction.Create(name: string; graph: TFuncGraph; inputs, outputs: TFTensors; attrs: TDictionary<string, string>);
var
  operations  : TArray<TFOperation>;
  input_ops   : TArray<TFOperation>;
  output_names: TArray<string>;
  i           : integer;
begin
    fnum_outputs := outputs.Count;
    for i:= 0 to inputs.Count - 1 do
      input_ops := input_ops + [ inputs[i].op ];

    var IOperation := graph.get_operations;
    for i := 0 to Length(IOperation) - 1 do
    begin
        var operation := IOperation[i];
        if not Tarray.Contains<TFOperation>(input_ops, operation.op) then
          operations := operations + [ operation as TFOperation ];
    end;
    func_graph := TFuncGraph.Create(graph.Handle, name, attrs);
    func_graph.ToGraph(operations, inputs.ToArray, outputs.ToArray, output_names);
end;

function EagerDefinedFunction.Call(args: TFTensors): TFTensors;
begin
    var attrs : TArray<TValue> := ['executor_type', '', 'config_proto', tf.Context.FunctionCallOptions.config_proto_serialized];

    var results := tf.Runner.TFE_Execute(tf.Context, tf.Context.DeviceName, func_graph.FuncName, args.ToArray, attrs, Fnum_outputs);

    Result := TFTensors.Create(results);
end;

function EagerDefinedFunction.Get_Name: string;
begin
    Result := func_graph.FuncName
end;

{ TapeGradientFunctions }

constructor TapeGradientFunctions.Create(f_func_graph: TFuncGraph; need_gradients_for_jvps : Boolean);
begin
    Ffunc_graph := f_func_graph;
end;

function TapeGradientFunctions.BuildFunctionsForOutputs(outputs, inference_args: TFTensors): Tuple<EagerDefinedFunction, TFuncGraph, ConcreteFunction>;
begin
    var trainable_outputs := TList<TFTensor>.Create;
    var trainable_indices := TList<Integer>.Create;
    for var i := 0 to outputs.Count -1 do
    begin
        var output := outputs[i];
        if gradients_util.IsTrainable(output) then
        begin
            trainable_outputs.Add(output);
            trainable_indices.Add(i);
        end;
    end;

    var gradients_wrt_outputs := TList<TFTensor>.Create;
    var backwards_graph       := TFuncGraph.Create(_BACKWARD_PREFIX+'_'+Ffunc_graph.FuncName+'_'+ Tops.uid.ToString);
    backwards_graph.as_default;
    for var output in trainable_outputs do
        gradients_wrt_outputs.Add(tf.placeholder(output.dtype, output.shape));

    var gradients_wrt_inputs := gradients_util._GradientsHelper(trainable_outputs.ToArray, Ffunc_graph.Inputs.ToArray, gradients_wrt_outputs.ToArray,'gradients', False, False, 0, nil, Ffunc_graph);

    var captures_from_forward : TArray<TFTensor> := [];
    for var i := 0 to Length(backwards_graph.external_captures) - 1 do
    begin
         var it := backwards_graph.external_captures[i];
         if ((it is TEagerTensor) = False) and ((it is TNDArray) = False) and (it.graph = Ffunc_graph) then
            captures_from_forward := captures_from_forward + [ it ];
    end;
    for var capture in captures_from_forward do
    begin
        if not Ffunc_graph.Outputs.Contains(capture) then
            Ffunc_graph.Outputs.Add(capture);
    end;
    backwards_graph.gExit;

    var forward_function_name  := _FORWARD_PREFIX +'_'+Ffunc_graph.FuncName+'_'+ Tops.uid.ToString;
    var backward_function_attr := TDictionary<string, string>.Create;

    backward_function_attr.AddOrSetValue(FORWARD_FUNCTION_ATTRIBUTE_NAME, forward_function_name);
    gradients_wrt_outputs.AddRange(backwards_graph.internal_captures);
    backwards_graph.Inputs  := TFTensors.Create(gradients_wrt_outputs.ToArray);
    backwards_graph.Outputs := TFTensors.Create(gradients_wrt_inputs);

    var backward_function     := ConcreteFunction.Create(backwards_graph, backward_function_attr);
    var forward_function_attr := TDictionary<string, string>.Create;
    forward_function_attr.AddOrSetValue(BACKWARD_FUNCTION_ATTRIBUTE_NAME, backward_function.Name);

    var forward_function := EagerDefinedFunction.Create(forward_function_name, Ffunc_graph, Ffunc_graph.Inputs, Ffunc_graph.Outputs, forward_function_attr);

    Result := Tuple.Create(forward_function, Ffunc_graph, backward_function);
end;

function TapeGradientFunctions.Forward(inference_args: TFTensors): EagerDefinedFunction;
begin
    Result := ForwardAndBackwardFunctions(inference_args);
end;

function TapeGradientFunctions.ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction;
begin
    raise Exception.Create('ForwardAndBackwardFunctions .  Virtual method');
end;

procedure TapeGradientFunctions.&Record(flat_outputs, inference_args: TFTensors);
begin
   var tWrapFun          := _wrap_backward_function(Fforward_graph, Fbackward, flat_outputs);
   var backward_function := tWrapFun.Value1;
   var to_record         := tWrapFun.Value2;

   tf.Runner.RecordGradient(Fforward.Name, inference_args.ToArray, [], to_record.ToArray, backward_function);
end;

function TapeGradientFunctions._wrap_backward_function(forward_graph: TFuncGraph; backward: ConcreteFunction; outputs: TFTensors): Tuple<BackwardFunction, TFTensors>;
begin
    var backward_function_inputs := Length(backward.Inputs) - Length(backward.CapturedInputs);
    var recorded_outputs := TFTensors.Create;
    var trainable_recorded_outputs := 0;
    for  var i := 0 to outputs.Count - 1 do
    begin
        var output       :=  outputs[i];
        if trainable_recorded_outputs < backward_function_inputs then
            recorded_outputs.Add(output);
        if gradients_util.IsTrainable(output) then
            trainable_recorded_outputs := trainable_recorded_outputs + 1;
    end;

    if not Assigned(Fbackward_function_wrapper) then
    begin
        var capture_mapping := TDictionary<Int64, TFTensor>.Create;
        for var i := 0 to outputs.Count -1 do
            capture_mapping.AddOrSetValue(forward_graph.Outputs[i].Id, outputs[i]);

        var remapped_captures := TFTensors.Create;
        for var capture in backward.CapturedInputs do
        begin
            if capture_mapping.ContainsKey(capture.Id) then
                remapped_captures.Add( capture_mapping[capture.Id] );
        end;

        var skip_positions := TList<Integer>.Create;
        for  var i := 0 to outputs.Count - 1 do
        begin
            var output_index :=  i;
            var output       :=  outputs[i];
            if not gradients_util.IsTrainable(output) then
               skip_positions.Add(output_index);
        end;

        Fbackward_function_wrapper := function (args : TArray<TFTensor>; unneeded_gradients: TArray<Int64>): TArray<TFTensor>
                begin
                    var processed_args := TFTensors.Create;
                    var input_index    := 0;
                    for var output_index := 0 to Length(args) - 1 do
                    begin
                        var arg := args[output_index];
                        if skip_positions.Contains(output_index) then
                            continue;
                        if arg = nil then
                            raise Exception.Create('Not Implemented');
                        processed_args.Add(arg);
                        input_index := input_index + 1;
                        if input_index >= backward_function_inputs then
                            break;
                    end;

                    tf.LogMsg('Invoke backward function: '+backward.Name);
                    var gradients := backward.CallFlat(processed_args.ToArray, remapped_captures.ToArray);

                    for var unneeded_gradient_index in unneeded_gradients do
                    begin
                        var index : Integer := unneeded_gradient_index;
                        if gradients.Count <= index then
                            gradients.Insert(index, nil);
                    end;

                    Result := gradients.ToArray;
                end;
    end;

    Result := Tuple<BackwardFunction, TFTensors>.Create(Fbackward_function_wrapper, recorded_outputs);
end;

{ FirstOrderTapeGradientFunctions }

constructor FirstOrderTapeGradientFunctions.Create(func_graph: TFuncGraph; need_gradients_for_jvps: Boolean);
begin
    inherited Create(func_graph, need_gradients_for_jvps);
end;

function FirstOrderTapeGradientFunctions.ForwardAndBackwardFunctions(inference_args: TFTensors): EagerDefinedFunction;
begin
    var outputs := Ffunc_graph.Outputs;
    var tRes     := BuildFunctionsForOutputs(outputs, inference_args);

    Fforward       := tRes.Value1;
    Fforward_graph := tRes.Value2;
    Fbackward      := tRes.Value3;
    Fforwardprop_output_indices := nil;
    Fnum_forwardprop_outputs    := 0;

    Result := Fforward;
end;

end.
