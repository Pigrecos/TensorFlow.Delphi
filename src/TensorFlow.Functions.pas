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
     uses System.SysUtils,
          System.Generics.Collections,

          Spring,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.DApiBase,
          Tensorflow.Graph,
          TensorFlow.Framework;
type

   ConcreteFunction  = class
     private
        function Get_CaptInput: TArray<TFTensor>;
        function Get_Inputs: TArray<TFTensor>;
        function Get_Name: string;
     protected
        func_graph : TFuncGraph;
        {TODO -oMax -cImplementare : Implementare ForwardBackwardCall}
        //forward_backward: ForwardBackwardCall;
        Outputs         : TArray<TFTensor>;
        ReturnType      : PTypeInfo;
     public
        OutputStructure : TArray<TensorSpec>;

        constructor Create(_name: string); overload;
        constructor Create(graph: TFuncGraph; attrs: TDictionary<string, string> = nil); overload;
        constructor Create(func: TFunc<TFTensor, TFTensor>; dtype: TF_DataType; func_name: string = ''); overload;
        procedure   ToGraph(inputs: TFTensors; outputs: TFTensors);
        function    FilteredCall(inputs: TFTensors): TFTensors;
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


function ConcreteFunction.CallFlat(args, captured_inputs: TArray<TFTensor>): TFTensors;
begin
    var executing_eagerly := tf.Context.executing_eagerly;
    var default_graph     := Tops.get_default_graph;
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

   { if (forward_backward == null)
        forward_backward = SelectForwardAndBackwardFunctions(args, possible_gradient_type, executing_eagerly);
    var (forward_function, args_with_tangents) = forward_backward.Forward();
    Tensors flat_outputs = null;
    if (executing_eagerly)
        flat_outputs = forward_function.Call(args_with_tangents);
    forward_backward.Record(flat_outputs);
    return flat_outputs;}
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

function ConcreteFunction.ToString: string;
begin
    Result := Name
end;

end.
