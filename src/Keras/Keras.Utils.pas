unit Keras.Utils;
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
       uses System.SysUtils,
            System.Math,
            System.Character,
            System.Generics.Collections,

            Spring,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,
            TensorFlow.Variable,

            Keras.Layer;

type
  base_layer_utils = record
    private

    public
        /// <summary>
        /// Adds a new variable to the layer.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        class function make_variable(args: VariableArgs): IVariableV1; static;
        /// <summary>
        /// Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        class function unique_layer_name(name: string; name_uid_map : TDictionary<string, Integer>= nil; avoid_names : TArray<string>= []; zero_based: Boolean = false): string ; static;
        class function needs_keras_history(inputs: TFTensors): Boolean; static;
        class function get_default_graph_uid_map: TDictionary<string, Integer>;  static;
        class function create_keras_history(inputs: TFTensors): TArray<Layer>; static;
        class procedure CreateKerasHistoryHelper(tensors: TFTensors; processed_ops: TList<TFOperation>; created_layers: TList<Layer>) ; static;
        class function  uses_keras_history(op_input: TFTensor): Boolean; static;
  end;

  layer_utils = record
    private

    public
      class function count_params(_layer: Layer; weights: TList<IVariableV1>) : Integer ; static;
  end;

  generic_utils = record
    private

    public
      class function to_snake_case(name: string): string; static;
  end;

  conv_utils = record
    private

    public
      class function convert_data_format(data_format: string; ndim: Integer): string; static;
      class function normalize_tuple(value: TArray<Integer>; n: Integer; name: string): TArray<Integer> ; static;
      class function normalize_padding(value: string): string; static;
      class function normalize_data_format(value: string): string; static;
      class function deconv_output_length(input_length: Integer; filter_size: Integer; padding: string; output_padding: Integer = -1; stride: Integer = 0; dilation: Integer = 1): Integer; static;
  end;

implementation
          uses Tensorflow,
               Tensorflow.Utils,
               TensorFlow.Initializer,
               Tensorflow.NameScope,
               TensorFlow.Ops,

               Keras.ArgsDefinition,

               ProtoGen.variable;

{ base_layer_utils }

class function base_layer_utils.make_variable(args: VariableArgs): IVariableV1;
begin
    var init_val : TFunc<TFTensor> := function: TFTensor
                                        begin
                                            Result := args.Initializer.Apply(InitializerArgs.Create(args.Shape, args.DType));
                                        end;
    var variable_dtype := Tdtypes.as_base_dtype(args.DType);
    var s := args.Shape;
    var Ps : PTFShape := nil;
    if not s.IsNil then
       Ps := @s;
    Result := tf.Variable(init_val, args.Trainable, args.ValidateShape, args.UseResource, args.Name, variable_dtype, TVariableAggregation.VARIABLE_AGGREGATION_NONE,Ps );
end;

class function base_layer_utils.needs_keras_history(inputs: TFTensors): Boolean;
begin
    for var it in inputs do
    begin
        if it.KerasHistory = nil then
          Exit(True);
    end;
    Result := false;
end;

class function base_layer_utils.unique_layer_name(name: string; name_uid_map: TDictionary<string, Integer>; avoid_names: TArray<string>; zero_based: Boolean): string;
begin
    if name_uid_map = nil then
        name_uid_map := get_default_graph_uid_map;

    var proposed_name : string := '';
    while (proposed_name = '') or ( TArray.Contains<string>(avoid_names, proposed_name) ) do
    begin
        if not name_uid_map.ContainsKey(name) then
            name_uid_map.Add(name, 0);

        if zero_based then
        begin
            var number: integer := name_uid_map[name];
            if number > 0 then
                proposed_name := name+'_'+ IntToStr(number)
            else
                proposed_name := name;

            name_uid_map[name] := name_uid_map[name] + 1;
        end else
        begin
            name_uid_map[name] := name_uid_map[name] + 1;
            proposed_name      := name+'_'+ IntToStr(name_uid_map[name]);
        end;
    end;

    Result := proposed_name;
end;

class function base_layer_utils.create_keras_history(inputs: TFTensors): TArray<Layer>;
begin
    var processed_ops  := TList<TFOperation>.Create;
    var created_layers := TList<Layer>.Create;
    CreateKerasHistoryHelper(inputs, processed_ops, created_layers);
    Result := created_layers.ToArray;
end;

class procedure base_layer_utils.CreateKerasHistoryHelper(tensors: TFTensors; processed_ops: TList<TFOperation>; created_layers: TList<Layer>);
begin
    for var tensor in tensors do
    begin
        if tensor.KerasHistory <> nil then
            continue;

        var op := tensor.op;
        if not processed_ops.Contains(op) then
        begin
            var layer_inputs := TList<TFTensor>.Create;
            var constants    := TDictionary<integer, TNDArray>.Create;
            for var i : Integer := 0 to Length(op.inputs.Inputs) - 1 do
            begin
                var op_input := op.inputs.Inputs[i];
                if uses_keras_history(op_input) then
                    layer_inputs.Add(op_input)
                else begin
                    TUtils.tf_with<TNameScope>( Tops.init_scope,
                          procedure(v1: TNameScope)
                            begin
                                constants[i] := tf.keras.backend.eval_in_eager_or_function(TFTensors.Create(op_input));
                            end );
                end;
            end;

            // recursively
            CreateKerasHistoryHelper(TFTensors.Create(layer_inputs.ToArray), processed_ops, created_layers);

            var opLayerArgs := TensorFlowOpLayerArgs.Create;
            opLayerArgs.NodeDef   := op.NodeDef;
            opLayerArgs.Constants := constants;
            opLayerArgs.Name      := op.name;

            var op_layer := TensorFlowOpLayer.Create(opLayerArgs);
            created_layers.Add(op_layer);
            op_layer.SetConnectivityMetadata(TFTensors.Create(layer_inputs.ToArray), TFTensors.Create(op.outputs));
            processed_ops.Add(op);
        end;
    end;
end;

class function base_layer_utils.uses_keras_history(op_input: TFTensor): Boolean;
begin
    if op_input.KerasHistory <> nil then
        Exit( true );

    for var input in op_input.op.Inputs.Inputs do
        if uses_keras_history(input) then
            Exit( true );

    Result := false;
end;

class function base_layer_utils.get_default_graph_uid_map: TDictionary<string, Integer>;
begin
    var graph := Tops.get_default_graph;
    var name_uid_map : TDictionary<string, Integer> ;
    if tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS.ContainsKey(graph) then
    begin
        name_uid_map := tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS[graph];
    end else
    begin
        name_uid_map := TDictionary<string, Integer>.Create;
        tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS.Add(graph, name_uid_map);
    end;

    Result := name_uid_map;
end;

{ layer_utils }

class function layer_utils.count_params(_layer: Layer; weights: TList<IVariableV1>): Integer;
begin
    var total := 0;
    var weight_shapes : TArray<TFShape> := [];
    for var i := 0 to weights.Count - 1 do
         weight_shapes := weight_shapes + [ weights[i].Shape ];

    for var i := 0 to Length(weight_shapes) - 1 do
       total := total + weight_shapes[i].Size ;

    Result := total;
end;

{ generic_utils }

class function generic_utils.to_snake_case(name: string): string;
begin
    Result := '';
    for var i := 1 to name.Length do
    begin
       if (name[i].IsUpper) and ( not name[i - 1].IsDigit ) then Result := Result + '_'+ name[i]
       else                                                      Result := Result + name[i]
    end;
end;

{ conv_utils }

class function conv_utils.convert_data_format(data_format: string; ndim: Integer): string;
begin
    if data_format = 'channels_last' then
    begin
        if      ndim = 3 then Exit('NWC')
        else if ndim = 4 then Exit('NHWC')
        else if ndim = 5 then Exit('NDHWC')
        else
            raise Exception.Create('Input rank not supported: '+ ndim.ToString );
    end
    else if data_format = 'channels_first' then
    begin
        if      ndim = 3 then Exit('NCW')
        else if ndim = 4 then Exit('NCHW')
        else if ndim = 5 then Exit('NCDHW')
        else
            raise Exception.Create('Input rank not supported: ' +ndim.ToString );
    end
    else
       raise Exception.Create('Invalid data_format: '+ data_format);
end;

class function conv_utils.normalize_tuple(value: TArray<Integer>; n: Integer; name: string): TArray<Integer>;
begin
    if Length(value) = 1 then
    begin
         for var i := 0 to n-1 do
           Result := Result + [  value[0] ];
    end else
    begin
        Result := value;
    end;
end;

class function conv_utils.deconv_output_length(input_length, filter_size: Integer; padding: string; output_padding, stride, dilation: Integer): Integer;
begin
    // Get the dilated kernel size
    filter_size := filter_size + (filter_size - 1) * (dilation - 1);

    // Infer length if output padding is None, else compute the exact length
    var _length : Integer := -1;
    if output_padding = -1 then
    begin
        if      padding = 'valid' then  _length := input_length * stride + max(filter_size - stride, 0)
        else if padding = 'full'  then  _length := input_length * stride - (stride + filter_size - 2)
        else if padding = 'same'  then  _length := input_length * stride;
    end else
    begin
        raise Exception.Create('Not Implemented');
    end;
    Result := _length;
end;

class function conv_utils.normalize_data_format(value: string): string;
begin                 
    if string.IsNullOrEmpty(value) then
        Exit ( 'channels_last' );
    Result := value.ToLower;
end;

class function conv_utils.normalize_padding(value: string): string;
begin
    Result := value.ToLower;
end;

end.
