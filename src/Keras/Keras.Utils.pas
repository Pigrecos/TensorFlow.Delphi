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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
       uses System.SysUtils,
            Winapi.Windows,
            System.Math,
            System.Character,
            System.Generics.Collections,
            System.IOUtils,
            System.Net.HttpClient,
            System.Classes,
            System.ZLib,
            System.Zip,

            Spring,
            Spring.Collections.Enumerable,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,
            TensorFlow.Variable,

            Keras.Layer,
            Keras.Engine,
            Keras.Models;

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
      class function get_source_inputs(tensor: TFTensor; layer : ILayer = nil ;  node_index : Integer= -1): TFTensors; static;
      // for model.summary
      //
      class procedure print_summary(mModel: Model; line_length: Integer = -1; positions:TArray<Single>= []); static;
      class procedure print_row(fields: TArray<string>; positions: TArray<Integer>); static;
      /// <summary>
      /// Prints a summary for a single layer.
      /// </summary>
      /// <param name="layer"></param>
      class procedure print_layer_summary(lLayer: ILayer; positions: TArray<Integer>); static;
      class procedure print_layer_summary_with_connections(lLayer: ILayer; positions: TArray<Integer>; relevant_nodes: TList<INode>); static;
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

  losses_utils = record
    public
      class function compute_weighted_loss(losses: TFTensor; sample_weight: TFTensor = nil; reduction: string = ''; name: string = ''): TFTensor; static;
      class function scale_losses_by_sample_weight(losses: TFTensor; sample_weight: TFTensor): TFTensor; static;
      class function squeeze_or_expand_dimensions(y_pred: TFTensor; sample_weight: TFTensor): Tuple<TFTensor,TFTensor>; static;
      class function reduce_weighted_loss(weighted_losses: TFTensor; reduction: string): TFTensor; static;
      class function _safe_mean(losses: TFTensor; num_present: TFTensor): TFTensor; static;
      class function _num_elements(losses: TFTensor): TFTensor; static;
  end;

  KerasUtils = class
    public
      /// <summary>
      /// Downloads a file from a URL if it not already in the cache.
      /// </summary>
      /// <param name="fname">Name of the file</param>
      /// <param name="origin">Original URL of the file</param>
      /// <param name="untar"></param>
      /// <param name="md5_hash"></param>
      /// <param name="file_hash"></param>
      /// <param name="cache_subdir"></param>
      /// <param name="hash_algorithm"></param>
      /// <param name="extract"></param>
      /// <param name="archive_format"></param>
      /// <param name="cache_dir"></param>
      /// <returns></returns>
      function get_file(fname: string; origin: string; untar: Boolean = false; md5_hash: string = ''; file_hash: string = ''; cache_subdir: string = 'datasets'; hash_algorithm: string = 'auto'; extract: Boolean = false; archive_format: string = 'auto'; cache_dir: string = '') : string;
  end;

implementation
          uses Tensorflow,
               Tensorflow.Utils,
               TensorFlow.Initializer,
               Tensorflow.NameScope,
               TensorFlow.Ops,
               Tensorflow.math_ops,
               Tensorflow.array_ops,

               Keras.ArgsDefinition,
               Keras.LossFunc,

               ProtoGen.variable,

               ipztar, ipzzip,ipzgzip;

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
    Result := tf.Variable<TFunc<TFTensor>>(init_val, args.Trainable, args.ValidateShape, args.UseResource, args.Name, variable_dtype, TVariableAggregation.VARIABLE_AGGREGATION_NONE,Ps ) ;
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

class function layer_utils.get_source_inputs(tensor: TFTensor; layer: ILayer; node_index: Integer): TFTensors;
var
   kT  : Tuple<ILayer, Integer, Integer>;
begin
    if layer = nil then
    begin
        kT := (tensor.KerasHistory as TKerasHistory).ToTuple;
        layer      := kT.Value1;
        node_index := kT.Value2;
    end;
    if (layer.InboundNodes = nil) or (layer.InboundNodes.Count = 0) then
        Exit( TFTensors.Create(tensor) )
    else
    begin
        var node := layer.InboundNodes[node_index];
        if node.is_input then
            Exit( node.input_tensors )
        else
        begin
            var source_tensors := TList<TFTensor>.Create;
            for var _layer in node.iterate_inbound do
            begin
                layer      := _layer.Value1;
                node_index := _layer.Value2;
                tensor     := _layer.Value4;
                var previous_sources := get_source_inputs(tensor, layer, node_index);
                for var x in previous_sources do
                begin
                    // should be check if exist?
                    source_tensors.add(x);
                end
            end;
            Result := TFTensors.Create(source_tensors);
        end;
    end;
end;

class procedure layer_utils.print_summary(mModel: Model; line_length: Integer; positions: TArray<Single>);
var
  sequential_like: Boolean;
  nodes          : TList<INode>;
  layer          : ILayer;
  node           : INode;
  flag           : Boolean;
  to_display     : TArray<string>;
  relevant_nodes : TList<INode>;
  positions_int  : TArray<Integer>;
  i              : Integer;
  ePosition      : Enumerable<Single>;
begin
    sequential_like := mModel is Sequential;
    if not sequential_like then
    begin
        sequential_like := True;
        nodes := TList<INode>.Create;
        for var v in mModel.NodesByDepth do
        begin
            if (v.Value.Count > 1) or ((v.Value.Count = 1) and (v.Value[0].KerasInputs.Count > 1)) then
            begin
                sequential_like := False;
                Break;
            end;
            nodes.AddRange(v.Value);
        end;
        if sequential_like then
        begin
            for layer in mModel.Layers do
            begin
                flag := False;
                for node in layer.InboundNodes do
                begin
                    if nodes.Contains(node) then
                    begin
                        if flag then
                        begin
                            sequential_like := False;
                            Break;
                        end
                        else
                          flag := True;
                    end;
                    if not sequential_like then
                      Break;
                end;
            end;
        end;
    end;

    relevant_nodes := TList<INode>.Create;
    if sequential_like then
    begin
      if line_length < 0 then
        line_length := 65;

      if positions = nil then
        positions := [0.45, 0.85, 1.0];

      ePosition := Enumerable<Single>.Create(positions);
      if positions[Length(positions) - 1] <= 1 then
        positions := ePosition.Select(
                      function (p: Single): Single
                      begin
                          Result := line_length * p;
                      end).ToArray;
      to_display := ['Layer (type)', 'Output Shape', 'Param #'];
    end else
    begin
      if line_length < 0 then
        line_length := 98;

      if positions = nil then
        positions := [0.33, 0.55, 0.67, 1.0];

      ePosition := Enumerable<Single>.Create(positions);
      if positions[Length(positions) - 1] <= 1 then
        positions := ePosition.Select(
                      function (p: Single): Single
                      begin
                        Result := line_length * p;
                      end).ToArray;
      to_display := ['Layer (type)', 'Output Shape', 'Param #', 'Connected to'];
      for var v in mModel.NodesByDepth do
        relevant_nodes.AddRange(v.Value);
    end;
    var ePositionI := Enumerable<Integer>.Create(positions_int);
    positions_int := ePositionI.Select(
                        function (x: Integer): Integer
                        begin
                          Result := Trunc(x);
                        end).ToArray;

    tf.Logger.Info(Format('Model: %s', [mModel.Name]),'Summary');

    var sLinea := '';
    for i := 0 to line_length -1 do sLinea := sLinea + '_';
    tf.Logger.Info(sLinea,'Summary');

    print_row(to_display, positions_int);

    var sLinea1 := '';
    for i := 0 to line_length -1 do sLinea1 := sLinea1 + '=';
    tf.Logger.Info(sLinea1,'Summary');

    for i := 0 to mModel.Layers.Count- 1 do
    begin
        layer := mModel.Layers[i] ;
        if sequential_like then  print_layer_summary(layer, positions_int)
        else                     print_layer_summary_with_connections(layer, positions_int, relevant_nodes);

        if i = mModel.Layers.Count - 1 then  tf.Logger.Info(sLinea1,'Summary')
        else                                 tf.Logger.Info(sLinea,'Summary');
    end;

    var trainable_count     := count_params(mModel, mModel.trainable_variables);
    var non_trainable_count := count_params(mModel, mModel.non_trainable_variables);
    tf.Logger.Info(Format('Total params: %d', [trainable_count + non_trainable_count]),'Summary');
    tf.Logger.Info(Format('Trainable params: %d', [trainable_count]),'Summary');
    tf.Logger.Info(Format('Non-trainable params: %d', [non_trainable_count]),'Summary');

end;

class procedure layer_utils.print_row(fields: TArray<string>; positions: TArray<Integer>);
var
  line: string;
  i: Integer;
  spaces : TArray<String>;
begin
    line := '';
    for i := 0 to Length(fields) - 1 do
    begin
        if i > 0 then
          line := line + ' ';

        line := line + fields[i];
        line := Copy(line, 1, positions[i]);

        SetLength(spaces, positions[i] - Length(line));
        FillChar(spaces[0], Length(spaces) * SizeOf(Char), ' ');
        line := line + string.Join('', spaces);
    end;
    tf.Logger.Info(line,'Summary');
end;

class procedure layer_utils.print_layer_summary(lLayer: ILayer; positions: TArray<Integer>);
var
  sName: string;
  fields: TArray<string>;
begin
  sName := lLayer.Name;

  var v : TValue := TValue.From<ILayer>(lLayer);
  var nTipoL := v.TypeInfo.Name;

  fields := [sName + ' (' + nTipoL + ')',  lLayer.output_shape.ToString ,  lLayer.count_params.ToString ] ;

  print_row(fields, positions);
end;

class procedure layer_utils.print_layer_summary_with_connections(lLayer: ILayer; positions: TArray<Integer>; relevant_nodes: TList<INode>);
var
  connections: TList<string>;
  node: INode;
  name: string;
  first_connection: string;
  i: Integer;
  fields: TArray<string>;
begin
    connections := TList<string>.Create;
    for node in lLayer.InboundNodes do
    begin
        if not relevant_nodes.Contains(node) then
          Continue;
        for var tEnum in node.iterate_inbound do
        begin
            var inbound_layer := tEnum.Value1;
            var node_index    := tEnum.Value2;
            var tensor_index  := tEnum.Value3;
            connections.Add(inbound_layer.Name + '[' + node_index.ToString + '][' + tensor_index.ToString + ']');
        end;
    end;
    name := lLayer.Name;
    first_connection := '';
    if connections.Count > 0 then
      first_connection := connections[0];

    var v : TValue := TValue.From<ILayer>(lLayer);
    var nTipoL := v.TypeInfo.Name;

    fields := [ name + '(' +nTipoL+ ')', lLayer.output_shape.ToString, lLayer.count_params.ToString ];

    print_row(fields, positions);
    if connections.Count > 1 then
    begin
        for i := 1 to connections.Count - 1 do
        begin
            fields := ['',  '', '', connections[i] ];
            print_row(fields, positions);
        end;
    end;
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

{ KerasUtils }

function KerasUtils.get_file(fname, origin: string; untar: Boolean; md5_hash, file_hash, cache_subdir, hash_algorithm: string; extract: Boolean; archive_format,
  cache_dir: string): string;
var
  Web     : THTTPClient;
  MS      : TMemoryStream;

begin
    if string.IsNullOrEmpty(cache_dir) then
        cache_dir := TPath.GetTempPath;

    var datadir_base := cache_dir;
    TDirectory.CreateDirectory(datadir_base);

    var datadir := TPath.Combine(datadir_base, cache_subdir);
    TDirectory.CreateDirectory(datadir);

    Web := THTTPClient.Create;
    try
      MS := TMemoryStream.Create;
      try
        Web.Get(origin, MS);
        MS.SaveToFile(datadir+'\'+fname);

        var archive := TPath.Combine(datadir, fname);

        if archive.EndsWith('.zip') then
        begin
           var AZipFile := TipzZip.Create(nil);
           try
             AZipFile.ArchiveFile   := archive;
             AZipFile.ExtractToPath := datadir;
             AZipFile.ExtractAll;
           finally
              AZipFile.Free;
           end;
        end
        else if archive.EndsWith('.tgz')then
        begin
           var ATgzFile := TipzTar.Create(nil);
           try
             ATgzFile.ArchiveFile        := archive;
             ATgzFile.UseGzipCompression := true;
             ATgzFile.ExtractToPath      := datadir;
             ATgzFile.ExtractAll;
           finally
             ATgzFile.Free;
           end;
        end
        else if archive.EndsWith('.gz') then
        begin
           var AGzFile := TipzGzip.Create(nil);
           try
             AGzFile.ArchiveFile     := archive;
             AGzFile.ExtractToPath   := datadir;
             AGzFile.ExtractAll;
           finally
             AGzFile.Free;
           end;
        end;
      finally
        MS.Free;
      end;
    finally
      Web.Free;
    end;

    Result := datadir;
end;

{ losses_utils }

class function losses_utils.compute_weighted_loss(losses, sample_weight: TFTensor; reduction, name: string): TFTensor;
begin
    if sample_weight = nil then
    begin
        if losses.dtype = TF_DataType.TF_DOUBLE then sample_weight := tf.constant(Double(1.0))
        else                                         sample_weight := tf.constant(Single(1.0));
    end;
    var weighted_losses := scale_losses_by_sample_weight(losses, sample_weight);
    // Apply reduction function to the individual weighted losses.
    var loss := reduce_weighted_loss(weighted_losses, reduction);
    // Convert the result back to the input type.
    // loss = math_ops.cast(loss, losses.dtype);
    Result := loss;
end;

class function losses_utils.scale_losses_by_sample_weight(losses, sample_weight: TFTensor): TFTensor;
begin
    // losses = math_ops.cast(losses, dtypes.float32);
    // sample_weight = math_ops.cast(sample_weight, dtypes.float32);
    // Update dimensions of `sample_weight` to match with `losses` if possible.
    // (losses, sample_weight) = squeeze_or_expand_dimensions(losses, sample_weight);
    Result := math_ops.multiply(losses, sample_weight);
end;

class function losses_utils.squeeze_or_expand_dimensions(y_pred, sample_weight: TFTensor): Tuple<TFTensor, TFTensor>;
begin
    var weights_shape := sample_weight.shape;
    var weights_rank  := weights_shape.ndim;
    if weights_rank = 0 then
        Exit( Tuple.Create(y_pred, sample_weight) );
    raise TFException.Create('Not Implemented ');
end;

class function losses_utils.reduce_weighted_loss(weighted_losses: TFTensor; reduction: string): TFTensor;
begin
    if reduction = ReductionV2.NONE then
        Exit(weighted_losses)
    else begin
        var loss := math_ops.reduce_sum(weighted_losses);
        if reduction = ReductionV2.SUM_OVER_BATCH_SIZE then
            loss := _safe_mean(loss, _num_elements(weighted_losses));
        Result := loss;
    end;
end;

class function losses_utils._safe_mean(losses, num_present: TFTensor): TFTensor;
begin
    var total_loss := math_ops.reduce_sum(losses);
    Result := math_ops.div_no_nan(total_loss, num_present, 'value');
end;

class function losses_utils._num_elements(losses: TFTensor): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('num_elements'),
                          function(v1: TNameScope): TFTensor
                            begin
                                var scope : string := v1.ToString;
                                Result := math_ops.cast(array_ops.size(losses, scope), losses.dtype);
                            end );
end;


end.
