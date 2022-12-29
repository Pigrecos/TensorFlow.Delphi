unit Keras.Models;
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
          System.Generics.Collections,

          Spring,
          Spring.Collections.Enumerable,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.Variable,

          Keras.ArgsDefinition,
          Keras.Engine,
          Keras.Layer,
          Keras.Optimizer,
          Keras.LossFunc,
          Keras.Container,
          Keras.Data;

type
     ModelConfig = class
      public
        Name        : string;
        Layers      : TList<LayerConfig>;
        InputLayers : TList<NodeConfig>;
        OutputLayers: TList<NodeConfig>;

        function ToString: string; override;
     end;

    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    Model = class(Layer, IModel)
      private
        Fis_compiled             : Boolean;
        Fsteps_per_execution     : IVariableV1;
        Ftrain_counter           : IVariableV1;
        Ftest_counter            : IVariableV1;
        Fpredict_counter         : IVariableV1;
        Fbase_model_initialized  : Boolean;
        // Model.Compile
        compiled_loss            : LossesContainer;
        compiled_metrics         : MetricsContainer;

        function GetLayers: TList<ILayer>;  virtual;
        function Gettrainable_variables: TList<IVariableV1>;
      protected
        Fis_graph_network : Boolean;
        Finputs           : TFTensors;
        Foutputs          : TFTensors;

      public
        loss         : ILossFunc;
        optimizer    : OptimizerV2;
        output_names : TArray<String>;
        stop_training: Boolean;
        data_handler : DataHandler;

        constructor Create(_args: ModelArgs);
        procedure _configure_steps_per_execution(steps_per_execution: Integer);
        procedure _reset_compile_cache;
        procedure _init_batch_counters;
        // Model.Compile
        procedure compile(_optimizer : OptimizerV2= nil; _loss: ILossFunc = nil; metrics : TArray<string>= nil); overload;
        procedure compile(_optimizer: string; _loss: string; metrics: TArray<string>); overload;

        property Layers              : TList<ILayer> read GetLayers ;
        property trainable_variables : TList<IVariableV1> read Gettrainable_variables ;
    end;

    /// <summary>
    /// A `Functional` model is a `Model` defined as a directed graph of layers.
    /// </summary>
    Functional = class(Model)
      private
        Foutput_layers      : TList<ILayer>;
        Finput_layers       : TList<ILayer>;
        Finput_coordinates  : TList<TKerasHistory>;
        Foutput_coordinates : TList<TKerasHistory>;
        Ftensor_usage_count : TDictionary<Int64, Integer>;
      protected
        procedure _init_graph_network(inputs: TFTensors; outputs: TFTensors);
        function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      public
        NetworkNodes : TArray<string>;

        constructor Create(inputs: TFTensors; outputs: TFTensors; name: string = '');
        destructor Destroy; override;
        class function from_config(config: ModelConfig): Functional;
        /// <summary>
        /// Reconstructs graph from config object.
        /// </summary>
        /// <param name="config"></param>
        /// <returns></returns>
        class function  reconstruct_from_config(config: ModelConfig): Tuple<TFTensors, TFTensors, TDictionary<string, ILayer> >;
        class procedure process_layer(created_layers: TDictionary<string, ILayer>; layer_data: LayerConfig; unprocessed_nodes: TDictionary<ILayer, NodeConfig>; node_count_by_layer: TDictionary<ILayer, Integer>);
        class procedure process_node(layer: ILayer; node_data: NodeConfig; created_layers: TDictionary<string, ILayer>; node_count_by_layer: TDictionary<ILayer, Integer>; node_index_map: TDictionary< Tuple<string, Integer>, Integer>) ;
        class function  _should_skip_first_node(layer: ILayer): Boolean;
        /// <summary>
        /// Adds layers that are not connected to the outputs to the model.
        /// </summary>
        /// <param name="created_layers"></param>
        procedure connect_ancillary_layers(created_layers: TDictionary<string, ILayer>) ;
        /// <summary>
        /// Validates a network's topology and gather its layers and nodes.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        function MapGraphNetwork(inputs: TFTensors; outputs: TFTensors): Tuple< TArray<string>, TDictionary<Integer, TList<INode>>, TList<ILayer>, TDictionary<Integer, TList<ILayer>> >;
        /// <summary>
        /// Assigns unique names to the Network's outputs.
        /// </summary>
        procedure _set_output_names;
        /// <summary>
        /// This method topologically sorts nodes in order from inputs to outputs.
        /// </summary>
        /// <param name="outputs"></param>
        function  BuildMap(outputs: TFTensors): Tuple< TList<INode>, TDictionary<ILayer, Integer> >;
        procedure BuildMapHelper(tensor: TFTensor; finished_nodes: TList<INode>; nodes_in_progress: TList<INode>; nodes_in_decreasing_depth: TList<INode>; layer_indices: TDictionary<ILayer, Integer> );
        procedure ComputeTensorUsageCount;
        function  MakeNodeKey(layer_name: string; node_index: Integer): string;
    end;

    /// <summary>
    /// `Sequential` groups a linear stack of layers into a `tf.keras.Model`.
    /// `Sequential` provides training and inference features on this model.
    /// </summary>
    Sequential = class(Functional)
      private
        Fcompute_output_and_mask_jointly : Boolean;
        Fauto_track_sub_layers           : Boolean;
        Finferred_input_shape            : TFShape;
        Fhas_explicit_input_shape        : Boolean;
        Fgraph_initialized               : Boolean;
        Fcreated_nodes                   : TList<INode>;

        function GetOutShape: TFShape;
        function GetLayers: TList<ILayer>; override;
      protected
        function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      public
        args : SequentialArgs;

        constructor Create(_args : SequentialArgs);
        procedure add(tensor: TFTensor); overload;
        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// <param name="layer"></param>
        procedure add(layer: ILayer); overload;
        procedure _handle_deferred_layer_dependencies(layers: TArray<ILayer>);
        procedure _build_graph_network_for_inferred_shape(input_shape: TFShape; input_dtype: TF_DataType);
        procedure clear_previously_created_nodes(layer: ILayer; created_nodes: TList<INode>);
        procedure track_nodes_created_by_last_call(layer: ILayer; created_nodes: TList<INode>);

        property output_shape : TFShape       read GetOutShape;
        property Layers       : TList<ILayer> read GetLayers;
    end;

implementation
         uses Tensorflow,
              TensorFlow.Ops,
              Tensorflow.Utils,

              Keras.Utils,

              ProtoGen.variable;

{ Model }

constructor Model.Create(_args: ModelArgs);
begin
    inherited Create(_args);

    _init_batch_counters;
end;

function Model.GetLayers: TList<ILayer>;
var
  res   : TArray<ILayer>;
begin
    res := _flatten_layers(false, false);

    Result := TList<ILayer>.Create(res) ;
end;

function Model.Gettrainable_variables: TList<IVariableV1>;
var
  variables : TList<IVariableV1> ;
begin
    variables := TList<IVariableV1>.Create;

    if not Trainable then
    begin
        Exit(variables);
    end;

    for var trackable_obj in Fself_tracked_trackables do
    begin
        if trackable_obj.Trainable then
            variables.AddRange(trackable_obj.trainable_variables);
    end;

    for var layer in Flayers do
    begin
        if layer.Trainable then
            variables.AddRange(layer.trainable_variables);
    end;

    // variables.AddRange(_trainable_weights);

    Result := variables;
end;

procedure Model._configure_steps_per_execution(steps_per_execution: Integer);
begin
    Fsteps_per_execution := tf.Variable(steps_per_execution, True, True, True, '', TF_DataType.TF_INT64, VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA);
end;

procedure Model._init_batch_counters;
begin
   Ftrain_counter   := tf.Variable(Int64(0), True, True, True, '', TF_DataType.TF_INT64, VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA);
   Ftest_counter    := tf.Variable(Int64(0), True, True, True, '', TF_DataType.TF_INT64, VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA);
   Fpredict_counter := tf.Variable(Int64(0), True, True, True, '', TF_DataType.TF_INT64, VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA);
end;

procedure Model._reset_compile_cache;
begin
    // Used to cache `trainable` attr of `Layer`s for `fit`.
    Fcompiled_trainable_state := _get_trainable_state;
    tf.keras.backend._GRAPH := nil;
end;

procedure Model.compile(_optimizer: OptimizerV2; _loss: ILossFunc; metrics: TArray<string>);
begin
    if Assigned(_optimizer) then Self.optimizer := _optimizer
    else                         Self.optimizer := TRMSprop.Create( RMSpropArgs.Create ) ;

    if Assigned(_loss) then Self.loss := _loss
    else                    Self.loss := TMeanSquaredError.Create ;

    compiled_loss    := LossesContainer.Create(loss, output_names);
    compiled_metrics := MetricsContainer.Create(metrics, output_names);

    var experimental_steps_per_execution : Integer := 1;
    _configure_steps_per_execution(experimental_steps_per_execution);

    // Initialize cache attrs.
    _reset_compile_cache;
    Fis_compiled := true;
end;

procedure Model.compile(_optimizer, _loss: string; metrics: TArray<string>);
var
  _opt  : OptimizerV2;
  l_Loss: ILossFunc;
begin
    if _optimizer = 'rmsprop' then _opt := TRMSprop.Create( RMSpropArgs.Create )
    else raise Exception.Create('compile - Optimizer :'+ _optimizer+' Not Implemented');

    if      _loss = 'mse' then l_Loss := TMeanSquaredError.Create
    else if _loss = 'mae' then l_Loss := TMeanAbsoluteError.Create
    else raise Exception.Create('compile - LossFunc :'+ _loss+' Not Implemented');

    compile(_opt, l_Loss, metrics);
end;

{ Functional }

constructor Functional.Create(inputs, outputs: TFTensors; name: string);
var
  mArgs : ModelArgs;
begin
    mArgs := ModelArgs.Create;
    mArgs.Name   := name;
    mArgs.Inputs := inputs;
    mArgs.Outputs:= outputs;

    inherited Create(mArgs);

    Finput_layers      := TList<ILayer>.Create;
    Foutput_layers     := TList<ILayer>.Create;
    Finput_coordinates := TList<TKerasHistory>.Create;
    Foutput_coordinates:= TList<TKerasHistory>.Create;
    Ftensor_usage_count:= TDictionary<Int64, Integer>.Create;

    if self is Sequential then Exit;

    _init_graph_network(inputs, outputs);
end;

destructor Functional.Destroy;
begin
    Finput_layers.Free;
    Foutput_layers.Free;
    Finput_coordinates.Free;
    Foutput_coordinates.Free;
    Ftensor_usage_count.Free;

    inherited;
end;

class function Functional.from_config(config: ModelConfig): Functional;
var
  tK : Tuple<TFTensors, TFTensors, TDictionary<string, ILayer>>;
begin
    tK := reconstruct_from_config(config);
    var input_tensors := tK.Value1;
    var output_tensors:= tK.Value2;
    var created_layers:= tK.Value3;

    var model := Functional.Create(input_tensors, output_tensors, config.Name);
    model.connect_ancillary_layers(created_layers);
    Result := model;
end;

class function Functional.reconstruct_from_config(config: ModelConfig): Tuple<TFTensors, TFTensors, TDictionary<string, ILayer>>;
var
  created_layers      : TDictionary<string, ILayer>;
  node_index_map      : TDictionary<Tuple<string, Integer>, integer>;
  node_count_by_layer : TDictionary<ILayer, Integer>;
  unprocessed_nodes   : TDictionary<ILayer, NodeConfig>;
  input_tensors       : TList<TFTensor>;
  output_tensors      : TList<TFTensor>;
begin
    // Layer instances created during the graph reconstruction process.
    created_layers      := TDictionary<string, ILayer>.Create;
    node_index_map      := TDictionary<Tuple<string, Integer>, Integer>.Create;
    node_count_by_layer := TDictionary<ILayer, Integer>.Create;
    unprocessed_nodes   := TDictionary<ILayer, NodeConfig>.Create;

    // First, we create all layers and enqueue nodes to be processed
    for var layer_data in config.Layers do
        process_layer(created_layers, layer_data, unprocessed_nodes, node_count_by_layer);

    // Then we process nodes in order of layer depth.
    // Nodes that cannot yet be processed (if the inbound node
    // does not yet exist) are re-enqueued, and the process
    // is repeated until all nodes are processed.
    while unprocessed_nodes.Count > 0 do
    begin
        for var layer_data in config.Layers do
        begin
            var layer := created_layers[layer_data.Name];
            if unprocessed_nodes.ContainsKey(layer) then
            begin
                var node_data := unprocessed_nodes[layer];
                // for var node_data in unprocessed_nodes[layer] do
                begin
                    process_node(layer, node_data, created_layers, node_count_by_layer, node_index_map);
                    unprocessed_nodes.Remove(layer);
                end
            end;
        end;
    end;

    input_tensors := TList<TFTensor>.Create;
    for var layer_data in config.InputLayers do
    begin
        var layer                := created_layers[layer_data.Name];
        var layer_output_tensors := layer.InboundNodes[layer_data.NodeIndex].Outputs;
        input_tensors.add(layer_output_tensors[layer_data.TensorIndex]);
    end;

    output_tensors := TList<TFTensor>.Create;
    for var layer_data in config.OutputLayers do
    begin
        var layer                := created_layers[layer_data.Name];
        var layer_output_tensors := layer.InboundNodes[layer_data.NodeIndex].Outputs;
        output_tensors.add(layer_output_tensors[layer_data.TensorIndex]);
    end;

    Result := Tuple.Create(TFTensors.Create(input_tensors), TFTensors.Create(output_tensors), created_layers);
end;

class procedure Functional.process_layer(created_layers: TDictionary<string, ILayer>; layer_data: LayerConfig; unprocessed_nodes: TDictionary<ILayer, NodeConfig>;
  node_count_by_layer: TDictionary<ILayer, Integer>);
var
  layer              : ILayer;
  layer_name         : string;
  inbound_nodes_data : TList<NodeConfig>;
begin
    layer      := nil;
    layer_name := layer_data.Name;
    if created_layers.ContainsKey(layer_name) then
        layer := created_layers[layer_name]
    else begin
        if      layer_data.ClassName = 'InputLayer' then layer := InputLayer.from_config(layer_data.Config)
        else if layer_data.ClassName = 'Dense'      then layer := Dense.from_config(layer_data.Config)
        else raise Exception.Create('Not Implemented');

        created_layers.AddOrSetValue(layer_name, layer);
    end;
    if _should_skip_first_node(layer) then node_count_by_layer.AddOrSetValue(layer, 1)
    else                                   node_count_by_layer.AddOrSetValue(layer, 0) ;

    inbound_nodes_data := layer_data.InboundNodes;
    for var node_data in inbound_nodes_data do
    begin
        if not unprocessed_nodes.ContainsKey(layer) then
            unprocessed_nodes.Add(layer, node_data)
        else
            unprocessed_nodes.AddOrSetValue(layer, node_data);
    end;
end;

class procedure Functional.process_node(layer: ILayer; node_data: NodeConfig; created_layers: TDictionary<string, ILayer>; node_count_by_layer: TDictionary<ILayer, Integer>;
  node_index_map: TDictionary<Tuple<string, Integer>, Integer>);
var
  input_tensors : TList<TFTensor>;
  inbound_layer : ILayer;
begin
    input_tensors := TList<TFTensor>.Create;

    inbound_layer    := created_layers[node_data.Name];
    var inbound_node := inbound_layer.InboundNodes[ node_data.NodeIndex ];
    input_tensors.Add(inbound_node.Outputs[ node_data.NodeIndex ]);

    var output_tensors := layer.Apply(TFTensors.Create(input_tensors));

    // Update node index map.
    var output_index := (output_tensors[0].KerasHistory as TKerasHistory).node_index;
    var t : Tuple<string, Integer> := Tuple.Create(layer.Name, node_count_by_layer[layer]) ;
    node_index_map.AddOrSEtValue(t, output_index);
    node_count_by_layer.AddOrSetValue(layer, node_count_by_layer[layer] + 1);
end;

class function Functional._should_skip_first_node(layer: ILayer): Boolean;
begin
     Result :=  (layer is Functional) and (layer.Layers[0] is InputLayer);
end;

procedure Functional.ComputeTensorUsageCount;
var
  available_tensors : TList<Int64>;
  depth_keys        : TArray<Integer>;
  eKeys             : Enumerable<Integer>;
  OrdFun            : TFunc<integer,Integer>;
  input_tensors     : TArray<Int64>;
begin
    OrdFun := Function(x:integer): Integer
               begin
                    Result := x;
               end ;

    available_tensors := TList<Int64>.Create;
    for var tensor in Finputs do
       available_tensors.Add(tensor.id);

    eKeys := Enumerable<Integer>.Create( NodesByDepth.Keys.ToArray);
    depth_keys := eKeys.OrderBy<Integer>(OrdFun).Reversed.Skip(1).ToArray;

    for var depth in depth_keys do
    begin
        for var node in NodesByDepth[depth] do
        begin
            for var tensor in node.KerasInputs do
               input_tensors := input_tensors + [ tensor.id ];

            if TUtils.IsSubSet<Int64>(input_tensors,available_tensors) then
            begin
                for var tensor in node.KerasInputs do
                begin
                    if not Ftensor_usage_count.ContainsKey(tensor.Id) then
                        Ftensor_usage_count.AddOrSetValue(tensor.Id, 0);
                    Ftensor_usage_count[tensor.Id] := Ftensor_usage_count[tensor.Id] + 1;
                end;

                for var output_tensor in node.Outputs do
                    available_tensors.Add(output_tensor.Id);
            end;
        end;
    end;

    for var tensor in Foutputs do
    begin
        if  not Ftensor_usage_count.ContainsKey(tensor.Id) then
            Ftensor_usage_count.AddOrSetValue(tensor.Id, 0);
        Ftensor_usage_count[tensor.Id] := Ftensor_usage_count[tensor.Id] + 1;
    end;
end;

procedure Functional.connect_ancillary_layers(created_layers: TDictionary<string, ILayer>);
begin

end;

procedure Functional._set_output_names;
var
  uniquified   : TList<string>;
  output_names : TList<string>;
  prefix_count : TDictionary<string, Integer>;
begin
    uniquified   := TList<string>.Create;
    output_names := TList<string>.Create;
    prefix_count := TDictionary<string, Integer>.Create;
    try
      for var layer in Foutput_layers do
      begin
          var proposal := layer.Name;
          while output_names.Contains(proposal) do
          begin
              var existing_count := TUtils.Get<string, Integer>(prefix_count, layer.Name, 1);
              proposal := layer.Name+'_'+existing_count.ToString;
              prefix_count.AddOrSetValue(layer.Name, existing_count + 1);
          end;
          output_names.add(proposal);
          uniquified.add(proposal);
      end;

      Self.output_names := uniquified.ToArray;
    finally
      uniquified.Free;
      output_names.Free;
      prefix_count.Free;
    end;
end;

function Functional.MakeNodeKey(layer_name: string; node_index: Integer): string;
begin
   Result := layer_name + '_ib-' + node_index.ToString;
end;

function Functional.BuildMap(outputs: TFTensors): Tuple<TList<INode>, TDictionary<ILayer, Integer>>;
var
  finished_nodes            : TList<INode>;
  nodes_in_progress         : TList<INode>;
  nodes_in_decreasing_depth : TList<INode>;
  layer_indices             : TDictionary<ILayer, integer>;
begin
    finished_nodes            := TList<INode>.Create;
    nodes_in_progress         := TList<INode>.Create;
    nodes_in_decreasing_depth := TList<INode>.Create;
    layer_indices             := TDictionary<ILayer, Integer>.Create;
    for var output in outputs do
        BuildMapHelper(output, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices);

    Result := Tuple.Create(nodes_in_decreasing_depth, layer_indices);
end;

procedure Functional.BuildMapHelper(tensor: TFTensor; finished_nodes, nodes_in_progress, nodes_in_decreasing_depth: TList<INode>; layer_indices: TDictionary<ILayer, Integer>);
var
  kT   : Tuple<ILayer, Integer, Integer>;
begin
    kT := (tensor.KerasHistory as TKerasHistory).ToTuple;
    var layer        := kT.Value1;
    var node_index   := kT.Value2;

    var node := layer.InboundNodes[node_index] as Node;

    // Don't repeat work for shared subgraphs
    if finished_nodes.Contains(node) then
        exit;

    // Prevent cycles.
    if nodes_in_progress.Contains(node) then
       raise Exception.Create('The tensor '+tensor.name+'  at layer '+layer.Name + ' is part of a cycle.');

    // Store the traversal order for layer sorting.
    if not layer_indices.ContainsKey(layer) then
        layer_indices.Add(layer, layer_indices.Count);

    // Propagate to all previous tensors connected to this node.
    nodes_in_progress.Add(node);
    if  not node.is_input then
    begin
        for var k_tensor in node.KerasInputs do
           BuildMapHelper(k_tensor, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices);
    end;

    finished_nodes.Add(node);
    nodes_in_progress.Remove(node);
    nodes_in_decreasing_depth.Add(node);
end;

function Functional.MapGraphNetwork(inputs, outputs: TFTensors): Tuple<TArray<string>, TDictionary<Integer, TList<INode>>, TList<ILayer>, TDictionary<Integer, TList<ILayer>>>;
var
  tMap                      : Tuple<TList<INode>, TDictionary<ILayer, Integer>>;
  nodes_in_decreasing_depth : TList<INode>;
  layers                    : TList<ILayer>;
  layer_indices             : TDictionary<ILayer, Integer>;
  network_nodes             : TArray<string>;
  depth_keys                : TArray<Integer>;
  nodes_depths              : TDictionary<INode, Integer>;
  layers_depths             : TDictionary<ILayer, Integer>;
  nodes_by_depth            : TDictionary<Integer, TList<INode>>;
  layers_by_depth           : TDictionary<Integer, TList<ILayer>>;
  kT                        : Tuple<ILayer, Integer, Integer>;
  eKeys                     : Enumerable<Integer>;
  OrdFun                    : TFunc<integer,Integer>;
  OrdFun1                   : TFunc<ILayer,Integer>;
begin
   OrdFun := Function(x:integer): Integer
           begin
                Result := x;
           end ;
   OrdFun1:= Function(x:ILayer): Integer
           begin
                Result := layer_indices[x];
           end ;

    tMap := BuildMap(outputs);
    nodes_in_decreasing_depth := tMap.Value1;
    layer_indices             := tMap.Value2;

    network_nodes := [];
    for var node in nodes_in_decreasing_depth do
    begin
         var sNode : string := MakeNodeKey(node.Layer.Name, node.Layer.InboundNodes.IndexOf(node));
         network_nodes := network_nodes + [ sNode ];
    end;

    nodes_depths  := TDictionary<INode, integer>.Create;
    layers_depths := TDictionary<ILayer, Integer>.Create;

    nodes_in_decreasing_depth.Reverse;
    for var node in nodes_in_decreasing_depth do
    begin
        // If the depth is not set, the node has no outbound nodes (depth 0).
        var depth : Integer := TUtils.SetDefault<INode, integer>(nodes_depths,node, 0);
        // Update the depth of the corresponding layer
        var previous_depth : Integer := TUtils.Get<ILayer, Integer>(layers_depths,node.Layer, 0);
        // If we've seen this layer before at a higher depth,
        // we should use that depth instead of the node depth.
        // This is necessary for shared layers that have inputs at different
        // depth levels in the graph.
        depth := Max(depth, previous_depth);
        layers_depths.AddOrSetValue(node.Layer, depth);
        nodes_depths.AddOrSetValue(node, depth);

        // Update the depth of inbound nodes.
        // The "depth" of a node is the max of the depths
        // of all nodes it is connected to + 1.
        for var node_dep in node.ParentNodes do
        begin
            previous_depth := TUtils.Get<INode, Integer>(nodes_depths,node_dep, 0);
            nodes_depths.AddOrSetValue(node_dep, Max(depth + 1, previous_depth));
        end;
    end;

    // Handle inputs that are not connected to outputs.
    // We do not error out here because the inputs may be used to compute losses
    // and metrics.
    for var input_t in inputs do
    begin
        kT := (input_t.KerasHistory as TKerasHistory).ToTuple;
        var input_layer := kT.Value1;
        if not layers_depths.ContainsKey(input_layer) then
        begin
            layers_depths.AddOrSetValue(input_layer, 0);
            layer_indices.AddOrSetValue(input_layer, -1);
            nodes_depths.AddOrSetValue(input_layer.InboundNodes[0], 0);
            network_nodes := network_nodes + [ MakeNodeKey(input_layer.Name, 0) ];
        end;
    end;

    // Build a dict {depth: list of nodes with this depth}
    nodes_by_depth := TDictionary<Integer, TList<INode>>.Create;
    for var dNode in nodes_depths do
    begin
        var node := dNode.Key;
        var depth:= dNode.Value;
        if not nodes_by_depth.ContainsKey(depth) then
            nodes_by_depth.AddOrSetValue(depth, TList<INode>.Create);
        nodes_by_depth[depth].add(node);
    end;

    layers_by_depth := TDictionary<Integer, TList<ILayer>>.Create;
    for var dLayer in layers_depths do
    begin
        var layer := dLayer.Key;
        var depth:= dLayer.Value;
        if not layers_by_depth.ContainsKey(depth) then
            layers_by_depth.AddOrSetValue(depth, TList<ILayer>.Create);
        layers_by_depth[depth].add(layer);
    end;

    // Get sorted list of layer depths.
    eKeys      := Enumerable<Integer>.Create( layers_by_depth.Keys.ToArray );
    depth_keys := eKeys.OrderBy<Integer>(OrdFun).Reversed.ToArray;

    // Set self.layers ordered by depth.
    layers := TList<ILayer>.Create;
    for var depth in depth_keys do
    begin
        var layers_for_depth : TList<ILayer> := layers_by_depth[depth];

        // Network.layers needs to have a deterministic order:
        // here we order them by traversal order.
        var e : Enumerable<ILayer> := Enumerable<ILayer>.Create(layers_for_depth.ToArray);
        layers_for_depth := TList<ILayer>.Create( e.OrderBy<Integer>(OrdFun1).ToArray );
        layers.AddRange(layers_for_depth);
    end;

    // Get sorted list of node depths.
    var e := Enumerable<Integer>.Create( nodes_by_depth.Keys.ToArray );
    depth_keys := eKeys.OrderBy<Integer>(OrdFun).Reversed.ToArray;

    Result := Tuple.Create (network_nodes, nodes_by_depth, layers, layers_by_depth);
end;

procedure Functional._init_graph_network(inputs, outputs: TFTensors);
 var
   kT  : Tuple<ILayer, Integer, Integer>;
   mapT: Tuple<TArray<string>, TDictionary<Integer, TList<INode>>, TList<ILayer>, TDictionary<Integer, TList<ILayer>>>;
begin
    Fis_graph_network := true;
    Finputs           := inputs;
    Foutputs          := outputs;
    Fbuilt            := true;
    if base_layer_utils.needs_keras_history(outputs) then
            base_layer_utils.create_keras_history(outputs);
    // Build self._output_layers:
    for var x in outputs do
    begin
        kT := (x.KerasHistory as TKerasHistory).ToTuple;
        var layer        := kT.Value1;
        var node_index   := kT.Value2;
        var tensor_index:=  kT.Value3;
        Foutput_layers.Add(layer);
        Foutput_coordinates.Add(TKerasHistory.Create(layer, node_index, tensor_index));
    end;
     // Build self._input_layers:
    for var x in inputs do
    begin
        kT := (x.KerasHistory as TKerasHistory).ToTuple;
        var layer        := kT.Value1;
        var node_index   := kT.Value2;
        var tensor_index:=  kT.Value3;
        Finput_layers.Add(layer);
        Finput_coordinates.Add(TKerasHistory.Create(layer, node_index, tensor_index));
    end;
    // Keep track of the network's nodes and layers.
    mapT := MapGraphNetwork(inputs, outputs);
    var nodes          := mapT.Value1;
    var nodes_by_depth := mapT.Value2;
    var layers         := mapT.Value3;
    NetworkNodes := nodes;
    NodesByDepth := nodes_by_depth;
    if Flayers.Count = 0 then
      Flayers := layers;
    Fself_tracked_trackables := layers;
    _set_output_names;
    ComputeTensorUsageCount;
end;

function Functional.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
 selFunc     : TFunc<Integer, TFTensor>;
 tensor_dict : TDictionary<Int64, TQueue<TFTensor>>;
 x,y         : TFTensor;
 eKeys       : Enumerable<Integer>;
 depth_keys  : TArray<Integer>;
 OrdFun      : TFunc<integer,Integer>;
begin
    selFunc := function(v : Integer):TFTensor
                begin
                    Result := y;
                end;
    OrdFun := Function(z:integer): Integer
                begin
                     Result := z;
                end ;

    x := nil;

    tensor_dict := TDictionary<Int64, TQueue<TFTensor>>.Create;

    // map input values
    for var x_y in TUtils.zip<TFTensor>(Self.Finputs, inputs) do
    begin
        x := x_y.Value1;
        y := x_y.Value2;

        var enu : TArray<TFTensor> := TUtils.range(0, Ftensor_usage_count[x.Id]).Select<TFTensor>(selFunc).ToArray;
        var q := TQueue<TFTensor>.Create( TList<TFTensor>.Create(enu) );

        tensor_dict.AddOrSetValue(x.Id, q);
    end ;

    eKeys := Enumerable<Integer>.Create( NodesByDepth.Keys.ToArray);
    depth_keys := eKeys.OrderBy<Integer>(OrdFun).Reversed.ToArray;

    for var depth in depth_keys do
    begin
        var nodes := NodesByDepth[depth];
        for var i := 0 to nodes.count-1 do
        begin
            var nNode: Node := nodes[i] as Node;
            // Input tensors already exist.
            if nNode.is_input then
                continue;

            var layer_inputs := nNode.MapArguments(tensor_dict);

            tf.Logger.Debug( Format('Depth %s: %s: %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name]),'Layer');

            var isTarainig : Boolean := False;
            if assigned(training) then isTarainig := training^;

            var outputs := nNode.Layer.Apply(layer_inputs, nil, isTarainig);
            for var output in outputs do
            begin
                if output <> nil  then
                   tf.Logger.Info( Format('Depth %s: %s: %s %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name, output.shape.ToString]),'Layer');
            end;
            // Update tensor_dict for next or later input
            for var z := 0 to nNode.Outputs.Count - 1 do
            begin
                var x_id := outputs[z].id;
                y        := outputs[z];

                var enu : TArray<TFTensor> := TUtils.range(0, Ftensor_usage_count[x_id]).Select<TFTensor>(selFunc).ToArray;
                var q := TQueue<TFTensor>.Create( TList<TFTensor>.Create(enu) );

               tensor_dict.AddOrSetValue(x.Id, q);
            end;
        end;
    end;

    var output_tensors := TFTensors.Create;

    for x in Foutputs do
        output_tensors.Add(tensor_dict[x.Id].Dequeue);

    Result := output_tensors;
end;

{ Sequential }

constructor Sequential.Create(_args: SequentialArgs);
begin
    inherited Create(_args.Inputs, _args.Outputs, _args.Name);

    args := _args;
    if args.Layers = nil then
        args.Layers := TList<ILayer>.Create;

    // SupportsMasking = true;
    Fcompute_output_and_mask_jointly := true;
    Fauto_track_sub_layers           := false;
    Fhas_explicit_input_shape        := false;
    Fis_graph_network                := false;
    Fcreated_nodes                   := TList<INode>.Create;

    // Add to the model any layers passed to the constructor.
    if args.Layers <> nil then
    begin
        for var layer in args.Layers do
            add(layer);
    end;
end;

procedure Sequential.add(tensor: TFTensor);
begin
    var layer := (tensor.KerasHistory as TKerasHistory).Layer;
    add(layer);
end;

procedure Sequential.add(layer: ILayer);
begin
    Fbuilt         := false;
    var set_inputs := false;
    if Flayers.Count = 0 then
    begin
        if layer is InputLayer then
        begin
            set_inputs := true;
        end else
        begin
            if not layer.BatchInputShape.isNil then
            begin
                // Instantiate an input layer.
                var x := tf.keras.Input(default(TFShape), layer.BatchInputShape, -1, layer.DType, layer.Name + '_input');

                // This will build the current layer
                // and create the node connecting the current layer
                // to the input layer we just created.
                layer.Apply( TFTensors.Create(x));
                set_inputs := true;
            end;
        end;

        if set_inputs then
        begin
            // If an input layer (placeholder) is available.
            Foutputs := layer.InboundNodes.Last.Outputs;
            Finputs  := layer_utils.get_source_inputs(Foutputs[0]);
            Fbuilt   := true;
            Fhas_explicit_input_shape := true;
        end;
    end
    else if Foutputs <> nil then
    begin
        Foutputs := layer.Apply(Foutputs);
        Fbuilt   := true;
    end;

    if (set_inputs) or (Fis_graph_network) then
    begin
        _init_graph_network(Finputs, Foutputs);
        Fis_graph_network := true;
    end else
    begin
        Fself_tracked_trackables.add(layer);
        _handle_deferred_layer_dependencies([layer]);
    end;
end;

function Sequential.GetLayers: TList<ILayer>;
var
   _layers : TList<ILayer>;
begin
    _layers := inherited Layers;

    Result := TList<ILayer>.Create;

    for var i := 0 to _layers.Count-1 do
    begin
        if _layers[i] is InputLayer then  Continue ;
        Result.Add(_layers[i])
    end;

end;

function Sequential.GetOutShape: TFShape;
begin
    Result := Foutputs[0].shape;
end;

function Sequential.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if not Fhas_explicit_input_shape then
       _build_graph_network_for_inferred_shape(inputs.shape, inputs.dtype);

    if Fgraph_initialized then
    begin
        if not Fbuilt then
            _init_graph_network(Finputs, Foutputs);
        Result  := inherited Call(inputs, state, training);
        Exit;
    end;

    Result := inherited Call(inputs, state, training);
end;

procedure Sequential.clear_previously_created_nodes(layer: ILayer; created_nodes: TList<INode>);
var
  outNodes, inNodes: TList<INode>;
begin
    outNodes := TList<INode>.Create;
    inNodes  := TList<INode>.Create;

    for var node in layer.InboundNodes do
    begin
        for var prev_layer in node.InboundLayers do
        begin
            for var x in prev_layer.OutboundNodes do
            begin
                if not created_nodes.Contains(x) then
                  outNodes.Add(x);
            end;
            prev_layer.OutboundNodes.Clear;
            prev_layer.OutboundNodes.AddRange(outNodes);
        end;
    end;

    for var x in layer.InboundNodes do
    begin
      if not created_nodes.Contains(x) then
        inNodes.Add(x);
    end;
    layer.InboundNodes.Clear;
    layer.InboundNodes.AddRange(inNodes);
end;

procedure Sequential.track_nodes_created_by_last_call(layer: ILayer; created_nodes: TList<INode>);
begin
    var node := layer.InboundNodes.Last;
    created_nodes.Add(node);
    for var prev_layer in node.InboundLayers do
        created_nodes.add(prev_layer.OutboundNodes.Last);
end;

procedure Sequential._build_graph_network_for_inferred_shape(input_shape: TFShape; input_dtype: TF_DataType);
var
  layer_input  : TFTensors;
  layer_output : TFTensors;
  outputs      : TFTensors;
  created_nodes: TList<INode>;
begin
    if Finferred_input_shape = input_shape then
       Exit;
    TOps.init_scope;
    var inputs := tf.keras.Input(default(TFShape), input_shape, -1, input_dtype, Flayers[0].Name+'_input');
    layer_input  := TFTensors.Create(inputs);
    outputs      := nil;
    created_nodes:= TList<INode>.Create;
    for var layer in Flayers do
    begin
      clear_previously_created_nodes(layer, Fcreated_nodes);
      layer_output := layer.Apply(layer_input);
      // Keep track of nodes just created above
      track_nodes_created_by_last_call(layer, created_nodes);
      layer_input := layer_output;
      outputs     := layer_output;
    end;
    Fcreated_nodes        := created_nodes;
    _init_graph_network(TFTensors.Create(inputs), outputs);
    Fgraph_initialized    := true;
    Finferred_input_shape := input_shape;

end;

procedure Sequential._handle_deferred_layer_dependencies(layers: TArray<ILayer>);
begin
    Flayers.AddRange(layers);
end;

{ ModelConfig }

function ModelConfig.ToString: string;
begin
    Result := Format('%s, %d Layers, %d Input Layers %d Output Layers',[Name, Layers.Count, InputLayers.Count, OutputLayers.Count])
end;

end.
