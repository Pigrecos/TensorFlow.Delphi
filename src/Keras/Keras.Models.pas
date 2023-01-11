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
          Tensorflow.Gradient,

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
       // Fbase_model_initialized  : Boolean;
        // Model.Compile
        compiled_loss            : LossesContainer;
        compiled_metrics         : MetricsContainer;

        function GetLayers: TList<ILayer>;  virtual;
        function Gettrainable_variables: TList<IVariableV1>;
        function GetMetrics: TList<Metric>;
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
        procedure Build(input_shape: TFShape); override;
        procedure _configure_steps_per_execution(steps_per_execution: Integer);
        procedure _reset_compile_cache;
        procedure _init_batch_counters;
        procedure reset_metrics;
        // Model.Compile
        //
        procedure compile(_optimizer : OptimizerV2= nil; _loss: ILossFunc = nil; metrics : TArray<string>= nil); overload;
        procedure compile(_optimizer: string; _loss: string; metrics: TArray<string>); overload;
        // Model.Predict
        //
        /// <summary>
        /// Generates output predictions for the input samples.
        /// </summary>
        /// <param name="x">Input samples</param>
        /// <param name="batch_size">Number of samples per batch</param>
        /// <param name="verbose">Verbosity mode</param>
        /// <param name="steps">
        /// Total number of steps (batches of samples)
        /// before declaring the prediction round finished.
        /// </param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <returns></returns>
        function predict(x: TFTensor; batch_size : Integer= -1; verbose: Integer = 0; steps: Integer = -1; max_queue_size: Integer = 10; workers: Integer = 1; use_multiprocessing : Boolean = false): TFTensors;
        function run_predict_step(iterator: OwnedIterator): TFTensors;
        function predict_step(data: TFTensor): TFTensors;
        // Model.Evaluate
        //
        /// <summary>
        /// Returns the loss value & metrics values for the model in test mode.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="batch_size"></param>
        /// <param name="verbose"></param>
        /// <param name="steps"></param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <param name="return_dict"></param>
        procedure evaluate(x                  : TNDArray;
                          y                   : TNDArray;
                          batch_size          : Integer= -1;
                          verbose             : Integer = 1;
                          steps               : Integer = -1;
                          max_queue_size      : Integer= 10;
                          workers             : Integer= 1;
                          use_multiprocessing : Boolean = false;
                          return_dict         : Boolean= false); overload;
        function evaluate(x: IDatasetV2):  TArray<TPair<string, Single> >;overload;
        function test_function(iterator: OwnedIterator) : TList< Tuple<string, TFTensor> >;
        function test_step(x: TFTensor; y: TFTensor): TList< Tuple<string, TFTensor> >;
        // Model.fit
        //
        /// <summary>
        /// Trains the model for a fixed number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="batch_size"></param>
        /// <param name="epochs"></param>
        /// <param name="verbose"></param>
        /// <param name="validation_split"></param>
        /// <param name="shuffle"></param>
        procedure fit(x: TNDArray; y      : TNDArray;
                      batch_size          : Integer= -1;
                      epochs              : Integer= 1;
                      verbose             : Integer = 1;
                      validation_split    : Single= 0.0;
                      shuffle             : Boolean= true;
                      initial_epoch       : Integer= 0;
                      max_queue_size      : Integer= 10;
                      workers             : Integer= 1;
                      use_multiprocessing : Boolean= false); overload;
        procedure fit(dataset             : IDatasetV2;
                      batch_size          : Integer= -1;
                      epochs              : Integer= 1;
                      verbose             : Integer = 1;
                      validation_split    : Single= 0.0;
                      shuffle             : Boolean= true;
                      initial_epoch       : Integer= 0;
                      max_queue_size      : Integer= 10;
                      workers             : Integer= 1;
                      use_multiprocessing : Boolean= false); overload;
        procedure FitInternal(epochs: Integer; verbose: Integer);
        procedure on_epoch_begin(epoch: Integer; epochs: Integer);
        procedure on_train_batch_begin(verbose: Integer; step: Int64; elapse: Int64; results: TList<Tuple<string, TFTensor>>) ;
        // Model.Train
        //
        function train_step_function(iterator: OwnedIterator): TList<Tuple<string, TFTensor>>;
        /// <summary>
        /// The logic for one training step.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        function  train_step(x: TFTensor; y: TFTensor): TList<Tuple<string, TFTensor>>;
        procedure _minimize(tape: TGradientTape; optimizer: OptimizerV2; loss: TFTensor; trainable_variables: TList<IVariableV1>);
        // MOdel.Summary
        //
        /// <summary>
        /// Prints a string summary of the network.
        /// </summary>
        procedure summary(line_length: Integer = -1; positions: TArray<Single> = []);

        property Layers              : TList<ILayer> read GetLayers ;
        property TrainableVariables  : TList<IVariableV1> read Gettrainable_variables ;
        property Metrics             : TList<Metric> read GetMetrics ;
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

        property OutputShape  : TFShape       read GetOutShape;
        property Layers       : TList<ILayer> read GetLayers;
    end;

implementation
         uses Tensorflow,
              TensorFlow.Tensor,
              TensorFlow.Ops,
              Tensorflow.Utils,
              TensorFlow.Slice,
              Tensorflow.Graph,

              NumPy.NDArray,

              Keras.Utils,

              ProtoGen.variable;

{ Model }

constructor Model.Create(_args: ModelArgs);
begin
    inherited Create(_args);

    _init_batch_counters;
end;

procedure Model.evaluate(x, y: TNDArray; batch_size, verbose, steps, max_queue_size, workers: Integer; use_multiprocessing, return_dict: Boolean);
var
  res     : TList<Tuple<string, TFTensor> > ;
  dataArgs: DataHandlerArgs;
begin
    dataArgs := DataHandlerArgs.Create;
    dataArgs.X             :=  x;
    dataArgs.Y             :=  y;
    dataArgs.BatchSize     :=  batch_size;
    dataArgs.StepsPerEpoch :=  steps;
    dataArgs.InitialEpoch  :=  0;
    dataArgs.Epochs        :=  1;
    dataArgs.MaxQueueSize  :=  max_queue_size;
    dataArgs.Workers       :=  workers;
    dataArgs.UseMultiprocessing :=  use_multiprocessing;
    dataArgs.Model         :=  Self;
    dataArgs.StepsPerExecution :=  Fsteps_per_execution;

    data_handler := DataHandler.Create(dataArgs);

    tf.LogMsg('Testing...');
    for var epoch_iterator in data_handler.enumerate_epochs do
    begin
        var epoch   := epoch_iterator.Value1;
        var iterator:= epoch_iterator.Value2;
        reset_metrics;
        // callbacks.on_epoch_begin(epoch)
        // data_handler.catch_stop_iteration();
        res := nil ;
        for var step in data_handler.steps do
        begin
            // callbacks.on_train_batch_begin(step)
            res := test_function(iterator);
        end;

        var sArray : TArray<string> := [];
        for var i := 0 to res.Count - 1 do
        begin
            var tTensor     : TTensor := res[i].Value2;
            var floatTensor : Single := Single(tTensor);
            sArray := sArray + [ res[i].Value1, FloatToStr(floatTensor) ];
        end;
        var sRes := string.Join(', ',sArray) ;

        tf.LogMsg('iterator: '+ IntToStr(epoch + 1 ) +', ' + sRes);
    end;
end;

function Model.evaluate(x: IDatasetV2): TArray<TPair<string, Single>>;
var
  logs     : TList<Tuple<string, TFTensor> > ;
  dataArgs : DataHandlerArgs;
begin
    dataArgs := DataHandlerArgs.Create;
    dataArgs.Dataset       :=  x;
    dataArgs.Model         :=  Self;
    dataArgs.StepsPerExecution :=  Fsteps_per_execution;

    data_handler := DataHandler.Create(dataArgs);

    tf.LogMsg('Testing...');
    logs := nil ;
    for var epoch_iterator in data_handler.enumerate_epochs do
    begin
        var epoch   := epoch_iterator.Value1;
        var iterator:= epoch_iterator.Value2;
        reset_metrics;
        // callbacks.on_epoch_begin(epoch)
        // data_handler.catch_stop_iteration();
        for var step in data_handler.steps do
        begin
            // callbacks.on_train_batch_begin(step)
            logs := test_function(iterator);
        end;

        var sArray : TArray<string> := [];
        for var i := 0 to logs.Count - 1 do
        begin
            var tTensor     : TTensor := logs[i].Value2;
            var floatTensor : Single := Single(tTensor);
            sArray := sArray + [ logs[i].Value1, FloatToStr(floatTensor) ];
        end;
        var sRes := string.Join(', ',sArray) ;

        tf.LogMsg('iterator: '+ IntToStr(epoch + 1 ) +', ' + sRes);
    end;
    Result := [];
    for var i := 0 to logs.Count - 1 do
    begin
        var tTensor     : TTensor := logs[i].Value2;
        var floatTensor : Single := Single(tTensor);
        Result := Result + [ TPair<string, Single>.create(logs[i].Value1, floatTensor) ];
    end;
end;

procedure Model.fit(x, y: TNDArray; batch_size, epochs, verbose: Integer; validation_split: Single; shuffle: Boolean; initial_epoch, max_queue_size, workers: Integer;
  use_multiprocessing: Boolean);
var
   dataArgs        : DataHandlerArgs;
   train_count     : Integer;
   train_x,train_y : NDArray;
begin
    train_count := trunc(x.dims[0] * (1 - validation_split));
    train_x     := x[ [Slice.Create(0, train_count)] ];
    train_y     := y[ [Slice.Create(0, train_count)] ];

    dataArgs := DataHandlerArgs.Create;

    dataArgs.X             := train_x;
    dataArgs.Y             := train_y;
    dataArgs.BatchSize     := batch_size;
    dataArgs.InitialEpoch  := initial_epoch;
    dataArgs.Epochs        := epochs;
    dataArgs.Shuffle       := shuffle;
    dataArgs.MaxQueueSize  := max_queue_size;
    dataArgs.Workers       := workers;
    dataArgs.UseMultiprocessing := use_multiprocessing;
    dataArgs.Model         := Self;
    dataArgs.StepsPerExecution := Fsteps_per_execution;

    data_handler := DataHandler.Create(dataArgs);

    FitInternal(epochs, verbose);
end;

procedure Model.fit(dataset: IDatasetV2; batch_size, epochs, verbose: Integer; validation_split: Single; shuffle: Boolean; initial_epoch, max_queue_size, workers: Integer;
  use_multiprocessing: Boolean);
var
   dataArgs        : DataHandlerArgs;
begin
    dataArgs := DataHandlerArgs.Create;

    dataArgs.Dataset            := dataset;
    dataArgs.BatchSize          := batch_size;
    dataArgs.InitialEpoch       := initial_epoch;
    dataArgs.Epochs             := epochs;
    dataArgs.Shuffle            := shuffle;
    dataArgs.MaxQueueSize       := max_queue_size;
    dataArgs.Workers            := workers;
    dataArgs.UseMultiprocessing := use_multiprocessing;
    dataArgs.Model              := Self;
    dataArgs.StepsPerExecution  := Fsteps_per_execution;

    data_handler := DataHandler.Create(dataArgs);

    FitInternal(epochs, verbose);

end;

procedure Model.FitInternal(epochs, verbose: Integer);
var
  iterator: OwnedIterator;
  epoch   : Integer;
  step    : Integer;
  results : TList<Tuple<string, TFTensor>>;
  sw      : TStopWatch;
begin
    stop_training := False;
    if      Ftrain_counter is RefVariable          then (Ftrain_counter as RefVariable)         .assign_add(Integer(0))
    else if Ftrain_counter is BaseResourceVariable then (Ftrain_counter as BaseResourceVariable).assign_add(Integer(0))
    else raise Exception.Create('Model.FitInterna Error!');

    sw := TStopWatch.Create;
    for var it in data_handler.enumerate_epochs do
    begin
        epoch   := it.Value1;
        iterator:= it.Value2;
        reset_metrics;
        on_epoch_begin(epoch, epochs);
        // data_handler.catch_stop_iteration();
        for step in data_handler.steps do
        begin
            sw.Start;
            results := train_step_function(iterator);
            sw.Stop;
            on_train_batch_begin(verbose, step, sw.ElapsedMilliseconds, results);

            // recycle memory more frequency
            if sw.ElapsedMilliseconds > 100 then
            begin
               // GC.Collect;
            end;
            sw.Reset;
        end;
        tf.LogMsg('');

        // GC.Collect;
        // GC.WaitForPendingFinalizers;
    end;

end;

procedure Model.on_epoch_begin(epoch, epochs: Integer);
begin
    tf.LogMsg(Format('Epoch: %.3d/%.3d',[epoch+1, epochs]));
end;

procedure Model.on_train_batch_begin(verbose: Integer; step, elapse: Int64; results: TList<Tuple<string, TFTensor>>);
var
  resultPairs: string;
  progress, remaining: string;
  i, j: Integer;
begin
    if verbose = 1 then
    begin
        resultPairs := '';
        for i := 0 to results.Count - 1 do
        begin
            var tTensor : TTensor := results[i].Value2;
            resultPairs := resultPairs + Format('%s: %.6f', [results[i].Value1, Single(tTensor)]);
            if i < results.Count - 1 then
              resultPairs := resultPairs + ', ';
        end;

        progress := '';
        for i := 0 to step do
          for j := 0 to 30 div data_handler.Inferredsteps - 1 do
            progress := progress + '=';
        progress := progress + '>';

        remaining := '';
        for i := 1 to 30 - Length(progress) - 1 do
          remaining := remaining + '.';

        tf.LogMsg(Format('%.4d/%.4d [%s%s] - %dms/step %s', [step + 1, data_handler.Inferredsteps, progress, remaining, elapse, resultPairs]));
    end;
end;

function Model.test_function(iterator: OwnedIterator): TList<Tuple<string, TFTensor>>;
begin
    var data   := iterator.next;
    var outputs:= test_step(data[0], data[1]);
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Ftest_counter is RefVariable          then Result := (Ftest_counter as RefVariable)         .assign_add(Integer(1))
            else if Ftest_counter is BaseResourceVariable then Result := (Ftest_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.test_function Error!');
        end);
    Result := outputs;
end;

function Model.test_step(x, y: TFTensor): TList<Tuple<string, TFTensor>>;
begin
    var x_y  := data_handler.DataAdapter.Expand1d(x, y);
    x := x_y.Value1;
    y := x_y.Value2;
    var y_pred := Apply(TFTensors.Create(x), nil, false);
    compiled_loss.Call(y, y_pred.first);

    compiled_metrics.update_state(y, y_pred.first);

    Result := TList<Tuple<string, TFTensor>>.Create;
    for var i := 0 to Metrics.Count -1 do
    begin
        var t := Tuple<string, TFTensor>.Create( Metrics[i].Name, Metrics[i].R_result );
        Result.Add(t);
    end;
end;

function Model.train_step_function(iterator: OwnedIterator): TList<Tuple<string, TFTensor>>;
begin
    var data    := iterator.next();
    var outputs := train_step(data[0], data[1]);
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Ftrain_counter is RefVariable          then Result := (Ftrain_counter as RefVariable)         .assign_add(Integer(1))
            else if Ftrain_counter is BaseResourceVariable then Result := (Ftrain_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.train_step_function Error!');
        end);
    Result := outputs;
end;

function Model.train_step(x, y: TFTensor): TList<Tuple<string, TFTensor>>;
begin
    var x_y := data_handler.DataAdapter.Expand1d(x, y);
    x := x_y.Value1;
    y := x_y.Value2;

    var tape   := tf.GradientTape;
    var y_pred := Apply(TFTensors.Create(x), nil, true);
    var loss   := compiled_loss.Call(y, y_pred.First);

    // For custom training steps, users can just write:
    // trainable_variables = self.trainable_variables
    // gradients = tape.gradient(loss, trainable_variables)
    // self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    // The _minimize call does a few extra steps unnecessary in most cases,
    // such as loss scaling and gradient clipping.
    _minimize(tape, optimizer, loss, TrainableVariables);
    compiled_metrics.update_state(y, y_pred.First);

    Result := TList<Tuple<string, TFTensor>>.Create;
    for var i := 0 to metrics.Count - 1 do
       Result.Add(Tuple.Create(metrics[i].Name, metrics[i].R_result));
end;

procedure Model._minimize(tape: TGradientTape; optimizer: OptimizerV2; loss: TFTensor; trainable_variables: TList<IVariableV1>);
var
  gradients : TArray<TFTensor>;
  gradientsAndVariables: TList<Tuple<TFTensor, IVariableV1>>;
  gradientsAndResVariables: TList<Tuple<TFTensor, ResourceVariable>>;
begin
    gradients := tape.gradient(loss, trainable_variables.ToArray);
    gradientsAndVariables := TList<Tuple<TFTensor, IVariableV1>>.Create;
    try
      for var i := 0 to Length(gradients) - 1 do
        gradientsAndVariables.Add(Tuple<TFTensor, IVariableV1>.Create(gradients[i], trainable_variables[i]));

      gradients := optimizer._aggregate_gradients(gradientsAndVariables.ToArray);
      gradients := optimizer._clip_gradients(gradients);

      gradientsAndResVariables := TList<Tuple<TFTensor, ResourceVariable>>.Create;
      try
        for var i := 0 to Length(gradients) - 1 do
          gradientsAndResVariables.Add(Tuple<TFTensor, ResourceVariable>.Create(gradients[i], trainable_variables[i] as ResourceVariable));

        optimizer.apply_gradients(gradientsAndResVariables.ToArray, '', False);
      finally
        gradientsAndResVariables.Free;
      end;

    finally
     gradientsAndVariables.Free;
    end;
end;

function Model.GetLayers: TList<ILayer>;
var
  res   : TArray<ILayer>;
begin
    res := _flatten_layers(false, false);

    Result := TList<ILayer>.Create(res) ;
end;

function Model.GetMetrics: TList<Metric>;
begin
    var _metrics := TList<Metric>.Create;

    if Fis_compiled then
    begin
        if compiled_loss <> nil then
            _metrics.AddRange(compiled_loss.metrics);
        if compiled_metrics <> nil then
            _metrics.AddRange(compiled_metrics.metrics);
    end;

    (*foreach (var layer in _flatten_layers())
        _metrics.extend(layer.metrics);*)

    Result := _metrics;
end;

procedure Model.reset_metrics;
begin
    for var metric in Metrics  do
       metric.reset_states;
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
            variables.AddRange(trackable_obj.TrainableVariables);
    end;

    for var layer in Fself_tracked_trackables do
    begin
        if layer.Trainable then
            variables.AddRange(layer.TrainableVariables);
    end;

    // variables.AddRange(_trainable_weights);

    Result := variables;
end;

function Model.predict(x: TFTensor; batch_size, verbose, steps, max_queue_size, workers: Integer; use_multiprocessing: Boolean): TFTensors;
var
  dataArgs : DataHandlerArgs;
begin
    dataArgs := DataHandlerArgs.Create;

    dataArgs.X             := x;
    dataArgs.BatchSize     := batch_size;
    dataArgs.InitialEpoch  := 0;
    dataArgs.Epochs        := 1;
    dataArgs.MaxQueueSize  := max_queue_size;
    dataArgs.Workers       := workers;
    dataArgs.UseMultiprocessing := use_multiprocessing;
    dataArgs.Model         := self;
    dataArgs.StepsPerExecution := Fsteps_per_execution;

    data_handler := DataHandler.Create(dataArgs);

    var outputs : TFTensors := nil;

    if      Fpredict_counter is RefVariable          then (Fpredict_counter as RefVariable)         .assign(Integer(1))
    else if Fpredict_counter is BaseResourceVariable then (Fpredict_counter as BaseResourceVariable).assign(Integer(1))
    else raise Exception.Create('Model.run_predict_step Error!');

    // callbacks.on_predict_begin()
    for  var it in data_handler.enumerate_epochs do
    begin
        var epoch   := it.Value1;
        var iterator:= it.Value2;
        for var step in data_handler.steps do
        begin
            // callbacks.on_predict_batch_begin(step)
            var batch_outputs := run_predict_step(iterator);
            outputs           := batch_outputs;
            var end_step      := step + data_handler.StepIncrement;
            // callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
        end;

    end ;
    // callbacks.on_predict_end()
    Result := outputs;
end;

function Model.run_predict_step(iterator: OwnedIterator): TFTensors;
begin
    var data   := iterator.next;
    var outputs:= predict_step(data[0]);
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Fpredict_counter is RefVariable          then Result := (Fpredict_counter as RefVariable)         .assign_add(Integer(1))
            else if Fpredict_counter is BaseResourceVariable then Result := (Fpredict_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.run_predict_step Error!');
        end);
    Result := outputs;
end;

procedure Model.summary(line_length: Integer; positions: TArray<Single>);
begin
   layer_utils.print_summary(self, line_length, positions);
end;

function Model.predict_step(data: TFTensor): TFTensors;
begin
    Result := Apply(TFTensors.Create(data), nil, false);
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

procedure Model.Build(input_shape: TFShape);
var
  graph : TFGraph;
begin
    if tf.executing_eagerly then graph := TFuncGraph.Create('build_graph')
    else                      graph := tf.keras.backend.get_graph;

    graph.as_default;
    var x := tf.placeholder(DType, input_shape);
    var pTraining := False;
    Call(TFTensors.Create(x), nil, @pTraining );
    graph.gExit;

    inherited Build(input_shape);
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
   layer : ILayer;
   kT    : Tuple<ILayer, Integer, Integer>;
   mapT  : Tuple<TArray<string>, TDictionary<Integer, TList<INode>>, TList<ILayer>, TDictionary<Integer, TList<ILayer>>>;
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
        layer            := kT.Value1;
        var node_index   := kT.Value2;
        var tensor_index :=  kT.Value3;
        Foutput_layers.Add(layer);
        Foutput_coordinates.Add(TKerasHistory.Create(layer, node_index, tensor_index));
    end;
     // Build self._input_layers:
    for var x in inputs do
    begin
        kT := (x.KerasHistory as TKerasHistory).ToTuple;
        layer            := kT.Value1;
        var node_index   := kT.Value2;
        var tensor_index :=  kT.Value3;
        Finput_layers.Add(layer);
        Finput_coordinates.Add(TKerasHistory.Create(layer, node_index, tensor_index));
    end;
    // Keep track of the network's nodes and layers.
    mapT := MapGraphNetwork(inputs, outputs);
    NetworkNodes             := mapT.Value1;
    NodesByDepth             := mapT.Value2;
    Fself_tracked_trackables := mapT.Value3;

    _set_output_names;
    ComputeTensorUsageCount;
end;

function Functional.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
 tensor_dict : TDictionary<Int64, TQueue<TFTensor>>;
 x,y         : TFTensor;
 eKeys       : Enumerable<Integer>;
 depth_keys  : TArray<Integer>;
 outputs     : TFTensors;
begin
    tensor_dict := TDictionary<Int64, TQueue<TFTensor>>.Create;

    // map input values
    for var x_y in TUtils.zip<TFTensor>(Self.Finputs, inputs) do
    begin
        x := x_y.Value1;
        y := x_y.Value2;

        var enu : TArray<TFTensor> := TUtils.range(0, Ftensor_usage_count[x.Id]).Select<TFTensor>(function(v : Integer):TFTensor
                                                                                                    begin
                                                                                                        Result := y;
                                                                                                    end).ToArray;
        var q := TQueue<TFTensor>.Create( TList<TFTensor>.Create(enu) );

        tensor_dict.AddOrSetValue(x.Id, q);
    end ;

    eKeys := Enumerable<Integer>.Create( NodesByDepth.Keys.ToArray);
    depth_keys := eKeys.OrderBy<Integer>(Function(z:integer): Integer
                                            begin
                                                 Result := z;
                                            end).Reversed.ToArray;

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

            tf.LogMsg(Format('Depth %s:  %s:  %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name]));

            var isTarainig : Boolean := False;
            if assigned(training) then isTarainig := training^;

            outputs := nNode.Layer.Apply(layer_inputs, nil, isTarainig);
            for var output in outputs do
            begin
                if output <> nil  then
                   tf.LogMsg( Format('Depth %s: %s: %s %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name, output.shape.ToString]));
            end;
            // Update tensor_dict for next or later input
            for var z := 0 to nNode.Outputs.Count - 1 do
            begin
                var x_id := nNode.outputs[z].id;
                y        := outputs[z];

                var enu : TArray<TFTensor> := TUtils.range(0, Ftensor_usage_count[x_id]).Select<TFTensor>(function(v : Integer):TFTensor
                                                                                                            begin
                                                                                                                Result := y;
                                                                                                            end).ToArray;
                var q := TQueue<TFTensor>.Create( TList<TFTensor>.Create(enu) );

                tensor_dict.AddOrSetValue(x_id, q);
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
    if Fself_tracked_trackables.Count = 0 then
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
    var inputs := tf.keras.Input(default(TFShape), input_shape, -1, input_dtype, Fself_tracked_trackables[0].Name+'_input');
    layer_input  := TFTensors.Create(inputs);
    outputs      := nil;
    created_nodes:= TList<INode>.Create;
    for var layer in Fself_tracked_trackables do
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
    Fself_tracked_trackables.AddRange(layers);
end;

{ ModelConfig }

function ModelConfig.ToString: string;
begin
    Result := Format('%s, %d Layers, %d Input Layers %d Output Layers',[Name, Layers.Count, InputLayers.Count, OutputLayers.Count])
end;

end.
