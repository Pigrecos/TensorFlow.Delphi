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
          TensorFlow.Core,

          keras.Callbacks,
          Keras.Core,
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
      strict private
        FOnEpochBegin      : TCB_On_Epoch_Begin;
        FOnEpochEnd        : TCB_On_Epoch_End;
        FOnTrainBatchBegin : TCB_On_Train_Batch_Begin ;
        FOnTrainBatchEnd   : TCB_On_Train_Batch_End ;
        FOnEndSummary      : TCB_On_End_Summary ;

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
        function GetTrainW: TList<IVariableV1>;
        function GetNotTrainW: TList<IVariableV1>;
        function GetMetrics: TList<IMetricFunc>;
        //
        function  Get_OnEpochBegin: TCB_On_Epoch_Begin;
        procedure Set_OnEpochBegin(Value: TCB_On_Epoch_Begin);
        //
        function  Get_OnEpochEnd: TCB_On_Epoch_End;
        procedure Set_OnEpochEnd(Value: TCB_On_Epoch_End);
        //
        function  Get_OnTrainBatchBegin: TCB_On_Train_Batch_Begin;
        procedure Set_OnTrainBatchBegin(Value: TCB_On_Train_Batch_Begin);
        //
        function  Get_OnTrainBatchEnd: TCB_On_Train_Batch_End;
        procedure Set_OnTrainBatchEnd(Value: TCB_On_Train_Batch_End);
        //
        function  Get_OnEndSummary: TCB_On_End_Summary;
        procedure Set_OnEndSummary(Value: TCB_On_End_Summary);

      protected
        Fis_graph_network : Boolean;
        Finputs           : TFTensors;
        Foutputs          : TFTensors;

      public
        loss         : ILossFunc;
        optimizer    : IOptimizer;
        output_names : TArray<String>;
        stop_training: Boolean;

        constructor Create(_args: ModelArgs);
        destructor  Destroy; override;
        procedure Initialize(_args: LayerArgs); override;
        procedure Build(input_shape: TFShape); override;
        procedure _configure_steps_per_execution(steps_per_execution: Integer);
        procedure _reset_compile_cache;
        procedure _init_batch_counters;
        procedure reset_metrics;
        // Model.Compile
        //
        procedure compile(_optimizer : IOptimizer= nil; _loss: ILossFunc = nil; metrics : TArray<string>= nil); overload;
        procedure compile(_optimizer : string;          _loss: string;          metrics: TArray<string>); overload;
        procedure compile(_optimizer : IOptimizer= nil; _loss: ILossFunc = nil; metrics : TArray<IMetricFunc>= nil); overload;
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
        function predict(x                   : TFTensors;
                         batch_size          : Integer= -1;
                         verbose             : Integer = 0;
                         steps               : Integer = -1;
                         max_queue_size      : Integer = 10;
                         workers             : Integer = 1;
                         use_multiprocessing : Boolean = false): TFTensors; overload;
        function predict(dataset             : IDatasetV2;
                         batch_size          : Integer= -1;
                         verbose             : Integer = 0;
                         steps               : Integer = -1;
                         max_queue_size      : Integer = 10;
                         workers             : Integer = 1;
                         use_multiprocessing : Boolean = false): TFTensors; overload;
        function PredictInternal(data_handler: DataHandler; verbose: Integer): TFTensors;
        function run_predict_step(iterator: OwnedIterator): TFTensors;
        function predict_step(data: TFTensors): TFTensors; overload;
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
        function test_function(data_handler : DataHandler; iterator: OwnedIterator) : TList< Tuple<string, TFTensor> >;
        function test_step(data_handler : DataHandler; x: TFTensor; y: TFTensor): TList< Tuple<string, TFTensor> >;
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
        function fit(x: TNDArray; y      : TNDArray;
                      batch_size          : Integer= -1;
                      epochs              : Integer= 1;
                      verbose             : Integer = 1;
                      validation_split    : Single= 0.0;
                      shuffle             : Boolean= true;
                      initial_epoch       : Integer= 0;
                      max_queue_size      : Integer= 10;
                      workers             : Integer= 1;
                      use_multiprocessing : Boolean= false): ICallback; overload;
        function fit( x: TArray<TNDArray>; y      : TNDArray;
                  batch_size          : Integer= -1;
                  epochs              : Integer= 1;
                  verbose             : Integer = 1;
                  validation_split    : Single= 0.0;
                  shuffle             : Boolean= true;
                  initial_epoch       : Integer= 0;
                  max_queue_size      : Integer= 10;
                  workers             : Integer= 1;
                  use_multiprocessing : Boolean= false): ICallback; overload;
        function fit(dataset             : IDatasetV2;
                      validation_data     : IDatasetV2= nil;
                      batch_size          : Integer= -1;
                      epochs              : Integer= 1;
                      verbose             : Integer = 1;
                      validation_split    : Single= 0.0;
                      shuffle             : Boolean= true;
                      initial_epoch       : Integer= 0;
                      max_queue_size      : Integer= 10;
                      workers             : Integer= 1;
                      use_multiprocessing : Boolean= false): History; overload;
        function FitInternal(data_handler : DataHandler; epochs: Integer; verbose: Integer; validation_data: IDatasetV2; train_step_func : TFunc< DataHandler, OwnedIterator, TDictionary<string, single>>): History;
        // Model.Train
        //
        function train_step_function(data_handler : DataHandler; iterator: OwnedIterator): TDictionary<string, single>;
        function train_step_multi_inputs_function(data_handler: DataHandler; iterator: OwnedIterator): TDictionary<string, single>;
        /// <summary>
        /// The logic for one training step.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        function  train_step(data_handler : DataHandler; x: TFTensors; y: TFTensors): TDictionary<string, single>;
        procedure _minimize(tape: TGradientTape; optimizer: IOptimizer; loss: TFTensor; trainable_variables: TList<IVariableV1>);
        // MOdel.Summary
        //
        /// <summary>
        /// Prints a string summary of the network.
        /// </summary>
        procedure summary(line_length: Integer = -1; positions: TArray<Single> = []);
        procedure load_weights(filepath: string; by_name: Boolean = false; skip_mismatch : Boolean= false; options: TObject = nil);
        procedure save_weights(filepath: string; overwrite : Boolean= true; save_format: string = ''; options: TObject = nil);
        procedure save(filepath: string; overwrite : Boolean= true; include_optimizer: Boolean = true; save_format: string = 'tf'; {SaveOptions? options = null,} signatures : ConcreteFunction = nil; save_traces : Boolean= true);

        property Layers              : TList<ILayer>      read GetLayers ;
        property TrainableWeights    : TList<IVariableV1> read GetTrainW ;
        property NonTrainableWeights : TList<IVariableV1> read GetNotTrainW ;
        property Metrics             : TList<IMetricFunc> read GetMetrics ;
        property IsGraphNetwork      : Boolean            read Fis_graph_network;
        // callbacks
        property OnEpochBegin      : TCB_On_Epoch_Begin       read Get_OnEpochBegin      write Set_OnEpochBegin;
        property OnEpochEnd        : TCB_On_Epoch_End         read Get_OnEpochEnd        write Set_OnEpochEnd;
        property OnTrainBatchBegin : TCB_On_Train_Batch_Begin read Get_OnTrainBatchBegin write Set_OnTrainBatchBegin;
        property OnTrainBatchEnd   : TCB_On_Train_Batch_End   read Get_OnTrainBatchEnd   write Set_OnTrainBatchEnd;
        property OnEndSummary      : TCB_On_End_Summary       read Get_OnEndSummary      write Set_OnEndSummary;
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
        procedure InitLayers(layers: TList<ILayer>);
      protected
        function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      public
        args : SequentialArgs;

        constructor Create(_args : SequentialArgs);
        destructor  Destroy; override;
        procedure add(tensor: TFTensor); overload;
        procedure add(tensor: TFTensors); overload;
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

              NumPy.NDArray,

              Keras.Utils,
              Keras.Saving,

              TensorFlow.Proto,

              hdf5dll,
              hdf5;

{ Model }

constructor Model.Create(_args: ModelArgs);
begin
    inherited Create(_args);

    FOnEpochBegin      := nil ;
    FOnTrainBatchBegin := nil ;
    _init_batch_counters;
end;

procedure Model.Initialize(_args: LayerArgs);
begin
  _init_batch_counters;
  inherited Initialize(_args);
end;

destructor Model.Destroy;
begin
  FOnEpochBegin := nil;
  FOnTrainBatchBegin := nil;
  FOnEndSummary      := nil;

  compiled_loss.Free;
  compiled_metrics.Free;

  inherited Destroy;
end;

procedure Model.evaluate(x, y: TNDArray; batch_size, verbose, steps, max_queue_size, workers: Integer; use_multiprocessing, return_dict: Boolean);
var
  dataArgs: DataHandlerArgs;

  cCallbacks : CallbackList;
  cbParam    : CallbackParams;
begin
    if x.dims[0] <> y.dims[0] then
        raise Exception.Create('The array x and y should have same value at dim 0, but got '+x.dims[0].ToString+' and '+y.dims[0].ToString);

    dataArgs := DataHandlerArgs.Create;
    dataArgs.X             :=  TFTensors.Create(x);
    dataArgs.Y             :=  TFTensors.Create(y);
    dataArgs.BatchSize     :=  batch_size;
    dataArgs.StepsPerEpoch :=  steps;
    dataArgs.InitialEpoch  :=  0;
    dataArgs.Epochs        :=  1;
    dataArgs.MaxQueueSize  :=  max_queue_size;
    dataArgs.Workers       :=  workers;
    dataArgs.UseMultiprocessing :=  use_multiprocessing;
    dataArgs.Model         :=  Self;
    dataArgs.StepsPerExecution :=  Fsteps_per_execution;

    var data_handler := DataHandler.Create(dataArgs);

    cbParam    := CallbackParams.Create;
    cbParam.mModel  := Self;
    cbParam.Verbose := verbose;
    cbParam.Steps   := data_handler.Inferredsteps;

    cCallbacks := CallbackList.Create(cbParam) ;

    cCallbacks.on_test_begin;
    for var epoch_iterator in data_handler.enumerate_epochs do
    begin
        //var epoch   := epoch_iterator.Value1;
        var iterator:= epoch_iterator.Value2;
        reset_metrics;

        for var step in data_handler.steps do
        begin
            cCallbacks.on_test_batch_begin(step);
            var logs := test_function(data_handler, iterator);

            var end_step := step + data_handler.StepIncrement;
            cCallbacks.on_test_batch_end(end_step, logs)
        end;
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

    var data_handler := DataHandler.Create(dataArgs);

    logs := nil ;
    for var epoch_iterator in data_handler.enumerate_epochs do
    begin
        //var epoch   := epoch_iterator.Value1;
        var iterator:= epoch_iterator.Value2;
        reset_metrics;
        // callbacks.on_epoch_begin(epoch)
        // data_handler.catch_stop_iteration();
        for var step in data_handler.steps do
        begin
            // callbacks.on_train_batch_begin(step)
            logs := test_function(data_handler, iterator);
        end;
    end;
    Result := [];
    for var i := 0 to logs.Count - 1 do
    begin
        var tTensor     : TTensor := logs[i].Value2;
        var floatTensor : Single := Single(tTensor);
        Result := Result + [ TPair<string, Single>.create(logs[i].Value1, floatTensor) ];
    end;
end;

function Model.fit(x, y: TNDArray; batch_size, epochs, verbose: Integer; validation_split: Single; shuffle: Boolean; initial_epoch, max_queue_size, workers: Integer;
  use_multiprocessing: Boolean): ICallback;
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

    var data_handler := DataHandler.Create(dataArgs);

    Result := FitInternal(data_handler, epochs, verbose, nil, train_step_function);
end;

function Model.fit(dataset: IDatasetV2; validation_data : IDatasetV2; batch_size, epochs, verbose: Integer; validation_split: Single; shuffle: Boolean; initial_epoch, max_queue_size, workers: Integer;
  use_multiprocessing: Boolean): History;
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

    var data_handler := DataHandler.Create(dataArgs);

    Result := FitInternal(data_handler, epochs, verbose, validation_data, train_step_function);

end;

function Model.fit(x: TArray<TNDArray>; y: TNDArray; batch_size, epochs, verbose: Integer; validation_split: Single; shuffle: Boolean; initial_epoch, max_queue_size,
  workers: Integer; use_multiprocessing: Boolean): ICallback;
var
   dataArgs        : DataHandlerArgs;
   train_count     : Integer;
   train_x         : TArray<TFTensor>;
   train_y         : NDArray;
begin
    for var tx in x do
    begin
        if tx.dims[0] <> y.dims[0] then
           raise Exception.Create('The array x and y should have same value at dim 0, but got '+ tx.dims[0].ToString+' and '+ y.dims[0].ToString);
    end;
    train_count := trunc(y.dims[0] * (1 - validation_split));

    train_x := [];
    for var tx in x do
    begin
        train_x := train_x + [ tx[ [Slice.Create(0, train_count)] ] ]
    end;
    train_y     := y[ [Slice.Create(0, train_count)] ];

    dataArgs := DataHandlerArgs.Create;

    dataArgs.X             := TFTensors.Create(train_x);
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

    var data_handler := DataHandler.Create(dataArgs);

    if (Length(data_handler.DataAdapter.GetDataset.structure) > 2) or
       (data_handler.DataAdapter.GetDataset.FirstInputTensorCount > 1) then
    begin
        Result := FitInternal(data_handler, epochs, verbose, nil,train_step_multi_inputs_function);
    end else
    begin
        Result := FitInternal(data_handler, epochs, verbose, nil,train_step_function);
    end;
end;

function Model.FitInternal(data_handler : DataHandler; epochs: Integer; verbose: Integer; validation_data: IDatasetV2; train_step_func : TFunc< DataHandler, OwnedIterator, TDictionary<string, single>>): History;
var
  iterator: OwnedIterator;
  epoch   : Integer;
  step    : Integer;

  cCallbacks : CallbackList;
  cbParam    : CallbackParams;
begin
    stop_training := False;
    if      Ftrain_counter is RefVariable          then (Ftrain_counter as RefVariable)         .assign_add(Integer(0))
    else if Ftrain_counter is BaseResourceVariable then (Ftrain_counter as BaseResourceVariable).assign_add(Integer(0))
    else raise Exception.Create('Model.FitInterna Error!');

    cbParam    := CallbackParams.Create;
    cbParam.mModel  := Self;
    cbParam.Verbose := verbose;
    cbParam.Epochs  := epochs;
    cbParam.Steps   := data_handler.Inferredsteps;

    cCallbacks := CallbackList.Create(cbParam);
    cCallbacks.on_train_begin;

    for var it in data_handler.enumerate_epochs do
    begin
        epoch   := it.Value1;
        iterator:= it.Value2;

        reset_metrics;
        cCallbacks.on_epoch_begin(epoch);

        if Assigned(FOnEpochBegin) then
           FOnEpochBegin(cCallbacks.sLog);

        // data_handler.catch_stop_iteration();
        var logs := TDictionary<string, Single>.Create;
        for step in data_handler.steps do
        begin
            cCallbacks.on_train_batch_begin(step);

            if Assigned(FOnTrainBatchBegin) then
               FOnTrainBatchBegin(cCallbacks.sLog) ;

            logs         := train_step_func(data_handler, iterator);
            var end_step := step + data_handler.StepIncrement;
            cCallbacks.on_train_batch_end(end_step, logs);

            if Assigned(FOnTrainBatchEnd) then
               FOnTrainBatchEnd(cCallbacks.sLog);
        end;

        if validation_data <> nil then
        begin
            var val_logs := evaluate(validation_data);
            for var log in val_logs do
            begin
                logs.AddOrSetValue('val_' + log.Key, log.Value);
            end
        end;

        cCallbacks.on_epoch_end(epoch, logs);
        if Assigned(FOnEpochEnd) then
           FOnEpochEnd(cCallbacks.sLog);
    end;
    Result := cCallbacks.hHistory;
end;

function Model.test_function(data_handler : DataHandler; iterator: OwnedIterator): TList<Tuple<string, TFTensor>>;
begin
    var data   := iterator.next;
    var outputs:= test_step(data_handler, data[0], data[1]);
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Ftest_counter is RefVariable          then Result := (Ftest_counter as RefVariable)         .assign_add(Integer(1))
            else if Ftest_counter is BaseResourceVariable then Result := (Ftest_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.test_function Error!');
        end);
    Result := outputs;
end;

function Model.test_step(data_handler : DataHandler; x, y: TFTensor): TList<Tuple<string, TFTensor>>;
begin
    var x_y  := data_handler.DataAdapter.Expand1d(TFTensors.Create(x) , TFTensors.Create(y));
    x := x_y.Value1.First;
    y := x_y.Value2.First;
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

function Model.train_step_multi_inputs_function(data_handler : DataHandler; iterator: OwnedIterator): TDictionary<string, single>;
begin
    var data := iterator.next;
    var eData := Enumerable<TFTensor>.Create(data);
    var x_size := data_handler.DataAdapter.GetDataset.FirstInputTensorCount;

    var outputs := train_step(data_handler, TFTensors.Create(eData.Take(x_size).ToArray), TFTensors.Create(eData.Skip(x_size).ToArray));
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Ftrain_counter is RefVariable          then Result := (Ftrain_counter as RefVariable)         .assign_add(Integer(1))
            else if Ftrain_counter is BaseResourceVariable then Result := (Ftrain_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.train_step_function Error!');
        end);
    Result := outputs;
end;

function Model.train_step_function(data_handler : DataHandler; iterator: OwnedIterator): TDictionary<string, single>;
begin
    var data    := iterator.next();
    var outputs := train_step(data_handler, TFTensors.Create(data[0]), TFTensors.Create(data[1]));
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Ftrain_counter is RefVariable          then Result := (Ftrain_counter as RefVariable)         .assign_add(Integer(1))
            else if Ftrain_counter is BaseResourceVariable then Result := (Ftrain_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.train_step_function Error!');
        end);
    Result := outputs;
end;

function Model.train_step(data_handler : DataHandler; x, y: TFTensors): TDictionary<string, single>;
begin
    var x_y := data_handler.DataAdapter.Expand1d(x, y);
    x := x_y.Value1;
    y := x_y.Value2;

    var tape   := tf.GradientTape;
    var y_pred := Apply(x, nil, true);
    var loss   := compiled_loss.Call(y.First, y_pred.First);

    // For custom training steps, users can just write:
    // trainable_variables = self.trainable_variables
    // gradients = tape.gradient(loss, trainable_variables)
    // self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    // The _minimize call does a few extra steps unnecessary in most cases,
    // such as loss scaling and gradient clipping.
    _minimize(tape, optimizer, loss, TrainableVariables);
    compiled_metrics.update_state(y.First, y_pred.First);

    Result := TDictionary<string, single>.Create;
    for var i := 0 to metrics.Count - 1 do
    begin
       var res : TTensor := metrics[i].R_result;

       if TFTensor(res).ndim > 0 then
           res  := tf.reduce_mean(res);

       var f : Single := Single(res);
       Result.Add(metrics[i].Name, f);
    end;
end;

procedure Model._minimize(tape: TGradientTape; optimizer: IOptimizer; loss: TFTensor; trainable_variables: TList<IVariableV1>);
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

      gradients := optimizer.aggregate_gradients(gradientsAndVariables.ToArray);
      gradients := optimizer.clip_gradients(gradients);

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

function Model.GetMetrics: TList<IMetricFunc>;
begin
    var _metrics := TList<IMetricFunc>.Create;

    if Fis_compiled then
    begin
        if compiled_loss <> nil then
        begin
           for var i := 0 to compiled_loss.metrics.Count -1 do
             _metrics.Add(compiled_loss.metrics[i]);
        end;
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

function Model.GetTrainW: TList<IVariableV1>;
var
  variables : TList<IVariableV1> ;
begin
    variables := TList<IVariableV1>.Create;
    try
      if not Trainable then
      begin
          Exit(variables);
      end;

      for var trackable_obj in Fself_tracked_trackables do
      begin
          if trackable_obj.Trainable then
              variables.AddRange(trackable_obj.TrainableVariables);
      end;
      variables.AddRange( FTrainableWeights );

      Result := TList<IVariableV1>.Create( Enumerable<IVariableV1>.Create(variables.ToArray).Distinct.ToArray );
    finally
      variables.Free;
    end;
end;

function Model.Get_OnEpochBegin: TCB_On_Epoch_Begin;
begin
    Result := FOnEpochBegin;
end;

procedure Model.Set_OnEpochBegin(Value: TCB_On_Epoch_Begin);
begin
    FOnEpochBegin := Value;
end;

function Model.Get_OnEpochEnd: TCB_On_Epoch_End;
begin
     Result := FOnEpochEnd;
end;

procedure Model.Set_OnEpochEnd(Value: TCB_On_Epoch_End);
begin
    FOnEpochEnd := Value;
end;

function Model.Get_OnTrainBatchBegin: TCB_On_Train_Batch_Begin;
begin
    Result := FOnTrainBatchBegin;
end;

procedure Model.Set_OnTrainBatchBegin(Value: TCB_On_Train_Batch_Begin);
begin
    FOnTrainBatchBegin := Value;
end;

function Model.Get_OnTrainBatchEnd: TCB_On_Train_Batch_End;
begin
    Result := FOnTrainBatchEnd
end;

procedure Model.Set_OnTrainBatchEnd(Value: TCB_On_Train_Batch_End);
begin
    FOnTrainBatchEnd := Value;
end;

function Model.Get_OnEndSummary: TCB_On_End_Summary;
begin
    Result := FOnEndSummary
end;

procedure Model.Set_OnEndSummary(Value: TCB_On_End_Summary);
begin
    FOnEndSummary  := Value;
end;

function Model.GetNotTrainW: TList<IVariableV1>;
var
  variables : TList<IVariableV1> ;
begin
    variables := TList<IVariableV1>.Create;
    try
      for var trackable_obj in Fself_tracked_trackables do
          variables.AddRange(trackable_obj.NonTrainableWeights);

      if  not Trainable then
      begin
          var trainable_variables := TList<IVariableV1>.Create;
          for var trackable_obj in Fself_tracked_trackables do
              variables.AddRange(trackable_obj.TrainableWeights);

          variables.AddRange(trainable_variables);
          variables.AddRange(FTrainableWeights);
          variables.AddRange(FNonTrainableWeights);
      end;

      Result := TList<IVariableV1>.Create( Enumerable<IVariableV1>.Create(variables.ToArray).Distinct.ToArray );
    finally
      variables.Free;
    end;
end;

function Model.PredictInternal(data_handler: DataHandler; verbose: Integer): TFTensors;
var
  cCallbacks : CallbackList;
  cbParam    : CallbackParams;
begin
    cbParam    := CallbackParams.Create;
    cbParam.mModel  := Self;
    cbParam.Verbose := verbose;
    cbParam.Epochs  := 1;
    cbParam.Steps   := data_handler.Inferredsteps;

    cCallbacks := CallbackList.Create(cbParam);

    var batch_outputs: TFTensor := nil;

    if      Fpredict_counter is RefVariable          then (Fpredict_counter as RefVariable)         .assign(Integer(0))
    else if Fpredict_counter is BaseResourceVariable then (Fpredict_counter as BaseResourceVariable).assign(Integer(0))
    else raise Exception.Create('Model.PredictInternal Error!');

    cCallbacks.on_predict_begin;
    for  var it in data_handler.enumerate_epochs do
    begin
        // var epoch   := it.Value1;
        var iterator:= it.Value2;

        for var step in data_handler.steps do
        begin
            cCallbacks.on_predict_batch_begin(step);
            var tmp_batch_outputs := run_predict_step(iterator);
            if batch_outputs = nil then
            begin
                batch_outputs := tmp_batch_outputs[0];
            end else
            begin
                batch_outputs := tf.concat([ batch_outputs, tmp_batch_outputs[0] ], 0);
            end;

            var end_step := step + data_handler.StepIncrement;

            var logs := TDictionary<string, TFTensors>.Create;
            logs.Add('outputs', TFTensors.Create(batch_outputs)) ;
            cCallbacks.on_predict_batch_end(end_step, logs);
        end;

    end ;
    Result := TFTensors.Create(batch_outputs);
end;

function Model.predict(dataset: IDatasetV2; batch_size, verbose, steps, max_queue_size, workers: Integer; use_multiprocessing: Boolean): TFTensors;
var
  dataArgs : DataHandlerArgs;
begin
    dataArgs := DataHandlerArgs.Create;

    dataArgs.Dataset       := dataset;
    dataArgs.BatchSize     := batch_size;
    dataArgs.InitialEpoch  := 0;
    dataArgs.Epochs        := 1;
    dataArgs.MaxQueueSize  := max_queue_size;
    dataArgs.Workers       := workers;
    dataArgs.UseMultiprocessing := use_multiprocessing;
    dataArgs.Model         := self;
    dataArgs.StepsPerExecution := Fsteps_per_execution;

    var data_handler := DataHandler.Create(dataArgs);

    Result := PredictInternal(data_handler, verbose);

end;

function Model.predict(x: TFTensors; batch_size, verbose, steps, max_queue_size, workers: Integer; use_multiprocessing: Boolean): TFTensors;
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

    var data_handler := DataHandler.Create(dataArgs);

    Result := PredictInternal(data_handler, verbose);
end;

function Model.run_predict_step(iterator: OwnedIterator): TFTensors;
begin
    var data   := iterator.next;
    var outputs:= predict_step( TFTensors.Create(data) );
    TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([]),
      function(d : TControlDependenciesController ): TFtensor
        begin
            if      Fpredict_counter is RefVariable          then Result := (Fpredict_counter as RefVariable)         .assign_add(Integer(1))
            else if Fpredict_counter is BaseResourceVariable then Result := (Fpredict_counter as BaseResourceVariable).assign_add(Integer(1))
            else raise Exception.Create('Model.run_predict_step Error!');
        end);
    Result := outputs;
end;

procedure Model.load_weights(filepath: string; by_name, skip_mismatch: Boolean; options: TObject);
var
  fileId   : Int64;
  msuccess : Boolean;
  lsuccess : Boolean;
begin
    fileId := THdf5.OpenFile(filepath, true);
    if fileId < 0 then
    begin
        tf.LogMsg('Can''t find weights file :'+ filepath);
        Exit;
    end;
    msuccess := THdf5.GroupExists(fileId, 'model_weights');
    lsuccess := THdf5.GroupExists(fileId, 'layer_names');

    if (not lsuccess) and (msuccess) then
        fileId := THdf5.FH5.H5Gopen2(fileId, 'model_weights', H5P_DEFAULT);

    if by_name then
        //fdf5_format.load_weights_from_hdf5_group_by_name();
        raise Exception.Create('Not ImplementedException')
    else begin
        hdf5_format.load_weights_from_hdf5_group(fileId, Layers);
        THdf5.CloseFile(fileId);
    end;
end;

procedure Model.save(filepath: string; overwrite, include_optimizer: Boolean; save_format: string; signatures: ConcreteFunction; save_traces: Boolean);
begin

end;

procedure Model.save_weights(filepath: string; overwrite: Boolean; save_format: string; options: TObject);
var
  fileId   : Int64;
begin
    fileId := THdf5.CreateFile(filepath);
    hdf5_format.save_weights_to_hdf5_group(fileId, Layers);
    THdf5.CloseFile(fileId);
end;

procedure Model.summary(line_length: Integer; positions: TArray<Single>);
begin
   var strSummary := layer_utils.print_summary(self, line_length, positions);

   if Assigned(FOnEndSummary) then
     FOnEndSummary(strSummary)
end;

function Model.predict_step(data: TFTensors): TFTensors;
begin
    Result := Apply(data, nil, false);
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
    TKerasApi.keras.backend._GRAPH := nil;
end;

procedure Model.Build(input_shape: TFShape);
var
  graph : TFGraph;
begin
    if (self is Functional) or (self is Sequential)  then
    begin
        inherited build(input_shape);
        Exit;
    end;

    if tf.executing_eagerly then graph := TFuncGraph.Create('build_graph')
    else                         graph := TKerasApi.keras.backend.get_graph;

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

procedure Model.compile(_optimizer: IOptimizer; _loss: ILossFunc; metrics: TArray<string>);
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

procedure Model.compile(_optimizer: IOptimizer; _loss: ILossFunc; metrics: TArray<IMetricFunc>);
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

    inherited Destroy;
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
        //if      layer_data.ClassName = 'InputLayer' then layer := InputLayer.from_config(layer_data.Config)
        //else if layer_data.ClassName = 'Dense'      then layer := Dense.from_config(layer_data.Config)
        //else raise Exception.Create('Not Implemented');

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
        for var i := 0 to node.KerasInputs.Count - 1 do
        begin
           var k_tensor := node.KerasInputs[i];
           BuildMapHelper(k_tensor, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices);
        end;
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

    //if Fself_tracked_trackables.Count = 0 then
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

            tf.LogMsg(Format('[Debug] - Depth %s:  %s:  %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name]));

            var isTarainig : Boolean := False;
            if assigned(training) then isTarainig := training^;

            outputs := nNode.Layer.Apply(layer_inputs, nil, isTarainig);
            for var output in outputs do
            begin
                if output <> nil  then
                   tf.LogMsg( Format('[Information] - Depth %s: %s: %s %s',[depth.toString,(nNode.Layer as TObject).ClassName,nNode.Layer.Name, output.shape.ToString]));
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

    // SupportsMasking = true;
    Fcompute_output_and_mask_jointly := true;
    Fauto_track_sub_layers           := false;
    Fhas_explicit_input_shape        := false;
    Fis_graph_network                := false;
    Fcreated_nodes                   := TList<INode>.Create;

    // Add to the model any layers passed to the constructor.
    if args.Layers <> nil then
      InitLayers(args.Layers) ;
end;

destructor Sequential.Destroy;
begin
  args.Free;
  Fcreated_nodes.Free;

  inherited Destroy;
end;

procedure Sequential.InitLayers(layers: TList<ILayer>);
begin
    for var layer in args.Layers do
      add(layer);
end;

procedure Sequential.add(tensor: TFTensor);
begin
    var layer := (tensor.KerasHistory as TKerasHistory).Layer;
    add(layer);
end;

procedure Sequential.add(tensor: TFTensors);
begin
    var t := tensor.First;
    var layer := (t.KerasHistory as TKerasHistory).Layer;
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
                var shp := layer.BatchInputShape;
                // Instantiate an input layer.
                var x := tf.keras.Input(System.default(TFShape), -1, layer.Name + '_input', layer.DType, False, nil, False, nil, @shp );

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
    _layers := inherited GetLayers;

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

    var sName : string;
    if Fself_tracked_trackables[0].Name.EndsWith('_input') then sName :=  Fself_tracked_trackables[0].Name
    else                                                        sName :=  Fself_tracked_trackables[0].Name+'_input';

    var inputs := tf.keras.Input(System.default(TFShape), -1, sName, input_dtype, False, nil, False, nil, @input_shape );

    layer_input  := TFTensors.Create(inputs);
    outputs      := nil;
    created_nodes:= TList<INode>.Create;
    for var layer in args.Layers do
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


