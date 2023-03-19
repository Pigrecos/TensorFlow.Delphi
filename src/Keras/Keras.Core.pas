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

unit Keras.Core;
{$POINTERMATH ON}
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
     uses System.SysUtils,
          System.Generics.Collections,

          Spring,
          Spring.Collections.Enumerable,
          Spring.Container.Common,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          Tensorflow.Initializer,
          TensorFlow.Core,

          TensorFlow.Proto,

          Numpy.Axis;

const null = $11111111;

type
  AutotuneAlgorithm =( HILL_CLIMB = 0, GRADIENT_DESCENT = 1 );
  type DataFormat = ( channels_first = 0, channels_last = 1 );
  TActivation = Reference To function(features: TFTensor; name: string = ''): TFTensor;

  ILayer       = interface;
  IRegularizer = interface;

  NodeConfig = class;
  LayerArgs  = class;
  RegularizerArgs = class;
  Cropping2DArgs = class;

  {$REGION 'interfaces'}
  IInitializersApi = interface
     ['{D6AC2579-8B83-43A6-9C10-1BF9B293B3C4}']

      function Orthogonal(gain: Single = 1.0; seed : pInteger= nil) : IInitializer;
      function he_normal(seed : PInteger= nil) : IInitializer;
  end;

  INode = interface
    ['{F604E1FF-BF7E-4A41-967E-FEB9C4A2B52F}']
    function  GetInputTensor: TFTensors;
    function  GetOutptus: TFTensors;
    function  GetILayer: ILayer;
    function  GetlstLayer:  TList<TFTensor>;
    procedure SetlstLayer(const lst: TList<TFTensor>);
    function  GetINodes: TArray<INode>;
    function  GetInLayers: TArray<ILayer>;
    function  GetIsInput: Boolean;

    function iterate_inbound :  Enumerable< Tuple<ILayer, Integer, Integer, TFTensor> >;
    function serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;

    property input_tensors: TFTensors      read GetInputTensor;
    property Outputs      : TFTensors      read GetOutptus;
    property Layer        : ILayer         read GetILayer;
    property KerasInputs  : TList<TFTensor>read GetlstLayer write SetlstLayer;
    property ParentNodes  : TArray<INode>  read GetINodes;
    property InboundLayers: TArray<ILayer> read GetInLayers;
    property is_input     : Boolean        read GetIsInput;
  end;

  IPoolFunction = interface
    ['{17352CCA-5C47-49AD-9265-C9762F7F3FB6}']

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
  end;

  IAccumulator = interface
    ['{353EB77F-F529-43A0-8667-D513E6B5A45C}']
  end;

  /// <summary>
  /// Functional object that defines a shardable computation.
  /// </summary>
  ICombiner = interface
    ['{0C519749-3A1D-4545-813F-A81FEC5C3B69}']

    procedure Compute(values: TFTensor; accumulator : IAccumulator= nil);
    procedure Merge;
    procedure Extract;
    function  Restore: IAccumulator;
    procedure Serialize;
    procedure Deserialize;
  end;

  ILayer = interface
    ['{8B096EA0-F97B-47DA-BA05-49674F9282C5}']
    function GetName:string;
    function GetTrainable : Boolean;
    function GetBuilt     : Boolean;
    function GetLayers    : TList<ILayer>;
    function GetInNodes   : TList<INode>;
    function GetOutNodes  : TList<INode>;
    function GetTrainVars : TList<IVariableV1>;
    function GetTrainW    : TList<IVariableV1>;
    function GetNotTrainW : TList<IVariableV1>;
    function GetWeights   : TList<IVariableV1>;
    function GetOutShape  : TFShape;
    function GetBatchShape: TFShape;
    function GetDtype     : TF_DataType;

    function Apply(inputs: TFTensors; state: TFTensor = nil; training: Boolean = false): TFTensors;
    function count_params: Integer;
    function get_config: LayerArgs;
    procedure build(input_shape: TFShape);

    property Name                    : string              read GetName;
    property Trainable               : Boolean             read GetTrainable;
    property Built                   : Boolean             read GetBuilt;
    property Layers                  : TList<ILayer>       read GetLayers;
    property InboundNodes            : TList<INode>        read GetInNodes;
    property OutboundNodes           : TList<INode>        read GetOutNodes;
    property TrainableVariables      : TList<IVariableV1>  read GetTrainVars;
    property TrainableWeights        : TList<IVariableV1>  read GetTrainW;
    property NonTrainableWeights     : TList<IVariableV1>  read GetNotTrainW;
    property Weights                 : TList<IVariableV1>  read GetWeights;
    property OutputShape             : TFShape             read GetOutShape;
    property BatchInputShape         : TFShape             read GetBatchShape;
    property DType                   : TF_DataType         read GetDtype;
  end;

  IOptimizer  = interface
      ['{1B287EFE-CFFD-4E65-A439-D8D0E43309E7}']
      function  aggregate_gradients(grads_and_vars : TArray< Tuple<TFTensor, IVariableV1> > ) : TArray<TFTensor>;
      function  clip_gradients(grads: TArray<TFTensor>): TArray<TFTensor>;
      function  apply_gradients(grads_and_vars: Tuple<TFTensor, ResourceVariable>;           name : string= ''; experimental_aggregate_gradients : Boolean = True) : TFOperation; overload;
      function  apply_gradients(grads_and_vars: TArray< Tuple<TFTensor, ResourceVariable> >; name : string= ''; experimental_aggregate_gradients : Boolean = True) : TFOperation; overload;
  end;

  ICallback = interface
    ['{4FCFF764-DBF2-4F28-92E6-14C32F4E1C2D}']
        function GetLog: string;
        function Get_history: TDictionary<string, TList<Single>>;
        procedure Set_history(Value: TDictionary<string, TList<Single>>);

        procedure on_train_begin;
        procedure on_epoch_begin(epoch: Integer);
        procedure on_train_batch_begin(step: Int64);
        procedure on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
        procedure on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
        //
        procedure on_predict_begin;
        procedure on_predict_batch_begin(step: Int64);
        procedure on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
        procedure on_predict_end;
        //
        procedure on_test_begin;
        procedure on_test_batch_begin(step: Int64);
        procedure on_test_batch_end(end_step: Int64; logs : TList<Tuple<string, TFTensor>>);
        //
        property sLog    : string read GetLog;
        property hHistory : TDictionary<string, TList<Single>> read Get_history write Set_history;
  end;

  IMetricFunc = interface
    ['{65939C7A-38CD-4297-900D-6B3D29B07035}']

    /// <summary>
    /// Accumulates metric statistics.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="sample_weight"></param>
    /// <returns></returns>
    function update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor;
    function R_result: TFTensor;
    procedure reset_states;
    function GetName: string;

    property Name : string read GetName;
  end;

  IMetricsApi = interface
   ['{E2E4D504-75DC-4386-8DD4-2763F74B4290}']
      function binary_accuracy(y_true: TFTensor; y_pred: TFTensor): TFTensor;
      function categorical_accuracy(y_true: TFTensor; y_pred: TFTensor): TFTensor;
      function mean_absolute_error(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
      function mean_absolute_percentage_error(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
      /// <summary>
      /// Calculates how often predictions matches integer labels.
      /// </summary>
      /// <param name="y_true">Integer ground truth values.</param>
      /// <param name="y_pred">The prediction values.</param>
      /// <returns>Sparse categorical accuracy values.</returns>
      function sparse_categorical_accuracy(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
      /// <summary>
      /// Computes how often targets are in the top `K` predictions.
      /// </summary>
      /// <param name="y_true"></param>
      /// <param name="y_pred"></param>
      /// <param name="k"></param>
      /// <returns></returns>
      function top_k_categorical_accuracy(y_true: TFTensor; y_pred: TFTensor; k : Integer= 5) : TFTensor;
      /// <summary>
      /// Computes how often targets are in the top K predictions.
      /// </summary>
      /// <param name="k"></param>
      /// <returns></returns>
      function TopKCategoricalAccuracy(k: Integer = 5; name: string = 'top_k_categorical_accuracy'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes the recall of the predictions with respect to the labels.
      /// </summary>
      /// <param name="thresholds"></param>
      /// <param name="top_k"></param>
      /// <param name="class_id"></param>
      /// <param name="name"></param>
      /// <param name="dtype"></param>
      /// <returns></returns>
      function Recall(thresholds: Single = 0.5; top_k: Integer = 0; class_id : Integer = 0; name: string = 'recall'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes the precision of the predictions with respect to the labels.
      /// </summary>
      /// <param name="thresholds"></param>
      /// <param name="top_k"></param>
      /// <param name="class_id"></param>
      /// <param name="name"></param>
      /// <param name="dtype"></param>
      /// <returns></returns>
      function Precision(thresholds: Single = 0.5; top_k: Integer = 0; class_id : Integer= 0; name: string = 'recall'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Calculates how often predictions match binary labels.
      /// </summary>
      /// <returns></returns>
      function BinaryAccuracy(name: string = 'binary_accuracy'; dtype: TF_DataType = TF_FLOAT; threshold : Single= 05): IMetricFunc;
      /// <summary>
      /// Calculates how often predictions match one-hot labels.
      /// </summary>
      /// <returns></returns>
      function CategoricalCrossentropy(name: string = 'categorical_crossentropy'; dtype: TF_DataType = TF_FLOAT; from_logits: Boolean = false; label_smoothing : Single= 0; axis: PAxis = nil): IMetricFunc;
      /// <summary>
      /// Computes the crossentropy metric between the labels and predictions.
      /// </summary>
      /// <returns></returns>
      function CategoricalAccuracy(name: string = 'categorical_accuracy';dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      function categorical_crossentropy(y_true: TFTensor; y_pred: TFTensor; from_logits: Boolean = false; label_smoothing :Single= 0; axis : PAxis= nil): TFTensor;
      /// <summary>
      /// Calculates how often predictions equal labels.
      /// </summary>
      /// <returns></returns>
      function Accuracy(name: string = 'accuracy'; dtype : TF_DataType =TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes the cosine similarity between the labels and predictions.
      /// </summary>
      /// <returns></returns>
      function CosineSimilarity(name: string = 'cosine_similarity'; dtype : TF_DataType = TF_FLOAT; axis : PAxis= nil): IMetricFunc;
      /// <summary>
      /// Computes F-1 Score.
      /// </summary>
      /// <returns></returns>
      function F1Score(num_classes: Integer; average: string = ''; threshold : PSingle= nil; name : string= 'f1_score'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes F-Beta score.
      /// </summary>
      /// <returns></returns>
      function FBetaScore(num_classes: Integer; average: string = ''; beta: Single = 0.1; threshold: PSingle = nil; name: string = 'fbeta_score'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes hamming loss.
      /// </summary>
      /// <param name="mode">multiclass or multilabel</param>
      /// <param name="threshold"></param>
      /// <param name="name"></param>
      /// <param name="dtype"></param>
      /// <returns></returns>
      function HammingLoss(mode: string; threshold: PSingle = nil; name: string = 'hamming_loss'; dtype : TF_DataType=TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes the sparse categorical crossentropy loss.
      /// </summary>
      /// <param name="y_true"></param>
      /// <param name="y_pred"></param>
      /// <param name="from_logits"></param>
      /// <param name="ignore_class"></param>
      /// <param name="axis"></param>
      /// <returns></returns>
      function sparse_categorical_crossentropy(y_true: TFTensor; y_pred: TFTensor; from_logits : Boolean= false; ignore_class : PInteger= nil; axis : PAxis= nil): TFTensor;
      /// <summary>
      /// Computes the crossentropy metric between the labels and predictions.
      /// </summary>
      /// <returns></returns>
      function SparseCategoricalCrossentropy(name: string = 'sparse_categorical_crossentropy'; dtype: TF_DataType = TF_FLOAT; from_logits: Boolean = false; ignore_class: PInteger = nil; axis: PAxis = nil): IMetricFunc;
      /// <summary>
      /// Calculates how often predictions match integer labels.
      /// </summary>
      /// <returns></returns>
      function SparseCategoricalAccuracy(name: string = 'sparse_categorical_accuracy'; dtype : TF_DataType= TF_FLOAT): IMetricFunc;
      /// <summary>
      /// Computes how often integer targets are in the top K predictions.
      /// </summary>
      /// <param name="k"></param>
      /// <returns></returns>
      function SparseTopKCategoricalAccuracy(k: Integer = 5; name: string = 'sparse_top_k_categorical_accuracy'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
  end;

  ILossFunc = interface
   ['{AE7EE5F7-1243-45C6-86B1-C43DB3146C9F}']
     //function GetReduction : string;
     //function GetName : string;

     function Call(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor;

    // property Reduction : string read GetReduction;
    // property Name      : string read GetName;
  end;

  ILossesApi = interface
  ['{67692E83-81E1-4498-9065-154223708B2E}']
       function  BinaryCrossentropy(from_logits : Boolean= false; label_smoothing: Single = 0; axis: Integer = -1; reduction : string= 'auto'; name: string = 'binary_crossentropy'): ILossFunc;
       function  SparseCategoricalCrossentropy(reduction : string= ''; name : string= '';from_logits: Boolean = false) : ILossFunc;
       function  CategoricalCrossentropy(reduction : string= ''; name : string= ''; from_logits: Boolean = false): ILossFunc;
       function  MeanSquaredError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanSquaredLogarithmicError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanAbsolutePercentageError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanAbsoluteError(reduction : string= ''; name : string= '') : ILossFunc;
       function  CosineSimilarity(reduction : string= ''; name : string= '';axis: Integer=-1): ILossFunc;
       function  Huber(reduction : string= ''; name : string= ''; delta: TFTensor=nil): ILossFunc;
       function  LogCosh(reduction : string= ''; name : string= ''): ILossFunc;
       /// <summary>
       /// Implements the focal loss function.
       /// </summary>
       /// <param name="from_logits"></param>
       /// <param name="alpha"></param>
       /// <param name="gamma"></param>
       /// <param name="reduction"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       function  SigmoidFocalCrossEntropy(from_logits: Boolean = false; alpha: Single = 0.25; gamma: Single = 2.0; reduction: string = 'none'; name: string = 'sigmoid_focal_crossentropy'): ILossFunc;
   end;

  IPreprocessing   = interface
  ['{E1668123-6E93-45AB-9E71-7FEFECD54A05}']
      function Resizing(height: Integer; width: Integer; interpolation: string = 'bilinear'): ILayer;
      function TextVectorization(standardize: TFunc<TFTensor, TFTensor> = nil; split : string= 'whitespace'; max_tokens: Integer = -1; output_mode: string = 'int'; output_sequence_length: Integer = -1): ILayer;
  end;

  ILayersApi = interface
  ['{BE5055DF-9B19-4B7C-AA61-5CC8FE742744}']
     function  ReadProc : IPreprocessing;

     function Add: ILayer;

     function AveragePooling2D(pool_size: PTFShape = nil; strides: PTFShape = nil; padding: string = 'valid'; data_format: string = ''): ILayer; overload;
     function AveragePooling2D(pool_size: TFShape ; strides: TFShape ;             padding: string = 'valid'; data_format: string = ''): ILayer; overload;

     function BatchNormalization(axis                       : Integer = -1;
                                momentum                    : Single = 0.99;
                                epsilon                     : Single = 0.001;
                                center                      : Boolean= true;
                                scale                       : Boolean= true;
                                beta_initializer            : IInitializer= nil;
                                gamma_initializer           : IInitializer= nil;
                                moving_mean_initializer     : IInitializer = nil;
                                moving_variance_initializer : IInitializer= nil;
                                trainable                   : Boolean = true;
                                name                        : string= '';
                                renorm                      : Boolean= false;
                                renorm_momentum             : Single= 0.99): ILayer;

     /// <summary>
     /// A preprocessing layer which encodes integer features.
     /// </summary>
     /// <param name="num_tokens">The total number of tokens the layer should support.</param>
     /// <param name="output_mode">Specification for the output of the layer.</param>
     /// <returns></returns>
     function CategoryEncoding(num_tokens: Integer; output_mode : string= 'one_hot'; sparse: Boolean = false; count_weights : TNDArray= nil): ILayer;

     function Conv1D(filters           : Integer;
                    kernel_size        : TFShape;
                    strides            : Integer= 1;
                    padding            : string= 'valid';
                    data_format        : string= 'channels_last';
                    dilation_rate      : Integer = 1;
                    groups             : Integer= 1;
                    activation         : string= '';
                    use_bias           : Boolean= true;
                    kernel_initializer : string = 'glorot_uniform';
                    bias_initializer   : string= 'zeros'): ILayer; overload;

     function Conv1D(filters           : Integer;
                    kernel_size        : TFShape;
                    activation         : string): ILayer; overload;

     function Conv2D(filters             : Integer;
                    kernel_size          : PTFShape= nil;
                    strides              : PTFShape= nil;
                    padding              : string = 'valid';
                    data_format          : string = '';
                    dilation_rate        : PTFShape= nil;
                    groups               : Integer= 1;
                    activation           : TActivation= nil;
                    use_bias             : Boolean= true;
                    kernel_initializer   : IInitializer = nil;
                    bias_initializer     : IInitializer = nil;
                    kernel_regularizer   : IRegularizer = nil;
                    bias_regularizer     : IRegularizer= nil;
                    activity_regularizer : IRegularizer= nil): ILayer; overload;

     function Conv2D(filters          : Integer;
                    kernel_size       : PTFShape= nil;
                    strides           : PTFShape= nil;
                    padding           : string = 'valid';
                    data_format       : string= '';
                    dilation_rate     : PTFShape = nil;
                    groups            : Integer = 1;
                    activation        : string = '';
                    use_bias          : Boolean= true;
                    kernel_initializer: string = 'glorot_uniform';
                    bias_initializer  : string = 'zeros'): ILayer; overload;

     function Conv2D(filters           : Integer;
                     kernel_size       : TFShape;
                     activation        : string;
                     padding           : string): ILayer; overload;

     function Conv2DTranspose(filters                : Integer;
                                kernel_size          : PTFShape= nil;
                                strides              : PTFShape= nil;
                                output_padding       : string = 'valid';
                                data_format          : string = '';
                                dilation_rate        : PTFShape = nil;
                                activation           : string = '';
                                use_bias             : Boolean= true;
                                kernel_initializer   : string = '';
                                bias_initializer     : string = '';
                                kernel_regularizer   : string = '';
                                bias_regularizer     : string = '';
                                activity_regularizer : string = ''): ILayer; overload;

     function Conv2DTranspose(filters                : Integer;
                                kernel_size          : TFShape;
                                strides              : TFShape;
                                output_padding       : string = 'valid';
                                data_format          : string = '';
                                dilation_rate        : PTFShape = nil;
                                activation           : string = '';
                                use_bias             : Boolean= true;
                                kernel_initializer   : string = '';
                                bias_initializer     : string = '';
                                kernel_regularizer   : string = '';
                                bias_regularizer     : string = '';
                                activity_regularizer : string = ''): ILayer; overload;

     function Dense(units: Integer): ILayer; overload;

     function Dense(units: Integer; activation: string; input_shape: PTFShape = nil): ILayer; overload;

     function Dense(units             : Integer;
                    activation        : TActivation;
                    kernel_initializer: IInitializer = nil;
                    use_bias          : Boolean= true;
                    bias_initializer  : IInitializer = nil;
                    input_shape       : PTFShape= nil): ILayer; overload;

     function Dropout(rate: Single; noise_shape: PTFShape = nil; seed: pInteger= nil): ILayer;

     function Embedding(input_dim              : Integer;
                        output_dim             : Integer;
                        embeddings_initializer : IInitializer= nil;
                        mask_zero              : Boolean= false;
                        input_shape            : PTFShape= nil;
                        input_length           : Integer= -1): ILayer;

     function EinsumDense(equation            : string;
                          output_shape        : TFShape;
                          bias_axes           : string;
                          activation          : TActivation= nil;
                          kernel_initializer  : IInitializer= nil;
                          bias_initializer    : IInitializer= nil;
                          kernel_regularizer  : IRegularizer= nil;
                          bias_regularizer    : IRegularizer= nil;
                          activity_regularizer: IRegularizer = nil;
                          kernel_constraint   : TProc= nil;
                          bias_constraint     : TProc= nil): ILayer;

      function Flatten(data_format: string = ''): ILayer;
      function GlobalAveragePooling1D(data_format : string= 'channels_last'): ILayer;
      function GlobalAveragePooling2D: ILayer; overload;
      function GlobalAveragePooling2D(data_format: string = 'channels_last'): ILayer; overload;
      function GlobalMaxPooling1D(data_format: string = 'channels_last'): ILayer;
      function GlobalMaxPooling2D(data_format: string = 'channels_last'): ILayer;

      function Input(shape      : TFShape;
                     batch_size : Integer = -1;
                     name       : string = '';
                     dtype      : TF_DataType = DtInvalid;
                     sparse     : Boolean = false;
                     tensor     : TFTensor = nil;
                     ragged     : Boolean= false;
                     type_spec  : TypeSpec= nil;
                     batch_input_shape: PTFShape= nil;
                     batch_shape: PTFShape= nil): TFTensors;

      function InputLayer(input_shape: TFShape;
                          name       : string= '';
                          sparse     : Boolean= false;
                          ragged     : Boolean= false): ILayer;

      function LayerNormalization(axis             : TAxis;
                                  epsilon          : Single= 1e-3;
                                  center           : Boolean= true;
                                  scale            : Boolean= true;
                                  beta_initializer : IInitializer= nil;
                                  gamma_initializer: IInitializer= nil): ILayer;

      function LeakyReLU(alpha: Single = 0.3): ILayer;

      function LSTM(units                : Integer;
                    activation           : TActivation= nil;
                    recurrent_activation : TActivation= nil;
                    use_bias             : Boolean= true;
                    kernel_initializer   : IInitializer= nil;
                    recurrent_initializer: IInitializer= nil;
                    bias_initializer     : IInitializer= nil;
                    unit_forget_bias     : Boolean= true;
                    dropout              : Single= 0;
                    recurrent_dropout    : Single= 0;
                    _implementation      : Integer=2;
                    return_sequences     : Boolean= false;
                    return_state         : Boolean= false;
                    go_backwards         : Boolean= false;
                    stateful             : Boolean= false;
                    time_major           : Boolean= false;
                    unroll               : Boolean= false): ILayer;

      function MaxPooling1D(pool_size : PInteger= nil; strides : PInteger= nil; padding : string= 'valid'; data_format: string = ''): ILayer;
      function MaxPooling2D(pool_size : PTFShape= nil; strides : PTFShape= nil; padding : string= 'valid'; data_format: string = ''): ILayer; overload;
      function MaxPooling2D(pool_size : TFShape;       strides : TFShape;       padding : string= 'valid'; data_format: string = ''): ILayer; overload;

      function Permute(dims: TArray<Integer>): ILayer;

      function Rescaling(scale: Single; offset : Single= 0; input_shape : PTFShape= nil): ILayer;

      function SimpleRNN( units                : Integer;
                          activation           : string  = 'tanh';
                          kernel_initializer   : string  = 'glorot_uniform';
                          recurrent_initializer: string  = 'orthogonal';
                          bias_initializer     : string  = 'zeros';
                          return_sequences     : Boolean = False;
                          return_state         : Boolean = False ): ILayer;

      function Subtract: ILayer;

      // ILayerApi.Activation
      //
      function ELU(alpha: Single = 0.1): ILayer;
      function SELU: ILayer;
      function Softmax(axis: TAxis): ILayer;
      function Softplus: ILayer;
      function HardSigmoid: ILayer;
      function Softsign: ILayer;
      function Swish: ILayer;
      function Tanh: ILayer;
      function Exponential: ILayer;

      // ILayerApi.Activation
      //
      function Attention(use_scale :  Boolean= false; score_mode: string = 'dot'; causal: Boolean = false; dropout : Single= 0): ILayer;

      function MultiHeadAttention(num_heads           : Integer;
                                  key_dim             : Integer;
                                  value_dim           : PInteger= nil;
                                  dropout             : Single= 0;
                                  use_bias            : Boolean= true;
                                  output_shape        : PTFShape= nil;
                                  attention_axes      : PTFShape= nil;
                                  kernel_initializer  : IInitializer= nil;
                                  bias_initializer    : IInitializer= nil;
                                  kernel_regularizer  : IRegularizer= nil;
                                  bias_regularizer    : IRegularizer= nil;
                                  activity_regularizer: IRegularizer= nil;
                                  kernel_constraint   : TProc = nil;
                                  bias_constraint     : TProc = nil): ILayer;

      // ILayerApi.Cropping
      //
      function Cropping1D(cropping: TNDArray): ILayer;
      function Cropping2D(cropping: TNDArray; data_format : DataFormat = DataFormat.channels_last): ILayer;
      function Cropping3D(cropping: TNDArray; data_format : DataFormat = DataFormat.channels_last): ILayer;

      // ILayerApi.Merging
      //
      function Concatenate(axis: Integer = -1): ILayer;

      // ILayerApi.Reshaping
      //
      function Reshape(target_shape: TFShape): ILayer; overload;
      function Reshape(target_shape: TArray<TValue>): ILayer; overload;
      function UpSampling2D(size : PTFShape= nil; data_format : string = ''; interpolation: string = 'nearest'): ILayer;
      function ZeroPadding2D(padding: TNDArray): ILayer;

      property preprocessing : IPreprocessing read ReadProc;
  end;

  IRegularizer = interface
    ['{F08A34E9-848E-4BF7-915F-6216D1C4E078}']

    function Apply(args: RegularizerArgs): TFTensor;
  end;

  TCB_On_Epoch_Begin       = reference to procedure(msg: string);
  TCB_On_Epoch_End         = reference to procedure(msg: string);
  TCB_On_Train_Batch_Begin = reference to procedure(msg: string);
  TCB_On_Train_Batch_End   = reference to procedure(msg: string);
  TCB_On_End_Summary       = reference to procedure(msg: string);

  IModel = interface(ILayer)
    ['{2B165250-B224-4DF2-B0B1-D1577D054A70}']

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

    procedure compile(_optimizer : IOptimizer= nil; _loss: ILossFunc = nil; metrics : TArray<string>= nil); overload;
    procedure compile(_optimizer : string;          _loss: string;          metrics: TArray<string>); overload;

    function fit( x: TNDArray; y      : TNDArray;
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
    procedure load_weights(filepath: string; by_name: Boolean = false; skip_mismatch : Boolean= false; options: TObject = nil);
    procedure save_weights(filepath: string; overwrite : Boolean= true; save_format: string = ''; options: TObject = nil);
    procedure save(filepath: string; overwrite : Boolean= true; include_optimizer: Boolean = true; save_format: string = 'tf'; {SaveOptions? options = null,} signatures : ConcreteFunction = nil; save_traces : Boolean= true);
    procedure evaluate( x  : TNDArray; y : TNDArray;
                        batch_size          : Integer= -1;
                        verbose             : Integer = 1;
                        steps               : Integer = -1;
                        max_queue_size      : Integer= 10;
                        workers             : Integer= 1;
                        use_multiprocessing : Boolean = false;
                        return_dict         : Boolean= false);
    function predict(x                   : TFTensors;
                     batch_size          : Integer= -1;
                     verbose             : Integer = 0;
                     steps               : Integer = -1;
                     max_queue_size      : Integer = 10;
                     workers             : Integer = 1;
                     use_multiprocessing : Boolean = false): TFTensors;
    procedure summary(line_length: Integer = -1; positions: TArray<Single> = []);

    // callbacks
    property OnEpochBegin      : TCB_On_Epoch_Begin       read Get_OnEpochBegin      write Set_OnEpochBegin;
    property OnEpochEnd        : TCB_On_Epoch_End         read Get_OnEpochEnd        write Set_OnEpochEnd;
    property OnTrainBatchBegin : TCB_On_Train_Batch_Begin read Get_OnTrainBatchBegin write Set_OnTrainBatchBegin;
    property OnTrainBatchEnd   : TCB_On_Train_Batch_End   read Get_OnTrainBatchEnd   write Set_OnTrainBatchEnd;
    property OnEndSummary      : TCB_On_End_Summary       read Get_OnEndSummary      write Set_OnEndSummary;
  end;
  {$ENDREGION}

  RegularizerArgs  = class
    public
      X : TFTensor;

     constructor Create(t: TFTensor);
  end;

  LayerArgs = class
     public
        /// <summary>
        /// Indicates whether the layer's weights are updated during training
        /// and whether the layer's updates are run during training.
        /// </summary>
        Trainable : Boolean;
        Name      : string;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        DType     : TF_DataType;
        /// <summary>
        /// Whether the `call` method can be used to build a TF graph without issues.
        /// This attribute has no effect if the model is created using the Functional
        /// API. Instead, `model.dynamic` is determined based on the internal layers.
        /// </summary>
        bDynamic  : boolean;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        InputShape : TFShape;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        BatchInputShape : TFShape;
        BatchSize       : Integer;
        /// <summary>
        /// Initial weight values.
        /// </summary>
        Weights         : TArray<Single>;
        /// <summary>
        /// Regularizer function applied to the output of the layer(its "activation").
        /// </summary>
        ActivityRegularizer : IRegularizer;
        Autocast            : Boolean;
        IsFromConfig        : Boolean;
        constructor Create;
  end;

  NodeConfig = class
     public
        Name        : string;
        NodeIndex   : integer;
        TensorIndex : Integer;

        function ToString: string; override;
  end;

  LayerConfig = class
     public
        Name        : string;
        ClassName   : string;
        Config      : LayerArgs;
        InboundNodes: TList<NodeConfig>;

        constructor Create;
  end; 
  
  NodeArgs = class
    InboundLayers : TArray<ILayer>;
    NodeIndices   : TArray<Integer>;
    TensorIndices : TArray<Integer>;
    InputTensors  : TFTensors;
    Outputs       : TFTensors;
  end; 
  
  /// <summary>
  /// A `Node` describes the connectivity between two layers.
  ///
  /// Each time a layer is connected to some new input,
  /// a node is added to `layer._inbound_nodes`.
  /// Each time the output of a layer is used by another layer,
  /// a node is added to `layer._outbound_nodes`.
  /// </summary>
  Node = class(TInterfacedObject,INode)
    private
       Fargs        : NodeArgs;
       Flayer       : ILayer;
       FKerasInputs :  TList<TFTensor>;

       function  GetInputTensor: TFTensors;
       function  GetOutptus: TFTensors;
       function  GetlstLayer:  TList<TFTensor>;
       procedure SetlstLayer(const lst: TList<TFTensor>);
       function  GetILayer: ILayer;
       function  GetIsInput: Boolean;
       function  GetINodes: TArray<INode>;
       function  GetInLayers: TArray<ILayer>;
    protected

    public
     constructor Create(_args:NodeArgs);
     procedure  Connect(_layer: ILayer);
     /// <summary>
     /// Maps Keras Tensors to computed Tensors using `tensor_dict`.
     /// </summary>
     /// <param name="tensor_dict"></param>
     /// <returns></returns>
     function MapArguments(tensor_dict: TDictionary< Int64, TQueue<TFTensor> >): TFTensors;
     function iterate_inbound :  Enumerable< Tuple<ILayer, Integer, Integer, TFTensor> >;
     function serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;


      property input_tensors: TFTensors      read GetInputTensor;
      property Outputs      : TFTensors      read GetOutptus;
      property KerasInputs  : TList<TFTensor>read GetlstLayer write SetlstLayer;
      property Layer        : ILayer         read GetILayer;
      property is_input     : Boolean        read GetIsInput;
      property ParentNodes  : TArray<INode>  read GetINodes;
      property InboundLayers: TArray<ILayer> read GetInLayers;
  end; 

  CallContextManager = class(TInterfacedObject, IDisposable)
     private
        Fbuild_graph : Boolean;
     protected
        procedure Dispose;
     public
        constructor Create(build_graph: Boolean);
  end;

  TCallContext = class(CallContextManager)
     constructor Create;
     function Enter(build_graph: Boolean): CallContextManager;
  end;

  /// <summary>
  /// Specifies the ndim, dtype and shape of every input to a layer.
  /// </summary>
  TInputSpec = class
     private
        FnDim      : Nullable<Integer>;
        Fmin_ndim  : Nullable<Integer>;
        Faxes      : TDictionary<Integer,Integer>;
        FAllAxisDim: TArray<Integer>;
        Fshape     : TFShape;
     public
        constructor Create(dtype: TF_DataType = DtInvalid; _ndim : Integer= null; _min_ndim: Integer = null; _axes: TDictionary<Integer,Integer> = nil; shape : PTFShape= nil);
        destructor Destroy; override;

        property nDim       : Nullable<Integer> read FnDim;
        property min_ndim   : Nullable<Integer> read Fmin_ndim;
        property axes       : TDictionary<Integer,Integer> read Faxes;
        property AllAxisDim : TArray<Integer>   read FAllAxisDim;
  end;

  /// <summary>
  /// Tracks the Layer call that created a Tensor, for Keras Graph Networks.
  /// </summary>
  TKerasHistory = class(TObject)
    private
      FLayer        : ILayer;
      Fnode_index   : Integer;
      Ftensor_index : Integer;
    public
      constructor Create(_layer: ILayer; _node_index: Integer; _tensor_index: Integer);
      function ToTuple: Tuple<ILayer,Integer,Integer>;

      property Layer        : ILayer  read FLayer;
      property node_index   : Integer read Fnode_index;
      property tensor_index : Integer read Ftensor_index;
  end;

  IActivationsApi = interface
    ['{1C028BD3-46A1-494B-9B50-9387E8AAFF71}']
     function Get_Linear : TActivation;
     function Get_Relu   : TActivation;
     function Get_Sigmoid: TActivation;
     function Get_Softmax: TActivation;
     function Get_Tanh   : TActivation;
     function Get_mish   : TActivation;

     property Linear : TActivation read Get_Linear;
     property Relu   : TActivation read Get_Relu;
     property Sigmoid: TActivation read Get_Sigmoid;
     property Softmax: TActivation read Get_Softmax;
     property Tanh   : TActivation read Get_Tanh;
     property mish   : TActivation read Get_mish;
  end;

  // Keras.Activations
  //
  TActivations = class(TInterfacedObject, IActivationsApi)
    private
       FLinear : TActivation;
       FRelu   : TActivation;
       FSigmoid: TActivation;
       FSoftmax: TActivation;
       FTanh   : TActivation;
       Fmish   : TActivation;

       function Get_Linear : TActivation;
       function Get_Relu   : TActivation;
       function Get_Sigmoid: TActivation;
       function Get_Softmax: TActivation;
       function Get_Tanh   : TActivation;
       function Get_mish   : TActivation;
    public
       constructor Create;
       destructor  Destroy; override;
   end; 

   {$REGION 'Keras.ArgsDefinitions'} 
   // Keras.ArgsDefinitions
   //   
   ModelArgs = class(LayerArgs)
   	public
   	Inputs  : TFTensors;
   	Outputs : TFTensors;
   
   	constructor Create;
   end;
   
   SequentialArgs = class(ModelArgs)
   	public
   	Layers : TList<ILayer>;
   
   	constructor Create;
   end;
   
   CategoryEncodingArgs = class(LayerArgs)
   	private
   
   	public
   	NumTokens     : Integer;
   		OutputMode   : string;
   		Sparse       : boolean ;
   		CountWeights : TNDArray;
   end;
   
   ConvolutionalArgs = class(LayerArgs)
   	private
   
   	public
   	Rank           : Integer;
   	Filters        : Integer;
   	NumSpatialDims : Integer;
   	KernelSize     : TFShape;
   	/// <summary>
   	/// specifying the stride length of the convolution.
   	/// </summary>
   	Strides            : TFShape;
   	Padding            : string;
   	DataFormat         : string;
   	DilationRate       : TFShape;
   	Groups             : Integer;
   	Activation         : TActivation;
   	UseBias            : Boolean;
   	KernelInitializer  : IInitializer ;
   	BiasInitializer    : IInitializer;
   	KernelRegularizer  : IRegularizer;
   	BiasRegularizer    : IRegularizer;
   	KernelConstraint   : TProc;
   	BiasConstraint     : TProc;
   
   	constructor Create;
   end;
   
   Conv1DArgs = class(ConvolutionalArgs)
   	public
   	constructor Create;
   end;
   
   Conv2DArgs = class(ConvolutionalArgs)
   	public
   	constructor Create;
   end;
   
   RNNArgs = class(LayerArgs)
   	public
   		type
   		IRnnArgCell = interface(ILayer)
   			function GetState_size : TValue;
   
   			property state_size : TValue read GetState_size;
   		end;
   	public
   		Cell            : IRnnArgCell;
   		ReturnSequences : Boolean;
   		ReturnState     : Boolean;
   		GoBackwards     : Boolean;
   		Stateful        : Boolean;
   		Unroll          : Boolean;
   		TimeMajor       : Boolean;
   		Kwargs          : TDictionary<string,TValue>;
   
   		Units               : Integer;
   		Activation          : TActivation;
   		RecurrentActivation : TActivation;
   		UseBias             : boolean;
   		KernelInitializer   : IInitializer;
   		RecurrentInitializer: IInitializer;
   		BiasInitializer     : IInitializer;
   
   		Constructor Create;
   end;
   
   SimpleRNNArgs = class(RNNArgs)
   	public
   		Constructor Create;
   end;
   
   OptimizerV2Args = class
   	public
   		Name         : string;
   		LearningRate : Single;
   		InitialDecay : Single;
   		ClipNorm     : Single;
   		ClipValue    : Single;
   
   		Constructor Create;
   end;
   
   RMSpropArgs = class(OptimizerV2Args)
   	public
   		RHO      : Single;
   		Momentum : Single;
   		Epsilon  : Single;
   		Centered : Boolean;
   
   		Constructor Create;
   end;
   
   ELUArgs = class(LayerArgs)
   	public
   	Alpha   : Single ;
   	constructor Create;
   end;
   
   LeakyReLuArgs = class(LayerArgs)
   	public
   	Alpha   : Single ;
   	constructor Create;
   end;
   
   SoftmaxArgs = class(LayerArgs)
   	public
   	axis   : TAxis ;
   	constructor Create;
   end;
   
   BaseDenseAttentionArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// Boolean. Set to `true` for decoder self-attention. Adds a mask such
   	/// that position `i` cannot attend to positions `j > i`. This prevents the
   	/// flow of information from the future towards the past.
   	/// </summary>
   	causal : boolean ;
   
   	/// <summary>
   	/// Float between 0 and 1. Fraction of the units to drop for the
   	/// attention scores.
   	/// </summary>
   	dropout : Single;
   	constructor Create;
   end;
   
   AttentionArgs = class(BaseDenseAttentionArgs)
   	public
   	/// <summary>
   	/// If `true`, will create a scalar variable to scale the attention scores.
   	/// </summary>
   	use_scale : Boolean;
   
   	/// <summary>
   	/// Function to use to compute attention scores, one of
   	/// `{"dot", "concat"}`. `"dot"` refers to the dot product between the query
   	/// and key vectors. `"concat"` refers to the hyperbolic tangent of the
   	/// concatenation of the query and key vectors.
   	/// </summary>
   	score_mode : string ;
   	constructor Create;
   end;
   
   MultiHeadAttentionArgs = class(LayerArgs)
   	public
   	NumHeads          : Integer;
   	KeyDim            : Integer;
   	ValueDim          : Nullable<Integer>;
   	Dropout           : Single;
   	UseBias           : Boolean;
   	OutputShape       : TFShape;
   	AttentionAxis     : TFShape ;
   	KernelInitializer : IInitializer;
   	BiasInitializer   : IInitializer ;
   	KernelRegularizer : IRegularizer;
   	BiasRegularizer   : IRegularizer;
   	KernelConstraint  : TProc;
   	BiasConstraint    : TProc;
   
   	constructor Create;
   end;
   
   DropoutArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// Float between 0 and 1. Fraction of the input units to drop.
   	/// </summary>
   	Rate : Single;
   
   	/// <summary>
   	/// 1D integer tensor representing the shape of the
   	/// binary dropout mask that will be multiplied with the input.
   	/// </summary>
   	NoiseShape : TFShape;
   
   	/// <summary>
   	/// random seed.
   	/// </summary>
   	Seed  : Nullable<Integer>;
   
   	SupportsMasking : Boolean;
   
   	constructor Create;
   end;
   
   DenseArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// Positive integer, dimensionality of the output space.
   	/// </summary>
   	Units : Integer;
   
   	/// <summary>
   	/// Activation function to use.
   	/// </summary>
   	Activation : TActivation;
   
   	/// <summary>
   	/// Whether the layer uses a bias vector.
   	/// </summary>
   	UseBias : Boolean;
   
   	/// <summary>
   	/// Initializer for the `kernel` weights matrix.
   	/// </summary>
   	KernelInitializer : IInitializer;
   
   	/// <summary>
   	/// Initializer for the bias vector.
   	/// </summary>
   	BiasInitializer : IInitializer;
   
   	/// <summary>
   	/// Regularizer function applied to the `kernel` weights matrix.
   	/// </summary>
   	KernelRegularizer : IRegularizer;
   
   	/// <summary>
   	/// Regularizer function applied to the bias vector.
   	/// </summary>
   	BiasRegularizer : IRegularizer;
   
   	/// <summary>
   	/// Constraint function applied to the `kernel` weights matrix.
   	/// </summary>
   	KernelConstraint : TProc;
   
   	/// <summary>
   	/// Constraint function applied to the bias vector.
   	/// </summary>
   	BiasConstraint : TProc;
   
   	constructor Create;
   end;
   
   EinsumDenseArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// An equation describing the einsum to perform. This equation must
   	/// be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
   	/// `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
   	/// expression sequence.
   	/// </summary>
   	Equation : string;
   
   	/// <summary>
   	/// The expected shape of the output tensor (excluding the batch
   	/// dimension and any dimensions represented by ellipses). You can specify
   	/// None for any dimension that is unknown or can be inferred from the input
   	/// shape.
   	/// </summary>
   	OutputShape : TFShape;
   
   	/// <summary>
   	/// A string containing the output dimension(s) to apply a bias to.
   	/// Each character in the `bias_axes` string should correspond to a character
   	/// in the output portion of the `equation` string.
   	/// </summary>
   	BiasAxes : string;
   
   	/// <summary>
   	/// Activation function to use.
   	/// </summary>
   	Activation : TActivation;
   
   	/// <summary>
   	/// Initializer for the `kernel` weights matrix.
   	/// </summary>
   	KernelInitializer : IInitializer;
   
   	/// <summary>
   	/// Initializer for the bias vector.
   	/// </summary>
   	BiasInitializer : IInitializer;
   
   	/// <summary>
   	/// Regularizer function applied to the `kernel` weights matrix.
   	/// </summary>
   	KernelRegularizer : IRegularizer;
   
   	/// <summary>
   	/// Regularizer function applied to the bias vector.
   	/// </summary>
   	BiasRegularizer : IRegularizer;
   
   	/// <summary>
   	/// Constraint function applied to the `kernel` weights matrix.
   	/// </summary>
   	KernelConstraint : TProc;
   
   	/// <summary>
   	/// Constraint function applied to the bias vector.
   	/// </summary>
   	BiasConstraint : TProc;
   
   	constructor Create;
   end;
   
   EmbeddingArgs = class(LayerArgs)
   	public
   	InputDim    : Integer;
   	OutputDim   : Integer;
   	MaskZero    : Boolean;
   	InputLength : Integer;
   	EmbeddingsInitializer : IInitializer;
   
   	constructor Create;
   end;
   
   InputLayerArgs = class(LayerArgs)
   	public
   	InputTensor : TFTensor;
   	Sparse      : Boolean;
   	Ragged      : Boolean;
   
   	constructor Create;
   end;
   
   CroppingArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// Accept length 1 or 2
   	/// </summary>
   	cropping : TNDArray;
   
   	constructor Create;
   end;
   
   Cropping2DArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// channel last: (b, h, w, c)
   	/// channels_first: (b, c, h, w)
   	/// </summary>
    public
   	/// <summary>
   	/// Accept: int[1][2], int[1][1], int[2][2]
   	/// </summary>
   	cropping    : TNDarray;
   	data_format : DataFormat;
   
   	Constructor Create;
   end;
   
   Cropping3DArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// channel last: (b, h, w, c)
   	/// channels_first: (b, c, h, w)
   	/// </summary>
   	public
   	/// <summary>
   	/// Accept: int[1][3], int[1][1], int[3][2]
   	/// </summary>
   	cropping    : TNDarray;
   	data_format : DataFormat;
   
   	Constructor Create;
   end;
   
   LSTMArgs = class(RNNArgs)
   	public
   	UnitForgetBias  : Boolean;
   	Dropout         : Single;
   	RecurrentDropout: Single;
   	&Implementation : Integer;
   
   	Constructor Create;
   end;
   
   LSTMCellArgs = class(LayerArgs)
   	public
   	Constructor Create;
   end;
   
   MergeArgs = class(LayerArgs)
   	public
   	Inputs : TFTensors;
   	Axis   : Integer;
   
   	Constructor Create;
   end;
   
   LayerNormalizationArgs = class(LayerArgs)
   	public
   	Axis            : TAxis;
   	Epsilon         : Single;
   	Center          : Boolean;
   	Scale           : Boolean;
   	BetaInitializer : IInitializer;
   	GammaInitializer: IInitializer;
   	BetaRegularizer : IRegularizer;
   	GammaRegularizer: IRegularizer;
   
   	Constructor Create;
   end;
   
   BatchNormalizationArgs = class(LayerArgs)
   	public
   	Axis            : TFShape;
   	Momentum        : Single;
   	Epsilon         : Single;
   	Center          : Boolean;
   	Scale           : Boolean;
   	BetaInitializer : IInitializer;
   	GammaInitializer: IInitializer;
   	MovingMeanInitializer : IInitializer;
   	MovingVarianceInitializer : IInitializer;
   	BetaRegularizer : IRegularizer;
   	GammaRegularizer: IRegularizer;
   	Renorm          : Boolean;
   	RenormMomentum  : Single;
   
   	Constructor Create;
   end;
   
   Pooling1DArgs = class(LayerArgs)
   	private
   	Fstrides : Nullable<Integer>;
   	function  GetStrides: Integer;
   	procedure SetStrides(const Value: Integer);
   	public
   	/// <summary>
   	/// The pooling function to apply, e.g. `tf.nn.max_pool2d`.
   	/// </summary>
   	PoolFunction : IPoolFunction;
   
   	/// <summary>
   	/// specifying the size of the pooling window.
   	/// </summary>
   	PoolSize : Integer;
   
   	/// <summary>
   	/// The padding method, either 'valid' or 'same'.
   	/// </summary>
   	Padding : string ;
   
   	/// <summary>
   	/// one of `channels_last` (default) or `channels_first`.
   	/// </summary>
   	DataFormat : string;
   
   	Constructor Create;
   
   	/// <summary>
   	/// specifying the strides of the pooling operation.
   	/// </summary>
   	property Strides : Integer read GetStrides write SetStrides;
   end;
   
   Pooling2DArgs = class(LayerArgs)
   	public
   	/// <summary>
   	/// The pooling function to apply, e.g. `tf.nn.max_pool2d`.
   	/// </summary>
   	PoolFunction : IPoolFunction;
   
   	/// <summary>
   	/// specifying the size of the pooling window.
   	/// </summary>
   	PoolSize : TFShape;
   
   	/// <summary>
   	/// specifying the strides of the pooling operation.
   	/// </summary>
   	Strides : TFShape;
   
   	/// <summary>
   	/// The padding method, either 'valid' or 'same'.
   	/// </summary>
   	Padding : string ;
   
   	/// <summary>
   	/// one of `channels_last` (default) or `channels_first`.
   	/// </summary>
   	DataFormat : string;
   
   	Constructor Create;
   end;
   
   PreprocessingLayerArgs = class(LayerArgs)
   	public
   	Constructor Create;
   end;
   
   ResizingArgs = class(PreprocessingLayerArgs)
   public
   	Height        : Integer;
   	Width         : Integer;
   	Interpolation : string;
   
   	Constructor Create;
   end;
   
   TextVectorizationArgs = class(PreprocessingLayerArgs)
   public
   	Standardize          : TFunc<TFTensor, TFTensor>;
   	Split                : string ;
   	MaxTokens            : Integer;
   	OutputMode           : string;
   	OutputSequenceLength : Integer;
   	Vocabulary           : TArray<String>;
   
   	Constructor Create;
   end;
   
   RescalingArgs = class(LayerArgs)
   public
   	Scale : Single;
   	Offset: Single;
   
   	Constructor Create;
   end;
   
   ZeroPadding2DArgs = class(LayerArgs)
   public
   	Padding : TNDArray;
   
   	Constructor Create;
   end;
   
   FlattenArgs = class(LayerArgs)
   public
   	DataFormat : string;
   
   	Constructor Create;
   end;
   
   PermuteArgs = class(LayerArgs)
   public
   	dims : TArray<Integer>;
   
   	Constructor Create;
   end;
   
   ReshapeArgs = class(LayerArgs)
   public
   	TargetShape       : TFShape;
   	TargetShapeObjects: TArray<TValue>;
   
   	Constructor Create;
   end;
   
   UpSampling2DArgs = class(LayerArgs)
   public
   	Size          : TFShape;
   	DataFormat    : string;
   	/// <summary>
   	/// 'nearest', 'bilinear'
   	/// </summary>
   	Interpolation : string;
   
   	Constructor Create;
   end;
   
   TensorFlowOpLayerArgs = class(LayerArgs)
   	private
   
   	public
   	NodeDef   : TNodeDef ;
   	Constants : TDictionary<Integer, TNDArray>;
   
   	constructor Create;
   end;		  
   {$ENDREGION}

implementation
        uses Tensorflow.Utils,
             Tensorflow.Tensor,
             Tensorflow;

{ RegularizerArgs }

constructor RegularizerArgs.Create(t: TFTensor);
begin
    X := t;
end;

{ LayerArgs }

constructor LayerArgs.Create;
begin
    Trainable := true;
    Name      := '';
    DType     := TF_DataType.TF_FLOAT;
    bDynamic  := false;
    BatchSize := -1;
end;

{ NodeConfig }

function NodeConfig.ToString: string;
begin
    Result := Format('%s, %d, %d',[Name, NodeIndex, TensorIndex])
end;

{ LayerConfig }

constructor LayerConfig.Create;
begin

end;

{ Node }

constructor Node.Create(_args: NodeArgs);
begin
    FKerasInputs := TList<TFTensor>.Create;
    Fargs := _args;
end;

procedure Node.Connect(_layer: ILayer);
begin
    Flayer := _layer;

    if Fargs.InputTensors <> nil then
        FKerasInputs.AddRange(Fargs.InputTensors);

    // Wire up Node to Layers.
    Flayer.InboundNodes.Add(self);

    for var kt in FKerasInputs do
    begin
        if kt.KerasHistory = nil then
            continue;
        var kKt := kt.KerasHistory as TKerasHistory;
        var t   := kKt.ToTuple;
        var inbound_layer := t.Value1;
        if inbound_layer <> nil then
            inbound_layer.OutboundNodes.Add(Self);
    end;

    // Set metadata on outputs.
    var node_index := Flayer.InboundNodes.Count - 1;
    for var i := 0 to Outputs.Count - 1 do
    begin
        var tensor          := Outputs[i];
        var name := tensor.Op.Tipo;
        {tensor.KerasHistory} Outputs[i].KerasHistory :=  TKerasHistory.Create(layer, node_index, i);
    end;
end;

function Node.serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;
begin
    Result := TList<NodeConfig>.Create;

    for var i := 0 to FKerasInputs.Count-1 do
    begin
        var kh  := FKerasInputs[i].KerasHistory;
        var kkh := kh as TKerasHistory;

        var node_key       := make_node_key(kkh.Layer.Name, kkh.Node_Index);
        var new_node_index := TUtils.Get<string, Integer>(node_conversion_map, node_key, 0);

        var resNode := NodeConfig.Create;
        resNode.Name        := kkh.Layer.Name;
        resNode.NodeIndex   := new_node_index;
        resNode.TensorIndex := kkh.tensor_index;

        Result.Add(resNode);
    end;
end;

function Node.GetILayer: ILayer;
begin
    Result := Flayer;
end;

function Node.GetInLayers: TArray<ILayer>;
begin
    Result := [];
    var t := iterate_inbound.ToArray;
    for var i := 0 to Length(t)-1 do
      Result := Result + [ t[i].Value1 ] ;
end;

function Node.GetINodes: TArray<INode>;
begin
    var node_deps := TList<INode>.Create;
    for var kt in FKerasInputs do
    begin
        var kkh := kt.KerasHistory as TKerasHistory;
        var t   := kkh.ToTuple;
        var llayer     := t.Value1;
        var node_index := t.Value2;
        if llayer <> nil then
            node_deps.Add(llayer.InboundNodes[node_index]);
    end;
    Result := node_deps.ToArray;
end;

function Node.GetInputTensor: TFTensors;
begin
   if is_input then Result := Outputs
   else             Result := Fargs.InputTensors;
end;

function Node.GetIsInput: Boolean;
begin
    Result := Fargs.InputTensors = nil;
end;

function Node.GetlstLayer: TList<TFTensor>;
begin
    Result := FKerasInputs;
end;

procedure Node.SetlstLayer(const lst: TList<TFTensor>);
begin
    FKerasInputs := lst;
end;

function Node.GetOutptus: TFTensors;
begin
    Result := Fargs.Outputs
end;

function Node.iterate_inbound: Enumerable<Tuple<ILayer, Integer, Integer, TFTensor>>;
begin
    var res : TArray< Tuple<ILayer, Integer, Integer, TFTensor> > := [];
    for var kt in FKerasInputs do
    begin
        var kKt := kt.KerasHistory as TKerasHistory;
        var t   := kKt.ToTuple;
        var layer        := t.Value1 ;
        var node_index   := t.Value2;
        var tensor_index := t.Value3;
        res := res + [ tuple.Create(layer, node_index, tensor_index, kt) ] ;
    end;
    Result := Enumerable<Tuple<ILayer, Integer, Integer, TFTensor>>.Create(res);
end;

function Node.MapArguments(tensor_dict: TDictionary<Int64, TQueue<TFTensor>>): TFTensors;
begin
    if FKerasInputs.Count = 1 then
    begin
        var kt_id := FKerasInputs[0].Id;
        Result    := TFTensors.Create( tensor_dict[kt_id].Dequeue );
    end else
    begin
        var flat_arguments := FKerasInputs.ToArray;
        for var kt_index := 0 to  FKerasInputs.Count-1 do
        begin
            var kt                   := FKerasInputs[kt_index] ;
            flat_arguments[kt_index] := tensor_dict[kt.Id].Dequeue;
        end;
        Result := TFTensors.Create( flat_arguments );
    end;
end;

{ CallContextManager }

constructor CallContextManager.Create(build_graph: Boolean);
begin
    Fbuild_graph := build_graph;
end;

procedure CallContextManager.Dispose;
begin

end;

{ CallContext }

constructor TCallContext.Create;
begin

end;

function TCallContext.Enter(build_graph: Boolean): CallContextManager;
begin
   Result := CallContextManager.Create(build_graph);
end;

{ TInputSpec }

constructor TInputSpec.Create(dtype: TF_DataType; _ndim, _min_ndim: Integer; _axes: TDictionary<Integer, Integer>; shape: PTFShape);
begin
    if _ndim = null then FnDim := nil
    else                FnDim := _ndim ;

   if _axes = nil then
      _axes := TDictionary<Integer, Integer>.Create;
  Faxes := _axes;

  if _min_ndim = null then Fmin_ndim := nil
  else                     Fmin_ndim := _min_ndim ;

  if shape <> nil then Fshape := shape^
  else                 Fshape := System.default(TFShape);

  if (Fndim = nil) and (shape <> nil) then
      Fndim := Fshape.ndim;

  FAllAxisDim := [];
  if Faxes <> nil then
    FAllAxisDim := Faxes.Values.ToArray;
end;

destructor TInputSpec.Destroy;
begin
  Faxes.Free;
  inherited;
end;

{ KerasHistory }

constructor TKerasHistory.Create(_layer: ILayer; _node_index, _tensor_index: Integer);
begin
    Flayer        := _layer;
    Fnode_index   := _node_index;
    Ftensor_index := _tensor_index;
end;

function TKerasHistory.ToTuple: Tuple<ILayer, Integer, Integer>;
begin
    Result := Tuple<ILayer, Integer, Integer>.Create(Flayer,Fnode_index,Ftensor_index)
end;

{ Activations }

constructor TActivations.Create;
begin
    FLinear :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := features;
                end;

    FRelu :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                    Result := tf.Context.ExecuteOp('Relu', name, ExecuteOpArgs.Create([features])).First;
                end;

    FSigmoid :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                   Result := tf.Context.ExecuteOp('Sigmoid', name, ExecuteOpArgs.Create([features])).First;
                end;

    FSoftmax :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := tf.Context.ExecuteOp('Softmax', name, ExecuteOpArgs.Create([features])).First;
                end;

    FTanh :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := tf.Context.ExecuteOp('Tanh', name, ExecuteOpArgs.Create([features])).First;
                end;
    Fmish :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := TTensor(features) * tf.math.tanh(tf.math.softplus(features));
                end;
end;

destructor TActivations.Destroy;
begin
     FLinear  := nil;
     FRelu    := nil;
     FSigmoid := nil;
     FSoftmax := nil;
     FTanh    := nil;
     Fmish    := nil;
end;

function TActivations.Get_Linear: TActivation;
begin
    Result := FLinear;
end;

function TActivations.Get_mish: TActivation;
begin
    Result := Fmish
end;

function TActivations.Get_Relu: TActivation;
begin
    Result := FRelu
end;

function TActivations.Get_Sigmoid: TActivation;
begin
    Result := FSigmoid
end;

function TActivations.Get_Softmax: TActivation;
begin
    Result := FSoftmax
end;

function TActivations.Get_Tanh: TActivation;
begin
    Result := FTanh
end;

{$REGION 'Keras.ArgsDefinitions'}
{ ConvolutionalArgs }

constructor ConvolutionalArgs.Create;
begin
    inherited Create;

    Rank           := 2 ;
    NumSpatialDims := -1;
    KernelSize     := 5;
    /// <summary>
    /// specifying the stride length of the convolution.
    /// </summary>
    Strides            := TFShape.Create([1,1]);
    Padding            := 'valid';
    DilationRate       := TFShape.Create([1,1]);
    Groups             := 1;
    KernelInitializer  := tf.glorot_uniform_initializer;
    BiasInitializer    := tf.zeros_initializer;
    
end;

{ RNNArgs }

constructor RNNArgs.Create;
begin
    inherited Create;

    Cell            := nil;
    ReturnSequences := false;
    ReturnState     := false;
    GoBackwards     := false;
    Stateful        := false;
    Unroll          := false;
    TimeMajor       := false;
    Kwargs          := nil;
    UseBias         := True;
end;

{ SimpleRNNArgs }

constructor SimpleRNNArgs.Create;
begin
    inherited Create;
end;

{ OptimizerV2Args }

constructor OptimizerV2Args.Create;
begin
    Name         := '';
    LearningRate := 0.001;
    InitialDecay := 0.0;
    ClipNorm     := 0.0;
    ClipValue    := 0.0;
end;

{ RMSpropArgs }

constructor RMSpropArgs.Create;
begin
    inherited Create;

    RHO       := 0.9;
    Momentum  := 0.0;
    Epsilon   := 1e-7;
    Centered  := false;
end;

{ Cropping2DArgs }

constructor Cropping2DArgs.Create;
begin
    inherited Create;

    data_format := DataFormat.channels_last;
end;

{ Cropping3DArgs }

constructor Cropping3DArgs.Create;
begin
     inherited Create;

     data_format := DataFormat.channels_last;
end;

{ ELUArgs }

constructor ELUArgs.Create;
begin
   inherited Create;

   Alpha := 0.1
end;

{ LeakyReLuArgs }

constructor LeakyReLuArgs.Create;
begin
    inherited Create;

    Alpha := 0.3
end;

{ SoftmaxArgs }

constructor SoftmaxArgs.Create;
begin
     inherited Create;

     axis := -1;
end;

{ BaseDenseAttentionArgs }

constructor BaseDenseAttentionArgs.Create;
begin
    inherited Create;

    causal := False;
    dropout := 0;
end;

{ AttentionArgs }

constructor AttentionArgs.Create;
begin
    inherited Create;

    use_scale := False;
    score_mode:= 'dot';
end;

{ MultiHeadAttentionArgs }

constructor MultiHeadAttentionArgs.Create;
begin
    inherited Create;

    ValueDim          := nil;
    Dropout           := 0;
    UseBias           := True;
    OutputShape       := System.Default(TFShape);
    AttentionAxis     := System.Default(TFShape);
    KernelInitializer := tf.glorot_uniform_initializer;
    BiasInitializer   := tf.zeros_initializer;
    KernelRegularizer := nil;
    BiasRegularizer   := nil;
    KernelConstraint  := nil;
    BiasConstraint    := nil;
end;

{ DropoutArgs }

constructor DropoutArgs.Create;
begin
    inherited Create;
end;

{ DenseArgs }

constructor DenseArgs.Create;
begin
     inherited Create;

     UseBias           := True;
     KernelInitializer := tf.glorot_uniform_initializer;
     BiasInitializer   := tf.zeros_initializer;
end;

{ TensorFlowOpLayerArgs }

constructor TensorFlowOpLayerArgs.Create;
begin
    inherited Create;
end;

{ EinsumDenseArgs }

constructor EinsumDenseArgs.Create;
begin
    inherited Create;

    BiasAxes          := '';
    KernelInitializer := tf.glorot_uniform_initializer;
    BiasInitializer   := tf.zeros_initializer;
end;

{ EmbeddingArgs }

constructor EmbeddingArgs.Create;
begin
    inherited Create;

    InputLength := -1;
end;

{ InputLayerArgs }

constructor InputLayerArgs.Create;
begin
    inherited Create;
end;

{ Conv1DArgs }

constructor Conv1DArgs.Create;
begin
   inherited Create;
end;

{ Conv2DArgs }

constructor Conv2DArgs.Create;
begin
   inherited Create;
end;

{ CroppingArgs }

constructor CroppingArgs.Create;
begin
    inherited Create;
end;

{ LSTMArgs }

constructor LSTMArgs.Create;
begin
   inherited Create;
end;

{ LSTMCellArgs }

constructor LSTMCellArgs.Create;
begin
   inherited Create;
end;

{ MergeArgs }

constructor MergeArgs.Create;
begin
    inherited Create;
end;

{ LayerNormalizationArgs }

constructor LayerNormalizationArgs.Create;
begin
    inherited Create;

    Axis             := -1;
    Epsilon          := 1e-3;
    Center           := true;
    Scale            := true;
    BetaInitializer  := tf.zeros_initializer;
    GammaInitializer := tf.ones_initializer;
end;

{ BatchNormalizationArgs }

constructor BatchNormalizationArgs.Create;
begin
    inherited Create;

    Axis             := -1;
    Momentum         := 0.99;
    Epsilon          := 1e-3;
    Center           := true;
    Scale            := true;
    BetaInitializer  := tf.zeros_initializer;
    GammaInitializer := tf.ones_initializer;
    MovingMeanInitializer := tf.zeros_initializer;
    MovingVarianceInitializer := tf.ones_initializer;
    RenormMomentum   := 0.99;
end;

{ Pooling1DArgs }

constructor Pooling1DArgs.Create;
begin
    inherited Create;

    Fstrides := nil;
    Padding  := 'valid';
end;

function Pooling1DArgs.GetStrides: Integer;
begin
    if Fstrides.HasValue then   Result := Fstrides.Value
    else                        Result := PoolSize;
end;

procedure Pooling1DArgs.SetStrides(const Value: Integer);
begin
    Fstrides := Value;
end;

{ Pooling2DArgs }

constructor Pooling2DArgs.Create;
begin
    inherited Create;

    Padding  := 'valid';
end;

{ PreprocessingLayerArgs }

constructor PreprocessingLayerArgs.Create;
begin
   inherited Create;
end;

{ ResizingArgs }

constructor ResizingArgs.Create;
begin
    inherited Create;

    Interpolation := 'bilinear';
end;

{ TextVectorizationArgs }

constructor TextVectorizationArgs.Create;
begin
    inherited Create;

    Split     := 'standardize';
    MaxTokens := -1;
    OutputMode:= 'int';
    OutputSequenceLength := -1;

end;

{ RescalingArgs }

constructor RescalingArgs.Create;
begin
    inherited Create;
end;

{ ZeroPadding2DArgs }

constructor ZeroPadding2DArgs.Create;
begin
    inherited Create;
end;

{ FlattenArgs }

constructor FlattenArgs.Create;
begin
    inherited Create;
end;

{ PermuteArgs }

constructor PermuteArgs.Create;
begin
    inherited Create;
end;

{ ReshapeArgs }

constructor ReshapeArgs.Create;
begin
    inherited Create;

    TargetShape        := System.default(TFShape);
    TargetShapeObjects := [];
end;

{ UpSampling2DArgs }

constructor UpSampling2DArgs.Create;
begin
    inherited Create;

    Size         := System.default(TFShape);
    Interpolation:= 'nearest';
end;

{ ModelArgs }

constructor ModelArgs.Create;
begin
    inherited Create;
end;

{ SequentialArgs }

constructor SequentialArgs.Create;
begin
     inherited Create;
end;
{$ENDREGION}

end.
