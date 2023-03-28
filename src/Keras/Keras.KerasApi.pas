unit Keras.KerasApi;
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
          System.Generics.Collections,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.Initializer,
          TensorFlow.Core,

          Keras.Data,

          Keras.Optimizer,
          Keras.Regularizers,
          Keras.Backend,
          Keras.Preprocessing,
          Keras.MetricsApi,
          Keras.LayersApi,
          Keras.LossFunc,
          Keras.Utils,
          Keras.Core,
          Keras.Models;

type
  InitializersApi = class(TInterfacedObject, IInitializersApi)
    public
      /// <summary>
      /// He normal initializer.
      /// </summary>
      /// <param name="seed"></param>
      /// <returns></returns>
      function he_normal(seed : PInteger= nil) : IInitializer;
      function Orthogonal(gain: Single = 1.0; seed : pInteger= nil) : IInitializer;
  end;

  IKerasApi = interface
    function GetLayers: ILayersApi;
    function GetLosses: ILossesApi;
    function GetMetrics:IMetricsApi;
    function GetInizializer:IInitializersApi;
    function GetActivations:IActivationsApi;
    function GetOptimizers:IOptimizerApi;

    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <returns></returns>
    function Model(inputs: TFTensors; outputs: TFTensors; name: string = ''): IModel;
    /// <summary>
    /// Instantiate a Keras tensor.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="batch_size"></param>
    /// <param name="dtype"></param>
    /// <param name="name"></param>
    /// <param name="sparse">
    /// A boolean specifying whether the placeholder to be created is sparse.
    /// </param>
    /// <param name="ragged">
    /// A boolean specifying whether the placeholder to be created is ragged.
    /// </param>
    /// <param name="tensor">
    /// Optional existing tensor to wrap into the `Input` layer.
    /// If set, the layer will not create a placeholder tensor.
    /// </param>
    /// <returns></returns>
    function  Input( shape      : TFShape;
                     batch_size : Integer = -1;
                     name       : string = '';
                     dtype      : TF_DataType = DtInvalid;
                     sparse     : Boolean = false;
                     tensor     : TFTensor = nil;
                     ragged     : Boolean= false;
                     type_spec  : TypeSpec= nil;
                     batch_input_shape: PTFShape= nil;
                     batch_shape: PTFShape= nil): TFTensors; overload;
    function  Input(shape: TFShape; name : string): TFTensor; overload;

    property Inizializers : IInitializersApi read GetInizializer;
    property layers       : ILayersApi       read GetLayers;
    property losses       : ILossesApi       read GetLosses;
    property activations  : IActivationsApi  read GetActivations;
    property optimizers   : IOptimizerApi    read GetOptimizers;
    property metrics      : IMetricsApi      read GetMetrics;
    // IModelsApi models { get; }
  end;

  TRegularizers = class
    public
      function l2(_l2: Single = 0.01): IRegularizer;
  end;

  KerasInterface = class(TInterfacedObject,IKerasApi)
    private
      FInizializers : IInitializersApi;
      Flayers       : ILayersApi;
      Flosses       : ILossesApi;
      Factivations  : IActivationsApi;
      Foptimizers   : IOptimizerApi;
      Fmetrics      : IMetricsApi;

      function GetInizializer:IInitializersApi;
      function GetLayers: ILayersApi;
      function GetLosses: ILossesApi;
      function GetActivations:IActivationsApi;
      function GetOptimizers:IOptimizerApi;
      function GetMetrics:IMetricsApi;
    public
      datasets     : KerasDataset;
      Regularizers : TRegularizers;
      preprocessing: TPreprocessing;
      backend      : BackendImpl;
      //public ModelsApi models { get; } = new ModelsApi();
      utils        : KerasUtils;
      constructor Create;
      destructor Destroy; override;
      /// <summary>
      /// Instantiate a Keras tensor.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="batch_size"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="sparse">
      /// A boolean specifying whether the placeholder to be created is sparse.
      /// </param>
      /// <param name="ragged">
      /// A boolean specifying whether the placeholder to be created is ragged.
      /// </param>
      /// <param name="tensor">
      /// Optional existing tensor to wrap into the `Input` layer.
      /// If set, the layer will not create a placeholder tensor.
      /// </param>
      /// <returns></returns>
      function  Input( shape      : TFShape;
                       batch_size : Integer = -1;
                       name       : string = '';
                       dtype      : TF_DataType = DtInvalid;
                       sparse     : Boolean = false;
                       tensor     : TFTensor = nil;
                       ragged     : Boolean= false;
                       type_spec  : TypeSpec= nil;
                       batch_input_shape: PTFShape= nil;
                       batch_shape: PTFShape= nil): TFTensors; overload;

      function  Input(shape: TFShape; name : string): TFTensor; overload;
      function  Sequential(layers: TList<ILayer> = nil; name : string= ''): Sequential;
      /// <summary>
      /// `Model` groups layers into an object with training and inference features.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="output"></param>
      /// <returns></returns>
      function Model(inputs: TFTensors; outputs: TFTensors; name: string = ''): IModel;
  end;

implementation
       uses Tensorflow;

{ InitializersApi }

function InitializersApi.he_normal(seed: PInteger): IInitializer;
begin
   Result := VarianceScaling.Create(2.0, 'fan_in', false, seed);
end;

function InitializersApi.Orthogonal(gain: Single; seed: pInteger): IInitializer;
begin
    Result := TensorFlow.Initializer.Orthogonal.Create(gain, seed) ;
end;

{ Regularizers }

function TRegularizers.l2(_l2: Single): IRegularizer;
begin
   Result :=  TL2.Create(_l2);
end;

{ KerasInterface }

constructor KerasInterface.Create;
begin
    FInizializers := InitializersApi.Create;
    Regularizers  := TRegularizers.Create;
    Foptimizers   := OptimizerApi.Create;
    Fmetrics      := MetricsApi.Create;
    utils         := KerasUtils.Create;
    Flayers       := LayersApi.Create;
    Flosses       := LossesApi.Create;
    Factivations  := TActivations.Create;
    preprocessing := TPreprocessing.Create;

    datasets     := KerasDataset.Create;
    backend      := BackendImpl.Create;
end;

destructor KerasInterface.Destroy;
begin
    Regularizers.Free;
    utils.Free;
    preprocessing.Free;
    datasets.free;
    backend.Free;

    inherited Destroy;
end;

function KerasInterface.GetActivations: IActivationsApi;
begin
    Result := Factivations;
end;

function KerasInterface.GetInizializer: IInitializersApi;
begin
    Result :=  FInizializers;
end;

function KerasInterface.GetLayers: ILayersApi;
begin
    Result := Flayers;
end;

function KerasInterface.GetLosses: ILossesApi;
begin
    Result := Flosses;
end;

function KerasInterface.GetMetrics: IMetricsApi;
begin
    Result := Fmetrics;
end;

function KerasInterface.GetOptimizers: IOptimizerApi;
begin
    Result := Foptimizers;
end;

function KerasInterface.Input(shape: TFShape; name : string): TFTensor;
begin
    Result := Input(shape,-1, name).First;
end;

function KerasInterface.Input(shape : TFShape; batch_size : Integer; name : string; dtype : TF_DataType; sparse : Boolean; tensor : TFTensor; ragged : Boolean; type_spec : TypeSpec; batch_input_shape: PTFShape; batch_shape: PTFShape): TFTensors;
begin
    Result := tf.keras.layers.Input(shape, batch_size, name, dtype, sparse, tensor, ragged, type_spec, batch_input_shape, batch_shape)
end;

function KerasInterface.Model(inputs, outputs: TFTensors; name: string): IModel;
begin
    Result := Functional.Create(inputs, outputs, name);
end;

function KerasInterface.Sequential(layers: TList<ILayer>; name: string): Sequential;
 var
   _args : SequentialArgs;
begin
    _args := SequentialArgs.Create;
    _args.Layers := layers;
    _args.Name   := name;

    Result := Keras.Models.Sequential.Create(_args);
end;

end.
