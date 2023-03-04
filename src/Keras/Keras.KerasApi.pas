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

          Keras.Data,
          Keras.ILayersApi,
          Keras.Layer,
          Keras.Activations,
          Keras.Optimizer,
          Keras.Regularizers,
          Keras.Backend,
          Keras.Preprocessing,
          Keras.MetricsApi,
          Keras.LayersApi,
          Keras.LossFunc,
          Keras.Utils,
          Keras.Engine,
          Keras.Models;

type

  IKerasApi = interface
    function GetLayers: ILayersApi;

    property layers : ILayersApi read GetLayers;
  end;

  TInitializers = class
    private

    public
      /// <summary>
      /// He normal initializer.
      /// </summary>
      /// <param name="seed"></param>
      /// <returns></returns>
      function he_normal(seed : PInteger= nil) : IInitializer;
  end;

  TRegularizers = class
    private

    public
      function l2(_l2: Single = 0.01): IRegularizer;
  end;

  KerasInterface = class(TInterfacedObject,IKerasApi)
    private
      Flayers  : ILayersApi;

      function GetLayers: ILayersApi;
    public
      datasets     : KerasDataset;
      Inizializers : TInitializers;
      Regularizers : TRegularizers;
      layers       : ILayersApi;
      losses       : ILossesApi;
      activations  : TActivations;
      preprocessing: TPreprocessing;
      backend      : BackendImpl;
      optimizers   : OptimizerApi;
      metrics      : IMetricsApi;
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
      function  Input(shape             : TFShape;
                      batch_input_shape : TFShape;
                      batch_size        : Integer= -1;
                      dtype             : TF_DataType = DtInvalid;
                      name              : string = '';
                      sparse            : Boolean = false;
                      ragged            : Boolean = false;
                      tensor            : TFTensor = nil): TFTensor; overload;
      function  Input(shape             : TFShape): TFTensor; overload;
      function  Input(shape: TFShape; name : string): TFTensor; overload;
      function  Sequential(layers: TList<ILayer> = nil; name : string= ''): Sequential;
      /// <summary>
      /// `Model` groups layers into an object with training and inference features.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="output"></param>
      /// <returns></returns>
      function Model(inputs: TFTensors; outputs: TFTensors; name: string = ''): Functional;
  end;

implementation
       uses Keras.ArgsDefinition;

{ Initializers }

function TInitializers.he_normal(seed: PInteger): IInitializer;
begin
   Result := VarianceScaling.Create(2.0, 'fan_in', false, seed);
end;

{ Regularizers }

function TRegularizers.l2(_l2: Single): IRegularizer;
begin
   Result :=  TL2.Create(_l2);
end;

{ KerasInterface }

constructor KerasInterface.Create;
begin
    Inizializers := TInitializers.Create;
    Regularizers := TRegularizers.Create;
    optimizers   := OptimizerApi.Create;
    metrics      := MetricsApi.Create;
    utils        := KerasUtils.Create;
    layers       := LayersApi.Create;
    losses       := LossesApi.Create;
    activations  := TActivations.Create;
    preprocessing:= TPreprocessing.Create;

    datasets     := KerasDataset.Create;

    backend      := BackendImpl.Create;
end;

destructor KerasInterface.Destroy;
begin
    Inizializers.Free;
    Regularizers.Free;
    optimizers.Free;
    utils.Free;

    activations.Free;
    preprocessing.Free;

    backend.Free;
    datasets.free;

    inherited Destroy;
end;

function KerasInterface.GetLayers: ILayersApi;
begin
    Result := Flayers;
end;

function KerasInterface.Input(shape: TFShape): TFTensor;
begin
    Result := Input(shape, default(TFShape), -1, DtInvalid, '', false, false, nil);
end;

function KerasInterface.Input(shape: TFShape; name : string): TFTensor;
begin
    Result := Input(shape, default(TFShape), -1, DtInvalid, name, false, false, nil);
end;

function KerasInterface.Input(shape, batch_input_shape: TFShape; batch_size: Integer; dtype: TF_DataType; name: string; sparse, ragged: Boolean; tensor: TFTensor): TFTensor;
var
  args : InputLayerArgs;
begin
   if not batch_input_shape.isNull then
   begin
        var a :=  batch_input_shape.dims;
        Delete(a,0,1);
        shape := a;
   end;

   args := InputLayerArgs.Create;
   args.Name            := name;
   args.InputShape      := shape;
   args.BatchInputShape := batch_input_shape;
   args.BatchSize       := batch_size;
   args.DType           := dtype;
   args.Sparse          := sparse;
   args.Ragged          := ragged;
   args.InputTensor     := tensor;

   var input_layer := Keras.Layer.InputLayer.Create(args);
   Result := input_layer.InboundNodes[0].Outputs.first;
   
end;

function KerasInterface.Model(inputs, outputs: TFTensors; name: string): Functional;
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
