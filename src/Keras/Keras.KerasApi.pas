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

          TensorFlow.Initializer,

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
          Keras.Utils;


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
      //public KerasDataset datasets { get; } = new KerasDataset();
      Inizializers : TInitializers;
      Regularizers : TRegularizers;
      layers       : ILayersApi;
      losses       : LossesApi;
      activations  : TActivations;
      preprocessing: TPreprocessing;
      backend      : BackendImpl;
      optimizers   : OptimizerApi;
      metrics      : MetricsApi;
      //public ModelsApi models { get; } = new ModelsApi();
      utils        : KerasUtils;
      constructor Create;
  end;

implementation

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

    backend      := BackendImpl.Create;
    optimizers   := OptimizerApi.Create;
end;

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

function KerasInterface.GetLayers: ILayersApi;
begin
    Result := Flayers;
end;

end.
