unit Keras.KerasApi;

interface
     uses System.SysUtils,

          TensorFlow.Initializer,

          Keras.ILayersApi,
          Keras.Layer,
          Keras.Activations,
          Keras.Optimizer,
          Keras.Regularizers,
          Keras.Backend;


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
      //public LossesApi losses { get; } = new LossesApi();
      //public Activations activations { get; } = new Activations();
      //public Preprocessing preprocessing { get; } = new Preprocessing();
      backend      : BackendImpl;
      optimizers   : OptimizerApi;
      //public MetricsApi metrics { get; } = new MetricsApi();
      //public ModelsApi models { get; } = new ModelsApi();
      //public KerasUtils utils { get; } = new KerasUtils();
      constructor Create;
  end;

implementation

{ KerasInterface }

constructor KerasInterface.Create;
begin
    Inizializers := TInitializers.Create;
    Regularizers := TRegularizers.Create;
    optimizers   := OptimizerApi.Create;

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
