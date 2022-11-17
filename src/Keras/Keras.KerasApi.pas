unit Keras.KerasApi;

interface
     uses System.SysUtils,

          Keras.Layer,
          Keras.Activations,
          Keras.Optimizer;


type
  KerasInterface = class
    private

    public
      optimizers :  OptimizerApi;

      constructor Create;
  end;

var
  kKeras : KerasInterface;


implementation

{ KerasInterface }

constructor KerasInterface.Create;
begin
    optimizers := OptimizerApi.Create;
end;

initialization
begin
    kKeras := KerasInterface.Create;
end;

finalization
begin
    kKeras.Free;
end;

end.
