unit Keras.ArgsDefinition;

interface
       uses System.SysUtils,
            Spring,
            Spring.Collections,
            Spring.Collections.Lists,
            Spring.Collections.Dictionaries,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable,
            TensorFlow.Initializer,

            Keras.Activations,
            Keras.Layer;

type
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
      KernelConstraint   : procedure;
      BiasConstraint     : procedure;

      constructor Create;
  end;

  RNNArgs = class(LayerArgs)
    private

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

        Constructor Create;
  end;


implementation
          uses Tensorflow;

{ ConvolutionalArgs }

constructor ConvolutionalArgs.Create;
begin
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
    Cell            := nil;
    ReturnSequences := false;
    ReturnState     := false;
    GoBackwards     := false;
    Stateful        := false;
    Unroll          := false;
    TimeMajor       := false;
    Kwargs          := nil;
end;

end.
