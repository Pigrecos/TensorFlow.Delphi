unit Keras.ArgsDefinition;
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

            Spring,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable,
            TensorFlow.Initializer,

            Keras.Regularizers,
            Keras.Activations,

            Keras.Engine,

            ProtoGen.nodeDef;

type

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
      type DataFormat = ( channels_first = 0, channels_last = 1 );
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
      type DataFormat = ( channels_first_ = 0, channels_last_ = 1 );
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

implementation
          uses Tensorflow;

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

     data_format := DataFormat.channels_last_;
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
    OutputShape       := Default(TFShape);
    AttentionAxis     := Default(TFShape);
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

    TargetShape        := default(TFShape);
    TargetShapeObjects := [];
end;

{ UpSampling2DArgs }

constructor UpSampling2DArgs.Create;
begin
    inherited Create;

    Size         := default(TFShape);
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

end.
