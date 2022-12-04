unit Keras.ArgsDefinition;
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

 Cropping2DArgs = class(LayerArgs)
    private

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
    private

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

        Unitis              : Integer;
        Activation          : TActivation;
        RecurrentActivation : TActivation;
        UseBias             : boolean;
        KernelInitializer   : IInitializer;
        RecurrentInitializer: IInitializer;
        BiasInitializer     : IInitializer;

        Constructor Create;
  end;

  OptimizerV2Args = class
    private

    public
        Name         : string;
        LearningRate : Single;
        InitialDecay : Single;
        ClipNorm     : Single;
        ClipValue    : Single;

        Constructor Create;
  end;

  RMSpropArgs = class(OptimizerV2Args)
    private

    public
        RHO      : Single;
        Momentum : Single;
        Epsilon  : Single;
        Centered : Boolean;

        Constructor Create;
  end;

  ELUArgs = class(LayerArgs)
    private

    public
      Alpha   : Single ;
      constructor Create;
  end;

  LeakyReLuArgs = class(LayerArgs)
    private

    public
      Alpha   : Single ;
      constructor Create;
  end;

  SoftmaxArgs = class(LayerArgs)
    private

    public
      axis   : TAxis ;
      constructor Create;
  end;

  BaseDenseAttentionArgs = class(LayerArgs)
    private

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

  TensorFlowOpLayerArgs = class(LayerArgs)
    private

    public
      NodeDef   : TNodeDef ;
      Constants : TDictionary<Integer, TNDArray>;
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
    data_format := DataFormat.channels_last;
end;

{ Cropping3DArgs }

constructor Cropping3DArgs.Create;
begin
     data_format := DataFormat.channels_last_;
end;

{ ELUArgs }

constructor ELUArgs.Create;
begin
   Alpha := 0.1
end;

{ LeakyReLuArgs }

constructor LeakyReLuArgs.Create;
begin
    Alpha := 0.3
end;

{ SoftmaxArgs }

constructor SoftmaxArgs.Create;
begin
     axis := -1;
end;

{ BaseDenseAttentionArgs }

constructor BaseDenseAttentionArgs.Create;
begin
    causal := False;
    dropout := 0;
end;

end.
