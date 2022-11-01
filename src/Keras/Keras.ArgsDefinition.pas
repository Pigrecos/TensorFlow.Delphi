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
