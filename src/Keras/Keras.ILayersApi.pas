unit Keras.ILayersApi;
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
          System.Rtti,

          TensorFlow.DApi,
          TensorFlow.Initializer,

          NumPy.NDArray,
          Numpy.Axis,

          Keras.Engine,
          Keras.Layer,
          Keras.Activations,
          Keras.Optimizer,
          Keras.Regularizers,
          Keras.ArgsDefinition;


type

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

      function Input(shape: TFShape; batch_size : Integer = -1; name : string = ''; sparse: Boolean = false; ragged : Boolean= false): TFTensors;

      function InputLayer(input_shape: TFShape; name : string= ''; sparse : Boolean= false; ragged : Boolean= false): ILayer;

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
      function MaxPooling2D(pool_size : PTFShape= nil; strides : PTFShape= nil; padding : string= 'valid'; data_format: string = ''): ILayer;

      function Permute(dims: TArray<Integer>): ILayer;

      function Rescaling(scale: Single; offset : Single= 0; input_shape : PTFShape= nil): ILayer;

      function SimpleRNN( units                : Integer;
                          activation           : string= 'tanh';
                          kernel_initializer   : string= 'glorot_uniform';
                          recurrent_initializer: string= 'orthogonal';
                          bias_initializer     : string= 'zeros'): ILayer;

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
      function Cropping2D(cropping: TNDArray; data_format : Cropping2DArgs.DataFormat = Cropping2DArgs.DataFormat.channels_last): ILayer;
      function Cropping3D(cropping: TNDArray; data_format : Cropping3DArgs.DataFormat = Cropping3DArgs.DataFormat.channels_last_): ILayer;

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

implementation

end.
