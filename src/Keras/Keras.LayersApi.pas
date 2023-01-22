unit Keras.LayersApi;
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

          Spring,

          TensorFlow.DApi,
          TensorFlow.Initializer,
          Numpy.Axis,

          Keras.Engine,
          Keras.Activations,
          Keras.Regularizers,
          Keras.ILayersApi,
          Keras.ArgsDefinition,
          Keras.Layer,
          Keras.Preprocessing;

type
  LayersApi = class(TInterfacedObject, ILayersApi)
    private
       function ReadProc: IPreprocessing;
    protected
       FPreProcessing : IPreprocessing;
    public

       constructor Create;

       function Add: ILayer;

       /// <summary>
       /// Average pooling operation for spatial data.
       /// </summary>
       /// <param name="pool_size"></param>
       /// <param name="strides"></param>
       /// <param name="padding"></param>
       /// <param name="data_format"></param>
       /// <returns></returns>
       function AveragePooling2D(pool_size: PTFShape = nil; strides: PTFShape = nil; padding: string = 'valid'; data_format: string = ''): ILayer; overload;
       function AveragePooling2D(pool_size: TFShape ; strides: TFShape ;             padding: string = 'valid'; data_format: string = ''): ILayer; overload;

       /// <summary>
       /// Layer that normalizes its inputs.
       /// Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
       /// Importantly, batch normalization works differently during training and during inference.
       ///
       /// http://arxiv.org/abs/1502.03167
       /// </summary>
       /// <param name="axis">The axis that should be normalized (typically the features axis).
       /// For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
       /// </param>
       /// <param name="momentum">Momentum for the moving average.</param>
       /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
       /// <param name="center">If True, add offset of beta to normalized tensor. If False, beta is ignored.</param>
       /// <param name="scale">If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.</param>
       /// <param name="beta_initializer">Initializer for the beta weight.</param>
       /// <param name="gamma_initializer">Initializer for the gamma weight.</param>
       /// <param name="moving_mean_initializer">Initializer for the moving mean.</param>
       /// <param name="moving_variance_initializer">Initializer for the moving variance.</param>
       /// <param name="trainable">Boolean, if True the variables will be marked as trainable.</param>
       /// <param name="name">Layer name.</param>
       /// <param name="renorm">Whether to use Batch Renormalization. This adds extra variables during training. The inference is the same for either value of this parameter.</param>
       /// <param name="renorm_momentum">Momentum used to update the moving means and standard deviations with renorm.
       /// Unlike momentum, this affects training and should be neither too small (which would add noise) nor too large (which would give stale estimates).
       /// Note that momentum is still applied to get the means and variances for inference.
       /// </param>
       /// <returns>Tensor of the same shape as input.</returns>
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

       /// <summary>
       /// 1D convolution layer (e.g. temporal convolution).
       /// This layer creates a convolution kernel that is convolved with the layer input over a single spatial(or temporal) dimension to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
       /// </summary>
       /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
       /// <param name="kernel_size">An integer specifying the width of the 1D convolution window.</param>
       /// <param name="strides">An integer specifying the stride of the convolution window . Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
       /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
       /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
       /// <param name="dilation_rate">An integer specifying the dilation rate to use for dilated convolution.Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
       /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
       /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
       /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
       /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see keras.initializers).</param>
       /// <param name="bias_initializer">Initializer for the bias vector (see keras.initializers).</param>
       /// <returns>A tensor of rank 3 representing activation(conv1d(inputs, kernel) + bias).</returns>
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

       /// <summary>
       /// 2D convolution layer (e.g. spatial convolution over images).
       /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
       /// If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
       /// </summary>
       /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
       /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
       /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
       /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
       /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
       /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
       /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
       /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
       /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
       /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see keras.initializers).</param>
       /// <param name="bias_initializer">Initializer for the bias vector (see keras.initializers).</param>
       /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
       /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see keras.regularizers).</param>
       /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
       /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
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

       /// <summary>
       /// 2D convolution layer (e.g. spatial convolution over images).
       /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
       /// If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
       /// </summary>
       /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
       /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
       /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
       /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
       /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
       /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
       /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
       /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
       /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
       /// <param name="kernel_initializer">The name of the initializer for the kernel weights matrix (see keras.initializers).</param>
       /// <param name="bias_initializer">The name of the initializer for the bias vector (see keras.initializers).</param>
       /// <param name="kernel_regularizer">The name of the regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
       /// <param name="bias_regularizer">The name of the regularizer function applied to the bias vector (see keras.regularizers).</param>
       /// <param name="activity_regularizer">The name of the regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
       /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
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

       /// <summary>
        /// Transposed convolution layer (sometimes called Deconvolution).
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="output_padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">The name of the initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">The name of the initializer for the bias vector (see keras.initializers).</param>
        /// <param name="kernel_regularizer">The name of the regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
        /// <param name="bias_regularizer">The name of the regularizer function applied to the bias vector (see keras.regularizers).</param>
        /// <param name="activity_regularizer">The name of the regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
        /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
        function Conv2DTranspose(filters             : Integer;
                                kernel_size          : PTFShape= nil;
                                strides              : PTFShape= nil;
                                output_padding       : string = 'valid';
                                data_format          : string = '';
                                dilation_rate        : PTFShape = nil;
                                activation           : string = '';
                                use_bias             : Boolean= true;
                                kernel_initializer   : string = '';
                                bias_initializer     : string = '';
                                kernel_regularizer   : string = '';
                                bias_regularizer     : string = '';
                                activity_regularizer : string = ''): ILayer;

       /// <summary>
       /// Just your regular densely-connected NN layer.
       ///
       /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the
       /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer,
       /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
       /// </summary>
       /// <param name="units">Positive integer, dimensionality of the output space.</param>
       /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
       function Dense(units: Integer): ILayer; overload;

       /// <summary>
       /// Just your regular densely-connected NN layer.
       ///
       /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the
       /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer,
       /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
       /// </summary>
       /// <param name="units">Positive integer, dimensionality of the output space.</param>
       /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
       /// <param name="input_shape">N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).</param>
       /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
       function Dense(units: Integer; activation: string = ''; input_shape: PTFShape = nil): ILayer; overload;

       /// <summary>
       /// Just your regular densely-connected NN layer.
       ///
       /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the
       /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer,
       /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
       /// </summary>
       /// <param name="units">Positive integer, dimensionality of the output space.</param>
       /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
       /// <param name="kernel_initializer">Initializer for the kernel weights matrix.</param>
       /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
       /// <param name="bias_initializer">Initializer for the bias vector.</param>
       /// <param name="input_shape">N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).</param>
       /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
       function Dense(units             : Integer;
                      activation        : TActivation= nil;
                      kernel_initializer: IInitializer = nil;
                      use_bias          : Boolean= true;
                      bias_initializer  : IInitializer = nil;
                      input_shape       : PTFShape= nil): ILayer; overload;

       /// <summary>
       /// Applies Dropout to the input.
       /// The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time,
       /// which helps prevent overfitting.Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
       /// </summary>
       /// <param name="rate">Float between 0 and 1. Fraction of the input units to drop.</param>
       /// <param name="noise_shape">1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance,
       /// if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps,
       /// you can use noise_shape=(batch_size, 1, features).
       /// </param>
       /// <param name="seed">An integer to use as random seed.</param>
       /// <returns></returns>
       function Dropout(rate: Single; noise_shape: PTFShape = nil; seed: pInteger= nil): ILayer;

       /// <summary>
       /// Turns positive integers (indexes) into dense vectors of fixed size.
       /// This layer can only be used as the first layer in a model.
       /// e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
       /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
       /// </summary>
       /// <param name="input_dim">Size of the vocabulary, i.e. maximum integer index + 1.</param>
       /// <param name="output_dim">Dimension of the dense embedding.</param>
       /// <param name="embeddings_initializer">Initializer for the embeddings matrix (see keras.initializers).</param>
       /// <param name="mask_zero"></param>
       /// <returns></returns>
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

        /// <summary>
        /// Flattens the input. Does not affect the batch size.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...).
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json.
        /// If you never set it, then it will be "channels_last".
        /// </param>
        /// <returns></returns>
        function Flatten(data_format: string = ''): ILayer;

        /// <summary>
        /// Global average pooling operation for temporal data.
        /// </summary>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        function GlobalAveragePooling1D(data_format : string= 'channels_last'): ILayer;

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <returns></returns>
        function GlobalAveragePooling2D: ILayer; overload;

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).</param>
        /// <returns></returns>
        function GlobalAveragePooling2D(data_format: string = 'channels_last'): ILayer; overload;

        /// <summary>
        /// Global max pooling operation for 1D temporal data.
        /// Downsamples the input representation by taking the maximum value over the time dimension.
        /// </summary>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        function GlobalMaxPooling1D(data_format: string = 'channels_last'): ILayer;

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).</param>
        /// <returns></returns>
        function GlobalMaxPooling2D(data_format: string = 'channels_last'): ILayer;

        /// <summary>
        /// `Input()` is used to instantiate a Keras tensor.
        ///  Keras tensor is a TensorFlow symbolic tensor object, which we augment with certain attributes that allow us
        ///  to build a Keras model just by knowing the inputs and outputs of the model.
        /// </summary>
        /// <param name="shape">A shape tuple not including the batch size.</param>
        /// <param name="name">An optional name string for the layer. Should be unique in a model (do not reuse the same name twice). It will be autogenerated if it isn't provided.</param>
        /// <param name="sparse">A boolean specifying whether the placeholder to be created is sparse. Only one of 'ragged' and 'sparse' can be True.
        /// Note that, if sparse is False, sparse tensors can still be passed into the input - they will be densified with a default value of 0.
        /// </param>
        /// <param name="ragged">A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True.
        /// In this case, values of 'None' in the 'shape' argument represent ragged dimensions. For more information about RaggedTensors, see this guide.
        /// </param>
        /// <returns>A tensor.</returns>
        function Input(shape: TFShape; batch_size : Integer = -1; name : string = ''; sparse: Boolean = false; ragged : Boolean= false): TFTensors;

        function InputLayer(input_shape: TFShape; name : string= ''; sparse : Boolean= false; ragged : Boolean= false): ILayer;

        function LayerNormalization(axis             : TAxis;
                                    epsilon          : Single= 1e-3;
                                    center           : Boolean= true;
                                    scale            : Boolean= true;
                                    beta_initializer : IInitializer= nil;
                                    gamma_initializer: IInitializer= nil): ILayer;

        /// <summary>
        /// Leaky version of a Rectified Linear Unit.
        /// </summary>
        /// <param name="alpha">Negative slope coefficient.</param>
        /// <returns></returns>
        function LeakyReLU(alpha: Single = 0.3): ILayer;

        /// <summary>
        /// Long Short-Term Memory layer - Hochreiter 1997.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use. If you pass null, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step. If you pass null, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean (default True), whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs. Default: glorot_uniform.</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. Default: orthogonal.</param>
        /// <param name="bias_initializer">Initializer for the bias vector. Default: zeros.</param>
        /// <param name="unit_forget_bias">Boolean (default True). If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al..</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.</param>
        /// <param name="implementation"></param>
        /// <param name="return_sequences">Boolean. Whether to return the last output. in the output sequence, or the full sequence. Default: False.</param>
        /// <param name="return_state">Whether to return the last state in addition to the output. Default: False.</param>
        /// <param name="go_backwards">Boolean (default false). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="time_major">
        /// The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be in shape [timesteps, batch, feature],
        /// whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the
        /// beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.</param>
        /// <param name="unroll">
        /// Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN,
        /// although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
        /// </param>
        /// <returns></returns>
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

        /// <summary>
        ///
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">The name of the activation function to use. Default: hyperbolic tangent (tanh)..</param>
        /// <returns></returns>
        function SimpleRNN( units                : Integer;
                            activation           : string= 'tanh';
                            kernel_initializer   : string= 'glorot_uniform';
                            recurrent_initializer: string= 'orthogonal';
                            bias_initializer     : string= 'zeros';
                            return_sequences     : Boolean= False;
                            return_state         : Boolean = false): ILayer;
        /// <summary>
        /// Max pooling operation for 1D temporal data.
        /// </summary>
        /// <param name="pool_size">Integer, size of the max pooling window.</param>
        /// <param name="strides">Integer, or null. Specifies how much the pooling window moves for each pooling step. If null, it will default to pool_size.</param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding.
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        function MaxPooling1D(pool_size : PInteger= nil; strides : PInteger= nil; padding : string= 'valid'; data_format: string = ''): ILayer;

        /// <summary>
        /// Max pooling operation for 2D spatial data.
        /// Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis.
        /// The window is shifted by strides in each dimension. The resulting output when using "valid" padding option has a shape(number of rows or columns)
        /// of: output_shape = (input_shape - pool_size + 1) / strides)
        /// The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides
        /// </summary>
        /// <param name="pool_size">
        /// Integer or tuple of 2 integers, window size over which to take the maximum.
        /// (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
        /// </param>
        /// <param name="strides">
        /// Integer, tuple of 2 integers, or null. Strides values. Specifies how far the pooling window moves for each pooling step.
        /// If null, it will default to pool_size.
        /// </param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding.
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to
        /// inputs with shape (batch, channels, height, width).
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json.
        /// If you never set it, then it will be "channels_last"</param>
        /// <returns></returns>
        function MaxPooling2D(pool_size : PTFShape= nil; strides : PTFShape= nil; padding : string= 'valid'; data_format: string = ''): ILayer;

        /// <summary>
        /// Max pooling layer for 2D inputs (e.g. images).
        /// </summary>
        /// <param name="inputs">The tensor over which to pool. Must have rank 4.</param>
        /// <param name="pool_size">
        /// Integer or tuple of 2 integers, window size over which to take the maximum.
        /// (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
        /// </param>
        /// <param name="strides">
        /// Integer, tuple of 2 integers, or null. Strides values. Specifies how far the pooling window moves for each pooling step.
        /// If null, it will default to pool_size.
        /// </param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding.
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to
        /// inputs with shape (batch, channels, height, width).
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json.
        /// If you never set it, then it will be "channels_last"</param>
        /// <param name="name">A name for the layer</param>
        /// <returns></returns>
        function  max_pooling2d(inputs: TFTensor; pool_size : TArray<Integer>; strides: TArray<Integer>; padding: string = 'valid'; data_format: string = 'channels_last'; name: string = ''): TFTensor;

        function Permute(dims: TArray<Integer>): ILayer;

        /// <summary>
        ///
        /// </summary>
        /// <param name="scale"></param>
        /// <param name="offset"></param>
        /// <param name="input_shape"></param>
        /// <returns></returns>
        function Rescaling(scale: Single; offset : Single= 0; input_shape : PTFShape= nil): ILayer;

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
        /// <summary>
        /// Layer that concatenates a list of inputs.
        /// </summary>
        /// <param name="axis">Axis along which to concatenate.</param>
        /// <returns></returns>
        function Concatenate(axis: Integer = -1): ILayer;

        // ILayerApi.Reshaping
        //
        function Reshape(target_shape: TFShape): ILayer; overload;
        function Reshape(target_shape: TArray<TValue>): ILayer; overload;
        function UpSampling2D(size : PTFShape= nil; data_format : string = ''; interpolation: string = 'nearest'): ILayer;
        function ZeroPadding2D(padding: TNDArray): ILayer;

        /// <summary>
        /// Get an activation function layer from its name.
        /// </summary>
        /// <param name="name">The name of the activation function. One of linear, relu, sigmoid, and tanh.</param>
        /// <returns></returns>
        function GetActivationByName(name: string) : TActivation;

        /// <summary>
        /// Get an weights initializer from its name.
        /// </summary>
        /// <param name="name">The name of the initializer. One of zeros, ones, and glorot_uniform.</param>
        /// <returns></returns>
        function GetInitializerByName(name: string): IInitializer;

        property PreProcessing : IPreprocessing read ReadProc;
  end;

implementation
         uses Tensorflow;

{ LayersApi }

constructor LayersApi.Create;
begin
    FPreProcessing := TPreProcessing.Create
end;

function LayersApi.ReadProc: IPreprocessing;
begin
    Result := FPreProcessing;
end;

function LayersApi.Add: ILayer;
begin
    Result := Keras.Layer.Add.Create( MergeArgs.create);
end;

function LayersApi.Subtract: ILayer;
begin
    Result := Keras.Layer.Subtract.Create( MergeArgs.create);
end;

function LayersApi.Attention(use_scale: Boolean; score_mode: string; causal: Boolean; dropout: Single): ILayer;
var
  args : AttentionArgs;
begin
    args := AttentionArgs.Create;

    args.use_scale  := use_scale;
    args.score_mode := score_mode;
    args.causal     := causal;
    args.dropout    := dropout;

    Result := Keras.Layer.Attention.Create( args );
end;

function LayersApi.AveragePooling2D(pool_size, strides: TFShape; padding, data_format: string): ILayer;
 var
   args    : Pooling2DArgs;
   sPSize  : TFShape;
   sStrides: TFShape;
begin
    args   := Pooling2DArgs.Create;

    sPSize := TFShape.Create([2,2]);
    if not pool_size.IsNil then  sPSize:= pool_size ;

    sStrides := default(TFShape) ;
    if not strides.IsNil then  sStrides:= strides ;

    args.PoolSize  := sPSize;
    args.Strides   := sStrides;
    args.Padding   := padding;
    args.DataFormat:= data_format;

    Result := Keras.Layer.AveragePooling2D.Create( args );
end;

function LayersApi.AveragePooling2D(pool_size, strides: PTFShape; padding, data_format: string): ILayer;
 var
   args    : Pooling2DArgs;
   sPSize  : TFShape;
   sStrides: TFShape;
begin
    args   := Pooling2DArgs.Create;

    sPSize := TFShape.Create([2,2]);
    if pool_size <> nil then  sPSize:= pool_size^ ;

    sStrides := default(TFShape) ;
    if strides <> nil then  sStrides:= strides^ ;

    args.PoolSize  := sPSize;
    args.Strides   := sStrides;
    args.Padding   := padding;
    args.DataFormat:= data_format;

    Result := Keras.Layer.AveragePooling2D.Create( args );
end;

function LayersApi.BatchNormalization(axis: Integer; momentum, epsilon: Single; center, scale: Boolean; beta_initializer, gamma_initializer, moving_mean_initializer,
  moving_variance_initializer: IInitializer; trainable: Boolean; name: string; renorm: Boolean; renorm_momentum: Single): ILayer;
var
   args      : BatchNormalizationArgs;
   bIniz,
   gIniz,
   mMeanIniz,
   mVarIniz  : IInitializer ;
begin
    args   := BatchNormalizationArgs.Create;

    bIniz := beta_initializer;
    if not Assigned(bIniz) then  bIniz:= tf.zeros_initializer ;

    gIniz := gamma_initializer;
    if not Assigned(gIniz) then  gIniz:= tf.ones_initializer ;

    mMeanIniz := moving_mean_initializer;
    if not Assigned(mMeanIniz) then  mMeanIniz:= tf.zeros_initializer ;

    mVarIniz := moving_variance_initializer;
    if not Assigned(mVarIniz) then  mVarIniz:= tf.ones_initializer ;

    args.Axis                      := axis;
    args.Momentum                  := momentum;
    args.Epsilon                   := epsilon;
    args.Center                    := center;
    args.Scale                     := scale;
    args.BetaInitializer           := bIniz;
    args.GammaInitializer          := gIniz;
    args.MovingMeanInitializer     := mMeanIniz;
    args.MovingVarianceInitializer := mVarIniz;
    args.Renorm                    := renorm;
    args.RenormMomentum            := renorm_momentum;
    args.Trainable                 := trainable;
    args.Name                      := name;

    Result := Keras.Layer.BatchNormalization.Create( args );
end;

function LayersApi.Concatenate(axis: Integer): ILayer;
var
  args : MergeArgs;
begin
    args := MergeArgs.Create;
    args.Axis  := axis;

    Result := Keras.Layer.Concatenate.Create( args );
end;

function LayersApi.Conv1D(filters: Integer; kernel_size: TFShape; strides: Integer; padding, data_format: string; dilation_rate, groups: Integer; activation: string;
  use_bias: Boolean; kernel_initializer, bias_initializer: string): ILayer;
 var
   args    : Conv1DArgs;
   sKSize  : TFShape;
begin
    args   := Conv1DArgs.Create;

    sKSize := TFShape.Create([1,5]);
    if not kernel_size.isNull  then  sKSize := kernel_size ;

    args.Rank              := 1;
    args.Filters           := filters;
    args.KernelSize        := sKSize;
    args.Strides           := strides;
    args.Padding           := padding;
    args.DataFormat        := data_format;
    args.DilationRate      := dilation_rate;
    args.Groups            := groups;
    args.UseBias           := use_bias;
    args.Activation        := GetActivationByName(activation);
    args.KernelInitializer := GetInitializerByName(kernel_initializer);
    args.BiasInitializer   := GetInitializerByName(bias_initializer);

    Result := Keras.Layer.Conv1D.Create( args );
end;

function LayersApi.Conv1D(filters: Integer; kernel_size: TFShape; activation: string): ILayer;
begin
    Result := Conv1D(filters, kernel_size, 1, 'valid', 'channels_last', 1, 1, activation, true, 'glorot_uniform', 'zeros')
end;

function LayersApi.Conv2D(filters: Integer; kernel_size, strides: PTFShape; padding, data_format: string; dilation_rate: PTFShape; groups: Integer; activation: TActivation;
  use_bias: Boolean; kernel_initializer, bias_initializer: IInitializer; kernel_regularizer, bias_regularizer, activity_regularizer: IRegularizer): ILayer;
 var
   args     : Conv2DArgs;
   sKSize,
   sStrides,
   sDil_rate: TFShape;

   kIniz,
   bIniz    : IInitializer;
   aAct     : TActivation;
begin
    args   := Conv2DArgs.Create;

    sKSize := TFShape.Create([5,5]);
    if Assigned(kernel_size)  then  sKSize := kernel_size^ ;

    sStrides := TFShape.Create([1,1]);
    if Assigned(strides)  then  sStrides := strides^ ;

    sDil_rate := TFShape.Create([1,1]);
    if Assigned(dilation_rate)  then  sDil_rate := dilation_rate^ ;

    kIniz := kernel_initializer;
    if not Assigned(kIniz) then  kIniz:= tf.glorot_uniform_initializer ;

    bIniz := bias_initializer;
    if not Assigned(bIniz) then  bIniz:= tf.zeros_initializer ;

    aAct := activation;
    if not Assigned(aAct) then  aAct:= tf.keras.activations.Linear;

    args.Rank                := 2;
    args.Filters             := filters;
    args.KernelSize          := sKSize;
    args.Strides             := sStrides;
    args.Padding             := padding;
    args.DataFormat          := data_format;
    args.DilationRate        := sDil_rate;
    args.Groups              := groups;
    args.UseBias             := use_bias;
    args.KernelRegularizer   := kernel_regularizer;
    args.KernelInitializer   := kIniz;
    args.BiasInitializer     := bIniz;
    args.BiasRegularizer     := bias_regularizer;
    args.ActivityRegularizer := activity_regularizer;
    args.Activation          := aAct;

    Result := Keras.Layer.Conv2D.Create( args );
end;

function LayersApi.Conv2D(filters: Integer; kernel_size, strides: PTFShape; padding, data_format: string; dilation_rate: PTFShape; groups: Integer; activation: string;
  use_bias: Boolean; kernel_initializer, bias_initializer: string): ILayer;
 var
   args     : Conv2DArgs;
   sKSize,
   sStrides,
   sDil_rate: TFShape;
begin
    args   := Conv2DArgs.Create;

    sKSize := TFShape.Create([5,5]);
    if Assigned(kernel_size)  then  sKSize := kernel_size^ ;

    sStrides := TFShape.Create([1,1]);
    if Assigned(strides)  then  sStrides := strides^ ;

    sDil_rate := TFShape.Create([1,1]);
    if Assigned(dilation_rate)  then  sDil_rate := dilation_rate^ ;

    args.Rank                := 2;
    args.Filters             := filters;
    args.KernelSize          := sKSize;
    args.Strides             := sStrides;
    args.Padding             := padding;
    args.DataFormat          := data_format;
    args.DilationRate        := sDil_rate;
    args.Groups              := groups;
    args.UseBias             := use_bias;
    args.KernelInitializer   := GetInitializerByName(kernel_initializer);
    args.BiasInitializer     := GetInitializerByName(bias_initializer);
    args.Activation          := GetActivationByName(activation);

    Result := Keras.Layer.Conv2D.Create( args );
end;

function LayersApi.Conv2DTranspose(filters: Integer; kernel_size, strides: PTFShape; output_padding, data_format: string; dilation_rate: PTFShape; activation: string;
  use_bias: Boolean; kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer: string): ILayer;
 var
   args     : Conv2DArgs;
   sKSize,
   sStrides,
   sDil_rate: TFShape;
begin
    args   := Conv2DArgs.Create;

    sKSize := TFShape.Create([5,5]);
    if Assigned(kernel_size)  then  sKSize := kernel_size^ ;

    sStrides := TFShape.Create([1,1]);
    if Assigned(strides)  then  sStrides := strides^ ;

    sDil_rate := TFShape.Create([1,1]);
    if Assigned(dilation_rate)  then  sDil_rate := dilation_rate^ ;

    args.Rank                := 2;
    args.Filters             := filters;
    args.KernelSize          := sKSize;
    args.Strides             := sStrides;
    args.Padding             := output_padding;
    args.DataFormat          := data_format;
    args.DilationRate        := sDil_rate;
    args.UseBias             := use_bias;
    args.KernelInitializer   := GetInitializerByName(kernel_initializer);
    args.BiasInitializer     := GetInitializerByName(bias_initializer);
    args.Activation          := GetActivationByName(activation);

    Result := Keras.Layer.Conv2DTranspose.Create( args );
end;

function LayersApi.Cropping1D(cropping: TNDArray): ILayer;
 var
   args     : CroppingArgs;
begin
   args   := CroppingArgs.Create;

   args.cropping := cropping;

   Result := Keras.Layer.Cropping1D.Create( args );
end;

function LayersApi.Cropping2D(cropping: TNDArray; data_format: Cropping2DArgs.DataFormat): ILayer;
 var
   args     : Cropping2DArgs;
begin
   args   := Cropping2DArgs.Create;

   args.cropping    := cropping;
   args.data_format := data_format;

   Result := Keras.Layer.Cropping2D.Create( args );
end;

function LayersApi.Cropping3D(cropping: TNDArray; data_format: Cropping3DArgs.DataFormat): ILayer;
 var
   args     : Cropping3DArgs;
begin
   args   := Cropping3DArgs.Create;

   args.cropping    := cropping;
   args.data_format := data_format;

   Result := Keras.Layer.Cropping3D.Create( args );
end;

function LayersApi.Dense(units: Integer; activation: TActivation; kernel_initializer: IInitializer; use_bias: Boolean; bias_initializer: IInitializer;
  input_shape: PTFShape): ILayer;
var
   args  : DenseArgs;
   iShape: TFShape;
   aAct  : TActivation;
   kIniz,
   bIniz : IInitializer;
begin
   iShape := default(TFShape);
   if Assigned(input_shape) then  iShape := input_shape^;

   aAct := activation;
   if not Assigned(aAct) then  aAct:= tf.keras.activations.Linear;

   kIniz := kernel_initializer;
   if not Assigned(kIniz) then kIniz := tf.glorot_uniform_initializer;

   bIniz := bias_initializer;
   if not Assigned(bIniz) then
   begin
       if use_bias then  bIniz := tf.zeros_initializer
       else              bIniz := nil;
   end;

   args := DenseArgs.Create;

   args.Units             := units;
   args.Activation        := aAct;
   args.KernelInitializer := kIniz;
   args.BiasInitializer   := bIniz;
   args.InputShape        := iShape;

   Result := Keras.Layer.Dense.Create( args );
end;

function LayersApi.Dense(units: Integer; activation: string; input_shape: PTFShape): ILayer;
var
   args  : DenseArgs;
   iShape: TFShape;
begin
   iShape := default(TFShape);
   if Assigned(input_shape) then  iShape := input_shape^;


   args := DenseArgs.Create;

   args.Units      := units;
   args.Activation := GetActivationByName(activation);
   args.InputShape := iShape;

   Result := Keras.Layer.Dense.Create( args );
end;

function LayersApi.Dense(units: Integer): ILayer;
var
   args : DenseArgs;
begin
   args := DenseArgs.Create;

   args.Units      := units;
   args.Activation := GetActivationByName('linear');

   Result := Keras.Layer.Dense.Create( args );
end;

function LayersApi.Dropout(rate: Single; noise_shape: PTFShape; seed: pInteger): ILayer;
var
   args  : DropoutArgs;
   nShape: TFShape;
   iSeed : Nullable<Integer>;
begin
   nShape := default(TFShape);
   if Assigned(noise_shape) then  nShape := noise_shape^;
   if seed = nil then iSeed := nil
   else               iSeed := seed^;

   args := DropoutArgs.Create;

   args.Rate       := rate;
   args.NoiseShape := nShape;
   args.Seed       := iSeed;

   Result := Keras.Layer.Dropout.Create( args );
end;

function LayersApi.EinsumDense(equation: string; output_shape: TFShape; bias_axes: string; activation: TActivation; kernel_initializer, bias_initializer: IInitializer;
  kernel_regularizer, bias_regularizer, activity_regularizer: IRegularizer; kernel_constraint, bias_constraint: TProc): ILayer;
var
   args  : EinsumDenseArgs;
   kIniz,
   bIniz : IInitializer;
begin
   kIniz := kernel_initializer;
   if not Assigned(kIniz) then kIniz := tf.glorot_uniform_initializer;

   bIniz := bias_initializer;
   if not Assigned(bIniz) then bIniz := tf.zeros_initializer;

   args := EinsumDenseArgs.Create;

   args.Equation            := equation;
   args.OutputShape         := output_shape;
   args.BiasAxes            := bias_axes;
   args.Activation          := activation;
   args.KernelInitializer   := kIniz;
   args.BiasInitializer     := bIniz;
   args.KernelRegularizer   := kernel_regularizer;
   args.BiasRegularizer     := bias_regularizer;
   args.ActivityRegularizer := activity_regularizer;
   args.KernelConstraint    := kernel_constraint;
   args.BiasConstraint      := bias_constraint;

   Result := Keras.Layer.EinsumDense.Create( args );
end;

function LayersApi.ELU(alpha: Single): ILayer;
var
   args : ELUArgs;
begin
   args := ELUArgs.Create;

   args.Alpha      := alpha;

   Result := Keras.Layer.ELU.Create( args );
end;

function LayersApi.Embedding(input_dim, output_dim: Integer; embeddings_initializer: IInitializer; mask_zero: Boolean; input_shape: PTFShape; input_length: Integer): ILayer;
var
   args  : EmbeddingArgs;
   iShape: TFShape;
begin
   iShape := default(TFShape);
   if Assigned(input_shape) then  iShape := input_shape^;

   args := EmbeddingArgs.Create;

   args.InputDim              := input_dim;
   args.OutputDim             := output_dim;
   args.MaskZero              := mask_zero;
   args.InputShape            := iShape;
   args.InputLength           := input_length;
   args.EmbeddingsInitializer := embeddings_initializer;

   Result := Keras.Layer.Embedding.Create( args );
end;

function LayersApi.Flatten(data_format: string): ILayer;
var
   args : FlattenArgs;
begin
   args := FlattenArgs.Create;

   args.DataFormat      := data_format;

   Result := Keras.Layer.Flatten.Create( args );
end;

function LayersApi.GlobalAveragePooling1D(data_format: string): ILayer;
var
   args : Pooling1DArgs;
begin
   args := Pooling1DArgs.Create;

   args.DataFormat      := data_format;

   Result := Keras.Layer.GlobalAveragePooling1D.Create( args );
end;

function LayersApi.GlobalAveragePooling2D: ILayer;
var
   args : Pooling2DArgs;
begin
   args := Pooling2DArgs.Create;

   Result := Keras.Layer.GlobalAveragePooling2D.Create( args );
end;

function LayersApi.GlobalAveragePooling2D(data_format: string): ILayer;
var
   args : Pooling2DArgs;
begin
   args := Pooling2DArgs.Create;

   args.DataFormat      := data_format;

   Result := Keras.Layer.GlobalAveragePooling2D.Create( args );
end;

function LayersApi.GlobalMaxPooling1D(data_format: string): ILayer;
var
   args : Pooling1DArgs;
begin
   args := Pooling1DArgs.Create;

   args.DataFormat      := data_format;

   Result := Keras.Layer.GlobalMaxPooling1D.Create( args );
end;

function LayersApi.GlobalMaxPooling2D(data_format: string): ILayer;
var
   args : Pooling2DArgs;
begin
   args := Pooling2DArgs.Create;

   args.DataFormat      := data_format;

   Result := Keras.Layer.GlobalMaxPooling2D.Create( args );
end;

function LayersApi.Input(shape: TFShape; batch_size: Integer; name: string; sparse, ragged: Boolean): TFTensors;
var
  args : InputLayerArgs;
begin
   args := InputLayerArgs.Create;
   args.InputShape  := shape;
   args.BatchSize   := batch_size;
   args.Name        := name;
   args.Sparse      := sparse;
   args.Ragged      := ragged ;

   var input_layer := Keras.Layer.InputLayer.Create(args);

   Result := input_layer.InboundNodes[0].Outputs;
end;

function LayersApi.InputLayer(input_shape: TFShape; name: string; sparse, ragged: Boolean): ILayer;
var
  args : InputLayerArgs;
begin
   args := InputLayerArgs.Create;

   args.InputShape  := input_shape;
   args.Name        := name;
   args.Sparse      := sparse;
   args.Ragged      := ragged ;

   Result := Keras.Layer.InputLayer.Create(args);
end;

function LayersApi.LayerNormalization(axis: TAxis; epsilon: Single; center, scale: Boolean; beta_initializer, gamma_initializer: IInitializer): ILayer;
var
  args : LayerNormalizationArgs;
  bIniz: IInitializer;
begin
   bIniz := beta_initializer;
   if not Assigned(bIniz) then bIniz := tf.zeros_initializer;

   args := LayerNormalizationArgs.Create;

   args.Axis            := axis;
   args.Epsilon         := epsilon;
   args.Center          := center;
   args.Scale           := scale;
   args.BetaInitializer := bIniz;

   Result := Keras.Layer.LayerNormalization.Create(args);
end;

function LayersApi.LeakyReLU(alpha: Single): ILayer;
var
  args : LeakyReLuArgs;
begin
   args        := LeakyReLuArgs.Create;
   args.Alpha  := alpha;

   Result := Keras.Layer.LeakyReLu.Create(args);
end;

function LayersApi.SimpleRNN(units: Integer; activation, kernel_initializer, recurrent_initializer, bias_initializer: string;return_sequences : Boolean; return_state : Boolean): ILayer;
var
  args : SimpleRNNArgs;
begin
   args  := SimpleRNNArgs.Create;

   args.Units               := units;
   args.Activation          := GetActivationByName(activation);
   args.KernelInitializer   := GetInitializerByName(kernel_initializer);
   args.RecurrentInitializer:= GetInitializerByName(recurrent_initializer);
   args.BiasInitializer     := GetInitializerByName(bias_initializer);
   args.ReturnSequences     := return_sequences;
   args.ReturnState         := return_state;

   Result := Keras.Layer.SimpleRNN.Create(args);
end;

function LayersApi.LSTM(units: Integer; activation, recurrent_activation: TActivation; use_bias: Boolean; kernel_initializer, recurrent_initializer,
  bias_initializer: IInitializer; unit_forget_bias: Boolean; dropout, recurrent_dropout: Single; _implementation: Integer; return_sequences, return_state, go_backwards, stateful,
  time_major, unroll: Boolean): ILayer;
var
  args  : LSTMArgs;
  aAct,
  rAct  : TActivation;
  kIniz,
  RIniz,
  bIniz : IInitializer;
begin
   aAct := activation;
   if not Assigned(aAct) then aAct :=  tf.keras.activations.Tanh;
   rAct := recurrent_activation;
   if not Assigned(rAct) then rAct :=  tf.keras.activations.Sigmoid;

   kIniz := kernel_initializer;
   if not Assigned(kIniz) then kIniz := tf.glorot_uniform_initializer;

   RIniz := recurrent_initializer;
   if not Assigned(RIniz) then RIniz := tf.orthogonal_initializer;

   bIniz := bias_initializer;
   if not Assigned(bIniz) then bIniz := tf.zeros_initializer;

   args := LSTMArgs.Create;

   args.Units                := units;
   args.Activation           := aAct;
   args.RecurrentActivation  := rAct;
   args.KernelInitializer    := kIniz;
   args.RecurrentInitializer := RIniz;
   args.BiasInitializer      := bIniz;
   args.Dropout              := dropout;
   args.RecurrentDropout     := recurrent_dropout;
   args.&Implementation      := _implementation;
   args.ReturnSequences      := return_sequences;
   args.ReturnState          := return_state;
   args.GoBackwards          := go_backwards;
   args.Stateful             := stateful;
   args.TimeMajor            := time_major;
   args.Unroll               := unroll ;

   Result := Keras.Layer.LSTM.Create(args);

end;

function LayersApi.MaxPooling1D(pool_size, strides: PInteger; padding, data_format: string): ILayer;
var
  args     : Pooling1DArgs;
  iPSize   : Integer;
  iStrides : Integer;
begin
   args  := Pooling1DArgs.Create;

   iPSize := 2;
   if Assigned(pool_size) then iPSize := pool_size^ ;

   iStrides := 2;
   if Assigned(strides) then iStrides := strides^
   else begin
        if Assigned(pool_size) then iStrides := pool_size^
   end;

   args.PoolSize  := iPSize;
   args.Strides   := iStrides;
   args.Padding   := padding;
   args.DataFormat:= data_format;

   Result := Keras.Layer.MaxPooling1D.Create(args);
end;

function LayersApi.MaxPooling2D(pool_size, strides: PTFShape; padding, data_format: string): ILayer;
var
  args     : Pooling2DArgs;
  sPSize   : TFShape;
  sStrides : TFShape;
begin
   args  := Pooling2DArgs.Create;

   sPSize := TFShape.Create([2,2]);
   if Assigned(pool_size) then sPSize := pool_size^ ;

   sStrides := default(TFShape);
   if Assigned(strides) then sStrides := strides^;

   args.PoolSize  := sPSize;
   args.Strides   := sStrides;
   args.Padding   := padding;
   args.DataFormat:= data_format;

   Result := Keras.Layer.MaxPooling2D(args);
end;

function LayersApi.max_pooling2d(inputs: TFTensor; pool_size, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
var
  args     : Pooling2DArgs;
begin
   args  := Pooling2DArgs.Create;

   args.PoolSize  := pool_size;
   args.Strides   := strides;
   args.Padding   := padding;
   args.DataFormat:= data_format;

   var layer := Keras.Layer.MaxPooling2D.Create(args);

   Result := layer.Apply( TFTensors.Create(inputs) ).First;
end;

function LayersApi.MultiHeadAttention(num_heads, key_dim: Integer; value_dim: PInteger; dropout: Single; use_bias: Boolean; output_shape, attention_axes: PTFShape;
  kernel_initializer, bias_initializer: IInitializer; kernel_regularizer, bias_regularizer, activity_regularizer: IRegularizer; kernel_constraint,
  bias_constraint: TProc): ILayer;
var
  args   : MultiHeadAttentionArgs;
  iV_dim : Nullable<Integer>;
  oShape : TFShape;
  aAxes  : TFShape;
  kIniz,
  bIniz  : IInitializer;
begin
   args  := MultiHeadAttentionArgs.Create;

   iV_dim := nil;
   if Assigned(value_dim) then iV_dim := value_dim^;

   oShape := default(TFShape);
   if Assigned(output_shape) then oShape := output_shape^ ;

   aAxes := default(TFShape);
   if Assigned(attention_axes) then aAxes := attention_axes^;

   kIniz := kernel_initializer;
   if not Assigned(kIniz) then kIniz := tf.glorot_uniform_initializer;

   bIniz := bias_initializer;
   if not Assigned(bIniz) then bIniz := tf.zeros_initializer;

   args.NumHeads            := num_heads;
   args.KeyDim              := key_dim;
   args.ValueDim            := iV_dim;
   args.Dropout             := dropout;
   args.UseBias             := use_bias;
   args.OutputShape         := oShape;
   args.AttentionAxis       := aAxes;
   args.KernelInitializer   := kIniz;
   args.BiasInitializer     := bIniz;
   args.KernelRegularizer   := kernel_regularizer;
   args.BiasRegularizer     := bias_regularizer;
   args.ActivityRegularizer := activity_regularizer;
   args.KernelConstraint    := kernel_constraint;
   args.BiasConstraint      := bias_constraint;

   Result := Keras.Layer.MultiHeadAttention.Create(args);

end;

function LayersApi.Permute(dims: TArray<Integer>): ILayer;
var
   args : PermuteArgs;
begin
   args     := PermuteArgs.Create;
   args.dims := dims;

   Result := Keras.Layer.Permute.Create( args );
end;

function LayersApi.Rescaling(scale, offset: Single; input_shape: PTFShape): ILayer;
var
   args   : RescalingArgs;
   iShape : TFShape;
begin
   iShape := default(TFShape);
   if Assigned(input_shape) then iShape := input_shape^ ;

   args := RescalingArgs.Create;

   args.Scale     := scale;
   args.Offset    := offset;
   args.InputShape:= iShape;

   Result := Keras.Layer.Rescaling.Create( args );
end;

function LayersApi.Reshape(target_shape: TArray<TValue>): ILayer;
var
   args : ReshapeArgs;
begin
   args                    := ReshapeArgs.Create;
   args.TargetShapeObjects := target_shape;

   Result := Keras.Layer.Reshape.Create( args );
end;

function LayersApi.Reshape(target_shape: TFShape): ILayer;
var
   args   : ReshapeArgs;
begin
   args             := ReshapeArgs.Create;
   args.TargetShape := target_shape;

   Result := Keras.Layer.Reshape.Create( args );
end;

function LayersApi.SELU: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.SELU.Create( args );
end;

function LayersApi.Softmax(axis: TAxis): ILayer;
var
   args : SoftmaxArgs;
begin
   args     := SoftmaxArgs.Create;
   args.axis := axis;

   Result := Keras.Layer.Softmax.Create( args );
end;

function LayersApi.Softplus: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.Softplus.Create( args );
end;

function LayersApi.Softsign: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.Softsign.Create( args );
end;

function LayersApi.HardSigmoid: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.HardSigmoid.Create( args );
end;

function LayersApi.Swish: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.Swish.Create( args );
end;

function LayersApi.Tanh: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.Tanh.Create( args );
end;

function LayersApi.Exponential: ILayer;
var
   args : LayerArgs;
begin
   args := LayerArgs.Create;
   Result := Keras.Layer.Exponential.Create( args );
end;

function LayersApi.UpSampling2D(size: PTFShape; data_format, interpolation: string): ILayer;
var
   args  : UpSampling2DArgs;
   sSize : TFShape;
begin
   sSize := TFShape.Create([2,2]);
   if Assigned(size) then sSize := size^ ;

   args     := UpSampling2DArgs.Create;
   args.Size:= sSize;

   Result := Keras.Layer.UpSampling2D.Create( args );
end;

function LayersApi.ZeroPadding2D(padding: TNDArray): ILayer;
var
   args : ZeroPadding2DArgs;
begin
   args         := ZeroPadding2DArgs.Create;
   args.Padding := padding;

   Result := Keras.Layer.ZeroPadding2D.Create( args );
end;

function LayersApi.GetActivationByName(name: string): TActivation;
begin
    if      name = 'linear'  then Result := tf.keras.activations.Linear
    else if name = 'relu'    then Result := tf.keras.activations.Relu
    else if name = 'sigmoid' then Result := tf.keras.activations.Sigmoid
    else if name = 'tanh'    then Result := tf.keras.activations.Tanh
    else if name = 'softmax' then Result := tf.keras.activations.Softmax
    else raise Exception.Create('Activation '+name+' not found')
end;

function LayersApi.GetInitializerByName(name: string): IInitializer;
begin
    if      name = 'glorot_uniform'  then Result := tf.glorot_uniform_initializer
    else if name = 'zeros'           then Result := tf.zeros_initializer
    else if name = 'ones'            then Result := tf.ones_initializer
    else if name = 'orthogonal'      then Result := tf.orthogonal_initializer
    else                                  Result := tf.glorot_uniform_initializer
end;

end.
