unit Keras.Layer;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}
{$WARN SYMBOL_DEPRECATED OFF}

interface
       uses System.SysUtils,
            System.Rtti,
            System.TypInfo,
            System.JSON,
            System.Generics.Collections,
            System.RegularExpressions,

            Spring,
            Spring.Collections,
            Spring.Collections.Enumerable,

            Neon.Core.Persistence,
            Neon.Core.Persistence.JSON,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,
            TensorFlow.Core,
            Numpy.Axis,
            TensorFlow.Training,
            TensorFlow.Initializer,
            TensorFlow.NnOps,

            Keras.Core,
            Keras.Data,

            TensorFlow.Proto;


threadvar  tls_CallContext : TCallContext;

type
{$REGION 'Layer'}
  /// <summary>
  /// Base layer class.
  /// A layer is a class implementing common neural networks operations, such
  /// as convolution, batch norm, etc. These operations require managing weights,
  /// losses, updates, and inter-layer connectivity.
  /// </summary>
  Layer = class(AutoTrackable, ILayer)
    private
       Fargs           : LayerArgs;
       Fdynamic        : Boolean;
       FSupportsMasking: Boolean;
       FNodesByDepth   : TDictionary<Integer, TList<INode>>;

       function GetBuilt    : Boolean;
       function GetTrainable: Boolean;
       function GetName:string;
       function GetDtype: TF_DataType;
       function GetTrainW: TList<IVariableV1>;
       function GetTrainVars: TList<IVariableV1>;
       function GetNonTrainVars: TList<IVariableV1>;
       function GetNotTrainW: TList<IVariableV1>;
       function GetWeights   : TList<IVariableV1>;
       procedure SetWeights(value: TList<IVariableV1>);
       function GetVars: TList<IVariableV1>;
       function GetBatchShape: TFShape;
       function GetInNodes: TList<INode>;
       function GetOutNodes: TList<INode>;
       function GetCallCtx: TCallContext;
       function GetInput: TArray<TFTensor>;
       function GetoutShape: TFShape;
       //
       procedure _set_connectivity_metadata_(inputs: TFTensors; outputs: TFTensors);
       procedure _handle_activity_regularization(inputs: TFTensors; outputs: TFTensors);
       procedure _set_mask_metadata(inputs: TFTensors; outputs: TFTensors; previous_mask: TFTensors);
       function  compute_mask(inputs: TFTensor; mask: TFTensor = nil): TFTensor;
       function  GetLayers: TList<ILayer>; virtual;
       procedure SetCallCtx(const Value: TCallContext);
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
    protected
       Fbuilt    : Boolean;
       Fstateful : Boolean;
       FId       : Integer;
       Ftrainable_state          : TDictionary<ILayer, boolean>;
       Fcompiled_trainable_state : TDictionary<ILayer, boolean>;
       /// <summary>
       /// Provides information about which inputs are compatible with the layer.
       /// </summary>
       FinputSpec              : TInputSpec;
       FTrainableWeights       : TList<IVariableV1>;
       FNonTrainableWeights    : TList<IVariableV1>;
       FName                   : string;
       FBase_Name              : string;
       FcomputePreviousMask    : Boolean;
       Fupdates                : TList<TFOperation>;
       FInboundNodes           : TList<INode>;
       FOutboundNodes          : TList<INode>;
       Fself_tracked_trackables: TList<ILayer>;
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;

       function add_weight( name            : string;
                            shape           : TFShape;
                            dtype           : TF_DataType = TF_DataType.TF_FLOAT;
                            initializer     : IInitializer = nil;
                            regularizer     : IRegularizer = nil;
                            synchronization : TVariableSynchronization= VARIABLE_SYNCHRONIZATION_AUTO;
                            aggregation     : TVariableAggregation    = VARIABLE_AGGREGATION_NONE;
                            trainable       : Boolean= true;
                            getter          : TFunc<VariableArgs, IVariableV1>= nil): IVariableV1; virtual;
       /// <summary>
       /// Get the `trainable` state of each sublayer.
       /// </summary>
       /// <returns></returns>
       function _get_trainable_state: TDictionary<ILayer, boolean>;
       procedure StackLayers(_layers : TArray<ILayer>);
       procedure MaybeBuild(inputs: TFTensors);
       function  ComputeOutputShape(input_shape: TFShape): TFShape; virtual;
       function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; virtual;
       function  _name_scope: string;virtual;
       procedure Build(input_shape: TFShape); virtual;
       procedure add_loss(losses: TFunc<TFTensor>); virtual;
       procedure _init_set_name(_name: string; zero_based: Boolean = true); virtual;
       function  _gather_children_variables(include_trainable: Boolean = false; include_non_trainable: Boolean = false): TList<IVariableV1>;
       procedure Initialize(_args: LayerArgs); virtual;
    public
        constructor Create(_args: LayerArgs);
        destructor Destroy; override;
        /// <summary>
        /// Loads all layer weights, either from a TensorFlow or an HDF5 weight file.
        /// </summary>
        /// <param name="filepath"></param>
        procedure load_weights(filepath: string);
        function  _in_functional_construction_mode(inputs: TFTensors): Boolean;
        procedure SetConnectivityMetadata(inputs: TFTensors; outputs: TFTensors);
        procedure _handle_weight_regularization(name: string; variable: IVariableV1; regularizer: IRegularizer);
        function  Apply(inputs: TFTensors; state: TFTensor = nil; training: Boolean = false): TFTensors;
        function  count_params: Integer;
        function  get_config : LayerArgs;
        function  FunctionalConstructionCall(inputs: TFTensors): TFTensors;
        function _flatten_layers(recursive : Boolean= true; include_self : Boolean= true): TArray<ILayer>;
        {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}

        /// <summary>
        /// Indicates whether `build` needs to be called upon layer call, to create
        /// the layer's weights.
        /// </summary>
        property Built   : Boolean   read GetBuilt;
        /// <summary>
        /// Arguments initialize layer.
        /// </summary>
        property args     : LayerArgs   read Fargs;
        property Trainable: Boolean     read GetTrainable;
        property DType    : TF_DataType read GetDtype;
        /// <summary>
        /// A stateful layer is a layer whose updates are run during inference too,
        /// for instance stateful RNNs.
        /// </summary>
        property stateful             : Boolean             read Fstateful;
        property dynamicc             : Boolean             read Fdynamic;
        property SupportsMasking      : Boolean             read FSupportsMasking;
        property inputSpec            : TInputSpec          read FinputSpec;
        property TrainableVariables   : TList<IVariableV1>  read GetTrainVars;
        property NonTrainableVariables: TList<IVariableV1>  read GetNonTrainVars;
        property Variables            : TList<IVariableV1>  read GetVars;
        property TrainableWeights     : TList<IVariableV1>  read GetTrainW;
        property NonTrainableWeights  : TList<IVariableV1>  read GetNotTrainW;
        property weights              : TList<IVariableV1>  read GetWeights write SetWeights;
        property Id                   : Integer             read FId;
        property Name                 : string              read GetName;
        property BatchInputShape      : TFShape             read GetBatchShape;
        property InboundNodes         : TList<INode>        read GetInNodes;
        property OutboundNodes        : TList<INode>        read GetOutNodes;
        property CallContext          : TCallContext        read GetCallCtx write SetCallCtx;
        property input                : TArray<TFTensor>    read GetInput;
        property NodesByDepth         : TDictionary<Integer, TList<INode>> read FNodesByDepth write FNodesByDepth;
        property OutputShape          : TFShape             read GetoutShape;
        property Layers               : TList<ILayer>       read GetLayers;

  end;
{$ENDREGION}

  {$REGION 'Activation'}
  /// <summary>
  /// ELU Layer:
  /// x = 0 when x > 0, x = alpha( e^x-1 ) elsewhere
  /// </summary>
  ELU = class(Layer)
  private
      function GetAlpha: Single;
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args  : ELUArgs;

      constructor Create(_args: ELUArgs);

      property alpha : Single read GetAlpha;
  end;

  Exponential = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  HardSigmoid = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  /// <summary>
  /// Leaky version of a Rectified Linear Unit.
  /// </summary>
  LeakyReLu = class(Layer)
  private
      function GetAlpha: Single;
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args  : LeakyReLuArgs;

      constructor Create(_args: LeakyReLuArgs);

      property alpha : Single read GetAlpha;
  end;

  /// <summary>
  /// SELU Layer:
  /// similar to ELU, but has pre-defined alpha and scale
  /// </summary>
  SELU = class(Layer)
    protected
      const alpha : Single = 1.67326324;
      const scale : Single = 1.05070098;

      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  Softmax = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      axis : TAxis;

      constructor Create(_args: SoftmaxArgs);
  end;

  Softplus = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  Swish = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  Softsign = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;

  Tanh = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public

      constructor Create(_args: LayerArgs);
  end;
  {$ENDREGION}

  {$REGION 'Regularization'}
  Dropout = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args  : DropoutArgs;
      function get_noise_shape(inputs: TFTensor ): TFTensor;

      constructor Create(_args: DropoutArgs);
  end;
  {$ENDREGION}

  {$REGION 'Core'}
  /// <summary>
  /// Just your regular densely-connected NN layer.
  /// </summary>
  Dense  = class(Layer)
  private
      function getAct: TActivation;
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args   : DenseArgs;
      kernel : IVariableV1;
      bias   : IVariableV1;

      constructor Create(_args: DenseArgs);

      property activation : TActivation read getAct;
  end;

  // A layer that uses `tf.einsum` as the backing computation.
  //   This layer can perform einsum calculations of arbitrary dimensionality.
  //   Args:
  //     equation: An equation describing the einsum to perform. This equation must
  //       be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
  //       `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
  //       expression sequence.
  //     output_shape: The expected shape of the output tensor (excluding the batch
  //       dimension and any dimensions represented by ellipses). You can specify
  //       None for any dimension that is unknown or can be inferred from the input
  //       shape.
  //     activation: Activation function to use. If you don't specify anything, no
  //       activation is applied (that is, a "linear" activation: `a(x) = x`).
  //     bias_axes: A string containing the output dimension(s) to apply a bias to.
  //       Each character in the `bias_axes` string should correspond to a character
  //       in the output portion of the `equation` string.
  //     kernel_initializer: Initializer for the `kernel` weights matrix.
  //     bias_initializer: Initializer for the bias vector.
  //     kernel_regularizer: Regularizer function applied to the `kernel` weights
  //       matrix.
  //     bias_regularizer: Regularizer function applied to the bias vector.
  //     activity_regularizer: Regularizer function applied to the output of the
  //       layer (its "activation").
  //     kernel_constraint: Constraint function applied to the `kernel` weights
  //       matrix.
  //     bias_constraint: Constraint function applied to the bias vector.
  //   Examples:
  //   **Biased dense layer with einsums**
  //   This example shows how to instantiate a standard Keras dense layer using
  //   einsum operations. This example is equivalent to
  //   `tf.keras.layers.Dense(64, use_bias=True)`.
  //   >>> layer = tf.keras.layers.EinsumDense("ab,bc->ac",
  //   ...                                     output_shape=64,
  //   ...                                     bias_axes="c")
  //   >>> input_tensor = tf.keras.Input(shape=[32])
  //   >>> output_tensor = layer(input_tensor)
  //   >>> output_tensor
  //   <... shape=(None, 64) dtype=...>
  //   **Applying a dense layer to a sequence**
  //   This example shows how to instantiate a layer that applies the same dense
  //   operation to every element in a sequence. Here, the `output_shape` has two
  //   values (since there are two non-batch dimensions in the output); the first
  //   dimension in the `output_shape` is `None`, because the sequence dimension `b`
  //   has an unknown shape.
  //   >>> layer = tf.keras.layers.EinsumDense("abc,cd->abd",
  //   ...                                     output_shape=(None, 64),
  //   ...                                     bias_axes="d")
  //   >>> input_tensor = tf.keras.Input(shape=[32, 128])
  //   >>> output_tensor = layer(input_tensor)
  //   >>> output_tensor
  //   <... shape=(None, 32, 64) dtype=...>
  //   **Applying a dense layer to a sequence using ellipses**
  //   This example shows how to instantiate a layer that applies the same dense
  //   operation to every element in a sequence, but uses the ellipsis notation
  //   instead of specifying the batch and sequence dimensions.
  //   Because we are using ellipsis notation and have specified only one axis, the
  //   `output_shape` arg is a single value. When instantiated in this way, the layer
  //   can handle any number of sequence dimensions - including the case where no
  //   sequence dimension exists.
  //   >>> layer = tf.keras.layers.EinsumDense("...x,xy->...y",
  //   ...                                     output_shape=64,
  //   ...                                     bias_axes="y")
  //   >>> input_tensor = tf.keras.Input(shape=[32, 128])
  //   >>> output_tensor = layer(input_tensor)
  //   >>> output_tensor
  //   <... shape=(None, 32, 64) dtype=...>
  //
  EinsumDense = class(Layer)
    private
      Fequation             : string;
      Factivation           : TActivation;
      Fbias                 : IVariableV1 ;
      Fkernel               : IVariableV1 ;
      Fbias_axes            : string;
      Fkernel_initializer   : IInitializer;
      Fbias_initializer     : IInitializer;
      Fkernel_constraint    : TProc;
      Fbias_constraint      : TProc;
      Fbias_regularizer     : IRegularizer;
      Fkernel_regularizer   : IRegularizer;
      Ffull_output_shape    : TFShape;
      Fpartial_output_shape : TFshape;

    protected
      procedure Build(input_shape: TFShape); override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      /// <summary>
      /// Analyzes an einsum string to determine the required weight shape.
      /// </summary>
      class function _analyze_einsum_string(equation: string; bias_axes: string; input_shape: TFShape; output_shape: TFShape): Tuple<TFShape, TFShape, TFShape>;
      /// <summary>
      /// Analyze an pre-split einsum string to find the weight shape.
      /// </summary>
      class function _analyze_split_string(split_string: TMatch; bias_axes: string; input_shape: TFShape; output_shape: TFShape; left_elided : Boolean= false): Tuple<TFShape, TFShape, TFShape>;

      constructor Create(_args: EinsumDenseArgs);

      property equation             : string        read Fequation;
      property activation           : TActivation   read Factivation;
      property bias                 : IVariableV1   read Fbias;
      property kernel               : IVariableV1   read Fkernel;
      property bias_axes            : string        read Fbias_axes;
      property kernel_initializer   : IInitializer  read Fkernel_initializer;
      property bias_initializer     : IInitializer  read Fbias_initializer;
      property kernel_constraint    : TProc         read Fkernel_constraint;
      property bias_constraint      : TProc         read Fbias_constraint;
      property bias_regularizer     : IRegularizer  read Fbias_regularizer;
      property kernel_regularizer   : IRegularizer  read Fkernel_regularizer;
      property full_output_shape    : TFShape       read Ffull_output_shape;
      property partial_output_shape : TFshape       read Fpartial_output_shape;
  end;

  /// <summary>
  /// Turns positive integers (indexes) into dense vectors of fixed size.
  /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
  /// </summary>
  Embedding = class(Layer)
    private
      function getInput_dim: Integer;
      function getmask_zero: Boolean;
      function getoutput_dim: Integer;

    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args      : EmbeddingArgs;
      embeddings: IVariableV1;
      embeddings_initializer : IInitializer;

      constructor Create(_args: EmbeddingArgs);

      property input_dim : Integer  read getInput_dim;
      property output_dim: Integer  read getoutput_dim;
      property mask_zero : Boolean  read getmask_zero;
  end;

  /// <summary>
  /// Layer to be used as an entry point into a Network (a graph of layers).
  /// </summary>
  InputLayer = class(Layer)
    public
      args         : InputLayerArgs;
      isPlaceholder: Boolean;
      typeSpec     : TensorSpec;

      constructor Create(_args: InputLayerArgs);
  end;

  {$ENDREGION}

  {$REGION 'Attention'}
  /// <summary>
  /// Base class for attention layers that can be used in sequence DNN/CNN models.
  ///This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
  ///Attention is formed by three tensors: Query, Key and Value.
  /// </summary>

  /// <summary>
  /// Base Attention class for Dense networks.
  /// This class is suitable for Dense or CNN networks, and not for RNN networks.
  /// Implementations of attention mechanisms should inherit from this class, and
  /// reuse the `apply_attention_scores()` method.
  /// </summary>
  BaseDenseAttention = class(Layer)
  private
    function getCausal: Boolean;
    function getDropout: Single;
    protected
      Fsupports_masking : Boolean;

      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  Call(inputs: TFTensors; mask:TFTensors = nil; training : pBoolean= nil; return_attention_scores : Boolean= false): TFTensors; reintroduce; overload ;
      /// <summary>
      /// Calculates attention scores.
      /// </summary>
      /// <param name="query">query: Query tensor of shape `[batch_size, Tq, dim]`.</param>
      /// <param name="key">key: Key tensor of shape `[batch_size, Tv, dim]`.</param>
      /// <returns>Tensor of shape `[batch_size, Tq, Tv]`.</returns>
      function _calculate_scores(query: TFTensor; key: TFTensor): TFTensor; virtual;
    public
      args : BaseDenseAttentionArgs;

      constructor Create(_args: BaseDenseAttentionArgs);
      /// <summary>
      /// Applies attention scores to the given value tensor.
      /// To use this method in your attention layer, follow the steps:
      /// <para>
      ///     * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      ///       `[batch_size, Tv]` to calculate the attention `scores`.
      /// </para>
      /// <para>
      ///     * Pass `scores` and `value` tensors to this method. The method applies
      ///       `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      ///       returns `matmul(attention_distribution, value).
      /// </para>
      /// <para>
      ///     * Apply `query_mask` and return the result.
      /// </para>
      /// </summary>
      /// <param name="scores">Scores float tensor of shape `[batch_size, Tq, Tv]`.</param>
      /// <param name="value">Value tensor of shape `[batch_size, Tv, dim]`.</param>
      /// <param name="scores_mask">
      /// A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
      /// [batch_size, Tq, Tv]`. If given, scores at positions where
      /// `scores_mask==False` do not contribute to the result. It must contain
      /// at least one `True` value in each line along the last dimension.
      /// </param>
      /// <param name="training">
      /// Boolean indicating whether the layer should behave in
      /// training mode (adding dropout) or in inference mode (no dropout).
      /// </param>
      /// <returns>
      /// <para>
      /// Tensor of shape `[batch_size, Tq, dim]`.
      /// </para>
      /// <para>
      /// Attention scores after masking and softmax with shape
      /// [batch_size, Tq, Tv]`.
      /// </para>
      /// </returns>
      function _apply_scores(scores: TFTensor; value: TFTensor; scores_mask: TFTensor = nil; training : PBoolean= nil): Tuple<TFTensor,TFTensor>;
      function compute_mask(inputs: TFTensors; mask: TFTensors = nil) : TFTensor;
      /// <summary>
      /// Validates arguments of the call method.
      /// </summary>
      procedure _validate_call_args(inputs: TFTensors; mask: TFTensors) ;
      class function _lower_triangular_mask(shape: TFShape): TFTensor;
      class function _merge_masks(x: TFTensor; y: TFTensor): TFTensor;
      function get_config: LayerArgs;

      property causal : Boolean read getCausal;
      property dropout: Single  read getDropout;
  end;

  /// <summary>
  /// Dot-product attention layer, a.k.a. Luong-style attention.
  /// Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
  /// shape `[batch_size, Tv, dim]` and `key` tensor of shape
  /// `[batch_size, Tv, dim]`. The calculation follows the steps:
  /// <para>
  /// 1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
  ///    product: `scores = tf.matmul(query, key, transpose_b=True)`.
  /// </para>
  /// <para>
  /// 2. Use scores to calculate a distribution with shape
  ///    `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
  /// </para>
  /// <para>
  /// 3. Use `distribution` to create a linear combination of `value` with
  ///    shape `[batch_size, Tq, dim]`:
  ///    `return tf.matmul(distribution, value)`.
  /// </para>
  /// </summary>
  /// <example> 0
  /// <code>
  /// //Variable-length int sequences.
  /// var query_input = keras.Input((1000), dtype: TF_DataType.TF_INT32);
  /// var value_input = keras.Input((1000), dtype: TF_DataType.TF_INT32);
  /// // Embedding lookup.
  /// var token_embedding = keras.layers.Embedding(input_dim: 1000, output_dim: 64);
  /// // Query embeddings of shape [batch_size, Tq, dimension].
  /// var query_embeddings = token_embedding.Apply(query_input);
  /// // Value embeddings of shape [batch_size, Tv, dimension].
  /// var value_embeddings = token_embedding.Apply(value_input);
  /// // CNN layer.
  /// var cnn_layer = keras.layers.Conv1D(
  ///     filters: 100,
  ///     kernel_size: 4,
  ///     // Use 'same' padding so outputs have the same shape as inputs.
  ///     padding: "same");
  /// var cnn_layer2 = keras.layers.Conv1D(
  ///     filters: 100,
  ///     kernel_size: 4,
  ///     // Use 'same' padding so outputs have the same shape as inputs.
  ///     padding: "same");
  /// // Query encoding of shape [batch_size, Tq, filters].
  /// var query_seq_encoding = cnn_layer.Apply(query_embeddings);
  /// // Value encoding of shape [batch_size, Tv, filters].
  /// var value_seq_encoding = cnn_layer.Apply(value_embeddings);
  /// // Query-value attention of shape [batch_size, Tq, filters].
  /// var query_value_attention_seq = keras.layers.Attention().Apply(
  ///    (query_seq_encoding, value_seq_encoding));
  /// // Reduce over the sequence axis to produce encodings of shape
  /// // [batch_size, filters].
  /// var query_encoding = keras.layers.GlobalAveragePooling1D().Apply(
  ///     query_seq_encoding);
  /// var query_value_attention = keras.layers.GlobalAveragePooling1D().Apply(
  ///     query_value_attention_seq);
  /// // Concatenate query and document encodings to produce a DNN input layer.
  /// var input_layer = keras.layers.Concatenate().Apply(
  ///     (query_encoding, query_value_attention));
  /// // Add DNN layers, and create Model.
  /// // ...
  /// </code>
  /// </example>
  Attention = class(BaseDenseAttention)
    private
    function GetScoreMode: string;
    function GetUse_Scale: Boolean;

    protected
      // Creates variable when `use_scale` is True or `score_mode` is `concat`.
      procedure Build(input_shape: TFShape); override;

    public
      concat_score_weight : IVariableV1 ;
      scale               : IVariableV1 ;
      args                : AttentionArgs;

      constructor Create(_args: AttentionArgs);
      /// <summary>
      /// Calculates attention scores as a query-key dot product.
      /// </summary>
      /// <param name="query">query: Query tensor of shape `[batch_size, Tq, dim]`.</param>
      /// <param name="key">key: Key tensor of shape `[batch_size, Tv, dim]`.</param>
      /// <returns>Tensor of shape `[batch_size, Tq, Tv]`.</returns>
      function _calculate_scores(query: TFTensor; key: TFTensor): TFTensor; override;
      function get_config: LayerArgs;

      property score_mode: string read  GetScoreMode;
      property use_scale : Boolean read GetUse_Scale;
  end;

  MultiHeadAttention = class(Layer)
    private
        const _CHR_IDX : string = 'abcdefghijklmnopqrstuvwxyz';
    protected
        Fquery_shape          : TFShape;
        Fkey_shape            : TFShape;
        Fvalue_shape          : TFShape ;
        Fbuilt_from_signature : Boolean;
        Fquery_dense          : EinsumDense;
        Fkey_dense            : EinsumDense;
        Fvalue_dense          : EinsumDense;
        Foutput_dense         : EinsumDense;
        Fdot_product_equation : string;
        Fcombine_equation     : string;
        Fsoftmax              : Softmax ;
        Fdropout_layer        : Dropout ;

        function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
        function  Call(inputs: TFTensors; attention_mask: TFTensor; training : PBoolean= nil; return_attention_scores: Boolean = false): TFTensors; reintroduce; overload ;
    public
        args : MultiHeadAttentionArgs;

        constructor Create(_args: MultiHeadAttentionArgs);
        /// <summary>
        /// Builds einsum equations for the attention computation.
        /// Query, key, value inputs after projection are expected to have the shape as:
        /// `(bs, [non-attention dims], [attention dims], num_heads, channels)`.
        /// `bs` and `[non-attention dims]` are treated as `[batch dims]`.
        ///
        /// <para>
        /// The attention operations can be generalized:
        /// </para>
        /// <para>
        ///   (1) Query-key dot product:
        ///   `([batch dims], [query attention dims], num_heads, channels), ([batch dims],
        ///   [key attention dims], num_heads, channels) -> ([batch dim],
        ///   num_heads, [query attention dims], [key attention dims])`
        ///   </para><para>
        ///   (2) Combination:
        ///   `([batch dims], num_heads, [query attention dims], [key attention dims]),
        ///   ([batch dims], [value attention dims], num_heads, channels) -> ([batch dims],
        ///   [query attention dims], num_heads, channels)`
        /// </para>
        /// </summary>
        /// <param name="rank">Rank of query, key, value tensors.</param>
        /// <param name="attn_axes">List/tuple of axes, `[-1, rank)`,
        ///                        that attention will be applied to.</param>
        /// <returns></returns>
        class function _build_attention_equation(rank: Integer; attn_axes: TFShape): Tuple<string, string, Integer>;
        /// <summary>
        /// Builds an einsum equation for projections inside multi-head attention.
        /// </summary>
        class function _build_proj_equation(free_dims: Integer; bound_dims: Integer; output_dims: Integer): Tuple<string, string, Integer>;
        class function _get_output_shape(output_rank: Integer; known_last_dims: TFShape): TFShape;
        procedure _build_from_signature(query: TFTensor; value: TFTensor; key: TFTensor = nil); overload;
        procedure _build_from_signature(query: TFShape; value: TFShape; key: TFShape ); overload;
        function _get_dense(equation: string; output_shape: TFShape; bias_axes: string; name: string): EinsumDense;
        function _build_output_dense(free_dims: Integer; name: string): EinsumDense;
        procedure _build_attention(rank: Integer);
        function _masked_softmax(attention_scores: TFTensor; attention_mask : TFTensor = nil): TFTensor;
        function _compute_attention(query: TFTensor; key: TFTensor; value: TFTensor; attention_mask : TFTensor= nil; training: Boolean = false): TFTensors;
  end;
  {$ENDREGION}

  {$REGION 'Convolutional'}
  Convolutional = class(Layer)
    private
      Fconvolution_op : ConvolutionInternal;

      function Getactivation: TActivation;
      function Getbias_initializer: IInitializer;
      function Getdata_format: string;
      function Getdilation_rate: TFShape;
      function GetFilter: Integer;
      function Getkernel_initializer: IInitializer;
      function Getkernel_regularizer: IRegularizer;
      function Getkernel_size: TFShape;
      function Getpadding: string;
      function GetRank: Integer;
      function Getstrides: TFShape;
      function Getuse_bias: Boolean;
    protected
      kernel             : IVariableV1;
      bias               : IVariableV1;
      _tf_data_format    : string;

      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  _get_channel_axis: Integer; virtual;
    public
      args : ConvolutionalArgs;

      constructor Create(_args: ConvolutionalArgs) ;

      property rank               : Integer      read GetRank;
      property filters            : Integer      read GetFilter;
      property kernel_size        : TFShape      read Getkernel_size;
      property strides            : TFShape      read Getstrides;
      property padding            : string       read Getpadding;
      property data_format        : string       read Getdata_format;
      property dilation_rate      : TFShape      read Getdilation_rate;
      property activation         : TActivation  read Getactivation;
      property use_bias           : Boolean      read Getuse_bias;
      property kernel_initializer : IInitializer read Getkernel_initializer;
      property kernel_regularizer : IRegularizer read Getkernel_regularizer;
      property bias_initializer   : IInitializer read Getbias_initializer;
  end;

  Conv1D = class(Convolutional)
    public
      constructor Create(_args: Conv1DArgs) ;
  end;

    Conv2D = class(Convolutional)
    public
      constructor Create(_args: Conv2DArgs) ;
  end;

  Conv2DTranspose = class(Conv2D)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      constructor Create(_args: Conv2DArgs) ;
  end;
  {$ENDREGION}

  {$REGION 'Cropping'}
  Cropping1D = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args : CroppingArgs;

      constructor Create(_args: CroppingArgs) ;
  end;

  /// <summary>
  /// Crop the input along axis 1 and 2.
  /// <para> For example: </para>
  /// <para> shape (1, 5, 5, 5) -- crop2D ((1, 2), (1, 3)) --> shape (1, 2, 1, 5) </para>
  /// </summary>
  Cropping2D = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args : Cropping2DArgs;

      constructor Create(_args: Cropping2DArgs) ;
  end;

  /// <summary>
  /// Similar to copping 2D
  /// </summary>
  Cropping3D = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args : Cropping3DArgs;

      constructor Create(_args: Cropping3DArgs) ;
  end;
  {$ENDREGION}

  {$REGION 'Lstm-Rnn'}
  LSTMCell = class(Layer)
    private

    protected

    public
      args : LSTMCellArgs;

      constructor Create(_args: LSTMCellArgs) ;
  end;

  RNN = class(Layer)
    private
      Fargs           : RNNArgs;
     // Finput_spec     : TObject; // or NoneValue??
     // Fstate_spec     : TObject;
     // Fstates         : TObject;
     // Fconstants_spec : TObject;
     // Fnum_constants  : Integer;

      function PreConstruct(_args: RNNArgs) : RNNArgs;
      function _generate_zero_filled_state_for_cell(cell: LSTMCell; batch_size: TFTensor): TFTensor;
    protected
      kernel : IVariableV1;
      bias   : IVariableV1;
      cell   : ILayer;

      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  get_initial_state(inputs: TFTEnsor): TFTensor;
    public

      // Check whether the state_size contains multiple states.
      class function _is_multiple_state(state_size: TValue): Boolean;
      procedure Build(input_shape: TFShape); override;
      constructor Create(_args: RNNArgs) ;
  end;

  /// <summary>
  /// Long Short-Term Memory layer - Hochreiter 1997.
  ///
  /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  /// for details about the usage of RNN API.
  /// </summary>
  LSTM = class(RNN)
    private
      function getUnits: Integer;
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args      : LSTMArgs;
      state_spec: TArray<TInputSpec> ;

      constructor Create(_args: LSTMArgs) ;

      property units     : Integer read getUnits;
  end;

  SimpleRNNCell = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
     args               : SimpleRNNArgs;
     kernel             : IVariableV1;
     bias               : IVariableV1;
     recurrent_kernel   : IVariableV1;

     constructor Create(_args: SimpleRNNArgs) ;
  end;

  SimpleRNN = class(RNN)
    public
      args               : SimpleRNNArgs;

      constructor Create(_args: SimpleRNNArgs) ;
  end;

  StackedRNNCells = class(Layer, RNNArgs.IRnnArgCell)
  private
    function GetOuputSize: TValue;
    public
      Cells              : IList<RnnCell> ;
      reverse_state_order: Boolean;

      {$REGION 'Property Accessors'}
      function GetState_size : TValue;
      {$ENDREGION}

      constructor Create(_args: StackedRNNCellsArgs) ;

      property output_size : TValue read GetOuputSize;
  end;
  {$ENDREGION}

  {$REGION 'Merging'}
  Merge = class(Layer)
    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  _merge_function(inputs: TFTensors): TFTensors; virtual;
    public

      constructor Create(_args: MergeArgs) ;
  end;

  Add = class(Merge)
    private

    protected
      function  _merge_function(inputs: TFTensors): TFTensors; override;
    public

      constructor Create(_args: MergeArgs) ;
  end;

  Subtract = class(Merge)
    private

    protected
      function  _merge_function(inputs: TFTensors): TFTensors; override;
    public

      constructor Create(_args: MergeArgs) ;
  end;

  Concatenate = class(Merge)
    private
      function GetAxis: Integer;

    protected
      procedure Build(input_shape: TFShape); override;
      function  _merge_function(inputs: TFTensors): TFTensors; override;
    public
      args : MergeArgs;

      constructor Create(_args: MergeArgs) ;

      property axis : Integer read GetAxis;
  end;
  {$ENDREGION}

  {$REGION 'Normalization'}
  LayerNormalization = class(Layer)
    private
      function getbeta_initializer: IInitializer;
      function getcenter: Boolean;
      function getepsilon: Single;
      function getgamma_initializer: IInitializer;
      function getgamma_regularizer: IRegularizer;
      function getscale: Boolean;

    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args           : LayerNormalizationArgs;
      fused          : Boolean ;
      axis           : TArray<Integer>;
      _data_format   : string;
      kernel_size    : TFShape;
      gamma          : IVariableV1;
      beta           : IVariableV1;
      moving_mean    : IVariableV1;
      moving_variance: IVariableV1;

      constructor Create(_args: LayerNormalizationArgs) ;
      function _fused_can_be_used(ndims: Integer) : Boolean;

      property epsilon  : Single  read getepsilon;
      property center   : Boolean read getcenter;
      property scale    : Boolean read getscale;
      property beta_initializer  : IInitializer read getbeta_initializer;
      property gamma_initializer : IInitializer read getgamma_initializer;
      property gamma_regularizer : IRegularizer read getgamma_regularizer;

  end;

  BatchNormalization = class(Layer)
    private
      function getbeta_initializer: IInitializer;
      function getcenter: Boolean;
      function getepsilon: Single;
      function getgamma_initializer: IInitializer;
      function getgamma_regularizer: IRegularizer;
      function getMomentum: Single;
      function getmoving_mean_initializer: IInitializer;
      function getmoving_variance_initializer: IInitializer;
      function getrenorm: Boolean;
      function getscale: Boolean;
      function _moments(inputs: TFTensors; reduction_axes: TArray<Integer>; keep_dims: Boolean): Tuple<TFTensor, TFTensor>;
      function _calculate_mean_and_var(inputs: TFTensors; reduction_axes: TArray<Integer>; keep_dims: Boolean): Tuple<TFTensor, TFTensor>;
      function _support_zero_size_input: Boolean;
      function _fused_batch_norm(inputs, training: TFTensor): TFTensor;
      procedure _assign_moving_average(variable: IVariableV1; value, momentum: TFTensor);
      procedure _assign_new_value(variable: IVariableV1; value: TFTensor);

    protected
      procedure Build(input_shape: TFShape); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
       args : BatchNormalizationArgs;
       fused          : Boolean ;
       axis           : TArray<Integer>;
       _data_format   : string;
       kernel_size    : TFShape;
       gamma          : IVariableV1;
       beta           : IVariableV1;
       moving_mean    : IVariableV1;
       moving_variance: IVariableV1;

       constructor Create(_args: BatchNormalizationArgs) ;

       property momentum : Single  read getMomentum;
       property epsilon  : Single  read getepsilon;
       property center   : Boolean read getcenter;
       property scale    : Boolean read getscale;
       property renorm   : Boolean read getrenorm;
       property beta_initializer            : IInitializer read getbeta_initializer;
       property gamma_initializer           : IInitializer read getgamma_initializer;
       property moving_mean_initializer     : IInitializer read getmoving_mean_initializer;
       property moving_variance_initializer : IInitializer read getmoving_variance_initializer;
       property gamma_regularizer           : IRegularizer read getgamma_regularizer;
  end;
  {$ENDREGION}

  {$REGION 'Pooling'}
  Pooling1D = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      args      : Pooling1DArgs;
      input_spec: TInputSpec;

      constructor Create(_args: Pooling1DArgs) ;
  end;

  Pooling2D = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      args      : Pooling2DArgs;
      input_spec: TInputSpec;

      constructor Create(_args: Pooling2DArgs) ;
  end;

  MaxPooling1D = class(Pooling1D)
    public

      constructor Create(_args: Pooling1DArgs) ;
  end;

  MaxPooling2D = class(Pooling2D)
    public

      constructor Create(_args: Pooling2DArgs) ;
  end;

  AveragePooling2D = class(Pooling2D)
    public

      constructor Create(_args: Pooling2DArgs) ;
  end;

  GlobalPooling1D = class(Layer)
    private
      function getDataFormat: string;
    protected

      Finput_spec  : TInputSpec;
    public
      args      : Pooling1DArgs;

      constructor Create(_args: Pooling1DArgs) ;

      property data_format : string read getDataFormat;
  end;

  GlobalPooling2D = class(Layer)
    private
      function getDataFormat: string;
    protected

      Finput_spec  : TInputSpec;
    public
      args      : Pooling2DArgs;

      constructor Create(_args: Pooling2DArgs) ;

      property data_format : string read getDataFormat;
  end;

  GlobalMaxPooling1D = class(GlobalPooling1D)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      constructor Create(_args: Pooling1DArgs) ;
  end;

  GlobalMaxPooling2D = class(GlobalPooling2D)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      constructor Create(_args: Pooling2DArgs) ;
  end;

  GlobalAveragePooling1D = class(GlobalPooling1D)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      constructor Create(_args: Pooling1DArgs) ;
  end;

  GlobalAveragePooling2D = class(GlobalPooling2D)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;

    public
      constructor Create(_args: Pooling2DArgs) ;
  end;
  {$ENDREGION}

  {$REGION 'PreProcessing'}
  PreprocessingLayer = class(Layer)
    public
      constructor Create(_args: PreprocessingLayerArgs) ;
  end;

  /// <summary>
  /// Resize the batched image input to target height and width.
  /// The input should be a 4-D tensor in the format of NHWC.
  /// </summary>
  Resizing = class(PreprocessingLayer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; overload ;override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args : ResizingArgs;

      function from_config(config: TJSONValue): Resizing;
      constructor Create(_args: ResizingArgs) ;
  end;

  CombinerPreprocessingLayer = class(Layer)
    protected
       Fcombiner           : ICombiner;
       Fpreviously_updated : Boolean;
    public
       args : PreprocessingLayerArgs;

       procedure adapt(data: IDatasetV2; reset_state: Boolean = true); virtual;
       constructor Create(_args: PreprocessingLayerArgs) ;
  end;

  IndexLookupAccumulator = class(TInterfacedObject,IAccumulator)
    protected

    public
       CountDict : TDictionary<string,Integer>;
       constructor Create ;
  end;

  /// <summary>
  /// Combiner for the IndexLookup preprocessing layer.
  /// </summary>
  IndexLookupCombiner = class(TInterfacedObject,ICombiner)
    protected
       Fvocab_size : Integer;
       Fmask_value : string;
    public
       procedure Compute(values: TFTensor; accumulator : IAccumulator= nil);
       procedure Merge;
       procedure Extract;
       function  Restore: IAccumulator;
       procedure Serialize;
       procedure Deserialize;

       constructor Create(vocab_size : Integer= -1; mask_value: string = '') ;
  end;

  IndexLookup = class(CombinerPreprocessingLayer)
    protected

    public
       procedure adapt(data: IDatasetV2; reset_state: Boolean = true); override;

       constructor Create(max_tokens: Integer = -1; num_oov_indices : Integer= 1; mask_token : string=''; oov_token: string = '[UNK]'; encoding: string = 'utf-8'; invert : Boolean = false) ;
  end;

  /// <summary>
  /// Maps strings from a vocabulary to integer indices.
  /// </summary>
  StringLookup = class(IndexLookup)
    protected

    public

       constructor Create(max_tokens: Integer = -1; num_oov_indices : Integer= 1; mask_token : string=''; vocabulary: TArray<string> = []; oov_token: string = '[UNK]'; encoding: string = 'utf-8'; invert : Boolean = false);
  end;

  TextVectorization = class(CombinerPreprocessingLayer)
    protected
       Findex_lookup_layer : IndexLookup;

       procedure Build(input_shape: TFShape); override;
    public
       args : TextVectorizationArgs;

       constructor Create(_args: TextVectorizationArgs) ;
       function _preprocess(inputs: TFTensors): TFTensors;
       procedure adapt(data: IDatasetV2; reset_state: Boolean = true); override;

  end;
  {$ENDREGION}

  {$REGION 'Rescaling'}
  /// <summary>
  /// Multiply inputs by `scale` and adds `offset`.
  /// </summary>
  Rescaling = class(Layer)
    protected
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args  :  RescalingArgs;
      scale : TFTensor;
      offset: TFTensor;

      constructor Create(_args: RescalingArgs) ;
  end;
  {$ENDREGION}

  {$REGION 'Reshaping'}
  /// <summary>
  /// Zero-padding layer for 2D input (e.g. picture).
  ///
  /// This layer can add rows and columns of zeros
  /// at the top, bottom, left and right side of an image tensor.
  /// </summary>
  ZeroPadding2D = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      data_format : string;
      padding     : TNDArray;
      input_spec  : TInputSpec;

      constructor Create(_args: ZeroPadding2DArgs) ;
  end;

  Flatten = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args            : FlattenArgs;
      input_spec      : TInputSpec;
      _channels_first : Boolean;

      constructor Create(_args: FlattenArgs) ;
  end;

  Permute = class(Layer)
    protected
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      procedure Build(input_shape: TFShape); override;
    public
      dims, permute : TArray<Integer>;

      constructor Create(_args: PermuteArgs) ;
  end;

  /// <summary>
  /// Layer that reshapes inputs into the given shape.
  /// </summary>
  Reshape = class(Layer)
    protected
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args : ReshapeArgs;

      constructor Create(_args: ReshapeArgs) ;
  end;

  /// <summary>
  /// Layer that reshapes inputs into the given shape.
  /// </summary>
  UpSampling2D = class(Layer)
  private
      function GetInterpolation: string;
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args          : UpSampling2DArgs;
      size          : TArray<Integer>;
      data_format   : string;

      constructor Create(_args: UpSampling2DArgs) ;

      property interpolation : string  read GetInterpolation;
  end;
  {$ENDREGION}

  {$REGION 'Metric'}
  /// <summary>
  /// Encapsulates metric logic and state.
  /// </summary>
  Metric = class(Layer, IMetricFunc)
    protected
      total      : IVariableV1;
      count      : IVariableV1;
      Freduction : string;
      Fdtype     : TF_DataType;

      function add_weight(name            : string;
                          shape           : TFShape;
                          dtype           : TF_DataType = TF_DataType.TF_FLOAT;
                          initializer     : IInitializer = nil;
                          regularizer     : IRegularizer = nil;
                          synchronization : TVariableSynchronization= VARIABLE_SYNCHRONIZATION_AUTO;
                          aggregation     : TVariableAggregation    = VARIABLE_AGGREGATION_NONE;
                          trainable       : Boolean= true;
                          getter          : TFunc<VariableArgs, IVariableV1>= nil): IVariableV1; override;
    public
      constructor Create(name: string = ''; dtype: TF_DataType = DtInvalid);
      function  update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; virtual;
      procedure reset_states; virtual;
      function  R_result: TFTensor; virtual;
      function  ToString: string; override;
  end;
  {$ENDREGION}

  {$REGION 'CategoryEncoding'}
  /// <summary>
  /// This layer provides options for condensing data into a categorical encoding when the total number of tokens are known in advance.
  /// </summary>
  CategoryEncoding = class(Layer)
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args : CategoryEncodingArgs;

      constructor Create(_args: CategoryEncodingArgs) ;
      function  encode_categorical_inputs(inputs: TFTensor; output_mode: string; depth: Integer; dtype : TF_DataType = TF_FLOAT; sparse: Boolean = false; count_weights: TFTensor = nil) : TFTensors;

  end;
  {$ENDREGION}

  {$REGION 'TensorFlowOpLayer'}
  TensorFlowOpLayer = class(Layer)
  private
      function GetConsts: TDictionary<Integer, TNDArray>;
      function GetNodeDef: TNodeDef;
    protected
      F_function  : ConcreteFunction;

      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args     : TensorFlowOpLayerArgs;

      OpType   : string;

      TF_OP_LAYER_NAME_PREFIX : string;

      constructor Create(_args: TensorFlowOpLayerArgs) ;
      function DeFunCall(inputs: TFTensors): TFTensors;
      function mark_as_return(tensors: TFTensors): TFTensors;
      function MakOp(inputs: TFTensors): TFTensors;
      function GetOpLayer(_args: TensorFlowOpLayerArgs):Layer;

      property constants: TDictionary<Integer, TNDArray> read GetConsts;
      property node_def : TNodeDef                       read GetNodeDef ;
  end;
  {$ENDREGION}

implementation
        uses
             Keras.Backend,

             Tensorflow,
             Tensorflow.Variable,
             TensorFlow.Tensor,
             TensorFlow.Ops,
             Tensorflow.Utils,
             TensorFlow.nn_ops,
             TensorFlow.nn_impl,
             Tensorflow.math_ops,
             Tensorflow.array_ops,
             TensorFlow.gen_math_ops,
             TensorFlow.embedding_ops,
             TensorFlow.image_ops_impl,
             TensorFlow.Slice,
             NumPy.NDArray,

             Keras.Utils;

{$REGION 'Layer'}
{ Layer }

constructor Layer.Create(_args: LayerArgs);
begin
   Initialize(_args)
end;

procedure Layer.Initialize(_args: LayerArgs);
begin
    Fargs := _args;
    // A stateful layer is a layer whose updates are run during inference too,
    // for instance stateful RNNs.
    Fstateful := false;
    // Indicates whether `build` needs to be called upon layer call, to create
    // the layer's weights.
    Fbuilt := false;
    FSupportsMasking := false;
    Fdynamic         := True;

    Fid := Tops.uid_layer;
    _init_set_name(Fargs.Name);
    FtrainableWeights       := TList<IVariableV1>.Create;
    FnonTrainableWeights    := TList<IVariableV1>.Create;
    FcomputePreviousMask    := false;
    Fupdates                := TList<TFOperation>.Create;
    Fself_tracked_trackables:= TList<ILayer>.Create;

    FinboundNodes  := TList<INode>.Create;
    FoutboundNodes := TList<INode>.Create;

    // Manage input shape information if passed.
    if (Fargs.BatchInputShape.isNull) and (not Fargs.InputShape.isNull) then
    begin
        var aShape : TArray<Int64> := [Fargs.BatchSize] + Fargs.InputShape.dims;
        Fargs.BatchInputShape :=  aShape;
    end;
end;

destructor Layer.Destroy;
begin
    Fargs.Free;

    if Assigned(FNodesByDepth) then
      FNodesByDepth.Free;
    //if Assigned(Ftrainable_state) then
    //  Ftrainable_state.Free;
    //if Assigned(Fcompiled_trainable_state) then
    //  Fcompiled_trainable_state.Free;
    if Assigned(FinputSpec) then
      FinputSpec.Free;
    if Assigned(FTrainableWeights) then
      FTrainableWeights.Free;
    if Assigned(FNonTrainableWeights) then
      FNonTrainableWeights.Free;
    if Assigned(Fupdates) then
      Fupdates.Free;
    if Assigned(FInboundNodes) then
      FInboundNodes.Free;
    if Assigned(FOutboundNodes) then
      FOutboundNodes.Free;
    if Assigned(Fself_tracked_trackables) then
      Fself_tracked_trackables.Free;

    inherited Destroy;
end;

procedure Layer.add_loss(losses: TFunc<TFTensor>);
begin

end;

procedure Layer.Build(input_shape: TFShape);
begin
    Fbuilt := true;
end;

function Layer.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    Result := inputs;
end;

function Layer.compute_mask(inputs, mask: TFTensor): TFTensor;
begin
   Result := nil;
end;

function Layer.count_params: Integer;
begin
    if Trainable  then
    begin
        Result := layer_utils.count_params(self, weights);
        Exit;
    end;
    Result := 0;
end;

function Layer.GetBatchShape: TFShape;
begin
    Result := Fargs.BatchInputShape;
end;

function Layer.GetBuilt: Boolean;
begin
   Result := Fbuilt;
end;

function Layer.GetCallCtx: TCallContext;
begin
   Result := tls_CallContext;
end;

procedure Layer.SetCallCtx(const Value: TCallContext);
begin

end;

function Layer.GetDtype: TF_DataType;
begin
    Result := Fargs.DType;
end;

function Layer.GetName: string;
begin
    Result := FName;
end;

function Layer.GetNonTrainVars: TList<IVariableV1>;
begin
    Result := NonTrainableWeights;
end;

function Layer.GetInNodes: TList<INode>;
begin
    Result := FInboundNodes;
end;

function Layer.GetInput: TArray<TFTensor>;
begin
   Result := FInboundNodes[0].input_tensors.ToArray
end;

function Layer.GetLayers: TList<ILayer>;
begin
    Result := Fself_tracked_trackables;
end;

function Layer.GetOutNodes: TList<INode>;
begin
    Result := FOutboundNodes;
end;

function Layer.GetoutShape: TFShape;
begin
    Result := System.Default(TFShape) ;
    if FInboundNodes.Count < 1 then Exit;

    Result := FInboundNodes[0].Outputs.shape;
end;

function Layer.GetTrainable: Boolean;
begin
   Result := Fargs.Trainable;
end;

function Layer.GetTrainVars: TList<IVariableV1>;
begin
    Result := TrainableWeights;
end;

function Layer.GetTrainW: TList<IVariableV1>;
begin
    if not Trainable then
      Exit( TList<IVariableV1>.Create );

    var children_weights := _gather_children_variables(true);

    var enu_children_weights := Enumerable<IVariableV1>.Create(children_weights.ToArray);
    var Res : Enumerable<IVariableV1> := enu_children_weights.Concat(FTrainableWeights.ToArray);
    Res     := Res.Distinct;

    Result := TList<IVariableV1>.Create(Res.ToArray);
end;

function Layer.GetNotTrainW: TList<IVariableV1>;
begin
    if not Trainable then
    begin
      var children_weights := _gather_children_variables(true,True);

      var enu_children_weights := Enumerable<IVariableV1>.Create(children_weights.ToArray);
      var Res : Enumerable<IVariableV1> := enu_children_weights.Concat(FTrainableWeights.ToArray);
      Res     := Res.Concat(FNonTrainableWeights.ToArray);
      Res     := Res.Distinct;

      Result := TList<IVariableV1>.Create(Res.ToArray);
    end else
    begin
      var children_weights := _gather_children_variables(False,True);

      var enu_children_weights := Enumerable<IVariableV1>.Create(children_weights.ToArray);
      var Res : Enumerable<IVariableV1> := enu_children_weights.Concat(FNonTrainableWeights.ToArray);
      Res     := Res.Distinct;

      Result := TList<IVariableV1>.Create(Res.ToArray);
    end;
end;

function Layer.GetWeights: TList<IVariableV1>;
begin
    Result := TList<IVariableV1>.Create( TrainableWeights.ToArray + NonTrainableWeights.ToArray) ;
end;

procedure Layer.SetWeights(value: TList<IVariableV1>);
begin
    if weights.Count <> value.Count then raise TFException.Create(
                                'You called `set_weights` on layer \'+ Fname +
                                'with a weight list of length '+ IntToStr(value.Count)+', but the layer was ' +
                                'expecting '+ IntToStr(weights.count)+ ' weights.');

    for var t in TUtils.zip<IVariableV1>(weights, value) do
    begin
        var this_w := t.Value1;
        var v_w    := t.Value2;

        if      this_w is RefVariable          then  (this_w as RefVariable).assign(v_w, False , '',true)
        else if this_w is BaseResourceVariable then  (this_w as BaseResourceVariable).assign(v_w, False ,'',true)
        else
           raise Exception.Create('state_ops.assign Error!');
    end;
end;

function Layer.GetVars: TList<IVariableV1>;
begin
    Result := Weights
end;

function Layer.get_config: LayerArgs;
begin
    Result := Fargs
end;

procedure Layer.load_weights(filepath: string);
begin

end;

procedure Layer.StackLayers(_layers: TArray<ILayer>);
begin
    Fself_tracked_trackables.AddRange(_layers);
end;

function Layer.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    raise TFException.Create('Not Implemented');
end;

procedure Layer.MaybeBuild(inputs: TFTensors);
begin
    // Check input assumptions set before layer building, e.g. input rank.
    if Fbuilt then
        Exit;
    if DType = TF_DataType.DtInvalid then
        Fargs.DType := inputs.dtype;

    tf.init_scope;

    var need_restore_mode : Boolean := false;
    var enuInputs := Enumerable<TFTensor>.Create(inputs.ToArray);
    var bAny := enuInputs.Any( function(const x: TFTensor) : Boolean
                                  begin
                                     Result := x is TEagerTensor;
                                  end);

    if (bAny) or (tf.Context.is_build_function) then
    begin
        need_restore_mode := true;
        tf.Context.eager_mode(tf.Context.is_build_function);
    end;

    build(inputs.shape);

    if need_restore_mode then
        tf.Context.restore_mode;

    Fbuilt := true;
end;

function Layer.FunctionalConstructionCall(inputs: TFTensors): TFTensors;
begin
    //var mask_arg_passed_by_framework     : Boolean := false;
    //var training_arg_passed_by_framework : Boolean := false;
    //var training_value : TFTensor := nil;
    //if training_value = nil then
    //   training_arg_passed_by_framework := true;


    if base_layer_utils.needs_keras_history(inputs) then
        base_layer_utils.create_keras_history(inputs);

    var outputs : TFTensors ;
    CallContext.enter(true);

    var graph := tf.keras.backend.get_graph;
    graph.as_default;

    var scope := Tops.name_scope(_name_scope);
    scope._enter_;

    MaybeBuild(inputs);

    // Wrapping `call` function in autograph to allow for dynamic control
    // flow and control dependencies in call. We are limiting this to
    // subclassed layers as autograph is strictly needed only for
    // subclassed layers and models.
    // tf_convert will respect the value of autograph setting in the
    // enclosing tf.function, if any.
    if not Fdynamic then
      raise Exception.Create(' Not Implemented');

    outputs := Call(inputs);

    _set_connectivity_metadata_(inputs, outputs);
    _handle_activity_regularization(inputs, outputs);
    _set_mask_metadata(inputs, outputs, nil);

    scope._exit_;
    graph.gExit;

    Result := outputs;
end;

function Layer.Apply(inputs: TFTensors; state: TFTensor; training: Boolean): TFTensors;
begin
    if callContext = nil then
        callContext := TCallContext.Create;

    if _in_functional_construction_mode(inputs) then
        Exit( FunctionalConstructionCall(inputs) );

    var eager := tf.executing_eagerly;
    CallContext.enter(false);

    var nameScope : string := _name_scope;
    if eager then  nameScope := Name ;

    var scope := Tops.name_scope(nameScope);
    scope._enter_;

    if not Fbuilt then
        MaybeBuild(inputs);

    var outputs := Call(inputs, state, @training);

    // memory leak
    // _set_connectivity_metadata_(inputs, outputs);
    _handle_activity_regularization(inputs, outputs);
    _set_mask_metadata(inputs, outputs, nil);

    scope._exit_;

    Result := outputs;
end;

procedure Layer.SetConnectivityMetadata(inputs, outputs: TFTensors);
begin
   _set_connectivity_metadata_(inputs, outputs)
end;

procedure Layer._handle_activity_regularization(inputs, outputs: TFTensors);
begin
    //if(_activity_regularizer != null)
    begin

    end
end;

procedure Layer._handle_weight_regularization(name: string; variable: IVariableV1; regularizer: IRegularizer);
begin
   var res := TUtils.tf_with<TNameScope,TFunc<TFTensor>>( TOps.name_scope(name + '/Regularizer'),
                    function(v1: TNameScope): TFunc<TFTensor>
                      begin
                           Result := function: TFTensor
                                      begin
                                         Result := regularizer.Apply(RegularizerArgs.Create(variable.AsTensor) );
                                      end;
                      end );
end;

procedure Layer._init_set_name(_name: string; zero_based: Boolean);
begin
    Fbase_name := _name;
    Fname      := _name;
    if _name = '' then
    begin
        Fbase_name := generic_utils.to_snake_case( TObject(self).ClassName);
        Fname      := base_layer_utils.unique_layer_name(Fbase_name,nil, [], zero_based);
    end;
end;

function Layer._in_functional_construction_mode(inputs: TFTensors): Boolean;
begin
    var Count : Integer := 0;
    var aI := inputs.ToArray;
    for var i := 0 to Length(aI)-1 do
    begin
      if ( not(aI[i] is TEagerTensor) ) and ( not(aI[i] is TNDArray) ) then
          Inc(Count);
    end;

    Result := (tf.Context.executing_eagerly) and (Count = inputs.Count )
end;

function Layer._name_scope: string;
begin
    Result := FName;
end;

procedure Layer._set_connectivity_metadata_(inputs, outputs: TFTensors);
begin
   var nArgs := NodeArgs.Create;
   nArgs.InputTensors := inputs;
   nArgs.Outputs      := outputs;

   var node := Node.Create(nArgs);
   node.Connect(self);
end;

procedure Layer._set_mask_metadata(inputs, outputs, previous_mask: TFTensors);
begin

end;

function Layer._flatten_layers(recursive, include_self: Boolean): TArray<ILayer>;
begin
    if include_self then
    begin
        Result := [ self ] ;
        Exit;
    end;

    var seen_object_ids := TList<integer>.Create;
    var deque           := TQueue<ILayer>.Create(Fself_tracked_trackables);
    while deque.Count > 0 do
    begin
        var layer_or_container    := deque.Dequeue;
        var layer_or_container_id : Integer := TObject(layer_or_container).GetHashCode;
        if seen_object_ids.Contains(layer_or_container_id) then
            continue;
        seen_object_ids.Add(layer_or_container_id);
        Result := Result + [ layer_or_container ];
        if recursive then
           TUtils.extendleft<Ilayer>(deque, layer_or_container.Layers);
    end;
end;

function Layer._gather_children_variables(include_trainable, include_non_trainable: Boolean): TList<IVariableV1>;
begin
    var res := TList<IVariableV1>.Create;
    var nested_layers := _flatten_layers(false, false);
    for var _layer in nested_layers do
    begin
        if _layer is Layer  then
        begin
            if (include_trainable = true) and (include_non_trainable = true) then
            begin
                res.AddRange(Layer(_layer).Variables);
            end
            else if (include_trainable = true) and (include_non_trainable = false) then
            begin
                res.AddRange(Layer(_layer).TrainableVariables);
            end
            else if(include_trainable = false) and (include_non_trainable = true) then
            begin
                res.AddRange(Layer(_layer).NonTrainableVariables);
            end
        end;
    end;
    Result := res;
end;

function Layer._get_trainable_state: TDictionary<ILayer, boolean>;
begin
    Ftrainable_state := TDictionary<ILayer, Boolean>.Create;
    var flatLayer := _flatten_layers;

    for var i := 0 to Length(flatLayer) - 1 do
        Ftrainable_state.AddOrSetValue(flatLayer[i], flatLayer[i].Trainable);

    Result := Ftrainable_state;
end;

function Layer.add_weight(name: string; shape: TFShape; dtype: TF_DataType; initializer: IInitializer; regularizer: IRegularizer; synchronization: TVariableSynchronization;
  aggregation: TVariableAggregation; trainable: Boolean; getter: TFunc<VariableArgs, IVariableV1>): IVariableV1;
begin
    // Initialize variable when no initializer provided
    if initializer = nil then
    begin
        // If dtype is DT_FLOAT, provide a uniform unit scaling initializer
        if Tdtypes.is_floating(dtype)     then   initializer := tf.glorot_uniform_initializer
        else if Tdtypes.is_integer(dtype) then    initializer := tf.zeros_initializer
        else raise Exception.Create( Format('An initializer for variable %s of type %s is required for layer %s',[name,Tdtypes.ToString( Tdtypes.as_base_dtype(dtype) ),name ]) );
    end;

    if synchronization = TVariableSynchronization.VARIABLE_SYNCHRONIZATION_ON_READ then
        trainable := false;

    var args : VariableArgs;

    args.Name            := name;
    args.Shape           := shape;
    args.DType           := dtype;

    if Assigned(getter)  then args.Getter := getter
    else                      args.Getter :=  base_layer_utils.make_variable;

    args.Overwrite       := true;
    args.Initializer     := initializer;
    args.Synchronization := synchronization;
    args.Aggregation     := aggregation;
    args.Trainable       := trainable;

    var variable := _add_variable_with_custom_getter(args);

    if regularizer <> nil then
    begin
        var name_in_scope := variable.Name.Split([':'])[0];
        _handle_weight_regularization(name_in_scope, variable, regularizer);
    end;

    //backend.track_variable(variable);
    if trainable = true then FTrainableWeights.Add(variable)
    else                     FNonTrainableWeights.Add(variable);

    Result := variable;
end;

{$IFNDEF AUTOREFCOUNT}
function Layer.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure Layer.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := Layer(Obj).FRefCount;
  until AtomicCmpExchange(Layer(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure Layer.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure Layer.BeforeDestruction;
begin
  //if RefCount <> 0 then
  //  Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function Layer.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  Layer(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function Layer.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function Layer._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function Layer._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{$REGION 'Activation'}
{ ELU }

constructor ELU.Create(_args: ELUArgs);
begin
    inherited Create(_args);
    args := _args;
end;

function ELU.GetAlpha: Single;
begin
    Result := args.Alpha;
end;

procedure ELU.Build(input_shape: TFShape);
begin
  if alpha < 0  then
    raise TFException.Create('Alpha must be a number greater than 0.');

  inherited Build(input_shape);
end;

function ELU.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.first;
    output := tf.where(TTensor(output) > Single(0), output,
              tf.multiply(alpha, tf.sub(tf.exp(output), Single(1))));
    Result := TFTensors.Create(output)
end;

function ELU.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Exponential }

constructor Exponential.Create(_args: LayerArgs);
begin
    inherited Create(_args);
end;

procedure Exponential.Build(input_shape: TFShape);
begin
  inherited Build(input_shape);
end;

function Exponential.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.first;

    var _out := tf.exp(output);
    Result := TFTensors.Create(_out);
end;

function Exponential.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ HardSigmoid }

constructor HardSigmoid.Create(_args: LayerArgs);
begin
   inherited Create(_args);
end;

function HardSigmoid.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var x : TFTensor := inputs.first;

    var res := tf.clip_by_value( tf.add(tf.multiply(x, Single(0.2)), Single(0.5)), Single(0), Single(1)) ;
    Result := TFTensors.Create(res);
end;

function HardSigmoid.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
   Result := input_shape ;
end;

{ LeakyReLu }

constructor LeakyReLu.Create(_args: LeakyReLuArgs);
begin
    inherited Create(_args);

    args := _args;
end;

function LeakyReLu.GetAlpha: Single;
begin
    Result := args.Alpha;
end;

function LeakyReLu.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
   var Res := tf.nn.leaky_relu(inputs.first, alpha);
   Result := TFTensors.Create(res);
end;

{ SELU }

constructor SELU.Create(_args: LayerArgs);
begin
   inherited Create(_args);
end;

procedure SELU.Build(input_shape: TFShape);
begin
  if alpha < 0  then
    raise TFException.Create('Alpha must be a number greater than 0.');

  inherited Build(input_shape);
end;

function SELU.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.first;

    var res := tf.where(TTensor(output) > Single(0),
                        tf.multiply(scale, output),
                        tf.multiply(scale, tf.multiply(alpha, tf.sub(tf.exp(output), Single(1)))));

    Result := TFTensors.Create(res);
end;

function SELU.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Softmax }

constructor Softmax.Create(_args: SoftmaxArgs);
begin
    inherited Create(_args);

    axis := _args.axis;
end;

function Softmax.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var x : TFTensor ;
    if inputs.Count = 2 then x := inputs.First + ((Double(1.0) - TTensor(tf.cast(inputs[1], inputs.dtype))) * Double(1e-9))
    else                     x := inputs.First;

    var e : TFTensor := tf.exp(tf.sub(x, tf.reduce_max(x, @axis, true)));
    var s : TFTensor := tf.reduce_sum(e, @axis,  nil, true);

    var Res := tf.&div(e, s);
    Result := TFTensors.Create(res);
end;

function Softmax.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Softplus }

constructor Softplus.Create(_args: LayerArgs);
begin
    inherited Create(_args);
end;

function Softplus.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var x : TFTensor := inputs.first;
    var res := tf.log( tf.add(tf.exp(x), Single(1)) );
    Result  := TFTensors.Create(res);
end;

function Softplus.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Softsign }

constructor Softsign.Create(_args: LayerArgs);
begin
   inherited Create(_args);
end;

function Softsign.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var x : TFTensor := inputs.first;
    // x / (abs(x) + 1)
    var res := tf.&div(x, tf.add(Single(1), tf.abs(x)));
    Result  := TFTensors.Create(res);
end;

function Softsign.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Swish }

constructor Swish.Create(_args: LayerArgs);
begin
    inherited Create(_args);
end;

function Swish.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
     var x : TFTensor := inputs.first;
     // x / (1 + exp(-x))
     var res := tf.&div(x, (tf.add(Single(1), tf.exp(tf.negative(x)))));
     Result  := TFTensors.Create(res);
end;

function Swish.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape ;
end;

{ Tanh }

constructor Tanh.Create(_args: LayerArgs);
begin
     inherited Create(_args);
end;

function Tanh.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var x : TFTensor := inputs.first;

    var res := tf.tanh(x);
    Result  := TFTensors.Create(res);
end;

function Tanh.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
   Result := input_shape ;
end;
{$ENDREGION}

{$REGION 'Attention'}

{ BaseDenseAttention }

constructor BaseDenseAttention.Create(_args: BaseDenseAttentionArgs);
begin
    inherited Create(_args);
    args              := _args;
    Fsupports_masking := true;
end;

function BaseDenseAttention._calculate_scores(query, key: TFTensor): TFTensor;
begin
    raise Exception.Create('_calculate_scores');
end;

function BaseDenseAttention._apply_scores(scores, value, scores_mask: TFTensor; training: PBoolean): Tuple<TFTensor, TFTensor>;
begin
    if scores_mask <> nil then
    begin
        var padding_mask := tf.logical_not(scores_mask);
        // Bias so padding positions do not contribute to attention distribution.
        // Note 65504. is the max float16 value.
        if scores.dtype = tf.float16_t then
            scores := TTensor(scores) - ( Single(65504) * TTensor(tf.cast(padding_mask, scores.dtype)) )
        else
            scores := TTensor(scores) - ( Single(1000000000) * TTensor(tf.cast(padding_mask, scores.dtype)) );
    end;
    var _training : Boolean;
    var b := False;
    training := @b; // TODO: Delete this line when backend.learning_phase is available
    if training = nil then
    begin
        if tf.keras.backend.learning_phase = GraphLearningPhase.train_mode then _training := True
        else                                                                    _training := False;
    end else
    begin
        _training := training^;
    end;
    var weights := tf.nn.softmax(scores);
    var d := dropout;
    var dropped_weights : TFunc<TFTensor> := function : TFTensor
                                              begin
                                                 Result := tf.nn.dropout(weights, nil, nil, nil, '', @d);
                                              end;

    var false_pred : TFunc<TFTensor> := function : TFTensor
                                              begin
                                                 Result := tf.identity(weights);
                                              end;

    weights := smart_module.smart_cond(_training, dropped_weights, false_pred);
    //return (tf.matmul(weights, value), weights);
    Result := Tuple.Create(tf.linalg.einsum('bij,bjk->bik', TFTensors.Create([weights, value])), weights );
end;

function BaseDenseAttention.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  _inp, _mask : TFTensors;
  count       : Integer;
  has_bool,
  return_attention_scores  : Boolean;
begin
    _mask := nil;
    _inp  := nil;
    count := inputs.Count;

    if (count < 2) or (count > 6)  then
         raise Exception.Create(name + ' layer accepts inputs list of length from 2 to 6, namely [query, value, (key), (query_mask), (value_mask),'+
                                       '  (return_attention_scores)]. Received length: '+ Inttostr(count));

    has_bool                := inputs[count - 1].dtype = TF_DataType.TF_BOOL;
    return_attention_scores := false;
    if has_bool then
    begin
        return_attention_scores := Boolean( TTensor(inputs[count - 1]) );
        Dec(count);
    end;

    case count of
     2: begin
            _inp := TFTensors.Create( [inputs[0], inputs[1]] );
        end;
     3: begin
            _inp := TFTensors.Create( [ inputs[0], inputs[1], inputs[2] ] );
        end;
     4: begin
            if inputs[0].shape = inputs[2].shape then
            begin
                if inputs[1].shape = inputs[3].shape then
                begin
                    _inp := TFTensors.Create( [ inputs[0], inputs[1] ] );
                    _mask:= TFTensors.Create( [ inputs[2], inputs[3] ] );
                end;
            end else
                raise Exception.Create('BaseDenseAttention.Call'); //TODO:Add discriptions for this err
        end;
     5: begin
            _inp  := TFTensors.Create([ inputs[0], inputs[1], inputs[2] ]);
            _mask := TFTensors.Create([inputs[3], inputs[4]]);
        end;
     else
        raise Exception.Create('BaseDenseAttention.Call'); //TODO:Add discriptions for this err
    end;

    Result := call(_inp, _mask, training, return_attention_scores);
end;

function BaseDenseAttention.Call(inputs, mask: TFTensors; training: pBoolean; return_attention_scores: Boolean): TFTensors;
begin
    var causal_mask : TFTensor ;
    //this._validate_call_args(inputs: inputs, mask: mask);
    var q := inputs[0];
    var v := inputs[1];
    var k : TFTensor;
    var q_mask : TFTensor := nil;
    var v_mask : TFTensor := nil;
    if inputs.Count > 2 then k := inputs[2]
    else                     k := v;
    if mask <> nil then q_mask := mask[0];
    if mask <> nil then v_mask := mask[1];
    var scores := _calculate_scores(q, k);
    if v_mask <> nil then
        // Mask of shape [batch_size, 1, Tv].
        v_mask := tf.expand_dims(v_mask, -2);
    if causal then
    begin
        // Creates a lower triangular mask, so position i cannot attend to
        // positions j>i. This prevents the flow of information from the future
        // into the past.
        var scores_shape := tf.shape(scores);
        // causal_mask_shape = [1, Tq, Tv].
        var causal_mask_shape := tf.concat( [ tf.ones_like( tf.slice<Integer,Integer>(scores_shape, [0], [-2]) ),
                                              tf.concat( [scores_shape[-2], scores_shape[-1]], 0) ], 0);

        var _causal_mask_shape := TFShape.Create(causal_mask_shape.ToArray<Integer>);
        causal_mask            := _lower_triangular_mask(_causal_mask_shape);
    end else
    begin
        causal_mask := nil;
    end;
    var scores_mask := _merge_masks(v_mask, causal_mask);
    var tup := _apply_scores(scores, v, scores_mask, training);
    var res             := tup.Value1;
    var attention_scores:= tup.Value2;

    if q_mask <> nil then
    begin
        // Mask of shape [batch_size, Tq, 1].
        q_mask := tf.expand_dims(q_mask, -1);
        res    := TTensor(res) * tf.cast(q_mask, res.dtype);
    end;
    if return_attention_scores then
    begin
        Result := TFTensors.Create([res, attention_scores]);
        Exit;
    end;
    Result := TFTensors.Create(res);
end;

function BaseDenseAttention.compute_mask(inputs, mask: TFTensors): TFTensor;
begin
    _validate_call_args(inputs, mask);
    if mask <> nil then
    begin
        var q_mask := mask[0];
        if q_mask = nil then
            Exit( nil );
        Result := tf.convert_to_tensor(q_mask);
        Exit;
    end;
    Result := nil;
end;

procedure BaseDenseAttention._validate_call_args(inputs, mask: TFTensors);
begin
    if (inputs.Count < 2) or (inputs.Count > 3)  then
        raise TFException.Create(name+' layer accepts inputs list of length 2 or 3, namely [query, value] or [query, value, key]. Received length: '+ inputs.Count.ToString);
    if mask <> nil then
        if (mask.Count < 2) or (mask.Count > inputs.Count) then
           raise TFException.Create(name + ' layer mask must be a list of length 2, namely [query_mask, value_mask]. Received length: '+ mask.Count.ToString);
end;

class function BaseDenseAttention._lower_triangular_mask(shape: TFShape): TFTensor;
begin
    var row_index := tf.cumsum(tf.ones(shape, tf.int32_t), -2);
    var col_index := tf.cumsum(tf.ones(shape, tf.int32_t), -1);
    Result := tf.greater_equal(row_index, col_index);
end;

class function BaseDenseAttention._merge_masks(x, y: TFTensor): TFTensor;
begin
    if x = nil then
        Exit( y );
    if y = nil  then
        Exit( x );

    Result := tf.logical_and(x, y);
end;

function BaseDenseAttention.get_config: LayerArgs;
begin
   Result := args;
end;

function BaseDenseAttention.getCausal: Boolean;
begin
    Result := args.causal
end;

function BaseDenseAttention.getDropout: Single;
begin
    Result := args.dropout
end;

{ Attention }

constructor Attention.Create(_args: AttentionArgs);
begin
    inherited Create(_args);
    args := _args ;
    if (score_mode <> 'dot') and (score_mode <> 'concat') then
       raise TFException.Create('Received: score_mode='+score_mode+'. Acceptable values are: ["dot", "concat"]');
end;

procedure Attention.Build(input_shape: TFShape);
begin
    if use_scale  then
      scale := add_weight('scale', 1, DType, tf.ones_initializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, true)
    else
      scale := nil;

    if score_mode = 'concat' then
        concat_score_weight := add_weight('concat_score_weight', 1, DType, tf.ones_initializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, true)
    else
        concat_score_weight := nil;

    inherited Build(input_shape);
end;

function Attention._calculate_scores(query, key: TFTensor): TFTensor;
begin
    var scores: TFTensor := nil;
    if score_mode = 'dot' then
    begin
        //scores = tf.matmul(query, key, transpose_b: true);
        //scores = tf.matmul(tf.squeeze(query),tf.squeeze(key), transpose_b: true);
        scores := tf.linalg.einsum('bij,bkj->bik', TFTensors.Create([query, key]));
        if scale <> nil then
            scores := TTensor(scores) * scale.AsTensor;
    end
    else if score_mode = 'concat' then
    begin
        // Reshape tensors to enable broadcasting.
        // Reshape into [batch_size, Tq, 1, dim].
        var q_reshaped := tf.expand_dims(query, -2);
        // Reshape into [batch_size, 1, Tv, dim].
        var k_reshaped := tf.expand_dims(key, -3);
        var tA : TAxis := -1;
        if scale <> nil then
            scores := TTensor(concat_score_weight.AsTensor) * tf.reduce_sum(tf.tanh(TTensor(scale.AsTensor) * (TTensor(q_reshaped) + k_reshaped)), @tA)
        else
            scores := TTensor(concat_score_weight.AsTensor) * tf.reduce_sum(tf.tanh(TTensor(q_reshaped) + k_reshaped), @tA);
    end;
    Result := scores;
end;

function Attention.get_config: LayerArgs;
begin
    Result := args;
end;

function Attention.GetScoreMode: string;
begin
    Result := args.score_mode;
end;

function Attention.GetUse_Scale: Boolean;
begin
    Result := args.use_scale;
end;

{$ENDREGION}

{$REGION 'Regularization'}
{ Dropout }

constructor Dropout.Create(_args: DropoutArgs);
begin
    inherited Create(_args);

    args := _args;
end;

function Dropout.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var b : Boolean ;
    if training = nil then b := False
    else                   b := training^;
    training := @b;
    // function nn_internal.dropout(x, keep_prob, noise_shape: TFTensor; seed: PInteger; name: string; rate: PSingle): TFTensor;
    var true_pred : TFunc<TFTensor> := function : TFTensor
                                              begin
                                                 Result := tf.nn.dropout(inputs.First, nil,  get_noise_shape(inputs.First), @args.Seed, '', @args.Rate);
                                              end;

    var false_pred : TFunc<TFTensor> := function : TFTensor
                                              begin
                                                 Result := tf.identity(inputs.First);
                                              end;

    var output := smart_module.smart_cond(training^, true_pred, true_pred );

    Result := TFTensors.Create(output);
end;

function Dropout.get_noise_shape(inputs: TFTensor): TFTensor;
begin
    if args.NoiseShape.isNull then
        Exit(nil);

    Result := nil;
end;
{$ENDREGION}

{$REGION 'Core'}
{ Dense }

constructor Dense.Create(_args: DenseArgs);
begin
    inherited Create(_args);

    args := _args;
    FSupportsMasking := true;
    FinputSpec       := TInputSpec.Create(DtInvalid,null, 2);
end;

procedure Dense.Build(input_shape: TFShape);
begin

    var last_dim              := input_shape.dims[ High(input_shape.dims) ];
    var axes := TDictionary<Integer, Integer>.Create;
    axes.AddOrSetValue(-1,last_dim);
    FinputSpec := TInputSpec.Create(DtInvalid,null, 2, axes);

    kernel := add_weight('kernel', TFShape.Create([last_dim, args.Units]), DType, args.KernelInitializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE,True);

    if args.UseBias then
        bias := add_weight('bias', TFShape.Create([args.Units]), DType, args.BiasInitializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE,True);

    Fbuilt := true;
end;

function Dense.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  outputs : TFTensor;
  rank    : Integer;
  nda     : TArray< TArray<Integer> >;
begin
    rank    := inputs.rank;
    nda     := [ [ rank - 1 ], [ 0 ] ] ;
    if rank > 2 then
       outputs := tf.linalg.tensordot(inputs.First, kernel.AsTensor, TNDArray.Create(nda) )
    else
       outputs := gen_math_ops.mat_mul(inputs.First, kernel.AsTensor);

    if args.UseBias then
        outputs := tf.nn.bias_add(outputs, bias);
    if args.Activation <> nil then
        outputs := activation(outputs);

    Result := TFTensors.Create(outputs);
end;

function Dense.getAct: TActivation;
begin
    Result := args.Activation;
end;

{ EinsumDense }

constructor EinsumDense.Create(_args: EinsumDenseArgs);
begin
    inherited Create(_args);

    Fequation             := _args.Equation;
    Fpartial_output_shape := _args.OutputShape;
    Fbias_axes            := _args.BiasAxes;
    Factivation           := _args.Activation;
    Fkernel_initializer   := _args.KernelInitializer;
    Fbias_initializer     := _args.BiasInitializer;
    Fkernel_regularizer   := _args.KernelRegularizer;
    Fbias_regularizer     := _args.BiasRegularizer;
    Fkernel_constraint    := _args.KernelConstraint;
    Fbias_constraint      := _args.BiasConstraint;
end;

procedure EinsumDense.Build(input_shape: TFShape);
var
 kernel_shape,
 bias_shape  : TFShape;
 shape_data  : Tuple<TFShape, TFShape, TFShape>;
begin
    shape_data     := _analyze_einsum_string(Fequation, Fbias_axes, input_shape, Fpartial_output_shape);

    kernel_shape       := shape_data.Value1;
    bias_shape         := shape_data.Value2;
    Ffull_output_shape := shape_data.Value3;

    Fkernel := add_weight('kernel', kernel_shape, DType, Fkernel_initializer, Fkernel_regularizer, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE,True);

    if bias_shape <> nil then
        Fbias := add_weight('bias', bias_shape, DType, Fbias_initializer, Fbias_regularizer, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE,True)
    else
        Fbias := nil;

    inherited build(input_shape);
end;

function EinsumDense.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := Ffull_output_shape;
end;

function EinsumDense.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var ret := tf.linalg.einsum(Fequation, TFTensors.Create([inputs.First,Fkernel.AsTensor]));
    if Fbias <> nil then
        ret := TTensor(ret) + Fbias.AsTensor;

    if Assigned(Factivation) then
        ret := Factivation(ret);

    Result := TFTensors.Create(ret);
end;

class function EinsumDense._analyze_einsum_string(equation, bias_axes: string; input_shape, output_shape: TFShape): Tuple<TFShape, TFShape, TFShape>;
begin
    var dot_replaced_string := TRegex.Replace(equation, '\.\.\.', '0');

    // This is the case where no ellipses are present in the string.
    var split_string := TRegex.Match(dot_replaced_string, '([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)');
    if split_string.Success then
        Exit( _analyze_split_string(split_string, bias_axes, input_shape, output_shape) );

    // This is the case where ellipses are present on the left.
    split_string := TRegex.Match(dot_replaced_string, '0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)');
    if split_string.Success then
        Exit( _analyze_split_string(split_string, bias_axes, input_shape, output_shape, true) );

    // This is the case where ellipses are present on the right.
    split_string := TRegex.Match(dot_replaced_string, '([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0');
    if split_string.Success then
        Exit( _analyze_split_string(split_string, bias_axes, input_shape, output_shape) );

    raise TFException.Create('Invalid einsum equation "'+ equation +'." Equations must be in the form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....');
end;

class function EinsumDense._analyze_split_string(split_string: TMatch; bias_axes: string; input_shape, output_shape: TFShape;
  left_elided: Boolean): Tuple<TFShape, TFShape, TFShape>;
var
  bias_shape,
  _output_shape  : TList<Integer>;
  output_dim_map : TDictionary<char, Integer>;
  input_dim_map  : TDictionary<char, Integer>;
  input_spec,
  weight_spec,
  output_spec    : string;
  elided         : Integer;
begin
    input_spec := split_string.Groups[1].Value;
    weight_spec:= split_string.Groups[2].Value;
    output_spec:= split_string.Groups[3].Value;

    elided := input_shape.ndim - input_spec.Length;
    _output_shape := TList<Integer>.Create;
    _output_shape.Add(input_shape[0]);
    _output_shape.AddRange(output_shape.as_int_list);

    if (elided > 0) and (left_elided) then
        for var i := 1 to elided - 1 do
            // We already inserted the 0th input dimension at dim 0, so we need to
            // start at location 1 here.
            _output_shape.Insert(1, input_shape[i])
    else if (elided > 0) and ( not left_elided) then
        for var i := input_shape.ndim - elided  to (input_shape.ndim - (input_shape.ndim - elided)) - 1 do
            _output_shape.Add(input_shape[i]);

    if left_elided then
    begin
        // If we have beginning dimensions elided, we need to use negative indexing
        // to determine where in the input dimension our values are.
        //input_dim_map = { dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec) }
        input_dim_map := TDictionary<char, Integer>.Create;
        for var i := 1 to input_spec.Length do
            input_dim_map.AddOrSetValue(input_spec[i], i + elided - input_shape.ndim);

        // Because we've constructed the full output shape already, we don't need
        // to do negative indexing.
        //output_dim_map = { dim: (i + elided) for i, dim in enumerate(output_spec)}
        output_dim_map := TDictionary<char, Integer>.Create;
        for var i := 1 to output_spec.Length do
            output_dim_map.AddOrSetValue(output_spec[i], i + elided);
    end else
    begin
        input_dim_map := TDictionary<char, Integer>.Create;
        for var i := 1 to input_spec.Length do
            input_dim_map.AddOrSetValue(input_spec[i], i);

        output_dim_map := TDictionary<char, Integer>.Create;
        for var i := 1 to output_spec.Length do
            output_dim_map.AddOrSetValue(output_spec[i], i);
    end;

    for var dim in input_spec do
    begin
        var input_shape_at_dim := input_shape[ input_dim_map[dim]-1 ];
        var index : Integer;
        if output_dim_map.TryGetValue(dim, index) then
        begin
            var output_shape_at_dim := _output_shape[index-1];
            if (output_shape_at_dim <> -1) and (output_shape_at_dim <> input_shape_at_dim) then
               raise TFException.Create('Input shape and output shape do not match at shared dimension '+ dim + sLineBreak +
                                        '.Input shape is '+input_shape_at_dim.ToString+', and output shape is '+ output_shape[output_dim_map[dim]].ToString+'.');
        end;
    end;

    for var dim in output_spec do
    begin
        if ( not input_spec.Contains(dim) ) and ( not weight_spec.Contains(dim) )  then
        begin
           raise TFException.Create('Dimension '+dim+' was specified in the output '+ output_spec + sLineBreak +
                                    'but has no corresponding dim in the input spec '+ input_spec + 'or weight spec '+ output_spec );
        end;
    end;

    var weight_shape := TList<Int64>.Create;
    for var dim in weight_spec do
    begin
        if input_dim_map.ContainsKey(dim) then
            weight_shape.add( input_shape[ input_dim_map[dim]-1 ] )
        else if output_dim_map.ContainsKey(dim) then
            weight_shape.add(_output_shape[ output_dim_map[dim]-1 ])
        else
           raise TFException.Create('Weight dimension '+dim+' did not have a match in ' + sLineBreak +
                                    'either the input spec '+ input_spec +
                                    'or the output spec '+ output_spec + '. For this layer, the weight must be fully specified.');
    end;

    if bias_axes <> '' then
    begin
        var num_left_elided : Integer;
        if left_elided then num_left_elided := elided
        else                num_left_elided := 0;

        var idx_map := TDictionary<char, Integer>.Create;
        for var i := 1 to output_spec.Length do
            idx_map.AddOrSetValue( output_spec[i], _output_shape[(i-1) + num_left_elided] );

        for var _char in bias_axes do
            if not output_spec.Contains(_char) then
               raise TFException.Create('Bias dimension '+ _char +' was requested, but is not part of the output spec '+ output_spec);

        var lstIdx := TList<Integer>.Create;
        for var  _char in bias_axes do
        begin
            lstIdx.Add(output_spec.IndexOf(_char));
        end;
        var e := Enumerable<Integer>.Create(lstIdx.ToArray);
        var first_bias_location := e.Min;

        var bias_output_spec := output_spec.Substring(first_bias_location);

        lstIdx.Clear;
        lstIdx.Pack;
        for var  _char in bias_output_spec do
        begin
           if bias_axes.Contains(_char) then  lstIdx.Add( idx_map[_char] )
           else                               lstIdx.Add( 1 );
        end;
        bias_shape := lstIdx ;

        if not left_elided then
        begin
            for var x := 0 to  elided -1 do
                bias_shape.add(1);
        end;
    end
    else bias_shape := nil;

    var s1 : TFShape := weight_shape.ToArray;
    var s2 : TFShape;
    var s3 : TFShape := _output_shape.ToArray;
    if bias_shape <> nil then s2 := bias_shape.ToArray
    else                      s2 := TList<Integer>.Create.ToArray;

    Result := Tuple.Create( s1,s2,s3);
end;

{ Embedding }

constructor Embedding.Create(_args: EmbeddingArgs);
begin
    var lArg : LayerArgs := LayerArgs.Create;

    lArg.DType      := _args.DType;
    lArg.Name       := _args.Name;
    lArg.InputShape := _args.InputShape;
    lArg.BatchSize  := _args.BatchSize;

    if _args.InputShape.isNull then
        lArg.InputShape := _args.InputLength;

    var a : TArray<Int64> := [ _args.BatchSize ] + _args.InputShape.dims;
    if _args.BatchInputShape.isNull then
        lArg.BatchInputShape := a;

    inherited Create(lArg);

    args := _args;

    if Assigned( args.EmbeddingsInitializer) then embeddings_initializer :=  args.EmbeddingsInitializer
    else                                          embeddings_initializer :=  tf.random_uniform_initializer;
    FSupportsMasking := mask_zero;
end;

procedure Embedding.Build(input_shape: TFShape);
begin
    tf.Context.eager_mode;

    embeddings := add_weight('embeddings', TFShape.Create([input_dim, output_dim]), TF_DataType.TF_FLOAT, embeddings_initializer );

    tf.Context.graph_mode;
    Fbuilt := true;
end;

function Embedding.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var dtype := inputs.dtype;
    if (dtype <> tf.int32_t) and (dtype <> tf.int64_t) then
        inputs := TFTensors.Create(  math_ops.cast(inputs.First, tf.int32_t) );

    var outputs := embedding_ops.embedding_lookup(embeddings, inputs.First);
    Result := TFTensors.Create(outputs);
end;

function Embedding.getInput_dim: Integer;
begin
    Result := args.InputDim;
end;

function Embedding.getmask_zero: Boolean;
begin
    Result := args.MaskZero
end;

function Embedding.getoutput_dim: Integer;
begin
    Result := args.OutputDim;
end;

{ InputLayer }

constructor InputLayer.Create(_args: InputLayerArgs);
begin
    inherited Create(_args);

    args             := _args;
    Fbuilt           := true;
    FSupportsMasking := true;

    if not BatchInputShape.isNull then
    begin
        args.BatchSize  := BatchInputShape.dims[0];
        var a :=  BatchInputShape.dims;
        Delete(a,0,1);
        args.InputShape := a;
    end;

    // moved to base class
    if string.IsNullOrEmpty(args.Name) then
    begin
        var prefix := 'input';
        Fname      := prefix + '_' + tf.keras.backend.get_uid(prefix).ToString;
        args.Name  := name;
    end;

    if args.DType = TF_DataType.DtInvalid then
    begin
        if args.InputTensor = nil then args.DType := tf.float32_t
        else                           args.DType := args.InputTensor.dtype;
    end;

    if args.InputTensor = nil then
    begin
        if not args.InputShape.isNull then
        begin
            var aBatch : TArray<Int64> := [args.BatchSize] + args.InputShape.dims;
            args.BatchInputShape := aBatch;
        end else
        begin
            args.BatchInputShape := System.default(TFShape);
        end;

        var graph := tf.keras.backend.get_graph;
        graph.as_default;

        var batch := BatchInputShape;
        args.InputTensor := tf.keras.backend.placeholder(@batch, -1, DType, args.Sparse, Name, args.Ragged);

        graph.gExit;

        isPlaceholder := true;
    end;

    // Create an input node to add to self.outbound_node
    // and set output_tensors' _keras_history.
    // input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
    // input_tensor._keras_mask = None
    var nA : NodeArgs := NodeArgs.Create;
    nA.Outputs := TFTensors.Create( args.InputTensor );
    var node := Node.Create(nA);
    node.Connect(Self);

    typeSpec := TensorSpec.Create(args.InputTensor.shape, args.InputTensor.dtype, Name);
end;

{ MultiHeadAttention }

constructor MultiHeadAttention.Create(_args: MultiHeadAttentionArgs);
begin
    inherited Create(_args) ;

    Fquery_shape          := System.Default(TFShape);
    Fkey_shape            := System.Default(TFShape);
    Fvalue_shape          := System.Default(TFShape);
    Fbuilt_from_signature := False;
    Fquery_dense          := nil;
    Fkey_dense            := nil;
    Fvalue_dense          := nil;
    Foutput_dense         := nil;
    Fdot_product_equation := '';
    Fcombine_equation     := '';
    Fsoftmax              := nil;
    Fdropout_layer        := nil;

    args := _args;
end;

class function MultiHeadAttention._build_attention_equation(rank: Integer; attn_axes: TFShape): Tuple<string, string, Integer>;
begin
    var target_notation := _CHR_IDX.Substring(0, rank);
    // `batch_dims` includes the head dim.
    // batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    // Since range(rank) is an IEnumerable like (0, 1, 2 ...) whose index is equal to its value
    // use IEnumerable.Except instead of np.delete which is unavailable
    var e  : Enumerable<integer> := TUtils.range(rank);
    var e1 : TArray<integer> := attn_axes.as_int_list + [rank - 1];
    var batch_dims := e.ExceptWith(e1);

    var letter_offset   := rank;
    var source_notation := '';
    for var i := 0 to rank-1 do
    begin
        if (batch_dims.Contains(i)) or ( i = rank - 1) then
            source_notation := source_notation + target_notation[i+1]
        else begin
            source_notation := source_notation + _CHR_IDX[1+letter_offset];
            letter_offset   := letter_offset + 1;
        end;
    end;
    var tn  : string :='';
    var tn1 : string :='';
    var sn  : string :='';
    for var i in batch_dims do            tn  := tn  + target_notation[i+1];
    for var i in attn_axes.as_int_list do tn1 := tn1 + target_notation[i+1];
    for var i in attn_axes.as_int_list do sn  := sn  + source_notation[i+1];

    var product_notation := tn + tn1 + sn;

    var dot_product_equation := Format('%s,%s->%s',[source_notation,target_notation,product_notation]);
    var attn_scores_rank     := product_notation.Length;
    var combine_equation     := Format('%s,%s->%s',[product_notation,source_notation,target_notation]);

    Result := Tuple.Create(dot_product_equation, combine_equation, attn_scores_rank);
end;

class function MultiHeadAttention._build_proj_equation(free_dims, bound_dims, output_dims: Integer): Tuple<string, string, Integer>;
begin
    var _char : char;
    var input_str  := '';
    var kernel_str := '';
    var output_str := '';
    var bias_axes  := '';
    var letter_offset := 0;
    for var i in TUtils.range(free_dims) do
    begin
        _char      := _CHR_IDX[i+1 + letter_offset];
        input_str  := input_str +_char;
        output_str := output_str + _char;
    end;
    letter_offset := letter_offset + free_dims;
    for var i in TUtils.range(bound_dims) do
    begin
        _char      := _CHR_IDX[i+1 + letter_offset];
        input_str  := input_str + _char;
        kernel_str := kernel_str +_char;
    end;
    letter_offset := letter_offset +  bound_dims;
    for var i in TUtils.range(output_dims) do
    begin
        _char := _CHR_IDX[i+1 + letter_offset];
        kernel_str := kernel_str + _char;
        output_str := output_str + _char;
        bias_axes  := bias_axes  + _char;
    end;
    var equation := Format('%s,%s->%s',[input_str,kernel_str,output_str]);
    Result := Tuple.Create(equation, bias_axes, output_str.Length);
end;

class function MultiHeadAttention._get_output_shape(output_rank: Integer; known_last_dims: TFShape): TFShape;
begin
    var a : TArray<Integer> := [];
    for var i in TUtils.range(output_rank - known_last_dims.rank) do
        a := a + [-1];

    Result := a + known_last_dims.as_int_list;
end;

procedure MultiHeadAttention._build_from_signature(query, value, key: TFTensor);
begin
    var s : TFShape := System.default(TFShape);
    if Assigned(key)  then s := key.shape;

    _build_from_signature(query.shape, value.shape, s);
end;

procedure MultiHeadAttention._build_from_signature(query, value, key: TFShape);
begin
    Fbuilt_from_signature := true;
    Fquery_shape          := query;
    Fvalue_shape          := value;

    if key = nil then   Fkey_shape := Fvalue_shape
    else                Fkey_shape := key;
    // Any setup work performed only once should happen in an `init_scope`
    // to avoid creating symbolic Tensors that will later pollute any eager
    // operations.
    TUtils.tf_with<TNameScope>(tf.init_scope,
                 procedure(v1: TNameScope)
                        begin
                            var free_dims := Fquery_shape.rank - 1;
                            var t_be := _build_proj_equation(free_dims, 1, 2);
                            var einsum_equation := t_be.Value1;
                            var bias_axes       := t_be.Value2;
                            var output_rank     := t_be.Value3;

                            var sBias : string := '';
                            if args.UseBias then  sBias := bias_axes;

                            Fquery_dense := _get_dense(einsum_equation, _get_output_shape(output_rank - 1, TFShape.Create([args.NumHeads, args.KeyDim])), sBias, 'query');

                            var t_bp := _build_proj_equation(Fkey_shape.rank - 1, 1, 2);
                            einsum_equation := t_bp.Value1;
                            bias_axes       := t_bp.Value2;
                            output_rank     := t_bp.Value3;
                            Fkey_dense := _get_dense(einsum_equation, _get_output_shape(output_rank - 1, TFShape.Create([args.NumHeads, args.KeyDim])), sBias, 'key');

                            var t_bp1 := _build_proj_equation(Fvalue_shape.rank - 1, 1, 2);
                            einsum_equation := t_bp1.Value1;
                            bias_axes       := t_bp1.Value2;
                            output_rank     := t_bp1.Value3;

                            var vDim : Integer ;
                            if args.ValueDim = nil then vDim  := args.KeyDim
                            else                        vDim  := args.ValueDim;
                            Fvalue_dense := _get_dense(einsum_equation, _get_output_shape(output_rank - 1, TFShape.Create([args.NumHeads, vDim])), sBias, 'value');
                            // Builds the attention computations for multi-head dot product attention.
                            // These computations could be wrapped into the keras attention layer once
                            // it support mult-head einsum computations.
                            _build_attention(output_rank);
                            Foutput_dense := _build_output_dense(free_dims, 'attention_output');
                        end);


    StackLayers([Fquery_dense, Fkey_dense, Fvalue_dense, Foutput_dense]);
end;

function MultiHeadAttention._get_dense(equation: string; output_shape: TFShape; bias_axes, name: string): EinsumDense;
begin
    var eD : EinsumDenseArgs := EinsumDenseArgs.Create;

    eD.Equation          := equation;
    eD.OutputShape       := output_shape;
    eD.BiasAxes          := bias_axes;
    eD.Name              := name;
    eD.KernelInitializer := args.KernelInitializer;
    eD.BiasInitializer   := args.BiasInitializer;
    eD.KernelRegularizer := args.KernelRegularizer;
    eD.BiasRegularizer   := args.BiasRegularizer;
    eD.KernelConstraint  := args.KernelConstraint;
    eD.BiasConstraint    := args.BiasConstraint;

    Result := EinsumDense.Create(eD);
end;

function MultiHeadAttention._build_output_dense(free_dims: Integer; name: string): EinsumDense;
begin
    if args.OutputShape.IsNull then args.OutputShape := TFShape.Create([ Fquery_shape[-1] ]);
    var bE := _build_proj_equation(free_dims, 2,  args.OutputShape.ndim);
    var einsum_equation := bE.Value1;
    var bias_axes       := bE.Value2;
    var output_rank     := bE.Value3;

    var sBias : string := '';
    if args.UseBias then  sBias := bias_axes;
    Result := _get_dense(einsum_equation, _get_output_shape(output_rank - 1, args.OutputShape), sBias, name);
end;

procedure MultiHeadAttention._build_attention(rank: Integer);
begin
    if args.AttentionAxis = nil then
        args.AttentionAxis := TFShape.Create(TUtils.range(1, rank - 2).ToArray);

    var attn_scores_rank : Integer;
    var bA := _build_attention_equation(rank, args.AttentionAxis);
    Fdot_product_equation := bA.Value1;
    Fcombine_equation     := bA.Value2;
    attn_scores_rank      := bA.Value3;

    var norm_axes := TUtils.range(attn_scores_rank - args.AttentionAxis.ndim, attn_scores_rank).ToArray;

    var sArg := SoftmaxArgs.Create; sArg.axis := norm_axes;
    Fsoftmax := Softmax.Create(sArg);

    var dArg       := DropoutArgs.Create; dArg.Rate := args.Dropout;
    Fdropout_layer := Dropout.Create(dArg);
end;

function MultiHeadAttention._masked_softmax(attention_scores, attention_mask: TFTensor): TFTensor;
begin
    if attention_mask <> nil then
    begin
        var mask_expansion_axis := -args.AttentionAxis.ndim * 2 - 1;
        for var i := 0 to (attention_scores.shape.ndim - attention_mask.shape.ndim) - 1 do
            attention_mask := tf.expand_dims(attention_mask, mask_expansion_axis);
    end;
    var tScore : TFTensors;
    if attention_mask = nil then tScore := TFTensors.Create(attention_scores)
    else                         tScore := TFTensors.Create([attention_scores, attention_mask]) ;
    Result := Fsoftmax.Apply(tScore).First;
end;

function MultiHeadAttention._compute_attention(query, key, value, attention_mask: TFTensor; training: Boolean): TFTensors;
var
  s : TFShape;
begin
    // Note: Applying scalar multiply at the smaller end of einsum improves
    // XLA performance, but may introduce slight numeric differences in
    // the Transformer attention head.
    s := query.shape;
    query := tf.multiply( query, Single(1) / TTensor(tf.sqrt( tf.convert_to_tensor( Single(args.KeyDim) ) ) ));
    s := query.shape;
    // Take the dot product between "query" and "key" to get the raw
    // attention scores.
    var attention_scores := tf.linalg.einsum(Fdot_product_equation, TFTensors.Create([key, query]));
    attention_scores     := _masked_softmax(attention_scores, attention_mask);
    // This is actually dropping out entire tokens to attend to, which might
    // seem a bit unusual, but is taken from the original Transformer paper.
    var attention_scores_dropout := Fdropout_layer.Apply(TFTensors.Create(attention_scores), nil, training);
    // `context_layer` = [B, T, N, H]
    var attention_output := tf.linalg.einsum(Fcombine_equation, TFTensors.Create([attention_scores_dropout.First, value]));
    Result := TFTensors.Create([attention_output, attention_scores]);
end;

function MultiHeadAttention.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var _inp : TFTensors;
    var _mask: TFTensor := nil;

    var count : Integer := inputs.Count;
    if (count < 2) or (count > 5) then
      raise Exception.Create(Format('%s layer accepts inputs list of length from 2 to 5, ' +
                                    'namely [query, value, (key), (attention_mask), (return_attention_scores)].' +
                                    'Received length: %d.',[name,count]));

    var has_bool : Boolean := inputs[count - 1].dtype = TF_DataType.TF_BOOL;
    var return_attention_scores : Boolean := false;
    if has_bool then
    begin
        return_attention_scores := Boolean(TTensor(inputs[count - 1]));
        Dec(count);
    end;

    Case count of
      2: begin
            _inp := TFTensors.Create([inputs[0], inputs[1]]);
         end;
      3: begin
            if inputs[2].shape[-1] = inputs[1].shape[-1] then
                _inp :=  TFTensors.Create( [ inputs[0], inputs[1], inputs[2] ] )
            else begin
                _inp  := TFTensors.Create( [inputs[0], inputs[1]] );
                _mask := inputs[2];
            end;
         end;
      4: begin
            _inp  := TFTensors.Create( [ inputs[0], inputs[1], inputs[2] ] );
            _mask := inputs[3];
         end;
    else
        raise Exception.Create(''); //TODO:Add discriptions for this err
    end;

    Result := call(_inp, _mask, training, return_attention_scores);
end;

function MultiHeadAttention.Call(inputs: TFTensors; attention_mask: TFTensor; training: PBoolean; return_attention_scores: Boolean): TFTensors;
begin
    var query := inputs[0];
    var value := inputs[1];
    var key : TFTensor ;
    if inputs.Count = 3 then  key := inputs[2]
    else                      key := nil;
    if not Fbuilt_from_signature then
       _build_from_signature(query, value, key);
    if key = nil then
        key := value;

    // TODO: Add RaggedTensor support
    //var query_is_ragged = query is tf.RaggedTensor;
    //if (query_is_ragged)
    //{
    //    var query_lengths = query.nested_row_lengths();
    //    query = query.to_tensor();
    //}
    //var key_is_ragged = key is tf.RaggedTensor;
    //var value_is_ragged = value is tf.RaggedTensor;
    //if (key_is_ragged && value_is_ragged)
    //{
    //    // Ensure they have the same shape.
    //    var bounding_shape = tf.math.maximum(key.bounding_shape(), value.bounding_shape());
    //    key = key.to_tensor(shape: bounding_shape);
    //    value = value.to_tensor(shape: bounding_shape);
    //}
    //else if (key_is_ragged)
    //{
    //    key = key.to_tensor(shape: tf.shape(value));
    //}
    //else if (value_is_ragged)
    //{
    //    value = value.to_tensor(shape: tf.shape(key));
    //}

    //   N = `num_attention_heads`
    //   H = `size_per_head`
    // `query` = [B, T, N ,H]
    query := Fquery_dense.Apply( TFTensors.Create(query) ).First;
    // `key` = [B, S, N, H]
    key := Fkey_dense.Apply( TFTensors.Create(key) ).First;
    // `value` = [B, S, N, H]
    value := Fvalue_dense.Apply( TFTensors.Create(value) ).First;
    var bTrain : Boolean := False;
    if training <> nil then bTrain := training^;

    var ts := _compute_attention(query, key, value, attention_mask, bTrain);
    var attention_output := ts[0];
    var attention_scores := ts[1];

    attention_output := Foutput_dense.Apply(TFTensors.Create(attention_output)).First;

    //if (query_is_ragged)
    //{
    //    attention_output = tf.RaggedTensor.from_tensor(attention_output, lengths: query_lengths);
    //}

    if return_attention_scores then
        Exit( TFTensors.Create([attention_output, attention_scores]) );

    Result := TFTensors.Create(attention_output);
end;

{$ENDREGION}

{$REGION 'Convolutional'}

{ Convolutional }

constructor Convolutional.Create(_args: ConvolutionalArgs);
begin
    inherited Create(_args) ;

    args := _args;

    args.KernelSize   := conv_utils.normalize_tuple(args.KernelSize.as_int_list, args.Rank, 'kernel_size');
    args.Strides      := conv_utils.normalize_tuple(args.Strides.as_int_list, args.Rank, 'strides');
    args.Padding      := conv_utils.normalize_padding(args.Padding);
    args.DataFormat   := conv_utils.normalize_data_format(args.DataFormat);
    args.DilationRate := conv_utils.normalize_tuple(args.DilationRate.as_int_list, args.Rank, 'dilation_rate');
    FinputSpec        := TInputSpec.Create(DtInvalid, rank + 2);
    _tf_data_format   := conv_utils.convert_data_format(data_format, rank + 2);
end;

function Convolutional.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var outputs := Fconvolution_op.Apply(inputs.first, kernel.AsTensor);
    if use_bias then
    begin
        if data_format = 'channels_first' then
           raise Exception.Create('call channels_first')
        else
           outputs := nn_ops.bias_add(outputs, bias, 'NHWC');
    end;

    if activation <> nil then
        outputs := activation(outputs);

    Result := TFTensors.create(outputs);
end;

procedure Convolutional.Build(input_shape: TFShape);
begin
    var channel_axis : Integer;
    if data_format = 'channels_first' then channel_axis := 1
    else                                   channel_axis := -1;

    var input_channel : Int64;
    if channel_axis < 0  then input_channel := input_shape.dims[input_shape.ndim + channel_axis]
    else                      input_channel := input_shape.dims[channel_axis];

    var kernel_shape : TFShape := kernel_size.dims + [ input_channel div args.Groups, filters ];

    kernel := add_weight('kernel', kernel_shape, DType, kernel_initializer, kernel_regularizer, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, true);

    if use_bias then
        bias := add_weight('bias', TFShape.Create([filters]), DType, bias_initializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, True);

    var axes := TDictionary<Integer, Integer>.Create;
    axes.Add(-1, input_channel);
    FinputSpec := TInputSpec.Create(DtInvalid, null, rank + 2, axes);

    var tf_padding : string;
    if padding = 'causal' then
        tf_padding := 'VALID'
    else
        tf_padding := padding.ToUpper;

    var tf_op_name : string := self.name;

    Fconvolution_op := nn_ops.convolution_internal(tf_padding, strides, dilation_rate, rank, tf_op_name, _tf_data_format);

    Fbuilt := true;
end;

function Convolutional._get_channel_axis: Integer;
begin
    if data_format = 'channels_first'  then Result := -1 - rank
    else                                    Result := -1 ;
end;

function Convolutional.Getactivation: TActivation;
begin
    Result := args.Activation;
end;

function Convolutional.Getbias_initializer: IInitializer;
begin
    Result := args.BiasInitializer;
end;

function Convolutional.Getdata_format: string;
begin
    Result := args.DataFormat;
end;

function Convolutional.Getdilation_rate: TFShape;
begin
    Result := args.DilationRate;
end;

function Convolutional.GetFilter: Integer;
begin
    Result := args.Filters;
end;

function Convolutional.Getkernel_initializer: IInitializer;
begin
    Result := args.KernelInitializer;
end;

function Convolutional.Getkernel_regularizer: IRegularizer;
begin
    Result := args.KernelRegularizer;
end;

function Convolutional.Getkernel_size: TFShape;
begin
    Result := args.KernelSize;
end;

function Convolutional.Getpadding: string;
begin
    Result := args.Padding;
end;

function Convolutional.GetRank: Integer;
begin
    Result := args.Rank;
end;

function Convolutional.Getstrides: TFShape;
begin
    Result := args.Strides;
end;

function Convolutional.Getuse_bias: Boolean;
begin
    Result := args.UseBias;
end;

{ Conv1D }

constructor Conv1D.Create(_args: Conv1DArgs);
begin
    inherited Create(_args) ;
end;

{ Conv2D }

constructor Conv2D.Create(_args: Conv2DArgs);
begin
    inherited Create(_args) ;
end;

{ Conv2DTranspose }

constructor Conv2DTranspose.Create(_args: Conv2DArgs);
begin
    inherited Create(_args) ;
end;

procedure Conv2DTranspose.Build(input_shape: TFShape);
begin
    if input_shape.ndim <> 4 then
       raise TFException.Create('Inputs should have rank 4. Received input shape: '+input_shape.ToString);

    //var channel_axis := _get_channel_axis;
    var input_dim    := input_shape[-1];
    var kernel_shape := TFShape.Create([kernel_size[0], kernel_size[1], filters, input_dim]);

    kernel := add_weight('kernel', kernel_shape, TF_FLOAT, kernel_initializer, kernel_regularizer, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, true);

    if use_bias then
        bias := add_weight('bias', filters, TF_FLOAT, bias_initializer, nil, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_NONE, true);

    Fbuilt := true;

end;

function Conv2DTranspose.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var inputs_shape := array_ops.shape(inputs.first);
    var batch_size   := inputs_shape[0];
    var h_axis := 1;
    var w_axis := 2 ;
    if data_format = 'channels_first' then
    begin
        h_axis := 2;
        w_axis := 3 ;
    end;
    var height := -1;
    var width  := -1;
    if inputs.shape.ndim > -1 then
    begin
        var dims := inputs.shape.dims;
        height := dims[h_axis];
        width  := dims[w_axis];
    end;

    var kernel_h := kernel_size.dims[0];
    var kernel_w := kernel_size.dims[1];
    var stride_h := strides.dims[0];
    var stride_w := strides.dims[1];

    var out_pad_h := -1;
    var out_pad_w := -1;

    // Infer the dynamic output shape:
    var out_height := conv_utils.deconv_output_length(height, kernel_h, padding, out_pad_h, stride_h, dilation_rate[0]);
    var out_width  := conv_utils.deconv_output_length(width,  kernel_w, padding, out_pad_w, stride_w, dilation_rate[1]);

    var output_shape_tensor : TFTensor;
    if data_format = 'channels_first' then
    begin
        var v : TValue := Tvalue.From<TArray<TValue>> ([batch_size, filters, out_height, out_width]);
        output_shape_tensor := array_ops.stack(v);
    end else
    begin
         var v : TValue := Tvalue.From<TArray<TValue>> ([batch_size, out_height, out_width, filters]);
        output_shape_tensor := array_ops.stack(v);
    end;
    //(x: TFTensor; kernel: IVariableV1; output_shape: TFTensor; strides: PTFShape = nil; padding: string = 'valid'; data_format : string= ''; dilation_rate: PTFShape = nil):
    var sShape  := strides;
    var sdShape := dilation_rate;
    var outputs := tf.keras.backend.conv2d_transpose(inputs.First, kernel, output_shape_tensor, @sShape, padding, data_format, @sdShape);

    if not tf.Context.executing_eagerly then
    begin
        var out_shape := ComputeOutputShape(inputs.shape);
        outputs.shape := out_shape;
    end;

    if use_bias then
        outputs := tf.nn.bias_add(outputs, bias, conv_utils.convert_data_format(data_format, 4));

    if activation <> nil then
    begin
        var res :=  activation(outputs);

        Exit ( TFTensors.Create(res) );
    end;

    Result := TFTensors.Create(outputs);
end;

function Conv2DTranspose.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    var output_shape := input_shape.dims;
    var c_axis := 3;
    var h_axis := 1;
    var w_axis := 2;
    if data_format = 'channels_first' then
    begin
        c_axis := 1;
        h_axis := 2;
        w_axis := 3;
    end;

    var kernel_h := kernel_size.dims[0];
    var kernel_w := kernel_size.dims[1];
    var stride_h := strides.dims[0];
    var stride_w := strides.dims[1];

    var out_pad_h := -1;
    var out_pad_w := -1;

    output_shape[c_axis] := filters;

    output_shape[h_axis] := conv_utils.deconv_output_length(output_shape[h_axis], kernel_h, padding, out_pad_h, stride_h, dilation_rate[0]);
    output_shape[w_axis] := conv_utils.deconv_output_length(output_shape[w_axis], kernel_w, padding, out_pad_w, stride_w, dilation_rate[1]);

    Result := TFShape.Create(output_shape);
end;
{$ENDREGION}

{$REGION 'Cropping'}

{ Cropping1D }

constructor Cropping1D.Create(_args: CroppingArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;

procedure Cropping1D.Build(input_shape: TFShape);
begin
    if args.cropping.rank <> 1 then
    begin
        // throw an ValueError exception
        raise TFException.Create('Cropping1D.Build');
    end
    else if (args.cropping.shape[0] > 2) or (args.cropping.shape[0] < 1) then
    begin
        raise TFException.Create('The `cropping` argument must be a tuple of 2 integers.');
    end;
    Fbuilt := true;

end;

function Cropping1D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.First;
    if output.rank <> 3 then
    begin
        // throw an ValueError exception
        raise TFException.Create('Expected dim=3, found dim='+ output.rank.ToString);
    end;
    if args.cropping.shape[0] = 1 then
    begin
        var crop_start : Integer := NDarray(args.cropping[0]);
        output := output[ [ Slice.Create(nil,nil), Slice.Create(crop_start, output.shape[1] - crop_start), Slice.Create(nil,nil)] ];
    end else
    begin
        var crop_start : Integer := NDarray(args.cropping[0]);
        var crop_end   : Integer := NDarray(args.cropping[1]);
        output := output[ [ Slice.Create(nil,nil), Slice.Create(crop_start, output.shape[1] - crop_end),  Slice.Create(nil,nil)] ];
    end;
    Result := TFTensors.Create(output);
end;

function Cropping1D.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    if args.cropping.shape[0] = 1 then
    begin
        var crop: Integer  := NDarray(args.cropping[0]);
        Result := TFShape.Create ( [input_shape[0], input_shape[1] - crop * 2, input_shape[2]] );
    end else
    begin
        var crop_start : Integer := NDarray(args.cropping[0]);
        var crop_end   : Integer := NDarray(args.cropping[1]);
        Result := TFShape.Create ( [input_shape[0], input_shape[1] - crop_start - crop_end, input_shape[2]]);
    end
end;

{ Cropping2D }

constructor Cropping2D.Create(_args: Cropping2DArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;

procedure Cropping2D.Build(input_shape: TFShape);
begin
  Fbuilt := true;
end;

function Cropping2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  output : TFTensor;
begin
    output := inputs.First;
    if output.rank <> 4 then
    begin
          // throw an ValueError exception
          raise TFException.Create('Expected dim=4, found dim=' + output.rank.ToString);
    end;
    if args.cropping.shape = TFShape.Create([1]) then
    begin
          var crop : Integer := NDarray(args.cropping[0]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(crop, output.shape[1] - crop),
                                              Slice.Create(crop, output.shape[2] - crop),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(crop, output.shape[2] - crop),
                                              Slice.Create(crop, output.shape[3] - crop)] ];
          end;
    end
    // a tuple of 2 integers
    else if args.cropping.shape = TFShape.Create([2]) then
    begin
          var crop_1 : Integer := NDArray(args.cropping[0]);
          var crop_2 : Integer := NDArray(args.cropping[1]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(crop_1, output.shape[1] - crop_1),
                                              Slice.Create(crop_2, output.shape[2] - crop_2),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(crop_1, output.shape[2] - crop_1),
                                              Slice.Create(crop_2, output.shape[3] - crop_2)] ];
          end;
    end
    else if (args.cropping.shape[0] = 2) and (args.cropping.shape[1] = 2) then
    begin
          var x_start : Integer := NDArray(args.cropping[[0, 0]]);
          var x_end   : Integer := NDArray(args.cropping[[0, 1]]);

          var y_start : Integer := NDArray(args.cropping[[1, 0]]);
          var y_end   : Integer := NDArray(args.cropping[[1, 1]]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(x_start, output.shape[1] - x_end),
                                              Slice.Create(y_start, output.shape[2] - y_end),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(x_start, output.shape[2] - x_end),
                                              Slice.Create(y_start, output.shape[3] - y_end)
                                              ] ];
          end;
    end;
    Result := TFTensors.Create(output);
end;

function Cropping2D.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    if args.cropping.shape = TFShape.Create([1]) then
    begin
          var crop : Integer := NDarray(args.cropping[0]);
          if args.data_format = DataFormat.channels_last then
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1] - crop * 2, input_shape[2] - crop * 2, input_shape[3]]);
          end else
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - crop * 2, input_shape[3] - crop * 2]);
          end;
    end
    // a tuple of 2 integers
    else if args.cropping.shape =  TFShape.Create([2]) then
    begin
          var crop_1 : Integer := NDArray(args.cropping[0]);
          var crop_2 : Integer := NDArray(args.cropping[1]);
          if args.data_format = DataFormat.channels_last then
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1] - crop_1 * 2, input_shape[2] - crop_2 * 2, input_shape[3]]);
          end else
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - crop_1 * 2, input_shape[3] - crop_2 * 2]);
          end;
    end
    else if args.cropping.shape = TFShape.Create([2, 2]) then
    begin
          var crop_1_start : Integer := NDArray(args.cropping[[0, 0]]);
          var crop_1_end   : Integer := NDArray(args.cropping[[0, 1]]);

          var crop_2_start : Integer := NDArray(args.cropping[[1, 0]]);
          var crop_2_end   : Integer := NDArray(args.cropping[[1, 1]]);

          if args.data_format = DataFormat.channels_last  then
          begin
              Result := TFShape.Create([input_shape[0], input_shape[1] - crop_1_start - crop_1_end, input_shape[2] - crop_2_start - crop_2_end, input_shape[3]]);
          end else
          begin
              Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - crop_1_start - crop_1_end, input_shape[3] - crop_2_start - crop_2_end]);
          end;
    end else
    begin
         raise TFException.Create('Cropping2D.ComputeOutputShape');
    end;
end;

{ Cropping3D }

constructor Cropping3D.Create(_args: Cropping3DArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;

procedure Cropping3D.Build(input_shape: TFShape);
begin
   Fbuilt := true;
end;

function Cropping3D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  output : TFTensor;
begin
    output := inputs.First;
    if output.rank <> 5 then
    begin
          // throw an ValueError exception
          raise TFException.Create('Expected dim=5, found dim=' + output.rank.ToString);
    end;
    if args.cropping.shape = TFShape.Create([1]) then
    begin
          var crop : Integer := NDarray(args.cropping[0]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(crop, output.shape[1] - crop),
                                              Slice.Create(crop, output.shape[2] - crop),
                                              Slice.Create(crop, output.shape[3] - crop),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(crop, output.shape[2] - crop),
                                              Slice.Create(crop, output.shape[3] - crop),
                                              Slice.Create(crop, output.shape[4] - crop)] ];
          end;
    end
    // int[1][3] equivalent to a tuple of 3 integers
    else if args.cropping.shape = TFShape.Create([3]) then
    begin
          var crop_1 : Integer := NDArray(args.cropping[0]);
          var crop_2 : Integer := NDArray(args.cropping[1]);
          var crop_3 : Integer := NDArray(args.cropping[2]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(crop_1, output.shape[1] - crop_1),
                                              Slice.Create(crop_2, output.shape[2] - crop_2),
                                              Slice.Create(crop_3, output.shape[3] - crop_3),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(crop_1, output.shape[2] - crop_1),
                                              Slice.Create(crop_2, output.shape[3] - crop_2),
                                              Slice.Create(crop_3, output.shape[4] - crop_3)] ];
          end;
    end
    else if (args.cropping.shape[0] = 3) and (args.cropping.shape[1] = 2) then
    begin
          var x       : Integer := NDArray(args.cropping[[0, 0]]);
          var x_end   : Integer := NDArray(args.cropping[[0, 1]]);

          var y       : Integer := NDArray(args.cropping[[1, 0]]);
          var y_end   : Integer := NDArray(args.cropping[[1, 1]]);

          var z       : Integer := NDArray(args.cropping[[2, 0]]);
          var z_end   : Integer := NDArray(args.cropping[[2, 1]]);
          if args.data_format = DataFormat.channels_last then
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(x, output.shape[1] - x_end),
                                              Slice.Create(y, output.shape[2] - y_end),
                                              Slice.Create(z, output.shape[3] - z_end),
                                              Slice.Create(nil,nil)] ];
          end else
          begin
                output := output[ [Slice.Create(nil,nil),
                                              Slice.Create(nil,nil),
                                              Slice.Create(x, output.shape[2] - x_end),
                                              Slice.Create(y, output.shape[3] - y_end),
                                              Slice.Create(z, output.shape[4] - z_end)
                                              ] ];
          end;
    end;
    Result := TFTensors.Create(output);

end;

function Cropping3D.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    if args.cropping.shape = TFShape.Create([1]) then
    begin
          var crop : Integer := NDarray(args.cropping[0]);
          if args.data_format = DataFormat.channels_last then
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1] - crop * 2, input_shape[2] - crop * 2, input_shape[3] - crop * 2, input_shape[4]]);
          end else
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - crop * 2, input_shape[3] - crop * 2, input_shape[4] - crop * 2]);
          end;
    end
    // a tuple of 2 integers
    else if args.cropping.shape =  TFShape.Create([3]) then
    begin
          var crop_start_1 : Integer := NDArray(args.cropping[0]);
          var crop_start_2 : Integer := NDArray(args.cropping[1]);
          var crop_start_3 : Integer := NDArray(args.cropping[2]);
          if args.data_format = DataFormat.channels_last then
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1] - crop_start_1 * 2, input_shape[2] - crop_start_2 * 2, input_shape[3] - crop_start_3 * 2, input_shape[4]]);
          end else
          begin
                Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - crop_start_1 * 2, input_shape[3] - crop_start_2 * 2, input_shape[4] - crop_start_3 * 2]);
          end;
    end
    else if args.cropping.shape = TFShape.Create([3, 2]) then
    begin
          var x       : Integer := NDArray(args.cropping[[0, 0]]);
          var x_end   : Integer := NDArray(args.cropping[[0, 1]]);

          var y       : Integer := NDArray(args.cropping[[1, 0]]);
          var y_end   : Integer := NDArray(args.cropping[[1, 1]]);

          var z       : Integer := NDArray(args.cropping[[2, 0]]);
          var z_end   : Integer := NDArray(args.cropping[[2, 1]]);

          if args.data_format = DataFormat.channels_last  then
          begin
              Result := TFShape.Create([input_shape[0], input_shape[1] - x - x_end, input_shape[2] - y - y_end, input_shape[3] - z - z_end, input_shape[4]]);
          end else
          begin
              Result := TFShape.Create([input_shape[0], input_shape[1], input_shape[2] - x - x_end, input_shape[3] - y - y_end, input_shape[4] - z - z_end]);
          end;
    end else
    begin
         raise TFException.Create('Cropping3D.ComputeOutputShape');
    end;
end;

{$ENDREGION}

{$REGION 'Lstm-Rnn'}

{ LSTMCell }

constructor LSTMCell.Create(_args: LSTMCellArgs);
begin
   inherited Create(_args) ;

   args := _args;
end;

{ RNN }

function RNN.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
  Result :=  inherited Call(inputs, state, training);
end;

constructor RNN.Create(_args: RNNArgs);
begin
    inherited Create(PreConstruct(_args)) ;
    Fargs := _args;
    FSupportsMasking := True;
end;

function RNN.get_initial_state(inputs: TFTEnsor): TFTensor;
begin
    Result := _generate_zero_filled_state_for_cell(nil, nil);
end;

function RNN.PreConstruct(_args: RNNArgs): RNNArgs;
begin
    if _args.Kwargs = nil then
      _args.Kwargs := TDictionary<string,TValue>.Create;

    // If true, the output for masked timestep will be zeros, whereas in the
    // false case, output from previous timestep is returned for masked timestep.
    var zeroOutputForMask : Boolean := TUtils.Get<string,TValue>(_args.Kwargs,'zero_output_for_mask', false).AsType<Boolean>;

    var input_shape: TFShape;
    var propIS : TFShape := TUtils.Get<string,TValue>(_args.Kwargs,'input_shape', TValue.From<TFShape>(System.Default(TFShape))).AsType<TFShape>;
    var propID : Integer := TUtils.Get<string,TValue>(_args.Kwargs,'input_dim', $CC).AsType<Integer>;
    var propIL : Integer := TUtils.Get<string,TValue>(_args.Kwargs,'input_length', $CC).AsType<Integer>;

    if (propIS.isNil) and ((propID <> $CC) or (propIL <> $CC)) then
    begin
        if propIL = $CC then  propIL := -1;
        if propID = $CC then  propID := -1;

        input_shape := TFShape.Create([propIL, propID]) ;
        _args.Kwargs.AddOrSetValue('input_shape', TValue.From<TFShape>(input_shape));
    end;

    Result := _args;
end;

function RNN._generate_zero_filled_state_for_cell(cell: LSTMCell; batch_size: TFTensor): TFTensor;
begin
    raise Exception.Create('Not Implemented');
end;

class function RNN._is_multiple_state(state_size: TValue): Boolean;
var
  tt : TRttiType;
  p  : TRttiProperty;
  ctx: TRttiContext;
begin
    ctx := TRttiContext.Create;
    try
      tt := ctx.GetType(state_size.TypeInfo);
      p  := tt.GetProperty('Count');

      Result := p <> nil;
    finally
      ctx.Free;
    end;
end;

procedure RNN.build(input_shape: TFShape);
begin
  if not cell.Built then
     cell.build(input_shape);

end;

{ LSTM }

constructor LSTM.Create(_args: LSTMArgs);
begin
   inherited Create(_args) ;

   args := _args;

   var a : TArray<Integer>:= [units, units];
   for var i := 0 to Length(a)- 1  do
   begin
     var sShape : TFShape := TFShape.Create([-1, a[i]]);
     state_spec := state_spec + [ TInputSpec.Create(DtInvalid, null, null,nil, @sShape) ];
   end;

end;

function LSTM.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    Result := inherited Call(inputs, state, training);
end;

function LSTM.ComputeOutputShape(input_shape: TFShape): TFShape;
begin

end;

function LSTM.getUnits: Integer;
begin
   Result := args.Units;
end;

{ SimpleRNNCell }

constructor SimpleRNNCell.Create(_args: SimpleRNNArgs);
begin
    inherited Create(_args) ;
end;

procedure SimpleRNNCell.Build(input_shape: TFShape);
begin
  var input_dim := input_shape[-1];

  kernel := add_weight('kernel',           TFShape.Create([input_shape[-1], args.Units]), TF_FLOAT, args.KernelInitializer);
  kernel := add_weight('recurrent_kernel', TFShape.Create([args.Units, args.Units]),      TF_FLOAT, args.RecurrentInitializer);
  if args.UseBias then
    bias := add_weight('bias', TFShape.Create([args.Units]), TF_FLOAT, args.BiasInitializer );

  Fbuilt := true;
end;

function SimpleRNNCell.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    Result := inherited Call(inputs, state, training);
end;

{ SimpleRNN }

constructor SimpleRNN.Create(_args: SimpleRNNArgs);
begin
    inherited Create(_args) ;

    args := _args;
    cell := SimpleRNNCell.Create(_args)
end;

{ StackedRNNCells }

constructor StackedRNNCells.Create(_args: StackedRNNCellsArgs);
begin
    inherited Create(_args) ;

    if _args.Kwargs = nil then
       _args.Kwargs := TDictionary<string,TValue>.Create;

    Cells := _args.Cells;
    reverse_state_order := TUtils.Get<string,TValue>(_args.Kwargs,'reverse_state_order', false).AsType<Boolean>;

    if reverse_state_order then
    raise Exception.Create('reverse_state_order=True in StackedRNNCells will soon ' +
                           'be deprecated. Please update the code to work with the ' +
                           'natural order of states if you rely on the RNN states, eg RNN(return_state=True).');

end;

function StackedRNNCells.GetOuputSize: TValue;
begin
    var lastCell := Cells[Cells.Count - 1];

    if lastCell.output_size <> -1 then
    begin
        Result := lastCell.output_size;
    end
    else if RNN._is_multiple_state(lastCell.state_size) then
    begin
        // return ((dynamic)Cells[-1].state_size)[0];
        raise Exception.Create('Not Implemented');
    end else
    begin
        Result := Cells[-1].state_size;
    end;
end;

function StackedRNNCells.GetState_size: TValue;
begin

end;

{$ENDREGION}

{$REGION 'Merging'}
{ Merge }

constructor Merge.Create(_args: MergeArgs);
begin
    inherited Create(_args) ;
end;

procedure Merge.Build(input_shape: TFShape);
begin
  // output_shape = input_shape.dims[1^];
end;

function Merge.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    Result := _merge_function(inputs);
end;

function Merge._merge_function(inputs: TFTensors): TFTensors;
begin
    Result := nil;
end;

{ Add }

constructor Add.Create(_args: MergeArgs);
begin
    inherited Create(_args) ;
end;

function Add._merge_function(inputs: TFTensors): TFTensors;
begin
    var output := inputs[0];
    for var i := 1 to inputs.count-1 do
        output := TTensor(output) + inputs[i];

    Result := TFTensors.Create(output);
end;

{ Subtract }

constructor Subtract.Create(_args: MergeArgs);
begin
    inherited Create(_args) ;
end;

function Subtract._merge_function(inputs: TFTensors): TFTensors;
begin
    if inputs.Count <> 2 then
        raise TFException.Create('A `Subtract` layer should be called on exactly 2 inputs');

    var output : TFTensor := TTensor(inputs[0]) - inputs[1];
    Result := TFTensors.Create(output);
end;

{ Concatenate }

constructor Concatenate.Create(_args: MergeArgs);
begin
    inherited Create(_args) ;
    args := _args;
end;

procedure Concatenate.Build(input_shape: TFShape);
begin
  (*var shape_set = new HashSet<Shape>();
    var reduced_inputs_shapes = inputs.Select(x => x.shape).ToArray();
    for (var i = 0; i < reduced_inputs_shapes.Length; i++)
    {
        int seq = -1;
        Shape shape = reduced_inputs_shapes[i].Where(x =>
        {
            seq++;
            return seq != i;
        }).ToArray();
        shape_set.Add(shape);
    }*)
end;

function Concatenate.GetAxis: Integer;
begin
    Result := args.Axis;
end;

function Concatenate._merge_function(inputs: TFTensors): TFTensors;
begin
    var Res := tf.keras.backend.concatenate(inputs, axis);
    Result := TFTensors.Create(Res);
end;
{$ENDREGION}

{$REGION 'Normalization'}

{ LayerNormalization }

constructor LayerNormalization.Create(_args: LayerNormalizationArgs);
begin
    inherited Create(_args) ;
    args := _args;

    axis := args.Axis.axis;
end;

procedure LayerNormalization.Build(input_shape: TFShape);
begin
    var ndims := input_shape.ndim;
    for var idx := 0 to Length(axis) -1 do
    begin
        var x := axis[idx];
        if x < 0 then
            axis[idx] := ndims + x;
    end;

    var axis_to_dim := TDictionary<Integer, Integer>.Create;
     for var x in axis do
        axis_to_dim.AddOrSetValue(x, input_shape[x]);

    FinputSpec := TInputSpec.Create(DtInvalid, ndims, null, axis_to_dim);
    var param_dtype : TF_DataType;
    if DType = TF_DataType.DtInvalid  then  param_dtype := TF_DataType.TF_FLOAT
    else                                    param_dtype :=  DType;
    var param_shape := inputSpec.AllAxisDim;

    if scale then
        gamma := add_weight('gamma', param_shape, param_dtype, gamma_initializer{, trainable: true});

    if center then
        beta := add_weight('beta', param_shape, param_dtype, beta_initializer{,trainable: true});

    fused := _fused_can_be_used(ndims);

    Fbuilt := true;
end;

function LayerNormalization.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var outputs : TFTensor := nil;
    //var inputs_dtype := TDtypes.as_base_dtype(inputs.dtype);
    var input_shape  := inputs.shape;
    var ndims        := input_shape.ndim;

    var broadcast_shape : TArray<Integer>;
    for var i := 0 to ndims - 1 do
      broadcast_shape := broadcast_shape + [1];

    for var dim in axis do
        broadcast_shape[dim] := input_shape.as_int_list[dim];

    var _broadcast  : TFunc<IVariableV1, TFTensor> := function(v : IVariableV1): TFTensor
    begin
        if (v.shape.ndim <> ndims) and (not TUtils.SequenceEqual<Integer>(axis, [ ndims - 1 ]))  then
            Exit( tf.reshape(v.AsTensor, broadcast_shape) );
        Result := v.AsTensor;
    end;

    if fused then
    begin
        var tensor_shape := tf.shape(inputs.First);
        var pre_dim      := tf.constant(Integer(1));
        var in_dim       := tf.constant(Integer(1));
        for var dim in TUtils.range(ndims) do
        begin
            var dim_tensor := tensor_shape[dim];
            if dim < axis[0] then
                pre_dim := TTensor(pre_dim) * dim_tensor
            else
                in_dim := TTensor(in_dim) * dim_tensor;
        end;
        var vObj : TArray<TValue> := [ Integer(1), pre_dim, in_dim, Integer (1) ];
        inputs := TFTensors.Create( tf.reshape(inputs.first, vObj) );

        var scale  := tf.ones (TFShape.Create([Integer(TTensor(pre_dim))]), DType);
        var offset := tf.zeros(TFShape.Create([Integer(TTensor(pre_dim))]), DType);

        outputs := tf.nn.fused_batch_norm(inputs.first,scale, offset, nil, nil, epsilon, 'NCHW')[0];

        outputs := tf.reshape(outputs, tensor_shape);

        scale   := _broadcast(gamma);
        offset  := _broadcast(beta);

        outputs := TTensor(outputs) * tf.cast(scale, outputs.dtype);
        outputs := TTensor(outputs) + tf.cast(offset, outputs.dtype);
    end else
    begin

    end;

    // If some components of the shape got lost due to adjustments, fix that.
    outputs.shape := input_shape;

    Result := TFTensors.Create(outputs);
end;

function LayerNormalization.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    Result := input_shape;
end;

function LayerNormalization._fused_can_be_used(ndims: Integer) : Boolean;
begin
    var can_use_fused := false;
    if (axis[ High(axis) ] = ndims - 1) and (axis[ High(axis) ] - axis[0] = Length(axis) - 1) then
        can_use_fused := true;
    if (epsilon < 1.001e-5) or (DType <> tf.float32_t)  then
        can_use_fused := false;
    Result := can_use_fused;
end;

function LayerNormalization.getbeta_initializer: IInitializer;
begin
    Result := args.BetaInitializer
end;

function LayerNormalization.getcenter: Boolean;
begin
    Result := args.Center
end;

function LayerNormalization.getepsilon: Single;
begin
    Result := args.Epsilon
end;

function LayerNormalization.getgamma_initializer: IInitializer;
begin
    Result := args.GammaInitializer
end;

function LayerNormalization.getgamma_regularizer: IRegularizer;
begin
   Result := args.GammaRegularizer
end;

function LayerNormalization.getscale: Boolean;
begin
    Result := args.Scale;
end;

{ BatchNormalization }

constructor BatchNormalization.Create(_args: BatchNormalizationArgs);
begin
    inherited Create(_args) ;
    args := _args;

    axis := [];
    for var i := 0 to Length(args.Axis.dims)-1 do
       axis := axis + [ args.Axis.dims[i] ]

end;

procedure BatchNormalization.Build(input_shape: TFShape);
begin
    var ndims := input_shape.ndim;
    for var idx := 0 to Length(axis) -1 do
    begin
        var x := axis[idx];
        if x < 0 then
            axis[idx] := ndims + x;
    end;

    fused := ndims = 4;

    if fused then
    begin
        if TUtils.SequenceEqual<Integer>(axis, [ 1 ]) then
            _data_format := 'NCHW'
        else if TUtils.SequenceEqual<Integer>(axis, [ 3 ]) then
            _data_format := 'NHWC'
        else
            raise TFException.Create('Unsupported axis, fused batch norm only supports axis == [1] or axis == [3]');
    end;

    var axis_to_dim := TDictionary<Integer, Integer>.Create;
    for var x in axis do
        axis_to_dim.AddOrSetValue(x,input_shape[x]);

    FinputSpec := TInputSpec.Create(DtInvalid, ndims, null, axis_to_dim);
    var param_dtype : TF_DataType;
    if DType = TF_DataType.DtInvalid  then  param_dtype := TF_DataType.TF_FLOAT
    else                                    param_dtype :=  DType;
    var param_shape := inputSpec.AllAxisDim;

    if scale then
        gamma := add_weight('gamma', param_shape, param_dtype, gamma_initializer{, trainable: true})
    else
        raise TFException.Create('add_weight gamma');

    if center then
        beta := add_weight('beta', param_shape, param_dtype, beta_initializer{,trainable: true})
    else
        raise TFException.Create('add_weight beta');

    moving_mean := add_weight('moving_mean', param_shape, param_dtype, moving_mean_initializer, nil, VARIABLE_SYNCHRONIZATION_ON_READ, VARIABLE_AGGREGATION_MEAN, false);

    moving_variance := add_weight('moving_variance', param_shape, param_dtype, moving_variance_initializer, nil, VARIABLE_SYNCHRONIZATION_ON_READ, VARIABLE_AGGREGATION_MEAN, false);

    if renorm then
       raise TFException.Create('build when renorm is true');

    Fbuilt := true;
end;

function BatchNormalization.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
   Result := input_shape;
end;

function BatchNormalization._moments(inputs: TFTensors; reduction_axes: TArray<Integer>; keep_dims: Boolean): Tuple<TFTensor, TFTensor>;
begin
    var mean_variance := _calculate_mean_and_var(inputs, reduction_axes, keep_dims);
    if _support_zero_size_input then
       raise TFException.Create('Not Implemented');

    Result :=  mean_variance;
end;

function BatchNormalization._calculate_mean_and_var(inputs: TFTensors; reduction_axes: TArray<Integer>; keep_dims: Boolean): Tuple<TFTensor, TFTensor>;
begin
   Result := nn_impl.moments(inputs.First, reduction_axes, '', keep_dims);
end;

function BatchNormalization._support_zero_size_input: Boolean;
begin
    Result := false;
end;

function BatchNormalization._fused_batch_norm(inputs: TFTensor; training: TFTensor): TFTensor;
var
  training_value: Nullable<Boolean>;
begin
    var input_batch_size      : TFShape := System.default(TFShape);
    var use_fused_avg_updates : Boolean := true;
    var exponential_avg_factor: Single  := 0;
    if use_fused_avg_updates then
        exponential_avg_factor := 1.0 - momentum;

    var _fused_batch_norm_training : TFunc< TArray<TFTensor> > :=
              function : TArray<TFTensor>
               begin
                   Result := tf.nn.fused_batch_norm(inputs, gamma.AsTensor, beta.AsTensor, moving_mean.AsTensor, moving_variance.AsTensor, epsilon, _data_format, true, '', exponential_avg_factor);
               end;
    var _fused_batch_norm_inference : TFunc< TArray<TFTensor> > :=
              function : TArray<TFTensor>
               begin
                   Result := tf.nn.fused_batch_norm(inputs, gamma.AsTensor, beta.AsTensor, moving_mean.AsTensor, moving_variance.AsTensor, epsilon, _data_format, false);
               end;

    if (use_fused_avg_updates) and (not input_batch_size.isNull) then
       raise TFException.Create('Not Implemented');

    var res := smart_module.smart_cond(training, _fused_batch_norm_training, _fused_batch_norm_inference);
    var output   := res[0];
    var mean     := res[1];
    var variance := res[2];
    training_value := smart_module.smart_constant_value(training);

    if (not training_value.HasValue) or (training_value.HasValue and training_value.Value) then
    begin
        var momentum_tensor : TFTensor := nil;
        if not use_fused_avg_updates then
        begin
            if training_value = nil then
                momentum_tensor := smart_module.smart_cond(training, function : TArray<TFTensor>
                                                                       begin
                                                                           var f := TOps.convert_to_tensor(momentum);
                                                                           Result := [ f ]
                                                                       end,
                                                                       function : TArray<TFTensor>
                                                                       begin
                                                                           var f := TOps.convert_to_tensor(Single(1.0));
                                                                           Result := [ f ]
                                                                        end)[0]
            else
                momentum_tensor := Tops.convert_to_tensor(momentum);
        end;

        if use_fused_avg_updates then
            _assign_new_value(moving_mean, mean)
        else
            _assign_moving_average(moving_variance, variance, momentum_tensor);

        if use_fused_avg_updates then
            _assign_new_value(moving_variance, variance)
        else
            _assign_moving_average(moving_variance, variance, momentum_tensor);

        // var mean_update = _assign_moving_average(moving_mean.AsTensor(), mean, momentum_tensor);
        // var variance_update = _assign_moving_average(moving_variance.AsTensor(), variance, momentum_tensor);
        // add_update(new Tensor[] { mean_update }, inputs: true);
        // add_update(new Tensor[] { variance_update }, inputs: true);
    end;

    Result := output;
end;

procedure BatchNormalization._assign_new_value(variable: IVariableV1; value: TFTensor);
begin
    var vValues : TArray<TValue> := [TValue.From<IVariableV1>(variable), value, momentum];
    TUtils.tf_with<TNameScope>( TOps.name_scope('AssignNewValue', '', @vValues),
        procedure (v1: TNameScope)
          begin
              var scope := v1.ToString;
              // var cm = ops.colocate_with(variable);
              variable.assign_lazy_load(value, scope);
          end );
end;

procedure BatchNormalization._assign_moving_average(variable: IVariableV1; value: TFTensor; momentum: TFTensor);
begin
    var vValues : TArray<TValue> := [TValue.From<IVariableV1>(variable), value, momentum];
    TUtils.tf_with<TNameScope>( TOps.name_scope('AssignMovingAvg', '', @vValues),
        procedure (v1: TNameScope)
          begin
              // var cm = ops.colocate_with(variable);
              var scope := v1.ToString;
              var decay        := Tops.convert_to_tensor(Single(1.0) - TTensor(momentum), Dtinvalid, 'decay');
              var update_delta := (TTensor(variable.AsTensor) - math_ops.cast(value, variable.dtype)) * decay;
              variable.assign_sub_lazy_load(update_delta, scope);
          end );
end;

function BatchNormalization.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  outputs,
  training_tensor : TFTensor ;
  input_shape     : TFShape;
  ndims           : Integer;
  training_value  : Nullable<Boolean>;
  keep_dims       : Boolean;
  reduction_axes  : TArray<Integer>;
  broadcast_shape : TArray<Integer>;
  scale,offset    : IVariableV1;
  mean            : TFTensor;
  variance        : TFTensor;
  offset_tensor   : TFTensor;
  scale_tensor    : TFTensor;
  mean_variance   : Tuple<TFTensor, TFTensor>;
begin
    if training = nil then training_tensor := tf.placeholder(tf.bool_t, TFShape.Scalar)
    else                   training_tensor := tf.logical_and(training^, Trainable);
    if fused then
    begin
        // var training = tf.convert_to_tensor(training);
        outputs := _fused_batch_norm(inputs.First, training_tensor);
        Result := TFTensors.Create(outputs);
        Exit;
    end;

    //var inputs_dtype := Tdtypes.as_base_dtype(inputs.dtype);
    input_shape  := inputs.shape;
    ndims        := input_shape.ndim;

    for var i := 0 to  ndims - 1 do
    begin
        if not TArray.Contains<Integer>(axis, i) then
           reduction_axes := reduction_axes + [ i ];
    end;

    // Broadcasting only necessary for single-axis batch norm where the axis is
    // not the last dimension
    for var i := 0 to  ndims - 1 do
        broadcast_shape := broadcast_shape + [ 1 ];

    broadcast_shape[ axis[0] ] := input_shape.dims[ axis[0] ];

    scale  := gamma;
    offset := beta;
    offset_tensor := math_ops.cast(offset, inputs.dtype);
    scale_tensor  := math_ops.cast(scale, inputs.dtype);

    training_value := smart_module.smart_constant_value(training_tensor);

    if (training_value.HasValue) and (training_value.Value = false) then
    begin
        mean     := moving_mean.AsTensor;
        variance := moving_variance.AsTensor;
    end else
    begin
        keep_dims := Length(axis) > 1;
        mean_variance := _moments(inputs, reduction_axes, keep_dims);
        mean     := mean_variance.Value1;
        variance := mean_variance.Value2;

        mean := smart_module.smart_cond(training_tensor,
                                               function : TArray<TFTensor>
                                                begin
                                                    Result := [mean]
                                                end,
                                               function : TArray<TFTensor>
                                                begin
                                                    Result := [ Tops.convert_to_tensor(TValue.From<IVariableV1>(moving_mean)) ]
                                                end)[0] ;

        variance := smart_module.smart_cond(training_tensor,
                                                       function : TArray<TFTensor>
                                                         begin
                                                             Result := [variance]
                                                         end,
                                                       function : TArray<TFTensor>
                                                         begin
                                                             Result := [ Tops.convert_to_tensor(TValue.From<IVariableV1>(moving_variance)) ]
                                                         end)[0] ;
    end;

    mean          := math_ops.cast(mean, inputs.dtype);
    variance      := math_ops.cast(variance, inputs.dtype);
    outputs       := nn_impl.batch_normalization(inputs.First, mean, variance, offset_tensor, scale_tensor, epsilon);
    // If some components of the shape got lost due to adjustments, fix that.
    outputs.shape     := input_shape;
    Result := TFTensors.Create( outputs );
end;

function BatchNormalization.getbeta_initializer: IInitializer;
begin
   Result := args.BetaInitializer
end;

function BatchNormalization.getcenter: Boolean;
begin
    Result := args.Center
end;

function BatchNormalization.getepsilon: Single;
begin
    Result := args.Epsilon
end;

function BatchNormalization.getgamma_initializer: IInitializer;
begin
    Result := args.GammaInitializer
end;

function BatchNormalization.getgamma_regularizer: IRegularizer;
begin
    Result := args.GammaRegularizer
end;

function BatchNormalization.getMomentum: Single;
begin
    Result := args.Momentum
end;

function BatchNormalization.getmoving_mean_initializer: IInitializer;
begin
    Result := args.MovingMeanInitializer
end;

function BatchNormalization.getmoving_variance_initializer: IInitializer;
begin
    Result := args.MovingVarianceInitializer;
end;

function BatchNormalization.getrenorm: Boolean;
begin
    Result := args.Renorm;
end;

function BatchNormalization.getscale: Boolean;
begin
    Result := args.Scale
end;
{$ENDREGION}

{$REGION 'Pooling'}

{ Pooling1D }

constructor Pooling1D.Create(_args: Pooling1DArgs);
begin
   inherited Create(_args) ;

   args := _args;

   args.Padding    := conv_utils.normalize_padding(args.Padding);
   args.DataFormat := conv_utils.normalize_data_format(args.DataFormat);
   input_spec      := TInputSpec.Create(DtInvalid, 3);

end;

function Pooling1D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  pad_axis,ndim : Integer;
  pool_shape    : TArray<Integer>;
  strides       : TArray<Integer>;
begin

    if args.DataFormat = 'channels_first' then pad_axis:= 2
    else                                       pad_axis:= 3;

    inputs := TFTensors.Create( tf.expand_dims(inputs.first, pad_axis) );
    pool_shape := [ args.PoolSize, 1 ];
    strides    := [ args.Strides, 1 ];
    ndim       := inputs[0].ndim;

    if args.DataFormat = 'channels_last' then
    begin
        pool_shape := [ 1 ] + pool_shape + [ 1 ];
        strides    := [ 1 ] + strides    + [ 1 ];
    end else
    begin
        pool_shape := [ 1, 1 ] + pool_shape;
        strides    := [ 1, 1 ] + strides;
    end;

    var outputs := args.PoolFunction.Apply( inputs.First, pool_shape, strides, args.Padding.ToUpper, conv_utils.convert_data_format(args.DataFormat, ndim));

    Result := TFTensors.Create( tf.squeeze(outputs, pad_axis) );
end;

{ Pooling2D }

constructor Pooling2D.Create(_args: Pooling2DArgs);
begin
   inherited Create(_args) ;

   args := _args;

   args.PoolSize   := conv_utils.normalize_tuple(args.PoolSize, 2, 'pool_size');
   args.Strides    := conv_utils.normalize_tuple(args.Strides,  2, 'strides');
   args.Padding    := conv_utils.normalize_padding(args.Padding);
   args.DataFormat := conv_utils.normalize_data_format(args.DataFormat);
   input_spec      := TInputSpec.Create(DtInvalid, 4);

end;

function Pooling2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var pool_shape : TArray<Integer>;
    var strides : TArray<Integer>;
    if args.DataFormat = 'channels_last' then
    begin
        pool_shape := [ 1, args.PoolSize.dims[0], args.PoolSize.dims[1], 1 ];
        strides    := [ 1, args.Strides.dims[0],  args.Strides.dims[1], 1 ];
    end else
    begin
        pool_shape := [ 1, 1, args.PoolSize.dims[0], args.PoolSize.dims[1] ];
        strides    := [ 1, 1, args.Strides.dims[0],  args.Strides.dims[1]  ];
    end;

    var outputs := args.PoolFunction.Apply( inputs.First, pool_shape, strides, args.Padding.ToUpper, conv_utils.convert_data_format(args.DataFormat, 4));

    Result := TFTensors.Create( outputs );
end;

{ MaxPooling1D }

constructor MaxPooling1D.Create(_args: Pooling1DArgs);
begin
    inherited Create(_args) ;

    args.PoolFunction := MaxPoolFunction.Create;
end;

{ MaxPooling2D }

constructor MaxPooling2D.Create(_args: Pooling2DArgs);
begin
    inherited Create(_args) ;

    args.PoolFunction := MaxPoolFunction.Create;
end;

{ AveragePooling2D }

constructor AveragePooling2D.Create(_args: Pooling2DArgs);
begin
    inherited Create(_args) ;

    args.PoolFunction := AveragePoolFunction.Create;
end;

{ GlobalPooling1D }

constructor GlobalPooling1D.Create(_args: Pooling1DArgs);
begin
   inherited Create(_args) ;

   args := _args;
   args.DataFormat := conv_utils.normalize_data_format(args.DataFormat);
   Finput_spec     := TInputSpec.Create(DtInvalid, 3);
end;

function GlobalPooling1D.getDataFormat: string;
begin
    Result := args.DataFormat
end;

{ GlobalPooling2D }

constructor GlobalPooling2D.Create(_args: Pooling2DArgs);
begin
   inherited Create(_args) ;

   args := _args;
   args.DataFormat := conv_utils.normalize_data_format(args.DataFormat);
   Finput_spec     := TInputSpec.Create(DtInvalid, 4);
end;

function GlobalPooling2D.getDataFormat: string;
begin
   Result := args.DataFormat
end;

{ GlobalMaxPooling1D }

constructor GlobalMaxPooling1D.Create(_args: Pooling1DArgs);
begin
    inherited Create(_args) ;
end;

function GlobalMaxPooling1D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if data_format = 'channels_last' then
    begin
        var a : TAxis := 1;
        var Res := math_ops.reduce_max(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end else
    begin
        var a : TAxis := 2;
        var Res := math_ops.reduce_max(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end;
end;

{ GlobalMaxPooling2D }

constructor GlobalMaxPooling2D.Create(_args: Pooling2DArgs);
begin
     inherited Create(_args) ;
end;

function GlobalMaxPooling2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if data_format = 'channels_last' then
    begin
        var a : TAxis := [1,2];
        var Res := math_ops.reduce_max(inputs.First, @a, false) ;
        Result := TFTensors.Create(Res);
    end else
    begin
        var a : TAxis := [2,3];
        var Res := math_ops.reduce_max(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end;
end;

{ GlobalAveragePooling1D }

constructor GlobalAveragePooling1D.Create(_args: Pooling1DArgs);
begin
     inherited Create(_args) ;
end;

function GlobalAveragePooling1D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if data_format = 'channels_last' then
    begin
        var a : TAxis := 1;
        var Res := math_ops.reduce_mean(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end else
    begin
        var a : TAxis := 2;
        var Res := math_ops.reduce_mean(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end;
end;

{ GlobalAveragePooling2D }

constructor GlobalAveragePooling2D.Create(_args: Pooling2DArgs);
begin
     inherited Create(_args) ;
end;

function GlobalAveragePooling2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if data_format = 'channels_last' then
    begin
        var a : TAxis := [1,2];
        var Res := math_ops.reduce_mean(inputs.First, @a, false) ;
        Result := TFTensors.Create(Res);
    end else
    begin
        var a : TAxis := [2,3];
        var Res := math_ops.reduce_mean(inputs.First, @a, false);
        Result := TFTensors.Create(Res);
    end;
end;
{$ENDREGION}

{$REGION 'PreProcessing'}

{ PreprocessingLayer }

constructor PreprocessingLayer.Create(_args: PreprocessingLayerArgs);
begin
   inherited Create(_args) ;
end;

{ Resizing }

constructor Resizing.Create(_args: ResizingArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;


function Resizing.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
   var a : TArray<Integer> := [ args.Height, args.Width ];
   var Res := image_ops_impl.resize_images_v2(inputs.First, a, args.Interpolation);
   Result  := TFTensors.Create(Res);
end;

function Resizing.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
     Result := TFShape.Create([input_shape.dims[0], args.Height, args.Width, input_shape.dims[3]]);
end;

function Resizing.from_config(config: TJSONValue): Resizing;
var
  LJSON: TJSONValue;
  LReader: TNeonDeserializerJSON;
  rREs   : Resizing;
begin
    rREs := nil;

    LJSON := TJSONObject.ParseJSONValue(config.ToString);
    if not Assigned(LJSON) then
      raise Exception.Create('Error parsing JSON string');

    var AConfig := TNeonConfiguration.Default;
    try
      LReader := TNeonDeserializerJSON.Create(AConfig);
      try
        LReader.JSONToObject(rREs, LJSON);
        //LogError(LReader.Errors);
      finally
        LReader.Free;
      end;
    finally
      LJSON.Free;
    end;
    Result := rREs;
end;

{ CombinerPreprocessingLayer }

constructor CombinerPreprocessingLayer.Create(_args: PreprocessingLayerArgs);
begin
    inherited Create(_args) ;

    args                := _args;
    Fpreviously_updated := false;
end;

procedure CombinerPreprocessingLayer.adapt(data: IDatasetV2; reset_state: Boolean);
begin
    var accumulator : IAccumulator;
    if not reset_state then
        accumulator := Fcombiner.Restore;

    var next_data    := data.make_one_shot_iterator;
    var data_element := next_data.next;
end;

{ IndexLookupAccumulator }

constructor IndexLookupAccumulator.Create;
begin
    CountDict := TDictionary<string,Integer>.Create;
end;

{ IndexLookupCombiner }

constructor IndexLookupCombiner.Create(vocab_size: Integer; mask_value: string);
begin
    Fvocab_size := vocab_size;
    Fmask_value := mask_value;
end;


procedure IndexLookupCombiner.Compute(values: TFTensor; accumulator: IAccumulator);
begin
    if accumulator = nil  then
      accumulator := IndexLookupAccumulator.Create;
end;

procedure IndexLookupCombiner.Deserialize;
begin

end;

procedure IndexLookupCombiner.Extract;
begin

end;

procedure IndexLookupCombiner.Merge;
begin

end;

function IndexLookupCombiner.Restore: IAccumulator;
begin

end;

procedure IndexLookupCombiner.Serialize;
begin

end;

{ IndexLookup }

constructor IndexLookup.Create(max_tokens, num_oov_indices: Integer; mask_token, oov_token, encoding: string; invert: Boolean);
var
  num_mask_tokens, vocab_size : Integer;

begin
    if mask_token = '' then num_mask_tokens := 0
    else                    num_mask_tokens := 1;
    vocab_size := max_tokens - (num_oov_indices + num_mask_tokens);
    Fcombiner := IndexLookupCombiner.Create(vocab_size, mask_token)
end;

procedure IndexLookup.adapt(data: IDatasetV2; reset_state: Boolean);
begin
  if not reset_state then
      raise Exception.Create('IndexLookup does not support streaming adapts.');

  inherited adapt(data,reset_state);
end;

{ StringLookup }

constructor StringLookup.Create(max_tokens, num_oov_indices: Integer; mask_token: string; vocabulary: TArray<string>; oov_token, encoding: string; invert: Boolean);
begin

end;

{ TextVectorization }

constructor TextVectorization.Create(_args: TextVectorizationArgs);
begin
    inherited Create(_args) ;

    args       := _args;
    args.DType := TF_DataType.TF_STRING;
    // string standardize = "lower_and_strip_punctuation",

    var mask_token := '';
    Findex_lookup_layer := StringLookup.Create(args.MaxTokens, 1, mask_token,  args.Vocabulary);
end;

procedure TextVectorization.Build(input_shape: TFShape);
begin
  inherited Build(input_shape);

end;

procedure TextVectorization.adapt(data: IDatasetV2; reset_state: Boolean);
begin
  var shape := data.output_shapes[0];
  if shape.ndim = 1 then
      data := data.map( function(tensor: TFTensors): TFTensors
                         begin
                             var Res := array_ops.expand_dims(tensor.first, -1);
                             Result := TFTensors.Create(Res);
                         end);
  build(data.variant_tensor.shape);
  var preprocessed_inputs := data.map(_preprocess);
  Findex_lookup_layer.adapt(preprocessed_inputs);

end;

function TextVectorization._preprocess(inputs: TFTensors): TFTensors;
var
  input_tensor : TFTensor;
begin
    input_tensor := nil;
    if Assigned(args.Standardize) then
        input_tensor := args.Standardize(inputs.First);
    if not string.IsNullOrEmpty(args.Split) then
    begin
        if inputs.shape.ndim > 1 then
            input_tensor := array_ops.squeeze(inputs.First, [ -1 ]);
        if args.Split = 'whitespace' then
            input_tensor := tf.strings.split(input_tensor).ToTensor;
    end;
    Result := TFTensors.Create( input_tensor );
end;

{ Rescaling }

constructor Rescaling.Create(_args: RescalingArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;

function Rescaling.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    scale   := constant_op.constant(args.Scale, args.DType,'Const');
    offset  := constant_op.constant(args.Offset, args.DType,'Const');
    var Res := TTensor(math_ops.cast(inputs.First, args.DType)) * scale + offset;
    Result := TFTensors.Create(Res) ;
end;

function Rescaling.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
   Result := input_shape;
end;

{$ENDREGION}

{$REGION 'Reshaping'}
{ ZeroPadding2D }

constructor ZeroPadding2D.Create(_args: ZeroPadding2DArgs);
begin
    inherited Create(_args) ;

    data_format  := conv_utils.normalize_data_format(data_format);
    padding      := _args.Padding;
    input_spec   := TInputSpec.Create(DtInvalid, 4);
end;

function ZeroPadding2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var Res := tf.keras.backend.spatial_2d_padding(inputs.First, padding, data_format);
    Result := TFTensors.Create(Res) ;
end;

{ Flatten }

constructor Flatten.Create(_args: FlattenArgs);
begin
    inherited Create(_args) ;

    args             := _args;
    args.DataFormat  := conv_utils.normalize_data_format(args.DataFormat);
    input_spec   := TInputSpec.Create(DtInvalid, 4);
    _channels_first := False;
    if args.DataFormat = 'channels_first' then
       _channels_first := True;
end;

function Flatten.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if _channels_first then
      raise Exception.Create('Not Implemented');

    if tf.executing_eagerly then
    begin
        var Res := array_ops.reshape(inputs.First, TFShape.Create([ inputs.shape[0], -1 ]));
        Result := TFTensors.Create(Res);
        Exit;
    end else
    begin
        var input_shape := inputs.shape;
        var rank        := inputs.shape.ndim;
        if rank = 1 then
        begin
            var Res := array_ops.expand_dims(inputs.First, 1);
            Result := TFTensors.Create(Res);
            Exit;
        end;
        var batch_dim : Integer := tensor_shape.dimension_value(input_shape[0]);
        if batch_dim <> -1 then
        begin
            var Res := array_ops.reshape(inputs.First,TFshape.Create([batch_dim, -1 ]));
            Result := TFTensors.Create(Res);
            Exit;
        end;

        var non_batch_dims : TArray<Integer>;
        var aShapeDims : TArray<Integer> := input_shape;
        for var i := 1 to Length(aShapeDims)- 1 do
             non_batch_dims := non_batch_dims + [ aShapeDims[i] ];

        var num := 1;
        if Length(non_batch_dims) > 0 then
        begin
            for var i := 0 to Length(non_batch_dims) - 1 do
                num := num * non_batch_dims[i];
        end;
        var Res := array_ops.reshape(inputs.First, [ inputs.shape[0], num ]);
        Result := TFTensors.Create(Res);
    end;
end;

{ Permute }

constructor Permute.Create(_args: PermuteArgs);
begin
    inherited Create(_args) ;

    dims := _args.dims;
end;

procedure Permute.Build(input_shape: TFShape);
begin
    var rank := input_shape.rank;
    if length(dims) <> (rank - 1) then
       raise Exception.Create('Dimensions must match.');

    SetLength(permute, input_shape.rank);
    for var i := 1 to Length(dims)  do
       permute[i] := dims[i-1] ;

    Fbuilt := true;
end;

function Permute.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
var
  outputs : TFTensor;
begin
    outputs := inputs.First;
    var aAxis : TAxis := permute;
    var Res := tf.transpose(outputs, aAxis);
    Result := TFTensors.Create(Res);
end;

function Permute.ComputeOutputShape(input_shape: TFShape): TFShape;
var
  output_shape : TFShape;
begin
    output_shape := TFShape(input_shape.dims);
    for var i := 0 to Length(dims) -1 do
    begin
        var d := dims[i];
        var target_dim := input_shape[d];
        output_shape[i + 1] := target_dim;
    end;
    Result := output_shape;
end;

{ Reshape }

constructor Reshape.Create(_args: ReshapeArgs);
begin
    inherited Create(_args) ;

    args := _args;
end;

function Reshape.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var shapes := TList<TFTensor>.Create;
    shapes.Add( array_ops.shape(inputs.First)[0] );
    var dtype := shapes[0].dtype;

    if args.TargetShapeObjects <> nil then
        // shapes.AddRange(args.TargetShapeObjects);
        raise Exception.Create('Not Implemented');

    if not args.TargetShape.isnull then
    begin
        for var i := 0 to Length(args.TargetShape.dims) - 1 do
        begin
            var t : TFTensor := constant_op.constant(args.TargetShape.dims[i], dtype,'Const') ;
            shapes.Add(t);
        end;
    end;

    var shape := Tops.convert_to_tensor(shapes);

    var res := array_ops.reshape(inputs.First, shape);

    if not tf.Context.executing_eagerly then
        res.shape := ComputeOutputShape(inputs.shape);

    Result := TFTensors.Create(res);
end;

function Reshape.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
    var bFound : Boolean := False;
    for var i := 1 to Length(input_shape.dims) - 1 do
    begin
       if input_shape.dims[i] = - 1 then
       begin
           bFound := True;
           Break;
       end;
    end;

    if bFound then
    begin
        raise Exception.Create('Not Implemented');
    end else
    begin
        input_shape      := TFShape(input_shape.dims[0]);
        var output_shape := input_shape.concatenate(args.TargetShape.dims);
        Result := output_shape;
    end;
end;
{ UpSampling2D }

constructor UpSampling2D.Create(_args: UpSampling2DArgs);
begin
    inherited Create(_args) ;

    args             := _args;
    data_format      := conv_utils.normalize_data_format(args.DataFormat);
    size             := conv_utils.normalize_tuple(args.Size, 2, 'size');
    FinputSpec       := TInputSpec.Create(DtInvalid, 4);
end;

function UpSampling2D.GetInterpolation: string;
begin
   Result := args.Interpolation;
end;

function UpSampling2D.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
   var Res := tf.keras.backend.resize_images(inputs.First, size[0], size[1], data_format, interpolation);
   Result := TFTensors.Create(res);
end;

{$ENDREGION}

{$REGION 'Metric'}
{ Metric }

constructor Metric.Create(name: string; dtype: TF_DataType);
var
  lArg : LayerArgs;
begin
  lArg := LayerArgs.Create;
  lArg.Name := name;
  lArg.DType:= dtype;

   inherited Create(lArg) ;

   Fstateful := true;
   Fbuilt    := true;
end;

function Metric.add_weight(name: string; shape: TFShape; dtype: TF_DataType; initializer: IInitializer; regularizer: IRegularizer; synchronization: TVariableSynchronization;
  aggregation: TVariableAggregation; trainable: Boolean; getter: TFunc<VariableArgs, IVariableV1>): IVariableV1;
begin
    Result := TUtils.tf_with<TNameScope,IVariableV1>( TOps.name_scope(name),
                    function(v1: TNameScope): IVariableV1
                      begin
                          Result := inherited add_weight(name, shape, dtype, initializer, nil, synchronization, aggregation, false);
                      end);
end;

procedure Metric.reset_states;
begin
    for var v in weights do
    begin
        if      v is RefVariable          then  (v as RefVariable).assign(Integer(0))
        else if v is BaseResourceVariable then  (v as BaseResourceVariable).assign(Integer(0))
        else
           raise Exception.Create('Metric.reset_states Error!');
    end;
end;

function Metric.R_result: TFTensor;
begin
    raise TFException.Create('Not Implemented');
end;

function Metric.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    raise TFException.Create('Not Implemented');
end;

function Metric.ToString: string;
begin
    var tot : NDArray := total.numpy;
    var c   : NDArray := count.numpy;
    Result := Format('%s %s/%s',[Name,FormatFloat('0.0000', tot), FormatFloat('0.0000', c)]);
end;
{$ENDREGION}

{$REGION 'TensorFlowOpLayer'}
{ TensorFlowOpLayer }

constructor TensorFlowOpLayer.Create(_args: TensorFlowOpLayerArgs);
begin
    TF_OP_LAYER_NAME_PREFIX := 'tf_op_layer_';

    var l := LayerArgs.Create;
    l.Name      := TF_OP_LAYER_NAME_PREFIX + _args.Name;
    l.Trainable := _args.Trainable;
    l.DType     := _args.DType;
    l.Autocast  := false;

    F_function := nil;

    inherited Create(l) ;

    args   := _args;
    Fbuilt := True;
end;

function TensorFlowOpLayer.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    if tf.Context.executing_eagerly then
      Exit( DeFunCall(inputs) );

    Result := MakOp(inputs);
end;

function TensorFlowOpLayer.DeFunCall(inputs: TFTensors): TFTensors;
var
  graph_inputs : TArray<TFTensor>;
begin
   if F_function = nil then
   begin
        F_function := ConcreteFunction.Create(name);
        F_function.Enter;

        var i : Integer := 0;
        for i := 0 to inputs.Count - 1 do
           graph_inputs := graph_inputs + [ tf.placeholder(inputs[i].Dtype, inputs[i].shape, 'defun_inputs_'+i.ToString) ];

        var graph_outputs := MakOp(TFTensors.Create(graph_inputs));
        graph_outputs     := mark_as_return(graph_outputs);

        F_function.ToGraph(TFTensors.Create(graph_inputs), graph_outputs);
        F_function._Exit;
   end;

    var outputs := F_function.FilteredCall(inputs);
    Result := outputs;
end;

function TensorFlowOpLayer.mark_as_return(tensors: TFTensors): TFTensors;
begin
    var res := TFTensors.Create;
    for var tensor in tensors do
        res.Add(array_ops.identity(tensor));
    Result := res;
end;

function TensorFlowOpLayer.MakOp(inputs: TFTensors): TFTensors;
begin
    var graph := inputs.graph;
    graph.as_default;
    for var it in constants do
    begin
        var constant := it.Value;
        var g := node_def.Inputs[it.Key];
        var value    := constant_op.constant(constant, DtInvalid, node_def.Inputs[it.Key]);
        inputs.Insert(it.Key, value);
    end;

    var t_op := Tops._create_c_op(graph, node_def, inputs.ToArray, [], nil);
    var op  := graph._create_op_from_tf_operation(t_op.value1);
    op._control_flow_post_processing;

    // Record the gradient because custom-made ops don't go through the
    // code-gen'd eager call path
    var op_type := op.NodeDef.Op;

    tf.Runner.RecordGradient(op_type, op.inputs.inputs, nil, op.outputs);

    graph.gExit;
    Result := TFTensors.create(op.outputs);
end;

function TensorFlowOpLayer.GetConsts: TDictionary<Integer, TNDArray>;
begin
    Result := args.Constants;
end;

function TensorFlowOpLayer.GetNodeDef: TNodeDef;
begin
    Result := args.NodeDef;
end;

function TensorFlowOpLayer.GetOpLayer(_args: TensorFlowOpLayerArgs): Layer;
begin
   Result := TensorFlowOpLayer.Create(_args)
end;

{$ENDREGION}
{ CategoryEncoding }

constructor CategoryEncoding.Create(_args: CategoryEncodingArgs);
begin
    inherited Create(_args);
    args := _args;
end;

function CategoryEncoding.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var depth := args.NumTokens;
    var max_value := tf.reduce_max(inputs.First);
    var min_value := tf.reduce_min(inputs.First);
    (*var condition = tf.logical_and(tf.greater(tf.cast(constant_op.constant(depth), max_value.dtype), max_value),
        tf.greater_equal(min_value, tf.cast(constant_op.constant(0), min_value.dtype)));*)
    var bincounts := encode_categorical_inputs(inputs.First, args.OutputMode, depth, args.DType, args.Sparse, args.CountWeights);
    if args.OutputMode <> 'tf_idf' then
    begin
        Result := bincounts;
        Exit;
    end;
    Result := inputs;
end;

function CategoryEncoding.ComputeOutputShape(input_shape: TFShape): TFShape;
begin
     Result := input_shape;
end;

function CategoryEncoding.encode_categorical_inputs(inputs: TFTensor; output_mode: string; depth: Integer; dtype: TF_DataType; sparse: Boolean;
  count_weights: TFTensor): TFTensors;
begin
    var binary_output : Boolean := false;
    if output_mode = 'one_hot' then
    begin
        binary_output := true;
        if inputs.shape[-1] <> 1 then
        begin
            inputs := tf.expand_dims(inputs, -1);
        end;
    end
    else if output_mode = 'multi_hot' then
    begin
        binary_output := true;
    end;
    var depth_tensor := constant_op.constant(depth);
    var s : TFShape := -1;
    var Res := tf.math.bincount(inputs, count_weights, depth_tensor, depth_tensor, dtype, '',  @s, binary_output);
    Result := TFTensors.Create(Res);
end;

end.
