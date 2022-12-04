unit Keras.Layer;
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
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
       uses System.SysUtils,
            System.Generics.Collections,

            Spring,
            Spring.Collections.Enumerable,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable,
            TensorFlow.Training,
            TensorFlow.Initializer,

            Keras.Regularizers,
            Keras.ArgsDefinition,
            Keras.Engine,

            ProtoGen.nodeDef,
            ProtoGen.variable;

type

  IPoolFunction = interface
    ['{17352CCA-5C47-49AD-9265-C9762F7F3FB6}']

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
  end;

  threadvar  tls_CallContext : TCallContext;

type
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
       function GetDtype: TF_DataType;
       function GetTrainW: TList<IVariableV1>;
       function GetTrainVars: TList<IVariableV1>;
       function GetNotTrainW: TList<IVariableV1>;
       function GetName: string;
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
       function  GetW: TList<IVariableV1>;
       procedure SetW(const Value: TList<IVariableV1>);
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
       FinputSpec              : InputSpec;
       Ftrainable_weights      : TList<IVariableV1>;
       Fnon_trainable_weights  : TList<IVariableV1>;
       FName                   : string;
       FBase_Name              : string;
       FcomputePreviousMask    : Boolean;
       Fupdates                : TList<TFOperation>;
       FInboundNodes           : TList<INode>;
       FOutboundNodes          : TList<INode>;
       Fself_tracked_trackables: TList<ILayer>;
       FLayers                 : TList<ILayer>;
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
       procedure build(inputs: TFTensors); virtual;
       procedure add_loss(losses: TFunc<TFTensor>); virtual;
       procedure _init_set_name(_name: string; zero_based: Boolean = true); virtual;
    public
        constructor Create(_args: LayerArgs);

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
        property trainable_weights    : TList<IVariableV1>  read GetTrainW;
        property trainable_variables  : TList<IVariableV1>  read GetTrainVars;
        property non_trainable_weights: TList<IVariableV1>  read GetNotTrainW;
        property Id                   : Integer             read FId;
        property Name                 : string              read GetName;
        property BatchInputShape      : TFShape             read GetBatchShape;
        property InboundNodes         : TList<INode>        read GetInNodes;
        property OutboundNodes        : TList<INode>        read GetOutNodes;
        property CallContext          : TCallContext        read GetCallCtx write SetCallCtx;
        property input                : TArray<TFTensor>    read GetInput;
        property NodesByDepth         : TDictionary<Integer, TList<INode>> read FNodesByDepth write FNodesByDepth;
        property output_shape         : TFShape             read GetoutShape;
        property Layers               : TList<ILayer>       read GetLayers;
        property weights              : TList<IVariableV1>  read GetW write SetW;

  end;

  {$REGION 'Activation'}
  /// <summary>
  /// ELU Layer:
  /// x = 0 when x > 0, x = alpha( e^x-1 ) elsewhere
  /// </summary>
  ELU = class(Layer)
    protected
      procedure Build(inputs: TFTensors); override;
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      function  ComputeOutputShape(input_shape: TFShape): TFShape; override;
    public
      args  : ELUArgs;
      alpha : Single;

      constructor Create(_args: ELUArgs);
  end;

  Exponential = class(Layer)
    protected
      procedure Build(inputs: TFTensors); override;
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
    protected
      function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
    public
      args  : LeakyReLuArgs;
      alpha : Single;
      constructor Create(_args: LeakyReLuArgs);
  end;

  /// <summary>
  /// SELU Layer:
  /// similar to ELU, but has pre-defined alpha and scale
  /// </summary>
  SELU = class(Layer)
    protected
      const alpha : Single = 1.67326324;
      const scale : Single = 1.05070098;

      procedure Build(inputs: TFTensors); override;
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
      function  Call(inputs: TFTensors; mask:TFTensors = nil; training : pBoolean= nil; return_attention_scores : Boolean= false): TFTensors; overload ;
    public
      args : BaseDenseAttentionArgs;

      constructor Create(_args: BaseDenseAttentionArgs);
      /// <summary>
      /// Calculates attention scores.
      /// </summary>
      /// <param name="query">query: Query tensor of shape `[batch_size, Tq, dim]`.</param>
      /// <param name="key">key: Key tensor of shape `[batch_size, Tv, dim]`.</param>
      /// <returns>Tensor of shape `[batch_size, Tq, Tv]`.</returns>
      function _calculate_scores(query: TFTensor; key: TFTensor): TFTensor; virtual;
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

  end;
  {$ENDREGION}

  TensorFlowOpLayer = class(Layer)
    private

    protected

    public
      args     : TensorFlowOpLayerArgs;
      constants: TDictionary<Integer, TNDArray>;
      node_def : TNodeDef;
      OpType   : string;

      TF_OP_LAYER_NAME_PREFIX : string;

      constructor Create(_args: TensorFlowOpLayerArgs) ;
  end;

implementation
        uses
             Keras.Backend,

             Tensorflow,
             TensorFlow.Tensor,
             TensorFlow.Ops,
             Tensorflow.Utils,
             Tensorflow.NameScope,
             TensorFlow.EagerTensor,
             TensorFlow.Framework,

             Keras.Utils;

{ Layer }

constructor Layer.Create(_args: LayerArgs);
begin
    Fargs := _args;
    // A stateful layer is a layer whose updates are run during inference too,
    // for instance stateful RNNs.
    Fstateful := false;
    // Indicates whether `build` needs to be called upon layer call, to create
    // the layer's weights.
    Fbuilt := false;
    FSupportsMasking := false;

    Fid := Tops.uid_layer;
    _init_set_name(args.Name);
    Ftrainable_weights      := TList<IVariableV1>.Create;
    Fnon_trainable_weights  := TList<IVariableV1>.Create;
    FcomputePreviousMask    := false;
    Fupdates                := TList<TFOperation>.Create;
    Fself_tracked_trackables:= TList<ILayer>.Create;

    FinboundNodes  := TList<INode>.Create;
    FoutboundNodes := TList<INode>.Create;

    // Manage input shape information if passed.
    if (Fargs.BatchInputShape.isNull) and (not args.InputShape.isNull) then
    begin
        var aShape : TArray<Int64> := [args.BatchSize] + Fargs.InputShape.dims;
        args.BatchInputShape :=  aShape;
    end;

    FLayers := TList<ILayer>.Create;
end;

procedure Layer.add_loss(losses: TFunc<TFTensor>);
begin

end;

procedure Layer.build(inputs: TFTensors);
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

function Layer.GetNotTrainW: TList<IVariableV1>;
begin
    Result := Fnon_trainable_weights;
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
    Result := FLayers;
end;

function Layer.GetOutNodes: TList<INode>;
begin
    Result := FOutboundNodes;
end;

function Layer.GetoutShape: TFShape;
begin
    Result := FInboundNodes[0].Outputs.shape;
end;

function Layer.GetTrainable: Boolean;
begin
   Result := Fargs.Trainable;
end;

function Layer.GetTrainVars: TList<IVariableV1>;
begin
    Result := Ftrainable_weights;
end;

function Layer.GetTrainW: TList<IVariableV1>;
begin
   Result := Ftrainable_weights;
end;

function Layer.GetW: TList<IVariableV1>;
begin
    var weights := TList<IVariableV1>.Create;
    weights.AddRange(Ftrainable_weights);
    weights.AddRange(Fnon_trainable_weights);
    Result := weights;
end;

function Layer.get_config: LayerArgs;
begin
    Result := Fargs
end;

procedure Layer.load_weights(filepath: string);
begin

end;

procedure Layer.SetW(const Value: TList<IVariableV1>);
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

procedure Layer.StackLayers(_layers: TArray<ILayer>);
begin
    Flayers.AddRange(_layers);
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

    build(inputs);

    if need_restore_mode then
        tf.Context.restore_mode;

    Fbuilt := true;
end;

function Layer.FunctionalConstructionCall(inputs: TFTensors): TFTensors;
begin
    var mask_arg_passed_by_framework     : Boolean := false;
    var training_arg_passed_by_framework : Boolean := false;
    var training_value : TFTensor := nil;
    if training_value = nil then
       training_arg_passed_by_framework := true;


    if base_layer_utils.needs_keras_history(inputs) then
        base_layer_utils.create_keras_history(inputs);

    var outputs : TFTensors := nil;
    var ctxManager  := CallContext.enter(true);

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
    var ctxManager := CallContext.enter(false);

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
        Fbase_name := generic_utils.to_snake_case( Ptypeinfo(TypeInfo(Layer))^.Name);
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
    var deque           := TQueue<ILayer>.Create(Flayers);
    while not deque.Count <= 0 do
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

function Layer._get_trainable_state: TDictionary<ILayer, boolean>;
begin
    Ftrainable_state := TDictionary<ILayer, Boolean>.Create;
    for var llayer in _flatten_layers do
        Ftrainable_state.AddOrSetValue(llayer, llayer.Trainable);
    Result := Ftrainable_state;
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
    if trainable = true then Ftrainable_weights.Add(variable)
    else                     Fnon_trainable_weights.Add(variable);

    Result := variable;
end;

procedure Layer.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure Layer.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
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

{ TensorFlowOpLayer }

constructor TensorFlowOpLayer.Create(_args: TensorFlowOpLayerArgs);
begin
    var l := LayerArgs.Create;
    l.Name      := TF_OP_LAYER_NAME_PREFIX + args.Name;
    l.Trainable := args.Trainable;
    l.DType     := args.DType;
    l.Autocast  := false;

    inherited Create(l) ;

    args   := _args;
    Fbuilt := True;
end;

{$REGION 'Activation'}
{ ELU }

constructor ELU.Create(_args: ELUArgs);
begin
    inherited Create(_args);
    args := _args;
end;

procedure ELU.Build(inputs: TFTensors);
begin
  if alpha < 0  then
    raise TFException.Create('Alpha must be a number greater than 0.');

  Fbuilt := true;
end;

function ELU.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.first;
    output := tf.where(TTensor(output) > 0, output,
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

procedure Exponential.Build(inputs: TFTensors);
begin
   Fbuilt := true;
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

procedure SELU.Build(inputs: TFTensors);
begin
  if alpha < 0  then
    raise TFException.Create('Alpha must be a number greater than 0.');

  Fbuilt := true;
end;

function SELU.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    var output : TFTensor := inputs.first;

    var res := tf.where(TTensor(output) > 0,
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
    //training := @false; // TODO: Delete this line when backend.learning_phase is available
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

    weights := Tensorflow.Framework.smart_module.smart_cond(_training, dropped_weights, false_pred);
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
{$ENDREGION}

end.
