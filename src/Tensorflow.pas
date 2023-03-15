unit Tensorflow;
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

interface
  uses System.SysUtils, System.Rtti,  System.TypInfo, System.Classes,

       System.Generics.Collections,
       Spring,

       TF4D.Core.CApi,
       TensorFlow.DApiBase,
       TensorFlow.DApi,
       TensorFlow.Context,
       TensorFlow.EagareRunner,
       TensorFlow.DApiEager,
       Tensorflow.Utils,
       TensorFlow.OpDefLibrary,
       Tensorflow.Gradient,
       Tensorflow.String_ops,
       TensorFlow.Variable,
       TensorFlow.Tensors.Ragged,
       TensorFlow.Initializer,
       TensorFlow.Activation,
       Keras.KerasApi,
       TensorFlow.Training,
       TensorFlow.bitwise_ops,
       TensorFlow.linalg_ops,
       Tensorflow.NameScope,
       TensorFlow.Framework,
       Numpy,
       Numpy.Axis,

       Keras.Data,

       ProtoGen.Tensor,
       Protogen.tensorShape,
       ProtoGen.attrValue,
       ProtoGen.types,
       ProtoGen.opDef,
       protogen.config,
       ProtoGen.variable;



const
  C_GRAPH_MODE : Integer = 0;
  C_EAGER_MODE : Integer = 1;

type
{$REGION 'CompatV1Api'}
  CompatV1Api = class
     private

     public
       constructor Create;
       destructor  Destroy; override;
       procedure   disable_eager_execution;
       function    Session: TFSession;
       function    global_variables_initializer: TFOperation;
       function    get_variable(name            : string;
                                shape           : PTFShape= nil;
                                dtype           : TF_DataType = TF_DataType.DtInvalid;
                                initializer     : TObject = nil;{IInitializer or Tensor}
                                trainable       : PBoolean= nil;
                                collections     : TList<string> = nil;
                                use_resource    : PBoolean= nil;
                                validate_shape  : Boolean = true;
                                synchronization : TVariableSynchronization = TVariableSynchronization.VARIABLE_SYNCHRONIZATION_AUTO;
                                aggregation     : TVariableAggregation     = TVariableAggregation.VARIABLE_AGGREGATION_NONE):IVariableV1;
  end;
{$ENDREGION}

{$REGION 'CompatApi'}
  CompatApi = class
     public
       v1 : CompatV1Api;

       constructor Create;
       destructor  Destroy; override;
  end;
{$ENDREGION}

{$REGION 'StringsApi'}
  StringsApi = class
    private

    public
      ops : string_ops;

      constructor Create;
      destructor Destroy; override;
      /// <summary>
      /// Converts all uppercase characters into their respective lowercase replacements.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="encoding"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function lower(input: TFTensor; encoding: string = ''; name : string = ''): TFTensor;
      /// <summary>
      ///
      /// </summary>
      /// <param name="input"></param>
      /// <param name="pattern"></param>
      /// <param name="rewrite"></param>
      /// <param name="replace_global"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function regex_replace(input: TFTensor; pattern: string; rewrite: string; replace_global: Boolean = true; name : string = ''): TFTensor;
      /// <summary>
      /// Return substrings from `Tensor` of strings.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="pos"></param>
      /// <param name="len"></param>
      /// <param name="name"></param>
      /// <param name="uint"></param>
      /// <returns></returns>
      function substr(input: TFTensor; pos: Integer; len: Integer; name: string  = ''; &uint: string = 'BYTE'): TFTensor; overload;
      function substr(input: String; pos: Integer; len: Integer; name: string = ''; &uint: string = 'BYTE'): TFTensor;overload;
      /// <summary>
      /// String lengths of `input`.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="name"></param>
      /// <param name="unit"></param>
      /// <returns></returns>
      function string_length(input: TFTensor; name: string = ''; &unit : string= 'BYTE') : TFTensor ;
      function format(template: string; inputs: TArray<TFTEnsor>; placeholder: string = '{}'; summarize: Integer = 3; name: string = '') : TFTensor;
      function split(input: TFTensor; sep : AnsiChar = ' '; maxsplit : Integer= -1; name: string = ''): RaggedTensor;
      function unicode_decode_with_offsets(input: TFTensor; input_encoding: string; errors: string = 'replace'; replacement_char: Integer = $FFFD; replace_control_characters: Boolean = false; name: string = ''): Tuple<RaggedTensor, RaggedTensor>;

  end;
{$ENDREGION}

{$REGION 'GraphKeys'}
  /// <summary>
  /// Standard names to use for graph collections.
  /// The standard library uses various well-known names to collect and
  /// retrieve values associated with a graph. For example, the
  /// `tf.Optimizer` subclasses default to optimizing the variables
  /// collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  /// specified, but it is also possible to pass an explicit list of
  /// variables.
  /// </summary>
  TGraphKeys = record
    public
    const
      /// <summary>
      /// Key to collect concatenated sharded variables.
      /// </summary>
      CONCATENATED_VARIABLES_      :  string  = 'concatenated_variables';
      /// <summary>
      /// the subset of `Variable` objects that will be trained by an optimizer.
      /// </summary>
      TRAINABLE_VARIABLES_         :  string  = 'trainable_variables';
      /// <summary>
      /// Trainable resource-style variables.
      /// </summary>
      TRAINABLE_RESOURCE_VARIABLES_:  string  = 'trainable_resource_variables';
      /// <summary>
      /// Key for streaming model ports.
      /// </summary>
      _STREAMING_MODEL_PORTS_      :  string  = 'streaming_model_ports';
      /// <summary>
      /// Key to collect losses
      /// </summary>
      LOSSES_                      :  string  = 'losses';
      LOCAL_VARIABLES_             :  string  = 'local_variables';
      METRIC_VARIABLES_            :  string  = 'metric_variables';
      MODEL_VARIABLES_             :  string  = 'model_variables';
      MOVING_AVERAGE_VARIABLES_    :  string  = 'moving_average_variables';
      /// <summary>
      /// Key to collect Variable objects that are global (shared across machines).
      /// Default collection for all variables, except local ones.
      /// </summary>
      GLOBAL_VARIABLES_ :  string  = 'variables';
      TRAIN_OP_         :  string  = 'train_op';
      GLOBAL_STEP_      :  string  = 'global_step';
      /// <summary>
      /// List of all collections that keep track of variables.
      /// </summary>
      _VARIABLE_COLLECTIONS_ : Array[0..7] of string = (
          'variables',
          'local_variables',
          'metric_variables',
          'model_variables',
          'trainable_variables',
          'moving_average_variables',
          'concatenated_variables',
          'trainable_resource_variables');
      /// <summary>
      /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
      /// </summary>
      SAVEABLE_OBJECTS_     :  string = 'saveable_objects';
      /// <summary>
      /// Key to collect update_ops
      /// </summary>
      UPDATE_OPS_           :  string = 'update_ops';
      // Key to collect summaries.
      SUMMARIES_            :  string = 'summaries';
      // Used to store v2 summary names.
      _SUMMARY_COLLECTION_  :  string = '_SUMMARY_V2';
      // Key for control flow context.
      COND_CONTEXT_         :  string = 'cond_context';
      WHILE_CONTEXT_        :  string = 'while_context';
    class var
      CONCATENATED_VARIABLES : String;
      /// <summary>
      /// the subset of `Variable` objects that will be trained by an optimizer.
      /// </summary>
      TRAINABLE_VARIABLES : String;
      /// <summary>
      /// Trainable resource-style variables.
      /// </summary>
      TRAINABLE_RESOURCE_VARIABLES : String;
      /// <summary>
      /// Key for streaming model ports.
      /// </summary>
      _STREAMING_MODEL_PORTS : String;
      /// <summary>
      /// Key to collect local variables that are local to the machine and are not
      /// saved/restored.
      /// </summary>
      LOCAL_VARIABLES : String;
      /// <summary>
      /// Key to collect losses
      /// </summary>
      LOSSES : String;
      METRIC_VARIABLES : String;
      MOVING_AVERAGE_VARIABLES : string;
      /// <summary>
      /// Key to collect Variable objects that are global (shared across machines).
      /// Default collection for all variables, except local ones.
      /// </summary>
      GLOBAL_VARIABLES : String;
      TRAIN_OP : String;
      GLOBAL_STEP : String;
      GLOBAL_STEP_READ_KEY : string;
      _VARIABLE_COLLECTIONS : TArray<string>;
      /// <summary>
      /// Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
      /// </summary>
      SAVEABLE_OBJECTS : String;
      /// <summary>
      /// Key to collect update_ops
      /// </summary>
      UPDATE_OPS : String;
      // Key to collect summaries.
      SUMMARIES : String;
      // Used to store v2 summary names.
      _SUMMARY_COLLECTION : String;
      // Key for control flow context.
      COND_CONTEXT : String;
      WHILE_CONTEXT : String;
      class function Create: TGraphKeys; static;
  end;
{$ENDREGION}

{$REGION 'TRandom'}
  TRandom  = class
    private

    public
      function stateless_normal(shape: TFShape; mean : Single= 0.0; stddev : Single= 1.0; dtype : TF_DataType = TF_FLOAT; name: string = ''): TFTensor;
      /// <summary>
      /// Outputs random values from a normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="mean"></param>
      /// <param name="stddev"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Function normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed : pInteger = nil; name: string = ''): TFTensor;
      /// <summary>
      /// Outputs random values from a truncated normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="mean"></param>
      /// <param name="stddev"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Function truncated_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = ''): TFTensor;
      Function categorical(logits: TFTensor; num_samples: Integer; seed: pInteger = nil; name: string = ''; output_dtype: TF_DataType = DtInvalid): TFTensor;
      Function uniform(shape: TFShape; minval: Single = 0; maxval: Single = 1; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTensor;
      Function random_uniform(shape: TFShape; minval: Single = 0; maxval: SIngle = 1; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTensor;
      /// <summary>
      /// Randomly shuffles a tensor along its first dimension.
      /// </summary>
      /// <param name="value"></param>
      /// <param name="seed"></param>
      /// <param name="name"></param>
      /// <returns>
      /// A tensor of same shape and type as value, shuffled along its
      /// first dimension.
      /// </returns>
      Function  random_shuffle(value: TFTensor; seed: Integer = 0; name: string = '') : TFTensor;
      procedure set_random_seed(seed: Integer);
      Function  multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger = nil; name: string = ''; output_dtype: TF_DataType = DtInvalid): TFTensor;
  end;
{$ENDREGION}

{$REGION 'MathApi'}
MathApi = class
  private

  public
    function argmax(input: TFTensor; axis: TAxis ; name: string = ''; dimension: PInteger = nil; output_type: TF_DataType = TF_INT64): TFTensor;
    function log(x: TFTensor; name: string = ''): TFTensor;
    /// <summary>
    /// Computes the Gauss error function of `x` element-wise.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    function erf(x: TFTensor; name: string = ''): TFTensor;
    function sum(x: TFTensor;      axis: TAxis; name: string = ''): TFTensor;
    /// <summary>
    ///
    /// </summary>
    /// <param name="arr"></param>
    /// <param name="weights"></param>
    /// <param name="minlength"></param>
    /// <param name="maxlength"></param>
    /// <param name="dtype"></param>
    /// <param name="name"></param>
    /// <param name="axis"></param>
    /// <param name="binary_output"></param>
    /// <returns></returns>
    function bincount(arr: TFTensor; weights: TFTensor = nil; minlength: TFTensor = nil; maxlength: TFTensor = nil; dtype: TF_DataType = TF_INT32;  name: string = ''; axis: PTFShape = nil; binary_output: Boolean = false): TFTensor;
    function in_top_k(predictions: TFTensor; targets: TFTensor; k: Integer; name: string = 'InTopK'): TFTensor;
end;
{$ENDREGION}

{$REGION 'nn_internal'}
nn_internal = class
  private

  public
    function relu: IActivation;overload;
    function swish: IActivation;
    function tanh: IActivation; overload;
    function softmax: IActivation; overload;
    function tanh(x:TFTensor; name: string = '') : TFTensor ;  overload;
    function relu(features:TFTensor; name: string = '') : TFTensor ; overload;
    /// <summary>
    /// Computes sigmoid of `x` element-wise.
    /// Specifically, `y = 1 / (1 + exp(-x))`.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="x"></param>
    /// <param name="name">A name for the operation (optional).</param>
    /// <returns>A Tensor with the same type as `x`.</returns>
    function sigmoid<T>(x: T; name: string = ''): TFTensor ;
    function softmax(logits: TFTensor; axis: Integer = -1; name: string = ''): TFTensor ; overload;
    function sigmoid_cross_entropy_with_logits(labels: TFTensor; logits: TFTensor; name: string = '') : TFTensor;
    /// <summary>
    /// Computes dropout.
    /// </summary>
    /// <param name="x">A floating point tensor.</param>
    /// <param name="keep_prob">(deprecated) A deprecated alias for `(1-rate)`.</param>
    /// <param name="noise_shape"></param>
    /// <param name="seed">Used to create random seeds.</param>
    /// <param name="name"></param>
    /// <param name="rate">A scalar `Tensor` with the same type as `x`.</param>
    /// <returns>A Tensor of the same shape of `x`.</returns>
    function dropout(x: TFTensor; keep_prob: TFTensor = nil; noise_shape: TFTensor = nil; seed: PInteger = nil; name: string = ''; rate: PSingle = nil): TFTensor ;
    function l2_loss(t: TFTensor; name: string = ''): TFTensor ;
    function softmax_cross_entropy_with_logits_v2(labels: TFTensor; logits: TFTensor; axis: Integer = -1; name : string= ''): TFTensor ;
    function leaky_relu(features: TFTensor; alpha: Single = 0.2; name: string = ''): TFTensor ;
    function bias_add(value: TFTensor; bias: IVariableV1; data_format: string = ''; name: string = ''): TFTensor ;
    /// <summary>
    /// Computes sparse softmax cross entropy between `logits` and `labels`.
    /// </summary>
    /// <param name="labels"></param>
    /// <param name="logits"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    function sparse_softmax_cross_entropy_with_logits(labels : TFTensor= nil; logits : TFTensor= nil; name : string= ''): TFTensor;
    function fused_batch_norm(x,
                              scale,
                              offset     : TFTensor;
                              mean       : TFTensor = nil;
                              variance   : TFTensor = nil;
                              epsilon    : Single = 0.001;
                              data_format: string = 'NHWC';
                              is_training: Boolean= true;
                              name       : string = '';
                              exponential_avg_factor : Single= 1.0): TArray<TFTensor>;
end;
{$ENDREGION}

{$REGION 'train_internal'}
train_internal = class
   private

   public
     function GradientDescentOptimizer(learning_rate: Single): Optimizer; overload;
     function GradientDescentOptimizer(learning_rate: TFTensor): Optimizer; overload;
     function AdamOptimizer(learning_rate: Single; epsilon: Single = 1e-8; name: string = 'Adam'): Optimizer; overload;
     function AdamOptimizer(learning_rate: Single; dtype: TF_DataType ;    name: string = 'Adam'): Optimizer; overload;
     function AdamOptimizer(learning_rate: IVariableV1;                    name: string = 'Adam'): Optimizer; overload;
     function AdamOptimizer(learning_rate: TFTensor;                       name: string = 'Adam'): Optimizer; overload;
end;
{$ENDREGION}

{$REGION 'LinalgApi'}
  LinalgApi = class
     public
       ops : linalg_ops;

       constructor Create;
       destructor  Destroy; override;

       function einsum(equation: string; inputs: TFTensors; name: string = ''): TFTensor;
       function eye(num_rows: Integer; num_columns: Integer = -1; batch_shape : PTFShape= nil; dtype: TF_DataType = TF_DOUBLE; name: string = ''): TFTensor;
       function diag(diagonal: TFTensor; name: string = ''): TFTensor;
       function norm(tensor: TFTensor; _ord: string = 'euclidean'; axis: PAxis = nil; name: string = ''; keepdims: Boolean = true): TFTensor;
       function inv(input: TFTensor; adjoint: Boolean = false; name : string= ''): TFTensor;
       function global_norm(t_list: TArray<TFTensor>; name : string= ''): TFTensor;
       function lstsq(matrix: TFTensor; rhs: TFTensor; l2_regularizer : TNDArray= nil; fast: Boolean = true; name: string = '') : TFTensor;
       function tensordot(x: TFTensor; y: TFTensor; axes: TNDArray; name : string= '') : TFTensor;
       function matmul(a: TFTensor; b: TFTensor) : TFTensor;overload;
       function qr(input: TFTensor; full_matrices: Boolean = true; name: string = ''): TFTensors;
       function tensor_diag_part(input: TFTensor; name: string = ''): TFTensor;

  end;
{$ENDREGION}

{$REGION 'image_internal'}
  image_internal = class
     public
       function resize_images_v2(images: TFTensor; size: TFShape;  method: string = 'bilinear'; preserve_aspect_ratio : Boolean= false; antialias : Boolean= false; name: string = ''): TFTensor; overload;
       function resize_images_v2(images: TFTensor; size: TFTensor; method: string = 'bilinear'; preserve_aspect_ratio : Boolean= false; antialias : Boolean= false; name: string = ''): TFTensor; overload;
end;
{$ENDREGION}

{$REGION 'DataOps'}
  DataOps = class
     private
       FDataset : DatasetManager;
     public
       const AUTOTUNE             : Integer = -1;
       const INFINITE_CARDINALITY : Integer = -1;
       const UNKNOWN_CARDINALITY  : Integer = -2;

       constructor Create;
       destructor Destroy; override;

       property Dataset : DatasetManager read FDataset;
  end;
{$ENDREGION}

{$REGION 'TTensorflow'}
  TTensorflow = class(TFDisposable)
    private
      FtapeSet : TGradientTape;
      FMemoLog : TStringList;

      function  GetVersion: string;
    protected
		  procedure NativeDispose(hnd: Pointer); override;

    public
     const
      byte8_t   = TF_DataType.TF_UINT8;
      int8_t    = TF_DataType.TF_INT8;
      int16_t   = TF_DataType.TF_INT16;
      int32_t   = TF_DataType.TF_INT32;
      int64_t   = TF_DataType.TF_INT64;
      float16_t = TF_DataType.TF_HALF;
      float32_t = TF_DataType.TF_FLOAT;
      float64_t = TF_DataType.TF_DOUBLE;
      bool_t    = TF_DataType.TF_BOOL;
      chars_t   = TF_DataType.TF_STRING;
      string_t  = TF_DataType.TF_STRING;
     var
      Status   : TFStatus;
      Context  : TContext;
      OpDefLib : OpDefLibrary;
      Runner   : TEagerRunner;
      keras    : KerasInterface;
      compat   : CompatApi;
      strings  : StringsApi;
      GraphKeys: TGraphKeys;
      data     : DataOps;
      //
      random   : TRandom;
      /// <summary>
      /// NumPy API on TensorFlow
      /// https://www.tensorflow.org/api_docs/python/tf/experimental/numpy
      /// </summary>
      numpy  : NumPyImpl;
      math   : MathApi;
      nn     : nn_internal;
      bitwise: bitwise_ops;
      train  : train_internal;
      linalg : LinalgApi;
      image  : image_internal;
      // Inizializer
      glorot_uniform_initializer : IInitializer;
      zeros_initializer          : IInitializer;
      ones_initializer           : IInitializer;
      random_uniform_initializer : IInitializer;
      orthogonal_initializer     : IInitializer;

      procedure LogMsg(Msg: string);
      function  constant_initializer<T>(value: T; dtype: TF_DataType = TF_FLOAT; verify_shape : Boolean= false): IInitializer;
      function  variable_scope(name: string;         default_name : string = ''; values: TArray<TFTensor> = []; reuse: PBoolean = nil; auxiliary_name_scope : Boolean= true): variable_scope; overload;
      function  variable_scope(scope: VariableScope; default_name : string = ''; values: TArray<TFTensor> = []; reuse: PBoolean = nil; auxiliary_name_scope : Boolean= true): variable_scope; overload;

      constructor Create;
      destructor  Destroy ; override;
      procedure   enable_eager_execution;
      function    executing_eagerly:Boolean;
      function    get_default_graph: TFgraph;
      procedure   reset_default_graph;
      function    peak_default_graph: TFgraph;
      /// <summary>
      /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
      /// When eager execution is enabled, code inside an init_scope block runs with
      /// eager execution enabled even when tracing a `tf.function`.
      /// </summary>
      function  init_scope: TNameScope;
      /// <summary>
      ///     Creates a new graph.
      /// </summary>
      ///<remarks>Has no interaction with graph defaulting. Equivalent to new Graph();</remarks>
      function Graph: TFGraph;
      function placeholder(dtype: TF_DataType; shape: TFShape ; name: string = ''): TFTensor; overload;
      function placeholder(dtype: TF_DataType): TFTensor; overload;
      /// <summary>
      /// A placeholder op that passes through `input` when its output is not fed.
      /// </summary>
      /// <typeparam name="T"></typeparam>
      /// <param name="input">A `Tensor`. The default value to produce when output is not fed.</param>
      /// <param name="shape">
      /// A `tf.Shape` or list of `int`s. The (possibly partial) shape of
      /// the tensor.
      /// </param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>A `Tensor`. Has the same type as `input`.</returns>
      function placeholder_with_default<T>(input :T ; shape: TArray<Integer>; name : string= ''): TFTensor;
      function Session(graph: TFGraph; config: TConfigProto = nil): TFSession;overload;
      function Session: TFSession;overload;
      function get_default_session: TFSession;
      function Variable<T>(data           : T;
                           trainable      : Boolean= true;
                           validate_shape : Boolean = true;
                           use_resource   : Boolean = true;
                           name           : string= '';
                           dtype          : TF_DataType = TF_DataType.DtInvalid;
                           aggregation    : TVariableAggregation = TVariableAggregation.VARIABLE_AGGREGATION_NONE;
                           shape          : PTFShape= nil):ResourceVariable; overload;
      function Variable<T>(data: T;  name : string; dtype: TF_DataType = TF_DataType.DtInvalid):ResourceVariable;  overload;
      // tf.tensor
      function convert_to_tensor(value: TValue; dtype: TF_DataType= DtInvalid; name: string= ''; preferred_dtype: TF_DataType=DtInvalid): TFTensor;

      {$REGION 'tf.linalg'}
      // tf.linalg
      //
      function diag(diagonal: TFTensor; name: string = ''): TFTensor;
      function matmul(a: TFTensor; b: TFTensor; transpose_a : Boolean= false; transpose_b : Boolean= false) : TFTensor; overload;
      /// <summary>
      /// Multiply slices of the two matrices "x" and "y".
      /// </summary>
      /// <remarks>
      /// The `BatchMatMul` operation is embedded into the
      /// `MatMul` operation on the DLL side. However the expected
      /// attributes are not the same, hence we need to expose this
      /// method to have the right args list on the `_apply_op_helper`
      /// function.
      ///
      /// For each rank > 2 the first rank - 2 dimensions are considered
      /// as fixed, and have to be consistent across the two matrices. A
      /// common matrix multiplication is then applied over the residual
      /// 2 dimensions.
      ///
      /// e.g.
      ///     x is (3, 6, 12); y is (3, 12, 6)
      ///     batch_matmul(x, y) ==> (3, 6, 6)
      /// </remarks>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="adj_x"></param>
      /// <param name="adj_y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function batch_matmul(x: TFTensor; y: TFTensor; adj_x : Boolean = false; adj_y : Boolean= false; name : string= ''): TFTensor;
      {$ENDREGION}

      {$REGION 'tf.ops'}
      // tf.ops
      //
      procedure add_to_collection<T>(name: string; value: T);
      procedure add_to_collections<T>(names: TList<string>; value: T);
      function  clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor = nil; name: string = '') : Tuple<TFTensors, TFTensor>;
      function  assign(ref: IVariableV1; value: TValue; validate_shape: Boolean = true; use_locking: Boolean = true; name: string = ''): TFTensor;
      procedure device(device_name: string);
      function  get_collection<T>(key: string; scope: string = ''): TList<T>;
      /// <summary>
      /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
      /// When eager execution is enabled, code inside an init_scope block runs with
      /// eager execution enabled even when tracing a `tf.function`.
      /// </summary>
      /// <summary>
      /// Returns a context manager that creates hierarchical names for operations.
      /// </summary>
      /// <param name="name">The name argument that is passed to the op function.</param>
      /// <param name="default_name">The default name to use if the name argument is None.</param>
      /// <param name="values">The list of Tensor arguments that are passed to the op function.</param>
      /// <returns>The scope name.</returns>
  //    function name_scope(name: string; default_name: string = ''; values: PValue = nil): NameScope;
      /// <summary>
      /// Does nothing. Only useful as a placeholder for control edges.
      /// </summary>
      /// <param name="name"></param>
      /// <returns></returns>
      function no_op(name: string = ''): TFOperation;
      /// <summary>
      /// map on the list of tensors unpacked from `elems` on dimension 0.
      /// </summary>
      /// <param name="fn"></param>
      /// <param name="elems"></param>
      /// <param name="dtype"></param>
      /// <param name="parallel_iterations"></param>
      /// <param name="back_prop"></param>
      /// <param name="swap_memory"></param>
      /// <param name="infer_shape"></param>
      /// <param name="name"></param>
      /// <returns>A tensor or (possibly nested) sequence of tensors.</returns>
      function map_fn(fn : TFunc<TFTensor, TFTensor> ; elems: TFTensor; dtype : TF_DataType = DtInvalid; parallel_iterations : Integer= -1; back_prop: Boolean = true; swap_memory : Boolean = false; infer_shape : Boolean = true; name: string = ''): TFTensor;
      {$ENDREGION}

      {$REGION 'tf.constant'}
      // tf.constant
      //
      /// <summary>
      ///
      /// </summary>
      /// <param name="value"></param>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function constant(value: TValue; dtype : TF_DataType = DtInvalid; shape : PTFShape= nil; name : AnsiString = 'Const'): TFTensor; overload;
      function constant(value: TValue; name : AnsiString ): TFTensor; overload;
      function constant(value: TValue; shape : TFShape ): TFTensor; overload;
      function zeros(shape: TFShape; dtype:  TF_DataType = TF_DataType.TF_FLOAT; name: string = ''): TFTensor; overload;
      function zeros(shape: TFTensor; dtype: TF_DataType = TF_DataType.TF_FLOAT; name: string = ''): TFTensor; overload;
      function ones(shape: TFShape; dtype: TF_DataType = TF_DataType.TF_FLOAT; name: string = ''): TFTensor;
      function size(input: TFTensor; name: string = ''; out_type: TF_DataType = TF_DataType.TF_INT32): TFTensor;
      {$ENDREGION}

      // tf.reshape
      function  reshape(tensor: TFTensor; shape: TFShape; name: string = ''): TFTensor; overload;
      function  reshape(tensor: TFTensor; shape: TFTensor; name: string = ''): TFTensor; overload;
      function  reshape(tensor: TFTensor; shape: TArray<TValue>; name: string = ''): TFTensor; overload;

      {$REGION 'tf.gradients'}
      //tf.gradients
      //
      // Gradient
      /// <summary>
      /// Record operations for automatic differentiation.
      /// </summary>
      /// <param name="persistent"></param>
      /// <param name="watch_accessed_variables"></param>
      /// <returns>Tape set</returns>
      function GradientTape(persistent: Boolean = false; watch_accessed_variables: Boolean = true): TGradientTape;
      function GetTapeSet: TStack<ITape>;
      {$ENDREGION}


      {$REGION 'tf.variable'}
      // tf.variable
      //
      /// <summary>
      /// Returns an Op that initializes a list of variables.
      /// </summary>
      /// <param name="var_list">List of `Variable` objects to initialize.</param>
      /// <param name="name">Optional name for the returned operation.</param>
      /// <returns>An Op that run the initializers of all the specified variables.</returns>
      function variables_initializer(var_list: TArray<IVariableV1>; name : string= 'init'):TFOperation;
      function global_variables_initializer: TFOperation;
      function global_variables(scope: string = '') : TArray<IVariableV1>;
      function trainable_variables(scope: string = '') : TArray<IVariableV1>;
      {$ENDREGION}

      {$REGION 'tf.array'}
      // tf.array
      //
      /// <summary>
      /// Inserts a dimension of 1 into a tensor's shape.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="axis"></param>
      /// <param name="name"></param>
      /// <returns>
      /// A `Tensor` with the same data as `input`, but its shape has an additional
      /// dimension of size 1 added.
      /// </returns>
      function expand_dims(input: TFTensor; axis: Integer = -1; name: string = ''): TFTensor;
      /// <summary>
      /// Concatenates tensors along one dimension.
      /// </summary>
      /// <param name="values">A list of `Tensor` objects or a single `Tensor`.</param>
      /// <param name="axis"></param>
      /// <param name="name"></param>
      /// <returns>A `Tensor` resulting from concatenation of the input tensors.</returns>
      function concat(values: TList<TFTensor>; axis: Integer; name: string = 'concat'): TFTensor; overload;
      function concat(values: TArray<TFTensor>; axis: Integer; name: string = 'concat'): TFTensor; overload;
      /// <summary>
      /// Return a tensor with the same shape and contents as input.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function identity(input: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// BatchToSpace for N-D tensors of type T.
      /// </summary>
      /// <typeparam name="T"></typeparam>
      /// <param name="input"></param>
      /// <param name="block_shape"></param>
      /// <param name="crops"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function batch_to_space_nd<T>(input: T; block_shape: TArray<Integer>; crops: TArray< TArray<Integer> >; name: string = ''): TFTensor;
      /// <summary>
      /// Apply boolean mask to tensor.
      /// </summary>
      /// <typeparam name="T1"></typeparam>
      /// <typeparam name="T2"></typeparam>
      /// <param name="tensor">N-D tensor.</param>
      /// <param name="mask">K-D boolean tensor, K &lt;= N and K must be known statically.</param>
      /// <param name="name"></param>
      /// <param name="axis">A 0-D int Tensor representing the axis in tensor to mask from. </param>
      /// <returns>(N-K+1)-dimensional tensor populated by entries in tensor corresponding to True values in mask.</returns>
      function  boolean_mask<T1, T2>(tensor: T1; mask: T2; name: string = 'boolean_mask'; axis: Integer = 0): TFTensor;
      /// <summary>
      /// Gather slices from params axis axis according to indices.
      /// </summary>
      /// <param name="params"></param>
      /// <param name="indices"></param>
      /// <param name="name"></param>
      /// <param name="axis"></param>
      /// <returns></returns>
      function gather(params: TFTensor; indices: TFTensor; name: string = ''; axis: Integer = 0): TFTensor;
      /// <summary>
      /// Extracts a slice from a tensor.
      /// </summary>
      /// <param name="input">A `Tensor`.</param>
      /// <param name="begin">An `int32` or `int64` `Tensor`.</param>
      /// <param name="size">An `int32` or `int64` `Tensor`.</param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>A `Tensor` the same type as `input`.</returns>
      function slice<Tb, Ts>(input: TFTensor; _begin : TArray<Tb>; size : TArray<Ts>; name: string = ''): TFTensor;
      /// <summary>
      /// Transposes `a`. Permutes the dimensions according to `perm`.
      /// </summary>
      /// <param name="a"></param>
      /// <param name="perm"></param>
      /// <param name="name"></param>
      /// <param name="conjugate"></param>
      /// <returns></returns>
      function transpose<T1>(a: T1; perm: PAxis = nil; name: string = 'transpose'; conjugate: Boolean = false): TFTensor;
      /// <summary>
      /// Creates a tensor with all elements set to zero.
      /// </summary>
      /// <param name="tensor"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="optimize"></param>
      /// <returns>A `Tensor` with all elements set to zero.</returns>
      function zeros_like(tensor: TFTensor; dtype: TF_DataType = DtInvalid; name : string= ''; optimize : Boolean= true): TFTensor; overload;
      function zeros_like(nd    : TNDArray; dtype: TF_DataType = DtInvalid; name : string= ''; optimize : Boolean= true): TFTensor; overload;
      /// <summary>
      /// Creates a tensor with all elements set to 1.
      /// </summary>
      /// <param name="tensor"></param>
      /// <param name="dtype"></param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <param name="optimize">
      /// if true, attempt to statically determine the shape of 'tensor' and
      /// encode it as a constant.
      /// </param>
      /// <returns>A `Tensor` with all elements set to 1.</returns>
      function ones_like(tensor:TFTensor ; dtype: TF_DataType = DtInvalid; name: string = ''; optimize: Boolean = true): TFTensor; overload;
      function ones_like(nd:TNDArray     ; dtype: TF_DataType = DtInvalid; name: string = ''; optimize: Boolean = true): TFTensor; overload;
      /// <summary>
      /// Return the elements, either from `x` or `y`, depending on the `condition`.
      /// </summary>
      /// <returns></returns>
      function where<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the shape of a tensor.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="name"></param>
      /// <param name="out_type"></param>
      /// <returns></returns>
      function shape(input: TFTensor; name: string = ''; out_type: TF_DataType = TF_INT32): TFTensor;
      function TensorArray(dtype: TF_DataType; size : Integer= 0; dynamic_size: Boolean = false; clear_after_read : Boolean= true; element_shape: PTFShape = nil; colocate_with_first_write_call: Boolean = true; infer_shape: Boolean = true): TTensorArray ; overload;
      function TensorArray(dtype: TF_DataType; size : TFTensor; dynamic_size: Boolean = false; clear_after_read : Boolean= true; element_shape: PTFShape = nil; colocate_with_first_write_call: Boolean = true; infer_shape: Boolean = true): TTensorArray ; overload;
      function squeeze(input: TFTensor; axis: Integer;              name: string= ''; squeeze_dims: Integer = -1): TFTensor;overload;
      function squeeze(input: TFTensor; axis: TArray<Integer> = []; name: string= ''; squeeze_dims: Integer = -1): TFTensor;overload;
      {$ENDREGION}

      {$REGION 'tf.sparse'}
      // tf.sparse
      //
      /// <summary>
      /// Converts a sparse representation into a dense tensor.
      /// </summary>
      /// <typeparam name="T"></typeparam>
      /// <param name="sparse_indices"></param>
      /// <param name="output_shape"></param>
      /// <param name="sparse_values"></param>
      /// <param name="default_value"></param>
      /// <param name="validate_indices"></param>
      /// <param name="name"></param>
      /// <returns>Dense `Tensor` of shape `output_shape`.  Has the same type as `sparse_values`.</returns>
      function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFShape; sparse_values: T; default_value: T; validate_indices: Boolean = true; name : string = ''): TFTensor;overload;
      function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFShape; sparse_values: T): TFTensor;overload;
      function SparseTensor(indices: TArray< TArray<Int64> >; values: TValue; dense_shape:TArray<Int64>) : TSparseTensor;  overload;
      function SparseTensor(indices: TArray<TArray<Int64>>;   values: TArray<Integer>; dense_shape: TArray<Int64>): TSparseTensor; overload;
      function sparse_tensor_to_dense(sp_input: TSparseTensor; default_value: TValue; validate_indices : Boolean= true; name: string = ''): TFTensor;
      {$ENDREGION}

      {$REGION 'tf.math'}
      // tf.math
      //
      /// <summary>
      /// Computes the sum of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input"></param>
      /// <param name="axis"></param>
      /// <returns></returns>
      function reduce_sum(input: TFTensor; axis: PAxis = nil; reduction_indices: PAxis = nil; keepdims: Boolean = false; name: string = '') : TFTensor;
      function range(start: TValue; limit: TValue; delta: TValue; dtype: Nullable<TF_DataType>; name: string = 'range'): TFTensor; overload;
      function range(start: TValue; limit: TValue): TFTensor; overload;
      function negative(x: TFTensor; name: string = ''): TFTensor;
      function add(a: TFTensor; b: TFTensor; name: string = ''): TFTensor; overload;
      function add<Tx, Ty>(a: Tx; b: Ty; name: string = ''): TFTensor; overload;
      function multiply(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; overload;
      /// <summary>
      /// return x * y
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function multiply<Tx, Ty>(x: Tx; y: Ty; name: string = ''): TFTensor; overload;
      function pow<T1, T2>(x:T1; y: T2; name: string = 'pow'): TFTensor;
      function abs(x: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes acos of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function acos(x: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes asin of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function asin(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Adds all input tensors element-wise.
      /// </summary>
      /// <param name="inputs"></param>
      /// <param name="name"></param>
      /// <returns>A `Tensor` of same shape and type as the elements of `inputs`.</returns>
      function add_n(inputs: TArray<TFTensor>; name: string = ''): TFTensor;
      /// <summary>
      /// Computes atan of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function atan(x : TFTensor; name: string = ''): TFTensor;
      function arg_max(input : TFTensor; dimension: Integer; output_type: TF_DataType = TF_INT64; name: string = ''): TFTensor;
      function arg_min(input : TFTensor; dimension: Integer; output_type: TF_DataType = TF_INT64; name: string = ''): TFTensor;
      function is_finite(input : TFTensor; name: string = ''): TFTensor;
      function is_nan(input : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Returns element-wise smallest integer not less than x.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function ceil(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes sin of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function sin(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes hyperbolic sine of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function sinh(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes cos of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function cos(x : TFTensor; name: string = ''): TFTensor; overload;
      function cos(x: Single; name: string = ''): TFTensor; overload;
      /// <summary>
      /// Computes hyperbolic cosine of x element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function cosh(x : TFTensor; name: string = ''): TFTensor;
      function tan(x : TFTensor; name: string = ''): TFTensor;
      function tanh(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Returns element-wise largest integer not greater than x.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function floor(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the truth value of (x > y) element-wise.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function greater<Tx, Ty>(x : Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the truth value of (x >= y) element-wise.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function greater_equal<Tx, Ty>(x : Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the truth value of (x &lt; y) element-wise.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function less<Tx, Ty>(x : Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the log of the absolute value of `Gamma(x)` element-wise.
      /// </summary>
      /// <param name="x">A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
      /// <param name="name">A name for the operation (optional).</param>
      /// <returns>A `Tensor`. Has the same type as `x`.</returns>
      function lgamma(x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the truth value of (x &lt;= y) element-wise.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function less_equal<Tx, Ty>(x : Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Computes natural logarithm of (1 + x) element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function log1p(x : TFTensor; name: string = ''): TFTensor;
      function logical_and<T>(x: T; y: T; name: string = ''): TFTensor;
      function logical_not(x : TFTensor; name: string = ''): TFTensor;
      function logical_or(x : TFTensor;  y: TFTensor; name: string = ''): TFTensor;
      function logical_xor(x : TFTensor; y: TFTensor; name : string = 'LogicalXor'): TFTensor;
      /// <summary>
      /// Clips tensor values to a specified min and max.
      /// </summary>
      /// <param name="t"></param>
      /// <param name="clip_value_min"></param>
      /// <param name="clip_value_max"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function _clip_by_value(t: TFTensor; clip_value_min: TFTensor; clip_value_max: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      ///    Clips tensor values to a specified min and max.
      /// </summary>
      /// <param name="t">
      ///    A <c>Tensor</c>.
      /// </param>
      /// <param name="clip_value_min">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The minimum value to clip by.
      /// </param>
      /// <param name="clip_value_max">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The maximum value to clip by.
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ClipByValue'.
      /// </param>
      /// <returns>
      ///    A clipped <c>Tensor</c> with the same shape as input 't'.
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      /// <remarks>
      ///    Given a tensor <c>t</c>, this operation returns a tensor of the same type and
      ///    shape as <c>t</c> with its values clipped to <c>clip_value_min</c> and <c>clip_value_max</c>.
      ///    Any values less than <c>clip_value_min</c> are set to <c>clip_value_min</c>. Any values
      ///    greater than <c>clip_value_max</c> are set to <c>clip_value_max</c>.
      /// </remarks>
      function clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max:T2; name: string = 'ClipByValue'): TFTensor;
      function sub<Tx, Ty>(a: Tx; b: Ty; name: string = ''): TFTensor;
      function sqrt(a: TFTensor; name: string = ''): TFTensor;
      function sign(a: TFTensor; name: string = ''): TFTensor;
      function subtract<T>(x : TFTensor;  y: TArray<T>; name: string = ''): TFTensor; overload;
      function log(x: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// return x - y
      /// </summary>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function subtract(x : TFTensor; y: TFTensor; name: string = ''): TFTensor; overload;
      function equal(x : TFTensor; y: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
      /// </summary>
      /// <param name="y"></param>
      /// <param name="x"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function atan2(y: TFTensor; x : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the maximum of elements across dimensions of a tensor.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="input"></param>
      /// <param name="axis"></param>
      /// <param name="keep_dims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function max<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the minimum of elements across dimensions of a tensor.
      /// </summary>
      /// <typeparam name="Tx"></typeparam>
      /// <typeparam name="Ty"></typeparam>
      /// <param name="input"></param>
      /// <param name="axis"></param>
      /// <param name="keep_dims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function min<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean = false; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
      /// </summary>
      /// <typeparam name="T1"></typeparam>
      /// <typeparam name="T2"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function maximum<T1, T2>(x: T1; y: T2; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the min of x and y (i.e. x &lt; y ? x : y) element-wise.
      /// </summary>
      /// <typeparam name="T1"></typeparam>
      /// <typeparam name="T2"></typeparam>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function minimum<T1, T2>(x: T1; y: T2; name: string = ''): TFTensor;
      /// <summary>
      /// Returns the truth value of (x != y) element-wise.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns>A `Tensor` of type bool with the same size as that of x or y.</returns>
      function not_equal<Tx, Ty>(x: Tx; y: Ty; name: string = ''): TFTensor;
      /// <summary>
      /// Divides x / y elementwise (using Python 2 division operator semantics).
      /// </summary>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function &div(x : TFTensor; y: TFTensor; name: string = ''): TFTensor;
      function divide(a: TFTensor; b: TFTensor): TFTensor; overload;
      function divide<T>(x : TFTensor; y: TArray<T>; name: string = ''): TFTensor; overload;
      /// <summary>
      /// Divides `x / y` elementwise, rounding toward the most negative integer.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns>`x / y` rounded down.</returns>
      function floordiv(x : TFTensor; y: TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Divides x / y elementwise (using Python 3 division operator semantics).
      /// </summary>
      /// <param name="x"></param>
      /// <param name="y"></param>
      /// <param name="name"></param>
      /// <returns>`x / y` evaluated in floating point.</returns>
      class function truediv(x : TFTensor; y: TFTensor; name: string = ''): TFTensor;
      function real(input : TFTensor; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the "logical or" of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor">The boolean tensor to reduce.</param>
      /// <param name="axis">The dimensions to reduce.</param>
      /// <param name="keepdims">If true, retains reduced dimensions with length 1.</param>
      /// <param name="name"></param>
      /// <returns>The reduced tensor.</returns>
      function reduce_any(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the "logical and" of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor"></param>
      /// <param name="axis"></param>
      /// <param name="keepdims"></param>
      /// <param name="name"></param>
      /// <returns>The reduced tensor.</returns>
      function reduce_all(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the product of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor"></param>
      /// <param name="axis"></param>
      /// <param name="keepdims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function reduce_prod(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      /// <summary>
      /// Computes the maximum of elements across dimensions of a tensor.
      /// </summary>
      /// <param name="input_tensor"></param>
      /// <param name="axis"></param>
      /// <param name="keepdims"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function reduce_max     (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      function reduce_min     (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      function reduce_std     (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      function reduce_variance(input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''): TFTensor;
      function reduce_mean    (input_tensor: TFTensor; axis: PAxis = nil; keepdims: Boolean = false; name: string = ''; reduction_indices: PInteger = nil): TFTensor;
      function sigmoid<T>(x: T; name: string = ''): TFTensor;
      function sum(input : TFTensor; axis: Integer; keep_dims: Boolean = false; name: string = ''): TFTensor;
      function round(x : TFTensor; name: string = ''): TFTensor;
      function cast(x : TFTensor; dtype: TF_DataType; name: string = ''): TFTensor;
      function cumsum(x : TFTensor; axis: Integer = 0; exclusive: Boolean = false; reverse: Boolean = false; name: string = ''): TFTensor;
      function square(x : TFTensor; name: string = ''): TFTensor;
      function squared_difference(x : TFTensor; y: TFTensor; name: string = '') : TFTensor;
      function exp(x: TFTensor; name: string = ''): TFTensor;
      {$ENDREGION}

      {$REGION 'tf.random'}
      // tf.random
      //
      function  random_uniform(shape: TFShape; minval: Single = 0; maxval: Single = 1; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = ''): TFTensor;
      Function  truncated_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = ''): TFTensor;
      Function  random_shuffle(value: TFTensor; seed: Integer = 0; name: string = '') : TFTensor;
      procedure set_random_seed(seed: Integer);
      Function  multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger = nil; name: string = ''; output_dtype: TF_DataType = DtInvalid): TFTensor;
      {$ENDREGION}

      property Version : string read GetVersion;
      property MemoLog : TStringList  read FMemoLog write FMemoLog;

  end;
{$ENDREGION}

  var
   tf : TTensorflow;

implementation
   uses Oz.Pb.Classes, Oz.SGL.Collections,oz.Pb.StrBuffer, pbPublic, pbInput, pbOutput,

        NumPy.NDArray,
        TensorFlow.EagerTensor,
        TensorFlow.Ops ,
        TensorFlow.Constant_op,
        Tensorflow.math_ops,
        TensorFlow.gen_math_ops,
        Tensorflow.gen_array_ops,
        Tensorflow.array_ops,
        TensorFlow.clip_ops,
        TensorFlow.gen_control_flow_ops,
        tensorflow.gen_sparse_ops,
        TensorFlow.random_ops,
        TensorFlow.gen_nn_ops,
        TensorFlow.nn_ops,
        TensorFlow.nn_impl,
        TensorFlow.image_ops_impl,
        tensorflow.stateless_random_ops,
        TensorFlow.Tensor;

{$REGION 'TTensorflow'}
{ TTensorflow }

constructor TTensorflow.Create;
begin
    Context   := TContext.Create;
    Status    := TFStatus.Create;
    OpDefLib  := OpDefLibrary.Create;
    runner    := TEagerRunner.Create ;

    FtapeSet  := TGradientTape.Create;
    compat    := CompatApi.Create;
    strings   := StringsApi.Create;
    GraphKeys := TGraphKeys.Create;
    data      := DataOps.Create;
    // Get gradient Function
    TOps.RegisterFromAssembly;
    //
    random    := TRandom.Create;
    numpy     := NumPyImpl.Create;
    math      := MathApi.Create;
    nn        := nn_internal.Create;
    bitwise   := bitwise_ops.Create;
    train     := train_internal.Create;
    linalg    := LinalgApi.Create;
    image     := image_internal.Create;
    //
    glorot_uniform_initializer := GlorotUniform.Create;
    zeros_initializer          := TensorFlow.Initializer.Zeros.Create;
    ones_initializer           := TensorFlow.Initializer.Ones.Create;
    random_uniform_initializer := RandomUniform.Create;
    orthogonal_initializer     := Orthogonal.Create;

    FMemoLog := TStringList.Create;
end;

destructor TTensorflow.Destroy;
begin
  Context.Free;
  Status.Free;
  OpDefLib.Free;
  Runner.Free;
  FtapeSet.Free;
  compat.Free;
  strings.Free;
  GraphKeys := System.Default(TGraphKeys);
  data.Free;
  op_def_registry.FreeDictionary;
  //
  random.Free;
  numpy.Free;
  math.Free;
  nn.Free;
  bitwise.Free;
  train.Free;
  linalg.Free;
  image.Free;
  //
  glorot_uniform_initializer.Free;
  zeros_initializer.Free;
  ones_initializer.Free;
  random_uniform_initializer.Free;
  orthogonal_initializer.Free;
  //
  FMemoLog.Free;
  //
  if Assigned(gradientFunctions) then  gradientFunctions.Free;

  inherited Destroy;

end;

procedure TTensorflow.LogMsg(Msg: string);
begin
    if Assigned(FMemoLog) then
       FMemoLog.Add(Msg);
end;

function TTensorflow.diag(diagonal: TFTensor; name: string): TFTensor;
begin
   Result := gen_array_ops.diag(diagonal, name);
end;

function TTensorflow.matmul(a, b: TFTensor; transpose_a, transpose_b: Boolean): TFTensor;
begin
   Result := math_ops.matmul(a, b, transpose_a, transpose_b);
end;

function TTensorflow.batch_matmul(x, y: TFTensor; adj_x, adj_y: Boolean; name: string): TFTensor;
begin
    Result := math_ops.batch_matmul(x, y, adj_x, adj_y, name);
end;

function TTensorflow.convert_to_tensor(value: TValue; dtype: TF_DataType; name: string; preferred_dtype: TF_DataType): TFTensor;
begin
    Result := TOps.convert_to_tensor(value,dtype, name,False,preferred_dtype);
end;

function TTensorflow.cos(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.cos(x, name);
end;

function TTensorflow.cos(x: Single; name: string): TFTensor;
begin
    Result := gen_math_ops.cos(x, name);
end;

function TTensorflow.cosh(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.cosh(x, name);
end;

function TTensorflow.concat(values: TList<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    if values.Count = 1 then
    begin
        Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var tensor := Tops.convert_to_tensor(axis, TDtypes.cint32, 'concat_dim');
                                                Assert(tensor.shape.ndim = 0);
                                                Result := identity(values.First,  v1.toString);
                                            end );
        Exit;
    end;
    Result := gen_array_ops.concat_v2(values.ToArray, axis, name);
end;

function TTensorflow.abs(x: TFTensor; name: string): TFTensor;
begin
    Result := math_ops.abs(x, name);
end;

function TTensorflow.acos(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.acos(x, name);
end;

function TTensorflow.&div(x, y: TFTensor; name: string): TFTensor;
begin
    Result := math_ops.div(x, y, name);
end;

function TTensorflow.add(a, b: TFTensor; name: string): TFTensor;
begin
    Result :=  gen_math_ops.add(a, b, name);
end;

function TTensorflow.add<Tx, Ty>(a: Tx; b: Ty; name: string): TFTensor;
begin
    Result :=  gen_math_ops.add(a, b, name);
end;

function TTensorflow.add_n(inputs: TArray<TFTensor>; name: string): TFTensor;
begin
    Result := math_ops.add_n(inputs, name);
end;

procedure TTensorflow.device(device_name: string);
begin
    get_default_graph.device(device_name);
end;

procedure TTensorflow.add_to_collection<T>(name: string; value: T);
begin
    get_default_graph.add_to_collection<T>(name, value);
end;

procedure TTensorflow.add_to_collections<T>(names: TList<string>; value: T);
begin
    get_default_graph.add_to_collection<T>(names, value);
end;

function TTensorflow.assign(ref: IVariableV1; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
    Result := state_ops.assign(ref, value, validate_shape, use_locking, name);
end;

function TTensorflow.map_fn(fn: TFunc<TFTensor, TFTensor>; elems: TFTensor; dtype: TF_DataType; parallel_iterations: Integer; back_prop, swap_memory, infer_shape: Boolean;
  name: string): TFTensor;
begin
    Result := TFOperation.map_fn(fn, elems, dtype,parallel_iterations, back_prop, swap_memory, infer_shape, name)
end;

function TTensorflow.get_collection<T>(key, scope: string): TList<T>;
begin
    Result := get_default_graph.get_collection<T>(key, scope);
end;

function TTensorflow.clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor; name: string): Tuple<TFTensors, TFTensor>;
begin
   Result := clip_ops.clip_by_global_norm(t_list, clip_norm, use_norm, name);
end;

function TTensorflow.no_op(name: string): TFOperation;
begin
    Result := gen_control_flow_ops.no_op(name);
end;

function TTensorflow.arg_max(input: TFTensor; dimension: Integer; output_type: TF_DataType; name: string): TFTensor;
begin
    Result := gen_math_ops.arg_max(input, dimension, output_type, name);
end;

function TTensorflow.arg_min(input: TFTensor; dimension: Integer; output_type: TF_DataType; name: string): TFTensor;
begin
    Result := gen_math_ops.arg_min(input, dimension, output_type, name);
end;

function TTensorflow.asin(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.asin(x, name);
end;

function TTensorflow.atan(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.atan(x, name);
end;

function TTensorflow.atan2(y, x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.atan2(y, x, name);
end;

function TTensorflow.max<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name: string): TFTensor;
begin
    Result :=  gen_math_ops._max(input, axis, keep_dims, name);
end;

function TTensorflow.maximum<T1, T2>(x: T1; y: T2; name: string): TFTensor;
begin
    Result := gen_math_ops.maximum(x, y, name);
end;

function TTensorflow.min<Tx, Ty>(input: Tx; axis: Ty; keep_dims: Boolean; name: string): TFTensor;
begin
    Result := gen_math_ops._min(input, axis, keep_dims, name);
end;

function TTensorflow.minimum<T1, T2>(x: T1; y: T2; name: string): TFTensor;
begin
    Result := gen_math_ops.minimum(x, y, name);
end;

function TTensorflow.multiply(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.mul(x, y, name);
end;

function TTensorflow.multiply<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.mul(x, y, name);
end;

function TTensorflow.batch_to_space_nd<T>(input: T; block_shape: TArray<Integer>; crops: TArray<TArray<Integer>>; name: string): TFTensor;
begin
    Result := gen_array_ops.batch_to_space_nd(input, block_shape, crops, name)
end;

function TTensorflow.boolean_mask<T1, T2>(tensor: T1; mask: T2; name: string; axis: Integer): TFTensor;
begin
    Result := array_ops.boolean_mask(tensor, mask, name, axis);
end;

function TTensorflow.cast(x: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result :=  math_ops.cast(x, dtype, name);
end;

function TTensorflow.ceil(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.ceil(x, name);
end;

function TTensorflow.clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string): TFTensor;
begin
    Result := clip_ops.clip_by_value(t, clip_value_min, clip_value_max, name);
end;

function TTensorflow.concat(values: TArray<TFTensor>; axis: Integer; name: string): TFTensor;
begin
    Result := concat(TList<TFTensor>.Create(values),axis,name);
end;

function TTensorflow.constant(value: TValue; shape: TFShape): TFTensor;
begin
    var Sshape : PTFShape;
    if shape.IsNil then Sshape := nil
    else                Sshape := @shape;

    Result := constant(value, DtInvalid, Sshape, 'Const');
end;

function TTensorflow.constant(value: TValue; name: AnsiString): TFTensor;
begin
    Result := constant(value, DtInvalid, nil, name);
end;

function TTensorflow.constant(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString): TFTensor;
begin
    Result :=constant_op.constant(value,
                                  dtype,
                                  shape,
                                  False,
                                  True,
                                  name);
end;

function TTensorflow.constant_initializer<T>(value: T; dtype: TF_DataType; verify_shape: Boolean): IInitializer;
begin
    Result := Constant<T>.Create(value, dtype, verify_shape)
end;

function TTensorflow.cumsum(x: TFTensor; axis: Integer; exclusive, reverse: Boolean; name: string): TFTensor;
begin
    Result := math_ops.cumsum(x, axis, exclusive, reverse, name);
end;

function TTensorflow.divide(a, b: TFTensor): TFTensor;
begin
    Result := TTensor(a) / b
end;

function TTensorflow.divide<T>(x: TFTensor; y: TArray<T>; name: string): TFTensor;
begin
    Result := TTensor(x) / Tops.convert_to_tensor( TValue.From< TArray<T> >(y), Tdtypes.as_base_dtype(x.dtype), 'y')
end;

procedure TTensorflow.enable_eager_execution;
begin
    Context.eager_mode;
end;

function TTensorflow.equal(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.equal(x, y, True, name);
end;

function TTensorflow.executing_eagerly: Boolean;
begin
    Result := Context.executing_eagerly;
end;

function TTensorflow.exp(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.exp(x, name);
end;

function TTensorflow.expand_dims(input: TFTensor; axis: Integer; name: string): TFTensor;
begin
    Result := array_ops.expand_dims(input, axis, name);
end;

function TTensorflow.floor(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.floor(x, name);
end;

function TTensorflow.floordiv(x, y: TFTensor; name: string): TFTensor;
begin
   Result := math_ops.floordiv(x, y, name);
end;

function TTensorflow.gather(params, indices: TFTensor; name: string; axis: Integer): TFTensor;
begin
    Result := array_ops.gather(params, indices, name, axis);
end;

function TTensorflow.GetVersion: string;
begin
     Result := string(AnsiString(TF_Version));
end;

procedure TTensorflow.NativeDispose(hnd: Pointer);
begin
  inherited;

  Context.Free;
  Status.Free;
  OpDefLib.Free;
  Runner.Free;
  FtapeSet.Free;

end;

function TTensorflow.negative(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.neg(x, name);
end;

function TTensorflow.not_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := math_ops.not_equal(x, y, name);
end;

function TTensorflow.get_default_graph: TFgraph;
begin
    Result := TOps.get_default_graph;
end;

function TTensorflow.get_default_session: TFSession;
begin
    Result := Tops.get_default_session;
end;

function TTensorflow.global_variables(scope: string): TArray<IVariableV1>;
begin
    var Value := TOps.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) ;
    Result := Value.AsType< TList<IVariableV1> >.ToArray;
end;

function TTensorflow.global_variables_initializer: TFOperation;
begin
    Result := tf.compat.v1.global_variables_initializer;
end;

function TTensorflow.Graph: TFGraph;
begin
    Result := TFGraph.Create;
end;

function TTensorflow.greater<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.greater(x, y, name);
end;

function TTensorflow.greater_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.greater_equal(x, y, name);
end;

function TTensorflow.identity(input: TFTensor; name: string): TFTensor;
begin
    Result := array_ops.identity(input, name);
end;

function TTensorflow.init_scope: TNameScope;
begin
    Result := Tops.init_scope
end;

function TTensorflow.is_finite(input: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.is_finite(input, name);
end;

function TTensorflow.is_nan(input: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.is_nan(input, name);
end;

function TTensorflow.less<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.less(x, y, name);
end;

function TTensorflow.less_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.less_equal(x, y, name);
end;

function TTensorflow.lgamma(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.lgamma(x, name);
end;

function TTensorflow.log(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.log(x, name);
end;

function TTensorflow.log1p(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.log1p(x, name);
end;

function TTensorflow.logical_and<T>(x, y: T; name: string): TFTensor;
begin
   Result := gen_math_ops.logical_and(x, y, name);
end;

function TTensorflow.logical_not(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.logical_not(x, name);
end;

function TTensorflow.logical_or(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.logical_or(x, y, name);
end;

function TTensorflow.logical_xor(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.logical_xor(x, y, name);
end;

function TTensorflow.peak_default_graph: TFgraph;
begin
    Result := TOps.peak_default_graph;
end;

function TTensorflow.placeholder(dtype: TF_DataType): TFTensor;
begin
    Result := array_ops.placeholder(dtype,nil,'');
end;

function TTensorflow.placeholder_with_default<T>(input: T; shape: TArray<Integer>; name: string): TFTensor;
begin
    Result := gen_array_ops.placeholder_with_default(input, shape, name);
end;

function TTensorflow.pow<T1, T2>(x: T1; y: T2; name: string): TFTensor;
begin
    Result := math_ops.pow(x, y, name);
end;

function TTensorflow.placeholder(dtype: TF_DataType; shape: TFShape; name: string): TFTensor;
begin
    Result := array_ops.placeholder(dtype,@shape,name);
end;

function TTensorflow.range(start, limit, delta: TValue; dtype: Nullable<TF_DataType>; name: string): TFTensor;
begin
   var pl : PValue := nil;
   if limit.typeInfo <> nil then  pl := @limit;

   var pD : PValue := nil;
   if delta.typeInfo <> nil then  pD := @delta;

   Result :=  math_ops.range(start, pl, pD, dtype, name);
end;

function TTensorflow.random_uniform(shape: TFShape; minval, maxval: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    Result := random.uniform(shape, minval, maxval, dtype, seed, name);
end;

Function TTensorflow.truncated_normal(shape: TFShape; mean: Single; stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    Result := random_ops.truncated_normal(shape, mean, stddev, dtype, seed, name);
end;

Function TTensorflow.random_shuffle(value: TFTensor; seed: Integer; name: string) : TFTensor;
begin
    Result := random_ops.random_shuffle(value, seed, name);
end;

procedure TTensorflow.set_random_seed(seed: Integer);
begin
    if executing_eagerly then
        Context.set_global_seed(seed)
    else
        Tops.get_default_graph.seed := seed;
end;

function TTensorflow.shape(input: TFTensor; name: string; out_type: TF_DataType): TFTensor;
begin
    Result := array_ops.shape_internal(input, name, true, out_type);
end;

function TTensorflow.TensorArray(dtype: TF_DataType; size: Integer; dynamic_size, clear_after_read: Boolean; element_shape: PTFShape; colocate_with_first_write_call,
  infer_shape: Boolean): TTensorArray;
begin
    if tf.executing_eagerly then
      Result := TEagerTensorArray.Create(dtype, constant_op.constant(size), dynamic_size, clear_after_read, '', nil, nil, infer_shape, element_shape, colocate_with_first_write_call)
    else
      Result := TGraphTensorArray.Create(dtype, constant_op.constant(size), dynamic_size, clear_after_read, '', nil, nil, infer_shape, element_shape, colocate_with_first_write_call)
end;

function TTensorflow.TensorArray(dtype: TF_DataType; size: TFTensor; dynamic_size, clear_after_read: Boolean; element_shape: PTFShape; colocate_with_first_write_call,
  infer_shape: Boolean): TTensorArray;
begin
    if tf.executing_eagerly then
      Result := TEagerTensorArray.Create(dtype, size, dynamic_size, clear_after_read, '', nil, nil, infer_shape, element_shape, colocate_with_first_write_call)
    else
      Result := TGraphTensorArray.Create(dtype, size, dynamic_size, clear_after_read, '', nil, nil, infer_shape, element_shape, colocate_with_first_write_call)
end;

Function TTensorflow.multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger; name: string; output_dtype: TF_DataType): TFTensor;
begin
    Result := random_ops.multinomial(logits, num_samples, seed, name, output_dtype);
end;

function TTensorflow.range(start, limit: TValue): TFTensor;
begin
    var v     : TValue                := System.default(TValue);
    var nTipo : Nullable<TF_DataType> := dtinvalid ;

    Result := range(start, limit,v, nTipo,'range');
end;

function TTensorflow.real(input: TFTensor; name: string): TFTensor;
begin
    Result := math_ops.real(input, name);
end;

function TTensorflow.reduce_all(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
   Result := math_ops.reduce_all(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_any(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    Result := math_ops.reduce_any(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_max(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    Result := math_ops.reduce_max(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_mean(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string; reduction_indices: PInteger): TFTensor;
begin
    Result := math_ops.reduce_mean(input_tensor, axis, keepdims, name, reduction_indices);
end;

function TTensorflow.reduce_min(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    Result := math_ops.reduce_min(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_prod(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
   Result := math_ops.reduce_prod(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_std(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
   Result := math_ops.reduce_std(input_tensor, axis, keepdims, name);
end;

function TTensorflow.reduce_sum(input: TFTensor; axis, reduction_indices: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    if keepdims then
    begin
        if axis <> nil then Result := math_ops.reduce_sum(input, constant_op.constant(TValue.From<TAxis>(axis^)), keepdims, name)
        else begin
              var v : TValue := System.default(TValue);
              if reduction_indices <> nil then v := TValue.From<TAxis>(reduction_indices^);

              Result := math_ops.reduce_sum(input, constant_op.constant(v), keepdims, name);
        end;
    end else
    begin
        if axis <> nil then Result := math_ops.reduce_sum( input, constant_op.constant(TValue.From<TAxis>(axis^)) )
        else begin
              var v : TValue := System.default(TValue);
              if reduction_indices <> nil then v := TValue.From<TAxis>(reduction_indices^);

              Result := math_ops.reduce_sum( input, constant_op.constant(v) );
        end;
    end;
end;

function TTensorflow.reduce_variance(input_tensor: TFTensor; axis: PAxis; keepdims: Boolean; name: string): TFTensor;
begin
    Result := math_ops.reduce_variance(input_tensor, axis^, keepdims, name);
end;

procedure TTensorflow.reset_default_graph;
begin
    TOps.reset_default_graph
end;

function TTensorflow.reshape(tensor: TFTensor; shape: TFShape; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

function TTensorflow.reshape(tensor: TFTensor; shape: TArray<TValue>; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

function TTensorflow.reshape(tensor, shape: TFTensor; name: string): TFTensor;
begin
    Result := gen_array_ops.reshape(tensor, shape, name);
end;

function TTensorflow.round(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.round(x, name);
end;

function TTensorflow.Session: TFSession;
begin
    Result := compat.v1.Session;
end;

function TTensorflow.Session(graph: TFGraph; config: TConfigProto): TFSession;
begin
    Result := TFSession.Create(graph, config).as_default;
end;

function TTensorflow.ones(shape: TFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := array_ops.ones(shape, dtype, name);
end;

function TTensorflow.ones_like(tensor: TFTensor; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
   Result := array_ops.ones_like(tensor, dtype, name, optimize);
end;

function TTensorflow.ones_like(nd: TNDArray; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
   Result := array_ops.ones_like(nd, dtype, name, optimize);
end;

function TTensorflow.sigmoid<T>(x: T; name: string): TFTensor;
begin
    Result := math_ops.sigmoid(x, name);
end;

function TTensorflow.sign(a: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sign(a, name);
end;

function TTensorflow.sin(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sin(x, name);
end;

function TTensorflow.sinh(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sinh(x, name);
end;

function TTensorflow.size(input: TFTensor; name: string; out_type: TF_DataType): TFTensor;
begin
    Result := array_ops.size(input, name, true, out_type);
end;

function TTensorflow.slice<Tb, Ts>(input: TFTensor; _begin: TArray<Tb>; size: TArray<Ts>; name: string): TFTensor;
begin
    Result := array_ops.slice(input, _begin, size, name);
end;

function TTensorflow.SparseTensor(indices: TArray<TArray<Int64>>; values: TValue; dense_shape: TArray<Int64>): TSparseTensor;
begin
    Result := TSparseTensor.Create(indices, values, dense_shape);
end;

function TTensorflow.SparseTensor(indices: TArray<TArray<Int64>>; values: TArray<Integer>; dense_shape: TArray<Int64>): TSparseTensor;
begin
    Result := SparseTensor(indices, TValue.From< TArray<Integer> >(values), dense_shape);
end;

function TTensorflow.sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFShape; sparse_values, default_value: T; validate_indices: Boolean; name: string): TFTensor;
begin
    Result := gen_sparse_ops.sparse_to_dense(sparse_indices,output_shape,sparse_values, default_value, validate_indices, name)
end;

function TTensorflow.sparse_tensor_to_dense(sp_input: TSparseTensor; default_value: TValue; validate_indices: Boolean; name: string): TFTensor;
begin
    var v : TFTensor := tf.constant( 0 ) ;
    Result :=  gen_sparse_ops.sparse_to_dense(sp_input.indices,sp_input.dense_shape, sp_input.values,v,True,'')
end;

function TTensorflow.sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFShape; sparse_values: T): TFTensor;
var
   v : T;
begin
   var FValue : TValue := TValue.from<T>(sparse_values) ;
   if FValue.IsOrdinal then
   begin
       var v1 : TValue := 0;
       v := v1.asType<T>;
   end
   else if FValue.IsType<TFTensor> then
   begin
       var v1 : TValue := TValue.From<TFTensor>( TFTensor.Create(0) ) ;
       v := v1.asType<T>;
   end
   else if FValue.IsType<TNDArray> then
   begin
       var v1 : TValue := TValue.From<TNDArray>( TNDArray.Create(0) ) ;
       v := v1.asType<T>;
   end;

   Result := sparse_to_dense(sparse_indices,output_shape,sparse_values, v, True, '')
end;

function TTensorflow.sqrt(a: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sqrt(a, name);
end;

function TTensorflow.square(x: TFTensor; name: string): TFTensor;
begin
   Result := gen_math_ops.square(x, name);
end;

function TTensorflow.squared_difference(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.squared_difference(x, y, name);
end;

function TTensorflow.squeeze(input: TFTensor; axis: Integer; name: string; squeeze_dims: Integer): TFTensor;
begin
    Result := array_ops.squeeze(input, [ axis ], name);
end;

function TTensorflow.squeeze(input: TFTensor; axis: TArray<Integer>; name: string; squeeze_dims: Integer): TFTensor;
begin
    Result := array_ops.squeeze(input, axis, name);
end;

function TTensorflow.sub<Tx, Ty>(a: Tx; b: Ty; name: string): TFTensor;
begin
   Result := gen_math_ops.sub(a, b, name);
end;

function TTensorflow.subtract(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.sub(x, y, name);
end;

function TTensorflow.subtract<T>(x: TFTensor; y: TArray<T>; name: string): TFTensor;
begin
    Result := gen_math_ops.sub(x, Tops.convert_to_tensor(TValue.From< TArray<T> >(y),  Tdtypes.as_base_dtype(x.dtype), 'y'), name);
end;

function TTensorflow.sum(input: TFTensor; axis: Integer; keep_dims: Boolean; name: string): TFTensor;
begin
    Result := gen_math_ops._sum(input, axis, keep_dims, name);
end;

function TTensorflow.tan(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.tan(x, name);
end;

function TTensorflow.tanh(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.tanh(x, name);
end;

function TTensorflow.trainable_variables(scope: string): TArray<IVariableV1>;
begin
    var Value := variables.trainable_variables;
    Result := Value.AsType< TList<IVariableV1> >.ToArray;
end;

function TTensorflow.transpose<T1>(a: T1; perm: PAxis; name: string; conjugate: Boolean): TFTensor;
begin
    Result := array_ops.transpose(a, perm, name, conjugate);
end;

class function TTensorflow.truediv(x, y: TFTensor; name: string): TFTensor;
begin
    Result := math_ops.truediv(x, y, name);
end;

function TTensorflow.Variable<T>(data: T; name: string; dtype: TF_DataType): ResourceVariable;
begin
    var dData := TValue.from<T>(data);
    Result := ResourceVariable.Create(@dData, {trainable}true,nil, {validate_shape}true, '',name, nil, dtype, '',{aggregation}TVariableAggregation.VARIABLE_AGGREGATION_NONE, {shape}nil)
end;

function TTensorflow.Variable<T>(data: T;  trainable : Boolean; validate_shape: Boolean; use_resource: Boolean; name : string;dtype: TF_DataType; aggregation: TVariableAggregation; shape : PTFShape):ResourceVariable;
begin
    var dData := TValue.from<T>(data);
    Result := ResourceVariable.Create(@dData, trainable,nil, validate_shape, '',name, nil, dtype, '',aggregation, shape)
end;

function TTensorflow.variables_initializer(var_list: TArray<IVariableV1>; name: string): TFOperation;
begin
    Result := variables.variables_initializer(var_list, name)
end;

function TTensorflow.variable_scope(scope: VariableScope; default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean): variable_scope;
begin
    Result := Tensorflow.Variable.variable_scope.Create(scope, default_name, values, reuse, auxiliary_name_scope)
end;

function TTensorflow.variable_scope(name, default_name: string; values: TArray<TFTensor>; reuse: PBoolean; auxiliary_name_scope: Boolean): variable_scope;
begin
    Result := Tensorflow.Variable.variable_scope.Create(name, default_name, values, reuse, auxiliary_name_scope)
end;

function TTensorflow.zeros(shape: TFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := array_ops.zeros(shape, dtype, name);
end;

function TTensorflow.zeros(shape: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := array_ops.zeros(shape, dtype, name);
end;

function TTensorflow.zeros_like(nd: TNDArray; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
   Result := array_ops.zeros_like(nd, dtype, name, optimize);
end;

function TTensorflow.zeros_like(tensor: TFTensor; dtype: TF_DataType; name: string; optimize: Boolean): TFTensor;
begin
    Result := array_ops.zeros_like(tensor, dtype, name, optimize);
end;

function TTensorflow.where<Tx, Ty>(condition: TFTensor; x: Tx; y: Ty; name: string): TFTensor;
begin
    var v_x :=  TValue.From<Tx>(x);
    var v_y :=  TValue.From<Ty>(y);
    var x_obj : TObject := nil;
    var y_obj : TObject := nil;
    if v_x.IsObject then
       x_obj := v_x.AsObject;
    if v_y.IsObject then
       y_obj := v_y.AsObject;

    Result := array_ops.where(condition, x_obj, y_obj, name);
end;

function TTensorflow._clip_by_value(t, clip_value_min, clip_value_max: TFTensor; name: string): TFTensor;
begin
    Result := clip_ops.clip_by_value(t, clip_value_min, clip_value_max, name);
end;

function TTensorflow.GradientTape(persistent: Boolean = false; watch_accessed_variables: Boolean = true): TGradientTape;
begin
    var tape := FtapeSet.PushTape(persistent, watch_accessed_variables);
    tape.StartRecord;
    Result := FtapeSet;
end;

function TTensorflow.GetTapeSet: TStack<ITape>;
begin
    Result := FtapeSet.GetTapeSet;
end;
{$ENDREGION}

{$REGION 'CompatV1Api'}

{ CompatV1Api }

constructor CompatV1Api.Create;
begin
  inherited;

end;

destructor CompatV1Api.Destroy;
begin

  inherited;
end;

procedure CompatV1Api.disable_eager_execution;
begin
    tf.Context.graph_mode ;
end;

function CompatV1Api.get_variable(name: string; shape: PTFShape; dtype: TF_DataType; initializer: TObject; trainable: PBoolean; collections: TList<string>;
  use_resource: PBoolean; validate_shape: Boolean; synchronization: TVariableSynchronization; aggregation: TVariableAggregation): IVariableV1;
begin
    var scope := variable_scope.get_variable_scope();
    var store := variable_scope._get_default_variable_store();
    Result := scope.get_variable(store, name, shape, dtype, initializer, trainable, collections, use_resource, validate_shape);
end;

function CompatV1Api.global_variables_initializer: TFOperation;
begin
    var g := variables.global_variables;
    Result:= variables.variables_initializer(g.ToArray);
end;

function CompatV1Api.Session: TFSession;
begin
    Result := TFSession.Create;
    Result.as_default;
end;
{$ENDREGION}

{$REGION 'CompatApi'}
{ CompatApi }

constructor CompatApi.Create;
begin
  inherited Create;
  v1 := CompatV1Api.Create;
end;

destructor CompatApi.Destroy;
begin
  v1.Free;
  inherited;
end;
{$ENDREGION}

{$REGION 'StringsApi'}
{ StringsApi }

constructor StringsApi.Create;
begin
    ops := string_ops.Create;
end;

destructor StringsApi.Destroy;
begin
    ops.Free;
end;

function StringsApi.lower(input: TFTensor; encoding, name: string): TFTensor;
begin
    Result := ops.lower(input, encoding, name);
end;

function StringsApi.regex_replace(input: TFTensor; pattern, rewrite: string; replace_global: Boolean; name: string): TFTensor;
begin
    Result := ops.regex_replace(input, pattern, rewrite, replace_global, name);
end;

function StringsApi.substr(input: TFTensor; pos, len: Integer; name, uint: string): TFTensor;
begin
    Result := ops.substr(input, pos, len, uint, name);
end;

function StringsApi.substr(input: String; pos, len: Integer; name, uint: string): TFTensor;
begin
    Result := ops.substr(input, pos, len, uint, name);
end;

function StringsApi.string_length(input: TFTensor; name, &unit: string): TFTensor;
begin
    Result :=  ops.string_length(input, &unit);
end;

function StringsApi.format(template: string; inputs: TArray<TFTEnsor>; placeholder: string; summarize: Integer; name: string): TFTensor;
begin
    Result := ops.string_format(inputs, template, placeholder, summarize, name);
end;

function StringsApi.split(input: TFTensor; sep: AnsiChar; maxsplit: Integer; name: string): RaggedTensor;
begin
    Result := ops.string_split_v2(input, sep, maxsplit, name);
end;

function StringsApi.unicode_decode_with_offsets(input: TFTensor; input_encoding, errors: string; replacement_char: Integer; replace_control_characters: Boolean;
  name: string): Tuple<RaggedTensor, RaggedTensor>;
begin
    Result := ops.unicode_decode_with_offsets(input, input_encoding, errors, replacement_char, replace_control_characters, name)
end;

{$ENDREGION}

{$REGION 'GraphKeys'}
{ GraphKeys }

class function TGraphKeys.Create: TGraphKeys;
begin
    CONCATENATED_VARIABLES       := CONCATENATED_VARIABLES_;
    TRAINABLE_VARIABLES          := TRAINABLE_VARIABLES_;
    TRAINABLE_RESOURCE_VARIABLES := TRAINABLE_RESOURCE_VARIABLES_;
    _STREAMING_MODEL_PORTS       := _STREAMING_MODEL_PORTS_;
    LOCAL_VARIABLES              := LOCAL_VARIABLES_;
    LOSSES                       := LOSSES_;
    METRIC_VARIABLES             := METRIC_VARIABLES_;
    MOVING_AVERAGE_VARIABLES     := MOVING_AVERAGE_VARIABLES_;
    GLOBAL_VARIABLES             := GLOBAL_VARIABLES_;
    TRAIN_OP                     := TRAIN_OP_;
    GLOBAL_STEP                  := GLOBAL_STEP_;
    GLOBAL_STEP_READ_KEY         := 'global_step_read_op_cache';
    _VARIABLE_COLLECTIONS := TArray.Copy<string>(_VARIABLE_COLLECTIONS_);
    SAVEABLE_OBJECTS             := SAVEABLE_OBJECTS_;
    UPDATE_OPS                   := UPDATE_OPS_;
    SUMMARIES                    := SUMMARIES_;
    _SUMMARY_COLLECTION          := _SUMMARY_COLLECTION_;
    COND_CONTEXT                 := COND_CONTEXT_;
    WHILE_CONTEXT                := WHILE_CONTEXT_;
end;

{$ENDREGION}

{$REGION 'TRandom'}
{ TRandom }

function TRandom.normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    Result := random_ops.random_normal(shape, mean, stddev, dtype, seed, name);
end;

function TRandom.truncated_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    Result := random_ops.truncated_normal(shape, mean, stddev, dtype, seed, name);
end;

function TRandom.categorical(logits: TFTensor; num_samples: Integer; seed: pInteger; name: string; output_dtype: TF_DataType): TFTensor;
begin
    Result := random_ops.multinomial(logits, num_samples, seed, name, output_dtype);
end;

function TRandom.uniform(shape: TFShape; minval, maxval: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    if TDtypes.is_integer(dtype) then Result := random_ops.random_uniform_int(shape, Trunc(minval), Trunc(maxval), seed, name)
    else                              Result := random_ops.random_uniform(shape, minval, maxval, dtype, seed, name);
end;

function TRandom.random_uniform(shape: TFShape; minval, maxval: SIngle; dtype: TF_DataType; seed: pInteger; name: string): TFTensor;
begin
    Result := uniform(shape, minval, maxval, dtype, seed, name);
end;

function TRandom.random_shuffle(value: TFTensor; seed: Integer; name: string): TFTensor;
begin
    Result := random_ops.random_shuffle(value, seed, name);
end;

procedure TRandom.set_random_seed(seed: Integer);
begin
    if tf.executing_eagerly then tf.Context.set_global_seed(seed)
    else                         Tops.get_default_graph.seed := seed;
end;

function TRandom.stateless_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; name: string): TFTensor;
begin
   Result :=  stateless_random_ops.stateless_random_normal(shape, mean, stddev, dtype, [], name);
end;

function TRandom.multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger; name: string; output_dtype: TF_DataType): TFTensor;
begin
   Result := random_ops.multinomial(logits, num_samples, seed, name, output_dtype);
end;
{$ENDREGION}

{$REGION 'nn_internal'}
{ nn_internal }

function nn_internal.bias_add(value: TFTensor; bias: IVariableV1; data_format, name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([value, TValue.From<IVariableV1>(bias)]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'BiasAdd', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 name   := v1.ToString;
                                 Result := gen_nn_ops.bias_add(value, bias, data_format, name);
                            end );
end;

function nn_internal.dropout(x, keep_prob, noise_shape: TFTensor; seed: PInteger; name: string; rate: PSingle): TFTensor;
begin
    var keep: TFTensor := nil;
    if keep_prob <> nil then
        keep := 1.0 - TTensor(keep_prob);
    var rate_tensor : TFTensor ;
    if rate <> nil  then rate_tensor := tf.constant(rate^,'')
    else                 rate_tensor := keep;
    Result := nn_ops.dropout_v2(x, rate_tensor, noise_shape, seed,  name);
end;

function nn_internal.fused_batch_norm(x, scale, offset, mean, variance: TFTensor; epsilon: Single; data_format: string; is_training: Boolean; name: string;
  exponential_avg_factor: Single): TArray<TFTensor>;
begin
   Result := nn_impl.fused_batch_norm(x, scale, offset, mean, variance, epsilon, data_format, is_training, name, exponential_avg_factor)
end;

function nn_internal.l2_loss(t: TFTensor; name: string): TFTensor;
begin
   Result := nn_ops.l2_loss(t, name);
end;

function nn_internal.leaky_relu(features: TFTensor; alpha: Single; name: string): TFTensor;
begin
   Result := nn_ops.leaky_relu(features, alpha, name);
end;

function nn_internal.relu: IActivation;
begin
    Result :=  TensorFlow.Activation.relu.Create;
end;

function nn_internal.swish: IActivation;
begin
    Result :=  TensorFlow.Activation.swish.Create;
end;

function nn_internal.tanh: IActivation;
begin
    Result :=  TensorFlow.Activation.tanh.Create;
end;

function nn_internal.softmax: IActivation;
begin
    Result :=  TensorFlow.Activation.softmax.Create;
end;

function nn_internal.relu(features: TFTensor; name: string): TFTensor;
begin
    Result := gen_nn_ops.relu(features, name);
end;

function nn_internal.sigmoid<T>(x: T; name: string): TFTensor;
begin
   Result := math_ops.sigmoid(x, name);
end;

function nn_internal.sigmoid_cross_entropy_with_logits(labels, logits: TFTensor; name: string): TFTensor;
begin
    result := nn_impl.sigmoid_cross_entropy_with_logits(labels, logits, name);
end;

function nn_internal.softmax(logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
   Result :=  gen_nn_ops.softmax(logits, name);
end;

function nn_internal.softmax_cross_entropy_with_logits_v2(labels, logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
   Result := nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits, axis, name)
end;

function nn_internal.sparse_softmax_cross_entropy_with_logits(labels, logits: TFTensor; name: string): TFTensor;
begin
   Result := nn_ops.sparse_softmax_cross_entropy_with_logits(labels, logits, name);
end;

function nn_internal.tanh(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_nn_ops.tanh(x, name);
end;
{$ENDREGION}

{$REGION 'train_internal'}
{ train_internal }

function train_internal.AdamOptimizer(learning_rate: IVariableV1; name: string): Optimizer;
begin
    Result := TensorFlow.Training.AdamOptimizer.Create(learning_rate.AsTensor, 0.9, 0.999, 1e-8, False, TF_FLOAT,name);
end;

function train_internal.AdamOptimizer(learning_rate: Single; dtype: TF_DataType; name: string): Optimizer;
begin
    Result := TensorFlow.Training.AdamOptimizer.Create(learning_rate, 0.9, 0.999, 1e-8, False, dtype,name);
end;

function train_internal.AdamOptimizer(learning_rate, epsilon: Single; name: string): Optimizer;
begin
    Result := TensorFlow.Training.AdamOptimizer.Create(learning_rate, 0.9, 0.999, epsilon, False, TF_FLOAT,name);
end;

function train_internal.AdamOptimizer(learning_rate: TFTensor; name: string): Optimizer;
begin
    Result := TensorFlow.Training.AdamOptimizer.Create(learning_rate, 0.9, 0.999, 1e-8, False, TF_FLOAT,name);
end;

function train_internal.GradientDescentOptimizer(learning_rate: TFTensor): Optimizer;
begin
   Result := TensorFlow.Training.GradientDescentOptimizer.Create(learning_rate);
end;

function train_internal.GradientDescentOptimizer(learning_rate: Single): Optimizer;
begin
    Result := TensorFlow.Training.GradientDescentOptimizer.Create(learning_rate);
end;
{$ENDREGION}

{$REGION 'MathApi'}
{ MathApi }

function MathApi.argmax(input: TFTensor; axis: TAxis; name: string; dimension: PInteger; output_type: TF_DataType): TFTensor;
begin
    Result := gen_math_ops.arg_max(input, axis, output_type, name);
end;

function MathApi.log(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.log(x, name);
end;

function MathApi.erf(x: TFTensor; name: string): TFTensor;
begin
    Result := math_ops.erf(x, name);
end;

function MathApi.in_top_k(predictions, targets: TFTensor; k: Integer; name: string): TFTensor;
begin
    Result := nn_ops.in_top_k(predictions, targets, k, name)
end;

function MathApi.sum(x: TFTensor; axis: TAxis; name: string): TFTensor;
begin
   Result :=  math_ops.reduce_sum(x, axis, False, name);
end;

function MathApi.bincount(arr, weights, minlength, maxlength: TFTensor; dtype: TF_DataType; name: string; axis: PTFShape; binary_output: Boolean): TFTensor;
begin
    Result := math_ops.bincount(arr, weights, minlength, maxlength, dtype, name, axis, binary_output);
end;
{$ENDREGION}

{$REGION 'LinalgApi'}

{ LinalgApi }

constructor LinalgApi.Create;
begin
    ops := linalg_ops.Create;
end;

destructor LinalgApi.Destroy;
begin
  ops.Free;
  inherited;
end;

function LinalgApi.diag(diagonal: TFTensor; name: string): TFTensor;
begin
   Result := gen_array_ops.diag(diagonal, name);
end;

function LinalgApi.einsum(equation: string; inputs: TFTensors; name: string): TFTensor;
begin
    Result := math_ops.einsum(equation, inputs, name);
end;

function LinalgApi.eye(num_rows, num_columns: Integer; batch_shape: PTFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    Result := ops.eye(num_rows, num_columns, batch_shape, dtype, name);
end;

function LinalgApi.global_norm(t_list: TArray<TFTensor>; name: string): TFTensor;
begin
   Result := clip_ops.global_norm(t_list, name);
end;

function LinalgApi.inv(input: TFTensor; adjoint: Boolean; name: string): TFTensor;
begin
    Result := ops.matrix_inverse(input, adjoint, name);
end;

function LinalgApi.lstsq(matrix, rhs: TFTensor; l2_regularizer: TNDArray; fast: Boolean; name: string): TFTensor;
begin
    Result := ops.matrix_solve_ls(matrix, rhs, l2_regularizer, fast, name);
end;

function LinalgApi.matmul(a, b: TFTensor): TFTensor;
begin
   Result := math_ops.matmul(a, b);
end;

function LinalgApi.norm(tensor: TFTensor; _ord: string; axis: PAxis; name: string; keepdims: Boolean): TFTensor;
begin
    Result := ops.norm(tensor, _ord, axis, name);
end;

function LinalgApi.qr(input: TFTensor; full_matrices: Boolean; name: string): TFTensors;
begin
    Result := ops.qr(input, full_matrices, name);
end;

function LinalgApi.tensordot(x, y: TFTensor; axes: TNDArray; name: string): TFTensor;
begin
    Result := math_ops.tensordot(x, y, axes, name);
end;

function LinalgApi.tensor_diag_part(input: TFTensor; name: string): TFTensor;
begin
   Result := gen_array_ops.diag_part(input, name);
end;

{$ENDREGION}

{$REGION 'DataOps'}
{ DataOps }

constructor DataOps.Create;
begin
    FDataset := DatasetManager.Create;
end;
{$ENDREGION}

destructor DataOps.Destroy;
begin
  FDataset.Free;
  inherited;
end;

{ image_internal }

function image_internal.resize_images_v2(images, size: TFTensor; method: string; preserve_aspect_ratio, antialias: Boolean; name: string): TFTensor;
begin
    Result := image_ops_impl.resize_images_v2(images, size, method, preserve_aspect_ratio, antialias, name);
end;

function image_internal.resize_images_v2(images: TFTensor; size: TFShape; method: string; preserve_aspect_ratio, antialias: Boolean; name: string): TFTensor;
begin
    Result := image_ops_impl.resize_images_v2(images, size, method, preserve_aspect_ratio, antialias, name);
end;

initialization
begin
    tf          := TTensorflow.Create;
    tf.keras    := KerasInterface.Create;
end;

finalization
begin
     tf.keras.free;
     tf.Free;
end;

end.


