unit Keras.Data;
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
          Spring.Container.Common,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.dataset_ops,
          TensorFlow.Framework,
          TensorFlow.functions,
          TensorFlow.Variable,

          Keras.Engine,
          Keras.CommonDef;

type
  IDatasetV2    = class;
  OwnedIterator = class;

  DataHandlerArgs = class
    public
      X                  : TFTensor;
      Y                  : TFTensor;
      Dataset            : IDatasetV2;
      BatchSize          : Integer;
      StepsPerEpoch      : Integer;
      InitialEpoch       : Integer;
      Epochs             : Integer;
      Shuffle            : Boolean;
      MaxQueueSize       : Integer;
      Workers            : Integer;
      UseMultiprocessing : Boolean;
      Model              : IModel;
      StepsPerExecution  : IVariableV1;

      constructor Create;
  end;

  DataAdapterArgs = class
    public
      X                  : TFTensor;
      Y                  : TFTensor;
      Dataset            : IDatasetV2;
      BatchSize          : Integer;
      Steps              : Integer;
      Epochs             : Integer;
      Shuffle            : Boolean;
      MaxQueueSize       : Integer;
      Worker             : Integer;
      UseMultiprocessing : Boolean;
      Model              : IModel;

      constructor Create;
  end;

  /// <summary>
  /// In TF 2.0, tf.data is the preferred API for user to feed in data. In order
  /// to simplify the training code path, all the input data object will be
  /// converted to `tf.data.Dataset` if possible.
  /// </summary>
  IDataAdapter  = interface
    /// <summary>
    /// Whether the current DataAdapter could handle the input x and y.
    /// </summary>
    /// <param name="x">input features</param>
    /// <param name="y">target labels</param>
    /// <returns></returns>
    function CanHandle(x: TFTensor; y: TFTensor = nil): Boolean;
    function GetDataset: IDatasetV2;
    function GetSize: Integer;
    function Expand1d(x: TFTensor; y: TFTensor): Tuple<TFTensor, TFTensor>;
    function ShouldRecreateIterator: Boolean;
  end;

  DataAdapter  = class abstract
    protected
       Fargs   : DataAdapterArgs;
       Fdataset: IDatasetV2 ;
    public
       function CanHandle(x: TFTensor; y: TFTensor = nil): Boolean; virtual;
       function GetDataset: IDatasetV2; virtual;
       function GetSize: Integer; virtual;
       Function Expand1d(x: TFTensor; y: TFTensor): Tuple<TFTensor, TFTensor>; virtual;
       function ShouldRecreateIterator: Boolean;
  end;

  DatasetAdapter = class(DataAdapter, IDataAdapter)
    private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
    protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
    public
       {$IFNDEF AUTOREFCOUNT}
         procedure AfterConstruction; override;
         procedure BeforeDestruction; override;
         class function NewInstance: TObject; override;
         property RefCount: Integer read GetRefCount;
       {$ENDIF}

       constructor Create(_args: DataAdapterArgs);
       function GetSize: Integer; override;
  end;

  DatasetManager  = class
    public
      function from_generator<T>(generator: TArray<T>; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>): IDatasetV2;
      /// <summary>
      /// Creates a `Dataset` with a single element, comprising the given tensors.
      /// </summary>
      /// <param name="tensors"></param>
      /// <returns></returns>
      function from_tensors(nTensors: TNDArray): IDatasetV2; overload;
      function from_tensors(tTensors: TFTensors): IDatasetV2; overload;
      function from_tensor_slices(tFeatures: TFTensor; tLabels: TFTensor): IDatasetV2;overload;
      function from_tensor_slices(tTensor: TFTensor): IDatasetV2;overload;
      function from_tensor_slices(sArray: TArray<string>): IDatasetV2;overload;
      function from_tensor_slices(nArray: TNDArray) : IDatasetV2;overload;
      function range(count: Integer; output_type: TF_DataType = TF_INT64): IDatasetV2;overload;
      function range(start: Integer; stop: Integer; step: Integer = 1; output_type: TF_DataType = TF_INT64): IDatasetV2; overload;
      function zip(ds: TArray<IDatasetV2>): IDatasetV2;
  end;

   /// <summary>
  /// Handles iterating over epoch-level `tf.data.Iterator` objects.
  /// </summary>
  DataHandler = class
    private
      Fargs                      : DataHandlerArgs;
      Fadapter                   : IDataAdapter ;
      Fdataset                   : IDatasetV2;
      Finferred_steps            : Int64;
      Fcurrent_step              : Int64;
      Fstep_increment            : Int64;
      Finsufficient_data         : Boolean;
      Fsteps_per_execution       : IVariableV1;
      Fsteps_per_execution_value : Int64;

      function Get_epochs: Integer;
      function Get_initial_epoch: Integer;
    public
      constructor Create(_args : DataHandlerArgs) ;
      destructor  Destroy; override;
      function    _infer_steps(steps_per_epoch: Integer; dataset: IDatasetV2): Int64;
      function    enumerate_epochs:  TEnumerable< Tuple<Integer, OwnedIterator> >;
      function    steps: TEnumerable<Int64>;

      property Inferredsteps : Int64         read Finferred_steps;
      property DataAdapter   : IDataAdapter  read Fadapter;
      property StepIncrement : Int64         read Fstep_increment;
      property initial_epoch : Integer       read Get_initial_epoch;
      property epochs        : Integer       read Get_epochs;
  end;

  /// <summary>
  /// Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy.
  /// </summary>
  TensorLikeDataAdapter = class(DataAdapter, IDataAdapter)
    private
       Fsize               : Integer;
       Fbatch_size         : Integer;
       Fnum_samples        : Integer;
       Fnum_full_batches   : Integer;
       Fpartial_batch_size : Integer;

       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
    protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
    public
       {$IFNDEF AUTOREFCOUNT}
         procedure AfterConstruction; override;
         procedure BeforeDestruction; override;
         class function NewInstance: TObject; override;
         property RefCount: Integer read GetRefCount;
       {$ENDIF}

       constructor Create(_args: DataAdapterArgs);
       function permutation(tensor: TFTensors): TFTensors;
       /// <summary>
       /// Convert a Tensor of indices into a dataset of batched indices.
       /// </summary>
       /// <param name="tensor"></param>
       /// <returns></returns>
       function  slice_batch_indices(indices: TFTensor): IDatasetV2;
       function  slice_inputs(indices_dataset: IDatasetV2; elements: TFTensors): IDatasetV2;
       procedure _process_tensorlike;
       function GetSize: Integer; override;
  end;

  DatasetOptions = class
  end;

  /// <summary>
  /// An iterator producing tf.Tensor objects from a tf.data.Dataset.
  /// </summary>
  OwnedIterator = class(TInterfacedObject, IDisposable)
     public
       _dataset           : IDatasetV2;
       _element_spec      : TArray<TensorSpec>;
       ops                : dataset_ops;
       _deleter           : TFTensor;
       _iterator_resource : TFTensor;

       constructor Create(dataset: IDatasetV2);
       procedure   _create_iterator(dataset: IDatasetV2);
       function    next: TArray<TFTensor>;
       procedure Dispose;
  end;

  IDatasetV2 = class(TEnumerable< Tuple<TFTensor, TFTensor> >)
    private
      Fclass_name    : TArray<String>;
      Fvariant_tensor: TFTensor;
      Fstructure     : TArray<TensorSpec>;
      Foutput_shapes : TArray<TFShape> ;
      Foutput_types  : TArray<TF_DataType> ;

      function  Get_output_shapes:TArray<TFShape>; virtual; abstract;
      function  Get_output_types:TArray<TF_DataType>; virtual; abstract;
      function  Get_element_spec:TArray<TensorSpec>;  virtual; abstract;
      function  Get_Length: Integer; virtual; abstract;
    public
      /// <summary>
      /// Caches the elements in this dataset.
      /// </summary>
      /// <param name="filename"></param>
      /// <returns></returns>
      function cache(filename : string= ''): IDatasetV2; virtual; abstract;

      /// <summary>
      /// Creates a `Dataset` by concatenating the given dataset with this dataset.
      /// </summary>
      /// <param name="dataset"></param>
      /// <returns></returns>
      function concatenate(dataset: IDatasetV2): IDatasetV2; virtual; abstract;

      /// <summary>
      ///
      /// </summary>
      /// <param name="count"></param>
      /// <returns></returns>
      function &repeat(count: Integer = -1): IDatasetV2;virtual; abstract;

      /// <summary>
      /// Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
      /// </summary>
      /// <param name="num_shards">The number of shards operating in parallel</param>
      /// <param name="index">The worker index</param>
      /// <returns></returns>
      function shard(num_shards: Integer; index: Integer): IDatasetV2;virtual; abstract;

      function shuffle(buffer_size: Integer; seed : pInteger = nil; reshuffle_each_iteration: Boolean = true): IDatasetV2; virtual; abstract;

      /// <summary>
      /// Creates a `Dataset` that skips `count` elements from this dataset.
      /// </summary>
      /// <param name="count"></param>
      /// <returns></returns>
      function skip(count: Integer): IDatasetV2; virtual; abstract;

      function batch(batch_size: Integer; drop_remainder: Boolean = false): IDatasetV2;virtual; abstract;

      function prefetch(buffer_size : Integer= -1; slack_period: PInteger = nil): IDatasetV2;virtual; abstract;

      function take(count: Integer): IDatasetV2; virtual; abstract;

      function optimize(optimizations: TArray<string>; optimization_configs: TArray<string>): IDatasetV2;virtual; abstract;

      function map(map_func: TFunc<TFTensors, TFTensors>; use_inter_op_parallelism : Boolean= true; preserve_cardinality: Boolean = true; use_legacy_function: Boolean = false): IDatasetV2; overload;  virtual; abstract;
      function map(map_func: TFunc<TFTensors, TFTensors>; num_parallel_calls: Integer): IDatasetV2; overload; virtual; abstract;

      function filter(map_func: TFunc<TFTensors, TFTensors>): IDatasetV2; overload; virtual; abstract;
      function filter(map_func: TFunc<TFTensor, Boolean>): IDatasetV2; overload; virtual; abstract;

      function make_one_shot_iterator: OwnedIterator; virtual; abstract;

      function flat_map(map_func: TFunc<TFTensor, IDatasetV2>): IDatasetV2; virtual; abstract;

      function model(algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64): IDatasetV2; virtual; abstract;

      function with_options(options: DatasetOptions): IDatasetV2; virtual; abstract;

      /// <summary>
      /// Apply options, such as optimization configuration, to the dataset.
      /// </summary>
      /// <returns></returns>
      function apply_options: IDatasetV2; virtual; abstract;

      /// <summary>
      /// Returns the cardinality of `dataset`, if known.
      /// </summary>
      /// <param name="name"></param>
      /// <returns></returns>
      function cardinality(name: string = ''): TFTensor; virtual; abstract;

      property class_names    : TArray<String>      read Fclass_name       write Fclass_name;
      property variant_tensor : TFTensor            read Fvariant_tensor   write Fvariant_tensor;
      property structure      : TArray<TensorSpec>  read Fstructure        write Fstructure;
      property output_shapes  : TArray<TFShape>     read Get_output_shapes write Foutput_shapes;
      property output_types   : TArray<TF_DataType> read Get_output_types  write Foutput_types;
      property element_spec   : TArray<TensorSpec>  read Get_element_spec;
      property iLength        : Integer             read Get_Length;
  end;

  /// <summary>
  /// Abstract class representing a dataset with no inputs.
  /// </summary>
  DatasetV2 = class(IDatasetV2)
    private
      function  Get_output_shapes:TArray<TFShape>; override;
      function  Get_output_types:TArray<TF_DataType>; override;
      function  Get_element_spec:TArray<TensorSpec>;  override;
      function  Get_Length: Integer; override;
    protected
      Fops : dataset_ops;

      function DoGetEnumerator: TEnumerator< Tuple<TFTensor, TFTensor> >; override;
    public
      constructor Create;

      function cache(filename : string= ''): IDatasetV2;override ;
      function concatenate(dataset: IDatasetV2): IDatasetV2; override ;
      function &repeat(count: Integer = -1): IDatasetV2; override ;
      function shard(num_shards: Integer; index: Integer): IDatasetV2; override;
      function shuffle(buffer_size: Integer; seed : pInteger = nil; reshuffle_each_iteration: Boolean = true): IDatasetV2;  override  ;
      function skip(count: Integer): IDatasetV2; override ;
      function batch(batch_size: Integer; drop_remainder: Boolean = false): IDatasetV2;override ;
      function prefetch(buffer_size : Integer= -1; slack_period: PInteger = nil): IDatasetV2; override ;
      function take(count: Integer): IDatasetV2;  override;
      function optimize(optimizations: TArray<string>; optimization_configs: TArray<string>): IDatasetV2; override;
      function map(map_func: TFunc<TFTensors, TFTensors>; use_inter_op_parallelism : Boolean= true; preserve_cardinality: Boolean = true; use_legacy_function: Boolean = false): IDatasetV2; overload;override;
      function map(map_func: TFunc<TFTensors, TFTensors>; num_parallel_calls: Integer): IDatasetV2; overload; override;
      function filter(map_func: TFunc<TFTensors, TFTensors>): IDatasetV2; overload; override;
      function filter(map_func: TFunc<TFTensor, Boolean>): IDatasetV2; overload;   override;
      function make_one_shot_iterator: OwnedIterator; override;
      function flat_map(map_func: TFunc<TFTensor, IDatasetV2>): IDatasetV2; override;
      function model(algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64): IDatasetV2;  override;
      function with_options(options: DatasetOptions): IDatasetV2; override ;
      function apply_options: IDatasetV2;override ;
      function cardinality(name: string = ''): TFTensor;override ;

      type
        TDatasetEnumerator = class(TEnumerator< Tuple<TFTensor, TFTensor> >)
        private
          FDatasetV2     : IDatasetV2;
          FownedIterator : ownedIterator;
          FIndex         : Integer;
          FCurrent       : Tuple<TFTensor, TFTensor>;
          function GetCurrent: Tuple<TFTensor, TFTensor>;
        protected
          function DoGetCurrent: Tuple<TFTensor, TFTensor>; override;
          function DoMoveNext: Boolean; override;
        public
          constructor Create(const ADataSet: IDatasetV2);
          property Current: Tuple<TFTensor, TFTensor> read GetCurrent;
          function MoveNext: Boolean;
        end;
      function GetEnumerator: TDatasetEnumerator; reintroduce; inline;
  end;

  /// <summary>
  /// Abstract class representing a dataset with one input.
  /// </summary>
  UnaryDataset = class(DatasetV2)
    protected
      Finput_dataset : IDatasetV2;
    public
      constructor Create(input_dataset: IDatasetV2);
  end;

  /// <summary>
  /// A `Dataset` that batches contiguous elements from its input.
  /// </summary>
  BatchDataset = class(UnaryDataset)
    private
      Fbatch_size    : TFTensor;
      Fdrop_remainder: TFTensor;
    public
      constructor Create(input_dataset: IDatasetV2; batch_size: Integer; drop_remainder : Boolean= false) ;
  end;

  /// <summary>
  /// A `Dataset` that filters its input according to a predicate function.
  /// </summary>
  FilterDataset = class(UnaryDataset)
    public
      constructor Create(input_dataset: IDatasetV2; predicate_func: TFunc<TFTensor, Boolean>; predicate_func_name: string = 'func'); overload;
      constructor Create(input_dataset: IDatasetV2; predicate_func: TFunc<TFTensors, TFTensors>;  predicate_func_name: string = 'func'); overload;
  end;

  FlatMapDataset = class(UnaryDataset)
    public
      constructor Create(input_dataset: IDatasetV2; map_func : TFunc<TFTensor, IDatasetV2> ) ;
  end;

  MapDataset = class(UnaryDataset)
    private
      FMapFunc               : TFunc<TFTensors, TFTensors>;
      FUseInterOpParallelism : Boolean;
      FPreserveCardinality   : Boolean;
      FUseLegacyFunction     : Boolean;
    public
      constructor Create(const InputDataset: IDatasetV2; const MapFunc: TFunc<TFTensors, TFTensors>; UseInterOpParallelism: Boolean = True; PreserveCardinality: Boolean = False; UseLegacyFunction: Boolean = False; MapFunc_Name : string = 'func');
  end;

  //A `Dataset` that maps a function over elements in its input in parallel.
  ParallelMapDataset = class(UnaryDataset)
    public
      constructor Create(const InputDataset: IDatasetV2; const Map_Func: TFunc<TFTensors, TFTensors>; num_parallel_calls : Integer = -1; Use_Inter_Op_Parallelism: Boolean = True; Preserve_Cardinality: Boolean = False; Use_Legacy_Function: Boolean = False; Map_Func_Name : string = 'func');
  end;

  /// <summary>
  /// Represents a unary dataset with the same input and output structure.
  /// </summary>
  UnaryUnchangedStructureDataset = class(UnaryDataset)
    public
      constructor Create(input_dataset: IDatasetV2) ;
  end;

  DatasetSource = class(DatasetV2)
    protected
      Ftensors : TArray<TFTensor>;
    public
      constructor Create ;
  end;

  GeneratorDataset = class(DatasetSource)
    public
      constructor Create ;
  end;

  RangeDataset = class(DatasetSource)
    public
      constructor Create(stop: Integer; start: Integer = 0; step: Integer = 1; output_type : TF_DataType= TF_DataType.TF_INT64) ;
  end;

  /// <summary>
  /// A `Dataset` with a single element.
  /// </summary>
  TensorDataset = class(DatasetSource)
    public
      constructor Create(elements: TFTensors); overload;
      constructor Create(element: TNDArray); overload;
  end;

  TensorSliceDataset = class(DatasetSource)
    public
      constructor Create(sArray: TArray<string>); overload;
      constructor Create(nArray: TNDArray); overload;
      constructor Create(tTensor: TFTensor); overload;
      constructor Create(tTensor: TFTensor; labels: TFTensor); overload;
  end;

  /// <summary>
  /// A `Dataset` that concatenates its input with given dataset.
  /// </summary>
  ConcatenateDataset = class(DatasetV2)
    private
      Finput_dataset          : IDatasetV2;
      Fdataset_to_concatenate : IDatasetV2;
    public
      constructor Create(input_dataset: IDatasetV2; dataset_to_concatenate: IDatasetV2) ;
  end;

  CacheDataset = class(UnaryUnchangedStructureDataset)
    private
      Ffilename : TFTensor;
    public
      constructor Create(input_dataset: IDatasetV2; filename: string) ;
  end;

  /// <summary>
  /// A `Dataset` that acts as an identity, and models performance.
  /// </summary>
  ModelDataset = class(UnaryUnchangedStructureDataset)
    public
      constructor Create(input_dataset: IDatasetV2; algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64) ;
  end;

  /// <summary>
  /// A `Dataset` that acts as an identity, and applies optimizations.
  /// </summary>
  OptimizeDataset = class(UnaryUnchangedStructureDataset)
    public
      constructor Create(dataset: IDatasetV2; optimizations_enabled : TArray<String>= nil; optimizations_disabled : TArray<String>= nil; optimizations_default : TArray<String>= nil; optimization_configs : TArray<String>= nil) ;
  end;

  /// <summary>
  /// An identity `Dataset` that stores options.
  /// </summary>
  OptionsDataset = class(UnaryUnchangedStructureDataset)
    private
      Foptions : DatasetOptions;
    public
      constructor Create(input_dataset: IDatasetV2; options: DatasetOptions) ;
  end;

  /// <summary>
  /// Creates a `Dataset` that prefetches elements from this dataset.
  /// </summary>
  PrefetchDataset = class(UnaryUnchangedStructureDataset)
    public
      constructor Create(input_dataset: IDatasetV2; buffer_size : int64= -1; slack_period : Integer = 0) ;
  end;

  /// <summary>
  /// A `Dataset` that repeats its input several times.
  /// </summary>
  RepeatDataset =  class(UnaryUnchangedStructureDataset)
    public
      constructor Create(input_dataset: IDatasetV2; count : Integer = -1) ;
  end;

  /// <summary>
  /// A `Dataset` for sharding its input.
  /// </summary>
  ShardDataset = class(UnaryUnchangedStructureDataset)
    private
      FNum_shards : TFTensor;
      FIndex      : TFTensor;
    public
      constructor Create(input_dataset: IDatasetV2; num_shards: Integer; index: Integer) ;
  end;

  /// <summary>
  /// Randomly shuffles the elements of this dataset.
  /// </summary>
  ShuffleDataset = class(UnaryUnchangedStructureDataset)
    private
      Fbuffer_size : TFTensor;
      Fseed        : TFTensor;
      Fseed2       : TFTensor;
      Freshuffle_each_iteration : Boolean;
    public
      constructor Create(input_dataset: IDatasetV2; buffer_size: Int64; seed : pInteger= nil; reshuffle_each_iteration : Boolean= true) ;
  end;

  /// <summary>
  /// A `Dataset` skipping the first `count` elements from its input.
  /// </summary>
  SkipDataset = class(UnaryUnchangedStructureDataset)
    private
      FCount : TFTensor;
    public
      constructor Create(input_dataset: IDatasetV2; count: Integer) ;
  end;

  TakeDataset = class(UnaryUnchangedStructureDataset)
    private
      FCount : TFTensor;
    public
      constructor Create(input_dataset: IDatasetV2; count: Integer) ;
  end;

  ZipDataset = class(DatasetV2)
    private
      // keep all dataset references
      Finputs : TArray< IDatasetV2>;
    public
      constructor Create(ds : TArray< IDatasetV2>) ;
  end;

  ConcreteFunction_helper  = class Helper for ConcreteFunction
    public
       constructor Create(func: TFunc<TFTensor, IDatasetV2>; dtype: TF_DataType; funct_Name: String= 'func'); overload;
  end;

implementation
          uses  Winapi.Windows,
                System.Math,

                Tensorflow,
                TensorFlow.Context,
                TensorFlow.Ops,
                Tensorflow.Graph,
                Tensorflow.Utils,
                TensorFlow.Constant_op,
                Tensorflow.math_ops,
                Tensorflow.gen_array_ops,
                Tensorflow.array_ops,
                TensorFlow.random_ops,

                NumPy.NDArray;

{ ConcreteFunction_helper }

constructor ConcreteFunction_helper.Create(func: TFunc<TFTensor, IDatasetV2>; dtype: TF_DataType; funct_Name: String);
var
  func_name : string;
  input     : TFTensor;
  output    : IDatasetV2;
  opers     : TArray<TFOperation>;
begin
    func_name := funct_Name+'_'+ Tops.uid_function.ToString;

    func_graph := TFuncGraph.Create(func_name);
    func_graph.as_default;

    input  := tf.placeholder(dtype);
    output := func(input);

    OutputStructure := output.structure;

    opers := [];
    var items := func_graph.nodes_by_name.Values.ToArray;
    for var i := 0 to Length(items)-1 do
         opers := opers + [ items[i] as TFOperation ];

    func_graph.ToGraph(opers, [ input ], [ output.variant_tensor ], nil);
    func_graph.gExit;
end;

 { OwnedIterator }

constructor OwnedIterator.Create(dataset: IDatasetV2);
begin
    _create_iterator(dataset);
end;

procedure OwnedIterator.Dispose;
begin
   tf.Runner.Execute(tf.Context, 'DeleteIterator', 0, [ _iterator_resource, _deleter ], []);
end;

function OwnedIterator.next: TArray<TFTensor>;
begin
    try
      var res := ops.iterator_get_next(_iterator_resource, _dataset.output_types, _dataset.output_shapes);
      for var i := 0 to Length(res) - 1 do
      begin
          res[i].shape := _element_spec[i].shape;
      end;
      Result := res;

    except
       on e : ERangeError do
        MessageBoxA(0,'Error',PAnsiChar(AnsiString(e.Message)),mb_Ok)
    end;
end;

procedure OwnedIterator._create_iterator(dataset: IDatasetV2);
begin
    dataset  := dataset.apply_options();
    _dataset := dataset;
    _element_spec := dataset.element_spec;
    // _flat_output_types =
    var tI := ops.anonymous_iterator_v2(_dataset.output_types, _dataset.output_shapes);
    _iterator_resource := tI.Value1;
    _deleter           := tI.Value2;
    ops.make_iterator(dataset.variant_tensor, _iterator_resource);
end;

{ DataHandlerArgs }

constructor DataHandlerArgs.Create;
begin
      BatchSize          := 32;
      StepsPerEpoch      := -1;
      InitialEpoch       := 0;
      Epochs             := 1;
      Shuffle            := false;
      MaxQueueSize       := 10;
      Workers            := 1;
      UseMultiprocessing := False;
end;

{ DataAdapterArgs }

constructor DataAdapterArgs.Create;
begin
    BatchSize := 32;
end;

{ DataAdapter }

function DataAdapter.CanHandle(x, y: TFTensor): Boolean;
begin
    raise Exception.Create('Not Implemented');
end;

function DataAdapter.Expand1d(x, y: TFTensor): Tuple<TFTensor, TFTensor>;
begin
    if x.shape.ndim = 1 then
        x := array_ops.expand_dims(x, -1);
    if y.shape.ndim = 1 then
        y := array_ops.expand_dims(y, -1);
    Result := Tuple.Create(x, y);
end;

function DataAdapter.GetDataset: IDatasetV2;
begin
    Result := Fdataset;
end;

function DataAdapter.GetSize: Integer;
begin
    raise Exception.Create('Not Implemented');
end;

function DataAdapter.ShouldRecreateIterator: Boolean;
begin
    Result := True;
end;

{ DatasetAdapter }

constructor DatasetAdapter.Create(_args: DataAdapterArgs);
begin
    Fargs    := _args;
    Fdataset := _args.Dataset;
end;

function DatasetAdapter.GetSize: Integer;
begin
    Result := -1;
end;

{$IFNDEF AUTOREFCOUNT}
function DatasetAdapter.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure DatasetAdapter.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := DatasetAdapter(Obj).FRefCount;
  until AtomicCmpExchange(DatasetAdapter(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure DatasetAdapter.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure DatasetAdapter.BeforeDestruction;
begin
  if RefCount <> 0 then
    System.Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function DatasetAdapter.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  DatasetAdapter(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function DatasetAdapter.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function DatasetAdapter._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function DatasetAdapter._Release: Integer;
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

{ TensorLikeDataAdapter }

constructor TensorLikeDataAdapter.Create(_args: DataAdapterArgs);
var
  batch_size      : Integer;
  indices_dataset : IDatasetV2;
  inputs          : TFTensors;
begin
    Self.Fargs := _args;
    _process_tensorlike;
    Fnum_samples := _args.X.shape[0];

    if _args.BatchSize = -1 then  batch_size := 32
    else                          batch_size := _args.BatchSize;

    Fbatch_size := batch_size;
    Fsize := Ceil(Fnum_samples / (batch_size + 0.0));
    Fnum_full_batches   := Fnum_samples div Fbatch_size;
    Fpartial_batch_size := Fnum_samples mod Fbatch_size;

    indices_dataset := tf.data.Dataset.range(1);

    indices_dataset := indices_dataset.&repeat(_args.Epochs);
    indices_dataset := indices_dataset.map(permutation).prefetch(1);
    indices_dataset := indices_dataset.flat_map(slice_batch_indices);

    inputs := TFTensors.Create;
    if _args.X <> nil then
        inputs.Add(_args.X);
    if _args.Y <> nil then
        inputs.Add(_args.Y);

    Fdataset := slice_inputs(indices_dataset, inputs);
end;

function TensorLikeDataAdapter.permutation(tensor: TFTensors): TFTensors;
var
  indices : TFTensor;
begin
    indices := math_ops.range(Fnum_samples, nil, nil, Tdtypes.cint64);
    if Fargs.Shuffle then
        indices := random_ops.random_shuffle(indices);
    Result := TFTensors.Create(indices);
end;

function TensorLikeDataAdapter.GetSize: Integer;
begin
   Result := FSize;
end;

function TensorLikeDataAdapter.slice_batch_indices(indices: TFTensor): IDatasetV2;
begin
     var num_in_full_batch := Fnum_full_batches * Fbatch_size;

     var abegin : TArray<Integer> := [0];
     var aSize  : TArray<Integer> := [num_in_full_batch];
     var first_k_indices          := array_ops.slice(indices, abegin, aSize);

     first_k_indices  := array_ops.reshape(first_k_indices, [ Fnum_full_batches, Fbatch_size ]);
     var flat_dataset := tf.data.Dataset.from_tensor_slices(first_k_indices);

     if Fpartial_batch_size > 0 then
     begin
          var sArray          := array_ops.slice(indices, [ constant_op.constant(num_in_full_batch)], [constant_op.constant(Fpartial_batch_size)]);
          var index_remainder := tf.data.Dataset.from_tensors(TFTensors.Create(sArray));
          flat_dataset        := flat_dataset.concatenate(index_remainder);
     end;

     Result := flat_dataset;
end;

function TensorLikeDataAdapter.slice_inputs(indices_dataset: IDatasetV2; elements: TFTensors): IDatasetV2;
var
  map_fun : TFunc<TFTensors, TFTensors>;
begin
    var dataset := tf.data.Dataset.from_tensors(elements).&repeat;
    dataset     := tf.data.Dataset.zip([indices_dataset, dataset]);

    map_fun := function(inputs: TFTensors): TFTensors
               begin
                   var indices := inputs[0];
                   var results : TArray<TFTensor> := [];
                   for var i := 1 to  inputs.count -1 do
                   begin
                      var res := gen_array_ops.gather_v2(inputs[i], indices, 0) ;
                      results := results + [ res ];
                   end;
                   Result := TFTensors.Create(results);
               end;

    dataset := dataset.map(map_fun, -1);

    Result := dataset.with_options( DatasetOptions.Create );
end;

procedure TensorLikeDataAdapter._process_tensorlike;
begin

end;

{$IFNDEF AUTOREFCOUNT}
function TensorLikeDataAdapter.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TensorLikeDataAdapter.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TensorLikeDataAdapter(Obj).FRefCount;
  until AtomicCmpExchange(TensorLikeDataAdapter(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TensorLikeDataAdapter.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TensorLikeDataAdapter.BeforeDestruction;
begin
  if RefCount <> 0 then
    System.Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TensorLikeDataAdapter.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TensorLikeDataAdapter(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TensorLikeDataAdapter.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TensorLikeDataAdapter._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TensorLikeDataAdapter._Release: Integer;
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

{ DatasetV2 }

constructor DatasetV2.Create;
begin
    Fops := default(dataset_ops);
end;

function DatasetV2.Get_output_types: TArray<TF_DataType>;
begin
    Result := [];
    for var i := 0 to Length(Fstructure)-1 do
        Result := Result + [ Fstructure[i].dtype ];
end;

function DatasetV2.Get_element_spec: TArray<TensorSpec>;
begin
    Result := Fstructure;
end;

function DatasetV2.Get_Length: Integer;
begin
    var nd : NDArray := cardinality.numpy;
    Result := nd;
end;

function DatasetV2.Get_output_shapes: TArray<TFShape>;
begin
    Result := [];
    for var i := 0 to Length(Fstructure)-1 do
        Result := Result + [ Fstructure[i].shape ];
end;

function DatasetV2.cache(filename: string): IDatasetV2;
begin
    Result := CacheDataset.Create(Self, filename);
end;

function DatasetV2.concatenate(dataset: IDatasetV2): IDatasetV2;
begin
    Result := ConcatenateDataset.Create(Self, dataset);
end;

function DatasetV2.take(count: Integer): IDatasetV2;
begin
     Result := TakeDataset.Create(Self, count);
end;

function DatasetV2.batch(batch_size: Integer; drop_remainder: Boolean): IDatasetV2;
begin
     Result := BatchDataset.Create(Self, batch_size,drop_remainder);
end;

function DatasetV2.prefetch(buffer_size: Integer; slack_period: PInteger): IDatasetV2;
begin
    var nslackPer : Integer := 0;
    if slack_period <> nil then nslackPer := slack_period^;

    Result := PrefetchDataset.Create(Self, buffer_size,nslackPer);
end;

function DatasetV2.&repeat(count: Integer): IDatasetV2;
begin
    Result := RepeatDataset.Create(Self, count);
end;

function DatasetV2.shard(num_shards, index: Integer): IDatasetV2;
begin
    Result := ShardDataset.Create(Self, num_shards, index);
end;

function DatasetV2.shuffle(buffer_size: Integer; seed: pInteger; reshuffle_each_iteration: Boolean): IDatasetV2;
begin
    Result := ShuffleDataset.Create(Self, buffer_size, seed, reshuffle_each_iteration);
end;

function DatasetV2.skip(count: Integer): IDatasetV2;
begin
    Result := SkipDataset.Create(Self, count);
end;

function DatasetV2.optimize(optimizations, optimization_configs: TArray<string>): IDatasetV2;
begin
    Result := OptimizeDataset.Create(Self, optimizations, optimization_configs);
end;

function DatasetV2.map(map_func: TFunc<TFTensors, TFTensors>; use_inter_op_parallelism, preserve_cardinality, use_legacy_function: Boolean): IDatasetV2;
begin
    Result := MapDataset.Create(Self, map_func, use_inter_op_parallelism, preserve_cardinality, use_legacy_function);
end;

function DatasetV2.map(map_func: TFunc<TFTensors, TFTensors>; num_parallel_calls: Integer): IDatasetV2;
begin
    Result := ParallelMapDataset.Create(Self, map_func, num_parallel_calls, True, True);
end;

function DatasetV2.filter(map_func: TFunc<TFTensors, TFTensors>): IDatasetV2;
begin
    Result := FilterDataset.Create(Self, map_func);
end;

function DatasetV2.filter(map_func: TFunc<TFTensor, Boolean>): IDatasetV2;
begin
    Result := FilterDataset.Create(Self, map_func);
end;

function DatasetV2.make_one_shot_iterator: OwnedIterator;
begin
    if tf.Context.executing_eagerly then
    begin
        // with ops.colocate_with(self._variant_tensor)
        Result := OwnedIterator.Create(Self);
        Exit;
    end;
    raise Exception.Create('Not Implemented');
end;

function DatasetV2.flat_map(map_func: TFunc<TFTensor, IDatasetV2>): IDatasetV2;
begin
   Result := FlatMapDataset.Create(Self, map_func);
end;

function DatasetV2.model(algorithm: AutotuneAlgorithm; cpu_budget, ram_budget: Int64): IDatasetV2;
begin
  Result := ModelDataset.Create(Self, algorithm, cpu_budget, ram_budget);
end;

function DatasetV2.with_options(options: DatasetOptions): IDatasetV2;
begin
   Result := OptionsDataset.Create(Self, options);
end;

function DatasetV2.apply_options: IDatasetV2;
var
  dataset    : IDatasetV2;
  autotune   : Boolean;
  cpu_budget : Int64;
  ram_budget : Int64;
  graph_rewrites : TArray<string>;
  graph_rewrite_configs : TArray<string>;
begin
    dataset := self;
    // (1) Apply threading options

    // (2) Apply autotune options
    autotune   := true;
    cpu_budget := 0;
    ram_budget := 0;
    if autotune then
        dataset := dataset.model(AutotuneAlgorithm.HILL_CLIMB, cpu_budget, ram_budget);

    // (3) Apply graph rewrite options
    graph_rewrites        := ['map_and_batch_fusion', 'map_parallelization', 'noop_elimination', 'shuffle_and_repeat_fusion' ];
    graph_rewrite_configs := ['autotune_buffer_sizes:autotune:true', 'batch_parallelization:autotune:true', 'disable_prefetch_legacy_autotune:autotune:true',
                              'enable_gradient_descent:autotune:true', 'map_parallelization:autotune:true' ];

    dataset := OptimizeDataset.Create(dataset, [], [], graph_rewrites, graph_rewrite_configs);

    // (4) Apply stats aggregator options

    Result := dataset;
end;

function DatasetV2.cardinality(name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DatasetCardinality', name, ExecuteOpArgs.Create([variant_tensor])).First;
end;

function DatasetV2.DoGetEnumerator: TEnumerator<Tuple<TFTensor, TFTensor>>;
begin
    Result := GetEnumerator;
end;

function DatasetV2.GetEnumerator: TDatasetEnumerator;
begin
     Result := TDatasetEnumerator.Create(Self);
end;

{ DatasetV2.TDatasetEnumerator }

constructor DatasetV2.TDatasetEnumerator.Create(const ADataSet: IDatasetV2);
begin
  inherited Create;
  FIndex := -1;
  FDatasetV2 := ADataSet;
  FownedIterator := OwnedIterator.Create(FDatasetV2);
end;

function DatasetV2.TDatasetEnumerator.DoGetCurrent: Tuple<TFTensor, TFTensor>;
begin
  Result := GetCurrent;
end;

function DatasetV2.TDatasetEnumerator.DoMoveNext: Boolean;
begin
  Result := MoveNext;
end;

function DatasetV2.TDatasetEnumerator.GetCurrent: Tuple<TFTensor, TFTensor>;
begin
   Result := FCurrent;
end;

function DatasetV2.TDatasetEnumerator.MoveNext: Boolean;
var
  res : TArray<TFTensor>;
begin
    Result := False;
    try
      res := FownedIterator.next;
      if Length(res) = 1 then FCurrent := Tuple<TFTensor, TFTensor>.Create(res[0], nil)
      else                    FCurrent := Tuple<TFTensor, TFTensor>.Create(res[0], res[1]) ;

      Inc(FIndex);
      Result := True;
    except
       Exit;
    end;
end;

{ UnaryDataset }

constructor UnaryDataset.Create(input_dataset: IDatasetV2);
begin
     Finput_dataset := input_dataset;
     Fstructure     := input_dataset.structure;
end;

{ BatchDataset }

constructor BatchDataset.Create(input_dataset: IDatasetV2; batch_size: Integer; drop_remainder: Boolean);
begin
    inherited Create(input_dataset);

    Finput_dataset := input_dataset;
    Fbatch_size    := tf.convert_to_tensor(batch_size, TF_DataType.TF_INT64, 'batch_size');
    Fdrop_remainder:= tf.convert_to_tensor(drop_remainder, TF_DataType.TF_BOOL, 'drop_remainder');

    if drop_remainder then
    begin
        raise Exception.Create('Not Implemented');
    end else
    begin
        Fstructure := [];
        for var i := 0 to Length(input_dataset.element_spec)-1 do
            Fstructure := Fstructure + [ input_dataset.element_spec[i]._batch(-1) ];
    end;

    variant_tensor := Fops.batch_dataset_v2(input_dataset.variant_tensor, Fbatch_size, Fdrop_remainder, output_types, output_shapes);
end;

{ FilterDataset }

constructor FilterDataset.Create(input_dataset: IDatasetV2; predicate_func: TFunc<TFTensor, Boolean>;  predicate_func_name: string);
var
  predicate_func_update : TFunc<TFTensors, TFTensors> ;
  func                  : ConcreteFunction;
begin
    inherited Create(input_dataset);

    predicate_func_update := function(x: TFTensors):TFTensors
      begin
          var res := predicate_func(x.First);
          Result := TFTensors.Create( constant_op.constant(res) );
      end;

    func := ConcreteFunction.Create(predicate_func_name + '_' + Tops.uid_function.ToString);
    func.Enter;
    var inputs := TFTensors.Create;
    for var input in input_dataset.element_spec do
        inputs.Add( tf.placeholder(input.dtype, input.shape, 'arg'));
    var outputs := predicate_func_update(inputs);
    func.ToGraph(inputs, outputs);
    func._Exit;

    Fstructure := func.OutputStructure;

    variant_tensor := Fops.filter_dataset(input_dataset.variant_tensor, func, output_types, output_shapes);
end;

constructor FilterDataset.Create(input_dataset: IDatasetV2; predicate_func: TFunc<TFTensors, TFTensors>;  predicate_func_name: string);
var
 func : ConcreteFunction;
begin
    inherited Create(input_dataset);

    func := ConcreteFunction.Create(predicate_func_name + '_' + Tops.uid_function.ToString);
    func.Enter;
    var inputs := TFTensors.Create;
    for var input in input_dataset.element_spec do
        inputs.Add( tf.placeholder(input.dtype, input.shape, 'arg'));
    var outputs := predicate_func(inputs);
    func.ToGraph(inputs, outputs);
    func._Exit;

    Fstructure := func.OutputStructure;

    variant_tensor := Fops.filter_dataset(input_dataset.variant_tensor, func, output_types, output_shapes);
end;

{ FlatMapDataset }

constructor FlatMapDataset.Create(input_dataset: IDatasetV2; map_func: TFunc<TFTensor, IDatasetV2>);
var
 func : ConcreteFunction;
begin
    inherited Create(input_dataset);

    func := ConcreteFunction.Create(map_func, input_dataset.element_spec[0].dtype);
    Fstructure := func.OutputStructure;

    variant_tensor := Fops.flat_map_dataset(input_dataset.variant_tensor, func, output_types, output_shapes);
end;

{ MapDataset }

constructor MapDataset.Create(const InputDataset: IDatasetV2; const MapFunc: TFunc<TFTensors, TFTensors>; UseInterOpParallelism: Boolean; PreserveCardinality: Boolean; UseLegacyFunction: Boolean; MapFunc_Name : string);
var
  Func   : ConcreteFunction;
  Inputs : TFTensors;
  Outputs: TFTensors;
begin
    inherited Create(InputDataset);

    FMapFunc               := MapFunc;
    FUseInterOpParallelism := UseInterOpParallelism;
    FPreserveCardinality   := PreserveCardinality;
    FUseLegacyFunction     := UseLegacyFunction;

    Func := ConcreteFunction.Create(Format('%s_%d', [MapFunc_Name, Tops.uid_function]));
    Func.Enter;
    Inputs := TFTensors.Create;
    for var Input in InputDataset.element_spec do
      Inputs.Add(tf.placeholder(Input.dtype, Input.shape, 'arg'));
    Outputs := MapFunc(Inputs);
    Func.ToGraph(Inputs, Outputs);
    Func._Exit;

    structure := Func.OutputStructure;

    variant_tensor := Fops.map_dataset(InputDataset.variant_tensor, Func, output_types, output_shapes, UseInterOpParallelism, PreserveCardinality);
end;


{ ParallelMapDataset }

constructor ParallelMapDataset.Create(const InputDataset: IDatasetV2; const Map_Func: TFunc<TFTensors, TFTensors>; num_parallel_calls: Integer; Use_Inter_Op_Parallelism,
  Preserve_Cardinality, Use_Legacy_Function: Boolean; Map_Func_Name: string);
var
  Func   : ConcreteFunction;
  Inputs : TFTensors;
  Outputs: TFTensors;
begin
    inherited Create(InputDataset);

    Func := ConcreteFunction.Create(Format('%s_%d', [Map_Func_Name, Tops.uid_function]));
    Func.Enter;
    Inputs := TFTensors.Create;
    for var Input in InputDataset.element_spec do
      Inputs.Add(tf.placeholder(Input.dtype, Input.shape, 'arg'));
    Outputs := Map_Func(Inputs);
    Func.ToGraph(Inputs, Outputs);
    Func._Exit;

    structure := Func.OutputStructure;

    var _num_parallel_calls := tf.convert_to_tensor(num_parallel_calls, tf.int64_t, 'num_parallel_calls');

    variant_tensor := Fops.parallel_map_dataset_v2(InputDataset.variant_tensor, _num_parallel_calls, func, output_types, output_shapes, use_inter_op_parallelism,'default', preserve_cardinality);
end;

{ UnaryUnchangedStructureDataset }

constructor UnaryUnchangedStructureDataset.Create(input_dataset: IDatasetV2);
begin
    inherited Create(input_dataset);
end;

{ DatasetSource }

constructor DatasetSource.Create;
begin

end;

{ GeneratorDataset }

constructor GeneratorDataset.Create;
begin

end;

{ RangeDataset }

constructor RangeDataset.Create(stop, start, step: Integer; output_type: TF_DataType);
begin
    var start_tensor := tf.convert_to_tensor(Int64(start));
    var step_tensor  := tf.convert_to_tensor(Int64(step));
    var stop_tensor  := tf.convert_to_tensor(Int64(stop));

    var s : TArray<Integer> := [];
    Fstructure := [ TensorSpec.Create( s, output_type) ];
    variant_tensor := Fops.range_dataset(start_tensor, stop_tensor, step_tensor, output_types, output_shapes);
end;

{ TensorDataset }

constructor TensorDataset.Create(elements: TFTensors);
begin
    Ftensors := elements.ToArray;
    Fstructure := [];
    for var i := 0 to Length(Ftensors)-1 do
      Fstructure := Fstructure + [ Ftensors[i].ToTensorSpec ];

    variant_tensor := Fops.tensor_dataset(Ftensors, output_shapes);
end;

constructor TensorDataset.Create(element: TNDArray);
begin
    Ftensors := [ tf.convert_to_tensor(element) ];
    Fstructure := [];
    for var i := 0 to Length(Ftensors)-1 do
      Fstructure := Fstructure + [ Ftensors[i].ToTensorSpec ];

    variant_tensor := Fops.tensor_dataset(Ftensors, output_shapes);
end;

{ TensorSliceDataset }

constructor TensorSliceDataset.Create(sArray: TArray<string>);
begin
     var element      := tf.constant(sArray);
     Ftensors         := [ element ];
     var batched_spec := [ element.ToTensorSpec ];

     for var i := 0 to Length(batched_spec)-1 do
      Fstructure := Fstructure + [ batched_spec[i]._unbatch ];

     variant_tensor := Fops.tensor_slice_dataset(Ftensors, output_shapes);
end;

constructor TensorSliceDataset.Create(nArray: TNDArray);
begin
     var element      := tf.constant(nArray);
     Ftensors         := [ element ];
     var batched_spec := [ element.ToTensorSpec ];

     for var i := 0 to Length(batched_spec)-1 do
      Fstructure := Fstructure + [ batched_spec[i]._unbatch ];

     variant_tensor := Fops.tensor_slice_dataset(Ftensors, output_shapes);
end;

constructor TensorSliceDataset.Create(tTensor: TFTensor);
begin
     Ftensors         := [ tTensor ];
     var batched_spec := [ tTensor.ToTensorSpec ];

     for var i := 0 to Length(batched_spec)-1 do
      Fstructure := Fstructure + [ batched_spec[i]._unbatch ];

     variant_tensor := Fops.tensor_slice_dataset(Ftensors, output_shapes);
end;

constructor TensorSliceDataset.Create(tTensor, labels: TFTensor);
begin
     Ftensors         := [ tTensor, labels];
     var batched_spec := [ tTensor.ToTensorSpec, labels.ToTensorSpec ];

     for var i := 0 to Length(batched_spec)-1 do
      Fstructure := Fstructure + [ batched_spec[i]._unbatch ];

     variant_tensor := Fops.tensor_slice_dataset(Ftensors, output_shapes);
end;

{ ConcatenateDataset }

constructor ConcatenateDataset.Create(input_dataset, dataset_to_concatenate: IDatasetV2);
begin
    Finput_dataset          := input_dataset;
    Fdataset_to_concatenate := dataset_to_concatenate;
    var _structure := TList<TensorSpec>.Create;
    for var i := 0 to Length(dataset_to_concatenate.element_spec) - 1 do
    begin
        var spec  := dataset_to_concatenate.element_spec[i];
        var shape := Finput_dataset.output_shapes[i].most_specific_compatible_shape(spec.shape);
        _structure.Add(TensorSpec.Create(shape, spec.dtype));
    end;
    Fstructure := _structure.ToArray();

    variant_tensor := Fops.concatenate_dataset(input_dataset.variant_tensor, dataset_to_concatenate.variant_tensor, output_types, output_shapes);
end;

{ CacheDataset }

constructor CacheDataset.Create(input_dataset: IDatasetV2; filename: string);
begin
    inherited Create(input_dataset);

    Ffilename      := tf.convert_to_tensor(filename, TF_DataType.TF_STRING, 'filename');
    variant_tensor := Fops.cache_dataset_v2(input_dataset.variant_tensor, Ffilename, Fops.dummy_memory_cache, output_types, output_shapes);
end;

{ ModelDataset }

constructor ModelDataset.Create(input_dataset: IDatasetV2; algorithm: AutotuneAlgorithm; cpu_budget, ram_budget: Int64);
begin
    inherited Create(input_dataset);

    variant_tensor := Fops.model_dataset(input_dataset.variant_tensor, output_types, output_shapes, algorithm, cpu_budget, ram_budget);
end;

{ OptimizeDataset }

constructor OptimizeDataset.Create(dataset: IDatasetV2; optimizations_enabled, optimizations_disabled, optimizations_default, optimization_configs: TArray<String>);
begin
    inherited Create(dataset);

    var _optimizations_enabled  := tf.convert_to_tensor(optimizations_enabled,  TF_STRING, 'optimizations_enabled');
    var _optimizations_disabled := tf.convert_to_tensor(optimizations_disabled, TF_STRING, 'optimizations_disabled');
    var _optimizations_default  := tf.convert_to_tensor(optimizations_default,  TF_STRING, 'optimizations_default');

    variant_tensor := Fops.optimize_dataset_v2(Finput_dataset.variant_tensor, _optimizations_enabled, _optimizations_disabled, _optimizations_default, output_types, output_shapes, optimization_configs);
end;

{ OptionsDataset }

constructor OptionsDataset.Create(input_dataset: IDatasetV2; options: DatasetOptions);
begin
    inherited Create(input_dataset);

    Foptions := options;
    variant_tensor := input_dataset.variant_tensor;
end;

{ PrefetchDataset }

constructor PrefetchDataset.Create(input_dataset: IDatasetV2; buffer_size: int64; slack_period: Integer);
begin
    inherited Create(input_dataset);

    var buffer_size_tensor := tf.convert_to_tensor(buffer_size, TF_INT64, 'buffer_size');

    variant_tensor := Fops.prefetch_dataset(input_dataset.variant_tensor, buffer_size_tensor, input_dataset.output_types, input_dataset.output_shapes, slack_period);
end;

{ RepeatDataset }

constructor RepeatDataset.Create(input_dataset: IDatasetV2; count: Integer);
begin
    inherited Create(input_dataset);

    var count_tensor := constant_op.constant(count, TF_INT64, 'count');
    variant_tensor   := Fops.repeat_dataset(input_dataset.variant_tensor, count_tensor, input_dataset.output_types, input_dataset.output_shapes);
end;

{ ShardDataset }

constructor ShardDataset.Create(input_dataset: IDatasetV2; num_shards, index: Integer);
begin
    inherited Create(input_dataset);

    FNum_shards := tf.convert_to_tensor(num_shards, TF_INT64, 'num_shards');
    FIndex      := tf.convert_to_tensor(index, TF_INT64, 'index');

    variant_tensor := Fops.shard_dataset(input_dataset.variant_tensor, FNum_shards, FIndex, input_dataset.output_types, input_dataset.output_shapes)
end;

{ ShuffleDataset }

constructor ShuffleDataset.Create(input_dataset: IDatasetV2; buffer_size: Int64; seed: pInteger; reshuffle_each_iteration: Boolean);
begin
    inherited Create(input_dataset);

    Fbuffer_size := tf.convert_to_tensor(buffer_size, TF_INT64, 'buffer_size');

    var nSeed : TNullableInteger := nil;
    if seed <> nil then nSeed := seed^;

    var tSeed := random_seed.get_seed_tensor(nSeed);
    Fseed := tSeed.Value1;
    Fseed2:= tSeed.Value2;

    Freshuffle_each_iteration := reshuffle_each_iteration;

    var seed_generator := Fops.dummy_seed_generator;
    if tf.Context.executing_eagerly then
        variant_tensor := Fops.shuffle_dataset_v3(input_dataset.variant_tensor, Fbuffer_size, Fseed, Fseed2, seed_generator, output_types, output_shapes, Freshuffle_each_iteration)
    else
        raise Exception.Create('Not Implemented');
end;

{ SkipDataset }

constructor SkipDataset.Create(input_dataset: IDatasetV2; count: Integer);
begin
    inherited Create(input_dataset);

    Fcount := tf.convert_to_tensor(count, Tdtypes.cint64, 'count');
    variant_tensor := Fops.skip_dataset(input_dataset.variant_tensor, Fcount, output_types, output_shapes);
end;

{ TakeDataset }

constructor TakeDataset.Create(input_dataset: IDatasetV2; count: Integer);
begin
    inherited Create(input_dataset);

    Fcount := tf.convert_to_tensor(count, Tdtypes.cint64, 'count');
    variant_tensor := Fops.take_dataset(input_dataset.variant_tensor, Fcount, output_types, output_shapes);
end;

{ ZipDataset }

constructor ZipDataset.Create(ds: TArray<IDatasetV2>);
begin
    Finputs := ds;
    var input_datasets : TArray<TFTensor> := [];
    for var i := 0 to Length(ds)-1  do
       input_datasets := input_datasets + [ ds[i].variant_tensor ];

    var _structure := TList<TensorSpec>.Create;
    for var dataset in ds do
        _structure.AddRange(dataset.structure)
        ;
    Fstructure := _structure.ToArray;

    variant_tensor := Fops.zip_dataset(input_datasets, output_types, output_shapes);
end;

{ DatasetManager }

function DatasetManager.from_generator<T>(generator: TArray<T>; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>): IDatasetV2;
begin
    Result := GeneratorDataset.Create;
end;

function DatasetManager.from_tensors(tTensors: TFTensors): IDatasetV2;
begin
    Result := TensorDataset.Create(tTensors);
end;

function DatasetManager.from_tensors(nTensors: TNDArray): IDatasetV2;
begin
    Result := TensorDataset.Create(nTensors);
end;

function DatasetManager.from_tensor_slices(nArray: TNDArray): IDatasetV2;
begin
    Result := TensorSliceDataset.Create(nArray);
end;

function DatasetManager.from_tensor_slices(sArray: TArray<string>): IDatasetV2;
begin
    Result := TensorSliceDataset.Create(sArray);
end;

function DatasetManager.from_tensor_slices(tTensor: TFTensor): IDatasetV2;
begin
    Result := TensorSliceDataset.Create(tTensor);
end;

function DatasetManager.from_tensor_slices(tFeatures, tLabels: TFTensor): IDatasetV2;
begin
    Result := TensorSliceDataset.Create(tFeatures, tLabels);
end;

function DatasetManager.range(count: Integer; output_type: TF_DataType): IDatasetV2;
begin
    Result := RangeDataset.Create(count, 0, 1, output_type);
end;

function DatasetManager.range(start, stop, step: Integer; output_type: TF_DataType): IDatasetV2;
begin
    Result := RangeDataset.Create(stop, start, step, output_type);
end;

function DatasetManager.zip(ds: TArray<IDatasetV2>): IDatasetV2;
begin
    Result := ZipDataset.Create(ds);
end;

{ DataHandler }

constructor DataHandler.Create(_args: DataHandlerArgs);
begin
    Fargs := _args;
    if Fargs.StepsPerExecution = nil then
    begin
        Fsteps_per_execution       := tf.Variable(Int64(1));
        Fsteps_per_execution_value := 1;
    end else
    begin
        Fsteps_per_execution       := Fargs.StepsPerExecution;
        var numpuVal : NDArray     := Fargs.StepsPerExecution.numpy;
        Fsteps_per_execution_value := numpuVal;
    end;

    if Fargs.Dataset = nil then
    begin
        var aDataAdaptArg : DataAdapterArgs := DataAdapterArgs.Create;
        aDataAdaptArg.X                  := Fargs.X;
        aDataAdaptArg.Y                  := Fargs.Y;
        aDataAdaptArg.BatchSize          := Fargs.BatchSize;
        aDataAdaptArg.Steps              := Fargs.StepsPerEpoch;
        aDataAdaptArg.Epochs             := Fargs.Epochs - Fargs.InitialEpoch;
        aDataAdaptArg.Shuffle            := Fargs.Shuffle;
        aDataAdaptArg.MaxQueueSize       := Fargs.MaxQueueSize;
        aDataAdaptArg.Worker             := Fargs.Workers;
        aDataAdaptArg.UseMultiprocessing := Fargs.UseMultiprocessing;
        aDataAdaptArg.Model              := Fargs.Model;

        Fadapter := TensorLikeDataAdapter.Create(aDataAdaptArg);
    end else
    begin
        var aDataAdaptArg : DataAdapterArgs := DataAdapterArgs.Create;
        aDataAdaptArg.Dataset            := Fargs.Dataset;
        aDataAdaptArg.BatchSize          := Fargs.BatchSize;
        aDataAdaptArg.Steps              := Fargs.StepsPerEpoch;
        aDataAdaptArg.Epochs             := Fargs.Epochs - Fargs.InitialEpoch;
        aDataAdaptArg.Shuffle            := Fargs.Shuffle;
        aDataAdaptArg.MaxQueueSize       := Fargs.MaxQueueSize;
        aDataAdaptArg.Worker             := Fargs.Workers;
        aDataAdaptArg.UseMultiprocessing := Fargs.UseMultiprocessing;
        aDataAdaptArg.Model              := Fargs.Model;

        Fadapter := DatasetAdapter.Create(aDataAdaptArg);
    end;

    Fdataset        := Fadapter.GetDataset;
    Finferred_steps := _infer_steps(Fargs.StepsPerEpoch, Fdataset);
    Fcurrent_step   := 0;
    Fstep_increment := Fsteps_per_execution_value - 1;
    Finsufficient_data := false;
end;

destructor DataHandler.Destroy;
begin
  Fargs.Free;

  inherited Destroy;
end;

function DataHandler.Get_epochs: Integer;
begin
    Result := Fargs.Epochs;
end;

function DataHandler.Get_initial_epoch: Integer;
begin
    Result := Fargs.InitialEpoch;
end;

function DataHandler._infer_steps(steps_per_epoch: Integer; dataset: IDatasetV2): Int64;
begin
    if steps_per_epoch > -1 then
        Exit( steps_per_epoch );

    var adapter_steps := Fadapter.GetSize;
    if adapter_steps > -1 then
        Exit ( adapter_steps );

    var size               := dataset.cardinality;
    var numpuVal : NDArray := size.numpy;
    Result                 := numpuVal;
end;

function DataHandler.enumerate_epochs: TEnumerable<Tuple<Integer, OwnedIterator>>;
var
  epoch        : Integer;
  data_iterator: OwnedIterator;
begin
    Result := TList<Tuple<Integer, OwnedIterator>>.Create;
    for epoch := initial_epoch to epochs - 1 do
    begin
      if Finsufficient_data then
        Break;

      data_iterator := OwnedIterator.Create(Fdataset);
      TList<Tuple<Integer, OwnedIterator>>(Result).Add(Tuple<Integer, OwnedIterator>.Create(epoch, data_iterator));
    end;
    // _adapter.on_epoch_end()
end;

function DataHandler.steps: TEnumerable<Int64>;
var
  can_run_full_execution : Boolean;
  steps_remaining        : Int64;
begin
    Result := TList<Int64>.Create;

    Fcurrent_step := 0;
    while Fcurrent_step < Finferred_steps do
    begin
        if Finsufficient_data then
          Break;

        can_run_full_execution := (Fsteps_per_execution_value = 1) or (Finferred_steps < 0) or (Finferred_steps - Fcurrent_step >= Fsteps_per_execution_value);
        if can_run_full_execution then
        begin
            Fstep_increment := Fsteps_per_execution_value - 1;
            TList<Int64>(Result).Add(Fcurrent_step);
            Fcurrent_step := Fcurrent_step + Fsteps_per_execution_value;
        end else
        begin
            steps_remaining := Finferred_steps - Fcurrent_step;

            if      Fsteps_per_execution is RefVariable          then  (Fsteps_per_execution as RefVariable).assign(steps_remaining)
            else if Fsteps_per_execution is BaseResourceVariable then  (Fsteps_per_execution as BaseResourceVariable).assign(steps_remaining)
            else    raise Exception.Create(' DataHandler.steps Error!');

            Fstep_increment := steps_remaining - 1;

            TList<Int64>(Result).Add(Fcurrent_step);
            Fcurrent_step := Fcurrent_step + steps_remaining;

            if      Fsteps_per_execution is RefVariable          then  (Fsteps_per_execution as RefVariable).assign(Fsteps_per_execution_value)
            else if Fsteps_per_execution is BaseResourceVariable then  (Fsteps_per_execution as BaseResourceVariable).assign(Fsteps_per_execution_value)
            else    raise Exception.Create(' DataHandler.steps Error!');
        end;
    end;
end;

end.

