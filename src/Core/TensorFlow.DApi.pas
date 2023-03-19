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

unit TensorFlow.DApi;
{$POINTERMATH ON}
{$WARN DUPLICATE_CTOR_DTOR OFF}
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
uses
  System.SysUtils,
  System.Types,
  Winapi.Windows,
  System.Rtti,
  system.TypInfo,
  System.Generics.Collections,

  Spring,
  Spring.Collections,
  Spring.Collections.Base,
  Spring.Collections.Enumerable,

  TF4D.Core.CApi,
  TF4D.Core.CApiEager,
  TensorFlow.DApiBase,
  TensorFlow.Slice,
  Tensorflow.Interfaces,

  TensorFlow.Proto;

type
TFGraph         = class;
TFOperation     = class;
TFTensor        = class;
TNDArray        = class;
TFOperationDesc = class;
TFSession       = class;
//
WhileContext    = class;

{$REGION 'ParsedSliceArgs'}
ParsedSliceArgs = record
   aBegin         : TArray<Integer>;
   tPackedBegin   : TFTEnsor;
   aEnd           : TArray<Integer>;
   tPackedEnd     : TFTEnsor;
   aStrides       : TArray<Integer>;
   tPackedStrides : TFTEnsor;
   iBeginMask     : Integer;
   iEndMask       : Integer;
   iShrinkAxisMask: Integer;
   iNewAxisMask   : Integer;
   iEllipsisMask  : Integer;
end;
{$ENDREGION}

{$REGION 'TFShape'}
/// <summary>
/// Represents the shape of a tensor
/// </summary>
/// <remarks>
/// <para>
/// The shapes can be created by calling the constructor with the number of dimensions
/// in the shape.   The null value is used to specify that the shape is unknown,
/// an empty array is used to create a scalar, and other values are used to specify
/// the number of dimensions.
/// </para>
/// <para>
/// For the Unknown case, you can use <see cref="P:TensorFlor.TFShape.Unknown"/>, for
/// scalars, you can use the <see cref="P:TensorFlor.TFShape.Scalar"/> shape.
/// </para>
/// <para>
/// To create a 2-element vector, use:
/// new TFShape (2)
/// </para>
/// <para>
/// To create a 2x3 matrix, use:
/// new TFShape (2, 3)
/// </para>
/// <para>
/// To create a shape with an unknown number of elements, you can pass the value
/// -1.  This is typically used to indicate the shape of tensors that represent a
/// variable-sized batch of values.
/// </para>
/// <para>
/// To create a matrix with 4 columns and an unknown number of rows:
/// var batch = new TFShape (-1, 4)
/// </para>
/// </remarks>
PTFShape = ^TFShape;
TFShape = record
 private
   FHandle : Pointer;
   FaDims  : TArray<TF_int64_t>;
   FSize   : Int64;
   Fndim   : Integer;
   FStrides: TArray<TF_int64_t>;
   function  GetIsFullDef: Boolean;
   function  GetSize: Int64;
   function  GetnDim: Integer;
   function  GetRank: Integer;
   function  GetStrid: TArray<Int64>;
   function  GetItem(idx: Integer): Int64;overload;
   function  GetItem(sSlice: Slice): TFShape;overload;
   procedure SetItem(idx: Integer; const Value: Int64);
 public
   constructor Create(dims: TArray<TF_int64_t>); overload;
   constructor Create(dims: TArray<Integer>); overload;
   constructor Create(proto: TTensorShapeProto); overload;
   class function Scalar: TFShape; static;
   class function null: TFShape; static;
   // Da slicehelper a shape
   class function AlignWithShape(shape: TFShape; slices: TArray<Slice>):TArray<Slice>; static;
   class function GetShape(shape1: TFShape; slices: TArray<Slice>):TFShape;static;
   class function GetStrides(shape: TFShape):TArray<Int64>;static;
   class function GetOffset(shape: TFShape; indices: TArray<Integer>): Int64; static;

   class operator Implicit(a: PTFShape): TFShape;
   class operator Explicit(const Value: TFShape): TValue;
   class operator Implicit(a : TFShape): TFTensor;
   class operator Implicit(a : TFShape): TArray<Integer>;
   class operator Implicit(a : TArray<Integer>): TFShape;
   class operator Implicit(a : TArray<Int64>): TFShape;
   class operator Implicit(a : Integer): TFShape;
   class operator Equal(a,b : TFShape): Boolean;
   class operator NotEqual(a,b : TFShape): Boolean;

   /// <summary>
   /// Returns an unknown Shape, optionally with a known rank.
   /// </summary>
   /// <param name="rank"></param>
   /// <returns></returns>
   function unknown_shape(rank_: Integer = -1): TFShape;
   /// <summary>
   /// Returns a `Shape` combining the information in `self` and `other`.
   /// </summary>
   /// <param name="other"></param>
   /// <returns></returns>
   function merge_with(other: TFShape): TFShape;
   function with_rank(rank_: Integer): TFShape;
   function with_rank_at_least(rank_: Integer): TFShape;
   function ToString: string;
   function is_compatible_with(shape2:TFShape):Boolean;
   function most_specific_compatible_shape(other: TFShape): TFShape;
   function as_int_list : TArray<Integer>;
   function IsScalar: Boolean;
   function IsNull: Boolean;
   function IsNil: Boolean;
   function Equals(target: TValue): Boolean; reintroduce;
   /// <summary>
   ///     Returns the concatenation of the dimension in `self` and `other`.
   /// </summary>
   /// <param name="other"></param>
   /// <returns></returns>
   function concatenate(other: TFShape): TFShape; overload;
   function concatenate(other: TArray<Int64>): TFShape; overload;
   //
   property ndim: Integer read GetnDim;
   /// <summary>
   ///     Returns the size this shape represents.
   /// </summary>
   property Size: Int64 read GetSize;
   property rank: Integer read GetRank;
   property Strides : TArray<Int64> read GetStrid;

   property Dims               : TArray<TF_int64_t> read FaDims write FaDims;
   property IsFullyDefined     : Boolean            read GetIsFullDef;
   property Item[idx: Integer ]: Int64              read GetItem write SetItem; default;
   property Item[sSlice: Slice]: TFShape            read GetItem ; default;

end;
{$ENDREGION}

{$REGION 'TParameter'}
TParameter = record
  sNome : string;
  vValue: TValue;
end;
{$ENDREGION}

{$REGION 'ITensorOrOperation'}
/// <summary>
/// in order to limit function return value
/// is Tensor or Operation
/// </summary>
ITensorOrOperation = class abstract(TFDisposable)
  private
    FName   : string;
    FDtype  : TF_DataType;
    FDevice : string;
    FOp     : TFOperation;
    FOutputs: TArray<TFTensor>;
    /// <summary>
	  /// Returns the data type for the tensor.
	  /// </summary>
	  /// <value>The type of the tensor.</value>
    function GetType: TF_DataType; virtual;abstract;
    function GetName: string; virtual; abstract;
    function GetDevice: string; virtual; abstract;
  public
     //function numpy:TNDArray; virtual ; abstract;

     property Dtype  : TF_DataType  read GetType;
     property Name   : string       read GetName ;
     property Device : string       read GetDevice;
     property Op     : TFOperation  read FOp;
     property Outputs: TArray<TensorFlow.DApi.TFTensor>  read FOutputs;
end;
{$ENDREGION}

{$REGION 'TControlFlowContext'}
/// <summary>
/// The base class for control flow context.
///
/// The usage pattern is a sequence of(Enter, Exit) followed by a final
/// ExitResult.
///
/// We maintain the following state for control flow contexts during graph
/// construction:
/// 1. graph has _control_flow_context: the current context used to
/// construct new nodes.Changed by ctxt.Enter() and ctxt.Exit()
/// 2. op has _control_flow_context: the context to which the op belongs.
/// Set at the time the op is created.Immutable.
/// 3. A ControlFlowContext has _outer_context: the context in which this
/// context is created.Set at the time a context is created.Immutable.
/// 4. A ControlFlowContext has _context_stack.
/// Pushed and popped by ctxt.Enter() and ctxt.Exit()
/// </summary>
TControlFlowContext = class(TInterfacedObject,ITensorFlowObject)
   private

   protected
     Fvalues : TLIst<string>;

     Fname : string;
     /// <summary>
     /// The predicate tensor in this branch
     /// </summary>
     Fpivot : TFTensor;
     /// <summary>
     /// The boolean tensor for the cond predicate
     /// </summary>
     Fpred : TFTensor;
     /// <summary>
     /// 0 or 1 representing this branch
     /// </summa
     Fbranch : Integer;
     Fcontext_stack  : TStack<TControlFlowContext>;
     Fouter_context  : TControlFlowContext;
     /// <summary>
     /// The keys are the names of tensors referenced by but external to this
     /// context. Each value is the Tensor that should be used by this context to
     /// access the key value (e.g. a switch output guarding a cond input value).
     /// </summary>
     Fexternal_values: TDictionary<string, ITensorOrOperation>;
     /// <summary>
     /// Add `op` to the current context.
     /// </summary>
     procedure _AddOpInternal(op: TFOperation); virtual;
     /// <summary>
     /// Remove any external control dependency on this op.
     /// </summary>
     /// <param name="op"></param>
     function _RemoveExternalControlEdges(op: TFOperation): Tuple<TArray<TFOperation>, TArray<TFOperation>> ; Virtual;
     /// <summary>
     /// Initializes values and external_values from `ValuesDef` protocol buffer.
     /// </summary>
     /// <param name="values_def"></param>
     /// <param name="import_scope"></param>
     procedure _init_values_from_proto(values_def: TValuesDef; import_scope: string = '');
     function OpInContext(op: TFOperation): Boolean;
   public
     constructor Create;
     destructor Destroy;  override;
     procedure _Enter_;
     procedure _Exit_;
     /// <summary>
     /// Enter this control flow context.
     /// </summary>
     procedure Enter_; virtual;
     /// <summary>
     /// Exit this control flow context.
     /// </summary>
     procedure Exit_ ; virtual;
     function IsWhileContext: Boolean; virtual;
     function IsCondContext: Boolean; virtual;
     /// <summary>
     /// Return the while context containing this context
     /// </summary>
     function GetWhileContext : WhileContext ;virtual;
     /// <summary>
     /// Notifies a scope about an operator added to an inner scope.
     /// </summary>
     /// <param name="op"></param>
     procedure AddInnerOp(op: TFOperation); virtual;
     /// <summary>
     /// Add `op` to the current context.
     /// </summary>
     procedure AddOp(op: TFOperation) ;virtual;
     /// <summary>
     /// Add `val` to the current context and its outer context recursively.
     /// </summary>
     /// <param name="val"></param>
     /// <returns></returns>
     function AddValue(val: TFTensor): TFTensor; virtual;
     procedure __init__(values_def: TValuesDef; import_scope: string = '') ; overload;
     procedure __init__ ;  overload;
     procedure ExitResult(res: TArray<TFTensor>);
     /// <summary>
     /// Returns true if `maybe_containing_ctxt` is or contains `ctxt`.
     /// </summary>
     function IsContainingContext(ctxt: TControlFlowContext; maybe_containing_ctxt: TControlFlowContext): Boolean;

     property values : TLIst<string> read  Fvalues write Fvalues;
     property pivot  : TFTensor read Fpivot;
     property pred   : TFTensor read Fpred;
     property branch : Integer  read Fbranch;
     property Nome   : String   read FName;
     property outer_context  : TControlFlowContext read Fouter_context ;
end;
{$ENDREGION}

{$REGION 'GradLoopState'}
/// <summary>
/// The state used for constructing the gradient graph for a while loop.
/// </summary>
GradLoopState  = class
   private
      Fgrad_context : WhileContext ;
      //    # The loop counter added by AddBackpropLoopCounter. It is the value
      //    # of the loop counter for the current iteration.
      //    self._grad_index = None
      //    # A sync op for backprop.
      //    self._grad_sync = None
      //    # Information needed by backprop.
      //Fhistory_map : THashtable;
      Fswitch_map : TDictionary<TFOperation, TFTensor>;
      /// <summary>
      /// The while loop context for forward.
      /// </summary>
      Fforward_context : WhileContext;
      /// <summary>
      /// The grad loop state for the outer while loop.
      /// </summary>
      Fouter_grad_state  : GradLoopState;
      Fforward_index     : TFTensor;
      Fgrad_index        : TFTensor;
      Fforward_loop_exits: TArray<TFTensor>;
      Fdeferred_exits    : TList<TFTensor>;
      Funused_exits      : TList<TFTensor>;
      Fgrad_sync         : TFOperation;
   public
      /// <summary>
      /// The number of exits we expect to see but haven't.
      /// </summary>
      pending_exits_count : Integer;

      constructor Create(forward_ctxt: WhileContext; outer_grad_state_: GradLoopState);
      destructor  Destroy; override;

      property grad_context    : WhileContext read Fgrad_context;
      //property Hashtable history_map => _history_map;
      property switch_map      : TDictionary<TFOperation, TFTensor> read Fswitch_map;
      property forward_context : WhileContext                       read Fforward_context;
      property outer_grad_state: GradLoopState                      read Fouter_grad_state;
      property forward_index   : TFTensor                           read Fforward_index;
      /// <summary>
      /// The list of exits of the forward loop.
      /// </summary>
      property forward_loop_exits : TArray<TFTensor> read Fforward_loop_exits;
      property deferred_exits     : TList<TFTensor>  read Fdeferred_exits;
      property unused_exits       : TList<TFTensor>  read Funused_exits;
end;
{$ENDREGION}

{$REGION 'LoopVar'}
LoopVar<TItem> = class(TInterfacedObject, ICanBeFlattened, IPackable<LoopVar<TItem>>)
   public
     Counter : TFTensor;
     Item    : TItem;
     Function Flatten: TArray<TValue>;
     function Pack(sequences: TArray<TValue>): LoopVar<TItem> ;

     constructor Create(_counter: TFTensor; _item: TItem);
end;
{$ENDREGION}

{$REGION 'WhileContext'}
/// <summary>
/// Creates a `WhileContext`.
/// </summary>
WhileContext = class(TControlFlowContext)
   private
      Fback_prop          : Boolean ;
      Fgrad_state         : GradLoopState;
      Fmaximum_iterations : TFTensor;
      Fparallel_iterations: Integer;
      Fswap_memory        : Boolean;
      Fpivot_for_pred     : TFTensor;
      Fpivot_for_body     : TFTensor;
      Floop_exits         : TList<TFTensor>;
      Floop_enters        : TList<TFTensor>;
      Fgraph              : TFGraph;
      procedure _init_from_args(maximum_iterations: TFTensor; parallel_iterations: Integer; back_prop: Boolean; swap_memory: Boolean; name: string);
      procedure _init_from_proto(context_def: TWhileContextDef; import_scope: string = '');
      function _convert_tensorarray_to_flow(tensor_or_tensor_array: TValue): TFTensor;
      function _get_shape_invariant(_var: TFTensor; shape: TArray<Integer>= []): TFShape;
      /// <summary>
      /// Add the loop termination condition and body to the graph.
      /// </summary>
      /// <typeparam name="TItem"></typeparam>
      /// <param name="pred"></param>
      /// <param name="body"></param>
      /// <param name="original_loop_vars"></param>
      /// <param name="loop_vars"></param>
      /// <param name="shape_invariants"></param>
      /// <returns></returns>
      function _BuildLoop<TItem>(pred : TFunc<LoopVar<TItem>, TFTensor>; body: TFunc<LoopVar<TItem>, LoopVar<TItem>>; original_loop_vars: LoopVar<TItem>; loop_vars: TArray<TFTensor>; shape_invariants: TArray<TFShape>): Tuple<LoopVar<TItem>, TArray<TFTensor>>;

   public
      /// <summary>
      /// Add the loop termination condition and body to the graph.
      /// </summary>
      function BuildLoop<TItem>(pred: TFunc<LoopVar<TItem>, TFTensor>; body: TFunc<LoopVar<TItem>, LoopVar<TItem>>; loop_vars: LoopVar<TItem>; shape_invariants: TArray<TFShape>; return_same_structure: Boolean): LoopVar<TItem>;

      constructor Create(maximum_iterations : TFTensor= nil;
                         parallel_iterations: Integer = 10;
                         back_prop          : Boolean = true;
                         swap_memory        : Boolean = false;
                         name               : string = 'while_context';
                         grad_state         : GradLoopState= nil;
                         context_def        : TWhileContextDef= nil;
                         import_scope       : string= '');

      property   maximum_iterations : TFTensor        read Fmaximum_iterations;
      property   parallel_iterations: Integer         read Fparallel_iterations;
      property   swap_memory        : Boolean         read Fswap_memory;
      property   loop_exits         : TList<TFTensor> read Floop_exits;
      property   loop_enters        : TList<TFTensor> read Floop_enters;
      property   grad_state         : GradLoopState   read Fgrad_state;
      property   back_prop          : Boolean         read Fback_prop;
end;
{$ENDREGION}

{$REGION 'TControlDependenciesController'}
TControlDependenciesController = class(TInterfacedObject,ITensorFlowObject)
   private
      FGraph             : TFGraph;
      Fcontrol_inputs_val: TList<ITensorOrOperation> ;
      Fseen_nodes        : TList<ITensorOrOperation> ;
      Fold_stack         : TList<TControlDependenciesController> ;
      Fnew_stack         : Boolean;
      Fold_control_flow_context: TControlFlowContext;
      function GetCtxInput: TArray<ITensorOrOperation>;
   public
      procedure _Enter_;
      procedure _Exit_;

      /// <summary>
      /// Create a new `_ControlDependenciesController`.
      ///
      /// A `_ControlDependenciesController` is the context manager for
      /// `with tf.control_dependencies()` blocks.These normally nest,
      /// as described in the documentation for `control_dependencies()`.
      ///
      /// The `control_inputs` argument list control dependencies that must be
      /// added to the current set of control dependencies.Because of
      /// uniquification the set can be empty even if the caller passed a list of
      /// ops.The special value `None` indicates that we want to start a new
      /// empty set of control dependencies instead of extending the current set.
      ///
      /// In that case we also clear the current control flow context, which is an
      /// additional mechanism to add control dependencies.
      /// </summary>
      /// <param name="graph">The graph that this controller is managing.</param>
      /// <param name="control_inputs">List of ops to use as control inputs in addition
      /// to the current control dependencies.None to indicate that
      /// the dependencies should be cleared.
      /// </param>
      constructor Create(graph: TFGraph; control_inputs: TList<ITensorOrOperation>);
      destructor Destroy; override;
      function  op_in_group(op: ITensorOrOperation): Boolean;
      procedure add_op(op: ITensorOrOperation);

      property  control_inputs: TArray<ITensorOrOperation> read GetCtxInput;

end;
{$ENDREGION}

{$REGION 'FeedItem'}
  /// <summary>
  /// Feed dictionary item
  /// </summary>
  PFeedItem = ^FeedItem;
  FeedItem = record
    private
      FKey  : TValue;
      FValue : TValue;
    public
       constructor Create(k: TValue; v: TValue);
       procedure   Deconstruct(var k:TValue; var v: TValue);

       class Operator Implicit(t: Tuple<TValue,TValue>):FeedItem;

       property Key   : TValue Read FKey;
       property Value : TValue Read FValue;
  end;
{$ENDREGION}

{$REGION 'FetchMapper'}
  FetchMapper = class
    private
      Funique_fetches : TList<ITensorOrOperation>;
      Fvalue_indices  : TList< TArray<Integer> >;
    public
      constructor Create;
      destructor  Destroy; override;
      class function  for_fetch(fetch: TValue; graph : TFGraph= nil): FetchMapper;
      function  build_results(values: TList<TNDArray>): TArray<TNDArray>; virtual;
      function  unique_fetches: TList<ITensorOrOperation>; virtual ;
  end;
{$ENDREGION}

{$REGION 'ListFetchMapper'}
  ListFetchMapper = class(FetchMapper)
    private
      Fmappers : TArray<FetchMapper> ;
    public
      constructor Create(fetches: TArray<TValue>);
      destructor  Destroy; override;
      function  _uniquify_fetches(fetch_mappers: TArray<FetchMapper>): Tuple< TList<ITensorOrOperation>, TList< TArray<Integer>> >;
  end;
{$ENDREGION}

{$REGION 'ElementFetchMapper'}
  /// <summary>
  /// Fetch mapper for singleton tensors and ops.
  /// </summary>
  ElementFetchMapper = class(FetchMapper)
    private
       Fcontraction_fn : TFunc< TList<TNDArray>,TValue>;
    public
       constructor Create(fetches: TArray<TValue>; contraction_fn: TFunc< TList<TNDArray>,TValue>; graph : TFGraph= nil);
       destructor Destroy; override;
       /// <summary>
       /// Build results matching the original fetch shape.
       /// </summary>
       /// <param name="values"></param>
       /// <returns></returns>
       function  build_results(values: TList<TNDArray>): TArray<TNDArray>; override;
  end;
{$ENDREGION}

{$REGION 'FetchHandler'}
  /// <summary>
  /// Handler for structured fetches.
  /// </summary>
  FetchHandler  = class
    private
       Ffetch_mapper : FetchMapper;
       Ffetches      : TList<TFTensor>;
       Fops          : TList<Boolean>;
       Ffinal_fetches: TList<TFTensor>;
       Ftargets      : TList<TValue>;

       procedure _assert_fetchable(graph: TFGraph; op: TFOperation);
    public
       constructor Create(graph: TFGraph; fetches: TValue; feeds : TDictionary<TValue, TValue>= nil; feed_handles : TProc = nil) ;
       destructor Destroy;override;
       function  build_results(tensor_values: TArray<TNDArray>): TArray<TNDArray>;
       function fetches: TList<TFTensor>;
       function targets: TList<TValue>;
  end;
{$ENDREGION}

{$REGION 'TFSessionOptions'}
/// <summary>
/// The session options object holds configuration options that you want to use during your session, like the TensorFlow target or the configuration.
/// </summary>
TFSessionOptions = class(TFDisposable)
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFSessionOptions"/> class.
		/// </summary>
   constructor Create;overload;
   constructor Create(target : TF_TString= ''; config: TConfigProto = nil); overload;
		/// <summary>
		/// Delete the instance.
		/// </summary>
   destructor  Destroy; override;
   /// <summary>
   /// Sets the target in options.
   /// </summary>
   /// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
   /// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
	 procedure SetTarget(target: TF_TString);
	 /// <summary>
	 /// Sets the configuration information for the session.
	 /// </summary>
	 /// <param name="protoData">protocol buffer for the tensorflow.ConfigProto message.</param>
   /// <remarks>
	 /// The configuration option is a Protocol Buffer representing the tensorflow.ConfigProto
	 /// </remarks>
	 procedure SetConfig(protoData: TConfigProto);
end;
{$ENDREGION}

{$REGION 'TFSession'}
/// <summary>
/// Drives the execution of a graph
/// </summary>
/// <remarks>
/// <para>
/// This creates a new context to execute a TFGraph.   You can use the
/// constructor to create an empty session, or you can load an existing
/// model using the <see cref="FromSavedModel"/> static method in this class.
/// </para>
/// <para>
/// To execute operations with the graph, call the <see cref="GetRunner"/>  method
/// which returns an object that you can use to build the operation by providing
/// the inputs, requesting the operations that you want to execute and the desired outputs.
/// </para>
/// <para>
/// The <see cref="GetRunner"/> method is a high-level helper function that wraps a
/// call to the <see cref="Run"/> method which just takes too many parameters that must
/// be kept in sync.
/// </para>
/// </remarks>
TFSession = class(TFDisposable)
 private
   FGraph:  TFGraph;
   /// <summary>
   /// If a tensor handle that is fed to a device incompatible placeholder,
   /// we move the tensor to the right device, generate a new tensor handle,
   /// and update feed_dict to use the new handle.
   /// </summary>
   function _update_with_movers: TList<TValue>;
   /// <summary>
   /// Runs a step based on the given fetches and feeds.
   /// </summary>
   /// <param name="target_list">A list of operations to be run, but not fetched.</param>
   /// <param name="fetch_list"></param>
   /// <param name="feed_dict"></param>
   /// <returns>
   /// A list of numpy ndarrays, corresponding to the elements of
   /// `fetch_list`.  If the ith element of `fetch_list` contains the
   /// name of an operation, the first Tensor output of that operation
   /// will be returned for that element.
   /// </returns>
   function _do_run(const target_list: TList<TFOperation>; const fetch_list: TList<TFTensor>; const feed_dict: TDictionary<TValue, TValue>): TArray<TNDArray>;
   function _run(fetches: TValue; feed_dict : TArray<FeedItem>= nil): TArray<TNDArray>;
   function _call_tf_sessionrun(feed_dict: TArray< TPair<TF_Output, TFTensor> >; fetch_list: TArray<TF_Output>; target_list: TList<TFOperation>): TArray<TNDArray>;
   procedure _extend_graph;
   function  fetchValue(output: Pointer): TNDArray;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   // BaseSession
   constructor Create(hnd   : Pointer;         graph  : TFGraph); overload;
   constructor Create(target: TF_TString = ''; g      : TFGraph = nil;    config : TConfigProto= nil; status: TFStatus = nil); overload;
   // Session
   constructor Create(g     : TFGraph;         config : TConfigProto= nil; status: TFStatus = nil); overload;
   destructor  Destroy; override;

   procedure   run(op:      TFOperation;        feed_dict: TArray<FeedItem>= []); overload;
   function    run(fetche:  TFTensor;           feed_dict: TArray<FeedItem>): TNDArray ;overload;
   function    run(fetche:  ITensorOrOperation; feed_dict: TArray<FeedItem>=[]): TNDArray;overload;
   function    run(fetches: TArray<TValue>;     feed_dict: TArray<FeedItem>= []): TArray<TNDArray>;overload;
   function    run(fetches: TArray<TValue>)                                 : TArray<TNDArray>;overload;
   function    run(fetches: TFTensor)                                       : TNDArray;overload;

   function    as_default:  TFSession;
   function    eval(tensor: TFTensor): TFTensor;
   //
   property    Graph: TFGraph read FGraph write FGraph;
end;
{$ENDREGION}

{$REGION 'TFTensor'}
/// <summary>
/// TFTensor holds a multi-dimensional array of elements of a single data type.
/// </summary>
/// <remarks>
/// <para>
/// You can create tensors with the various constructors in this class, or using
/// the implicit conversions from various data types into a TFTensor.
///</para>
/// <para>
/// The implicit conversions for basic types produce tensors of one dimesion with
/// a single element, while the implicit conversion from an array, expects a multi-dimensional
/// array that is converted into a tensor of the right dimensions.
/// </para>
/// <para>
/// The special "String" tensor data type that you will find in TensorFlow documentation
/// really represents a byte array.   You can create string tensors by using the <see cref="M:TensorFlow.TFTensor.CreateString"/>
/// method that takes a byte array buffer as input.
/// </para>
/// </remarks>
TFTensor = class(ITensorOrOperation,ITensorOrTensorArray)
 private
   FEagerTensorHandle    : PTFE_TensorHandle;
   FlDeallocator_called  : Boolean;
   FIsCreatedInGraphMode : Boolean;
   FIsList               : Boolean;
   /// <summary>
   ///     The Operation that produces this tensor as an output.
   /// </summary>
   Ftf_output            : Nullable<TF_Output>;
   FValue_index          : Integer;
   FOverride_dtype       : TF_DataType;
   /// <summary>
   ///     The Graph that contains this tensor.
   /// </summary>
   FGraph                : TFGraph;
   FRank                 : Integer;
   FShape                : TFShape;
   /// <summary>
   ///     Used for keep other pointer when do implicit operating
   /// </summary>
   FTag : TValue;
   /// <summary>
   /// Keras History: (Layer, (node_index, tensor_index))
   /// </summary>
   FKerasHistory : TObject;

   function GetByteSize: UInt64;
   function GetDataTypeSize: UInt64;
   function GetSize: UInt64;
   function GetData: Pointer;
   function GetRank: Integer;
   // inherited from ITensorOrOperation
   function  GetName: string; override;
   function GetType: TF_DataType; override;
   function GetDevice: string; override;

   function GetTensorDataPointer: Pointer;
   procedure UpdateTensoData;
   procedure InitTensor(shape: TFShape; dtype: TF_DataType);overload;
   class function GetBestDType<Tx, Ty>(x: Tx; y: Ty) : TF_DataType;
   class function TF_NewTensor(shape: TFShape; dtype: TF_DataType; data: Pointer):PTF_Tensor; overload;
   class function TF_NewTensor(data: TArray<Byte>; shape: TFShape; dtype: TF_DataType):PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<T>; shape: TFShape): PTF_Tensor; overload;
   function GetDims: TArray<Int64>;
   function GetItem(slices: TArray<Slice>): TFTensor;overload;
   function GetItem(idx: Integer): TFTensor;overload;
   function GetItem(slices: TArray<string>): TFTensor;overload;
   function GetItem(_slice: string): TFTensor;overload;
   function GetItem(start:TFTensor; stop:TFTensor = nil; step: TFTensor = nil): TFTensor;overload;
   function GetnDim: Integer;
 protected
   FId : Int64;

   procedure NativeDispose(hnd: Pointer); override;
   function  GetNDArray(ddtype: TF_DataType): TNDArray;
   function  GetShape: TFShape;  virtual;
   procedure Setshape(const Value: TFShape);  virtual;
 public
   // Scalar
   constructor Create(hnd: Pointer);   overload;
   constructor Create(const value: Boolean);  overload; virtual;
   constructor Create(const value: Byte);     overload; virtual;
   constructor Create(const value: Int8);     overload; virtual;
   constructor Create(const value: UInt16);   overload; virtual;
   constructor Create(const value: Int16);    overload; virtual;
   constructor Create(const value: Cardinal); overload; virtual;
   constructor Create(const value: Integer);  overload; virtual;
   constructor Create(const value: UInt64);   overload; virtual;
   constructor Create(const value: Int64);    overload; virtual;
   constructor Create(const value: Single);   overload; virtual;
   constructor Create(const value: Double);   overload; virtual;
   constructor Create(const value: TF_TString); overload; virtual;
   /// <summary>
   ///     Create a Tensor object from an existing TF handle
   /// </summary>
   /// <param name="handle">Handle to a <see cref="Tensor"/> object.</param>
   constructor Create(hhandle: Pointer; clone: Boolean);overload; virtual;

   /// <summary>
   /// Create a new Tensor from the given unmanaged memory pointer (which must be allocated, fixed or pinned by the caller)
   /// Note: the caller is responsible for freeing the memory. Calling Dispose on this object will dispose the TensorFlow tensor
   /// but not the memory itself!
   /// </summary>
   /// <param name="data_ptr">Pointer to unmanaged, fixed or pinned memory which the caller owns</param>
   /// <param name="shape">Tensor shape</param>
   /// <param name="dType">TF data type</param>
   constructor Create(data  : Pointer;shape: TFShape; dtype: TF_DataType); overload;
   constructor Create(shape : TFShape; dtype:TF_DataType); overload;
   constructor Create(bytes : TArray<Byte>;shape : TFShape; dtype:TF_DataType); overload;
   constructor Create(op    : TFOperation; value_index: Integer; dtype:TF_DataType); overload;
   // Array of T
   class function Create<T>(aArray: TArray<T>;                        shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<T>>;                shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<TArray<T>>>;        shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: PTFShape=nil):TFTensor; overload;

   class function  InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType):PTF_Tensor; overload;
   class function  InitTensor<T>(aArray: TArray<T>;                        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function  InitTensor<T>(aArray: TArray<TArray<T>>;                shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function  InitTensor<T>(aArray: TArray<TArray<TArray<T>>>;        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function  InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   //
   class function    StringTensor(srcArray: TArray<TArray<Byte>>; shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<TF_TString>;   shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<string>;       shape: TFShape):PTF_Tensor;overload;
   //
   function StringBytes:TArray< TArray<Byte> >; overload;
   function StringBytes(index: Integer): TArray<byte>; overload;
   function StringData(index: integer): AnsiString; overload;
   function StringData: TArray<TF_TString>; overload;
   //
   function ToString: string;override;
   function Equals(y: Integer): TFTensor; reintroduce;
   function NotEquals(y: Integer): TFTensor;
   //
   destructor  Destroy; override;

   /// <summary>
   ///     Returns a list of Operations that consume this tensor.
   /// </summary>
   /// <returns></returns>
   function consumers: TArray<TFOperation>;
   function _shape_tuple: TArray<Integer>;
   /// <summary>
   ///     Updates the shape of this tensor.
   /// </summary>
   procedure set_shape(shape: TFTensor);
   /// <summary>
   ///
   /// </summary>
   /// <typeparam name="T"></typeparam>
   /// <returns></returns>
   function ToArray<T>:TArray<T>;
   /// <summary>
   /// Copy of the contents of this Tensor into a NumPy array or scalar.
   /// </summary>
   /// <returns>
   /// A NumPy array of the same shape and dtype or a NumPy scalar, if this
   /// Tensor has rank 0.
   /// </returns>
   function numpy: TNDArray;
   function _as_tf_output : TF_Output;
   class function BinaryOpWrapper<Tx, Ty>(name: string; x: Tx; y: Ty) : TFTensor;

   /// <summary>
   /// Copies the memory of current buffer onto newly allocated array.
   /// </summary>
   /// <returns></returns>
   function BufferToArray: TArray<Byte>;
   /// <summary>
   ///     Evaluates this tensor in a `Session`.
   /// </summary>
   /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
   /// <param name="session">The `Session` to be used to evaluate this tensor.</param>
   /// <returns>A <see cref="NumPy"/> array corresponding to the value of this tensor.</returns>
   function  eval(session : TFSession; feed_dict : TArray<FeedItem>= nil) : TNDArray;
   function  _slice(start: Integer): TFTensor; overload;
   function  _slice(_sl: slice): TFTensor; overload;

   property  KerasHistory  : TObject        read FKerasHistory write FKerasHistory;
   property  Tag           : TValue         read FTag write FTag;
   property  bytesize      : UInt64         read GetByteSize;
   property  dtypesize     : UInt64         read GetDataTypeSize;
   property  size          : UInt64         read GetSize;
   property  buffer        : Pointer        read GetData;
   property  value_index   : Integer        read FValue_index;
   property  override_dtype: TF_DataType    read FOverride_dtype;
   property  id            : Int64          read FId;
   property  graph         : TFGraph        read FGraph;
   property  Shape         : TFShape        read GetShape write Setshape;
   property  isList        : Boolean        read FIsList write FIsList;
   /// <summary>
   /// number of dimensions <br></br>
   /// -1 Unknown  <br></br>
   /// 0	Scalar (magnitude only) <br></br>
   /// 1	Vector (magnitude and direction) <br></br>
   /// 2	Matrix (table of numbers) <br></br>
   /// 3	3-Tensor (cube of numbers) <br></br>
   /// n	n-Tensor (you get the idea)
   /// </summary>
   /// <remarks>https://www.tensorflow.org/api_docs/python/tf/rank</remarks>
   property  rank             :    Integer          read GetRank;
   property  dims             :    TArray<Int64>    read GetDims;
   property  ndim             :    Integer          read GetnDim;
   property  DeallocatorCalled:    Boolean          read FlDeallocator_called;
   property  isCreatedInGraphMode: Boolean          read FIsCreatedInGraphMode;
   property  TensorDataPointer:    Pointer          read GetTensorDataPointer;
   property  EagerTensorHandle: PTFE_TensorHandle   read FEagerTensorHandle write FEagerTensorHandle;
   property  Item[slices: TArray<Slice> ]: TFTensor read GetItem ; default;
   property  Item[idx: Integer ]         : TFTensor read GetItem ; default;
   property  Item[slices: TArray<string>]: TFTensor read GetItem ; default;
   property  Item[slice: string]: TFTensor          read GetItem ; default;
   property  Item[start:TFTensor; stop:TFTensor = nil; step: TFTensor = nil]: TFTensor  read GetItem ; default;
end;
{$ENDREGION}

{$REGION 'TNDArray'}
TNDArray = class(TFTensor, IEnumerable )
   private
     procedure NewEagerTensorHandle;
     function  GetItem(indices: TArray<Integer>): TNDArray; overload;
     function  GetItem(indices: TArray<Slice>): TNDArray; overload;
     function  GetItem(indice: Integer): TNDArray; overload;
     procedure SetItem(indices: TArray<Integer>; const Value: TNDArray); overload;
     function  GetItem(mask: TNDArray): TNDArray; overload;
     procedure SetItem(mask: TNDArray; const Value: TNDArray); overload;
     procedure SetItem(indices: TArray<Slice>; const Value: TNDArray); overload;
     procedure SetItem(indice: Integer; const Value: TNDArray); overload;
     function  GetData(slices: TArray<Slice>): TNDArray; overload;
     function  GetData(indices: TArray<Integer>; axis: Integer = 0): TNDArray; overload;
     function  GetData(Mask : TNDArray): TNDArray; overload;
     procedure SetData(Mask : TNDArray; value: TNDArray); overload;
     procedure MaskData(Mask : TNDArray; value: TNDArray);
     procedure SetData(slices: TArray<Slice>; aArray: TNDArray); overload;
     procedure SetData(src: TNDArray; slices: TArray<Slice>; indices: TArray<Integer>; currentNDim: integer);overload;
     function  GetScalar(offset : UInt64 = 0): TNDArray;
     function  GetArrayData(newshape: TFShape; indices: TArray<Integer>): TNDArray;
     function  GetDataPointer: Pointer;
    private
     type
     TEnumerator_NDArray = class(TEnumeratorBase<TNDArray>)
      private
        fIndex: Integer;
        fSource : TNDArray;
      protected
        function GetCurrent: TNDArray; override;
      public
        constructor Create(source : TNDArray);
        destructor Destroy; override;
        function MoveNext: Boolean; override;
      end;

   protected
    {$REGION 'Property Accessors'}
    function GetCount: Integer; virtual;
    function GetElementType: PTypeInfo; virtual;
    function GetIsEmpty: Boolean; virtual;
    {$ENDREGION}
   public
     function AsObject: TObject;
     function GetEnumerator: IEnumerator;

     constructor Create(const value: Boolean); overload; override;
     constructor Create(const value: Byte);    overload; override;
     constructor Create(const value: Word);    overload; override;
     constructor Create(const value: Integer); overload; override;
     constructor Create(const value: Int64);   overload; override;
     constructor Create(const value: Single);  overload; override;
     constructor Create(const value: Double);  overload; override;
     //
     constructor Create(bytes: TArray<TF_TString>;shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Boolean>;                        shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Boolean>>;                shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Boolean>>>;        shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>;shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Byte>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Byte>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Byte>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int8>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int8>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int8>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int8>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int16>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int16>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int16>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<UInt16>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<UInt16>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<UInt16>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<UInt16>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int32>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int32>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int32>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<UInt32>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<UInt32>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<UInt32>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<UInt32>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Int64>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Int64>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Int64>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<UInt64>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<UInt64>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<UInt64>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Single>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Single>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Single>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(bytes: TArray<Double>;                         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<Double>>;                 shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<Double>>>;         shape: PTFShape= nil);overload;
     constructor Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: PTFShape= nil);overload;
     //
     constructor Create(value: TValue;shape: PTFShape= nil); overload;
     //
     constructor Create(shape: TFShape;                      dtype: TF_DataType); overload;
     constructor Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType); overload;
     constructor Create(value: TArray<Int64>;                dtype: TF_DataType); overload;
     constructor Create(address: Pointer;   shape: PTFShape; dtype: TF_DataType); overload;
     constructor Create(tensor: TFTensor; clone : Boolean= false); overload;

     destructor  Destroy; override;

     class function Scalar<T>(value: T):TNDArray;

     function To_Array: TArray<TNDArray>;
     function reshape(newshape: TFShape): TNDArray;
     function astype(dtype: TF_DataType): TNDArray;
     function ToByteArray: TArray<Byte>;
     function Equals(y: TNDArray): Boolean; reintroduce;

     property Item[indices: Integer ]         : TNDArray read GetItem write SetItem; default;
     property Item[indices: TArray<Integer> ] : TNDArray read GetItem write SetItem; default;
     property Item[slices: TArray<Slice> ]    : TNDArray read GetItem write SetItem; default;
     property Item[mask: TNDArray ]           : TNDArray read GetItem write SetItem; default;
     property data : Pointer read GetDataPointer;


  end;
{$ENDREGION}

{$REGION 'Tensors.Ragged'}
  /// <summary>
  /// Abstract base class for Tensor-like objects that are composed from Tensors.
  /// </summary>
  CompositeTensor = class abstract
  end;

  Dimension = record
    private
      Fvalue : Int64;
    public
      constructor Create(value_   : Int64);
      function merge_with(other: Dimension):Dimension;
      function ToString: string;

      class operator implicit(value_:Int64): Dimension;
      class operator implicit(value_:Dimension): Int64;

      property value  : Int64 read Fvalue;
  end;

  /// <summary>
  /// Represents a sparse tensor.
  /// </summary>
  TSparseTensor = record
    private

    public
      indices    : TFTensor;
      values     : TFTensor;
      dense_shape: TFTensor;
      constructor Create(indices_: TFTensor; values_: TFTensor;dense_shape_ : TFTensor);overload;
      constructor Create(indices_: TArray< TArray<Int64> >; values_: TValue;dense_shape_ : TArray<Int64>); overload;
      procedure _init;
  end;

  tensor_shape = class
     public
      class function dimension_at_index(shape: TFShape; index: Integer): Dimension;
      class function dimension_value(dimension: Dimension) : Integer;
  end;

  /// <summary>
  /// Partitioning of a sequence of values into contiguous subsequences ("rows").
  /// </summary>
  RowPartition = class(CompositeTensor)
    private
       Frow_splits  : TFTensor;
       Frow_lengths : TFTensor;
       Fvalue_rowids: TFTensor;
       Fnrows       : TFTensor;

       function GetStaticRow: Integer;
       function GetSt_uni_row_l: Integer;
    public
      constructor Create(row_splits: TFTensor; row_lengths : TFTensor = nil; value_rowids: TFTensor = nil; nrows: TFTensor = nil; uniform_row_length : TFTensor= nil);
      /// <summary>
      /// Creates a `RowPartition` with rows partitioned by `value_rowids`.
      /// </summary>
      /// <param name="value_rowids"></param>
      /// <param name="nrows"></param>
      /// <param name="validate"></param>
      /// <param name="preferred_dtype"></param>
      /// <returns></returns>
      class function from_value_rowids(value_rowids: TFTensor; nrows: TFTensor = nil; validate : Boolean = true; preferred_dtype: TF_DataType = DtInvalid): RowPartition;
      class function from_row_splits(row_splits: TFTensor; validate : Boolean= true; preferred_dtype : TF_DataType = DtInvalid): RowPartition;

      property row_splits   : TFTensor read Frow_splits;
      property static_nrows : Integer  read GetStaticRow;
      property static_uniform_row_length : Integer read GetSt_uni_row_l;

  end;

  /// <summary>
  /// Represents a ragged tensor.
  /// </summary>
  RaggedTensor = class (CompositeTensor)
     private
        Fvalues        : TFTensor;
        Frow_partition : RowPartition;
        Frow_splits    : TFTensor;

        function GetDtype: TF_DataType;
        function GetRow_splits: TFTensor;
        function getShape: TFShape;
        function GetItem(index: Integer): TFTensor; overload;
        function GetItem(slices: TArray<Slice>): RaggedTensor; overload;
        function _ragged_getitem(row_key: Integer) : TFTensor;
        function _ragged_getitem_inner_dimensions(input: RaggedTensor; slices: TArray<Slice>): RaggedTensor;
        function getNest_row_splits: TArray<TFTensor>;

     public
        constructor Create(values: TFTensor; internal : Boolean = true; row_partition : RowPartition= nil);
        class function from_row_partition(values: TFTensor; row_partition: RowPartition; validate: Boolean = true): RaggedTensor;
        /// <summary>
        /// Creates a `RaggedTensor` with rows partitioned by `value_rowids`.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="value_rowids"></param>
        /// <param name="nrows"></param>
        /// <param name="name"></param>
        /// <param name="validate"></param>
        /// <returns></returns>
        class function from_value_rowids(values: TFTensor; value_rowids: TFTensor; nrows: TFTensor = nil; name: string = ''; validate: Boolean = true): RaggedTensor;
        class function from_row_splits(values: TFTensor; row_splits: TFTensor; name : string = ''; validate: Boolean = true) : RaggedTensor;
        function _to_variant(batched_input : Boolean= false; name : string= '') : TFTensor;
        class function FromTensor(t: TFTensor): RaggedTensor;
        function ToTensor: TFTensor;

        property flat_values                : TFTensor         read Fvalues;
        property row_splits                 : TFTensor         read GetRow_splits;
        property dtype                      : TF_DataType      read GetDtype;
        property shape                      : TFShape          read getShape;
        property nested_row_splits          : TArray<TFTensor> read getNest_row_splits;
        property Item[i: Integer]           : TFTensor         read GetItem; default;
        property Item[slices: TArray<Slice>]: RaggedTensor     read GetItem; default;
  end;

  /// <summary>
  /// TensorArray is designed to hide an underlying implementation object
  /// and as such accesses many of that object's hidden fields.
  ///
  /// "Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.
  /// This class is meant to be used with dynamic iteration primitives such as
  /// `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  /// "flow" control flow dependencies.
  /// </summary>
  TTensorArray = class( TInterfacedObject, ITensorOrTensorArray)
    protected
      Fdtype : TF_DataType;
      Fhandle: TFTensor;
      Fflow  : TFTensor;
      Finfer_shape : Boolean;
      Fcolocate_with_first_write_call : Boolean;
    public
      function unstack(value: TFTensor; name: string = ''): TTensorArray; virtual; abstract;
      function stack(name: string = ''): TFTensor; virtual; abstract;
      function gather(indices: TFTensor; name: string = ''): TFTensor; virtual; abstract;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; virtual; abstract;

      function read<T>(index: T; name: string = ''): TFTensor;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; overload;

      property  dtype                           : TF_DataType read Fdtype;
      property  handle                          : TFTensor    read FHandle;
      property  flow                            : TFTensor    read Fflow;
      property  infer_shape                     : Boolean     read Finfer_shape;
      property  colocate_with_first_write_call  : Boolean     read Fcolocate_with_first_write_call;
  end;

  BodyItem = class(TInterfacedObject, ICanBeFlattened, IPackable<BodyItem>, IFromMergeVars<BodyItem>)
    private
      FI      : TFTensor;
      FAccs_ta: TArray<TTensorArray>;
    public
      constructor Create; overload;
      constructor Create(v_I : TFTensor; v_accs_ta: TArray<TTensorArray>); overload;

      function Flatten: TArray<TValue>;
      function FromMergeVars(mergeVars: TArray<ITensorOrTensorArray>): BodyItem ;
      function Pack(sequences: TArray<TValue>): BodyItem ;

      property I       : TFTensor             read FI;
      property Accs_ta : TArray<TTensorArray> read FAccs_ta;
  end;

  TGraphTensorArray = class(TTensorArray)
    private
      Fdynamic_size     : Boolean;
      Felement_shape    : TList<TFShape>;
      Fcolocate_with    : TList<TFTensor>;
      Fclear_after_read : Boolean;
     // Ftensor_array     : TList<TFTensor>;

      function size(name: string = ''): TFTensor;
    protected

    public
      constructor Create(_dtype: TF_DataType; size: TFTensor; dynamic_size: Boolean = false;
                         clear_after_read : Boolean= true; tensor_array_name : string = ''; _handle : TFTensor= nil;
                         _flow : TFTensor = nil; _infer_shape: Boolean = true; _element_shape : PTFShape= nil;
                         _colocate_with_first_write_call : Boolean= true; _name: string = '');
      function  scatter(indices: TFTensor; value: TFTensor; name: string = ''): TTensorArray;
      procedure _merge_element_shape(shape: TFShape);
      procedure _maybe_colocate_with(value: TFTensor);
      function read<T>(index: T; name: string = ''): TFTensor; reintroduce;
      function unstack(value: TFTensor; name: string = ''): TTensorArray; override;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; override;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; reintroduce; overload;
      function stack(name: string = ''): TFTensor; override;
      function gather(indices: TFTensor; name: string = ''): TFTensor; override;
  end;
{$ENDREGION}

{$REGION 'TEagerTensor'}
TEagerTensor = class(TFTensor)
    protected
       procedure NewEagerTensorHandle(h:Pointer);
       procedure NativeDispose(hnd: Pointer); override;
    private
       m_Device : string;
       procedure Resolve;
       function GetDeviceName: string;
    protected
       function GetShape: TFShape;  override;
       procedure Setshape(const Value: TFShape);  override;

    public
       constructor Create(h:Pointer);overload;
       constructor Create(h: Pointer;NewEagerTensor: Boolean);overload; override;
       constructor Create(shape: TFShape;dType: TF_DataType);overload;
       constructor Create(value: TValue; shape: PTFShape); overload;

       constructor Create(bytes: TArray<TF_TString>;shape: TFShape);overload;

       constructor Create(bytes: TArray<Boolean>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Boolean>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Boolean>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Byte>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Byte>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Byte>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int16>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int16>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int16>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int32>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int32>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int32>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int64>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int64>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int64>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<UInt64>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<UInt64>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<UInt64>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Single>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Single>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Single>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Single>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Double>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Double>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Double>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Double>>>>;shape: TFShape; dtype: TF_DataType);overload;
       destructor Destroy; override;

       class function GetRank(eTensor:TEagerTensor): Integer;
       class function GetDims(eTensor:TEagerTensor): TArray<Integer>;

       /// <summary>
       /// _create_substitute_placeholder
       /// </summary>
       /// <returns></returns>
       function  AsPlaceholder(name: string = ''): TFTensor;
       function  AsConstant(name: string = ''): TFTensor;
       procedure copy_handle_data(target_t: TFTensor);
       function  ToString: string;override;

       property Device : string read GetDeviceName;

end;

TEagerTensorArray = class(TTensorArray)
    private
      Fdynamic_size     : Boolean;
      Felement_shape    : TFShape;
      Fcolocate_with    : TList<TFTensor>;
      Fclear_after_read : Boolean;
      Ftensor_array     : TList<TFTensor>;

      function size(name: string = ''): TFTensor;
    protected

    public
      constructor Create(_dtype: TF_DataType; size: TFTensor; dynamic_size: Boolean = false;
                         clear_after_read : Boolean= true; tensor_array_name : string = ''; _handle : TFTensor= nil;
                         _flow : TFTensor = nil; _infer_shape: Boolean = true; _element_shape : PTFShape= nil;
                         _colocate_with_first_write_call : Boolean= true; _name: string = '');
      function  scatter(indices: TFTensor; value: TFTensor; name: string = ''): TTensorArray;
      procedure _merge_element_shape(shape: TFShape);
      procedure _maybe_colocate_with(value: TFTensor);
      function read<T>(index: T; name: string = ''): TFTensor; reintroduce;
      function unstack(value: TFTensor; name: string = ''): TTensorArray; override;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; override;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; reintroduce; overload;
      function stack(name: string = ''): TFTensor; override;
      function gather(indices: TFTensor; name: string = ''): TFTensor; override;
end;
{$ENDREGION}

{$REGION 'TFTensors'}
TFTensors = class (TObjectList<TFTensor>)
 private
   FItems  : TList<TFTensor>;
   Fdtype  : TF_DataType;
   Fshape  : TFShape;
   FRank   : Integer;
   Fgraph  : TFGraph;
   FIsList : Boolean;
   FiLength: Integer;
   FIsCreatedInGraphMode : Boolean;

   function  GetTensor(idx: Integer): TFTensor;
   procedure SetTensor(idx: Integer; const Value: TFTensor);
    function GetShape: TFShape;
 protected

 public
   destructor  Destroy; override;
   constructor Create; overload;
   constructor Create(const tensors: array of TFTensor); overload;
   constructor Create(tensor: TFTensor); overload;
   constructor Create(t: Tuple<TFTensor,TFTensor>); overload;
   constructor Create(tensors : TList<TFTensor>); overload;
   function    Add(const tensor : TFTensor): Integer;
   procedure   AddRange(tensors : TArray<TFTensor>);
   procedure   Insert(idx : Integer ; const tensor : TFTensor);
   function    ToTensor(tensors: TFTensors): TFTensor;
   function    ToArray: TArray<TFTensor>; reintroduce;
   function    ToString: string; override;
   procedure   Deconstruct(var a: TFTensor; var b : TFTensor);


   property Item[idx: Integer] : TFTensor read GetTensor write SetTensor; default;
   property dtype   : TF_DataType read Fdtype ;
   property shape   : TFShape     read GetShape ;
   property rank    : Integer     read FRank ;
   property graph   : TFGraph     read Fgraph ;
   property IsList  : Boolean     read FIsList write FIsList ;
   property iLength : Integer     read FiLength ;
   property IsCreatedInGraphMode  : Boolean     read FIsCreatedInGraphMode ;
end;
{$ENDREGION}

{$REGION 'TFOperationDesc'}
/// <summary>
/// Low-level TensorFlow operation builder
/// </summary>
/// <remarks>
/// <para>This is the low-level API that is used to create operations by manually specificying all
/// the parameters of an operation (inputs, outputs, attribute descriptions) that can then
/// be attached into a graph.
/// </para>
/// <para>
/// Generally, you will instead be using the methods surfaced in <see cref="T:TensorFlow.TFGraph"/>
/// that surfaces a C# high-level API that has already been bound to the built-in TensorFlow
/// nodes.
/// </para>
/// <para>
/// You create instances bound to a graph, add inputs, attributes and so on, and when you are done
/// you can call the <see cref="FinishOperation"/> method that will turn this TFOperationDesc
/// into a <see cref="T:TensorFlow.TFOperation"/>.
/// </para>
/// </remarks>
TFOperationDesc = class(TFDisposable)
 private
   FsOpType   : TF_TString;
   FsOperName : TF_TString;
   FGraph     : TFGraph;
    function GetHandleOperation: Pointer;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(graph: TFGraph; opType, operName: TF_TString);
   destructor  Destroy; override;
   procedure SetAttrType(attr_name: AnsiString; value: TF_DataType);
   procedure SetAttrShape(attr_name: AnsiString; dims: TArray<Int64>);
   /// <summary>
	 /// Adds the specified input to the operation
	 /// </summary>
	 /// <returns>The input.</returns>
	 /// <param name="input">Input.</param>
	 procedure AddInput(input: TF_Output);
   /// <summary>
   /// Adds a series of inputs to the operation.
   /// </summary>
   /// <param name="inputs">Inputs, this is a params array for your convenience.</param>
   procedure AddInputs (inputs: TArray<TF_Output>);
   /// <summary>
   /// Turns the operation description into an actual operation in the graph.
   /// </summary>
   /// <returns>The operation on success, or null on error.</returns>
   /// <param name="status">Optional status, on failure the operation is not added to the graph.
   /// If you pass null (the default), this operation throws on error conditions.</param>
   function FinishOperation (status: TFStatus = Nil): TFOperation;

   property op : Pointer read GetHandleOperation;
end;
{$ENDREGION}

{$REGION 'TInputList'}
TInputList = class (TList<TFTensor>)
   private
      Finputs : TArray<TFTensor>;

    function GetItem(index: Integer): TFTensor; reintroduce;
    function GetLength: Integer;
   public

     constructor Create(const tensors: array of TFTensor);
     destructor  Destroy; override;

     property Len    : Integer                 read GetLength;
     property inputs : TArray<TFTensor>        read Finputs;
     property Item[index: Integer]: TFTensor read GetItem; default;
end;
{$ENDREGION}

{$REGION 'TFOperation'}
/// <summary>
/// Represents a computation node in the graph.  Tensorflow operations are attached to a <see cref="T:Tensorflow.TFGraph"/>.
/// </summary>
/// <remarks>
/// TFOperations are usually created by  invoking one of the methods in
/// <see cref="T:Tensorflow.TFGraph"/>, but they can also be constructed
/// manually using the low-level <see cref="T:Tensorflow.TFOperationDesc"/> API.
/// </remarks>
TFOperation = class(ITensorOrOperation)
 private
    FTipo                 : string;
    // Pointer to the graph, to keep it from collecting if there are TFOperations alive.
    FGraph                : TFGraph;
    FInputs_val           : TInputList;
    Fcontrol_flow_context : TControlFlowContext;
    Fid_value             : Integer;
    Fis_stateful          : Boolean;
    Fcontrol_inputs       : TArray<TFOperation>;

    function OutputType(index: Integer): TF_DataType;
    function GeNumOutputs: Integer;
    function GetOutput: TFTensor;
    // Inherited from ITensorOrOperation
    function GetDevice: string; override;
    function GetType: TF_DataType; override;
    function GetName: string;override;
    function GeNumInputs: Integer;
    function GetInputList: TInputList; virtual;
    function GetNodeDef: TNodeDef;
    function GetipoOp: string;
    procedure _assert_same_graph(tensor: TFTensor);
    function GetCtrlInputs: TArray<TFOperation>;
    function GetControlInputs: TArray<TFOperation>;
    function GetNumCtrlInputs: Integer;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(hnd: Pointer; graph: TFGraph= nil);overload;
   constructor Create(node_def       : TNodeDef;
                      g              : TFGraph;
                      inputs         : TArray<TFTensor> = [];
                      output_types   : TArray<TF_DataType> = [];
                      control_inputs : TArray<ITensorOrOperation> = [];
                      input_types    : TArray<TF_DataType> = [];
                      original_op    : string = '';
                      op_def         : TOpDef = nil);overload;
   destructor  Destroy; override;
   /// <summary>
   /// Get operation by handle
   /// </summary>
   /// <param name="handle"></param>
   /// <returns></returns>
   function GetOperation(h: Pointer): TFOperation;
   function InputListLength(Name: string): Integer;
   function Input(index: Integer): TF_Output;
   function InputType(index: Integer): TF_DataType;
   //
   /// <summary>
   /// Create and return a new TF_Output for output_idx'th output of this op.
   /// </summary>
   function _tf_output(output_idx: Integer): TF_Output;
   function _tf_input(input_idx: Integer): TF_Input;
   /// <summary>
   /// Add this op to its control flow context.
   ///
   /// This may add new ops and change this op's inputs. self.inputs must be
   /// available before calling this method.
   /// </summary>
   procedure _control_flow_post_processing;
   procedure _add_control_input(op: TFOperation);
   procedure _add_control_inputs(ops: TArray<TFOperation>);
   procedure _set_control_flow_context(ctx: TControlFlowContext);
   function  _get_control_flow_context: TControlFlowContext;
   procedure run(feed_dict : TArray<FeedItem>= []; session: TFSession = nil);
   function  get_attr(name:string): TValue; overload; virtual;
   function  get_attr<T>(name: string): T;  overload;
   function  get_attr_list<T>(name: string): TArray<T>;
   function  GetWhileContext: WhileContext;
   /// <summary>
   /// Update the input to this operation at the given index.
   ///
   /// NOTE: This is for TF internal use only.Please don't use it.
   /// </summary>
   /// <param name="index">the index of the input to update.</param>
   /// <param name="tensor"> the Tensor to be used as the input at the given index.</param>
   procedure _update_input(index: Integer; tensor: TFTensor);
   /// <summary>
   /// Transforms `elems` by applying `fn` to each element unstacked on axis 0.
   /// </summary>
   /// <param name="fn"></param>
   /// <param name="elems">A tensor or (possibly nested) sequence of tensors, each of which will
   ///                     be unstacked along their first dimension.  `fn` will be applied to the
   ///                     nested sequence of the resulting slices.  `elems` may include ragged and
   ///                     sparse tensors. `elems` must consist of at least one tensor.</param>
   /// <param name="dtype">Deprecated: Equivalent to `fn_output_signature`.</param>
   /// <param name="parallel_iterations">(optional) The number of iterations allowed to run in
   ///                                   parallel. When graph building, the default value is 10. While executing
   ///                                   eagerly, the default value is set to 1.</param>
   /// <param name="back_prop">(optional) False disables support for back propagation.</param>
   /// <param name="swap_memory">(optional) True enables GPU-CPU memory swapping.</param>
   /// <param name="infer_shape">(optional) False disables tests for consistent output shapes.</param>
   /// <param name="name">(optional) Name prefix for the returned tensors.</param>
   /// <returns>A tensor or (possibly nested) sequence of tensors.</returns>
   class function map_fn(fn                 : TFunc<TFTensor, TFTensor> ;
                        elems               : TFTensor;
                        dtype               : TF_DataType = DtInvalid;
                        parallel_iterations : Integer= 10;
                        back_prop           : Boolean = true;
                        swap_memory         : Boolean = false;
                        infer_shape         : Boolean = true;
                        name                : string = ''): TFTensor;
   //
   property Graph     : TFGraph    read FGraph;
   property NumOutputs: Integer    read GeNumOutputs;
   property NumInputs : Integer    read GeNumInputs;
   property id_value  : Integer    read Fid_value    write Fid_value;
   property id        : Integer    read Fid_value    write Fid_value;
   property Output    : TFTensor   read GetOutput;
   property inputs    : TInputList read GetInputList;
   property NodeDef   : TNodeDef   read GetNodeDef;
   property Tipo      : string     read GetipoOp;
   property NumControlInputs : Integer           read GetNumCtrlInputs;
   property control_inputs : TArray<TFOperation> read GetCtrlInputs;

end;
{$ENDREGION}

{$REGION 'EagerOperation'}
EagerOperation = class(TFOperation)
  private
    F_Inputs          : TArray<TFTensor>;
    F_Attrs           : TArray<TValue>;
    F_SkipInputIndices: TArray<Int64>;
    FName             : string;
    FNumInputs        : Integer;
    FNumOutputs       : Integer;

    function GetInputList: TInputList; override;
    function GetOutpts: TArray<TensorFlow.DApi.TFTensor>;
    procedure SetOutputs(const Value: TArray<TensorFlow.DApi.TFTensor>);
  public
    constructor Create;
    function    get_attr(attr_name:string): TValue; override;

    property SkipInputIndices: TArray<Int64> read F_SkipInputIndices write F_SkipInputIndices;
    property Outputs   : TArray<TensorFlow.DApi.TFTensor>  read GetOutpts write SetOutputs;
    property Inputs    : TArray<TFTensor>  read F_Inputs    write F_Inputs;
    property Attrs     : TArray<TValue>    read F_Attrs     write F_Attrs  ;
    property name      : String            read FName       write FName;
    property NumInputs : Integer           read FNumInputs  write FNumInputs;
    property NumOutputs: Integer           read FNumOutputs write FNumOutputs;
end;
{$ENDREGION}

{$REGION 'TFGraph'}
/// <summary>
/// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
/// </summary>
/// <remarks>
/// <para>
/// Graphs consist of operations (represented by TFOperation objects), these can be named, or
/// the runtime will automatically assign a name.
/// </para>
/// <para>
/// For debugging purposes, you might want to group operations together, for this, call the
/// WithScope method with your new scope, which will create a new namespace for your object names.
/// </para>
/// <para>
/// For example, if you call WithScope ("demo"), and add an operation named "add" inside the
/// scope, the full name of the operation will be "demo/add", if you create a new scope inside, say
/// "hot", and add a "sub" operation there the result will be "demo/hot/sub".
/// </para>
/// </remarks>
TFGraph = class(TFDisposable)
 private
   Fnodes_by_id        : TDictionary<Integer, ITensorOrOperation>;
   Fnodes_by_name      : TDictionary<string, ITensorOrOperation>;
   Fnames_in_use       : TDictionary<string, Integer>;
   Fversion            : Integer;
   Fnext_id_counter    : Integer;
   Funfetchable_ops    : TList<TFOperation>;
   Funfeedable_tensors : TList<TFTensor>;
   Fname_stack         : string;
   Flast_loss_reduction: string;
   Fis_loss_scaled_by_optimizer : Boolean;
   // <summary>
   /// True if the graph is considered "finalized".  In that case no
   /// new operations can be added.
   /// </summary>
   F_finalized        : Boolean;
   /// <summary>
   /// Arbitrary collections of objects.
   /// </summary>
   Fcollections       : TDictionary<string, TValue>;
   Fcontainer         : string;
   Fseed              : Integer;
   // Current control flow context. It could be either CondContext or WhileContext
   Fcontrol_flow_context : TControlFlowContext ;
   // represents the nested with(...) statements
   Fcontrol_dependencies_stack : TList<TControlDependenciesController>;

   procedure _check_not_finalized;
   function  _as_graph_element(Value: TVAlue): TFTEnsor;
   procedure _create_op_helper(op: TFOperation; compute_device: Boolean = true);
   function  _as_graph_element_locked(obj: TValue; allow_tensor: Boolean = true; allow_operation: Boolean = true): ITensorOrOperation;
 protected
   Fouter_graph       : TFGraph;
   Fgraph_key         : string;
   Fbuilding_function : Boolean;

   procedure NativeDispose(hnd: Pointer); override;
 public
   function name_scope(name: TF_TString) : TF_TString;
   function get_name_scope: TF_TString;
   /// <summary>
   /// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
   /// </summary>
   constructor Create; overload;
   constructor Create(hnd: Pointer); overload;
   destructor  Destroy; override;
   function    NextId: Integer;

   function get_operations: TArray<ITensorOrOperation>;
   /// <summary>
   /// Creates an `Operation` in this graph from the supplied TF_Operation.
   ///
   /// This method is like create_op() except the new Operation is constructed
   /// using `c_op`. The returned Operation will have `c_op` as its _c_op
   /// field.This is used to create Operation objects around TF_Operations created
   /// indirectly by the C API(e.g.by TF_ImportGraphDef, TF_FinishWhile).
   ///
   /// This function does not call Operation._control_flow_post_processing or
   /// Graph._control_dependencies_for_inputs (since the inputs may not be
   /// available yet). The caller is responsible for calling these methods.
   /// </summary>
   /// <param name="c_op">a wrapped TF_Operation</param>
   /// <param name="compute_device">(Optional.) If True, device functions will be executed
   /// to compute the device property of the Operation.</param>
   /// <returns>An `Operation` object.</returns>
   function _create_op_from_tf_operation(c_op: Pointer; compute_device : Boolean= true) : TFOperation;
   procedure device(device_name: string);
   function  as_graph_element(obj: TValue; allow_tensor: Boolean = true; allow_operation: Boolean = true): ITensorOrOperation;
   procedure _pop_control_dependencies_controller(controller: TControlDependenciesController);
   procedure _push_control_dependencies_controller(controller: TControlDependenciesController);
   /// <summary>
   /// Returns a context manager that specifies control dependencies.
   ///
   /// Use with the `with` keyword to specify that all operations constructed
   /// within the context should have control dependencies on
   /// `control_inputs`.
   /// </summary>
   function control_dependencies(control_inputs: TArray<TValue>): TControlDependenciesController;
   /// <summary>
   /// Returns the current control flow context.
   /// </summary>
   /// <returns>A context object.</returns>
   function _get_control_flow_context: TControlFlowContext;
   /// <summary>
   /// Sets the current control flow context.
   /// </summary>
   /// <param name="ctx">a context object.</param>
   procedure  _set_control_flow_context(ctx : TControlFlowContext);
   /// <summary>
   /// Record that the given op depends on all registered control dependencies.
   /// </summary>
   procedure _record_op_seen_by_control_dependencies(op: TFOperation);
   /// <summary>
   /// For an op that takes `input_ops` as inputs, compute control inputs.
   /// </summary>
   /// <param name="input_ops">The data input ops for an op to be created.</param>
   /// <returns>A list of control inputs for the op to be created.</returns>
   function _control_dependencies_for_inputs(input_ops: TArray<ITensorOrOperation>): TArray<ITensorOrOperation>;
   /// <summary>
   /// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name,
   /// or null if the named operation does not exist in the graph.
   /// </summary>
   /// <param name="name">Name to lookup.</param>
   function GetOpByName(const Name: TF_TString): TFOperation;
   /// <summary>
   /// Return a unique operation name for `name`.
   ///
   /// Note: You rarely need to call `unique_name()` directly.Most of
   /// the time you just need to create `with g.name_scope()` blocks to
   /// generate structured names.
   ///
   /// `unique_name` is used to generate structured names, separated by
   /// `"/"`, to help identify operations when debugging a graph.
   /// Operation names are displayed in error messages reported by the
   /// TensorFlow runtime, and in various visualization tools such as
   /// TensorBoard.
   ///
   /// If `mark_as_used` is set to `True`, which is the default, a new
   /// unique name is created and marked as in use.If it's set to `False`,
   /// the unique name is returned without actually being marked as used.
   /// This is useful when the caller simply wants to know what the name
   /// to be created will be.
   /// </summary>
   /// <param name="name">The name for an operation.</param>
   /// <param name="mark_as_used"> Whether to mark this name as being used.</param>
   /// <returns>A string to be passed to `create_op()` that will be used
   /// to name the operation being created.</returns>
   function  unique_name(name: TF_TString;mark_as_used:Boolean = True): TF_TString;
   procedure add_to_collection<T>(name: string;value: T); overload;
   procedure add_to_collection<T>(names: TList<string>; value: T); overload;
   function  get_collection(name: string; scope: string = '') : TValue; overload;
   function  get_collection<T>(name: string; scope: string = ''): TList<T>;overload;
   function  get_collection_ref<T>(name: string): TList<T>;
   procedure colocate_with_for_gradient(op: TFOperation; gradient_uid: string;ignore_existing: Boolean = false);
   procedure prevent_feeding(tensor: TFTensor);
   procedure prevent_fetching(op: TFOperation);
   function  GetOpDef(tipo : string): TOpDef;
   procedure gExit; virtual;
   function  NewOperation(opType,opName: string):TFOperationDesc;
   procedure Add_op(var op: TFOperation);
   function  is_fetchable<T>(tensor_or_op: T ) : Boolean;
   /// <summary>
   /// Returns a context manager that makes this `Graph` the default graph.
   /// Must call Exit() to pop graph
   /// </summary>
   /// <returns></returns>
   function as_default: TFGraph; virtual;
   function Create_op(op_type        : TF_TString ;
                      inputs         : TArray<TFTensor>;
                      dtypes         : TArray<TF_DataType>;
                      input_types    : TArray<TF_DataType> = [];
                      Name           : TF_TString= '';
                      attrs          : TDictionary<string, TAttrValue> = nil;
                      op_def         : TOpDef= nil;
                      compute_device : Boolean = True) : TFOperation; virtual;
   //
   property nodes_by_name       : TDictionary<string, ITensorOrOperation> read Fnodes_by_name;
   property version             : Integer read Fversion;
   property name_stack          : string  read Fname_stack write Fname_stack;
   property graph_key           : string  read Fgraph_key;
   property last_loss_reduction : string  read Flast_loss_reduction;
   property is_loss_scaled_by_optimizer : Boolean read Fis_loss_scaled_by_optimizer write Fis_loss_scaled_by_optimizer;
   property building_function   : Boolean read Fbuilding_function;
   property container           : string  read Fcontainer;
   property seed                : Integer read Fseed write Fseed;
   property OuterGraph          : TFGraph read Fouter_graph;
   property control_flow_context: TControlFlowContext read Fcontrol_flow_context;
   property control_dependencies_stack : TList<TControlDependenciesController> read Fcontrol_dependencies_stack;

end;
{$ENDREGION}

  SliceHelper = record helper for Slice
    public
      class function AlignWithShape(shape: TFShape; slices: TArray<slice>): TArray<slice>; static;
  end;

implementation
   uses System.Math,

        Tensorflow.Core,
        Tensorflow.Utils,
        TensorFlow.Ops,

        TensorFlow.Variable,
        TensorFlow.Operations,
        TensorFlow.Tensor,
        TensorFlow,
        Numpy,
        NumPy.NDArray,

        Oz.Pb.Classes,
        Oz.Pb.StrBuffer,

        ProtoGen.Main;
        

class function SliceHelper.AlignWithShape(shape: TFShape; slices: TArray<slice>): TArray<slice>;
var
  i, j, sliceIndex, ndim: Integer;
  newSlices: TList<Slice>;
begin
    ndim := shape.ndim;
    if (ndim = Length(slices)) then
      Result := slices
    else begin
        // align slices
        newSlices := TList<Slice>.Create();
        try
          sliceIndex := 0;
          i := 0;
          while i < ndim  do
          begin
              if (sliceIndex > Length(slices) - 1) then
              begin
                  newSlices.Add(Slice.All);
                  Continue;
              end;

              if (slices[sliceIndex] = Slice.All) then
              begin
                  newSlices.Add(Slice.All);
                  for j := 0 to ndim - Length(slices) - 1 do
                  begin
                    newSlices.Add(Slice.All);
                    Inc(i);
                  end;
              end
              else
              begin
                  newSlices.Add(slices[sliceIndex]);
              end;
              Inc(sliceIndex);
              Inc(i);
          end;
          Result := newSlices.ToArray();
        finally
          newSlices.Free;
        end;
    end;
end;

{$REGION 'FeedItem'}
{ FeedItem }

constructor FeedItem.Create(k, v: TValue);
begin
    FKey := k;
    FValue := v;
end;

procedure FeedItem.Deconstruct(var k, v: TValue);
begin
    k := FKey;
    v := FValue;
end;

class operator FeedItem.Implicit(t: Tuple<TValue, TValue>): FeedItem;
begin
    Result := FeedItem.Create(t.Value1,t.Value2);
end;
{$ENDREGION}

{$REGION 'FetchMapper'}
{ FetchMapper }

constructor FetchMapper.Create;
begin
    Funique_fetches := TList<ITensorOrOperation>.Create;
    Fvalue_indices  := TList< TArray<Integer> >.Create;
end;

destructor FetchMapper.Destroy;
begin
    Fvalue_indices.Free;
    Funique_fetches.Free;
    inherited Destroy;
end;

function FetchMapper.build_results(values: TList<TNDArray>): TArray<TNDArray>;
begin
    Result := values.ToArray;
end;

function FetchMapper.unique_fetches: TList<ITensorOrOperation>;
begin
    Result := Funique_fetches;
end;

class function FetchMapper.for_fetch(fetch: TValue; graph: TFGraph): FetchMapper;
begin
    var fetches : TArray<TValue>;
    if fetch.IsArray then
    begin
        fetches :=  fetch.AsType< TArray<TValue> >
    end else
    begin
        fetches := [fetch]
    end;

    if fetch.IsType< TList<TValue> > then
    begin
        var fetches1 := fetch.AsType< TList<TValue> > ;
        Result := ListFetchMapper.Create(fetches1.ToArray);
        Exit;
    end;
    if fetch.IsArray then
    begin
        Result := ListFetchMapper.Create(fetches);
        Exit;
    end else
    begin
        Result := ElementFetchMapper.Create(fetches, function(fetched_vals : TList<TNDArray>): TValue
                                                      begin
                                                          if fetched_vals[0] <> nil then
                                                             Result := TValue.From<TNDArray>(fetched_vals[0])
                                                      end , graph);
    end;
end;
{$ENDREGION}

{$REGION 'ListFetchMapper'}
{ ListFetchMapper }

constructor ListFetchMapper.Create(fetches: TArray<TValue>);
begin
    Fmappers := [];
    for var i := 0 to Length(fetches)-1 do
      Fmappers := Fmappers + [ FetchMapper.for_fetch(fetches[i]) ] ;

    var tTuple := _uniquify_fetches(Fmappers);
    Funique_fetches := tTuple.Value1;
    Fvalue_indices  := tTuple.Value2;
end;

destructor ListFetchMapper.Destroy;
begin

  inherited;
end;

function ListFetchMapper._uniquify_fetches(fetch_mappers: TArray<FetchMapper>): Tuple<TList<ITensorOrOperation>, TList<TArray<Integer>>>;
begin
    var unique_fetches := TList<ITensorOrOperation>.Create;
    var value_indices  := TList< TArray<Integer> >.Create;
    var seen_fetches   := TDictionary<ITensorOrOperation, Integer>.Create;
    try
      for var m in fetch_mappers do
      begin
          var m_value_indices := TList<Integer>.Create;
          try
            for var uf in m.unique_fetches do
            begin
                if uf is TFTensor then
                begin
                    var f := uf as TFTensor;
                    if not seen_fetches.ContainsKey(f) then
                    begin
                         seen_fetches.AddOrSetValue(f, seen_fetches.Count);
                         unique_fetches.Add(f);
                    end;
                    m_value_indices.Add(seen_fetches.Count - 1);
                end
                else if uf is TFOperation then
                begin
                    var f := uf as TFOperation;
                    if not seen_fetches.ContainsKey(f) then
                    begin
                         seen_fetches.AddOrSetValue(f, seen_fetches.Count);
                         unique_fetches.Add(f);
                    end;
                    m_value_indices.Add(seen_fetches.Count - 1);
                end else
                begin
                    raise TFException.Create('Not ImplementedException "_uniquify_fetches"');
                end;
            end;
            value_indices.Add(m_value_indices.ToArray());
          finally
            m_value_indices.Free;
          end;
      end;
      Result := Tuple<TList<ITensorOrOperation>, TList<TArray<Integer>>>.Create(unique_fetches, value_indices);
    finally
      seen_fetches.Free;
    end;
end;
{$ENDREGION}

{$REGION 'ElementFetchMapper'}
{ ElementFetchMapper }

constructor ElementFetchMapper.Create(fetches: TArray<TValue>; contraction_fn: TFunc<TList<TNDArray>, TValue>; graph: TFGraph);
begin
   inherited Create;

    var g : TFGraph;
    if Assigned(graph) then g := graph
    else                    g := Tops.get_default_graph ;

    for var fetch in fetches do
    begin
        var el := g.as_graph_element(fetch, true, true);
        Funique_fetches.Add(el);
    end;
   Fcontraction_fn := contraction_fn;
end;

destructor ElementFetchMapper.Destroy;
begin

  inherited;
end;

function ElementFetchMapper.build_results(values: TList<TNDArray>): TArray<TNDArray>;
begin
    Result := [];

    if (values.Count > 0) and (values[0] <> nil) then
    begin
        var ret := Fcontraction_fn(values);
        if ret.IsType<TNDArray> then
        begin
            var value := ret.AsType<TNDArray> ;
            Result := Result + [ value ];
        end
        else if ret.IsType<Boolean> then
        begin
            var value := ret.AsType<Boolean> ;
            Result := Result + [ TNDArray.Scalar<Boolean>(value) ];
        end
        else if ret.IsType<Byte> then
        begin
            var value := ret.AsType<Byte> ;
            Result := Result + [ TNDArray.Scalar<Byte>(value) ];
        end
        else if ret.IsType<Integer> then
        begin
            var value := ret.AsType<Integer> ;
            Result := Result + [ TNDArray.Scalar<Integer>(value) ];
        end
        else if ret.IsType<Single> then
        begin
            var value := ret.AsType<Single> ;
            Result := Result + [ TNDArray.Scalar<Single>(value) ];
        end
    end;
end;
{$ENDREGION}

{$REGION 'FetchHandler'}
{ FetchHandler }

constructor FetchHandler.Create(graph: TFGraph; fetches: TValue; feeds: TDictionary<TValue, TValue>; feed_handles: TProc);
begin
    Ffetch_mapper := FetchMapper.for_fetch(fetches, graph);

    Ffetches      := TList<TFTensor>.Create;
    Fops          := TList<Boolean>.Create;
    Ffinal_fetches:= TList<TFTensor>.Create;
    Ftargets      := TList<TValue>.Create;

    for var fetch in Ffetch_mapper.unique_fetches do
    begin
        if fetch is TFOperation then
        begin
            var val := fetch as TFOperation;
            _assert_fetchable(graph, val);
            Ftargets.Add(val);
            Fops.Add(true);
        end
        else if fetch is TFTensor then
        begin

            var val := fetch as TFTensor;
            _assert_fetchable(graph, val.op);
            Ffetches.Add(val);
            Fops.Add(false);
        end else
        begin
           raise TFException.Create('Not Implemented - FetchHandler fetch');
        end;

    end;
    Ffinal_fetches := Ffetches;
end;

destructor FetchHandler.Destroy;
begin
   Ffetch_mapper.free;
   Ffetches.Free;
   Fops.Free;
   //Ffinal_fetches.Free;
   //Ftargets.Free;

   inherited Destroy;
end;

function FetchHandler.build_results(tensor_values: TArray<TNDArray>): TArray<TNDArray>;
begin
    var full_values := TList<TNDArray>.Create;
    if Ffinal_fetches.Count <> Length(tensor_values) then
       raise TFException.Create('Ffinal_fetches mismatch tensor_values');

    var j : Integer := 0;
    for var is_op in Fops do
    begin
        if is_op then
        begin
            if Length(tensor_values) > 0 then
                full_values.Add(TNDArray.Create(Single.NaN))
            else
                full_values.Add(nil);
        end else
        begin
            var value := tensor_values[j];
            inc(j);
            full_values.Add(value);
        end;
    end;
    if j <> Length(tensor_values) then
       raise TFException.Create('j mismatch tensor_values');

    Result := Ffetch_mapper.build_results(full_values);
end;

function FetchHandler.fetches: TList<TFTensor>;
begin
    Result := Ffinal_fetches
end;

function FetchHandler.targets: TList<TValue>;
begin
    Result := Ftargets
end;

procedure FetchHandler._assert_fetchable(graph: TFGraph; op: TFOperation);
begin
    if not graph.is_fetchable(op) then
       raise TFException.Create('Operation '+Op.Name+ ' has been marked as not fetchable.');

end;
{$ENDREGION}

{$REGION 'TFSessionOptions'}
//------------------------------------------------------------------------------
//----------------------------- TFSessionOptions -------------------------------
//------------------------------------------------------------------------------
constructor TFSessionOptions.Create;
begin
    inherited Create(TF_NewSessionOptions());
end;

destructor  TFSessionOptions.Destroy;
begin
    inherited Destroy;
end;

constructor TFSessionOptions.Create(target: TF_TString; config: TConfigProto);
begin
    inherited Create( TF_NewSessionOptions );

    TF_SetTarget(Handle, PAnsiChar(target));

    if config <> nil  then
       SetConfig(config)
end;

procedure TFSessionOptions.NativeDispose(hnd: Pointer);
begin
   if Assigned(hnd) then
     TF_DeleteSessionOptions(hnd);
end;

procedure TFSessionOptions.SetTarget(target: TF_TString);
begin
   if not Assigned(Handle) then
     ObjectDisposedException();
   TF_SetTarget(Handle, PAnsiChar(target));
end;

procedure TFSessionOptions.SetConfig(protoData: TConfigProto);
var
   Saver    : TpbSaver;
   Status   : TFStatus;
begin
   if not Assigned(Handle) then
     ObjectDisposedException();

    Status := TFStatus.Create;
   try
     Saver.Init;
     TpbSaver.SaveConfigProto(Saver,protoData);
     var config_str := Saver.Pb.GetBytes;

     TF_SetConfig(Handle ,@config_str[0],Length(config_str),Status.Handle);
     Status.RaiseEx;
   finally
      Status.Free;
   end;

end;
{$ENDREGION}

{$REGION 'TFSession'}
//------------------------------------------------------------------------------
//----------------------------- TFSession --------------------------------------
//------------------------------------------------------------------------------
constructor TFSession.Create(hnd: Pointer; graph: TFGraph);
begin
   inherited Create(hnd);
   if Assigned(graph)  then FGraph := graph
   else                     FGraph := Tops.get_default_graph;

end;

constructor TFSession.Create(target: TF_TString; g : TFGraph; config : TConfigProto; status: TFStatus);
begin
   if Assigned(graph)  then FGraph := graph
   else                     FGraph := Tops.get_default_graph;

   if not FGraph.building_function then
   begin
       if Tops.get_default_graph <> FGraph then
          FGraph.as_default;
   end;

   var opts := TFSessionOptions.Create(target, config);
   try
     var sStatus : TFStatus;
     if Assigned(status) then sStatus := status
     else                     sStatus := tf.status;

     var l_pSession:= TF_NewSession(graph.handle, opts.Handle, sStatus.Handle);
     sStatus.CheckMaybeRaise(status,False);

     inherited Create(l_pSession);
   finally
     opts.Free;
   end;

end;

constructor TFSession.Create(g: TFGraph; config: TConfigProto; status: TFStatus);
begin
    Create('', g, config, status)
end;

destructor  TFSession.Destroy;
var
 l_oStatus: TFStatus;
begin
   if Assigned(Handle) then
   begin
       l_oStatus := TFStatus.Create;
       TF_CloseSession (Handle, l_oStatus.Handle);
       TF_DeleteSession(Handle, l_oStatus.Handle);
       Handle := Nil;           // <- Don't execute NativeDispose
       FreeAndNil(l_oStatus);
       FGraph.Handle := Nil;  // <- Already with session deleted.
       FreeAndNil(FGraph);
   end;
   inherited Destroy;
end;

function TFSession.eval(tensor: TFTensor): TFTensor;
var
  l_pInputs, l_pOutputs          : PTF_Output;
  l_pInputValues, l_pOutputValues: PTF_Tensor;
  l_pTargets                     : PTF_Operation;
begin
    var status := tf.Status;

    l_pInputs      := nil;
    l_pInputValues := nil;
    l_pTargets     := nil;

    var output_values : TArray<PTF_Tensor> := [ nil ];
    l_pOutputValues := @(output_values[0]);

    var fetch_list := [ tensor._as_tf_output ];
    l_pOutputs     := @(fetch_list[0]);

    TF_SessionRun(Handle,
        // RunOptions
        nil,
        // Inputs Tensors
        l_pInputs, l_pInputValues, 0,
        // Outputs Tensors
        l_pOutputs, l_pOutputValues, 1,
        // Target operations
        l_pTargets, 0,
        // RunMetadata
        nil,
        // Output Status
        status.Handle);
    status.RaiseEx;
    var ss := status.StatusMessage;
    Result := TFTensor.Create( output_values[0]);
end;

function TFSession.as_default: TFSession;
begin
    Result := Tops.set_default_session(self);
end;

function TFSession.fetchValue(output: Pointer): TNDArray;
begin
    var tensor := TFTensor.Create(output);
    try
      Result     := tensor.numpy;
    finally
      tensor.Free;
    end;
end;

procedure TFSession.NativeDispose(hnd: Pointer);
var
 l_pStatus : PTF_Status;
begin
   if Assigned(hnd) then
   begin
       l_pStatus := TF_NewStatus();
       TF_DeleteSession(hnd,l_pStatus);
       TF_DeleteStatus(l_pStatus);
   end;
end;

function TFSession.run(fetches: TFTensor): TNDArray;
begin
    var feed_items : TArray<FeedItem> := [];
    Result := _run(fetches, feed_items)[0];
end;

function TFSession.run(fetches: TArray<TValue>): TArray<TNDArray>;
begin
    var feed_items : TArray<FeedItem> := [];
    Result := _run(TValue.From<TArray<TValue>>(fetches), feed_items);
end;

function TFSession.run(fetches: TArray<TValue>; feed_dict: TArray<FeedItem>): TArray<TNDArray>;
begin
    Result := _run(TValue.From<TArray<TValue>>(fetches), feed_dict);
end;

function TFSession.run(fetche: ITensorOrOperation; feed_dict: TArray<FeedItem>): TNDArray;
begin
    var Res := _run(fetche, feed_dict);
    if fetche is TFTensor then
      Result := Res[0]
    else
      Result := nil;
end;

function TFSession.run(fetche: TFTensor; feed_dict: TArray<FeedItem>): TNDArray;
begin
    Result := _run(fetche, feed_dict)[0];
end;

procedure TFSession.run(op: TFOperation; feed_dict: TArray<FeedItem>);
begin
    _run(op, feed_dict);
end;

function TFSession._call_tf_sessionrun(feed_dict: TArray<TPair<TF_Output, TFTensor>>; fetch_list: TArray<TF_Output>; target_list: TList<TFOperation>): TArray<TNDArray>;
var
  inputs       : TArray<TF_Output>;
  input_values : TArray<PTF_Tensor>;
  output       : TArray<TF_Output>;
  output_values: TArray<PTF_Tensor>;
  target_opers : TArray<PTF_Operation>;
begin
    // Ensure any changes to the graph are reflected in the runtime.
    _extend_graph();
    var status := tf.Status;

    // Input tensors
    //
    inputs := [];
    for var i := 0 to Length(feed_dict) - 1 do
       inputs := inputs + [ feed_dict[i].Key ];
    var pInputs : PTF_Output := nil;
    if Length(inputs) > 0 then  pInputs := @(inputs[0]);

    input_values := [];
    for var i := 0 to Length(feed_dict) - 1 do
       input_values := input_values + [ feed_dict[i].Value.Handle ];
    var pInput_values : PTF_Tensor := nil;
    if Length(input_values) > 0 then  pInput_values := @(input_values[0]);

    // Output tensors
    //
    output := [];
    for var i := 0 to Length(fetch_list) - 1 do
       output := output + [ fetch_list[i] ];
    var pOutput : PTF_Output := nil;
    if Length(output) > 0 then  pOutput := @(output[0]);

    output_values := [];
    for var i := 0  to Length(fetch_list) - 1 do
        output_values := output_values + [ nil ];
    var pOutput_values : PTF_Tensor := nil;
    if Length(output_values) > 0 then  pOutput_values := @(output_values[0]);

    // Target operations
    //
    target_opers := [];
    for var i := 0 to target_list.Count - 1 do
       target_opers := target_opers + [ target_list[i].Handle ];
    var pTarget_opers : PTF_Operation := nil;
    if Length(target_opers) > 0 then  pTarget_opers :=  @(target_opers[0]);

    TF_SessionRun(Handle,
                  // RunOptions
                  nil,
                  // Input tensors
                  pInputs, pInput_values,Length(input_values),
                  // Output tensors
                  pOutput, pOutput_values, Length(output_values),
                  // Target operations
                  pTarget_opers, target_list.Count,
                  // RunMetadata
                  nil,
                  // Output status
                  status.Handle);
    status.RaiseEx;
    SetLength(Result,Length(fetch_list));
    for var i := 0 to Length(fetch_list) - 1 do
        Result[i] := fetchValue( output_values[i] );
end;

function TFSession._do_run(const target_list: TList<TFOperation>; const fetch_list: TList<TFTensor>; const feed_dict: TDictionary<TValue, TValue>): TArray<TNDArray>;
var
  x : TPair<TValue, TValue>;
begin
    var feeds : TArray< TPair<TF_Output, TFTensor> >;
    SetLength(feeds,feed_dict.Count);

    var i : Integer := 0;
    for x in feed_dict do
    begin
        if x.Key.IsType<TFTensor> then
        begin
            var key := x.Key.AsType<TFTensor>;
            if x.Value.IsType<TFTensor> then
            begin
                var v := x.Value.AsType<TFTensor>;
                if v.dtype <> key.dtype then
                     raise TFException.Create('Tensor '+v.ToString+' does not match the expected dtype {key.dtype}, actual dtype: {v.dtype}');
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, v);
                inc(i);
            end
            else if x.Value.IsType<Pointer> then
            begin
                var v := x.Value.AsType<Pointer>;
                var tensor := TFTensor.Create(v);
                if tensor.dtype <> key.dtype then
                     raise TFException.Create('Tensor '+tensor.ToString+' does not match the expected dtype {key.dtype}, actual dtype: {v.dtype}');
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, tensor);
                inc(i);
            end
            else if x.Value.IsType<Boolean> then
            begin
                var v := x.Value.AsType<Boolean>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsType<Byte> then
            begin
                var v := x.Value.AsType<Byte>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsType<Integer> then
            begin
                var v := x.Value.AsType<Integer>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsType<Int64> then
            begin
                var v := x.Value.AsType<Int64>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsType<Single> then
            begin
                var v := x.Value.AsType<Single>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(Single(v)));
                inc(i);
            end
            else if x.Value.IsType<Double> then
            begin
                var v := x.Value.AsType<Double>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(Double(v)));
                inc(i);
            end
            else if x.Value.IsType<TF_TString> then
            begin
                var v := x.Value.AsType<TF_TString>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsArray then
            begin
                var s := TUtils.GetShape(x.Value);
                {TODO -oMax -cSession : Add Tensor Generic Array creation}
               (*var v := x.Value.AsType<TF_TString>;
                feeds[i++] = new KeyValuePair<TF_Output, Tensor>(key._as_tf_output(), new Tensor(v, v.GetShape()));
               *)
                inc(i);
            end else
            begin
              raise TFException.Create('Not Implemented');
            end;
        end else
        begin
           raise TFException.Create('Not Implemented');
        end;
    end;
    var ft := Enumerable<TFTensor>.Create(fetch_list.toArray).Select<TF_Output>(function (Arg1: TFTensor): TF_Output
                                                                       begin
                                                                           Result := Arg1._as_tf_output;
                                                                       end );
    var fetches := ft.ToArray;
    Result := _call_tf_sessionrun(feeds, fetches, target_list);
end;

procedure TFSession._extend_graph;
begin

end;

function TFSession._run(fetches: TValue; feed_dict : TArray<FeedItem>): TArray<TNDArray>;
var
  fh            : FetchHandler;
  final_fetches : TList<TFTensor>;
  final_targets : TList<TValue>;
begin
    if Fgraph.version = 0 then
      raise TFException.Create('The Session graph is empty. Add operations to the graph before calling run().');

    var feed_dict_tensor := TDictionary<TValue, TValue>.Create;
    // Validate and process feed_dict.
    if feed_dict <> nil then
    begin
        for var subfeed in feed_dict do
        begin
            var subfeed_t := FGraph.as_graph_element(subfeed.Key, true, false);
            feed_dict_tensor.AddOrSetValue(subfeed_t, subfeed.Value);
        end;
    end;

    // Create a fetch handler to take care of the structure of fetches.
    fh := FetchHandler.Create(Fgraph, fetches, feed_dict_tensor);
    try
      // Run request and get response.
      // We need to keep the returned movers alive for the following _do_run().
      // These movers are no longer needed when _do_run() completes, and
      // are deleted when `movers` goes out of scope when this _run() ends.
      _update_with_movers();
      final_fetches  := TList<TFTensor>.Create(fh.fetches.ToArray);
      final_targets  := TList<TValue>.Create(fh.targets.ToArray);
      try
        // We only want to really perform the run if fetches or targets are provided,
        // or if the call is a partial run that specifies feeds.
        var ft := Enumerable<TValue>.Create(final_targets.ToArray).Select<TFOperation>(function (Arg1: TValue): TFOperation
                                                                           begin
                                                                               Result := Arg1.AsType<TFOperation>;
                                                                           end );
        var target  := TList<TFOperation>.Create(ft.ToArray);
        var results := _do_run(target, final_fetches, feed_dict_tensor);

        Result := fh.build_results(results);
      finally
        final_fetches.Free;
        final_targets.Free;
      end;
    finally
      feed_dict_tensor.Free;
      fh.Free;
    end;
end;

function TFSession._update_with_movers: TList<TValue>;
begin
    Result := TList<TValue>.Create;
end;
{$ENDREGION}

{$REGION 'TFTensor'}

{ TFTensor }

constructor TFTensor.Create(hnd: Pointer);
begin
 inherited Create(hnd);
 FlDeallocator_called := False;

 UpdateTensoData;

end;

constructor TFTensor.Create(const value: Boolean);
begin
   inherited Create(InitTensor<Boolean>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Byte);
begin
   inherited Create(InitTensor<Byte>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int8);
begin
   inherited Create(InitTensor<Int8>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: UInt16);
begin
   inherited Create(InitTensor<UInt16>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int16);
begin
   inherited Create(InitTensor<Int16>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Cardinal);
begin
   inherited Create(InitTensor<Cardinal>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Integer);
begin
   inherited Create(InitTensor<Integer>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: UInt64);
begin
   inherited Create(InitTensor<UInt64>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int64);
begin
   inherited Create(InitTensor<Int64>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Single);
begin
   inherited Create(InitTensor<Single>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Double);
begin
   inherited Create(InitTensor<Double>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(hhandle: Pointer; clone: Boolean);
begin
    Handle := hhandle;
    if (clone) and (hhandle <> nil) then
        Handle := TF_NewTensor(shape, dtype, TensorDataPointer);
end;

constructor TFTensor.Create(const value: TF_TString);
begin
   inherited Create(InitTensor<TF_TString>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(data: Pointer;shape: TFShape; dtype: TF_DataType);
begin
   inherited Create( TFTensor.TF_NewTensor(shape, dtype, data ) );

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(shape : TFShape; dtype:TF_DataType);
begin
   InitTensor(shape, dtype);
   inherited Create(Handle);

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    InitTensor(shape,bytes,dtype) ;
    Create(Handle);

    FIsCreatedInGraphMode := not tf.executing_eagerly;
    FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<T>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if (shape = nil) or (shape.IsNil) then  sShape := TUtils.GetShape( TValue.From< TArray<T> >( aArray ) )
    else                                    sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<T>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if (shape = nil) or (shape.IsNil) then  sShape := TUtils.GetShape( TValue.From< TArray<TArray<T>> >( aArray ) )
    else                                    sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<TArray<T>>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if(shape = nil) or (shape.IsNil) then  sShape := TUtils.GetShape( TValue.From<  TArray<TArray<TArray<T>>> >( aArray ) )
    else                                   sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if (shape = nil) or (shape.IsNil) then  sShape := TUtils.GetShape( TValue.From<  TArray<TArray<TArray<TArray<T>>>> >( aArray ) )
    else                                    sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0][0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(op: TFOperation; value_index: Integer; dtype: TF_DataType);
begin
     Fop     := op;
     FGraph := Fop.Graph;

     if Assigned(op.Handle) then
      FDevice := string( AnsiString(TF_OperationDevice(op.Handle)) ) ;

     FValue_index := value_index;
     FOverride_dtype := dtype;
     FId             := TOps.uid;
     FIsCreatedInGraphMode := not tf.executing_eagerly;
     FEagerTensorHandle    := nil;
end;

class function TFTensor.TF_NewTensor(data: TArray<Byte>; shape: TFShape; dtype: TF_DataType):PTF_Tensor;
begin
     var _length : TF_size_t := Length(data);
     var dims     := shape.Dims;

     var pDims : PTF_int64_t;
     pDims :=  PTF_int64_t(Pointer(@dims)^);

     var hHandle := TF_AllocateTensor(dtype, pDims, shape.ndim, _length);
     var ttensor := TF_TensorData(hHandle);
     if ttensor = nil then
        raise TFException.Create('AllocateTensor failed.');

     if Assigned(data)  then
       Move(data[0], ttensor^, _length);

     Result := hHandle;

     inherited Create(Result);
end;

class function TFTensor.TF_NewTensor(shape: TFShape; dtype: TF_DataType; data: Pointer):PTF_Tensor;
begin
     var _length : TF_size_t := shape.Size * Tdtypes.get_datatype_size(dtype);
     var dims     := shape.Dims;

     var pDims : PTF_int64_t ;
     pDims :=  PTF_int64_t(Pointer(@dims)^);

     var hHandle := TF_AllocateTensor(dtype, pDims, shape.ndim, _length);
     var ttensor := TF_TensorData(hHandle);
     if ttensor = nil then
        raise TFException.Create('AllocateTensor failed.');

     if Assigned(data)  then
       Move(data^, ttensor^, _length);

     Result := hHandle;

     inherited Create(Result);
end;

destructor  TFTensor.Destroy;
begin
 if FlDeallocator_called then
   Handle := Nil;
 inherited Destroy;
end;

function TFTensor.eval(session: TFSession; feed_dict: TArray<FeedItem>): TNDArray;
begin
    if self = nil then
      raise TFException.Create('Run in Empty Session. TFTensor.eval(). Run in Graph mode?');
    Result := Tops._eval_using_default_session(self, feed_dict, graph, session);
end;

procedure TFTensor.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteTensor(hnd);
end;

function TFTensor.Equals(y: Integer): TFTensor;
begin
    Result := gen_math_ops.equal(self, constant_op.constant(y, self.dtype, 'Const') );
end;

function TFTensor.NotEquals(y: Integer): TFTensor;
begin
    Result := gen_math_ops.not_equal(self, constant_op.constant(y, self.dtype, 'Const') );
end;

function TFTensor.StringBytes: TArray< TArray<Byte> >;
var
  i    : Integer;
  buf  : TArray< TArray<Byte> > ;
begin
    if dtype <> TF_DataType.TF_STRING then
      raise TFException.Create('Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})');
    //
    // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
    // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
    //
    var size : NativeInt := 1;
    for var s in shape.dims do
          size := size * s;

    SetLength(buf,size);
    for i := 0 to  size - 1 do
        buf[i] := StringBytes(i);

    Result := buf;
end;

function TFTensor.StringBytes(index: Integer): TArray<byte>;
var
 data : Pointer;

begin
    if dtype <> TF_DataType.TF_STRING then
      raise TFException.Create('Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})');

    Result := [] ;
    var tstrings : PTF_TString:= TensorDataPointer;

    for var i := 0 to  shape.size - 1 do
    begin
        if index = i then
        begin
            data     := TF_StringGetDataPointer(tstrings);

            var len  := TF_StringGetSize(tstrings);

            SetLength(Result,len);
            CopyMemory(@Result[0], PByte(data), len)
        end;
        Inc(PByte(tstrings), TF_TSRING_SIZE);
    end;
end;

function TFTensor.StringData(index: integer): AnsiString;
Begin
    var bytes := StringBytes(index);
    Result := AnsiString (TEncoding.UTF8.GetString(bytes));
end;

function TFTensor.StringData: TArray<TF_TString>;
begin
    var buffer := StringBytes;
    var _str : TArray<TF_TString>;
    SetLength(_str, Length(buffer) );
    for var i := 0  to Length(_str) - 1 do
        _str[i] := AnsiString(TEncoding.UTF8.GetString( buffer[i]));
    Result :=  _str;
end;

class function TFTensor.StringTensor(srcArray: TArray<TArray<byte>>; shape: TFShape):PTF_Tensor;
var

  dims     : TArray<Int64>;
  pDims    : PTF_int64_t ;
  i        : Integer ;

  //ts       : TArray<TF_TString> ;
begin
     dims     := shape.Dims;
     pDims    :=  PTF_int64_t(Pointer(@dims)^);

     var pTensor := TF_AllocateTensor(TF_DataType.TF_STRING, pDims, shape.ndim, shape.size * TF_TSRING_SIZE);
     var tstr  : PTF_TString  := TF_TensorData(pTensor);

    // Modified by Max 20/11/2022 20:36:00 Testing for byte string non UTF8 char
    //SetLength(ts,Length(srcArray));
    //for i := 0 to Length(srcArray)- 1 do
      //ts[i] := TF_TString( TEncoding.UTF8.GetString(srcArray[i]) );


    for i := 0 to Length(srcArray) - 1 do
    begin
          TF_StringInit(tstr);

          TF_StringCopy(tstr, @srcArray[i][0], Length(srcArray[i]));

          Inc(pbyte(tstr), TF_TSRING_SIZE);
    end;
    Result := pTensor;
end;

class function TFTensor.StringTensor(srcArray: TArray<TF_TString>; shape: TFShape):PTF_Tensor;
var
  buffer : TArray<TArray<Byte>>;
begin
    // convert string array to byte[][]
    //
    SetLength(buffer,Length(srcArray));
    for var i := 0 to Length(srcArray)- 1 do
      buffer[i] := TEncoding.UTF8.GetBytes( string(srcArray[i]) );

    Result := StringTensor(buffer,shape);
end;

class function TFTensor.StringTensor(srcArray: TArray<string>; shape: TFShape):PTF_Tensor;
var
  buffer : TArray<TArray<Byte>>;
begin
    // convert string array to byte[][]
    //
    SetLength(buffer,Length(srcArray));
    for var i := 0 to Length(srcArray)- 1 do
      buffer[i] := TEncoding.UTF8.GetBytes( srcArray[i] );

    Result := StringTensor(buffer,shape);
end;

function TFTensor.ToArray<T>: TArray<T>;
var
  l_pData,l_pVal  : Pointer ;
  l_iFullByteSize : NativeInt;
  res             : TArray<T>;
begin
    if Tdtypes.as_tf_dtype( TypeInfo(T) ) <> Dtype then
      raise TFException.Create('Required dtype {dtype} mismatch with {typeof(T).as_tf_dtype()}.');

    l_pData := TF_TensorData(Handle);

    if (ndim = 0) or (size = 1) then
    begin
        SetLength(res,1);
        l_pVal  := @res[0];
        l_iFullByteSize := dtypesize;

        Move(l_pData^, l_pVal^, l_iFullByteSize);
        Result := res;
        Exit;
    end;

    SetLength(res,size);
    l_pVal  := nil;
    if Length(res) > 0 then  l_pVal := @res[0];

    l_iFullByteSize :=  size * dtypesize;
    if l_pVal <> nil then
      Move(l_pData^, l_pVal^, l_iFullByteSize);

    Result := res;
end;

function TFTensor.ToString: string;
begin
    Result := Format('tf.Tensor "%s" shape=%s dtype=%s',[name,Shape.ToString,Tdtypes.as_numpy_name(dtype)]);
end;

procedure TFTensor.UpdateTensoData;
begin
   FEagerTensorHandle    := nil;
   FlDeallocator_called  := False;
   FIsCreatedInGraphMode := False;
   FIsList               := False;

   Ftf_output            := nil;
   FValue_index          := 0;
   FOverride_dtype       := TF_DataType.DtInvalid;
   FId                   := 0;

   FGraph                := nil;

   GetRank;
   GetShape;
   GetName;
   GetType;
   GetDevice;
 end;

 function TFTensor.GetNDArray(ddtype: TF_DataType): TNDArray;
 begin
     if ddtype = TF_DataType.TF_STRING then
    begin
        var s : TFShape := Shape;
        var str := StringData;
        Result := TNDArray.Create(str,@s);
        Exit;
    end;
    Result := TNDArray.Create(self, true);
 end;

 function TFTensor.GetnDim: Integer;
begin
   Result := rank;
end;

function TFTensor.numpy: TNDArray;
 begin
     Result := GetNDArray(dtype);
 end;

function TFTensor._as_tf_output: TF_Output;
begin
    if not Ftf_output.HasValue then
    begin
     if Fop <> nil then
        Ftf_output := TF_Output.Create(Fop.Handle,FValue_index)
     else
        Ftf_output := TF_Output.Create(Handle,FValue_index)

    end ;


    Result := Ftf_output;
end;

class function TFTensor.GetBestDType<Tx, Ty>(x: Tx; y: Ty): TF_DataType;
begin
    var dtype1 := TUtils.GetDataType( TValue.From<Tx>(x) );
    var dtype2 := TUtils.GetDataType( TValue.From<Ty>(y) );
    if (Tdtypes.is_integer(dtype1)) and (Tdtypes.is_floating(dtype2)) then
        Exit(dtype2)
    else if (Tdtypes.is_floating(dtype1)) and (Tdtypes.is_integer(dtype2)) then
        Exit(dtype1)
    else
        Result := dtype1;
end;

class function TFTensor.BinaryOpWrapper<Tx, Ty>(name: string; x: Tx; y: Ty): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ TValue.From<Tx>(x) ];
    vValues := vValues + [ TValue.From<Ty>(y) ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', name, @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var dtype := GetBestDType(x, y);
                                                var x1 := Tops.convert_to_tensor(vValues[0], dtype, 'x');
                                                var y1 := Tops.convert_to_tensor(vValues[1], dtype,'y' );
                                                var newname := v1.ToString;
                                                if string.LowerCase(name)  = 'add' then
                                                  Result := math_ops.add_v2(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'div' then
                                                  Result := math_ops.&div(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'floordiv' then
                                                  Result := gen_math_ops.floor_div(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'truediv' then
                                                  Result := math_ops.truediv(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'mul' then
                                                  Result := math_ops.multiply(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'sub' then
                                                  Result := gen_math_ops.sub(x1, y1, newname)
                                                else if string.LowerCase(name)  = 'mod' then
                                                  Result := gen_math_ops.floor_mod(x1, y1, newname)
                                                else
                                                 raise TFException.Create('Not Implemented BinaryOpWrapper'+ name);
                                            end );
end;

function TFTensor.BufferToArray: TArray<Byte>;
var
  l_pData,l_pVal  : Pointer ;
  res             : TArray<Byte>;
begin
    if Dtype = TF_String then
    Begin
        Result := StringBytes(0);
        Exit;
    End;

    SetLength(res,NativeInt(bytesize));
    l_pData := TF_TensorData(Handle);

    l_pVal  := @res[0];

    Move(l_pData^, l_pVal^, bytesize);

    Result := res;
end;

function TFTensor.consumers: TArray<TFOperation>;
begin
    var output         := _as_tf_output;
    var consumer_names := TF_OperationOutputConsumers_wrapper(output);
    Result := [];
    for var i := 0 to Length(consumer_names) - 1 do
       Result := Result + [ graph.GetOpByName(consumer_names[i]) ];

end;

procedure TFTensor.InitTensor(shape: TFShape; dtype: TF_DataType);
begin
    Handle := TF_NewTensor(shape,dtype,nil)  ;
end;

class function TFTensor.InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType): PTF_Tensor;
begin
     if dtype = TF_DataType.TF_STRING then
     begin
         var buf : TArray<TArray<byte>>;
         SetLength(buf,1);
         for var i := 0 to Length(bytes) - 1 do
           buf[0] := buf[0] + [ bytes[i] ];
         Result := StringTensor( buf, TFShape.Scalar);
     end else
     begin
         Result  := TF_NewTensor(bytes,shape,dtype) ;
     end;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<T>; shape: TFShape): PTF_Tensor;
begin
    var dtype := TUtils.GetDataType( TValue.From< TArray<T> >(aArray)) ;

    if shape.isNil then
       shape :=  TUtils.GetShape(TValue.From< TArray<T> >(aArray));

    Result := InitTensor<T>(aArray, shape,dtype)
end;

class function TFTensor.InitTensor<T>(aArray: TArray<T>;shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     if TypeInfo(T) = TypeInfo(TF_TString) then
     begin
         var v := TValue.From< TArray<T> >(aArray);
         var v1 := v.AsType< TArray<TF_TString> > ;
         Result := StringTensor(v1, shape);
     end
     else if TypeInfo(T) = TypeInfo(string)then
     begin
         var v := TValue.From< TArray<T> >(aArray);
         var v1 := v.AsType< TArray<string> > ;
         Result := StringTensor(v1, shape);
     end else
     begin
         l_pData := nil;
         if Length(aArray) > 0  then
            l_pData := PByte(@aArray[0]);

         Result  := TF_NewTensor(shape,dtype,l_pData) ;
     end;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<T>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     var _length : NativeInt := shape.Size;
     var a : TArray<T>; SetLength(a,_length) ;
     var j : Integer := 0;
     for var i := 0 to Length(aArray) - 1 do
     begin
       CopyMemory(@a[j], @aArray[i][0], Length(aArray[i]) * Tdtypes.get_datatype_size(dtype)) ;
       Inc(j,Length(aArray[i]));
     end;

     l_pData := PByte(@a[0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<T>>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     var _length : NativeInt := shape.Size;
     var a : TArray<T>; SetLength(a,_length) ;
     var j : Integer := 0;
     for var i := 0 to Length(aArray) - 1 do
     begin
        for var k := 0 to Length(aArray[i]) - 1 do
        begin
           CopyMemory(@a[j], @aArray[i][k][0], Length(aArray[i][k]) * Tdtypes.get_datatype_size(dtype)) ;
           Inc(j,Length(aArray[i][k]));
        end;
     end;

     l_pData := PByte(@a[0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     var _length : NativeInt := shape.Size;
     var a : TArray<T>; SetLength(a,_length) ;
     var j : Integer := 0;
     for var i := 0 to Length(aArray) - 1 do
     begin
        for var k := 0 to Length(aArray[i]) - 1 do
        begin
            for var x := 0 to Length(aArray[i][k]) - 1 do
            begin
               CopyMemory(@a[j], @aArray[i][k][x][0], Length(aArray[i][k][x]) * Tdtypes.get_datatype_size(dtype)) ;
               Inc(j,Length(aArray[i][k][x]));
            end;
        end;
     end;

     l_pData := PByte(@a[0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

function TFTensor.GetTensorDataPointer: Pointer;
begin
     if Handle = nil then Result := nil
     else                 Result := TF_TensorData(Handle);
end;

function TFTensor.GetType: TF_DataType;
begin
    if Handle = nil then
      FDtype := FOverride_dtype
    else
      FDtype := TF_DataType( TF_TensorType(Handle) );

    Result := FDtype;
end;

function TFTensor.GetByteSize: UInt64;
begin
    if Assigned(Handle) then
     Result := TF_TensorByteSize(Handle)
   else
     Result := 0;
end;
function TFTensor.GetData: Pointer;
begin
    if Assigned(Handle) then
     Result := TF_TensorData(Handle)
   else
     Result := nil;
end;

function TFTensor.GetDataTypeSize: UInt64;
begin
    Result := Tdtypes.get_datatype_size(Dtype);
end;

function TFTensor.GetDevice: string;
begin
    Result := '';
    if FOp <> nil then
      Result := FOp.Device;
    FDevice := Result;
end;

function TFTensor.GetDims: TArray<Int64>;
begin
   Result := shape.Dims;
end;

function TFTensor.GetItem(_slice: string): TFTensor;
begin
    var slices: TArray<string> := [_slice];
    var sl    : TArray<Slice>  := [];
    for var i := 0 to Length(slices) -1 do
    begin
        sl := sl + [ Slice.Create( slices[i] ) ]
    end;
    Result := item[sl];
end;

function TFTensor.GetItem(slices: TArray<string>): TFTensor;
begin
    var sl: TArray<Slice> := [];
    for var i := 0 to Length(slices) -1 do
    begin
        sl := sl + [ Slice.Create( slices[i] ) ]
    end;
    Result := item[sl];
end;

function TFTensor.GetItem(idx: Integer): TFTensor;
begin
    Result := _slice(idx)
end;

function TFTensor.GetItem(slices: TArray<Slice>): TFTensor;
begin
    var args := TUtils.ParseSlices(slices);
    var newVal : TValue := TValue.From< ParsedSliceArgs >(args);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'strided_slice', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var name : string := v1.ToString;
                                                if args.aBegin <> nil then
                                                begin
                                                    var packed_begin  := array_ops.stack( TValue.from< TArray<Integer> >(args.aBegin) );
                                                    var packed_end    := array_ops.stack( TValue.from< TArray<Integer> >(args.aEnd) );
                                                    var packed_strides:= array_ops.stack( TValue.from< TArray<Integer> >(args.aStrides) );
                                                    Result := gen_array_ops.strided_slice(self,
                                                                                          packed_begin,
                                                                                          packed_end,
                                                                                          packed_strides,
                                                                                          args.iBeginMask,
                                                                                          args.iEndMask,
                                                                                          args.iEllipsisMask,
                                                                                          args.iNewAxisMask,
                                                                                          args.iShrinkAxisMask,
                                                                                          name);
                                                    Exit;
                                                end;
                                                raise  TFException.Create('Not Implemented');
                                            end );
end;

function TFTensor.GetItem(start, stop, step: TFTensor): TFTensor;
begin
    var args := TUtils.ParseSlices(start, stop, step);
    var newVal : TValue := TValue.From< ParsedSliceArgs >(args);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'strided_slice', @newVal),
                function(v1: TNameScope): TFTensor
                  begin
                      var name : string := v1.ToString;
                      var tensor := gen_array_ops.strided_slice(self,
                                                                args.tPackedBegin,
                                                                args.tPackedEnd,
                                                                args.tPackedStrides,
                                                                args.iBeginMask,
                                                                args.iEndMask,
                                                                args.iEllipsisMask,
                                                                args.iNewAxisMask,
                                                                args.iShrinkAxisMask,
                                                                name);
                      Result := tensor;
                  end);
end;

function TFTensor.GetName: string;
var
 opname : string;
begin
    opname := '<unnamed>';
    if Fop <> nil then
      opname := Fop.name;

   FName := Result;
   Result := Format('%s:%d',[opname,value_index]);
end;

function TFTensor.GetRank: Integer;
begin
    if not Assigned(Handle) then
    begin
        var output :=  _as_tf_output;
        var ndim : Integer := TF_GraphGetTensorNumDims(FGraph.Handle,output,tf.Status.Handle);
        Result := ndim;
    end else
    begin
        Result := TF_NumDims(Handle)
    end;
    FRank := Result;
end;

function TFTensor.GetShape: TFShape;
begin
    FShape := System.default(TFShape);

    if rank < 0 then
        Exit(FShape);

    var irank : TArray<Int64>; SetLength(irank,rank);
    var dims := TFShape.Create(irank);

    if not Assigned(Handle) then
    begin

        var pDim : PInt64 := nil;
        if Length(dims.Dims) > 0 then pDim := @dims.Dims[0];

        TF_GraphGetTensorShape(op.graph.Handle, _as_tf_output(), pDim, rank, tf.Status.Handle);
        FShape := dims;
    end else
    begin
        for var i := 0 to rank -1 do
           dims.Dims[i] := TF_Dim(Handle, i);

        FShape := dims;
    end;
    Result := FShape;
end;

procedure TFTensor.Setshape(const Value: TFShape);
begin
    if (value.IsNil) or (Value.ndim < 1) then
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, nil, -1, tf.Status.Handle)
    else
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, @value.dims[0], value.ndim, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

procedure TFTensor.set_shape(shape: TFTensor);
begin
    // ReSharper disable once MergeConditionalExpression
    if shape = nil then Fshape := System.Default(TFShape)
    else                Fshape := shape.shape;
end;

function TFTensor._shape_tuple: TArray<Integer>;
begin
    if rank < 0 then  Result := []
    else begin
        Result := [];
        for var i := 0 to Length(shape.dims)- 1 do
          Result := Result + [ dims[i] ];
    end;
end;

function TFTensor._slice(_sl: slice): TFTensor;
begin
    var slice_spec : TArray<Integer> := [ _sl.start ];
    var aBegin   := TList<Integer>.Create;
    var aEnd     := TList<Integer>.Create;
    var strides  := TList<Integer>.Create;
    try
      var index            : Integer := 0;
      var new_axis_mask    : Integer := 0;
      var shrink_axis_mask : Integer := 0;
      var begin_mask       : Integer := 0;
      var end_mask         : Integer := 0;
      var ellipsis_mask    : Integer := 0;
      for var s in slice_spec do
      begin
          aBegin.Add(s);
          if _sl.Stop.HasValue then
          begin
              aEnd.Add(_sl.Stop.Value);
          end else
          begin
              aEnd.Add(0);
              end_mask := end_mask or (1 shl index);
          end;

          strides.Add(_sl.Step);

          Inc(index);
      end;

      var vValues : TArray<TValue>;
      vValues := vValues + [ TValue.From< TList<Integer> >(aBegin) ];
      vValues := vValues + [ TValue.From< TList<Integer> >(aEnd) ];
      vValues := vValues + [ TValue.From< TList<Integer> >(strides) ];
      var newVal : TValue := TValue.From<TArray<TValue>>(vValues);

      Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'strided_slice', @newVal),
                                            function(v1: TNameScope): TFTensor
                                              begin
                                                  var name : string := v1.ToString;
                                                  if aBegin <> nil then
                                                  begin
                                                      var packed_begin  := array_ops.stack( TValue.from< TArray<Integer> >(aBegin.ToArray) );
                                                      var packed_end    := array_ops.stack( TValue.from< TArray<Integer> >(aEnd.ToArray) );
                                                      var packed_strides:= array_ops.stack( TValue.from< TArray<Integer> >(strides.ToArray) );
                                                      Result := gen_array_ops.strided_slice(self,
                                                                                            packed_begin,
                                                                                            packed_end,
                                                                                            packed_strides,
                                                                                            begin_mask,
                                                                                            end_mask,
                                                                                            ellipsis_mask,
                                                                                            new_axis_mask,
                                                                                            shrink_axis_mask,
                                                                                            name);
                                                      Exit;
                                                  end;
                                                  raise  TFException.Create('Not Implemented');
                                              end );
    finally
      aBegin.Free;
      aEnd.Free;
      strides.Free;
    end;
end;

function TFTensor._slice(start: Integer): TFTensor;
begin
    var slice_spec : TArray<Integer> := [ start ];
    var aBegin   := TList<Integer>.Create;
    var aEnd     := TList<Integer>.Create;
    var strides  := TList<Integer>.Create;
    try
      var index            : Integer := 0;
      var new_axis_mask    : Integer := 0;
      var shrink_axis_mask : Integer := 0;
      var begin_mask       : Integer := 0;
      var end_mask         : Integer := 0;
      var ellipsis_mask    : Integer := 0;
      for var s in slice_spec do
      begin
          aBegin.Add(s);
          aEnd.Add(s + 1);
          strides.Add(1);
          shrink_axis_mask := shrink_axis_mask or (1 shl index);
          Inc(index);
      end;

      var vValues : TArray<TValue>;
      vValues := vValues + [ TValue.From< TList<Integer> >(aBegin) ];
      vValues := vValues + [ TValue.From< TList<Integer> >(aEnd) ];
      vValues := vValues + [ TValue.From< TList<Integer> >(strides) ];
      var newVal : TValue := TValue.From<TArray<TValue>>(vValues);

      Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'strided_slice', @newVal),
                                            function(v1: TNameScope): TFTensor
                                              begin
                                                  var name : string := v1.ToString;
                                                  if aBegin <> nil then
                                                  begin
                                                      var packed_begin  := array_ops.stack( TValue.from< TArray<Integer> >(aBegin.ToArray) );
                                                      var packed_end    := array_ops.stack( TValue.from< TArray<Integer> >(aEnd.ToArray) );
                                                      var packed_strides:= array_ops.stack( TValue.from< TArray<Integer> >(strides.ToArray) );
                                                      Result := gen_array_ops.strided_slice(self,
                                                                                            packed_begin,
                                                                                            packed_end,
                                                                                            packed_strides,
                                                                                            begin_mask,
                                                                                            end_mask,
                                                                                            ellipsis_mask,
                                                                                            new_axis_mask,
                                                                                            shrink_axis_mask,
                                                                                            name);
                                                      Exit;
                                                  end;
                                                  raise  TFException.Create('Not Implemented');
                                              end );
    finally
      aBegin.Free;
      aEnd.Free;
      strides.Free;
    end;
end;

function TFTensor.GetSize: UInt64;
begin
    if Handle = nil then Exit(0);

    Result := bytesize div dtypesize;
end;
{$ENDREGION}

{$REGION 'TNDArray'}

{ TNDArray }

constructor TNDArray.Create(const value: Integer);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Int64);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Single);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Boolean);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Byte);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Word);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(const value: Double);
begin
    inherited Create(value);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(tensor: TFTensor; clone: Boolean);
begin
    inherited Create(tensor.Handle, clone);
    if handle = nil then
    begin
        tensor := tf.get_default_session.eval(tensor);
        Handle := tensor.Handle;
    end;
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<Boolean>; shape: PTFShape);
begin
     inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape) );
     NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Boolean>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Boolean>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_BOOL;

    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Byte>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Byte>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Byte>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_UINT8;

    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int8>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Int8>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int8>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int8>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int8>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int8>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_UINT8;

    inherited Create( TFTensor.InitTensor<Int8>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int16>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int16>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int16>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT16;

    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<UInt16>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<UInt16>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<UInt16>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<UInt16>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<UInt16>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<UInt16>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT16;

    inherited Create( TFTensor.InitTensor<UInt16>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int32>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int32>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int32>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_INT32;

    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<UInt32>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<UInt32>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<UInt32>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<UInt32>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<UInt32>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<UInt32>(bytes);
        shape := @v;
    end;

    dtype:=TF_DataType.TF_INT32;

    inherited Create( TFTensor.InitTensor<UInt32>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Int64>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Int64>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Int64>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT64;

    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<UInt64>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<UInt64>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<UInt64>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<UInt64>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_INT64;

    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Single>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Single>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Single>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Single>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_FLOAT;

    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Double>; shape: PTFShape);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape) );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<Double>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<Double>>>; shape: PTFShape);
begin
    Create(TValue.From(Bytes),shape );
    NewEagerTensorHandle;
end;

constructor TNDArray.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: PTFShape);
var
  dtype : TF_DataType;
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TUtils.GetShape<Double>(bytes);
        shape := @v;
    end;

    dtype:= TF_DataType.TF_DOUBLE;

    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: TArray<Int64>; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(value,shape) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(address: Pointer; shape: PTFShape; dtype: TF_DataType);
begin
    inherited Create(address,shape^,dtype);
    NewEagerTensorHandle ;
end;

destructor TNDArray.Destroy;
begin
  if Assigned(EagerTensorHandle) then
    TFE_DeleteTensorHandle(EagerTensorHandle);
  inherited Destroy;
end;

function TNDArray.Equals(y: TNDArray): Boolean;
begin
    if ndim <> y.ndim        then exit(False)
    else if size <> y.size   then exit(False)
    else if dtype <> y.dtype then exit(False);
    Result := TUtils.SequenceEqual<byte>(ToByteArray, y.ToByteArray);
end;

function TNDArray.GetItem(indices: TArray<Integer>): TNDArray;
begin
     var aSlice : TArray<Slice> := [];
     for var i := 0 to Length(indices) - 1 do
     begin
        var x := indices[i];
        aSlice := aSlice + [ Slice.Create(x, x+1, 1, true) ];
     end;
     Result := GetData(aSlice)
end;

procedure TNDArray.SetItem(indices: TArray<Integer>; const Value: TNDArray);
begin
     var aSlice : TArray<Slice> := [];
     for var i := 0 to Length(indices) - 1 do
     begin
        var x := indices[i];
        if x < 0 then
          x := dims[0] + x;

        aSlice := aSlice + [ Slice.Create(x,x+1,1,true) ];
     end;
     SetData(aSlice,Value);
end;

function TNDArray.GetItem(indices: TArray<Slice>): TNDArray;
begin
    Result := GetData(indices)
end;

procedure TNDArray.SetItem(indice: Integer; const Value: TNDArray);
begin
    SetItem([indice],Value)
end;


procedure TNDArray.SetItem(indices: TArray<Slice>; const Value: TNDArray);
begin
    SetData(indices,Value);
end;

function TNDArray.GetItem(indice: Integer): TNDArray;
begin
    Result := GetData([Slice(indice)])
end;

function TNDArray.GetData(slices: TArray<Slice>): TNDArray;
begin
    if shape.IsScalar then
        Exit( GetScalar );
    var indices1 : TArray<Integer>;
    if Slice.AreAllIndex(slices, indices1) then
    begin
        var newshape := TFShape.GetShape(shape, slices);
        if newshape.IsScalar then
        begin
            var offset := TFShape.GetOffset(shape, indices1);
            Result := GetScalar( UInt64(offset) );
            Exit;
        end else
        begin
            Result := GetArrayData(newshape, indices1);
            Exit;
        end;
    end
    else if Length(slices) = 1 then
    begin
        var slice := slices[0];
        if slice.Step = 1 then
        begin
            var newshape := TFShape.GetShape(shape, [slice]);
            var ndarray :=  TNDArray.Create(newshape, dtype);
            var new_dims : TArray<Integer>;
            SetLength(new_dims,shape.ndim);
            if slice.Start = nil then  new_dims[0] := 0
            else                       new_dims[0] := slice.Start;
            var offset := TFShape.GetOffset(shape, new_dims);
            var src := PByte(data) + ( UInt64(offset) * dtypesize) ;
            var dst := PByte(ndarray.data);
            var len := UInt64(newshape.size) * dtypesize;
            CopyMemory(dst,src,len);
            Result := ndarray;
            Exit;
        end;
    end;
    // default, performance is bad
    var tensor := inherited Item[slices];
    if tensor.Handle = nil then
    begin
        if tf.executing_eagerly then
            tensor := tf.get_default_session.eval(tensor);
    end;
    Result := TNDArray.Create(tensor, tf.executing_eagerly);

end;

function TNDArray.GetData(indices: TArray<Integer>; axis: Integer): TNDArray;
begin
    if shape.IsScalar then
        exit( GetScalar );

    if axis = 0 then
    begin
        var dims := shape.as_int_list;
        dims[0] := Length(indices);

        var nd := np.np_ndarray(dims, dtype);

        dims[0] := 1;
        var len := TFShape.Create(dims).size * TDTypes.get_datatype_size(dtype);

        var dst_index : Integer := 0;
        for var _pos in indices do
        begin
            var src_offset : UInt64 := TFShape.GetOffset(shape, [_pos]);
            var dst_offset : UInt64 := TFShape.GetOffset(nd.shape, [dst_index]);
            Inc(dst_index);
            var src := PByte(data) + src_offset * dtypesize;
            var dst := PByte(nd.data) + dst_offset * dtypesize;
            CopyMemory(dst,src,len);
        end;

        Result := nd;
    end else
       raise Exception.Create('Not Implemented ');
end;

function TNDArray.GetData(Mask: TNDArray): TNDArray;
begin
    if mask.dtype = TF_DataType.TF_BOOL then
    begin
        var bArray   := Mask.ToArray<Boolean>;
        var intArray : TArray<Integer> := [];
        for var i := 0 to Length(bArray) -1  do
          if bArray[i] then
             intArray := intArray + [ i ];

        Result := GetData(intArray);
    end
    else if mask.dtype = TF_DataType.TF_INT32 then
    begin
        Result := GetData(mask.ToArray<Integer>)
    end
    else if mask.dtype = TF_DataType.TF_INT64 then
    begin
        var i64Array   := Mask.ToArray<Int64>;
        var intArray : TArray<Integer> := [];
        for var i := 0 to Length(i64Array) -1  do
           intArray := intArray + [ i64Array[i] ];

        Result := GetData(intArray);
    end
    else if mask.dtype = TF_DataType.TF_FLOAT then
    begin
        var sArray   := Mask.ToArray<Single>;
        var intArray : TArray<Integer> := [];
        for var i := 0 to Length(sArray) -1  do
           intArray := intArray + [ Trunc(sArray[i]) ];

        Result := GetData(intArray);
    end else
      raise Exception.Create('Not Implemented ');
end;

procedure TNDArray.SetData(Mask: TNDArray; value: TNDArray);
begin
    if mask.dtype = TF_DataType.TF_BOOL then
        MaskData(mask, value)
    else
        raise Exception.Create('Not Implemented ');
end;

procedure TNDArray.MaskData(Mask, value: TNDArray);
begin
    var masks := mask.ToArray<Boolean>;
    (*var s1 = new Shape(dims.Skip(mask.rank).ToArray());
    var val = tf.fill(s1, value).numpy();
    for (int i = 0; i < masks.Length; i++)
    {
        if (masks[i])
            this[i] = val;
    } *)
end;

function TNDArray.GetItem(mask: TNDArray): TNDArray;
begin
    Result := GetData(mask)
end;

procedure TNDArray.SetItem(mask: TNDArray; const Value: TNDArray);
begin
    SetData(mask, Value);
end;

function TNDArray.GetDataPointer: Pointer;
begin
    Result := TensorDataPointer;
end;

function TNDArray.GetIsEmpty: Boolean;
var
  enumerator: IEnumerator;
begin
   enumerator := GetEnumerator;
   Result := not enumerator.MoveNext;
end;

function TNDArray.GetElementType: PTypeInfo;
begin
   Result := PTypeInfo(TypeInfo(TNDArray))
end;

function TNDArray.GetEnumerator: IEnumerator;
begin
    Result := TEnumerator_NDArray.Create(Self);
end;

function TNDArray.GetCount: Integer;
var
  enumerator: IEnumerator;
begin
   Result := 0;
   enumerator := GetEnumerator;
   while enumerator.MoveNext do
     Inc(Result);
end;

function TNDArray.AsObject: TObject;
begin
    Result := Self;
end;

procedure TNDArray.SetData(slices: TArray<Slice>; aArray: TNDArray);
var
  indici : TArray<Integer>;
begin
    SetLength(indici,shape.ndim);
    SetData(aarray, slices, indici, -1);
end;

procedure TNDArray.SetData(src: TNDArray; slices: TArray<Slice>; indices: TArray<Integer>; currentNDim: integer);
var
  i, offset, srcIndex, step: Integer;
  start, stop : TNullableInteger;
  dst         : Pointer;
  Locslice    : Slice;
begin
    if (dtype <> src.dtype) then
      raise Exception.CreateFmt('Required dtype %s but %s is assigned.', [Tdtypes.ToString(Dtype) , Tdtypes.ToString(src.dtype)]);

    if (Length(slices) = 0) then
      Exit;

    if (shape = src.shape) then
    begin
        System.Move(src.data^, data^, src.bytesize);
        Exit;
    end;

    // first iteration
    if (currentNDim = -1) then
      slices := Slice.AlignWithShape(shape, slices);

    // last dimension
    if (currentNDim = ndim - 1) then
    begin
        offset := TFShape.GetOffset(shape, indices);
        dst := Pointer(NativeInt(data) + offset * Integer(dtypesize));
        System.Move(src.data^, dst^, src.bytesize);
        Exit;
    end;

    Inc(currentNDim);
    Locslice := slices[currentNDim];

    start := Locslice.Start;
    if start = nil then  start := 0;

    stop := Locslice.Stop;
    {$WARN SYMBOL_DEPRECATED OFF}
    if stop = nil then stop := dims[currentNDim];
    {$WARN SYMBOL_DEPRECATED ON}

    step := Locslice.Step;

    if (step <> 1) then
    begin
        for i := start to stop.value - 1 do
        begin
            if (i >= dims[currentNDim]) then
              raise Exception.CreateFmt('Index should be in [0, %d] but got %d', [dims[currentNDim], i]);

            indices[currentNDim] := i;
            if (currentNDim < ndim - src.ndim) then
              SetData(src, slices, indices, currentNDim)
            else
            begin
                srcIndex := (i - start.value) div step;
                SetData(src[srcIndex], slices, indices, currentNDim);
            end;
        end;
    end
    else
    begin
        for i := start to stop.value - 1 do
        begin
            if (i >= dims[currentNDim]) then
              raise Exception.CreateFmt('Index should be in [0, %d] but got %d', [dims[currentNDim], i]);

            indices[currentNDim] := i;
            if (currentNDim < ndim - src.ndim) then
              SetData(src, slices, indices, currentNDim)
            // last dimension
            else if (currentNDim = ndim - 1) then
            begin
                SetData(src, slices, indices, currentNDim);
                Break;
            end
            else if (Slice.IsContinuousBlock(slices, currentNDim)) then
            begin
                offset := TFShape.GetOffset(shape, indices);
                dst := Pointer(NativeInt(data) + offset * Integer(dtypesize));
                System.Move(src.data^, dst^, src.bytesize);
                Exit;
            end else
            begin
                srcIndex := i - start.value;
                SetData(src[srcIndex], slices, indices, currentNDim);
            end;
        end;
    end;
    // reset indices
    indices[currentNDim] := 0;
end;

function TNDArray.GetScalar(offset: UInt64): TNDArray;
begin
    var nd_array := TNDArray.Create(Shape.Scalar, dtype);
    var src     := PByte(data) + (offset * dtypesize);
    CopyMemory(nd_array.TensorDataPointer,src,dtypesize);
    Result := nd_array;
end;

function TNDArray.GetArrayData(newshape: TFShape; indices: TArray<Integer>): TNDArray;
begin
    var offset := TFShape.GetOffset(shape, indices);
    var len := UInt64(newshape.size) * dtypesize;
    var nd_array := TNDArray.Create(newshape, dtype);
    var src     := PByte(data) + (UInt64(offset) * dtypesize);
    CopyMemory(nd_array.TensorDataPointer,src,len);
    Result := nd_array;
end;

constructor TNDArray.Create(bytes: TArray<TF_TString>; shape: PTFShape);
begin
    var v : TFShape;
    if shape = nil then
    begin
        v := TFShape.Create([Length(bytes)]);
        shape := @v ;
    end;

    inherited Create( StringTensor(bytes,shape) );
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(value: TValue; shape: PTFShape);
var
  aDim  : Integer;
  dtype : TF_DataType;
  vValue: TValue;
  lIsArray : Boolean;
begin
    aDim     := 0;
    vValue   := value;
    lIsArray := False;
    dtype    := dtInvalid;
    while vValue.IsArray do
    begin
        if value.GetArrayLength < 1 then
        begin
           var ttt := value.TypeData^.DynArrElType^ ;
           dtype := TDTypes.as_tf_dtype(ttt);
           Break;
        end;

        lIsArray := True;
        vValue := vValue.GetArrayElement(0);
        inc(aDim)
    end;

    var v : TFShape;
    if (shape = nil) or (shape.IsNil) then
    begin
        v := TUtils.GetShape(value);
        shape := @v;
    end;

    if value.GetArrayLength >0 then
      dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
        NewEagerTensorHandle ;
        Exit;
    end;

    if (lIsArray) and (dtype = TF_DOUBLE) and (string.LowerCase(vValue.TypeInfo.Name).Contains('extended')) then
    begin
        case aDim of
          1: begin
             var aValue : TArray<Double> := [];
             var a := value.AsType< TArray<Extended> > ;
             for var i := 0 to Length(a)- 1 do
               aValue := aValue + [ Double(a[i]) ];

             Value := TValue.From< TArray<Double> >(aValue);
          end;
          2:begin
             var aValue : TArray< TArray<Double> > ;
             var a := value.AsType< TArray< TArray<Extended>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                   aValue[i] := aValue[i] + [ Double(a[i][j]) ];
             end;

             Value := TValue.From< TArray<TArray<Double>> >(aValue);
          end;
          3: begin
             var aValue : TArray<TArray<TArray<Single>>> ;
             var a := value.AsType< TArray<TArray<TArray<Single>>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                 begin
                      SetLength(aValue[i],Length(aValue[i])+1);
                      for var k := 0 to Length(a[i][j])- 1 do
                        aValue[i][j] := aValue[i][j] + [ Double(a[i][j][k]) ];
                 end;
             end;
          end;
        end;
    end;


    case aDim of
       0 : begin
         case dtype of
           TF_FLOAT:  inherited Create( Single(value.AsExtended) );
           TF_DOUBLE: inherited Create( double(value.AsExtended) );
           TF_INT32:  inherited Create( value.AsInteger );
           TF_UINT8:  inherited Create( Byte(value.AsOrdinal) );
           TF_INT16:  inherited Create( int16(value.AsOrdinal) );
           TF_INT8:   inherited Create( int8(value.AsOrdinal) ) ;
           TF_STRING: inherited Create( AnsiString(value.AsString) );
           TF_INT64:  inherited Create( value.AsInt64 );
           TF_BOOL:   inherited Create( value.AsBoolean );
           TF_UINT16: inherited Create( word(value.AsOrdinal) );
           TF_UINT32: inherited Create( Cardinal(value.AsOrdinal) );
           TF_UINT64: inherited Create( value.AsUInt64 );
         end;
       end;
       1 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<Single> >,shape);
           TF_DOUBLE: Create( value.AsType< TArray<Double> >, shape);
           TF_INT32:  Create( value.AsType< TArray<Int32> > , shape);
           TF_UINT8:  Create( value.AsType< TArray<UInt8> >,shape);
           TF_INT16:  Create( value.AsType< TArray<Int16> >,shape);
           TF_INT8:   Create( value.AsType< TArray<Int8> >,shape);
           TF_STRING: Create( TFTensor.InitTensor<string>(value.AsType< TArray<string> >, shape,dtype) ); //Create( value.AsType< TArray<string> >,shape);
           TF_INT64:  Create( value.AsType< TArray<Int64> >,shape);
           TF_BOOL:   Create( value.AsType< TArray<Boolean> >,shape);
           TF_UINT16: Create( value.AsType< TArray<UInt16> >,shape);
           TF_UINT32: Create( value.AsType< TArray<UInt32> >,shape);
           TF_UINT64: Create( value.AsType< TArray<UInt64> >,shape);
         end;
       end;
       2 : begin
         case dtype of
           TF_FLOAT:  Create( TFTensor.InitTensor<Single>(value.AsType< TArray<TArray<Single>> >,  shape,dtype) );
           TF_DOUBLE: Create( TFTensor.InitTensor<Double>(value.AsType< TArray<TArray<Double>> >,  shape,dtype) );
           TF_INT32:  Create( TFTensor.InitTensor<Int32>(value.AsType< TArray<TArray<Int32>> >,    shape,dtype) );
           TF_UINT8:  Create( TFTensor.InitTensor<UInt8>(value.AsType< TArray<TArray<UInt8>> >,    shape,dtype) );
           TF_INT16:  Create( TFTensor.InitTensor<Int16>(value.AsType< TArray<TArray<Int16>> >,    shape,dtype) );
           TF_INT8:   Create( TFTensor.InitTensor<Int8>(value.AsType< TArray<TArray<Int8>> >,      shape,dtype) );
           TF_STRING: Create( TFTensor.InitTensor<string>(value.AsType< TArray<TArray<string>> >,  shape,dtype) );
           TF_INT64:  Create( TFTensor.InitTensor<Int64>(value.AsType< TArray<TArray<Int64>> >,    shape,dtype) );
           TF_BOOL:   Create( TFTensor.InitTensor<Boolean>(value.AsType< TArray<TArray<Boolean>> >,shape,dtype) );
           TF_UINT16: Create( TFTensor.InitTensor<UInt16>(value.AsType< TArray<TArray<UInt16>> >,  shape,dtype) );
           TF_UINT32: Create( TFTensor.InitTensor<UInt32>(value.AsType< TArray<TArray<UInt32>> >,  shape,dtype) );
           TF_UINT64: Create( TFTensor.InitTensor<UInt64>(value.AsType< TArray<TArray<UInt64>> >,  shape,dtype) );
         end;
         NewEagerTensorHandle ;
       end;
       3 : begin
         case dtype of
           TF_FLOAT:  Create( TFTensor.InitTensor<Single>(value.AsType< TArray<TArray<TArray<Single>>> >,  shape,dtype) );
           TF_DOUBLE: Create( TFTensor.InitTensor<Double>(value.AsType< TArray<TArray<TArray<Double>>> >,  shape,dtype) );
           TF_INT32:  Create( TFTensor.InitTensor<Int32>(value.AsType< TArray<TArray<TArray<Int32>>> >,    shape,dtype) );
           TF_UINT8:  Create( TFTensor.InitTensor<UInt8>(value.AsType< TArray<TArray<TArray<UInt8>>> >,    shape,dtype) );
           TF_INT16:  Create( TFTensor.InitTensor<Int16>(value.AsType< TArray<TArray<TArray<Int16>>> >,    shape,dtype) );
           TF_INT8:   Create( TFTensor.InitTensor<Int8>(value.AsType< TArray<TArray<TArray<Int8>>> >,      shape,dtype) );
           TF_STRING: Create( TFTensor.InitTensor<string>(value.AsType< TArray<TArray<TArray<string>>> >,  shape,dtype) );
           TF_INT64:  Create( TFTensor.InitTensor<Int64>(value.AsType< TArray<TArray<TArray<Int64>>> >,    shape,dtype) );
           TF_BOOL:   Create( TFTensor.InitTensor<Boolean>(value.AsType< TArray<TArray<TArray<Boolean>>> >,shape,dtype) );
           TF_UINT16: Create( TFTensor.InitTensor<UInt16>(value.AsType< TArray<TArray<TArray<UInt16>>> >,  shape,dtype) );
           TF_UINT32: Create( TFTensor.InitTensor<UInt32>(value.AsType< TArray<TArray<TArray<UInt32>>> >,  shape,dtype) );
           TF_UINT64: Create( TFTensor.InitTensor<UInt64>(value.AsType< TArray<TArray<TArray<UInt64>>> >,  shape,dtype) );
         end;
         NewEagerTensorHandle ;
       end;
       4 : begin
         case dtype of
           TF_FLOAT:  Create( TFTensor.InitTensor<Single>(value.AsType< TArray<TArray<TArray<TArray<Single>>>> >,  shape,dtype) );
           TF_DOUBLE: Create( TFTensor.InitTensor<Double>(value.AsType< TArray<TArray<TArray<TArray<Double>>>> >,  shape,dtype) );
           TF_INT32:  Create( TFTensor.InitTensor<Int32>(value.AsType< TArray<TArray<TArray<TArray<Int32>>>> >,    shape,dtype) );
           TF_UINT8:  Create( TFTensor.InitTensor<UInt8>(value.AsType< TArray<TArray<TArray<TArray<UInt8>>>> >,    shape,dtype) );
           TF_INT16:  Create( TFTensor.InitTensor<Int16>(value.AsType< TArray<TArray<TArray<TArray<Int16>>>> >,    shape,dtype) );
           TF_INT8:   Create( TFTensor.InitTensor<Int8>(value.AsType< TArray<TArray<TArray<TArray<Int8>>>> >,      shape,dtype) );
           TF_STRING: Create( TFTensor.InitTensor<string>(value.AsType< TArray<TArray<TArray<TArray<string>>>> >,  shape,dtype) );
           TF_INT64:  Create( TFTensor.InitTensor<Int64>(value.AsType< TArray<TArray<TArray<TArray<Int64>>>> >,    shape,dtype) );
           TF_BOOL:   Create( TFTensor.InitTensor<Boolean>(value.AsType< TArray<TArray<TArray<TArray<Boolean>>>> >,shape,dtype) );
           TF_UINT16: Create( TFTensor.InitTensor<UInt16>(value.AsType< TArray<TArray<TArray<TArray<UInt16>>>> >,  shape,dtype) );
           TF_UINT32: Create( TFTensor.InitTensor<UInt32>(value.AsType< TArray<TArray<TArray<TArray<UInt32>>>> >,  shape,dtype) );
           TF_UINT64: Create( TFTensor.InitTensor<UInt64>(value.AsType< TArray<TArray<TArray<TArray<UInt64>>>> >,  shape,dtype) );
         end;
         NewEagerTensorHandle ;
       end;

    end;

end;

constructor TNDArray.Create(shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(shape,dtype);
    NewEagerTensorHandle ;
end;

constructor TNDArray.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TNDArray.InitTensor<Byte>(bytes,shape ,dtype) );
    NewEagerTensorHandle ;
end;

procedure TNDArray.NewEagerTensorHandle;
begin
    if Assigned(Handle) then
      eagerTensorHandle := TEagerTensor.Create(Handle,true).EagerTensorHandle;
end;

function TNDArray.astype(dtype: TF_DataType): TNDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create( math_ops.cast(self, dtype) );

    if FChangedMode then
       tf.Context.restore_mode;
end;

function TNDArray.reshape(newshape: TFShape): TNDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create( tf.reshape(self, newshape) ) ;

    if FChangedMode then
      tf.Context.restore_mode;
end;

class function TNDArray.Scalar<T>(value: T): TNDArray;
begin
   var ttipo: pTypeInfo := TypeInfo(T);
   var x := TValue.From<T>(value);

   if      ttipo = TypeInfo(Boolean) then  Result := TNDArray.Create(x.AsBoolean)
   else if ttipo = TypeInfo(byte)    then  Result := TNDArray.Create(Byte(x.AsInteger))
   else if ttipo = TypeInfo(Integer) then  Result := TNDArray.Create(x.AsInteger)
   else if ttipo = TypeInfo(Int64)   then  Result := TNDArray.Create(x.AsInt64)
   else if ttipo = TypeInfo(single)  then  Result := TNDArray.Create(Single(x.AsExtended))
   else if ttipo = TypeInfo(Double)  then  Result := TNDArray.Create(Double(x.AsExtended))
   else
     raise TFException.Create('NotImplementedException');
end;

function TNDArray.To_Array: TArray<TNDArray>;
var
  Value: TNDArray;
begin
  Result := nil;
  for Value in Self do
   Result := Result + [ Value ];
end;

function TNDArray.ToByteArray: TArray<Byte>;
begin
    Result :=  BufferToArray;
end;

{$ENDREGION}

{$REGION 'Tensors.Ragged'}
{ SparseTensor }

constructor TSparseTensor.Create(indices_, values_, dense_shape_: TFTensor);
begin
    self.indices    := indices_;
    self.values     := values_;
    self.dense_shape:= dense_shape_;
    _init();
end;

constructor TSparseTensor.Create(indices_: TArray<TArray<Int64>>; values_: TValue; dense_shape_: TArray<Int64>);
begin
   Self := TUtils.tf_with<TNameScope,TSparseTensor>( TOps.name_scope('', 'SparseTensor'),
              function(v1: TNameScope): TSparseTensor
                begin
                    Result.indices    := Tops.convert_to_tensor(TValue.From< TArray<TArray<Int64>> >( indices_), TDtypes.cint64, 'indices');
                    Result.values     := Tops.convert_to_tensor(values_, TF_DataType.DtInvalid, 'values');
                    Result.dense_shape:= Tops.convert_to_tensor(TValue.From< TArray<Int64> >(dense_shape_), TDtypes.cint64, 'dense_shape');
                end );
    _init;
end;

procedure TSparseTensor._init;
begin
    var indices_shape    := indices.shape.with_rank(2);
    var values_shape     := values.shape.with_rank(1);
    var dense_shape_shape:= dense_shape.shape.with_rank(1);
    indices_shape['0'].merge_with(TFShape.Create( [ values_shape[0] ]));
    indices_shape['1'].merge_with(TFShape.Create( [ dense_shape_shape[0] ] ));
end;

{ Dimension }

constructor Dimension.Create(value_: Int64);
begin
    Fvalue := value_;
end;

class operator Dimension.implicit(value_: Int64): Dimension;
begin
    Result := Dimension.Create(value_)
end;

class operator Dimension.implicit(value_: Dimension): Int64;
begin
    Result := value_.value
end;

function Dimension.merge_with(other: Dimension): Dimension;
begin
    if Fvalue = -1 then Result := Dimension.Create(other.value)
    else                Result := Dimension(Fvalue);
end;

function Dimension.ToString: string;
begin
    Result := 'Dimension('+ IntToStr(Fvalue) +')';
end;

{ RowPartition }

constructor RowPartition.Create(row_splits, row_lengths, value_rowids, nrows, uniform_row_length: TFTensor);
begin
    Frow_splits   := row_splits;
    Frow_lengths  := row_lengths;
    Fvalue_rowids := value_rowids;
    Fnrows        := nrows;
end;

class function RowPartition.from_value_rowids(value_rowids, nrows: TFTensor; validate: Boolean; preferred_dtype: TF_DataType): RowPartition;
begin
    Result := TUtils.tf_with<TNameScope,RowPartition>( TOps.name_scope('', 'RowPartitionFromValueRowIds'),
                  function(v1: TNameScope): RowPartition
                    begin
                        var value_rowids_int32 := math_ops.cast(value_rowids, Tdtypes.cint32);
                        var nrows_int32        := math_ops.cast(nrows, Tdtypes.cint32);
                        var row_lengths        := tf.math.bincount(value_rowids_int32, nil, nrows_int32, nrows_int32,  value_rowids.dtype);
                        var a1 := Tops.convert_to_tensor( TArray<Int64>.create( 0 ) );
                        var a2 := tf.cumsum(row_lengths);
                        var row_splits := array_ops.concat([ a1,a2 ], 0);
                        Result := RowPartition.Create(row_splits, row_lengths, value_rowids, nrows);
                    end );
end;

class function RowPartition.from_row_splits(row_splits: TFTensor; validate: Boolean; preferred_dtype: TF_DataType): RowPartition;
begin
    Result := TUtils.tf_with<TNameScope,RowPartition>( TOps.name_scope('', 'RowPartitionFromRowSplits'),
                  function(v1: TNameScope): RowPartition
                    begin
                        Result :=  RowPartition.Create(row_splits);
                    end );
end;

function RowPartition.GetStaticRow: Integer;
begin
    Result := Frow_splits.shape[0] - 1;
end;

function RowPartition.GetSt_uni_row_l: Integer;
begin
    Result := -1
end;

{ RaggedTensor }

constructor RaggedTensor.Create(values: TFTensor; internal: Boolean; row_partition: RowPartition);
begin
     Fvalues        := values;
     Frow_partition := row_partition;
end;

class function RaggedTensor.FromTensor(t: TFTensor): RaggedTensor;
begin
    Result := t.Tag.AsType<RaggedTensor>;
end;

function RaggedTensor.ToTensor: TFTensor;
begin
    Result := Self._to_variant;
end;

class function RaggedTensor.from_row_partition(values: TFTensor; row_partition: RowPartition; validate: Boolean): RaggedTensor;
begin
    Result := RaggedTensor.Create(values, true, row_partition);
end;

function RaggedTensor.GetDtype: TF_DataType;
begin
    Result := Fvalues.Dtype;
end;

function RaggedTensor.GetRow_splits: TFTensor;
begin
    Result := Frow_partition.row_splits
end;

function RaggedTensor.getShape: TFShape;
begin
    var nrows := Frow_partition.static_nrows;
    var ncols := Frow_partition.static_uniform_row_length;
    Result := TFShape.Create([nrows, ncols]);
end;

function RaggedTensor._ragged_getitem(row_key: Integer): TFTensor;
begin
    var starts := Frow_splits[':-1'];
    var limits := Frow_splits['1:'];
    var row    := Fvalues[ starts[row_key], limits[row_key] ];
    Result := row;
end;

function RaggedTensor._ragged_getitem_inner_dimensions(input: RaggedTensor; slices: TArray<Slice>): RaggedTensor;
begin
    Result := input;
end;

function RaggedTensor._to_variant(batched_input: Boolean; name: string): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'RaggedToVariant'),
                  function(v1: TNameScope): TFTensor
                    begin
                        var Args := ExecuteOpArgs.Create([nested_row_splits, flat_values]);
                        Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                       begin
                                           Result := [];
                                           var pParam : TParameter;

                                           pParam.sNome := 'RAGGED_RANK' ;
                                           pParam.vValue:= op.get_attr<Integer>('RAGGED_RANK');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'Tvalues' ;
                                           pParam.vValue:= op.get_attr('Tvalues');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'Tsplits' ;
                                           pParam.vValue:= op.get_attr('Tsplits');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'batched_input' ;
                                           pParam.vValue:= op.get_attr<Boolean>('batched_input');
                                           Result := Result + [ pParam ] ;
                                       end;

                        Result := tf.Context.ExecuteOp('RaggedTensorToVariant', name, Args.SetAttributes(['batched_input',batched_input])).First;
                    end );
end;

class function RaggedTensor.from_value_rowids(values: TFTensor; value_rowids: TFTensor; nrows: TFTensor; name: string; validate: Boolean): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'RaggedFromValueRowIds'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        var row_partition := RowPartition.from_value_rowids(value_rowids, nrows, validate);
                        Result := from_row_partition(values, row_partition, validate);
                    end );
end;

class function RaggedTensor.from_row_splits(values, row_splits: TFTensor; name: string; validate: Boolean): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'RaggedFromRowSplits'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        var row_partition := RowPartition.from_row_splits(row_splits, validate);
                        Result := from_row_partition(values, row_partition, validate);
                    end );
end;

function RaggedTensor.GetItem(index: Integer): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'RaggedGetItem'),
                  function(v1: TNameScope): TFTensor
                    begin
                        Result   := _ragged_getitem(index);
                    end );
end;

function RaggedTensor.GetItem(slices: TArray<Slice>): RaggedTensor;
begin
    var row_key := slices[0];
    var inner_keys := slices;
    Delete(inner_keys,0,1);
    var args := TUtils.ParseSlices(slices);
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope('', 'RaggedGetItem'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        Result :=_ragged_getitem_inner_dimensions(self, inner_keys);
                    end );
end;

function RaggedTensor.getNest_row_splits: TArray<TFTensor>;
begin
    Result := [ Frow_splits ]
end;

{ tensor_shape }

class function tensor_shape.dimension_at_index(shape: TFShape; index: Integer): Dimension;
begin
    if   shape.ndim < 0  then Result := Dimension.Create(-1)
    else                      Result := Dimension.Create(shape.dims[index])
end;

class function tensor_shape.dimension_value(dimension: Dimension): Integer;
begin
    Result := dimension.value
end;

{ BodyItem }

constructor BodyItem.Create;
begin

end;

constructor BodyItem.Create(v_I: TFTensor; v_accs_ta: TArray<TTensorArray>);
begin
    FI       := v_I;
    FAccs_ta := v_accs_ta;
end;

function BodyItem.Flatten: TArray<TValue>;
begin
    var elements := TList<TValue>.Create([ FI ]);
    var a : TArray<TValue> := [];
    for var i := 0 to Length(FAccs_ta)-1 do
        a := a + [ TValue.From<TTensorArray>(FAccs_ta[i]) ];

    elements.AddRange(a);
    Result := elements.ToArray;
end;

function BodyItem.FromMergeVars(mergeVars: TArray<ITensorOrTensorArray>): BodyItem;
begin
    FI       := mergeVars[1] as TFTensor;
    FAccs_ta := [ mergeVars[2]  as TTensorArray ];
    Result := self;
end;

function BodyItem.Pack(sequences: TArray<TValue>): BodyItem;
begin
    FI       := sequences[0].AsType<TFTensor>;
    FAccs_ta := [ sequences[1].AsType<TTensorArray> ];

    Result := BodyItem.Create(FI, FAccs_ta);
end;

{ TTensorArray }

function TTensorArray.read<T>(index: T; name: string): TFTensor;
begin
 Result := nil;
end;

function TTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    Result := nil;
end;

{ TGraphTensorArray }

constructor TGraphTensorArray.Create(_dtype: TF_DataType; size: TFTensor; dynamic_size, clear_after_read: Boolean; tensor_array_name: string; _handle, _flow: TFTensor;
  _infer_shape: Boolean; _element_shape: PTFShape; _colocate_with_first_write_call: Boolean; _name: string);
begin
    Fclear_after_read := clear_after_read;
    Fdynamic_size     := dynamic_size;
    Fdtype            := _dtype;

    Fcolocate_with_first_write_call := _colocate_with_first_write_call;
    if Fcolocate_with_first_write_call then
        Fcolocate_with := TList<TFTensor>.Create;

    // Record the current static shape for the array elements. The element
    // shape is defined either by `element_shape` or the shape of the tensor
    // of the first write. If `infer_shape` is true, all writes checks for
    // shape equality.
    if _element_shape = nil then
    begin
        Finfer_shape   := _infer_shape;
        Felement_shape := TList<TFShape>.Create;
    end else
    begin
        Finfer_shape   := true;
        Felement_shape := TList<TFShape>.Create([ _element_shape ]);
    end;

    var vvalue := TValue.From< TArray<TFTensor> >([_handle, size, _flow]);
    TUtils.tf_with<TNameScope>( TOps.name_scope(_name, 'TensorArray', @vvalue),
        procedure(v1: TNameScope)
          begin
              var scope : string := v1.toString;
              if _handle <> nil then
              begin
                  Fhandle := _handle;
                  Fflow   := _flow;
              end else
              begin
                  var create : TFunc< Tuple<TFTensor, TFTensor> > := function:Tuple<TFTensor, TFTensor>
                        begin
                            Result :=  gen_data_flow_ops.tensor_array_v3(size, _dtype, _element_shape, dynamic_size, clear_after_read, _infer_shape, tensor_array_name, scope )
                        end;

                  // Construct the TensorArray with an empty device.  The first
                  // write into the TensorArray from a Tensor with a set device
                  // will retroactively set the device value of this op.
                  if _colocate_with_first_write_call then
                  begin
                      Tops.colocate_with(true);
                      var t1 := create;
                      Fhandle := t1.Value1;
                      Fflow   := t1.Value2;
                  end else
                  begin
                      var t1 := create;
                      Fhandle := t1.Value1;
                      Fflow   := t1.Value2;
                  end;
              end;
          end);
end;

function TGraphTensorArray.gather(indices: TFTensor; name: string): TFTensor;
var
  element_shape : TFShape;
  value         : TFTensor;
begin
    element_shape := TFShape.Null;

    if Felement_shape.Count > 0 then
       element_shape := Felement_shape[0];

    value := gen_data_flow_ops.tensor_array_gather_v3(Fhandle, indices, Fflow, Fdtype, @element_shape, name) ;

    //if (element_shape != null)
    //value.set_shape(-1, element_shape.dims);

    Result := value;
end;

function TGraphTensorArray.read<T>(index: T; name: string): TFTensor;
begin
    var value := gen_data_flow_ops.tensor_array_read_v3(Fhandle, constant_op.constant(TValue.From<T>(index)), Fflow, Fdtype, name);

    if Felement_shape <> nil then
        value.shape := Felement_shape[0].dims;

    Result := value;
end;

function TGraphTensorArray.scatter(indices, value: TFTensor; name: string): TTensorArray;
begin
    raise Exception.Create('Error Not Implemented scatter');
end;

function TGraphTensorArray.size(name: string): TFTensor;
begin
    Result := gen_data_flow_ops.tensor_array_size_v3(Fhandle, Fflow, name);
end;

function TGraphTensorArray.stack(name: string): TFTensor;
begin
    Tops.colocate_with(Fhandle);

    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'TensorArrayStack', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              var Limit : TValue :=  size;
                              Result := gather(math_ops.range(0, @Limit), name);
                          end);
end;

function TGraphTensorArray.unstack(value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayUnstack', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              var num_elements : TFTensor := array_ops.shape(value)[0];
                              var Limit : TValue :=  num_elements;
                              Result := scatter(math_ops.range(0, @Limit), value, name);
                          end);
end;

function TGraphTensorArray.write(index, value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle,index, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayWrite', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              _maybe_colocate_with(value);
                              var flow_out := gen_data_flow_ops.tensor_array_write_v3(Fhandle, index, value, Fflow, name);

                              Result := tensor_array_ops.build_ta_with_new_flow(Self, flow_out);
                          end);
end;

function TGraphTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    var value_tensor := Tops.convert_to_tensor(TValue.From<T>(value), DtInvalid, 'value', False, Fdtype );
    var index_tensor := Tops.convert_to_tensor(index, DtInvalid, 'index');
    Result := write(index_tensor, value_tensor, name);
end;

procedure TGraphTensorArray._maybe_colocate_with(value: TFTensor);
begin
    Fcolocate_with.Add(value);
end;

procedure TGraphTensorArray._merge_element_shape(shape: TFShape);
begin
    Felement_shape.Add(shape);
end;
{$ENDREGION}

{$REGION 'TEagerTensor'}
{ TEagerTensor }

constructor TEagerTensor.Create(h: Pointer);
begin
    FId  := TOps.uid;
    EagerTensorHandle := h;
    Resolve;
end;

constructor TEagerTensor.Create(h: Pointer;NewEagerTensor: Boolean);
begin
     NewEagerTensorHandle(h);
end;

constructor TEagerTensor.Create(shape: TFShape;dType: TF_DataType);
begin
    inherited Create(shape,dType);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Boolean>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Boolean>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor(shape,bytes,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Byte>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int16>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int16>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int32>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int32>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int64>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int64>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<UInt64>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<UInt64>>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<UInt64>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Single>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(TFTensor.InitTensor<Single>(bytes,shape,dtype));
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Single>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Single>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Double>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Double>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Double>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(value: TValue; shape: PTFShape);
var
  aDim  : Integer;
  dtype : TF_DataType;
  vValue: TValue;
  lIsArray : Boolean;
begin
    aDim     := 0;
    vValue   := value;
    lIsArray := False;
    dtype    := dtInvalid;
    while vValue.IsArray do
    begin
        if value.GetArrayLength < 1 then
        begin
           var ttt := value.TypeData^.DynArrElType^ ;
           dtype := TDTypes.as_tf_dtype(ttt);
           Break;
        end;
        lIsArray := True;
        vValue := vValue.GetArrayElement(0);
        inc(aDim)
    end;

    var v : TFShape;
    if (shape = nil) or (shape.IsNil) then
    begin
        v := TUtils.GetShape(value);
        shape := @v;
    end;

    if value.GetArrayLength > 0 then
      dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
        NewEagerTensorHandle(Handle);
        Exit;
    end;

    if (lIsArray) and (dtype = TF_DOUBLE) and (string.LowerCase(vValue.TypeInfo.Name).Contains('extended')) then
    begin
        case aDim of
          1: begin
             var aValue : TArray<Double> := [];
             var a := value.AsType< TArray<Extended> > ;
             for var i := 0 to Length(a)- 1 do
               aValue := aValue + [ Double(a[i]) ];

             Value := TValue.From< TArray<Double> >(aValue);
          end;
          2:begin
             var aValue : TArray< TArray<Double> > ;
             var a := value.AsType< TArray< TArray<Extended>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                   aValue[i] := aValue[i] + [ Double(a[i][j]) ];
             end;

             Value := TValue.From< TArray<TArray<Double>> >(aValue);
          end;
          3: begin
             var aValue : TArray<TArray<TArray<Single>>> ;
             var a := value.AsType< TArray<TArray<TArray<Single>>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                 begin
                      SetLength(aValue[i],Length(aValue[i])+1);
                      for var k := 0 to Length(a[i][j])- 1 do
                        aValue[i][j] := aValue[i][j] + [ Double(a[i][j][k]) ];
                 end;
             end;
          end;
        end;
    end;


    case aDim of
       0 : begin
         case dtype of
           TF_FLOAT:  Create( Single(value.AsExtended) );
           TF_DOUBLE: Create( double(value.AsExtended) );
           TF_INT32:  Create( value.AsInteger );
           TF_UINT8:  Create( Byte(value.AsOrdinal) );
           TF_INT16:  Create( int16(value.AsOrdinal) );
           TF_INT8:   Create( int8(value.AsOrdinal) ) ;
           TF_STRING: Create( AnsiString(value.AsString) );
           TF_INT64:  Create( value.AsInt64 );
           TF_BOOL:   Create( value.AsBoolean );
           TF_UINT16: Create( word(value.AsOrdinal) );
           TF_UINT32: Create( Cardinal(value.AsOrdinal) );
           TF_UINT64: Create( value.AsUInt64 );
         end;
       end;
       1 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<Single> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<Double> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<Int32> >,    shape,dtype) ;
           TF_UINT8:  Create( value.AsType< TArray<UInt8> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<Int16> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<Int8> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<string> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<Int64> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<Boolean> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<UInt16> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<UInt32> >,  shape,dtype);
           TF_UINT64: Create( value.AsType< TArray<UInt64> >,  shape,dtype) ;
         end;
       end;
       2 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<TArray<Single>> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<TArray<Double>> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<TArray<Int32>> >,    shape,dtype);
           TF_UINT8:  Create( value.AsType< TArray<TArray<UInt8>> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<TArray<Int16>> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<TArray<Int8>> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<TArray<string>> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<TArray<Int64>> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<TArray<Boolean>> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<TArray<UInt16>> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<TArray<UInt32>> >,  shape,dtype) ;
           TF_UINT64: Create( value.AsType< TArray<TArray<UInt64>> >,  shape,dtype) ;
         end;
       end;
       3 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<TArray<TArray<Single>>> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<TArray<TArray<Double>>> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<TArray<TArray<Int32>>> >,    shape,dtype) ;
           TF_UINT8:  Create( value.AsType< TArray<TArray<TArray<UInt8>>> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<TArray<TArray<Int16>>> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<TArray<TArray<Int8>>> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<TArray<TArray<string>>> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<TArray<TArray<Int64>>> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<TArray<TArray<Boolean>>> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<TArray<TArray<UInt16>>> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<TArray<TArray<UInt32>>> >,  shape,dtype);
           TF_UINT64: Create( value.AsType< TArray<TArray<TArray<UInt64>>> >,  shape,dtype) ;
         end;

       end;

    end;

end;


procedure TEagerTensor.copy_handle_data(target_t: TFTensor);
begin
    if (target_t.dtype = TF_DataType.TF_RESOURCE) or (target_t.dtype = TF_DataType.TF_VARIANT) then
    begin
        // need to export
        //(target_t.graph, target_t._as_tf_output(), 0, new IntPtr[0], new int[0], new DataType[0], tf.Status.Handle);
    end;
end;

function TEagerTensor.AsConstant(name: string): TFTensor;
begin
   Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(nil),
                                          function(v1: TControlDependenciesController): TFTensor
                                            begin
                                                Result := tf.constant(numpy, DtInvalid, nil, name)
                                            end );
end;

function TEagerTensor.AsPlaceholder(name: string): TFTensor;
begin
    var placeholder := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(nil),
                                          function(v1: TControlDependenciesController): TFTensor
                                            begin
                                                Result := tf.placeholder(dtype, nil, name)
                                            end );
    copy_handle_data(placeholder);
    Result := placeholder;
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

destructor TEagerTensor.Destroy;
begin
   inherited Destroy;
end;

procedure TEagerTensor.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TFE_DeleteTensorHandle(hnd);
end;

function TEagerTensor.GetDeviceName: string;
begin
    m_Device :=  string(AnsiString( TFE_TensorHandleDeviceName(EagerTensorHandle, tf.Status.Handle)));
    Result := m_Device;
end;

class function TEagerTensor.GetDims(eTensor: TEagerTensor): TArray<Integer>;
var
  dims : TArray<Integer>;

begin
    var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
    SetLength(dims, TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle));
    for var i := 0 to Length(dims)-1 do
        dims[i] := TFE_TensorHandleDim(tfe_tensor_handle, i, tf.Status.Handle);
    Result := dims;
end;

class function TEagerTensor.GetRank(eTensor: TEagerTensor): Integer;
begin
     var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
     Result := TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TF_TString>; shape: TFShape);
begin
    inherited Create( StringTensor(bytes,shape) );
    NewEagerTensorHandle(Handle);
end;

procedure TEagerTensor.NewEagerTensorHandle(h: Pointer);
begin
    FId := Tops.uid;
    EagerTensorHandle := TFE_NewTensorHandle(h,tf.Status.Handle) ;
    tf.Status.RaiseEx;
end;

procedure TEagerTensor.Resolve;
begin
    Handle := TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

function TEagerTensor.ToString: string;
begin
    var nd  := TNDArray.Create(Self);
    var str := NDArrayRender.ToString(nd);
    Result  := Format('tf.Tensor: shape=%s, dtype=%s, numpy=%s',[Shape.ToString, Tdtypes.ToString(dtype), str]);
end;

procedure TEagerTensor.Setshape(const Value: TFShape);
begin
    if not Shape.is_compatible_with(Value) then
        raise Exception.Create('Tensor''s shape is not compatible.');
end;

function TEagerTensor.GetShape: TFShape;
begin
    var sShape := System.default(TFShape);

    if rank < 0 then
        Exit(sShape);

    var numDims := TFE_TensorHandleNumDims(eagerTensorHandle, tf.Status.Handle);
    var dims : TArray<Integer> ;
    if numDims > 0 then
    begin
        SetLength(dims,numDims) ;

        for var i := 0 to Length(dims)- 1 do
          dims[i] := TFE_TensorHandleDim(eagerTensorHandle, i, tf.Status.Handle);
    end;
    Result := dims;
end;

{ TEagerTensorArray }

constructor TEagerTensorArray.Create(_dtype: TF_DataType; size: TFTensor; dynamic_size, clear_after_read: Boolean; tensor_array_name: string; _handle, _flow: TFTensor;
  _infer_shape: Boolean; _element_shape: PTFShape; _colocate_with_first_write_call: Boolean; _name: string);
begin
    Fflow             := constant_op.constant(Integer(0));
    Finfer_shape      := _infer_shape;
    if Assigned(_element_shape) then Felement_shape := _element_shape^
    else                             Felement_shape := TFShape.null;

    Fdtype            := Tdtypes.as_base_dtype(_dtype);
    Fdynamic_size     := dynamic_size;
    Fclear_after_read := clear_after_read;
    Ftensor_array     := TList<TFTensor>.Create;
    Fcolocate_with_first_write_call := colocate_with_first_write_call;
end;

function TEagerTensorArray.gather(indices: TFTensor; name: string): TFTensor;
var
  element_shape : TFShape;
  value         : TFTensor;
begin
    element_shape := TFShape.Null;

    value := gen_data_flow_ops.tensor_array_gather_v3(Fhandle, indices, Fflow, Fdtype, @element_shape, name) ;

    //if (element_shape != null)
    //value.set_shape(-1, element_shape.dims);

    Result := value;
end;

function TEagerTensorArray.read<T>(index: T; name: string): TFTensor;
var
  index_int : Integer;
begin
    index_int := -1;
    var v : TValue := TValue.From<T>(index);
    if v.TypeInfo = TypeInfo(Integer) then
    begin
        index_int := v.AsInteger
    end
    else if v.TypeInfo = TypeInfo(TFTensor) then
    begin
        var nd : NDArray := v.AsType<TFTensor>.numpy;
        index_int := nd;
    end else
        raise Exception.Create('read<T> Error');

    if Fclear_after_read then
      Ftensor_array[index_int] := nil;

    Result := Ftensor_array[index_int];
end;

function TEagerTensorArray.scatter(indices, value: TFTensor; name: string): TTensorArray;
begin
    raise Exception.Create('Error Not Implemented scatter');
end;

function TEagerTensorArray.size(name: string): TFTensor;
begin
    Result := gen_data_flow_ops.tensor_array_size_v3(Fhandle, Fflow, name);
end;

function TEagerTensorArray.stack(name: string): TFTensor;
begin
    Tops.colocate_with(Fhandle);

    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'TensorArrayStack', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              var Limit : TValue :=  size;
                              Result := gather(math_ops.range(0, @Limit), name);
                          end);
end;

function TEagerTensorArray.unstack(value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayUnstack', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              var num_elements : TFTensor := array_ops.shape(value)[0];
                              var Limit : TValue :=  num_elements;
                              Result := scatter(math_ops.range(0, @Limit), value, name);
                          end);
end;

function TEagerTensorArray.write(index, value: TFTensor; name: string): TTensorArray;
begin
    if Finfer_shape then
        Felement_shape := Felement_shape.merge_with(value.shape);

    Ftensor_array.add(value);

    Result := self;
end;

function TEagerTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    var value_tensor := Tops.convert_to_tensor(TValue.From<T>(value), DtInvalid, 'value', False, Fdtype );
    var index_tensor := Tops.convert_to_tensor(index, DtInvalid, 'index');
    Result := write(index_tensor, value_tensor, name);
end;

procedure TEagerTensorArray._maybe_colocate_with(value: TFTensor);
begin
    Fcolocate_with.Add(value);
end;

procedure TEagerTensorArray._merge_element_shape(shape: TFShape);
begin
    Felement_shape.concatenate(shape);
end;
{$ENDREGION}

{$REGION 'TFOperationDesc'}
//------------------------------------------------------------------------------
//----------------------------- TFOperationDesc --------------------------------
//------------------------------------------------------------------------------
constructor TFOperationDesc.Create(graph: TFGraph; opType, operName: TF_TString);
begin
   inherited Create(Nil);
   if not Assigned(graph) then
     raise TFException.Create('TFOperationDesc.Create: Argument Null Exception - graph');

   self.Handle := TF_NewOperation(graph.Handle, PAnsiChar(opType), PAnsiChar(operName));

   FGraph     := graph;
   FsOpType   := opType;
   FsOperName := operName;
end;
destructor TFOperationDesc.Destroy;
begin
 inherited Destroy;
end;
procedure TFOperationDesc.NativeDispose(hnd: Pointer);
begin
 if Assigned(Handle) then
   TensorFlow.DApiBase.WriteTFProt(Format('TFOperationDescription(%s,%s) was never turned into an TFOperation',[FsOpType,FsOperName]));
end;

procedure TFOperationDesc.SetAttrShape(attr_name: AnsiString; dims: TArray<Int64>);
begin
     TF_SetAttrShape(Handle, PAnsiChar(@attr_name), PInt64(@dims[0]), Length(dims));
end;

procedure TFOperationDesc.SetAttrType(attr_name: AnsiString; value: TF_DataType);
begin
    TF_SetAttrType(Handle, PAnsiChar(@attr_name), value);
end;

procedure TFOperationDesc.AddInput(input: TF_Output);
begin
     if not Assigned(Handle) then
       TFDisposable.ObjectDisposedException();

     TF_AddInput(self.Handle, input);
end;

procedure TFOperationDesc.AddInputs(inputs: TArray<TF_Output>);
begin
    if not Assigned(Handle) then
       TFDisposable.ObjectDisposedException();

    TF_AddInputList(self.Handle, PTF_Output(@inputs[0]), Length(inputs));
end;

function TFOperationDesc.FinishOperation(status: TFStatus = Nil): TFOperation;
var
 l_oStatus: TFStatus;
 l_pOp: PTF_Operation;
begin
   if not Assigned(Handle) then
     TFDisposable.ObjectDisposedException();

   l_oStatus := TFStatus.Setup(status);
   l_pOp := TF_FinishOperation(Handle, l_oStatus.Handle);
   l_oStatus.CheckMaybeRaise(status);

   Handle := Nil;
   if Assigned(status) and status.Error then
     Result := Nil
   else
     Result := TFOperation.Create(l_pOp);

   self.DisposeOf;
end;

function TFOperationDesc.GetHandleOperation: Pointer;
begin
    Result := Handle
end;
{$ENDREGION}

{$REGION 'TFOperation'}
//------------------------------------------------------------------------------
//----------------------------- TFOperation ------------------------------------
//------------------------------------------------------------------------------
constructor TFOperation.Create(node_def       : TNodeDef;
                               g              : TFGraph;
                               inputs         : TArray<TFTensor>;
                               output_types   : TArray<TF_DataType>;
                               control_inputs : TArray<ITensorOrOperation> ;
                               input_types    : TArray<TF_DataType> ;
                               original_op    : string;
                               op_def         : TOpDef );
begin
   inherited Create;
   Fgraph := g;
   // Build the list of control inputs.
   var control_input_ops := TList<TFOperation>.Create;
   try
     if control_inputs <>  nil then
     begin
         for var c in control_inputs do
         begin
              if c is TFOperation then
                 control_input_ops.Add(TFOperation(c))
              else if c is TFTensor then
                 control_input_ops.Add(TFTensor(c).op)
         end;
     end;
     Fid_value := FGraph.NextId;

     // Dict mapping op name to file and line information for op colocation
     // context managers.
     Fcontrol_flow_context := graph._get_control_flow_context;

     // This will be set by self.inputs.
     if op_def = nil then
     begin
       var op := g.GetOpDef(node_def.Op);
       op_def := op;
     end;

     var t := TOps._create_c_op(g, node_def, inputs, control_input_ops.ToArray, op_def);
     var _handle := t.Value1;
     //var pDesc   := t.Value2;
     Fis_stateful := op_def.IsStateful;

     Handle := _handle;

     // Initialize self._outputs.
     SetLength(output_types,NumOutputs) ;
     for var i := 0 to NumOutputs - 1 do
        output_types[i] := OutputType(i);

     SetLength(FOutputs,NumOutputs);
     for var i := 0 to NumOutputs - 1 do
        FOutputs[i] := TFTensor.Create(self, i, output_types[i]);

     FGraph.Add_op(Self);

     FOp := self;
     GetName;
     GetType;
     GetipoOp;

     if Handle <> nil then
         _control_flow_post_processing;

   finally
     control_input_ops.free;
   end;
end;

constructor TFOperation.Create(hnd: Pointer; graph: TFGraph);
begin
     inherited Create;

     if not Assigned(hnd) then
         Exit;

     Handle := hnd;
     if Assigned(graph)  then
        FGraph := graph
     else
        FGraph := TOps.get_default_graph;

     SetLength(FOutputs,NumOutputs);
     for var i := 0 to NumOutputs - 1 do
      FOutputs[i] := TFTensor.Create(self, i, OutputType(i));

     FOp := self;
     GetName;
     GetType;
     GetipoOp;

      // Dict mapping op name to file and line information for op colocation
      // context managers.
      Fcontrol_flow_context := FGraph._get_control_flow_context;
      // Note: _control_flow_post_processing() must not be called here, the caller is responsible for calling it when using this constructor.
end;
destructor  TFOperation.Destroy;
begin
 Handle := Nil;
 inherited Destroy;
end;

function TFOperation.GeNumOutputs: Integer;
begin
    Result := - 1;

    if Assigned(Handle) then
      Result := TF_OperationNumOutputs(Handle)
end;

function TFOperation.GetNumCtrlInputs: Integer;
begin
    Result := 0;
    if Assigned(Handle) then
      Result := TF_OperationNumControlInputs(Handle);
end;

function TFOperation.GetControlInputs: TArray<TFOperation>;
begin
    var ctrl_inputs : TArray<TFOperation>; SetLength(ctrl_inputs, NumControlInputs);

    if NumControlInputs > 0 then
    begin
        var control_input_handle : TArray<Pointer> ; SetLength(control_input_handle, NumControlInputs);
        TF_OperationGetControlInputs(Handle, @control_input_handle[0], NumControlInputs);
        for var i := 0 to NumControlInputs - 1 do
        begin
            var hHandle := control_input_handle[i];
            ctrl_inputs[i] := TFOperation.Create(hHandle);
        end;

    end;

    Result := ctrl_inputs;
end;

function TFOperation.GetCtrlInputs: TArray<TFOperation>;
begin
    if (Fcontrol_inputs = nil) or (Length(Fcontrol_inputs) = 0) then
        Fcontrol_inputs := GetControlInputs;
    Result := Fcontrol_inputs;
end;

function TFOperation.GetDevice: string;
begin
    FDevice := '';
    if Assigned(Handle) then
      FDevice := string( AnsiString(TF_OperationDevice(Handle)) ) ;

    Result := FDevice;
end;

function TFOperation.Input(index: Integer): TF_Output;
begin
    Result := TF_OperationInput( TF_Input.Create(Handle, index) );
end;

function TFOperation.InputType(index: Integer): TF_DataType;
begin
    Result := TF_OperationInputType( TF_Input.Create(Handle, index) );
end;

class function TFOperation.map_fn(fn: TFunc<TFTensor, TFTensor>; elems: TFTensor; dtype: TF_DataType; parallel_iterations: Integer; back_prop, swap_memory, infer_shape: Boolean;
  name: string): TFTensor;
var
  input_is_sequence           : Boolean;
  output_is_sequence          : Boolean;
  output_flatten,input_flatten: TFunc<TValue, TArray<TFTensor>>;
  output_pack,input_pack      : TFunc<TArray<TFTensor>, TFTensor> ;
  elems_flat                  : TArray<TFTensor>;
  dtype_flat                  : TArray<TF_DataType>;
  static_shape                : TFShape;
  elems_ta                    : TArray<TTensorArray>;
  accs_ta                     : TArray<TTensorArray>;
  cond                        : TFunc<BodyItem, TFTensor> ;
  compute                     : TFunc<BodyItem, BodyItem>;
  r_a                         : BodyItem;
  results_flat                : TArray<TFTensor>;
  n_static                    : Dimension;
begin
    input_is_sequence := nest.is_sequence(elems);
    input_flatten := function (x: TValue): TArray<TFTensor>
                           begin
                               var x_a : TFTensor := x ;
                               if input_is_sequence then Result := nest.flatten<TFTensor>(x_a).ToArray
                               else                      Result := [ x ];
                           end;
    input_pack := function (x: TArray<TFTensor>): TFTensor
                           begin
                               var lst := TList<TFTensor>.Create(x);
                               if input_is_sequence then Result := nest.pack_sequence_as(elems, TList<TObject>(lst)) as TFTensor
                               else                      Result := x[0] ;
                           end;

    if dtype = TF_DataType.DtInvalid then
    begin
        output_is_sequence := input_is_sequence;
        output_flatten     := input_flatten;
        output_pack        := input_pack;
    end else
    begin
        output_is_sequence := nest.is_sequence(dtype);

        output_flatten := function (x: TValue): TArray<TFTensor>
                           begin
                               var x_a : TFTensor := x ;
                               if output_is_sequence then Result := nest.flatten<TFTensor>(x_a).ToArray
                               else                       Result := [ x ];
                           end;
        output_pack := function (x: TArray<TFTensor>): TFTensor
                           begin
                               var lst := TList<TFTensor>.Create(x);
                               if output_is_sequence then Result := nest.pack_sequence_as(elems, TList<TObject>(lst)) as TFTensor
                               else                       Result := x[0] ;
                           end;
    end;

    elems_flat := input_flatten(elems);
    var vvalue := TValue.From<TArray<TFTensor>>(elems_flat);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'map', @vvalue),
        function(v1: TNameScope): TFTensor
          begin
              name := string(v1.ToString);
              //if in_graph_mode:
              //# Any get_variable calls in fn will cache the first call locally
              //# and not issue repeated network I/O requests for each iteration.
              //varscope = vs.get_variable_scope()
              //varscope_caching_device_was_none = False
              //if varscope.caching_device is None:
              //  # TODO(ebrevdo): Change to using colocate_with here and in other
              //  # methods.
              //  varscope.set_caching_device(lambda op: op.device)
              //  varscope_caching_device_was_none = True

              for var i := 0 to Length(elems_flat)-1 do
                elems_flat[i] := Tops.convert_to_tensor(elems_flat[i], DtInvalid, 'elem');

              dtype      := elems_flat[0].Dtype;
              dtype_flat := [ dtype ];

              // Convert elems to tensor array. n may be known statically.
              static_shape := elems_flat[0].shape;

              var n : Int64 := static_shape[0];

              // TensorArrays are always flat
              elems_ta := [];
              for var i := 0 to Length(elems_flat)-1 do
                   elems_ta := elems_ta + [ tf.TensorArray(elems_flat[i].dtype, n, false, True, nil, True, true) ];

              // Unpack elements
              var elems_ta_1 := TList<TTensorArray>.Create;
              try
                for var elem_ta_elem in TUtils.zip<TTensorArray,TFTensor>(elems_ta, elems_flat) do
                    elems_ta_1.Add (elem_ta_elem.Value1.unstack(elem_ta_elem.Value2) );

                elems_ta := elems_ta_1.ToArray;
              finally
                elems_ta_1.Free;
              end;

              var i := constant_op.constant(Integer(0));

              for var x := 0 to Length(dtype_flat)-1 do
                   accs_ta := accs_ta + [ tf.TensorArray(dtype_flat[x], n, false, True, nil, True, infer_shape) ];

              compute := function (item: BodyItem): BodyItem
                  var
                    packed_values    : TFTensor;
                    packed_fn_values : TFTensor;
                    flat_fn_values   : TArray<TFTensor>;
                  begin
                      var tmpA : TArray<TFTensor> :=[];
                      for var j := 0 to Length(elems_ta)-1 do
                         tmpA  := tmpA + [ elems_ta[j].read( item.I ) ];

                      packed_values    := input_pack(tmpA);
                      packed_fn_values := fn(packed_values);
                      //nest.assert_same_structure(dtype or elems, packed_fn_values)

                      flat_fn_values := output_flatten(packed_fn_values);
                      for var j := 0 to Length(item.Accs_ta) - 1 do
                      begin
                          item.Accs_ta[j].write(item.I, flat_fn_values[j]);
                      end;

                      Result := BodyItem.Create(TTensor(item.I) + Integer(1), item.Accs_ta);
                  end;

              cond  := function(x: BodyItem): TFTensor
                        begin
                           Result := TTensor(x.I) < n;
                        end;

              r_a := control_flow_ops.while_loop<BodyItem>(cond, compute, BodyItem.Create(i, accs_ta), [], parallel_iterations, back_prop, swap_memory, '', tf.constant(n));

              for var j := 0 to Length(r_a.Accs_ta) - 1 do
                   results_flat := results_flat + [ r_a.Accs_ta[j].stack ];

              n_static :=  Dimension.Create( tensor_shape.dimension_value(elems_flat[0].shape.with_rank_at_least(1).dims[0]) );

              for var j := 1 to Length(elems_flat) -1 do
              begin
                  var elem := elems_flat[j];
                  n_static.merge_with(Dimension.Create(tensor_shape.dimension_value(elem.shape.with_rank_at_least(1).dims[0])));
              end;

              for var r in results_flat do
              begin
                 var aDims : TArray<Int64>:= [];
                 for var j := 1 to Length(elems_flat) -1 do
                    aDims := aDims + [ r.dims[j] ];

                 r.shape := TFShape.Create([n_static]).concatenate(aDims);
              end;


              // todo get working when the above caching_device is fixed
              //if (in_graph_mode && varscope_caching_device_was_none) {
              //    varscope.set_caching_device(None);
              //}

              Result := output_pack(results_flat);

          end);

end;

function TFOperation.InputListLength(Name: string): Integer;
begin
    var num : Integer ;
    num := TF_OperationInputListLength(Handle, PAnsiChar(AnsiString(Name)) , tf.Status.Handle);
    tf.Status.RaiseEx;
    Result := num;
end;

function TFOperation.GeNumInputs: Integer;
begin
    if Handle = nil then  Exit(-1);

    Result := TF_OperationNumInputs(Handle);
end;

function TFOperation.GetInputList: TInputList;
begin
    if Finputs_val = nil  then
    begin
        var retval : TArray<TFTensor>;
        SetLength(retval, NumInputs);
        for var i := 0 to NumInputs - 1 do
        begin
            var tf_output := Input(i);
            var op        := GetOperation(tf_output.oper);
            retval[i]     := op.outputs[ tf_output.index ];
        end;
        Finputs_val := TInputList.Create(retval);
    end;
    Result := Finputs_val;
end;

function TFOperation.GetipoOp: string;
begin
    Result := '';
    if Assigned(Handle) then
      Result := string( AnsiString(TF_OperationOpType(Handle)) ) ;
    FTipo := Result;
end;

function TFOperation.Getname: string;
begin
    Result := '';
    if Assigned(Handle) then
      Result := string( AnsiString(TF_OperationName(Handle)) ) ;
    FName := Result;
end;

function TFOperation.GetOperation(h: Pointer): TFOperation;
var
  nodes : TDictionary<string, ITensorOrOperation>;
begin
    nodes := tf.get_default_graph.nodes_by_name;
    for var node in nodes.Values do
    begin
        if node is TFOperation then
        begin
            var op := node as TFOperation ;
            if op.Handle = h then
                Exit(op);
        end;
    end;
    Result := TFOperation.Create(h);
end;

function TFOperation.GetOutput: TFTensor;
begin
    var lst := TList<TFTensor>.Create(FOutputs);

    Result := nil;
    if lst.Count > 0 then
       Result := lst.First
end;

function TFOperation.GetType: TF_DataType;
begin
   Result := DtInvalid;
   if Assigned(Output)  then
      Result := Output.dtype;
end;

function TFOperation.GetWhileContext: WhileContext;
begin
    Result := Fcontrol_flow_context as WhileContext;
end;

function TFOperation.get_attr(name: string): TValue;
var
  Loader  : TpbLoader;
  var buf : TFBuffer;
begin
    buf := TFBuffer.Create;
    try
      TF_OperationGetAttrValueProto(Handle, PAnsiChar(AnsiString(name)), buf.Handle, tf.Status.Handle);
      tf.Status.RaiseEx;
      var aBuf := buf.toArray;
      Loader.Init;
      Loader.Pb.Init(@aBuf[0],Length(aBuf),false);

      var AttrValue : TAttrValue ;
      loader.LoadAttrValue(AttrValue);

      Result := AttrValue.Value.value ;
    finally
     buf.Free;
    end;
end;

function TFOperation.get_attr<T>(name: string): T;
begin
     var v := get_attr(name);

     Result := v.AsType<T>;
end;

function TFOperation.get_attr_list<T>(name: string): TArray<T>;
begin
    if tf.executing_eagerly then
    begin
        var v := get_attr(name);

        Result := v.AsType<TArray<T>>;
        Exit;
    end;

    var buf := TFBuffer.Create;
    try
      TF_OperationGetAttrValueProto(Handle, PAnsiChar(AnsiString(name)), buf.Handle, tf.Status.Handle);
      tf.Status.RaiseEx;
      var aBuf := buf.toArray;

      var Loader  : TpbLoader;
      Loader.Init;
      Loader.Pb.Init(@aBuf[0],Length(aBuf),false);

      var AttrValue : TAttrValue ;
      loader.LoadAttrValue(AttrValue);

      if TypeInfo(T) = TypeInfo(Int32) then
      begin
          var lst :=   AttrValue.Value.value.AsType< TListValue > ;
          Result := [];
          for var i := 0 to lst.&Is.count-1 do
          begin
              var Lvalue := TValue.From<Int32>( Integer(lst.&Is[i]) );
              Result := Result + [ Lvalue.AsType<T>  ];
          end;
      end
      else if TypeInfo(T) = TypeInfo(Int64) then
      begin
          var lst :=   AttrValue.Value.value.AsType< TListValue > ;
          Result := [];
          for var i := 0 to lst.&Is.count-1 do
          begin
              var Lvalue := TValue.From<Int64>( lst.&Is[i] );
              Result := Result + [ Lvalue.AsType<T>  ];
          end;
      end
    finally
     buf.Free;
    end;
end;

function TFOperation.GetNodeDef: TNodeDef;
var
  Loader  : TpbLoader;
  var buf : TFBuffer;
begin
    buf := TFBuffer.Create;
    try
      TF_OperationToNodeDef(Handle, buf.Handle, tf.Status.Handle);
      tf.Status.RaiseEx;

      var aBuf := buf.toArray;
      Loader.Init;
      Loader.Pb.Init(@aBuf[0],Length(aBuf),false);
      // Loader.Pb.SaveToFile('test.proto');   For testing

      var NodeDef : TNodeDef ;
      loader.LoadNodeDef(NodeDef);

      Result := NodeDef ;
    finally
     buf.Free;
    end;

end;

procedure TFOperation.NativeDispose(hnd: Pointer);
begin
 // No Delete!
end;

function TFOperation.OutputType(index: Integer): TF_DataType;
begin
    Result := TF_OperationOutputType(_tf_output(index));
end;

procedure TFOperation.run(feed_dict: TArray<FeedItem>; session: TFSession);
begin
    Tops._run_using_default_session(Self, feed_dict, graph, session);
end;

procedure TFOperation._add_control_input(op: TFOperation);
begin
    // c_api.TF_AddControlInput(_opDesc, op);
    // c_api.AddControlInput(graph, _handle, op);
end;

procedure TFOperation._add_control_inputs(ops: TArray<TFOperation>);
begin
    for var op in ops do
     _add_control_input(op);
end;

procedure TFOperation._control_flow_post_processing;
begin
    for var input_tensor: TFTensor in inputs do
        control_flow_util.CheckInputFromValidContext(self, input_tensor.op);

    if Fcontrol_flow_context <> nil then
        Fcontrol_flow_context.AddOp(Self);
end;

function TFOperation._get_control_flow_context: TControlFlowContext;
begin
    Result := Fcontrol_flow_context;
end;

procedure TFOperation._set_control_flow_context(ctx: TControlFlowContext);
begin
    Fcontrol_flow_context := ctx;
end;

function TFOperation._tf_input(input_idx: Integer): TF_Input;
begin
    Result :=  TF_Input.Create(Handle, input_idx);
end;

function TFOperation._tf_output(output_idx: Integer): TF_Output;
begin
     Result :=  TF_Output.Create(Handle, output_idx);
end;

procedure TFOperation._update_input(index: Integer; tensor: TFTensor);
begin
    _assert_same_graph(tensor);

    // var input = _tf_input(index);
    // var output = tensor._as_tf_output();
    // Reset cached inputs.
    Finputs_val := nil;
    // _node_def = null;
    // after the c_api call next time _inputs is accessed
    // the updated inputs are reloaded from the c_api
    // lock (Locks.ProcessWide)
    // {
        // disable
        // c_api.TF_UpdateEdge(_graph, output, input, tf.Status.Handle);
        //var updated_inputs = inputs;
        // tf.Status.Check();
    // }
end;

procedure TFOperation._assert_same_graph(tensor: TFTensor);
begin
    {TODO -oMAs -cGeneral : Implement}
end;

{$ENDREGION}

{$REGION 'TFShape'}
{ TFShape }

function TFShape.GetIsFullDef: Boolean;
begin
    Result := (ndim > -1){ and ( Length(FaDims) > 0 )};
    for var i := 0 to Length(FaDims)-1 do
    begin
      if FaDims[i] < 1 then
      begin
          Result := False;
          Break;
      end;
    end;
end;

function TFShape.GetItem(sSlice: Slice): TFShape;
begin
    if not sSlice.Stop.HasValue then
        sSlice.Stop := Nullable<Integer>( Length(dims) - Integer(sSlice.Start.Value) + 1 );
    if (sSlice.Start.HasValue = false) or (sSlice.Len.HasValue = false) then
       raise TFException.Create('Slice must has Start and Length.');
    var r := Enumerable<Int64>.Create(dims) ;
    Result := TFShape( r.Skip(sSlice.Start.Value).Take(sSlice.Len.Value).ToArray );
end;

class operator TFShape.Implicit(a: PTFShape): TFShape;
begin
    if Assigned(a) then
       Result := a^;
end;

class operator TFShape.Explicit(const Value: TFShape): TValue;
begin
    TValue.From<TFShape>(Value);
end;

class operator TFShape.Implicit(a: TFShape): TFTensor;
begin
   var v : TValue := TValue.From<TFShape>(a);

    Result := constant_op.constant(v, dtInvalid,'Const') ;
end;

class operator TFShape.Implicit(a: TFShape): TArray<Integer>;
var
  i : Integer;
begin
    Result := [];
    for i := 0 to Length(a.FaDims) - 1 do
     Result := Result + [ a.FaDims[i] ] ;
end;

class operator TFShape.Implicit(a: TArray<Integer>): TFShape;
begin
    Result := TFShape.Create(a)
end;

class operator TFShape.Implicit(a: TArray<Int64>): TFShape;
begin
    Result := TFShape.Create(a)
end;

class operator TFShape.Implicit(a: Integer): TFShape;
begin
    Result := TFShape.Create([a])
end;

function TFShape.IsNil: Boolean;
begin
    Result := Self.FHandle = nil;
end;

function TFShape.IsNull: Boolean;
begin
    Result := FaDims = nil;
end;

constructor TFShape.Create(dims: TArray<TF_int64_t>);
begin
    Self := System.Default(TFShape);

    Self.FaDims  := dims;
    Self.Fndim   := GetnDim;
    Self.FSize   := GetSize;
    Self.FStrides:= [];

    Self.FHandle := Pointer($10000000);
end;

constructor TFShape.Create(proto: TTensorShapeProto);
begin
    Self := System.Default(TFShape);

    for var i := 0 to proto.Dims.Count - 1 do
    begin
        Self.FaDims := Self.FaDims + [ proto.Dims[i].Size ];
    end;

    Self.Fndim   := GetnDim;
    Self.FSize   := GetSize;
    Self.FStrides:= [];

    Self.FHandle := Pointer($10000000);
end;

class function TFShape.Scalar: TFShape;
begin
    var v : TFShape;

    v.FaDims  := nil;
    v.Fndim   := 0;
    v.FSize   := 1;
    v.FStrides:= [];

    v.FHandle := Pointer($10000000);
    Result := v ;

end;

function TFShape.GetItem(idx: Integer): Int64;
begin
     if idx < 0 then Result := dims[ndim + idx]
     else            Result := dims[idx]
end;

procedure TFShape.SetItem(idx: Integer; const Value: Int64);
begin
    dims[idx] := value;
end;

class function TFShape.AlignWithShape(shape: TFShape; slices: TArray<Slice>): TArray<Slice>;
begin
    var indim := shape.ndim;
    if indim = Length(slices) then
        Exit(slices) ;
    // align slices
    //
    var new_slices := TList<Slice>.Create;
    var slice_index := 0;
    var i : Integer := 0;
    while i < indim do
    begin
        if slice_index > Length(slices) - 1 then
        begin
            new_slices.Add(Slice.All);
            Inc(i);
            continue;
        end;
        if slices[slice_index] = Slice.All then
        begin
            new_slices.Add(Slice.All);
            for var j := 0 to (indim - Length(slices) )-1 do
            begin
                new_slices.Add(Slice.All);
                Inc(i);
            end;
        end else
        begin
            new_slices.Add(slices[slice_index]);
        end;
        Inc(slice_index);
        Inc(i);
    end;
    Result:= new_slices.ToArray();
end;

function TFShape.as_int_list: TArray<Integer>;
begin
    Result := [];
    for var i := 0 to Length(FaDims)-1 do
      Result := Result + [ Integer(FaDims[i]) ];
end;

function TFShape.concatenate(other: TArray<Int64>): TFShape;
begin
    Result := concatenate( TFShape.Create(other) );
end;

function TFShape.concatenate(other: TFShape): TFShape;
begin
    var otherShape := other;

    if (ndim < 0) or (otherShape.ndim < 0)  then
        Exit( TFShape.Null)
    else begin
        var concatenate_dims : TArray<Int64>;
        SetLength(concatenate_dims,ndim + otherShape.ndim);
        for var i := 0 to ndim - 1 do
            concatenate_dims[i] := dims[i];
        for var i := 0 to  otherShape.ndim - 1 do
            concatenate_dims[ndim + i] := otherShape.dims[i];
        Result := TFShape.Create(concatenate_dims);
    end;
end;

constructor TFShape.Create(dims: TArray<Integer>);
begin
    var i64Dim :  TArray<TF_int64_t> := [];
    for var i := 0 to Length(dims)-1 do
       i64Dim := i64Dim + [ Int64(dims[i]) ] ;

    Self := System.Default(TFShape);

    Self.FaDims  := i64Dim;
    Self.Fndim   := GetnDim;
    Self.FSize   := GetSize;
    Self.FStrides:= [];

    Self.FHandle := Pointer($10000000);
end;

class operator TFShape.Equal(a, b: TFShape): Boolean;
begin
    Result := a.Equals(  TValue.From<TFShape>(b) )
end;

class operator TFShape.NotEqual(a, b: TFShape): Boolean;
begin
    Result := not (a = b)
end;

class function TFShape.null: TFShape;
begin
  Result := System.Default(TFShape);
end;

function TFShape.Equals(target: TValue): Boolean;
begin
    if target.IsType<TFShape> then
    begin
        var vval := target.AsType<TFShape>;
        if (ndim = -1) and (vval.ndim = -1) then Exit(false)
        else if ndim <> vval.ndim then           Exit(false);
        var vVector : Vector<TF_int64_t> := self.Dims;
        Result := vVector.Equals(vval.Dims)
    end
    else if target.IsType< TArray<Int64> > then
    begin
        var vval := target.AsType< TArray<Int64> >;
        if ndim <> Length(vval) then Exit(False);

        var vVector : Vector<TF_int64_t> := self.Dims;
        Result := vVector.Equals(vval)
    end
    else if target.IsType< TArray<Integer> > then
    begin
        var vval := target.AsType< TArray<Integer> >;
        if ndim <> Length(vval) then Exit(False);

        var vval2 : Vector<Integer>;
        for var i := 0 to Length(self.Dims) - 1 do
           vval2 := vval2 + [ self.Dims[i] ];

        Result := vval2.Equals(vval) ;
    end
    else if target.IsType< TList<Integer> > then
    begin
        var vval := target.AsType< TList<Integer> >;
        if ndim <> vval.Count then Exit(False);

        var vval2 : Vector<Integer>;
        for var i := 0 to Length(self.Dims) - 1 do
           vval2 := vval2 + [ self.Dims[i] ];

        Result := vval2.Equals(vval.ToArray) ;
    end
    else if target.IsType< TList<Int64> > then
    begin
        var vval := target.AsType<  TList<Int64> >;
        if ndim <> vval.Count then Exit(False);

        var vval2 : Vector<Int64> := self.Dims;
        Result := vval2.Equals(vval.ToArray) ;
    end else
    begin
        Result := False;
    end;
end;

function TFShape.GetnDim: Integer;
begin
    //if FaDims = nil then Exit(-1);

    Result := Length(FaDims);
end;

function TFShape.GetRank: Integer;
begin
    Result := ndim;
end;

function TFShape.GetSize: Int64;
begin
    // scalar
    FSize := 1;

    if ndim = 0 then Exit(FSize);
    var computed : Int64 := 1;
    for var i := 0 to ndim - 1 do
    begin
        var val := dims[i];
        if val = 0 then
            Exit(0)
        else if val < 0 then
            continue;
        computed := computed * val;
    end;
    FSize := computed;
    Result := FSize;
end;

function TFShape.GetStrid: TArray<Int64>;
begin
    if FStrides = nil then
      FStrides := GetStrides(self);

    Result := FStrides
end;

class function TFShape.GetStrides(shape: TFShape): TArray<Int64>;
begin
    var strides : TArray<Int64>;
    SetLength(strides,shape.ndim) ;
    if shape.ndim = 0 then
        Exit(strides);
    strides[Length(strides) - 1] := 1;
    for var idx := Length(strides) - 1 downto 1 do
        strides[idx - 1] := strides[idx] * shape.dims[idx];
    Result := strides;
end;

class function TFShape.GetShape(shape1: TFShape; slices: TArray<Slice>): TFShape;
begin
    var new_dims := shape1.dims;
    slices       := AlignWithShape(shape1, slices);
    for var i := 0 to  Length(shape1.dims) - 1 do
    begin
        var sSlice : Slice := slices[i];
        if sSlice = Slice.All then
            new_dims[i] := shape1.dims[i]
        else if sSlice.IsIndex then
            new_dims[i] := 1
        else begin// range
            var iStart :Integer;
            if sSlice.Start = nil then iStart  := 0
            else                       iStart  := sSlice.Start;
            var iStop :Integer;
            if sSlice.Stop = nil then iStop  := shape1.dims[i]
            else                      iStop  := sSlice.Stop;
            new_dims[i] := iStop-iStart;
        end;
    end;
    // strip first dim if is index
    var return_dims := TList<Int64>.Create;
    try
      for var i := 0 to Length(new_dims) - 1 do
      begin
          if slices[i].IsIndex then
              continue;
          return_dims.add(new_dims[i]);
      end;
      Result := TFShape.Create(return_dims.ToArray);
    finally
      return_dims.Free;
    end;
end;

class function TFShape.GetOffset(shape: TFShape; indices: TArray<Integer>): Int64;
begin
    if (shape.ndim = 0) and (Length(indices)  = 1) then
        Exit(indices[0]);
    var offset : Int64 := 0;
    var strides        := shape.strides;
    for var i := 0 to Length(indices) - 1 do
        offset := offset + (strides[i] * indices[i]);
    if offset < 0 then
        raise TFException.Create('NotImplemented');
    Result := offset;
end;

function TFShape.IsScalar: Boolean;
begin
     Result := (ndim = 0) and (FHandle <> nil);
end;

function TFShape.is_compatible_with(shape2: TFShape): Boolean;
begin
    if (TArray.IndexOf<Int64>(dims,-1)> -1) or (TArray.IndexOf<Int64>(shape2.dims,-1) > -1) then
        Exit(true);
    if size <> shape2.size then
        Exit(false);
    Result := true;
end;

function TFShape.ToString: string;
begin
    case Fndim of
      -1 : Result := '<unknown>';
      0  : Result := '()';
      1  : Result := '(' + IntToStr(FaDims[0]).Replace('-1','None') +')';
    else
      var sStrings : TArray<String>;
      for var i := 0 to Length(FaDims) - 1 do
           sStrings := sStrings + [ IntToStr(FaDims[i]).Replace('-1','None') ];
      Result := '(' + Result.Join(',', sStrings) + ')';
    end;
end;

function TFShape.merge_with(other: TFShape): TFShape;
begin
    if Length(dims) = 0 then
        Exit( other );
    var new_dims := TList<Int64>.Create;
    try
      for  var i in TEnumerable.Range(0, ndim) do
      begin
          var dim := Dimension.create(dims[i]);
          var merged := dim.merge_with( Dimension.Create(other.dims[i]));
          new_dims.Add(merged.value);
      end;
      Result := TFShape.Create(new_dims.ToArray);
    finally
      new_dims.Free
    end;
end;

function TFShape.most_specific_compatible_shape(other: TFShape): TFShape;
var
  _Dims : TArray<Int64>;
begin
    _Dims := [];
    for var i := 0 to ndim - 1 do
       _Dims := _Dims + [ -1 ];

    for var i := 0 to Length(Dims)-1 do
    begin
        if Dims[i] = other.Dims[i] then
          _Dims[i] := Dims[i];
    end;
    Result := TFShape.Create(_Dims) ;
end;

function TFShape.unknown_shape(rank_: Integer): TFShape;
begin
    if rank_ = -1 then Result := TFShape.null
    else               Result := TFShape.Create(TEnumerable.Repeated<Int64>(-1, rank).ToArray);
end;

function TFShape.with_rank(rank_: Integer): TFShape;
begin
    Result := merge_with(unknown_shape(rank_));
end;

function TFShape.with_rank_at_least(rank_: Integer): TFShape;
begin
    if ndim < rank then
       raise  TFException.Create(format('Shape {this} must have rank at least {rank}',[Self.ToString,rank_]))
    else
       Result := Self;
end;

{$ENDREGION}

{$REGION 'TFGraph'}
//------------------------------------------------------------------------------
//----------------------------- TFGraph ----------------------------------------
//------------------------------------------------------------------------------
procedure TFGraph.add_to_collection<T>(name: string; value: T);
begin
    _check_not_finalized ;
    if Fcollections.ContainsKey(name) then
    begin
        var v : TList<T> := Fcollections[name].AsType< TList<T> >;
        v.Add( value );
        Fcollections.AddOrSetValue(name,TValue.From< TList<T> >(v))
    end
    else  begin
       var list := TList<T>.Create;
       list.Add(value);
       Fcollections.Add(name, list) ;
    end;
end;

procedure TFGraph.Add_op(var op: TFOperation);
begin
    op.id_value := NextId;
    Fnodes_by_id.AddOrSetValue(op.id_value, op);
    Fnodes_by_name.AddOrSetValue(op.name, op);
    Fversion := Max(Fversion, op.id_value);
end;

procedure TFGraph.add_to_collection<T>(names: TList<string>; value: T);
begin
    for var name in names do
      Add_to_collection(name,value);
end;

function TFGraph.as_default: TFGraph;
begin
    tf.Context.graph_mode(false);
    Result := TOps.set_default_graph(Self);
end;

function TFGraph.get_collection(name, scope: string): TValue;
begin
    if Fcollections.ContainsKey(name) then
       Result := Fcollections[name]
    else
       Result := nil;
end;

function TFGraph.get_collection<T>(name, scope: string): TList<T>;
begin
     var collection : TList<T>;
     if Fcollections.ContainsKey(name) then
         collection := Fcollections[name].AsType< TList<T> >
     else
         collection := TList<T>.Create;

     Result :=  collection;
end;

function TFGraph.get_collection_ref<T>(name: string): TList<T>;
begin
     if Fcollections.ContainsKey(name) then
         Result := Fcollections[name].AsType< TList<T> >
     else
         Result := TList<T>.Create;
end;

procedure TFGraph.colocate_with_for_gradient(op: TFOperation; gradient_uid: string;ignore_existing: Boolean = false);
begin

end;

procedure TFGraph.gExit;
begin
    tf.Context.restore_mode;
    TOps.pop_graph;
end;

function TFGraph.is_fetchable<T>(tensor_or_op: T): Boolean;
begin
    var value := TValue.From<T>(tensor_or_op);
    if value.IsType<TFTensor> then
    begin
        var tensor := value.AsType<TFTensor>;
        Result := not Funfetchable_ops.Contains(tensor.Op);
        Exit;
    end
    else if value.IsType<TFOperation> then
    begin
        var op := value.AsType<TFOperation>;
        Result := not Funfetchable_ops.Contains(op);
        Exit;
    end;
    Result := false;
end;

constructor TFGraph.Create;
begin
    inherited Create( TF_NewGraph );

    Fcontrol_flow_context        := TControlFlowContext.Create ;
    Fcontrol_dependencies_stack  := TList<TControlDependenciesController>.Create;

    Fnodes_by_id        := TDictionary<Integer, ITensorOrOperation>.Create;
    Fnodes_by_name      := TDictionary<string, ITensorOrOperation>.Create;
    Fnames_in_use       := TDictionary<string, Integer>.Create;
    Fversion            := 0;
    Fnext_id_counter    := 0;
    Funfetchable_ops    := TList<TFOperation>.Create;
    Funfeedable_tensors := TList<TFTensor>.Create;
    Fname_stack         := '';;
    Fgraph_key          := 'graph-' + IntToStr(TOps.GraphUniqueId)+'/';
    Flast_loss_reduction:= '';
    Fis_loss_scaled_by_optimizer := false;
    F_finalized         := False;
    Fcollections        := TDictionary<string, TValue>.Create;
    Fbuilding_function  := False;
    Fcontainer          := '';
    Fseed               := 0;
    Fouter_graph        := nil;
end;

constructor TFGraph.Create(hnd: Pointer);
begin
    Handle := hnd;

    Fcontrol_flow_context        := TControlFlowContext.Create ;
    Fcontrol_dependencies_stack  := TList<TControlDependenciesController>.Create;

    Fnodes_by_id        := TDictionary<Integer, ITensorOrOperation>.Create;
    Fnodes_by_name      := TDictionary<string, ITensorOrOperation>.Create;
    Fnames_in_use       := TDictionary<string, Integer>.Create;
    Fversion            := 0;
    Fnext_id_counter    := 0;
    Funfetchable_ops    := TList<TFOperation>.Create;
    Funfeedable_tensors := TList<TFTensor>.Create;
    Fname_stack         := '';;
    Fgraph_key          := 'graph-' + IntToStr(TOps.GraphUniqueId);
    Flast_loss_reduction:= '';
    Fis_loss_scaled_by_optimizer := false;
    F_finalized         := False;
    Fcollections        := TDictionary<string, TValue>.Create;
    Fbuilding_function  := False;
    Fcontainer          := '';
    Fseed               := 0;
    Fouter_graph        := nil;
end;

destructor  TFGraph.Destroy;
begin
   Fcollections.Free;
   Fnodes_by_id.Free;
   Fnodes_by_name.Free;
   Fnames_in_use.Free;
   Funfetchable_ops.Free;
   Funfeedable_tensors.Free;
   Fcontrol_flow_context.Free ;
   Fcontrol_dependencies_stack.Free;

   inherited Destroy;
end;
procedure TFGraph.device(device_name: string);
begin

end;

procedure TFGraph.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteGraph(hnd);
end;

function TFGraph.NewOperation(opType, opName: string): TFOperationDesc;
begin
    Result := TFOperationDesc.Create(self, TF_TString(opType), TF_TString(opName));
end;

function TFGraph.NextId: Integer;
begin
   Inc(Fnext_id_counter);

   Result := Fnext_id_counter;
end;

procedure TFGraph.prevent_feeding(tensor: TFTensor);
begin
    Funfeedable_tensors.Add(tensor);
end;

procedure TFGraph.prevent_fetching(op: TFOperation);
begin
    Funfetchable_ops.Add(op);
end;

function TFGraph.get_name_scope: TF_TString;
begin
    Result:= Fname_stack;
end;

function TFGraph.get_operations: TArray<ITensorOrOperation>;
begin
    Result := Fnodes_by_name.Values.ToArray;
end;

function TFGraph.name_scope(name: TF_TString): TF_TString;
begin
    var new_stack : TF_TString := '';

    if string.IsNullOrEmpty(string(name)) then
        new_stack := ''
    else if string(name).EndsWith('/') then
        new_stack := TOps.name_from_scope_name(string(name))
    else
        new_stack := unique_name(name);
    Fname_stack := new_stack;
    if String.IsNullOrEmpty(string(new_stack)) then Result := ''
    else                                            Result := new_stack + '/';
end;

function TFGraph.unique_name(name: TF_TString;mark_as_used:Boolean): TF_TString;
begin
    if not String.IsNullOrEmpty(Fname_stack) then
        name := Fname_stack + '/' + name;
    // For the sake of checking for names in use, we treat names as case
    // insensitive (e.g. foo = Foo).
    var name_key := String(name).ToLower;
    var i : Integer := 0;
    if Fnames_in_use.ContainsKey(name_key) then
        i := Fnames_in_use[name_key];
    // Increment the number for "name_key".
    if mark_as_used then
        Fnames_in_use.AddOrSetValue(name_key, i + 1);
    if i > 0 then
    begin
        // Make sure the composed name key is not already used.
        var base_name_key := name_key;
        while Fnames_in_use.ContainsKey(name_key) do
        begin
            name_key := base_name_key+'_'+IntToStr(i);
            i := i + 1;
        end;
        // Mark the composed name_key as used in case someone wants
        // to call unique_name("name_1").
        if mark_as_used then
            Fnames_in_use.AddOrSetValue(name_key, 1);
        // Return the new name with the original capitalization of the given name.
        name := name+'_' + IntToStr(i - 1);
    end;
    Result := name;
end;

procedure TFGraph._check_not_finalized;
begin
    if F_finalized then
       raise TFException.Create('Graph is finalized and cannot be modified.');
end;

function  TFGraph._as_graph_element(Value: TVAlue): TFTEnsor;
begin
    if Value.IsType<RefVariable> then
    begin
        var vVar := Value.AsType<RefVariable>;
        Result :=vVar._as_graph_element;
    end
    else if Value.IsType<RefVariable> then
    begin
        var vVar := Value.AsType<ResourceVariable>;
        Result := vVar.GraphElement;
    end else
    begin
        Result := nil;
    end;
end;

function TFGraph.as_graph_element(obj: TValue; allow_tensor: Boolean = true; allow_operation: Boolean = true): ITensorOrOperation;
begin
    Result := _as_graph_element_locked(obj, allow_tensor, allow_operation);
end;

function TFGraph._as_graph_element_locked(obj: TValue; allow_tensor, allow_operation: Boolean): ITensorOrOperation;
begin
    var types_str : string := '';

    if allow_tensor and  allow_operation then
    begin
        types_str := 'Tensor or Operation';
    end
    else if allow_tensor then
    begin
        types_str := 'Tensor';
    end
    else if allow_operation then
    begin
        types_str := 'Operation';
    end;
    var temp_obj := _as_graph_element(obj);
    if temp_obj <> nil then
        obj := temp_obj;
    // If obj appears to be a name...
    if (obj.IsType<string>) or (obj.IsType<AnsiString>) then
    begin
        var name : string := obj.AsString;
        if (name.Contains(':')) and (allow_tensor) then
        begin
            var op_name : string := name.Split([':'])[0];
            var out_n : Integer := integer.Parse(name.Split([':'])[1]);
            if Fnodes_by_name.ContainsKey(op_name) then
                Exit ( Fnodes_by_name[op_name].outputs[out_n] )
            else
               raise TFException.Create( format('The name %s refers to a Tensor which does not ' +
                                              'exist. The operation, %s, does not exist in the graph.',[ name, op_name]));
        end
        else if (not name.Contains(':')) and (allow_operation) then
        begin
            if not Fnodes_by_name.ContainsKey(name) then
                raise TFException.Create('The name '+name+' refers to an Operation not in the graph.');
            Exit( Fnodes_by_name[name] );
        end
        else if (not name.Contains(':')) and  (not allow_operation) then
        begin
            // Looks like an Operation name but can't be an Operation.
            if Fnodes_by_name.ContainsKey(name) then
                // Yep, it's an Operation name
                raise TFException.Create('The name '+name+' refers to an Operation, not a '+types_str+'.')
            else
               raise TFException.Create( format('The name %s looks like an (invalid) Operation name, not a %s' +
                                      'Tensor names must be of the form \"<op_name>:output_index>\".',[name,types_str]));
        end;
    end;
    if (obj.IsType<TFTensor>) and (allow_tensor) then
    begin
        var tensor := obj.AsType<TFTensor>;
        if tensor.graph.Equals(Self) then
        begin
            Exit(tensor);
        end else
        begin
            raise TFException.Create('Tensor '+ obj.TypeInfo.Name +' is not an element of this graph.');
        end;
    end
    else if (obj.IsType<TFOperation>) and (allow_operation) then
    begin
        var op := obj.AsType<TFOperation>;
        if op.graph.Equals(self) then
        begin
            Exit(op);
        end else
        begin
            raise TFException.Create('Operation '+ obj.TypeInfo.Name +' is not an element of this graph.');
        end;
    end;
    raise TFException.Create('Can not convert a '+obj.TypeInfo.Name+ ' into a '+types_str+'.');
end;

function TFGraph.control_dependencies(control_inputs: TArray<TValue>): TControlDependenciesController;
begin
    if (control_inputs = nil) or (tf.Context.executing_eagerly) then
        exit ( TControlDependenciesController.Create(Self, nil) );
    var control_ops := TList<ITensorOrOperation>.Create;
    for var c in control_inputs do
    begin
        if string.LowerCase(c.TypeInfo.Name) = 'tftensor' then
          control_ops.Add(c.AsType<TFTEnsor>.Op )
        else if string.LowerCase(c.TypeInfo.Name) = 'tfoperation' then
          control_ops.Add(c.AsType<TFOperation> )
        else begin
                var t1 := _as_graph_element(c);
                if t1 = nil then
                   raise TFException.Create('Control input must be Operation or Tensor:{c}');
                control_ops.Add(t1.op);
        end;
    end;
    Result := TControlDependenciesController.Create(self, control_ops);
end;

function TFGraph._get_control_flow_context: TControlFlowContext;
begin
    Result := Fcontrol_flow_context;
end;

procedure TFGraph._push_control_dependencies_controller(controller: TControlDependenciesController);
begin
    Fcontrol_dependencies_stack.Add(controller);
end;

procedure TFGraph._pop_control_dependencies_controller(controller: TControlDependenciesController);
begin
    Fcontrol_dependencies_stack.Delete(Fcontrol_dependencies_stack.Count - 1);
end;

procedure TFGraph._set_control_flow_context(ctx: TControlFlowContext);
begin
   Fcontrol_flow_context := ctx;
end;

procedure TFGraph._record_op_seen_by_control_dependencies(op: TFOperation);
var
  i : Integer;
begin
   for i := 0 to Fcontrol_dependencies_stack.Count - 1 do
   begin
    if Assigned( Fcontrol_dependencies_stack[i]) then
         Fcontrol_dependencies_stack[i].add_op(op);
   end;
end;

function TFGraph._control_dependencies_for_inputs(input_ops: TArray<ITensorOrOperation>): TArray<ITensorOrOperation>;
var
  controller : TControlDependenciesController;
begin
    var ret := TList<ITensorOrOperation>.Create;

    if Fcontrol_dependencies_stack.Count < 1 then
       Exit(ret.ToArray);

    for var i := 0 to Fcontrol_dependencies_stack.Count - 1 do
    begin
        controller := Fcontrol_dependencies_stack[i] ;
        var dominated : Boolean := false;
        // If any of the input_ops already depends on the inputs from controller,
        // we say that the new op is dominated (by that input), and we therefore
        // do not need to add control dependencies for this controller's inputs.
        for var op in input_ops do
        begin
            if controller.op_in_group(op) then
            begin
                dominated := true;
                break;
            end;
        end;
        var x := TCollections.CreateList<ITensorOrOperation>(controller.control_inputs);

        var x1 := x.Where(function(const aItem : ITensorOrOperation): Boolean
                               begin
                                    Result := not TArray.Contains<ITensorOrOperation>(input_ops,aItem);
                               end);
        if not dominated then
           ret.AddRange(x1.ToArray);
    end;
    Result := ret.ToArray;
end;

function TFGraph._create_op_from_tf_operation(c_op: Pointer; compute_device: Boolean): TFOperation;
begin
    var ret := TFOperation.Create(c_op, self);
    add_op(ret);

    var name_key := ret.name.ToLower;
    if not Fnames_in_use.ContainsKey(name_key) then
        Fnames_in_use.Add(name_key, 1);

    _create_op_helper(ret, compute_device);

    Result := ret;
end;

procedure TFGraph._create_op_helper(op: TFOperation; compute_device: Boolean);
begin
    _record_op_seen_by_control_dependencies(op);
end;

function TFGraph.GetOpByName(const Name: TF_TString): TFOperation;
var
   l_pOp: PTF_Operation;
begin
   if not Assigned(Handle) then
     ObjectDisposedException();
   l_pOp := TF_GraphOperationByName(Handle, PAnsiChar(Name));
   if Assigned(l_pOp) then
     Result := TFOperation.Create(l_pOp,self)
   else
     Result := Nil;
end;

function TFGraph.GetOpDef(tipo: string): TOpDef;
begin
    Result := op_def_registry.GetOpDef(tipo)
end;

function TFGraph.Create_op(op_type       : TF_TString;
                                 inputs        : TArray<TFTensor>;
                                 dtypes,
                                 input_types   : TArray<TF_DataType>;
                                 Name          : TF_TString;
                                 attrs         : TDictionary<string, TAttrValue>;
                                 op_def        : TOpDef;
                                 compute_device: Boolean): TFOperation;
var
  Tt : Enumerable<TFTensor>;
begin
    if Length(inputs) = 0 then
        inputs := [];//[ TFTensor.Create( [Int32(0)]) ];
    if string.IsNullOrEmpty(Name) then
        name := op_type;

    // If a names ends with a '/' it is a "name scope" and we use it as-is,
    // after removing the trailing '/'.
    // This was causing duplicate graph node name errors, when testing a conv2d autoencoder
    // https://keras.io/guides/functional_api/#:~:text=keras.,graph%20(DAG)%20of%20layers.
    if string(name).EndsWith('/') then
        name := TOps.name_from_scope_name(name)
    else
        Name := unique_name(name);

    var node_def := TOps._NodeDef(op_type, name, attrs);

    var input_ops   : TArray<ITensorOrOperation>;
    for var i := 0 to Length(inputs)-1 do
      input_ops := input_ops  + [ inputs[i].op ];

    Tt := Enumerable<TFTensor>.Create(inputs);
    var operations := Tt.Select<ITensorOrOperation>(function(x: TFTensor): ITensorOrOperation
                                              begin
                                                Result := x.op;
                                              end).ToArray;

    var control_inputs := _control_dependencies_for_inputs(input_ops);

    var op := TFOperation.Create(node_def,Self,inputs,dtypes,control_inputs,input_types,'',op_def);

    _create_op_helper(op, compute_device);

    Result := op;

end;
{$ENDREGION}

{$REGION 'TFTensors'}
{ TFTensors }

constructor TFTensors.Create;
begin
   inherited Create;

   FItems := TList<TFTensor>.Create;

   FiLength := FItems.Count;
end;

constructor TFTensors.Create(const tensors: array of TFTensor);
begin
   inherited Create(tensors);

   FItems := TList<TFTensor>.Create;

   FItems.AddRange(tensors);

   var fFirst : TFTensor := nil;
   if FItems.Count > 0 then
    fFirst := FItems.First;

   if (fFirst <> nil ) and ((fFirst.Handle <> nil) or (fFirst.FOp <> nil))  then
   begin
     FiLength := FItems.Count;
     Fdtype := fFirst.Dtype;
     Fshape := fFirst.Shape;
     FRank  := fFirst.rank;
     Fgraph := fFirst.graph;
     FIsCreatedInGraphMode := fFirst.isCreatedInGraphMode;
   end;

end;

constructor TFTensors.Create(tensor: TFTensor);
begin
    if tensor = nil then
    begin
         tensor := TFTensor.Create;

         Create([tensor]);

         FiLength := 0;
         Exit;
    end;
    Create([tensor])
end;

constructor TFTensors.Create(t: Tuple<TFTensor, TFTensor>);
begin
    Create([t.Value1,t.Value2])
end;

constructor TFTensors.Create(tensors: TList<TFTensor>);
begin
    Create(tensors.ToArray)
end;

procedure TFTensors.Deconstruct(var a, b: TFTensor);
begin
    a := FItems[0];
    b := FItems[1];
end;

destructor TFTensors.Destroy;
begin
    //for var item in FItems do
    //  item.Free;

    FItems.Clear;
    FItems.Free;

    inherited;
end;

function TFTensors.GetShape: TFShape;
begin
   var fFirst := FItems.First;

   if (fFirst.Handle <> nil) or (fFirst.FOp <> nil)  then
     Fshape := fFirst.Shape
   else
     Fshape := System.default(TFShape);

   Result := Fshape;
end;

function TFTensors.GetTensor(idx: Integer): TFTensor;
begin
    Result := FItems[idx];
end;

procedure TFTensors.SetTensor(idx: Integer; const Value: TFTensor);
begin
   FItems[idx] := Value;
end;

function TFTensors.ToArray: TArray<TFTensor>;
begin
    Result := FItems.ToArray
end;

function TFTensors.ToString: string;
begin
    if FItems.Count = 1 then
      Exit(FItems.First.ToString) ;

    Format('%d Tensors.',[]);
end;

function TFTensors.ToTensor(tensors: TFTensors): TFTensor;
var
  s : string;
begin
    if FItems.Count < 1 then Exit(nil);
    s := '';
    for var i := 0 to FItems.Count - 1 do
      s := s + FItems[i].name +',' ;

    s := s.Remove(s.Length-1,1) ;
    Result := FItems.First;
end;

function TFTensors.Add(const tensor: TFTensor): Integer;
begin
    Result := inherited Add(tensor) ;

    FItems.Add(tensor);

    Fdtype := FItems.First.Dtype;
    Fshape := FItems.First.Shape;
    FRank  := FItems.First.rank;
    Fgraph := FItems.First.graph;
    FIsCreatedInGraphMode := FItems.First.isCreatedInGraphMode;
    FiLength := FItems.Count;
end;

procedure TFTensors.AddRange(tensors: TArray<TFTensor>);
begin
    FItems.AddRange(tensors);

    Fdtype := FItems.First.Dtype;
    Fshape := FItems.First.Shape;
    FRank  := FItems.First.rank;
    Fgraph := FItems.First.graph;
    FIsCreatedInGraphMode := FItems.First.isCreatedInGraphMode;
    FiLength := FItems.Count;
end;

procedure TFTensors.Insert(idx: Integer; const tensor: TFTensor);
begin
    inherited Insert(idx,tensor);

    FItems.Insert(idx,tensor);
end;

{$ENDREGION}

{$REGION 'TControlFlowContext'}
{ TControlFlowContext }

constructor TControlFlowContext.Create;
begin
    Fcontext_stack   := TStack<TControlFlowContext>.Create;
    Fexternal_values := TDictionary<string, ITensorOrOperation>.Create;
    Fvalues          := TLIst<string>.Create;
end;

destructor TControlFlowContext.Destroy;
var
    LEntry: TControlFlowContext;
begin
    while Fcontext_stack.Count > 0 do
    begin
      LEntry := Fcontext_stack.Pop;
      LEntry.Free;
    end;
    Fcontext_stack.Free;
    Fexternal_values.Free;
    Fvalues.Free;
end;

function TControlFlowContext.AddValue(val: TFTensor): TFTensor;
begin
    // to be overridden
    Result := nil;
end;

procedure TControlFlowContext.AddInnerOp(op: TFOperation);
begin
    if Fouter_context <> nil then
       Fouter_context.AddInnerOp(op);
end;

procedure TControlFlowContext.AddOp(op: TFOperation);
begin
    _AddOpInternal(op);
end;

procedure TControlFlowContext._AddOpInternal(op: TFOperation);
begin
    if op = nil then
    begin
        raise TFException.Create('Not Implemented');
    end else
    begin
        for var index := 0 to op.inputs.Count - 1 do
        begin
            var x      := op.inputs[index];
            var real_x := AddValue(x);
            if real_x <> x then
                op._update_input(index, real_x);
        end;
    end;
end;

procedure TControlFlowContext._Enter_;
begin

end;

procedure TControlFlowContext._Exit_;
begin

end;

procedure TControlFlowContext._init_values_from_proto(values_def: TValuesDef; import_scope: string);
begin
    Fouter_context := Tops.get_default_graph._get_control_flow_context;
    if values_def <> nil then
        _init_values_from_proto(values_def, import_scope)
    else begin
        Fvalues := TLIst<string>.Create;
        Fexternal_values := TDictionary<string, ITensorOrOperation>.Create;
    end;
end;

function TControlFlowContext._RemoveExternalControlEdges(op: TFOperation): Tuple<TArray<TFOperation>, TArray<TFOperation>>;
begin
    var while_ctxt := GetWhileContext;

    var internal_control_inputs := TList<TFOperation>.Create;
    // A control input of `op` is internal if it is in the same while
    // loop context as the enclosing while loop context of self.
    if while_ctxt = nil then
    begin
        internal_control_inputs.AddRange(op.control_inputs);
    end else
    begin
        for var x: TFOperation in op.control_inputs do
        begin
            var ctxt := control_flow_util.GetOutputContext(x);
            if (ctxt <> nil) and (ctxt.GetWhileContext = while_ctxt) then
                internal_control_inputs.Add(x);
        end
    end;

    var external_control_inputs := TList<TFOperation>.Create;
    if internal_control_inputs.Count <> Length(op.control_inputs) then
       raise Exception.Create('Not Implemented');

    Result := Tuple.Create(internal_control_inputs.ToArray, external_control_inputs.ToArray);
end;

procedure TControlFlowContext.__init__;
begin

end;

procedure TControlFlowContext.__init__(values_def: TValuesDef; import_scope: string);
begin
    Fexternal_values := TDictionary<string, ITensorOrOperation>.Create;
    for  var value in values_def.Valuess do
        Fvalues.Add(value);
    var g := Tops.get_default_graph;
    for var value in values_def.ExternalValues do
    begin
        var k := Tops.prepend_name_scope(value.Key, import_scope);
        var v := value.Value;
        Fexternal_values[k] := g.as_graph_element(Tops.prepend_name_scope(v, import_scope));
    end;

    var op_names : TArray<String>;
    for var value in Fvalues do
    begin
        if not Fexternal_values.ContainsKey(value) then
        begin
            var s : TArray<String>:= value.split([':']);
            op_names := op_names + [ s[0] ];
        end;
    end;
    for var op in op_names do
       (g.as_graph_element(op) as TFOperation)._set_control_flow_context(self);
end;

procedure TControlFlowContext.Enter_;
begin
    var graph :=Tops.get_default_graph;
    Fcontext_stack.Push(graph._get_control_flow_context);
    graph._set_control_flow_context(Self);
end;

procedure TControlFlowContext.ExitResult(res: TArray<TFTensor>);
begin
    if Fouter_context <> nil then
       raise Exception.Create('Not Implemented "ExitResult"');
end;

procedure TControlFlowContext.Exit_ ;
begin
    var graph := Tops.get_default_graph;
    var last_context := Fcontext_stack.Pop;
    graph._set_control_flow_context(last_context);
end;

function TControlFlowContext.GetWhileContext: WhileContext;
begin
    if Fouter_context <> nil then
      Exit ( Fouter_context.GetWhileContext);
   Result := nil;
end;

function TControlFlowContext.IsCondContext: Boolean;
begin
   Result := False;
end;

function TControlFlowContext.OpInContext(op: TFOperation): Boolean;
begin
    Result := IsContainingContext(op._get_control_flow_context, self);
end;

function TControlFlowContext.IsContainingContext(ctxt, maybe_containing_ctxt: TControlFlowContext): Boolean;
begin
    while ctxt <> maybe_containing_ctxt do
    begin
        if ctxt = nil then
            Exit(false);
        ctxt := ctxt.outer_context;
    end;
    Result := true;
end;

function TControlFlowContext.IsWhileContext: Boolean;
begin
   Result := False;
end;

{$ENDREGION}

{$REGION 'TControlDependenciesController'}
{ TControlDependenciesController }

constructor TControlDependenciesController.Create(graph: TFGraph; control_inputs: TList<ITensorOrOperation>);
begin
    Fgraph := graph;
    if control_inputs = nil then
    begin
        Fcontrol_inputs_val := TList<ITensorOrOperation>.Create;
        Fnew_stack          := true;
    end else
    begin
        Fcontrol_inputs_val := control_inputs;
        Fnew_stack          := false;
    end;
    Fseen_nodes := TList<ITensorOrOperation>.Create;
    Fold_stack  := nil;
    Fold_control_flow_context := nil;
end;

destructor TControlDependenciesController.Destroy;
begin
    Fseen_nodes.Free;
end;

function TControlDependenciesController.GetCtxInput: TArray<ITensorOrOperation>;
begin
    Result := Fcontrol_inputs_val.ToArray;
end;

procedure TControlDependenciesController._Enter_;
begin
    if Fnew_stack then
    begin
        // Clear the control_dependencies graph.
        Fold_stack := Fgraph.Fcontrol_dependencies_stack;
        Fgraph.Fcontrol_dependencies_stack := TList<TControlDependenciesController>.Create;
        // Clear the control_flow_context too.
        Fold_control_flow_context := Fgraph._get_control_flow_context;
        Fgraph._set_control_flow_context(nil);
    end;
    Fgraph._push_control_dependencies_controller(Self);
end;

procedure TControlDependenciesController._Exit_;
begin
    Fgraph._pop_control_dependencies_controller(self);
    if Fnew_stack then
    begin
        Fgraph.Fcontrol_dependencies_stack := Fold_stack;
        Fgraph._set_control_flow_context(Fold_control_flow_context);
    end;
end;

function  TControlDependenciesController.op_in_group(op: ITensorOrOperation): Boolean;
begin
    Result := Fseen_nodes.Contains(op);
end;

procedure TControlDependenciesController.add_op(op: ITensorOrOperation);
begin
    Fseen_nodes.Add(op);
end;
{$ENDREGION}

{$REGION 'TInputList'}
{ TInputList }

constructor TInputList.Create(const tensors: array of TFTensor);
begin
  inherited Create(tensors);

  Finputs := [];
  for var i := 0 to Length(tensors) - 1 do
    Finputs := Finputs + [ tensors[i] ];
end;

destructor TInputList.Destroy;
begin
  Finputs := [];

  inherited Destroy;
end;

function TInputList.GetItem(index: Integer): TFTensor;
begin
    if index = - 1 then
      index := Length (Finputs) -1;
    Result := Finputs[index]
end;

function TInputList.GetLength: Integer;
begin
   Result := Length(Finputs)
end;
{$ENDREGION}

{ TNDArray.TEnumerator_NDArray }

constructor TNDArray.TEnumerator_NDArray.Create(source : TNDArray);
begin
   fSource := source;
   fIndex := 0;
end;

destructor TNDArray.TEnumerator_NDArray.Destroy;
begin

  inherited Destroy;
end;

function TNDArray.TEnumerator_NDArray.GetCurrent: TNDArray;
begin
    Result := fSource[findex];
end;

function TNDArray.TEnumerator_NDArray.MoveNext: Boolean;
begin
   Result := fIndex < fSource.dims[0];
   if Result then
    Inc(fIndex);
end;

{$REGION 'GradLoopState'}
{ GradLoopState }

constructor GradLoopState.Create(forward_ctxt: WhileContext; outer_grad_state_: GradLoopState);
begin
    Fgrad_context := nil;
    Fswitch_map   := TDictionary<TFOperation, TFTensor>.Create;
end;

destructor GradLoopState.Destroy;
begin
   if Assigned(Fgrad_context) then  Fgrad_context.Free ;
   if Assigned(Fswitch_map) then  Fswitch_map.Free;
   if Assigned(Fforward_context) then  Fforward_context.Free;
   if Assigned(Fouter_grad_state) then  Fouter_grad_state.Free;
   if Assigned(Fforward_index) then  Fforward_index.Free;
   if Assigned(Fgrad_index) then  Fgrad_index.Free;
   Fforward_loop_exits := [];
   if Assigned(Fdeferred_exits) then  Fdeferred_exits.Free;
   if Assigned(Funused_exits) then  Funused_exits.Free;
   if Assigned(Fgrad_sync) then  Fgrad_sync.Free;

   inherited;
end;
{$ENDREGION}

{$REGION 'EagerOperation'}

{ EagerOperation }

constructor EagerOperation.Create;
begin
    inherited Create(nil);
end;

function EagerOperation.GetInputList: TInputList;
begin
    if Finputs_val = nil  then
    begin
        Finputs_val := TInputList.Create(inputs);
    end;
    Result := Finputs_val;
end;

function EagerOperation.GetOutpts: TArray<TensorFlow.DApi.TFTensor>;
begin
    if Length(FOutputs) < 1 then
    begin
        FOutputs := Outputs;
    end;
    Result := FOutputs;
end;

procedure EagerOperation.SetOutputs(const Value: TArray<TensorFlow.DApi.TFTensor>);
begin
    FOutputs := Value;
end;

function EagerOperation.get_attr(attr_name: string): TValue;
begin
    // var attrType = c_api.TFE_OpNameGetAttrType(tf.Context.Handle, Name, attr_name, ref isList, tf.Status.Handle);
    var i : Integer := 0 ;
    while i < Length(Attrs) do
    begin
        if ( Attrs[i].AsString = attr_name ) then
            Exit( Attrs[i + 1] );
       Inc(i,2);
    end;
    Result := nil;
end;
{$ENDREGION}

{$REGION 'LoopVar'}
{ LoopVar<TItem> }

constructor LoopVar<TItem>.Create(_counter: TFTensor; _item: TItem);
begin
     Counter := _counter;
     Item    := _item;
end;

function LoopVar<TItem>.Flatten: TArray<TValue>;
begin

end;

function LoopVar<TItem>.Pack(sequences: TArray<TValue>): LoopVar<TItem>;
begin

end;
{$ENDREGION}

{$REGION 'WhileContext'}
{ WhileContext }

function WhileContext._get_shape_invariant(_var: TFTensor; shape: TArray<Integer>): TFShape;
begin
    Result := _var.shape;
end;

function WhileContext._convert_tensorarray_to_flow(tensor_or_tensor_array: TValue): TFTensor;
begin
    if tensor_or_tensor_array.IsType<TTensorArray> then
    begin
        Result := tensor_or_tensor_array.AsType<TTensorArray>.flow;
        Exit;
    end
    else if tensor_or_tensor_array.IsType<TFTensor> then
    begin
        Result := tensor_or_tensor_array.AsType<TFTensor>;
        Exit;
    end;

    raise Exception.Create('_convert_tensorarray_to_flow');
end;

function WhileContext._BuildLoop<TItem>(pred: TFunc<LoopVar<TItem>, TFTensor>; body: TFunc<LoopVar<TItem>, LoopVar<TItem>>; original_loop_vars: LoopVar<TItem>;
  loop_vars: TArray<TFTensor>; shape_invariants: TArray<TFShape>): Tuple<LoopVar<TItem>, TArray<TFTensor>>;
begin
  raise Exception.Create('not Implemented . "_BuildLoop"');
  (*  var flat_loop_vars = nest.flatten2(original_loop_vars)
        .Select(x => (ITensorOrTensorArray)x)
        .ToArray();

    // Let the context know the loop variables so the loop variables
    // would be added in the outer contexts properly.
    _InitializeValues(loop_vars);
    var real_vars = loop_vars;
    Tensor[] enter_vars = null;
    tf_with(ops.control_dependencies(null), delegate
    {
        enter_vars = real_vars.Select(x => control_flow_ops._Enter(x,
            _name,
            is_constant: false,
            parallel_iterations: _parallel_iterations,
            use_input_shape: shape_invariants == null))
        .ToArray();

        foreach (var x in enter_vars)
        {
            x.graph.prevent_feeding(x);
            if (_outer_context != null)
                _outer_context.AddInnerOp(x.op);
        }
    });

    // Finds the closest enclosing non-None control pivot.
    var outer_context = _outer_context;
    object control_pivot = null;
    while (outer_context != null && control_pivot == null)
    {

    }

    if (control_pivot != null)
    {

    }

    _SetShapeInvariants(real_vars, enter_vars, shape_invariants);

    // Fix the control inputs and control flow context of these enter ops.
    _FixControlInputsAndContext(enter_vars);
    _InitializeValues(enter_vars);
    _loop_enters = enter_vars.ToList();

    var merge_vars = enter_vars
        .Select(x => merge(new[] { x, x }))
        .Select(m => (Tensor)m)
        .ToArray();

    _pivot_for_pred = merge_vars[0];

    // Build the graph for pred.
    var merge_vars_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_loop_vars, merge_vars);
    var packed_vars = new LoopVar<TItem>(
        (Tensor)merge_vars_with_tensor_arrays[0],
        new TItem().FromMergeVars(merge_vars_with_tensor_arrays));
    var pp = pred(packed_vars);
    var c = ops.convert_to_tensor(pp);
    _pivot = gen_control_flow_ops.loop_cond(c, name: "LoopCond");
    var switch_vars = merge_vars.Select(x => _SwitchRefOrTensor(x, _pivot))
        .ToArray();

    // Build the graph for body.
    var vars_for_body = switch_vars.Select(x => _Identity(x[1])).ToArray();
    _pivot_for_body = vars_for_body[0];
    // Convert TensorArray flow variables inside the context back into
    // their associated TensorArrays for calling the body.
    var vars_for_body_with_tensor_arrays = _convert_flows_to_tensorarrays(flat_loop_vars, vars_for_body);
    var packed_vars_for_body = nest.pack_sequence_as2(original_loop_vars, vars_for_body_with_tensor_arrays);
    var pre_summaries = ops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);
    var body_result = body(packed_vars_for_body);
    var post_summaries = ops.get_collection(tf.GraphKeys._SUMMARY_COLLECTION);

    // Store body_result to keep track of TensorArrays returned by body
    var original_body_result = body_result;
    // Convert TensorArrays returned by body into their flow variables
    var result = nest.flatten2(body_result)
        .Select(x => _convert_tensorarray_to_flow(x))
        .ToArray();
    // result = ops.convert_n_to_tensor_or_composite(result);
    var next_vars = new List<Tensor>();
    foreach (var (m, v) in zip(merge_vars, result))
        next_vars.Add(_AddNextAndBackEdge(m, v));

    // Add the exit ops.
    var exit_vars = switch_vars.Select(x => exit(x[0])).ToList();
    _loop_exits = exit_vars;

    // Exit the loop.
    // ExitResult(exit_vars);
    return (original_body_result, exit_vars.ToArray());
 *)
end;

function WhileContext.BuildLoop<TItem>(pred: TFunc<LoopVar<TItem>, TFTensor>; body: TFunc<LoopVar<TItem>, LoopVar<TItem>>; loop_vars: LoopVar<TItem>;
  shape_invariants: TArray<TFShape>; return_same_structure: Boolean): LoopVar<TItem>;
begin
    // Keep original_loop_vars to identify which are TensorArrays
    var original_loop_vars := loop_vars;

    // Convert TensorArrays to their flow variables
    var loopFlat := loop_vars.Flatten;
    var loop_vars_tensors : TArray<TFTensor> := [];
    for var i := 0 to Length(loopFlat) -1  do
        loop_vars_tensors := loop_vars_tensors + [ _convert_tensorarray_to_flow(loopFlat[i]) ];

    if Length(shape_invariants) < 1 then
    begin
        for var i := 0 to Length(loop_vars_tensors) -1  do
        shape_invariants := shape_invariants + [ _get_shape_invariant(loop_vars_tensors[i]) ];
    end;

    Enter_;
    var tLoop := _BuildLoop<TItem>(pred, body, original_loop_vars, loop_vars_tensors, shape_invariants);
    var original_body_result := tLoop.Value1;
    var exit_vars            := tLoop.Value2;
    Exit_;

    var resultFlat := original_body_result.Flatten;
    var flat_result : TArray<ITensorOrTensorArray> := [];
    for var i := 0 to Length(resultFlat) -1  do
        flat_result := flat_result + [ resultFlat[i].AsType<TFTensor> ];

    var exit_vars_with_tensor_arrays := control_flow_ops._convert_flows_to_tensorarrays(flat_result, exit_vars);

    var a : TArray<TValue>;
    for var i := 0 to Length(exit_vars_with_tensor_arrays) -1  do
        a := a + [ TVAlue.From<ITensorOrTensorArray>(exit_vars_with_tensor_arrays[i]) ];
    var packed_exit_vars := original_body_result.Pack(a);

   Result := packed_exit_vars;
end;

constructor WhileContext.Create(maximum_iterations: TFTensor; parallel_iterations: Integer; back_prop, swap_memory: Boolean; name: string; grad_state: GradLoopState;
  context_def: TWhileContextDef; import_scope: string);
begin
    if context_def <> nil then
    begin
        _init_from_proto(context_def, import_scope);
    end else
    begin
        _init_from_args(maximum_iterations, parallel_iterations, back_prop, swap_memory, name);
    end;

    Fgrad_state := grad_state;
end;

procedure WhileContext._init_from_args(maximum_iterations: TFTensor; parallel_iterations: Integer; back_prop, swap_memory: Boolean; name: string);
begin
    Fname                := Tops.get_default_graph.unique_name(name);
    Fmaximum_iterations  := maximum_iterations;
    Fparallel_iterations := parallel_iterations;
    Fback_prop           := back_prop;
    Fswap_memory         := swap_memory;
    Floop_exits          := TList<TFTensor>.Create;
    Floop_enters         := TList<TFTensor>.Create;
    Fgraph               := Tops.get_default_graph();
end;

procedure WhileContext._init_from_proto(context_def: TWhileContextDef; import_scope: string);
begin
    var g := Tops.get_default_graph;
    Fname := Tops.prepend_name_scope(context_def.ContextName, import_scope);
    if not string.IsNullOrEmpty(context_def.MaximumIterationsName) then
        Fmaximum_iterations := g.as_graph_element(Tops.prepend_name_scope(context_def.MaximumIterationsName, import_scope)) as TFTensor;
    Fparallel_iterations := context_def.ParallelIterations;
    Fback_prop := context_def.BackProp;
    Fswap_memory := context_def.SwapMemory;
    Fpivot_for_pred := g.as_graph_element(Tops.prepend_name_scope(context_def.PivotForPredName, import_scope)) as TFTensor;
    // We use this node to control constants created by the body lambda.
    Fpivot_for_body := g.as_graph_element(Tops.prepend_name_scope(context_def.PivotForBodyName, import_scope)) as TFTensor;
    // The boolean tensor for loop termination condition.
    Fpivot := g.as_graph_element(Tops.prepend_name_scope(context_def.PivotName, import_scope)) as TFTensor;
    // The list of exit tensors for loop variables.
    Floop_exits := TList<TFTensor>.Create;
    for var i := 0 to context_def.LoopExitNamess.Count- 1 do
    begin
        var exit_name := context_def.LoopExitNamess[i];
        Floop_exits.Add(g.as_graph_element(Tops.prepend_name_scope(exit_name, import_scope)) as TFTensor);
    end;
    // The list of enter tensors for loop variables.
    Floop_enters := TList<TFTensor>.Create;
    for var i := 0 to context_def.LoopEnterNamess.Count-1 do
    begin
        var enter_name := context_def.LoopEnterNamess[i];
        Floop_enters.Add(g.as_graph_element(Tops.prepend_name_scope(enter_name, import_scope)) as TFTensor);

    end;

    __init__(context_def.ValuesDef, import_scope);
end;
{$ENDREGION}


end.





