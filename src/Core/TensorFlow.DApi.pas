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
  Spring.Collections.Dictionaries,

  Spring.Collections.Enumerable,
  Spring.Collections.Lists,

  TF4D.Core.CApi,
  TensorFlow.DApiEager,
  TensorFlow.DApiBase,
  TensorFlow.Slice,

  ProtoGen.tensorShape,
  ProtoGen.attrValue,
  ProtoGen.opDef,
  ProtoGen.nodeDef,
  ProtoGen.config;

type
TFGraph         = class;
TFOperation     = class;
TFTensor        = class;
TNDArray        = class;
TFOperationDesc = class;
TFSession       = class;

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
    FDtype  : TF_DataType;
    FName   : string;
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
     property Name   : string       read GetName;
     property Device : string       read GetDevice;
     property Op     : TFOperation  read FOp;
     property Outputs: TArray<TensorFlow.DApi.TFTensor>  read FOutputs;
end;
{$ENDREGION}

{$REGION 'TControlFlowContext'}
TControlFlowContext = class(TInterfacedObject,ITensorFlowObject)
   private

   protected
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
   public
     constructor Create;
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

     property pivot  : TFTensor read Fpivot;
     property pred   : TFTensor read Fpred;
     property branch : Integer  read Fbranch;
     property Nome   : String   read FName;
     property outer_context  : TControlFlowContext read Fouter_context ;
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
   constructor Create(target : TF_TString= ''; config: PConfigProto = nil); overload;
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
   constructor Create(target: TF_TString = ''; g      : TFGraph = nil;    config : PConfigProto= nil; status: TFStatus = nil); overload;
   // Session
   constructor Create(g     : TFGraph;         config : PConfigProto= nil; status: TFStatus = nil); overload;
   destructor  Destroy; override;

   procedure   run(op:      TFOperation;        feed_dict: TArray<FeedItem>); overload;
   function    run(fetche:  TFTensor;           feed_dict: TArray<FeedItem>): TNDArray ;overload;
   function    run(fetche:  ITensorOrOperation; feed_dict: TArray<FeedItem>): TNDArray;overload;
   function    run(fetches: TValue;             feed_dict: TArray<FeedItem>): TArray<TNDArray>;overload;
   function    run(fetches: TValue)                                         : TArray<TNDArray>;overload;
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
TFTensor = class(ITensorOrOperation)
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
   FId                   : Int64;
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

   function GetByteSize: UInt64;
   function GetDataTypeSize: UInt64;
   function GetSize: UInt64;
   function GetData: Pointer;
   function GetRank: Integer;
   function GetShape: TFShape;
   procedure Setshape(const Value: TFShape);
   // inherited from ITensorOrOperation
   function GetName: string; override;
   function GetType: TF_DataType; override;
   function GetDevice: string; override;

   function GetTensorDataPointer: Pointer;
   function StringBytes:TArray< TArray<Byte> >; overload;
   procedure UpdateTensoData;
   procedure InitTensor(shape: TFShape; dtype: TF_DataType);overload;
   Procedure InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType); overload;
   class function GetBestDType<Tx, Ty>(x: Tx; y: Ty) : TF_DataType;
   class function TF_NewTensor(shape: TFShape; dtype: TF_DataType; data: Pointer):PTF_Tensor; overload;
   class function TF_NewTensor(data: TArray<Byte>; shape: TFShape; dtype: TF_DataType):PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<T>; shape: TFShape): PTF_Tensor; overload;
   function GetDims: TArray<Int64>;
   function GetItem(slices: TArray<Slice>): TFTensor;overload;
   function GetItem(idx: Integer): TFTensor;overload;
   function GetItem(slices: TArray<string>): TFTensor;overload;
   function GetnDim: Integer;
 protected
   procedure NativeDispose(hnd: Pointer); override;
   function  GetNDArray(ddtype: TF_DataType): TNDArray;
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

   class function InitTensor<T>(aArray: TArray<T>;                        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<T>>;                shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<T>>>;        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   //
   class function    StringTensor(srcArray: TArray<TArray<Byte>>; shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<TF_TString>;   shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<string>;       shape: TFShape):PTF_Tensor;overload;
   //
   function StringBytes(index: Integer): TArray<byte>; overload;
   function StringData(index: integer): AnsiString; overload;
   function StringData: TArray<TF_TString>; overload;
   //
   function ToString: string;override;
   //
   destructor  Destroy; override;

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
   function  _slice(start: Integer): TFTensor;

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

end;
{$ENDREGION}

{$REGION 'TNDArray'}
TNDArray = class(TFTensor)
   private
     procedure NewEagerTensorHandle;
     function  GetItem(indices: TArray<Integer>): TNDArray; overload;
     function  GetItem(indices: TArray<Slice>): TNDArray; overload;
     function  GetItem(indice: Integer): TNDArray; overload;
     procedure SetItem(indices: TArray<Integer>; const Value: TNDArray); overload;
     procedure SetItem(indices: TArray<Slice>; const Value: TNDArray); overload;
     procedure SetItem(indice: Integer; const Value: TNDArray); overload;
     function  GetData(slices: TArray<Slice>): TNDArray;
     procedure SetData(slices: TArray<Slice>; aArray: TNDArray);
     function  GetScalar(offset : UInt64 = 0): TNDArray;
     function  GetArrayData(newshape: TFShape; indices: TArray<Integer>): TNDArray;
     function  GetDataPointer: Pointer;
   public
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

     function reshape(newshape: TFShape): TNDArray;
     function astype(dtype: TF_DataType): TNDArray;
     function ToByteArray: TArray<Byte>;

     property Item[indices: Integer ]         : TNDArray read GetItem write SetItem; default;
     property Item[indices: TArray<Integer> ] : TNDArray read GetItem write SetItem; default;
     property Item[slices: TArray<Slice> ]    : TNDArray read GetItem write SetItem; default;
     property data : Pointer read GetDataPointer;

     type
       TNDArrayEnum = class(TIteratorBase<TNDArray>, IEnumerator<TNDArray>)
          protected
             function GetCurrent: TNDArray;
             function TryMoveNext(var current: TNDArray): Boolean; override;
          public
             constructor Create; override;
             function GetEnumerator: IEnumerator<TNDArray>; override;
             function MoveNext: Boolean;
       end;

  end;
{$ENDREGION}

{$REGION 'TFTensors'}
TFTensors = class (TList<TFTensor>)
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
 protected

 public
   destructor  Destroy; override;
   constructor Create; overload; override;
   constructor Create(const tensors: array of TFTensor); overload; override;
   constructor Create(tensor: TFTensor); overload;
   constructor Create(t: Tuple<TFTensor,TFTensor>); overload;
   constructor Create(tensors : TList<TFTensor>); overload;
   function    Add(const tensor : TFTensor): Integer;  override;
   procedure   AddRange(tensors : TArray<TFTensor>);
   procedure   Insert(idx : Integer ; const tensor : TFTensor);  override;
   function    ToTensor(tensors: TFTensors): TFTensor;
   function    ToArray: TArray<TFTensor>; reintroduce;
   function    ToString: string; override;
   procedure   Deconstruct(var a: TFTensor; var b : TFTensor);


   property Item[idx: Integer] : TFTensor read GetTensor write SetTensor; default;
   property dtype   : TF_DataType read Fdtype ;
   property shape   : TFShape     read Fshape ;
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

     constructor Create(const tensors: array of TFTensor); override;
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
    // Pointer to the graph, to keep it from collecting if there are TFOperations alive.
    FGraph                : TFGraph;
    FInputs_val           : TInputList;
    Fcontrol_flow_context : TControlFlowContext;
    Fname                 : string;
    Fid_value             : Integer;
    Fis_stateful          : Boolean;

    function OutputType(index: Integer): TF_DataType;
    function GeNumOutputs: Integer;
    function GetOutput: TFTensor;
    // Inherited from ITensorOrOperation
    function GetDevice: string; override;
    function GetType: TF_DataType; override;
    function GetName: string;override;
    function GeNumInputs: Integer;
    function GetInputList: TInputList;
    function GetNodeDef: TNodeDef;
    function GetipoOp: string;
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
                      op_def         : POpDef = nil);overload;
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
   function  get_attr(name:string): TValue; overload;
   function  get_attr<T>(name: string): T;  overload;
   function  get_attr_list<T>(name: string): TArray<T>;
   //
   property Graph     : TFGraph    read FGraph;
   property NumOutputs: Integer    read GeNumOutputs;
   property NumInputs : Integer    read GeNumInputs;
   property id_value  : Integer    read Fid_value write Fid_value;
   property id        : Integer    read Fid_value write Fid_value;
   property Output    : TFTensor   read GetOutput;
   property inputs    : TInputList read GetInputList;
   property NodeDef   : TNodeDef   read GetNodeDef;
   property Tipo      : string     read GetipoOp;
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
   Fgraph_key          : string;
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
   Fbuilding_function : Boolean;
   Fcontainer         : string;
   Fseed              : Integer;
   Fouter_graph       : TFGraph;
   // Current control flow context. It could be either CondContext or WhileContext
   Fcontrol_flow_context : TControlFlowContext ;
   // represents the nested with(...) statements
   Fcontrol_dependencies_stack : TList<TControlDependenciesController>;

   procedure _check_not_finalized;
   function  _as_graph_element(Value: TVAlue): TFTEnsor;
   procedure _create_op_helper(op: TFOperation; compute_device: Boolean = true);
   function  _as_graph_element_locked(obj: TValue; allow_tensor: Boolean = true; allow_operation: Boolean = true): ITensorOrOperation;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   _name_stack        :     TF_TString;
   function name_scope(name: TF_TString) : TF_TString;
   function get_name_scope: TF_TString;
   /// <summary>
   /// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
   /// </summary>
   constructor Create; overload;
   constructor Create(hnd: Pointer); overload;
   destructor  Destroy; override;
   function    NextId: Integer;

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
   procedure gExit;
   function  NewOperation(opType,opName: string):TFOperationDesc;
   procedure Add_op(var op: TFOperation);
   function  is_fetchable<T>(tensor_or_op: T ) : Boolean;
   /// <summary>
   /// Returns a context manager that makes this `Graph` the default graph.
   /// Must call Exit() to pop graph
   /// </summary>
   /// <returns></returns>
   function as_default: TFGraph;
   function Create_op(op_type        : TF_TString ;
                      inputs         : TArray<TFTensor>;
                      dtypes         : TArray<TF_DataType>;
                      input_types    : TArray<TF_DataType> = [];
                      Name           : TF_TString= '';
                      attrs          : TDictionary<string, TAttrValue> = nil;
                      op_def         : POpDef= nil;
                      compute_device : Boolean = True) : TFOperation;
   //
   property nodes_by_name       : TDictionary<string, ITensorOrOperation> read Fnodes_by_name;
   property version             : Integer read Fversion;
   property name_stack          : string  read Fname_stack;
   property graph_key           : string  read Fgraph_key;
   property last_loss_reduction : string  read Flast_loss_reduction;
   property is_loss_scaled_by_optimizer : Boolean read Fis_loss_scaled_by_optimizer write Fis_loss_scaled_by_optimizer;
   property building_function   : Boolean read Fbuilding_function;
   property container           : string  read Fcontainer;
   property seed                : Integer read Fseed write Fseed;
   property outer_graph         : TFGraph read Fouter_graph;
   property control_flow_context: TControlFlowContext read Fcontrol_flow_context;
   property control_dependencies_stack : TList<TControlDependenciesController> read Fcontrol_dependencies_stack;

end;
{$ENDREGION}

implementation
   uses System.Math,

        Oz.Pb.Classes,
        Oz.Pb.StrBuffer,

        TensorFlow,TensorFlow.Ops,
        TensorFlow.Constant_op,
        Tensorflow.NameScope,
        TensorFlow.EagerTensor,
        TensorFlow.Variable,
        Tensorflow.Utils,
        TensorFlow.Framework,
        Tensorflow.math_ops,
        TensorFlow.gen_math_ops,
        Tensorflow.array_ops,
        Tensorflow.gen_array_ops,
        TensorFlow.Tensors.Ragged;


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
    for var m in fetch_mappers do
    begin
        var m_value_indices := TList<Integer>.Create;
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
    end;
    Result := Tuple<TList<ITensorOrOperation>, TList<TArray<Integer>>>.Create(unique_fetches, value_indices);
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

    if values.Count > 0 then
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
   Ffinal_fetches.Free;
   Ftargets.Free;

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

constructor TFSessionOptions.Create(target: TF_TString; config: PConfigProto);
begin
    inherited Create( TF_NewSessionOptions );

    TF_SetTarget(Handle, PAnsiChar(target));

    if config <> nil  then
       SetConfig(config^)
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

constructor TFSession.Create(target: TF_TString; g : TFGraph; config : PConfigProto; status: TFStatus);
begin
   if Assigned(graph)  then FGraph := graph
   else                     FGraph := Tops.get_default_graph;

   if not FGraph.building_function then
   begin
       if Tops.get_default_graph <> FGraph then
          FGraph.as_default;
   end;

   var opts := TFSessionOptions.Create(target, config);
   var sStatus : TFStatus;
   if Assigned(status) then sStatus := status
   else                     sStatus := tf.status;

   var l_pSession:= TF_NewSession(graph.handle, opts.Handle, sStatus.Handle);
   sStatus.CheckMaybeRaise(status,False);

   inherited Create(l_pSession);

end;

constructor TFSession.Create(g: TFGraph; config: PConfigProto; status: TFStatus);
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
    Result     := tensor.numpy;
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

function TFSession.run(fetches: TValue): TArray<TNDArray>;
begin
    var feed_items : TArray<FeedItem> := [];
    Result := _run(fetches, feed_items);
end;

function TFSession.run(fetches: TValue; feed_dict: TArray<FeedItem>): TArray<TNDArray>;
begin
    Result := _run(fetches, feed_dict);
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
begin
    // Ensure any changes to the graph are reflected in the runtime.
    _extend_graph();
    var status := tf.Status;

    // Input tensors
    //
    var inputs : TArray<TF_Output> := [];
    for var i := 0 to Length(feed_dict) - 1 do
       inputs := inputs + [ feed_dict[i].Key ];

    var input_values : TArray<PTF_Tensor> := [];
    for var i := 0 to Length(feed_dict) - 1 do
       input_values := input_values + [ feed_dict[i].Value.Handle ];

    // Output tensors
    //
    var output : TArray<TF_Output> := [];
    for var i := 0 to Length(fetch_list) - 1 do
       output := output + [ fetch_list[i] ];

    var output_values : TArray<PTF_Tensor>;
    for var i := 0  to Length(fetch_list) - 1 do
        output_values := output_values + [ nil ];

    // Target operations
    //
    var target_opers : TArray<PTF_Operation> := [];
    for var i := 0 to target_list.Count - 1 do
       target_opers := target_opers + [ target_list[i].Handle ];

    TF_SessionRun(Handle,
                  // RunOptions
                  nil,
                  // Input tensors
                  @(inputs[0]), @(input_values[0]),Length(input_values),
                  // Output tensors
                  @(output[0]), @(output_values[0]), Length(output_values),
                  // Target operations
                  @(target_opers[0]), target_list.Count,
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
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
                inc(i);
            end
            else if x.Value.IsType<Double> then
            begin
                var v := x.Value.AsType<Double>;
                feeds[i] := TPair<TF_Output, TFTensor>.Create(key._as_tf_output, TFTensor.Create(v));
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
    var ft := TEnumerable.Select<TFTensor,TF_Output>(fetch_list, function (Arg1: TFTensor): TF_Output
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
    try

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
        var ft := TEnumerable.Select<TValue,TFOperation>(final_targets, function (Arg1: TValue): TFOperation
                                                                           begin
                                                                               Result := Arg1.AsType<TFOperation>;
                                                                           end );
        var target  := TList<TFOperation>.Create(ft.ToArray);
        var results := _do_run(target, final_fetches, feed_dict_tensor);

        Result := fh.build_results(results);
      finally
        //final_fetches.Free;
        //final_targets.Free;
      end;
    finally
      feed_dict_tensor.Free;
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
     FId := TOps.uid;
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
       Move(@data[0], ttensor^, _length);

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
    var size : Int64 := 1;
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

  ts       : TArray<TF_TString> ;
begin
     dims     := shape.Dims;
     pDims    :=  PTF_int64_t(Pointer(@dims)^);

     var pTensor := TF_AllocateTensor(TF_DataType.TF_STRING, pDims, shape.ndim, shape.size * TF_TSRING_SIZE);
     var tstr  : PTF_TString  := TF_TensorData(pTensor);

    SetLength(ts,Length(srcArray));
    for i := 0 to Length(srcArray)- 1 do
      ts[i] := TF_TString( TEncoding.UTF8.GetString(srcArray[i]) );

    for i := 0 to Length(ts) - 1 do
    begin
          TF_StringInit(tstr);

          TF_StringCopy(tstr, @ts[i][1], Length(ts[i]));

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
  l_iFullByteSize : Integer;
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
        Exit;
    end;

    //if (ndim[0] > 1) then
    //   raise TFException.Create('ToArray - ndim[0] > 1  !!!.');

    SetLength(res,size);
    l_pVal  := @res[0];
    l_iFullByteSize :=  size * dtypesize;
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

    SetLength(res,bytesize);
    l_pData := TF_TensorData(Handle);

    l_pVal  := @res[0];

    Move(l_pData^, l_pVal^, bytesize);

    Result := res;
end;

procedure TFTensor.InitTensor(shape: TFShape; dtype: TF_DataType);
begin
    Handle := TF_NewTensor(shape,dtype,nil)  ;
end;

Procedure TFTensor.InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType);
begin
     if dtype = TF_DataType.TF_STRING then
     begin
         var buf : TArray<TArray<byte>>;
         SetLength(buf,1);
         for var i := 0 to Length(bytes) - 1 do
           buf[0][i] := bytes[i];
         Handle := StringTensor( buf, TFShape.Scalar);
     end else
     begin
         Handle  := TF_NewTensor(bytes,shape,dtype) ;
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
         l_pData := PByte(@aArray[0]);
         Result  := TF_NewTensor(shape,dtype,l_pData) ;
     end;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<T>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     var _length := shape.Size;
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
     l_pData := PByte(@aArray[0][0][0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0][0][0][0]);
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

function TFTensor.GetName: string;
var
 opname : string;
begin
    opname := '<unnamed>';
    if Fop <> nil then
      opname := Fop.name;

    FName := Format('%s:%d',[opname,value_index]);
    Result := FName
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
        var pDim : PInt64 := @dims.Dims[0];
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
    if value.IsNil then
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, nil, -1, tf.Status.Handle)
    else
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, @value.dims[0], value.ndim, tf.Status.Handle);
    tf.Status.RaiseEx;
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

        aSlice := aSlice + [ Slice.Create(x,x,1,true) ];
     end;
     SetData(aSlice,Value);
end;

function TNDArray.GetItem(indices: TArray<Slice>): TNDArray;
begin
    Result := GetData(indices)
end;

procedure TNDArray.SetItem(indices: TArray<Slice>; const Value: TNDArray);
begin
    SetData(indices,Value);
end;

function TNDArray.GetItem(indice: Integer): TNDArray;
begin
    Result := GetData([indice])
end;

procedure TNDArray.SetItem(indice: Integer; const Value: TNDArray);
begin
    SetData([indice],Value)
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

function TNDArray.GetDataPointer: Pointer;
begin
   Result := TensorDataPointer;
end;

procedure TNDArray.SetData(slices: TArray<Slice>; aArray: TNDArray);
begin
    {TODO -oMax -cNDArray : Add Setdata}
    //SetData(array, slices.ToArray(), new int[shape.ndim].ToArray(), -1);
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
    aDim := 0;
    vValue := value;
    lIsArray := False;
    while vValue.IsArray do
    begin
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

    dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
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
begin
    Result := TNDArray.Create( math_ops.cast(self, dtype) )
end;

function TNDArray.reshape(newshape: TFShape): TNDArray;
begin
    Result := TNDArray.Create( tf.reshape(self, newshape) )
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

function TNDArray.ToByteArray: TArray<Byte>;
begin
    Result :=  BufferToArray;
end;

{ TNDArray.TNDArrayEnum }

constructor TNDArray.TNDArrayEnum.Create;
begin
  inherited;

end;

function TNDArray.TNDArrayEnum.GetCurrent: TNDArray;
begin

end;

function TNDArray.TNDArrayEnum.GetEnumerator: IEnumerator<TNDArray>;
begin

end;

function TNDArray.TNDArrayEnum.MoveNext: Boolean;
begin

end;

function TNDArray.TNDArrayEnum.TryMoveNext(var current: TNDArray): Boolean;
begin

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
                               op_def         : POpDef );
begin
   inherited Create;
   Fgraph := g;
   // Build the list of control inputs.
   var control_input_ops := TList<TFOperation>.Create;
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

   // This will be set by self.inputs.
   if op_def = nil then
   begin
     var op := g.GetOpDef(node_def.Op);
     op_def := @op;
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

   if Handle <> nil then
       _control_flow_post_processing;

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
end;

function TFOperation.Getname: string;
begin
    Fname := '';
    if Assigned(Handle) then
      Fname := string( AnsiString(TF_OperationName(Handle)) ) ;

    Result := Fname;
end;

function TFOperation.GetOperation(h: Pointer): TFOperation;
begin
    var nodes := tf.get_default_graph.nodes_by_name;
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

    Result := lst.FirstOrDefault(nil)
end;

function TFOperation.GetType: TF_DataType;
begin
   Result := Output.dtype;
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

        Result := [ v.AsType<T> ];
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
              var Lvalue := TValue.From<Int32>( Integer(lst.&Is[i]^) );
              Result := Result + [ Lvalue.AsType<T>  ];
          end;
      end
      else if TypeInfo(T) = TypeInfo(Int64) then
      begin
          var lst :=   AttrValue.Value.value.AsType< TListValue > ;
          Result := [];
          for var i := 0 to lst.&Is.count-1 do
          begin
              var Lvalue := TValue.From<Int64>( lst.&Is[i]^ );
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
    {TODO -oMax -cDApi : Add controlFlow Processing}
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
     Result := ndim = 0;
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
      Result := Result.Join(',', sStrings);
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
    Result:= _name_stack;
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
    _name_stack := new_stack;
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
        var x := TList<ITensorOrOperation>.Create(controller.control_inputs);
        try
          var x1 := x.Where(function(const aItem : ITensorOrOperation): Boolean
                               begin
                                    Result := not TArray.Contains<ITensorOrOperation>(input_ops,aItem);
                               end);
          if not dominated then
              ret.AddRange(x1);
        finally
         // x.Free;
        end;

    end;
    Result := ret.ToArray;
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
                                 op_def        : POpDef;
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
    {TODO -oMax -cException : Fix Exception}
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

   var fFirst := FItems.First;

   Fdtype := fFirst.Dtype;
   Fshape := fFirst.Shape;
   FRank  := fFirst.rank;
   Fgraph := fFirst.graph;
   FIsCreatedInGraphMode := fFirst.isCreatedInGraphMode;
   FiLength := FItems.Count;
end;

constructor TFTensors.Create(tensor: TFTensor);
begin
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
    for var item in FItems do
      item.Free;

    FItems.Free;

    inherited;
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
end;

procedure TControlFlowContext._Enter_;
begin

end;

procedure TControlFlowContext._Exit_;
begin

end;

procedure TControlFlowContext.Enter_;
begin
    var graph :=Tops.get_default_graph;
    Fcontext_stack.Push(graph._get_control_flow_context);
    graph._set_control_flow_context(Self);
end;

procedure TControlFlowContext.Exit_ ;
begin
    var graph := Tops.get_default_graph;
    var last_context := Fcontext_stack.Pop;
    graph._set_control_flow_context(last_context);
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

Initialization
begin

end;
finalization
begin

end;
end.



