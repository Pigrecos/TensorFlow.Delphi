{$REGION 'Licence'}
{ Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Org. Source:  https://github.com/tensorflow/tensorflow
Org. Docu:    https://www.tensorflow.org
The structure of this Delphi porting was oriented to C# porting by Miguel Deicaza.
C# Source:    https://github.com/migueldeicaza/TensorFlowSharp
Delphi porting version: 1.2
==============================================================================}
{$ENDREGION}
unit TensorFlow.DApi;
{$POINTERMATH ON}
{$WARN DUPLICATE_CTOR_DTOR OFF}
interface
uses
  System.SysUtils, System.Classes, System.Types, Generics.Collections,
  System.AnsiStrings, System.Rtti, Spring,spring.Collections.Base,
  TensorFlow.LowLevelAPI, TensorFlow.DApiEager, TensorFlow.DApiBase, TensorFlow._Helpers,

  ProtoGen.opDef,
  ProtoGen.nodeDef;

type
TFGraph         = class;
TFOperation     = class;
TFSession       = class;
TFMMEnv         = class;
TFTensor        = class;
TFOperationDesc = class;

PTFShape = ^TFShape;
TFShape = record
 private
   FHandle : Pointer;
   FaDims  : TArray<TF_int64_t>;
   FSize   : Int64;
   Fndim   : Integer;
 public
   class operator Initialize (out Dest: TFShape);
   class operator Finalize   (var Dest: TFShape);
   //class operator Implicit(a: TFShape): PTFShape;
   class operator Implicit(a: PTFShape): TFShape;

   property Dims: TArray<TF_int64_t> read FaDims write FaDims;
end;

TParameter = record
  sNome : string;
  vValue: TValue;
end;

  /// <summary>
  /// in order to limit function return value
  /// is Tensor or Operation
  /// </summary>
  ITensorOrOperation = class(TFDisposable)
    public
       FDevice : string;
       FOp     : TFOperation;
       FName   : string;
       FDtype  : TF_DataType;
       FOutputs: TArray<TFTensor>;

       //function numpy:TNDArray; virtual ; abstract;
    public
       
  end;


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
   constructor Create;
		/// <summary>
		/// Delete the instance.
		/// </summary>
   destructor  Destroy; override;
   /// <summary>
   /// Sets the target in options.
   /// </summary>
   /// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
   /// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
	 procedure SetTarget(target: TFString);
	 /// <summary>
	 /// Sets the configuration information for the session.
	 /// </summary>
	 /// <param name="protoData">Serialized protocol buffer for the tensorflow.ConfigProto message.</param>
	 /// <param name="length">Length of the buffer.</param>
	 /// <param name="status">If config was not parsed successfully as a ConfigProto, the error is recorded here.</param>
	 /// <remarks>
	 /// The configuration option is a Protocol Buffer representing the tensorflow.ConfigProto
	 /// </remarks>
	 procedure SetConfig(protoData: Pointer; lng: Integer; status: TFStatus = Nil);
end;

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
   m_eagerTensorHandle   : PTFE_TensorHandle;
   m_lDeallocator_called : Boolean;
   m_isCreatedInGraphMode: Boolean;
   m_IsList              : Boolean;
   /// <summary>
   ///     The Operation that produces this tensor as an output.
   /// </summary>
   m_op                  : TFOperation;
   m_tf_output           : Nullable<TF_Output>;
   m_value_index         : Integer;
   m_override_dtype      : TF_DataType;
   m_id                  : Int64;
   F_name                : string;
   /// <summary>
   ///     The Graph that contains this tensor.
   /// </summary>
   m_graph               : TFGraph;
   m_rank                : Integer;
   m_shape               : TFShape;
	 /// <summary>
	 /// Returns the data type for the tensor.
	 /// </summary>
	 /// <value>The type of the tensor.</value>
   function GetTensorDataType(): TF_DataType;
	 /// <summary>
	 /// Returns the value of the Tensor if possible, or IsEmpty if the data type can not be represented in Delphi
	 /// </summary>
	 /// <param name="jagged">
	 /// The default is set to false, which returns .NET multi-dimensional arrays for multi-dimensional
	 /// tensors.    This is useful to feed the data back as a TFTensor created from an array.   Set to
	 /// true if you want to get arrays pointing to arrays, which are slightly more convenient to work
	 /// with from C#
	 /// </param>
	 /// <remarks>
	 /// Jagged arrays create various intermediate arrays, while multi-dimensional arrays are more
	 /// efficient memory-wise.
	 /// </remarks>
	 /// <returns>The value encodes the contents of the tensor, and could include simple values, arrays and multi-dimensional values.</returns>
   function GetValue(): TValue;
   function GetByteSize: UInt64;
   function GetDataTypeSize: UInt64;
   function GetSize: UInt64;
   function GetData: Pointer;
   function GetDim: Tarray<UInt64>;
   function GetRank: Integer;
   function GetShape: TFShape;
   procedure Setshape(const Value: TFShape);
   function GetName: string;
   function GetType: TF_DataType;

 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(hnd: Pointer);   overload;
   constructor Create(value: Byte);    overload; virtual;
   constructor Create(value: Boolean); overload; virtual;
   constructor Create(value: Integer); overload; virtual;
   constructor Create(value: Int16);   overload; virtual;
   constructor Create(value: Int64);   overload; virtual;
   constructor Create(value: Single);  overload; virtual;
   constructor Create(value: Double);  overload; virtual;
   constructor Create(op: TFOperation; value_index: Integer; dtype:TF_DataType); overload;
   constructor Create(const value: TFString); overload;
   constructor Create(values: TArray<Byte>); overload;
   constructor Create(values: TArray<TArray<Byte>>); overload;
   constructor Create(values: TArray<Boolean>); overload;
   constructor Create(values: TArray<TArray<Boolean>>); overload;
   constructor Create(values: TArray<Integer>); overload;
   constructor Create(values: TArray<TArray<Integer>>); overload;
   constructor Create(values: TArray<Int16>); overload;
   constructor Create(values: TArray<TArray<Int16>>); overload;
   constructor Create(values: TArray<Int64>); overload;
   constructor Create(values: TArray<TArray<Int64>>); overload;
   constructor Create(values: TArray<Single>); overload;
   constructor Create(values: TArray<TArray<Single>>); overload;
   constructor Create(values: TArray<Double>); overload;
   constructor Create(values: TArray<TArray<Double>>); overload;
   constructor Create(values: TArray<TFString>); overload;
   constructor Create(values: TArray<TArray<TFString>>); overload;
   constructor Create(shape : TFShape;dtype:TF_DataType; data: Pointer); overload;
   constructor Create(bytes : TArray<Byte>;shape : TFShape; dtype:TF_DataType); overload;
   constructor Create(shape : TFShape;dtype:TF_DataType); overload;
   class function    StringTensor(srcArray: TArray<TArray<Byte>>; shape: TFShape):TFTensor;overload;
   class function    StringTensor(srcArray: TArray<TFString>; shape: TFShape):TFTensor;overload;
   //
   function ToString: string;override;
   //
   destructor  Destroy; override;
   //
   function MMLck(): TFTensor;
   //
	 /// <summary>
	 /// The tensor is a array.
	 /// </summary>
   function IsArray(var dim1, dim2: TF_int64_t): Boolean;
   /// <summary>
   ///
   /// </summary>
   /// <typeparam name="T"></typeparam>
   /// <returns></returns>
   function ToArray<T>:TArray<T>;
   //
   function GetArray(var values: TArray<Byte>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Byte>>): Boolean; overload;
   function GetArray(var values: TArray<Boolean>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Boolean>>): Boolean; overload;
   function GetArray(var values: TArray<Integer>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Integer>>): Boolean; overload;
   function GetArray(var values: TArray<Int16>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Int16>>): Boolean; overload;
   function GetArray(var values: TArray<Int64>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Int64>>): Boolean; overload;
   function GetArray(var values: TArray<Single>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Single>>): Boolean; overload;
   function GetArray(var values: TArray<Double>): Boolean; overload;
   function GetArray(var values: TArray<TArray<Double>>): Boolean; overload;
   function GetArray(var values: TArray<TFString>): Boolean; overload;
   function GetArray(var values: TArray<TArray<TFString>>): Boolean; overload;
   class function InitTensor(aArray: TArray<Byte>;shape: TFShape; dtype: TF_DataType): TFTensor; overload;

   class function InitTensor<T>(aArray: TArray<T>;                        shape: TFShape; dtype: TF_DataType): TFTensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<T>>;                shape: TFShape; dtype: TF_DataType): TFTensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<T>>>;        shape: TFShape; dtype: TF_DataType): TFTensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: TFShape; dtype: TF_DataType): TFTensor; overload;

   function _as_tf_output : TF_Output;

   /// <summary>
   /// Copies the memory of current buffer onto newly allocated array.
   /// </summary>
   /// <returns></returns>
   function BufferToArray: TArray<Byte>;
   //
	 /// <summary>
	 /// Returns the data type for the tensor.
	 /// </summary>
	 /// <value>The type of the tensor.</value>
   property  TensorDataType: TF_DataType    read GetTensorDataType;
   property  bytesize      : UInt64         read GetByteSize;
   property  dtypesize     : UInt64         read GetDataTypeSize;
   property  size          : UInt64         read GetSize;
   property  buffer        : Pointer        read GetData;
   property  ndim          : Tarray<UInt64> read GetDim;
   property  op            : TFOperation    read m_Op;
   property  value_index   : Integer        read m_value_index;
   property  override_dtype: TF_DataType    read m_override_dtype;
   property  id            : Int64          read m_id;
   property  graph         : TFGraph        read m_graph;
   property  Shape         : TFShape        read GetShape write Setshape;
   property  Device        : string         read FDevice;
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
   property  isList           :    Boolean         read m_isList write m_isList;
   property  rank             :    Integer         read GetRank;
   property  DeallocatorCalled:    Boolean         read m_lDeallocator_called;
   property  isCreatedInGraphMode: Boolean         read m_isCreatedInGraphMode;
   property  Value: TValue                         read GetValue;
   property  EagerTensorHandle: PTFE_TensorHandle  read m_eagerTensorHandle write m_eagerTensorHandle;
   property  name: string                          read GetName;

   property Dtype  : TF_DataType       read GetType;
end;

TFTensors = class (TEnumerableBase<TFTensor>)
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
   constructor Create(tensors: TArray<TFTensor>); overload;
   constructor Create(tensor: TFTensor); overload;
   constructor Create(t: Tuple<TFTensor,TFTensor>); overload;
   constructor Create(tensors : TList<TFTensor>); overload;
   procedure   Add(tensor : TFTensor);
   procedure   AddRange(tensors : TArray<TFTensor>);
   procedure   Insert(idx : Integer ;tensor : TFTensor);
   function    ToTensor(tensors: TFTensors): TFTensor;
   function    ToArray(tensors: TFTensors): TArray<TFTensor>; reintroduce;
   function    ToString: string; override;
   procedure   Deconstruct(var a: TFTensor; var b : TFTensor);

   function GetEnumerator: IEnumerator<TFTensor>; reintroduce;

   property Item[idx: Integer] : TFTensor read GetTensor write SetTensor; default;
   property dtype   : TF_DataType read Fdtype ;
   property shape   : TFShape     read Fshape ;
   property rank    : Integer     read FRank ;
   property graph   : TFGraph     read Fgraph ;
   property IsList  : Boolean     read FIsList write FIsList ;
   property iLength : Integer     read FiLength ;
   property IsCreatedInGraphMode  : Boolean     read FIsCreatedInGraphMode ;
end;


/// <summary>
/// Represents a specific output of an operation on a tensor.
/// </summary>
/// <remarks>
/// <para>
/// TFOutput objects represent one of the outputs of an operation in the graph
/// (TFGraph).  Outputs have a data type, and eventually a shape that you can
/// retrieve by calling the <see cref="M:TensorFlow.TFGraph.GetShape"/> method.
/// </para>
/// <para>
/// These can be passed as an input argument to a function for adding operations
/// to a graph, or to the TFSession's Run and GetRunner method as values to be
/// fetched.
/// </para>
/// </remarks>
TFOutput = class
private
  m_oOper: TFOperation;
  /// <summary>The index of the output within oper.</summary>
  m_iIdx:  Integer;
  function GetNumConsumers(): Integer;
  function GetDataType(): TF_DataType;
public
  /// <summary>
  /// Initializes a new TFOutput instance.
  /// </summary>
  /// <param name="operation">The operation to which to attach the output.</param>
  /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
  constructor Create(operation: TFOperation; idx: Integer = 0); overload;
  /// <summary>
  /// Initializes a new TFOutput instance from another TFOutput
  /// </summary>
  /// <param name="operation">The other TFOutput that is having its operation attached.</param>
  /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
  constructor Create(output: TFOutput; idx: Integer = 0); overload;
  destructor  Destroy; override;
  //
  function MMLck(): TFOutput;
  //
  /// <summary>
  /// Convert this instance of TFOutput to native TF_Output.
  /// </summary>
  function ToTF_Output(): TF_Output;
   /// <summary>String representation of this record.</summary>
  function ToString(): TFString; reintroduce;
  /// <summary>
  /// Gets the number consumers.
  /// </summary>
  /// <value>The number consumers.</value>
  /// <remarks>
  /// This number can change when new operations are added to the graph.
  /// </remarks>
  property NumConsumers: Integer read GetNumConsumers;
  property Oper:  TFOperation    read m_oOper write m_oOper;
  property Idx:   Integer        read m_iIdx  write m_iIdx;
  property OutputType: TF_DataType read GetDataType;
end;
/// <summary>
/// TFGraph name scope handle
/// </summary>
/// <remarks>
/// Instances of this class when disposed restore the CurrentNameScope to the
/// value they had when the TFGraph.WithScope method was called.
/// </remarks>
TFScope = class(TInterfacedObject)
 private
   m_oContainer: TFGraph;
   m_sName:      TFString;
 public
   constructor Create(container: TFGraph);
   destructor  Destroy; override;
   /// <summary>
   /// Pops the name space to the previous namescope in use.
   /// </summary>
   /// <remarks>Call <see cref="Dispose"/> when you are finished using the <see cref="T:TensorFlow.TFScope"/>
   /// to restore the previous name scope in use in the <see cref="T:TensorFlow.TFGraph"/>.
   /// </remarks>
   procedure Dispose;
end;
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
TFShapeHelper = record Helper for TFShape
 private
   function  GetSize: Int64;
   function  GetnDim: Integer;
 public
   constructor Create(dims: TArray<TF_int64_t>); overload;
   class function Scalar: TFShape; static;

   class operator Implicit(a : TFShape): TFTensor;
   class operator Implicit(a : TFShape): TArray<Integer>;
   
   function ToString: string;
   function is_compatible_with(shape2:TFShape):Boolean;
   function IsScalar: Boolean;
   function IsNull: Boolean;
   function IsNil: Boolean;
   function Equals(target: TValue): Boolean; reintroduce;
   //
   property ndim: Integer read GetnDim;
   /// <summary>
   ///     Returns the size this shape represents.
   /// </summary>
   property Size: Int64 read GetSize;
end;
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
   m_sOpType:   TFString;
   m_sOperName: TFString;
   m_oGraph:    TFGraph;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(graph: TFGraph; opType, operName: TFString);
   destructor  Destroy; override;
   function SetDevice(device: TFString): TFOperationDesc;
   function SetAttrType(const attrName: TFString; dataType: TF_DataType): TFOperationDesc; overload;
   function SetAttrType(const attrName: TFString; dataTypes: TArray<TF_DataType>): TFOperationDesc; overload;
   function SetAttrShape(attrName: TFString; shape: TFShape): TFOperationDesc; overload;
   function SetAttrShape(attrName: TFString; shape: TArray<TFShape>): TFOperationDesc; overload;
 	 function SetAttr(const attrName: TFString; value: TFString): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; values: TArray<TFString>): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; value: Int64): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; values: TArray<Int64>): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; value: Single): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; values: TArray<Single>): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; value: Boolean): TFOperationDesc; overload;
   function SetAttr(const attrName: TFString; values: TArray<Boolean>): TFOperationDesc; overload;
	 function SetAttr(const attrName: TFString; tensor: TFTensor; status: TFStatus = Nil): TFOperationDesc; overload;
	 /// <summary>
	 /// Adds the specified input to the operation
	 /// </summary>
	 /// <returns>The input.</returns>
	 /// <param name="input">Input.</param>
	 function AddInput(input: TFOutput): TFOperationDesc;
   /// <summary>
   /// Adds a series of inputs to the operation.
   /// </summary>
   /// <param name="inputs">Inputs, this is a params array for your convenience.</param>
   function AddInputs (inputs: TArray<TFOutput>): TFOperationDesc;
   /// <summary>
   /// Turns the operation description into an actual operation in the graph.
   /// </summary>
   /// <returns>The operation on success, or null on error.</returns>
   /// <param name="status">Optional status, on failure the operation is not added to the graph.
   /// If you pass null (the default), this operation throws on error conditions.</param>
   function FinishOperation (status: TFStatus = Nil): TFOperation;
end;

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
   m_oGraph  :  TFGraph;
   m_name    : string;
   m_id_value: Integer;
   m_is_stateful : Boolean;
   m_NumOutputs  : Integer;
    function Getname: string;
    function OutputType(index: Integer): TF_DataType;
    function GeNumOutputs: Integer;
    function GetDevice: string;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(graph: TFGraph; hnd: Pointer);overload;
   constructor Create(node_def       : TNodeDef;
                      g              : TFGraph;
                      inputs         : TArray<TFTensor> = [];
                      output_types   : TArray<TF_DataType> = [];
                      control_inputs : TArray<ITensorOrOperation> = [];
                      input_types    : TArray<TF_DataType> = [];
                      original_op    : string = '';
                      op_def         : POpDef = nil);overload;
   destructor  Destroy; override;
   //
   function MMLck(): TFOperation;
   //
   property Graph     : TFGraph  read m_oGraph;
   property Device    : string   read GetDevice;
   property name      : string   read Getname;
   property NumOutputs: Integer  read GeNumOutputs;
   property id_value: Integer  read m_id_value write m_id_value;
   // Inherithed fom ITensorOrOperation
   property Outputs: TArray<TensorFlow.DApi.TFTensor>  read FOutputs;

end;
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
   m_sCurrentNameScope:     TFString;
   m_oValues:               TDictionary<String, Integer>;
   m_oPendingInitVariables: TList<TFOperation>;
   m_oScopes:               TList<TFScope>;
   m__next_id_counter:      Integer;
   /// <summary>
   /// True if the graph is considered "finalized".  In that case no
   /// new operations can be added.
   /// </summary>
   m_building_function : Boolean;
   m_finalized   : Boolean;
   m_collections: TDictionary<string, TObject>;
   function MakeUnique(const name: TFString): TFString;
   procedure _check_not_finalized;
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   _name_stack        :     TFString;
   function name_scope(name: TFString) : TFString;
   function get_name_scope: TFString;
   /// <summary>
   /// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
   /// </summary>
   constructor Create;
   destructor  Destroy; override;
   function NextId:Integer;
   //
   /// <summary>
   /// Creates a new namescope by setting the scope to the description provided.
   /// </summary>
   /// <returns>A new scope that will remain in use until the return TFScope is disposed.</returns>
   /// <param name="nameScopeDesc">The namescope description, if the value is null, this
   /// will reset the toplevel namescope to be the empty value. </param>
   /// <remarks>
   /// <para>
   /// To more easily name your operations and group then, you can use the
   /// WithScope method to set a current name scope that alter the complete name
   /// of an operation added to the graph.
   /// </para>
   /// <para>
   /// The graph starts with a scope set to the empty string, you can introduce new
   /// scopes by calling WithScope, and can be conveniently used with the C# using
   /// statement, like this:
   /// </para>
   /// <code>
   /// Assert(graph.CurrentNamescope, '');
   /// with (var nested = graph.WithScope('nested')) do begin
   ///    Assert(graph.CurrentNameScope, 'nested');
   ///    with (var inner = graph.WithScope('inner')) do begin
   ///        Assert(graph.CurrentNameScope, 'nested/inner');
   ///    end;
   /// end;
   /// </code>
   /// </remarks>
   function WithScope(nameScopeDesc: TFString): TFScope;
   /// <summary>
   /// Make a ident name with current scope name
   /// </summary>
   function MakeName(const operName, userName: TFString; var nameBuf: TFString): PTFChar;
   /// <summary>
   /// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name,
   /// or null if the named operation does not exist in the graph.
   /// </summary>
   /// <param name="name">Name to lookup.</param>
   function GetOpByName(const Name: TFString): TFOperation;
   /// <summary>
   ///  Returns the tensor shape for the specific output pparameters as an array of longs.
   /// </summary>
   /// <returns>Nil for single dimension.</returns>
   /// <param name="output">The output operation to probe.</param>
   /// <param name="dims">Array for dimensions (var).</param>
   /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
   function GetShape(output: TFOutput; var dims: TArray<TF_int64_t>; status: TFStatus = Nil): PTF_int64_t;
   /// <summary>
   /// Registers a specified variable as an initialization variable.
   /// </summary>
   /// <param name="variable">Variable to register.</param>
   /// <remarks>
   /// <para>
   /// This is a convenience method to track the variables that need to be initialized in the graph,
   /// you can retrieve the list of all those variables by calling the <see cref="M:TensorFlow.TFGraph.GetGlobalVariablesInitializer"/>
   /// which will return this list and clear the state at that point.
   /// </para>
   /// <para>
   /// You typically use this method from helper methods to register all the variables that you want
   /// initialized, and a higher level method will retrieve all these variables and initialize them
   /// at their convenience.
   /// </para>
   /// </remarks>
   procedure AddInitVariable(variable: TFOperation);
   /// <summary>
   /// Gets the list of all registered global variables.
   /// </summary>
   /// <returns>The array of variables that should be initialized.</returns>
   /// <remarks>
   /// After this method is invoked the list of pending initialization variables
   /// is cleared.
   /// </remarks>
   function GetGlobalVariablesInitializer(var variables: TArray<TFOperation>): Boolean;
   //
   procedure add_to_collection<T>(name: string;value: T); overload;
   procedure add_to_collection<T>(names: TList<string>; value: T); overload;
   function  get_collection(name: string; scope: string = '') : TObject; overload;
   function  get_collection<T>(name: string; scope: string = ''): TList<T>;overload;
   function  get_collection_ref<T>(name: string): TList<T>;
   function  GetOpDef(tipo : string): TOpDef;
   procedure gExit;
   function  NewOperation(opType,opName: string):TFOperationDesc;
   /// <summary>
   /// Returns a context manager that makes this `Graph` the default graph.
   /// Must call Exit() to pop graph
   /// </summary>
   /// <returns></returns>
   function as_default: TFGraph;
   //
   property CurrentNameScope: TFString read m_sCurrentNameScope write m_sCurrentNameScope;
   property building_function: Boolean read m_building_function;
   property next_id_counter  : Integer read  m__next_id_counter;
end;
/// <summary>
/// Use the TFSessionRunner class to easily configure inputs, outputs and targets to be passed to the session runner.
/// </summary>
/// <remarks>
/// <para>
/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
/// to construct the parameters that will be passed to the TFSession.Run method.
/// </para>
/// <para>
/// Instances of this class are created by calling the GetRunner method on the TFSession.
/// </para>
/// <para>
/// The various methods in this class return an instance to the Runner itsel, to allow
/// to easily construct chains of execution like this:
/// </para>
/// <code>
/// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ();
/// </code>
/// <para>
/// You do not need to chain the operations, this works just the same:
/// </para>
/// <code>
/// runner = session.GetRunner ();
/// runner.AddInput(myInput);
/// runner.Fetch(myOutput);
/// var results = runner.Run();
/// </code>
/// </remarks>
TFSessionRunner = class
 private
   m_oSession:      TFSession;
   m_aInputs:       TList<TFOutput>;
   m_aInputValues:  TList<TFTensor>;
   m_aOutputs:      TList<TFOutput>;
   m_aOutputValues: TList<TFTensor>;
   m_aTargets:      TList<TFOperation>;
   m_oRunMetadata:  TFBuffer;
   m_oRunOptions:   TFBuffer;
   /// <summary>
   /// Parses user strings that contain both the operation name and an index.
   /// </summary>
   function  ParseOutput(operationName: TFString): TFOutput;
   procedure ResetOutputs();
   procedure ResetOutputValues();
   procedure DeleteLists();
 public
   constructor Create(session: TFSession);
   destructor  Destroy; override;
   procedure Reset();
   /// <summary>
   /// Adds an input to the session
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="input">Incoming port.</param>
   /// <param name="value">Value to assing to the incoming port.</param>
   function AddInput(input: TFOutput; value: TFTensor): TFSessionRunner; overload;
   /// <summary>
   /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
   /// <param name="value">Value to assing to the incoming port.</param>
   function AddInput(inputName: TFString; value: TFTensor): TFSessionRunner; overload;
   /// <summary>
   /// Add the specified operation as the ones to be retrieved.
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="target">One target.</param>
   function AddTarget(target: TFOperation): TFSessionRunner; overload;
   /// <summary>
   /// Adds the specified operations as the ones to be retrieved.
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="targets">One or more targets.</param>
   function AddTarget(targets: TArray<TFOperation>): TFSessionRunner; overload;
   /// <summary>
   /// Adds the specified operation name as the ones to be retrieved.
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="targetName">One target names.</param>
   function AddTarget(targetName: TFString): TFSessionRunner; overload;
   /// <summary>
   /// Adds the specified operation names as the ones to be retrieved.
   /// </summary>
   /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
   /// <param name="targetNames">One or more target names.</param>
   function AddTarget(targetNames: TArray<TFString>): TFSessionRunner; overload;
   /// <summary>
   /// Makes the Run method return the index-th output of the tensor referenced by operation.
   /// </summary>
   /// <returns>The instance of runner, to allow chaining operations.</returns>
   /// <param name="operationName">The name of the operation in the graph.</param>
   /// <param name="idx">The index of the output in the operation.</param>
   function Fetch(operationName: TFString; idx: Integer): TFSessionRunner; overload;
   /// <summary>
   /// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
   /// </summary>
   /// <returns>The instance of runner, to allow chaining operations.</returns>
   /// <param name="operationName">The name of the operation in the graph, which might be a simple name, or it might be name:index,
   /// where the index is the .</param>
   function Fetch(operationName: TFString): TFSessionRunner; overload;
   /// <summary>
   /// Makes the Run method return the output of the tensor referenced by output
   /// </summary>
   /// <returns>The instance of runner, to allow chaining operations.</returns>
   /// <param name="output">The output referencing a specified tensor.</param>
   function Fetch(output: TFOutput): TFSessionRunner; overload;
   /// <summary>
   /// Makes the Run method return the output of all the tensor referenced by outputs.
   /// </summary>
   /// <returns>The instance of runner, to allow chaining operations.</returns>
   /// <param name="outputs">The outputs referencing a specified tensor.</param>
   function Fetch(outputs: TArray<TFOutput>): TFSessionRunner; overload;
   /// <summary>
   /// Makes the Run method return the output of all the tensor referenced by outputs.
   /// </summary>
   /// <returns>The instance of runner, to allow chaining operations.</returns>
   /// <param name="outputs">The output sreferencing a specified tensor.</param>
   function Fetch(outputNames: TArray<TFString>): TFSessionRunner; overload;
   /// <summary>
   /// Helper function to convert one TFOutput to tensorflow data array TF_Output
   /// </summary>
   class function ToArray_TF_Output(outp: TFOutput; var outputBuf: TArray<TF_Output>): PTF_Output; overload;
   /// <summary>
   /// Helper function to convert a list of TFOutput's to tensorflow data array TF_Output
   /// </summary>
   class function ToArray_TF_Output(outp: TList<TFOutput>; var outputBuf: TArray<TF_Output>): PTF_Output; overload;
   /// <summary>
   /// Helper function to convert a list of TFTensor's to tensorflow data array PTF_Tensor
   /// </summary>
   class function ToArray_TF_Tensor(tensors: TList<TFTensor>; var tensorBuf: TArray<PTF_Tensor>): PTF_Tensor;
   /// <summary>
   /// Helper function to convert a list of TFOperation's to tensorflow data array PTF_Operation
   /// </summary>
   class function ToArray_TF_Operation(operations: TList<TFOperation>; var opsBuf: TArray<PTF_Operation>): PTF_Operation;
   /// <summary>
   ///  Execute the graph fragments necessary to compute all requested fetches.
   /// </summary>
   /// <returns>One TFTensor for each call to Fetch that you made, in the order that you made them.</returns>
   /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
   function Run(status: TFStatus = Nil): TFTensor;  overload;
   /// <summary>
   /// Run the specified operation, by adding it implicity to the output, single return value
   /// </summary>
   /// <param name="operation">The output of the operation.</param>
   /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
   /// <remarks>
   /// This method is a convenience method, and when you call it, it will clear any
   /// calls that you might have done to Fetch() and use the specified operation to Fetch
   /// instead.
   /// </remarks>
   function Run(operation: TFOutput; status: TFStatus = Nil): TFTensor; overload;
   //
   property RunMetadata: TFBuffer read m_oRunMetadata write m_oRunMetadata;
   property RunOptions:  TFBuffer read m_oRunOptions  write m_oRunOptions;
end;
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
   m_oGraph:  TFGraph;
   m_oSessionRunner: TFSessionRunner;
   procedure Init(graph: TFGraph; status: TFStatus = Nil);
 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   constructor Create(status: TFStatus = Nil); overload;
   constructor Create(hnd: Pointer; graph: TFGraph); overload;
   constructor Create(graph: TFGraph; sessionOptions: TFSessionOptions; status: TFStatus = Nil); overload;
   constructor Create(graph: TFGraph; status: TFStatus = Nil); overload;
   destructor  Destroy; override;
   /// <summary>
   /// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
   /// </summary>
   /// <returns>The runner.</returns>
   /// <remarks>
   /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
   /// to construct the parameters that will be passed to the TFSession.Run method.
   ///
   /// The Run method will return an array of TFTensor values, one for each invocation to the Fetch method.
   /// </remarks>
   function GetRunner(): TFSessionRunner;
   /// <summary>
   /// Executes a pipeline given the specified inputs, inputValues, outputs, targetOpers, runMetadata and runOptions.
   /// A simpler API is available by calling the <see cref="M:GetRunner"/> method which performs all the bookkeeping
   /// necessary.
   /// </summary>
   /// <returns>An array of tensors fetched from the requested outputs.</returns>
   /// <param name="inputs">Inputs nodes.</param>
   /// <param name="inputValues">Input values.</param>
   /// <param name="outputs">Output nodes.</param>
   /// <param name="outputValues">Output values.</param>
   /// <param name="targetOpers">Target operations to execute.</param>
   /// <param name="runMetadata">Run metadata.</param>
   /// <param name="runOptions">Run options.</param>
   /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
   function Run(inputs: TArray<TF_Output>; inputValues: TArray<PTF_Tensor>;
                outputs: TArray<TF_Output>; var outputValues: TArray<PTF_Tensor>;
                targetOpers: TArray<PTF_Operation> = Nil;
                runMetadata: PTF_Buffer = Nil; runOptions: PTF_Buffer = Nil;
                status: PTF_Status = Nil): Boolean;
   //
   property    Graph: TFGraph read m_oGraph write m_oGraph;
end;
/// <summary>
/// With this memory management enviroment you can collect tensor's,
/// operation's and TFOutput's.
/// </summary>
TFMMEnv = class
   m_lIsActive: Boolean;
   m_lSessionActive: Boolean;
   m_aTensorListOutside:  TList<TFTensor>;
   m_aTensorList:  TList<TFTensor>;
   m_aOutputList:  TList<TFOutput>;
   m_aOperationList: TList<TFOperation>;
public
   //
   class var g_oMMEnv: TFMMEnv;
   //
   constructor Create;
   destructor  Destroy; override;
   //
   class function GetMMEnv(): TFMMEnv;
   //
   class procedure StartMM;
   class procedure EndMM;
   //
   /// <summary>
   /// Add Tensor to MMEnv
   /// </summary>
   /// <param name="tensor">Tensor instance.</param>
   procedure AddTensor(tensor: TFTensor);
   /// <summary>
   /// Add a TFOutput to MMEnv
   /// </summary>
   /// <param name="output">TFOutput instance.</param>
   procedure AddOutput(output: TFOutput);
   /// <summary>
   /// Add a TFOperation to MMEnv
   /// </summary>
   /// <param name="output">TFOperation instance.</param>
   procedure AddOperation(oper: TFOperation);
   /// <summary>
   /// Reset the session lists.
   /// </summary>
   procedure ResetSessionLists;
   /// <summary>
   /// Reset the all lists.
   /// </summary>
   procedure ResetAllLists;
   //
   property  IsActive:      Boolean read m_lIsActive      write m_lIsActive;
   property  SessionActive: Boolean read m_lSessionActive write m_lSessionActive;
end;

// Tensor-Helpers
// -----------------------------------------------------------------------------
/// <summary>
/// Converts an integer into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the integer value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(value: Integer): TFTensor; overload; inline;
/// <summary>
/// Converts a float into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the float value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(value: Single): TFTensor; overload; inline;
/// <summary>
/// Converts a string into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the string value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(const value: TFString): TFTensor;  overload; inline;
/// <summary>
/// Converts a Integer array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing the Integer value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<Integer>): TFTensor; overload; inline;
/// <summary>
/// Converts a Integer 2D-array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing the Integer value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<TArray<Integer>>): TFTensor; overload; inline;
/// <summary>
/// Converts a float array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing the float values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<Single>): TFTensor; overload; inline;
/// <summary>
/// Converts a float 2D-array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing the float values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<TArray<Single>>): TFTensor; overload; inline;
/// <summary>
/// Converts a string array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing string values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<TFString>): TFTensor; overload; inline;
/// <summary>
/// Converts a string 2D-array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing string values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _T(values: TArray<TArray<TFString>>): TFTensor; overload; inline;
/// <summary>
/// Converts a byte into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the byte value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TByte(value: Byte): TFTensor; inline;
/// <summary>
/// Converts a boolean into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the integer value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TBool(value: Boolean): TFTensor; inline;
/// <summary>
/// Converts a Int16 into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the short value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TInt16(value: Int16): TFTensor; inline;
/// <summary>
/// Converts a Int64 into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the long value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TInt64(value: Int64): TFTensor; inline;
/// <summary>
/// Converts a double into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing the double value.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TDouble(value: double): TFTensor; inline;
/// <summary>
/// Converts a byte array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing byte values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TAByte(values: TArray<Byte>): TFTensor; overload; inline;
/// <summary>
/// Converts a byte 2D-array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing byte values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TAByte(values: TArray<TArray<Byte>>): TFTensor; overload; inline;
/// <summary>
/// Converts a boolean array into a 1-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing boolean values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TABool(values: TArray<Boolean>): TFTensor; overload; inline;
/// <summary>
/// Converts a boolean array into a 2-dimensional, 1-valued tensor.
/// </summary>
/// <returns>The tensor representing boolean values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TABool(values: TArray<TArray<Boolean>>): TFTensor; overload; inline;
/// <summary>
/// Converts a Int64 array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing Int64 values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TAInt64(values: TArray<Int64>): TFTensor; overload; inline;
/// <summary>
/// Converts a Int64 array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing Int64 values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TAInt64(values: TArray<TArray<Int64>>): TFTensor; overload; inline;
/// <summary>
/// Converts a double array into a 1-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing double values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TADouble(values: TArray<double>): TFTensor; overload; inline;
/// <summary>
/// Converts a double array into a 2-dimensional, array tensor.
/// </summary>
/// <returns>The tensor representing double values.</returns>
/// <param name="value">Value to initialize the tensor with.</param>
function _TADouble(values: TArray<TArray<double>>): TFTensor; overload; inline;
function get_datatype_size(tipo: TF_DataType): Integer;

implementation
   uses TensorFlow,TensorFlow.Ops, Tensorflow.Utils,TensorFlow.Framework;

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
procedure TFSessionOptions.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteSessionOptions(hnd);
end;
procedure TFSessionOptions.SetTarget(target: TFString);
begin
 if not Assigned(Handle) then
   ObjectDisposedException();
 TF_SetTarget(Handle, _PTFChar(target));
end;
procedure TFSessionOptions.SetConfig(protoData: Pointer; lng: Integer; status: TFStatus = Nil);
var
 l_oStatus: TFStatus;
begin
 if not Assigned(Handle) then
   ObjectDisposedException();
 l_oStatus := TFStatus.Setup(status);
 TF_SetConfig(Handle, protoData, TF_size_t(lng), l_oStatus.Handle);
end;
//------------------------------------------------------------------------------
//----------------------------- TFTensor ---------------------------------------
//------------------------------------------------------------------------------
constructor TFTensor.Create(hnd: Pointer);
begin
 inherited Create(hnd);
 m_lDeallocator_called := False;
 self.MMLck();
end;
constructor TFTensor.Create(value: Boolean);
var
 l_iLen:     TF_size_t;
 l_pBoolean: PBoolean;
begin
 inherited Create(Nil);
 l_iLen   := sizeof(Boolean);
 GetMem(l_pBoolean, l_iLen);
 l_pBoolean^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_BOOL), Nil, 0,
                      l_pBoolean, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Integer);
var
 l_iLen:   TF_size_t;
 l_pInt32: PInt32;
begin
 inherited Create(Nil);
 Assert(sizeof(Integer) = sizeof(Int32));
 l_iLen   := sizeof(Int32);
 GetMem(l_pInt32, l_iLen);
 l_pInt32^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT32), Nil, 0,
                      l_pInt32, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Int16);
var
 l_iLen:   TF_size_t;
 l_pInt16: PInt16;
begin
 inherited Create(Nil);
 l_iLen   := 1 * sizeof(Int16);
 GetMem(l_pInt16, l_iLen);
 l_pInt16^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT16), Nil, 0,
                      l_pInt16, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Int64);
var
 l_iLen:   TF_size_t;
 l_pInt64: PInt64;
begin
 inherited Create(Nil);
 l_iLen   := 1 * sizeof(Int64);
 GetMem(l_pInt64, l_iLen);
 l_pInt64^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT64), Nil, 0,
                      l_pInt64, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Byte);
var
 l_iLen:  TF_size_t;
 l_pByte: PByte;
begin
 inherited Create(Nil);
 l_iLen   := sizeof(Byte);
 GetMem(l_pByte, l_iLen);
 l_pByte^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT8), Nil, 0,
                      l_pByte, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Single);
var
 l_iLen:    TF_size_t;
 l_pSingle: PSingle;
begin
 inherited Create(Nil);
 l_iLen   := sizeof(Single);
 GetMem(l_pSingle, l_iLen);
 l_pSingle^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_FLOAT), Nil, 0,
                      l_pSingle, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(value: Double);
var
 l_iLen:    TF_size_t;
 l_pDouble: PDouble;
begin
 inherited Create(Nil);
 l_iLen   := sizeof(Double);
 GetMem(l_pDouble, l_iLen);
 l_pDouble^ := value;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_Double), Nil, 0,
                      l_pDouble, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor  TFTensor.Create(const value: TFString);
var
 l_iLen:    TF_size_t;
 l_pChar, l_pSrc:  PTFChar;
 l_sEncodeStr: TFString;
begin
 inherited Create(Nil);
 l_sEncodeStr := _EncodeStr(value);
 l_iLen   := Length(l_sEncodeStr) * sizeof(TFChar);
 GetMem(l_pChar, l_iLen);
 l_pSrc := _PTFChar(l_sEncodeStr);
 Move(l_pSrc^, l_pChar^, l_iLen);
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_STRING), Nil, 0,
                      l_pChar, l_iLen, Deallocator_For_TensorDatas,
                      @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Byte>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAByte(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT8), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Byte>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAByte(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT8), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Boolean>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemABool(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_BOOL), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Boolean>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemABool(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_BOOL), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Integer>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMem(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT32), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Integer>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMem(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT32), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Int16>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAInt16(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT16), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Int16>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAInt16(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT16), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Int64>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAInt64(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT64), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Int64>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemAInt64(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_INT64), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Single>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMem(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_FLOAT), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Single>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMem(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_FLOAT), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<Double>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemADouble(values, l_iDim, l_iByteSize);
 l_aDims[0] := l_iDim;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_DOUBLE), PTF_int64_t(@l_aDims[0]), 1,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;
constructor TFTensor.Create(values: TArray<TArray<Double>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
begin
 inherited Create(Nil);
 l_pData := _AllocMemADouble(values, l_iDim1, l_iDim2, l_iByteSize);
 l_aDims[0] := l_iDim1;
 l_aDims[1] := l_iDim2;
 m_lDeallocator_called := False;
 Handle := TF_NewTensor(Int32(TF_DOUBLE), PTF_int64_t(@l_aDims[0]), 2,
                        l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                        @m_lDeallocator_called);
 self.MMLck();
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
end;

constructor TFTensor.Create(values: TArray<TFString>);
var
 l_iDim, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..0] of TF_int64_t;
 l_aEncodeValues: TArray<TFString>;
begin
 inherited Create(Nil);
 if _EncodeStrings(values, l_aEncodeValues) then begin
   l_pData := _AllocMem(l_aEncodeValues, l_iDim, l_iByteSize);
   l_aDims[0] := l_iDim;
   Handle := TF_NewTensor(Int32(TF_STRING), PTF_int64_t(@l_aDims[0]), 1,
                          l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                          @m_lDeallocator_called);
 end;
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 self.MMLck();
end;
constructor TFTensor.Create(values: TArray<TArray<TFString>>);
var
 l_iDim1, l_iDim2, l_iByteSize: TF_int64_t;
 l_pData: Pointer;
 l_aDims: array[0..1] of TF_int64_t;
 l_aEncodeValues: TArray<TArray<TFString>>;
begin
 inherited Create(Nil);
 if _EncodeStrings(values, l_aEncodeValues) then begin
   l_pData := _AllocMem(l_aEncodeValues, l_iDim1, l_iDim2, l_iByteSize);
   l_aDims[0] := l_iDim1;
   l_aDims[1] := l_iDim2;
   Handle := TF_NewTensor(Int32(TF_STRING), PTF_int64_t(@l_aDims[0]), 2,
                          l_pData, l_iByteSize, Deallocator_For_TensorDatas,
                          @m_lDeallocator_called);
 end;
 m_isCreatedInGraphMode := not tf.executing_eagerly;
 m_eagerTensorHandle    := nil;
 self.MMLck();
end;

constructor TFTensor.Create(shape: TFShape; dtype: TF_DataType);
begin
    Self.Create(shape,dtype,nil);
end;

constructor TFTensor.Create(op: TFOperation; value_index: Integer; dtype: TF_DataType);
begin
     m_op    := op;
     m_graph := m_op.Graph;

     m_value_index := value_index;
     m_override_dtype := dtype;
     m_id := TOps.uid;
     m_isCreatedInGraphMode := not tf.executing_eagerly;
     m_eagerTensorHandle    := nil;
end;

constructor TFTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    Create( InitTensor<Byte>(bytes,shape,dtype) );
    m_isCreatedInGraphMode := not tf.executing_eagerly;
    m_eagerTensorHandle    := nil;
end;

constructor TFTensor.Create(shape: TFShape; dtype: TF_DataType; data: Pointer);
var
 i           : Integer;
 l_iNumValues: TF_int64_t;
 l_pTensor   : PTF_Tensor;
 l_pData     : Pointer;
begin
     inherited Create(Nil);
     l_iNumValues := 1;
     var num_dims := Length(shape.Dims);
     var dims     := shape.Dims;
     var dTypeSize:= get_datatype_size(dtype);

     for i := 0 to num_dims-1 do
       l_iNumValues := l_iNumValues * dims[i];

     var _length : TF_size_t := l_iNumValues * dTypeSize;

     l_pTensor := TF_AllocateTensor(Integer(dtype), PTF_int64_t(@dims[0]), num_dims, _length);
     l_pData   := TF_TensorData(l_pTensor);
     if l_pData = nil then
        raise Exception.Create('AllocateTensor failed.');

     if Assigned(data)  then
       Move(data^, l_pData^, _length);

     Handle := l_pTensor;

     m_isCreatedInGraphMode := not tf.executing_eagerly;
     m_eagerTensorHandle    := nil;
     self.MMLck();
end;

destructor  TFTensor.Destroy;
begin
 if m_lDeallocator_called then
   Handle := Nil;
 inherited Destroy;
end;

procedure TFTensor.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteTensor(hnd);
end;

class function TFTensor.StringTensor(srcArray: TArray<TArray<byte>>; shape: TFShape):TFTensor;
var
  l_pTensor   : PTF_Tensor;
  size        : TF_int64_t;
  tstr        : PTFString;
begin

     var num_dims := Length(shape.Dims);
     var dims     := shape.Dims;
     size := 0;
     for var i := 0 to num_dims-1 do
       size := size + dims[i];

     l_pTensor := TF_AllocateTensor(Int32(TF_STRING), PTF_int64_t(@dims[0]), num_dims, size * TF_TSRING_SIZE);
     tstr      := TF_TensorData(l_pTensor);

     for var i := 0 to Length(srcArray) - 1 do
     begin
          TF_StringInit(tstr);
          TF_StringCopy(tstr, PTFChar(@srcArray[i]), Length(srcArray[i]));
          Inc(PByte(tstr), TF_TSRING_SIZE);
     end;

     Result := TFTensor.Create(l_pTensor);
end;

class function TFTensor.StringTensor(srcArray: TArray<TFString>; shape: TFShape):TFTensor;
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

function TFTensor.ToArray<T>: TArray<T>;
var
  l_pData,l_pVal  : Pointer ;
  l_iFullByteSize : Integer;
  res             : TArray<T>;
begin
    if Tdtypes.as_tf_dtype( TypeInfo(T) ) <> Dtype then
      raise Exception.Create('Required dtype {dtype} mismatch with {typeof(T).as_tf_dtype()}.');

    l_pData := TF_TensorData(Handle);


    if (ndim[0] = 0) or (size = 1) then
    begin
        SetLength(res,1);
        l_pVal  := @res[0];
        l_iFullByteSize := dtypesize;

        Move(l_pData^, l_pVal^, l_iFullByteSize);
        Exit;
    end;

    if (ndim[0] > 1) then
       raise Exception.Create('ToArray - ndim[0] > 1  !!!.');

    SetLength(res,size);
    l_pVal  := @res[0];
    l_iFullByteSize :=  size * dtypesize;
    Move(l_pData^, l_pVal^, l_iFullByteSize);

    Result := res;
end;

function TFTensor.ToString: string;
begin
    Result := Format('tf.Tensor "%s" shape=%s dtype=%s',[name,Shape.ToString,Tdtypes.as_numpy_name(TensorDataType)]);
end;

function TFTensor._as_tf_output: TF_Output;
begin
    if not m_tf_output.HasValue then
    begin
        var o := TFOutput.Create(m_op,m_value_index);
        m_tf_output := o.ToTF_Output;
    end;
    Result := m_tf_output;
end;

function TFTensor.BufferToArray: TArray<Byte>;
var
  l_pData,l_pVal  : Pointer ;
  res             : TArray<Byte>;
begin
    SetLength(res,bytesize);
    l_pData := TF_TensorData(Handle);

    l_pVal  := @res[0];

    Move(l_pData^, l_pVal^, bytesize);


    Result := res;
end;

class function TFTensor.InitTensor(aArray: TArray<Byte>;shape: TFShape; dtype: TF_DataType): TFTensor;
var
  l_pData     : Pointer;
begin
     if dtype = TF_DataType.TF_STRING then
     begin
         var buf : TArray<TArray<byte>>;
         SetLength(buf,1);
         for var i := 0 to Length(aArray) - 1 do
           buf[0][i] := aArray[i];
         Result := StringTensor( buf, TFShape.Create([0]));
     end else
     begin
         l_pData := PByte(@aArray[0]);
         Result  := TFTensor.Create(shape,dtype,l_pData) ;
     end;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<T>;shape: TFShape; dtype: TF_DataType): TFTensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0]);
     Result  := TFTensor.Create(shape,dtype,l_pData) ;
end;
class function TFTensor.InitTensor<T>(aArray: TArray<TArray<T>>; shape: TFShape; dtype: TF_DataType): TFTensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0]);
     Result := TFTensor.Create(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<T>>>; shape: TFShape; dtype: TF_DataType): TFTensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0]);
     Result := TFTensor.Create(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: TFShape; dtype: TF_DataType): TFTensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0]);
     Result := TFTensor.Create(shape,dtype,l_pData) ;
end;

function TFTensor.GetTensorDataType(): TF_DataType;
begin
 if Assigned(Handle) then
   Result := TF_DataType(TF_TensorType(Handle))
 else
   Result := TF_DATATYPE_UNKNOWN;
end;

function TFTensor.GetType: TF_DataType;
begin
    if Handle = nil then
      FDtype := m_override_dtype
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
    if Assigned(Handle) then
     Result := TF_DataTypeSize(Integer(TensorDataType))
   else
     Result := 0;
end;

function TFTensor.GetDim: Tarray<UInt64>;
begin
    if Assigned(Handle) then
     Result := [TF_NumDims(Handle)]
   else begin
      var output := _as_tf_output;
      var ndim := TF_GraphGetTensorNumDims(op.graph, output, tf.Status.Handle);
      Result := [ndim];
   end;
end;

function TFTensor.GetName: string;
var
 opname : string;
begin
    opname := '<unnamed>';
    if m_op <> nil then
      opname := m_op.name;

    F_name := Format('%s:%d',[opname,value_index]);
    Result := F_name
end;

function TFTensor.GetRank: Integer;
begin
    if not Assigned(Handle) then
    begin
        var output :=  _as_tf_output;
        var ndim : Integer := TF_GraphGetTensorNumDims(m_graph.Handle,output,tf.Status.Handle);
        Result := ndim;
    end else
    begin
        Result := TF_NumDims(Handle)
    end;
    m_rank := Result;
end;

function TFTensor.GetShape: TFShape;
begin
    m_shape := default(TFShape);

    if rank < 0 then
        Exit(m_shape);

    var dims := TFShape.Create([rank]);

    if not Assigned(Handle) then
    begin
        TF_GraphGetTensorShape(op.graph.Handle, _as_tf_output(), @dims, rank, tf.Status.Handle);
    end else
    begin
        for var i := 0 to rank -1 do
           dims.Dims[i] := TF_Dim(Handle, i);

        m_shape := dims;
    end;
    Result := m_shape;
end;

procedure TFTensor.Setshape(const Value: TFShape);
begin
    if Self is TEagerTensor then
      if not Shape.is_compatible_with(Value) then
         raise Exception.Create('Tensor''s shape is not compatible.');

    if value.IsNil then
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, nil, -1, tf.Status.Handle)
    else
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, @value.dims, value.ndim, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

function TFTensor.GetSize: UInt64;
begin
    Result := bytesize div dtypesize;
end;

function TFTensor.MMLck(): TFTensor;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 if Assigned(env) then
   env.AddTensor(self);
 Result := self;
end;
function TFTensor.IsArray(var dim1, dim2: TF_int64_t): Boolean;
var
  l_iDims: Integer;
begin
 Result := False;
 dim1   := 0;
 dim2   := 0;
 if Assigned(Handle) then begin
   l_iDims := TF_NumDims(Handle);
   if l_iDims > 2 then
     raise TFException.Create('TFTensor.IsArray: Multidimension Array! -> Change this API!');
   if l_iDims >= 1 then begin
     dim1 := TF_Dim(Handle,0);
     if l_iDims = 2 then begin
       dim2 := TF_Dim(Handle,1);
       if (dim1*dim2) >= 1 then
         Result := True;
     end
     else if dim1 >= 1 then
       Result := True;
   end;
 end;
end;
function TFTensor.GetValue(): TValue;
var
 dim1, dim2: TF_int64_t;
 l_enDataType: TF_DataType;
 l_pData:  Pointer;
 l_pFloat:  PSingle;
 l_pDouble: PDouble;
 l_pInt8:   PInt8;
 l_pInt16:  PInt16;
 l_pInt32:  PInt32;
 l_pInt64:  PInt64;
 l_pBool:   PBoolean;
 l_pChar :  PTFChar;
 l_sDecodeStr: TFString;

begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_DATATYPE_UNKNOWN then begin
   if self.IsArray(dim1, dim2) then
     raise TFException.Create('TFTensor.GetValue: Tensor data is an array!');
   l_pData    := TF_TensorData(Handle);
   case l_enDataType of
     TF_FLOAT: begin
       l_pFloat := PSingle(l_pData);
       Result   := TValue(l_pFloat^);
     end;
     TF_DOUBLE: begin
       l_pDouble:= PDouble(l_pData);
       Result   := TValue(l_pDouble^);
     end;
     TF_INT32: begin
       l_pInt32 := PInt32(l_pData);
       Result   := TValue(l_pInt32^);
     end;
     TF_UINT8: begin
       raise TFException.Create('TFTensor.GetValue: Tensor data type is TF_UINT8!');
     end;
     TF_INT16: begin
       l_pInt16 := PInt16(l_pData);
       Result   := TValue(Integer(l_pInt16^));
     end;
     TF_INT8: begin
       l_pInt8 := PInt8(l_pData);
       Result  := TValue(Integer(l_pInt8^));
     end;
     TF_STRING: begin
       l_pChar   := PTFChar(l_pData);
       l_sDecodeStr := AnsiString(l_pChar);
       Result := TValue(String(l_sDecodeStr));
     end;
     TF_INT64: begin
       l_pInt64 := PInt64(l_pData);
       Result  := TValue(l_pInt64^);
     end;
     TF_BOOL: begin
       l_pBool := PBoolean(l_pData);
       Result  := TValue(l_pBool^);
     end;
   end;
 end;
end;
function TFTensor.GetArray(var values: TArray<Byte>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT8 then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data type is not TF_INT8!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAByte(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Byte>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT8 then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data type is not TF_INT8!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Byte): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAByte(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Boolean>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_BOOL then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data type is not TF_BOOL!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayABool(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Boolean>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_BOOL then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data type is not TF_BOOL!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Bool): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayABool(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Integer>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT32 then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data type is not TF_INT32!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArray(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Integer>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT32 then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data type is not TF_INT32!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Integer): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArray(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Int16>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT16 then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data type is not TF_INT16!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAInt16(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Int16>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT16 then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data type is not TF_INT16!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Int16): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAInt16(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Int64>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT64 then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data type is not TF_INT64!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAInt64(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Int64>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_INT64 then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data type is not TF_INT64!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Int64): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayAInt64(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Single>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_FLOAT then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data type is not TF_FLOAT!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArray(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Single>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_FLOAT then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data type is not TF_FLOAT!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Float): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArray(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<Double>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_DOUBLE then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data type is not TF_DOUBLE!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayADouble(l_pData, dim1, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<Double>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2: TF_int64_t;
 l_pData:  Pointer;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_DOUBLE then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data type is not TF_DOUBLE!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray (Double): Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 Result := _GetArrayADouble(l_pData, dim1, dim2, values);
end;
function TFTensor.GetArray(var values: TArray<TFString>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2, total_byte_size: TF_int64_t;
 l_pData:  Pointer;
 srcArray: TArray<TFString>;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_STRING then
   raise TFException.Create('TFTensor.GetArray (TFString): Tensor data type is not TF_STRING!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray (TFString): Tensor data is not an array!');
 if dim2 > 0 then
   raise TFException.Create('TFTensor.GetArray (TFString): Tensor data array has 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 total_byte_size := TF_TensorByteSize(Handle);
 Result := _GetArray(l_pData, dim1, total_byte_size, srcArray);
 if Result then
   Result := _DecodeStrings(srcArray, values);
end;
function TFTensor.GetArray(var values: TArray<TArray<TFString>>): Boolean;
var
 l_enDataType: TF_DataType;
 dim1, dim2, total_byte_size: TF_int64_t;
 l_pData:  Pointer;
 srcArray: TArray<TArray<TFString>>;
begin
 l_enDataType := self.GetTensorDataType();
 if l_enDataType <> TF_STRING then
   raise TFException.Create('TFTensor.GetArray: Tensor data type is not TF_STRING!');
 if not self.IsArray(dim1, dim2) then
   raise TFException.Create('TFTensor.GetArray: Tensor data is not an array!');
 if dim2 = 0 then
   raise TFException.Create('TFTensor.GetArray: Tensor data array has not 2 dimensions!');
 l_pData:= TF_TensorData(Handle);
 total_byte_size := TF_TensorByteSize(Handle);
 Result := _GetArray(l_pData, dim1, dim2, total_byte_size, srcArray);
 if Result then
   Result := _DecodeStrings(srcArray, values);
end;

// Tensor-Helpers
// -----------------------------------------------------------------------------
function _T(value: Integer): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _T(value: Single): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _T(const value: TFString): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _T(values: TArray<Integer>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _T(values: TArray<TArray<Integer>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _T(values: TArray<Single>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _T(values: TArray<TArray<Single>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _T(values: TArray<TFString>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _T(values: TArray< TArray<TFString>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TByte(value: Byte): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _TBool(value: Boolean): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _TInt16(value: Int16): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _TInt64(value: Int64): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _TDouble(value: double): TFTensor;
begin
 Result := TFTensor.Create(value);
end;
function _TAByte(values: TArray<Byte>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TAByte(values: TArray<TArray<Byte>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TABool(values: TArray<Boolean>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TABool(values: TArray<TArray<Boolean>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TAInt64(values: TArray<Int64>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TAInt64(values: TArray<TArray<Int64>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TADouble(values: TArray<double>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
function _TADouble(values: TArray<TArray<double>>): TFTensor;
begin
 Result := TFTensor.Create(values);
end;
//------------------------------------------------------------------------------
//----------------------------- TFOutput ---------------------------------------
//------------------------------------------------------------------------------
constructor TFOutput.Create(operation: TFOperation; idx: Integer = 0);
begin
 if not Assigned(operation) then
   raise TFException.Create('TFOutput.Create: Argument Null Exception - operation');
 self.m_oOper := operation;
 self.m_iIdx  := idx;
 self.MMLck();
end;
constructor TFOutput.Create(output: TFOutput; idx: Integer = 0);
begin
 if not Assigned(output.Oper) then
   raise TFException.Create('TFOutput.Create: Outputs does not have a valid TFOperation');
 self.m_oOper := output.Oper;
 self.m_iIdx  := output.Idx;
 self.MMLck();
end;
destructor  TFOutput.Destroy;
begin
 inherited Destroy;
end;
function TFOutput.MMLck(): TFOutput;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 if Assigned(env) then
   env.AddOutput(self);
 Result := self;
end;
function TFOutput.ToTF_Output(): TF_Output;
begin
 Result.oper  := self.Oper.Handle;
 Result.index := self.Idx;
end;
function TFOutput.GetDataType(): TF_DataType;
var
 l_oOutp: TF_Output;
begin
 l_oOutp:= self.ToTF_Output();
 Result := TF_DataType(TF_OperationOutputType(l_oOutp));
end;
function TFOutput.GetNumConsumers(): Integer;
begin
 Result := TF_OperationOutputNumConsumers(ToTF_Output());
end;
function  TFOutput.ToString(): TFString;
begin
 Result := TFString(Format('[TFOutput: Operation=%d Index=%d ]', [Integer(self.Oper), self.Idx]));
end;
//------------------------------------------------------------------------------
//----------------------------- TFScope ----------------------------------------
//------------------------------------------------------------------------------
constructor TFScope.Create(container: TFGraph);
begin
 inherited Create;
 m_oContainer := container;
 m_sName      := container.CurrentNameScope;
end;
destructor  TFScope.Destroy;
begin
 inherited Destroy;
end;
procedure TFScope.Dispose;
begin
 m_oContainer.CurrentNameScope := m_sName;
end;
//------------------------------------------------------------------------------
//----------------------------- TFOperationDesc --------------------------------
//------------------------------------------------------------------------------
constructor TFOperationDesc.Create(graph: TFGraph; opType, operName: TFString);
begin
 inherited Create(Nil);
 if not Assigned(graph) then
   raise TFException.Create('TFOperationDesc.Create: Argument Null Exception - graph');
 self.Handle := TF_NewOperation(graph.Handle, _PTFChar(opType), _PTFChar(operName));
 m_oGraph  := graph;
 m_sOpType := opType;
 m_sOperName := operName;
end;
destructor TFOperationDesc.Destroy;
begin
 inherited Destroy;
end;
procedure TFOperationDesc.NativeDispose(hnd: Pointer);
begin
 if Assigned(Handle) then
   TensorFlow.DApiBase.WriteTFProt(Format('TFOperationDescription(%s,%s) was never turned into an TFOperation',[m_sOpType,m_sOperName]));
end;
function TFOperationDesc.SetDevice(device: TFString): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(device) = 0 then
   raise TFException.Create('TFOperationDesc.SetDevice: Argument Null Exception - device');
 TF_SetDevice(PTF_OperationDescription(self.Handle), _PTFChar(device));
 Result := self;
end;
function TFOperationDesc.SetAttrType(const attrName: TFString; dataType: TF_DataType): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttrType: Argument Null Exception - attrName');
 TF_SetAttrType(Handle, _PTFChar(attrName), Integer(dataType));
 Result := self;
end;
function TFOperationDesc.SetAttrType(const attrName: TFString; dataTypes: TArray<TF_DataType>): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttrType: Argument Null Exception - attrName');
 if not Assigned(dataTypes) or (Length(dataTypes) = 0) then
   raise TFException.Create('TFOperationDesc.SetAttrType(...,TArray<TF_DataType>): Argument Null Exception - dataTypes');
 TF_SetAttrTypeList(Handle, _PTFChar(attrName), @(dataTypes[0]), Length(dataTypes));
 Result := self;
end;
function TFOperationDesc.SetAttrShape(attrName: TFString; shape: TFShape): TFOperationDesc;
begin
 if not Assigned(handle) then
   ObjectDisposedException ();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttrShape: Argument Null Exception - attrName');
 if (shape.IsNil) or not Assigned(shape.Dims) then
   TF_SetAttrShape(Handle, _PTFChar(attrName), Nil, -1)
 else
   TF_SetAttrShape(Handle, _PTFChar(attrName), @(shape.Dims[0]), Length(shape.Dims));
 Result := self;
end;
function TFOperationDesc.SetAttrShape(attrName: TFString; shape: TArray<TFShape>): TFOperationDesc;
begin
 //TODO:
end;
function TFOperationDesc.SetAttr(const attrName: TFString; value: TFString): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TFString): Argument Null Exception - attrName');
 TF_SetAttrString(Handle, _PTFChar(attrName), _PTFChar(value), TF_size_t(Length(value)));
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; values: TArray<TFString>): TFOperationDesc;
var
 l_aLengths: TArray<TF_size_t>;
 l_aPTFChar: TArray<PTFChar>;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<TFString>): Argument Null Exception - attrName');
 if not Assigned(values) or (Length(values) = 0) then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<TFString>): Argument Null Exception - values');
 _GetPTFCharArray(values,l_aLengths,l_aPTFChar);
 TF_SetAttrStringList(Handle, _PTFChar(attrName), @(l_aPTFChar[0]), @(l_aLengths[0]), Length(values));
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; value: Int64): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,Int64): Argument Null Exception - attrName');
 TF_SetAttrInt(Handle, _PTFChar(attrName), value);
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; values: TArray<Int64>): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Int64>): Argument Null Exception - attrName');
 if not Assigned(values) or (Length(values) = 0) then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Int64>): Argument Null Exception - values');
 TF_SetAttrIntList(Handle, _PTFChar(attrName), @(values[0]), Length(values));
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; value: Single): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,Single): Argument Null Exception - attrName');
 TF_SetAttrFloat(Handle, _PTFChar(attrName), value);
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; values: TArray<Single>): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Single>): Argument Null Exception - attrName');
 if not Assigned(values) or (Length(values) = 0) then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Single>): Argument Null Exception - values');
 TF_SetAttrFloatList(Handle, _PTFChar(attrName), @(values[0]), Length(values));
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; value: Boolean): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,Boolean): Argument Null Exception - attrName');
 TF_SetAttrBool(Handle, _PTFChar(attrName), value);
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; values: TArray<Boolean>): TFOperationDesc;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Boolean>): Argument Null Exception - attrName');
 if not Assigned(values) or (Length(values) = 0) then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TArray<Boolean>): Argument Null Exception - values');
 TF_SetAttrBoolList(Handle, _PTFChar(attrName), @(values[0]), Length(values));
 Result := self;
end;
function TFOperationDesc.SetAttr(const attrName: TFString; tensor: TFTensor; status: TFStatus = Nil): TFOperationDesc;
var
 l_oStatus: TFStatus;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 if Length(attrName) = 0 then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TFTensor): Argument Null Exception - attrName');
 if not Assigned(tensor) then
   raise TFException.Create('TFOperationDesc.SetAttr(...,TFTensor): Argument Null Exception - tensor');
 l_oStatus := TFStatus.Setup(status);
 TF_SetAttrTensor(Handle, _PTFChar(attrName), tensor.handle, l_oStatus.Handle);
 l_oStatus.CheckMaybeRaise(status);
 Result := self;
end;
function TFOperationDesc.AddInput(input: TFOutput): TFOperationDesc;
var
 l_oInp: TF_Output;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 l_oInp := input.ToTF_Output();
 TF_AddInput(self.Handle, l_oInp);
 Result := self;
end;
function TFOperationDesc.AddInputs(inputs: TArray<TFOutput>): TFOperationDesc;
var
 n, i: Integer;
 tf_outp: TArray<TF_Output>;
begin
 if not Assigned(Handle) then
   TFDisposable.ObjectDisposedException();
 n := Length(inputs);
 if Assigned(inputs) and (n > 0) then begin
   SetLength(tf_outp,n);
   for i := 0 to n-1 do
     tf_outp[i] := inputs[i].ToTF_Output();
   TF_AddInputList(self.Handle, PTF_Output(@tf_outp[0]), Length(inputs));
 end;
 Result := self;
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
   Result := TFOperation.Create(m_oGraph,l_pOp);
 self.DisposeOf;
end;

function TFOperation.OutputType(index: Integer):TF_DataType;
begin
    Result := TF_DataType (TF_OperationOutputType(TFOutput.Create(Self,index).ToTF_Output ) )
end;
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
   m_oGraph := g;
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
   m_id_value := m_oGraph.NextId;

   // This will be set by self.inputs.
   if op_def = nil then
   begin
     var op := g.GetOpDef(node_def.Op);
     op_def := @op;
   end;

   var t := TOps._create_c_op(g, node_def, inputs, control_input_ops.ToArray, op_def);
   var _handle := t.Value1;
   var pDesc   := t.Value2;
   m_is_stateful := op_def.IsStateful;

   Handle := _handle;

   // Initialize self._outputs.
   SetLength(output_types,NumOutputs) ;
   for var i := 0 to NumOutputs - 1 do
      output_types[i] := OutputType(i);
   SetLength(FOutputs,NumOutputs);
   for var i := 0 to NumOutputs - 1 do
      FOutputs[i] := TFTensor.Create(self, i, output_types[i]);

   //m_oGraph._add_op(Self);
   // if _handle <>nil
   //    _control_flow_post_processing();
   //self.MMLck();

end;

constructor TFOperation.Create(graph: TFGraph; hnd: Pointer);
begin
     inherited Create;

     if not Assigned(hnd) then
         Exit;

     Handle := hnd;
     if Assigned(graph)  then
        m_oGraph := graph
     else
        m_oGraph := TOps.get_default_graph;

     SetLength(FOutputs,NumOutputs);
     for var i := 0 to NumOutputs - 1 do
      FOutputs[i] := TFTensor.Create(self, i, OutputType(i));

     self.MMLck();

      // Dict mapping op name to file and line information for op colocation
      // context managers.
      // _control_flow_context = _graph._get_control_flow_context();
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

function TFOperation.Getname: string;
begin
    m_name := '';
    if Assigned(Handle) then
      m_name := string( AnsiString(TF_OperationName(Handle)) ) ;

    Result := m_name;
end;

function TFOperation.MMLck(): TFOperation;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 if Assigned(env) then
   env.AddOperation(self);
 Result := self;
end;
procedure TFOperation.NativeDispose(hnd: Pointer);
begin
 // No Delete!
end;
//------------------------------------------------------------------------------
//----------------------------- TFShape ----------------------------------------
//------------------------------------------------------------------------------
(*class operator TFShape.Implicit(a: TFShape): PTFShape;
begin
    Result := PTFShape(Intptr(@a));
end; *)

class operator TFShape.Implicit(a: PTFShape): TFShape;
begin
    if Assigned(a) then
       Result := a^;
end;

class operator TFShape.Initialize(out Dest: TFShape);
begin
    //Dest := Default(TFShape);
    //Dest.FHandle := nil;
    //Dest.FaDims := Nil;
end;

class operator TFShape.Finalize(var Dest: TFShape);
begin
    //Dest := Default(TFShape);
    //SetLength(Dest.FaDims,0);
    //Dest.FHandle := nil;
    //Dest.FaDims  := nil;
end;

{ TFShapeHelper }

class operator TFShapeHelper.Implicit(a: TFShape): TFTensor;
begin
   var v : TValue := TValue.From<TFShape>(a);

    Result := TConstant_op.constant(@v)
end;

class operator TFShapeHelper.Implicit(a: TFShape): TArray<Integer>;
var
  i : Integer;
begin
    Result := [];
    for i := 0 to Length(a.FaDims) - 1 do
     Result := Result + [ a.FaDims[i] ] ;
end;

function TFShapeHelper.IsNil: Boolean;
begin
    Result := Self.FHandle = nil;
end;

function TFShapeHelper.IsNull: Boolean;
begin
    Result := FaDims = nil;
end;

constructor TFShapeHelper.Create(dims: TArray<TF_int64_t>);
begin
    Self := Default(TFShape);

    Self.FSize   := 0;
    Self.Fndim   := 0;
    Self.FHandle := Pointer($10000000);
    Self.FaDims := dims;
end;

class function TFShapeHelper.Scalar: TFShape;
begin
    var v : TFShape;

    v.FaDims := nil;
    v.FSize   := 0;
    v.Fndim   := 0;
    v.FHandle := Pointer($10000000);
    Result := v ;

end;

function TFShapeHelper.Equals(target: TValue): Boolean;
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

function TFShapeHelper.GetnDim: Integer;
begin
    //if FaDims = nil then Exit(-1);

    Result := Length(FaDims);
end;

function TFShapeHelper.GetSize: Int64;
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

function TFShapeHelper.IsScalar: Boolean;
begin
     Result := ndim = 0;
end;

function TFShapeHelper.is_compatible_with(shape2: TFShape): Boolean;
begin
    if (TArray.IndexOf<Int64>(dims,-1)> -1) or (TArray.IndexOf<Int64>(shape2.dims,-1) > -1) then
        Exit(true);
    if size <> shape2.size then
        Exit(false);
    Result := true;
end;

function TFShapeHelper.ToString: string;
begin
    case Fndim of
      -1 : Result := '<unknown>';
      0  : Result := '()';
      1  : Result := '(' + IntToStr(FaDims[0]).Replace('-1','None') +')';
    else
      var sStrings : TArray<String>;
      for var i := 0 to Length(FaDims) - 1 do
           sStrings := sStrings + [ IntToStr(FaDims[i]).Replace('-1','None') ];
      Result.Join(',', sStrings);
    end;
end;

//------------------------------------------------------------------------------
//----------------------------- TFGraph ----------------------------------------
//------------------------------------------------------------------------------
procedure TFGraph.add_to_collection<T>(name: string; value: T);
begin
    _check_not_finalized ;
    if m_collections.ContainsKey(name) then
      (m_collections[name] as TList<T>).Add(value)
    else  begin
       var list := TList<T>.Create;
       list.Add(value);
       m_collections.Add(name, list) ;
    end;
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

function TFGraph.get_collection(name, scope: string): TObject;
begin
    if m_collections.ContainsKey(name) then
       Result := m_collections[name]
    else
       Result := nil;
end;

function TFGraph.get_collection<T>(name, scope: string): TList<T>;
begin
     var collection : TList<T>;
     if m_collections.ContainsKey(name) then
         collection := m_collections[name] as TList<T>
     else
         collection := TList<T>.Create;

     Result :=  collection;
end;

function TFGraph.get_collection_ref<T>(name: string): TList<T>;
begin
     if m_collections.ContainsKey(name) then
         Result := m_collections[name] as TList<T>
     else
         Result := TList<T>.Create;
end;

procedure TFGraph.gExit;
begin
    tf.Context.restore_mode;
    TOps.pop_graph;
end;

constructor TFGraph.Create;
begin
 inherited Create(TF_NewGraph());
 _name_stack := '';
 m_sCurrentNameScope := '';
 m_oValues := TDictionary<String, Integer>.Create;
 m_oPendingInitVariables := Nil;
 m_oScopes := Nil;

 m_finalized   := False;
 m_collections:= TDictionary<string, TObject>.Create;
end;
destructor  TFGraph.Destroy;
var
 i: Integer;
begin
   m_oValues.Clear;
   FreeAndNil(m_oValues);
   if Assigned(m_oPendingInitVariables) then begin
     m_oPendingInitVariables.Clear;
     FreeAndNil(m_oPendingInitVariables);
   end;
   if Assigned(m_oScopes) then begin
     for i := 0 to m_oScopes.Count-1 do
       m_oScopes[i].Free;
     m_oScopes.Clear;
     FreeAndNil(m_oScopes);
   end;
   m_collections.Free;

   inherited Destroy;
end;
procedure TFGraph.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteGraph(hnd);
end;

function TFGraph.NewOperation(opType, opName: string): TFOperationDesc;
begin
    Result := TFOperationDesc.Create(self, TFString(opType), TFString(opName));
end;

function TFGraph.NextId: Integer;
begin
   Inc(m__next_id_counter);

   Result := m__next_id_counter;
end;

function TFGraph.get_name_scope: TFString;
begin
    Result:= _name_stack;
end;

function TFGraph.name_scope(name: TFString): TFString;
begin
    var new_stack : TFString := '';

    if string.IsNullOrEmpty(string(name)) then
        new_stack := ''
    else if string(name).EndsWith('/') then
        new_stack := TOps.name_from_scope_name(string(name))
    else
        new_stack := MakeUnique(name);
    _name_stack := new_stack;
    if String.IsNullOrEmpty(string(new_stack)) then Result := ''
    else                                            Result := new_stack + '/';
end;

function TFGraph.MakeUnique(const name: TFString): TFString;
var
 val: Integer;
 l_sName: String;
begin
 val := 0;
 l_sName := String(name);
 if not m_oValues.TryGetValue(l_sName, val) then begin
   val := 0;
   m_oValues.Add(l_sName,val);
 end
 else begin
   Inc(val);
   m_oValues[l_sName] := val;
 end;
 Result := name + TFString(IntToStr(val));
end;

function TFGraph.WithScope(nameScopeDesc: TFString): TFScope;
var
 scope: TFScope;
begin
 scope := TFScope.Create(self);
 if not Assigned(m_oScopes) then
   m_oScopes := TList<TFScope>.Create;
 m_oScopes.Add(scope);
 if Length(CurrentNameScope) = 0 then
   CurrentNameScope := nameScopeDesc
 else
   CurrentNameScope := CurrentNameScope + '/' + nameScopeDesc;
 Result := scope;
end;
procedure TFGraph._check_not_finalized;
begin
    if m_finalized then
       raise TFException.Create('Graph is finalized and cannot be modified.');
end;

function TFGraph.MakeName(const operName, userName: TFString; var nameBuf: TFString): PTFChar;
var
 k: TFString;
begin
 if Length(userName) = 0 then begin
   if Length(CurrentNameScope) = 0 then
     k := operName
   else
     k := CurrentNameScope + '/' + operName;
   nameBuf := MakeUnique(k);
 end
 else begin
   if Length(CurrentNameScope) = 0 then
     nameBuf := userName
   else
     nameBuf := CurrentNameScope + '/' + userName;
 end;
 Result := _PTFChar(nameBuf);
end;
function TFGraph.GetOpByName(const Name: TFString): TFOperation;
var
 l_pOp: PTF_Operation;
begin
 if not Assigned(Handle) then
   ObjectDisposedException();
 l_pOp := TF_GraphOperationByName(Handle, _PTFChar(Name));
 if Assigned(l_pOp) then
   Result := TFOperation.Create(self, l_pOp)
 else
   Result := Nil;
end;
function TFGraph.GetOpDef(tipo: string): TOpDef;
begin
    Result := op_def_registry.GetOpDef(tipo)
end;

function TFGraph.GetShape(output: TFOutput; var dims: TArray<TF_int64_t>; status: TFStatus = Nil): PTF_int64_t;
var
 l_lOk: Boolean;
 ndims: Integer;
 l_oStatus: TFStatus;
 l_oOutp: TF_Output;
begin
 Result := Nil;
 SetLength(dims,0);
 if not Assigned(Handle) then
   ObjectDisposedException();
 l_oStatus := TFStatus.Setup(status);
 l_oOutp := output.ToTF_Output();
 ndims := TF_GraphGetTensorNumDims(Handle, l_oOutp, l_oStatus.Handle);
 l_lOk := l_oStatus.CheckMaybeRaise(status, False);
 if l_lOk and (ndims > 0) then begin
   SetLength(dims,ndims);
   Result := @(dims[0]);
   TF_GraphGetTensorShape(Handle, l_oOutp, Result, ndims, l_oStatus.Handle);
   l_oStatus.CheckMaybeRaise(status);
 end
 else
   l_oStatus.Free;
end;

procedure TFGraph.AddInitVariable(variable: TFOperation);
begin
 if not Assigned(m_oPendingInitVariables) then
  	m_oPendingInitVariables := TList<TFOperation>.Create;
 m_oPendingInitVariables.Add(variable);
end;
function TFGraph.GetGlobalVariablesInitializer(var variables: TArray<TFOperation>): Boolean;
var
 i, n: Integer;
begin
 if Assigned(m_oPendingInitVariables) then begin
   n := m_oPendingInitVariables.Count;
   SetLength(variables,n);
   for i := 0 to n-1 do
     variables[i] := m_oPendingInitVariables[i];
   Result := True;
 end
 else begin
   SetLength(variables,0);
   Result := False;
 end;
end;
//------------------------------------------------------------------------------
//----------------------------- TFSessionRunner --------------------------------
//------------------------------------------------------------------------------
constructor TFSessionRunner.Create(session: TFSession);
begin
 inherited Create;
 m_oSession := session;
 m_aInputs      := TList<TFOutput>.Create;
 m_aInputValues := TList<TFTensor>.Create;
 m_aOutputs     := TList<TFOutput>.Create;
 m_aOutputValues:= TList<TFTensor>.Create;
 m_aTargets     := TList<TFOperation>.Create;
 m_oRunMetadata := Nil;
 m_oRunOptions  := Nil;
end;
destructor  TFSessionRunner.Destroy;
begin
 DeleteLists();
 inherited Destroy;
end;
procedure TFSessionRunner.ResetOutputs();
var
 l_enOutp: TList<TFOutput>.TEnumerator;
 l_oOutp:  TFOutput;
begin
 l_enOutp := m_aOutputs.GetEnumerator;
 while(l_enOutp.MoveNext) do begin
   l_oOutp := l_enOutp.Current;
   l_oOutp.Free;
 end;
 m_aOutputs.Clear;
 l_enOutp.Free;
end;
procedure TFSessionRunner.ResetOutputValues();
var
 l_enTens: TList<TFTensor>.TEnumerator;
 l_oTens:  TFTensor;
begin
 l_enTens := m_aOutputValues.GetEnumerator;
 while(l_enTens.MoveNext) do begin
   l_oTens := l_enTens.Current;
   l_oTens.Free;
 end;
 m_aOutputValues.Clear;
 l_enTens.Free;
end;
procedure TFSessionRunner.Reset;
begin
 m_aInputs.Clear;
 m_aInputValues.Clear;
 m_aOutputs.Clear;
 m_aOutputValues.Clear;
 m_aTargets.Clear;
end;
procedure TFSessionRunner.DeleteLists();
begin
 self.Reset();
 FreeAndNil(m_aInputs);
 FreeAndNil(m_aInputValues);
 FreeAndNil(m_aOutputs);
 FreeAndNil(m_aOutputValues);
 FreeAndNil(m_aTargets);
end;
function TFSessionRunner.ParseOutput(operationName: TFString): TFOutput;
var
 n, l_iIdx: Integer;
 l_sOp, l_sIdx: TFString;
 l_oOp: TFOperation;
begin
 // Parses user strings that contain both the operation name and an index.
 l_iIdx := 0;
 n := Pos(':',string(operationName));
 if n > 0 then begin
   l_sOp := Copy(operationName,1,n-1);
   l_sIdx:= Copy(operationName,n+1,999);
   try
     l_iIdx := StrToInt(String(l_sIdx));
   except
     l_iIdx := 0;
   end;
 end
 else
   l_sOp := operationName;
 l_oOp  := m_oSession.Graph.GetOpByName(l_sOp);
 if Assigned(l_oOp) then
   Result := TFOutput.Create(l_oOp,l_iIdx)
 else
   Result := Nil;
end;
function TFSessionRunner.AddInput(input: TFOutput; value: TFTensor): TFSessionRunner;
begin
 if not Assigned(input) then
   raise TFException.Create('TFSessionRunner.AddInput: Argument Null Exception - input');
 m_aInputs.Add(input);
 m_aInputValues.Add(value);
 Result := self;
end;
function TFSessionRunner.AddInput(inputName: TFString; value: TFTensor): TFSessionRunner;
var
 l_oOutp: TFOutput;
begin
 if Length(inputName) = 0 then
   raise TFException.Create('TFSessionRunner.AddInput: Argument Null Exception - inputName');
 l_oOutp := ParseOutput(inputName);
 if Assigned(l_oOutp) then begin
   m_aInputs.Add(l_oOutp);
   m_aInputValues.Add(value);
 end
 else
   raise TFException.Create('TFSessionRunner.AddInput: Unknown inputName - ' + inputName);
 Result := self;
end;
function TFSessionRunner.AddTarget(target: TFOperation): TFSessionRunner;
begin
 if not Assigned(target) then
   raise TFException.Create('TFSessionRunner.AddTarget: Argument Null Exception - target');
 m_aTargets.Add(target);
 Result := self;
end;
function TFSessionRunner.AddTarget(targets: TArray<TFOperation>): TFSessionRunner;
var
 i: Integer;
begin
 if Length(targets) = 0 then
   raise TFException.Create('TFSessionRunner.AddTarget: Argument Null Exception - targets');
 for i := 0 to Length(targets)-1 do
   m_aTargets.Add(targets[i]);
 Result := self;
end;
function TFSessionRunner.AddTarget(targetName: TFString): TFSessionRunner;
var
 l_oOp: TFOperation;
begin
 if Length(targetName) = 0 then
   raise TFException.Create('TFSessionRunner.AddTarget: Argument Null Exception - targetName');
 l_oOp := m_oSession.Graph.GetOpByName(targetName);
 if Assigned(l_oOp) then
   m_aTargets.Add(l_oOp)
 else
   raise TFException.Create('TFSessionRunner.AddTarget: Unknown targetName - ' + targetName);
 Result := self;
end;
function TFSessionRunner.AddTarget(targetNames: TArray<TFString>): TFSessionRunner;
var
 i: Integer;
begin
 if Length(targetNames) = 0 then
   raise TFException.Create('TFSessionRunner.AddTarget: Argument Null Exception - targetNames');
 for i := 0 to Length(targetNames)-1 do
   self.AddTarget(targetNames[i]);
 Result := self;
end;
function TFSessionRunner.Fetch(operationName: TFString; idx: Integer): TFSessionRunner;
var
 l_oOp: TFOperation;
begin
 l_oOp := m_oSession.Graph.GetOpByName(operationName);
 if Assigned(l_oOp) then begin
   m_aOutputs.Add(TFOutput.Create(l_oOp,idx))
 end
 else
   raise TFException.Create('TFSessionRunner.Fetch: Unknown operationName - ' + operationName);
 Result := self;
end;
function TFSessionRunner.Fetch(operationName: TFString): TFSessionRunner;
var
 l_oOp: TFOperation;
begin
 l_oOp := m_oSession.Graph.GetOpByName(operationName);
 if Assigned(l_oOp) then begin
   m_aOutputs.Add(TFOutput.Create(l_oOp))
 end
 else
   raise TFException.Create('TFSessionRunner.Fetch: Unknown operationName - ' + operationName);
 Result := self;
end;
function TFSessionRunner.Fetch(output: TFOutput): TFSessionRunner;
begin
 m_aOutputs.Add(output);
 Result := self;
end;
function TFSessionRunner.Fetch(outputs: TArray<TFOutput>): TFSessionRunner;
var
 i: Integer;
begin
 for i := 0 to Length(outputs)-1 do
   m_aOutputs.Add(outputs[i]);
 Result := self;
end;
function TFSessionRunner.Fetch(outputNames: TArray<TFString>): TFSessionRunner;
var
 i: Integer;
begin
 if Length(outputNames) = 0 then
   raise TFException.Create('TFSessionRunner.Fetch: Argument Null Exception - outputNames');
 for i := 0 to Length(outputNames)-1 do
   self.Fetch(outputNames[i]);
 Result := self;
end;
class function TFSessionRunner.ToArray_TF_Output(outp: TFOutput; var outputBuf: TArray<TF_Output>): PTF_Output;
begin
 if Assigned(outp) then begin
   SetLength(outputBuf,1);
   outputBuf[0].oper  := outp.Oper.Handle;
   outputBuf[0].index := outp.Idx;
   Result := @(outputBuf[0]);
 end
 else begin
   SetLength(outputBuf,0);
   Result := Nil
 end;
end;
class function TFSessionRunner.ToArray_TF_Output(outp: TList<TFOutput>; var outputBuf: TArray<TF_Output>): PTF_Output;
var
 i, n: Integer;
begin
 if Assigned(outp) then begin
   n := outp.Count;
   SetLength(outputBuf,n);
   for i := 0 to n-1 do begin
     outputBuf[i].oper  := outp[i].Oper.Handle;
     outputBuf[i].index := outp[i].Idx;
   end;
   Result := @(outputBuf[0]);
 end
 else begin
   SetLength(outputBuf,0);
   Result := Nil
 end;
end;
class function TFSessionRunner.ToArray_TF_Tensor(tensors: TList<TFTensor>; var tensorBuf: TArray<PTF_Tensor>): PTF_Tensor;
var
 i, n: Integer;
begin
 if Assigned(tensors) then begin
   n := tensors.Count;
   SetLength(tensorBuf,n);
   for i := 0 to n-1 do
     tensorBuf[i] := tensors[i].Handle;
   Result := @(tensorBuf[0]);
 end
 else begin
   SetLength(tensorBuf,0);
   Result := Nil
 end;
end;
class function TFSessionRunner.ToArray_TF_Operation(operations: TList<TFOperation>; var opsBuf: TArray<PTF_Operation>): PTF_Operation;
var
 i, n: Integer;
begin
 if Assigned(operations) then begin
   n := operations.Count;
   SetLength(opsBuf,n);
   for i := 0 to n-1 do
     opsBuf[i] := operations[i].Handle;
   Result := @(opsBuf[0]);
 end
 else begin
   SetLength(opsBuf,0);
   Result := Nil
 end;
end;
function TFSessionRunner.Run(status: TFStatus = Nil): TFTensor;
var
 i, l_iInputsCnt, l_iOutputsCnt, l_iTargetsCnt: Integer;
 l_aInputsBuf, l_aOutputsBuf: TArray<TF_Output>;
 l_aInputValBuf, l_aOutputValBuf: TArray<PTF_Tensor>;
 l_aTargetsBuf: TArray<PTF_Operation>;
 l_pInputs, l_pOutputs: PTF_Output;
 l_pInputValues, l_pOutputValues: PTF_Tensor;
 l_pTargets: PTF_Operation;
 l_pRunMetadata, l_pRunOptions: PTF_Buffer;
 l_oStatus: TFStatus;
begin
 if m_aInputs.Count <> m_aInputValues.Count then
   raise TFException.Create('TFSessionRunner.Run: inputs and inputValues have different count');
 l_pInputs      := TFSessionRunner.ToArray_TF_Output(m_aInputs, l_aInputsBuf);
 l_pInputValues := TFSessionRunner.ToArray_TF_Tensor(m_aInputValues, l_aInputValBuf);
 l_pOutputs     := TFSessionRunner.ToArray_TF_Output(m_aOutputs, l_aOutputsBuf);
 l_pTargets     := TFSessionRunner.ToArray_TF_Operation(m_aTargets, l_aTargetsBuf);
 l_iInputsCnt   := Length(l_aInputsBuf);
 l_iOutputsCnt  := Length(l_aOutputsBuf);
 l_iTargetsCnt  := Length(l_aTargetsBuf);
 ResetOutputValues();
 SetLength(l_aOutputValBuf,l_iOutputsCnt);
 for i := 0 to l_iOutputsCnt-1 do
   l_aOutputValBuf[i] := Nil;
 l_pOutputValues := @(l_aOutputValBuf[0]);
 if Assigned(m_oRunOptions) then
   l_pRunOptions := m_oRunOptions.Handle
 else
   l_pRunOptions := Nil;
 if Assigned(m_oRunMetadata) then
   l_pRunMetadata := m_oRunMetadata.Handle
 else
   l_pRunMetadata := Nil;
 l_oStatus := TFStatus.Setup(status);
 TF_SessionRun(m_oSession.Handle, l_pRunOptions, l_pInputs, l_pInputValues,
               l_iInputsCnt, l_pOutputs, l_pOutputValues,
               l_iOutputsCnt, l_pTargets, l_iTargetsCnt, l_pRunMetadata, l_oStatus.Handle);
 l_oStatus.CheckMaybeRaise(status);
 for i := 0 to l_iOutputsCnt-1 do begin
   m_aOutputValues.Add(TFTensor.Create(l_aOutputValBuf[i]));
 end;
 if l_iOutputsCnt > 0 then
   Result := m_aOutputValues[0]
 else
   Result := Nil;
end;
function TFSessionRunner.Run(operation: TFOutput; status: TFStatus = Nil): TFTensor;
begin
 self.ResetOutputs();
 self.Fetch(operation);
 Result := self.Run(status);
end;
//------------------------------------------------------------------------------
//----------------------------- TFSession --------------------------------------
//------------------------------------------------------------------------------
constructor TFSession.Create(status: TFStatus = Nil);
var
 l_oGraph: TFGraph;
begin
 inherited Create(Nil);
 l_oGraph := TFGraph.Create;
 m_oSessionRunner := Nil;
 self.Init(l_oGraph,status);
end;
constructor TFSession.Create(hnd: Pointer; graph: TFGraph);
var
 l_oEnv: TFMMEnv;
begin
 inherited Create(hnd);
 m_oGraph := graph;
 l_oEnv := TFMMEnv.GetMMEnv();
 l_oEnv.SessionActive := True;
 m_oSessionRunner := Nil;
end;
constructor TFSession.Create(graph: TFGraph; sessionOptions: TFSessionOptions; status: TFStatus = Nil);
var
 l_oEnv: TFMMEnv;
 l_oStatus: TFStatus;
 l_pSession: PTF_Session;
begin
 inherited Create(Nil);
 m_oGraph  := graph;
 l_oStatus := TFStatus.Setup(status);
 l_pSession:= TF_NewSession(graph.handle, sessionOptions.handle, l_oStatus.handle);
 l_oStatus.CheckMaybeRaise(status);
 Handle := l_pSession;
 l_oEnv := TFMMEnv.GetMMEnv();
 l_oEnv.SessionActive := True;
 m_oSessionRunner := Nil;
end;
constructor TFSession.Create(graph: TFGraph; status: TFStatus = Nil);
begin
 inherited Create(Nil);
 self.Init(graph,status);
 m_oSessionRunner := Nil;
end;
procedure TFSession.Init(graph: TFGraph; status: TFStatus = Nil);
var
 l_oEnv: TFMMEnv;
 l_oStatus: TFStatus;
 l_pSession: PTF_Session;
 l_pOpts: PTF_SessionOptions;
begin
 m_oGraph  := graph;
 l_oStatus := TFStatus.Setup(status);
 l_pOpts   := TF_NewSessionOptions();
 l_pSession:= TF_NewSession(graph.handle, l_pOpts, l_oStatus.handle);
 TF_DeleteSessionOptions(l_pOpts);
 l_oStatus.CheckMaybeRaise(status);
 l_oEnv := TFMMEnv.GetMMEnv();
 l_oEnv.SessionActive := True;
 Handle := l_pSession;
end;
destructor  TFSession.Destroy;
var
 l_oEnv: TFMMEnv;
 l_oStatus: TFStatus;
begin
 if Assigned(m_oSessionRunner) then
   FreeAndNil(m_oSessionRunner);
 if Assigned(Handle) then begin
   l_oStatus := TFStatus.Create;
   TF_CloseSession (Handle, l_oStatus.Handle);
   TF_DeleteSession(Handle, l_oStatus.Handle);
   Handle := Nil;           // <- Don't execute NativeDispose
   FreeAndNil(l_oStatus);
   m_oGraph.Handle := Nil;  // <- Already with session deleted.
   FreeAndNil(m_oGraph);
 end;
 l_oEnv := TFMMEnv.GetMMEnv();
 l_oEnv.SessionActive := False;
 l_oEnv.ResetSessionLists;
 inherited Destroy;
end;
procedure TFSession.NativeDispose(hnd: Pointer);
var
 l_pStatus : PTF_Status;
begin
 if Assigned(hnd) then begin
   l_pStatus := TF_NewStatus();
   TF_DeleteSession(hnd,l_pStatus);
   TF_DeleteStatus(l_pStatus);
 end;
end;
function TFSession.GetRunner(): TFSessionRunner;
begin
 if Assigned(m_oSessionRunner) then
   m_oSessionRunner.Reset()
 else
   m_oSessionRunner := TFSessionRunner.Create(self);
 Result := m_oSessionRunner;
end;
function TFSession.Run(inputs: TArray<TF_Output>; inputValues: TArray<PTF_Tensor>;
                outputs: TArray<TF_Output>; var outputValues: TArray<PTF_Tensor>;
                targetOpers: TArray<PTF_Operation> = Nil;
                runMetadata: PTF_Buffer = Nil; runOptions: PTF_Buffer = Nil;
                status: PTF_Status = Nil): Boolean;
var
 l_iInputsCnt, l_iOutputsCnt: Integer;

begin
 if not Assigned(Handle) then
   ObjectDisposedException ();
 if not Assigned(inputs) then
   raise TFException.Create('TFSession.Run: Argument Null Exception - inputs');
 if not Assigned(inputValues) then
   raise TFException.Create('TFSession.Run: Argument Null Exception - inputValues');
 if not Assigned(outputs) then
   raise TFException.Create('TFSession.Run: Argument Null Exception - outputs');
 l_iInputsCnt := Length(inputs);
 if l_iInputsCnt <> Length(inputValues) then
   raise TFException.Create('TFSession.Run: inputs and inputValues have different lengths');
 l_iOutputsCnt := Length(outputs);
 //TODO:
end;

//------------------------------------------------------------------------------
//----------------------------- TFMMEnv ----------------------------------------
//------------------------------------------------------------------------------
constructor TFMMEnv.Create;
begin
 inherited Create;
 m_lSessionActive := False;
 m_lIsActive      := False;
 m_aTensorListOutside := TList<TFTensor>.Create;
 m_aTensorList := TList<TFTensor>.Create;
 m_aOutputList := TList<TFOutput>.Create;
 m_aOperationList := TList<TFOperation>.Create;
end;
destructor  TFMMEnv.Destroy;
begin
 self.ResetAllLists;
 m_aTensorListOutside.Free;
 m_aTensorList.Free;
 m_aOutputList.Free;
 m_aOperationList.Free;
 inherited Destroy;
end;
class function TFMMEnv.GetMMEnv(): TFMMEnv;
begin
 if not Assigned(TFMMEnv.g_oMMEnv) then
   TFMMEnv.g_oMMEnv := TFMMEnv.Create;
 Result := TFMMEnv.g_oMMEnv;
end;
class procedure TFMMEnv.StartMM;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 env.SessionActive := False;
 env.IsActive      := True;
 env.ResetAllLists;
end;
class procedure TFMMEnv.EndMM;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 env.SessionActive := False;
 env.IsActive      := False;
 env.ResetAllLists;
end;
procedure TFMMEnv.AddTensor(tensor: TFTensor);
begin
   if self.SessionActive then
   begin
       for var item in  m_aTensorListOutside do
         if item.Handle = tensor.Handle then Exit;
       m_aTensorList.Add(tensor)
   end
   else begin
       for var item in  m_aTensorListOutside do
         if item.Handle = tensor.Handle then Exit;
       m_aTensorListOutside.Add(tensor);
   end;
end;
procedure TFMMEnv.AddOutput(output: TFOutput);
begin
 m_aOutputList.Add(output);
end;
procedure TFMMEnv.AddOperation(oper: TFOperation);
begin
 m_aOperationList.Add(oper);
end;
procedure TFMMEnv.ResetSessionLists;
var
 i: Integer;
 l_oTensor: TFTensor;
 l_oOutput: TFOutput;
 l_oOper:   TFOperation;
begin
 if m_aTensorList.Count > 0 then begin
   for i := 0 to m_aTensorList.Count-1 do begin
     l_oTensor := m_aTensorList[i];
     if Assigned(l_oTensor) then begin
       if not l_oTensor.DeallocatorCalled then
         l_oTensor.Free;
       m_aTensorList[i] := Nil;
     end;
   end;
   m_aTensorList.Clear;
 end;
 if m_aOutputList.Count > 0 then begin
   for i := 0 to m_aOutputList.Count-1 do begin
     l_oOutput := m_aOutputList[i];
     if Assigned(l_oOutput) then begin
       l_oOutput.Free;
     end;
   end;
   m_aOutputList.Clear;
 end;
 if m_aOperationList.Count > 0 then begin
   for i := 0 to m_aOperationList.Count-1 do begin
     l_oOper := m_aOperationList[i];
     if Assigned(l_oOper) then begin
       l_oOper.Free;
     end;
   end;
   m_aOperationList.Clear;
 end;
end;
function as_base_dtype(tipo: TF_DataType): TF_DataType;
begin
     if Ord(tipo) > 100 then
       Result := TF_DataType(Ord(tipo) - 100)
     else
       Result := tipo;
end;
function get_datatype_size(tipo: TF_DataType): Integer;
begin
    case as_base_dtype(tipo) of
      TF_DataType.TF_BOOL   : Result :=  sizeof(Boolean);
      TF_DataType.TF_UINT8  : Result :=  sizeof(byte);
      TF_DataType.TF_INT8   : Result :=  sizeof(Int8);
      TF_DataType.TF_UINT16 : Result :=  sizeof(Word);
      TF_DataType.TF_INT16  : Result :=  sizeof(Int16);
      TF_DataType.TF_UINT32 : Result :=  sizeof(UInt32);
      TF_DataType.TF_INT32  : Result :=  sizeof(Int32);
      TF_DataType.TF_UINT64 : Result :=  sizeof(UInt64);
      TF_DataType.TF_INT64  : Result :=  sizeof(Int64);
      TF_DataType.TF_FLOAT  : Result :=  sizeof(Single);
      TF_DataType.TF_DOUBLE : Result :=  sizeof(Double);
    else
       raise Exception.Create('Not Implemented Tipo');
    end;
end;
procedure TFMMEnv.ResetAllLists;
var
 i: Integer;
 l_oTensor: TFTensor;
begin
 self.ResetSessionLists();
 if m_aTensorListOutside.Count > 0 then begin
   for i := 0 to m_aTensorListOutside.Count-1 do begin
     l_oTensor := m_aTensorListOutside[i];
     if Assigned(l_oTensor) then begin
       if not l_oTensor.DeallocatorCalled then
         l_oTensor.Free;
       m_aTensorListOutside[i] := Nil;
     end;
   end;
   m_aTensorListOutside.Clear;
 end;
end;

{ TFTensors }

constructor TFTensors.Create(tensors: TArray<TFTensor>);
begin
   FItems := TList<TFTensor>.Create;

   FItems.AddRange(tensors);

   Fdtype := FItems.First.TensorDataType;
   Fshape := FItems.First.Shape;
   FRank  := FItems.First.rank;
   Fgraph := FItems.First.graph;
   FIsCreatedInGraphMode := FItems.First.isCreatedInGraphMode;
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

function TFTensors.ToArray(tensors: TFTensors): TArray<TFTensor>;
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

procedure TFTensors.Add(tensor: TFTensor);
begin
    FItems.Add(tensor);

    Fdtype := FItems.First.TensorDataType;
    Fshape := FItems.First.Shape;
    FRank  := FItems.First.rank;
    Fgraph := FItems.First.graph;
    FIsCreatedInGraphMode := FItems.First.isCreatedInGraphMode;
    FiLength := FItems.Count;
end;

procedure TFTensors.AddRange(tensors: TArray<TFTensor>);
begin
    FItems.AddRange(tensors);

    Fdtype := FItems.First.TensorDataType;
    Fshape := FItems.First.Shape;
    FRank  := FItems.First.rank;
    Fgraph := FItems.First.graph;
    FIsCreatedInGraphMode := FItems.First.isCreatedInGraphMode;
    FiLength := FItems.Count;
end;

procedure TFTensors.Insert(idx: Integer; tensor: TFTensor);
begin
    FItems.Insert(idx,tensor);
end;

function TFTensors.GetEnumerator: IEnumerator<TFTensor>;
begin
    inherited GetEnumerator;
end;

Initialization
begin
 TFMMEnv.g_oMMEnv := Nil;
end;
finalization
begin
 if Assigned(TFMMEnv.g_oMMEnv) then
   FreeAndNil(TFMMEnv.g_oMMEnv);
end;
end.
