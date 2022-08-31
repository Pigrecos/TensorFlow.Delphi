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
unit TensorFlow.LowLevelAPI;

interface

uses System.Types, Winapi.Windows, System.AnsiStrings;

const
  TF_TSRING_SIZE : Integer = 24;

type
 // --------------------------------------------------------------------------
 // Basistypen
 // --------------------------------------------------------------------------
 TF_size_t   = NativeUInt;
 PTF_size_t  = ^TF_size_t;
 TF_int64_t  = Int64;
 PTF_int64_t  = ^TF_int64_t;
 PPTF_int64_t = ^PTF_int64_t;
 TFCode = UINT8;
 PTF_UINT8 = ^UINT8;
 PTF_Boolean = ^Boolean;
 PInt8  = ^Int8;
 PInt32 = ^Int32;
 PInt16 = ^Int16;
 TFChar = AnsiChar;
 PTFChar= PAnsiChar;
 TFString = AnsiString;
 PTFString= PAnsiString;

 // -----------------------------------------------------------------------------

 TF_deallocator      = procedure (data: Pointer; len: TF_size_t; arg: Pointer); cdecl;
 TF_data_deallocator = procedure (data: Pointer; length: TF_size_t);  cdecl;
 TFEX_ProtCallback   = procedure (prot_buf: PTFChar; prot_buf_len: TF_size_t); cdecl;

 /// <summary>Represents the type of the elements in a Tensor.</summary>
 TF_DataType = (
  TF_DATATYPE_UNKNOWN = 0,
  TF_FLOAT = 1,
  TF_DOUBLE = 2,
  /// <summary>Int32 tensors are always in 'host' memory.</summary>
  TF_INT32 = 3,
  TF_UINT8 = 4,
  TF_INT16 = 5,
  TF_INT8 = 6,
  TF_STRING = 7,
  /// <summary>Single-precision complex</summary>
  TF_COMPLEX64 = 8,
  /// <summary>Old identifier kept for API backwards compatibility</summary>
  TF_COMPLEX = 8,
  TF_INT64 = 9,
  TF_BOOL = 10,
  /// <summary>Quantized int8</summary>
  TF_QINT8 = 11,
  /// <summary>Quantized uint8</summary>
  TF_QUINT8 = 12,
  /// <summary>Quantized int32</summary>
  TF_QINT32 = 13,
  /// <summary>Float32 truncated to 16 bits.  Only for cast ops.</summary>
  TF_BFLOAT16 = 14,
  /// <summary>Quantized int16</summary>
  TF_QINT16 = 15,
  /// <summary>Quantized uint16</summary>
  TF_QUINT16 = 16,
  TF_UINT16 = 17,
  /// <summary>Double-precision complex</summary>
  TF_COMPLEX128 = 18,
  TF_HALF = 19,
  TF_RESOURCE = 20,
  TF_VARIANT = 21,
  TF_UINT32 = 22,
  TF_UINT64 = 23,

  TF_FLOAT_REF = 101,
  TF_DOUBLE_REF = 102,
  TF_INT32_REF = 103,
  TF_UINT8_REF = 104,
  TF_INT16_REF = 105,
  TF_INT8_REF = 106,
  TF_STRING_REF = 107,
  TF_COMPLEX64_REF = 108,
  TF_INT64_REF = 109,
  TF_BOOL_REF = 110,
  TF_QINT8_REF = 111,
  TF_QUINT8_REF = 112,
  TF_QINT32_REF = 113,
  TF_BFLOAT16_REF = 114,
  TF_QINT16_REF = 115,
  TF_QUINT16_REF = 116,
  TF_UINT16_REF = 117,
  TF_COMPLEX128_REF = 118,
  TF_HALF_REF = 119,
  TF_RESOURCE_REF = 120,
  TF_VARIANT_REF = 121,
  TF_UINT32_REF = 122,
  TF_UINT64_REF = 123
 );
 PTF_DataType = ^TF_DataType;
TF_Code = (
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15
 );


/// <summary>TF_AttrType describes the type of the value of an attribute on an operation.</summary>
TF_AttrType = (
  TF_ATTR_STRING = 0,
  TF_ATTR_INT = 1,
  TF_ATTR_FLOAT = 2,
  TF_ATTR_BOOL = 3,
  TF_ATTR_TYPE = 4,
  TF_ATTR_SHAPE = 5,
  TF_ATTR_TENSOR = 6,
  TF_ATTR_PLACEHOLDER = 7,
  TF_ATTR_FUNC = 8
);
TF_AttrValueCase = (
  VALUE_NOT_SET = 0,
  kList = 1,
  kS = 2,
  kI = 3,
  kF = 4,
  kB = 5,
  kType = 6,
  kShape = 7,
  kTensor = 8,
  kPlaceholder = 9,
  kFunc = 10
);
// --------------------------------------------------------------------------
/// <summary>TF_Status holds error information. It either has an OK code, or
/// else an error code with an associated error message.</summary>
PTF_Status = Pointer;

// --------------------------------------------------------------------------
/// <summary>TF_Buffer holds a pointer to a block of data and its associated length.
/// Typically, the data consists of a serialized protocol buffer, but other data
/// may also be held in a buffer.</summary>
/// <remarks>By default, TF_Buffer itself does not do any memory management of the
/// pointed-to block. If need be, users of this struct should specify how to
/// deallocate the block by setting the <c>data_deallocator</c> function pointer.</remarks>
TF_Buffer = record
  data:   Pointer;
  length: TF_size_t;
  data_deallocator: TF_data_deallocator;
end;
PTF_Buffer = ^TF_Buffer;

// -----------------------------------------------------------------------------
/// <summary>TF_Tensor holds a multi-dimensional array of elements of a single data type.
/// For all types other than TF_STRING, the data buffer stores elements
/// in row major order.</summary>
/// <remarks>E.g. if data is treated as a vector of <c>TF_DataType</c>:
///   element 0:   index (0, ..., 0)
///   element 1:   index (0, ..., 1)
///   ...
/// The format for <c>TF_STRING</c> tensors is:
///   start_offset: array[uint64]
///   data:         byte[...]
///
///   The string length (as a varint), followed by the contents of the string
///   is encoded at data[start_offset[i]]]. <c>TF_StringEncode</c> and <c>TF_StringDecode</c>
///   facilitate this encoding.</remarks>
PTF_Tensor = Pointer;


// -----------------------------------------------------------------------------

/// <summary>TF_SessionOptions holds options that can be passed during session creation.</summary>
PTF_SessionOptions = Pointer;


// -----------------------------------------------------------------------------

/// <summary>Represents a computation graph.  Graphs may be shared between sessions.
/// Graphs are thread-safe when used as directed below.</summary>
PTF_Graph = Pointer;
/// <summary>Operation being built. The underlying graph must outlive this.</summary>
PTF_OperationDescription = Pointer;
/// <summary>Operation that has been added to the graph. Valid until the graph is
/// deleted -- in particular adding a new operation to the graph does not
/// invalidate old TF_Operation* pointers.</summary>
PTF_Operation = Pointer;
/// <summary>Represents a specific input of an operation.</summary>
TF_Input = record
  oper:  PTF_Operation;
  /// <summary>The index of the input within oper.</summary>
  index: Integer;
end;
PTF_Input = ^TF_Input;
/// <summary>Represents a specific output of an operation.</summary>
TF_Output = record
  oper:  PTF_Operation;
  /// <summary>The index of the output within oper.</summary>
  index: Integer;
  constructor Create(op:PTF_Operation;idx: Integer);
end;
PTF_Output = ^TF_Output;
/// <summary>TF_AttrMetadata describes the value of an attribute on an operation.</summary>
TF_AttrMetadata = record
  /// <summary>A boolean: 1 if the attribute value is a list, 0 otherwise.</summary>
  is_list: Boolean;
  /// <summary>Length of the list if is_list is true. Undefined otherwise.</summary>
  list_size: TF_Int64_t;
  /// <summary>Type of elements of the list if is_list != 0.
  /// <summary>Type of the single value stored in the attribute if is_list == 0.</summary>
  AttrType: TF_AttrType;
  /// <summary>Total size the attribute value.
  /// The units of total_size depend on is_list and type.
  /// (1) If type == TF_ATTR_STRING and is_list == 0
  ///     then total_size is the byte size of the string
  ///     valued attribute.
  /// (2) If type == TF_ATTR_STRING and is_list == 1
  ///     then total_size is the cumulative byte size
  ///     of all the strings in the list.
  /// (3) If type == TF_ATTR_SHAPE and is_list == 0
  ///     then total_size is the number of dimensions
  ///     of the shape valued attribute, or -1
  ///     if its rank is unknown.
  /// (4) If type == TF_ATTR_SHAPE and is_list == 1
  ///     then total_size is the cumulative number
  ///     of dimensions of all shapes in the list.
  /// (5) Otherwise, total_size is undefined.</summary>
  total_size: TF_Int64_t;
end;
PTF_AttrMetadata = ^TF_AttrMetadata;

TF_WhileParams = record
  /// <summary>The number of inputs to the while loop, i.e. the number of loop variables.
  /// This is the size of cond_inputs, body_inputs, and body_outputs.</summary>
  ninputs: Integer;
  /// <summary>The while condition graph. The inputs are the current values of the loop
  /// variables. The output should be a scalar boolean.</summary>
  cond_graph:   PTF_Graph;
  cond_inputs:  PTF_Output;
  cond_output:  TF_Output;
  /// <summary>The loop body graph. The inputs are the current values of the loop
  /// variables. The outputs are the updated values of the loop variables.</summary>
  body_graph:   PTF_Graph;
  body_inputs:  PTF_Output;
  body_outputs: PTF_Output;
  /// <summary>Unique null-terminated name for this while loop. This is used as a prefix
  /// for created operations.</summary>
  name: PTFChar;
end;
PTF_WhileParams = ^TF_WhileParams;


// -----------------------------------------------------------------------------
PTF_Session  = Pointer;
PTF_OpList   = Pointer;
PTF_OpDef    = Pointer;
PTF_OpDefArg = Pointer;
PTF_OpDefAttr= Pointer;
PTF_AttrValue= Pointer;
PTF_GraphDef = Pointer;
PTF_MetaGraphDef=Pointer;
PTF_NodeDef  = Pointer;
PTF_AttrMap  = Pointer;
PTF_DeviceList= Pointer;
PTF_Library  = Pointer;
PTF_SignatureDefMap=Pointer;
PTF_SignatureDef=Pointer;
/// <summary><c>TF_ImportGraphDefOptions</c> holds options that can be passed to
/// <c>TF_GraphImportGraphDef</c>.</summary>
PTF_ImportGraphDefOptions = Pointer;

const
  c_sNameOfTensorflowLib = 'tensorflow.dll';

// -----------------------------------------------------------------------------
// Helper-Functions / Procedures
procedure Deallocator_For_TensorDatas(data: Pointer; len: TF_size_t; arg: Pointer); cdecl;

/// <summary>TF_Version returns a string describing version information of the
/// TensorFlow library. TensorFlow using semantic versioning.</summary>
/// <returns>Pointer to versions string</returns>
function TF_Version(): PTFChar; cdecl;
{$EXTERNALSYM TF_Version}

/// <summary><c>TF_DataTypeSize</c> returns the sizeof() for the underlying type corresponding
/// to the given TF_DataType enum value. Returns 0 for variable length types
/// (eg. TF_STRING) or on failure.</summary>
/// <returns>The data type size.</returns>
function TF_DataTypeSize(dt: Int32): TF_size_t; cdecl;
{$EXTERNALSYM TF_DataTypeSize}

/// <summary>Return a new instance of status.</summary>
function TF_NewStatus(): PTF_Status; cdecl;
{$EXTERNALSYM TF_NewStatus}

/// <summary>Return the code record in s.</summary>
/// <param name="s">Pointer to status object</param>
/// <returns>Gets the status code for the status object</returns>
function TF_GetCode(const s: PTF_Status): TF_Code; cdecl;
{$EXTERNALSYM TF_GetCode}

/// <summary>Return a pointer to the (null-terminated) error message in s.  The
/// return value points to memory that is only usable until the next
/// mutation to s. Always returns an empty string if TF_GetCode(s) is
/// TF_OK.</summary>
/// <param name="s">Pointer to status object</param>
/// <returns>The status message.</returns>
function TF_Message(const s: PTF_Status): PTFChar; cdecl;
{$EXTERNALSYM TF_Message}

/// <summary>Record <code, msg> in s.  Any previous information is lost.</summary>
/// <remarks>A common use is to clear a status: <code>TF_SetStatus(s, TF_OK, "");</code></remarks>
procedure TF_SetStatus(s: PTF_Status; code: TF_Code; const msg: PTFChar); cdecl;
{$EXTERNALSYM TF_SetStatus}

/// <summary>Delete a previously created status object.</summary>
procedure TF_DeleteStatus(s: PTF_Status); cdecl;
{$EXTERNALSYM TF_DeleteStatus}

/// <summary>Makes a copy of the input and sets an appropriate deallocator.  Useful for
/// passing in read-only, input protobufs.</summary>
function TF_NewBufferFromString(proto: Pointer; proto_len: TF_size_t): PTF_Buffer; cdecl;
{$EXTERNALSYM TF_NewBufferFromString}

/// <summary>Useful for passing *out* a protobuf.</summary>
function TF_NewBuffer(): PTF_Buffer; cdecl;
{$EXTERNALSYM TF_NewBuffer}
/// <summary>Delete the buffer.</summary>
procedure TF_DeleteBuffer(buffer: PTF_Buffer); cdecl;
{$EXTERNALSYM TF_DeleteBuffer}
function TF_GetBuffer(buffer: PTF_Buffer): TF_Buffer; cdecl;
{$EXTERNALSYM TF_GetBuffer}

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator)(data, len, deallocator_arg)
// Clients must provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
function TF_NewTensor(dt: Int32; const dims: PTF_int64_t; num_dims: Integer;
                      data: Pointer; len: TF_size_t; deallocator: TF_deallocator;
                      deallocator_arg: Pointer): PTF_Tensor; cdecl;
{$EXTERNALSYM TF_NewTensor}

// Allocate and return a new Tensor.
//
// This function is an alternative to TF_NewTensor and should be used when
// memory is allocated to pass the Tensor to the C API. The allocated memory
// satisfies TensorFlow's memory alignment preferences and should be preferred
// over calling malloc and free.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
function TF_AllocateTensor(dt: Int32; const dims: PTF_int64_t; num_dims: Integer;
                           len: TF_size_t): PTF_Tensor; cdecl;
{$EXTERNALSYM TF_AllocateTensor}

// Deletes `tensor` and returns a new TF_Tensor with the same content if
// possible. Returns nullptr and leaves `tensor` untouched if not.
function TF_TensorMaybeMove(tensor: PTF_Tensor): PTF_Tensor; cdecl;
{$EXTERNALSYM TF_TensorMaybeMove}
// Destroy a tensor.
procedure TF_DeleteTensor(tensor: PTF_Tensor); cdecl;
{$EXTERNALSYM TF_DeleteTensor}
// Return the type of a tensor element.
function TF_TensorType(const tensor: PTF_Tensor): Int32; cdecl;
{$EXTERNALSYM TF_TensorType}
// Return the number of dimensions that the tensor has.
function TF_NumDims(const tensor: PTF_Tensor): Integer; cdecl;
{$EXTERNALSYM TF_NumDims}
// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
function TF_Dim(const tensor: PTF_Tensor; dim_index: Integer): TF_int64_t; cdecl;
{$EXTERNALSYM TF_Dim}
// Return the size of the underlying data in bytes.
function TF_TensorByteSize(const tensor: PTF_Tensor): TF_size_t; cdecl;
{$EXTERNALSYM TF_TensorByteSize}
// Return a pointer to the underlying data buffer.
function TF_TensorData(const tensor: PTF_Tensor): Pointer; cdecl;
{$EXTERNALSYM TF_TensorData}

// --------------------------------------------------------------------------
// Encode the string `src` (`src_len` bytes long) into `dst` in the format
// required by TF_STRING tensors. Does not write to memory more than `dst_len`
// bytes beyond `*dst`. `dst_len` should be at least
// TF_StringEncodedSize(src_len).
//
// On success returns the size in bytes of the encoded string.
// Returns an error into `status` otherwise.
//** Modified by Max removed api from tensorflow 2.4  11/12/2021 17:21:00
//function TF_StringEncode(const src: PTFChar; src_len: TF_size_t; dst: PTFChar; dst_len: TF_size_t;   status: PTF_Status): TF_size_t; cdecl;
//{$EXTERNALSYM TF_StringEncode}

// Decode a string encoded using TF_StringEncode.
//
// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
// its length. Returns the number of bytes starting at `src` consumed while
// decoding. `*dst` points to memory within the encoded buffer.  On failure,
// `*dst` and `*dst_len` are undefined and an error is set in `status`.
//
// Does not read memory more than `src_len` bytes beyond `src`.
//** Modified by Max removed api from tensorflow 2.4  11/12/2021 17:21:00
//function TF_StringDecode(const src: PTFChar; len: TF_size_t;  var dst: PTFChar; dst_len: PTF_size_t; status: PTF_Status): TF_size_t; cdecl;
//{$EXTERNALSYM TF_StringDecode}

// Return the size in bytes required to encode a string `len` bytes long into a
// TF_STRING tensor.
function TF_StringEncodedSize(len: TF_size_t): TF_size_t; cdecl;
{$EXTERNALSYM TF_StringEncodedSize}

procedure TF_StringInit(t: PTFString); cdecl;
{$EXTERNALSYM TF_StringInit}

procedure TF_StringCopy(dst: PTFString; const src: PTFChar; size: TF_size_t); cdecl;
{$EXTERNALSYM TF_StringCopy}

function TF_StringGetDataPointer(const tstr: PTFString): PTFChar; cdecl;
{$EXTERNALSYM TF_StringGetDataPointer}

function TF_StringGetSize(const tstr: PTFString):TF_size_t;cdecl;
{$EXTERNALSYM TF_StringGetSize}

function TF_StringGetCapacity(const tstr: PTFString):TF_size_t;cdecl;
{$EXTERNALSYM TF_StringGetCapacity}

procedure TF_StringDealloc(tstr: PTFString);
{$EXTERNALSYM TF_StringDealloc}

// --------------------------------------------------------------------------
// Return a new options object.
function TF_NewSessionOptions(): PTF_SessionOptions; cdecl;
{$EXTERNALSYM TF_NewSessionOptions}

/// <summary>
/// Set the target in TF_SessionOptions.options.
/// </summary>
/// <remarks>
/// target can be empty, a single entry, or a comma separated list of entries.
/// Each entry is in one of the following formats :
/// "local"
/// ip:port
/// host:port
/// </remarks>
procedure TF_SetTarget(options: PTF_SessionOptions; target: PTFChar);
{$EXTERNALSYM TF_SetTarget}

/// <summary>
/// Set the config in TF_SessionOptions.options.
/// </summary>
/// <remarks>
/// config should be a serialized tensorflow.ConfigProto proto.
/// If config was not parsed successfully as a ConfigProto, record the
/// error information in *status.
/// </remarks>
procedure TF_SetConfig(options: PTF_SessionOptions;
                       const proto: Pointer; proto_len: TF_size_t;
                       status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetConfig}

// Experimental C API for TensorFlow.
//
function TF_CreateConfig(enable_xla_compilation: Byte; gpu_memory_allow_growth: Byte; num_cpu_devices: UInt32): PTF_Buffer;
{$EXTERNALSYM TF_CreateConfig}

/// <summary>
/// Destroy an options object.
/// </summary>
procedure TF_DeleteSessionOptions(options: PTF_SessionOptions);  cdecl;
{$EXTERNALSYM TF_DeleteSessionOptions}

// --------------------------------------------------------------------------
// The new graph construction API, still under development.
/// <summary>
/// Return a new graph object.
/// </summary>
function TF_NewGraph(): PTF_Graph; cdecl;
{$EXTERNALSYM TF_NewGraph}

// Destroy an options object.  Graph will be deleted once no more
// TFSession's are referencing it.
procedure TF_DeleteGraph(graph: PTF_Graph); cdecl;
{$EXTERNALSYM TF_DeleteGraph}

// Sets the shape of the Tensor referenced by `output` in `graph` to
// the shape described by `dims` and `num_dims`.
//
// If the number of dimensions is unknown, `num_dims` must be
// set to -1 and dims can be null. If a dimension is unknown,
// the corresponding entry in the `dims` array must be -1.
//
// This does not overwrite the existing shape associated with `output`,
// but merges the input shape with the existing shape.  For example,
// setting a shape of [-1, 2] with an existing shape [2, -1] would set
// a final shape of [2, 2] based on shape merging semantics.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * An invalid shape is being set (e.g., the shape being set
//     is incompatible with the existing shape).
procedure TF_GraphSetTensorShape(graph: PTF_Graph; output: TF_Output;
                                 const dims: PTF_int64_t; const num_dims: Integer;
                                 status: PTF_Status); cdecl;
{$EXTERNALSYM TF_GraphSetTensorShape}

// Returns the number of dimensions of the Tensor referenced by `output`
// in `graph`.
//
// If the number of dimensions in the shape is unknown, returns -1.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
function TF_GraphGetTensorNumDims(graph: PTF_Graph; output: TF_Output;
                                  status: PTF_Status): Integer; cdecl;
{$EXTERNALSYM TF_GraphGetTensorNumDims}

// Returns the shape of the Tensor referenced by `output` in `graph`
// into `dims`. `dims` must be an array large enough to hold `num_dims`
// entries (e.g., the return value of TF_GraphGetTensorNumDims).
//
// If the number of dimensions in the shape is unknown or the shape is
// a scalar, `dims` will remain untouched. Otherwise, each element of
// `dims` will be set corresponding to the size of the dimension. An
// unknown dimension is represented by `-1`.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * `num_dims` does not match the actual number of dimensions.
procedure TF_GraphGetTensorShape(graph: PTF_Graph; output: TF_Output;
                                 dims: PTF_int64_t; num_dims: Integer;
                                 status: PTF_Status); cdecl;
{$EXTERNALSYM TF_GraphGetTensorShape}

// Operation will only be added to *graph when TF_FinishOperation() is
// called (assuming TF_FinishOperation() does not return an error).
// *graph must not be deleted until after TF_FinishOperation() is
// called.
function TF_NewOperation(graph: PTF_Graph; const op_type: PTFChar;
                         const oper_name: PTFChar): PTF_OperationDescription; cdecl;
{$EXTERNALSYM TF_NewOperation}
// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
procedure TF_SetDevice(desc: PTF_OperationDescription; const device: PTFChar); cdecl;
{$EXTERNALSYM TF_SetDevice}

// The calls to TF_AddInput and TF_AddInputList must match (in number,
// order, and type) the op declaration.  For example, the "Concat" op
// has registration:
//   REGISTER_OP("Concat")
//       .Input("concat_dim: int32")
//       .Input("values: N * T")
//       .Output("output: T")
//       .Attr("N: int >= 2")
//       .Attr("T: type");
// that defines two inputs, "concat_dim" and "values" (in that order).
// You must use TF_AddInput() for the first input (since it takes a
// single tensor), and TF_AddInputList() for the second input (since
// it takes a list, even if you were to pass a list with a single
// tensor), as in:
//   TF_OperationDescription* desc = TF_NewOperation(graph, "Concat", "c");
//   TF_Output concat_dim_input = {...};
//   TF_AddInput(desc, concat_dim_input);
//   TF_Output values_inputs[5] = {{...}, ..., {...}};
//   TF_AddInputList(desc, values_inputs, 5);
// For inputs that take a single tensor.
procedure TF_AddInput(desc: PTF_OperationDescription; input: TF_Output); cdecl;
{$EXTERNALSYM TF_AddInput}

// For inputs that take a list of tensors.
// inputs must point to TF_Output[num_inputs].
procedure TF_AddInputList(desc: PTF_OperationDescription;
                          const inputs: PTF_Output;
                          num_inputs: Integer); cdecl;
{$EXTERNALSYM TF_AddInputList}
// Call once per control input to `desc`.
procedure TF_AddControlInput(desc: PTF_OperationDescription; input: PTF_Operation); cdecl;
{$EXTERNALSYM TF_AddControlInput}

// Request that `desc` be co-located on the device where `op`
// is placed.
//
// Use of this is discouraged since the implementation of device placement is
// subject to change. Primarily intended for internal libraries
procedure TF_ColocateWith(desc: PTF_OperationDescription; op: PTF_Operation); cdecl;
{$EXTERNALSYM TF_ColocateWith}
// Call some TF_SetAttr*() function for every attr that is not
// inferred from an input and doesn't have a default value you wish to
// keep.
/// <summary><c>value</c> must point to a string of length <c>length</c> bytes.</summary>
procedure TF_SetAttrString(desc: PTF_OperationDescription;
                           const attr_name: PTFChar;
                           const value: Pointer; len: TF_size_t); cdecl;
{$EXTERNALSYM TF_SetAttrString}
/// <summary><c>values</c> and <c>lengths</c> each must have lengths <c>num_values</c>.
/// <c>values[i]</c> must point to a string of length <c>lengths[i]</c> bytes.</summary>
procedure TF_SetAttrStringList(desc: PTF_OperationDescription;
                               const attr_name: PTFChar;
                               const values: PTFChar;
                               const lengths: PTF_size_t;
                               num_values: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrStringList}
procedure TF_SetAttrInt(desc: PTF_OperationDescription;
                        const attr_name: PTFChar; value: TF_int64_t); cdecl;
{$EXTERNALSYM TF_SetAttrInt}
procedure TF_SetAttrIntList(desc: PTF_OperationDescription;
                            const attr_name: PTFChar;
                            const values: PTF_int64_t;
                            num_values: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrIntList}
procedure TF_SetAttrFloat(desc: PTF_OperationDescription;
                          const attr_name: PTFChar; value: Single); cdecl;
{$EXTERNALSYM TF_SetAttrFloat}
procedure TF_SetAttrFloatList(desc: PTF_OperationDescription;
                              const attr_name: PTFChar;
                              const values: PSingle;
                              num_values: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrFloatList}
procedure TF_SetAttrBool(desc: PTF_OperationDescription;
                         const attr_name: PTFChar;
                         value: Boolean); cdecl;
{$EXTERNALSYM TF_SetAttrBool}
procedure TF_SetAttrBoolList(desc: PTF_OperationDescription;
                             const attr_name: PTFChar;
                             const values: PBoolean;
                             num_values: Integer);
{$EXTERNALSYM TF_SetAttrBoolList}
procedure TF_SetAttrType(desc: PTF_OperationDescription;
                         const attr_name: PTFChar;
                         value: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrType}
procedure TF_SetAttrTypeList(desc: PTF_OperationDescription;
                             const attr_name: PTFChar;
                             const values: PInteger;
                             num_values: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrTypeList}

// Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
// `dims` points to an array of length `num_dims`.  `dims[i]` must be
// >= -1, with -1 meaning "unknown dimension".
procedure TF_SetAttrShape(desc: PTF_OperationDescription;
                          const attr_name: PTFChar;
                          const dims: PTF_Int64_t; num_dims: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrShape}
// `dims` and `num_dims` must point to arrays of length `num_shapes`.
// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
// must be >= -1, with -1 meaning "unknown dimension".
procedure TF_SetAttrShapeList(desc: PTF_OperationDescription;
                              const attr_name: PTFChar;
                              const dims: PTF_Int64_t;
                              const num_dims: PInteger;
                              num_shapes: Integer); cdecl;
{$EXTERNALSYM TF_SetAttrShapeList}
// `proto` must point to an array of `proto_len` bytes representing a
// binary-serialized TensorShapeProto.
procedure TF_SetAttrTensorShapeProto(desc: PTF_OperationDescription;
                              const attr_name: PTFChar; const proto: Pointer;
                              proto_len: TF_Size_t; status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetAttrTensorShapeProto}
// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
// `protos[i]` must point to an array of `proto_lens[i]` bytes
// representing a binary-serialized TensorShapeProto.
procedure TF_SetAttrTensorShapeProtoList(
                              desc: PTF_OperationDescription;
                              const attr_name: PTFChar;
                              var   protos: Pointer;
                              const proto_lens: PTF_Size_t;
                              num_shapes: Integer;
                              status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetAttrTensorShapeProtoList}
procedure TF_SetAttrTensor(   desc: PTF_OperationDescription;
                              const attr_name: PTFChar;
                              value: PTF_Tensor;
                              status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetAttrTensor}
procedure TF_SetAttrTensorList(desc: PTF_OperationDescription;
                               const attr_name: PTFChar;
                               var values: PTF_Tensor;
                               num_values: Integer;
                               status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetAttrTensorList}
// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
procedure TF_SetAttrValueProto(desc: PTF_OperationDescription;
                               const attr_name: PTFChar;
                               const proto: Pointer;
                               proto_len: TF_Size_t;
                               status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SetAttrValueProto}
// If this function succeeds:
//   * *status is set to an OK value,
//   * a TF_Operation is added to the graph,
//   * a non-null value pointing to the added operation is returned --
//     this value is valid until the underlying graph is deleted.
// Otherwise:
//   * *status is set to a non-OK value,
//   * the graph is not modified,
//   * a null value is returned.
// In either case, it deletes `desc`.
function TF_FinishOperation(desc: PTF_OperationDescription;
                            status: PTF_Status): PTF_Operation; cdecl;
{$EXTERNALSYM TF_FinishOperation}
// TF_Operation functions.  Operations are immutable once created, so
// these are all query functions.
function TF_OperationName(oper: PTF_Operation): PTFChar; cdecl;
{$EXTERNALSYM TF_OperationName}
function TF_OperationOpType(oper: PTF_Operation): PTFChar; cdecl;
{$EXTERNALSYM TF_OperationOpType}
function TF_OperationDevice(oper: PTF_Operation): PTFChar; cdecl;
{$EXTERNALSYM TF_OperationDevice}
function TF_OperationNumOutputs(oper: PTF_Operation): Integer; cdecl;
{$EXTERNALSYM TF_OperationNumOutputs}
function TF_OperationOutputType(oper_out: TF_Output): Int32; cdecl;
{$EXTERNALSYM TF_OperationOutputType}
function TF_OperationOutputListLength(oper: PTF_Operation;
                            const arg_name: PTFChar;
                            status: PTF_Status): Integer; cdecl;
{$EXTERNALSYM TF_OperationOutputListLength}
function TF_OperationNumInputs(oper: PTF_Operation): Integer; cdecl;
{$EXTERNALSYM TF_OperationNumInputs}
function TF_OperationInputType(oper_in: TF_Input): Int32; cdecl;
{$EXTERNALSYM TF_OperationInputType}
function TF_OperationInputListLength(oper: PTF_Operation;
                            const arg_name: PTFChar;
                            status: PTF_Status): Integer; cdecl;
{$EXTERNALSYM TF_OperationInputListLength}
// In this code:
//   TF_Output producer = TF_OperationInput(consumer);
// There is an edge from producer.oper's output (given by
// producer.index) to consumer.oper's input (given by consumer.index).
function TF_OperationInput(oper_in: TF_Input): TF_Output; cdecl;
{$EXTERNALSYM TF_OperationInput}
// Get the number of current consumers of a specific output of an
// operation.  Note that this number can change when new operations
// are added to the graph.
function TF_OperationOutputNumConsumers(oper_out: TF_Output): Integer; cdecl;
{$EXTERNALSYM TF_OperationOutputNumConsumers}
// Get list of all current consumers of a specific output of an
// operation.  `consumers` must point to an array of length at least
// `max_consumers` (ideally set to
// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
// modification of the graph can increase the number of consumers of
// an operation.  Returns the number of output consumers (should match
// TF_OperationOutputNumConsumers(oper_out)).
function TF_OperationOutputConsumers(oper_out: TF_Output;
                                     consumers: PTF_Input;
                                     max_consumers: Integer): Integer; cdecl;
{$EXTERNALSYM TF_OperationOutputConsumers}
// Get the number of control inputs to an operation.
function TF_OperationNumControlInputs(oper: PTF_Operation): Integer;
{$EXTERNALSYM TF_OperationNumControlInputs}
// Get list of all control inputs to an operation.  `control_inputs` must
// point to an array of length `max_control_inputs` (ideally set to
// TF_OperationNumControlInputs(oper)).  Returns the number of control
// inputs (should match TF_OperationNumControlInputs(oper)).
function TF_OperationGetControlInputs(oper: PTF_Operation;
                                      control_inputs: PTF_Operation;
                                      max_control_inputs: Integer): Integer; cdecl;
{$EXTERNALSYM TF_OperationGetControlInputs}
// Get the number of operations that have `*oper` as a control input.
// Note that this number can change when new operations are added to
// the graph.
function TF_OperationNumControlOutputs(oper: PTF_Operation): Integer; cdecl;
{$EXTERNALSYM TF_OperationNumControlOutputs}
// Get the list of operations that have `*oper` as a control input.
// `control_outputs` must point to an array of length at least
// `max_control_outputs` (ideally set to
// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
// modification of the graph can increase the number of control
// outputs.  Returns the number of control outputs (should match
// TF_OperationNumControlOutputs(oper)).
function TF_OperationGetControlOutputs(oper: PTF_Operation;
                                       var control_outputs: PTF_Operation;
                                       max_control_outputs: Integer): Integer; cdecl;

{$EXTERNALSYM TF_OperationGetControlOutputs}



// -----------------------------------------------------------------------------


// Returns metadata about the value of the attribute `attr_name` of `oper`.

function TF_OperationGetAttrMetadata(oper: PTF_Operation; attr_name: PTFChar;
                                     status: PTF_Status): TF_AttrMetadata;
{$EXTERNALSYM TF_OperationGetAttrMetadata}


// Fills in `value` with the value of the attribute `attr_name`.  `value` must

// point to an array of length at least `max_length` (ideally set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
procedure TF_OperationGetAttrString(oper: PTF_Operation; attr_name: PTFChar;
                                    value: Pointer; max_length: TF_size_t;
                                    status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrString}

// Get the list of strings in the value of the attribute `attr_name`.  Fills in
// `values` and `lengths`, each of which must point to an array of length at
// least `max_values`.
//
// The elements of values will point to addresses in `storage` which must be at
// least `storage_size` bytes in length.  Ideally, max_values would be set to
// TF_AttrMetadata.list_size and `storage` would be at least
// TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is too small to hold the requested number of strings.
procedure TF_OperationGetAttrStringList(
                                 oper: PTF_Operation; attr_name: PTFChar;
                                 values: Pointer; lengths: PTF_size_t;
                                 max_values: Integer; storage: Pointer;
                                 storage_size: TF_size_t; status: PTF_Status);
{$EXTERNALSYM TF_OperationGetAttrStringList}

procedure TF_OperationGetAttrInt(oper: PTF_Operation; attr_name: PTFChar;
                                 value: PTF_int64_t;  status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrInt}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
procedure TF_OperationGetAttrIntList(oper: PTF_Operation; attr_name: PTFChar;
                                     values: PTF_int64_t;
                                     max_values: Integer;  status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrIntList}

procedure TF_OperationGetAttrFloat(oper: PTF_Operation; attr_name: PTFChar;
                                   value: PSingle;  status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrFloat}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
procedure TF_OperationGetAttrFloatList(oper: PTF_Operation; attr_name: PTFChar;
                                       values: PSingle;
                                       max_values: Integer;
                                       status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrFloatList}

procedure TF_OperationGetAttrBool(oper: PTF_Operation; attr_name: PTFChar;
                                  value: PTF_UINT8; status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrBool}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
procedure TF_OperationGetAttrBoolList(oper: PTF_Operation; attr_name: PTFChar;
                                      values: PTF_UINT8;
                                      max_values: Integer;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrBoolList}

procedure TF_OperationGetAttrType(oper: PTF_Operation; attr_name: PTFChar;
                                      value: PInt32;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrType}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
procedure TF_OperationGetAttrTypeList(oper: PTF_Operation; attr_name: PTFChar;
                                      values: PInt32;
                                      max_values: Integer;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrTypeList}

// Fills in `value` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `num_dims` (ideally set to
// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
procedure TF_OperationGetAttrShape(oper: PTF_Operation; attr_name: PTFChar;
                                      value: PTF_int64_t;
                                      num_dims: Integer;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrShape}

// Fills in `dims` with the list of shapes in the attribute `attr_name` of
// `oper` and `num_dims` with the corresponding number of dimensions. On return,
// for every i where `num_dims[i]` > 0, `dims[i]` will be an array of
// `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
// i-th shape in the list is unknown.
//
// The elements of `dims` will point to addresses in `storage` which must be
// large enough to hold at least `storage_size` int64_ts.  Ideally, `num_shapes`
// would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is insufficient to hold the requested shapes.
procedure TF_OperationGetAttrShapeList(
                                      oper: PTF_Operation;
                                      attr_name: PTFChar;
                                      dims: Pointer;
                                      num_dims: PInteger;
                                      num_shapes: Integer;
                                      storage: PTF_int64_t;
                                      storage_size: Integer;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrShapeList}

// Sets `value` to the binary-serialized TensorShapeProto of the value of
// `attr_name` attribute of `oper`'.
procedure TF_OperationGetAttrTensorShapeProto(
                                      oper: PTF_Operation;
                                      attr_name: PTFChar; value: PTF_Buffer;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrTensorShapeProto}

// Fills in `values` with binary-serialized TensorShapeProto values of the
// attribute `attr_name` of `oper`. `values` must point to an array of length at
// least `num_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
procedure TF_OperationGetAttrTensorShapeProtoList(
                                      oper: PTF_Operation;
                                      attr_name: PTFChar;
                                      var values: PTF_Buffer;
                                      max_values: Integer; status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrTensorShapeProtoList}

// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
//
// Allocates a new TF_Tensor which the caller is expected to take
// ownership of (and can deallocate using TF_DeleteTensor).
procedure TF_OperationGetAttrTensor(oper: PTF_Operation; attr_name: PTFChar;
                                      var value: PTF_Tensor;
                                      status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrTensor}

// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
// `oper`. `values` must point to an array of TF_Tensor* of length at least
// `max_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
//
// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
// (which can be deleted using TF_DeleteTensor(values[i])).
procedure TF_OperationGetAttrTensorList(oper: PTF_Operation; attr_name: PTFChar;
                                       var values: PTF_Tensor;
                                       max_values: Integer;
                                       status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationGetAttrTensorList}

// Sets `output_attr_value` to the binary-serialized AttrValue proto
// representation of the value of the `attr_name` attr of `oper`.
procedure TF_OperationGetAttrValueProto(
                                       oper: PTF_Operation; attr_name: PTFChar;
                                       output_attr_value: PTF_Buffer;
                                       status: PTF_Status);  cdecl;
{$EXTERNALSYM TF_OperationGetAttrValueProto}

// Returns the operation in the graph with `oper_name`. Returns nullptr if
// no operation found.
function  TF_GraphOperationByName(graph: PTF_Graph; oper_name: PTFChar): PTF_Operation;  cdecl;
{$EXTERNALSYM TF_GraphOperationByName}

// Iterate through the operations of a graph.  To use:
// size_t pos = 0;
// TF_Operation* oper;
// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
//   DoSomethingWithOperation(oper);
// }
function  TF_GraphNextOperation(graph: PTF_Graph; pos: PTF_size_t): PTF_Operation;  cdecl;
{$EXTERNALSYM TF_GraphNextOperation}

// Write out a serialized representation of `graph` (as a GraphDef protocol
// message) to `output_graph_def` (allocated by TF_NewBuffer()).
// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
// is called.
//
// May fail on very large graphs in the future.
procedure TF_GraphToGraphDef(graph: PTF_Graph; output_graph_def: PTF_Buffer;
                                       status: PTF_Status);  cdecl;
{$EXTERNALSYM TF_GraphToGraphDef}

// -----------------------------------------------------------------------------

function  TF_NewImportGraphDefOptions(): PTF_ImportGraphDefOptions;  cdecl;
{$EXTERNALSYM TF_NewImportGraphDefOptions}

procedure TF_DeleteImportGraphDefOptions(opts: PTF_ImportGraphDefOptions);  cdecl;
{$EXTERNALSYM TF_DeleteImportGraphDefOptions}

// Set the prefix to be prepended to the names of nodes in `graph_def` that will
// be imported into `graph`.
procedure TF_ImportGraphDefOptionsSetPrefix(
                            opts: PTF_ImportGraphDefOptions;
                            const prefix: PTFChar);  cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsSetPrefix}

// Set any imported nodes with input `src_name:src_index` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references a node already existing in the graph being imported into.
procedure TF_ImportGraphDefOptionsAddInputMapping(
                            opts: PTF_ImportGraphDefOptions;
                            const src_name: PTFChar; src_index: Integer;
                            dst: TF_Output);  cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsAddInputMapping}

// Set any imported nodes with control input `src_name` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references an operation already existing in the graph being imported
// into.
procedure TF_ImportGraphDefOptionsRemapControlDependency(
                            opts: PTF_ImportGraphDefOptions;
                            const src_name: PTFChar;
                            dst: PTF_Operation);  cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsRemapControlDependency}

// Cause the imported graph to have a control dependency on `oper`. `oper`
// should exist in the graph being imported into.
procedure TF_ImportGraphDefOptionsAddControlDependency(
                            opts: PTF_ImportGraphDefOptions;
                            oper: PTF_Operation);  cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsAddControlDependency}

// Add an output in `graph_def` to be returned via the `return_outputs` output
// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
// mapping, the corresponding existing tensor in `graph` will be returned.
procedure TF_ImportGraphDefOptionsAddReturnOutput(
                            opts: PTF_ImportGraphDefOptions;
                            const oper_name: PTFChar; idx: Integer); cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsAddReturnOutput}

// Returns the number of return outputs added via
// TF_ImportGraphDefOptionsAddReturnOutput().
function TF_ImportGraphDefOptionsNumReturnOutputs(
                            const opts: PTF_ImportGraphDefOptions): Integer; cdecl;
{$EXTERNALSYM TF_ImportGraphDefOptionsNumReturnOutputs}

// Import the graph serialized in `graph_def` into `graph`.
//
// `num_return_outputs` must be the number of return outputs added (i.e. the
// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
// `num_return_outputs` is non-zero, `return_outputs` must be of length
// `num_return_outputs`. Otherwise it can be null.
procedure TF_GraphImportGraphDefWithReturnOutputs(
                            graph: PTF_Graph;
                            const graph_def: PTF_Buffer;
                            const opts: PTF_ImportGraphDefOptions;
                            return_outputs: PTF_Output;
                            num_return_outputs: Integer;
                            status: PTF_Status); cdecl;
{$EXTERNALSYM TF_GraphImportGraphDefWithReturnOutputs}

// Import the graph serialized in `graph_def` into `graph`.
// Convenience function for when no return outputs have been added.
procedure TF_GraphImportGraphDef(
                            graph: PTF_Graph;
                            const graph_def: PTF_Buffer;
                            const opts: PTF_ImportGraphDefOptions;
                            status: PTF_Status); cdecl;
{$EXTERNALSYM TF_GraphImportGraphDef}

// Note: The following function may fail on very large protos in the future.

procedure TF_OperationToNodeDef(oper: PTF_Operation;
                            output_node_def: PTF_Buffer;
                            status: PTF_Status); cdecl;
{$EXTERNALSYM TF_OperationToNodeDef}

// -----------------------------------------------------------------------------

// Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
// outputs that already exist in `g` used as initial values for the loop
// variables.
//
// The returned TF_WhileParams will have all fields initialized except
// `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
// allocated to size `ninputs`. The caller should build `cond_graph` and
// `body_graph` starting from the inputs, and store the final outputs in
// `cond_output` and `body_outputs`.
//
// If `status` is OK, the caller must call either TF_FinishWhile or
// TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
// returned TF_WhileParams is not valid, and the caller should not call
// TF_FinishWhile() or TF_AbortWhile().
//
// Missing functionality (TODO):
// - Gradients
// - Reference-type inputs
// - Directly referencing external tensors from the cond/body graphs (this is
//   possible in the Python API)
function  TF_NewWhile(g: PTF_Graph; inputs: PTF_Output;
                            ninputs: Integer;
                            status: PTF_Status): TF_WhileParams; cdecl;
{$EXTERNALSYM TF_NewWhile}

// Builds the while loop specified by `params` and returns the output tensors of
// the while loop in `outputs`. `outputs` should be allocated to size
// `params.ninputs`.
//
// `params` is no longer valid once this returns.
//
// Either this or TF_AbortWhile() must be called after a successful
// TF_NewWhile() call.
procedure TF_FinishWhile(const params: PTF_WhileParams;
                            status:  PTF_Status;
                            outputs: PTF_Output); cdecl;
{$EXTERNALSYM TF_FinishWhile}

// Frees `params`s resources without building a while loop. `params` is no
// longer valid after this returns. Either this or TF_FinishWhile() must be
// called after a successful TF_NewWhile() call.
procedure TF_AbortWhile(const params: PTF_WhileParams); cdecl;
{$EXTERNALSYM TF_AbortWhile}

// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
// `dx` are used as initial gradients (which represent the symbolic partial
// derivatives of some loss function `L` w.r.t. `y`).
// `dx` must be nullptr or have size `ny`.
// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
// shapes in `y`.
// The partial derivatives are returned in `dy`. `dy` should be allocated to
// size `nx`.
//
// WARNING: This function does not yet support all the gradients that python
// supports. See
// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
// for instructions on how to add C++ more gradients.
procedure TF_AddGradients(  g: PTF_Graph; y: PTF_Output; ny: Integer;
                            x: PTF_Output; nx: Integer; dx: PTF_Output;
                            status: PTF_Status; dy: PTF_Output); cdecl;

// TODO(josh11b): Register OpDef, available to all operations added
// to this graph.

// The following two may both benefit from a subgraph-definition API
// that re-uses most of the graph-definition API.
// TODO(andydavis): Add functions to a graph.

// -----------------------------------------------------------------------------
// API for driving Graph execution.

// Return a new execution session with the associated graph, or NULL on error.
//
// *graph must be a valid graph (not deleted or nullptr).  This function will
// prevent the graph from being deleted until TF_DeleteSession() is called.
// Does not take ownership of opts.
function TF_NewSession(     graph: PTF_Graph;
                            const opts: PTF_SessionOptions;
                            status: PTF_Status): PTF_Session; cdecl;
{$EXTERNALSYM TF_NewSession}
// This function creates a new TF_Session (which is created on success) using
// `session_options`, and then initializes state (restoring tensors and other
// assets) using `run_options`.
//
// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
// are valid.
//
// - `export_dir` must be set to the path of the exported SavedModel.
// - `tags` must include the set of tags used to identify one MetaGraphDef in
//    the SavedModel.
// - `graph` must be a graph newly allocated with TF_NewGraph().
//
// If successful, populates `graph` with the contents of the Graph and
// `meta_graph_def` with the MetaGraphDef of the loaded model.
function  TF_LoadSessionFromSavedModel(
                            const session_options: PTF_SessionOptions;
                            const run_options: PTF_Buffer;
                            const export_dir: PTFChar;
                            tags:  PTFChar;   tags_len: Integer;
                            graph: PTF_Graph; meta_graph_def: PTF_Buffer;
                            status: PTF_Status): PTF_Session; cdecl;
{$EXTERNALSYM TF_LoadSessionFromSavedModel}

// Close a session.
//
// Contacts any other processes associated with the session, if applicable.
// May not be called after TF_DeleteSession().
procedure TF_CloseSession(session: PTF_Session; status: PTF_Status); cdecl;
{$EXTERNALSYM TF_CloseSession}

// Destroy a session object.
//
// Even if error information is recorded in *status, this call discards all
// local resources associated with the session.  The session may not be used
// during or after this call (and the session drops its reference to the
// corresponding graph).
procedure TF_DeleteSession(session: PTF_Session; status: PTF_Status); cdecl;
{$EXTERNALSYM TF_DeleteSession}

// Run the graph associated with the session starting with the supplied inputs
// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
//
// Any NULL and non-NULL value combinations for (`run_options`,
// `run_metadata`) are valid.
//
//    - `run_options` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to a `TF_Buffer` containing the
//      serialized representation of a `RunOptions` protocol buffer.
//    - `run_metadata` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to an empty, freshly allocated
//      `TF_Buffer` that may be updated to contain the serialized representation
//      of a `RunMetadata` protocol buffer.
//
// The caller retains ownership of `input_values` (which can be deleted using
// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
// them.
//
// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
// output_values[]. Ownership of the elements of output_values[] is transferred
// to the caller, which must eventually call TF_DeleteTensor on them.
//
// On failure, output_values[] contains NULLs.
procedure TF_SessionRun(
    session: PTF_Session;
    // RunOptions
    const run_options: PTF_Buffer;
    // Input tensors
    const inputs: PTF_Output;
    input_values: PTF_Tensor; ninputs: Integer;
    // Output tensors
    const outputs: PTF_Output;
    output_values: PTF_Tensor; noutputs: Integer;
    // Target operations
    target_opers: PTF_Operation; ntargets: Integer;
    // RunMetadata
    run_metadata: PTF_Buffer;
    // Output status
    status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SessionRun}
// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
// sequence of partial run calls.
//
// On success, returns a handle that is used for subsequent PRun calls. The
// handle should be deleted with TF_DeletePRunHandle when it is no longer
// needed.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
// NOTE: This is EXPERIMENTAL and subject to change.
procedure TF_SessionPRunSetup(
    session: PTF_Session;
    // Input names
    const inputs: PTF_Output; ninputs: Integer;
    // Output names
    const outputs: PTF_Output; noutputs: Integer;
    // Target operations
    target_opers: PTF_Operation; ntargets: Integer;
    // Output handle
    var handle: PTFChar;
    // Output status
    status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SessionPRunSetup}
// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
// NOTE: This is EXPERIMENTAL and subject to change.
procedure TF_SessionPRun(
    session: PTF_Session; const handle: PTFChar;
    // Input tensors
    const inputs: PTF_Output;
    const input_values: PTF_Tensor; ninputs: Integer;
    // Output tensors
    const outputs: PTF_Output;
    const output_values: PTF_Tensor; noutputs: Integer;
    // Target operations
    target_opers: PTF_Operation; ntargets: Integer;
    // Output status
    status: PTF_Status); cdecl;
{$EXTERNALSYM TF_SessionPRun}

// Deletes a handle allocated by TF_SessionPRunSetup.
// Once called, no more calls to TF_SessionPRun should be made.
procedure TF_DeletePRunHandle(const handle: PTFChar); cdecl;
{$EXTERNALSYM TF_DeletePRunHandle}
// -----------------------------------------------------------------------------
// Lists all devices in a TF_Session.
//
// Caller takes ownership of the returned TF_DeviceList* which must eventually
// be freed with a call to TF_DeleteDeviceList.
function  TF_SessionListDevices(session: PTF_Session;
                                status: PTF_Status): PTF_DeviceList; cdecl;
{$EXTERNALSYM TF_SessionListDevices}

// Deallocates the device list.
procedure TF_DeleteDeviceList(list: PTF_DeviceList); cdecl;
{$EXTERNALSYM TF_DeleteDeviceList}

// Counts the number of elements in the device list.
function  TF_DeviceListCount(const list: PTF_DeviceList): Integer; cdecl;
{$EXTERNALSYM TF_DeviceListCount}
// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
// The return value will be a pointer to a null terminated string. The caller
// must not modify or delete the string. It will be deallocated upon a call to
// TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
function  TF_DeviceListName( const list: PTF_DeviceList;
                             idx: Integer; status: PTF_Status): PTFChar; cdecl;
{$EXTERNALSYM TF_DeviceListName}

// Retrieves the type of the device at the given index.
//
// The caller must not modify or delete the string. It will be deallocated upon
// a call to TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
function  TF_DeviceListType( const list: PTF_DeviceList;
                             idx: Integer; status: PTF_Status): PTFChar; cdecl;
{$EXTERNALSYM TF_DeviceListType}

// Retrieve the amount of memory associated with a given device.
//
// If index is out of bounds, an error code will be set in the status object,
// and -1 will be returned.
function  TF_DeviceListMemoryBytes(const list: PTF_DeviceList;
                             idx: Integer; status: PTF_Status): TF_int64_t; cdecl;
{$EXTERNALSYM TF_DeviceListMemoryBytes}
// -----------------------------------------------------------------------------
// Load the library specified by library_filename and register the ops and
// kernels present in that library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
//
// On success, place OK in status and return the newly created library handle.
// The caller owns the library handle.
//
// On failure, place an error status in status and return NULL.
function  TF_LoadLibrary(const library_filename: PTFChar;
                               status: PTF_Status): PTF_Library; cdecl;
{$EXTERNALSYM TF_LoadLibrary}

// Get the OpList of OpDefs defined in the library pointed by lib_handle.
//
// Returns a TF_Buffer. The memory pointed to by the result is owned by
// lib_handle. The data in the buffer will be the serialized OpList proto for
// ops defined in the library.
function  TF_GetOpList(lib_handle: PTF_Library): TF_Buffer; cdecl;
{$EXTERNALSYM TF_GetOpList}

// Frees the memory associated with the library handle.
// Does NOT unload the library.
procedure TF_DeleteLibraryHandle(lib_handle: PTF_Library); cdecl;
{$EXTERNALSYM TF_DeleteLibraryHandle}


// -----------------------------------------------------------------------------
// Get the OpList of all OpDefs defined in this address space.
// Returns a TF_Buffer, ownership of which is transferred to the caller
// (and can be freed using TF_DeleteBuffer).
//
// The data in the buffer will be the serialized OpList proto for ops registered
// in this address space.
function TF_GetAllOpList(): PTF_Buffer; cdecl;
{$EXTERNALSYM TF_GetAllOpList}

// -----------------------------------------------------------------------------
//
// Extension API
function TFEX_AllocGraphDefFromBuffer(graph_def_buf: PTF_Buffer): PTF_GraphDef; cdecl;
{$EXTERNALSYM TFEX_AllocGraphDefFromBuffer}
procedure TFEX_DeleteGraphDef(graph_def: PTF_GraphDef); cdecl;
{$EXTERNALSYM TFEX_DeleteGraphDef}
function  TFEX_AllocMetaGraphDefFromBuffer(metagraph_buf: PTF_Buffer): PTF_MetaGraphDef; cdecl;
{$EXTERNALSYM TFEX_AllocMetaGraphDefFromBuffer}

procedure TFEX_DeleteMetaGraphDef(metagraph_def: PTF_MetaGraphDef); cdecl;
{$EXTERNALSYM TFEX_DeleteMetaGraphDef}
function  TFEX_AllocNodeDefFromBuffer(node_def_buf: PTF_Buffer): PTF_NodeDef; cdecl;
{$EXTERNALSYM TFEX_AllocNodeDefFromBuffer}
procedure TFEX_DeleteNodeDef(op_list: PTF_OpList); cdecl;
{$EXTERNALSYM TFEX_DeleteNodeDef}
function  TFEX_AllocAttrValueFromBuffer(attr_value_buf: PTF_Buffer): PTF_AttrValue; cdecl;
{$EXTERNALSYM TFEX_AllocAttrValueFromBuffer}
procedure TFEX_DeleteAttrValue(attr_value_buf: PTF_Buffer); cdecl;
{$EXTERNALSYM TFEX_DeleteAttrValue}
function TFEX_AllocOpListFromBuffer(op_list_buf: PTF_Buffer): PTF_OpList; cdecl;
{$EXTERNALSYM TFEX_AllocOpListFromBuffer}
procedure TFEX_DeleteOpList(op_list: PTF_OpList); cdecl;
{$EXTERNALSYM TFEX_DeleteOpList}
function TFEX_GetOpListCount(op_list: PTF_OpList): Integer; cdecl;
{$EXTERNALSYM TFEX_GetOpListCount}
function TFEX_GetOpDef(op_list: PTF_OpList; idx: Integer): PTF_OpDef;  cdecl;
{$EXTERNALSYM TFEX_GetOpDef}

function TFEX_GetOpDefName(const op_def: PTF_OpDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefName}

function TFEX_GetOpDefInputArgCount(const op_def: PTF_OpDef): Integer; cdecl;
{$EXTERNALSYM TFEX_GetOpDefInputArgCount}

function TFEX_GetOpDefInputArg(const op_def: PTF_OpDef; idx: Integer): PTF_OpDefArg; cdecl;
{$EXTERNALSYM TFEX_GetOpDefInputArg}

function TFEX_GetOpDefOutputArgCount(const op_def: PTF_OpDef): Integer; cdecl;
{$EXTERNALSYM TFEX_GetOpDefOutputArgCount}

function TFEX_GetOpDefOutputArg(const op_def: PTF_OpDef; idx: Integer): PTF_OpDefArg; cdecl;
{$EXTERNALSYM TFEX_GetOpDefOutputArg}

function TFEX_GetOpDefArgDefName(const arg_def: PTF_OpDefArg): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefArgDefName}

function TFEX_GetOpDefArgDefDescription(const arg_def: PTF_OpDefArg): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefArgDefDescription}

function TFEX_GetOpDefArgDefDataType(const arg_def: PTF_OpDefArg): Int32; cdecl;
{$EXTERNALSYM TFEX_GetOpDefArgDefDataType}

function TFEX_GetOpDefArgDefTypeAttr(const arg_def: PTF_OpDefArg): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefArgDefTypeAttr}

function TFEX_GetOpDefAttrCount(const op_def: PTF_OpDef): Integer; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrCount}

function TFEX_GetOpDefAttr(const op_def: PTF_OpDef; idx: Integer): PTF_OpDefAttr; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttr}

function TFEX_GetOpDefAttrName(op_def_attr: PTF_OpDefAttr): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrName}

function TFEX_GetOpDefAttrDescription(op_def_attr: PTF_OpDefAttr): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrDescription}

function TFEX_GetOpDefAttrMetadata(op_def_attr: PTF_OpDefAttr): PTF_AttrMetadata;  cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrMetadata}

function TFEX_GetOpDefAttrType(op_def_attr: PTF_OpDefAttr): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrType}

function TFEX_GetOpDefAttrDefaultValue(op_def_attr: PTF_OpDefAttr): PTF_AttrValue; cdecl;
{$EXTERNALSYM TFEX_GetOpDefAttrDefaultValue}

function TFEX_AllocAttrValue(oper: PTF_Operation; attr_name: PTFChar;
                           status: PTF_Status): PTF_AttrValue; cdecl;
{$EXTERNALSYM TFEX_AllocAttrValue}

function TFEX_GetAttrValueCase(attr_value: PTF_AttrValue): TF_AttrValueCase; cdecl;
{$EXTERNALSYM TFEX_GetAttrValueCase}
function TFEX_GetAttrValueType(attr_value: PTF_AttrValue): Int32; cdecl;
{$EXTERNALSYM TFEX_GetAttrValueType}
function TFEX_AttrValueHasTensor(attr_value: PTF_AttrValue): Integer; cdecl;
{$EXTERNALSYM TFEX_AttrValueHasTensor}
function TFEX_GetAttrValue_tensor(attr_value: PTF_AttrValue): PTF_Tensor; cdecl;
{$EXTERNALSYM TFEX_GetAttrValue_tensor}
function TFEX_GetAttrValue_s(attr_value: PTF_AttrValue): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetAttrValue_s}
function TFEX_GetAttrValue_i(attr_value: PTF_AttrValue): Int64; cdecl;
{$EXTERNALSYM TFEX_GetAttrValue_i}
function TFEX_GetAttrValue_f(attr_value: PTF_AttrValue): Single; cdecl;
{$EXTERNALSYM TFEX_GetAttrValue_f}
// --------------------------------------------------------------------------
function TFEX_AllocGraphDefFromGraph(graph: PTF_Graph): PTF_GraphDef; cdecl;
{$EXTERNALSYM TFEX_AllocGraphDefFromGraph}
function TFEX_AddNodeDefToGraphDef(graph_def: PTF_GraphDef): PTF_NodeDef; cdecl;
{$EXTERNALSYM TFEX_AddNodeDefToGraphDef}
function TFEX_GetNodeDefsCount(graph_def: PTF_GraphDef): Integer; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefsCount}

function TFEX_GetNodeDef(graph_def: PTF_GraphDef; idx: Integer): PTF_NodeDef; cdecl;
{$EXTERNALSYM TFEX_GetNodeDef}

function TFEX_AllocGraphDefDebugString(graph_def: PTF_GraphDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_AllocGraphDefDebugString}
function TFEX_AllocNodeDefDebugString(node_def: PTF_NodeDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_AllocNodeDefDebugString}
procedure TFEX_DeleteDebugString(debug_str: PTFChar); cdecl;
{$EXTERNALSYM TFEX_DeleteDebugString}

function TFEX_GetNodeDefOp(node_def: PTF_NodeDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefOp}
function TFEX_GetNodeDefName(node_def: PTF_NodeDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefName}
function TFEX_GetNodeDefInputCount(node_def: PTF_NodeDef): Integer; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefInputCount}
function TFEX_GetNodeDefInput(node_def: PTF_NodeDef; idx: Integer): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefInput}
function  TFEX_GetNodeDefAttrMap(node_def: PTF_NodeDef): PTF_AttrMap; cdecl;
{$EXTERNALSYM TFEX_GetNodeDefAttrMap}
function  TFEX_GetAttrMapCount(const map: PTF_AttrMap): Integer; cdecl;
{$EXTERNALSYM TFEX_GetAttrMapCount}

function  TFEX_GetAttrMapAt(const map: PTF_AttrMap; key: PTFChar): PTF_AttrValue; cdecl;
{$EXTERNALSYM TFEX_GetAttrMapAt}
// --------------------------------------------------------------------------
function TFEX_GetSignatureDefMapFromMetaGraphDef(metagraph_def: PTF_MetaGraphDef): PTF_SignatureDefMap; cdecl;
{$EXTERNALSYM TFEX_GetSignatureDefMapFromMetaGraphDef}
function TFEX_GetSignatureDefFromMap(signature_def_map: PTF_SignatureDefMap; key: PTFChar): PTF_SignatureDef; cdecl;
{$EXTERNALSYM TFEX_GetSignatureDefFromMap}
function TFEX_GetInputNameFromSignatureDef(signature_def: PTF_SignatureDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetInputNameFromSignatureDef}
function TFEX_GetOutputNameFromSignatureDef(signature_def: PTF_SignatureDef): PTFChar; cdecl;
{$EXTERNALSYM TFEX_GetOutputNameFromSignatureDef}
// --------------------------------------------------------------------------
function TFEX_TensorIntValCount(const tensor: PTF_Tensor): Integer; cdecl;
{$EXTERNALSYM TFEX_TensorIntValCount}
function TFEX_TensorIntVal(const tensor: PTF_Tensor; const idx: Integer): Integer; cdecl;
{$EXTERNALSYM TFEX_TensorIntVal}
function TFEX_ParseTensorName(const name: PTFChar; first: PTFChar; firstLen: PInteger): Integer; cdecl;
{$EXTERNALSYM TFEX_ParseTensorName}
// --------------------------------------------------------------------------
function TFEX_ColocationAttrName(): PTFChar; cdecl;
{$EXTERNALSYM TFEX_ColocationAttrName}

function TFEX_ColocationGroupPrefix(): PTFChar; cdecl;
{$EXTERNALSYM TFEX_ColocationGroupPrefix}

function TFEX_SavedModelTagServe(): PTFChar; cdecl;
{$EXTERNALSYM TFEX_SavedModelTagServe}

function TFEX_SavedModelTagTrain(): PTFChar; cdecl;
{$EXTERNALSYM TFEX_SavedModelTagTrain}

// --------------------------------------------------------------------------

function TFEX_RegisterOpsForTesting(): Integer; cdecl;
{$EXTERNALSYM TFEX_RegisterOpsForTesting}

// --------------------------------------------------------------------------

function TFEX_SpecialTest1(protbuf: PTFChar; protbuf_len: TF_int64_t;
                           prot_callback: TFEX_ProtCallback): Integer; cdecl;
{$EXTERNALSYM TFEX_SpecialTest1}

function TFEX_SpecialTest2(protbuf: PTFChar; protbuf_len: TF_int64_t;
                           prot_callback: TFEX_ProtCallback): Integer; cdecl;
{$EXTERNALSYM TFEX_SpecialTest2}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------

function TFParseTensorName(const name: TFString; var first: TFString): Integer; cdecl;

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------

implementation

// --------------------------------------------------------------------------

procedure Deallocator_For_TensorDatas(data: Pointer; len: TF_size_t; arg: Pointer);
var
 l_pntBoolean: PBoolean;
begin
 FreeMem(data);
 l_pntBoolean := PBoolean(arg);
 l_pntBoolean^:= True;
end;


function TFParseTensorName(const name: TFString; var first: TFString): Integer;
var
 l_iIdx, l_iStrLng: Integer;
 l_sStrBuf: TFString;
 l_pStrBuf, l_pName: PTFChar;
begin
 SetLength(l_sStrBuf,255);
 l_pStrBuf := @(l_sStrBuf[1]);
 l_pName   := @(name[1]);
 l_iIdx := TFEX_ParseTensorName(l_pName, l_pStrBuf, @l_iStrLng);
 SetLength(l_sStrBuf,l_iStrLng);
 first := l_sStrBuf;
 Result := l_iIdx;
end;

procedure ProtCallback(prot_buf: PTFChar; prot_buf_len: TF_size_t);

var

 l_sProtText: TFString;

begin

 SetLength(l_sProtText,prot_buf_len);
 System.AnsiStrings.StrPCopy(PTFChar(l_sProtText),prot_buf);
end;


{------------------------------- TTensorFlowSetup -------------------------}

function  TF_Version;         external c_sNameOfTensorflowLib name 'TF_Version';
function  TF_DataTypeSize;    external c_sNameOfTensorflowLib name 'TF_DataTypeSize';
function  TF_NewStatus;       external c_sNameOfTensorflowLib name 'TF_NewStatus';
function  TF_GetCode;         external c_sNameOfTensorflowLib name 'TF_GetCode';
function  TF_Message;         external c_sNameOfTensorflowLib name 'TF_Message';
procedure TF_SetStatus;       external c_sNameOfTensorflowLib name 'TF_SetStatus';
procedure TF_DeleteStatus;    external c_sNameOfTensorflowLib name 'TF_DeleteStatus';
function  TF_NewBufferFromString; external c_sNameOfTensorflowLib name 'TF_NewBufferFromString';
function  TF_NewBuffer;       external c_sNameOfTensorflowLib name 'TF_NewBuffer';
procedure TF_DeleteBuffer;    external c_sNameOfTensorflowLib name 'TF_DeleteBuffer';
function  TF_GetBuffer;       external c_sNameOfTensorflowLib name 'TF_GetBuffer';
function  TF_NewTensor;       external c_sNameOfTensorflowLib name 'TF_NewTensor';
function  TF_AllocateTensor;  external c_sNameOfTensorflowLib name 'TF_AllocateTensor';
function  TF_TensorMaybeMove; external c_sNameOfTensorflowLib name 'TF_TensorMaybeMove';
procedure TF_DeleteTensor;    external c_sNameOfTensorflowLib name 'TF_DeleteTensor';
function  TF_TensorType;      external c_sNameOfTensorflowLib name 'TF_TensorType';
function  TF_NumDims;         external c_sNameOfTensorflowLib name 'TF_NumDims';
function  TF_Dim;             external c_sNameOfTensorflowLib name 'TF_Dim';
function  TF_TensorByteSize;  external c_sNameOfTensorflowLib name 'TF_TensorByteSize';
function  TF_TensorData;      external c_sNameOfTensorflowLib name 'TF_TensorData';
//** Modified by Max removed api from tensorflow 2.4  11/12/2021 17:21:00
//function  TF_StringEncode;    external c_sNameOfTensorflowLib name 'TF_StringEncode';
//function  TF_StringDecode;    external c_sNameOfTensorflowLib name 'TF_StringDecode';
procedure TF_StringInit;          external c_sNameOfTensorflowLib name 'TF_StringInit';
procedure TF_StringCopy;          external c_sNameOfTensorflowLib name 'TF_StringCopy';
function  TF_StringGetDataPointer;external c_sNameOfTensorflowLib name 'TF_StringGetDataPointer';
function  TF_StringGetSize;       external c_sNameOfTensorflowLib name 'TF_StringGetSize';
function  TF_StringGetCapacity;   external c_sNameOfTensorflowLib name 'TF_StringGetCapacity';
procedure TF_StringDealloc;       external c_sNameOfTensorflowLib name 'TF_StringDealloc';
function  TF_StringEncodedSize; external c_sNameOfTensorflowLib name 'TF_StringEncodedSize';
function  TF_NewSessionOptions; external c_sNameOfTensorflowLib name 'TF_NewSessionOptions';
procedure TF_SetTarget;       external c_sNameOfTensorflowLib name 'TF_SetTarget';
procedure TF_SetConfig;       external c_sNameOfTensorflowLib name 'TF_SetConfig';
function  TF_CreateConfig;     external c_sNameOfTensorflowLib name 'TF_CreateConfig';
procedure TF_DeleteSessionOptions; external c_sNameOfTensorflowLib name 'TF_DeleteSessionOptions';
function  TF_NewGraph;        external c_sNameOfTensorflowLib name 'TF_NewGraph';
procedure TF_DeleteGraph;     external c_sNameOfTensorflowLib name 'TF_DeleteGraph';
procedure TF_GraphSetTensorShape; external c_sNameOfTensorflowLib name 'TF_GraphSetTensorShape';
function  TF_GraphGetTensorNumDims; external c_sNameOfTensorflowLib name 'TF_GraphGetTensorNumDims';
procedure TF_GraphGetTensorShape; external c_sNameOfTensorflowLib name 'TF_GraphGetTensorShape';
function  TF_NewOperation;    external c_sNameOfTensorflowLib name 'TF_NewOperation';
procedure TF_SetDevice;       external c_sNameOfTensorflowLib name 'TF_SetDevice';
procedure TF_AddInput;        external c_sNameOfTensorflowLib name 'TF_AddInput';
procedure TF_AddInputList;    external c_sNameOfTensorflowLib name 'TF_AddInputList';
procedure TF_AddControlInput; external c_sNameOfTensorflowLib name 'TF_AddControlInput';
procedure TF_ColocateWith;    external c_sNameOfTensorflowLib name 'TF_ColocateWith';
procedure TF_SetAttrString;   external c_sNameOfTensorflowLib name 'TF_SetAttrString';
procedure TF_SetAttrStringList; external c_sNameOfTensorflowLib name 'TF_SetAttrStringList';
procedure TF_SetAttrInt;      external c_sNameOfTensorflowLib name 'TF_SetAttrInt';
procedure TF_SetAttrIntList;  external c_sNameOfTensorflowLib name 'TF_SetAttrIntList';
procedure TF_SetAttrFloat;    external c_sNameOfTensorflowLib name 'TF_SetAttrFloat';
procedure TF_SetAttrFloatList;external c_sNameOfTensorflowLib name 'TF_SetAttrFloatList';
procedure TF_SetAttrBool;     external c_sNameOfTensorflowLib name 'TF_SetAttrBool';
procedure TF_SetAttrBoolList; external c_sNameOfTensorflowLib name 'TF_SetAttrBoolList';
procedure TF_SetAttrType;     external c_sNameOfTensorflowLib name 'TF_SetAttrType';
procedure TF_SetAttrTypeList; external c_sNameOfTensorflowLib name 'TF_SetAttrTypeList';
procedure TF_SetAttrShape; external c_sNameOfTensorflowLib name 'TF_SetAttrShape';
procedure TF_SetAttrShapeList; external c_sNameOfTensorflowLib name 'TF_SetAttrShapeList';
procedure TF_SetAttrTensorShapeProto; external c_sNameOfTensorflowLib name 'TF_SetAttrTensorShapeProto';
procedure TF_SetAttrTensorShapeProtoList; external c_sNameOfTensorflowLib name 'TF_SetAttrTensorShapeProtoList';
procedure TF_SetAttrTensor; external c_sNameOfTensorflowLib name 'TF_SetAttrTensor';
procedure TF_SetAttrTensorList; external c_sNameOfTensorflowLib name 'TF_SetAttrTensorList';
procedure TF_SetAttrValueProto; external c_sNameOfTensorflowLib name 'TF_SetAttrValueProto';
function  TF_FinishOperation; external c_sNameOfTensorflowLib name 'TF_FinishOperation';
function  TF_OperationName; external c_sNameOfTensorflowLib name 'TF_OperationName';
function  TF_OperationOpType; external c_sNameOfTensorflowLib name 'TF_OperationOpType';
function  TF_OperationDevice; external c_sNameOfTensorflowLib name 'TF_OperationDevice';
function  TF_OperationNumOutputs; external c_sNameOfTensorflowLib name 'TF_OperationNumOutputs';
function  TF_OperationOutputType; external c_sNameOfTensorflowLib name 'TF_OperationOutputType';
function  TF_OperationOutputListLength; external c_sNameOfTensorflowLib name 'TF_OperationOutputListLength';
function  TF_OperationNumInputs; external c_sNameOfTensorflowLib name 'TF_OperationNumInputs';
function  TF_OperationInputType;         external c_sNameOfTensorflowLib name 'TF_OperationInputType';
function  TF_OperationInputListLength;   external c_sNameOfTensorflowLib name 'TF_OperationInputListLength';
function  TF_OperationInput;             external c_sNameOfTensorflowLib name 'TF_OperationInput';
function  TF_OperationOutputNumConsumers; external c_sNameOfTensorflowLib name 'TF_OperationOutputNumConsumers';
function  TF_OperationOutputConsumers;   external c_sNameOfTensorflowLib name 'TF_OperationOutputConsumers';
function  TF_OperationNumControlInputs;  external c_sNameOfTensorflowLib name 'TF_OperationNumControlInputs';
function  TF_OperationGetControlInputs;  external c_sNameOfTensorflowLib name 'TF_OperationGetControlInputs';
function  TF_OperationNumControlOutputs; external c_sNameOfTensorflowLib name 'TF_OperationNumControlOutputs';
function  TF_OperationGetControlOutputs; external c_sNameOfTensorflowLib name 'TF_OperationGetControlOutputs';
function  TF_OperationGetAttrMetadata;   external c_sNameOfTensorflowLib name 'TF_OperationGetAttrMetadata';
procedure TF_OperationGetAttrString;     external c_sNameOfTensorflowLib name 'TF_OperationGetAttrString';
procedure TF_OperationGetAttrStringList; external c_sNameOfTensorflowLib name 'TF_OperationGetAttrStringList';
procedure TF_OperationGetAttrInt;        external c_sNameOfTensorflowLib name 'TF_OperationGetAttrInt';
procedure TF_OperationGetAttrIntList;    external c_sNameOfTensorflowLib name 'TF_OperationGetAttrIntList';
procedure TF_OperationGetAttrFloat;      external c_sNameOfTensorflowLib name 'TF_OperationGetAttrFloat';
procedure TF_OperationGetAttrFloatList;  external c_sNameOfTensorflowLib name 'TF_OperationGetAttrFloatList';
procedure TF_OperationGetAttrBool;       external c_sNameOfTensorflowLib name 'TF_OperationGetAttrBool';
procedure TF_OperationGetAttrBoolList;   external c_sNameOfTensorflowLib name 'TF_OperationGetAttrBoolList';
procedure TF_OperationGetAttrType;       external c_sNameOfTensorflowLib name 'TF_OperationGetAttrType';
procedure TF_OperationGetAttrTypeList;   external c_sNameOfTensorflowLib name 'TF_OperationGetAttrTypeList';
procedure TF_OperationGetAttrShape;      external c_sNameOfTensorflowLib name 'TF_OperationGetAttrShape';
procedure TF_OperationGetAttrShapeList;  external c_sNameOfTensorflowLib name 'TF_OperationGetAttrShapeList';
procedure TF_OperationGetAttrTensorShapeProto;external c_sNameOfTensorflowLib name 'TF_OperationGetAttrTensorShapeProto';
procedure TF_OperationGetAttrTensorShapeProtoList;external c_sNameOfTensorflowLib name 'TF_OperationGetAttrTensorShapeProtoList';
procedure TF_OperationGetAttrTensor;     external c_sNameOfTensorflowLib name 'TF_OperationGetAttrTensor';
procedure TF_OperationGetAttrTensorList; external c_sNameOfTensorflowLib name 'TF_OperationGetAttrTensorList';
procedure TF_OperationGetAttrValueProto; external c_sNameOfTensorflowLib name 'TF_OperationGetAttrValueProto';
function  TF_GraphOperationByName;       external c_sNameOfTensorflowLib name 'TF_GraphOperationByName';
function  TF_GraphNextOperation;         external c_sNameOfTensorflowLib name 'TF_GraphNextOperation';
procedure TF_GraphToGraphDef;            external c_sNameOfTensorflowLib name 'TF_GraphToGraphDef';
function  TF_NewImportGraphDefOptions;   external c_sNameOfTensorflowLib name 'TF_NewImportGraphDefOptions';
procedure TF_DeleteImportGraphDefOptions;external c_sNameOfTensorflowLib name 'TF_DeleteImportGraphDefOptions';

procedure TF_ImportGraphDefOptionsSetPrefix;external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsSetPrefix';
procedure TF_ImportGraphDefOptionsAddInputMapping; external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsAddInputMapping';
procedure TF_ImportGraphDefOptionsRemapControlDependency; external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsRemapControlDependency';
procedure TF_ImportGraphDefOptionsAddControlDependency;external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsAddControlDependency';
procedure TF_ImportGraphDefOptionsAddReturnOutput; external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsAddReturnOutput';
function  TF_ImportGraphDefOptionsNumReturnOutputs;external c_sNameOfTensorflowLib name 'TF_ImportGraphDefOptionsNumReturnOutputs';
procedure TF_GraphImportGraphDefWithReturnOutputs; external c_sNameOfTensorflowLib name 'TF_GraphImportGraphDefWithReturnOutputs';
procedure TF_GraphImportGraphDef;        external c_sNameOfTensorflowLib name 'TF_GraphImportGraphDef';
procedure TF_OperationToNodeDef;         external c_sNameOfTensorflowLib name 'TF_OperationToNodeDef';
function  TF_NewWhile;                   external c_sNameOfTensorflowLib name 'TF_NewWhile';
procedure TF_FinishWhile;                external c_sNameOfTensorflowLib name 'TF_FinishWhile';
procedure TF_AbortWhile;                 external c_sNameOfTensorflowLib name 'TF_AbortWhile';
procedure TF_AddGradients;               external c_sNameOfTensorflowLib name 'TF_AddGradients';
function  TF_NewSession;                 external c_sNameOfTensorflowLib name 'TF_NewSession';
function  TF_LoadSessionFromSavedModel;  external c_sNameOfTensorflowLib name 'TF_LoadSessionFromSavedModel';
procedure TF_CloseSession;               external c_sNameOfTensorflowLib name 'TF_CloseSession';
procedure TF_DeleteSession;              external c_sNameOfTensorflowLib name 'TF_DeleteSession';
procedure TF_SessionRun;                 external c_sNameOfTensorflowLib name 'TF_SessionRun';
procedure TF_SessionPRunSetup;           external c_sNameOfTensorflowLib name 'TF_SessionPRunSetup';
procedure TF_SessionPRun;                external c_sNameOfTensorflowLib name 'TF_SessionPRun';
procedure TF_DeletePRunHandle;           external c_sNameOfTensorflowLib name 'TF_DeletePRunHandle';
function  TF_SessionListDevices;         external c_sNameOfTensorflowLib name 'TF_SessionListDevices';
procedure TF_DeleteDeviceList;           external c_sNameOfTensorflowLib name 'TF_DeleteDeviceList';
function  TF_DeviceListCount;            external c_sNameOfTensorflowLib name 'TF_DeviceListCount';
function  TF_DeviceListName;             external c_sNameOfTensorflowLib name 'TF_DeviceListName';
function  TF_DeviceListType;             external c_sNameOfTensorflowLib name 'TF_DeviceListType';
function  TF_DeviceListMemoryBytes;      external c_sNameOfTensorflowLib name 'TF_DeviceListMemoryBytes';

function  TF_LoadLibrary;                external c_sNameOfTensorflowLib name 'TF_LoadLibrary';
function  TF_GetOpList;                  external c_sNameOfTensorflowLib name 'TF_GetOpList';
procedure TF_DeleteLibraryHandle;        external c_sNameOfTensorflowLib name 'TF_DeleteLibraryHandle';

function  TF_GetAllOpList;               external c_sNameOfTensorflowLib name 'TF_GetAllOpList';

function  TFEX_AllocGraphDefFromBuffer;  external c_sNameOfTensorflowLib name 'TFEX_AllocGraphDefFromBuffer';
procedure TFEX_DeleteGraphDef;           external c_sNameOfTensorflowLib name 'TFEX_DeleteGraphDef';
function  TFEX_AllocMetaGraphDefFromBuffer;external c_sNameOfTensorflowLib name 'TFEX_AllocMetaGraphDefFromBuffer';
procedure TFEX_DeleteMetaGraphDef;       external c_sNameOfTensorflowLib name 'TFEX_DeleteMetaGraphDef';
function  TFEX_AllocNodeDefFromBuffer;   external c_sNameOfTensorflowLib name 'TFEX_AllocNodeDefFromBuffer';
procedure TFEX_DeleteNodeDef;            external c_sNameOfTensorflowLib name 'TFEX_DeleteNodeDef';
function  TFEX_AllocAttrValueFromBuffer; external c_sNameOfTensorflowLib name 'TFEX_AllocAttrValueFromBuffer';
procedure TFEX_DeleteAttrValue;          external c_sNameOfTensorflowLib name 'TFEX_DeleteAttrValue';
function  TFEX_AllocOpListFromBuffer;    external c_sNameOfTensorflowLib name 'TFEX_AllocOpListFromBuffer';
procedure TFEX_DeleteOpList;             external c_sNameOfTensorflowLib name 'TFEX_DeleteOpList';
function  TFEX_GetOpListCount;           external c_sNameOfTensorflowLib name 'TFEX_GetOpListCount';

function  TFEX_GetOpDef;                 external c_sNameOfTensorflowLib name 'TFEX_GetOpDef';
function  TFEX_GetOpDefName;             external c_sNameOfTensorflowLib name 'TFEX_GetOpDefName';
function  TFEX_GetOpDefInputArgCount;    external c_sNameOfTensorflowLib name 'TFEX_GetOpDefInputArgCount';
function  TFEX_GetOpDefInputArg;         external c_sNameOfTensorflowLib name 'TFEX_GetOpDefInputArg';
function  TFEX_GetOpDefOutputArgCount;   external c_sNameOfTensorflowLib name 'TFEX_GetOpDefOutputArgCount';
function  TFEX_GetOpDefOutputArg;        external c_sNameOfTensorflowLib name 'TFEX_GetOpDefOutputArg';
function  TFEX_GetOpDefArgDefName;       external c_sNameOfTensorflowLib name 'TFEX_GetOpDefArgDefName';
function  TFEX_GetOpDefArgDefDescription;external c_sNameOfTensorflowLib name 'TFEX_GetOpDefArgDefDescription';
function  TFEX_GetOpDefArgDefDataType;   external c_sNameOfTensorflowLib name 'TFEX_GetOpDefArgDefDataType';
function  TFEX_GetOpDefArgDefTypeAttr;   external c_sNameOfTensorflowLib name 'TFEX_GetOpDefArgDefTypeAttr';
function  TFEX_GetOpDefAttrCount;        external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrCount';
function  TFEX_GetOpDefAttr;             external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttr';
function  TFEX_GetOpDefAttrName;         external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrName';
function  TFEX_GetOpDefAttrDescription;  external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrDescription';
function  TFEX_GetOpDefAttrMetadata;     external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrMetadata';
function  TFEX_GetOpDefAttrType;         external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrType';
function  TFEX_GetOpDefAttrDefaultValue; external c_sNameOfTensorflowLib name 'TFEX_GetOpDefAttrDefaultValue';
function  TFEX_AllocAttrValue;           external c_sNameOfTensorflowLib name 'TFEX_AllocAttrValue';
function  TFEX_GetAttrValueCase;         external c_sNameOfTensorflowLib name 'TFEX_GetAttrValueCase';
function  TFEX_GetAttrValueType;         external c_sNameOfTensorflowLib name 'TFEX_GetAttrValueType';
function  TFEX_AttrValueHasTensor;       external c_sNameOfTensorflowLib name 'TFEX_AttrValueHasTensor';
function  TFEX_GetAttrValue_tensor;      external c_sNameOfTensorflowLib name 'TFEX_GetAttrValue_tensor';
function  TFEX_GetAttrValue_s;           external c_sNameOfTensorflowLib name 'TFEX_GetAttrValue_s';
function  TFEX_GetAttrValue_i;           external c_sNameOfTensorflowLib name 'TFEX_GetAttrValue_i';
function  TFEX_GetAttrValue_f;           external c_sNameOfTensorflowLib name 'TFEX_GetAttrValue_f';
function  TFEX_AllocGraphDefFromGraph;   external c_sNameOfTensorflowLib name 'TFEX_AllocGraphDefFromGraph';
function  TFEX_AddNodeDefToGraphDef;     external c_sNameOfTensorflowLib name 'TFEX_AddNodeDefToGraphDef';
function  TFEX_GetNodeDefsCount;         external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefsCount';
function  TFEX_GetNodeDef;               external c_sNameOfTensorflowLib name 'TFEX_GetNodeDef';

function  TFEX_AllocGraphDefDebugString; external c_sNameOfTensorflowLib name 'TFEX_AllocGraphDefDebugString';
function  TFEX_AllocNodeDefDebugString;  external c_sNameOfTensorflowLib name 'TFEX_AllocNodeDefDebugString';
procedure TFEX_DeleteDebugString;        external c_sNameOfTensorflowLib name 'TFEX_DeleteDebugString';

function  TFEX_GetNodeDefOp;             external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefOp';
function  TFEX_GetNodeDefName;           external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefName';
function  TFEX_GetNodeDefInputCount;     external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefInputCount';
function  TFEX_GetNodeDefInput;          external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefInput';
function  TFEX_GetNodeDefAttrMap;        external c_sNameOfTensorflowLib name 'TFEX_GetNodeDefAttrMap';
function  TFEX_GetAttrMapCount;          external c_sNameOfTensorflowLib name 'TFEX_GetAttrMapCount';
function  TFEX_GetAttrMapAt;             external c_sNameOfTensorflowLib name 'TFEX_GetAttrMapAt';

function  TFEX_GetSignatureDefMapFromMetaGraphDef;external c_sNameOfTensorflowLib name 'TFEX_GetSignatureDefMapFromMetaGraphDef';
function  TFEX_GetSignatureDefFromMap;   external c_sNameOfTensorflowLib name 'TFEX_GetSignatureDefFromMap';
function  TFEX_GetInputNameFromSignatureDef; external c_sNameOfTensorflowLib name 'TFEX_GetInputNameFromSignatureDef';
function  TFEX_GetOutputNameFromSignatureDef;external c_sNameOfTensorflowLib name 'TFEX_GetOutputNameFromSignatureDef';
function  TFEX_TensorIntValCount;        external c_sNameOfTensorflowLib name 'TFEX_TensorIntValCount';
function  TFEX_TensorIntVal;             external c_sNameOfTensorflowLib name 'TFEX_TensorIntVal';
function  TFEX_ParseTensorName;          external c_sNameOfTensorflowLib name 'TFEX_ParseTensorName';

function  TFEX_ColocationAttrName;       external c_sNameOfTensorflowLib name 'TFEX_ColocationAttrName';
function  TFEX_ColocationGroupPrefix;    external c_sNameOfTensorflowLib name 'TFEX_ColocationGroupPrefix';
function  TFEX_SavedModelTagServe;       external c_sNameOfTensorflowLib name 'TFEX_SavedModelTagServe';
function  TFEX_SavedModelTagTrain;       external c_sNameOfTensorflowLib name 'TFEX_SavedModelTagTrain';

function  TFEX_RegisterOpsForTesting;    external c_sNameOfTensorflowLib name 'TFEX_RegisterOpsForTesting';

function  TFEX_SpecialTest1;             external c_sNameOfTensorflowLib name 'TFEX_SpecialTest1';
function  TFEX_SpecialTest2;             external c_sNameOfTensorflowLib name 'TFEX_SpecialTest2';

// --------------------------------------------------------------------------

{ TF_Output }

constructor TF_Output.Create(op: PTF_Operation; idx: Integer);
begin
    oper  := oper;
    index := index;
end;

end.
