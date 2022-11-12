unit TF4D.Core.CApi;
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
{$MINENUMSIZE 4}
interface
   uses Winapi.Windows;
const
  TensorFlowDev = 'cpu';
  TensorFlowVer = '2.11.0';
  TensorFlowLib = 'tensorflow' + '-' + TensorFlowDev + '-win64-' + TensorFlowVer + '.dll';
const
  TF_TSRING_SIZE : Integer = 24;
type
 // --------------------------------------------------------------------------
 // Basistypen
 // --------------------------------------------------------------------------
 TF_size_t     = NativeInt;
 PTF_size_t    = ^TF_size_t;
 TF_int64_t    = Int64;
 PTF_int64_t   = PInt64;
 PPTF_int64_t  = ^PTF_int64_t;
 TFCode        = UINT8;
 PTFChar       = PAnsiChar;
 PInt8         = ^Int8;
 PInt16        = ^Int16;
 PInt32        = ^Int32;
 PTF_DeviceList= Pointer;
// -----------------------------------------------------------------------------
// Helper-Functions / Procedures
procedure Deallocator_For_TensorDatas(data: Pointer; len: TF_size_t; arg: Pointer); cdecl;
{$REGION 'c_api_macros.h'}
type
  TF_Boolean = Byte;
  PTF_Boolean = ^TF_Boolean;
{$ENDREGION}
{$REGION 'tf_attrtype.h'}
type
  /// <summary>
  ///   TF_AttrType describes the type of the value of an attribute on an operation.
  /// </summary>
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
{$ENDREGION}
{$REGION 'tf_datatype.h'}
  /// <summary>
  ///   TF_DataType holds the type for a scalar value. E.g., one slot in a tensor. The
  ///   enum values here are identical to corresponding values in types.proto.
  /// </summary>
  /// <remarks>
  ///   Quantization is the process of transforming deep learning models to use
  ///   parameters and computations at a lower precision. Traditionally, DNN training
  ///   and inference have relied on the IEEE single-precision floating-point format,
  ///   using 32 bits to represent the floating-point model weights and activation
  ///   tensors.
  /// </remarks>
  TF_DataType = (
    DtInvalid = 0,
    TF_FLOAT = 1,        // 32-bit single-precision floating-point.
    TF_DOUBLE = 2,       // 64-bit double-precision floating-point.
    TF_INT32 = 3,        // Int32 tensors are always in 'host' memory.
    TF_UINT8 = 4,        // UInt8
    TF_INT16 = 5,        // Int16
    TF_INT8 = 6,         // Int8
    TF_STRING = 7,       // String
    TF_COMPLEX64 = 8,    // 64-bit single-precision complex.
    TF_COMPLEX = 8,      // Old identifier kept for API backwards compatibility
    TF_INT64 = 9,        // Int64
    TF_BOOL = 10,        // Bool
    TF_QINT8 = 11,       // Quantized int8
    TF_QUINT8 = 12,      // Quantized uint8
    TF_QINT32 = 13,      // Quantized int32
    TF_BFLOAT16 = 14,    // 16-bit truncated floating-point, Float32 truncated to 16 bits.  Only for cast ops.
    TF_QINT16 = 15,      // Quantized int16
    TF_QUINT16 = 16,     // Quantized uint16
    TF_UINT16 = 17,      // UInt16
    TF_COMPLEX128 = 18,  // 128-bit double-precision complex.
    TF_HALF = 19,        // 16-bit half-precision floating-point.
    TF_RESOURCE = 20,    // Handle to a mutable resource.
    TF_VARIANT = 21,     // Values of arbitrary types.
    TF_UINT32 = 22,      // UInt32
    TF_UINT64 = 23,      // UInt64

    // Added of TensorFlow.NET
    DtFloatRef = 101,  // DT_FLOAT_REF
    DtDoubleRef = 102, // DT_DOUBLE_REF
    DtInt32Ref = 103,  // DT_INT32_REF
    DtUint8Ref = 104,
    DtInt16Ref = 105,
    DtInt8Ref = 106,
    DtStringRef = 107,
    DtComplex64Ref = 108,
    DtInt64Ref = 109,  // DT_INT64_REF
    DtBoolRef = 110,
    DtQint8Ref = 111,
    DtQuint8Ref = 112,
    DtQint32Ref = 113,
    DtBfloat16Ref = 114,
    DtQint16Ref = 115,
    DtQuint16Ref = 116,
    DtUint16Ref = 117,
    DtComplex128Ref = 118,
    DtHalfRef = 119,
    DtResourceRef = 120,
    DtVariantRef = 121,
    DtUint32Ref = 122,
    DtUint64Ref = 123
 );
 PTF_DataType = ^TF_DataType;
/// <summary>
///   TF_DataTypeSize returns the sizeof() for the underlying type corresponding
///   to the given TF_DataType enum value. Returns 0 for variable length types
///   (eg. TF_STRING) or on failure.
/// </summary>
function TF_DataTypeSize(dt: TF_DataType): NativeInt;
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'tf_status.h'}
type
  TF_Status = record
  end;
  PTF_Status = ^ TF_Status;
  /// <summary>
  ///   TF_Code holds an error code. The enum values here are identical to
  ///   corresponding values in error_codes.proto.
  /// </summary>
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
/// <summary>
/// Return a new status object.
/// </summary>
function TF_NewStatus: PTF_Status;
  cdecl; external TensorFlowLib;
/// <summary>
///   Delete a previously created status object.
/// </summary>
procedure TF_DeleteStatus(status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Record <code, msg> in *s. Any previous information is lost. A common use is
///   to clear a status: TF_SetStatus(s, TF_OK, "");
/// </summary>
procedure TF_SetStatus(status: PTF_Status; code: TF_Code; const msg: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Convert from an I/O error code (e.g., errno) to a TF_Status value.
///   Any previous information is lost. Prefer to use this instead of TF_SetStatus
///   when the error comes from I/O operations.
/// </summary>
procedure TF_SetStatusFromIOError(status: PTF_Status; error_code: Integer; const context: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Return the code record in *s.
/// </summary>
function TF_GetCode(const status: PTF_Status): TF_Code;
  cdecl; external TensorFlowLib;
/// <summary>
///   Return a pointer to the (null-terminated) error message in *s.  The
///   return value points to memory that is only usable until the next
///   mutation to *s.  Always returns an empty string if TF_GetCode(s) is
///   TF_OK.
/// </summary>
function TF_Message(const status: PTF_Status): PAnsiChar;
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'tf_tensor.h'}
type
  /// <summary>
  ///   TF_Tensor holds a multi-dimensional array of elements of a single data type.
  ///   For all types other than TF_STRING, the data buffer stores elements
  ///   in row major order.  E.g. if data is treated as a vector of TF_DataType:
  ///
  ///     element 0:   index (0, ..., 0)
  ///     element 1:   index (0, ..., 1)
  ///     ...
  ///
  ///   The format for TF_STRING tensors is:
  ///     start_offset: array[uint64]
  ///     data:         byte[...]
  ///
  ///   The string length (as a varint, start_offset[i + 1] - start_offset[i]),
  ///   followed by the contents of the string is encoded at data[start_offset[i]].
  ///   TF_StringEncode and TF_StringDecode facilitate this encoding.
  /// </summary>
  TF_Tensor = record
  end;
  PTF_Tensor = ^TF_Tensor;
  PPTF_Tensor = ^PTF_Tensor;
  TTensorDataDeallocator = procedure(data: Pointer; len: NativeInt; arg: Pointer); cdecl;
/// <summary>
///   Return a new tensor that holds the bytes data[0, len-1]. The data will be
///   deallocated by a subsequent call to TF_DeleteTensor via deallocator. Clients must
///   provide a custom deallocator function so they can pass in memory managed by
///   something like numpy. May return NULL (and invoke the deallocator) if the
///   provided data buffer (data, len) is inconsistent with a tensor of the given
///   TF_DataType and the shape specified by (dima, num_dims).
/// </summary>
function TF_NewTensor(dtype: TF_DataType; const dims: PInt64; num_dims: Integer; data: Pointer;
  len: NativeInt; deallocator: TTensorDataDeallocator; deallocator_arg: Pointer): PTF_Tensor;
  cdecl; external TensorFlowLib;
/// <summary>
///   Allocate and return a new Tensor. This function is an alternative to TF_NewTensor
///   and should be used when memory is allocated to pass the Tensor to the C API. The
///   allocated memory satisfies TensorFlow's memory alignment preferences and should
///   be preferred over calling malloc and free. The caller must set the Tensor values
///   by writing them to the pointer returned by TF_TensorData with length
///   TF_TensorByteSize.
/// </summary>
function TF_AllocateTensor(dtype: TF_DataType; const dims: PInt64; num_dims: Integer; len: NativeInt): PTF_Tensor;
  cdecl; external TensorFlowLib;
/// <summary>
///   Deletes `tensor` and returns a new TF_Tensor with the same content if
///   possible. Returns nullptr and leaves `tensor` untouched if not.
/// </summary>
function TF_TensorMaybeMove(tensor: PTF_Tensor): PTF_Tensor;
  cdecl; external TensorFlowLib;
/// <summary>
///   Destroy a tensor.
/// </summary>
procedure TF_DeleteTensor(tensor: PTF_Tensor);
  cdecl; external TensorFlowLib;
/// <summary>
///   Return the type of a tensor element.
/// </summary>
function TF_TensorType(const tensor: PTF_Tensor): TF_DataType;
  cdecl; external TensorFlowLib;
/// <summary>
///   Return the number of dimensions that the tensor has.
/// </summary>
function TF_NumDims(const tensor: PTF_Tensor): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Return the length of the tensor in the "dim_index" dimension.
///   REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
/// </summary>
function TF_Dim(const tensor: PTF_Tensor; dim_index: Integer): Int64;
  cdecl; external TensorFlowLib;
/// <summary>
///   Return the size of the underlying data in bytes.
/// </summary>
function TF_TensorByteSize(const tensor: PTF_Tensor): NativeInt;
  cdecl; external TensorFlowLib;
/// <summary>
///   Return a pointer to the underlying data buffer.
/// </summary>
function TF_TensorData(const tensor: PTF_Tensor): Pointer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the number of elements in the tensor.
/// </summary>
function TF_TensorElementCount(const tensor: PTF_Tensor): NativeInt;
  cdecl; external TensorFlowLib;
/// <summary>
///   Copy the internal data representation of `from` to `to`. `new_dims` and
///   `num_new_dims` specify the new shape of the `to` tensor, `type` specifies its
///   data type. On success, *status is set to TF_OK and the two tensors share the
///   same data buffer.
///
///   This call requires that the `from` tensor and the given type and shape (dims
///   and num_dims) are "compatible" (i.e. they occupy the same number of bytes).
///   Specifically, given from_type_size = TF_DataTypeSize(TF_TensorType(from)):
///
///   ShapeElementCount(dims, num_dims) * TF_DataTypeSize(type) must equal
///   TF_TensorElementCount(from) * from_type_size
///   where TF_ShapeElementCount would be the number of elements in a tensor with
///   the given shape.
///
///   In addition, this function requires:
///     * TF_DataTypeSize(TF_TensorType(from)) != 0
///     * TF_DataTypeSize(type) != 0
///
///   If any of the requirements are not met, *status is set to TF_INVALID_ARGUMENT.
/// </summary>
procedure TF_TensorBitcastFrom(const src: PTF_Tensor; dtype: TF_DataType; dst: PTF_Tensor;
  const new_dims: PInt64; num_new_dims: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns bool iff this tensor is aligned.
/// </summary>
function TF_TensorIsAligned(const tensor: PTF_Tensor): TF_Boolean;
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'tf_tstring.h'}
type
  TF_TString_Type = (
    /// <summary>
    ///   Small string optimization; the contents of strings less than 22-bytes are
    ///   stored in the TF_TString struct. This avoids any heap allocations.
    /// </summary>
    TF_TSTR_SMALL = 0,
    /// <summary>
    ///   Heap allocated string.
    /// </summary>
    TF_TSTR_LARGE = 1,
    /// <summary>
    ///   (currently unused) An offset defined string. The string buffer begins
    ///   at an internally defined little-endian offset from `str'; i.e.
    ///   GetDataPointer() = str + offset. This type is useful for memory mapping
    ///   or reading string tensors directly from file, without the need to
    ///   deserialize the data. For security reasons, it is imperative that
    ///   OFFSET based string tensors are validated before use, or are from a
    ///   trusted source.
    /// </summary>
    TF_TSTR_OFFSET = 2,
    /// <summary>
    ///   A view into an unowned character string.
    /// </summary>
    TF_TSTR_VIEW = 3
  );
  TF_TString  =  AnsiString;

  TF_TString_Large = record  // NOLINT
     size : size_t;
     cap  : size_t;
     ptr  : PAnsiChar;
  end;
  TF_TString_Offset  = record  // NOLINT
     size  : UInt32;
     Offset: UInt32;
     count : UInt32;
  end;
  TF_TString_View   = record  // NOLINT
     size : size_t;
     ptr  : PAnsiChar;
  end;
  TF_TString_Raw    = record  // NOLINT
     raw : Array[0..23] of Byte;
  end;
  TF_TString_Union  = record  // NOLINT
     case Integer of
       0:( large  : TF_TString_Large);
       1:( offset : TF_TString_Offset);
       2:( view   : TF_TString_View);
       3:( raw    : TF_TString_Raw);
   end;

const   TF_TString_SmallCapacity =  SizeOf(TF_TString_Union) - SizeOf(AnsiChar) - SizeOf(Byte) ;

type
  TF_TString_Small     = record  // NOLINT
    size : UInt8;
    str  : Array[0..TF_TString_SmallCapacity + Sizeof(AnsiChar)-1] of Byte;
  end;
  _TFString  = record  // NOLINT
     case Integer of
       0:( smll   : TF_TString_Small);
       1:( large  : TF_TString_Large);
       2:( offset : TF_TString_Offset);
       3:( view   : TF_TString_View);
       4:( raw    : TF_TString_Raw);
   end;
   TFString  = record
   end;
   PTF_TString = ^_TFString;

/// <summary>
///   Initialize a new tstring.  This must be called before using any function below.
/// </summary>
procedure TF_StringInit(tstr: PTF_TString);
  cdecl; external TensorFlowLib;
/// <summary>
///   Copies `src' to `dst'. `dst' will be an owned type (SMALL/LARGE). `src' should
///   not point to memory owned by `dst'.
/// </summary>
procedure TF_StringCopy(dst: PTF_TString; const src: PByte; size: NativeInt);
  cdecl; external TensorFlowLib;
/// <summary>
///   Sets `dst' as a VIEW type to `src'. `dst' will not take ownership of `src'. It is
///   the user's responsibility to ensure that the lifetime of `src' exceeds `dst'. Any
///   mutations to `dst' via Append, AppendN, or GetMutableDataPointer, will result in
///   a copy into an owned SMALL or LARGE type, and will not modify `src'.
/// </summary>
procedure TF_StringAssignView(dst: PTF_TString; const src: PAnsiChar; size: NativeInt);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns a const char pointer to the start of the underlying string. The
///   underlying character buffer may not be null-terminated.
/// </summary>
function TF_StringGetDataPointer(const tstr: PTF_TString): PAnsiChar;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the underlying type of the tstring.
/// </summary>
/// <remarks>
function TF_StringGetType(const tstr: PTF_TString): TF_TString_Type;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the size of the string.
/// </summary>
function TF_StringGetSize(const tstr: PTF_TString): NativeInt;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the capacity of the string buffer. It should not be considered safe to
///   write in the region between size and capacity - call Resize or
///   ResizeUninitialized before doing so.
/// </summary>
function TF_StringGetCapacity(const tstr: PTF_TString): NativeInt;
  cdecl; external TensorFlowLib;
/// <summary>
///   Deallocate a tstring.
/// </summary>
procedure TF_StringDealloc(tstr: PTF_TString);
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'c_api.h'}
/// <summary>
///   Returns a string describing version information of the TensorFlow library.
///   TensorFlow using semantic versioning.
/// </summary>
function TF_Version: PAnsiChar;
  cdecl; external TensorFlowLib;
type
  TBufferDataDeallocator = procedure(data: Pointer; length: NativeInt); cdecl;
  /// <summary>
  ///   TF_Buffer holds a pointer to a block of data and its associated length.
  ///   Typically, the data consists of a serialized protocol buffer, but other data
  ///   may also be held in a buffer. By default, TF_Buffer itself does not do any
  ///   memory management of the pointed-to block. If need be, users of this struct
  ///   should specify how to deallocate the block by setting the `data_deallocator`
  ///   function pointer.
  /// </summary>
  TF_Buffer = record
    data: Pointer;
    length: NativeInt;
    data_deallocator: TBufferDataDeallocator;
  end;
  PTF_Buffer = ^TF_Buffer;
  PPTF_Buffer = ^PTF_Buffer;
/// <summary>
///   Makes a copy of the input and sets an appropriate deallocator. Useful for passing
///   in read-only, input protobufs.
/// </summary>
function TF_NewBufferFromString(const proto: Pointer; proto_len: NativeInt): PTF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Creates a new buffer, useful for passing 'out' a protobuf.
/// </summary>
function TF_NewBuffer: PTF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Deletes the buffer.
/// </summary>
procedure TF_DeleteBuffer(buffer: PTF_Buffer);
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the buffer.
/// </summary>
function TF_GetBuffer(buffer: PTF_Buffer): TF_Buffer;
  cdecl; external TensorFlowLib;
type
  /// <summary>
  ///   Used to return strings across the C API. The caller does not take ownership
  ///   of the underlying data pointer and is not responsible for freeing it.
  /// </summary>
  TF_StringView = record
    data: PAnsiChar;
    len: NativeInt;
  end;
  /// <summary>
  ///   TF_SessionOptions holds options that can be passed during session creation.
  /// </summary>
  TF_SessionOptions = record
  end;
  PTF_SessionOptions = ^TF_SessionOptions;
/// <summary>
///   Return a new options object.
/// </summary>
function TF_NewSessionOptions: PTF_SessionOptions;
  cdecl; external TensorFlowLib;
/// <summary>
///   Set the target in TF_SessionOptions.options; target can be empty, a single entry,
///   or a comma separated list of entries. Each entry is in one of the following
///   formats: "local", ip:port, host:port.
/// </summary>
procedure TF_SetTarget(options: PTF_SessionOptions; const target: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set the config in TF_SessionOptions.options; config should be a serialized
///   tensorflow.ConfigProto proto. If config was not parsed successfully as a
///   ConfigProto, record the error information in *status.
/// </summary>
procedure TF_SetConfig(options: PTF_SessionOptions; const proto: Pointer; proto_len: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Destroy an options object.
/// </summary>
procedure TF_DeleteSessionOptions(options: PTF_SessionOptions);
  cdecl; external TensorFlowLib;
{$REGION 'Graph Construction API, still under development'}
type
  /// <summary>
  ///   Represents a computation graph. Graphs may be shared between sessions. Graphs
  ///   are thread-safe when used as directed below.
  /// </summary>
  TF_Graph = record
  end;
  PTF_Graph = ^TF_Graph;
  /// <summary>
  ///   Operation that has been added to the graph. Valid until the graph is deleted -
  ///   in particular adding a new operation to the graph does not invalidate old
  ///   TF_Operation* pointers.
  /// </summary>
  TF_Operation = record
  end;
  PTF_Operation = ^TF_Operation;
  PPTF_Operation = ^PTF_Operation;
  TF_OperationDescription = record
  end;
  /// <summary>
  ///   Operation being built. The underlying graph must outlive this.
  /// </summary>
  PTF_OperationDescription = ^TF_OperationDescription;
  /// <summary>
  ///   Represents a specific input of an operation.
  /// </summary>
  TF_Input = record
    oper: PTF_Operation;
    index: Integer;

    constructor Create(oper: PTF_Operation; index: Integer);
  end;
  PTF_Input = ^TF_Input;
  /// <summary>
  ///   Represents a specific output of an operation.
  /// </summary>
  TF_Output = record
    oper: PTF_Operation;
    index: Integer;

    constructor Create(oper: PTF_Operation; index: Integer);
  end;
  PTF_Output = ^TF_Output;
  TF_AttrMetadata = record
    is_list: TF_Boolean;
    list_size: Int64;
    &type: TF_AttrType;
    total_size: Int64;
  end;
  PTF_AttrMetadata = ^TF_AttrMetadata;
  /// <summary>
  ///   TF_ImportGraphDefOptions holds options that can be passed to
  ///   TF_GraphImportGraphDef.
  /// </summary>
  TF_ImportGraphDefOptions = record
  end;
  PTF_ImportGraphDefOptions = ^TF_ImportGraphDefOptions;
  /// <summary>
  ///   TF_ImportGraphDefResults holds results that are generated by
  ///   TF_GraphImportGraphDefWithResults().
  /// </summary>
  TF_ImportGraphDefResults = record
  end;
  PTF_ImportGraphDefResults = ^TF_ImportGraphDefResults;
  /// <summary>
  ///   TF_Function is a grouping of operations with defined inputs and outputs. Once
  ///   created and added to graphs, functions can be invoked by creating an <br />
  ///   operation whose operation type matches the function name.
  /// </summary>
  TF_Function = record
  end;
  PTF_Function = ^TF_Function;
  PPTF_Function = ^PTF_Function;
  /// <summary>
  ///   Function definition options.
  /// </summary>
  TF_FunctionOptions = record
  end;
  PTF_FunctionOptions = ^TF_FunctionOptions;
  TF_WhileParams = record
    /// <summary>
    ///   The number of inputs to the while loop, i.e. the number of loop variables.
    ///   This is the size of cond_inputs, body_inputs, and body_outputs.
    /// </summary>
    ninputs: Integer;
    /// <summary>
    ///   The while condition graph. The inputs are the current values of the loop
    ///   variables. The output should be a scalar boolean.
    /// </summary>
    cond_graph: PTF_Graph;
    cond_inputs: PTF_Input;
    cond_output: TF_Output;
    /// <summary>
    ///   The loop body graph. The inputs are the current values of the loop variables.
    ///   The outputs are the updated values of the loop variables.
    /// </summary>
    body_graph: PTF_Graph;
    body_inputs: PTF_Output;
    body_outputs: PTF_Output;
    /// <summary>
    ///   Unique null-terminated name for this while loop. This is used as a prefix for
    ///   created operations.
    /// </summary>
    name: PAnsiChar;
  end;
  PTF_WhileParams = ^TF_WhileParams;
/// <summary>
///   Return a new graph object.
/// </summary>
function TF_NewGraph: PTF_Graph;
  cdecl; external TensorFlowLib;
// Destroy an options object.  Graph will be deleted once no more
// TFSession's are referencing it.
procedure TF_DeleteGraph(graph: PTF_Graph);
   cdecl; external TensorFlowLib;

/// <summary>
///   Sets the shape of the Tensor referenced by `output` in `graph` to the shape
///   described by `dims` and `num_dims`. If the number of dimensions is unknown,
///   `num_dims` must be set to -1 and `dims` can be null. If a dimension is unknown,
///   the corresponding entry in the `dims` array must be -1. This does not overwrite
///   the existing shape associated with `output`, but merges the input shape with the
///   existing shape. For example, setting a shape of [-1, 2] with an existing shape
///   [2, -1] would set a final shape of [2, 2] based on shape merging semantics. <br />
/// </summary>
/// <param name="status">
///   Returns an error into `status` if: `output` is not in `graph`, or an
///   invalid shape is being set (e.g., the shape being set is incompatible with the
///   existing shape).
/// </param>
procedure TF_GraphSetTensorShape(graph: PTF_Graph; output: TF_Output; const dims: PInt64;
  const num_dims: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the number of dimensions of the Tensor referenced by `output` in `graph`.
///   If the number of dimensions in the shape is unknown, returns -1. Returns an error
///   into `status` if `output` is not in `graph`.
/// </summary>
function TF_GraphGetTensorNumDims(graph: PTF_Graph; output: TF_output; status: PTF_Status): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the shape of the Tensor referenced by `output` in `graph` into `dims`.
///   `dims` must be an array large enough to hold `num_dims` entries (e.g., the return
///   value of TF_GraphGetTensorNumDims). If the number of dimensions in the shape is
///   unknown or the shape is a scalar, `dims` will remain untouched. Otherwise,
///   each element of `dims` will be set corresponding to the size of the dimension. An
///   unknown dimension is represented by `-1`. Returns an error into `status` if
///   `output` is not in `graph`, or `num_dims` does not match the actual number of
///   dimensions.
/// </summary>
procedure TF_GraphGetTensorShape(graph: PTF_Graph; output: TF_Output; const dims: PInt64;
  num_dims: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Operation will only be added to *graph when TF_FinishOperation is called
///   (assuming TF_FinishOperation() does not return an error). graph must not be
///   deleted until after TF_FinishOperation is called.
/// </summary>
function TF_NewOperation(graph: PTF_Graph; const op_type: PAnsiChar; const oper_name: PAnsiChar): PTF_OperationDescription;
  cdecl; external TensorFlowLib;
/// <summary>
///   Specify the device for `desc`. Defaults to empty, meaning unconstrained
/// </summary>
procedure TF_SetDevice(desc: PTF_OperationDescription; const device: PAnsiChar);
  cdecl; external TensorFlowLib;
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
/// <summary>
///   For inputs that take a single tensor.
/// </summary>
procedure TF_AddInput(desc: PTF_OperationDescription; input: TF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   For inputs that take a list of tensors. inputs must point to
///   TF_Output[num_inputs].
/// </summary>
procedure TF_AddInputList(desc: PTF_OperationDescription; const inputs: PTF_Output; num_inputs: Integer);
  cdecl; external TensorFlowLib;
/// <summary>
///   Call once per control input to `desc`.
/// </summary>
procedure TF_AddControlInput(desc: PTF_OperationDescription; input: PTF_Operation);
  cdecl; external TensorFlowLib;
/// <summary>
///   Request that `desc` be co-located on the device where `op` is placed.
/// </summary>
/// <remarks>
///   Use of this is discouraged since the implementation of device placement is
///   subject to change. Primarily intended for internal libraries.
/// </remarks>
procedure TF_ColocateWith(desc: PTF_OperationDescription; op: PTF_Operation);
  cdecl; external TensorFlowLib;
// Call some TF_SetAttr*() function for every attr that is not inferred from an input
// and doesn't have a default value you wish to keep.
procedure TF_SetAttrString(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const value: Pointer; length: NativeInt);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrStringList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const values: PPointer; const lengths: PNativeInt; num_values: Integer);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrInt(desc: PTF_OperationDescription; const attr_name: PAnsiChar; value: Int64);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrIntList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const values: PInt64; num_values: Integer);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrFloat(desc: PTF_OperationDescription; const attr_name: PAnsiChar; value: Single);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrFloatList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const values: PSingle; num_values: Integer);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrBool(desc: PTF_OperationDescription; const attr_name: PAnsiChar; value: TF_Boolean);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrBoolList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const values: PTF_Boolean; num_values: Integer);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrType(desc: PTF_OperationDescription; const attr_name: PAnsiChar; value: TF_DataType);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrTypeList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const values: PTF_DataType; num_values: Integer);
  cdecl; external TensorFlowLib;
procedure TF_SetAttrPlaceholder(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const placeholder: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set a 'func' attribute to the specified name. `value` must point to a string of
///   length `length` bytes.
/// </summary>
procedure TF_SetAttrFuncName(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const value: PAnsiChar; length: NativeInt);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set `num_dims` to -1 to represent "unknown rank". Otherwise, `dims` points to an
///   array of length `num_dims`. `dims[i]` must be &gt;= -1, with -1 meaning "unknown
///   dimension".
/// </summary>
procedure TF_SetAttrShape(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const dims: PInt64; num_dims: Integer);
  cdecl; external TensorFlowLib;
type
  PPInt64 = ^PInt64;
/// <summary>
///   `dims` and `num_dims` must point to arrays of length `num_shapes`. Set
///   `num_dims[i]` to -1 to represent "unknown rank". Otherwise, `dims[i]` points to
///   an array of length `num_dims[i]`. `dims[i][j]` must be &gt;= -1, with -1 meaning
///   "unknown dimension".
/// </summary>
procedure TF_SetAttrShapeList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const dims: PPInt64; const num_dims: PInt64; num_shapes: Integer);
  cdecl; external TensorFlowLib;
/// <summary>
///   `proto` must point to an array of `proto_len` bytes representing a
///   binary-serialized TensorShapeProto.
/// </summary>
procedure TF_SetAttrTensorShapeProto(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const proto: Pointer; proto_len: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   `protos` and `proto_lens` must point to arrays of length `num_shapes`.
///   `protos[i]` must point to an array of `proto_lens[i]` bytes representing a
///   binary-serialized TensorShapeProto.
/// </summary>
procedure TF_SetAttrTensorShapeProtoList(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const protos: PPointer; const proto_lens: PNativeInt; num_shapes: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   `proto` should point to a sequence of bytes of length `proto_len` representing a
///   binary serialization of an AttrValue protocol buffer.
/// </summary>
procedure TF_SetAttrProto(desc: PTF_OperationDescription; const attr_name: PAnsiChar;
  const proto: Pointer; proto_len: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
procedure TF_SetAttrValueProto(desc: PTF_OperationDescription; const attr_name: PTFChar; const proto: Pointer; proto_len: TF_Size_t;
                               status: PTF_Status); cdecl; external TensorFlowLib;
/// <summary>
///   If this function succeeds:
///   <list type="bullet">
///     <item>
///       status is set to an OK value,
///     </item>
///     <item>
///       a TF_Operation is added to the graph,
///     </item>
///     <item>
///       a non-null value pointing to the added operation is returned -- this
///       value is valid until the underlying graph is deleted.
///     </item>
///   </list>
///   Otherwise:
///   <list type="bullet">
///     <item>
///       status is set to a non-OK value
///     </item>
///     <item>
///       the graph is not modified,
///     </item>
///     <item>
///       a null value is returned. <br />
///     </item>
///   </list>
///   In either case, it deletes `desc`.
/// </summary>
function TF_FinishOperation(desc: PTF_OperationDescription; status: PTF_Status): PTF_Operation;
  cdecl; external TensorFlowLib;
// TF_Operation functions. Operations are immutable once created, so
// these are all query functions.
function TF_OperationName(oper: PTF_Operation): PAnsiChar;
  cdecl; external TensorFlowLib;
function TF_OperationOpType(oper: PTF_Operation): PAnsiChar;
  cdecl; external TensorFlowLib;
function TF_OperationDevice(oper: PTF_Operation): PAnsiChar;
  cdecl; external TensorFlowLib;
function TF_OperationNumOutputs(oper: PTF_Operation): Integer;
  cdecl; external TensorFlowLib;
function TF_OperationOutputType(oper: TF_Output): TF_DataType;
  cdecl; external TensorFlowLib;
function TF_OperationOutputListLength(oper: PTF_Operation; const arg_name: PAnsiChar; status: PTF_Status): Integer;
  cdecl; external TensorFlowLib;
function TF_OperationNumInputs(oper: PTF_Operation): Integer;
  cdecl; external TensorFlowLib;
function TF_OperationInputType(oper: TF_Input): TF_DataType;
  cdecl; external TensorFlowLib;
function TF_OperationInputListLength(oper: PTF_Operation; const arg_name: PAnsiChar; status: PTF_Status): Integer;
  cdecl; external TensorFlowLib;

/// <summary>
///   In this code: TF_Output producer = TF_OperationInput(consumer). There is an edge
///   from producer.oper's output (given by producer.index) to consumer.oper's input
///   (given by consumer.index).
/// </summary>
function TF_OperationInput(oper_in: TF_Input): TF_Output;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get list of all inputs of a specific operation. `inputs` must point to an array
///   of length at least `max_inputs` (ideally set to TF_OperationNumInputs(oper)).
///   Beware that a concurrent modification of the graph can increase the number of
///   inputs of an operation.
/// </summary>
procedure TF_OperationAllInputs(oper: PTF_Operation; inputs: PTF_Output; max_inputs: Integer);
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the number of current consumers of a specific output of an operation. Note
///   that this number can change when new operations are added to the graph.
/// </summary>
function TF_OperationOutputNumConsumers(oper_out: TF_Output): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get list of all current consumers of a specific output of an length at least
///   `max_consumers` (ideally set to TF_OperationOutputNumConsumers(oper_out)).
///   Beware that a concurrent modification of the graph can increase the number of
///   consumers of an operation. Returns the number of output consumers (should match
///   TF_OperationOutputNumConsumers(oper_out)).
/// </summary>
function TF_OperationOutputConsumers(oper_out: TF_Output; consumers: PTF_Input;
  max_consumers: Integer): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the number of control inputs to an operation.
/// </summary>
function TF_OperationNumControlInputs(oper: PTF_Operation): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get list of all control inputs to an operation. `control_inputs` must point to an
///   array of length `max_control_inputs` (ideally set to TF_OperationNumControlInputs(oper)).
///   Returns the number of control inputs (should match TF_OperationNumControlInputs(oper)).
/// </summary>
function TF_OperationGetControlInputs(oper: PTF_Operation; control_inputs: PPTF_Operation;
  max_control_inputs: Integer): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the number of operations that have `*oper` as a control input. Note that this
///   number can change when new operations are added to the graph.
/// </summary>
function TF_OperationNumControlOutputs(oper: PTF_Operation): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the list of operations that have `*oper` as a control input.
///   `control_outputs` must point to an array of length at least `max_control_outputs`
///   (ideally set to TF_OperationNumControlOutputs(oper)). Beware that a
///   concurrent modification of the graph can increase the number of control outputs.
///   Returns the number of control outputs (should match TF_OperationNumControlOutputs(oper)).
/// </summary>
function TF_OperationGetControlOutputs(oper: PTF_Operation; control_outputs: PPTF_Operation;
  max_control_outputs: Integer): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns metadata about the value of the attribute `attr_name` of `oper`.
/// </summary>
function TF_OperationGetAttrMetadata(oper: PTF_Operation; const attr_name: PAnsiChar;
  status: PTF_Status): TF_AttrMetadata;
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `value` with the value of the attribute `attr_name`. `value` must point
///   to an array of length at least `max_length` (ideally set to
///   TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrString(oper: PTF_Operation; const attr_name: PAnsiChar;
  value: Pointer; max_length: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Get the list of strings in the value of the attribute `attr_name`. Fills in
///     `values` and `lengths`, each of which must point to an array of length at
///     least `max_values`.
///   </para>
///   <para>
///     The elements of values will point to addresses in `storage` which must be at
///     least `storage_size` bytes in length. Ideally, max_values would be set to <br />
///     TF_AttrMetadata.list_size and `storage` would be at least
///     TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
///     attr_name).
///   </para>
///   <para>
///     Fails if storage_size is too small to hold the requested number of strings.
///   </para>
/// </summary>
procedure TF_OperationGetAttrStringList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PPointer; lengths: PNativeInt; max_Values: Integer;
  storage: Pointer; storage_size: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
procedure TF_OperationGetAttrInt(oper: PTF_Operation; const attr_name: PAnsiChar;
  out value: Int64; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `values` with the value of the attribute `attr_name` of `oper`. `values`
///   must point to an array of length at least `max_values` (ideally set
///   TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrIntList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PInt64; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
procedure TF_OperationGetAttrFloat(oper: PTF_Operation; const attr_name: PAnsiChar;
  out value: Single; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `values` with the value of the attribute `attr_name` of `oper`. `values`
///   must point to an array of length at least `max_values` (ideally set
///   TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrFLoatList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PSingle; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
procedure TF_OperationGetAttrBool(oper: PTF_Operation; const attr_name: PAnsiChar;
  out value: TF_Boolean; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `values` with the value of the attribute `attr_name` of `oper`. `values`
///   must point to an array of length at least `max_values` (ideally set
///   TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrBoolList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PTF_Boolean; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
procedure TF_OperationGetAttrType(oper: PTF_Operation; const attr_name: PAnsiChar;
  out value: TF_DataType; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `values` with the value of the attribute `attr_name` of `oper`. `values`
///   must point to an array of length at least `max_values` (ideally set
///   TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrTypeList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PTF_DataType; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `value` with the value of the attribute `attr_name` of `oper`. `values`
///   must point to an array of length at least `num_dims` (ideally set to
///   TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrShape(oper: PTF_Operation; const attr_name: PAnsiChar;
  value: PInt64; num_dims: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Fills in `dims` with the list of shapes in the attribute `attr_name` of
///     `oper` and `num_dims` with the corresponding number of dimensions. On return,
///     for every i where `num_dims[i]` &gt; 0, `dims[i]` will be an array of
///     `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
///     i-th shape in the list is unknown.
///   </para>
///   <para>
///     The elements of `dims` will point to addresses in `storage` which must be
///     large enough to hold at least `storage_size` int64_ts. Ideally, `num_shapes`
///     would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
///     TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper, attr_name).
///   </para>
///   <para>
///     Fails if storage_size is insufficient to hold the requested shapes.
///   </para>
/// </summary>
procedure TF_OperationGetAttrShapeList(oper: PTF_Operation; const attr_name: PAnsiChar;
  dims: PPInt64; num_dims: PInteger; num_shapes: Integer;
  storage: PInt64; storage_size: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Sets `value` to the binary-serialized TensorShapeProto of the value of
///   `attr_name` attribute of `oper`'.
/// </summary>
procedure TF_OperationGetAttrTensorShapeProto(oper: PTF_Operation; const attr_name: PAnsiChar;
  value: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fills in `values` with binary-serialized TensorShapeProto values of the attribute
///   `attr_name` of `oper`. `values` must point to an array of length at least
///   `num_values` (ideally set to TF_AttrMetadata.list_size from
///   TF_OperationGetAttrMetadata(oper, attr_name)).
/// </summary>
procedure TF_OperationGetAttrTensorShapeProtoList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PPTF_Buffer; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Gets the TF_Tensor valued attribute of `attr_name` of `oper`. Allocates a new
///   TF_Tensor which the caller is expected to take ownership of (and can deallocate
///   using TF_DeleteTensor).
/// </summary>
procedure TF_OperationGetAttrTensor(oper: PTF_Operation; const attr_name: PAnsiChar;
  out value: PTF_Tensor; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
///     `oper`. `values` must point to an array of TF_Tensor* of length at least
///     `max_values` (ideally set to TF_AttrMetadata.list_size from
///     TF_OperationGetAttrMetadata(oper, attr_name)).
///   </para>
///   <para>
///     The caller takes ownership of all the non-null TF_Tensor* entries in `values`
///     (which can be deleted using TF_DeleteTensor(values[i])).
///   </para>
/// </summary>
procedure TF_OperationGetAttrTensorList(oper: PTF_Operation; const attr_name: PAnsiChar;
  values: PPTF_Tensor; max_Values: Integer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Sets `output_attr_value` to the binary-serialized AttrValue proto representation
///   of the value of the `attr_name` attr of `oper`.
/// </summary>
procedure TF_OperationGetAttrValueProto(oper: PTF_Operation; const attr_name: PAnsiChar;
  output_attr_value: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the operation in the graph with `oper_name`. Returns nullptr if no
///   operation found.
/// </summary>
function TF_GraphOperationByName(graph: PTF_Graph; const oper_name: PAnsiChar): PTF_Operation;
  cdecl; external TensorFlowLib;
/// <summary>
///   Iterate through the operations of a graph.
/// </summary>
/// <example>
///   <code lang="Delphi">var pos: NativeInt := 0;
/// ver oper: PTF_Operation := TF_GraphNextOperation(graph, pos);
/// while Assigned(oper) do
/// begin
///   DoSomethingWithOperation(oper);
///   oper := TF_GraphNextOperation(graph, pos);
/// end;</code>
/// </example>
function TF_GraphNextOperation(graph: PTF_Graph; out pos: NativeInt): PTF_Operation;
  cdecl; external TensorFlowLib;
/// <summary>
///   Write out a serialized representation of `graph` (as a GraphDef protocol message)
///   to `output_graph_def` (allocated by TF_NewBuffer()). `output_graph_def`'s
///   underlying buffer will be freed when TF_DeleteBuffer() is called. May fail on
///   very large graphs in the future.
/// </summary>
procedure TF_GraphToGraphDef(graph: PTF_Graph; output_graph_def: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the serialized OpDef proto with name `op_name`, or a bad status if no
///   such op exists. This can return OpDefs of functions copied into the graph.
/// </summary>
procedure TF_GraphGetOpDef(graph: PTF_Graph; const op_name: PAnsiChar; output_graph_def: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the serialized VersionDef proto for this graph.
/// </summary>
procedure TF_GraphVersions(graph: PTF_Graph; output_graph_def: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
function TF_NewImportGraphDefOptions: PTF_ImportGraphDefOptions;
  cdecl; external TensorFlowLib;
procedure TF_DeleteImportGraphDefOptions(opts: PTF_ImportGraphDefOptions);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set the prefix to be prepended to the names of nodes in `graph_def` that will be
///   imported into `graph`. `prefix` is copied and has no lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsSetPrefix(opts: PTF_ImportGraphDefOptions; const prefix: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set the execution device for nodes in `graph_def`. Only applies to nodes
///   where a device was not already explicitly specified. `device` is copied and has
///   no lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsSetDefaultDevice(opts: PTF_ImportGraphDefOptions; const device: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set whether to uniquify imported operation names. If true, imported operation
///   names will be modified if their name already exists in the graph. If false,
///   conflicting names will be treated as an error. Note that this option has no
///   effect if a prefix is set, since the prefix will guarantee all names are
///   unique. Defaults to false.
/// </summary>
procedure TF_ImportGraphDefOptionsSetUniquifyNames(opts: PTF_ImportGraphDefOptions; uniquify_names: TF_Boolean);
  cdecl; external TensorFlowLib;
/// <summary>
///   If true, the specified prefix will be modified if it already exists as an
///   operation name or prefix in the graph. If false, a conflicting prefix will be
///   treated as an error. This option has no effect if no prefix is specified.
/// </summary>
procedure TF_ImportGraphDefOptionsSetUniquifyPrefix(opts: PTF_ImportGraphDefOptions; uniquify_prefix: TF_Boolean);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set any imported nodes with input `src_name:src_index` to have that input
///   replaced with `dst`. `src_name` refers to a node in the graph to be imported,
///   `dst` references a node already existing in the graph being imported into.
///   `src_name` is copied and has no lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsAddInputMapping(opts: PTF_ImportGraphDefOptions; const src_name: PAnsiChar;
  src_index: Integer; dst: TF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   Set any imported nodes with control input `src_name` to have that input
///   replaced with `dst`. `src_name` refers to a node in the graph to be imported,
///   `dst` references an operation already existing in the graph being imported
///   into. `src_name` is copied and has no lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsRemapControlDependency(opts: PTF_ImportGraphDefOptions;
  const src_name: PAnsiChar; dst: PTF_Operation);
  cdecl; external TensorFlowLib;
/// <summary>
///   Cause the imported graph to have a control dependency on `oper`. `oper`
///   should exist in the graph being imported into.
/// </summary>
procedure TF_ImportGraphDefOptionsAddControlDependency(opts: PTF_ImportGraphDefOptions; oper: PTF_Operation);
  cdecl; external TensorFlowLib;
/// <summary>
///   Add an output in `graph_def` to be returned via the `return_outputs` output
///   parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
///   mapping, the corresponding existing tensor in `graph` will be returned.
///   `oper_name` is copied and has no lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsAddReturnOutput(opts: PTF_ImportGraphDefOptions;
  const oper_name: PAnsiChar; index: Integer);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the number of return outputs added via TF_ImportGraphDefOptionsAddReturnOutput().
/// </summary>
procedure TF_ImportGraphDefOptionsNumReturnOutputs(opts: PTF_ImportGraphDefOptions);
  cdecl; external TensorFlowLib;
/// <summary>
///   Add an operation in `graph_def` to be returned via the `return_opers` output
///   parameter of TF_GraphImportGraphDef(). `oper_name` is copied and has no
///   lifetime requirements.
/// </summary>
procedure TF_ImportGraphDefOptionsAddReturnOperation(opts: PTF_ImportGraphDefOptions; const oper_name: PAnsiChar);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the number of return operations added via TF_ImportGraphDefOptionsAddReturnOperation().
/// </summary>
procedure TF_ImportGraphDefOptionsNumReturnOperations(opts: PTF_ImportGraphDefOptions);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fetches the return outputs requested via TF_ImportGraphDefOptionsAddReturnOutput().
///   The number of fetched outputs is returned in `num_outputs`. The array of return
///   outputs is returned in `outputs`. `*outputs` is owned by and has the lifetime of `results`.
/// </summary>
procedure TF_ImportGraphDefResultsReturnOutputs(results: PTF_ImportGraphDefResults;
  out num_outputs: Integer; out outputs: PTF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   Fetches the return operations requested via TF_ImportGraphDefOptionsAddReturnOperation().
///   The number of fetched operations is returned in `num_opers`. The array of return operations is
///   returned in `opers`. `*opers` is owned by and has the lifetime of `results`
/// </summary>
procedure TF_ImportGraphDefResultsReturnOperations(results: PTF_ImportGraphDefResults;
  out num_opers: Integer; out opers: PPTF_Operation);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Fetches any input mappings requested via
///     TF_ImportGraphDefOptionsAddInputMapping() that didn't appear in the GraphDef
///     and weren't used as input to any node in the imported graph def.
///   </para>
///   <para>
///     The number of fetched mappings is returned in
///     `num_missing_unused_input_mappings`. The array of each mapping's source node
///     name is returned in `src_names`, and the array of each mapping's source index
///     is returned in `src_indexes`.
///   </para>
///   <para>
///     `*src_names`, `*src_indexes`, and the memory backing each string in
///     `src_names` are owned by and have the lifetime of `results`.
///   </para>
/// </summary>
procedure TF_ImportGraphDefResultsMissingUnusedInputMappings(results: PTF_ImportGraphDefResults;
  out num_missing_unused_input_mapppings: Integer; out src_names: PPAnsiChar; out src_indexes: PInteger);
  cdecl; external TensorFlowLib;
/// <summary>
///   Deletes a results object returned by TF_GraphImportGraphDefWithResults().
/// </summary>
procedure TF_DeleteImportGraphDefResults(results: PTF_ImportGraphDefResults);
  cdecl; external TensorFlowLib;
/// <summary>
///   Import the graph serialized in `graph_def` into `graph`. Returns nullptr and
///   a bad status on error. Otherwise, returns a populated
///   TF_ImportGraphDefResults instance. The returned instance must be deleted via
///   TF_DeleteImportGraphDefResults().
/// </summary>
function TF_GraphImportGraphDefWithResults(graph: PTF_Graph; const graph_Def: PTF_Buffer;
  const options: PTF_ImportGraphDefOptions; status: PTF_Status): PTF_ImportGraphDefResults;
  cdecl; external TensorFlowLib;
/// <summary>
///   Import the graph serialized in `graph_def` into `graph`. Convenience
///   function for when only return outputs are needed.
///   `num_return_outputs` must be the number of return outputs added (i.e. the
///   result of TF_ImportGraphDefOptionsNumReturnOutputs()). If
///   `num_return_outputs` is non-zero, `return_outputs` must be of length
///   `num_return_outputs`. Otherwise it can be null.
/// </summary>
function TF_GraphImportGraphDefWithReturnOutputs(graph: PTF_Graph; const graph_Def: PTF_Buffer;
  const options: PTF_ImportGraphDefOptions; return_outputs: PTF_output;
  num_return_outputs: Integer; status: PTF_Status): PTF_ImportGraphDefResults;
  cdecl; external TensorFlowLib;
/// <summary>
///   Import the graph serialized in `graph_def` into `graph`.Convenience
///   function for when no results are needed.
/// </summary>
procedure TF_GraphImportGraphDef(graph: PTF_Graph; const graph_Def: PTF_Buffer;
  const options: PTF_ImportGraphDefOptions; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Adds a copy of function `func` and optionally its gradient function `grad` to
///     `g`. Once `func`/`grad` is added to `g`, it can be called by creating an
///     operation using the function's name. Any changes to `func`/`grad` (including
///     deleting it) done after this method returns, won't affect the copy of
///     `func`/`grad` in `g`.
///   </para>
///   <para>
///     If `func` or `grad` are already in `g`, TF_GraphCopyFunction has no effect on
///     them, but can establish the function-&gt;gradient relationship between them
///     if `func` does not already have a gradient.
///   </para>
///   <para>
///     If `func` already has a gradient different from `grad`, an error is returned.
///     `func` must not be null. If `grad` is null and `func` is not in `g`, `func`
///     is added without a gradient.
///   </para>
///   <para>
///     If `grad` is null and `func` is in `g`, TF_GraphCopyFunction is a noop.
///     `grad` must have appropriate signature as described in the doc of GradientDef
///     in tensorflow/core/framework/function.proto.
///   </para>
///   <para>
///     If successful, status is set to OK and `func` and `grad` are added to `g`.
///     Otherwise, status is set to the encountered error and `g` is unmodified.
///   </para>
/// </summary>
procedure TF_GraphCopyFunction(g: PTF_Graph; const func: PTF_Function; const grad: PTF_Function; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the number of TF_Functions registered in `g`.
/// </summary>
function TF_GraphNumFunctions(graph: PTF_Graph): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Fills in `funcs` with the TF_Function* registered in `g`. `funcs` must point
///     to an array of TF_Function* of length at least `max_func`.
///   </para>
///   <para>
///     In usual usage, max_func should be set to the result of
///     TF_GraphNumFunctions(g). In this case, all the functions registered in `g`
///     will be returned. Else, an unspecified subset.
///   </para>
///   <para>
///     If successful, returns the number of TF_Function* successfully set in `funcs`
///     and sets status to OK.
///   </para>
///   <para>
///     The caller takes ownership of all the returned TF_Functions. They must be
///     deleted with TF_DeleteFunction. On error, returns 0, sets status to the
///     encountered error, and the contents of funcs will be undefined.
///   </para>
/// </summary>
function TF_GraphGetFunctions(graph: PTF_Graph; funcs: PPTF_Function; max_func: Integer; status: PTF_Status): Integer;
  cdecl; external TensorFlowLib;
/// <summary>
///   May fail on very large protos in the future.
/// </summary>
procedure TF_OperationToNodeDef(oper: PTF_Operation; out_node_def: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
///     outputs that already exist in `g` used as initial values for the loop
///     variables.
///   </para>
///   <para>
///     The returned TF_WhileParams will have all fields initialized except
///     `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
///     allocated to size `ninputs`. The caller should build `cond_graph` and
///     `body_graph` starting from the inputs, and store the final outputs in
///     `cond_output` and `body_outputs`.
///   </para>
///   <para>
///     If `status` is OK, the caller must call either TF_FinishWhile or
///     TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
///     returned TF_WhileParams is not valid, and the caller should not call
///     TF_FinishWhile() or TF_AbortWhile().
///   </para>
///   <para>
///     Missing functionality (TODO):
///   </para>
///   <list type="bullet">
///     <item>
///       Gradients
///     </item>
///     <item>
///       Reference-type inputs
///     </item>
///     <item>
///       Directly referencing external tensors from the cond/body graphs (this is
///       possible in the Python API)
///     </item>
///   </list>
/// </summary>
function TF_NewWhile(g: PTF_Graph; inputs: PTF_Output; ninputs: Integer; status: PTF_Status): TF_WhileParams;
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Builds the while loop specified by `params` and returns the output tensors of
///     the while loop in `outputs`. `outputs` should be allocated to size
///     `params.ninputs`.
///   </para>
///   <para>
///     `params` is no longer valid once this returns.
///   </para>
///   <para>
///     Either this or TF_AbortWhile() must be called after a successful
///     TF_NewWhile() call.
///   </para>
/// </summary>
procedure TF_FinishWhile(const params: PTF_WhileParams; status: PTF_Status; outputs: PTF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   Frees `params`s resources without building a while loop. `params` is no longer
///   valid after this returns. Either this or TF_FinishWhile() must be called after a
///   successful TF_NewWhile() call.
/// </summary>
procedure TF_AbortWhile(const params: PTF_WhileParams);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
///     i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
///   </para>
///   <para>
///     `dx` are used as initial gradients (which represent the symbolic partial
///     derivatives of some loss function `L` w.r.t. `y`). `dx` must be nullptr or
///     have size `ny`.
///   </para>
///   <para>
///     If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
///     shapes in `y`.
///   </para>
///   <para>
///     The partial derivatives are returned in `dy`. `dy` should be allocated to
///     size `nx`.
///   </para>
///   <para>
///     Gradient nodes are automatically named under the "gradients/" prefix. To
///     guarantee name uniqueness, subsequent calls to the same graph will append an
///     incremental tag to the prefix: "gradients_1/", "gradients_2/", ... See
///     TF_AddGradientsWithPrefix, which provides a means to specify a custom name
///     prefix for operations added to a graph to compute the gradients.
///   </para>
///   <para>
///     WARNING: This function does not yet support all the gradients that python
///     supports. See <see href="https:www.tensorflow.org/code/tensorflow/cc/gradients/README.md" />
///      for instructions on how to add C++ more gradients.
///   </para>
/// </summary>
procedure TF_AddGradients(g: PTF_Graph; y: PTF_Output; ny: Integer; x: PTF_Output; nx: Integer; dx: PTF_Output;
  status: PTF_Status; dy: PTF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
///     i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2... This is a variant
///     of TF_AddGradients that allows to caller to pass a custom name prefix to the
///     operations added to a graph to compute the gradients.
///   </para>
///   <para>
///     `dx` are used as initial gradients (which represent the symbolic partial
///     derivatives of some loss function `L` w.r.t. `y`). `dx` must be nullptr or
///     have size `ny`. If `dx` is nullptr, the implementation will use dx of
///     `OnesLike` for all shapes in `y`. The partial derivatives are returned in
///     `dy`. `dy` should be allocated to size `nx`.
///   </para>
///   <para>
///     `prefix` names the scope into which all gradients operations are being added.
///     `prefix` must be unique within the provided graph otherwise this operation
///     will fail. If `prefix` is nullptr, the default prefixing behaviour takes
///     place, see TF_AddGradients for more details.
///   </para>
///   <para>
///     WARNING: This function does not yet support all the gradients that python
///     supports. See <see href="https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md" />
///     for instructions on how to add C++ more gradients.
///   </para>
/// </summary>
procedure TF_AddGradientsWithPrefix(g: PTF_Graph; const prefix: PAnsiChar; y: PTF_Output; ny: Integer; x: PTF_Output;
  nx: Integer; dx: PTF_Output; status: PTF_Status; dy: PTF_Output);
  cdecl; external TensorFlowLib;
/// <summary>
///   Create a TF_Function from a TF_Graph
/// </summary>
/// <param name="fn_body">
///   The graph whose operations (or subset of whose operations) will be converted to
///   TF_Function.
/// </param>
/// <param name="fn_name">
///   The name of the new TF_Function. Should match the operation name (OpDef.name)
///   regexp [A-Z][A-Za-z0-9_.\\-/]*. If append_hash_to_fn_name` is false, `fn_name`
///   must be distinct from other function and operation names (at least those
///   registered in graphs where this function will be used).
/// </param>
/// <param name="append_hash_to_fn_name">
///   Must be 0 or 1. If set to 1, the actual name of the function will be `fn_name`
///   appended with '_&lt;hash_of_this_function's_definition&gt;'. If set to 0, the
///   function's name will be `fn_name`.
/// </param>
/// <param name="num_opers">
///   `num_opers` contains the number of elements in the `opers` array or a special
///   value of -1 meaning that no array is given. The distinction between an empty
///   array of operations and no array of operations is necessary to distinguish the
///   case of creating a function with no body (e.g. identity or permutation) and the
///   case of creating a function whose body contains all the nodes in the graph
///   (except for the automatic skipping, see below).
/// </param>
/// <param name="opers">
///   Array of operations to become the body of the function or null.
///   <list type="bullet">
///     <item>
///       If no array is given (`num_opers` = -1), all the operations in `fn_body`
///       will become part of the function except operations referenced in
///       `inputs`. These operations must have a single output (these operations
///       are typically placeholders created for the sole purpose of representing
///       an input. We can relax this constraint if there are compelling use
///       cases).
///     </item>
///     <item>
///       If an array is given (`num_opers` &gt;= 0), all operations in it will
///       become part of the function. In particular, no automatic skipping of
///       dummy input operations is performed.
///     </item>
///   </list>
/// </param>
/// <param name="ninputs">
///   Number of elements in `inputs` array
/// </param>
/// <param name="inputs">
///   Array of TF_Outputs that specify the inputs to the function. If `ninputs` is zero
///   (the function takes no inputs), `inputs` can be null. The names used for function
///   inputs are normalized names of the operations (usually placeholders) pointed to
///   by `inputs`. These operation names should start with a letter. Normalization will
///   convert all letters to lowercase and non-alphanumeric characters to '_' to make
///   resulting names match the "[a-z][a-z0-9_]*" pattern for operation argument names.
///   `inputs` cannot contain the same tensor twice.
/// </param>
/// <param name="noutputs">
///   Number of elements in `outputs` array
/// </param>
/// <param name="outputs">
///   Array of TF_Outputs that specify the outputs of the function. If `noutputs` is
///   zero (the function returns no outputs), `outputs` can be null. `outputs` can
///   contain the same tensor more than once.
/// </param>
/// <param name="output_names">
///   The names of the function's outputs. `output_names` array must either have the
///   same length as `outputs` (i.e. `noutputs`) or be null. In the former case, <br />
///   the names should match the regular expression for ArgDef names -
///   "[a-z][a-z0-9_]*". In the latter case, names for outputs will be generated
///   automatically.
/// </param>
/// <param name="opts">
///   Various options for the function, e.g. XLA's inlining control.
/// </param>
/// <param name="description">
///   Optional human-readable description of this function. status -
/// </param>
/// <param name="status">
///   Set to OK on success and an appropriate error on failure
/// </param>
/// <returns>
///   On success, a newly created TF_Function instance. It must be deleted by calling
///   TF_DeleteFunction. On failure, null.
/// </returns>
/// <remarks>
///   Note that when the same TF_Output is listed as both an input and an output, the
///   corresponding function's output will equal to this input, instead of the original
///   node's output. Callers must also satisfy the following constraints: <br /><list type="bullet">
///     <item>
///       `inputs` cannot refer to TF_Outputs within a control flow context. For
///       example, one cannot use the output of "switch" node as input.
///     </item>
///     <item>
///       `inputs` and `outputs` cannot have reference types. Reference types are
///       not exposed through C API and are being replaced with Resources. We
///       support reference types inside function's body to support legacy code. Do
///       not use them in new code.
///     </item>
///     <item>
///       Every node in the function's body must have all of its inputs (including
///       control inputs). In other words, for every node in the body, each input
///       must be either listed in `inputs` or must come from another node in the
///       body. In articular, it is an error to have a control edge going from a
///       node outside of the body into a node in the body. This applies to control
///       edges going from nodes referenced in `inputs` to nodes in the body when
///       the former nodes are not in the body (automatically skipped or not
///       included in explicitly specified body).
///     </item>
///   </list>
/// </remarks>
function TF_GraphToFunction(
  const fn_body: PTF_Graph;
  const fn_name: PAnsiChar;
  append_hash_to_fn_name: TF_Boolean;
  num_opers: Integer;
  const opers: PPTF_Operation;
  ninputs: Integer;
  const inputs: PTF_Output;
  noutputs: Integer;
  outputs: PTF_Output;
  const output_names: PPAnsiChar;
  const opts: PTF_FunctionOptions;
  const description: PAnsiChar;
  status: PTF_Status): PTF_Function;
  cdecl; external TensorFlowLib;
/// <summary>
///   Similar to TF_GraphToFunction but allows specifying control outputs of the
///   function.
/// </summary>
/// <param name="ncontrol_outputs">
///   Number of control outputs of the function.
/// </param>
/// <param name="control_outputs">
///   Vector of TF_Operation objects to be marked as control outputs of the function.
///   Operations marked as control outputs are guaranteed to execute.
/// </param>
/// <param name="control_output_names">
///   Optional. If not nullptr, vector of strings, one per control output, with their
///   names to be added to the function's OpDef.
/// </param>
function TF_GraphToFunctionWithControlOutputs(
  const fn_body: PTF_Graph;
  const fn_name: PAnsiChar;
  append_hash_to_fn_name: TF_Boolean;
  num_opers: Integer;
  const opers: PPTF_Operation;
  ninputs: Integer;
  const inputs: PTF_Output;
  noutputs: Integer;
  outputs: PTF_Output;
  const output_names: PPAnsiChar;
  ncontrol_outputs: Integer;
  const control_outputs: PPTF_Operation;
  const control_output_names: PPAnsiChar;
  const opts: PTF_FunctionOptions;
  const description: PAnsiChar;
  status: PTF_Status): PTF_Function;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the name of the graph function. The return value points to memory that is
///   only usable until the next mutation to *func.
/// </summary>
function TF_FunctionName(func: PTF_Function): PAnsiChar;
  cdecl; external TensorFlowLib;
/// <summary>
///   Write out a serialized representation of `func` (as a FunctionDef protocol
///   message) to `output_func_def` (allocated by TF_NewBuffer()). `output_func_def`'s
///   underlying buffer will be freed when TF_DeleteBuffer() is called. May fail on
///   very large graphs in the future.
/// </summary>
procedure TF_FunctionToFunctionDef(func: PTF_Function; output_func_def: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Construct and return the function whose FunctionDef representation is serialized
///   in `proto`. `proto_len` must equal the number of bytes pointed to by `proto`.
///   Returns: On success, a newly created TF_Function instance. It must be deleted by
///   calling TF_DeleteFunction. On failure, null.
/// </summary>
function TF_FunctionImportFunctionDef(const proto: Pointer; proto_len: NativeInt; status: PTF_Status): PTF_Function;
  cdecl; external TensorFlowLib;
/// <summary>
///   Sets function attribute named `attr_name` to value stored in `proto`. If this
///   attribute is already set to another value, it is overridden. `proto` should point
///   to a sequence of bytes of length `proto_len` representing a binary serialization
///   of an AttrValue protocol buffer.
/// </summary>
procedure TF_FunctionSetAttrValueProto(func: PTF_Function; const attr_name: PAnsiChar;
  const proto: Pointer; proto_len: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Sets `output_attr_value` to the binary-serialized AttrValue proto representation
///   of the value of the `attr_name` attr of `func`. If `attr_name` attribute is not
///   present, status is set to an error.
/// </summary>
procedure TF_FunctionGetAttrValueProto(func: PTF_Function; const attr_name: PAnsiChar;
  output_attr_value: PTF_Buffer; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Frees the memory used by the `func` struct. TF_DeleteFunction is a noop if `func`
///   is null. Deleting a function does not remove it from any graphs it was copied to.
/// </summary>
procedure TF_DeleteFunction(func: PTF_Function);
  cdecl; external TensorFlowLib;
/// <summary>
///   <para>
///     Attempts to evaluate `output`. This will only be possible if `output` doesn't
///     depend on any graph inputs (this function is safe to call if this isn't the
///     case though).
///   </para>
///   <para>
///     If the evaluation is successful, this function returns true and `output`s
///     value is returned in `result`. Otherwise returns false. An error status is
///     returned if something is wrong with the graph or input. Note that this may
///     return false even if no error status is set.
///   </para>
/// </summary>
function TF_TryEvaluateConstant(graph: PTF_Graph; output: TF_Output; result: PPTF_Tensor; status: PTF_Status): TF_Boolean;
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'API for driving Graph execution'}
type
  TF_Session = record
  end;
  PTF_Session = ^TF_Session;
/// <summary>
///   Return a new execution session with the associated graph, or NULL on error. Does
///   not take ownership of any input parameters. `graph` must be a valid graph (not
///   deleted or nullptr). `graph` will be kept alive for the lifetime of the returned
///   TF_Session. New nodes can still be added to `graph` after this call.
/// </summary>
function TF_NewSession(graph: PTF_Graph; const options: PTF_SessionOptions; status: PTF_Status): PTF_Session;
  cdecl; external TensorFlowLib;
/// <summary>
///   This function creates a new TF_Session (which is created on success) using
///   `session_options`, and then initializes state (restoring tensors and other
///   assets) using `run_options`. Any NULL and non-NULL value combinations for
///   (`run_options, `meta_graph_def`) are valid. If successful, populates `graph` with
///   the contents of the Graph and `meta_graph_def` with the MetaGraphDef of the
///   loaded model.
/// </summary>
/// <param name="export_dir">
///   Must be set to the path of the exported SavedModel
/// </param>
/// <param name="tags">
///   Must include the set of tags used to identify one MetaGraphDef in the SavedModel
/// </param>
/// <param name="graph">
///   Must be a graph newly allocated with TF_NewGraph().
/// </param>
function TF_LoadSessionFromSavedModel(
  const session_options: PTF_SessionOptions;
  const run_options: PTF_Buffer;
  const export_dir: PAnsiChar;
  const tags: PPAnsiChar;
  tags_len: Integer;
  graph: PTF_Graph;
  meta_graph_def: PTF_Buffer;
  status: PTF_Status
  ): PTF_Session;
  cdecl; external TensorFlowLib;
/// <summary>
///   Close a session. Contacts any other processes associated with the session, if
///   applicable. May not be called after TF_DeleteSession().
/// </summary>
procedure TF_CloseSession(session: PTF_Session; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Destroy a session object. Even if error information is recorded in *status, this
///   call discards all local resources associated with the session. The session may
///   not be used during or after this call (and the session drops its reference to the
///   corresponding graph).
/// </summary>
procedure TF_DeleteSession(session: PTF_Session; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Run the graph associated with the session starting with the supplied inputs
///   (inputs[0, ninputs-1] with corresponding values in input_values[0, ninputs-1]).
///   Any NULL and non-NULL value combinations for (`run_options`,`run_metadata`) are
///   valid. On success, the tensors corresponding to outputs[0, noutputs-1] are placed
///   in output_values[]. Ownership of the elements of output_values[] is transferred
///   to the caller, which must eventually call TF_DeleteTensor on them. On failure,
///   output_values[] contains NULLs.
/// </summary>
/// <param name="run_options">
///   Run options is owned by the caller, and may be NULL, in which case it will be
///   ignored; or non-NULL, in which case it must point to a `TF_Buffer` containing the
///   serialized representation of a `RunOptions` protocol buffer.
/// </param>
/// <param name="input_values">
///   The caller retains ownership of `input_values` (which can be deleted using
///   TF_DeleteTensor).
/// </param>
/// <param name="run_metadata">
///   Owned by the caller and may be NULL, in which case it will be ignored; or
///   non-NULL, in which case it must point to an empty, freshly allocated `TF_Buffer`
///   that may be updated to contain the serialized representation of a `RunMetadata`
///   protocol buffer
/// </param>
procedure TF_SessionRun(
  session: PTF_Session;
  // RunOptions
  const run_options: PTF_Buffer;
  // Input tensors
  const inputs: PTF_Output; const input_values: PTF_Tensor; ninputs: Integer;
  // Output tensors
  const outputs: PTF_Output; output_values: PTF_Tensor; noutputs: Integer;
  // Target operations
  const target_opers: PTF_Operation; ntargets: Integer;
  // RunMetadata
  run_metadata: PTF_Buffer;
  // Output status
  status: PTF_Status
  );
  cdecl; external TensorFlowLib;
/// <summary>
///   Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
///   sequence of partial run calls. On success, returns a handle that is used for
///   subsequent PRun calls. The handle should be deleted with TF_DeletePRunHandle when
///   it is no longer needed. On failure, out_status contains a tensorflow::Status with
///   an error message. *handle is set to nullptr.
/// </summary>
procedure TF_SessionPRun(
  session: PTF_Session;
  const handle: PAnsiChar;
  const inputs: PTF_Output;
  const input_values: PPTF_Tensor;
  ninputs: Integer;
  const outputs: PTF_Output;
  output_values: PPTF_Tensor;
  noutputs: Integer;
  const target_opers: PPTF_Operation;
  ntargets: Integer;
  status: PTF_Status
  );
  cdecl; external TensorFlowLib;
/// <summary>
///   Continue to run the graph with additional feeds and fetches. The execution state
///   is uniquely identified by the handle.
/// </summary>
procedure TF_DeletePRunHandle(const handle: PAnsiChar);
  cdecl; external TensorFlowLib;
{$ENDREGION}

// -----------------------------------------------------------------------------
// Lists all devices in a TF_Session.
//
// Caller takes ownership of the returned TF_DeviceList* which must eventually
// be freed with a call to TF_DeleteDeviceList.
function  TF_SessionListDevices(session: PTF_Session; status: PTF_Status): PTF_DeviceList;
   cdecl; external TensorFlowLib;

// Deallocates the device list.
procedure TF_DeleteDeviceList(list: PTF_DeviceList);
   cdecl; external TensorFlowLib;

// Counts the number of elements in the device list.
function  TF_DeviceListCount(const list: PTF_DeviceList): Integer;
   cdecl; external TensorFlowLib;

// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
// The return value will be a pointer to a null terminated string. The caller
// must not modify or delete the string. It will be deallocated upon a call to
// TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
function  TF_DeviceListName( const list: PTF_DeviceList; idx: Integer; status: PTF_Status): PTFChar;
   cdecl; external TensorFlowLib;

// Retrieves the type of the device at the given index.
//
// The caller must not modify or delete the string. It will be deallocated upon
// a call to TF_DeleteDeviceList.
//
// If index is out of bounds, an error code will be set in the status object,
// and a null pointer will be returned.
function  TF_DeviceListType( const list: PTF_DeviceList; idx: Integer; status: PTF_Status): PTFChar;
   cdecl; external TensorFlowLib;

// Retrieve the amount of memory associated with a given device.
//
// If index is out of bounds, an error code will be set in the status object,
// and -1 will be returned.
function  TF_DeviceListMemoryBytes(const list: PTF_DeviceList; idx: Integer; status: PTF_Status): TF_int64_t;
    cdecl; external TensorFlowLib;

function TF_OperationOutputConsumers_wrapper(oper_out: TF_Output): TArray<String>;

{$REGION 'Load plugins containing custom ops and kernels'}
type
  /// <summary>
  ///   TF_Library holds information about dynamically loaded TensorFlow plugins.
  /// </summary>
  TF_Library = record
  end;
  PTF_Library = ^TF_Library;
  /// <summary>
  ///   TF_ApiDefMap encapsulates a collection of API definitions for an operation.
  ///   This object maps the name of a TensorFlow operation to a description of the API
  ///   to generate for it, as defined by the ApiDef protocol buffer.
  ///   (<see href="https://www.tensorflow.org/code/tensorflow/core/framework/api_def.proto"/>).
  ///   The ApiDef messages are typically used to generate convenience wrapper
  ///   functions for TensorFlow operations in various language bindings.
  /// </summary>
  TF_ApiDefMap = record
  end;
  PTF_ApiDefMap = ^TF_ApiDefMap;
/// <summary>
///   Load the library specified by library_filename and register the ops and kernels
///   present in that library. Pass "library_filename" to a platform-specific
///   mechanism for dynamically loading a library. The rules for determining the exact
///   location of the library are platform-specific and are not documented here. On
///   success, place OK in status and return the newly created library handle. The
///   caller owns the library handle. On failure, place an error status in status and
///   return NULL.
/// </summary>
function TF_LoadLibrary(const library_filename: PAnsiChar; status: PTF_Status): PTF_Library;
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the OpList of OpDefs defined in the library pointed by lib_handle. Returns a
///   TF_Buffer. The memory pointed to by the result is owned by lib_handle. The data
///   in the buffer will be the serialized OpList proto for ops defined in the library.
/// </summary>
function TF_GetOpList(lib_handle: PTF_Library): TF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Frees the memory associated with the library handle. Does NOT unload the library.
/// </summary>
procedure TF_DeleteLibraryHandle(lib_handle: PTF_Library);
  cdecl; external TensorFlowLib;
/// <summary>
///   Get the OpList of all OpDefs defined in this address space. Returns a TF_Buffer,
///   ownership of which is transferred to the caller (and can be freed using
///   TF_DeleteBuffer). The data in the buffer will be the serialized OpList proto for
///   ops registered in this address space.
/// </summary>
function TF_GetAllOpList: PTF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Creates a new TF_ApiDefMap instance.
/// </summary>
/// <param name="op_list_buffer">
///   TF_Buffer instance containing serialized OpList protocol buffer. (See
///   https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto <br />for
///   the OpList proto definition).
/// </param>
/// <param name="status">
///   Set to OK on success and an appropriate error on failure.
/// </param>
function TF_NewApiDefMap(op_list_buffer: PTF_Buffer; status: PTF_Status): PTF_ApiDefMap;
  cdecl; external TensorFlowLib;
/// <summary>
///   Deallocates a TF_ApiDefMap.
/// </summary>
procedure TF_DeleteApiDefMap(apimap: PTF_ApiDefMap);
  cdecl; external TensorFlowLib;
/// <summary>
///   Add ApiDefs to the map. The provided ApiDefs will be merged with existing ones in
///   the map, with precedence given to the newly added version in case of conflicts
///   with previous calls to TF_ApiDefMapPut
/// </summary>
/// <param name="text">
///   Corresponds to a text representation of an ApiDefs protocol message.
///   (https://www.tensorflow.org/code/tensorflow/core/framework/api_def.proto).
/// </param>
procedure TF_ApiDefMapPut(api_def_map: PTF_ApiDefMap; const text: PAnsiChar; text_len: NativeInt; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns a serialized ApiDef protocol buffer for the TensorFlow operation named
///   `name`.
/// </summary>
function TF_ApiDefMapGet(api_def_map: PTF_ApiDefMap; const name: PAnsiChar; name_len: NativeInt; status: PTF_Status): PTF_Buffer;
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'Kernal definition information'}
/// <summary>
///   Returns a serialized KernelList protocol buffer containing KernelDefs for all
///   registered kernels.
/// </summary>
function TF_GetAllRegisteredKernels(status: PTF_Status): PTF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns a serialized KernelList protocol buffer containing KernelDefs for all
///   kernels registered for the operation named `name`.
/// </summary>
function TF_GetRegisteredKernelsForOp(const name: PAnsiChar; status: PTF_Status): PTF_Buffer;
  cdecl; external TensorFlowLib;
/// <summary>
///   Update edge, switch input/output in a node
/// </summary>
procedure TF_UpdateEdge(graph: PTF_Graph; new_src: TF_Output; dst: TF_Input; status: PTF_Status);
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$REGION 'In-process TensorFlow server functionality'}
// In-process TensorFlow server functionality, for use in distributed training.
// A Server instance encapsulates a set of devices and a Session target that
// can participate in distributed training. A server belongs to a cluster
// (specified by a ClusterSpec), and corresponds to a particular task in a
// named job. The server can communicate with any other server in the same
// cluster.
type
  /// <summary>
  ///   In-process TensorFlow server.
  /// </summary>
  TF_Server = record
  end;
  PTF_Server = ^TF_Server;
  TLogListener = procedure(const msg: PAnsiChar);
/// <summary>
///   Creates a new in-process TensorFlow server configured using a serialized
///   ServerDef protocol buffer provided via `proto` and `proto_len`. The server will
///   not serve any requests until TF_ServerStart is invoked. The server will stop
///   serving requests once TF_ServerStop or TF_DeleteServer is invoked.
/// </summary>
function TF_NewServer(const proto: Pointer; proto_len: NativeInt; status: PTF_Status): PTF_Server;
  cdecl; external TensorFlowLib;
/// <summary>
///   Starts an in-process TensorFlow server.
/// </summary>
procedure TF_ServerStart(server: PTF_Server; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Stops an in-process TensorFlow server.
/// </summary>
procedure TF_ServerStop(server: PTF_Server; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Blocks until the server has been successfully stopped (via TF_ServerStop or
///   TF_ServerClose).
/// </summary>
procedure TF_ServerJoin(server: PTF_Server; status: PTF_Status);
  cdecl; external TensorFlowLib;
/// <summary>
///   Returns the target string that can be provided to TF_SetTarget() to connect a
///   TF_Session to `server`. The returned string is valid only until TF_DeleteServer
///   is invoked.
/// </summary>
function TF_ServerTarget(server: PTF_Server): PAnsiChar;
  cdecl; external TensorFlowLib;
/// <summary>
///   Destroy an in-process TensorFlow server, frees memory. If server is running it
///   will be stopped and joined.
/// </summary>
procedure TF_DeleteServer(server: PTF_Server);
  cdecl; external TensorFlowLib;
/// <summary>
///   Register a listener method that processes printed messages. If any listeners are
///   registered, the print operator will call all listeners with the printed messages
///   and immediately return without writing to the logs.
/// </summary>
procedure TF_RegisterLogListener(listener: TLogListener);
  cdecl; external TensorFlowLib;
/// <summary>
///   Register a FileSystem plugin from filename `plugin_filename`. On success, place
///   OK in status. On failure, place an error status in status.
/// </summary>
procedure TF_RegisterFilesystemPlugin(const plugin_filename: PAnsiChar; status: PTF_Status);
  cdecl; external TensorFlowLib;
{$ENDREGION}
{$ENDREGION}
{$MINENUMSIZE 1}
implementation
procedure Deallocator_For_TensorDatas(data: Pointer; len: TF_size_t; arg: Pointer);
var
 l_pntBoolean: PBoolean;
begin
 FreeMem(data);
 l_pntBoolean := PBoolean(arg);
 l_pntBoolean^:= True;
end;

{ TF_Output }

constructor TF_Output.Create(oper: PTF_Operation; index: Integer);
begin
    Self.oper := oper;
    Self.index:= index;
end;

{ TF_Input }

constructor TF_Input.Create(oper: PTF_Operation; index: Integer);
begin
    Self.oper := oper;
    Self.index:= index;
end;

function TF_OperationOutputConsumers_wrapper(oper_out: TF_Output): TArray<String>;
begin
{$POINTERMATH ON}
    var num_consumers := TF_OperationOutputNumConsumers(oper_out);
    var size : Integer:= SizeOf(TF_Input);
    var handle        := AllocMem(size * num_consumers);
    var num : Integer := TF_OperationOutputConsumers(oper_out, handle, num_consumers);

    var consumers : TArray<String> ;
    SetLength(consumers,num_consumers);

    var inputptr := PTF_Input(handle);
    for var i : Integer := 0 to num - 1 do
    begin
        var oper     := inputptr[i].oper;
        consumers[i] := string(Ansistring(TF_OperationName(oper)));
    end;
    FreeMem(handle) ;

    Result := consumers;
{$POINTERMATH OFF}
end;

end.
