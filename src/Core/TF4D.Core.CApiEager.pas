unit TF4D.Core.CApiEager;
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
   uses System.SysUtils,TF4D.Core.CApi;

type
  /// <summary> Controls how to act when we try to run an operation on a given device but
  /// some input tensors are not on that device.
  /// LINT.IfChange
  /// Note: Keep in sync with internal copy of enum in eager/context.h. </summary>
  TFE_ContextDevicePlacementPolicy = (
    // <summary> Running operations with input tensors on the wrong device will fail. </summary>
    TFE_DEVICE_PLACEMENT_EXPLICIT = 0,
    // <summary> Copy the tensor to the right device but log a warning.</summary>
    TFE_DEVICE_PLACEMENT_WARN = 1,
    // <summary> Silently copy the tensor, which has a performance cost since the operation
    // will be blocked till the copy completes. This is the default placement </summary>
    // policy.
    TFE_DEVICE_PLACEMENT_SILENT = 2,
    // <summary> Placement policy which silently copies int32 tensors but not other dtypes. </summary>
    TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3);


  PTFE_ContextOptions = Pointer;

  // <summary> "Context" under which operations/functions are executed. It encapsulates
  // things like the available devices, resource manager etc.</summary>
  // <remarks> TFE_Context must outlive all tensor handles created using it. In other
  // words, TFE_DeleteContext() must be called after all tensor handles have
  // been deleted (with TFE_DeleteTensorHandle).</remarks>
  //
  // TODO(ashankar): Merge with TF_Session?
  PTFE_Context = Pointer;

  PTFE_Op  = Pointer;

  // A handle to a tensor on a device.
  //
  // Like a TF_Tensor, a TFE_TensorHandle refers to a tensor with a value, shape,
  // type etc. Unlike a TF_Tensor, a TFE_TensorHandle may refer to such tensors
  // placed in memory of different devices or remote address spaces.
  PTFE_TensorHandle = Pointer;

  /// <summary>
  /// Execute the operation defined by 'op' and return handles to computed
  /// tensors in `retvals`.
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="retvals">TFE_TensorHandle**</param>
  /// <param name="num_retvals">int*</param>
  /// <param name="status">TF_Status*</param>
  procedure TFE_Execute(op: PTFE_Op; retvals: PTFE_TensorHandle; var num_retvals: Integer; status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_Execute}

  /// <summary>
  ///
  /// </summary>
  /// <param name="ctx">TFE_Context*</param>
  /// <param name="op_or_function_name">const char*</param>
  /// <param name="status">TF_Status*</param>
  /// <returns></returns>
  function TFE_NewOp(ctx: PTFE_Context; op_or_function_name: PTFChar; status: PTF_Status): PTFE_Op; cdecl;
  {$EXTERNALSYM TFE_NewOp}

  /// <summary>
  /// Resets `op_to_reset` with `op_or_function_name` and `raw_device_name`. This
  /// is for performance optimization by reusing an exiting unused op rather than
  /// creating a new op every time. If `raw_device_name` is `NULL` or empty, it
  /// does not set the device name. If it's not `NULL`, then it attempts to parse
  /// and set the device name. It's effectively `TFE_OpSetDevice`, but it is faster
  /// than separately calling it because if the existing op has the same
  /// `raw_device_name`, it skips parsing and just leave as it is.
  /// </summary>
  /// <param name="op_to_reset">TFE_Op*</param>
  /// <param name="op_or_function_name">const char*</param>
  /// <param name="raw_device_name">const char*</param>
  /// <param name="status">TF_Status*</param>
  procedure TFE_OpReset(op_to_reset: PTFE_Op; op_or_function_name: PTFChar; raw_device_name: PTFChar; status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_OpReset}

  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  procedure TFE_DeleteOp(op: PTFE_Op); cdecl;
  {$EXTERNALSYM TFE_DeleteOp}
  /// <summary>
  ///
  /// </summary>
  /// <param name="op"></param>
  /// <param name="device_name"></param>
  /// <param name="status"></param>
  procedure TFE_OpSetDevice(op: PTFE_Op; device_name: PTFChar; status: PTF_Status); cdecl;
  {$EXTERNALSYM TFE_OpSetDevice}

  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="h">TFE_TensorHandle*</param>
  /// <param name="status">TF_Status*</param>
  procedure TFE_OpAddInput(op: PTFE_Op; h: PTFE_TensorHandle; status: PTF_Status); cdecl;
  {$EXTERNALSYM TFE_OpAddInput}

  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="attr_name">const char*</param>
  /// <param name="is_list">unsigned char*</param>
  /// <param name="status">TF_Status*</param>
  /// <returns></returns>
  function TFE_OpGetAttrType(op: PTFE_Op; attr_name: PTFChar; is_list: pbyte; status: PTF_Status): TF_AttrType; cdecl;
  {$EXTERNALSYM TFE_OpGetAttrType}

  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="attr_name">const char*</param>
  /// <param name="value">TF_DataType</param>
  procedure TFE_OpSetAttrType(op: PTFE_Op; attr_name: PTFChar; value: TF_DataType); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrType}
  procedure TFE_OpSetAttrInt(op: PTFE_Op; attr_name: PTFChar; value: Int64); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrInt}
  procedure TFE_OpSetAttrFloat(op: PTFE_Op; attr_name: PTFChar; value: Single); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrFloat}

  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="attr_name">const char*</param>
  /// <param name="dims">const int64_t*</param>
  /// <param name="num_dims">const int</param>
  /// <param name="out_status">TF_Status*</param>
  procedure TFE_OpSetAttrShape(op : PTFE_Op; attr_name: PTFChar; dims: PInt64; num_dims: Integer; out_status: PTF_Status); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrShape}
  procedure TFE_OpSetAttrShapeList(op : PTFE_Op; attr_name: PTFChar; dims: PInteger; num_dims: PInteger; num_values: Integer; out_status: PTF_Status); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrShapeList}
  procedure TFE_OpSetAttrStringList(op : PTFE_Op; attr_name: PTFChar; values: PTFChar; lengths: PUInt64; num_values: Integer);cdecl;
  {$EXTERNALSYM TFE_OpSetAttrStringList}
  procedure TFE_OpSetAttrBool(op : PTFE_Op; attr_name: PTFChar; value: Boolean); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrBool}
  procedure TFE_OpSetAttrFunctionName(op : PTFE_Op; attr_name: PTFChar; data:PTFChar; length: Integer); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrFunctionName}
  /// <summary>
  ///
  /// </summary>
  /// <param name="op">TFE_Op*</param>
  /// <param name="attr_name">const char*</param>
  /// <param name="value">const void*</param>
  /// <param name="length">size_t</param>
  procedure TFE_OpSetAttrString(op : PTFE_Op; attr_name: PTFChar; value: PTFChar; length: UInt64); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrString}
  procedure TFE_OpSetAttrTypeList(op : PTFE_Op; attr_name: PTFChar; values: PTF_DataType; num_values: Integer); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrTypeList}
  procedure TFE_OpSetAttrIntList(op : PTFE_Op; attr_name: PTFChar; values : PInt64; num_values: Integer); cdecl;
  {$EXTERNALSYM TFE_OpSetAttrIntList}

  //<summary> Return a new options object.</summary>
  function TFE_NewContextOptions: PTFE_ContextOptions; cdecl;
  {$EXTERNALSYM TFE_NewContextOptions}
  // <summary> Set the config in TF_ContextOptions.options.
  // config should be a serialized tensorflow.ConfigProto proto.
  // If config was not parsed successfully as a ConfigProto, record the
  // error information in *status.</summary>
  procedure TFE_ContextOptionsSetConfig(options: PTFE_ContextOptions; proto: Pointer; proto_len: TF_size_t; status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_ContextOptionsSetConfig}

  // <summary> Sets the default execution mode (sync/async). Note that this can be
  // overridden per thread using TFE_ContextSetExecutorForThread.</summary>
  procedure TFE_ContextOptionsSetAsync(opt: PTFE_ContextOptions; enable: Byte); cdecl;
  {$EXTERNALSYM TFE_ContextOptionsSetAsync}
  procedure TFE_ContextOptionsSetDevicePlacementPolicy(opt: PTFE_ContextOptions;dp: TFE_ContextDevicePlacementPolicy); cdecl;
  {$EXTERNALSYM TFE_ContextOptionsSetAsync}
  // <summary> Destroy an options object.</summary>
  procedure TFE_DeleteContextOptions(opt: PTFE_ContextOptions); cdecl;
  {$EXTERNALSYM TFE_DeleteContextOptions}

  function TFE_NewContext(const opts: PTFE_ContextOptions; status: PTF_Status): PTFE_Context;cdecl;
  {$EXTERNALSYM TFE_NewContext}

  procedure TFE_DeleteContext(ctx: PTFE_Context); cdecl;
  {$EXTERNALSYM TFE_DeleteContext}
  function TFE_ContextListDevices(ctx: PTFE_Context; status: PTF_Status): PTF_DeviceList;cdecl;
  {$EXTERNALSYM TFE_ContextListDevices}
  // <summary> Clears the internal caches in the TFE context. Useful when reseeding random
  // ops.</summary>
  procedure TFE_ContextClearCaches(ctx: PTFE_Context);cdecl;
  {$EXTERNALSYM TFE_ContextClearCaches}

  // <summary> Some TF ops need a step container to be set to limit the lifetime of some
  // resources (mostly TensorArray and Stack, used in while loop gradients in
  // graph mode). Calling this on a context tells it to start a step.</summary>
  procedure TFE_ContextStartStep(ctx: PTFE_Context);cdecl;
  {$EXTERNALSYM TFE_ContextStartStep}
  // <summary> Ends a step. When there is no active step (that is, every started step has
  // been ended) step containers will be cleared. Note: it is not safe to call
  // TFE_ContextEndStep while ops that rely on the step container may be running.</summary>
  procedure TFE_ContextEndStep(ctx: PTFE_Context);cdecl;
  {$EXTERNALSYM TFE_ContextEndStep}

  // <summary> Configure device placement policy logging for the eager executor. Note this
  // policy is applied to any subsequent op executions.</summary>
  procedure TFE_ContextSetLogDevicePlacement(ctx: PTFE_Context; enable: Byte; status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_ContextSetLogDevicePlacement}

  // Adds a function (created from TF_GraphToFunction or
  // TF_FunctionImportFunctionDef) to the context, allowing it to be executed with
  // TFE_Execute by creating an op with the same name as the function.
  procedure TFE_ContextAddFunction(ctx: PTFE_Context; _function: PTF_Function; status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_ContextAddFunction}

  // Checks whether a function is registered under `name`.
  function TFE_ContextHasFunction(ctx: PTFE_Context;  const name: PTFChar): Byte; cdecl;
  {$EXTERNALSYM TFE_ContextHasFunction}

  // Removes a function from the context. Once removed, you can no longer
  // TFE_Execute it or TFE_Execute any TFE_Op which has it as an attribute or any
  // other function which calls it as an attribute.
  procedure TFE_ContextRemoveFunction(ctx: PTFE_Context; name: PTFChar; status: PTF_Status);
  {$EXTERNALSYM TFE_ContextRemoveFunction}

  /// <summary>
  ///
  /// </summary>
  /// <param name="h">TFE_TensorHandle*</param>
  procedure TFE_DeleteTensorHandle(h: Pointer); cdecl;
  {$EXTERNALSYM TFE_DeleteTensorHandle}

  /// <summary>
  ///
  /// </summary>
  /// <param name="t">const tensorflow::Tensor&amp;</param>
  /// <returns>TFE_TensorHandle*</returns>
  function TFE_NewTensorHandle(t: Pointer;  status: PTF_Status):PTFE_TensorHandle;cdecl;
  {$EXTERNALSYM TFE_NewTensorHandle}

  /// <summary>
  /// This function will block till the operation that produces `h` has
  /// completed. The memory returned might alias the internal memory used by
  /// TensorFlow. Hence, callers should not mutate this memory (for example by
  /// modifying the memory region pointed to by TF_TensorData() on the returned
  /// TF_Tensor).
  /// </summary>
  /// <param name="h">TFE_TensorHandle*</param>
  /// <param name="status">TF_Status*</param>
  /// <returns></returns>
  function TFE_TensorHandleResolve(t: Pointer;  status: PTF_Status):PTFE_TensorHandle;cdecl;
  {$EXTERNALSYM TFE_TensorHandleResolve}

  function TFE_TensorHandleNumDims(t: Pointer;  status: PTF_Status):Integer;cdecl;
  {$EXTERNALSYM TFE_TensorHandleNumDims}

  function TFE_TensorHandleDim(t: Pointer;  dim: Integer ; status: PTF_Status):Integer;cdecl;
  {$EXTERNALSYM TFE_TensorHandleDim}

  function TFE_TensorHandleDeviceName(h : PTFE_TensorHandle ; status: PTF_Status): PTFChar;
  {$EXTERNALSYM TFE_TensorHandleDeviceName}

implementation

function  TFE_NewContextOptions;                      external TensorFlowLib name 'TFE_NewContextOptions';
procedure TFE_ContextOptionsSetConfig;                external TensorFlowLib name 'TFE_ContextOptionsSetConfig';
procedure TFE_ContextOptionsSetAsync;                 external TensorFlowLib name 'TFE_ContextOptionsSetAsync';
procedure TFE_ContextOptionsSetDevicePlacementPolicy; external TensorFlowLib name 'TFE_ContextOptionsSetDevicePlacementPolicy';
procedure TFE_DeleteContextOptions;                   external TensorFlowLib name 'TFE_DeleteContextOptions';
function  TFE_NewContext;                             external TensorFlowLib name 'TFE_NewContext';
procedure TFE_DeleteContext;                          external TensorFlowLib name 'TFE_DeleteContext';
function  TFE_ContextListDevices;                     external TensorFlowLib name 'TFE_ContextListDevices';
procedure TFE_ContextClearCaches;                     external TensorFlowLib name 'TFE_ContextClearCaches';
procedure TFE_ContextStartStep;                       external TensorFlowLib name 'TFE_ContextStartStep';
procedure TFE_ContextEndStep;                         external TensorFlowLib name 'TFE_ContextEndStep';
procedure TFE_ContextSetLogDevicePlacement;           external TensorFlowLib name 'TFE_ContextSetLogDevicePlacement';
procedure TFE_ContextAddFunction;                     external TensorFlowLib name 'TFE_ContextAddFunction';
function  TFE_ContextHasFunction;                     external TensorFlowLib name 'TFE_ContextHasFunction';
procedure TFE_ContextRemoveFunction;                  external TensorFlowLib name 'TFE_ContextRemoveFunction';
function  TFE_NewTensorHandle;                        external TensorFlowLib name 'TFE_NewTensorHandle';
procedure TFE_DeleteTensorHandle;                     external TensorFlowLib name 'TFE_DeleteTensorHandle';
function  TFE_TensorHandleResolve;                    external TensorFlowLib name 'TFE_TensorHandleResolve';
function  TFE_TensorHandleNumDims;                    external TensorFlowLib name 'TFE_TensorHandleNumDims';
function  TFE_TensorHandleDim;                        external TensorFlowLib name 'TFE_TensorHandleDim';
function  TFE_TensorHandleDeviceName;                 external TensorFlowLib name 'TFE_TensorHandleDeviceName';

procedure TFE_Execute;                                external TensorFlowLib name 'TFE_Execute';
function  TFE_NewOp;                                  external TensorFlowLib name 'TFE_NewOp';
procedure TFE_OpReset;                                external TensorFlowLib name 'TFE_OpReset';
procedure TFE_DeleteOp;                               external TensorFlowLib name 'TFE_DeleteOp';
procedure TFE_OpSetDevice;                            external TensorFlowLib name 'TFE_OpSetDevice';
procedure TFE_OpAddInput;                             external TensorFlowLib name 'TFE_OpAddInput';
function  TFE_OpGetAttrType;                          external TensorFlowLib name 'TFE_OpGetAttrType';
procedure TFE_OpSetAttrType;                          external TensorFlowLib name 'TFE_OpSetAttrType';
procedure TFE_OpSetAttrInt;                           external TensorFlowLib name 'TFE_OpSetAttrInt';
procedure TFE_OpSetAttrFloat;                         external TensorFlowLib name 'TFE_OpSetAttrFloat';
procedure TFE_OpSetAttrShape;                         external TensorFlowLib name 'TFE_OpSetAttrShape';
procedure TFE_OpSetAttrShapeList;                     external TensorFlowLib name 'TFE_OpSetAttrShapeList';
procedure TFE_OpSetAttrStringList;                    external TensorFlowLib name 'TFE_OpSetAttrStringList';
procedure TFE_OpSetAttrBool;                          external TensorFlowLib name 'TFE_OpSetAttrBool';
procedure TFE_OpSetAttrFunctionName;                  external TensorFlowLib name 'TFE_OpSetAttrFunctionName';
procedure TFE_OpSetAttrString;                        external TensorFlowLib name 'TFE_OpSetAttrString';
procedure TFE_OpSetAttrTypeList;                      external TensorFlowLib name 'TFE_OpSetAttrTypeList';
procedure TFE_OpSetAttrIntList;                       external TensorFlowLib name 'TFE_OpSetAttrIntList';


end.
