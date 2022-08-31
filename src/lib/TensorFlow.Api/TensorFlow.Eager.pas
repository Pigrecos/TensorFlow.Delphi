unit TensorFlow.Eager;

interface
   uses System.SysUtils, TensorFlow.LowLevelAPI;

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
  procedure TFE_ContextSetLogDevicePlacement(ctx: PTFE_Context; enable: Byte;status: PTF_Status);cdecl;
  {$EXTERNALSYM TFE_ContextSetLogDevicePlacement}

  /// <summary>
  ///
  /// </summary>
  /// <param name="t">const tensorflow::Tensor&amp;</param>
  /// <returns>TFE_TensorHandle*</returns>
  function TFE_NewTensorHandle(t: Pointer;  status: PTF_Status):PTF_Tensor;cdecl;
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
  function TFE_TensorHandleResolve(t: Pointer;  status: PTF_Status):PTF_Tensor;cdecl;
  {$EXTERNALSYM TFE_TensorHandleResolve}

  function TFE_TensorHandleNumDims(t: Pointer;  status: PTF_Status):Integer;cdecl;
  {$EXTERNALSYM TFE_TensorHandleNumDims}

  function TFE_TensorHandleDim(t: Pointer;  dim: Integer ; status: PTF_Status):Integer;cdecl;
  {$EXTERNALSYM TFE_TensorHandleDim}

  function TFE_TensorHandleDeviceName(h : PTF_Tensor ; status: PTF_Status): PTFChar;
  {$EXTERNALSYM TFE_TensorHandleDeviceName}

implementation

function  TFE_NewContextOptions;                      external c_sNameOfTensorflowLib name 'TFE_NewContextOptions';
procedure TFE_ContextOptionsSetConfig;                external c_sNameOfTensorflowLib name 'TFE_ContextOptionsSetConfig';
procedure TFE_ContextOptionsSetAsync;                 external c_sNameOfTensorflowLib name 'TFE_ContextOptionsSetAsync';
procedure TFE_ContextOptionsSetDevicePlacementPolicy; external c_sNameOfTensorflowLib name 'TFE_ContextOptionsSetDevicePlacementPolicy';
procedure TFE_DeleteContextOptions;                   external c_sNameOfTensorflowLib name 'TFE_DeleteContextOptions';
function  TFE_NewContext;                             external c_sNameOfTensorflowLib name 'TFE_NewContext';
procedure TFE_DeleteContext;                          external c_sNameOfTensorflowLib name 'TFE_DeleteContext';
function  TFE_ContextListDevices;                     external c_sNameOfTensorflowLib name 'TFE_ContextListDevices';
procedure TFE_ContextClearCaches;                     external c_sNameOfTensorflowLib name 'TFE_ContextClearCaches';
procedure TFE_ContextStartStep;                       external c_sNameOfTensorflowLib name 'TFE_ContextStartStep';
procedure TFE_ContextEndStep;                         external c_sNameOfTensorflowLib name 'TFE_ContextEndStep';
procedure TFE_ContextSetLogDevicePlacement;           external c_sNameOfTensorflowLib name 'TFE_ContextSetLogDevicePlacement';
function  TFE_NewTensorHandle;                        external c_sNameOfTensorflowLib name 'TFE_NewTensorHandle';
function  TFE_TensorHandleResolve;                    external c_sNameOfTensorflowLib name 'TFE_TensorHandleResolve';
function  TFE_TensorHandleNumDims;                    external c_sNameOfTensorflowLib name 'TFE_TensorHandleNumDims';
function  TFE_TensorHandleDim;                        external c_sNameOfTensorflowLib name 'TFE_TensorHandleDim';
function  TFE_TensorHandleDeviceName;                 external c_sNameOfTensorflowLib name 'TFE_TensorHandleDeviceName';

end.
