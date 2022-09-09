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
unit TensorFlow.DApiOperations;
interface
uses
  System.SysUtils, System.Variants, System.Classes, System.Types, System.IOUtils,
  System.AnsiStrings, System.StrUtils, TensorFlow.LowLevelAPI, TensorFlow._Helpers,
  TensorFlow.DApiBase, TensorFlow.DApi,

  Generics.Collections,

  Spring.Collections.Dictionaries,
  Spring.Collections.Enumerable,
  ProtoGen.opDef,
  ProtoGen.attrValue;
type
TFGraphHelper = class Helper for TFGraph
// ---------------------------- Fix, not generated -----------------------------
  function Create_op(op_type: TFString ;
                     inputs         : TArray<TFTensor>;
                     dtypes         : TArray<TF_DataType>;
                     input_types    : TArray<TF_DataType> = [];
                     Name           : TFString= '';
                     attrs          : System.Generics.Collections.TDictionary<string, TAttrValue> = nil;
                     op_def         : POpDef= nil;
                     compute_device : Boolean = True) : TFOperation;
	/// <summary>
	/// Creates a constant operation from a TFTensor or constant
	/// </summary>
	/// <param name="value">Value.</param>
	/// <param name="operName">Oper name.</param>
	/// <remarks>
	/// Since TFTensor have implicit conversion operators, you can call this method with
	/// a constant like this: graph.Const (23)
	/// </remarks>
	function OpConst(value: TFTensor; operName: TFString = ''): TFOutput; overload;
  /// <summary>
  ///   A placeholder op for a value that will be fed into the computation.
  /// </summary>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Placeholder'.
  /// </param>
  /// <param name="shape">
  ///   Optional argument
  ///   (Optional) The shape of the tensor. If the shape has 0 dimensions, the
  ///   shape is unconstrained.
  /// </param>
  /// <param name="dtype">
  ///   The type of elements in the tensor.
  /// </param>
  /// <returns>
  ///   A placeholder tensor that must be replaced using the feed mechanism.
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   N.B. This operation will fail with an error if it is executed. It is
  ///   intended as a way to represent a value that will always be fed, and to
  ///   provide attrs that enable the fed value to be checked at runtime.
  /// </remarks>
  function OpPlaceholder(dtype: TF_DataType; shape: PTFShape = Nil; operName: TFString = ''): TFOutput;
  /// <summary>
  /// Variable node, with a starting initial value.
  /// </summary>
  /// <param name="initialValue">Initial value.</param>
  /// <param name="init">Returns the operation that initializes the value of the variable.</param>
  /// <param name="value">Returns the value of the variable.</param>
  /// <param name="operName">Operation name, optional.</param>
  /// <returns>The returning TFOutput returns the handle to the variable.</returns>
  /// <remarks>
  /// Variables need to be initialized before the main execution so you will typically want to
  /// run the session on the variable
  /// </remarks>
  function Variable(initialValue: TFOutput; var init: TFOperation; var value: TFOutput; operName: TFString = ''): TFOutput; overload;
  /// <summary>
  /// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
  /// </summary>
  /// <param name="initialValue">Initial value.</param>
  /// <param name="value">Returns the value of the variable.</param>
  /// <param name="operName">Operation name, optional.</param>
  /// <returns>The returning TFOutput returns the handle to the variable.</returns>
  /// <remarks>
  /// Variables need to be initialized before the main execution so you will typically want to
  /// run the session on the variable.
  ///
  /// The init sequence for the variable is stored in the graph, you must manually initialize
  /// those by running the session on the global variables.
  /// </remarks>
  function Variable(initialValue: TFOutput; var value: TFOutput; operName: TFString = ''): TFOutput; overload;
  /// <summary>
  /// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
  /// </summary>
  /// <param name="initialValue">Initial value.</param>
  /// <param name="operName">Operation name, optional.</param>
  /// <returns>The returning TFOutput returns the handle to the variable.</returns>
  /// <remarks>
  /// Variables need to be initialized before the main execution so you will typically want to
  /// run the session on the variable.
  ///
  /// The init sequence for the variable is stored in the graph, you must manually initialize
  /// those by running the session on the global variables.
  /// </remarks>
  function Variable(initialValue: TFOutput; operName: TFString = ''): TFOutput; overload;
  /// <summary>
  ///   Creates a handle to a Variable resource.
  /// </summary>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'VarHandleOp'.
  /// </param>
  /// <param name="container">
  ///   Optional argument
  ///   the container this variable is placed in.
  /// </param>
  /// <param name="shared_name">
  ///   Optional argument
  ///   the name by which this variable is referred to.
  /// </param>
  /// <param name="dtype">
  ///   the type of this variable. Must agree with the dtypes
  ///   of all ops using this variable.
  /// </param>
  /// <param name="shape">
  ///   The (possibly partially specified) shape of this variable.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpVarHandle(dtype: TF_DataType; shape: TFShape;
                       container: TFString = ''; shared_name: TFString = '';
                       operName:  TFString = ''): TFOutput;
  /// <summary>
  ///   Assigns a new value to a variable.
  /// </summary>
  /// <param name="resource">
  ///   handle to the resource in which to store the variable.
  /// </param>
  /// <param name="value">
  ///   the value to set the new tensor to use.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'AssignVariableOp'.
  /// </param>
  /// <returns>
  ///   Returns the description of the operation
  /// </returns>
  /// <remarks>
  ///   Any ReadVariableOp with a control dependency on this op is guaranteed to return
  ///   this value or a subsequent newer value of the variable.
  /// </remarks>
  function OpAssignVariable(resource: TFOutput; value: TFOutput; operName: TFString = ''): TFOperation;
  /// <summary>
  ///   Reads the value of a variable.
  /// </summary>
  /// <param name="resource">
  ///   handle to the resource in which to store the variable.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'ReadVariableOp'.
  /// </param>
  /// <param name="dtype">
  ///   the dtype of the value.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   The tensor returned by this operation is immutable.
  ///
  ///   The value returned by this operation is guaranteed to be influenced by all the
  ///   writes on which this operation depends directly or indirectly, and to not be
  ///   influenced by any of the writes which depend directly or indirectly on this
  ///   operation.
  /// </remarks>
  function OpReadVariable(resource: TFOutput; dtype: TF_DataType; operName: TFString = ''): TFOutput;
  // Returns range(0, rank(x)) if reduction_indices is null
  function OpReduceDims(input: TFOutput; axis: TFOutput = Nil): TFOutput;
  /// <summary>
  /// Computes the sum of elements across dimensions of a tensor.
  /// </summary>
  /// <returns>The reduced tensor.</returns>
  /// <param name="input">The tensor to reduce. Should have numeric type.</param>
  /// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
  /// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
  /// <param name="operName">A name for the operation, optional.</param>
  /// <remarks>
  ///   Reduces input_tensor along the dimensions given in axis.
  /// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
  /// entry in axis. If keep_dims is true, the reduced dimensions
  /// are retained with length 1.
  ///
  /// If axis has no entries, all dimensions are reduced, and a
  /// tensor with a single element is returned.
  /// </remarks>
  function OpReduceSum(input: TFOutput; axis: TFOutput = Nil; keep_dims: Boolean = False; operName: TFString = ''): TFOutput;
// -----------------------------------------------------------------------------
  //
  /// <summary>
  ///   Raise a exception to abort the process when called.
  /// </summary>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Abort'.
  /// </param>
  /// <param name="error_msg">
  ///   Optional argument
  ///   A string which is the message associated with the exception.
  /// </param>
  /// <param name="exit_without_error">
  ///   Optional argument
  /// </param>
  /// <returns>
  ///   Returns the description of the operation
  /// </returns>
  /// <remarks>
  ///   If exit_without_error is true, the process will exit normally,
  ///   otherwise it will exit with a SIGABORT signal.
  ///
  ///   Returns nothing but an exception.
  /// </remarks>
  function OpAbort(error_msg: TFString = ''; exit_without_error: Boolean = True; operName: TFString = ''): TFOperation;
  /// <summary>
  ///   Computes the absolute value of a tensor.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Abs'.
  /// </param>
  /// <returns>
  ///   The eration can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   Given a tensor `x`, this operation returns a tensor containing the absolute
  ///   value of each element in `x`. For example, if x is an input element and y is
  ///   an output element, this operation computes \\(y = |x|\\).
  /// </remarks>
  function OpAbs(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Returns x + y element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="y">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Add'.
  /// </param>
  /// <returns>
  ///   The eration can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  ///   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  /// </remarks>
  function OpAdd(x, y: TFOutput; operName: TFString = ''): TFOutput;
  // ..........................
  /// <summary>
  ///   Add all input elements element wise.
  /// </summary>
  /// <param name="inputs">
  ///   Must all be the same size and shape.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'AddN'.
  /// </param>
  /// <returns>
  ///   The eration can be fetched from the resulting TFOutput, by fetching the Operation property from the result.
  /// </returns>
  function OpAddN(inputs: TArray<TFOutput>; operName: TFString = ''): TFOutput;
  // ..........................
  /// <summary>
  ///   Returns a constant tensor.
  /// </summary>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Const'.
  /// </param>
  /// <param name="value">
  ///   Attr `value` is the tensor to return.
  /// </param>
  /// <param name="dtype">
  /// </param>
  /// <returns>
  ///   The eration can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpConst(value: TFTensor; dtype: TF_DataType; operName: TFString = ''): TFOutput; overload;
  // ..........................
  /// <summary>
  ///   Returns x * y element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="y">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Mul'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   *NOTE*: `Mul` supports broadcasting. More about broadcasting
  ///   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  /// </remarks>
  function OpMul(x, y: TFOutput; operName: TFString = ''): TFOutput;
  // ..........................
  /// <summary>
  ///   Returns x / y element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="y">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Div'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   *NOTE*: `Div` supports broadcasting. More about broadcasting
  ///   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  /// </remarks>
  function OpDiv(x, y: TFOutput; operName: TFString = ''): TFOutput;
  // ..........................
  /// <summary>
  ///   Returns the index with the largest value across dimensions of a tensor.
  /// </summary>
  /// <param name="input">
  /// </param>
  /// <param name="dimension">
  ///   int32, 0 &amp;lt;= dimension &amp;lt; rank(input).  Describes which dimension
  ///   of the input Tensor to reduce across. For vectors, use dimension = 0.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'ArgMax'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   Note that in case of ties the identity of the return value is not guaranteed.
  /// </remarks>
  function OpArgMax(input, dimension: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Returns the index with the smallest value across dimensions of a tensor.
  /// </summary>
  /// <param name="input">
  /// </param>
  /// <param name="dimension">
  ///   int32, 0 &amp;lt;= dimension &amp;lt; rank(input).  Describes which dimension
  ///   of the input Tensor to reduce across. For vectors, use dimension = 0.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'ArgMin'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   Note that in case of ties the identity of the return value is not guaranteed.
  /// </remarks>
  function OpArgMin(input, dimension: TFOutput; operName: TFString = ''): TFOutput;
  // ..........................
  /// <summary>
  ///   Computes sin of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Sin'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpSin(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes cos of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Cos'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpCos(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes tan of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Tan'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpTan(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes asin of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Asin'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpASin(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes acos of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Acos'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpACos(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes atan of x element-wise.
  /// </summary>
  /// <param name="x">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Atan'.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  function OpATan(x: TFOutput; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Multiply the matrix "a" by the matrix "b".
  /// </summary>
  /// <param name="a">
  /// </param>
  /// <param name="b">
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'MatMul'.
  /// </param>
  /// <param name="transpose_a">
  ///   Optional argument
  ///   If true, "a" is transposed before multiplication.
  /// </param>
  /// <param name="transpose_b">
  ///   Optional argument
  ///   If true, "b" is transposed before multiplication.
  /// </param>
  /// <returns>
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   The inputs must be two-dimensional matrices and the inner dimension of
  ///   "a" (after being transposed if transpose_a is true) must match the
  ///   outer dimension of "b" (after being transposed if transposed_b is
  ///   true).
  ///
  ///   *Note*: The default kernel implementation for MatMul on GPUs uses
  ///   cublas.
  /// </remarks>
  function OpMatMul(a, b: TFOutput; transpose_a: Boolean = False; transpose_b: Boolean = False;
                    operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Computes the sum of elements across dimensions of a tensor.
  /// </summary>
  /// <param name="input">
  ///   The tensor to reduce.
  /// </param>
  /// <param name="reduction_indices">
  ///   The dimensions to reduce.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Sum'.
  /// </param>
  /// <param name="keep_dims">
  ///   Optional argument
  ///   If true, retain reduced dimensions with length 1.
  /// </param>
  /// <returns>
  ///   The reduced tensor.
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   Reduces `input` along the dimensions given in `reduction_indices`. Unless
  ///   `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  ///   `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  ///   retained with length 1.
  /// </remarks>
  function OpSum(input: TFOutput; reduction_indices: TFOutput; keep_dims: Boolean = False; operName: TFString = ''): TFOutput;
  /// <summary>
  ///   Creates a sequence of numbers.
  /// </summary>
  /// <param name="start">
  ///   0-D (scalar). First entry in the sequence.
  /// </param>
  /// <param name="limit">
  ///   0-D (scalar). Upper limit of sequence, exclusive.
  /// </param>
  /// <param name="delta">
  ///   0-D (scalar). Optional. Default is 1. Number that increments `start`.
  /// </param>
  /// <param name="operName">
  ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Range'.
  /// </param>
  /// <returns>
  ///   1-D.
  ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
  /// </returns>
  /// <remarks>
  ///   This operation creates a sequence of numbers that begins at `start` and
  ///   extends by increments of `delta` up to but not including `limit`.
  ///
  ///   For example:
  ///
  ///   ```
  ///   # 'start' is 3
  ///   # 'limit' is 18
  ///   # 'delta' is 3
  ///   tf.OpRange(start, limit, delta) ==&amp;gt; [3, 6, 9, 12, 15]
  ///   ```
  /// </remarks>
  function OpRange(start, limit, delta: TFOutput; operName: TFString = ''): TFOutput;
end;

implementation
          uses TensorFlow.Ops;

function TFGraphHelper.Create_op(op_type       : TFString;
                                 inputs        : TArray<TFTensor>;
                                 dtypes,
                                 input_types   : TArray<TF_DataType>;
                                 Name          : TFString;
                                 attrs         : System.Generics.Collections.TDictionary<string, TAttrValue>;
                                 op_def        : POpDef;
                                 compute_device: Boolean): TFOperation;
var
  l_sBuf2: TFString;
  var Tt : Enumerable<TFTensor>;
begin
    if Length(inputs) = 0 then
        inputs := [];//[ TFTensor.Create( [Int32(0)]) ];
    if string.IsNullOrEmpty(Name) then
        name := op_type;

    // If a names ends with a '/' it is a "name scope" and we use it as-is,
    // after removing the trailing '/'.
    if string(name).EndsWith('/') then
        name := TOps.name_from_scope_name(name)
    else
        Name := Name ; //unique_name(name)

    var node_def := TOps._NodeDef(op_type, name, attrs);

    var input_ops   : TArray<TFOperation>;
    for var i := 0 to Length(inputs)-1 do
      input_ops := input_ops  + [ inputs[i].op ];

    Tt := Enumerable<TFTensor>.Create(inputs);
    var operations := Tt.Select<ITensorOrOperation>(function(x: TFTensor): ITensorOrOperation
                                              begin
                                                Result := x.op;
                                              end).ToArray;

    var op := TFOperation.Create(node_def,Self,inputs,dtypes,operations,input_types,'',op_def);

    Result := op;

end;

// ---------------------------- Fix --------------------------------------------
function TFGraphHelper.OpConst(value: TFTensor; operName: TFString = ''): TFOutput;
begin
 Result := OpConst(value, value.TensorDataType, operName);
end;
function TFGraphHelper.OpPlaceholder (dtype: TF_DataType; shape: PTFShape = Nil; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Placeholder',l_sBuf1), MakeName('Placeholder', operName, l_sBuf2));
 l_oDesc.SetAttrType('dtype', dtype);
 if Assigned(shape) then
   l_oDesc.SetAttrShape('shape', shape);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.Variable(initialValue: TFOutput; var init: TFOperation; var value: TFOutput; operName: TFString = ''): TFOutput;
var
 scopeName: TFString;
 newScope, aScope, rScope: TFScope;
 dataType: TF_DataType;
 l_oShape: TFShape;
 dims: TArray<TF_int64_t>;
 l_pDims: PTF_int64_t;
 hnd: TFOutput;
begin
 Result := Nil;
 value  := Nil;
 MakeName('Variable', operName, scopeName);
 newScope := self.WithScope(scopeName);
 with newScope do begin
   dataType := initialValue.OutputType;
   l_pDims  := GetShape(initialValue, dims);
   l_oShape := TFShape.Create(dims);
   hnd      := OpVarHandle(dataType, l_oShape);
   aScope   := self.WithScope('Assign');
   with aScope do begin
     init  := OpAssignVariable(hnd, initialValue);
     rScope:= self.WithScope('Read');
     with rScope do begin
       value  := OpReadVariable(hnd, dataType);
       Result := hnd;
     end;
   end;

 end;
end;
function TFGraphHelper.Variable(initialValue: TFOutput; var value: TFOutput; operName: TFString = ''): TFOutput;
var
 scopeName: TFString;
 newScope, aScope, rScope: TFScope;
 dataType: TF_DataType;
 l_oShape: TFShape;
 dims: TArray<TF_int64_t>;
 l_pDims: PTF_int64_t;
 hnd: TFOutput;
 init: TFOperation;
begin
 Result := Nil;
 value  := Nil;
 MakeName('Variable', operName, scopeName);
 newScope := self.WithScope(scopeName);
 with newScope do begin
   dataType := initialValue.OutputType;
   l_pDims  := GetShape(initialValue, dims);
   l_oShape := TFShape.Create(dims);
   hnd      := OpVarHandle(dataType, l_oShape);
   aScope   := self.WithScope('Assign');
   with aScope do begin
     init  := OpAssignVariable(hnd, initialValue);
     self.AddInitVariable(init);
     rScope:= self.WithScope('Read');
     with rScope do begin
       value  := OpReadVariable(hnd, dataType);
       Result := hnd;
     end;
   end;

 end;
end;
function TFGraphHelper.Variable(initialValue: TFOutput; operName: TFString = ''): TFOutput;
var
 scopeName: TFString;
 newScope, aScope: TFScope;
 dataType: TF_DataType;
 l_oShape: TFShape;
 dims: TArray<TF_int64_t>;
 l_pDims: PTF_int64_t;
 hnd: TFOutput;
 init: TFOperation;
begin
 Result := Nil;
 MakeName('Variable', operName, scopeName);
 newScope := self.WithScope(scopeName);
 with newScope do begin
   dataType := initialValue.OutputType;
   l_pDims  := GetShape(initialValue, dims);
   l_oShape := TFShape.Create(dims);
   hnd      := OpVarHandle(dataType, l_oShape);
   aScope   := self.WithScope('Assign');
   with aScope do begin
     init  := OpAssignVariable(hnd, initialValue);
     self.AddInitVariable(init);
     Result := hnd;
   end;

 end;
end;
function TFGraphHelper.OpReadVariable(resource: TFOutput; dtype: TF_DataType; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('ReadVariableOp',l_sBuf1), MakeName('ReadVariableOp', operName, l_sBuf2));
 l_oDesc.AddInput(resource);
 l_oDesc.SetAttrType('dtype', dtype);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpReduceDims(input: TFOutput; axis: TFOutput = Nil): TFOutput;
var
 n, i: Integer;
 arr:  TArray<Integer>;
 dims: TArray<TF_int64_t>;
 l_pDims: PTF_int64_t;
begin
 if Assigned(axis) then
   Result := axis
 else begin
   // Fast path: avoid creating Rank and Range ops if ndims is known.
   l_pDims  := GetShape(input, dims);
   n := Length(dims);
   if n > 0 then begin
     // The python code distinguishes between tensor and sparsetensor
     SetLength(arr,n);
     for i := 0 to n-1 do
       arr[i] := i;    //TODO: <- Überprüfen!!!  Ist nicht  arr[i] = dims[i] gemeint?
     Result := OpConst(_T(arr), TF_DataType.TF_INT32);
   end
   else begin
     Result := OpRange(OpConst(_T(0)), OpConst(_T(0)), OpConst(_T(1)));
   end;
 end;
end;
function TFGraphHelper.OpReduceSum(input: TFOutput; axis: TFOutput = Nil; keep_dims: Boolean = False; operName: TFString = ''): TFOutput;
var
 reduction_indices: TFOutput;
begin
 reduction_indices := self.OpReduceDims(input, axis);
 Result := OpSum(input, reduction_indices, keep_dims, operName);
end;
// -----------------------------------------------------------------------------

function TFGraphHelper.OpAbort(error_msg: TFString = '';
                   exit_without_error: Boolean = True; operName: TFString = ''): TFOperation;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Abort',l_sBuf1), MakeName('Abort', operName, l_sBuf2));
 if Length(error_msg) > 0 then
   l_oDesc.SetAttr(TFString('error_msg'), error_msg);
 if exit_without_error then
   l_oDesc.SetAttr('exit_without_error', exit_without_error);
 Result := l_oDesc.FinishOperation ();
end;
function TFGraphHelper.OpAbs(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Abs',l_sBuf1), MakeName('Abs', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpAdd(x, y: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Add',l_sBuf1), MakeName('Add', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 l_oDesc.AddInput(y);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpAddN(inputs: TArray<TFOutput>; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('AddN',l_sBuf1), MakeName('AddN', operName, l_sBuf2));
 l_oDesc.AddInputs(inputs);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpConst(value: TFTensor; dtype: TF_DataType; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_oOp:   TFOperation;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Const',l_sBuf1), MakeName('Const', operName, l_sBuf2));
 l_oDesc.SetAttr('value',value);
 l_oDesc.SetAttrType('dtype', dtype);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpMul(x, y: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Mul',l_sBuf1), MakeName('Mul', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 l_oDesc.AddInput(y);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpDiv(x, y: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Div',l_sBuf1), MakeName('Div', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 l_oDesc.AddInput(y);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpArgMax(input, dimension: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('ArgMax',l_sBuf1), MakeName('ArgMax', operName, l_sBuf2));
 l_oDesc.AddInput(input);
 l_oDesc.AddInput(dimension);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpArgMin(input, dimension: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('ArgMin',l_sBuf1), MakeName('ArgMin', operName, l_sBuf2));
 l_oDesc.AddInput(input);
 l_oDesc.AddInput(dimension);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpSin(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Sin',l_sBuf1), MakeName('Sin', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpCos(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Cos',l_sBuf1), MakeName('Cos', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpTan(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Tan',l_sBuf1), MakeName('Tan', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpASin(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Asin',l_sBuf1), MakeName('Asin', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpACos(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Acos',l_sBuf1), MakeName('Acos', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpATan(x: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Atan',l_sBuf1), MakeName('Atan', operName, l_sBuf2));
 l_oDesc.AddInput(x);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpMatMul(a, b: TFOutput;
                                transpose_a: Boolean = False; transpose_b: Boolean = False;
                                operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('MatMul',l_sBuf1), MakeName('MatMul', operName, l_sBuf2));
 l_oDesc.AddInput(a);
 l_oDesc.AddInput(b);
 if transpose_a then
   l_oDesc.SetAttr ('transpose_a', transpose_a);
 if transpose_b then
   l_oDesc.SetAttr ('transpose_b', transpose_b);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpVarHandle(dtype: TF_DataType; shape: TFShape;
                       container: TFString = ''; shared_name: TFString = '';
                       operName:  TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('VarHandleOp',l_sBuf1), MakeName('VarHandleOp', operName, l_sBuf2));
 l_oDesc.SetAttrType('dtype', dtype);
 l_oDesc.SetAttrShape('shape', shape);
 if Length(container) > 0 then
   l_oDesc.SetAttr('container', container);
 if Length(shared_name) > 0 then
   l_oDesc.SetAttr('shared_name', shared_name);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpAssignVariable(resource: TFOutput; value: TFOutput; operName: TFString = ''): TFOperation;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('AssignVariableOp',l_sBuf1), MakeName('AssignVariableOp', operName, l_sBuf2));
 l_oDesc.AddInput(resource);
 l_oDesc.AddInput(value);
 Result := l_oDesc.FinishOperation();
end;
function TFGraphHelper.OpSum(input: TFOutput; reduction_indices: TFOutput; keep_dims: Boolean = False; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Sum',l_sBuf1), MakeName('Sum', operName, l_sBuf2));
 l_oDesc.AddInput (input);
 l_oDesc.AddInput (reduction_indices);
 if keep_dims then
   l_oDesc.SetAttr ('keep_dims', keep_dims);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
function TFGraphHelper.OpRange(start, limit, delta: TFOutput; operName: TFString = ''): TFOutput;
var
 l_oDesc: TFOperationDesc;
 l_sBuf1, l_sBuf2: TFString;
begin
 l_oDesc := TFOperationDesc.Create(self, _PTFChar('Range',l_sBuf1), MakeName('Range', operName, l_sBuf2));
 l_oDesc.AddInput(start);
 l_oDesc.AddInput(limit);
 l_oDesc.AddInput(delta);
 Result := TFOutput.Create(l_oDesc.FinishOperation());
end;
end.
