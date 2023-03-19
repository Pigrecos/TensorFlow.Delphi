unit TensorFlow.Variable;
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
     uses System.SysUtils,
          System.Rtti,
          System.TypInfo,
          System.Generics.Collections,

          Spring,
          Spring.Collections.Enumerable,

          TensorFlow.Slice,
          TensorFlow.Initializer,
          TF4D.Core.CApi,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          Tensorflow.Core,

          TensorFlow.Proto;

type

  variables = class
    private

    public
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        class  function trainable_variables : TValue;
        /// <summary>
        /// Returns all variables and `SaveableObject`s that must be checkpointed.
        /// </summary>
        /// <param name="scope"></param>
        /// <returns></returns>
        class  function _all_saveable_objects(scope: string = ''): TArray<IVariableV1>;
        /// <summary>
        /// Returns global variables.
        /// </summary>
        /// <param name="scope">
        /// (Optional.) A string. If supplied, the resulting list is filtered
        /// to include only items whose `name` attribute matches `scope` using
        /// `re.match`. Items without a `name` attribute are never returned if a
        /// scope is supplied. The choice of `re.match` means that a `scope` without
        /// special tokens filters by prefix.
        /// </param>
        /// <returns>A list of `Variable` objects.</returns>
        class  function  global_variables(scope: string = ''): TList<IVariableV1>;
        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        class  function  variables_initializer(var_list: TArray<IVariableV1>; name: string = 'init'): TFOperation;
  end;

  state_ops = class
    private

    public
      /// <summary>
      /// Create a variable Operation.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <returns></returns>
      class function variable_op_v2(shape: TArray<Integer>; dtype: TF_DataType; name: string = 'Variable'; container : string= ''; shared_name: string = ''): TFTensor;
      class function assign<T>(ref: T; value: TValue; validate_shape: Boolean = true; use_locking: Boolean = true; name: string = ''): TFTensor; overload;
      class function assign(ref: IVariableV1; value: TValue; validate_shape: Boolean = true; use_locking: Boolean = true; name: string = '') : TFTensor; overload;
      class function assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean = false; name : string= ''): TFTensor;
      //"""Update 'ref' by adding 'value' to it.
      //
      //  This operation outputs "ref" after the update is done.
      //  This makes it easier to chain operations that need to use the reset value.
      //
      //  Args:
      //    ref: A mutable `Tensor`. Must be one of the following types:
      //      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
      //      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      //      Should be from a `Variable` node.
      //    value: A `Tensor`. Must have the same type as `ref`.
      //      The value to be added to the variable.
      //    use_locking: An optional `bool`. Defaults to `False`.
      //      If True, the addition will be protected by a lock;
      //      otherwise the behavior is undefined, but may exhibit less contention.
      //    name: A name for the operation (optional).
      //
      //  Returns:
      //    Same as "ref".  Returned as a convenience for operations that want
      //    to use the new value after the variable has been updated.
      class function assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean = false; name: string = ''): TFTensor;
      class function scatter_add(ref: IVariableV1; indices: TFTensor; updates: TFTensor; use_locking: Boolean = false; name: string = '')  : TFTensor;
      class function is_variable_initialized(ref: RefVariable; name: string = '') : TFTensor;
  end;

/// <summary>
/// Wrapper record for ResourceVariable for Operator Overloading
/// </summary>
TResourceVariable = record
  private
      FResourceHandle : ResourceVariable;

      function value: TFTensor;
  public
      function assign<T>(value: T; use_locking: Boolean = false; name: string = ''; read_value: Boolean = true):TFTensor;
      function numpy: TNDArray;

      class operator Implicit(t : ResourceVariable): TResourceVariable;
      class operator Implicit(t : TResourceVariable): ResourceVariable;
      class operator Implicit(t : TResourceVariable): TFTensor;
      class operator Implicit(t : TResourceVariable): TEagerTensor;

      class operator Add(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Add(x: TResourceVariable; y: Single): TFTensor;
      class operator Add(x: TResourceVariable; y: Double) : TFTensor;
      class operator Add(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Add(x: TResourceVariable; y: TFTensor) : TFTensor;
      class operator Add(x: TFTensor;          y: TResourceVariable) : TFTensor;
      //
      class operator Subtract(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: Single): TFTensor;
      class operator Subtract(x: TResourceVariable; y: Double) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Subtract(x: TResourceVariable; y: TFTensor) : TFTensor;
      //
      class operator Multiply(x: TResourceVariable; y: Integer) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: Single): TFTensor;
      class operator Multiply(x: TResourceVariable; y: Double) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TResourceVariable) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TFTensor) : TFTensor;
      class operator Multiply(x: TResourceVariable; y: TNDArray) : TFTensor;
      //
      class operator LessThan(x: TResourceVariable; y: TFTensor) : TFTensor;
      class operator GreaterThan(x: TResourceVariable; y: TFTensor) : TFTensor;

end;


implementation
     uses Oz.Pb.Classes,

          Tensorflow.Gradient,
          TensorFlow.Tensor,

          Tensorflow,
          Tensorflow.Utils,
          TensorFlow.Ops,
          TensorFlow.Operations;




{ variables }

class function variables.global_variables(scope: string): TList<IVariableV1>;
begin
    Result := Tops.get_collection<IVariableV1>(tf.GraphKeys.GLOBAL_VARIABLES, scope);
end;

class function variables.trainable_variables: TValue;
begin
    Result := Tops.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES);
end;

class function variables.variables_initializer(var_list: TArray<IVariableV1>; name: string): TFOperation;
begin
    if Length(var_list) > 0 then
    begin
        var opList : TArray<TFOperation> := [];
        for var i := 0 to Length(var_list)- 1 do
          opList := opList + [ var_list[i].Initializer ] ;
        Result :=  control_flow_ops.group<TFOperation>(opList, name) ;
    end else
    begin
        Result := gen_control_flow_ops.no_op(name);
    end;
end;

class function variables._all_saveable_objects(scope: string): TArray<IVariableV1>;
begin
    var all := TList<IVariableV1>.Create;
    try

      all.AddRange(Tops.get_collection<IVariableV1>(tf.GraphKeys.GLOBAL_VARIABLES, scope));
      all.AddRange(Tops.get_collection<IVariableV1>(tf.GraphKeys.SAVEABLE_OBJECTS, scope));
      Result :=  all.ToArray;
    finally
      all.free;
    end;
end;



{ state_ops }

class function state_ops.assign(ref: IVariableV1; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
        Result := gen_state_ops.assign(ref, value,  validate_shape, use_locking, name)
    else begin
        if      ref is RefVariable          then Result := (ref as RefVariable).assign(value, False , name)
        else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign(value, False ,name)
        else
           raise Exception.Create('state_ops.assign Error!');
    end;
end;

class function state_ops.assign<T>(ref: T; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
    Result := gen_state_ops.assign(ref, value, validate_shape, use_locking, name)
end;

class function state_ops.assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean; name: string): TFTensor;
begin
    if tf.executing_eagerly then
    begin
        if      ref is RefVariable          then Result := (ref as RefVariable).assign_add(value, use_locking , name)
        else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign_add(value, use_locking ,name)
    end
    else
        Result := gen_state_ops.assign_add(ref, value, use_locking, name);
end;

class function state_ops.assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
      Result := gen_state_ops.assign_sub(ref, value, use_locking, name)
    else begin
      if      ref is RefVariable          then Result := (ref as RefVariable).assign_sub(value, use_locking , name)
      else if ref is BaseResourceVariable then Result := (ref as BaseResourceVariable).assign_sub(value, use_locking ,name)
      else
           raise Exception.Create('state_ops.assign_sub Error!');
    end;
end;

class function state_ops.is_variable_initialized(ref: RefVariable; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
    begin
        Result := gen_state_ops.is_variable_initialized(ref, name);
        Exit;
    end;
    raise TFException.Create('Not Implemented - is_variable_initialized');
end;

class function state_ops.scatter_add(ref: IVariableV1; indices, updates: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    if TDTypes.is_ref_dtype(ref.dtype) then
    begin
        Result := gen_state_ops.scatter_add(ref, indices, updates, use_locking, name);
        Exit;
    end;
    raise TFException.Create('Not Implemented - scatter_add');
end;

class function state_ops.variable_op_v2(shape: TArray<Integer>; dtype: TF_DataType; name, container, shared_name: string): TFTensor;
begin
    Result := gen_state_ops.variable_v2(shape, dtype, name, container,shared_name)
end;

{ TResourceVariable }

class operator TResourceVariable.Implicit(t: TResourceVariable): ResourceVariable;
begin
    Result := t.FResourceHandle;
end;

function TResourceVariable.numpy: TNDArray;
begin
    Result := FResourceHandle.numpy;
end;

function TResourceVariable.value: TFTensor;
begin
    Result := FResourceHandle.value;
end;

class operator TResourceVariable.Implicit(t: ResourceVariable): TResourceVariable;
begin
    Result.FResourceHandle := t;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t + t1;
end;

class operator TResourceVariable.Add(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t + y;
end;

class operator TResourceVariable.Add(x: TFTensor; y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := y.value;
    Result := t + x;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Subtract(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t - t1;
end;

class operator TResourceVariable.Subtract(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t - y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Integer): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Single): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: Double): TFTensor;
begin
    var t : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x, y: TResourceVariable): TFTensor;
begin
    var t  : TTensor := x.value;
    var t1 : TTensor := y.value;
    Result := t * t1;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.Multiply(x: TResourceVariable; y: TNDArray): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t * y;
end;

class operator TResourceVariable.LessThan(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t < y;
end;

class operator TResourceVariable.GreaterThan(x: TResourceVariable; y: TFTensor): TFTensor;
begin
    var t  : TTensor := x.value;
    Result := t > y;
end;

function TResourceVariable.assign<T>(value: T; use_locking: Boolean; name: string; read_value: Boolean): TFTensor;
begin
    Result := FResourceHandle.assign<T>(value,use_locking, name, read_value)
end;

class operator TResourceVariable.Implicit(t: TResourceVariable): TEagerTensor;
begin
    Result := t.FResourceHandle._dense_var_to_tensor as TEagerTensor;
end;

class operator TResourceVariable.Implicit(t: TResourceVariable): TFTensor;
begin
    Result := t.FResourceHandle._dense_var_to_tensor;
end;



end.

