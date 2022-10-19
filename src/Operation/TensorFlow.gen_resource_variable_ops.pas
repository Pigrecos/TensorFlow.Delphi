unit TensorFlow.gen_resource_variable_ops;
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
         Spring,
         spring.Collections.Dictionaries,
         Spring.Collections.Lists,

         TF4D.Core.CApi,
         TensorFlow.DApiBase,
         TensorFlow.DApi,
         TensorFlow.Variable;

type
  gen_resource_variable_ops = record

    public
      class function assign_sub_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      /// <summary>
      /// Adds a value to the current value of a variable.
      /// </summary>
      /// <param name="resource"></param>
      /// <param name="value"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function assign_add_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      class function assign_variable_op(resource: TFTensor; value: TFTensor; name: string = ''): TFOperation; static;
      class function var_is_initialized_op(resource: TFTensor; name: string = ''): TFTensor; static;
      /// <summary>
      /// Creates a handle to a Variable resource.
      /// </summary>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function var_handle_op(dtype: TF_DataType; shape: TFShape; container: string = ''; shared_name: string = ''; name: string = ''): TFTensor; static;
      class function destroy_resource_op(resource: TFTensor; ignore_lookup_error: Boolean = true; name: string = ''): TFTensor; static;
      /// <summary>
      /// Reads the value of a variable.
      /// </summary>
      /// <param name="resource"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function read_variable_op(resource: TFTensor; dtype: TF_DataType; name: string = ''): TFTensor; static;
      class function resource_gather(resource: TFTensor; indices: TFTensor; dtype: TF_DataType; batch_dims: Integer = 0; validate_indices: Boolean = true; name: string = ''): TFTensor; static;
  end;

implementation
     uses Tensorflow,
           Numpy,
           TensorFlow.Context,
           TensorFlow.Ops,
           Tensorflow.gen_array_ops,
           TensorFlow.gen_math_ops,
           Tensorflow.NameScope,
           Tensorflow.Utils,
           TensorFlow.Constant_op,
           TensorFlow.EagerTensor,
           TensorFlow.EagareRunner;

{ gen_resource_variable_ops }

class function gen_resource_variable_ops.assign_sub_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignSubVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    Result := nil;
end;

class function gen_resource_variable_ops.assign_add_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignAddVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    var _op := tf.OpDefLib._apply_op_helper('AssignAddVariableOp', name, [ GetArg('resource',resource), GetArg('value',value) ]);
    Result := _op;
end;

class function gen_resource_variable_ops.assign_variable_op(resource, value: TFTensor; name: string): TFOperation;
begin
    if tf.Context.executing_eagerly then
    begin
        tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('AssignVariableOp', name, [resource, value]));
        Exit(nil);
    end;
    var _op := tf.OpDefLib._apply_op_helper('AssignVariableOp', name, [ GetArg('resource',resource), GetArg('value',value) ]);
    Result := _op;
end;

class function gen_resource_variable_ops.var_is_initialized_op(resource: TFTensor; name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        var res := tf.Runner.TFE_FastPathExecute(TFastPathOpExecInfo.Create('VarIsInitializedOp', name, [resource]));
        Exit( res[0] );
    end;
    var _op := tf.OpDefLib._apply_op_helper('VarIsInitializedOp', name, [ GetArg('resource',resource) ]);
    Result := _op.Output;
end;

class function gen_resource_variable_ops.var_handle_op(dtype: TF_DataType; shape: TFShape; container, shared_name, name: string): TFTensor;
begin
    if tf.Context.executing_eagerly then
    begin
        var pdtype : TParameter;
        pdtype.sNome := 'dtype';
        pdtype.vValue:= TValue.From<Integer>( Ord(dtype) );

        var pshape : TParameter;
        pshape.sNome := 'shape';
        pshape.vValue:= TValue.From< TArray<Int64> >(shape.Dims);

        var pcontainer : TParameter;
        pcontainer.sNome := 'container';
        pcontainer.vValue:= container;

        var pshared_name : TParameter;
        pshared_name.sNome := 'shared_name';
        pshared_name.vValue:= shared_name;

        var pallowed_devices : TParameter;
        pallowed_devices.sNome := 'allowed_devices';
        var v : TArray<AnsiString> :=[];
        pallowed_devices.vValue :=   TValue.From<  TArray<AnsiString> >(v);

        var dAtrr := TUtils.ConvertToDict([pdtype,pshape,pcontainer,pshared_name,pallowed_devices]) ;
        var OpExecInfo := TFastPathOpExecInfo.Create('VarHandleOp', name,[]);
        OpExecInfo.attrs := dAtrr;

        var res := tf.Runner.TFE_FastPathExecute( OpExecInfo );
        Exit( res[0] );
    end;

    var _op := tf.OpDefLib._apply_op_helper('VarHandleOp', name, [ GetArg('dtype',dtype), GetArg('shape',TValue.From< TFShape >(shape)), GetArg('container',container), GetArg('shared_name',shared_name) ]);
    Result := _op.Output;
end;

class function gen_resource_variable_ops.destroy_resource_op(resource: TFTensor; ignore_lookup_error: Boolean; name: string): TFTensor;
begin
    Result :=  tf.Context.ExecuteOp('DestroyResourceOp', name, ExecuteOpArgs.Create([resource])
                                                      .SetAttributes(['ignore_lookup_error', ignore_lookup_error ])).FirstOrDefault(nil);
end;

class function gen_resource_variable_ops.read_variable_op(resource: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    Result :=  tf.Context.ExecuteOp('ReadVariableOp', name, ExecuteOpArgs.Create([resource])
                                                      .SetAttributes(['dtype', dtype ])).FirstOrDefault(nil);
end;

class function gen_resource_variable_ops.resource_gather(resource, indices: TFTensor; dtype: TF_DataType; batch_dims: Integer; validate_indices: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('VarHandleOp', name, [ GetArg('resource',resource),
                                                                   GetArg('indices',indices),
                                                                   GetArg('dtype',dtype),
                                                                   GetArg('batch_dims',batch_dims),
                                                                   GetArg('validate_indices',validate_indices) ]);
    Result := _op.Output;
end;

end.
