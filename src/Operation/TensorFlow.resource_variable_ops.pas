unit TensorFlow.resource_variable_ops;
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

interface
        uses System.SysUtils,
             Spring,
             spring.Collections.Dictionaries,
             Spring.Collections.Lists,

             TF4D.Core.CApi,
             TensorFlow.DApiBase,
             TensorFlow.DApi,
             TensorFlow.Variable,

             ProtoGen.types,
             ProtoGen.cppShapeInference;

type
  resource_variable_ops = record
    private
      /// <summary>
      /// Concats HandleData from tensors `handle` and `initial_value`.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="initial_value"></param>
      /// <returns></returns>
      class function  _combine_handle_data(handle: TFTensor; initial_value: TFTensor) : THandleData; static;
      class function  get_eager_safe_handle_data(handle: TFTensor) : THandleData; static;
      /// <summary>
      /// Sets the shape inference result HandleData on tensor.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="handle_data"></param>
      /// <param name="graph_mode"></param>
      class procedure _set_handle_shapes_and_types(tensor: TFTensor; handle_data: THandleData; graph_mode: Boolean);static;
    public
      /// <summary>
      /// Creates a variable handle with information to do shape inference.
      /// </summary>
      /// <param name="initial_value"></param>
      /// <param name="shape"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <param name="graph_mode"></param>
      /// <returns></returns>
      class function eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name: string; name: string; graph_mode: Boolean): TFTensor; Static;
      class function shape_safe_assign_variable_handle(tHandle: TFTensor;  shape: TArray<Integer>; value: TFTensor; name: string = '') : TFOperation; static;
      class function is_resource_variable(vVar :IVariableV1): Boolean; static;
      /// <summary>
      /// Create a new variable handle, optionally copying in `extra_handle_data`
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <param name="graph_mode"></param>
      /// <param name="initial_value"></param>
      /// <returns></returns>
      class function variable_handle_from_shape_and_dtype(shape: TFShape; dtype: TF_DataType; shared_name: string; name: string; graph_mode: Boolean; initial_value : TFTensor= nil) : TFTensor; Static;
  end;

implementation
        uses Oz.Pb.Classes,

             TensorFlow.Ops,
             Tensorflow.Utils,
             TensorFlow.gen_resource_variable_ops;

{ resorce_variable_ops }

class function resource_variable_ops.eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name, name: string; graph_mode: Boolean): TFTensor;
begin
    var dtype := TDtypes.as_base_dtype(initial_value.dtype);
    Result :=  variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name, graph_mode, initial_value);
end;

class function resource_variable_ops.is_resource_variable(vVar: IVariableV1): Boolean;
begin
    Result := vVar is ResourceVariable;
end;

class function resource_variable_ops.shape_safe_assign_variable_handle(tHandle: TFTensor; shape: TArray<Integer>; value: TFTensor; name: string): TFOperation;
begin
    var value_tensor := Tops.convert_to_tensor(value);
    Result           := gen_resource_variable_ops.assign_variable_op(tHandle, value_tensor, name)
end;

class function resource_variable_ops.variable_handle_from_shape_and_dtype(shape: TFShape; dtype: TF_DataType; shared_name, name: string; graph_mode: Boolean;
                                                 initial_value: TFTensor): TFTensor;
begin
    var container := Tops.get_default_graph.Container;
    var handle    := gen_resource_variable_ops.var_handle_op(dtype,shape, container, shared_name, name );
    if initial_value = nil then
        initial_value := handle;
    if graph_mode then
    begin
        var full_handle_data := _combine_handle_data(handle, initial_value);
        _set_handle_shapes_and_types(handle, full_handle_data, graph_mode);
        Result := handle;
        Exit;
    end else
    begin
        // We do not want two distinct ResourceVariable objects for the same
        // underlying resource in the runtime.
        // When in eager mode, explicitly ensure so here. When in graph mode, it's
        // ensured by always generating different variable names.
        var exists := gen_resource_variable_ops.var_is_initialized_op(handle);
        // We create an assert Op instead of checking right away in order to be
        // compatible with ASYNC execution mode. Further, since not all devices
        // support string tensors, we encode the assertion string in the Op name
        (*gen_logging_ops.assert(gen_math_ops.logical_not(exists),
            new[] { exists },
            name: "EagerVariableNameReuse");*)
        var handle_data : THandleData; handle_data.Init;
        var item : THandleShapeAndType;  item.init;
        item.Shape := TUtils.as_shape_proto(shape);
        item.Dtype := TDtypes.as_datatype_enum(dtype);
        handle_data.ShapeAndTypes.Add(@item);
        _set_handle_shapes_and_types(handle, handle_data, graph_mode);
        Result := handle;
    end;
end;

class function resource_variable_ops._combine_handle_data(handle, initial_value: TFTensor): THandleData;
begin
    var variable_handle_data := get_eager_safe_handle_data(initial_value);

    if initial_value.dtype <> Tdtypes.cvariant then
        Exit(variable_handle_data);
    raise TFException.Create('Not Implemented') ;
end;

class procedure resource_variable_ops._set_handle_shapes_and_types(tensor: TFTensor; handle_data: THandleData; graph_mode: Boolean);
begin
    if not graph_mode then
        Exit;
    var size := handle_data.ShapeAndTypes.Count;
    var types : TArray<TDataType> ; SetLength(types,size);
    var ranks : TArray<Integer>;   SetLength(ranks,size);
    for var i := 0 to size -1 do
    begin
        var shapeAndType := handle_data.ShapeAndTypes[i];
        types[i] := shapeAndType.Dtype;
        if shapeAndType.Shape.UnknownRank then ranks[i] := -1
        else                                   ranks[i] := shapeAndType.Shape.Dims.Count;
    end;
end;

class function resource_variable_ops.get_eager_safe_handle_data(handle: TFTensor): THandleData;
begin
    if handle.Handle = nil then
    begin
        var data : THandleData ; data.Init ;
        var item : THandleShapeAndType ; item.Init;
        item.Shape := TUtils.as_shape_proto(handle.shape);
        item.Dtype := TDtypes.as_datatype_enum(handle.dtype);
        data.ShapeAndTypes.Add(@item);
        Result := data;
    end else
    begin
        var protoByte := handle.BufferToArray;
        var Loader: TpbLoader; Loader.Init;
        Loader.Pb.Init(@protoByte[0],Length(protoByte),false);
        var protoHandle : THandleData;
        Loader.LoadHandleData(protoHandle);
        Result := protoHandle;
    end;
end;

end.
