unit TensorFlow.gen_data_flow_ops;
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
             TF4D.Core.CApi,
             TensorFlow.DApi,
             Numpy.Axis,

             TensorFlow.Context ;

type
  gen_data_flow_ops = record
    private

    public
      class function dynamic_stitch(indices: TArray<TFTensor>; data: TArray<TFTensor>; name: string = ''): TFTensor; static;
      class function tensor_array_size_v3(handle: TFTensor; flow_in: TFTensor; name: string = ''): TFTensor; static;
      class function tensor_array_gather_v3(handle: TFTensor; indices: TFTensor; flow_in: TFTensor; dtype: TF_DataType; element_shape : PTFShape= nil; name: string = ''): TFTensor; static;
      class function tensor_array_v3<T>(size: T; dtype: TF_DataType = DtInvalid; element_shape : PTFShape= nil; dynamic_size: Boolean = false; clear_after_read: Boolean = true;
                                        identical_element_shapes : Boolean= false; tensor_array_name: string = ''; name: string = '') : Tuple<TFTensor, TFTensor>; static;
      /// <summary>
      /// Read an element from the TensorArray into output `value`.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="index"></param>
      /// <param name="flow_in"></param>
      /// <param name="dtype"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function tensor_array_read_v3(handle: TFTensor; index: TFTensor; flow_in: TFTensor; dtype: TF_DataType; name: string = ''): TFTensor; static;
      class function tensor_array_write_v3(handle: TFTensor; index: TFTensor; value: TFTensor; flow_in: TFTensor; name : string= ''): TFTensor; static;
  end;

implementation
         uses Tensorflow.Utils,
              Tensorflow;

{ gen_data_flow_ops }

class function gen_data_flow_ops.dynamic_stitch(indices, data: TArray<TFTensor>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('DynamicStitch', name, [GetArg('indices',indices), GetArg('data',data)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_gather_v3(handle, indices, flow_in: TFTensor; dtype: TF_DataType; element_shape: PTFShape; name: string): TFTensor;
begin
    var sShape : TFShape := default(TFShape);
    if Assigned(element_shape) then
       sShape := element_shape^;

    var _op := tf.OpDefLib._apply_op_helper('TensorArrayGatherV3', name, [GetArg('handle',handle),
                                                                          GetArg('indices',indices),
                                                                          GetArg('dtype',dtype),
                                                                          GetArg('element_shape', TValue.From<TFShape>(sShape)),
                                                                          GetArg('flow_in',flow_in)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_read_v3(handle, index, flow_in: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArrayReadV3', name, [GetArg('handle',handle),
                                                                        GetArg('index',index),
                                                                        GetArg('flow_in',flow_in),
                                                                        GetArg('dtype', dtype)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_size_v3(handle, flow_in: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArraySizeV3', name, [GetArg('handle',handle), GetArg('flow_in',flow_in)]);
    Result  := _op.output;
end;

class function gen_data_flow_ops.tensor_array_v3<T>(size: T; dtype: TF_DataType; element_shape: PTFShape; dynamic_size, clear_after_read, identical_element_shapes: Boolean;
  tensor_array_name, name: string): Tuple<TFTensor, TFTensor>;
begin
    var sShape : TFShape := default(TFShape);
    if Assigned(element_shape) then
       sShape := element_shape^;

     var _op := tf.OpDefLib._apply_op_helper('TensorArrayV3', name, [GetArg('size', TValue.From<T>(size)),
                                                                     GetArg('dtype',dtype),
                                                                     GetArg('element_shape',TValue.From<TFShape>(sShape)),
                                                                     GetArg('dynamic_size', dynamic_size),
                                                                     GetArg('clear_after_read', clear_after_read),
                                                                     GetArg('identical_element_shapes', identical_element_shapes),
                                                                     GetArg('tensor_array_name', tensor_array_name)]);
    Result  :=  Tuple.Create(_op.outputs[0], _op.outputs[1]);
end;

class function gen_data_flow_ops.tensor_array_write_v3(handle, index, value, flow_in: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('TensorArrayWriteV3', name, [GetArg('handle',handle),
                                                                        GetArg('index',index),
                                                                        GetArg('value',value),
                                                                        GetArg('flow_in', flow_in)]);
    Result  := _op.output;
end;

end.
