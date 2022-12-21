unit TensorFlow.tensor_array_ops;
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

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.Tensors.Ragged;

type
  tensor_array_ops = record
     public
        /// <summary>
        /// Builds a TensorArray with a new `flow` tensor.
        /// </summary>
        /// <param name="old_ta"></param>
        /// <param name="flow"></param>
        /// <returns></returns>
        class function build_ta_with_new_flow(old_ta: TTensorArray; flow: TFTensor) : TTensorArray; overload; static;
        class function build_ta_with_new_flow(old_ta: TGraphTensorArray; flow: TFTensor) : TTensorArray; overload; static;

  end;
implementation
           uses Tensorflow;
{ tensor_array_ops }

class function tensor_array_ops.build_ta_with_new_flow(old_ta: TTensorArray; flow: TFTensor): TTensorArray;
begin
    var new_ta := tf.TensorArray(old_ta.dtype, 0, false, True, nil, old_ta.colocate_with_first_write_call, old_ta.infer_shape);
    Result := new_ta;
end;

class function tensor_array_ops.build_ta_with_new_flow(old_ta: TGraphTensorArray; flow: TFTensor): TTensorArray;
begin
    var new_ta := tf.TensorArray(old_ta.dtype, 0, false, True, nil, old_ta.colocate_with_first_write_call, old_ta.infer_shape);
    Result := new_ta;
end;

end.
