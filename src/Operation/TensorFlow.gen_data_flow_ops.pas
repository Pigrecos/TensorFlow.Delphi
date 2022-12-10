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

end.
