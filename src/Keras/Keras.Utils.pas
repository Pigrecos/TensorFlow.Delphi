unit Keras.Utils;
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

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,
            TensorFlow.Variable;

type
  base_layer_utils = record
    private

    public
        /// <summary>
        /// Adds a new variable to the layer.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        class function make_variable(args: VariableArgs): IVariableV1; static;
  end;


implementation
          uses Tensorflow,
               Tensorflow.Utils,
               TensorFlow.Initializer,

               ProtoGen.variable;

{ base_layer_utils }

class function base_layer_utils.make_variable(args: VariableArgs): IVariableV1;
begin
    var init_val : TFunc<TFTensor> := function: TFTensor
                                        begin
                                            Result := args.Initializer.Apply(InitializerArgs.Create(args.Shape, args.DType));
                                        end;

    var variable_dtype := Tdtypes.as_base_dtype(args.DType);

    var s := args.Shape;
    var Ps : PTFShape := nil;
    if not s.IsNil then
       Ps := @s;

    Result := tf.Variable(init_val, args.Trainable, args.ValidateShape, args.UseResource, args.Name, variable_dtype, TVariableAggregation.VARIABLE_AGGREGATION_NONE,Ps );
end;

end.
