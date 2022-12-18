unit Keras.Activations;
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

            TF4D.Core.CApi,
            TensorFlow.DApi,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable;

type
  TActivation = Reference To function(features: TFTensor; name: string = ''): TFTensor;

   TActivations = class
     Linear : TActivation;
     Relu   : TActivation;
     Sigmoid: TActivation;
     Softmax: TActivation;
     Tanh   : TActivation;

     constructor Create;
   end;

implementation
    uses Tensorflow;

{ Activations }

constructor TActivations.Create;
begin
    Linear :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := features;
                end;

    Relu :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                    Result := tf.Context.ExecuteOp('Relu', name, ExecuteOpArgs.Create([features])).First;
                end;

    Sigmoid :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                   Result := tf.Context.ExecuteOp('Sigmoid', name, ExecuteOpArgs.Create([features])).First;
                end;

    Softmax :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := tf.Context.ExecuteOp('Softmax', name, ExecuteOpArgs.Create([features])).First;
                end;

    Tanh :=  function(features: TFTensor; name: string = ''): TFTensor
                begin
                  Result := tf.Context.ExecuteOp('Tanh', name, ExecuteOpArgs.Create([features])).First;
                end;
end;

end.
