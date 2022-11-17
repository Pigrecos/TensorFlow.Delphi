unit TensorFlow.resource_variable_grad;
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
         System.Generics.Collections,
         Generics.Defaults,
         System.Math,

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         Tensorflow.Gradient;

type
    resource_variable_grad = class
      private
        FGradFunction     : TArray<TGradFunc>;
      public
        constructor Create;
        destructor Destroy;  override;

        property GradFunction  : TArray<TGradFunc> read FGradFunction;
    end;

implementation
      uses Tensorflow,
           TensorFlow.Constant_op,
           Tensorflow.Utils,
           TensorFlow.Ops,
           TensorFlow.Tensor,
           Tensorflow.math_ops,
           Tensorflow.gen_array_ops,
           TensorFlow.gen_math_ops,
           Tensorflow.array_ops;

// [RegisterGradient("ReadVariableOp")]
function _ReadGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ grads[0] ];
end;

{ resource_variable_grad }

constructor resource_variable_grad.Create;
begin
    FGradFunction := [TGradFunc.Create('ReadVariableOp',  _ReadGrad)] ;
end;

destructor resource_variable_grad.Destroy;
begin
   FGradFunction := [];
  inherited;
end;

end.
