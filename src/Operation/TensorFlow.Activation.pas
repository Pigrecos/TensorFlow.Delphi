unit TensorFlow.Activation;
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
         System.Math,
         Spring,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         Tensorflow.Utils,
         TensorFlow.Context;

type
  IActivation = interface
    ['{28DC26C6-54AF-4C11-BFB0-1219F3CAA365}']

    function Activate(features: TFTensor; name: string = ''): TFTensor;
  end;

  sigmoid = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  tanh = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  leakyrelu = class(TInterfacedObject, IActivation)
    private
      Falpha : Single;
    public
     constructor Create(alpha: single = 0.3);
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  elu = class(TInterfacedObject, IActivation)
    private
      Falpha : Single;
    public
     constructor Create(alpha: single = 0.1);
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  softmax = class(TInterfacedObject, IActivation)
    private
      Faxis : Integer;
    public
     constructor Create(axis: Integer = -1);
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  softplus = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  softsign = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  swish = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  linear = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  exponential = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  relu = class(TInterfacedObject, IActivation)
    private
      Fthreshold : Single;
      Falpha     : Single;
      FmaxValue  : Nullable<Single>;
    public
     constructor Create(threshold: Single = 0; alpha: Single = 0.2; max_value: PSingle = nil);
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  selu = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

  hard_sigmoid = class(TInterfacedObject, IActivation)
    private

    public
     function Activate(x: TFTensor; name: string = ''): TFTensor;
  end;

implementation
         uses Tensorflow,
         TensorFlow.Constant_op,
              TensorFlow.Tensor,
              Tensorflow.gen_ops,
              Tensorflow.array_ops,
              Tensorflow.math_ops,
              TensorFlow.nn_ops;
{ sigmoid }

function sigmoid.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.sigmoid(x);
end;

{ tanh }

function tanh.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.tanh(x);
end;

{ leakyrelu }

constructor leakyrelu.Create(alpha: single);
begin
    Falpha := alpha;
end;

function leakyrelu.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := nn_ops.leaky_relu(x, Falpha);
end;

{ elu }

constructor elu.Create(alpha: single);
begin
    Falpha := alpha;
end;

function elu.Activate(x: TFTensor; name: string): TFTensor;
begin
    var res := gen_ops.elu(x);
    if Abs(Falpha - 0.1) < 0.00001 then
     Exit(res);
    Result := array_ops.where(TTensor(x) > 0, res, Falpha * TTensor(res));
end;

{ softmax }

function softmax.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := nn_ops.softmax(x, Faxis);
end;

constructor softmax.Create(axis: Integer);
begin
    Faxis := axis;
end;

{ softplus }

function softplus.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_ops.softplus(x);
end;

{ softsign }

function softsign.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := gen_ops.softsign(x);
end;

{ swish }

function swish.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.multiply(x, tf.nn.sigmoid(x));
end;

{ linear }

function linear.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := x;
end;

{ exponential }

function exponential.Activate(x: TFTensor; name: string): TFTensor;
begin
    Result := tf.exp(x, name);
end;

{ relu }

constructor relu.Create(threshold, alpha: Single; max_value: PSingle);
begin
    Fthreshold := threshold;
    Falpha     := alpha;
    if max_value <> nil then
        FmaxValue := max_value^;
end;

function relu.Activate(x: TFTensor; name: string): TFTensor;
var
 tX : TTensor;
begin
    //based on keras/backend.py
    if Abs(Falpha) > 0.000001 then
    begin
        if (not FmaxValue.HasValue) and (Abs(Fthreshold) < 0.0001) then
        begin
            Result := nn_ops.leaky_relu(x, Falpha);
            Exit;
        end;
    end;
    tX := x;
    var negative_part : TFTensor;
    if Abs(Fthreshold) > 0.000001 then
    begin
        negative_part := gen_ops.relu(-tX + Fthreshold);
    end else
    begin
        negative_part := gen_ops.relu(-tX + Fthreshold);
    end;
    if Abs(Fthreshold) > 0.000001 then
    begin
        x := TTensor(x) * math_ops.cast(tf.greater(x, Fthreshold), TF_DataType.TF_FLOAT);
    end
    else if Abs(FmaxValue.Value - 6) < 0.0001 then
    begin
        x := gen_ops.relu6(x);
    end else
    begin
        x := gen_ops.relu(x);
    end;
    var clip_max : Boolean := FmaxValue.HasValue;
    if clip_max then
    begin
        var maxval : TFTensor := constant_op.constant(FmaxValue.value, TDTypes.as_base_dtype(x.dtype), 'Const');
        var zero              := constant_op.constant(Single(0.0),     TDTypes.as_base_dtype(x.dtype), 'Const');
        x := gen_ops.clip_by_value(x, zero, maxval);
    end;
    if Abs(Falpha) > 0.00001 then
    begin
        var a := constant_op.constant(Falpha, TDTypes.as_base_dtype(x.dtype), 'Const');
        x := x - (TTensor(a) * negative_part);
    end;
    Result := x;
end;

{ selu }

function selu.Activate(x: TFTensor; name: string): TFTensor;
const
  alpha : Single = 1.6732632423543772848170429916717;
  scale : Single = 1.0507009873554804934193349852946;
begin
     Result := scale * TTensor(elu.Create(alpha).Activate(x, name));
end;

{ hard_sigmoid }

function hard_sigmoid.Activate(x: TFTensor; name: string): TFTensor;
begin
    x := (0.2 * TTensor(x)) + 0.5;
    var zero := tf.convert_to_tensor(Single(0.0),  TDTypes.as_base_dtype(x.dtype));
    var one  := tf.convert_to_tensor(Single(1.0),  TDTypes.as_base_dtype(x.dtype));
    Result := tf.clip_by_value(x, zero, one);
end;

end.
