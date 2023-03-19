unit Keras.Regularizers;
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
            TensorFlow.DApi,

            Keras.Core;

type
  TL2 = class( TInterfacedObject, IRegularizer )
     protected
       Fl2 : Single;
     public
       constructor Create(_l2 : Single= 0.01);
       function Apply(args: RegularizerArgs): TFTensor;
  end;

  TL1 = class( TInterfacedObject, IRegularizer )
     protected
       Fl1 : Single;
     public
       constructor Create(_l1 : Single= 0.01);
       function Apply(args: RegularizerArgs): TFTensor;
  end;

  TL1L2 = class( TInterfacedObject, IRegularizer )
     protected
       Fl1 : Single;
       Fl2 : Single;
     public
       constructor Create(_l1 : Single= 0.0; _l2: Single = 0.0);
       function Apply(args: RegularizerArgs): TFTensor;
  end;

implementation
        uses Tensorflow,
             TensorFlow.Tensor,
             TensorFlow.Operations;

{ L2 }

constructor TL2.Create(_l2: Single);
begin
    Fl2 := _l2;
end;

function TL2.Apply(args: RegularizerArgs): TFTensor;
begin
     Result := Fl2 * TTensor( math_ops.reduce_sum(math_ops.square(args.X)) );
end;

{ TL1 }

constructor TL1.Create(_l1: Single);
begin
    Fl1 := _l1;
end;

function TL1.Apply(args: RegularizerArgs): TFTensor;
begin
    Result := Fl1 * TTensor( math_ops.reduce_sum(math_ops.abs(args.X)) );
end;

{ TL1L2 }

constructor TL1L2.Create(_l1, _l2: Single);
begin
    Fl1 := _l1;
    Fl2 := _l2;
end;

function TL1L2.Apply(args: RegularizerArgs): TFTensor;
begin
    var regularization : TTensor := tf.constant(Single(0.0), args.X.dtype);
    regularization := regularization + ( Fl1 * TTensor(math_ops.reduce_sum(math_ops.abs   (args.X))) );
    regularization := regularization + ( Fl2 * TTensor(math_ops.reduce_sum(math_ops.square(args.X))) );
    Result := regularization;
end;

end.
