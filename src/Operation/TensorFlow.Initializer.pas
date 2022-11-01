unit TensorFlow.Initializer;
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
         Spring.Collections,
         Spring.Collections.Enumerable,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase;

type
  InitializerArgs = class
     private

     public
       Name       : string;
       Shape      : TFShape;
       DType      : TF_DataType;
       VerifyShape: Boolean;

       constructor Create(_shape: TFShape; _dtype: TF_DataType = DtInvalid; _verify_shape: Boolean = false; _name: string = '');
  end;

  IInitializer = interface
    ['{3A2E973B-FC81-4836-B87B-22F5E41952BB}']

     function Apply(args: InitializerArgs): TFTensor;
  end;

  /// <summary>
  /// Initializer capable of adapting its scale to the shape of weights tensors.
  /// </summary>
  VarianceScaling = class (TInterfacedObject, IInitializer)
     private
        {$HINTS OFF}
        Fscale       : Single;
        Fmode        : string;
        Fdistribution: string;
        Fseed        : pInteger;
        Fdtype       : TF_DataType;
        Funiform     : Boolean;
         {$HINTS ON}

        function _compute_fans(shape: TArray<Integer>): TUple<Integer, Integer>;
     public
        constructor Create(factor: Single = 2.0; mode: string = 'FAN_IN'; uniform : Boolean= false; seed: pInteger = nil; dtype: TF_DataType = TF_FLOAT) ;
        function Apply(args: InitializerArgs): TFTensor;
  end;

  GlorotUniform = class(VarianceScaling)
     private

     public
        constructor Create(scale: Single = 1.0; mode: string = 'FAN_AVG'; uniform : Boolean= True; seed: pInteger = nil; dtype: TF_DataType = TF_FLOAT) ;
  end;

  Zeros = class (TInterfacedObject, IInitializer)
     private
        FShape       : TFShape;
        Fdtype       : TF_DataType;

     public
        constructor Create(shape: PTFShape = nil; dtype: TF_DataType = TF_FLOAT) ;
        function Apply(args: InitializerArgs): TFTensor;
  end;

  Ones = class (TInterfacedObject, IInitializer)
     private
        Fdtype       : TF_DataType;

     public
        constructor Create(dtype: TF_DataType = DtInvalid) ;
        function Apply(args: InitializerArgs): TFTensor;
  end;

  RandomUniform = class (TInterfacedObject, IInitializer)
     private
        Fseed        : pInteger;
        FMinVal      : Single;
        FMaxVal      : Single;
        Fdtype       : TF_DataType;

     public
        constructor Create(dtype: TF_DataType = TF_FLOAT; minval: Single = -0.05; maxval: Single = 0.05; seed : PInteger= nil) ;
        function Apply(args: InitializerArgs): TFTensor;
  end;

  Orthogonal = class (TInterfacedObject, IInitializer)
     private

     public
        function Apply(args: InitializerArgs): TFTensor;
  end;

  Constant<T> = class (TInterfacedObject, IInitializer)
     private
        Fdtype       : TF_DataType;
        FValue       : T;
        FVerify_shape: Boolean;

     public
        constructor Create(value :T ; dtype: TF_DataType = TF_FLOAT; verify_shape: Boolean = false) ;
        function Apply(args: InitializerArgs): TFTensor;
  end;

implementation
        uses TensorFlow.Constant_op,
             Tensorflow.Utils,
             TensorFlow.Context,
         TensorFlow.random_ops,
         Tensorflow.array_ops;

{ InitializerArgs }

constructor InitializerArgs.Create(_shape: TFShape; _dtype: TF_DataType; _verify_shape: Boolean; _name: string);
begin
    Shape       := _shape;
    DType       := _dtype;
    VerifyShape := _verify_shape;
    Name        := _name;
end;

{ VarianceScaling }

constructor VarianceScaling.Create(factor: Single; mode: string; uniform: Boolean; seed: pInteger; dtype: TF_DataType);
begin
    if not TDtypes.is_floating(dtype) then
       raise TFException.Create('Cannot create initializer for non-floating point type.');
    var s : TArray<String> := ['FAN_IN', 'FAN_OUT', 'FAN_AVG'];
    if not TArray.Contains<String>(s,mode) then
       raise TFException.Create('Unknown'+ mode + 'valid is [FAN_IN, FAN_OUT, FAN_AVG]');
    if factor < 0 then
       raise TFException.Create('"scale" must be positive float.');
    Fscale   := factor;
    Fmode    := mode;
    Fseed    := seed;
    Fdtype   := dtype;
    Funiform := uniform;
end;

function VarianceScaling._compute_fans(shape: TArray<Integer>): Tuple<Integer,Integer>;
begin
    if Length(shape) < 1 then
        Exit ( Tuple<Integer,Integer>.Create(1, 1) );
    if Length(shape) = 1 then
       Exit ( Tuple<Integer,Integer>.Create(shape[0], shape[0]) );
    if Length(shape) = 2 then
       Exit ( Tuple<Integer,Integer>.Create(shape[0], shape[1]) )
    else begin
        // Assuming convolution kernels (2D, 3D, or more).
        // kernel shape: (..., input_depth, depth)
        var receptive_field_size : Integer := 1;
        var lstShape := TCollections.CreateList<Integer>(shape);
        for var dim in lstShape.Take(lstShape.Count - 2) do
            receptive_field_size := receptive_field_size * dim;

        var fan_in  := lstShape[lstShape.Count - 2] * receptive_field_size;
        var fan_out := lstShape[lstShape.Count - 1] * receptive_field_size;
        Result := Tuple<Integer,Integer>.Create(fan_in, fan_out);
    end;
end;

function VarianceScaling.Apply(args: InitializerArgs): TFTensor;
begin
    if args.DType = TF_DataType.DtInvalid then
        args.DType := Fdtype;
    var n : Single := 0;
    var t := _compute_fans(args.Shape);
    var fan_in := t.Value1;
    var fan_out:= t.Value2;
    if      Fmode = 'FAN_IN' then  n := fan_in
    else if Fmode = 'FAN_OUT' then n := fan_out
    else if Fmode = 'FAN_AVG' then n := (fan_in + fan_out) / 2.0;
    if Funiform then
    begin
        var limit := Single( Sqrt(3.0 * Fscale / n) );
        Result := random_ops.random_uniform(args.Shape, -limit, limit, args.DType);
    end else
    begin
        var trunc_stddev := Single( Sqrt(1.3 * Fscale / n) );
        Result := random_ops.truncated_normal(args.Shape, 0.0, trunc_stddev, args.DType, Fseed);
    end;
end;

{ GlorotUniform }

constructor GlorotUniform.Create(scale: Single; mode: string; uniform: Boolean; seed: pInteger; dtype: TF_DataType);
begin
   inherited Create(scale, mode, uniform, seed, dtype);
end;

{ Zeros }

constructor Zeros.Create(shape: PTFShape; dtype: TF_DataType);
begin
    if shape <> nil then
      Fshape := shape^;
    Fdtype := dtype;
end;

function Zeros.Apply(args: InitializerArgs): TFTensor;
begin
    if args.DType = TF_DataType.DtInvalid then
        args.DType := Fdtype;
    if args.Shape.IsNull then
        args.Shape := Fshape;
    Result := array_ops.zeros(args.Shape, Fdtype);
end;

{ Ones }

constructor Ones.Create(dtype: TF_DataType);
begin
     Fdtype := dtype;
end;

function Ones.Apply(args: InitializerArgs): TFTensor;
begin
    if args.DType = TF_DataType.DtInvalid then
        args.DType := Fdtype;

    Result := array_ops.ones(args.Shape, Fdtype);
end;

{ RandomUniform }

constructor RandomUniform.Create(dtype: TF_DataType; minval, maxval: Single; seed: PInteger);
begin
    Fdtype  := dtype;
    Fminval := minval;
    Fmaxval := maxval;
    Fseed   := seed;
end;

function RandomUniform.Apply(args: InitializerArgs): TFTensor;
begin
    if (args.DType = TF_DataType.DtInvalid) then
                args.DType := Fdtype;
    Result := random_ops.random_uniform(args.Shape, Fminval, Fmaxval, Fdtype, Fseed)
end;

{ Orthogonal }

function Orthogonal.Apply(args: InitializerArgs): TFTensor;
begin
   raise TFException.Create('Not Implemented - Orthogonal --> IInitializer)');
end;

{ Constant<T> }

constructor Constant<T>.Create(value: T; dtype: TF_DataType; verify_shape: Boolean);
begin
    Fvalue        := value;
    Fdtype        := dtype;
    Fverify_shape := verify_shape;
end;

function Constant<T>.Apply(args: InitializerArgs): TFTensor;
begin
    if args.DType = TF_DataType.DtInvalid then
       args.DType := Fdtype;
    args.VerifyShape := Fverify_shape;
    var sh : TFShape := args.Shape;
    Result := constant_op.constant(TValue.From<T>(Fvalue), args.DType, @sh, args.VerifyShape, false,'Const')
end;

end.
