unit TensorFlow.Initializer;
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
         system.Generics.Collections,
         System.Math,

         Spring,
         Spring.Collections,

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

  IInitializer = class abstract

     function Apply(args: InitializerArgs): TFTensor; virtual;  abstract;
  end;

  /// <summary>
  /// Initializer capable of adapting its scale to the shape of weights tensors.
  /// </summary>
  VarianceScaling = class (IInitializer)
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
        function Apply(args: InitializerArgs): TFTensor; override;
  end;

  GlorotUniform = class(VarianceScaling)
     private

     public
        constructor Create(scale: Single = 1.0; mode: string = 'FAN_AVG'; uniform : Boolean= True; seed: pInteger = nil; dtype: TF_DataType = TF_FLOAT) ;
  end;

  Zeros = class (IInitializer)
     private
        FShape       : TFShape;
        Fdtype       : TF_DataType;

     public
        constructor Create(shape: PTFShape = nil; dtype: TF_DataType = TF_FLOAT) ;
        function Apply(args: InitializerArgs): TFTensor; override;
  end;

  Ones = class (IInitializer)
     private
        Fdtype       : TF_DataType;

     public
        constructor Create(dtype: TF_DataType = DtInvalid) ;
        function Apply(args: InitializerArgs): TFTensor; override;
  end;

  RandomNormal = class (IInitializer)
     private
        Fseed        : pInteger;
        Fmean        : Single;
        Fstddev      : Single;
        Fdtype       : TF_DataType;
        Fconfig      : TDictionary<string, TValue>;
        FClassName   : string;
     public

        constructor Create(mean : Single= 0.0; stddev: Single = 0.05; seed : PInteger= nil; dtype: TF_DataType = TF_FLOAT) ;
        destructor Destroy; override;
        function Apply(args: InitializerArgs): TFTensor;override;

        property cClassName : string read FClassName;
  end;

  RandomUniform = class (IInitializer)
     private
        Fseed        : pInteger;
        FMinVal      : Single;
        FMaxVal      : Single;
        Fdtype       : TF_DataType;

     public
        constructor Create(dtype: TF_DataType = TF_FLOAT; minval: Single = -0.05; maxval: Single = 0.05; seed : PInteger= nil) ;
        function Apply(args: InitializerArgs): TFTensor;override;
  end;

  Orthogonal = class (IInitializer)
     private
        FGain : Single;
        FSeed : pInteger;

        function _generate_init_val(shape: TFShape; dtype: TF_DataType): TFTensor;
     public
        constructor Create(gain: Single = 1.0; seed : pInteger= nil);
        function Apply(args: InitializerArgs): TFTensor; override;
  end;

  Constant<T> = class (IInitializer)
     private
        Fdtype       : TF_DataType;
        FValue       : T;
        FVerify_shape: Boolean;

     public
        constructor Create(value :T ; dtype: TF_DataType = TF_FLOAT; verify_shape: Boolean = false) ;
        function Apply(args: InitializerArgs): TFTensor;override;
  end;

implementation
        uses Tensorflow.Utils,
             Tensorflow.Core,
             TensorFlow.Operations,
             TensorFlow.Tensor,
             Tensorflow;

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

constructor Orthogonal.Create(gain: Single; seed: pInteger);
begin
    FGain := gain ;
    FSeed := seed;
end;

function Orthogonal.Apply(args: InitializerArgs): TFTensor;
begin
   var tipo : TF_DataType := TF_FLOAT;
   if args.DType <>  DtInvalid then   tipo := args.DType;

   Result := _generate_init_val(args.Shape, tipo);
end;

function Orthogonal._generate_init_val(shape: TFShape; dtype: TF_DataType): TFTensor;
begin
    var num_rows : Int64 := 1;
    for var i := 0 to Length(shape.dims) -2 do
    begin
        num_rows := num_rows * shape.dims[i];
    end;
    var num_cols := shape.dims[ High(shape.dims) ];
    var flat_shape := TFShape.Create([ Max(num_cols, num_rows), Min(num_cols, num_rows) ]);

    var a := tf.random.stateless_normal(flat_shape, 0.0, 1.0, dtype);
    // Compute the qr factorization
    var q_r := tf.linalg.qr(a, false);
    var q := q_r[0];
    var r:= q_r[1];
    // Make Q uniform
    var d := tf.linalg.tensor_diag_part(r);
    q := TTensor(q)  * tf.sign(d);

    if num_rows < num_cols then
    begin
        // q = tf.linalg.matrix_transpose(q);
        raise Exception.Create('Not Implemented');
    end;

    Result := Fgain * TTensor( tf.reshape(q, shape) );
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

{ RandomNormal }

constructor RandomNormal.Create(mean, stddev: Single; seed: PInteger; dtype: TF_DataType);
begin
    FClassName := 'RandomNormal';

    Fmean   := mean;
    Fstddev := stddev;
    Fseed   := seed;
    Fdtype  := dtype;

    Fconfig := TDictionary<string, TValue>.Create;

    Fconfig.Add('mean', Fmean);
    Fconfig.Add('stddev', Fstddev);
    Fconfig.Add('seed', TValue.From<PInteger>(Fseed));

end;

destructor RandomNormal.Destroy;
begin
    Fconfig.Free;
end;

function RandomNormal.Apply(args: InitializerArgs): TFTensor;
begin
    if args.DType = TF_DataType.DtInvalid then
       args.DType := Fdtype;

    Result := random_ops.random_normal(args.Shape, Fmean, Fstddev, args.DType, Fseed);
end;

end.
