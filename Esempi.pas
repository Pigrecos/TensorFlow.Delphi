unit Esempi;
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
     uses Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
          Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti,

          spring,

          TF4D.Core.CApi,

          Numpy,
          Tensorflow,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          Tensorflow.Utils,
          TensorFlow.Ops,
          TensorFlow.Context,
          Tensorflow.NameScope,
          TensorFlow.EagerTensor,

          Keras.KerasApi,

          TensorFlow.Variable,
          TensorFlow.Tensor,
          NumPy.NDArray;

type
  LinearRegression = class
    private

    public
       training_epochs : Integer;
       learning_rate   : Single;
       display_step    : Integer;

       n_samples       : Integer;
       train_X, train_Y: NDArray;

       constructor Create;
       procedure PrepareData;
       function  Run(mmo1: TMemo): Boolean;
  end;

  LinearRegressionEager = class
    private

    public
       training_epochs : Integer;
       training_steps  : Integer;
       learning_rate   : Single;
       display_step    : Integer;

       n_samples       : Integer;
       train_X, train_Y: TNDArray;

       constructor Create;
       procedure PrepareData;
       function  Run(mmo1: TMemo): Boolean;
  end;

  EagerModeTestBase = class
      constructor Create;
      procedure TestInit;
      function  Equal(f1: Single; f2: Single): Boolean; overload;
      function  Equal(f1: TArray<Single>; f2: TArray<Single>): Boolean; overload;
      function  Equal(f1: TArray<Double>; f2: TArray<Double>): Boolean; overload;

      procedure clip_by_global_norm;
  end;

  ActivationFunctionTest = class(EagerModeTestBase)
    public
      a : TFTensor;

      constructor Create;
      procedure Sigmoid;
      procedure ReLU;
      procedure TanH;
  end;

  BitwiseApiTest = class(EagerModeTestBase)
     public
       constructor Create;
       procedure BitwiseAnd;
       procedure BitwiseOr;
       procedure BitwiseXOR;
       procedure Invert;
       procedure LeftShift;
       procedure RightShift;
  end;

  ConstantTest = class(EagerModeTestBase)
      public
        constructor Create;
        procedure ScalarConst;
        procedure ZerosConst;
        procedure OnesConst;
        procedure OnesToHalves;
        procedure NDimConst;
        procedure Multiply;
        procedure Reshape;
  end;


implementation
        uses DUnitX.TestFramework;

{ LinearRegression }

constructor LinearRegression.Create;
begin
    training_epochs := 1000;

    // Parameters
    learning_rate := 0.01;
    display_step  := 50;
end;

procedure LinearRegression.PrepareData;
begin
    train_X := np.np_array<Single>([3.3, 4.4,  5.5,  6.71, 6.93,  4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]);
    train_Y := np.np_array<Single>([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465,  1.65,  2.904, 2.42,  2.94, 1.3]);
    n_samples := train_X.shape[0];

end;

function LinearRegression.Run(mmo1: TMemo): Boolean;
begin
    tf.compat.v1.disable_eager_execution;

    PrepareData;

    // tf Graph Input
    var X : TTensor := tf.placeholder(tf.float32_t);
    var Y : TTensor := tf.placeholder(tf.float32_t);

    // Set model weights
    // We can set a fixed init value in order to debug
    // var rnd1 = rng.randn<float>();
    // var rnd2 = rng.randn<float>();
    var W  := tf.Variable(Single(-0.06), 'weight');
    var b  := tf.Variable(Single(-0.73), 'bias');

    // Construct a linear model
    var pred : TTensor := tf.add(tf.multiply(X, W), b);
    //var pred1 := (X * W) + b;  OK

    // Mean squared error
    var cost := TTensor(tf.reduce_sum(tf.pow(pred - Y, 2.0)))  / (2.0 * n_samples) ;

    // Gradient descent
    // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    var optimizer := tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

    // Initialize the variables (i.e. assign their default value)
    var init := tf.global_variables_initializer;

    // Start training
    var sess := tf.Session;
    // Run the initializer
    sess.run(init);

    // Fit all training data
    var epoch: Integer ;
    for epoch := 0 to training_epochs -1 do
    begin
        for var zItem in TUtils.zip<Single>(train_X, train_Y) do
        begin
            var v_x : Single := zItem.Value1 ;
            var v_y : Single := zItem.Value2 ;
            sess.run(optimizer, [ Tuple<TValue,TValue>.Create(X, v_x), Tuple<TValue,TValue>.Create(Y, v_y) ]);
        end;

        // Display logs per epoch step
        if ((epoch + 1) mod display_step) = 0 then
        begin
            var fc : Single := NDArray(sess.run(cost, [ Tuple<TValue,TValue>.Create(X, train_X), Tuple<TValue,TValue>.Create(Y, train_Y) ]));
            var fW : Single := NDArray(sess.run( TResourceVariable(W) ));
            var fb : Single := NDArray(sess.run( TResourceVariable(b) ));
            mmo1.Lines.Add( Format('Epoch: %d cost=%.9f + "W=%.9f b=%.9f"',[epoch + 1,fc, fW,fb]) );
        end;
    end;

    mmo1.Lines.Add('Optimization Finished!');
    var training_cost : Single := NDArray(sess.run(cost, [ Tuple<TValue,TValue>.Create(X, train_X), Tuple<TValue,TValue>.Create(Y, train_Y) ]));
    var fW            : Single := NDArray(sess.run(  TResourceVariable(W) ));
    var fb            : Single := NDArray(sess.run(  TResourceVariable(b) ));

    mmo1.Lines.Add('');
    mmo1.Lines.Add(Format('Training cost=%.9f W=%.9f b=%.9f',[training_cost, fW, fb]));

    // Testing example
    var test_X : NDArray := np.np_array( TArray<Single>.Create(6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1) );
    var test_Y : NDArray := np.np_array( TArray<Single>.Create(1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03) );

    mmo1.Lines.Add('Testing... (Mean square loss Comparison)');

    var t_cost                 := TTensor(tf.reduce_sum(tf.pow(pred - Y, 2.0)))  / (2.0 * test_X.shape[0]) ;
    var testing_cost : Single  := NDArray(sess.run(t_cost, [ Tuple<TValue,TValue>.Create(X, test_X), Tuple<TValue,TValue>.Create(Y, test_Y) ]));

    mmo1.Lines.Add('');
    mmo1.Lines.Add( Format('Testing cost=%.9f',[testing_cost]) );
    var diff := Abs( training_cost - testing_cost);
    mmo1.Lines.Add( Format('Absolute mean square loss difference: %.9f',[diff]) );
    mmo1.Lines.Add('');

    Result := diff < 0.01;

end;

{ EagerModeTestBase }

function EagerModeTestBase.Equal(f1, f2: TArray<Single>): Boolean;
begin
    var ret: Boolean := false;
    var tolerance : Single := 000001;
    for var i := 0 to  Length(f1) - 1 do
    begin
        ret := Abs(f1[i] - f2[i]) <= tolerance;
        if  not ret then
            break;
    end;
    Result := ret;
end;

function EagerModeTestBase.Equal(f1, f2: Single): Boolean;
begin
     var tolerance : Single := 000001;
     Result := Abs(f1 - f2) <= tolerance;
end;

procedure EagerModeTestBase.clip_by_global_norm;
begin
    var t_list := TFTensors.Create( [ tf.constant( TArray<Single>.Create( 1, 2, 3, 4 ) ), tf.constant( TArray<Single>.Create( 5, 6, 7, 8 ) ) ] );
    var clip_norm : Single := 0.8;
    var tNorm := tf.clip_by_global_norm(t_list.ToArray, clip_norm);
    var res  := tNorm.Value1;
    var norm := tNorm.Value2;
    var expected  : TArray<Single> := [ 0.0560112074, 0.112022415, 0.16803363, 0.22404483 ];
    var actual := res[0].ToArray<Single>;
    Assert.IsTrue(Equal(expected, actual));
    expected  := [ 0.28005603, 0.336067259, 0.392078459, 0.448089659 ];
    actual    := res[1].ToArray<Single>;
    Assert.IsTrue(Equal(expected, actual));
    var nNorm : NDArray := norm.numpy;
    Assert.AreEqual<Single>( nNorm, 14.282857);
end;

constructor EagerModeTestBase.Create;
begin
    TestInit;
end;

function EagerModeTestBase.Equal(f1, f2: TArray<Double>): Boolean;
begin
    var ret: Boolean := false;
    var tolerance : Single := 000000000000001;
    for var i := 0 to  Length(f1) - 1 do
    begin
        ret := Abs(f1[i] - f2[i]) <= tolerance;
        if  not ret then
            break;
    end;
    Result := ret;
end;

procedure EagerModeTestBase.TestInit;
begin
    if not tf.executing_eagerly then
       tf.enable_eager_execution;
    tf.Context.ensure_initialized;
end;

{ ActivationFunctionTest }

constructor ActivationFunctionTest.Create;
begin
    a := tf.constant( TArray<Single>.Create( 1.0, -0.5, 3.4, -2.1, 0.0, -6.5 ) );
    TestInit;
end;

procedure ActivationFunctionTest.Sigmoid;
begin
    var b := tf.nn.sigmoid(a, 'sigmoid');
    var expected : TArray<Single> := [ 0.7310586, 0.37754068, 0.9677046, 0.10909683, 0.5, 0.00150118 ];
    var actual := b.ToArray<Single>;
    Assert.IsTrue( Equal(expected, actual) );
end;

procedure ActivationFunctionTest.ReLU;
begin
    var b := tf.nn.relu(a, 'ReLU');
    var expected : TArray<Single> := [ 1, 0, 3.4, 0, 0, 0 ];
    var actual := b.ToArray<Single>;
    Assert.IsTrue(Equal(expected, actual));
end;

procedure ActivationFunctionTest.TanH;
begin
    var b := tf.nn.tanh(a, 'TanH');
    var expected  : TArray<Single> := [ 0.7615942, -0.46211717, 0.9977749, -0.970452, 0, -0.99999547 ];
    var actual := b.ToArray<Single>;
    Assert.IsTrue(Equal(expected, actual));
end;

{ BitwiseApiTest }

constructor BitwiseApiTest.Create;
begin
    TestInit;
end;

procedure BitwiseApiTest.BitwiseAnd;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( 0, 5, 3, 14 ) );
    var rhs : TFTensor := tf.constant( TArray<Integer>.Create( 5, 0, 7, 11 ) );
    var bitwise_and_result := tf.bitwise.bitwise_and(lhs, rhs);
    var expected : TArray<Integer> := [ 0, 0, 3, 10 ];
    var actual := bitwise_and_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

procedure BitwiseApiTest.BitwiseOr;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( 0, 5, 3, 14 ) );
    var rhs : TFTensor := tf.constant( TArray<Integer>.Create( 5, 0, 7, 11 ) );
    var bitwise_or_result := tf.bitwise.bitwise_or(lhs, rhs);
    var expected : TArray<Integer> := [ 5, 5, 7, 15 ];
    var actual := bitwise_or_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

procedure BitwiseApiTest.BitwiseXOR;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( 0, 5, 3, 14 ) );
    var rhs : TFTensor := tf.constant( TArray<Integer>.Create( 5, 0, 7, 11 ) );
    var bitwise_xor_result := tf.bitwise.bitwise_xor(lhs, rhs);
    var expected : TArray<Integer> := [ 5, 5, 4, 5 ];
    var actual := bitwise_xor_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

procedure BitwiseApiTest.Invert;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( 0, 1, -3, integer.MaxValue ) );

    var invert_result := tf.bitwise.invert(lhs);
    var expected : TArray<Integer> := [ -1, -2, 2, Integer.MinValue ];
    var actual := invert_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

procedure BitwiseApiTest.LeftShift;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( -1, -5, -3, -14 ) );
    var rhs : TFTensor := tf.constant( TArray<Integer>.Create(5, 0, 7, 11 ));
    var left_shift_result := tf.bitwise.left_shift(lhs, rhs);
    var expected : TArray<Integer> := [ -32, -5, -384, -28672 ];
    var actual := left_shift_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

procedure BitwiseApiTest.RightShift;
begin
    var lhs : TFTensor := tf.constant( TArray<Integer>.Create( -2, 64, 101, 32 ) );
    var rhs : TFTensor := tf.constant( TArray<Integer>.Create( -1, -5, -3, -14 ) );
    var right_shift_result := tf.bitwise.right_shift(lhs, rhs);
    var expected : TArray<Integer> := [ -2, 64, 101, 32 ];
    var actual := right_shift_result.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(expected, actual));
end;

{ ConstantTest }

constructor ConstantTest.Create;
begin
    TestInit;
end;

procedure ConstantTest.Multiply;
begin
    var a : TTensor := tf.constant(Double(3.0));
    var b : TTensor := tf.constant(Double(2.0));
    var c : TTensor := a * b;
    Assert.AreEqual<Double>(6.0, Double(c));
end;

procedure ConstantTest.NDimConst;
begin
    var a : TArray<TArray<Integer>>:= [[3,1,1],[2,1,3]];
    var nd := np.np_array(a);

    var tensor := tf.constant(nd);
    var data := tensor.numpy.ToArray<Integer>;
    Assert.IsTrue( TUtils.SequenceEqual<Int64>  ([ 2, 3 ], tensor.shape.dims));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 3, 1, 1, 2, 1, 3 ], data));
end;

procedure ConstantTest.OnesConst;
begin
    var ones := tf.ones(TFShape.Create([3, 2]), tf.float32_t, 'ones');
    Assert.AreEqual(ones.dtype, tf.float32_t);
    Assert.AreEqual<Int64>(ones.shape[0], 3);
    Assert.AreEqual<Int64>(ones.shape[1], 2);
    Assert.IsTrue( TUtils.SequenceEqual<Single>( [1, 1, 1, 1, 1, 1 ], ones.numpy.ToArray<single>) );

end;

procedure ConstantTest.OnesToHalves;
begin
    var ones : TTensor   := tf.ones(TFShape.Create([3, 2]), tf.float64_t, 'ones');
    var halfes: TFTensor := ones * 0.5;
    Assert.AreEqual<Int64>(halfes.shape[0], 3);
    Assert.AreEqual<Int64>(halfes.shape[1], 2);
    Assert.IsTrue( TUtils.SequenceEqual<Double>( [ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ],halfes.numpy.ToArray<double>) );
end;

procedure ConstantTest.Reshape;
begin
    var ones := tf.ones(TFShape.Create([3, 2]), tf.float32_t, 'ones');
    var reshaped := tf.reshape(ones, TArray<Integer>.Create(2, 3) );
    Assert.AreEqual(reshaped.dtype, tf.float32_t);
    Assert.AreEqual<Int64>(reshaped.shape[0], 2);
    Assert.AreEqual<Int64>(reshaped.shape[1], 3);
    Assert.IsTrue( TUtils.SequenceEqual<Single>( [ 1, 1, 1, 1, 1, 1 ], ones.numpy.ToArray<Single>) );
end;

procedure ConstantTest.ScalarConst;
begin
    var tensor1 := tf.constant(8); // int
    Assert.AreEqual(tensor1.dtype, TF_DataType.TF_INT32);
    var tensor2 := tf.constant(Single(6.0)); // float
    Assert.AreEqual(tensor2.dtype, TF_DataType.TF_FLOAT);
    var tensor3 := tf.constant(Double(6.0)); // double
    Assert.AreEqual(tensor3.dtype, TF_DataType.TF_DOUBLE);
end;

procedure ConstantTest.ZerosConst;
begin
    // small size
    var tensor := tf.zeros(TArray<Integer>.Create(3, 2), tf.int32_t, 'small');
    Assert.AreEqual<Int64>(tensor.shape[0], 3);
    Assert.AreEqual<Int64>(tensor.shape[1], 2);
    Assert.IsTrue( TUtils.SequenceEqual<Integer>( [ 0, 0, 0, 0, 0, 0 ], tensor.numpy.ToArray<Integer>) );
    // big size
    tensor := tf.zeros(TArray<Integer>.Create(200, 100), tf.int32_t, 'big');
    Assert.AreEqual<Int64>(tensor.shape[0], 200);
    Assert.AreEqual<Int64>(tensor.shape[1], 100);
    var data := tensor.numpy.ToArray<Integer>;
    Assert.AreEqual<Integer>(0, data[0]);
    Assert.AreEqual<Integer>(0, data[500]);
    Assert.AreEqual<Integer>(0, data[Length(data) - 1]);
end;

{ LinearRegressionEager }

constructor LinearRegressionEager.Create;
begin
    training_epochs := 1000;
    training_steps  := 1000;

    // Parameters
    learning_rate := 0.01;
    display_step  := 50;
end;

procedure LinearRegressionEager.PrepareData;
begin
    train_X := np.np_array<Single>([3.3, 4.4,  5.5,  6.71, 6.93,  4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]);
    train_Y := np.np_array<Single>([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465,  1.65,  2.904, 2.42,  2.94, 1.3]);
    n_samples := train_X.shape[0];
end;

function LinearRegressionEager.Run(mmo1: TMemo): Boolean;
begin
    tf.enable_eager_execution;

    PrepareData;

    // Set model weights
    // We can set a fixed init value in order to debug
    // var rnd1 = rng.randn<float>();
    // var rnd2 = rng.randn<float>();
    var W : TResourceVariable := tf.Variable(Single(-0.06), 'weight');
    var b : TResourceVariable := tf.Variable(Single(-0.73), 'bias');
    var optimizer := kKeras.optimizers.SGD(learning_rate);

    // Run training for the given number of steps.
    for var step in TUtils.range(1, training_steps + 1) do
    begin
        // Run the optimization to update W and b values.
        // Wrap computation inside a GradientTape for automatic differentiation.
        var g := tf.GradientTape;
        // Linear regression (Wx + b).
        var pred : TTensor := (W * train_X) + b;
        // Mean square error.
        var sub  := pred - train_Y;
        var p    := tf.pow(sub, 2);
        var s    := tf.reduce_sum(p);
        var loss := TTensor(s) / (2 * n_samples);
        // should stop recording
        // Compute gradients.
        var t := Tuple<ResourceVariable, ResourceVariable>.Create(W, b);
        var gradients : Tuple<TFTensor,TFTensor> := g.gradient(loss, t);

        // Update W and b following gradients.
        //var n := ResourceVariable(W).name;
        optimizer.apply_gradients(TUtils.zip<TFTensor,ResourceVariable>(gradients, Tuple<ResourceVariable,ResourceVariable>.Create(W, b)));

        if step mod display_step = 0 then
        begin
            pred := W * train_X + b;
            loss := TTensor( tf.reduce_sum(tf.pow(pred - train_Y, 2)) ) / (2 * n_samples);
            Format('step: %d, loss: %.9f, W: %.9f, b: %.9f',[step, Single(loss.numpy),  W.numpy,  b.numpy]);

        end;
    end;
end;

end.
