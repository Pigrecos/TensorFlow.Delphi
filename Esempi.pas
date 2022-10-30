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

       procedure PrepareData;
       function  Run: Boolean;
  end;

  EagerModeTestBase = class
      procedure TestInit;
      function  Equal(f1: Single; f2: Single): Boolean; overload;
      function  Equal(f1: TArray<Single>; f2: TArray<Single>): Boolean; overload;
      function  Equal(f1: TArray<Double>; f2: TArray<Double>): Boolean; overload;
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


implementation
        uses DUnitX.TestFramework;

{ LinearRegression }

procedure LinearRegression.PrepareData;
begin
    train_X := np.np_array<Single>([3.3, 4.4,  5.5,  6.71, 6.93,  4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]);
    train_Y := np.np_array<Single>([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465,  1.65,  2.904, 2.42,  2.94, 1.3]);
    n_samples := train_X.shape[0];

end;

function LinearRegression.Run: Boolean;
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
    var W : ResourceVariable := tf.Variable(-0.06, 'weight');
    var b : ResourceVariable := tf.Variable(-0.73, 'bias');

    // Construct a linear model
    var pred : TTensor := tf.add(tf.multiply(X, W), b);
    var pred1 := (X * W) + b;

    Result := True;


    // Mean squared error
    var cost : TTensor := tf.reduce_sum(tf.pow(pred - Y, 2.0));
    cost := cost / (2.0 * n_samples);
    (*
    // Gradient descent
    // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

    // Initialize the variables (i.e. assign their default value)
    var init = tf.global_variables_initializer();

    // Start training
    using var sess = tf.Session();
    // Run the initializer
    sess.run(init);

    // Fit all training data
    for (int epoch = 0; epoch < training_epochs; epoch++)
    {
        foreach (var (x, y) in zip<float>(train_X, train_Y))
            sess.run(optimizer, (X, x), (Y, y));

        // Display logs per epoch step
        if ((epoch + 1) % display_step == 0)
        {
            var c = sess.run(cost, (X, train_X), (Y, train_Y));
            Console.WriteLine($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
        }
    }

    Console.WriteLine("Optimization Finished!");
    var training_cost = sess.run(cost, (X, train_X), (Y, train_Y));
    Console.WriteLine($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");

    // Testing example
    var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
    var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
    Console.WriteLine("Testing... (Mean square loss Comparison)");
    var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
        (X, test_X), (Y, test_Y));
    Console.WriteLine($"Testing cost={testing_cost}");
    var diff = Math.Abs((float)training_cost - (float)testing_cost);
    Console.WriteLine($"Absolute mean square loss difference: {diff}");

    return diff < 0.01;
    *)
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

end.
