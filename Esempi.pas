unit Esempi;
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
     uses Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
          Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti, System.Math,  System.Generics.Collections,

          spring,

          TF4D.Core.CApi,

          Numpy,
          Tensorflow,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          Tensorflow.Utils,
          TensorFlow.Ops,
          TensorFlow.Core,
          TensorFlow.Slice,

          Keras.Core,
          Keras.KerasApi,
          keras.Models,
          Keras.Layer,

          TensorFlow.Variable,
          TensorFlow.Tensor,
          NumPy.NDArray,
          Numpy.Axis;

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
       destructor Destroy; override;
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
       destructor Destroy; override;
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
      procedure NeuralNetworkTest_l2_loss;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/top_k_categorical_accuracy
      /// </summary>
      procedure top_k_categorical_accuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy
      /// </summary>
      procedure TopKCategoricalAccuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall
      /// </summary>
      procedure Recall;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision
      /// </summary>
      procedure Precision;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy
      /// </summary>
      procedure BinaryAccuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy
      /// </summary>
      procedure CategoricalAccuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy
      /// </summary>
      procedure CategoricalCrossentropy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy
      /// </summary>
      procedure Accuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CosineSimilarity
      /// </summary>
      procedure CosineSimilarity;
      /// <summary>
      /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/HammingLoss
      /// </summary>
      procedure HammingLoss;
      /// <summary>
      /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
      /// </summary>
      procedure F1Score;
      /// <summary>
      /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/FBetaScore
      /// </summary>
      procedure FBetaScore;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy
      /// </summary>
      procedure SparseCategoricalAccuracy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalCrossentropy
      /// </summary>
      procedure SparseCategoricalCrossentropy;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy
      /// </summary>
      procedure SparseTopKCategoricalAccuracy;
  end;

  PreprocessingTests = class(EagerModeTestBase)
    private
      texts           : TArray<string>;
      tokenized_texts : TArray<TArray<string>>;
      processed_texts : TArray<string>;
      OOV             : string;

    public

      constructor Create;
      destructor Destroy; override;

      procedure TokenizeWithNoOOV;
      procedure TokenizeWithNoOOV_Tkn;
      procedure TokenizeWithOOV;
      procedure TokenizeWithOOV_Tkn;
      procedure TokenizeTextsToSequences;
      procedure TokenizeTextsToSequences_Tkn;
      procedure PadSequencesWithDefaults;
      procedure TextToMatrixBinary;
      procedure TextToMatrixFrequency;


  end;

  /// <summary>
  /// https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers
  /// </summary>
  LayersTest = class(EagerModeTestBase)
    public
      procedure AveragePooling2D;
      procedure InputLayer;
      procedure Sequential;
      procedure Functional;
      /// <summary>
      /// Custom layer test, used in Dueling DQN
      /// </summary>
      procedure TensorFlowOpLayer;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
      /// </summary>
      procedure Embedding;
      /// <summary>
      /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
      /// </summary>
      procedure Dense;
      procedure EinsumDense;
      procedure SimpleRNN;
      procedure Resizing;
      procedure LayerNormalization;
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

  LinalgTest = class(EagerModeTestBase)
     private

      public
        constructor Create;
        procedure Einsum;
        procedure EyeTest;
        procedure GlobalNorm;
        procedure LSTSQ;
        procedure Tensordot;
  end;

  Keras_Layers_test = class(EagerModeTestBase)
      public
        constructor Create;

        procedure ActivationTest_LeakyReLU;
        procedure ActivationTest_ELU;
        procedure ActivationTest_SELU;
        procedure ActivationTest_Softmax;
        procedure ActivationTest_Softplus;
        procedure ActivationTest_Softsign;
        procedure ActivationTest_Exponential;
        procedure ActivationTest_HardSigmoid;
        procedure ActivationTest_Swish;
        /// <summary>
        /// https://www.tensorflow.org/addons/api_docs/python/tfa/activations/mish
        /// </summary>
        procedure ActivationTest_Mish;
        //
        procedure Attention_BaseDenseAttention;
        procedure Attention_Attention;
        procedure Attention_MultiHeadAttention;
        //
        procedure BasicConv1D;
        procedure BasicConv1D_ksize;
        procedure BasicConv1D_ksize_same;
        procedure BasicConv1D_ksize_strides;
        procedure BasicConv1D_ksize_dilations;
        procedure BasicConv1D_ksize_dilation_same;
        //
        procedure BasicConv2D;
        procedure BasicConv2D_ksize;
        procedure BasicConv2D_ksize_same;
        procedure BasicConv2D_ksize_strides;
        procedure BasicConv2D_ksize_dilations;
        procedure BasicConv2D_ksize_dilation_same;
        //
        procedure Cropping1D;
        procedure Cropping2D;
        procedure Cropping3D;
        //
        procedure Concatenate;
        //
        procedure ZeroPadding2D;
        procedure UpSampling2D;
        procedure Reshape;
        procedure Permute;
        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding
        /// </summary>
        procedure CategoryEncoding;
  end;

  Keras_Losses_test = class
      public
        // CosineSimilarity
        y_true_float   : TNDArray;
        y_pred_float   : TNDArray;
        // Huber
        y_true_float_H : TNDArray;
        y_pred_float_H : TNDArray;
        // LogCosh
        y_true_float_L : TNDArray;
        y_pred_float_L : TNDArray;
        // MeanAbsoluteError
        y_true_float_MAE : TNDArray;
        y_pred_float_MAE : TNDArray;

        constructor Create;

        // https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
        procedure BinaryCrossentropy;
        // https://keras.io/api/losses/regression_losses/
        procedure CosineSimilarity_Default;
        procedure CosineSimilarity_Sample_Weight;
        procedure CosineSimilarity_SUM;
        procedure CosineSimilarity_None;
        // https://keras.io/api/losses/regression_losses/#meansquarederror-class
        procedure Huber_Default;
        procedure Huber_Sample_Weight;
        procedure Huber_SUM;
        procedure Huber_None;
        // https://keras.io/api/losses/regression_losses/#meansquarederror-class
        procedure LogCosh_Default;
        procedure LogCosh_Sample_Weight;
        procedure LogCosh_SUM;
        procedure LogCosh_None;
        // https://keras.io/api/losses/regression_losses/
        procedure MeanAbsoluteError_Default;
        procedure MeanAbsoluteError_Sample_Weight;
        procedure MeanAbsoluteError_SUM;
        procedure MeanAbsoluteError_None;
        /// <summary>
        /// https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
        /// </summary>
        procedure SigmoidFocalCrossEntropy;
  end;

   procedure On_Epoch_Begin(msg: string);
   procedure On_Train_Batch_Begin(msg: string);
   procedure On_End_Summary(msg: string);

implementation
        uses untMain,

             DUnitX.TestFramework,

             Keras.LossFunc,
             Keras.Utils,
             keras.Preprocessing,

             Tensorflow.Proto;

procedure On_Epoch_Begin(msg: string);
begin
    frmMain.mmo1.Lines.Add(msg)
end;

procedure On_Train_Batch_Begin(msg: string);
begin
    frmMain.mmo1.Lines.Add(msg)
end;

procedure On_End_Summary(msg: string);
begin
    frmMain.mmo1.Lines.Add(msg)
end;

{ LinearRegression }

constructor LinearRegression.Create;
begin
    training_epochs := 1000;

    // Parameters
    learning_rate := 0.01;
    display_step  := 50;
end;

destructor LinearRegression.Destroy;
begin

  inherited;
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

procedure EagerModeTestBase.NeuralNetworkTest_l2_loss;
begin
    var vA : TArray< TArray<Single> > := [[1, 2, 3, 4],[5, 6, 7, 8]];
    var x := tf.Variable(np.np_array(vA), '',tf.float32_t);
    var l2 := tf.nn.l2_loss(x.totensor);
    var l2_numpy : NDArray := l2.numpy;
    Assert.AreEqual<Single>(l2_numpy, 102);
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

procedure EagerModeTestBase.top_k_categorical_accuracy;
begin
    var y_true := np.np_array< TArray<Integer>>([[ 0, 0, 1 ], [ 0, 1, 0 ]]);
    var y_pred := np.np_array< TArray<Single>>([[ 0.1, 0.9, 0.8 ], [ 0.05, 0.95, 0 ]]);
    var m := tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 3);
    var expected : TArray<Single> := [ 1, 1 ];
    var actual := m.numpy.ToArray<Single>;
    Assert.IsTrue(TUtils.SequenceEqual<Single>(expected, actual));
end;

procedure EagerModeTestBase.TopKCategoricalAccuracy;
begin
    var y_true := np.np_array< TArray<Integer>>([[ 0, 0, 1 ], [ 0, 1, 0 ]]);
    var y_pred := np.np_array< TArray<Single>>([[ 0.1, 0.9, 0.8 ], [ 0.05, 0.95, 0 ]]);
    var m      := tf.keras.metrics.TopKCategoricalAccuracy(1);
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.5);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.7, 0.3]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.3);
end;

procedure EagerModeTestBase.Recall;
begin
    var y_true := np.np_array< Integer >([ 0, 1, 1, 1 ]);
    var y_pred := np.np_array< Integer >([ 1, 0, 1, 1 ]);
    var m      := tf.keras.metrics.Recall;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.6666667);

    m.reset_states;
    var weights := np.np_array< Single >([ 0, 0, 1, 0 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 1);
end;

procedure EagerModeTestBase.Precision;
begin
    var y_true := np.np_array< Integer >([ 0, 1, 1, 1 ]);
    var y_pred := np.np_array< Integer >([ 1, 0, 1, 1 ]);
    var m      := tf.keras.metrics.Precision;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.6666667);

    m.reset_states;
    var weights := np.np_array< Single >([ 0, 0, 1, 0 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 1);

    // With top_k=2, it will calculate precision over y_true[:2]
    // and y_pred[:2]
    y_true := np.np_array< Integer >([ 0, 0, 1, 1 ]);
    y_pred := np.np_array< Integer >([ 1, 1, 1, 1 ]);
    m := tf.keras.metrics.Precision(0.5, 2);
    m.update_state(y_true, y_pred);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0);

    // With top_k=4, it will calculate precision over y_true[:4]
    // and y_pred[:4]
    y_true := np.np_array< Integer >([ 0, 0, 1, 1 ]);
    y_pred := np.np_array< Integer >([ 1, 1, 1, 1 ]);
    m := tf.keras.metrics.Precision(0.5, 4);
    m.update_state(y_true, y_pred);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.5);
end;

procedure EagerModeTestBase.BinaryAccuracy;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 1 ], [ 1 ],[ 0 ], [ 0 ] ]);
    var y_pred := np.np_array< TArray<Single> >([ [0.98 ], [ 1 ], [ 0 ], [ 0.6 ] ]);
    var m      := tf.keras.metrics.BinaryAccuracy;
    var weights := np.np_array< Single >([ 1, 0, 0, 1 ]);
    m.update_state(y_true, y_pred, weights);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.5);
end;

procedure EagerModeTestBase.CategoricalAccuracy;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 0, 0, 1 ], [ 0, 1, 0 ] ]);
    var y_pred := np.np_array< TArray<Single> >([ [ 0.1, 0.9, 0.8 ], [ 0.05, 0.95, 0 ] ]);
    var m      := tf.keras.metrics.CategoricalAccuracy;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.5);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.7, 0.3 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.3);
end;

procedure EagerModeTestBase.CategoricalCrossentropy;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 0, 1, 0 ], [ 0, 0, 1 ] ]);
    var y_pred := np.np_array< TArray<Single> >([ [ 0.05, 0.95, 0 ], [ 0.1, 0.8, 0.1 ] ]);
    var m      := tf.keras.metrics.CategoricalCrossentropy;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,1.1769392);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.3, 0.7 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 1.6271976);
end;

procedure EagerModeTestBase.Accuracy;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 1 ], [ 2 ], [ 3 ], [ 4 ] ]);
    var y_pred := np.np_array< TArray<Single> >([  [ 0 ], [ 2 ], [ 3 ], [ 4 ] ]);
    var m      := tf.keras.metrics.Accuracy;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.75);

    m.reset_states;
    var weights := np.np_array< Single >([ 1, 1, 0, 0 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.5);
end;

procedure EagerModeTestBase.CosineSimilarity;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 0, 1 ], [ 1, 1 ] ]);
    var y_pred := np.np_array< TArray<Single> >([  [ 1, 0 ], [ 1, 1 ] ]);
    var asse : TAxis := 1;
    var m      := tf.keras.metrics.CosineSimilarity('cosine_similarity', TF_FLOAT, @asse);
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.49999997);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.3, 0.7 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.6999999);
end;

procedure EagerModeTestBase.SparseCategoricalAccuracy;
begin
    var y_true := np.np_array< Integer >([ 2, 1 ]);
    var y_pred := np.np_array< TArray<Single> >([  [ 0.1, 0.6, 0.3 ], [ 0.05, 0.95, 0 ] ]);

    var m      := tf.keras.metrics.SparseCategoricalAccuracy;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.5);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.7, 0.3 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.3);
end;

procedure EagerModeTestBase.SparseCategoricalCrossentropy;
begin
    var y_true := np.np_array< Integer >([ 1, 2 ]);
    var y_pred := np.np_array< TArray<Single> >([  [ 0.05, 0.95, 0 ], [ 0.1, 0.8, 0.1 ] ]);

    var m      := tf.keras.metrics.SparseCategoricalCrossentropy;
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,1.1769392);
end;

procedure EagerModeTestBase.SparseTopKCategoricalAccuracy;
begin
    var y_true := np.np_array< Integer >([ 2, 1 ]);
    var y_pred := np.np_array< TArray<Single> >([  [ 0.1, 0.9, 0.8 ], [ 0.05, 0.95, 0 ] ]);

    var m      := tf.keras.metrics.SparseTopKCategoricalAccuracy(1);
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.5);

    m.reset_states;
    var weights := np.np_array< Single >([ 0.7, 0.3 ]);
    m.update_state(y_true, y_pred, weights);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r, 0.3);
end;

procedure EagerModeTestBase.HammingLoss;
begin
    // multi-class hamming loss
    var y_true := np.np_array< TArray<Integer> >([
        [ 1, 0, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ],
        [ 0, 1, 0, 0 ] ]);
    var y_pred := np.np_array< TArray<Single> >([
        [ 0.8, 0.1, 0.1, 0.0 ],
        [ 0.2, 0.0, 0.8, 0.0 ],
        [ 0.05, 0.05, 0.1, 0.8 ],
        [ 1.0, 0.0, 0.0, 0.0 ]]);

    var threshold  : Single := 0.6;
    var m      := tf.keras.metrics.HammingLoss('multiclass',@threshold);
    m.update_state(y_true, y_pred);
    var r : NDArray := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.25);

    // multi-label hamming loss
    y_true := np.np_array< TArray<Integer> >([
        [ 1, 0, 1, 0 ],
        [ 0, 1, 0, 1 ],
        [ 0, 0, 0, 1 ] ]);
    y_pred := np.np_array< TArray<Single> >([
        [ 0.82, 0.5, 0.9, 0.0 ],
        [ 0, 1, 0.4, 0.98 ],
        [ 0.89, 0.79, 0, 0.3 ]]);

    threshold  := 0.8;
    m      := tf.keras.metrics.HammingLoss('multilabel',@threshold);
    m.update_state(y_true, y_pred);
    r := m.R_result.numpy;
    Assert.AreEqual<Single>(r,0.16666667);
end;

procedure EagerModeTestBase.F1Score;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 1, 1, 1 ], [ 1, 0, 0 ], [ 1, 1, 0 ] ]);
    var y_pred := np.np_array< TArray<Single> >([ [ 0.2, 0.6, 0.7 ], [ 0.2, 0.6, 0.6 ], [ 0.6, 0.8, 0 ] ]);
    var threshold  : Single := 0.5;
    var m      := tf.keras.metrics.F1Score(3,'',@threshold);
    m.update_state(y_true, y_pred);
    var r := m.R_result.numpy.ToArray<Single>;
    var a : TArray<Single> := [ 0.5, 0.8, 0.6666667 ];
    Assert.IsTrue( TUtils.SequenceEqual<Single>(r,a)  );
end;

procedure EagerModeTestBase.FBetaScore;
begin
    var y_true := np.np_array< TArray<Integer> >([ [ 1, 1, 1 ], [ 1, 0, 0 ], [ 1, 1, 0 ] ]);
    var y_pred := np.np_array< TArray<Single> >([ [ 0.2, 0.6, 0.7 ], [ 0.2, 0.6, 0.6 ], [ 0.6, 0.8, 0 ] ]);
    var threshold  : Single := 0.5;
    var m      := tf.keras.metrics.FBetaScore(3,'',2.0, @threshold);
    m.update_state(y_true, y_pred);
    var r := m.R_result.numpy.ToArray<Single>;
    var a : TArray<Single> := [ 0.3846154, 0.90909094, 0.8333334 ];
    Assert.IsTrue( TUtils.SequenceEqual<Single>(r,a)  );
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

destructor LinearRegressionEager.Destroy;
begin
   train_X.Free;
   train_Y.Free;
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
    var optimizer := tf.Keras.optimizers.SGD(learning_rate);

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
        var gradients : Tuple<TFTensor,TFTensor> := g.gradient(loss, Tuple<ResourceVariable, ResourceVariable>.Create(W, b));
        // Update W and b following gradients.
        optimizer.apply_gradients(TUtils.zip<TFTensor,ResourceVariable>(gradients, Tuple<ResourceVariable,ResourceVariable>.Create(W, b)));

        if step mod display_step = 0 then
        begin
            pred := W * train_X + b;
            loss := TTensor( tf.reduce_sum(tf.pow(pred - train_Y, 2)) ) / (2 * n_samples);

            var fc : NDArray := loss.numpy;
            var fW : NDArray := W.numpy;
            var fb : NDArray := b.numpy;

            mmo1.Lines.Add( Format('step: %d, loss: %.9f, W: %.9f, b: %.9f',[step, Single(fc),  Single(fW), Single(fb)]) );

        end;
    end;
    mmo1.Lines.Add('');
    Result := True;
end;

{ LinalgTest }

procedure AssetSequenceEqual(expected: TArray<Single>; actual: TArray<Single>);
begin
    var eps: Single := 1e-5;
    for var i : Integer := 0 to Length(expected) - 1 do
        Assert.IsTrue(Abs(expected[i] - actual[i]) < eps * Max(1.0, Abs(expected[i]) ), Format('expected %.9f vs actual %.9f',[expected[i], actual[i]]) );
end;

constructor LinalgTest.Create;
begin
  TestInit;
end;

procedure LinalgTest.EyeTest;
begin
    var tensor := tf.linalg.eye(3);

    Assert.IsTrue(tensor.shape = TFShape.Create([3, 3]));
    var t1 : TTensor := tensor[[2, 0]];
    var t2 : TTensor := tensor[[2, 1]];
    var t3 : TTensor := tensor[[2, 2]];
    Assert.AreEqual<Double>(0.0,  Double(t1));
    Assert.AreEqual<Double>(0.0,  Double(t2));
    Assert.AreEqual<Double>(1.0,  Double(t3));
end;

/// <summary>
/// https://colab.research.google.com/github/biswajitsahoo1111/blog_notebooks/blob/master/Doing_Linear_Algebra_using_Tensorflow_2.ipynb#scrollTo=6xfOcTFBL3Up
/// </summary>
procedure LinalgTest.LSTSQ;
begin
    var aA_Over : TArray< TArray<Single>> := [ [ 1, 2 ], [ 2, 0.5 ], [ 3, 1 ], [ 4, 5.0] ];
    var A_over  := tf.constant(aA_Over);
    var aA_under : TArray< TArray<Single>> := [ [ 3, 1, 2, 5 ], [ 7, 9, 1, 4.0 ] ];
    var A_under := tf.constant(aA_under);
    var b_over  := tf.constant(TArray<Single>.Create( 3, 4, 5, 6.0), TFShape.Create([4, 1]) );
    var b_under  := tf.constant(TArray<Single>.Create( 7.2, -5.8),   TFShape.Create([2, 1]) );
    var x_over := tf.linalg.lstsq(A_over, b_over);
    var x := tf.matmul( tf.linalg.inv(tf.matmul(A_over, A_over,  true)), tf.matmul(A_over, b_over, true));
    Assert.IsTrue(x_over.shape = TFShape.Create([2, 1]));
    AssetSequenceEqual(x_over.ToArray<Single>, x.ToArray<Single>) ;
    var x_under := tf.linalg.lstsq(A_under, b_under);
    var y := tf.matmul(A_under, tf.matmul(tf.linalg.inv(tf.matmul(A_under, A_under, False, true)), b_under), true);
    Assert.IsTrue(x_under.shape = TFShape.Create([4, 1]));
    AssetSequenceEqual(x_under.ToArray<Single>, y.ToArray<Single>) ;

   (* var x_over_reg  := tf.linalg.lstsq(A_over, b_over, TNDArray.Create(Single(2.0)));
    var x_under_reg := tf.linalg.lstsq(A_under, b_under,TNDArray.Create(Single(2.0)));
    Assert.IsTrue(x_under_reg.shape = TFShape.Create([4, 1]));
    AssetSequenceEqual(x_under_reg.ToArray<Single>, [-0.04763567, -1.214508, 0.62748903, 1.299031])*) ;
end;

procedure LinalgTest.Einsum;
begin
    var m0 := tf.random.normal(TFShape.Create([2, 3]));
    var m1 := tf.random.normal(TFShape.Create([3, 5]));
    var e  := tf.linalg.einsum('ij,jk->ik', TFTensors.Create([m0, m1]) );
    Assert.IsTrue(e.shape = TFShape.Create([2, 5]));
end;

procedure LinalgTest.GlobalNorm;
begin
    var t_list := TFTensors.Create( [tf.constant(TArray<Single>.create(1, 2, 3, 4 )), tf.constant(TArray<Single>.create( 5, 6, 7, 8 ))] );
    var norm   := tf.linalg.global_norm(t_list.ToArray);
    var s1 : NDArray := norm.numpy;
    Assert.AreEqual<Single>(s1, 14.282857);
end;

procedure LinalgTest.Tensordot;
begin
    var a := tf.constant(TArray<Integer>.create( 1, 2 ));
    var b := tf.constant(TArray<Integer>.create( 2, 3 ));
    var c := tf.linalg.tensordot(a, b,  TNDArray.Create(Integer(0)));
    Assert.IsTrue(c.shape =  TFShape.Create([2, 2]));
    Assert.IsTrue(TUtils.SequenceEqual<Integer>(c.ToArray<integer>, [ 2, 3, 4, 6 ]) );
    c := tf.linalg.tensordot(a, b, TNDArray.Create( TArray<Integer>.Create( 0, 0 )) );
    Assert.AreEqual<Integer>(c.shape.ndim, 0);
    var s1 : NDArray := c.numpy;
    Assert.AreEqual<Integer>(s1, 8);
end;

{ Keras_Layers_test }

constructor Keras_Layers_test.Create;
begin
    inherited Create;
end;

procedure Keras_Layers_test.ActivationTest_LeakyReLU;
begin
    var layer := tf.keras.layers.LeakyReLU;

    var nd     : NDArray  := np.np_array<Single>([-3.0, -1.0, 0.0, 2.0]);
    var output : TTensor  := layer.Apply( nd );
    Equal( [ -0.9, -0.3, 0.0, 2.0 ], output.ToArray<Single>);
end;

procedure Keras_Layers_test.ActivationTest_Mish;
begin
    var aInput : TArray<Single> := [1.0, 0.0, 1.0];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.activations.Mish(input);

    var aexpected : TArray<Single> := [ 0.86509836, 0, 0.86509836 ];

    Assert.IsTrue( TUtils.SequenceEqual<Single>(aexpected, output.ToArray<Single>),'Assert - ActivationTest_Mish');
end;

procedure Keras_Layers_test.ActivationTest_ELU;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.ELU.Apply(input);

    var aexpected : TArray<Single> := [ -0.0950213, -0.08646648, -0.06321206, 0, 1, 2 ];
    var expected : TNDArray  :=  TNDArray.Create(aexpected);

    Assert.IsTrue(expected.numpy = output.numpy,'Assert - ActivationTest_ELU');
end;

procedure Keras_Layers_test.ActivationTest_SELU;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.SELU.Apply(input);

    var aexpected : TArray<Single> := [ -1.6705688, -1.5201665, -1.1113307, 0, 1.050701, 2.101402 ];
    var expected : TNDArray  :=  TNDArray.Create(aexpected);

    Assert.IsTrue(expected.numpy = output.numpy,'Assert - ActivationTest_SELU');
end;

procedure Keras_Layers_test.ActivationTest_Softmax;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.Softmax(TAxis(Integer(-1))).Apply(input);

    var expected : TArray<Single> := [ 0.0042697787, 0.011606461, 0.031549633, 0.085760795, 0.23312202, 0.6336913 ];

    Assert.IsTrue(Equal( expected, output.ToArray<Single>),'Assert - ActivationTest_Softmax');
end;

procedure Keras_Layers_test.ActivationTest_Softplus;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.Softplus.Apply(input);

    var aexpected : TArray<Single> := [ 0.04858733, 0.12692805, 0.31326166, 0.6931472, 1.3132616, 2.126928 ];
    var expected : TNDArray  :=  TNDArray.Create(aexpected);

    Assert.IsTrue(expected.numpy = output.numpy,'Assert - ActivationTest_Softplus');
end;

procedure Keras_Layers_test.ActivationTest_Softsign;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.Softsign.Apply(input);

    var aexpected : TArray<Single> := [ -0.75, -0.66666667, -0.5, 0, 0.5, 0.66666667 ];
    var expected : TNDArray  :=  TNDArray.Create(aexpected);

    Assert.IsTrue(expected.numpy = output.numpy,'Assert - ActivationTest_Softsign');
end;

procedure Keras_Layers_test.ActivationTest_Exponential;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.Exponential.Apply(input);

    var expected : TArray<Single> := [ 0.049787067, 0.13533528, 0.36787945, 1, 2.7182817, 7.389056 ];

    Assert.IsTrue(Equal( expected, output.ToArray<Single>),'Assert - ActivationTest_Exponential');
end;

procedure Keras_Layers_test.ActivationTest_HardSigmoid;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.HardSigmoid.Apply(input);

    var expected : TArray<Single> := [ 0, 0.099999994, 0.3, 0.5, 0.7, 0.9 ];

    Assert.IsTrue(Equal( expected, output.ToArray<Single>),'Assert - ActivationTest_HardSigmoid');
end;

procedure Keras_Layers_test.ActivationTest_Swish;
begin
    var aInput : TArray<Single> := [-3, -2, -1, 0, 1, 2];
    var input : TTensor := tf.constant( aInput );

    var output : TTensor := tf.keras.layers.Swish.Apply(input);

    var expected : TArray<Single> := [ -0.14227762, -0.23840584, -0.26894143, 0, 0.7310586, 1.761594 ];

    Assert.IsTrue(Equal( expected, output.ToArray<Single>),'Assert - ActivationTest_Swish');
end;

procedure Keras_Layers_test.Attention_BaseDenseAttention;

   procedure test_multi_dim_with_mask;
   begin
      // Scores tensor of shape [1, 1, 3]
      var scores := np.np_array<Single>([ [ [ 1, 0, 1 ] ] ], np.np_float32);
      // Value tensor of shape [1, 3, 1]
      var v := np.np_array<Single>([ [ [ 1.6 ], [ 0.7 ], [ -0.8 ] ] ], np.np_float32);
      // Scores mask tensor of shape [1, 1, 3]
      var scores_mask := np.np_array<Boolean>([ [ [ true, true, false ] ] ], np.np_bool);
      var _tup_1 :=  BaseDenseAttention.Create(BaseDenseAttentionArgs.Create )._apply_scores(scores, v, scores_mask);
      var actual        := _tup_1.Value1;
      var actual_scores := _tup_1.Value2;

      // Expected softmax scores = softmax(scores) with zeros in positions where
      // v_mask == False.
      // => softmax_scores000 = exp(1)/(exp(1) + exp(0)) = 0.73105857863
      //    softmax_scores001 = exp(0)/(exp(1) + exp(0)) = 0.26894142137
      //    softmax_scores002 = 0
      var expected_scores := np.np_array<Single>([ [ [ 0.73105857863, 0.26894142137, 0 ] ] ], np.np_float32);
      Assert.IsTrue(expected_scores.Equals( actual_scores.numpy));

      // Expected tensor of shape [1, 1, 1].
      // expected000 = 0.73105857863 * 1.6 + 0.26894142137 * 0.7 - 0 * 0.8
      //             = 1.35795272077
      //Actually the output is 1.3579528
      var expected := np.np_array<Single>([ [ [ 1.3579528 ] ] ], np.np_float32);
      Assert.IsTrue(expected.Equals(actual.numpy));
   end;

   procedure test_one_dim_batch_size_two;
   begin
      // Scores tensor of shape [2, 1, 2]
      var scores := np.np_array<Single>([ [ [ 1.1 ] ], [ [ 2.1 ] ] ], np.np_float32);
      // Value tensor of shape [2, 1, 1]
      var v := np.np_array<Single>([ [ [ 1.6 ] ], [ [ 2.6 ] ] ], np.np_float32);
      // Scores mask tensor of shape [2, 1, 1]
      var scores_mask := np.np_array<Boolean>([ [ [ true ] ], [ [ true ] ] ], np.np_bool);
      var _tup_1 :=  BaseDenseAttention.Create(BaseDenseAttentionArgs.Create )._apply_scores(scores, v, scores_mask);
      var actual        := _tup_1.Value1;
      var actual_scores := _tup_1.Value2;

      // Expected softmax_scores = [[[1]], [[1]]]
      var expected_scores := np.np_array<Single>([ [ [ 1 ] ], [ [ 1 ] ] ], np.np_float32);
      Assert.IsTrue(expected_scores.Equals( actual_scores.numpy));

      // Expected tensor of shape [2, 1, 1].
      // expected000 = softmax_scores[0, 0] * 1.6 = 1.6
      // expected100 = softmax_scores[1, 0] * 2.6 = 2.6
      var expected := np.np_array<Single>([ [ [ 1.6 ] ], [ [ 2.6 ] ] ], np.np_float32);
      Assert.IsTrue(expected.Equals(actual.numpy));
   end;

begin
    test_multi_dim_with_mask;
    test_one_dim_batch_size_two;
end;

procedure Keras_Layers_test.Attention_Attention;
    procedure test_calculate_scores_multi_dim ;
    begin
        // Query tensor of shape [1, 2, 4]
        var q := np.np_array<Single>([ [
                                        [  1, 1.1, 1.2, 1.3 ],
                                        [  2, 2.1, 2.2, 2.3 ] ] ], np.np_float32);

        // Key tensor of shape [1, 3, 4]
        var k := np.np_array<Single>([ [
                                        [  1.5, 1.6, 1.7, 1.8 ],
                                        [  2.5, 2.6, 2.7, 2.8 ],
                                        [  3.5, 3.6, 3.7, 3.8 ] ] ], np.np_float32);

        var attention_layer  := tf.keras.layers.Attention;
        //attention_layer.build(((1, 2, 4), (1, 3, 4)));
        var actual := Attention(attention_layer)._calculate_scores(q, k);
        // Expected tensor of shape [1, 2, 3].
        // expected000 = 1.*1.5+1.1*1.6+1.2*1.7+1.3*1.8 = 7.64
        // expected001 = 1.*2.5+1.1*2.6+1.2*2.7+1.3*2.8 = 12.24
        // expected002 = 1.*3.5+1.1*3.6+1.2*3.7+1.3*3.8 = 16.84
        // expected010 = 2.*1.5+2.1*1.6+2.2*1.7+2.3*1.8 = 14.24
        // expected011 = 2.*2.5+2.1*2.6+2.2*2.7+2.3*2.8 = 22.84
        // expected012 = 2.*3.5+2.1*3.6+2.2*3.7+2.3*3.8 = 31.44
        // Actually the output000 is 7.6400003, the output012 is 31.439999
        var expected := np.np_array<Single>([ [
                                        [  7.6400003, 12.24, 16.84     ],
                                        [  14.24,     22.84, 31.439999 ] ] ], np.np_float32);
        Assert.IsTrue(expected.Equals(actual.numpy));

    end;

    procedure test_calculate_scores_multi_dim_concat ;
    begin
        // Query tensor of shape [1, 2, 4]
        var q := np.np_array<Single>([ [
                                        [  1, 1.1, 1.2, 1.3 ],
                                        [  2, 2.1, 2.2, 2.3 ] ] ], np.np_float32);

        // Key tensor of shape [1, 3, 4]
        var k := np.np_array<Single>([ [
                                        [  1.5, 1.6, 1.7, 1.8 ],
                                        [  2.5, 2.6, 2.7, 2.8 ],
                                        [  3.5, 3.6, 3.7, 3.8 ] ] ], np.np_float32);

        var attention_layer  :=  Attention( tf.keras.layers.Attention(False,'concat') );
        //attention_layer.concat_score_weight = 1;
        var vArgs : VariableArgs;
        vArgs.Name := 'concat_score_weight';
        vArgs.Shape           := Integer(1);
        vArgs.DType           := TF_DataType.TF_FLOAT;
        vArgs.Overwrite       := true;
        vArgs.Initializer     := tf.ones_initializer;
        vArgs.Synchronization := VARIABLE_SYNCHRONIZATION_AUTO;
        vArgs.Aggregation     := VARIABLE_AGGREGATION_NONE;
        vArgs.Trainable       := true;
        attention_layer.concat_score_weight := base_layer_utils.make_variable(vArgs);
        //attention_layer.build(((1, 2, 4), (1, 3, 4)));
        //var actual = keras.backend.get_value(attention_layer._calculate_scores(query: q, key: k));
        var actual := attention_layer._calculate_scores(q, k);
        // pylint:disable=line-too-long
        // expected000 = tanh(1.+1.5) + tanh(1.1+1.6) + tanh(1.2+1.7) + tanh(1.3+1.8) = 3.96753427840
        // expected001 = tanh(1.+2.5) + tanh(1.1+2.6) + tanh(1.2+2.7) + tanh(1.3+2.8) = 3.99558784825
        // expected002 = tanh(1.+3.5) + tanh(1.1+3.6) + tanh(1.2+3.7) + tanh(1.3+3.8) = 3.99940254147
        // expected010 = tanh(2.+1.5) + tanh(2.1+1.6) + tanh(2.2+1.7) + tanh(2.3+1.8) = 3.99558784825
        // expected011 = tanh(2.+2.5) + tanh(2.1+2.6) + tanh(2.2+2.7) + tanh(2.3+2.8) = 3.99940254147
        // expected012 = tanh(2.+3.5) + tanh(2.1+3.6) + tanh(2.2+3.7) + tanh(2.3+3.8) = 3.99991913657
        //Actually the output012 is 3.9999194
        var expected := np.np_array<Single>([ [
                                        [  3.96753427840, 3.99558784825, 3.99940254147 ],
                                        [  3.99558784825, 3.99940254147, 3.9999194     ] ] ], np.np_float32);
        Assert.IsTrue(expected.Equals(actual.numpy));

    end;
begin
    test_calculate_scores_multi_dim;
    test_calculate_scores_multi_dim_concat;
end;

procedure Keras_Layers_test.Attention_MultiHeadAttention;
begin
    var batch_size := 3;

    var query       := tf.keras.Input(TFShape.Create([4, 8]));
    var value       := tf.keras.Input(TFShape.Create([2, 8]));
    var mask_tensor := tf.keras.Input(TFShape.Create([4, 2]));

    var attention_layer := tf.keras.layers.MultiHeadAttention(2, 2);
    attention_layer.Apply( TFTensors.Create([ query.First, value.First, mask_tensor.First ]) );

    var from_data := 10 * NDArray( np.random.randn([batch_size, 4, 8]) );
    var to_data   := 10 * NDArray( np.random.randn([batch_size, 2, 8]) );

    var sSize := TFShape.Create([4, 2]);
    var mask_data := np.random.randint(2, nil, @sSize);
    var masked_output_data := attention_layer.Apply( TFTensors.Create([ from_data, to_data, mask_data ]) );

    var null_mask_data       := np.ones(TFShape.Create([batch_size, 4, 2]));
    var unmasked_output_data := attention_layer.Apply(TFTensors.Create([ from_data, to_data, null_mask_data ]) );

    Assert.isTrue(masked_output_data.first.numpy.Equals(unmasked_output_data.first.numpy))
end;

procedure Keras_Layers_test.BasicConv1D;
begin
    var filters := 8;
    var conv := tf.keras.layers.Conv1D(filters, 3,  'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.IsTrue(y.shape = TFShape.Create([8, 6, 8]) );
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv1D_ksize;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv1D(filters, 3,  'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(3, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1] - 2, y.shape.Dims[1]);
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv1D_ksize_same;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv1D(filters, 3, 1, 'same', 'channels_last', 1, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(3, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1], y.shape.Dims[1]);
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv1D_ksize_strides;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv1D(filters, 3, 2, 'valid', 'channels_last', 1, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(3, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1]-5, y.shape.Dims[1]);
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv1D_ksize_dilations;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv1D(filters, 3, 1, 'valid', 'channels_last', 2, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(3, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1]-4, y.shape.Dims[1]);
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv1D_ksize_dilation_same;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv1D(filters, 3, 1, 'same', 'channels_last', 2, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(3, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1], y.shape.Dims[1]);
    Assert.AreEqual<Integer>(filters, y.shape[2]);
end;

procedure Keras_Layers_test.BasicConv2D;
begin
    var filters := 8;

    var conv := tf.keras.layers.Conv2D(filters, nil, nil, 'valid', '', nil, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1] - 4, y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2] - 4, y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.BasicConv2D_ksize;
begin
    var filters := 8;
    var sKsize : TFShape := 3;
    var conv := tf.keras.layers.Conv2D(filters, @sKsize, nil, 'valid', '', nil, 1, 'linear');

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1] - 2, y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2] - 2, y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.BasicConv2D_ksize_same;
begin
    var filters := 8;

    var sKsize : TFShape := 3;
    var conv := tf.keras.layers.Conv2D(filters, @sKsize, nil, 'same', '', nil, 1, 'linear');;

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1] , y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2] , y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.BasicConv2D_ksize_strides;
begin
    var filters := 8;

    var sKsize : TFShape := 3;
    var sstrides : TFShape := 2;
    var conv := tf.keras.layers.Conv2D(filters, @sKsize, @sstrides, 'valid', '', nil, 1, 'linear');;

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1]-5 , y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2]-5 , y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.BasicConv2D_ksize_dilations;
begin
    var filters := 8;

    var sKsize : TFShape := 3;
    var sdilation_rate : TFShape := 2;
    var conv := tf.keras.layers.Conv2D(filters, @sKsize, nil, 'valid', '', @sdilation_rate, 1, 'linear');;

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1]-4 , y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2]-4 , y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.BasicConv2D_ksize_dilation_same;
begin
    var filters := 8;

    var sKsize : TFShape := 3;
    var sdilation_rate : TFShape := 2;
    var conv := tf.keras.layers.Conv2D(filters, @sKsize, nil, 'same', '', @sdilation_rate, 1, 'linear');;

    var x := np.arange(Single(256.0)).reshape(TFShape.Create([1, 8, 8, 4]));
    var y := conv.Apply(TFTensors.Create(x));

    Assert.AreEqual<Integer>(4, y.shape.ndim);
    Assert.AreEqual<Int64>(x.shape.Dims[0], y.shape.Dims[0]);
    Assert.AreEqual<Int64>(x.shape.Dims[1] , y.shape.Dims[1]);
    Assert.AreEqual<Int64>(x.shape.Dims[2] , y.shape.Dims[2]);
    Assert.AreEqual<Integer>(filters, y.shape[3]);
end;

procedure Keras_Layers_test.Cropping1D;
begin
    var input_shape : TFShape := TFShape.Create([1, 5, 2]);
    var x := tf.zeros(input_shape);

    var aCropping := np.np_array<Integer>([1,2]);

    var cropping_1d := tf.keras.layers.Cropping1D(aCropping);
    var y           := cropping_1d.Apply(TFTensors.Create(x));

    Assert.IsTrue(TFShape.Create([1, 2, 2]) =  y.shape);
end;

procedure Keras_Layers_test.Cropping2D;
begin
    var input_shape : TFShape := TFShape.Create([1, 5, 6, 1]);
    var x := tf.zeros(input_shape);

    var aCropping := np.np_array<Integer>([ [1,2], [1,3]], np.np_int32);

    var cropping_2d := tf.keras.layers.Cropping2D(aCropping);
    var y           := cropping_2d.Apply(TFTensors.Create(x));

    Assert.IsTrue(TFShape.Create([1, 2, 2, 1]) =  y.shape);
end;

procedure Keras_Layers_test.Cropping3D;
begin
    var input_shape : TFShape := TFShape.Create([1, 5, 6, 7, 1]);
    var x := tf.zeros(input_shape);

    var aCropping := np.np_array<Integer>([ [1,2], [1,3], [1,4] ], np.np_int32);

    var cropping_3d := tf.keras.layers.Cropping3D(aCropping);
    var y           := cropping_3d.Apply(TFTensors.Create(x));

    Assert.IsTrue(TFShape.Create([1, 2, 2, 2, 1]) =  y.shape);
end;

procedure Keras_Layers_test.CategoryEncoding;
begin
    // one-hot
    var inputs := np.np_array<Integer>([3, 2, 0, 1 ]);
    var layer  := tf.keras.layers.CategoryEncoding(4);
    var output : TFTensor := layer.Apply(TFTensors.Create(inputs)).First;
    Assert.IsTrue(TFShape.Create([4, 4]) = output.shape);
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[0].ToArray<Single>,[ 0, 0, 0, 1 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[1].ToArray<Single>,[ 0, 0, 1, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[2].ToArray<Single>,[ 1, 0, 0, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[3].ToArray<Single>,[ 0, 1, 0, 0 ]));
    // multi-hot
    inputs := np.np_array<TArray<Integer>>([ [ 0, 1 ], [ 0, 0 ], [ 1, 2 ], [ 3, 1 ] ]);
    layer  := tf.keras.layers.CategoryEncoding(4, 'multi_hot');
    output := layer.Apply(TFTensors.Create(inputs)).First;
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[0].ToArray<Single>,[ 1, 1, 0, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[1].ToArray<Single>,[ 1, 0, 0, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[2].ToArray<Single>,[ 0, 1, 1, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[3].ToArray<Single>,[ 0, 1, 0, 1 ]));

    // using weighted inputs in "count" mode
    inputs := np.np_array<TArray<Integer>>([ [ 0, 1 ], [ 0, 0 ], [ 1, 2 ], [ 3, 1 ] ]);
    var weights := np.np_array<TArray<Single>>([ [  0.1, 0.2 ], [ 0.1, 0.1 ], [ 0.2, 0.3 ], [ 0.4, 0.2 ] ]);
    layer := tf.keras.layers.CategoryEncoding(4, 'count', False, weights);
    output := layer.Apply(TFTensors.Create(inputs)).First;
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[0].ToArray<Single>,[ 0.1, 0.2, 0, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[1].ToArray<Single>,[ 0.2, 0, 0, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[2].ToArray<Single>,[ 0, 0.2, 0.3, 0 ]));
    Assert.IsTrue(TUtils.SequenceEqual<Single>(output[3].ToArray<Single>,[ 0, 0.2, 0, 0.4 ]));
end;

procedure Keras_Layers_test.Concatenate;
begin
    var x := np.arange(20).reshape( TFShape.Create([2, 2, 5]) );
    var y := np.arange(20, 30).reshape( TFShape.Create([2, 1, 5]) );
    var z := tf.keras.layers.Concatenate(1).Apply(TFTensors.Create([x, y]) );
    Assert.IsTrue(TFShape.Create([2, 3, 5]) =  z.shape);
end;

procedure Keras_Layers_test.ZeroPadding2D;
begin
    var input_shape : TFShape := TFShape.Create([1, 1, 2, 2]);
    var x           := np.arange(input_shape.size).reshape( input_shape );

    var aZeroPad2D := np.np_array<Integer>([ [1,0], [1,0] ], np.np_int32);

    var zero_padding_2d := tf.keras.layers.ZeroPadding2D( aZeroPad2D );
    var y           := zero_padding_2d.Apply(TFTensors.Create(x));

    Assert.IsTrue(TFShape.Create([1, 2, 3, 2]) =  y.shape);
end;

procedure Keras_Layers_test.UpSampling2D;
begin
    var input_shape : TFShape := TFShape.Create([2, 2, 1, 3]);
    var x           := np.arange(input_shape.size).reshape( input_shape );

    var sSize : TFShape := TFShape.Create([1, 2]);

    var UpSampling2D := tf.keras.layers.UpSampling2D( @sSize );
    var y            := UpSampling2D.Apply(TFTensors.Create(x));

    Assert.IsTrue(TFShape.Create([2, 2, 2, 3]) =  y.shape);
end;

procedure Keras_Layers_test.Reshape;
begin
    var input_shape : TFShape := TFShape.Create([10, 5, 20]);
    var inputs      := tf.zeros(input_shape);
    var outputs     := tf.keras.layers.LeakyReLU().Apply( TFTensors.Create(inputs) );

    var sTargetShape : TFShape := TFShape.Create([20, 5]);
    var reshape := tf.keras.layers.Reshape(sTargetShape);

    outputs := reshape.Apply( outputs );

    Assert.IsTrue(TFShape.Create([10, 20, 5]) =  outputs.shape);
end;

procedure Keras_Layers_test.Permute;
begin
    var input_shape : TFShape := TFShape.Create([2, 3, 4, 5]);
    var inputs      := tf.zeros(input_shape);

    var sDims : TFShape := TFShape.Create([3, 2, 1]);
    var permute         := tf.keras.layers.Permute(sDims);
    var outputs         := permute.Apply(TFTensors.Create(inputs));

    Assert.IsTrue(TFShape.Create([2, 5, 4, 3]) =  outputs.shape);
end;

{ Keras_Losses_test }

constructor Keras_Losses_test.Create;
begin
    var aTrue : TArray< TArray<Single>> := [ [ 0.0, 1.0 ], [ 1.0, 1.0 ] ];
    var aPred : TArray< TArray<Single>> := [ [ 1.0, 0.0 ], [ 1.0, 1.0 ] ];
    y_true_float := TNDArray.Create(aTrue);
    y_pred_float := TNDArray.Create(aPred);

    var aTrue_H : TArray< TArray<Single>> := [ [ 0.0, 1.0 ], [ 0.0, 0.0 ] ];
    var aPred_H : TArray< TArray<Single>> := [ [ 0.6, 0.4 ], [ 0.4, 0.6 ] ];
    y_true_float_H := TNDArray.Create(aTrue_H);
    y_pred_float_H := TNDArray.Create(aPred_H);

    var aTrue_L : TArray< TArray<Single>> := [ [ 0.0, 1.0 ], [ 0.0, 0.0 ] ];
    var aPred_L : TArray< TArray<Single>> := [ [ 1.0, 1.0 ], [ 0.0, 0.0 ] ];
    y_true_float_L := TNDArray.Create(aTrue_L);
    y_pred_float_L := TNDArray.Create(aPred_L);

    var aTrue_MAE : TArray< TArray<Single>> := [ [ 0.0, 1.0 ], [ 0.0, 0.0 ] ];
    var aPred_MAE : TArray< TArray<Single>> := [ [ 1.0, 1.0 ], [ 1.0, 0.0 ] ];
    y_true_float_MAE := TNDArray.Create(aTrue_MAE);
    y_pred_float_MAE := TNDArray.Create(aPred_MAE);
end;


procedure Keras_Losses_test.BinaryCrossentropy;
begin
    // Example 1: (batch_size = 1, number of samples = 4)
    var aYT : TArray< Single> := [ 0, 1, 0, 0 ];
    var aYP : TArray< Single> := [ -18.6, 0.51, 2.94, -12.8 ];
    var y_true := tf.constant(aYT);
    var y_pred := tf.constant(aYP);
    var bce := tf.keras.losses.BinaryCrossentropy(true);
    var loss : NDArray := bce.Call(y_true, y_pred).numpy;
    Assert.AreEqual<Single>(Single(loss), 0.865458);

    // Example 2: (batch_size = 2, number of samples = 4)
    var aYT1 : TArray< TArray<Single>> := [ [ 0, 1 ], [ 0, 0 ] ];
    var aYP1 : TArray< TArray<Single>> := [ [ -18.6, 0.51 ], [ 2.94, -12.8 ] ];
    y_true := tf.constant(aYT1);
    y_pred := tf.constant(aYP1);
    bce := tf.keras.losses.BinaryCrossentropy(true);
    loss := bce.Call(y_true, y_pred).numpy;
    Assert.AreEqual<Single>(Single(loss), 0.865458);

    // Using 'sample_weight' attribute
    var sW : TArray< Single> := [0.8, 0.2 ];
    loss := bce.Call(y_true, y_pred, tf.constant(sW), ).numpy;
    Assert.AreEqual<Single>(Single(loss), 0.2436386);

    // Using 'sum' reduction` type.
    bce := tf.keras.losses.BinaryCrossentropy(true, 0, -1, Reduction.SUM);
    loss := bce.Call(y_true, y_pred).numpy;
    Assert.AreEqual<Single>(Single(loss), 1.730916);

    // Using 'none' reduction type.
    bce := tf.keras.losses.BinaryCrossentropy(true, 0, -1, Reduction.NONE);
    loss := bce.Call(y_true, y_pred).numpy;

    var expected := np.np_array<Single>([ 0.23515666, 1.4957594 ], np.np_float32);
    Assert.IsTrue(expected.Equals(loss));
end;

procedure Keras_Losses_test.CosineSimilarity_Default;
begin
    //>>> # Using 'auto'/'sum_over_batch_size' reduction type.
    //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1)
    //>>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
    //>>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
    //>>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    //>>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    //>>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
    //-0.5
    var loss := tf.keras.losses.CosineSimilarity('','', 1);
    var call := loss.Call(y_true_float, y_pred_float);

    var expected := TNDArray.create(Single(-0.49999997));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.CosineSimilarity_Sample_Weight;
begin
    //>>> # Calling with 'sample_weight'.
    //>>> cosine_loss(y_true, y_pred, sample_weight =[0.8, 0.2]).numpy()
    //- 0.0999
    var loss := tf.keras.losses.CosineSimilarity;
    var sample_weight := Numpy.np.np_array<Single>([0.8, 0.2],np.np_float32);
    var call := loss.Call(y_true_float, y_pred_float, sample_weight);

    var expected := TNDArray.create(Single(-0.099999994));
    Assert.IsTrue(expected.Equals(call.numpy));

end;

procedure Keras_Losses_test.CosineSimilarity_SUM;
begin
    //>>> # Using 'sum' reduction type.
    //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1,
    //...     reduction = tf.keras.losses.Reduction.SUM)
    //>>> cosine_loss(y_true, y_pred).numpy()
    //- 0.999
    var loss := tf.keras.losses.CosineSimilarity(ReductionV2.SUM,'', 1);
    var call := loss.Call(y_true_float, y_pred_float);

    var expected := TNDArray.create(Single(-0.99999994));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.CosineSimilarity_None;
begin
    //>>> # Using 'none' reduction type.
    //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1,
    //...     reduction = tf.keras.losses.Reduction.NONE)
    //>>> cosine_loss(y_true, y_pred).numpy()
    //array([-0., -0.999], dtype = float32)
    var loss := tf.keras.losses.CosineSimilarity(ReductionV2.NONE,'', 1);
    var call := loss.Call(y_true_float, y_pred_float);

    var expected := np.np_array<Single>([-0.0, -0.99999994],np.np_float32);
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.Huber_Default;
begin
    //>>> # Using 'auto'/'sum_over_batch_size' reduction type.
    //>>> h = tf.keras.losses.Huber()
    //>>> h(y_true, y_pred).numpy()
    //0.155
    var loss := tf.keras.losses.Huber;
    var call := loss.Call(y_true_float_H, y_pred_float_H);

    var expected := TNDArray.create(Single(0.155));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.Huber_Sample_Weight;
begin
    //>>> # Calling with 'sample_weight'.
    //>>> h(y_true, y_pred, sample_weight =[1, 0]).numpy()
    //0.09
    var loss := tf.keras.losses.Huber;
    var sample_weight := Numpy.np.np_array<Single>([0.1, 0.0],np.np_float32);
    var call := loss.Call(y_true_float_H, y_pred_float_H, sample_weight);

    var expected := TNDArray.create(Single(0.009000001));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.Huber_SUM;
begin
    //>>> # Using 'sum' reduction type.
    //>>> h = tf.keras.losses.Huber(
    //...     reduction = tf.keras.losses.Reduction.SUM)
    //>>> h(y_true, y_pred).numpy()
    //0.31
    var loss := tf.keras.losses.Huber(ReductionV2.SUM);
    var call := loss.Call(y_true_float_H, y_pred_float_H);

    var expected := TNDArray.create(Single(0.31));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.Huber_None;
begin
    //>>> # Using 'none' reduction type.
    //>>> h = tf.keras.losses.Huber(
    //...     reduction = tf.keras.losses.Reduction.NONE)
    //>>> h(y_true, y_pred).numpy()
    //array([0.18, 0.13], dtype = float32)
    var loss := tf.keras.losses.Huber(ReductionV2.NONE);
    var call := loss.Call(y_true_float_H, y_pred_float_H);

    var expected := np.np_array<Single>([0.18, 0.13000001],np.np_float32);
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.LogCosh_Default;
begin
    //>>> # Using 'auto'/'sum_over_batch_size' reduction type.
    //>>> l = tf.keras.losses.LogCosh()
    //>>> l(y_true, y_pred).numpy()
    //0.108
    var loss := tf.keras.losses.LogCosh;
    var call := loss.Call(y_true_float_L, y_pred_float_L);

    var expected := TNDArray.create(Single(0.1084452));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.LogCosh_Sample_Weight;
begin
    //>>> # Calling with 'sample_weight'.
    //>>> l(y_true, y_pred, sample_weight =[0.8, 0.2]).numpy()
    //0.087
    var loss := tf.keras.losses.LogCosh;
    var sample_weight := Numpy.np.np_array<Single>([0.8, 0.2],np.np_float32);
    var call := loss.Call(y_true_float_L, y_pred_float_L, sample_weight);

    var expected := TNDArray.create(Single(0.08675616));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.LogCosh_SUM;
begin
    //>>> # Using 'sum' reduction type.
    //>>> l = tf.keras.losses.LogCosh(
    //...     reduction = tf.keras.losses.Reduction.SUM)
    //>>> l(y_true, y_pred).numpy()
    //0.217
    var loss := tf.keras.losses.LogCosh(ReductionV2.SUM);
    var call := loss.Call(y_true_float_L, y_pred_float_L);

    var expected := TNDArray.create(Single(0.2168904));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.LogCosh_None;
begin
    //>>> # Using 'none' reduction type.
    //>>> l = tf.keras.losses.LogCosh(
    //...     reduction = tf.keras.losses.Reduction.NONE)
    //>>> l(y_true, y_pred).numpy()
    //array([0.217, 0.], dtype = float32)
    var loss := tf.keras.losses.LogCosh(ReductionV2.NONE);
    var call := loss.Call(y_true_float_L, y_pred_float_L);

    var expected := np.np_array<Single>([0.2168904, 0.0],np.np_float32);
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.MeanAbsoluteError_Default;
begin
    //>>> # Using 'auto'/'sum_over_batch_size' reduction type.
    //>>> mae = tf.keras.losses.MeanAbsoluteError()
    //>>> mae(y_true, y_pred).numpy()
    //0.5
    var loss := tf.keras.losses.MeanAbsoluteError;
    var call := loss.Call(y_true_float_MAE, y_pred_float_MAE);

    var expected := TNDArray.create(Single(0.5));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.MeanAbsoluteError_Sample_Weight;
begin
    //>>> # Calling with 'sample_weight'.
    //>>> mae(y_true, y_pred, sample_weight =[0.7, 0.3]).numpy()
    //0.25
    var loss := tf.keras.losses.MeanAbsoluteError;
    var sample_weight := np.np_array<Single>([0.7, 0.3],np.np_float32);
    var call := loss.Call(y_true_float_MAE, y_pred_float_MAE, sample_weight);

    var expected := TNDArray.create(Single(0.25));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.MeanAbsoluteError_SUM;
begin
    //>>> # Using 'sum' reduction type.
    //>>> mae = tf.keras.losses.MeanAbsoluteError(
    //...     reduction = tf.keras.losses.Reduction.SUM)
    //>>> mae(y_true, y_pred).numpy()
    //1.0
    var loss := tf.keras.losses.MeanAbsoluteError(ReductionV2.SUM);
    var call := loss.Call(y_true_float_MAE, y_pred_float_MAE);

    var expected := TNDArray.create(Single(1.0));
    Assert.IsTrue(expected.Equals(call.numpy));
end;

procedure Keras_Losses_test.SigmoidFocalCrossEntropy;
begin
    var y_true := np.expand_dims( np.np_array<Single>([1.0, 1.0, 0],np.np_float32),-1 );
    var y_pred := np.expand_dims( np.np_array<Single>([0.97, 0.91, 0.03],np.np_float32),-1 );
    var bce := tf.keras.losses.SigmoidFocalCrossEntropy;
    var loss := bce.Call(y_true, y_pred);
    var a : TArray<Single> := [ 6.8532745e-06, 1.909787e-04, 2.0559824e-05];
    Assert.IsTrue( TUtils.SequenceEqual<Single>(a, loss.numpy.ToArray<Single>) );
end;

procedure Keras_Losses_test.MeanAbsoluteError_None;
begin
    //>>> # Using 'none' reduction type.
    //>>> mae = tf.keras.losses.MeanAbsoluteError(
    //...     reduction = tf.keras.losses.Reduction.NONE)
    //>>> mae(y_true, y_pred).numpy()
    //array([0.5, 0.5], dtype = float32)
    var loss := tf.keras.losses.MeanAbsoluteError(ReductionV2.NONE);
    var call := loss.Call(y_true_float_MAE, y_pred_float_MAE);

    var expected := np.np_array<Single>([0.5, 0.5],np.np_float32);
    Assert.IsTrue(expected.Equals(call.numpy));
end;

{ LayersTest }

procedure LayersTest.AveragePooling2D;
begin
    var aInput : TArray< TArray<Single> > := [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]];
    var x  := tf.constant( aInput );
    x      := tf.reshape( x, TFShape.Create([1, 3, 3, 1]) );

    var avg_pool_2d    := tf.keras.layers.AveragePooling2D(TFShape.Create([2,2]), TFShape.Create([1,1]), 'valid');
    var avg : TFTensor := avg_pool_2d.Apply( TFTensors.Create(x)).First;

    Assert.IsTrue(TFShape.Create([1, 2, 2, 1])= avg.shape);
    var expected : TArray<Single> := [ 3, 4, 6, 7 ];
    Assert.IsTrue(Equal( expected, avg.ToArray<Single>),'Assert - AveragePooling2D');
end;

procedure LayersTest.InputLayer;
begin
    var lLayers := TList<ILayer>.Create;
    lLayers.Add( tf.keras.layers.InputLayer(TFShape.Create([4])) );
    lLayers.Add( tf.keras.layers.Dense(8) );

    var model := TKerasApi.keras.Sequential(lLayers );
    model.OnEpochBegin      :=  On_Epoch_Begin;
    model.OnTrainBatchBegin :=  On_Train_Batch_Begin;

    model.compile(tf.keras.optimizers.RMSprop(Single(0.001)), tf.keras.losses.MeanSquaredError, ['accuracy']);
    model.fit(np.zeros(TFShape.Create([10, 4]), tf.float32_t), np.ones(TFShape.Create([10, 8]), tf.float32_t));
end;

procedure LayersTest.Sequential;
begin
   var model := TKerasApi.keras.Sequential;
   model.add(tf.keras.Input(TFShape.Create([16])) );
end;

procedure LayersTest.Functional;
begin
    var layers := tf.keras.layers;
    var inputs := tf.keras.Input(TFShape.Create([784])) ;
    Assert.IsTrue(TFShape.Create([-1, 784]) = inputs.shape);

    var dense := layers.Dense(64, tf.keras.activations.Relu);
    var x     := dense.Apply( TFTensors.Create(inputs));

    x           := layers.Dense(64, tf.keras.activations.Relu).Apply(x);
    var outputs := layers.Dense(10).Apply(x);

    var model := tf.keras.Model( TFTensors.Create(inputs), outputs, 'mnist_model') ;
    model.OnEpochBegin      :=  On_Epoch_Begin;
    model.OnTrainBatchBegin :=  On_Train_Batch_Begin;
    model.OnEndSummary      :=  On_End_Summary;
    model.summary;
end;

procedure LayersTest.TensorFlowOpLayer;
var
  mean   : TTensor;
  adv    : TTensor;
  value  : TFTensor;
  inputs : TFTensors;
  x      : TFTensors;
begin

    var l_layers := tf.keras.layers;
    inputs   := l_layers.Input( TFShape.Create([24]) );
    x        := l_layers.Dense(128, 'relu').Apply(inputs);
    value    := l_layers.Dense(24).Apply(x).first;
    adv      := l_layers.Dense(1).Apply(x).First;

    var aAxis : TAxis := 1;
    mean         := adv - tf.reduce_mean(adv, @aAxis, true);;
    adv          := l_layers.Subtract.Apply(TFTensors.Create([adv, mean])).first;
    var outputs  := l_layers.Add.Apply(TFTensors.Create([value, adv]));
    var model    := tf.keras.Model(inputs, outputs);

    model.OnEpochBegin      :=  On_Epoch_Begin;
    model.OnTrainBatchBegin :=  On_Train_Batch_Begin;
    model.OnEndSummary      :=  On_End_Summary;

    model.compile(tf.keras.optimizers.RMSprop(Single(0.001)), tf.keras.losses.MeanSquaredError, [ 'acc' ]);
    model.summary;

    Assert.AreEqual(model.Layers.Count, 8);

    var res := model.predict(TFTensors.Create( tf.constant(np.arange(24).astype(np.np_float32)[ [np.newaxis, Slice.All] ]) ));

    Assert.Istrue(res.shape= TFShape.Create([1, 24]));
    model.fit(np.arange(24).astype(np.np_float32)[[np.newaxis, Slice.All]], np.arange(24).astype(np.np_float32)[[np.newaxis, Slice.All]],{Batch_Size} -1,{Epochs} 1,{Verbose} 0);
end;

procedure LayersTest.Embedding;
begin
    var model := TKerasApi.keras.Sequential;
    var layer := tf.keras.layers.Embedding(1000, 64, {embeddings_initializer}nil, {mask_zero}False, {input_shape}nil, {input_length}10);
    model.add(layer);

    var size : TFShape := TFShape.Create([32, 10]);
    var input_array := np.random.randint(1000, nil, @size);

    model.compile('rmsprop', 'mse', [ 'accuracy' ]);

    var output_array := model.predict(TFTensors.Create( input_array ));

    Assert.IsTrue(TFShape.Create([32, 10, 64]) =  output_array.shape);
end;

procedure LayersTest.Dense;
begin
    // Create a `Sequential` model and add a Dense layer as the first layer.
    var model := TKerasApi.keras.Sequential;
    model.add(tf.keras.Input(TFShape.Create([16])));
    model.add(tf.keras.layers.Dense(32, tf.keras.activations.Relu));
    // Now the model will take as input arrays of shape (None, 16)
    // and output arrays of shape (None, 32).
    // Note that after the first layer, you don't need to specify
    // the size of the input anymore:
    model.add(tf.keras.layers.Dense(32));
    Assert.IsTrue(TFShape.Create([-1, 32]) = model.OutputShape );
end;

procedure LayersTest.EinsumDense;
begin
    var ed := tf.keras.layers.EinsumDense('...b,bc->...c', TFShape.Create([4]), 'c', nil, tf.constant_initializer(0.5), tf.constant_initializer(0.03) );

    var inp := np.np_array<Single>([ [ 1, 2 ], [ 3, 4 ] ],np.np_float32);

    var expected_output := np.np_array<Single>( [[1.53, 1.53, 1.53, 1.53  ], [3.53, 3.53, 3.53, 3.53 ]],np.np_float32);
    var actual_output  := ed.Apply(TFtensors.create(inp))[0].numpy;
    Assert.IsTrue(expected_output.Equals(actual_output));
end;

procedure LayersTest.SimpleRNN;
begin
(*


            tf.UseKeras<KerasInterface>();
            var inputs = np.arange(6 * 10 * 8).reshape((6, 10, 8)).astype(np.float32);
            /*var simple_rnn = keras.layers.SimpleRNN(4);
            var output = simple_rnn.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);*/
            var simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences: true, return_state: true);
            var (whole_sequence_output, final_state) = simple_rnn.Apply(inputs);
*)
end;

procedure LayersTest.Resizing;
begin
    var inputs := tf.random.uniform(TFShape.Create([10, 32, 32, 3]));
    var layer  := tf.keras.layers.preprocessing.Resizing(16, 16);
    var output := layer.Apply(TFTensors.Create(inputs));
    Assert.IsTrue(TFShape.Create([10, 16, 16, 3]) = output.shape );
end;

procedure LayersTest.LayerNormalization;
begin
    var inputs := tf.constant( TTensor(np.arange(10).reshape(TFShape.Create([5, 2]))) * Integer(10) , tf.float32_t);
    var layer  := tf.keras.layers.LayerNormalization(1);
    var output : TFTensor := layer.Apply(TFTensors.Create(inputs)).first;
    Assert.IsTrue(TFShape.Create([5, 2]) = output.shape );

    var expected : TArray<Single> := [ -0.99998, 0.99998 ];

    Assert.IsTrue(Equal( expected, output[0].ToArray<Single>),'Assert - LayerNormalization');
end;

{ PreprocessingTests }

constructor PreprocessingTests.Create;
begin
    inherited Create;

    texts := ['It was the best of times, it was the worst of times.',
              'Mr and Mrs Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.',
              'It was the best of times, it was the worst of times.',
              'Mr and Mrs Dursley of number four, Privet Drive.'];

    tokenized_texts := [['It','was','the','best','of','times','it','was','the','worst','of','times'],
                        ['mr','and','mrs','dursley','of','number','four','privet','drive','were','proud','to','say','that','they','were','perfectly','normal','thank','you','very','much'],
                        ['It','was','the','best','of','times','it','was','the','worst','of','times'],
                        ['mr','and','mrs','dursley','of','number','four','privet','drive']];

    processed_texts := ['it was the best of times it was the worst of times',
                        'mr and mrs dursley of number four privet drive were proud to say that they were perfectly normal thank you very much',
                        'it was the best of times it was the worst of times',
                        'mr and mrs dursley of number four privet drive'];

    OOV := '<OOV>';
end;

destructor PreprocessingTests.Destroy;
begin
  texts := [];
  tokenized_texts := [];
  processed_texts := [];
  OOV := '';

  inherited Destroy;
end;

procedure PreprocessingTests.TokenizeWithNoOOV;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create;
    tTokenizer.fit_on_texts(texts);

    Assert.AreEqual(27, tTokenizer.word_index.Count);

    Assert.AreEqual(7, tTokenizer.word_index['worst']);
    Assert.AreEqual(13, tTokenizer.word_index['number']);
    Assert.AreEqual(9, tTokenizer.word_index['were']);
end;

procedure PreprocessingTests.TokenizeWithNoOOV_Tkn;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create;
    tTokenizer.fit_on_texts(tokenized_texts);

    Assert.AreEqual(27, tTokenizer.word_index.Count);

    Assert.AreEqual(7, tTokenizer.word_index['worst']);
    Assert.AreEqual(13, tTokenizer.word_index['number']);
    Assert.AreEqual(9, tTokenizer.word_index['were']);
end;

procedure PreprocessingTests.TokenizeWithOOV;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create(OOV);
    tTokenizer.fit_on_texts(texts);

    Assert.AreEqual(28, tTokenizer.word_index.Count);

    Assert.AreEqual(1,  tTokenizer.word_index[OOV]);
    Assert.AreEqual(8,  tTokenizer.word_index['worst']);
    Assert.AreEqual(14, tTokenizer.word_index['number']);
    Assert.AreEqual(10, tTokenizer.word_index['were']);
end;

procedure PreprocessingTests.TokenizeWithOOV_Tkn;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create(OOV);
    tTokenizer.fit_on_texts(tokenized_texts);

    Assert.AreEqual(28, tTokenizer.word_index.Count);

    Assert.AreEqual(1,  tTokenizer.word_index[OOV]);
    Assert.AreEqual(8,  tTokenizer.word_index['worst']);
    Assert.AreEqual(14, tTokenizer.word_index['number']);
    Assert.AreEqual(10, tTokenizer.word_index['were']);
end;

procedure PreprocessingTests.TokenizeTextsToSequences;
var
 tTokenizer : Tokenizer;
 sequences  : TList<TArray<Integer>>;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer;
    tTokenizer.fit_on_texts(texts);

    sequences := tTokenizer.texts_to_sequences( texts );
    Assert.AreEqual(4, sequences.Count);

    Assert.AreEqual(tTokenizer.word_index['worst'], sequences[0][9]);
    Assert.AreEqual(tTokenizer.word_index['proud'], sequences[1][10]);
end;

procedure PreprocessingTests.TokenizeTextsToSequences_Tkn;
var
 tTokenizer : Tokenizer;
 sequences  : TList<TArray<Integer>>;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer;
    tTokenizer.fit_on_texts(tokenized_texts);

    sequences := tTokenizer.texts_to_sequences( tokenized_texts );
    Assert.AreEqual(4, sequences.Count);

    Assert.AreEqual(tTokenizer.word_index['worst'], sequences[0][9]);
    Assert.AreEqual(tTokenizer.word_index['proud'], sequences[1][10]);
end;

procedure PreprocessingTests.PadSequencesWithDefaults;
var
 tTokenizer : Tokenizer;
 sequences  : TList<TArray<Integer>>;
 padded     : TNDArray;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create(OOV);
    tTokenizer.fit_on_texts(texts);

    sequences := tTokenizer.texts_to_sequences(texts);
    padded    := TKerasApi.keras.preprocessing.sequence.pad_sequences(sequences);

    Assert.AreEqual<Int64>(4, padded.dims[0]);
    Assert.AreEqual<Int64>(22, padded.dims[1]);

    var ndPadded : NDArray := padded;

    var iPad: Integer := ndPadded[[0, 19]];
    Assert.AreEqual(iPad, tTokenizer.word_index['worst']);

    for var i := 0 to 8 - 1 do
    begin
        iPad := ndPadded[[0, i]];
        Assert.AreEqual(iPad, 0);
    end;

    iPad := ndPadded[[1, 10]];
    Assert.AreEqual(iPad, tTokenizer.word_index['proud']);

    for var i := 0 to 20 -1 do
    begin
        iPad := ndPadded[[1, i]];
        Assert.AreNotEqual(iPad, 0);
    end;
end;

procedure PreprocessingTests.TextToMatrixBinary;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create;
    tTokenizer.fit_on_texts(texts);

    Assert.AreEqual(27, tTokenizer.word_index.Count);

    var matrix := tTokenizer.texts_to_matrix(texts);

    Assert.AreEqual<int64>(Length(texts), matrix.dims[0]);

    Assert.IsTrue(TUtils.SequenceEqual<double>([ 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], matrix[0].ToArray<double> ));
    Assert.IsTrue(TUtils.SequenceEqual<double>([ 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], matrix[1].ToArray<double> ));
end;

procedure PreprocessingTests.TextToMatrixFrequency;
var
 tTokenizer : Tokenizer;
begin
    tTokenizer := TKerasApi.keras.preprocessing.text.Tokenizer.Create;
    tTokenizer.fit_on_texts(texts);

    Assert.AreEqual(27, tTokenizer.word_index.Count);

    var matrix := tTokenizer.texts_to_matrix(texts, 'freq');

    Assert.AreEqual<Int64>(Length(texts), matrix.dims[0]);

    var t12 : double := 2.0 / 12.0;
    var o12 : double := 1.0 / 12.0;
    var t22 : double := 2.0 / 22.0;
    var o22 : double := 1.0 / 22.0;

    var r1 := matrix[0].ToArray<double>;
    var r2 := matrix[1].ToArray<double>;

    Assert.IsTrue(TUtils.SequenceEqual<double>([ 0, t12, t12, t12, t12, t12, 0, o12, 0, 0, 0, 0, o12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], matrix[0].ToArray<double> ));
    Assert.IsTrue(TUtils.SequenceEqual<double>([ 0, o22, 0, 0, 0, 0, o22, 0, o22, t22, o22, o22, 0, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22 ], matrix[1].ToArray<double> ));
end;



end.
