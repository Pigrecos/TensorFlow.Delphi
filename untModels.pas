unit untModels;

interface
        uses  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
              Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti, System.Math,  System.Generics.Collections,

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
              TensorFlow.Slice,

              Keras.Engine,
              keras.Data,
              Keras.KerasApi,
              Keras.ArgsDefinition,
              keras.Models,
              Keras.Layer,
              keras.Optimizer,

              TensorFlow.Variable,
              TensorFlow.Tensor,
              NumPy.NDArray,
              Numpy.Axis;

type
  ExampleConfig = class
    /// <summary>
    /// Example name
    /// </summary>
    Name : string;

    /// <summary>
    /// True to run example
    /// </summary>
    Enabled : boolean ;

    /// <summary>
    /// Set true to import the computation graph instead of building it.
    /// </summary>
    IsImportingGraph : boolean ;
  end;

  /// <summary>
  /// Interface of Example project
  /// Each example should implement IExample so the entry program will find it.
  /// </summary>
  IExample = interface
  ['{26A60407-D1B3-4FAA-B1B4-616376E5441A}']

      function  GetConf: ExampleConfig;
      procedure SetConf(const value: ExampleConfig);

      function InitConfig: ExampleConfig;
      function Run: Boolean;

      procedure BuildModel;

      /// <summary>
      /// Build dataflow graph, train and predict
      /// </summary>
      /// <returns></returns>
      procedure Train;
      function FreezeModel : string;
      procedure Test;

      procedure Predict;

      function ImportGraph: TFGraph;

      function BuildGraph: TFGraph;

      /// <summary>
      /// Prepare dataset
      /// </summary>
      procedure PrepareData;

      property Config : ExampleConfig read GetConf write SetConf;
  end;

  ConvNetArgs = class(ModelArgs)
     NumClasses  : Integer;

     constructor Create;
  end;

  ConvNet  = class(Model)
      private
        conv1    : ILayer;
        maxpool1 : ILayer;
        conv2    : ILayer;
        maxpool2 : ILayer ;
        flatten  : ILayer ;
        fc1      : ILayer;
        dropout  : ILayer ;
        output   : ILayer;
      protected
        function  Call(inputs: TFTensors; state: TFTensor = nil; training : pBoolean= nil): TFTensors; override;
      public
        constructor Create(args: ConvNetArgs );
  end;

  DigitRecognitionCnnKeras  = class(TInterfacedObject, IExample)
    private
      FConfig : ExampleConfig;
      // MNIST dataset parameters.
      num_classes : Integer;

      // Training parameters.
      learning_rate   : Single;
      training_steps  : Integer;
      batch_size      : Integer;
      display_step    : Integer;

      accuracy_test   : Single;

      train_data      : IDatasetV2;
      x_test, y_test,
      x_train, y_train: TNDArray;
      logMsg          : TStringList;

      function  GetConf: ExampleConfig;
      procedure SetConf(const value: ExampleConfig);
    protected


      procedure BuildModel;
      procedure Train;
      function  FreezeModel : string;
      procedure Test;
      procedure Predict;
      function  ImportGraph: TFGraph;
      function  BuildGraph: TFGraph;
      procedure PrepareData;
    public

      constructor Create;
      function    InitConfig: ExampleConfig;
      function    Run: Boolean;
      destructor  Destroy; override;
      procedure   run_optimization(conv_net: ConvNet; optimizer: OptimizerV2; x: TFTensor; y: TFTensor);
      function    cross_entropy_loss(x: TFTensor; y: TFTensor) : TFTensor;
      function    accuracy(y_pred: TFTensor; y_true: TFTensor): TFTensor;

      property Log : TStringList read logMsg;
  end;


implementation
            uses untMain;
{ ConvNetArgs }

constructor ConvNetArgs.Create;
begin
    inherited Create
end;

{ ConvNet }

constructor ConvNet.Create(args: ConvNetArgs);
begin
    inherited Create(args);

    var layers := tf.keras.layers;

    var kernel_size : TFShape := TFShape.Create([5]);
    // Convolution Layer with 32 filters and a kernel size of 5.
    conv1 := layers.Conv2D(32, @kernel_size, {strides}nil, {padding}'valid', {data_format}'', {dilation_rate}nil, {groups}1, tf.keras.activations.Relu);

    var poolSize : TFShape := TFShape.Create([2]);
    var strides  : TFShape := TFShape.Create([2]);
    // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
    maxpool1 := layers.MaxPooling2D(@poolSize, @strides);

    var kernel_size1 : TFShape := TFShape.Create([3]);
    // Convolution Layer with 64 filters and a kernel size of 3.
    conv2 := layers.Conv2D(64, @kernel_size1, {strides}nil, {padding}'valid', {data_format}'', {dilation_rate}nil, {groups}1, tf.keras.activations.Relu);

    var poolSize1 : TFShape := TFShape.Create([2]);
    var strides1  : TFShape := TFShape.Create([2]);
    // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
    maxpool2 := layers.MaxPooling2D(@poolSize1, @strides1);

    // Flatten the data to a 1-D vector for the fully connected layer.
    flatten := layers.Flatten;

    // Fully connected layer.
    fc1 := layers.Dense(1024);
    // Apply Dropout (if is_training is False, dropout is not applied).
    dropout := layers.Dropout(0.5);

    // Output layer, class prediction.
    output := layers.Dense(args.NumClasses);

    StackLayers([conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output]);
end;

function ConvNet.Call(inputs: TFTensors; state: TFTensor; training: pBoolean): TFTensors;
begin
    inputs := TFTensors.Create( tf.reshape(inputs.first, TFShape.Create([-1, 28, 28, 1])) );
    inputs := conv1.Apply(inputs);
    inputs := maxpool1.Apply(inputs);
    inputs := conv2.Apply(inputs);
    inputs := maxpool2.Apply(inputs);
    inputs := flatten.Apply(inputs);
    inputs := fc1.Apply(inputs);
    inputs := dropout.Apply(inputs, @training);
    inputs := output.Apply(inputs);

    if not training^ then
        inputs := TFTensors.Create(tf.nn.softmax(inputs.first));

    Result := inputs;
end;

{ DigitRecognitionCnnKeras }

constructor DigitRecognitionCnnKeras.Create;
begin
    // MNIST dataset parameters.
    num_classes := 10;

    // Training parameters.
    learning_rate := 0.001;
    training_steps:= 100;
    batch_size    := 128;
    display_step  := 10;

    accuracy_test := 0.0;

    logMsg  := TStringList.Create;
end;

destructor DigitRecognitionCnnKeras.Destroy;
begin
    logMsg.Free;
end;

function DigitRecognitionCnnKeras.InitConfig: ExampleConfig;
var
  config : ExampleConfig;
begin
    config := ExampleConfig.Create;
    config.Name := 'MNIST CNN (Keras Subclass)';
    config.Enabled := true;
    config.IsImportingGraph := false;

    Result := config;
end;

function DigitRecognitionCnnKeras.Run: Boolean;
begin
    tf.enable_eager_execution;

    PrepareData;

    Train;

    Result := accuracy_test > 0.85;
end;

procedure DigitRecognitionCnnKeras.Train;
var
  cArg      : ConvNetArgs;
  optimizer : OptimizerV2;
begin
    cArg := ConvNetArgs.Create;
    cArg.NumClasses := num_classes;
    // Build neural network model.
    var conv_net := ConvNet.Create(cArg);

    // ADAM optimizer.
    optimizer := tf.keras.optimizers.Adam(learning_rate);

    var step : Integer := 0;
    // Run training for the given number of steps.
    for var tTrain in train_data do
    begin
         Inc(step);
         var batch_x := tTrain.Value1;
         var batch_y := tTrain.Value2;
        // Run the optimization to update W and b values.
        run_optimization(conv_net, optimizer, batch_x, batch_y);

        if step mod display_step = 0 then
        begin
            var pred := conv_net.Apply(TFTensors.Create(batch_x));
            var loss : TTensor := cross_entropy_loss(pred.first, batch_y);
            var acc  : TTensor := accuracy(pred.first, batch_y);
            var fLoss := Single(loss);
            var facc  := Single(acc);
            logMsg.Add( Format('step: %d, loss: %.3f, accuracy: %.3f',[step, floss,facc]) );
            frmMain.mmo1.lines.Add(logMsg.Strings[ logMsg.Count-1 ])
        end;
    end;

    // Test model on validation set.
    x_test := x_test['::100'].numpy;
    y_test := y_test['::100'].numpy;
    var pred := conv_net.Apply(TFTensors.Create(x_test));
    var aTest : TTensor := accuracy(pred.First, y_test);
    accuracy_test :=  Single(aTest);
    logMsg.Add('Test Accuracy: ' + accuracy_test.ToString);
    frmMain.mmo1.lines.Add(logMsg.Strings[ logMsg.Count-1 ]) ;

    conv_net.save_weights( 'weights.h5', true);
end;

procedure DigitRecognitionCnnKeras.Test;
var
  cArg      : ConvNetArgs;
begin
    cArg := ConvNetArgs.Create;
    cArg.NumClasses := num_classes;
    // Build neural network model.
    var conv_net := ConvNet.Create(cArg);

     // Test model on validation set.
    x_test := x_test['::100'].numpy;
    y_test := y_test['::100'].numpy;

    conv_net.load_weights('weights.h5');

    var pred := conv_net.Apply(TFTensors.Create(x_test));
    var aTest : TTensor := accuracy(pred.First, y_test);
    accuracy_test :=  Single(aTest);
    logMsg.Add('Test Accuracy: ' + accuracy_test.ToString);

end;

procedure DigitRecognitionCnnKeras.run_optimization(conv_net: ConvNet; optimizer: OptimizerV2; x: TFTensor; y: TFTensor);
begin
    var g    := tf.GradientTape;
    var pred := conv_net.Apply(TFTensors.Create(x), nil, true);
    var loss := cross_entropy_loss(pred.first, y);

    // Compute gradients.
    var gradients := g.gradient(loss, conv_net.TrainableVariables.ToArray);

    // Update W and b following gradients.
    var g_v : TArray< Tuple<TFTensor,ResourceVariable> > := [];
    for var i := 0 to Length(gradients) - 1 do
    begin
        g_v := g_v + [ Tuple.Create(gradients[i], conv_net.TrainableVariables[i] as ResourceVariable) ];
    end;

    optimizer.apply_gradients(g_v);
end;

function DigitRecognitionCnnKeras.cross_entropy_loss(x: TFTensor; y: TFTensor) : TFTensor;
begin
    // Convert labels to int 64 for tf cross-entropy function.
    y := tf.cast(y, tf.int64_t);
    // Apply softmax to logits and compute cross-entropy.
    var loss := tf.nn.sparse_softmax_cross_entropy_with_logits(y, x);
    // Average loss across the batch.
    Result := tf.reduce_mean(loss);
end;

function DigitRecognitionCnnKeras.accuracy(y_pred: TFTensor; y_true: TFTensor): TFTensor;
begin
    // # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    var correct_prediction := tf.equal(tf.math.argmax(y_pred, 1), tf.cast(y_true, tf.int64_t));
    var axix : TAxis := -1;
    Result := tf.reduce_mean(tf.cast(correct_prediction, tf.float32_t), @axix);
end;

procedure DigitRecognitionCnnKeras.PrepareData;
begin
    var dp : DatasetPass := tf.keras.datasets.mnist.load_data;
    x_train := dp.Train.Value1;
    y_train := dp.Train.Value2;

    x_test  := dp.Test.Value1;
    y_test  := dp.Test.Value2;

    // Normalize images value from [0, 255] to [0, 1].
    x_train := NDArray(x_train) / Single(255.0);
    x_test  := NDArray(x_test) / Single(255.0);

    train_data := tf.data.Dataset.from_tensor_slices(x_train, y_train);
    train_data := train_data.&repeat.shuffle(5000).batch(batch_size).prefetch(1).take(training_steps);
end;

procedure DigitRecognitionCnnKeras.SetConf(const value: ExampleConfig);
begin
    FConfig := value;
end;

function DigitRecognitionCnnKeras.GetConf: ExampleConfig;
begin
    Result := FConfig
end;

procedure DigitRecognitionCnnKeras.Predict;
begin

end;

function DigitRecognitionCnnKeras.BuildGraph: TFGraph;
begin
   Result := nil;
end;

procedure DigitRecognitionCnnKeras.BuildModel;
begin

end;

function DigitRecognitionCnnKeras.FreezeModel: string;
begin

end;

function DigitRecognitionCnnKeras.ImportGraph: TFGraph;
begin
     Result := nil;
end;

end.
