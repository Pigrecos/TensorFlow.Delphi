unit untModels;

interface
        uses  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
              Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti, System.Math,  System.Generics.Collections,

              spring,

              TF4D.Core.CApi,

              Numpy,
              Tensorflow,
              Tensorflow.Core,
              TensorFlow.DApiBase,
              TensorFlow.DApi,
              Tensorflow.Utils,

              Keras.Core,
              keras.LayersApi,
              keras.Data,
              Keras.KerasApi,
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

  /// <summary>
  /// Build a convolutional neural network with TensorFlow v2.
  /// This example is using a low-level approach to better understand all mechanics behind building convolutional neural networks and the training process.
  /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
  /// </summary>
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
      procedure   run_optimization(conv_net: ConvNet; optimizer: IOptimizer; x: TFTensor; y: TFTensor);
      function    cross_entropy_loss(x: TFTensor; y: TFTensor) : TFTensor;
      function    accuracy(y_pred: TFTensor; y_true: TFTensor): TFTensor;

      property Log : TStringList read logMsg;
      property Config : ExampleConfig  read FConfig;
  end;

  MnistFnnKerasFunctional  = class(TInterfacedObject, IExample)
    private
      FConfig : ExampleConfig;
      // MNIST dataset parameters.
      mModel : IModel;
      layers : LayersApi;

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

      property Log : TStringList read logMsg;
      property Config : ExampleConfig  read FConfig;
  end;

  /// <summary>
  /// https://www.tensorflow.org/tutorials/generative/dcgan
  /// AtCode:JG5FLDRWHY9FEZ9S V559.83530 Provided by big crabs
  /// </summary>
  TMnistGAN = class(TInterfacedObject, IExample)
    private
      fTrainCount : Integer;

      FConfig : ExampleConfig;
      // MNIST dataset parameters.
      LeakyReLU_alpha : Single;
      imgpath    : string;
      modelpath  : string;
      img_shape  : TFShape;
      noise_dim  : Integer;
      img_rows   : Integer;
      img_cols   : Integer;
      channels   : Integer;
      EPOCHS     : Integer;
      BATCH_SIZE : Integer;
      BUFFER_SIZE: Integer;

      data         : DatasetPass;
      train_images : TNDArray;
      layers       : LayersApi;
      train_dataset: IDatasetV2;

      discriminator : Model;
      generator     : Model;

      discriminator_optimizer : IOptimizer;
      generator_optimizer     : IOptimizer;

      logMsg   : TStringList;

      function  Make_Generator_model: Model;
      function  Make_Discriminator_model: Model;
      function  GetConf: ExampleConfig;
      procedure SetConf(const value: ExampleConfig);
      procedure PredictImage(g: Model; step: Integer);
      function  cross_entropy(x, y: TTensor): TFTensor;
      procedure SaveImage(gen_imgs: TNDArray; step: Integer);
      procedure Train_step(images: TFTensor);
      function  discriminator_loss(real_output, fake_output: TFTensor): TFTensor;
      function  generator_loss(fake_output: TFTensor): TFTensor;
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

      property Log : TStringList read logMsg;
      property Config : ExampleConfig  read FConfig;
  end;

  TKerasUnitsTest = class
     class procedure LeNetModel;
  end;


implementation
            uses System.IOUtils,
                 untMain,Esempi;

class procedure TKerasUnitsTest.LeNetModel ;
begin
    var inputs := tf.keras.Input(TFShape.Create([28, 28, 1]));
    var conv1  := tf.keras.layers.Conv2D(16, TFShape.Create([3, 3]), 'relu', 'same').Apply(inputs);
    var pool1  := tf.keras.layers.MaxPooling2D(TFShape.Create([2, 2]), 2).Apply(conv1);
    var conv2  := tf.keras.layers.Conv2D(32, TFShape.Create([3, 3]), 'relu', 'same').Apply(pool1);
    var pool2  := tf.keras.layers.MaxPooling2D(TFShape.Create([2, 2]), 2).Apply(conv2);
    var flat1  := tf.keras.layers.Flatten.Apply(pool2);

    var inputs_2 := tf.keras.Input(TFShape.Create([28, 28, 1]));
    var conv1_2  := tf.keras.layers.Conv2D(16, TFShape.Create([3, 3]), 'relu', 'same').Apply(inputs_2);
    var pool1_2  := tf.keras.layers.MaxPooling2D(TFShape.Create([4, 4]), 4).Apply(conv1_2);
    var conv2_2  := tf.keras.layers.Conv2D(32, TFShape.Create([1, 1]), 'relu', 'same').Apply(pool1_2);
    var pool2_2  := tf.keras.layers.MaxPooling2D(TFShape.Create([2, 2]), 2).Apply(conv2_2);
    var flat1_2  := tf.keras.layers.Flatten.Apply(pool2_2);

    var concat := tf.keras.layers.Concatenate.Apply( TFTensors.Create(Tuple.Create(flat1.First, flat1_2.First)) );
    var dense1 := tf.keras.layers.Dense(512, 'relu').Apply(concat);
    var dense2 := tf.keras.layers.Dense(128, 'relu').Apply(dense1);
    var dense3 := tf.keras.layers.Dense(10,  'relu').Apply(dense2);
    var output := tf.keras.layers.Softmax(-1).Apply(dense3);

    var model := tf.keras.Model( TFTensors.Create(Tuple.Create(inputs.First, inputs_2.First)), output);

    model.OnEpochBegin      := On_Epoch_Begin;
    model.OnTrainBatchBegin := On_Train_Batch_Begin;
    model.OnEndSummary      := On_End_Summary;

    model.summary;

    var data_loader := MnistModelLoader.Create;

    var ms := ModelLoadSetting.Create;
    ms.TrainDir       := 'mnist';
    ms.OneHot         := false;
    ms.ValidationSize := 59900;

    var dataset := data_loader.LoadAsync(ms) ;

    var loss      := tf.keras.losses.SparseCategoricalCrossentropy;
    var optimizer := TAdam.Create(0.001);
    model.compile(optimizer, loss, ['accuracy']);

    var x1 : TNDArray := np.reshape(dataset.Train.Data, TFShape.Create([dataset.Train.Data.shape[0], 28, 28, 1]));
    var x2 : TNDArray := x1;

    var x : TArray<TNDArray> := [ x1, x2 ];
    model.fit(x, dataset.Train.Labels, 8, 3);

    x1 := np.ones(TFShape.Create([1, 28, 28, 1]), TF_DataType.TF_FLOAT);
    x2 := np.zeros(TFShape.Create([1, 28, 28, 1]), TF_DataType.TF_FLOAT);
    var pred := model.predict(TFTensors.Create([x1, x2]));
    //Console.WriteLine(pred);
end;
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
    InitConfig;
end;

destructor DigitRecognitionCnnKeras.Destroy;
begin
    logMsg.Free;
    FConfig.Free;
end;

function DigitRecognitionCnnKeras.InitConfig: ExampleConfig;
var
  config : ExampleConfig;
begin
    config := ExampleConfig.Create;
    config.Name := 'MNIST CNN (Keras Subclass)';
    config.Enabled := true;
    config.IsImportingGraph := false;

    Result  := config;
    FConfig := Result;
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
  optimizer : IOptimizer;
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
        run_optimization(conv_net, optimizer, batch_x.First, batch_y.First);

        if step mod display_step = 0 then
        begin
            var pred := conv_net.Apply(batch_x);
            var loss : TTensor := cross_entropy_loss(pred.first, batch_y.First);
            var acc  : TTensor := accuracy(pred.first, batch_y.First);
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

procedure DigitRecognitionCnnKeras.run_optimization(conv_net: ConvNet; optimizer: IOptimizer; x: TFTensor; y: TFTensor);
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
    var dp : DatasetPass := TKerasApi.keras.datasets.mnist.load_data;
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

{ MnistFnnKerasFunctional }

constructor MnistFnnKerasFunctional.Create;
begin
    layers  := LayersApi.Create;
    logMsg  := TStringList.Create;

    InitConfig;
end;

destructor MnistFnnKerasFunctional.Destroy;
begin
  layers.Free;
  Log.Free;
  FConfig.Free;
  inherited;
end;

function MnistFnnKerasFunctional.InitConfig: ExampleConfig;
var
  config : ExampleConfig;
begin
    config := ExampleConfig.Create;
    config.Name := 'MNIST FNN (Keras Functional)';
    config.Enabled := true;
    config.IsImportingGraph := false;

    Result  := config;
    FConfig := Result;
end;

function MnistFnnKerasFunctional.Run: Boolean;
begin
    tf.enable_eager_execution();

    PrepareData;
    BuildModel;
    Train;
    Result := true;
end;

procedure MnistFnnKerasFunctional.PrepareData;
begin
    var dp : DatasetPass := TKerasApi.keras.datasets.mnist.load_data;
    x_train := dp.Train.Value1;
    y_train := dp.Train.Value2;

    x_test  := dp.Test.Value1;
    y_test  := dp.Test.Value2;

    // Normalize images value from [0, 255] to [0, 1].
    x_train := NDArray(x_train.reshape(TFShape.Create([60000, 784]))) / Single(255.0);
    x_test  := NDArray(x_test.reshape(TFShape.Create([10000, 784])))  / Single(255.0);
end;

procedure MnistFnnKerasFunctional.BuildModel;
begin
    // input layer
    var inputs := tf.keras.Input(TFShape.Create([784]));
    // 1st dense layer
    var outputs := layers.Dense(64, tf.keras.activations.Relu).Apply(TFTensors.Create(inputs));
    // 2nd dense layer
    outputs := layers.Dense(64, tf.keras.activations.Relu).Apply(TFTensors.Create(outputs));
    // output layer
    outputs := layers.Dense(10).Apply(TFTensors.Create(outputs));
    // build keras model
    mModel := tf.keras.Model(TFTensors.Create(inputs), outputs, 'mnist_model');

    mModel.OnEpochBegin      := On_Epoch_Begin;
    mModel.OnTrainBatchBegin := On_Train_Batch_Begin;
    mModel.OnTestBatchEnd    := On_Epoch_Begin;
    mModel.OnEndSummary      := On_End_Summary;

    // show model summary
    mModel.summary;
    // compile keras model into tensorflow's static graph
    mModel.compile(tf.keras.optimizers.RMSprop, tf.keras.losses.SparseCategoricalCrossentropy('','', true),['accuracy']);
end;

procedure MnistFnnKerasFunctional.Train;
begin
    // train model by feeding data and labels.
    mModel.fit(x_train, y_train, {batch_size:} 64, {epochs:} 2, {verbose:}1, {validation_split:} 0.2);
    // evluate the model
    mModel.evaluate(x_test, y_test, {batch_size:}-1, {verbose:} 2);
    // save and serialize model
    // mModel.save('mnist_model');
    // recreate the exact same model purely from the file:
    // model = keras.models.load_model("path_to_my_model");
end;

procedure MnistFnnKerasFunctional.SetConf(const value: ExampleConfig);
begin
     FConfig := value;
end;

function MnistFnnKerasFunctional.GetConf: ExampleConfig;
begin
   Result := FConfig;
end;

function MnistFnnKerasFunctional.BuildGraph: TFGraph;
begin
   result := nil;
end;

function MnistFnnKerasFunctional.FreezeModel: string;
begin

end;

function MnistFnnKerasFunctional.ImportGraph: TFGraph;
begin
   result := nil
end;

procedure MnistFnnKerasFunctional.Predict;
begin

end;

procedure MnistFnnKerasFunctional.Test;
begin

end;

{ TMnistGAN }

constructor TMnistGAN.Create;
begin
   if not tf.executing_eagerly then
       tf.enable_eager_execution;

    LeakyReLU_alpha := 0.2;
    imgpath    := 'dcgan\imgs';
    modelpath  := 'dcgan\\models';
    noise_dim  := 100;
    img_rows   := 28;
    img_cols   := 28;
    channels   := 1;

    EPOCHS     := 20; // 50;
    BATCH_SIZE := 128;
    BUFFER_SIZE:= 60000;

    layers  := LayersApi.Create;
    logMsg  := TStringList.Create;

    fTrainCount := 0;

    InitConfig;
end;

destructor TMnistGAN.Destroy;
begin
  layers.Free;
  logMsg.Free;

  if Assigned(data) then
    data.Free;

  inherited;
end;

function TMnistGAN.InitConfig: ExampleConfig;
var
  config : ExampleConfig;
begin
    config := ExampleConfig.Create;
    config.Name := 'GAN MNIST';
    config.Enabled := true;
    config.IsImportingGraph := false;

    Result  := config;
    FConfig := Result;
end;

function TMnistGAN.GetConf: ExampleConfig;
begin
    Result := FConfig;
end;

procedure TMnistGAN.SetConf(const value: ExampleConfig);
begin
    FConfig := value;
end;

function TMnistGAN.Run: Boolean;
begin
    tf.enable_eager_execution;

    PrepareData;
    Train;
    
    Result := true;
end;

procedure TMnistGAN.PrepareData;
begin
    data  := TKerasApi.keras.datasets.mnist.load_data;

    train_images := data.Train.Value1.reshape(TFshape.Create([data.Train.Value1.Shape[0],28,28,1])).astype(np.np_float32);
    train_images := (NDArray(train_images) - 127.5) / 127.5 ;

    // # Batch and shuffle the data
    train_dataset := tf.Data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE);

    img_shape := [img_rows, img_cols, channels];

    if (img_cols mod 4 <> 0) or (img_rows mod 4 <> 0) then
        raise Exception.Create('The width and height of the image must be a multiple of 4');

    TDirectory.CreateDirectory(imgpath);
    TDirectory.CreateDirectory(modelpath);
end;

function TMnistGAN.Make_Generator_model: Model;
begin
    var mModel := TKerasApi.keras.Sequential(nil,'GENERATOR');

    mModel.OnEpochBegin      := On_Epoch_Begin;
    mModel.OnTrainBatchBegin := On_Train_Batch_Begin;
    mModel.OnEndSummary      := On_End_Summary;
    mModel.OnTestBatchBegin  := On_Train_Batch_Begin;

    mModel.Add( layers.Input(TFShape.Create([noise_dim])).first );
    mModel.Add( layers.Dense(7*7*256, {activation}nil ,{kernel_initializer}nil, {use_bias}False) );
    mModel.Add( layers.BatchNormalization);
    mModel.Add( layers.LeakyReLU);

    mModel.Add( layers.Reshape(TFShape.Create([7, 7, 256]))) ;
    Assert(mModel.OutputShape = TFShape.Create([-1, 7, 7, 256]));

    mModel.Add( layers.Conv2DTranspose(128, TFShape.Create([5, 5]), TFShape.Create([1, 1]), 'same', {data_format}'', {dilation_rate}nil, {activation}'relu', False));
    Assert(mModel.OutputShape = TFShape.Create([-1, 7, 7, 128]));
    mModel.Add( layers.BatchNormalization);
    mModel.Add( layers.LeakyReLU);

    mModel.Add( layers.Conv2DTranspose(64, TFShape.Create([5, 5]), TFShape.Create([2, 2]), 'same', {data_format}'', {dilation_rate}nil, {activation}'relu', False));
    Assert(mModel.OutputShape = TFShape.Create([-1, 14, 14, 64]));
    mModel.Add( layers.BatchNormalization);
    mModel.Add( layers.LeakyReLU);

    mModel.Add( layers.Conv2DTranspose(1, TFShape.Create([5, 5]), TFShape.Create([2, 2]), 'same', {data_format}'', {dilation_rate}nil, {activation}'tanh', False));
    Assert(mModel.OutputShape = TFShape.Create([-1, 28, 28, 1]));

    mModel.summary;
    Result := mModel;
end;

function TMnistGAN.Make_Discriminator_model: Model;
begin
    var model := TKerasApi.keras.Sequential(nil,'DISCRIMINATOR');

    model.OnEpochBegin      := On_Epoch_Begin;
    model.OnTrainBatchBegin := On_Train_Batch_Begin;
    model.OnEndSummary      := On_End_Summary;
    model.OnTestBatchBegin  := On_Train_Batch_Begin;

    model.Add( layers.Input(img_shape).first );
    model.add(layers.Conv2D(64, TFShape.Create([5, 5]), TFShape.Create([2, 2]), 'same'));
    model.add(layers.LeakyReLU);
    model.add(layers.Dropout(0.3));

    model.add(layers.Conv2D(128, TFShape.Create([5, 5]), TFShape.Create([2, 2]), 'same'));
    model.add(layers.LeakyReLU);
    model.add(layers.Dropout(0.3)) ;

    model.add(layers.Flatten) ;
    model.add(layers.Dense(1));

    model.summary;
    Result := model;
end;

procedure TMnistGAN.Train_step(images: TFTensor);
var
   t,t1 : TArray<Single>;
   s,s1 : TFShape;
begin
     try
       if fTrainCount = 38 then
          fTrainCount := fTrainCount;

       var sSize : TFShape := [ BATCH_SIZE, noise_dim  ];
       var noise    := np.random.normal(@sSize);
       noise        := noise.astype(np.np_float32);

       var gen_tape  := tf.GradientTape;
       var disc_tape := tf.GradientTape;

       var generated_images  := generator.Apply(TFTensors.Create(noise),nil, True);

       s := generated_images.First.Shape;
       if generated_images.First.Dtype = TF_float then
         t := generated_images.First.ToArray<Single>;

       var real_output := discriminator.Apply(TFTensors.Create(images),nil, true);
       var fake_output := discriminator.Apply(TFTensors.Create(generated_images), nil, true);

       s1 := fake_output.First.Shape;
       if fake_output.First.Dtype = TF_float then
         t1 := fake_output.First.ToArray<Single>;

       var gen_loss  := generator_loss(fake_output.First);
       var disc_loss := discriminator_loss(real_output.First, fake_output.First);

       var gradients_of_generator     := gen_tape.gradient(gen_loss, generator.TrainableVariables.ToArray);
       var gradients_of_discriminator := disc_tape.gradient(disc_loss, discriminator.TrainableVariables.ToArray);

       generator_optimizer.apply_gradients(gradients_of_generator, generator.TrainableVariables.ToArray);
       discriminator_optimizer.apply_gradients(gradients_of_discriminator, discriminator.TrainableVariables.ToArray);

       InC(fTrainCount);
     except
       ShowMessage('Raise on Error in TMnistGAN.Train_step in Train_step on epoch n#: '+fTrainCount.ToString);
     end;
end;

procedure TMnistGAN.Train;
begin
    discriminator := Make_Discriminator_model;
    generator     := Make_Generator_model;

    var d_lr : single := 1e-4;
    var g_lr : Single := 1e-4;
    discriminator_optimizer := tf.keras.optimizers.Adam(d_lr);
    generator_optimizer     := tf.keras.optimizers.Adam(g_lr);
    var showstep : Integer := 10;

    var i : Integer := 0;
    try
      while i < EPOCHS do
      begin
          while (FileExists('dcgan\models\Model_' + IntToStr(i + 100) + '_g.weights')) and (FileExists('dcgan\models\Model_' + IntToStr(i + 100) + '_d.weights')) do
              i := i + 100;

          if (FileExists('dcgan\models\Model_' + i.ToString + '_g.weights')) and (FileExists('dcgan\models\Model_' + i.ToString + '_d.weights')) then
          begin
              logMsg.Add('Loading weights for epoch ' + i.ToString);
              generator.load_weights('dcgan\models\Model_' + i.ToString + '_g.weights');
              discriminator.save_weights('dcgan\models\Model_' + i.ToString + '_d.weights');
              PredictImage(generator, i);
          end else
          begin
              for var image_batch in train_dataset do
              begin
                   train_step(image_batch.Value1.First) ;
                   if fTrainCount = 90 then  exit;
              end;

              if i mod 100 = 0 then
              begin
                  generator.save_weights('dcgan\models\Model_' + i.ToString + '_g.weights');
                  discriminator.save_weights('dcgan\models\Model_' + i.ToString + '_d.weights');
              end;
          end;
          inc(i);
      end;
    except
      ShowMessage('Errors at Epoch: '+ i.ToString)
    end;
end;

procedure TMnistGAN.PredictImage(g: Model; step: Integer);
begin
    var r := 5;
    var c := 5;
    var sSize : TFShape := [ r * c, 100 ];
    var noise := np.random.normal(@sSize);
    noise := noise.astype(np.np_float32);

    var tensor_result : TFTensor := g.predict(TFTensors.Create(noise)).first;
    var gen_imgs := tensor_result.numpy;
    SaveImage(gen_imgs, step);
end;

function TMnistGAN.cross_entropy(x: TTensor; y: TTensor) : TFTensor;
begin
   var bce := tf.keras.losses.BinaryCrossentropy(true);
   Result := bce.Call(x, y);
end;

function TMnistGAN.discriminator_loss(real_output: TFTensor; fake_output: TFTensor): TFTensor;
begin
    var real_loss := cross_entropy(tf.ones_like(real_output), real_output);
    var fake_loss := cross_entropy(tf.zeros_like(fake_output), fake_output);
    var total_loss := TTensor(real_loss) + fake_loss;
    Result := total_loss;
end;

function TMnistGAN.generator_loss(fake_output: TFTensor): TFTensor;
begin
    Result := cross_entropy(tf.ones_like(fake_output), fake_output);
end;


procedure TMnistGAN.Test;
begin
    var G := Make_Generator_model();
    G.load_weights(modelpath + '\Model_100_g.weights');
    PredictImage(G, 1);
end;

function FromArgb(alpha: Integer; red: Integer; green: Integer; blue: Integer): TColor;
begin
    Result := (byte(alpha) shl  24) or
              (byte(red)   shl  16) or
              (byte(green) shl  8 ) or
              (byte(blue)  shl  0 );
end;

procedure TMnistGAN.SaveImage(gen_imgs: TNDArray; step: Integer);
var
  image : TBitmap;
begin
    var  size : Integer := 4;
    gen_imgs := NDArray(gen_imgs) * 0.5 + 0.5; // 25x28x28x1 [0.0-1.0]
    var generatedImages := gen_imgs.To_Array; // Tensor.Numpy[25]...
    image := TBitmap.Create(28 * 5 * size, 28 * 5 * size);
    for var i : Integer := 0 to Length(generatedImages) -1 do
    begin
        var values := generatedImages[i].reshape(TFShape.Create([784])).ToArray<Single>;
        var min : Single := MinValue(values);
        var max : Single := MaxValue(values);
        var slope : Single := 0.0;
        if max > min then slope := 255.0 / (max - min);
        var canv := TCanvas.Create;
        canv.Brush.Bitmap := image;
        for var y := 0 to 28 - 1 do
        begin
            for var x : Integer := 0 to 28-1 do
            begin
                var value : Integer := Trunc(((values[y * 28 + x] - min) * slope));
                canv.Brush.Style := bsSolid;
                canv.Brush.Color := FromArgb(255, value, value, value);
                var origin := TPoint.Create(x * size + ((i mod 5) * 28 * size), y * size + ((i div 5) * 28 * size));
                canv.FillRect( TRect.Create(origin, size, size));
            end;

        end;
    end;
    image.SaveToFile(imgpath + '/image' + (step / 10).ToString() + '.jpg');
end;

function TMnistGAN.BuildGraph: TFGraph;
begin
     Result := nil;
end;

procedure TMnistGAN.BuildModel;
begin

end;

function TMnistGAN.FreezeModel: string;
begin

end;

function TMnistGAN.ImportGraph: TFGraph;
begin
    Result := nil;
end;

procedure TMnistGAN.Predict;
begin

end;

end.
