program TensorFlowDelphi;

{$WARN DUPLICATE_CTOR_DTOR OFF}

uses
  FastMM5,
  {$IFDEF EurekaLog}
  EMemLeaks,
  EResLeaks,
  EFastMM5Support,
  EDebugExports,
  EDebugJCL,
  EFixSafeCallException,
  EMapWin32,
  EAppVCL,
  EDialogWinAPIMSClassic,
  EDialogWinAPIEurekaLogDetailed,
  EDialogWinAPIStepsToReproduce,
  ExceptionLog7,
  {$ENDIF EurekaLog}
  Vcl.Forms,
  untMain in 'untMain.pas' {frmMain},
  Tensorflow in 'src\Tensorflow.pas',
  Tensorflow.Utils in 'src\Tensorflow.Utils.pas',
  NumPy.NDArray in 'src\NumpPy\NumPy.NDArray.pas',
  Numpy.Axis in 'src\NumpPy\Numpy.Axis.pas',
  TensorFlow.Ops in 'src\Operation\TensorFlow.Ops.pas',
  TensorFlow.Variable in 'src\TensorFlow.Variable.pas',
  Complex in 'src\lib\Complex.pas',
  TensorFlow.DApi in 'src\Core\TensorFlow.DApi.pas',
  TensorFlow.DApiBase in 'src\Core\TensorFlow.DApiBase.pas',
  TF4D.Core.CApiEager in 'src\Core\TF4D.Core.CApiEager.pas',
  TensorFlow.Tensor in 'src\TensorFlow.Tensor.pas',
  TF4D.Core.CApi in 'src\Core\TF4D.Core.CApi.pas',
  Numpy in 'src\NumpPy\Numpy.pas',
  TensorFlow.OpDefLibrary in 'src\Operation\TensorFlow.OpDefLibrary.pas',
  Tensorflow.Gradient in 'src\Gradient\Tensorflow.Gradient.pas',
  TensorFlow.Slice in 'src\TensorFlow.Slice.pas',
  Esempi in 'Esempi.pas',
  Keras.Layer in 'src\Keras\Keras.Layer.pas',
  TensorFlow.Initializer in 'src\Operation\TensorFlow.Initializer.pas',
  TensorFlow.NnOps in 'src\Operation\NnOps\TensorFlow.NnOps.pas',
  TensorFlow.Activation in 'src\Operation\TensorFlow.Activation.pas',
  TensorFlow.Interfaces in 'src\TensorFlow.Interfaces.pas',
  TensorFlow.Training in 'src\TensorFlow.Training.pas',
  TensorFlow.math_grad in 'src\Gradient\TensorFlow.math_grad.pas',
  Keras.Optimizer in 'src\Keras\Keras.Optimizer.pas',
  Keras.Utils in 'src\Keras\Keras.Utils.pas',
  Keras.KerasApi in 'src\Keras\Keras.KerasApi.pas',
  TensorFlow.resource_variable_grad in 'src\Gradient\TensorFlow.resource_variable_grad.pas',
  TensorFlow.array_grad in 'src\Gradient\TensorFlow.array_grad.pas',
  Keras.Regularizers in 'src\Keras\Keras.Regularizers.pas',
  Keras.LossFunc in 'src\Keras\Keras.LossFunc.pas',
  Keras.Backend in 'src\Keras\Keras.Backend.pas',
  Keras.Data in 'src\Keras\Keras.Data.pas',
  Keras.Models in 'src\Keras\Keras.Models.pas',
  Keras.Preprocessing in 'src\Keras\Keras.Preprocessing.pas',
  Keras.LayersApi in 'src\Keras\Keras.LayersApi.pas',
  Keras.MetricsApi in 'src\Keras\Keras.MetricsApi.pas',
  Keras.Container in 'src\Keras\Keras.Container.pas',
  TensorFlow.nn_grad in 'src\Gradient\TensorFlow.nn_grad.pas',
  ProtoGen.Main in 'src\Proto\ProtoGen.Main.pas',
  untModels in 'untModels.pas',
  hdf5dll in 'src\lib\hdf5dll.pas',
  Hdf5 in 'src\lib\Hdf5.pas',
  Keras.Saving in 'src\Keras\Keras.Saving.pas',
  Keras.Callbacks in 'src\Keras\Keras.Callbacks.pas',
  Keras.Core in 'src\Keras\Keras.Core.pas',
  TensorFlow.Core in 'src\TensorFlow.Core.pas',
  TensorFlow.Proto in 'src\Proto\TensorFlow.Proto.pas',
  TensorFlow.Operations in 'src\Operation\TensorFlow.Operations.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmMain, frmMain);
  Application.Run;
end.










