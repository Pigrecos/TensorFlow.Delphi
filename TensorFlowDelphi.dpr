program TensorFlowDelphi;

{$WARN DUPLICATE_CTOR_DTOR OFF}

uses
  Vcl.Forms,
  untMain in 'untMain.pas' {Form1},
  TensorFlow.LowLevelAPI in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow.LowLevelAPI.pas',
  TensorFlow._Helpers in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow._Helpers.pas',
  TensorFlow.DApi in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow.DApi.pas',
  TensorFlow.DApiBase in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow.DApiBase.pas',
  TensorFlow.DApiOperations in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow.DApiOperations.pas',
  Tensorflow in 'src\Tensorflow.pas',
  TensorFlow.Eager in '..\Machine Learning\TensorFlow4Delphi\api\TensorFlow.Eager.pas',
  Tensorflow.Utils in 'src\Tensorflow.Utils.pas',
  ProtoGen.config in 'src\Proto\ProtoGen.config.pas',
  NDArray in 'src\NDArray.pas',
  Numpy.Axis in 'src\Numpy.Axis.pas',
  ProtoGen.variable in 'src\Proto\ProtoGen.variable.pas',
  TensorFlow.Ops in 'src\TensorFlow.Ops.pas',
  Tensorflow.Graph in 'src\Tensorflow.Graph.pas',
  TensorFlow.Variable in 'src\TensorFlow.Variable.pas',
  TensorFlow.Framework in 'src\TensorFlow.Framework.pas',
  Complex in 'src\lib\Complex.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
