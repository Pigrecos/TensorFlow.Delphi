program TensorFlowDelphi;

{$WARN DUPLICATE_CTOR_DTOR OFF}

uses
  Vcl.Forms,
  untMain in 'untMain.pas' {Form1},
  Tensorflow in 'src\Tensorflow.pas',
  Tensorflow.Utils in 'src\Tensorflow.Utils.pas',
  ProtoGen.config in 'src\Proto\ProtoGen.config.pas',
  NumPy.NDArray in 'src\NumpPy\NumPy.NDArray.pas',
  Numpy.Axis in 'src\NumpPy\Numpy.Axis.pas',
  ProtoGen.variable in 'src\Proto\ProtoGen.variable.pas',
  TensorFlow.Ops in 'src\Operation\TensorFlow.Ops.pas',
  Tensorflow.Graph in 'src\Tensorflow.Graph.pas',
  TensorFlow.Variable in 'src\TensorFlow.Variable.pas',
  TensorFlow.Framework in 'src\TensorFlow.Framework.pas',
  Complex in 'src\lib\Complex.pas',
  TensorFlow.DApi in 'src\Core\TensorFlow.DApi.pas',
  TensorFlow.DApiBase in 'src\Core\TensorFlow.DApiBase.pas',
  TensorFlow.DApiEager in 'src\Core\TensorFlow.DApiEager.pas',
  TensorFlow.EagareRunner in 'src\TensorFlow.EagareRunner.pas',
  TensorFlow.Context in 'src\TensorFlow.Context.pas',
  TensorFlow.Tensor in 'src\TensorFlow.Tensor.pas',
  TF4D.Core.CApi in 'src\Core\TF4D.Core.CApi.pas',
  Numpy in 'src\NumpPy\Numpy.pas',
  Tensorflow.NameScope in 'src\Tensorflow.NameScope.pas',
  TensorFlow.OpDefLibrary in 'src\Operation\TensorFlow.OpDefLibrary.pas',
  TensorFlow.EagerTensor in 'src\TensorFlow.EagerTensor.pas',
  TensorFlow.Constant_op in 'src\TensorFlow.Constant_op.pas',
  TensorFlow.gen_math_ops in 'src\Operation\TensorFlow.gen_math_ops.pas',
  Tensorflow.gen_array_ops in 'src\Operation\Tensorflow.gen_array_ops.pas',
  Tensorflow.math_ops in 'src\Operation\Tensorflow.math_ops.pas',
  Tensorflow.array_ops in 'src\Operation\Tensorflow.array_ops.pas',
  Tensorflow.Gradient in 'src\Gradient\Tensorflow.Gradient.pas',
  TensorFlow.Slice in 'src\TensorFlow.Slice.pas',
  TensorFlow.String_ops in 'src\Operation\TensorFlow.String_ops.pas',
  TensorFlow.gen_state_ops in 'src\Operation\TensorFlow.gen_state_ops.pas',
  TensorFlow.gen_resource_variable_ops in 'src\Operation\TensorFlow.gen_resource_variable_ops.pas',
  TensorFlow.gen_control_flow_ops in 'src\Operation\TensorFlow.gen_control_flow_ops.pas',
  TensorFlow.control_flow_ops in 'src\Operation\TensorFlow.control_flow_ops.pas',
  TensorFlow.gen_sparse_ops in 'src\Operation\TensorFlow.gen_sparse_ops.pas',
  TensorFlow.Tensors.Ragged in 'src\TensorFlow.Tensors.Ragged.pas',
  TensorFlow.resource_variable_ops in 'src\Operation\TensorFlow.resource_variable_ops.pas',
  ProtoGen.cppShapeInference in 'src\Proto\ProtoGen.cppShapeInference.pas',
  Esempi in 'Esempi.pas',
  TensorFlow.gen_random_ops in 'src\Operation\TensorFlow.gen_random_ops.pas',
  TensorFlow.random_ops in 'src\Operation\TensorFlow.random_ops.pas',
  TensorFlow.clip_ops in 'src\Operation\TensorFlow.clip_ops.pas',
  Keras.Layer in 'src\Keras\Keras.Layer.pas',
  TensorFlow.gen_data_flow_ops in 'src\Operation\TensorFlow.gen_data_flow_ops.pas',
  TensorFlow.nn_ops in 'src\Operation\TensorFlow.nn_ops.pas',
  TensorFlow.Initializer in 'src\Operation\TensorFlow.Initializer.pas',
  Keras.ArgsDefinition in 'src\Keras\Keras.ArgsDefinition.pas',
  Keras.Activations in 'src\Keras\Keras.Activations.pas',
  TensorFlow.NnOps in 'src\Operation\NnOps\TensorFlow.NnOps.pas',
  TensorFlow.gen_nn_ops in 'src\Operation\TensorFlow.gen_nn_ops.pas',
  TensorFlow.Activation in 'src\Operation\TensorFlow.Activation.pas',
  TensorFlow.gen_ops in 'src\Operation\TensorFlow.gen_ops.pas',
  TensorFlow.Interfaces in 'src\TensorFlow.Interfaces.pas',
  TensorFlow.bitwise_ops in 'src\Operation\TensorFlow.bitwise_ops.pas',
  TensorFlow.Training in 'src\TensorFlow.Training.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.






