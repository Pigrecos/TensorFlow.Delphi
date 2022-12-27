program TensorFlowDelphi;

{$WARN DUPLICATE_CTOR_DTOR OFF}

uses
  Vcl.Forms,
  untMain in 'untMain.pas' {frmMain},
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
  TensorFlow.Training in 'src\TensorFlow.Training.pas',
  TensorFlow.ControlFlowState in 'src\Operation\TensorFlow.ControlFlowState.pas',
  TensorFlow.control_flow_util in 'src\Operation\TensorFlow.control_flow_util.pas',
  TensorFlow.math_grad in 'src\Gradient\TensorFlow.math_grad.pas',
  Keras.Optimizer in 'src\Keras\Keras.Optimizer.pas',
  Keras.Utils in 'src\Keras\Keras.Utils.pas',
  Keras.KerasApi in 'src\Keras\Keras.KerasApi.pas',
  TensorFlow.resource_variable_grad in 'src\Gradient\TensorFlow.resource_variable_grad.pas',
  TensorFlow.linalg_ops in 'src\Operation\TensorFlow.linalg_ops.pas',
  TensorFlow.array_grad in 'src\Gradient\TensorFlow.array_grad.pas',
  Keras.Regularizers in 'src\Keras\Keras.Regularizers.pas',
  Keras.ILayersApi in 'src\Keras\Keras.ILayersApi.pas',
  Keras.LossFunc in 'src\Keras\Keras.LossFunc.pas',
  Keras.Engine in 'src\Keras\Keras.Engine.pas',
  Keras.Backend in 'src\Keras\Keras.Backend.pas',
  TensorFlow.image_ops_impl in 'src\Operation\TensorFlow.image_ops_impl.pas',
  TensorFlow.gen_image_ops in 'src\Operation\TensorFlow.gen_image_ops.pas',
  TensorFlow.nn_impl in 'src\Operation\TensorFlow.nn_impl.pas',
  TensorFlow.embedding_ops in 'src\Operation\TensorFlow.embedding_ops.pas',
  Keras.Data in 'src\Keras\Keras.Data.pas',
  Keras.Models in 'src\Keras\Keras.Models.pas',
  TensorFlow.dataset_ops in 'src\Operation\TensorFlow.dataset_ops.pas',
  Keras.Preprocessing in 'src\Keras\Keras.Preprocessing.pas',
  Keras.LayersApi in 'src\Keras\Keras.LayersApi.pas',
  Keras.MetricsApi in 'src\Keras\Keras.MetricsApi.pas',
  TensorFlow.tensor_array_ops in 'src\Operation\TensorFlow.tensor_array_ops.pas',
  Keras.Saving in 'src\Keras\Keras.Saving.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmMain, frmMain);
  Application.Run;
end.






