unit untMain;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti,

  Spring,

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
  NumPy.NDArray,
  Numpy.Axis;

type
  GraphModeTestBase = class
    private
    protected
      Fgraph : TFGraph;
    public
      constructor Create;virtual;
      destructor  Destroy; reintroduce ;virtual;
  end;

  TUnitTest_Basic = class(GraphModeTestBase)
    private

    public
      constructor Create; override;
      destructor  Destroy; override;

      procedure Session_EvalTensor;
      procedure Session_Eval_SmallString_Scalar;
      procedure Session_Eval_LargeString_Scalar;
      procedure Session_Autocast_Case0;
      procedure Session_Autocast_Case1;
      procedure Session_Autocast_Case2;
      procedure Session_Autocast_Case3;
      //
      procedure Tensor_sparse_tensor_to_dense;
      procedure Tensor_sparse_to_dense;
      procedure Tensor_batch_to_space_nd;
      procedure Tensor_boolean_mask;
      //
      procedure Variabele_InitVariable;
      procedure Variabele_NewVariable;
      procedure Variabele_StringVar;
      procedure Variabele_VarSum;
      procedure Variabele_Assign1;
      procedure Variabele_Assign2;
      procedure Variabele_Assign3;
      /// <summary>
      /// Assign tensor to slice of other tensor.
      /// https://www.tensorflow.org/api_docs/python/tf/Variable#__getitem__
      /// </summary>
      procedure Variabele_SliceAssign;
      procedure Variabele_Accumulation;
      procedure Variabele_ShouldReturnNegative;
      procedure IdentityOriginalTensor;
      /// <summary>
      /// Test the function of setting random seed
      /// This will help regenerate the same result
      /// </summary>
      procedure TFRandomSeedTest;
      /// <summary>
      /// compare to Test above, seed is also added in params
      /// </summary>
      procedure TFRandomSeedTest2;
      /// <summary>
      /// This part we use funcs in tf.random rather than only tf
      /// </summary>
      procedure TFRandomRaodomSeedTest;
      /// <summary>
      /// compare to Test above, seed is also added in params
      /// </summary>
      procedure TFRandomRaodomSeedTest2;
  end;

  ManagedAPI = class
    private

    public
       constructor Create;
       destructor  Destroy;override;

       procedure Slice;
       procedure Gather;
       // Gradient
       procedure GradientFloatTest;
       procedure GradientDefaultTest;
       procedure GradientConcatTest;
       procedure GradientOperatorMulTest;
       procedure GradientSliceTest;
       // Tensor Operate
       procedure TransposeTest;
       procedure ConcatDoubleTest;
       procedure ConcatTest;
       procedure InitTensorTest;
       procedure TestZerosLike;
       // StringsApiTest
       procedure StringFromBytes;
       procedure StringEqual;
       procedure StringArray;
       procedure StringSplit;
  end;

  TfrmMain = class(TForm)
    btnTest: TBitBtn;
    mmo1: TMemo;
    btnLinReg: TBitBtn;
    btnLinReg1: TBitBtn;
    btnKerasLayers: TBitBtn;
    procedure btnTestClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure btnLinRegClick(Sender: TObject);
    procedure btnLinReg1Click(Sender: TObject);
    procedure btnKerasLayersClick(Sender: TObject);
  private
    procedure EnableEager;

  public
    procedure DisableEager;
  end;

var
  frmMain: TfrmMain;

implementation
         uses Esempi,

              System.Types,
              System.TypInfo,
              TensorFlow.Constant_op,
              Tensorflow.array_ops,
              Tensorflow.math_ops,

              TensorFlow.Slice,

              System.Generics.Collections,
              Spring.Collections.Stacks,

              DUnitX.TestFramework;
{$R *.dfm}

{ GraphModeTestBase }

constructor GraphModeTestBase.Create;
begin
    tf.compat.v1.disable_eager_execution;
    Fgraph := tf.Graph.as_default;
end;

destructor GraphModeTestBase.Destroy;
begin
    Fgraph.gExit
end;

{ SessionTest }

constructor TUnitTest_Basic.Create;
begin
  inherited Create;

end;

destructor TUnitTest_Basic.Destroy;
begin
  Fgraph.free;
  inherited Destroy;
end;

procedure TUnitTest_Basic.Session_Autocast_Case0;
begin
    var sess := tf.Session.as_default;
    var operation : ITensorOrOperation := tf.global_variables_initializer;
    // the cast to ITensorOrOperation is essential for the test of this method signature
    sess.run(operation);
end;

procedure TUnitTest_Basic.Session_Autocast_Case1;
begin
    var sess := tf.Session.as_default;
    var input := tf.placeholder(tf.int32_t, TFShape.Create([6]));
    var shape : TArray<Integer> := [ 2, 3 ];
    var op := tf.reshape(input, shape);
    sess.run(tf.global_variables_initializer);
    var input1 : TArray<Integer> :=[1, 2, 3, 4, 5, 6];
    var ret := sess.run(op, [ FeedItem.Create(input, np.np_array(input1)) ] );

    var s1 := ret.shape;
    if s1 <> TFShape.Create([2, 3]) then
      raise TFException.Create('Autocast_Case1- Shape not Equal');

    var aInput := ret.ToArray<Integer>;
    Assert.AreEqual(aInput,input1);
end;

procedure TUnitTest_Basic.Session_Autocast_Case2;
begin
    var sess := tf.Session.as_default;

    var input : TTensor := tf.placeholder(tf.float32_t, TFShape.Create([6]));
    var op := tf.reshape(input, TFShape.Create([2,3]));
    sess.run(tf.global_variables_initializer);

    // var input1 : TArray<Integer> :=[1, 2, 3, 4, 5, 6];
    // var aValue : NDArray  := np.np_array(input1).astype(np.np_float32) ; Valid
    var aValue : NDArray  := np.np_array<Integer>([1, 2, 3, 4, 5, 6]).astype(np.np_float32) ;
    aValue := aValue + Single(0.1);
    sess.run(op, [ FeedItem.Create( input, aValue ) ] );
end;

procedure TUnitTest_Basic.Session_Autocast_Case3;
begin
    var sess := tf.Session.as_default;

    var input : TTensor := tf.placeholder(tf.float32_t, TFShape.Create([6]));
    var op := tf.reshape(input, TFShape.Create([2,3]));
    sess.run(tf.global_variables_initializer);

    // var input1 : TArray<Integer> :=[1, 2, 3, 4, 5, 6];
    // var aValue : NDArray  := np.np_array(input1).astype(np.np_float32) ; Valid
    var aValue : NDArray  := np.np_array<Integer>([1, 2, 3, 4, 5, 6]).astype(np.np_float32) ;
    aValue := aValue + Single(0.1);
    var ret := sess.run(op, [ FeedItem.Create( input, aValue ) ] );

    if ret.shape <> TFShape.Create([2,3]) then
      raise Exception.Create('');

    var r := ret.ToArray<Single>;

    var t    := ret.ToString;
    Assert.AreEqual('tf.Tensor "<unnamed>:0" shape=2,3 dtype=float32',t);
end;

procedure TUnitTest_Basic.Session_EvalTensor;
begin
    var a : TTensor := constant_op.constant( np.np_array(3.0).reshape( TFShape.Create([1, 1])) );
    var b : TTensor := constant_op.constant(  np.np_array(2.0).reshape( TFShape.Create([1, 1])) );
    var c : TTensor := math_ops.matmul(a, b, 'matmul');

    var sess := tf.Session;

    var res : NDArray := c.eval(sess);
    Assert.AreEqual<Single>(NDArray(res[0]), 6.0,'EvalTensor Error!');
end;

procedure TUnitTest_Basic.Session_Eval_SmallString_Scalar;
begin

    var a := constant_op.constant( '123 heythere 123 ', TF_DataType.TF_STRING,'Const');
    var c := tf.strings.substr(a, 4, 8);
    var sess := tf.Session;
    var res : TArray<TF_TString> := c.eval(sess).StringData;
    //Assert.AreEqual<String>( res[0], 'heythere');

end;

procedure TUnitTest_Basic.Session_Eval_LargeString_Scalar;
begin
    var size  : Integer := 30000;
    var s := string.Create('a',size);
    var a := constant_op.constant(AnsiString(s), TF_DataType.TF_STRING,'Const');
    var c := tf.strings.substr(a, 0, size - 5000);
    var sess := tf.Session;
    var res := c.eval(sess).ToByteArray ;
    var sRes := TUTF8Encoding.UTF8.GetString(res);
end;

procedure TUnitTest_Basic.Tensor_boolean_mask;
begin
   var tensor := [ 0, 1, 2, 3 ];
   var mask   := np.np_array<Boolean>([ true, false, true, false ]);
   var masked := tf.boolean_mask(tensor, mask);

   var sess := tf.Session;
   var res := sess.run(masked);
   Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 2 ], res.ToArray<integer>));

end;

procedure TUnitTest_Basic.Tensor_sparse_tensor_to_dense;
begin
    var decoded_list := tf.SparseTensor([[0,0], [1,2]], [1,2], [3,4]);
    var onehot       := tf.sparse_tensor_to_dense(decoded_list,0);

    var s := onehot.shape;
    var sess := tf.Session;

    var res := sess.run(onehot);
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 1, 0, 0, 0 ], res[0].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 0, 2, 0 ], res[1].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 0, 0, 0 ], res[2].ToArray<integer>));
end;

procedure TUnitTest_Basic.Tensor_sparse_to_dense;
begin
    var indices := tf.reshape( tf.range(0, 5), TFShape.Create([ 5, 1 ]) );
    var labels  := tf.expand_dims(tf.constant( TValue.From< TArray<Integer>>([ 0, 1, 2, 3, 4 ]) ), 1);
    var st      := tf.concat([indices, labels ], 1);
    var onehot  := tf.sparse_to_dense(st, TFShape.create([5, 5]), 1);

    var s := onehot.shape;
    var sess := tf.Session;

    var res := sess.run(onehot);
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 1, 0, 0, 0, 0 ], res[0].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 1, 0, 0, 0 ], res[1].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 0, 1, 0, 0 ], res[2].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 0, 0, 1, 0 ], res[3].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0, 0, 0, 0, 1 ], res[4].ToArray<integer>));

end;

procedure TUnitTest_Basic.TFRandomSeedTest;
begin
    var initValue := np.arange(6).reshape(TFShape.create([3, 2]));
    tf.set_random_seed(1234);
    var a1 := tf.random_uniform(1);
    var b1 := tf.random_shuffle(tf.constant(initValue));

    // This part we consider to be a refresh
    tf.set_random_seed(10);
    tf.random_uniform(1);
    tf.random_shuffle(tf.constant(initValue));

    tf.set_random_seed(1234);
    var a2 := tf.random_uniform(1);
    var b2 := tf.random_shuffle(tf.constant(initValue));
    Assert.IsTrue(a1.numpy.Equals(a2.numpy));
    Assert.IsTrue(b1.numpy.Equals(b2.numpy));
end;

procedure TUnitTest_Basic.TFRandomSeedTest2;
begin
    var initValue := np.arange(6).reshape(TFShape.create([3, 2]));
    tf.set_random_seed(1234);
    var pSeed : Integer := 1234;
    var a1  := tf.random_uniform(1, 0, 1, TF_FLOAT,@pSeed);
    var b1  := tf.random_shuffle(tf.constant(initValue), pSeed);

    // This part we consider to be a refresh
    tf.set_random_seed(10);
    tf.random_uniform(1);
    tf.random_shuffle(tf.constant(initValue));

    tf.set_random_seed(1234);
    var a2 := tf.random_uniform(1);
    var b2 := tf.random_shuffle(tf.constant(initValue));
    Assert.IsTrue(a1.numpy.Equals(a2.numpy));
    Assert.IsTrue(b1.numpy.Equals(b2.numpy));
end;

procedure TUnitTest_Basic.TFRandomRaodomSeedTest;
begin
    tf.set_random_seed(1234);
    var a1 := tf.random.normal(1);
    var b1 := tf.random.truncated_normal(1);

    // This part we consider to be a refresh
    tf.set_random_seed(10);
    tf.random.normal(1);
    tf.random.truncated_normal(1);

    tf.set_random_seed(1234);
    var a2 := tf.random.normal(1);
    var b2 := tf.random.truncated_normal(1);

    Assert.IsTrue(a1.numpy.Equals(a2.numpy));
    Assert.IsTrue(b1.numpy.Equals(b2.numpy));
end;

procedure TUnitTest_Basic.TFRandomRaodomSeedTest2;
begin
    tf.set_random_seed(1234);
    var pSeed : Integer := 1234;
    var a1 := tf.random.normal(1, 0.0, 1.0, TF_FLOAT, @pSeed);
    var b1 := tf.random.truncated_normal(1);

    // This part we consider to be a refresh
    tf.set_random_seed(10);
    tf.random.normal(1);
    tf.random.truncated_normal(1);

    tf.set_random_seed(1234);
    var a2 := tf.random.normal(1, 0.0, 1.0, TF_FLOAT, @pSeed);
    var b2 := tf.random.truncated_normal(1,  0.0, 1.0, TF_FLOAT, @pSeed);

    Assert.IsTrue(a1.numpy.Equals(a2.numpy));
    Assert.IsTrue(b1.numpy.Equals(b2.numpy));
end;

procedure TUnitTest_Basic.Tensor_batch_to_space_nd;
begin
    var inputs := np.arange(24).reshape( TFShape.create([4, 2, 3]) );
    var block_shape : TArray<Integer> := [ 2, 2 ];
    var crops : TArray< TArray<Integer> > := [ [ 0, 0 ], [ 0, 0 ] ];
    var tensor := tf.batch_to_space_nd(inputs, block_shape, crops);

    var s := tensor.shape;
    var sess := tf.Session;

    var res := sess.run(tensor);
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 0,   6,  1,  7,  2, 8  ], res[[0,0]].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 12, 18, 13, 19, 14, 20 ], res[[0,1]].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 3,   9,  4, 10,  5, 11 ], res[[0,2]].ToArray<integer>));
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([ 15, 21, 16, 22, 17, 23 ], res[[0,3]].ToArray<integer>));
end;

procedure TUnitTest_Basic.Variabele_InitVariable ;
begin
    var aVar : TArray<Integer> := [ 1, 2] ;
    var v    := tf.Variable(aVar);
    var init : ITensorOrOperation := tf.compat.v1.global_variables_initializer;
    var sess := tf.compat.v1.Session;
    sess.run(init);
    // Usage passing the session explicitly.
    v.eval(sess);
    // Usage with the default session.  The 'with' block
    // above makes 'sess' the default session.
    v.eval;
end;

procedure TUnitTest_Basic.Variabele_NewVariable;
begin
    var x := tf.Variable(10, 'x');
    Assert.AreEqual(0, x.shape.ndim);
    if tf.context.executing_eagerly then
    begin
        var n : NDArray := x.numpy;
        Assert.AreEqual(Integer(n), 10);
    end;
end;

procedure TUnitTest_Basic.Variabele_StringVar;
begin
    {$HINTS OFF}
    var mammal1 := tf.Variable('Elephant', 'var1', tf.string_t);
    var mammal2 := tf.Variable('Tiger');

end;

procedure TUnitTest_Basic.Variabele_VarSum;
begin
    var x : TTensor := tf.constant(3,'x');
    var y := tf.Variable(x + 1, 'y');
    var intVar : NDArray := y.numpy;
    var i : Integer := intVar;
    Assert.AreEqual(i,4,'not equal')
end;
procedure TUnitTest_Basic.Variabele_Assign1;
begin
    var variable := tf.Variable(31, 'tree');
    var unread   := variable.assign(12);
    var intVar : NDArray := unread.numpy;
    var i : Integer := intVar;
    Assert.AreEqual(i,12,'not equal')
end;
procedure TUnitTest_Basic.Variabele_Assign2;
begin
    var v1 := tf.Variable(Single(10.0), 'v1');
    var v2 := v1.assign(TResourceVariable(v1) + Single(1.0) );
    var intVar  : NDArray  := v1.numpy;
    var intVar1 : NDArray  := v2.numpy;
    if intVar <> intVar1  then
      raise Exception.Create('not equal');
    var i : Single := intVar;
    Assert.AreEqual(i,Single(11),'not equal')
end;
procedure TUnitTest_Basic.Variabele_Assign3;
begin
    var v1 := tf.Variable(Single(10.0), 'v1');
    var v2 := tf.Variable(v1, 'v2');
    var intVar  : NDArray  := v1.numpy;
    var intVar1 : NDArray  := v2.numpy;
    if intVar <> intVar1  then
       raise Exception.Create('not equal');
    v1.assign(Single(30.0) );
    intVar := v1.numpy;
    intVar1:= v2.numpy;
    if intVar = intVar1  then
       raise Exception.Create('equal');
end;
procedure TUnitTest_Basic.Variabele_SliceAssign;
begin
    var a1 : TArray< TArray<Single> >;
    a1 := [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]] ;

    var nd := TNDArray.Create(a1);
    var x := tf.Variable(nd);

    // get slice form variable
    var sliced := x[[':2', ':2']];
    var n1   := nd[0][[':2']].toArray<Single> ;
    var n1_1 := sliced[0].numpy.toArray<Single>;
    Assert.IsTrue( TUtils.SequenceEqual<Single>(n1, n1_1));
    var n2   := nd[1][[':2']].toArray<Single> ;
    var n1_2 := sliced[1].numpy.toArray<Single>;
    Assert.IsTrue( TUtils.SequenceEqual<Single>(n2, n1_2));
end;
procedure TUnitTest_Basic.Variabele_Accumulation;
begin
    var x := tf.Variable(10, 'x');
    for var i := 0 to 5 - 1 do
        x.assign( TResourceVariable(x) + 1 );
    var xi : NDArray := x.numpy;
    Assert.AreEqual( Integer(xi), 15);
end;

procedure TUnitTest_Basic.Variabele_ShouldReturnNegative;
begin
    var a1 : TArray< TArray<Integer> >;
    a1 := [ [1, 2] ] ;
    var x := tf.constant( TValue.from<TArray< TArray<Integer> >>(a1) );
    var neg_x := tf.negative(x);
    Assert.IsTrue( TUtils.SequenceEqual<Int64>([1, 2 ], neg_x.shape.dims) ) ;
    Assert.IsTrue( TUtils.SequenceEqual<Integer>([-1, -2], neg_x.numpy.ToArray<Integer>) );
end;

procedure TUnitTest_Basic.IdentityOriginalTensor;
begin
    var a := tf.Variable(5);
    var a_identity := tf.identity( TResourceVariable(a) );
    a.assign_add(1);
    var an : NDArray := a_identity.numpy;
    Assert.AreEqual( Integer(an), 5);
    an := a.numpy;
    Assert.AreEqual(Integer(an), 6);
end;

procedure TfrmMain.FormShow(Sender: TObject);
begin
    frmMain.Caption := 'TensorFlow lib ver. : ('+tf.Version +') TensoFlow.NET Commit(3fde7558e2c0a457272075219107b0dee3c8e4e5)'
end;

type
  stack = class(TStack<Integer>)

  end;

procedure TfrmMain.btnKerasLayersClick(Sender: TObject);
var
  k_Layers  : Keras_Layers_test;
  k_losses  : Keras_Losses_test;
  k_Layer   : LayersTest;
begin
    //tf.keras.utils.get_file('flower_photos.tgz','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz');
    //tf.keras.utils.get_file('aal012022.dat.gz','https://ftp.nhc.noaa.gov/atcf/aid_public/aal012022.dat.gz') ;
    //tf.keras.utils.get_file('v0.70.2-NET6.zip','https://github.com/SciSharp/TensorFlow.NET/archive/refs/tags/v0.70.2-NET6.zip');

    mmo1.Clear;
    mmo1.Lines.Add('Test Keras Activation Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    k_Layers := Keras_Layers_test.Create;

    mmo1.Lines.Add('Keras Activation Layers[LeakyReLU]');
    k_Layers.ActivationTest_LeakyReLU;

    mmo1.Lines.Add('Keras Activation Layers[ELU]');
    k_Layers.ActivationTest_ELU;

    mmo1.Lines.Add('Keras Activation Layers[SELU]');
    k_Layers.ActivationTest_SELU;

    mmo1.Lines.Add('Keras Activation Layers[Softmax]');
    k_Layers.ActivationTest_Softmax;

    mmo1.Lines.Add('Keras Activation Layers[Softplus]');
    k_Layers.ActivationTest_Softplus;

    mmo1.Lines.Add('Keras Activation Layers[Softsign]');
    k_Layers.ActivationTest_Softsign;

    mmo1.Lines.Add('Keras Activation Layers[Exponential]');
    k_Layers.ActivationTest_Exponential;

    mmo1.Lines.Add('Keras Activation Layers[HardSigmoid]');
    k_Layers.ActivationTest_HardSigmoid;

    mmo1.Lines.Add('Keras Activation Layers[Swish]');
    k_Layers.ActivationTest_Swish;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Activation Layers...');
    mmo1.Lines.Add('========================================');

    // Attention Layers
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Attention Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Attention Layers[BaseDenseAttention]');
    k_Layers.Attention_BaseDenseAttention;

    mmo1.Lines.Add('Keras Attention Layers[Attention]');
    k_Layers.Attention_Attention;

    mmo1.Lines.Add('Keras Attention Layers[MultiHeadAttention]');
    k_Layers.Attention_MultiHeadAttention;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Attention Layers...');
    mmo1.Lines.Add('========================================');

    // Convolution Layers
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Convolution Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D]');
    k_Layers.BasicConv1D;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D_ksize]');
    k_Layers.BasicConv1D_ksize;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D_ksize_same]');
    k_Layers.BasicConv1D_ksize_same;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D_ksize_strides]');
    k_Layers.BasicConv1D_ksize_strides;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D_ksize_dilations]');
    k_Layers.BasicConv1D_ksize_dilations;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv1D_ksize_dilation_same]');
    k_Layers.BasicConv1D_ksize_dilation_same;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D]');
    k_Layers.BasicConv2D;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D_ksize]');
    k_Layers.BasicConv2D_ksize;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D_ksize_same]');
    k_Layers.BasicConv2D_ksize_same;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D_ksize_strides]');
    k_Layers.BasicConv2D_ksize_strides;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D_ksize_dilations]');
    k_Layers.BasicConv2D_ksize_dilations;

    mmo1.Lines.Add('Keras Convolution Layers[BasicConv2D_ksize_dilation_same]');
    k_Layers.BasicConv2D_ksize_dilation_same;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Convolution Layers...');
    mmo1.Lines.Add('========================================');

    // Cropping Layers
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Cropping Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Cropping Layers[Cropping1D]');
    k_Layers.Cropping1D;

    mmo1.Lines.Add('Keras Cropping Layers[Cropping2D]');
    k_Layers.Cropping2D;

    mmo1.Lines.Add('Keras Cropping Layers[Cropping3D]');
    k_Layers.Cropping3D;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Cropping Layers...');
    mmo1.Lines.Add('========================================');

    // Merging Layers
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Merging Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Merging Layers[Concatenate]');
    k_Layers.Concatenate;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Merging Layers...');
    mmo1.Lines.Add('========================================');

    // Reshaping Layers
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Reshaping Layers...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Reshaping Layers[ZeroPadding2D]');
    k_Layers.ZeroPadding2D;

    mmo1.Lines.Add('Keras Reshaping Layers[UpSampling2D]');
    k_Layers.UpSampling2D;

    mmo1.Lines.Add('Keras Reshaping Layers[Reshape]');
    k_Layers.Reshape;

    mmo1.Lines.Add('Keras Reshaping Layers[Permute]');
    k_Layers.Permute;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Cropping Layers...');
    mmo1.Lines.Add('========================================');

    // Free Layer test class
    //
    k_Layers.Free;

    //===========================
    //===========================

    k_losses  := Keras_Losses_test.Create;
    // CosineSimilarity
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras CosineSimilarity Losses...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras CosineSimilarity Losses[Default]');
    k_losses.CosineSimilarity_Default;

    mmo1.Lines.Add('Keras CosineSimilarity Losses[Sample_Weight]');
    k_losses.CosineSimilarity_Sample_Weight;

    mmo1.Lines.Add('Keras CosineSimilarity Losses[SUM]');
    k_losses.CosineSimilarity_SUM;

    mmo1.Lines.Add('Keras CosineSimilarity Losses[None]');
    k_losses.CosineSimilarity_None;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras CosineSimilarity Losses...');
    mmo1.Lines.Add('========================================');

    // Huber
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras Huber Losses...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Huber Losses[Default]');
    k_losses.Huber_Default;

    mmo1.Lines.Add('Keras Huber Losses[Sample_Weight]');
    k_losses.Huber_Sample_Weight;

    mmo1.Lines.Add('Keras Huber Losses[SUM]');
    k_losses.Huber_SUM;

    mmo1.Lines.Add('Keras Huber Losses[None]');
    k_losses.Huber_None;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras Huber Losses...');
    mmo1.Lines.Add('========================================');

    // LogCosh
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras LogCosh Losses...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras LogCosh Losses[Default]');
    k_losses.LogCosh_Default;

    mmo1.Lines.Add('Keras LogCosh Losses[Sample_Weight]');
    k_losses.LogCosh_Sample_Weight;

    mmo1.Lines.Add('Keras LogCosh Losses[SUM]');
    k_losses.LogCosh_SUM;

    mmo1.Lines.Add('Keras LogCosh Losses[None]');
    k_losses.LogCosh_None;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras LogCosh Losses...');
    mmo1.Lines.Add('========================================');

    // MeanAbsoluteError
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Keras MeanAbsoluteError Losses...');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras MeanAbsoluteError Losses[Default]');
    k_losses.MeanAbsoluteError_Default;

    mmo1.Lines.Add('Keras MeanAbsoluteError Losses[Sample_Weight]');
    k_losses.MeanAbsoluteError_Sample_Weight;

    mmo1.Lines.Add('Keras MeanAbsoluteError Losses[SUM]');
    k_losses.MeanAbsoluteError_SUM;

    mmo1.Lines.Add('Keras MeanAbsoluteError Losses[None]');
    k_losses.MeanAbsoluteError_None;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Keras MeanAbsoluteError Losses...');
    mmo1.Lines.Add('========================================');

    // Free Layer test class
    //
    k_losses.Free;

    //===========================
    //===========================

    k_Layer := LayersTest.Create;
    // Layer Test
    //
    mmo1.Lines.Add('');
    mmo1.Lines.Add('');
    mmo1.Lines.Add('Test Layers Keras Copmlex....');
    mmo1.Lines.Add('========================================');
    mmo1.Lines.Add('');

    mmo1.Lines.Add('Keras Layer[AveragePooling2D]');
    k_Layer.AveragePooling2D;

    mmo1.Lines.Add('Keras Layer[InputLayer]');
    k_Layer.InputLayer;
    for var i := 0 to tf.MemoLog.Count-1 do
          mmo1.Lines.add( tf.MemoLog[i]);

    mmo1.Lines.Add('Keras Layer[Sequential]');
    k_Layer.Sequential;
    mmo1.Lines.Add('');
    for var i := 0 to tf.MemoLog.Count-1 do
          mmo1.Lines.add( tf.MemoLog[i]);

    mmo1.Lines.Add('Keras Layer[Functional]');
    tf.MemoLog.Clear;
    k_Layer.Functional;
    mmo1.Lines.Add('');
    for var i := 0 to tf.MemoLog.Count-1 do
          mmo1.Lines.add( tf.MemoLog[i]);

    mmo1.Lines.Add('');
    mmo1.Lines.Add('Keras Layer[TensorFlowOpLayer]');
    tf.MemoLog.Clear;
    k_Layer.TensorFlowOpLayer;
    mmo1.Lines.Add('');
    for var i := 0 to tf.MemoLog.Count-1 do
          mmo1.Lines.add( tf.MemoLog[i]);

    mmo1.Lines.Add('Keras Layer[Embedding]');
    k_Layer.Embedding;

    mmo1.Lines.Add('Keras Layer[Dense]');
    k_Layer.Dense;

    mmo1.Lines.Add('Keras Layer[EinsumDense]');
    k_Layer.EinsumDense;

    mmo1.Lines.Add('Keras Layer[SimpleRNN]');
    k_Layer.SimpleRNN;

    mmo1.Lines.Add('Keras Layer[Resizing]');
    k_Layer.Resizing;

    mmo1.Lines.Add('Keras Layer[LayerNormalization]');
    k_Layer.LayerNormalization;

    mmo1.Lines.Add('');
    mmo1.Lines.Add('End Test Layers Keras Copmlex...');
    mmo1.Lines.Add('========================================');

    // Free Layer test class
    //
    k_Layer.Free;

    mmo1.Lines.Add('===== All tests completed Successfully. ========================================');

end;

procedure TfrmMain.btnLinReg1Click(Sender: TObject);
var
  lr_Eager : LinearRegressionEager;
begin
    mmo1.Clear;
    mmo1.Lines.Add('Linear Regression in Eager Mode Start...');
    mmo1.Lines.Add('========================================');

    lr_Eager := LinearRegressionEager.Create;
    lr_Eager.Run(mmo1);

    lr_Eager.Free;
    mmo1.Lines.Add('Linear Regression in Eager Mode End');
    mmo1.Lines.Add('===================================');
end;

procedure TfrmMain.btnLinRegClick(Sender: TObject);
var
  lr       : LinearRegression;
  lr_Eager : LinearRegressionEager;
begin
    mmo1.Clear;
    mmo1.Lines.Add('Linear Regression in Graph Mode Start...');
    mmo1.Lines.Add('========================================');

    lr := LinearRegression.Create;
    var lOk := lr.Run(mmo1);
    Assert.IsTrue(lOk,'Linear Regression in Graph Mode') ;
    lr.Free;
    mmo1.Lines.Add('Linear Regression in Graph Mode End');
    mmo1.Lines.Add('===================================');
end;

procedure TfrmMain.btnTestClick(Sender: TObject);
begin
    {$HINTS OFF}
    try
      mmo1.Clear;
      mmo1.Lines.Add('TensorFlow ver. : '+tf.Version);
      mmo1.Lines.Add('==================');

      // Init Test Graph Mode
      //
      mmo1.Lines.Add('Init Test Graph Mode');
      mmo1.Lines.Add('Session Test Start....');
      var UnitTest := TUnitTest_Basic.Create;

      UnitTest.Session_Autocast_Case2;
      UnitTest.Session_EvalTensor;
      UnitTest.Session_Eval_SmallString_Scalar;
      UnitTest.Session_Eval_LargeString_Scalar;
      UnitTest.Session_Autocast_Case0;
      UnitTest.Session_Autocast_Case1;
      UnitTest.Session_Autocast_Case2;
      UnitTest.Session_Autocast_Case3;
      mmo1.Lines.Add('Session Test End');
      // End Session Test
      //

      mmo1.Lines.Add('Random Test Start....');
      UnitTest.TFRandomSeedTest;
      UnitTest.TFRandomSeedTest2;
      UnitTest.TFRandomRaodomSeedTest;
      UnitTest.TFRandomRaodomSeedTest2;
      mmo1.Lines.Add('Random Test End....');

      mmo1.Lines.Add('Tensor Test Start....');
      UnitTest.Tensor_sparse_to_dense;
      UnitTest.Tensor_sparse_tensor_to_dense;
      UnitTest.Tensor_batch_to_space_nd;
      UnitTest.Tensor_boolean_mask;
      mmo1.Lines.Add('Tensor Test End....');
      // End Tensor Test
      //
      mmo1.Lines.Add('Variable Test Start....');
      UnitTest.Variabele_InitVariable;

      EnableEager  ;  //Eager Mode
      UnitTest.Variabele_NewVariable;
      UnitTest.Variabele_StringVar;
      UnitTest.Variabele_VarSum;
      UnitTest.Variabele_Assign1;
      UnitTest.Variabele_Assign2;
      UnitTest.Variabele_Assign3;
      UnitTest.Variabele_SliceAssign;
      UnitTest.Variabele_Accumulation;
      UnitTest.Variabele_ShouldReturnNegative;
      UnitTest.IdentityOriginalTensor;

      mmo1.Lines.Add('Variable Test End....');
      // End Variable Test
      //
      UnitTest.Free;

      var ma := ManagedAPI.Create;
      mmo1.Lines.Add('Init Test ManagedAPI');
      mmo1.Lines.Add('ArrayOpsTest Test Start....');

      ma.Slice;
      ma.Gather ;
      //gradient
      mmo1.Lines.Add('Gradient Test');
      ma.GradientFloatTest;
      ma.GradientDefaultTest;
      ma.GradientOperatorMulTest;
      ma.GradientSliceTest;
      ma.GradientConcatTest;
      // Tensor Operate
      mmo1.Lines.Add('Tensor Operate test');
      ma.TransposeTest ;
      ma.InitTensorTest;
      ma.ConcatTest;
      ma.ConcatDoubleTest;
      ma.TestZerosLike;

      // StringsApiTest
      mmo1.Lines.Add('Strings Api Test test');
      ma.StringFromBytes;
      ma.StringEqual;
      ma.StringArray;
      ma.StringSplit;

      mmo1.Lines.Add('Test ManagedAPI Test End....');
      ma.Free;

      mmo1.Lines.Add('Activation Test Start....');
      var ActTest := ActivationFunctionTest.Create;
      ActTest.Sigmoid;
      ActTest.ReLU;
      ActTest.TanH;
      mmo1.Lines.Add('Activation Test End....');
      ActTest.Free;

      mmo1.Lines.Add('clip_by_global_norm Test Start....');
      var Clips := EagerModeTestBase.Create;
      Clips.clip_by_global_norm;
      mmo1.Lines.Add('clip_by_global_norm Test End....');
      Clips.Free;

      mmo1.Lines.Add('Neural NetworkTest Test Start....');
      var NeuralNetworkTest := EagerModeTestBase.Create;
      NeuralNetworkTest.NeuralNetworkTest_l2_loss;
      mmo1.Lines.Add('Neural NetworkTest Test End....');
      NeuralNetworkTest.Free;

      mmo1.Lines.Add('Bitwise op. Test Start....');
      var Bitwise := BitwiseApiTest.Create;
      Bitwise.BitwiseAnd;
      Bitwise.BitwiseOr;
      Bitwise.BitwiseXOR ;
      Bitwise.Invert;
      Bitwise.LeftShift;
      Bitwise.RightShift;
      mmo1.Lines.Add('Bitwise op. Activation Test End....');
      Bitwise.Free;

      mmo1.Lines.Add('Constant Test Start....');
      var Constant := ConstantTest.Create;
      Constant.ScalarConst;
      Constant.ZerosConst;
      Constant.OnesConst ;
      Constant.OnesToHalves;
      Constant.NDimConst;
      Constant.Multiply;
      Constant.Reshape;
      mmo1.Lines.Add('Constant Test End....');
      Constant.Free;

      mmo1.Lines.Add('Linear Algebra Test Start....');
      var linAl := LinalgTest.Create;
      linAl.Einsum;
      linAl.EyeTest;
      linAl.GlobalNorm;
      linAl.LSTSQ;
      linAl.Tensordot;
      mmo1.Lines.Add('Linear Algebra End....');
      linAl.Free;
      {$HINTS ON}

       mmo1.Lines.Add('All Test Ok');
    except
       mmo1.Lines.Add('Test Failed!');
    end;
end;

procedure TfrmMain.EnableEager;
begin
  if not tf.executing_eagerly then
     tf.enable_eager_execution;
  tf.Context.ensure_initialized;
end;

procedure TfrmMain.DisableEager;
begin
   tf.compat.v1.disable_eager_execution;
end;

{ ManagedAPI }

constructor ManagedAPI.Create;
begin
    if not tf.executing_eagerly then
       tf.enable_eager_execution;
    tf.Context.ensure_initialized;
end;

destructor ManagedAPI.Destroy;
begin

end;

procedure ManagedAPI.TransposeTest;
begin
    // https://www.tensorflow.org/api_docs/python/tf/transpose#for_example_2
    var aX : TArray<TArray<Integer>> :=  [[1, 2, 3],[4, 5, 6]];
    var x := tf.constant(aX);
    var transpose_x := tf.transpose(x);
    var tr_numpy : TArray<Integer> := transpose_x[0].numpy.ToArray<Integer>;
    var aTest0   : TArray<Integer> := [1,4];
    Assert.IsTrue( TUtils.SequenceEqual<Integer>(aTest0, tr_numpy) );
    tr_numpy := transpose_x[1].numpy.ToArray<Integer>;
    aTest0   := [2,5];
    Assert.IsTrue( TUtils.SequenceEqual<Integer>(aTest0, tr_numpy) );
    tr_numpy := transpose_x[2].numpy.ToArray<Integer>;
    aTest0   := [3,6];
    Assert.IsTrue( TUtils.SequenceEqual<Integer>(aTest0, tr_numpy) );
    var aA :  TArray<TArray<TArray<TArray<Integer>>>> :=  [
                                                              [
                                                                  [
                                                                      [ 1, 11, 2, 22 ]
                                                                  ],
                                                                  [
                                                                      [ 3, 33, 4, 44 ]
                                                                  ]
                                                              ],
                                                              [
                                                                  [
                                                                      [ 5, 55, 6, 66 ]
                                                                  ],
                                                                  [
                                                                      [ 7, 77, 8, 88 ]
                                                                  ]
                                                              ]
                                                          ] ;
    var a := tf.constant( np.np_array(aA) );
    var aPerm : TAxis := [ 3, 1, 2, 0 ];
    var actual_transposed_a := tf.transpose(a, @aPerm);
    var aE :  TArray<TArray<TArray<TArray<Integer>>>> := [
                                                              [
                                                                  [ [ 1, 5 ] ], [ [ 3, 7 ] ]
                                                              ],
                                                              [
                                                                  [ [ 11, 55 ] ], [ [ 33, 77 ] ]
                                                              ],
                                                              [
                                                                  [
                                                                      [ 2, 6 ]
                                                                  ],
                                                                  [
                                                                      [ 4, 8 ]
                                                                  ]
                                                              ],
                                                              [
                                                                  [
                                                                      [ 22, 66 ]
                                                                  ],
                                                                  [
                                                                      [ 44, 88 ]
                                                                  ]
                                                              ]
                                                          ];
    var expected_transposed_a := tf.constant( np.np_array(aE) );
    Assert.IsTrue(TFShape.Create([4, 2, 1, 2])= actual_transposed_a.shape);
    Assert.IsTrue(expected_transposed_a.numpy.equals(actual_transposed_a.numpy) );
end;

procedure ManagedAPI.InitTensorTest;
begin
    var aX : TArray<TArray<TArray<Integer>>> :=  [ [ [ 1 ], [ 2 ], [ 3 ] ],
                                                   [ [ 4 ], [ 5 ], [ 6 ] ]
                                                  ];
    var a := tf.constant(np.np_array(aX));
    Assert.IsTrue( TUtils.SequenceEqual<Int64>([ 2, 3, 1 ], a.shape.dims));
    var b := tf.constant( aX );
    Assert.IsTrue( TUtils.SequenceEqual<Int64>([ 2, 3, 1 ], b.shape.dims));
end;

procedure ManagedAPI.ConcatTest;
begin
    var aA : TArray<TArray<Integer>> := [ [ 1, 2 ], [ 3, 4 ] ];
    var a := tf.constant( aA );
    aA    := [ [ 5, 6 ], [ 7, 8 ] ];
    var b := tf.constant( aA);
    aA    := [ [ 9, 10 ], [ 11, 12 ] ];
    var c := tf.constant( aA );
    var concatValue := tf.concat([ a, b, c ],  0);
    Assert.IsTrue( TUtils.SequenceEqual<Int64>([ 6, 2 ], concatValue.shape.dims) );
end;

procedure ManagedAPI.ConcatDoubleTest;
begin
    var aA : TArray<TArray<Double>> := [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ];
    var a := tf.constant( aA );
    aA    := [ [ 5.0, 6.0 ], [ 7.0, 8.0 ] ];
    var b := tf.constant( aA );
    aA    := [ [ 9.0, 10.0 ], [ 11.0, 12.0 ] ];
    var c := tf.constant( aA );
    var concatValue := tf.concat([ a, b, c ],  0);
    Assert.IsTrue( TUtils.SequenceEqual<Int64>([ 6, 2 ], concatValue.shape.dims) );
end;

procedure ManagedAPI.TestZerosLike;
begin
    var a2D : TArray<TArray<Integer>> := [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];
    var zeros2D := tf.zeros_like( TNdArray.Create(a2D) );
    var z  := zeros2D[0].numpy.ToArray<Integer>;
    var z1 := zeros2D[1].numpy.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>( [0, 0, 0 ], z) );
    Assert.IsTrue(TUtils.SequenceEqual<Integer>( [0, 0, 0 ], z1));
    var a1D : TArray<Integer> := [ 1, 2, 3 ];
    var zeros1D := tf.zeros_like( TNdArray.Create(a1D) );
    z  := zeros1D.numpy.ToArray<Integer>;
    Assert.IsTrue(TUtils.SequenceEqual<Integer>( [0, 0, 0 ], z) );
end;

procedure ManagedAPI.Slice;
begin
    // Tests based on example code in TF documentation
    {$HINTS OFF}
    var input_array := tf.constant( np.np_array<Integer>([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 ]).reshape(TFShape.Create( [3,2,3] )) );
    var indices     := tf.constant(np.np_array<Integer>([ 0, 2 ]));

    var r1 := array_ops.slice( input_array, TArray<Integer>.Create( 1, 0, 0 ), TArray<Integer>.Create( 1, 1, 3 ) );
    Assert.IsTrue(TFShape.Create([1,1,3])= r1.shape);
    var r1np := r1.numpy;
    Assert.AreEqual<Integer>(NDArray(r1np[[0, 0, 0]]), 3);
    Assert.AreEqual<Integer>(NDArray(r1np[[0, 0, 1]]), 3);
    Assert.AreEqual<Integer>(NDArray(r1np[[0, 0, 2]]), 3);

    var r2 := array_ops.slice( input_array, TArray<Integer>.Create( 1, 0, 0 ), TArray<Integer>.Create( 1, 2, 3 ) );
    Assert.IsTrue(TFShape.Create([1, 2, 3])= r2.shape);
    var r2np := r2.numpy;
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 0, 0]]), 3);
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 0, 1]]), 3);
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 0, 2]]), 3);
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 1, 0]]), 4);
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 1, 1]]), 4);
    Assert.AreEqual<Integer>(NDArray(r2np[[0, 1, 2]]), 4);

    var r3 := array_ops.slice( input_array, TArray<Integer>.Create( 1, 0, 0 ), TArray<Integer>.Create( 2, 1, 3 ) );
    Assert.IsTrue(TFShape.Create([2, 1, 3])= r3.shape);
    var r3np := r3.numpy;
    Assert.AreEqual<Integer>(NDArray(r3np[[0, 0, 0]]), 3);
    Assert.AreEqual<Integer>(NDArray(r3np[[0, 0, 1]]), 3);
    Assert.AreEqual<Integer>(NDArray(r3np[[0, 0, 2]]), 3);
    Assert.AreEqual<Integer>(NDArray(r3np[[1, 0, 0]]), 5);
    Assert.AreEqual<Integer>(NDArray(r3np[[1, 0, 1]]), 5);
    Assert.AreEqual<Integer>(NDArray(r3np[[1, 0, 2]]), 5);
end;

procedure ManagedAPI.StringFromBytes;
begin
    var jpg := tf.constant( TArray<Byte>.Create( $41, $ff, $d8, $ff ), tf.string_t);
    var strings := jpg.ToString;
    Assert.AreEqual(strings, 'tf.Tensor: shape=(), dtype=TF_STRING, numpy="A\xff\xd8\xff"');
end;

procedure ManagedAPI.StringEqual;
begin
    var str1 := tf.constant('Hello1');
    var str2 := tf.constant('Hello2');
    var res := tf.equal(str1, str2);
    var bRes : NDArray := res.numpy;
    Assert.IsFalse( Boolean(bRes) );
    var str3 := tf.constant('Hello1');
    res      := tf.equal(str1, str3);
    bRes     := res.numpy;
    Assert.IsTrue( Boolean(bRes) );
    var str4 := tf.strings.substr(str1, 0, 5);
    var str5 := tf.strings.substr(str2, 0, 5);
    res      := tf.equal(str4, str5);
    bRes     := res.numpy;
    Assert.IsTrue( Boolean(bRes) );
end;

procedure ManagedAPI.StringArray;
begin
    var strings : TArray<string> := [ 'map_and_batch_fusion', 'noop_elimination', 'shuffle_and_repeat_fusion' ];
    var tensor := tf.constant(strings, tf.string_t, nil, 'optimizations');
    var s0 : NDarray := tensor[0].numpy ;
    var s1 : NDarray := tensor[1].numpy ;
    var s2 : NDarray := tensor[2].numpy ;
    Assert.AreEqual<integer>(3, tensor.shape[0]);
    Assert.AreEqual<string>(s0, strings[0]);
    Assert.AreEqual<string>(s1, strings[1]);
    Assert.AreEqual<string>(s2, strings[2]);
end;

procedure ManagedAPI.StringSplit;
begin
    var tensor        := tf.constant(TArray<String>.Create('hello world', 'tensorflow .net csharp', 'fsharp' ));
    var ragged_tensor := tf.strings.split(tensor);
    Assert.IsTrue(TFShape.Create([3, -1])= ragged_tensor.shape);
end;

procedure ManagedAPI.Gather ;
begin
    var input_array := tf.constant(np.arange(12).reshape(TArray<Integer>.Create(3, 4)).astype(np.np_float32));
    var indices     := tf.constant(np.np_array(TArray<Integer>.Create( 0, 2 )));
    var res := array_ops.gather(input_array, indices);
    Assert.IsTrue(TFShape.Create([2, 4]) = res.shape);
    Assert.AreEqual<Single>(NDArray(res.numpy[[0, 0]]), 0.0);
    Assert.AreEqual<Single>(NDArray(res.numpy[[0, 1]]), 1.0);
    Assert.AreEqual<Single>(NDArray(res.numpy[[1, 3]]), 11.0);

    // Tests based on example code in Python doc string for tf.gather()
    var p1 := tf.random.normal(TFShape.Create([5, 6, 7, 8]));
    var i1 := tf.random_uniform(TFShape.Create([10, 11]), 0, 7, tf.int32_t);
    var r1 := tf.gather(p1, i1,'', 2);
    Assert.IsTrue(TFShape.Create([5, 6, 10, 11, 8]) = r1.shape);

    var p2 := tf.random.normal(TFShape.Create([4,3]));
    var a : TArray< TArray<Integer> > := [[0, 2]] ;
    var i2 := tf.constant(a);
    var r2 := tf.gather(p2, i2, '', 0);
    Assert.IsTrue(TFShape.Create([1, 2, 3]) = r2.shape);

    var r3 := tf.gather(p2, i2, '', 1);
    Assert.IsTrue(TFShape.Create([4,1,2]) = r3.shape);
end;

procedure ManagedAPI.GradientFloatTest;
begin
    var x : TResourceVariable := tf.Variable(3.0, '', tf.float32_t);
    var tape := tf.GradientTape;
    var y : TTensor:= tf.square(x);
    var y_grad := tape.gradient(TFTensor(y), ResourceVariable(x));
    Assert.AreEqual<Single>(9.0, Single(y));
end;

procedure ManagedAPI.GradientDefaultTest ;
begin
    var x : TResourceVariable := tf.Variable(3.0);
    var tape := tf.GradientTape;
    var y : TTensor:= tf.square(x);
    var y_grad := tape.gradient(TFTensor(y), ResourceVariable(x));
    Assert.AreEqual<Double>(9.0, Double(y));
end;

procedure ManagedAPI.GradientOperatorMulTest;
begin
    var x : TTensor := tf.constant(Single(0));
    var w := tf.Variable( TArray<Single>.Create( 1, 1 ) );
    var gt:= tf.GradientTape;
    var y := x * w;
    var cc := TFtensor(y).numpy.ToArray<Single>;
    var gr := gt.gradient(y, w);
    Assert.AreEqual<TArray<Single>>([ 0, 0 ], gr.numpy.ToArray<Single>);
end;

procedure ManagedAPI.GradientSliceTest;
begin
    var X : TTensor           := tf.zeros(10);
    var W : TResourceVariable := tf.Variable(Single(-0.06), 'weight');
    var b : TResourceVariable := tf.Variable(Single(-0.73), 'bias');
    var g := tf.GradientTape;
    var pred := W * X + b;
    var test := tf.slice<Integer,Integer>(pred, [ 0 ], pred.shape);
    var gradients : Tuple<TFTensor,TFTensor> := g.gradient(test, Tuple<ResourceVariable, ResourceVariable>.Create(W, b));

    Assert.AreEqual<Single>( Single(ttensor(gradients.Value1)), 0);
    Assert.AreEqual<Single>( Single(ttensor(gradients.Value2)), 10);
end;

procedure ManagedAPI.GradientConcatTest;
var
 pA : TAxis;
begin
    var w1 := tf.Variable( TArray< Tarray<Single> >.Create(Tarray<Single>.Create(1) ) );
    var w2 := tf.Variable( TArray< Tarray<Single> >.Create(Tarray<Single>.Create(3) ) );
    var g  := tf.GradientTape;
    var w  := tf.concat([ w1.toTensor, w2.toTensor ], 0);
    var x  := tf.ones(tfshape.Create([1, 2]));

    pA := 1;
    var y  := tf.reduce_sum(x, @pA);
    var r  := tf.matmul(w, x);
    var gradients := g.gradient(r, w);
    var t1 : TTensor := gradients[0][0];
    var t2 : TTensor := gradients[1][0];
    Assert.AreEqual<Single>(Single(t1), 2);
    Assert.AreEqual<Single>(Single(t2), 2);
end;

end.
