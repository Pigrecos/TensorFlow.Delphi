unit untMain;

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
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

  TensorFlow.Tensor,
  NumPy.NDArray;

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

  end;

  TForm1 = class(TForm)
    btn1: TBitBtn;
    mmo1: TMemo;
    procedure btn1Click(Sender: TObject);
    procedure FormShow(Sender: TObject);
  private

  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation
         uses System.Types,
              System.TypInfo,
              TensorFlow.Constant_op,
              Tensorflow.array_ops,
              Tensorflow.math_ops,

              TensorFlow.Slice,

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

  inherited Destroy;
end;

procedure TUnitTest_Basic.Session_Autocast_Case0;
begin
    var sess := tf.Session.as_default;
    var operation : ITensorOrOperation := tf.global_variables_initializer;
    // the cast to ITensorOrOperation is essential for the test of this method signature
    var ret := sess.run(operation);
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

    var input : TTensor := tf.placeholder(tf.float32_, TFShape.Create([6]));
    var op := tf.reshape(input, TFShape.Create([2,3]));
    sess.run(tf.global_variables_initializer);

    // var input1 : TArray<Integer> :=[1, 2, 3, 4, 5, 6];
    // var aValue : NDArray  := np.np_array(input1).astype(np.np_float32) ; Valid
    var aValue : NDArray  := np.np_array<Integer>([1, 2, 3, 4, 5, 6]).astype(np.np_float32) ;
    aValue := aValue + Single(0.1);
    var ret := sess.run(op, [ FeedItem.Create( input, aValue ) ] );
end;

procedure TUnitTest_Basic.Session_Autocast_Case3;
begin
    var sess := tf.Session.as_default;

    var input : TTensor := tf.placeholder(tf.float32_, TFShape.Create([6]));
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

    var tipo := ret.dtype;
    var t    := ret.ToString;
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
    Assert.AreEqual<String>( res[0], 'heythere');

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
    var init := tf.compat.v1.global_variables_initializer;
    var sess := tf.compat.v1.Session;
    sess.run(init);
    // Usage passing the session explicitly.
    v.eval(sess);
    // Usage with the default session.  The 'with' block
    // above makes 'sess' the default session.
    v.eval;
end;

procedure TForm1.FormShow(Sender: TObject);
begin
    Form1.Caption := 'TensorFlow lib ver. : ('+tf.Version +') TensoFlow.NET Commit(3fde7558e2c0a457272075219107b0dee3c8e4e5)'
end;

procedure TForm1.btn1Click(Sender: TObject);
begin
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
    mmo1.Lines.Add('Variable Test End....');
    // End Variable Test
    //
    UnitTest.Free;
end;

end.
