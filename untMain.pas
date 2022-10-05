unit untMain;

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,rtti,

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
      destructor  Destroy;virtual;
  end;

  TSessionTest = class(GraphModeTestBase)
  private

    public
      constructor Create; override;
      destructor  Destroy; override;

      procedure EvalTensor;
      procedure Eval_SmallString_Scalar;
      procedure Eval_LargeString_Scalar;
      procedure Autocast_Case0;
      procedure Autocast_Case1;
      procedure Autocast_Case2;
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

procedure TSessionTest.Autocast_Case0;
begin
    var sess := tf.Session.as_default;
    var operation : ITensorOrOperation := tf.global_variables_initializer;
    // the cast to ITensorOrOperation is essential for the test of this method signature
    var ret := sess.run(operation);
end;

procedure TSessionTest.Autocast_Case1;
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

    var tipo := ret.dtype;
    var t := ret.ToString;
end;

procedure TSessionTest.Autocast_Case2;
begin
    var sess := tf.Session.as_default;
    var input := tf.placeholder(tf.float32_, TFShape.Create([6]));
    var shape : TArray<Integer> := [ 2, 3 ];
    var op := tf.reshape(input, shape);
    sess.run(tf.global_variables_initializer);

    var input1 : TArray<Integer> :=[1, 2, 3, 4, 5, 6];
    var npArray : NDArray  := np.np_array(input1).astype(np._float32) ;
    var addFloat : NDArray := Single(0.1);
    var res : TNdArray  := npArray+addFloat;
    var ret := sess.run(op, [ FeedItem.Create(input, res ) ] );

    
end;

constructor TSessionTest.Create;
begin
  inherited Create;

end;

destructor TSessionTest.Destroy;
begin

  inherited Destroy;
end;

procedure TSessionTest.EvalTensor;
begin
    var a : TTensor := constant_op.constant( np.np_array(3.0).reshape( TFShape.Create([1, 1])) );
    var b : TTensor := constant_op.constant(  np.np_array(2.0).reshape( TFShape.Create([1, 1])) );
    var c : TTensor := math_ops.matmul(a, b, 'matmul');

    var sess := tf.Session;

    var res : NDArray := c.eval(sess);
    Assert.AreEqual<Single>(NDArray(res[0]), 6.0,'EvalTensor Error!');
end;

procedure TSessionTest.Eval_SmallString_Scalar;
begin

    var a := constant_op.constant( '123 heythere 123 ', TF_DataType.TF_STRING,'Const');
    var c := tf.strings.substr(a, 4, 8);
    var sess := tf.Session;
    var res : TArray<TF_TString> := c.eval(sess).StringData;
    Assert.AreEqual<String>( res[0], 'heythere');

end;

procedure TSessionTest.Eval_LargeString_Scalar;
begin
    var size  : Integer := 30000;
    var s := string.Create('a',size);
    var a := constant_op.constant(AnsiString(s), TF_DataType.TF_STRING,'Const');
    var c := tf.strings.substr(a, 0, size - 5000);
    var sess := tf.Session;
    var res := c.eval(sess).ToByteArray ;
    var sRes := TUTF8Encoding.UTF8.GetString(res);
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
    mmo1.Lines.Add('Session Test....');
    var SessionTest := TSessionTest.Create;
    SessionTest.EvalTensor;
    SessionTest.Eval_SmallString_Scalar;
    SessionTest.Eval_LargeString_Scalar;
    SessionTest.Autocast_Case0;
    SessionTest.Autocast_Case1;
    SessionTest.Autocast_Case2;

    // End Test
    //
    mmo1.Lines.Add('End Test ');
    SessionTest.Free;
end;

end.
