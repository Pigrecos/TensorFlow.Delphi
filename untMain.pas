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
  end;

  ManagedAPI = class
    private

    public
       constructor Create;
       destructor  Destroy;override;

       procedure Slice;
       procedure Gather;
  end;

  TForm1 = class(TForm)
    btnTest: TBitBtn;
    mmo1: TMemo;
    btnLinReg: TBitBtn;
    procedure btnTestClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure btnLinRegClick(Sender: TObject);
  private
    procedure EnableEager;

  public
    procedure DisableEager;
  end;

var
  Form1: TForm1;

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
    var g := sess.run(init);
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

procedure TForm1.FormShow(Sender: TObject);
begin
    Form1.Caption := 'TensorFlow lib ver. : ('+tf.Version +') TensoFlow.NET Commit(3fde7558e2c0a457272075219107b0dee3c8e4e5)'
end;

type
  stack = class(TStack<Integer>)

  end;

procedure TForm1.btnLinRegClick(Sender: TObject);
var
  lr : LinearRegression;
begin
    lr := LinearRegression.Create;

    lr.Run;

end;

procedure TForm1.btnTestClick(Sender: TObject);
begin
    {$HINTS OFF}
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
    {$HINTS ON}
end;

procedure TForm1.EnableEager;
begin
  if not tf.executing_eagerly then
     tf.enable_eager_execution;
  tf.Context.ensure_initialized;
end;

procedure TForm1.DisableEager;
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

procedure ManagedAPI.Slice;
begin
    // Tests based on example code in TF documentation

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

end.
