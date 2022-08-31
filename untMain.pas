unit untMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Buttons,
  TensorFlow.LowLevelAPI,
  Tensorflow,
  TensorFlow.DApiBase,
  TensorFlow.DApi,
  Tensorflow.Utils,
  TensorFlow.Ops,
  Spring,
  rtti,
  NDArray, CNClrLib.Comp;

type
  TForm1 = class(TForm)
    btn1: TBitBtn;
    mmo1: TMemo;
    procedure btn1Click(Sender: TObject);
  private
    function fill<T>(dims: TFTensor; value: T; name: AnsiString= ''): TFTensor;
    function zeros(shape: TFTensor; dtype: TF_DataType = TF_DataType.TF_FLOAT; name: AnsiString = ''): TFTensor; overload;
    function zeros(shape: TFShape;  dtype: TF_DataType = TF_DataType.TF_FLOAT; name: AnsiString = '') : TFTensor; overload;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

function TForm1.fill<T>(dims:TFTensor; value: T;name: AnsiString): TFTensor;
begin
    Result := tf.Context.ExecuteOp( 'Fill', name,ExecuteOpArgs.Create([dims, TValue.From<T>(value)]) ).FirstOrDefault;
end;

function TForm1.zeros(shape: TFShape; dtype: TF_DataType; name: AnsiString) : TFTensor;
begin
    var ddtype := TUtils.as_base_dtype(dtype);

    var ss := TValue.From<TFShape>(shape);
    var scope := TOps.name_scope(name, 'zeros', @ss).ToString;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros', @ss),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := scope;
                                                case ddtype of
                                                  TF_DataType.TF_DOUBLE: Result := TOps.constant(Double(0.0));
                                                  TF_DataType.TF_FLOAT:  Result := TOps.constant(Single(0.0));
                                                 else
                                                    Result := TOps.constant(0);
                                                end;
                                                fill(TFTensor.Create(shape), Result,  name);
                                            end );
end;

function TForm1.zeros(shape: TFTensor; dtype: TF_DataType; name: AnsiString) : TFTensor;

begin


    var ddtype := TUtils.as_base_dtype(dtype);

    var vShape := TValue.From<TFTensor>(shape) ;
    var scope := TOps.name_scope(name, 'zeros', @vShape).ToString;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'zeros', @vShape),
                                          function(v1: TNameScope): TFTensor
                                            begin

                                                name := scope;
                                                case ddtype of
                                                  TF_DataType.TF_BOOL:
                                                    Result := fill(shape, tf.constant(false, dtype), name);
                                                  TF_DataType.TF_DOUBLE:
                                                    Result := fill(shape, tf.constant(0.0, dtype),  name);
                                                  TF_DataType.TF_FLOAT:
                                                    Result := fill(shape, tf.constant(0.0, dtype),  name);
                                                  TF_DataType.TF_INT32:
                                                    Result := fill(shape, tf.constant(0, dtype),  name);
                                                  else
                                                    raise Exception.Create('can''t find type for zeros');
                                                end;

                                            end );



end;

procedure TForm1.btn1Click(Sender: TObject);
var
  tf         : TTensorflow;
  ctx        : TContext;
  te,te1,tes : TEagerTensor;
  testA      : TArray<TArray<TArray<TArray<Int32>>>> ;
  testA1     : TArray<TArray<TArray<Int32>>> ;

  testString : TArray<TFString> ;

  ndArrayTest : TNDArray;

begin
    mmo1.Clear;
    tf := TTensorflow.Create;
    mmo1.Lines.Add('Crazione tf.Context');
    mmo1.Lines.Add('TensorFlow ver. : '+tf.Version);

    //
    //
    ctx := tf.Context;
    var h := ctx.Handle_;
    //
    //
    var shape1 := TFShape.Create([4,5,3,4]);
    SetLength(testA,4);
    for var i := 0 to Length(testA) - 1 do
    begin
       SetLength( testA[i],5);
       for var k := 0 to Length(testA[i]) - 1 do
       begin
           SetLength( testA[i][k],3);
           for var c := 0 to Length(testA[i][k]) - 1 do
           begin
               for var j := 0 to 4 - 1 do
               begin
                  testA[i][k][c] := testA[i][k][c] + [ Random($FF)  ] ;
               end;
           end;
       end;
    end;

    te1 := TEagerTensor.Create(testA,shape1,TF_DataType.TF_UINT32);
    mmo1.Lines.Add('Creazione TEagerTensor : TArray<TArray<TArray<TArray<Int32>>>>');
    //var rr := te1.ToArray<UInt32>;

    //
    //
    var shape := TFShape.Create([4,5,3]);
    SetLength(testA1,4);
    for var i := 0 to Length(testA1) - 1 do
    begin
       SetLength( testA1[i],5);
       for var k := 0 to Length(testA1[i]) - 1 do
       begin
           for var j := 0 to 3 - 1 do
           begin
              testA1[i][k] := testA1[i][k] + [ Random($FF)  ]
           end;
       end;
    end;
    te := TEagerTensor.Create(testA1,shape,TF_DataType.TF_UINT32)  ;
    mmo1.Lines.Add('Creazione TEagerTensor : TArray<TArray<TArray<Int32>>>');

    var t1 := TEagerTensor.GetDims(te);
    var k1 := TEagerTensor.GetRank(te);

    var ssDev :=te.Device;
    mmo1.Lines.Add('Device : '+ ssDev);
    //
    //
    var shapeS := TFShape.Create([6]);
    testString := ['ABCD','123456','Abxyu','48778ER','Massimo','VVVVV123'];
    tes := TEagerTensor.Create(testString,shapeS);
    mmo1.Lines.Add('Creazione TEagerTensor : TArray<TFString>');



    var x0 : int8 := 102;
    var x1 : Byte := 102;
    var x2 : int16 := 102;
    var x6 : word := 102;
    var x3 : cardinal := 102;
    var x3_: Integer := 102;
    var x4 : int64 := 102;
    var x5 : uint64 := 102;
    var x7 : Boolean := True;
    var x8 : string := '31';
    var x9 : AnsiString := '32';

    ndArrayTest := TNDArray.Create(x0);
    ndArrayTest := TNDArray.Create(x1);
    ndArrayTest := TNDArray.Create(x2);
    ndArrayTest := TNDArray.Create(x3);
    ndArrayTest := TNDArray.Create(x3_);
    ndArrayTest := TNDArray.Create(x4);
    ndArrayTest := TNDArray.Create(x5);
    ndArrayTest := TNDArray.Create(x6);
    ndArrayTest := TNDArray.Create(x7);
    ndArrayTest := TNDArray.Create(x8);
    ndArrayTest := TNDArray.Create(x9);
    mmo1.Lines.Add('Creazione TNDArray : scalar explicit');

    ndArrayTest := TNDArray.scalar<byte>(x1);
    ndArrayTest := TNDArray.scalar<int64>(x4);

    var d : Double := 10.4;
    ndArrayTest := TNDArray.scalar<Double>(d);
    var s : single := 10.4;
    ndArrayTest := TNDArray.scalar<Single>(s);
    mmo1.Lines.Add('Creazione TNDArray : scalar Generic');

    var y : Tarray<int16> := [44,55,66];
    ndArrayTest := TNDArray.Create(y);
    mmo1.Lines.Add('Creazione TNDArray : Tarray<int16>');

    ndArrayTest := TNDArray.Create(testA1);
    mmo1.Lines.Add('Creazione TNDArray : TArray<TArray<TArray<Int32>>>');

    ndArrayTest := TNDArray.Create(testString);
    mmo1.Lines.Add('Creazione TNDArray : TArray<TFString>');

    ndArrayTest := TNDArray.Create(testA);


   /// test OpDefLibrary._apply_op_helper
   ///
   var dtype  : TF_DataType := TF_INT32;
   var shape0 : TFShape := nil;

   var re := System.Rtti.TValue.From<TFShape>(shape0);

   var Args : TArray<TParameter>;
   SetLength(Args,2);
   Args[0].sNome := 'dtype';
   Args[0].vValue:= TValue.From<Integer>(Ord(dtype));
   Args[1].sNome := 'shape';
   Args[1].vValue:= TValue.From<TFShape>(shape0);
   var _op := OpDefLibrary._apply_op_helper('Placeholder', 'test', Args);
   mmo1.Lines.Add('KerasNet : Placeholder');


   zeros( TFShape.Create([2, 3, 4, 5]) );

   var name := 'set_diag';
   var input    : TFTensor;
   var diagonal : TFTensor;
   var k        : Integer;
   //tf.Context.ExecuteOp('MatrixSetDiagV3',name,ExecuteOpArgs.Create([input, diagonal, k]))
end;

end.
