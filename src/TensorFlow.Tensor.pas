unit TensorFlow.Tensor;

interface
    uses
       System.SysUtils, System.Classes, System.Types, Generics.Collections, Winapi.Windows,
       System.AnsiStrings, System.Rtti, Spring,spring.Collections.Base,
       TensorFlow.LowLevelAPI, TensorFlow.DApiEager, TensorFlow.DApi;

type
 /// <summary>
/// TFTensor holds a multi-dimensional array of elements of a single data type.
/// </summary>
/// <remarks>
/// <para>
/// You can create tensors with the various constructors in this class, or using
/// the implicit conversions from various data types into a TFTensor.
///</para>
/// <para>
/// The implicit conversions for basic types produce tensors of one dimesion with
/// a single element, while the implicit conversion from an array, expects a multi-dimensional
/// array that is converted into a tensor of the right dimensions.
/// </para>
/// <para>
/// The special "String" tensor data type that you will find in TensorFlow documentation
/// really represents a byte array.   You can create string tensors by using the <see cref="M:TensorFlow.TFTensor.CreateString"/>
/// method that takes a byte array buffer as input.
/// </para>
/// </remarks>
TFTensor = class(ITensorOrOperation)
 private
   FEagerTensorHandle    : PTFE_TensorHandle;
   FlDeallocator_called  : Boolean;
   FIsCreatedInGraphMode : Boolean;
   FIsList               : Boolean;
   /// <summary>
   ///     The Operation that produces this tensor as an output.
   /// </summary>
   Ftf_output            : Nullable<TF_Output>;
   FValue_index          : Integer;
   FOverride_dtype       : TF_DataType;
   FId                   : Int64;
   /// <summary>
   ///     The Graph that contains this tensor.
   /// </summary>
   FGraph                : TFGraph;
   FRank                 : Integer;
   FShape                : TFShape;

   function GetByteSize: UInt64;
   function GetDataTypeSize: UInt64;
   function GetSize: UInt64;
   function GetData: Pointer;
   function GetDim: Tarray<UInt64>;
   function GetRank: Integer;
   function GetShape: TFShape;
   procedure Setshape(const Value: TFShape);
   function GetName: string;
   /// <summary>
	 /// Returns the data type for the tensor.
	 /// </summary>
	 /// <value>The type of the tensor.</value>
   function GetType: TF_DataType; reintroduce;
   function GetDevice: string;
   function GetTensorDataPointer: Pointer;

   procedure UpdateTensoData;

   class function    TF_NewTensor(shape: TFShape; dtype: TF_DataType; data: Pointer):PTF_Tensor; overload;
   class function    TF_NewTensor(data: TArray<Byte>; shape: TFShape; dtype: TF_DataType):PTF_Tensor; overload;
   class function    StringTensor(srcArray: TArray<TArray<Byte>>; shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<TFString>; shape: TFShape):PTF_Tensor;overload;
   class function    StringTensor(srcArray: TArray<string>; shape: TFShape):PTF_Tensor;overload;

   procedure InitTensor(shape: TFShape; dtype: TF_DataType);overload;
   Procedure InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType); overload;

   function InitTensor<T>(aArray: TArray<T>; shape: TFShape): PTF_Tensor; overload;

   class function InitTensor<T>(aArray: TArray<T>;                        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<T>>;                shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<T>>>;        shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   class function InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: TFShape; dtype: TF_DataType): PTF_Tensor; overload;
   function StringBytes(index: Integer): TArray<byte>; overload;
   function StringBytes:TArray< TArray<Byte> >; overload;
   function StringData(index: integer): AnsiString; overload;

 protected
   procedure NativeDispose(hnd: Pointer); override;
 public
   // Scalar
   constructor Create(hnd: Pointer);   overload;
   constructor Create(const value: Boolean);  overload; virtual;
   constructor Create(const value: Byte);     overload; virtual;
   constructor Create(const value: Int8);     overload; virtual;
   constructor Create(const value: UInt16);   overload; virtual;
   constructor Create(const value: Int16);    overload; virtual;
   constructor Create(const value: Cardinal); overload; virtual;
   constructor Create(const value: Integer);  overload; virtual;
   constructor Create(const value: UInt64);   overload; virtual;
   constructor Create(const value: Int64);    overload; virtual;
   constructor Create(const value: Single);   overload; virtual;
   constructor Create(const value: Double);   overload; virtual;
   constructor Create(const value: TFString); overload; virtual;

   constructor Create(shape : TFShape; dtype:TF_DataType); overload;
   constructor Create(bytes : TArray<Byte>;shape : TFShape; dtype:TF_DataType); overload;
   constructor Create(op: TFOperation; value_index: Integer; dtype:TF_DataType); overload;
   // Array of T
   class function Create<T>(aArray: TArray<T>;                        shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<T>>;                shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<TArray<T>>>;        shape: PTFShape=nil):TFTensor; overload;
   class function Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: PTFShape=nil):TFTensor; overload;

   function StringData: TArray<TFString>; overload;
   //
   function ToString: string;override;
   //
   destructor  Destroy; override;
   //
   function MMLck(): TFTensor;
   /// <summary>
   ///
   /// </summary>
   /// <typeparam name="T"></typeparam>
   /// <returns></returns>
   function ToArray<T>:TArray<T>;

   function _as_tf_output : TF_Output;

   /// <summary>
   /// Copies the memory of current buffer onto newly allocated array.
   /// </summary>
   /// <returns></returns>
   function BufferToArray: TArray<Byte>;

   class function TestTensor: Boolean;

   property  bytesize      : UInt64         read GetByteSize;
   property  dtypesize     : UInt64         read GetDataTypeSize;
   property  size          : UInt64         read GetSize;
   property  buffer        : Pointer        read GetData;
   property  ndim          : Tarray<UInt64> read GetDim;
   property  value_index   : Integer        read FValue_index;
   property  override_dtype: TF_DataType    read FOverride_dtype;
   property  id            : Int64          read FId;
   property  graph         : TFGraph        read FGraph;
   property  Shape         : TFShape        read GetShape write Setshape;
   property  isList        : Boolean        read FIsList write FIsList;
   /// <summary>
   /// number of dimensions <br></br>
   /// -1 Unknown  <br></br>
   /// 0	Scalar (magnitude only) <br></br>
   /// 1	Vector (magnitude and direction) <br></br>
   /// 2	Matrix (table of numbers) <br></br>
   /// 3	3-Tensor (cube of numbers) <br></br>
   /// n	n-Tensor (you get the idea)
   /// </summary>
   /// <remarks>https://www.tensorflow.org/api_docs/python/tf/rank</remarks>
   property  rank             :    Integer         read GetRank;
   property  DeallocatorCalled:    Boolean         read FlDeallocator_called;
   property  isCreatedInGraphMode: Boolean         read FIsCreatedInGraphMode;
   property  TensorDataPointer:    Pointer         read GetTensorDataPointer;
   property  EagerTensorHandle: PTFE_TensorHandle  read FEagerTensorHandle write FEagerTensorHandle;
   // inherithed from  ITensorOrOperation
   property Device : string            read GetDevice;
   property Op     : TFOperation       read FOp;
   property Name   : string            read GetName;
   property Dtype  : TF_DataType       read GetType;

end;

TTensor = record
  private
      FHandleTensor : TFTensor;

      class procedure EnsureScalar(t: TTensor);static;
      class procedure EnsureDType(t: TTensor; _is: TF_DataType); static;
    function GetShape: TFShape;

  public
      class operator Implicit(t : TFTensor): TTensor;
      class operator Implicit(t : TTensor): TFTensor;

      // Scalar
      constructor Create(hnd: Pointer);   overload;
      constructor Create(const value: Boolean);  overload;
      constructor Create(const value: Byte);     overload;
      constructor Create(const value: Int8);     overload;
      constructor Create(const value: UInt16);   overload;
      constructor Create(const value: Int16);    overload;
      constructor Create(const value: Cardinal); overload;
      constructor Create(const value: Integer);  overload;
      constructor Create(const value: UInt64);   overload;
      constructor Create(const value: Int64);    overload;
      constructor Create(const value: Single);   overload;
      constructor Create(const value: Double);   overload;
      constructor Create(const value: TFString); overload;

      constructor Create(shape : TFShape; dtype:TF_DataType); overload;
      constructor Create(bytes : TArray<Byte>;shape : TFShape; dtype:TF_DataType); overload;
      constructor Create(op: TFOperation; value_index: Integer; dtype:TF_DataType); overload;

      // Array of T
      class function Create<T>(aArray: TArray<T>;                        shape: PTFShape=nil):TFTensor; overload;static;
      class function Create<T>(aArray: TArray<TArray<T>>;                shape: PTFShape=nil):TFTensor; overload;static;
      class function Create<T>(aArray: TArray<TArray<TArray<T>>>;        shape: PTFShape=nil):TFTensor; overload;static;
      class function Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>;shape: PTFShape=nil):TFTensor; overload;static;

      // Class Operator
      class operator Explicit(t : TTensor): Boolean;
      class operator Explicit(t : TTensor): Byte;
      class operator Explicit(t : TTensor): Int8;
      class operator Explicit(t : TTensor): UInt16;
      class operator Explicit(t : TTensor): Int16;
      class operator Explicit(t : TTensor): UInt32;
      class operator Explicit(t : TTensor): Int32;
      class operator Explicit(t : TTensor): UInt64;
      class operator Explicit(t : TTensor): Int64;
      class operator Explicit(t : TTensor): Single;
      class operator Explicit(t : TTensor): Double;
      class operator Explicit(t : TTensor): AnsiString;

      class function ToStringArray(t: TTensor): TArray<AnsiString>; static;

      // Property
      property HTensor : TFTensor read FHandleTensor;
      property Shape   : TFShape  read GetShape;
end;

implementation
         uses Tensorflow,TensorFlow.Ops,Tensorflow.Utils;

//------------------------------------------------------------------------------
//----------------------------- TFTensor ---------------------------------------
//------------------------------------------------------------------------------
constructor TFTensor.Create(hnd: Pointer);
begin
 inherited Create(hnd);
 FlDeallocator_called := False;

 UpdateTensoData;

 self.MMLck();
end;

constructor TFTensor.Create(const value: Boolean);
begin
   Create(InitTensor<Boolean>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Byte);
begin
   Create(InitTensor<Byte>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int8);
begin
   Create(InitTensor<Int8>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: UInt16);
begin
   Create(InitTensor<UInt16>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int16);
begin
   Create(InitTensor<Int16>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Cardinal);
begin
   Create(InitTensor<Cardinal>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Integer);
begin
   Create(InitTensor<Integer>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: UInt64);
begin
   Create(InitTensor<UInt64>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Int64);
begin
   Create(InitTensor<Int64>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Single);
begin
   Create(InitTensor<Single>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: Double);
begin
   Create(InitTensor<Double>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(const value: TFString);
begin
   Create(InitTensor<TFString>([value], TFShape.Scalar));

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(shape : TFShape; dtype:TF_DataType);
begin
   InitTensor(shape, dtype);
   Create(Handle);

   FIsCreatedInGraphMode := not tf.executing_eagerly;
   FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    InitTensor(shape,bytes,dtype) ;
    Create(Handle);

    FIsCreatedInGraphMode := not tf.executing_eagerly;
    FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<T>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if shape = nil then  sShape := TUtils.GetShape( TValue.From< TArray<T> >( aArray ) )
    else                 sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<T>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if shape = nil then  sShape := TUtils.GetShape( TValue.From< TArray<TArray<T>> >( aArray ) )
    else                 sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<TArray<T>>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if shape = nil then  sShape := TUtils.GetShape( TValue.From<  TArray<TArray<TArray<T>>> >( aArray ) )
    else                 sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

class function TFTensor.Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: PTFShape):TFTensor;
begin
    var sShape : TFShape;
    if shape = nil then  sShape := TUtils.GetShape( TValue.From<  TArray<TArray<TArray<TArray<T>>>> >( aArray ) )
    else                 sShape := shape^ ;

    var dtype := TUtils.GetDataType( TValue.From<T>( aArray[0][0][0][0] ) );

    var HTensor := InitTensor<T>(aArray, sShape,dtype);

    Result := TFTensor.Create(HTensor);

    Result.FIsCreatedInGraphMode := not tf.executing_eagerly;
    Result.FEagerTensorHandle    := nil;
end;

constructor TFTensor.Create(op: TFOperation; value_index: Integer; dtype: TF_DataType);
begin
     Fop     := op;
     FGraph := Fop.Graph;

     FValue_index := value_index;
     FOverride_dtype := dtype;
     FId := TOps.uid;
     FIsCreatedInGraphMode := not tf.executing_eagerly;
     FEagerTensorHandle    := nil;
end;

class function TFTensor.TF_NewTensor(data: TArray<Byte>; shape: TFShape; dtype: TF_DataType):PTF_Tensor;
begin
     inherited Create(Nil);

     var _length : TF_size_t := Length(data);
     var dims     := shape.Dims;

     var pDims : PTF_int64_t;
     pDims :=  PTF_int64_t(Pointer(@dims)^);

     var hHandle := TF_AllocateTensor(Integer(dtype), pDims, shape.ndim, _length);
     var ttensor := TF_TensorData(hHandle);
     if ttensor = nil then
        raise Exception.Create('AllocateTensor failed.');

     if Assigned(data)  then
       Move(@data[0], ttensor^, _length);

     Result := hHandle;
end;

class function TFTensor.TF_NewTensor(shape: TFShape; dtype: TF_DataType; data: Pointer):PTF_Tensor;
begin
     inherited Create(Nil);

     var _length : TF_size_t := shape.Size * get_datatype_size(dtype);
     var dims     := shape.Dims;

     var pDims : PTF_int64_t ;
     pDims :=  PTF_int64_t(Pointer(@dims)^);

     var hHandle := TF_AllocateTensor(Integer(dtype), pDims, shape.ndim, _length);
     var ttensor := TF_TensorData(hHandle);
     if ttensor = nil then
        raise Exception.Create('AllocateTensor failed.');

     if Assigned(data)  then
       Move(data^, ttensor^, _length);

     Result := hHandle;
end;

destructor  TFTensor.Destroy;
begin
 if FlDeallocator_called then
   Handle := Nil;
 inherited Destroy;
end;

procedure TFTensor.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TF_DeleteTensor(hnd);
end;

function TFTensor.StringBytes: TArray< TArray<Byte> >;
var
  i    : Integer;
  buf  : TArray< TArray<Byte> > ;
begin
    if dtype <> TF_DataType.TF_STRING then
      raise Exception.Create('Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})');
    //
    // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
    // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
    //
    var size : Int64 := 1;
    for var s in shape.dims do
          size := size * s;

    SetLength(buf,size);
    for i := 0 to  size - 1 do
        buf[i] := StringBytes(i);

    Result := buf;
end;

function TFTensor.StringBytes(index: Integer): TArray<byte>;
var
 data : Pointer;

begin
    if dtype <> TF_DataType.TF_STRING then
      raise Exception.Create('Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})');

    Result := [] ;
    var tstrings := TensorDataPointer;
    for var i := 0 to  shape.size - 1 do
    begin
        if index = i then
        begin
            data := TF_StringGetDataPointer(tstrings);
            var len  := TF_StringGetSize(tstrings);

            SetLength(Result,len);
            CopyMemory(@Result[0], PByte(Pointer(data)^), len);
            break;
        end;
         Inc(PByte(tstrings), TF_TSRING_SIZE);
    end;

end;

function TFTensor.StringData(index: integer): AnsiString;
Begin
    var bytes := StringBytes(index);
    Result := AnsiString (TEncoding.UTF8.GetString(bytes));
end;

function TFTensor.StringData: TArray<TFString>;
begin
    var buffer := StringBytes;
    var _str : TArray<TFString>;
    SetLength(_str, Length(buffer) );
    for var i := 0  to Length(_str) - 1 do
        _str[i] := AnsiString(TEncoding.UTF8.GetString( buffer[i]));
    Result :=  _str;
end;

class function TFTensor.StringTensor(srcArray: TArray<TArray<byte>>; shape: TFShape):PTF_Tensor;
var
  l_pTensor   : PTF_Tensor;
  size        : TF_int64_t;
  tstr        : PTFString;
begin

     var num_dims := Length(shape.Dims);
     var dims     := shape.Dims;
     size := 0;
     for var i := 0 to num_dims-1 do
       size := size + dims[i];

     var pDims : PTF_int64_t ;
     pDims :=  PTF_int64_t(Pointer(@dims)^);

     l_pTensor := TF_AllocateTensor(Int32(TF_STRING), pDims, num_dims, shape.Size * TF_TSRING_SIZE);
     tstr      := TF_TensorData(l_pTensor);

     for var i := 0 to Length(srcArray) - 1 do
     begin
          TF_StringInit(tstr);
          TF_StringCopy(tstr, PTFChar(@srcArray[i]), Length(srcArray[i]));
          Inc(PByte(tstr), TF_TSRING_SIZE);
     end;

     Result := l_pTensor;
end;

class function TFTensor.StringTensor(srcArray: TArray<TFString>; shape: TFShape):PTF_Tensor;
var
  buffer : TArray<TArray<Byte>>;
begin
    // convert string array to byte[][]
    //
    SetLength(buffer,Length(srcArray));
    for var i := 0 to Length(srcArray)- 1 do
      buffer[i] := TEncoding.UTF8.GetBytes( string(srcArray[i]) );

    Result := StringTensor(buffer,shape);
end;

class function TFTensor.StringTensor(srcArray: TArray<string>; shape: TFShape):PTF_Tensor;
var
  buffer : TArray<TArray<Byte>>;
begin
    // convert string array to byte[][]
    //
    SetLength(buffer,Length(srcArray));
    for var i := 0 to Length(srcArray)- 1 do
      buffer[i] := TEncoding.UTF8.GetBytes( srcArray[i] );

    Result := StringTensor(buffer,shape);
end;

function TFTensor.ToArray<T>: TArray<T>;
var
  l_pData,l_pVal  : Pointer ;
  l_iFullByteSize : Integer;
  res             : TArray<T>;
begin
    if Tdtypes.as_tf_dtype( TypeInfo(T) ) <> Dtype then
      raise Exception.Create('Required dtype {dtype} mismatch with {typeof(T).as_tf_dtype()}.');

    l_pData := TF_TensorData(Handle);


    if (ndim[0] = 0) or (size = 1) then
    begin
        SetLength(res,1);
        l_pVal  := @res[0];
        l_iFullByteSize := dtypesize;

        Move(l_pData^, l_pVal^, l_iFullByteSize);
        Exit;
    end;

    if (ndim[0] > 1) then
       raise Exception.Create('ToArray - ndim[0] > 1  !!!.');

    SetLength(res,size);
    l_pVal  := @res[0];
    l_iFullByteSize :=  size * dtypesize;
    Move(l_pData^, l_pVal^, l_iFullByteSize);

    Result := res;
end;

function TFTensor.ToString: string;
begin
    Result := Format('tf.Tensor "%s" shape=%s dtype=%s',[name,Shape.ToString,Tdtypes.as_numpy_name(dtype)]);
end;

procedure TFTensor.UpdateTensoData;
begin
   FEagerTensorHandle    := nil;
   FlDeallocator_called  := False;
   FIsCreatedInGraphMode := False;
   FIsList               := False;

   Ftf_output            := nil;
   FValue_index          := 0;
   FOverride_dtype       := TF_DataType.TF_DATATYPE_UNKNOWN;
   FId                   := 0;

   FGraph                := nil;

   GetRank;
   GetShape;
   GetName;
   GetType;
   GetDevice;
 end;

function TFTensor._as_tf_output: TF_Output;
begin
    if not Ftf_output.HasValue then
    begin
        var o := TFOutput.Create(Fop,FValue_index);
        Ftf_output := o.ToTF_Output;
    end;
    Result := Ftf_output;
end;

function TFTensor.BufferToArray: TArray<Byte>;
var
  l_pData,l_pVal  : Pointer ;
  res             : TArray<Byte>;
begin
    SetLength(res,bytesize);
    l_pData := TF_TensorData(Handle);

    l_pVal  := @res[0];

    Move(l_pData^, l_pVal^, bytesize);

    Result := res;
end;

procedure TFTensor.InitTensor(shape: TFShape; dtype: TF_DataType);
begin
    Handle := TF_NewTensor(shape,dtype,nil)  ;
end;

Procedure TFTensor.InitTensor(shape: TFShape; bytes: TArray<Byte>; dtype: TF_DataType);
begin
     if dtype = TF_DataType.TF_STRING then
     begin
         var buf : TArray<TArray<byte>>;
         SetLength(buf,1);
         for var i := 0 to Length(bytes) - 1 do
           buf[0][i] := bytes[i];
         Handle := StringTensor( buf, TFShape.Scalar);
     end else
     begin
         Handle  := TF_NewTensor(bytes,shape,dtype) ;
     end;
end;

function TFTensor.InitTensor<T>(aArray: TArray<T>; shape: TFShape): PTF_Tensor;
begin
    var dtype := TUtils.GetDataType( TValue.From< TArray<T> >(aArray)) ;
    

    Result := InitTensor<T>(aArray, shape,dtype)
end;

class function TFTensor.InitTensor<T>(aArray: TArray<T>;shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     if TypeInfo(T) = TypeInfo(TFString) then
     begin
         var v := TValue.From< TArray<T> >(aArray);
         var v1 := v.AsType< TArray<TFString> > ;
         Result := StringTensor(v1, shape);
     end
     else if TypeInfo(T) = TypeInfo(string)then
     begin
         var v := TValue.From< TArray<T> >(aArray);
         var v1 := v.AsType< TArray<string> > ;
         Result := StringTensor(v1, shape);
     end else
     begin
         l_pData := PByte(@aArray[0]);
         Result  := TF_NewTensor(shape,dtype,l_pData) ;
     end;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<T>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0][0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<T>>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0][0][0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

class function TFTensor.InitTensor<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: TFShape; dtype: TF_DataType): PTF_Tensor;
var
  l_pData     : Pointer;
begin
     l_pData := PByte(@aArray[0][0][0][0]);
     Result := TF_NewTensor(shape,dtype,l_pData) ;
end;

function TFTensor.GetTensorDataPointer: Pointer;
begin
     if Handle = nil then Result := nil
     else                 Result := TF_TensorData(Handle);
end;

function TFTensor.GetType: TF_DataType;
begin
    if Handle = nil then
      FDtype := FOverride_dtype
    else
      FDtype := TF_DataType( TF_TensorType(Handle) );

    Result := FDtype;
end;

function TFTensor.GetByteSize: UInt64;
begin
    if Assigned(Handle) then
     Result := TF_TensorByteSize(Handle)
   else
     Result := 0;
end;
function TFTensor.GetData: Pointer;
begin
    if Assigned(Handle) then
     Result := TF_TensorData(Handle)
   else
     Result := nil;
end;

function TFTensor.GetDataTypeSize: UInt64;
begin
    Result := Tdtypes.get_datatype_size(Dtype);
end;

function TFTensor.GetDevice: string;
begin
    Result := '';
    if FOp <> nil then
      Result := FOp.Device;
    FDevice := Result;
end;

function TFTensor.GetDim: Tarray<UInt64>;
begin
    if Assigned(Handle) then
     Result := [TF_NumDims(Handle)]
   else begin
      var output := _as_tf_output;
      var ndim := TF_GraphGetTensorNumDims(op.graph, output, tf.Status.Handle);
      Result := [ndim];
   end;
end;

function TFTensor.GetName: string;
var
 opname : string;
begin
    opname := '<unnamed>';
    if Fop <> nil then
      opname := Fop.name;

    FName := Format('%s:%d',[opname,value_index]);
    Result := FName
end;

function TFTensor.GetRank: Integer;
begin
    if not Assigned(Handle) then
    begin
        var output :=  _as_tf_output;
        var ndim : Integer := TF_GraphGetTensorNumDims(FGraph.Handle,output,tf.Status.Handle);
        Result := ndim;
    end else
    begin
        Result := TF_NumDims(Handle)
    end;
    FRank := Result;
end;

function TFTensor.GetShape: TFShape;
begin
    FShape := default(TFShape);

    if rank < 0 then
        Exit(FShape);

    var irank : TArray<Int64>; SetLength(irank,rank);
    var dims := TFShape.Create(irank);

    if not Assigned(Handle) then
    begin
        TF_GraphGetTensorShape(op.graph.Handle, _as_tf_output(), @dims, rank, tf.Status.Handle);
    end else
    begin
        for var i := 0 to rank -1 do
           dims.Dims[i] := TF_Dim(Handle, i);

        FShape := dims;
    end;
    Result := FShape;
end;

procedure TFTensor.Setshape(const Value: TFShape);
begin
    if value.IsNil then
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, nil, -1, tf.Status.Handle)
    else
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, @value.dims, value.ndim, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

function TFTensor.GetSize: UInt64;
begin
    if Handle = nil then Exit(0);

    Result := bytesize div dtypesize;
end;

function TFTensor.MMLck(): TFTensor;
var
 env: TFMMEnv;
begin
 env := TFMMEnv.GetMMEnv();
 //if Assigned(env) then
 //  env.AddTensor(self);
 Result := self;
end;

class function TFTensor.TestTensor: Boolean;

begin
    var t1 := TTensor.Create(Byte($77));
    var B2 := Byte(t1);
    Assert( B2 = $77 ) ;

    t1 := TTensor.Create(Integer($61626364));
    var I2 := Integer(t1);
    Assert( I2 = $61626364 ) ;

    var t2 := TTensor.Create('Abcd');
    var ts2 := string(t2);
    Assert( ts2 = 'Abcd' ) ;

    var s2 := TTensor.Create<String>(['Abcd','12345']);
    var tshape := s2.Shape;
    var as2 := TTensor.ToStringArray(s2);

    var testA      : TArray<TArray<TArray<TArray<Int32>>>>;
    var shape1 := TFShape.Create([4,5,3,4]);

    shape1 := TFShape.Create([4,5,3,4]);
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

    var t3 := TTensor.Create<Int32>(testA,@shape1);
    Assert( t3.Shape.Equals( TValue.From<TFShape>(shape1) ) ) ;

    Result := True;
end;

{ tensor }

constructor TTensor.Create(hnd: Pointer);
begin
    FHandleTensor := TFTensor.Create(hnd)
end;

constructor TTensor.Create(const value: UInt16);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Int16);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Cardinal);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Integer);
begin
   FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Boolean);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Byte);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Int8);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: UInt64);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Int64);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Double);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: Single);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(const value: TFString);
begin
    FHandleTensor := TFTensor.Create(value)
end;

constructor TTensor.Create(shape: TFShape; dtype: TF_DataType);
begin
    FHandleTensor := TFTensor.Create(shape,dtype)
end;

constructor TTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
    FHandleTensor := TFTensor.Create(bytes,shape,dtype)
end;

constructor TTensor.Create(op: TFOperation; value_index: Integer; dtype: TF_DataType);
begin
    FHandleTensor := TFTensor.Create(op,value_index,dtype)
end;

class function TTensor.Create<T>(aArray: TArray<T>; shape: PTFShape): TFTensor;
begin
    Result := TFTensor.Create<T>(aArray,shape)
end;

class function TTensor.Create<T>(aArray: TArray<TArray<T>>; shape: PTFShape): TFTensor;
begin
    Result := TFTensor.Create<T>(aArray,shape)
end;

class function TTensor.Create<T>(aArray: TArray<TArray<TArray<T>>>; shape: PTFShape): TFTensor;
begin
    Result := TFTensor.Create<T>(aArray,shape)
end;

class function TTensor.Create<T>(aArray: TArray<TArray<TArray<TArray<T>>>>; shape: PTFShape): TFTensor;
begin
    Result := TFTensor.Create<T>(aArray,shape)
end;

class procedure TTensor.EnsureDType(t: TTensor; _is: TF_DataType);
begin
    if t.FHandleTensor.dtype <> _is then
       raise Exception.Create('Unable to cast scalar tensor {tensor.dtype} to {@is}');
end;

class Procedure TTensor.EnsureScalar(t: TTensor);
begin
    if t.HTensor = nil then
      raise Exception.Create('Null Tensor');
    if t.FHandleTensor.shape.ndim <> 0 then
      raise Exception.Create('Tensor must have 0 dimensions in order to convert to scalar');
    if t.FHandleTensor.shape.size <> 1 then
      raise Exception.Create('Tensor must have size 1 in order to convert to scalar');
end;

class operator TTensor.Implicit(t: TTensor): TFTensor;
begin
    Result := t.FHandleTensor;
end;

class operator TTensor.Implicit(t: TFTensor): TTensor;
begin
    Result.FHandleTensor := t;
end;

class operator TTensor.Explicit(t: TTensor): Boolean;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_BOOL);
    Result := PBoolean(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Byte;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_UINT8);
    Result := pbyte(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Int8;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_INT8);
    Result := PInt8(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Int16;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_INT16);
    Result := pInt16(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): UInt16;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_UINT16);
    Result := pWord(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): UInt32;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_UINT32);
    Result := pUInt32(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Int32;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_INT32);
    Result := pInt32(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Int64;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_INT64);
    Result := pInt64(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): UInt64;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_UINT64);
    Result := pUInt64(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Single;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_FLOAT);
    Result := PSingle(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): Double;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_DOUBLE);
    Result := PDouble(@t.FHandleTensor.buffer^)^;
end;

class operator TTensor.Explicit(t: TTensor): AnsiString;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_STRING);
    Result := t.FHandleTensor.StringData(0);
end;

function TTensor.GetShape: TFShape;
begin
    Result :=  FHandleTensor.Shape;
end;

class function TTensor.ToStringArray(t: TTensor): TArray<AnsiString>;
begin
    Result := t.FHandleTensor.StringData;
end;

initialization
  if tf = nil then
     tf := TTensorflow.Create;

   TFTensor.TestTensor

end.
