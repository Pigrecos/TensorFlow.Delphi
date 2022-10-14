unit Numpy;

interface
    uses System.SysUtils, Spring,


    TF4D.Core.CApi,
    TensorFlow.DApi,
    NumPy.NDArray;

type

np = record
    const
        np_bool       : TF_DataType = TF_DataType.TF_BOOL;
        np_char       : TF_DataType = TF_DataType.TF_INT8;
        np_byte       : TF_DataType = TF_DataType.TF_INT8;
        np_uint8      : TF_DataType = TF_DataType.TF_UINT8;
        np_ubyte      : TF_DataType = TF_DataType.TF_UINT8;
        np_int16      : TF_DataType = TF_DataType.TF_INT16;
        np_uint16     : TF_DataType = TF_DataType.TF_UINT16;
        np_int32      : TF_DataType = TF_DataType.TF_INT32;
        np_uint32     : TF_DataType = TF_DataType.TF_UINT32;
        np_int64      : TF_DataType = TF_DataType.TF_INT64;
        np_uint64     : TF_DataType = TF_DataType.TF_UINT64;
        np_float32    : TF_DataType = TF_DataType.TF_FLOAT;
        np_float64    : TF_DataType = TF_DataType.TF_DOUBLE;
        np_double     : TF_DataType = TF_DataType.TF_DOUBLE;
        np_decimal    : TF_DataType = TF_DataType.TF_DOUBLE;
        np_complex_   : TF_DataType = TF_DataType.TF_COMPLEX;
        np_complex64  : TF_DataType = TF_DataType.TF_COMPLEX64;
        np_complex128 : TF_DataType = TF_DataType.TF_COMPLEX128;

        class function np_array<T>(data: T): TNDArray;overload ;static;
        class function np_array<T>(data: TArray<T>): TNDArray;overload ;static;
        class function arange<T>(_end: T): TNDArray;static;
end;

implementation
        uses Tensorflow;

{ np }

class function np.np_array<T>(data: T): TNDArray;
begin
    var aValue := TValue.From<T>(data);
    if not aValue.IsArray then
    begin
      var aData : TArray<T> := [data];
      var a := TValue.From< TArray<T> >(aData);
      Result := TNDarray.create(a)
    end else
    begin
        Result := TNDarray.create(aValue) ;
        var s2 :=  Result.shape;
    end;
end;

class function np.arange<T>(_end: T): TNDArray;
begin
    var r : T := System.Default(T);
    var start := TValue.From<T>(r);
    var eEnd  := TValue.From<T>(_end);

    Result := TNDArray.Create (  tf.range( start, eEnd) );
end;

class function np.np_array<T>(data: TArray<T>): TNDArray;
begin
    var a := TValue.From< TArray<T> >(data);
    Result := TNDarray.create(a)
end;

end.
