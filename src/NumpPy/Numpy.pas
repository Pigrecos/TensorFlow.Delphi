unit Numpy;

interface
    uses System.SysUtils, Spring,


    TF4D.Core.CApi,
    TensorFlow.DApi,
    NumPy.NDArray;

type

np = record
    const
        _bool       : TF_DataType = TF_DataType.TF_BOOL;
        _char       : TF_DataType = TF_DataType.TF_INT8;
        _byte       : TF_DataType = TF_DataType.TF_INT8;
        _uint8      : TF_DataType = TF_DataType.TF_UINT8;
        _ubyte      : TF_DataType = TF_DataType.TF_UINT8;
        _int16      : TF_DataType = TF_DataType.TF_INT16;
        _uint16     : TF_DataType = TF_DataType.TF_UINT16;
        _int32      : TF_DataType = TF_DataType.TF_INT32;
        _uint32     : TF_DataType = TF_DataType.TF_UINT32;
        _int64      : TF_DataType = TF_DataType.TF_INT64;
        _uint64     : TF_DataType = TF_DataType.TF_UINT64;
        _float32    : TF_DataType = TF_DataType.TF_FLOAT;
        _float64    : TF_DataType = TF_DataType.TF_DOUBLE;
        _double     : TF_DataType = TF_DataType.TF_DOUBLE;
        _decimal    : TF_DataType = TF_DataType.TF_DOUBLE;
        _complex_   : TF_DataType = TF_DataType.TF_COMPLEX;
        _complex64  : TF_DataType = TF_DataType.TF_COMPLEX64;
        _complex128 : TF_DataType = TF_DataType.TF_COMPLEX128;

        class function np_array<T>(data: T): TNDArray;overload ;static;
        class function np_array<T>(data: TArray<T>): TNDArray;overload ;static;
end;

implementation

{ np }

class function np.np_array<T>(data: T): TNDArray;
begin
    var aData : TArray<T> := [data];
    Result := np_array<T>(aData)
end;

class function np.np_array<T>(data: TArray<T>): TNDArray;
begin
    var a := TValue.From< TArray<T> >(data);
    Result := TNDarray.create(a)
end;

end.
