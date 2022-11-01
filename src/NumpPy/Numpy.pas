unit Numpy;
{$REGION 'Licence'}
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
{$ENDREGION}

interface
    uses System.SysUtils,
         System.TypInfo,
         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,

         Numpy.Axis,
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

        // numPy.creation
        class function np_array<T>(data: T): TNDArray;overload ;static;
        class function np_array<T>(data: TArray<T>): TNDArray;overload ;static;
        class function arange<T>(_end: T): TNDArray;static;
        class function frombuffer(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType): TNDArray;static;
        // numPy.Math
        class function sum(x1: TNDArray; axis: PAxis = nil) : TNDArray;static;
        class function add(x: TNDArray; y: TNDArray): TNDArray;static;
        class function sqrt(x: TNDArray): TNDArray;static;
        class function sin(x: TNDArray): TNDArray;static;
        class function power(x: TNDArray;y: TNDArray): TNDArray;static;
        class function prod<T>(_array: TArray<T>): TNDArray;overload; static;
        class function prod(_array: TNDArray; axis: PAxis = nil; dtype : PTypeInfo= nil; keepdims: Boolean = false): TNDArray;overload; static;
        class function minimum(x1: TNDArray; x2: TNDArray) : TNDArray;static;
        class function maximum(x1: TNDArray; x2: TNDArray) : TNDArray;static;
        class function multiply(x1: TNDArray; x2: TNDArray): TNDArray;static;
        class function mean(x: TNDArray): TNDArray;static;
        class function log(x: TNDArray): TNDArray;static;
        class function floor(x: TNDArray): TNDArray;static;
        class function exp(x: TNDArray) : TNDArray;static;
        class function cos(x: TNDArray): TNDArray;static;
end;

NumPyImpl = class
   public
     function frombuffer(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType): TNDArray;
end;

implementation
        uses Tensorflow,
             Tensorflow.math_ops;

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

class function np.add(x, y: TNDArray): TNDArray;
begin
    Result := TNDArray.Create( math_ops.add(x, y) );
end;

class function np.arange<T>(_end: T): TNDArray;
begin
    var r : T := System.Default(T);
    var start := TValue.From<T>(r);
    var eEnd  := TValue.From<T>(_end);

    Result := TNDArray.Create (  tf.range( start, eEnd) );
end;

class function np.cos(x: TNDArray): TNDArray;
begin
    Result := TNDArray.Create( math_ops.cos(x) );
end;

class function np.exp(x: TNDArray): TNDArray;
begin
   Result := TNDArray.Create( tf.exp(x) );
end;

class function np.floor(x: TNDArray): TNDArray;
begin
   Result := TNDArray.Create( math_ops.floor(x) );
end;

class function np.frombuffer(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType): TNDArray;
begin
   Result := tf.numpy.frombuffer(bytes, shape, dtype);
end;

class function np.log(x: TNDArray): TNDArray;
begin
    Result := TNDArray.Create( tf.log(x) );
end;

class function np.maximum(x1, x2: TNDArray): TNDArray;
begin
   Result := TNDArray.Create( tf.maximum(x1, x2) );
end;

class function np.mean(x: TNDArray): TNDArray;
begin
    var a: TAxis := default(TAxis) ;
    Result := TNDArray.Create( math_ops.reduce_mean(x,a) );
end;

class function np.minimum(x1, x2: TNDArray): TNDArray;
begin
    Result := TNDArray.Create( tf.minimum(x1, x2) );
end;

class function np.multiply(x1, x2: TNDArray): TNDArray;
begin
   Result := TNDArray.Create( tf.multiply(x1, x2) );
end;

class function np.np_array<T>(data: TArray<T>): TNDArray;
begin
    var a := TValue.From< TArray<T> >(data);
    Result := TNDarray.create(a)
end;

class function np.power(x, y: TNDArray): TNDArray;
begin
    Result := TNDarray.create( tf.pow(x, y) );
end;

class function np.prod(_array: TNDArray; axis: PAxis; dtype: PTypeInfo; keepdims: Boolean): TNDArray;
begin
   Result := TNDarray.create( tf.reduce_prod(_array, axis) );
end;

class function np.prod<T>(_array: TArray<T>): TNDArray;
begin
     Result := TNDarray.create( tf.reduce_prod( TNDarray.create(_array) ) );
end;

class function np.sin(x: TNDArray): TNDArray;
begin
    Result := TNDarray.create( math_ops.sin(x) );
end;

class function np.sqrt(x: TNDArray): TNDArray;
begin
    Result :=  TNDarray.create( tf.sqrt(x) );
end;

class function np.sum(x1: TNDArray; axis: PAxis): TNDArray;
begin
    Result := TNDarray.create( tf.math.sum(x1, axis^) );
end;

{ NumPyImpl }

function NumPyImpl.frombuffer(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType): TNDArray;
begin
    Result := TNDArray.Create(bytes, shape, dtype);
end;

end.




