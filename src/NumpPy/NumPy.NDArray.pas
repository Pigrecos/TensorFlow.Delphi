unit NumPy.NDArray;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
   uses System.SysUtils,
        System.TypInfo,
        system.Rtti,
        TF4D.Core.CApi,
        TensorFlow.DApiBase,
        TensorFlow.DApi,
        TensorFlow.Slice;

type
  NDArrayConverter = class
    private

    public
      class function Scalar<T>(nd: TNDArray)  : T; overload; static;
      class function Scalar<T>(input: Byte)   : T; overload; static;
      class function Scalar<T>(input: Single) : T; overload; static;
      class function Scalar<T>(input: Double) : T; overload; static;
      class function Scalar<T>(input: Integer): T; overload; static;
      class function Scalar<T>(input: Int64)  : T; overload; static;
  end;

  NDArray = record
    private
      FHandleNDArray : TNDArray;

       function  GetItem(indices: TArray<Integer>): NDArray; overload;
       function  GetItem(indices: TArray<Slice>): NDArray; overload;
       function  GetItem(indice: Integer): NDArray; overload;
       procedure SetItem(indices: TArray<Integer>; const Value: NDArray); overload;
       procedure SetItem(indices: TArray<Slice>; const Value: NDArray); overload;
       procedure SetItem(indice: Integer; const Value: NDArray); overload;

       function GetShape: TFShape;
    public
      class operator Implicit(t : TNDArray): NDArray;  static;
      class operator Implicit(t : NDArray): TNDArray;  static;
      class operator Implicit(t : NDArray): TFTensors;  static;

      class operator Implicit(t : NDArray): Boolean;   static;
      class operator Implicit(t : NDArray): Byte;      static;
      class operator Implicit(t : NDArray): Int32;     static;
      class operator Implicit(t : NDArray): Int64;     static;
      class operator Implicit(t : NDArray): Single;    static;
      class operator Implicit(t : NDArray): Double;    static;
      //
      class operator Implicit(v : Boolean): NDArray;   static;
      class operator Implicit(v : Byte):   NDArray;    static;
      class operator Implicit(v : Int32):  NDArray;    static;
      class operator Implicit(v : Int64):  NDArray;    static;
      class operator Implicit(v : Single): NDArray;    static;
      class operator Implicit(v : Double): NDArray;    static;
      //
      class operator Implicit(v : NDArray): string;    static;

      Class Operator Add(lhs: NDArray; rhs: NDArray)          : NDArray; Overload;
      Class Operator Subtract(lhs: NDArray; rhs: NDArray)     : NDArray; Overload;
      Class Operator Multiply(lhs: NDArray; rhs: NDArray)     : NDArray; Overload;
      Class Operator Divide(lhs: NDArray; rhs: NDArray)       : NDArray; Overload;
      Class Operator Modulus(lhs: NDArray; rhs: NDArray)      : NDArray; Overload;
      Class Operator GreaterThan(lhs: NDArray; rhs: NDArray)  : NDArray; Overload;
      Class Operator LessThan(lhs: NDArray; rhs: NDArray)     : NDArray; Overload;
      Class Operator Negative(lhs: NDArray)                   : NDArray; Overload;
      Class Operator Equal(lhs: NDArray; rhs: NDArray)        : NDArray; Overload;
      Class Operator NotEqual(lhs: NDArray; rhs: NDArray)     : NDArray; Overload;

      function  numpy: NDArray;

      property Item[indices: Integer ]         : NDArray read GetItem write SetItem; default;
      property Item[indices: TArray<Integer> ] : NDArray read GetItem write SetItem; default;
      property Item[slices: TArray<Slice> ]    : NDArray read GetItem write SetItem; default;

      property HandleNDArray  : TNDArray read FHandleNDArray;
      property Shape          : TFShape  read GetShape;
  end;

implementation
      uses TensorFlow.Operations,
           TensorFlow.Tensor,
           Tensorflow;

{ NDArray }

class operator NDArray.Implicit(t: NDArray): TNDArray;
begin
    Result := t.FHandleNDArray;
end;

class operator NDArray.Implicit(t: TNDArray): NDArray;
begin
    Result.FHandleNDArray := t;
end;

class operator NDArray.Implicit(t: NDArray): Boolean;
begin
    if t.FHandleNDArray.Dtype  = TF_DataType.TF_BOOL then
      Result := PBoolean(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Boolean>(t.FHandleNDArray);
end;

class operator NDArray.Implicit(t: NDArray): Byte;
begin
    if t.FHandleNDArray.Dtype = TF_DataType.TF_UINT8 then
      Result := PByte(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Byte>(t.FHandleNDArray);
end;

class operator NDArray.Implicit(t: NDArray): Int32;
begin
    if t.FHandleNDArray.Dtype = TF_DataType.TF_INT32 then
      Result := PInteger(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Integer>(t.FHandleNDArray);
end;

class operator NDArray.Implicit(t: NDArray): Int64;
begin
    if t.FHandleNDArray.Dtype = TF_DataType.TF_INT64 then
      Result := PInt64(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Int64>(t.FHandleNDArray);
end;

class operator NDArray.Implicit(t: NDArray): Single;
begin
    if t.FHandleNDArray.Dtype = TF_DataType.TF_FLOAT then
      Result := PSIngle(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Single>(t.FHandleNDArray);
end;

function NDArray.GetItem(indices: TArray<Integer>): NDArray;
begin
    Result := FHandleNDArray.Item[indices]
end;

function NDArray.GetItem(indices: TArray<Slice>): NDArray;
begin
     Result := FHandleNDArray.Item[indices]
end;

class operator NDArray.Add(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (TFTensor.BinaryOpWrapper('add', lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Subtract(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (TFTensor.BinaryOpWrapper('sub', lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Multiply(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (TFTensor.BinaryOpWrapper('mul', lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Divide(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (TFTensor.BinaryOpWrapper('div', lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Modulus(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (TFTensor.BinaryOpWrapper('mod', lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.GreaterThan(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (gen_math_ops.greater(lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.LessThan(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (gen_math_ops.less(lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Negative(lhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TNDArray.Create (gen_math_ops.neg(lhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.Equal(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    if lhs.FHandleNDArray = nil then
      Result := TNDArray.scalar(False)
    else if rhs.FHandleNDArray = nil then
      Result := TNDArray.scalar(False)
    else
      Result := TNDArray.Create(math_ops.equal(lhs, rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

class operator NDArray.NotEqual(lhs, rhs: NDArray): NDArray;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    if (lhs.FHandleNDArray = nil) or (rhs.FHandleNDArray = nil) then
      Result := True
    else
      Result := TNDArray.Create (math_ops.not_equal(lhs,rhs));

    if FChangedMode then
       tf.Context.restore_mode;
end;

function NDArray.numpy: NDArray;
begin
   Result := FHandleNDArray.numpy;
end;

function NDArray.GetItem(indice: Integer): NDArray;
begin
     Result := FHandleNDArray.Item[indice]
end;

function NDArray.GetShape: TFShape;
begin
     Result :=  FHandleNDArray.Shape;
end;

class operator NDArray.Implicit(t: NDArray): Double;
begin
    if t.FHandleNDArray.Dtype = TF_DataType.TF_DOUBLE then
      Result := PDouble(t.FHandleNDArray.data)^
    else
      Result := NDArrayConverter.Scalar<Double>(t.FHandleNDArray);
end;

procedure NDArray.SetItem(indices: TArray<Integer>; const Value: NDArray);
begin
    FHandleNDArray.Item[indices] := Value
end;

procedure NDArray.SetItem(indices: TArray<Slice>; const Value: NDArray);
begin
    FHandleNDArray.Item[indices] := Value
end;

procedure NDArray.SetItem(indice: Integer; const Value: NDArray);
begin
    FHandleNDArray.Item[indice] := Value
end;

class operator NDArray.Implicit(v: Boolean): NDArray;
begin
    Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: Byte): NDArray;
begin
    Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: Int32): NDArray;
begin
    Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: Int64): NDArray;
begin
    Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: Single): NDArray;
begin
    Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: Double): NDArray;
begin
   Result := TNDArray.Create(v)
end;

class operator NDArray.Implicit(v: NDArray): string;
begin
    var t : TTensor := v.FHandleNDArray;
    Result := string(t)
end;

class operator NDArray.Implicit(t: NDArray): TFTensors;
var
  FChangedMode : Boolean;
begin
    FChangedMode := False;
    if not tf.executing_eagerly then
    begin
        tf.Context.eager_mode;
        FchangedMode := true;
    end;

    Result := TFTensors.Create(t.FHandleNDArray);

    if FChangedMode then
       tf.Context.restore_mode;
end;

{ NDArrayConverter }

class function NDArrayConverter.Scalar<T>(nd: TNDArray): T;
begin
    case nd.Dtype of
      TF_UINT8: Result := Scalar<T>( PByte(nd.data)^ );
      TF_FLOAT: Result := Scalar<T>( Single((nd.data)^) );
      TF_DOUBLE:Result := Scalar<T>( Double(PDouble(nd.data)^) );
      TF_INT32: Result := Scalar<T>( PInteger(nd.data)^ );
      TF_INT64: Result := Scalar<T>( PInt64(nd.data)^ );
    else
      raise TFException.Create('Not Implemented');
    end;
end;

class function NDArrayConverter.Scalar<T>(input: Byte): T;
begin
    var tipo := PTypeInfo(TypeInfo(T));

    if tipo = PTypeInfo(TypeInfo(Byte)) then
    begin
       var a : Byte := input;
       var v := TValue.From<Byte>(a);
       Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Single)) then
    begin
        var a : Single := input;
        var v := TValue.From<Single>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int32)) then
    begin
        var a : Int32 := input;
        var v := TValue.From<Int32>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int64)) then
    begin
        var a : Int64 := input;
        var v := TValue.From<Int64>(a);
        Result := v.AsType<T>;
    end else
    begin
       raise TFException.Create('Not Implemented');
    end;
end;

class function NDArrayConverter.Scalar<T>(input: Single): T;
begin
    var tipo := PTypeInfo(TypeInfo(T));

    if tipo = PTypeInfo(TypeInfo(Byte)) then
    begin
       var a : Byte := Trunc(input);
       var v := TValue.From<Byte>(a);
       Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Single)) then
    begin
        var a : Single := input;
        var v := TValue.From<Single>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Double)) then
    begin
        var a : Double := input;
        var v := TValue.From<Double>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int32)) then
    begin
        var a : Int32 := Trunc(input);
        var v := TValue.From<Int32>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int64)) then
    begin
        var a : Int64 := Trunc(input);
        var v := TValue.From<Int64>(a);;
        Result := v.AsType<T>;
    end else
    begin
       raise TFException.Create('Not Implemented');
    end;
end;

class function NDArrayConverter.Scalar<T>(input: Double): T;
begin
    var tipo := PTypeInfo(TypeInfo(T));

    if tipo = PTypeInfo(TypeInfo(Byte)) then
    begin
       var a : Byte := Trunc(input);
       var v := TValue.From<Byte>(a);
       Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Single)) then
    begin
        var a : Single := input;
        var v := TValue.From<Single>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Double)) then
    begin
        var a : Double := input;
        var v := TValue.From<Double>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int32)) then
    begin
        var a : Int32 := Trunc(input);
        var v := TValue.From<Int32>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int64)) then
    begin
        var a : Int64 := Trunc(input);
        var v := TValue.From<Int64>(a);;
        Result := v.AsType<T>;
    end else
    begin
       raise TFException.Create('Not Implemented');
    end;
end;


class function NDArrayConverter.Scalar<T>(input: Integer): T;
begin
    var tipo := PTypeInfo(TypeInfo(T));

    if tipo = PTypeInfo(TypeInfo(Byte)) then
    begin
       var a : Byte := input;
       var v := TValue.From<Byte>(a);
       Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Single)) then
    begin
        var a : Single := input;
        var v := TValue.From<Single>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int32)) then
    begin
        var a : Int32 := input;
        var v := TValue.From<Int32>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int64)) then
    begin
        var a : Int64 := input;
        var v := TValue.From<Int64>(a);
        Result := v.AsType<T>;
    end else
    begin
       raise TFException.Create('Not Implemented');
    end;
end;

class function NDArrayConverter.Scalar<T>(input: Int64): T;
begin
    var tipo := PTypeInfo(TypeInfo(T));

    if tipo = PTypeInfo(TypeInfo(Byte)) then
    begin
       var a : Byte := input;
       var v := TValue.From<Byte>(a);
       Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Single)) then
    begin
        var a : Single := input;
        var v := TValue.From<Single>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int32)) then
    begin
        var a : Int32 := input;
        var v := TValue.From<Int32>(a);
        Result := v.AsType<T>;
    end
    else if tipo = PTypeInfo(TypeInfo(Int64)) then
    begin
        var a : Int64 := input;
        var v := TValue.From<Int64>(a);
        Result := v.AsType<T>;
    end else
    begin
       raise TFException.Create('Not Implemented');
    end;
end;

end.

