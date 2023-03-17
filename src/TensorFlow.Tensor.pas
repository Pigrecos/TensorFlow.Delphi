unit TensorFlow.Tensor;
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
    uses
       System.SysUtils, System.Classes, System.Types, Generics.Collections, Winapi.Windows,
       System.AnsiStrings, System.Rtti, Spring,
       TF4D.Core.CApi,

       TensorFlow.DApi,
       TensorFlow.Variable,
       NumPy.NDArray;

type
TTensor = record
  private
      FHandleTensor : TFTensor;

      class procedure EnsureScalar(t: TTensor);static;
      class procedure EnsureDType(t: TTensor; _is: TF_DataType); static;
      function GetShape: TFShape;
      function GetTipo: TF_DataType;

  public
      class operator Implicit(t : TFTensor): TTensor;
      class operator Implicit(t : TTensor): TFTensor;
      class operator Implicit(t : TTensor): TFTensors;
      class operator Implicit(t : TFTensors): TTensor;

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
      constructor Create(const value: TF_TString); overload;

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
      class operator Explicit(t : TTensor): string;

      class function ToStringArray(t: TTensor): TArray<AnsiString>; static;
      //
      // Class Operator
      Class Operator Add(lhs: TTensor; rhs: ResourceVariable) : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator Add(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator Add(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator Add(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator Add(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator Add(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator Add(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator Add(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator Add(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator Add(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator Add(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Add(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator Add(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator Subtract(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator Subtract(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator Subtract(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator Subtract(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator Subtract(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator Subtract(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator Subtract(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator Subtract(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator Subtract(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator Subtract(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator Subtract(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Subtract(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator Subtract(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator Multiply(lhs: TTensor; rhs: ResourceVariable) : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator Multiply(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator Multiply(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator Multiply(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator Multiply(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator Multiply(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator Multiply(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator Multiply(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator Multiply(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator Multiply(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator Multiply(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Multiply(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator Multiply(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator Divide(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator Divide(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator Divide(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator Divide(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator Divide(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator Divide(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator Divide(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator Divide(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator Divide(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator Divide(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator Divide(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Divide(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator Divide(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator Modulus(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator Modulus(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator Modulus(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator Modulus(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator Modulus(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator Modulus(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator Modulus(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator Modulus(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator Modulus(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator Modulus(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator Modulus(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator Modulus(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator Modulus(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator GreaterThan(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator GreaterThan(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator GreaterThan(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator GreaterThan(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator GreaterThan(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator GreaterThan(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator GreaterThan(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator GreaterThan(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator GreaterThan(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator GreaterThan(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThan(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator GreaterThan(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator LessThan(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator LessThan(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator LessThan(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator LessThan(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator LessThan(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator LessThan(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator LessThan(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator LessThan(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator LessThan(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator LessThan(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator LessThan(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThan(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator LessThan(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator GreaterThanOrEqual(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      //
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: TNDArray)         : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TNDArray;rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Int8)             : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Int8;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Byte)             : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Byte;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Int16)            : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Int16;   rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Word)             : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Word;    rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Integer)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Integer; rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: UInt32)           : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: UInt32;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: UInt64)           : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: UInt64;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Int64)            : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Int64;   rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Single)           : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Single;  rhs: TTensor)          : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: TTensor; rhs: Double)           : TTensor; Overload;
      Class Operator LessThanOrEqual(lhs: Double;  rhs: TTensor)          : TTensor; Overload;
      Class Operator negative(x: TTensor): TTensor;

      function  eval(session : TFSession; feed_dict : TArray<FeedItem>= nil) : TNDArray;
      function  numpy: NDArray;
      function  ToArray<T>:TArray<T>;

      // Property
      property HTensor : TFTensor read FHandleTensor;
      property Shape   : TFShape  read GetShape;
      property dtype   : TF_DataType read GetTipo;
end;

implementation
         uses Tensorflow,
              TensorFlow.Ops,
              Tensorflow.Utils,
              TensorFlow.gen_math_ops;

{ TTensor }

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

constructor TTensor.Create(const value: TF_TString);
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
       raise Exception.Create(Format('Unable to cast scalar tensor %s to %s',[Tdtypes.ToString(t.dtype), Tdtypes.ToString(_is)]));
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

function TTensor.eval(session: TFSession; feed_dict: TArray<FeedItem>): TNDArray;
begin
    Result := FHandleTensor.eval(session,feed_dict);
end;

class operator TTensor.Implicit(t: TTensor): TFTensors;
begin
    Result := TFTensors.Create(t.FHandleTensor);
end;

class operator TTensor.Implicit(t: TFTensors): TTensor;
begin
    Result := t.First;
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

class operator TTensor.Explicit(t: TTensor): string;
begin
    EnsureScalar(t);
    EnsureDType(t, TF_STRING);
    Result := t.FHandleTensor.StringData(0);
end;

function TTensor.GetShape: TFShape;
begin
    Result :=  FHandleTensor.Shape;
end;

function TTensor.GetTipo: TF_DataType;
begin
    Result :=  FHandleTensor.Dtype;
end;

class function TTensor.ToStringArray(t: TTensor): TArray<AnsiString>;
begin
    Result := t.FHandleTensor.StringData;
end;

// Add
Class Operator TTensor.Add(lhs: TTensor; rhs: ResourceVariable): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Word): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Single): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: TTensor; rhs: Double): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;

Class Operator TTensor.Add(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('add', lhs, rhs);
end;
// subtract
Class Operator TTensor.Subtract(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Word): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Single): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: TTensor; rhs: Double): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;

Class Operator TTensor.Subtract(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('sub', lhs, rhs);
end;
// mul
Class Operator TTensor.Multiply(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Word): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Single): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: TTensor; rhs: Double): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

Class Operator TTensor.Multiply(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;
class operator TTensor.Multiply(lhs: TTensor; rhs: ResourceVariable): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mul', lhs, rhs);
end;

class operator TTensor.negative(x: TTensor): TTensor;
begin
   Result := gen_math_ops.neg(x);
end;

function TTensor.ToArray<T>: TArray<T>;
begin
    Result := FHandleTensor.ToArray<T>;
end;

function TTensor.numpy: NDArray;
begin
   Result := FHandleTensor.numpy;
end;

// Div
Class Operator TTensor.Divide(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('truediv', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Word): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Single): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: TTensor; rhs: Double): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;

Class Operator TTensor.Divide(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('div', lhs, rhs);
end;
// Mod
Class Operator TTensor.Modulus(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Word): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Single): TTensor;
begin
    Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: TTensor; rhs: Double): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;

Class Operator TTensor.Modulus(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result :=  TFTensor.BinaryOpWrapper('mod', lhs, rhs);
end;
//// GreaterThan
Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Word): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Single): TTensor;
begin
    Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: TTensor; rhs: Double): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;

Class Operator TTensor.GreaterThan(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater(lhs, rhs);
end;
// LessThan
Class Operator TTensor.LessThan(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Word): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Single): TTensor;
begin
    Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: TTensor; rhs: Double): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;

Class Operator TTensor.LessThan(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less(lhs, rhs);
end;
// GreaterThanOrEqual
Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Word): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Single): TTensor;
begin
    Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: TTensor; rhs: Double): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;

Class Operator TTensor.GreaterThanOrEqual(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.greater_equal(lhs, rhs);
end;
// LessThanOrEqual
Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: TNDArray): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TNDArray;rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Int8): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Int8;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Byte): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Byte;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Int16): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Int16;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Word): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Word;    rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Integer): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Integer; rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: UInt32): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: UInt32;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: UInt64): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: UInt64;  rhs: TTensor): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Int64): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Int64;   rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Single): TTensor;
begin
    Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Single;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: TTensor; rhs: Double): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

Class Operator TTensor.LessThanOrEqual(lhs: Double;  rhs: TTensor): TTensor;
begin
   Result := gen_math_ops.less_equal(lhs, rhs);
end;

end.

