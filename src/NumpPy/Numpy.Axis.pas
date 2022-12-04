unit Numpy.Axis;
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
          Spring,

          TensorFlow.DApi,
          TensorFlow.Constant_op;

type
  PAxis = ^TAxis;
  TAxis = Record
    private
     function GetSize: Integer;
    function GetItem(indices: Integer): Integer;
    public
      isScalar : Boolean;
      axis     : Nullable< TArray<Integer> >;

      class operator Implicit(const value: TAxis): TValue;
      class operator Implicit(const value: Integer): TAxis;
      class operator Implicit(const aValue: TArray<Integer>): TAxis;
      class operator Implicit(const aValue: TAxis): TFTensor;
      class operator Implicit(const aValue: TAxis): TArray<Integer>;
      class operator Implicit(const aValue: TAxis): PAxis;

      property size                    : Integer read GetSize;
      property Item[indices: Integer ] : Integer read GetItem; default;
  End;

implementation

{ TAxis }

function TAxis.GetItem(indices: Integer): Integer;
begin
   Result := axis.Value[indices]
end;

function TAxis.GetSize: Integer;
begin
   if axis = nil then Result := -1
   else               Result := Length(axis.Value);

end;

class operator TAxis.Implicit(const value: TAxis): TValue;
begin
    Result := TValue.From<TAxis>(Value);
end;

class operator TAxis.Implicit(const aValue: TArray<Integer>): TAxis;
begin
    Result := Default(TAxis);
    Result.axis := aValue;
end;

class operator TAxis.Implicit(const aValue: TAxis): TFTensor;
begin
    Result := constant_op.constant(aValue)
end;

class operator TAxis.Implicit(const aValue: TAxis): PAxis;
begin
    if aValue.axis = nil then  Result := nil
    else                       Result := @aValue;

end;

class operator TAxis.Implicit(const aValue: TAxis): TArray<Integer>;
begin
    Result := aValue.axis
end;

class operator TAxis.Implicit(const value: Integer): TAxis;
begin
    Result.axis     := [value];
    Result.isScalar := true;
end;

end.
