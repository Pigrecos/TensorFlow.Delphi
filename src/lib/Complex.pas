unit Complex;
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

uses
  System.SysUtils, System.Math;

type

  TPolar = record
  public
    Radius, Theta: Double;

    constructor Create(ARadius, ATheta: Double);

    function Real: Double;
    function Imag: Double;

    function Ln: TPolar;
    function Exp: TPolar;

  end;

  TComplex = record
  public
    Real, Imag: Double;

    constructor Create(AReal, AImag: Double);

    class operator Implicit(A: Double): TComplex;
    class operator Implicit(A: TPolar): TComplex;
    class operator Implicit(A: TComplex): TPolar;

    class operator Add(A, B: TComplex): TComplex;
    class operator Subtract(A, B: TComplex): TComplex;
    class operator Multiply(A, B: TComplex): TComplex;
    class operator Divide(A, B: TComplex): TComplex;
    class operator Positive(A: TComplex): TComplex;
    class operator Negative(A: TComplex): TComplex;

    class operator Equal(A, B: TComplex): Boolean;
    class operator NotEqual(A, B: TComplex): Boolean;

    function AbsSqr: Double;
    function Abs: Double;

    function Inverse: TComplex;
    function Ln: TComplex;
    function Exp: TComplex;
    function Sqr: TComplex;
    function Sqrt: TComplex;
    function Power(A: TComplex): TComplex;

    function Radius: Double;
    function Theta: Double;
    function AsPolar: TPolar;

    function ToString: string;

  end;

  TPolarHelper = record helper for TPolar
    function AsComplex: TComplex;

  end;

implementation

function PrettyFloat(AValue: Single): string; overload;
begin
  Result := AValue.ToString(ffGeneral, 7, 0, TFormatSettings.Invariant)
end;

function PrettyFloat(AValue: Double): string; overload;
begin
  Result := AValue.ToString(ffGeneral, 15, 0, TFormatSettings.Invariant)
end;

{ TComplex }

constructor TComplex.Create(AReal, AImag: Double);
begin
  Real := AReal;
  Imag := AImag;
end;

class operator TComplex.Implicit(A: Double): TComplex;
begin
  Result.Create(A, 0);
end;

class operator TComplex.Implicit(A: TPolar): TComplex;
begin
  Result := A.AsComplex;
end;

class operator TComplex.Implicit(A: TComplex): TPolar;
begin
  Result := A.AsPolar;
end;

class operator TComplex.Add(A, B: TComplex): TComplex;
begin
  Result.Create(A.Real + B.Real, A.Imag + B.Imag);
end;

class operator TComplex.Subtract(A, B: TComplex): TComplex;
begin
  Result.Create(A.Real - B.Real, A.Imag - B.Imag);
end;

class operator TComplex.Multiply(A, B: TComplex): TComplex;
begin
  Result.Create(
    A.Real * B.Real - A.Imag * B.Imag,
    A.Real * B.Imag + A.Imag * B.Real
    );
end;

class operator TComplex.Divide(A, B: TComplex): TComplex;
begin
  Result := A * B.Inverse;
end;

class operator TComplex.Equal(A, B: TComplex): Boolean;
begin
  Result := (A.Real = B.Real) and (A.Imag = B.Imag);
end;

class operator TComplex.Negative(A: TComplex): TComplex;
begin
  Result.Create(-A.Real, -A.Imag);
end;

class operator TComplex.NotEqual(A, B: TComplex): Boolean;
begin
  Result := (A.Real <> B.Real) or (A.Imag <> B.Imag);
end;

function TComplex.AbsSqr: Double;
begin
  Result := System.Sqr(Real) + System.Sqr(Imag);
end;

function TComplex.Abs: Double;
begin
  Result := System.Sqrt(AbsSqr);
end;

function TComplex.Inverse: TComplex;
var
  Denominator: Double;
begin
  Denominator := Radius;
  Result.Create(Real / Denominator, -Imag / Denominator);
end;

function TComplex.Ln: TComplex;
begin
  Result := AsPolar.Ln;
end;

function TComplex.Exp: TComplex;
begin
  Result := AsPolar.Exp;
end;

function TComplex.Sqr: TComplex;
begin
  Result.Create(System.Sqr(Real) - System.Sqr(Imag), 2 * Real * Imag);
end;

function TComplex.Sqrt: TComplex;
var
  LValue: Double;
begin
  if Real > 0 then
  begin
    LValue := Abs + Real;
    Result.Create(System.Sqrt(LValue / 2), Imag / System.Sqrt(LValue * 2));
  end
  else
  begin
    LValue := Abs - Real;
    if Imag < 0 then
      Result.Create(System.Abs(Imag) / System.Sqrt(LValue * 2), -System.Sqrt(LValue / 2))
    else
      Result.Create(System.Abs(Imag) / System.Sqrt(LValue * 2), System.Sqrt(LValue / 2));
  end;
end;

class operator TComplex.Positive(A: TComplex): TComplex;
begin
  Result := A;
end;

function TComplex.Power(A: TComplex): TComplex;
begin
  Result := (Ln * A).Exp;
end;

function TComplex.Radius: Double;
begin
  Result := System.Sqrt(System.Sqr(Real) + System.Sqr(Imag));
end;

function TComplex.Theta: Double;
begin
  Result := ArcTan2(Imag, Real);
end;

function TComplex.AsPolar: TPolar;
begin
  Result.Create(Radius, Theta);
end;

function TComplex.ToString: string;
begin
  if Imag = 0 then
    Exit(PrettyFloat(Real));
  if Real = 0 then
    Exit(PrettyFloat(Imag));
  Result := Format('(%s + %si)', [PrettyFloat(Real), PrettyFloat(Imag)]);
end;

{ TPolar }

constructor TPolar.Create(ARadius, ATheta: Double);
begin
  Radius := ARadius;
  Theta := ATheta;
end;

function TPolar.Real: Double;
begin
  Result := Radius * Cos(Theta);
end;

function TPolar.Exp: TPolar;
begin
  Result.Create(System.Exp(Radius), Theta);
end;

function TPolar.Imag: Double;
begin
  Result := Radius * Sin(Theta);
end;

function TPolar.Ln: TPolar;
begin
  Result.Create(System.Ln(Radius), Theta);
end;

{ TPolarHelper }

function TPolarHelper.AsComplex: TComplex;
begin
  Result.Create(Real, Imag);
end;

end.
