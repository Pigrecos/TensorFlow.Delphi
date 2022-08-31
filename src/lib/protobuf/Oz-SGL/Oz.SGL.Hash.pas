(* Standard Generic Library (SGL) for Pascal
  * Copyright (c) 2020 Marat Shaimardanov
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *      http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
*)

unit Oz.SGL.Hash;

interface

uses
  System.SysUtils, System.Math, System.TypInfo, System.Variants, Oz.SGL.Heap;

{$T+}

{$Region 'THashData'}

type
  THashKind = (hkMultiplicative, hkSHA1, hkSHA2, hkSHA5, hkMD5);

  TsgHash  = record
  type
    TUpdateProc = procedure(const key: PByte; Size: Cardinal);
    THashProc = function(const key: PByte; Size: Cardinal): Cardinal;
  private
    FUpdate: TUpdateProc;
    FHash: THashProc;
  public
    class function From(kind: THashKind): TsgHash; static;
    procedure Reset(kind: THashKind);

    class function HashMultiplicative(const key: PByte; Size: Cardinal): Cardinal; static;
    class function ELFHash(const digest: Cardinal; const key: PByte;
      const Size: Integer): Cardinal; static;

    // Update the Hash with the provided bytes
    procedure Update(const key; Size: Cardinal); overload;
    procedure Update(const key: TBytes; Size: Cardinal = 0); overload; inline;
    procedure Update(const key: string); overload; inline;
    // Hash function
    property Hash: THashProc read FHash;
  end;

{$EndRegion}

{$Region 'TsgHasher: GetHash and Equals operation'}

  PComparer = ^TComparer;
  TComparer = record
    Equals: TEqualsFunc;
    Hash: THashProc;
  end;

  PsgHasher = ^TsgHasher;
  TsgHasher = record
  private
    FComparer: PComparer;
  public
    class function From(m: PsgItemMeta): TsgHasher; overload; static;
    class function From(const Comparer: TComparer): TsgHasher; overload; static;
    function Equals(a, b: Pointer): Boolean;
    function GetHash(k: Pointer): Integer;
  end;

{$EndRegion}

function CompareRawByteString(const a, b: RawByteString): Integer;

implementation

type
  PTabInfo = ^TTabInfo;
  TTabInfo = record
    Selector: Boolean;
    Data: Pointer;
  end;

  TSelectProc = function(info: PTypeInfo; size: Integer): PComparer;

function CompareRawByteString(const a, b: RawByteString): Integer;
var
  L, La, Lb: Integer;
  Pa, Pb: PByte;
begin
  if Pointer(a) = Pointer(b) then
    Result := 0
  else if Pointer(a) = nil then
    Result := 0 - PInteger(PByte(b) - 4)^ // Length(b)
  else if Pointer(b) = nil then
    Result := PInteger(PByte(a) - 4)^ // Length(a)
  else
  begin
    Result := Integer(PByte(a)^) - Integer(PByte(b)^);
    if Result <> 0 then
      Exit;
    La := PInteger(PByte(a) - 4)^ - 1;  // Length(a);
    Lb := PInteger(PByte(b) - 4)^ - 1; // Length(b);
    L := La;
    if L > Lb then L := Lb;
    Pa := PByte(a) + 1;
    Pb := PByte(b) + 1;
    while L > 0 do
    begin
      Result := Integer(Pa^) - Integer(Pb^);
      if Result <> 0 then
        Exit;
      if L = 1 then break;
      Result := Integer(Pa[1]) - Integer(Pb[1]);
      if Result <> 0 then
        Exit;
      Inc(Pa, 2);
      Inc(Pb, 2);
      Dec(L, 2);
    end;
    Result := La - Lb;
  end;
end;

function EqualsByte(a, b: Pointer): Boolean;
begin
  Result := PByte(a)^ = PByte(b)^;
end;

function EqualsInt16(a, b: Pointer): Boolean;
begin
  Result := PWord(a)^ = PWord(b)^;
end;

function EqualsInt32(a, b: Pointer): Boolean;
begin
  Result := PInteger(a)^ = PInteger(b)^;
end;

function EqualsInt64(a, b: Pointer): Boolean;
begin
  Result := PInt64(a)^ = PInt64(b)^;
end;

function EqualsSingle(a, b: Pointer): Boolean;
begin
  Result := PSingle(a)^ = PSingle(b)^;
end;

function EqualsDouble(a, b: Pointer): Boolean;
begin
  Result := PDouble(a)^ = PDouble(b)^;
end;

function EqualsCurrency(a, b: Pointer): Boolean;
begin
  Result := PCurrency(a)^ = PCurrency(b)^;
end;

function EqualsComp(a, b: Pointer): Boolean;
begin
  Result := PComp(a)^ = PComp(b)^;
end;

function EqualsExtended(a, b: Pointer): Boolean;
begin
  Result := Extended(a^) = Extended(b^);
end;

function EqualsString(a, b: Pointer): Boolean;
begin
  Result := AnsiString(a^) = AnsiString(b^);
end;

function EqualsClass(a, b: Pointer): Boolean;
begin
  if TObject(a^) <> nil then
    Result := TObject(a^).Equals(TObject(b^))
  else if TObject(b^) <> nil then
    Result := TObject(b^).Equals(TObject(a^))
  else
    Result := True;
end;

function EqualsMethod(a, b: Pointer): Boolean;
begin
  Result := TMethod(a^) = TMethod(b^);
end;

function EqualsLString(a, b: Pointer): Boolean;
begin
  Result := CompareRawByteString(RawByteString(a^), RawByteString(b^)) = 0;
end;

function EqualsWString(a, b: Pointer): Boolean;
begin
  Result := WideString(a^) = WideString(b^);
end;

function EqualsUString(a, b: Pointer): Boolean;
begin
  Result := UnicodeString(a^) = UnicodeString(b^);
end;

function EqualsShortString(a, b: Pointer): Boolean;
begin
  Result := ShortString(a^) = ShortString(b^);
end;

function EqualsVariant(a, b: Pointer): Boolean;
var
  l, r: Variant;
begin
  l := PVariant(a)^;
  r := PVariant(b)^;
  Result := VarCompareValue(l, r) = vrEqual;
end;

function EqualsRecord(a, b: Pointer): Boolean;
begin
  Result := False;
end;

function EqualsPointer(a, b: Pointer): Boolean;
begin
  Result := NativeUInt(a^) = NativeUInt(b^);
end;

function EqualsI8(a, b: Pointer): Boolean;
begin
  Result := Int64(a^) = Int64(b^);
end;

function HashByte(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Byte));
end;

function HashInt16(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Word));
end;

function HashInt32(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Int32));
end;

function HashInt64(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Int64));
end;

function HashSingle(const key: PByte): Cardinal;
var
  m: Extended;
  e: Integer;
begin
  Frexp(PSingle(key)^, m, e);
  if m = 0 then
    m := Abs(m);
  Result := TsgHash.ELFHash(0, key, sizeof(Extended));
  Result := TsgHash.ELFHash(Result, key, sizeof(Integer));
end;

function HashDouble(const key: PByte): Cardinal;
var
  m: Extended;
  e: Integer;
begin
  Frexp(PDouble(key)^, m, e);
  if m = 0 then
    m := Abs(m);
  Result := TsgHash.ELFHash(0, key, sizeof(Extended));
  Result := TsgHash.ELFHash(Result, key, sizeof(Integer));
end;

function HashExtended(const key: PByte): Cardinal;
var
  m: Extended;
  e: Integer;
begin
  Frexp(PExtended(key)^, m, e);
  if m = 0 then
    m := Abs(m);
  Result := TsgHash.ELFHash(0, key, sizeof(Extended));
  Result := TsgHash.ELFHash(Result, key, sizeof(Integer));
end;

function HashComp(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Comp));
end;

function HashCurrency(const key: PByte): Cardinal;
begin
  Result := TsgHash.ELFHash(0, key, sizeof(Currency));
end;

function HashString(const key: PByte): Cardinal;
var
  s: string;
begin
  s := PString(key)^;
  Result := TsgHash.HashMultiplicative(key, Length(s));
end;

function HashClass(const key: PByte): Cardinal;
begin
  if TObject(key^) = nil then
    Result := 63
  else
    Result := TObject(key^).GetHashCode;
end;

type
  TMethodPointer = procedure of object;

function HashMethod(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(key, SizeOf(TMethodPointer));
end;

function HashLString(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(PByte(@PAnsiString(key)^[1]),
    Length(PAnsiString(key)^) * SizeOf(PAnsiString(key)^[1]));
end;

function HashWString(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(PByte(@PWideString(key)^[1]),
    Length(PWideString(key)^) * SizeOf(PWideString(key)^[1]));
end;

function HashUString(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(PByte(@PUnicodeString(key)^[1]),
    Length(PUnicodeString(key)^) * SizeOf(PUnicodeString(key)^[1]));
end;

function HashShortString(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(PByte(@PShortString(key)^[1]),
    Length(PShortString(key)^));
end;

function HashVariant(const key: PByte): Cardinal;
var
  v: string;
begin
  try
    v := PVariant(key)^;
    Result := HashUString(PByte(PChar(v)));
  except
    Result := TsgHash.HashMultiplicative(key, SizeOf(Variant));
  end;
end;

function HashPointer(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(key, sizeof(Pointer));
end;

function HashI8(const key: PByte): Cardinal;
begin
  Result := TsgHash.HashMultiplicative(key, sizeof(Int64));
end;

const
  // Integer
  EntryByte: TComparer = (Equals: EqualsByte; Hash: HashByte);
  EntryInt16: TComparer = (Equals: EqualsInt16; Hash: HashInt16);
  EntryInt32: TComparer = (Equals: EqualsInt32; Hash: HashInt32);
  EntryInt64: TComparer = (Equals: EqualsInt64; Hash: HashInt64);
  // Real
  EntryR4: TComparer = (Equals: EqualsSingle; Hash: HashSingle);
  EntryR8: TComparer = (Equals: EqualsDouble; Hash: HashDouble);
  EntryR10: TComparer = (Equals: EqualsExtended; Hash: HashExtended);
  EntryRI8: TComparer = (Equals: EqualsComp; Hash: HashComp);
  EntryRC8: TComparer = (Equals: EqualsCurrency; Hash: HashCurrency);
  // String
  EntryAnsiString: TComparer = (Equals: EqualsString; Hash: HashString);
  EntryLString: TComparer = (Equals: EqualsLString; Hash: HashLString);
  EntryWString: TComparer = (Equals: EqualsWString; Hash: HashWString);
  EntryUString: TComparer = (Equals: EqualsUString; Hash: HashUString);
  EntryShortString: TComparer = (Equals: EqualsShortString; Hash: HashShortString);

  EntryClass: TComparer = (Equals: EqualsClass; Hash: HashClass);
  EntryMethod: TComparer = (Equals: EqualsMethod; Hash: HashMethod);
  EntryVariant: TComparer = (Equals: EqualsVariant; Hash: HashVariant);
  EntryPointer: TComparer = (Equals: EqualsPointer; Hash: HashPointer);
  EntryI8: TComparer = (Equals: EqualsI8; Hash: HashI8);

function SelectBinary(info: PTypeInfo; size: Integer): PComparer;
begin
  case size of
    1: Result := @EntryByte;
    2: Result := @EntryInt16;
    4: Result := @EntryInt32;
    8: Result := @EntryInt64;
    else
    begin
      System.Error(reRangeError);
      exit(nil);
    end;
  end;
end;

function SelectInteger(info: PTypeInfo; size: Integer): PComparer;
begin
  case GetTypeData(info)^.OrdType of
    otSByte, otUByte: Result := @EntryByte;
    otSWord, otUWord: Result := @EntryInt16;
    otSLong, otULong: Result := @EntryInt32;
  else
    System.Error(reRangeError);
    exit(nil);
  end;
end;

function SelectFloat(info: PTypeInfo; size: Integer): PComparer;
begin
  case GetTypeData(info)^.FloatType of
    ftSingle: Result := @EntryR4;
    ftDouble: Result := @EntryR8;
    ftExtended: Result := @EntryR10;
    ftComp: Result := @EntryRI8;
    ftCurr: Result := @EntryRC8;
  else
    System.Error(reRangeError);
    exit(nil);
  end;
end;

const
  VTab: array [TTypeKind] of TTabInfo = (
    // tkUnknown
    (Selector: True; Data: @SelectBinary),
    // tkInteger
    (Selector: True; Data: @SelectInteger),
    // tkChar
    (Selector: True; Data: @SelectBinary),
    // tkEnumeration
    (Selector: True; Data: @SelectInteger),
    // tkFloat
    (Selector: True; Data: @SelectFloat),
    // tkString
    (Selector: False; Data: @EntryShortString),
    // tkSet
    (Selector: True; Data: @SelectBinary),
    // tkClass
    (Selector: False; Data: @EntryClass),
    // tkMethod
    (Selector: False; Data: @EntryMethod),
    // tkWChar
    (Selector: True; Data: @SelectBinary),
    // tkLString
    (Selector: False; Data: @EntryLString),
    // tkWString
    (Selector: False; Data: @EntryWString),
    // tkVariant
    (Selector: False; Data: @EntryVariant),
    // tkArray
    (Selector: True; Data: @SelectBinary),
    // tkRecord
    (Selector: False; Data: nil),
    // tkInterface
    (Selector: False; Data: @EntryPointer),
    // tkInt64
    (Selector: False; Data: @EntryI8),
    // tkDynArray
    (Selector: False; Data: nil),
    // tkUString
    (Selector: False; Data: @EntryUString),
    // tkClassRef
    (Selector: False; Data: @EntryPointer),
    // tkPointer
    (Selector: False; Data: @EntryPointer),
    // tkProcedure
    (Selector: False; Data: @EntryPointer),
    // tkMRecord
    (Selector: True; Data: nil));

{$Region 'TsgHash'}

class function TsgHash.From(kind: THashKind): TsgHash;
begin
  Result.Reset(kind);
end;

procedure TsgHash.Reset(kind: THashKind);
begin
  case kind of
    THashKind.hkMultiplicative:
      begin
        FHash := TsgHash.HashMultiplicative;
      end;
  end;
end;

procedure TsgHash.Update(const key; Size: Cardinal);
begin
  FUpdate(PByte(@key), Size);
end;

procedure TsgHash.Update(const key: TBytes; Size: Cardinal);
var
  L: Cardinal;
begin
  L := Size;
  if L = 0 then
    L := Length(key);
  FUpdate(PByte(key), L);
end;

procedure TsgHash.Update(const key: string);
begin
  Update(TEncoding.UTF8.GetBytes(key));
end;

class function TsgHash.HashMultiplicative(const key: PByte;
  Size: Cardinal): Cardinal;
var
  i, hash: Cardinal;
  p: PByte;
begin
  hash := 5381;
  p := key;
  for i := 1 to Size do
  begin
    hash := 33 * hash + p^;
    Inc(p);
  end;
  Result := hash;
end;

class function TsgHash.ELFHash(const digest: Cardinal; const key: PByte;
  const Size: Integer): Cardinal;
var
  i: Integer;
  p: PByte;
  t: Cardinal;
begin
  Result := digest;
  p := key;
  for i := 1 to Size do
  begin
    Result := (Result shl 4) + p^;
    Inc(p);
    t := Result and $F0000000;
    if t <> 0 then
      Result := Result xor (t shr 24);
    Result := Result and (not t);
  end;
end;

{$EndRegion}

{$Region 'TsgHasher'}

class function TsgHasher.From(m: PsgItemMeta): TsgHasher;
var
  info: PTabInfo;
begin
  if m.TypeInfo = nil then
    raise EsgError.Create('Invalid parameter');
  info := @VTab[PTypeInfo(m.TypeInfo)^.Kind];
  if info^.Selector then
    Result.FComparer := TSelectProc(info^.Data)(m.TypeInfo, m.ItemSize)
  else if info^.Data <> nil then
    Result.FComparer := PComparer(info^.Data)
  else
    raise EsgError.Create('TsgHasher: Type is not supported');
end;

class function TsgHasher.From(const Comparer: TComparer): TsgHasher;
begin
  Result.FComparer := @Comparer;
end;

function TsgHasher.Equals(a, b: Pointer): Boolean;
begin
  Result := PComparer(FComparer).Equals(a, b);
end;

function TsgHasher.GetHash(k: Pointer): Integer;
begin
  Result := PComparer(FComparer).Hash(k);
end;

{$EndRegion}

end.

