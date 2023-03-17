(* Standard Generic Library (SGL) for Pascal
 * Copyright (c) 2020, 2021 Marat Shaimardanov
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
unit Oz.SGL.Utils;

interface

type

{$Region 'TStd'}

  TStd = record
    // Copies exactly count values from the range beginning
    // at first to the range beginning at result.
    class function CopyN<T>(First: Pointer; Count: Cardinal; var R): Pointer; static;
    // Assigns the given value to the first count elements
    // in the range beginning at first if count > 0.
    class procedure FillN<T>(First: Pointer; Count: Cardinal; const Value: T); static;
  end;

{$EndRegion}

{$Region 'TSpan: a contiguous sequence of objects'}

  // The span describes an object that can refer to a contiguous sequence
  // of objects with the first element of the sequence at position zero.
  TSpan<T> = record
  type
    PItem = ^T;
  var
    FStart: PItem;
    FSize: Cardinal;
    function GetItem(Index: Integer): T; inline;
  public
    constructor From(start: PItem; size: Cardinal);
    function Empty: Boolean;
    property Size: Cardinal read FSize;
    property Items[Index: Integer]: T read GetItem;
  end;

{$EndRegion}

implementation

{$Region 'TStd'}

class function TStd.CopyN<T>(First: Pointer; Count: Cardinal; var R): Pointer;
type
  Pt = ^T;
var
  src, dest: Pt;
begin
  src := First;
  dest := @R;
  while Count > 0 do
  begin
    dest^ := src^;
    Inc(PByte(src), sizeof(T));
    Inc(PByte(dest), sizeof(T));
    Dec(Count);
  end;
  Result := @R;
end;

class procedure TStd.FillN<T>(First: Pointer; Count: Cardinal; const Value: T);
type
  Pt = ^T;
var
  p: Pt;
begin
  p := First;
  while Count > 0 do
  begin
    p^ := Value;
    Inc(PByte(p), sizeof(T));
    Dec(Count);
  end;
end;

{$EndRegion}

{$Region 'TSpan<T>'}

constructor TSpan<T>.From(start: PItem; size: Cardinal);
begin
  FStart := start;
  FSize := size;
end;

function TSpan<T>.Empty: Boolean;
begin
  Result := FSize = 0;
end;

function TSpan<T>.GetItem(Index: Integer): T;
begin
  Result := PItem(PByte(FStart) + sizeof(T) * Index)^;
end;

{$EndRegion}

end.

