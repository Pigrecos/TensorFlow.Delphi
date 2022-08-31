(* Protocol buffer code generator, for Delphi
 * Copyright (c) 2001-2020 Marat Shaimardanov
 *
 * This file is part of Protocol buffer code generator, for Delphi
 * is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this file. If not, see <https://www.gnu.org/licenses/>.
 *)

unit Oz.Cocor.Utils;

interface

uses
  System.Classes, System.SysUtils, System.Contnrs, System.Math,
  System.Generics.Defaults, System.Generics.Collections;

type
  FatalError = class(Exception);

{$Region 'TBitsBuffer, TBitSet: Set of 0 .. Size - 1'}

  TBitsBuffer = record
  private const
    Bpi = SizeOf(Integer) * 8;
    MaxBufSize = 32767;
    Fill: array [Boolean] of Byte = (0, $FF);
  private type
    TIntBit = 0 .. Bpi - 1;
    TIntSet = set of TIntBit;
    PIntSet = ^TIntSet;
    TBufSize = 0 .. MaxBufSize;
    PBitArray = ^TBitArray;
    TBitArray = array [0 .. MaxBufSize div Bpi] of TIntSet;
  private
    Size: TBufSize;
    Buf: Pointer;
    // check index range
    procedure Check(idx: TBufSize);
    // memory size calculation
    function CalcMemorySize(sz: TBufSize): TBufSize;
    // 'sz' the number of bit values, 'b' the value to assign to each bit
    procedure SetSize(sz: TBufSize; b: Boolean);
  public
    class function From(sz: TBufSize; defVal: Boolean): TBitsBuffer; static;
    // Get bit by idx
    function GetBit(idx: Integer): Boolean;
    // Set bit by idx
    procedure SetBit(idx: Integer; b: Boolean);
    // dest := self
    procedure AssignTo(var dest: TBitsBuffer);
    // self := self + bits
    procedure Unite(const bits: TBitsBuffer);
    // self := self * bits
    procedure Intersect(const bits: TBitsBuffer);
    // self := self - bits
    procedure Differ(const bits: TBitsBuffer);
    // set all bits to the specified value
    procedure SetAll(b: Boolean);
  end;

  TBitSet = class
  private
    FBits: TBitsBuffer;
    procedure SetBit(Index: Integer; b: Boolean);
    function GetBit(Index: Integer): Boolean;
  public
    constructor Create(Size: Cardinal; defVal: Boolean = false);
    destructor Destroy; override;
    function Equals(s: TBitSet): Boolean; reintroduce;
    function Elements: Integer;
    // a * b <> []
    function Intersects(s: TBitSet): Boolean;
    // dest := self
    function Clone: TBitSet;
    // The union of two sets.  self := self + s
    procedure Unite(s: TBitSet);
    // The intersection of two sets. self := self * s
    procedure Intersect(s: TBitSet);
    // The difference of two sets. self := self - s
    procedure Differ(s: TBitSet);
    // Set all bits to the specified value
    procedure SetAll(b: Boolean);
    // Print content
    procedure Print(const name: string; log: TTextWriter; margin: Integer);
    // The value of the bit at a specific position
    property Bits[index: Integer]: Boolean read GetBit write SetBit; default;
    // Set size.
    property Size: TBitsBuffer.TBufSize read FBits.Size;
  end;

{$EndRegion}

{$Region 'TCustomSet'}

  PRange = ^TRange;
  TRange = record
    lo, hi: Integer;
    next: PRange;
    procedure Init(lo, hi: Integer);
  end;

  // scan items until the function is false
  TScanFunction = reference to function(const r: TRange): Boolean;

  TCustomSet = class
  protected
    head: PRange;
    procedure FreeList(var head: PRange);
    function Get(i: Integer): Boolean;
  public
    destructor Destroy; override;
    procedure Incl(i: Integer);
    function Elements: Integer;
    function First: Integer;
    procedure Fill;
  end;

{$EndRegion}

{$Region 'TIntSet'}

  // todo: Implement an integer set using a tree or list
  TIntSet = class(TCustomSet)
  public
    procedure AddRange(lo, hi: Integer);
  end;

{$EndRegion}

{$Region 'TCharSet'}

  TCharSet = class(TCustomSet)
  public
    procedure Incl(ch: Char); overload;
    function Equals(s: TCharSet): Boolean; reintroduce;
    procedure Unite(s: TCharSet);
    procedure Intersect(s: TCharSet);
    function Subtract(s: TCharSet): TCharSet;
    function Includes(s: TCharSet): Boolean;
    function Intersects(s: TCharSet): Boolean;
    function Clone: TCharSet;
    function ToString: string; override;
    procedure Scan(f: TScanFunction);
    property Items[i: Integer]: Boolean read Get; default;
  end;

{$EndRegion}

function Between(ch, lo, hi: Char): Boolean;
function ToChar(i: Integer): string;
function Unquote(const s: string): string;
function AsCamel(const s: string): string;
function Plural(const s: string): string;

implementation

function Between(ch, lo, hi: Char): Boolean;
begin
  Result := (ch >= lo) and (ch <= hi);
end;

function ToChar(i: Integer): string;
begin
  if (Char(i) <> '''') and InRange(i, 32, 127)  then
    Result := Format('''%s''', [char(i)])
  else
    Result := Format('#%d', [i]);
end;

function Unquote(const s: string): string;
var
  n: Integer;
begin
  n := s.Length;
  Assert(s[1] = '"');
  Assert(s[n] = '"');
  Result := Copy(s, 2, n - 2);
end;

function AsCamel(const s: string): string;
var
  i: Integer;
  c: string;
  IsUp: Boolean;
begin
  Result := '';
  IsUp := True;
  for i := 1 to Length(s) do
  begin
    c := s[i];
    if c = '_' then
      IsUp := True
    else if not IsUp then
      Result := Result + c
    else
    begin
      Result := Result + c.ToUpperInvariant;
      IsUp := False;
    end;
  end;
end;

function Plural(const s: string): string;
begin
  Result := AsCamel(s) + 's';
end;

{$Region 'TBitsBuffer'}

class function TBitsBuffer.From(sz: TBufSize; defVal: Boolean): TBitsBuffer;
begin
  Result.Size := 0;
  Result.Buf := nil;
end;

procedure TBitsBuffer.SetSize(sz: TBufSize; b: Boolean);
var
  mem: Pointer;
  msz, old: Integer;
begin
  if sz <> Size then
  begin
    msz := CalcMemorySize(sz);
    old := CalcMemorySize(Size);
    if msz <> old then
    begin
      mem := nil;
      if msz <> 0 then
      begin
        GetMem(mem, msz);
        FillChar(mem^, msz, Fill[b]);
      end;
      if old <> 0 then
      begin
        if mem <> nil then
        begin
          if old < msz then
            msz := old;
          Move(Buf^, mem^, msz);
        end;
        FreeMem(Buf, old);
      end;
      Buf := mem;
    end;
    Size := sz;
  end;
end;

procedure TBitsBuffer.Check(idx: TBufSize);
begin
  if idx > Size then
    raise EBitsError.Create('Bits index out of range');
end;

function TBitsBuffer.CalcMemorySize(sz: TBufSize): TBufSize;
begin
  Result := (sz div Bpi + (sz mod Bpi + Bpi - 1) div Bpi) * SizeOf(Integer);
end;

procedure TBitsBuffer.SetBit(idx: Integer; b: Boolean);
var
  pint: PInteger;
  mask: Integer;
begin
  Check(idx);
  pint := Buf;
  Inc(pint, idx div Bpi);
  mask := 1 shl (idx mod Bpi);
  if b then
    pint^ := pint^ or mask
  else
    pint^ := pint^ and not mask;
end;

function TBitsBuffer.GetBit(idx: Integer): Boolean;
var
  pint: PInteger;
  mask: Integer;
begin
  Check(idx);
  pint := Buf;
  Inc(pint, idx div Bpi);
  mask := 1 shl (idx mod Bpi);
  Result := (pint^ and mask) <> 0;
end;

procedure TBitsBuffer.AssignTo(var dest: TBitsBuffer);
var msz: Integer;
begin
  Assert(Size = dest.Size);
  msz := CalcMemorySize(Size);
  Move(Buf^, dest.Buf^, msz);
end;

procedure TBitsBuffer.Unite(const bits: TBitsBuffer);
var
  i: Integer;
  a, b: PIntSet;
begin
  Assert(Size = bits.Size);
  a := Buf;
  b := bits.Buf;
  for i := 0 to Size div Bpi do
  begin
    a^ := a^ + b^;
    Inc(a);
    Inc(b);
  end;
end;

procedure TBitsBuffer.Intersect(const bits: TBitsBuffer);
var
  i: Integer;
  a, b: PIntSet;
begin
  Assert(Size = bits.Size);
  a := Buf;
  b := bits.Buf;
  for i := 0 to Size div Bpi do
  begin
    a^ := a^ * b^;
    Inc(a);
    Inc(b);
  end;
end;

procedure TBitsBuffer.Differ(const bits: TBitsBuffer);
var
  i: Integer;
  a, b: PIntSet;
begin
  Assert(Size = bits.Size);
  a := Buf;
  b := bits.Buf;
  for i := 0 to Size div Bpi do
  begin
    a^ := a^ - b^;
    Inc(a);
    Inc(b);
  end;
end;

procedure TBitsBuffer.SetAll(b: Boolean);
var msz: Integer;
begin
  msz := CalcMemorySize(Size);
  FillChar(Buf^, msz, Fill[b]);
end;

{$EndRegion}

{$Region 'TBitSet'}

constructor TBitSet.Create(Size: Cardinal; defVal: Boolean = false);
begin
  inherited Create;
  FBits.SetSize(Size, defVal);
end;

destructor TBitSet.Destroy;
begin
  FBits.SetSize(0, False);
  inherited Destroy;
end;

function TBitSet.Equals(s: TBitSet): Boolean;
var
  i: Integer;
begin
  for i := 0 to Size do
    if Self[i] <> s[i] then exit(False);
  Result := True;
end;

function TBitSet.Elements: Integer;
var
  i: Integer;
begin
  Result := 0;
  for i := 0 to Size do
    if Self[i] then Inc(Result);
end;

function TBitSet.Intersects(s: TBitSet): Boolean;
var
  i: Integer;
begin
  for i := 0 to Size do
    if Self[i] and s[i] then exit(True);
  Result := False;
end;

procedure TBitSet.SetBit(Index: Integer; b: Boolean);
begin
  FBits.SetBit(Index, b);
end;

function TBitSet.GetBit(Index: Integer): Boolean;
begin
  Result := FBits.GetBit(Index);
end;

function TBitSet.Clone: TBitSet;
begin
  Result := TBitSet.Create(Size);
  FBits.AssignTo(Result.FBits);
end;

procedure TBitSet.Unite(s: TBitSet);
begin
  FBits.Unite(s.FBits);
end;

procedure TBitSet.Intersect(s: TBitSet);
begin
  FBits.Intersect(s.FBits);
end;

procedure TBitSet.Differ(s: TBitSet);
begin
  FBits.Differ(s.FBits);
end;

procedure TBitSet.SetAll(b: Boolean);
begin
  FBits.SetAll(b);
end;

procedure TBitSet.Print(const name: string; log: TTextWriter; margin: Integer);
var
  i, col: Integer;
begin
  log.Write(name); log.Write('=[');
  col := 1;
  for i := 0 to Size - 1 do
  begin
    if Bits[i] then
      log.Write('x')
    else
      log.Write(' ');
    Inc(col);
    if col >= margin then
    begin
      log.WriteLine;
      col := 1;
    end;
  end;
  if Size = 0 then
    log.Write('-- empty set --');
  log.WriteLine(']');
end;

{$EndRegion}

{$Region 'TCustomSet'}

procedure TRange.Init(lo, hi: Integer);
begin
  Self.lo := lo;
  Self.hi := hi;
  Self.next := nil;
end;

procedure Include(var head: PRange; i: Integer);
var
  p, q, cn, nr: PRange;
begin
  p := head;
  q := nil;
  while (p <> nil) and (i >= p.lo - 1) do
  begin
    if i <= p.hi + 1 then
    begin
      // p.lo - 1 <= i <= p.hi + 1
      if i = p.lo - 1 then
        Dec(p.lo)
      else if i = p.hi + 1 then
      begin
        Inc(p.hi);
        cn := p.next;
        if (cn <> nil) and (p.hi = cn.lo - 1) then
        begin
          p.hi := cn.hi;
          p.next := cn.next;
          Dispose(cn);
        end;;
      end;
      exit;
    end;
    q := p;
    p := p.next;
  end;
  new(nr); nr.Init(i, i);
  nr.next := p;
  if q = nil then
    head := nr
  else
    q.next := nr;
end;

destructor TCustomSet.Destroy;
begin
  FreeList(head);
  inherited;
end;

procedure TCustomSet.FreeList(var head: PRange);
var
  p, q: PRange;
begin
  p := head;
  while p <> nil do
  begin
    q := p.next; Dispose(p);
    p := q;
  end;
  head := nil;
end;

function TCustomSet.Get(i: Integer): Boolean;
var
  p: PRange;
begin
  p := head;
  while p <> nil do
  begin
    if i < p.lo then
      exit(false)
    else if i <= p.hi then
      // p.lo <= i <= p.to
      exit(true);
    p := p.next;
  end;
  Result := false;
end;

procedure TCustomSet.Incl(i: Integer);
begin
  Include(head, i);
end;

function TCustomSet.Elements: Integer;
var
  n: Integer;
  p: PRange;
begin
  n := 0;
  p := head;
  while p <> nil do
  begin
    n := n + p.hi - p.lo + 1;
    p := p.next;
  end;
  Result := n;
end;

function TCustomSet.First: Integer;
begin
  if head <> nil then
    exit(head.lo);
  Result := -1;
end;

procedure TCustomSet.Fill;
begin
  New(head); head.Init(0, Ord(high(Char)));
end;

{$EndRegion}

{$Region 'TIntSet'}

procedure TIntSet.AddRange(lo, hi: Integer);
begin

end;

{$EndRegion}

{$Region 'TCharSet'}

procedure TCharSet.Incl(ch: Char);
begin
  Include(head, Ord(ch));
end;

function TCharSet.Clone: TCharSet;
var
  s: TCharSet;
  prev, cur, r: PRange;
begin
  s := TCharSet.Create;
  prev := nil;
  cur := head;
  while cur <> nil do
  begin
    New(r); r.Init(cur.lo, cur.hi);
    if prev = nil then
      s.head := r
    else
      prev.next := r;
    prev := r;
    cur := cur.next;
  end;
  Result := s;
end;

function TCharSet.Equals(s: TCharSet): Boolean;
var
  p, q: PRange;
begin
  p := head;
  q := s.head;
  while (p <> nil) and (q <> nil) do
  begin
    if (p.lo <> q.lo) or (p.hi <> q.hi) then
      exit(false);
    p := p.next;
    q := q.next;
  end;
  Result := p = q;
end;

procedure TCharSet.Unite(s: TCharSet);
var
  p: PRange;
  i: Integer;
begin
  p := s.head;
  while p <> nil do
  begin
    for i := p.lo to p.hi do
      Incl(i);
    p := p.next;
  end;
end;

procedure TCharSet.Intersect(s: TCharSet);
var
  p, x: PRange;
  i: Integer;
begin
  x := nil;
  p := head;
  while p <> nil do
  begin
    for i := p.lo to p.hi do
      if s[i] then
        Include(x, i);
    p := p.next;
  end;
  FreeList(head);
  head := x;
end;

procedure TCharSet.Scan(f: TScanFunction);
var
  r: PRange;
begin
  r := head;
  while r <> nil do
  begin
    if f(r^) then exit;
    r := r.next;
  end;
end;

function TCharSet.Subtract(s: TCharSet): TCharSet;
var
  r: PRange;
  i: Integer;
begin
  Result := TCharSet.Create;
  r := head;
  while r <> nil do
  begin
    for i := r.lo to r.hi do
      if not s[i] then
        Result.Incl(i);
    r := r.next;
  end;
end;

function TCharSet.ToString: string;
var
  r: PRange;
  s: string;
begin
  r := head;
  Result := '[';
  while r <> nil do
  begin
    if r.lo = r.hi then
      s := ToChar(r.lo)
    else
      s := ToChar(r.lo) + '..' + ToChar(r.hi);
    if r <> head then
      Result := Result + ',';
    Result := Result + s;
    r := r.next;
  end;
  Result := Result + ']';
end;

function TCharSet.Includes(s: TCharSet): Boolean;
var
  p: PRange;
  i: Integer;
begin
  p := s.head;
  while p <> nil do
  begin
    for i := p.lo to p.hi do
      if not Self[i] then
        exit(false);
    p := p.next;
  end;
  Result := true;
end;

function TCharSet.Intersects(s: TCharSet): Boolean;
var
  p: PRange;
  i: Integer;
begin
  p := s.head;
  while p <> nil do
  begin
    for i := p.lo to p.hi do
      if Self[i] then
        exit(true);
    p := p.next;
  end;
  Result := false;
end;

{$EndRegion}

end.

