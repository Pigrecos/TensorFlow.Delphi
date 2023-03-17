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

// The buffer for strings.
// The main purpose of the rapid format of long string.
// Features:
// 1. Minimize the handling of the memory manager.
// 2. One-time allocation of the specified size
// 3. The class is not multi-thread safe

unit Oz.Pb.StrBuffer;

interface

uses Classes, SysUtils;

const
  MaxBuffSize = Maxint div 16;
  SBufferIndexError = 'Buffer index out of bounds (%d)';
  SBufferCapacityError = 'Buffer capacity out of bounds (%d)';
  SBufferCountError = 'Buffer count out of bounds (%d)';

type

{$Region 'TStrBuffer: Unsegmented buffer'}

{ If you do not have enough space in the string than
  is taken a piece of memory twice the size
  and copies the data in this chunk of memory }
  TStrBuffer = class
  private
    FCount: Integer;
    FCapacity: Integer;
    FBuff: PByte;
  protected
    class procedure Error(const Msg: string; Data: Integer);
  public
    constructor Create;
    destructor Destroy; override;
    procedure SaveToStream(Stream: TStream);
    procedure SaveToFile(const FileName: string);
    procedure LoadFromFile(const FileName: string);
    procedure LoadFromStream(Stream: TStream);
    procedure Clear;
    procedure Add(const Value: TBytes); overload;
    procedure SetCapacity(NewCapacity: Integer);
    function GetBytes: TBytes;
    function GetCount: Integer;
    property Bytes: TBytes read GetBytes;
  end;

{$EndRegion}

{$Region 'TSegmentBuffer: memory is reserved by segments, without reallocation'}

  PSegment = ^TSegment;
  TSegment = record
    Next: PSegment;
    Size: Integer;
    Count: Integer;
    Data: array[0..0] of Byte;
  end;

  TSegmentBuffer = class
  private
    FCount: Integer;
    FCapacity: Integer;
    FFirst: PSegment;
    FLast: PSegment;
  protected
    class procedure Error(const Msg: string; Data: Integer);
  public
    constructor Create;
    destructor Destroy; override;
    procedure SaveToStream(Stream: TStream);
    procedure SaveToFile(const FileName: string);
    procedure LoadFromFile(const FileName: string);
    procedure LoadFromStream(Stream: TStream);
    procedure Clear;
    procedure AddSegment(Size: Integer);
    procedure Add(const Value: Byte); overload;
    procedure Add(const Value: TBytes); overload;
    procedure Add(const Value: PByte; Cnt: Integer); overload;
    function GetBytes: TBytes;
    function GetCount: Integer;
    property Bytes: TBytes read GetBytes;
  end;

{$EndRegion}

implementation

{$RANGECHECKS OFF}

{$Region 'TStrBuffer'}

class procedure TStrBuffer.Error(const Msg: string; Data: Integer);

  function ReturnAddr: Pointer;
  asm
    MOV EAX,[EBP+4]
  end;

begin
  raise EListError.CreateFmt(Msg, [Data])at ReturnAddr;
end;

constructor TStrBuffer.Create;
begin
  inherited Create;
  FCount := 0;
  FCapacity := 0;
  FBuff := nil;
end;

destructor TStrBuffer.Destroy;
begin
  Clear;
  inherited;
end;

procedure TStrBuffer.Clear;
begin
  FCount := 0;
  SetCapacity(0);
end;

procedure TStrBuffer.Add(const Value: TBytes);
var
  cnt, delta: Integer;
begin
  cnt := Length(Value);
  if FCount + cnt > FCapacity then
  begin
    delta := FCapacity div 2;
    if delta < cnt then
      delta := cnt * 2;
    SetCapacity(FCapacity + delta);
  end;
  System.Move(Pointer(Value)^, PByte(FBuff + FCount)^, cnt);
  Inc(FCount, cnt);
end;

function TStrBuffer.GetCount: Integer;
begin
  Result := FCount;
end;

function TStrBuffer.GetBytes: TBytes;
begin
  SetLength(Result, FCount);
  System.Move(FBuff^, Pointer(Result)^, FCount);
end;

procedure TStrBuffer.SetCapacity(NewCapacity: Integer);
begin
  if (NewCapacity < FCount) or (NewCapacity > MaxBuffSize) then
    Error(SBufferCapacityError, NewCapacity);
  if NewCapacity <> FCapacity then
  begin
    ReallocMem(FBuff, NewCapacity);
    FCapacity := NewCapacity;
  end;
end;

procedure TStrBuffer.SaveToStream(Stream: TStream);
begin
  Stream.WriteBuffer(FBuff, FCount);
end;

procedure TStrBuffer.SaveToFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TStrBuffer.LoadFromFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TStrBuffer.LoadFromStream(Stream: TStream);
var
  size: Integer;
  bytes: TBytes;
begin
  Clear;
  size := Stream.Size - Stream.Position;
  SetLength(bytes, size);
  Stream.Read(Pointer(bytes)^, size);
  Add(bytes);
end;

{$EndRegion}

{$Region 'TSegmentBuffer'}

class procedure TSegmentBuffer.Error(const Msg: string; Data: Integer);

  function ReturnAddr: Pointer;
  asm
    MOV EAX,[EBP+4]
  end;

begin
  raise EListError.CreateFmt(Msg, [Data])at ReturnAddr;
end;

constructor TSegmentBuffer.Create;
begin
  inherited Create;
  FCount := 0;
  FFirst := AllocMem(4096 + SizeOf(TSegment));
  FFirst.Next := nil;
  FFirst.Size := 4096;
  FFirst.Count := 0;
  FLast := FFirst;
end;

destructor TSegmentBuffer.Destroy;
begin
  Clear;
  FreeMem(FFirst, FFirst^.Size);
  inherited Destroy;
end;

procedure TSegmentBuffer.Clear;
var p1, p2: PSegment;
begin
  p1 := FFirst;
  while p1 <> FLast do
  begin
    p2 := p1;
    p1 := p1^.next;
    FreeMem(p2, p2^.Size);
  end;
  FFirst := FLast;
  FFirst^.Count := 0;
  FCount := 0;
end;

procedure TSegmentBuffer.AddSegment(Size: Integer);
var
  segment: PSegment;
begin
  segment := AllocMem(Size + SizeOf(TSegment) - SizeOf(Byte));
  segment.next := nil;
  segment.size := Size;
  segment.count := 0;
  FLast^.next := segment;
  FLast := segment;
  Inc(FCapacity, Size);
end;

function TSegmentBuffer.GetCount: Integer;
begin
  Result := FCount;
end;

function TSegmentBuffer.GetBytes: TBytes;
var
  p: PByte;
  segment: PSegment;
  len: Integer;
begin
  SetLength(Result, FCount);
  p := @Result[0];
  segment := FFirst;
  while segment <> nil do
  begin
    len := segment^.Count;
    System.Move(segment^.Data, p^, len);
    Inc(p, len);
    segment := segment^.Next;
  end;
end;

procedure TSegmentBuffer.Add(const Value: PByte; Cnt: Integer);
var
  p: PByte;
  tmp: Integer;
begin
  p := Value;
  // define size of unused memory in current buffer segment
  tmp := FLast^.Size - FLast^.Count;
  // if you do not have enough space in the buffer then copy the "unused" bytes
  // and reduce current segment
  if Cnt > tmp then
  begin
    System.Move(p^, FLast^.Data[FLast^.Count], tmp);
    Inc(FLast^.Count, tmp);
    Inc(FCount, tmp);
    Inc(p, tmp);
    Dec(Cnt, tmp);
    // add another segment of the larger buffer size
    tmp := FLast^.Size;
    if tmp < Cnt then tmp := Cnt;
    AddSegment(tmp * 2);
  end;
  if Cnt > 0 then
  begin
    Move(p^, FLast^.Data[FLast^.Count], Cnt);
    Inc(FCount, Cnt);
    Inc(FLast^.Count, Cnt);
  end;
end;

procedure TSegmentBuffer.Add(const Value: TBytes);
var len: Integer;
begin
  len := Length(Value);
  if len > 0 then
    Add(@Value[0], len);
end;

procedure TSegmentBuffer.Add(const Value: Byte);
begin
  Add(@Value, 1);
end;

procedure TSegmentBuffer.SaveToFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TSegmentBuffer.SaveToStream(Stream: TStream);
var segment: PSegment;
begin
  segment := FFirst;
  while segment <> nil do
  begin
    Stream.WriteBuffer(segment^.Data[0], segment^.Count);
    segment := segment^.Next;
  end;
end;

procedure TSegmentBuffer.LoadFromFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TSegmentBuffer.LoadFromStream(Stream: TStream);
var
  size: Integer;
  bytes: TBytes;
begin
  Clear;
  size := Stream.Size - Stream.Position;
  SetLength(bytes, size);
  Stream.Read(Pointer(bytes)^, size);
  Add(bytes);
end;

{$EndRegion}

{$RANGECHECKS ON}

end.
