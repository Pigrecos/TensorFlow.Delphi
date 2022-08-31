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

unit Oz.SGL.HandleManager;

interface

{$Region 'Uses'}

uses
  System.SysUtils, System.Math;

{$EndRegion}

{$T+}

{$Region 'Handles'}

type
  // Typed memory region handle
  hRegion = record
  const
    MaxIndex = 1024;
  type
    TIndex = 0 .. MaxIndex; // 10 bits
  var
    v: Cardinal;
  public
    function Index: TIndex; inline;
  end;

  // Handle for data instance of some type.
  hCollection = record
  const
    MaxIndex = 16 * 1024 - 1;
  type
    TIndex = 0 .. MaxIndex; // 14 bits
  var
    v: Cardinal;
  public
    constructor From(index: TIndex; counter: Byte; region: hRegion);
    // Index field.
    // These bits will make up the actual index in the handle manager,
    // so going from a handle to a pointer is a very fast operation.
    function Index: TIndex; inline;
    // Typed memory region handle.
    // This field allows to find out where
    // and what type of data the pointer refers to.
    function Region: hRegion; inline;
    // Reuse counter
    // The counter field is used to detect the validity of the old handle.
    // This field contains a number that is incremented each time an index slot
    // is reused. Whenever the Handle manager tries to convert a handle to a pointer,
    // it first checks to see if the counter field matches the stored record.
    // If not, it knows that the handle has expired and returns zero.
    function Counter: Byte; inline;
  end;

{$EndRegion}

{$Region 'TsgHandleManager: Handle manager'}

  // Efficient implementation of the handle manager.
  // The handle manager is implemented as an array of pointers,
  // and handles are indexes in this array.
  TsgHandleManager = record
  const
    MaxNodes = 16384; // use values 2 ^ n
    GuardNode = MaxNodes - 1;
  type
    TIndex = 0 .. MaxNodes - 1;

    // Array element.
    TNode = packed record
    public
      ptr: Pointer;  // a pointer to a data instance of some type
      prev: TIndex;
      next: TIndex;
      counter: Byte;
      active: Boolean;
      procedure Init(next, prev: TIndex);
    end;

    PNode = ^TNode;
    TNodes = array [TIndex] of TNode;
    TNodeProc = procedure(h: hCollection) of object;
  private
    FNodes: TNodes;
    FCount: Integer;
    FRegion: hRegion;
    FUsed: TIndex;
    FAvail: TIndex;
    // Move node from the src list to the dest list
    function MoveNode(idx: TIndex; var src, dest: TIndex): PNode; inline;
  public
    procedure Init(region: hRegion);
    // Add a pointer to a data instance of some type and return a handle
    function Add(ptr: Pointer): hCollection;
    // For the specified handle, replace a data instance of some type
    procedure Update(handle: hCollection; ptr: Pointer);
    // Remove an instance of some type of data
    procedure Remove(handle: hCollection);
    // Return an instance of some type of data
    function Get(handle: hCollection): Pointer;
    // Traverse all nodes in the handle manager
    // and execute the specified procedure on each of its nodes
    procedure Traversal(proc: TNodeProc);
    property Count: Integer read FCount;
  end;

{$EndRegion}

implementation

{$Region 'hRegion'}

function hRegion.Index: TIndex;
begin
  Result := v and $3FF;
end;

{$EndRegion}

{$Region 'hCollection'}

constructor hCollection.From(index: TIndex; counter: Byte; region: hRegion);
begin
  v := (((counter shl 10) or region.v) shl 14) or index;
end;

function hCollection.Index: TIndex;
begin
  // 2^14 - 1
  Result := v and $3FFF;
end;

function hCollection.Region: hRegion;
begin
  // 2^8 + 2^14
  Result.v := v shr 14 and $3FF;
end;

function hCollection.Counter: Byte;
begin
  // 2^8 0..255
  Result := (v shr 24) and $FF;
end;

{$EndRegion}

{$Region 'TsgHandleManager.TNode'}

procedure TsgHandleManager.TNode.Init(next, prev: TIndex);
begin
  ptr := nil;
  Self.next := next;
  Self.prev := prev;
  Self.counter := 1;
  Self.active := False;
end;

{$EndRegion}

{$Region 'TsgHandleManager'}

procedure TsgHandleManager.Init(region: hRegion);
var
  i: Integer;
  n: PNode;
begin
  FillChar(Self, sizeof(TsgHandleManager), 0);
  FRegion := region;
  FCount := 0;
  FUsed := GuardNode;
  FAvail := 0;
  for i := 0 to GuardNode - 1 do
  begin
    n := @FNodes[i];
    n.Init((i + 1) and GuardNode, (i - 1) and GuardNode);
  end;
  // guard node
  n := @FNodes[GuardNode];
  n.Init(GuardNode, GuardNode);
end;

function TsgHandleManager.MoveNode(idx: TIndex; var src, dest: TIndex): PNode;
var
  p: PNode;
begin
  Result := @FNodes[idx];
  // remove node from src list
  p := @FNodes[src];
  src := p.next;
  p.prev := GuardNode; // guard node
  // add node to dest list
  Result.next := dest;
  p := @FNodes[dest];
  p.prev := idx;
  dest := idx;
end;

function TsgHandleManager.Add(ptr: Pointer): hCollection;
var
  idx: Integer;
  n: PNode;
begin
{$IFDEF DEBUG}
  Assert(FCount < GuardNode - 1);
{$ENDIF}
  idx := FAvail;
{$IFDEF DEBUG}
  Assert(idx < GuardNode);
{$ENDIF}
  n := MoveNode(idx, FAvail, FUsed);
  n.counter := n.counter + 1;
  if n.counter = 0 then
    n.counter := 1;
{$IFDEF DEBUG}
  Assert(not n.active);
{$ENDIF}
  n.active := True;
  n.ptr := ptr;
  Inc(FCount);
  Result := hCollection.From(idx, n.counter, FRegion);
end;

procedure TsgHandleManager.Update(handle: hCollection; ptr: Pointer);
var
  n: PNode;
begin
  n := @FNodes[handle.Index];
{$IFDEF DEBUG}
  Assert(n.active);
  Assert(n.counter = handle.counter);
{$ENDIF}
  n.ptr := ptr;
end;

procedure TsgHandleManager.Remove(handle: hCollection);
var
  n: PNode;
begin
  n := MoveNode(handle.Index, FUsed, FAvail);
{$IFDEF DEBUG}
  Assert(n.active);
  Assert(n.counter = handle.counter);
{$ENDIF}
  n.active := False;
  Dec(FCount);
end;

function TsgHandleManager.Get(handle: hCollection): Pointer;
var
  n: PNode;
begin
  n := @FNodes[handle.Index];
  if (n.counter <> handle.counter) or not n.active then exit(nil);
  Result := n.ptr;
end;

procedure TsgHandleManager.Traversal(proc: TNodeProc);
var
  idx: Integer;
  n: PNode;
  h: hCollection;
begin
  if FCount = 0 then exit;
  idx := FUsed;
  while idx < GuardNode do
  begin
    n := @FNodes[idx];
    h := hCollection.From(idx, n.counter, FRegion);
    proc(h);
    idx := n.next;
  end;
end;

{$EndRegion}

end.

