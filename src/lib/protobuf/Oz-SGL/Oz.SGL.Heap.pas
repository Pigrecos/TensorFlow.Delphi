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

unit Oz.SGL.Heap;

interface

uses
  System.SysUtils, System.Math;

{$T+}

{$Region 'Forward declarations'}

type
  TCompareProc = function(const A, B): Integer;
  TFreeProc = procedure(p: Pointer);
  TEqualsFunc = function(a, b: Pointer): Boolean;
  THashProc = function(const key: PByte): Cardinal;
  TPredicateFunc = function(const p: Pointer): Boolean;

{$EndRegion}

{$Region 'EsgError'}

  EsgError = class(Exception)
  const
    NotImplemented = 0;
    InvalidParameters = 1;
    ListIndexError = 2;
    ListCountError = 3;
    CapacityError = 4;
    IncompatibleDataType = 5;
    NotEnoughMemory = 6;
    InvalidSize = 7;
    InvalidPointer = 8;
    ErrorMax = 8;
  private type
    TErrorMessages = array [0..ErrorMax] of string;
  private const
    ErrorMessages: TErrorMessages = (
      'Not implemented',
      'Invalid parameters',
      'List index error (%d)',
      'List count error (%d)',
      'List capacity error (%d)',
      'Incompatible data type',
      'Not enough memory',
      'Alloc: Invalid Size',
      'Dealloc: Invalid Pointer');
  public
    constructor Create(ErrNo: Integer); overload;
    constructor Create(ErrNo, IntParam: Integer); overload;
  end;

{$EndRegion}

{$Region 'TsgMemoryManager: Memory manager'}

  PsgFreeBlock = ^TsgFreeBlock;
  TsgFreeBlock = record
    Next: PsgFreeBlock;
    Size: Cardinal;
  end;

  TsgMemoryManager = record
  const
    MinSize = sizeof(Pointer) * 2;
  var
    Avail: PsgFreeBlock;
    Heap: Pointer;
    TopMemory: Pointer;
  public
    procedure Init(Heap: Pointer; HeapSize: Cardinal);
    // Allocate memory and return a pointer (nil for not enough memory)
    function Alloc(Size: Cardinal): Pointer;
    // Return memory to heap
    procedure FreeMem(Ptr: Pointer; Size: Cardinal);
    // Return memory to the heap with parameter validation
    procedure Dealloc(Ptr: Pointer; Size: Cardinal);
    // Reallocate memory
    function Realloc(Ptr: Pointer; OldSize, Size: Cardinal): Pointer;
  end;

{$EndRegion}

{$Region 'TsgItemMeta: Metadata for item of some type'}

  // Action to remove an element from the collection
  TRemoveAction = (
    HoldValue = 0,  // Hold the item value
    Clear = 1,      // Clear the item value
    Reuse = 2,      // Clear the item value and allow reuse
    Other = 3);     // Reserved

  // packed collection flags
  hMeta = packed record
  private const
    Seed: Word = 25117;
    function GetTypeKind: System.TTypeKind;
    function GetManagedType: Boolean;
    function GetHasWeakRef: Boolean;
    function GetSegmented: Boolean;
    procedure SetSegmented(const Value: Boolean);
    function GetRangeCheck: Boolean;
    procedure SetRangeCheck(const Value: Boolean);
    function GetNotification: Boolean;
    procedure SetNotification(const Value: Boolean);
    function GetOwnedObject: Boolean;
    procedure SetOwnedObject(const Value: Boolean);
    function GetRemoveAction: TRemoveAction;
    procedure SetRemoveAction(const Value: TRemoveAction);
  public
    constructor From(TypeKind: System.TTypeKind; ManagedType, HasWeakRef: Boolean);
    function Valid: Boolean; inline;
    // property
    property TypeKind: System.TTypeKind read GetTypeKind;
    // Has managed type
    property ManagedType: Boolean read GetManagedType;
    // Has weak reference
    property HasWeakRef: Boolean read GetHasWeakRef;
    property Segmented: Boolean read GetSegmented write SetSegmented;
    property RangeCheck: Boolean read GetRangeCheck write SetRangeCheck;
    property Notification: Boolean read GetNotification write SetNotification;
    property OwnedObject: Boolean read GetOwnedObject write SetOwnedObject;
    property RemoveAction: TRemoveAction read GetRemoveAction write SetRemoveAction;
  case Integer of
    0: (
      v: Integer);
    1: (
      MetaFlags: Byte;
      RegionFlags: Byte;
      SeedValue: Word);
  end;

  TRegionFlag = (
    rfSegmented,
    rfRangeCheck,
    rfNotification,
    rfOwnedObject);
  TRegionFlagSet = set of TRegionFlag;

  PsgItemMeta = ^TsgItemMeta;
  TFreeItem = procedure(meta: PsgItemMeta; p: Pointer);
  TAssignProc = procedure(meta: PsgItemMeta; Dest, Value: Pointer);

  TsgItemMeta = record
  var
    TypeInfo: Pointer;
    ItemSize: Cardinal;
    h: hMeta;
    OnFree: TFreeProc;
    FreeItem: TFreeItem;
    AssignItem: TAssignProc;
  strict private
    procedure InitMethods;
  public
    procedure Init<T>(OnFree: TFreeProc = nil); overload;
    procedure Init<T>(Flags: TRegionFlagSet; RemoveAction: TRemoveAction;
      OnFree: TFreeProc = nil); overload;
  end;
  PPsgItemMeta = ^PsgItemMeta;

{$EndRegion}

{$Region 'TsgMeta: unified metadata'}

{ We need a structure to store information for some type (metadata).
  Type (TypeInfo)
  Size
  Methods used to implement collections such as:
    Purge
    Assignment
    Value exchange
    Value comparison
    Hash generation
    Popular Serialization Methods
}
  TsgMeta = record
  private
    TypeInfo: Pointer;
    ItemSize: Cardinal;
    h: hMeta;
    FreeItem: TFreeItem;
    AssignItem: TAssignProc;
  strict private
    procedure InitMethods;
  public
    procedure Init<T>(FreeItem: TFreeItem);
  end;
  PsgMeta = ^TsgMeta;

{$EndRegion}

{$Region 'TMemSegment: Memory segment'}

// Right now the memory is being freed in the memory region.
// It makes sense to add the necessary information to the memory region
// in the form of metadata on the tuple.
// Then it would be easy enough to implement a universal memory release method.

  PMemSegment = ^TMemSegment;
  TMemSegment = record
  private
    Next: PMemSegment;  // Next segment
    TopMem: PByte;      // Reference to top memory
    FreePtr: PByte;     // Reference to free memory
    procedure CheckPointer(Ptr: Pointer; Size: Cardinal);
    procedure Init(HeapSize: Cardinal);
  public
    class procedure NewSegment(var s: PMemSegment; HeapSize: Cardinal); static;
    class procedure IncreaseHeapSize(var s: PMemSegment; NewHeapSize: Cardinal); static;
    // Allocate a piece of memory of the specified size
    function Occupy(Size: Cardinal): Pointer;
    // Return a reference to the beginning of the heap
    function GetHeapRef: PByte; inline;
    // Return the size of free memory
    function GetFreeSize: Cardinal; inline;
    // Return the size of the memory segment
    function GetHeapSize: Cardinal; inline;
    // Return the occupied size:
    function GetOccupiedSize: Cardinal; inline;
  end;

{$EndRegion}

{$Region 'TMemoryRegion: Typed memory region'}

  PMemoryRegion = ^TMemoryRegion;
  TMemoryRegion = record
  type
    TSwapProc = procedure(A, B: Pointer) of object;
  private
    Heap: PMemSegment;
    BlockSize: Cardinal;
    FCapacity: Integer;
    FCount: Integer;
    FMeta: TsgItemMeta;
    FTemporary: Pointer;
    // procedural types
    FSwapItems: TSwapProc;
    FCompareItems: TCompareProc;
    procedure GrowHeap(NewCount: Integer);
    function Grow(NewCount: Integer): Integer;
    function GetOccupiedCount(p: PMemSegment): Integer;
    procedure FreeHeap(var Heap: PMemSegment);
    procedure FreeItems(p: PMemSegment);
    function Valid: Boolean; inline;
    function GetMeta: PsgItemMeta;
  public
    // Segmented region provides immutable pointer addresses
    procedure Init(Meta: PsgItemMeta; BlockSize: Cardinal);
    // Free the region
    procedure Free;
    // Erases all elements from the memory region.
    procedure Clear;
    // Increase capacity
    function IncreaseCapacity(NewCount: Integer): Pointer;
    // Increase capacity and allocate
    function IncreaseAndAlloc(NewCount: Integer): Pointer;
    // Allocate memory of a specified size and return its pointer
    function Alloc(Size: Cardinal): Pointer;
    // Get a pointer to an element of an array of the specified type
    function GetItemPtr(Index: Cardinal): Pointer; inline;
    // Return a temporary variable
    function GetTemporary: Pointer;
    // propeties
    property Meta: PsgItemMeta read GetMeta;
    property Capacity: Integer read FCapacity;
    property ItemSize: Cardinal read FMeta.ItemSize;
    // item methods
    property FreeItem: TFreeItem read FMeta.FreeItem;
    property AssignItem: TAssignProc read FMeta.AssignItem;
    property SwapItems: TSwapProc read FSwapItems;
    property CompareItems: TCompareProc read FCompareItems;
  end;

{$EndRegion}

{$Region 'TUnbrokenRegion: Unbroken typed memory region for an array'}

  PUnbrokenRegion = ^TUnbrokenRegion;
  TUnbrokenRegion = record
  private
    FRegion: TMemoryRegion;
    function GetRegion: PMemoryRegion; inline;
    function GetMeta: PsgItemMeta; inline;
  public
    procedure Init(Meta: PsgItemMeta; BlockSize: Cardinal = 8192); overload;
    procedure InitWithCount(Meta: PsgItemMeta; Count: Cardinal); overload;
    // Free the region
    procedure Free; inline;
    // Erases all elements from the memory region.
    procedure Clear; inline;
    // Add an empty element
    function AddItem: PByte;
    // Insert an empty element
    procedure Insert(Index: Integer; const Value);
    // Delete element
    procedure Delete(Index: Integer);
    // Return number of elements
    function GetCount: Integer; inline;
    // Set number of elements
    procedure SetCount(NewCount: Integer);
    // Get a pointer to items
    function GetItems: PByte; inline;
    // Get a pointer to an element of an array of the specified type
    function GetItemPtr(Index: Cardinal): Pointer;
    // Increment pointer to an element
    function NextItem(Item: Pointer): Pointer;
    // Copy items from region
    procedure CopyFrom(const Region: PUnbrokenRegion; Index, Cnt: Integer);
    // Assign element
    procedure AssignItem(Dest, Src: Pointer);
    // Exchange elements
    procedure Exchange(Index1, Index2: Integer);
    // Propeties
    property Region: PMemoryRegion read GetRegion;
    property Meta: PsgItemMeta read GetMeta;
    property Capacity: Integer read FRegion.FCapacity;
    property Count: Integer read FRegion.FCount write SetCount;
    property ItemSize: Cardinal read FRegion.FMeta.ItemSize;
  end;

{$EndRegion}

{$Region 'TSegmentedRegion: Segmented typed region'}

  PSegmentedRegion = ^TSegmentedRegion;
  TSegmentedRegion = record
  private
    FRegion: TMemoryRegion;
    function GetMeta: PsgItemMeta; inline;
    function GetRegion: PMemoryRegion; inline;
    function GetCapacity: Integer;
    function GetCount: Integer;
  public
    procedure Init(Meta: PsgItemMeta; BlockSize: Cardinal);
    // Free the region
    procedure Free; inline;
    // Add item
    function AddItem: Pointer;
    // Dispose count items
    procedure Dispose(Items: Pointer; Count: Cardinal);
    // Assign
    procedure AssignItem(Dest, Src: Pointer);
    // Get a pointer to an element of an array of the specified type
    function GetItemPtr(Index: Cardinal): Pointer;
    // Propeties
    property Region: PMemoryRegion read GetRegion;
    // metadata
    property Meta: PsgItemMeta read GetMeta;
    // capacity
    property Capacity: Integer read GetCapacity;
    // number of elements
    property Count: Integer read GetCount;
    // item size
    property ItemSize: Cardinal read FRegion.FMeta.ItemSize;
  end;

{$EndRegion}

{$Region 'TsgItem: structure for a collection item of some type'}

  TsgItem = record
  private
    Ptr: Pointer;
    Region: PMemoryRegion;
  public
    procedure Init<T>(const Region: TMemoryRegion; var Value: T);
    procedure Assign(const Value);
    procedure Free;
  end;
  PsgItem = ^TsgItem;

{$EndRegion}

{$Region 'THeapPool: Memory Pool'}

  // List item
  PRegionItem = ^TRegionItem;

  TRegionItem = record
    r: TMemoryRegion;
    Next: PRegionItem;
  end;

  // List of memory regions
  TRegionItems = record
    root: PRegionItem;
    procedure Init; inline;
    // Add region
    procedure Add(p: PRegionItem);
    // Remove from the list
    function Remove: PRegionItem;
    // List is empty
    function Empty: Boolean; inline;
  end;

  THeapPool = class
  strict private
    FRegions: PMemoryRegion;
    FRealesed: TRegionItems;
    FBlockSize: Word;
    // Occupy region
    function FindOrCreateRegion(Meta: PsgItemMeta): PMemoryRegion;
  public
    constructor Create(BlockSize: Word = 8 * 1024);
    destructor Destroy; override;
    // Create a continuous region (e.g. memory for arrays)
    function CreateUnbrokenRegion(Meta: PsgItemMeta): PUnbrokenRegion;
    // Create a segmented region (for elements with a fixed address)
    function CreateRegion(Meta: PsgItemMeta): PSegmentedRegion;
    // Release the region
    procedure Release(r: PMemoryRegion);
  end;

{$EndRegion}

{$Region 'TsgContext: Processing context'}

  TsgContext = class
  private class var
    FPointerMeta: TsgItemMeta;
    FMemoryRegionMeta: TsgItemMeta;
  private
    function GetHeapPool: THeapPool;
  protected
    FHeapPool: THeapPool;
  public
    constructor Create;
    destructor Destroy; override;
    // Create a continuous region (e.g. memory for arrays)
    function CreateUnbrokenRegion(Meta: PsgItemMeta): PUnbrokenRegion; inline;
    // Create a segmented region (for elements with a fixed address)
    function CreateRegion(Meta: PsgItemMeta): PSegmentedRegion; inline;
    // Release the region
    procedure Release(r: PMemoryRegion); inline;
    // Clear main memory pool
    procedure ClearHeapPool;

    // Metadata for standard types
    class procedure InitMetadata;
    class property PointerMeta: TsgItemMeta read FPointerMeta;
    class property MemoryRegionMeta: TsgItemMeta read FMemoryRegionMeta;

    // Main memory pool
    property Pool: THeapPool read GetHeapPool;
  end;

{$EndRegion}

{$Region 'Subroutines'}

// Check the index entry into the range [0...Count - 1].
procedure CheckIndex(Index, Count: Integer); inline;
// if not ok raise error
procedure Check(ok: Boolean; const Msg: string = '');
// raise fatal error
procedure FatalError(const Msg: string);

{$EndRegion}

implementation

type
  PBytes = ^TBytes;
  PInterface = ^IInterface;

{$Region 'Subroutines'}

procedure CheckIndex(Index, Count: Integer); inline;
begin
  if Cardinal(Index) >= Cardinal(Count) then
    raise EsgError.Create(EsgError.ListIndexError, Index);
end;

procedure Check(ok: Boolean; const Msg: string = '');
begin
  if ok then
    exit;
  if Msg = '' then
    raise EsgError.Create('Check error')
  else
    raise EsgError.Create(Msg);
end;

procedure FatalError(const Msg: string);
begin
  raise EsgError.Create(Msg);
end;

{$EndRegion}

{$Region 'EsgError'}

constructor EsgError.Create(ErrNo: Integer);
var
  Msg: string;
begin
  if InRange(ErrNo, 0, ErrorMax) then
    Msg := ErrorMessages[ErrNo]
  else
    Msg := 'Error: ' + IntToStr(ErrNo);
  Create(Msg);
end;

constructor EsgError.Create(ErrNo, IntParam: Integer);
var
  Msg: string;
begin
  if InRange(ErrNo, 0, ErrorMax) then
    Msg := ErrorMessages[ErrNo]
  else
    Msg := 'Error: ' + IntToStr(ErrNo);
  CreateFmt(Msg, [IntParam]);
end;

{$EndRegion}

{$Region 'hMeta'}

constructor hMeta.From(TypeKind: System.TTypeKind; ManagedType, HasWeakRef: Boolean);
var
  b: Integer;
begin
  b := Seed shl 9;
  if HasWeakRef then b := b or 1;
  b := b shl 1;
  if ManagedType then b := b or 1;
  b := b shl 6;
  b := b + Ord(TypeKind) and $1F;
  v := b;
end;

function hMeta.Valid: Boolean;
begin
  Result := SeedValue = Seed;
end;

function hMeta.GetTypeKind: System.TTypeKind;
begin
  Result := System.TTypeKind(MetaFlags and $1F);
end;

function hMeta.GetManagedType: Boolean;
begin
  Result := False;
  if v and $40 <> 0 then
    Result := True;
end;

function hMeta.GetHasWeakRef: Boolean;
begin
  Result := False;
  if v and $80 <> 0 then
    Result := True;
end;

function hMeta.GetSegmented: Boolean;
begin
  Result := False;
  if v and $8000 <> 0 then
    Result := True;
end;

procedure hMeta.SetSegmented(const Value: Boolean);
begin
  if Value then
    v := v or $8000
  else
    v := v and not $8000;
end;

function hMeta.GetRangeCheck: Boolean;
begin
  Result := False;
  if v and $100 <> 0 then
    Result := True;
end;

procedure hMeta.SetRangeCheck(const Value: Boolean);
begin
  if Value then
    v := v or $100
  else
    v := v and not $100;
end;

function hMeta.GetNotification: Boolean;
begin
  Result := False;
  if v and $200 <> 0 then
    Result := True;
end;

procedure hMeta.SetNotification(const Value: Boolean);
begin
  if Value then
    v := v or $200
  else
    v := v and not $200;
end;

function hMeta.GetOwnedObject: Boolean;
begin
  Result := False;
  if v and $400 <> 0 then
    Result := True;
end;

procedure hMeta.SetOwnedObject(const Value: Boolean);
begin
  if Value then
    v := v or $400
  else
    v := v and not $400;
end;

function hMeta.GetRemoveAction: TRemoveAction;
begin
  Result := TRemoveAction((v shr 11) and $3);
end;

procedure hMeta.SetRemoveAction(const Value: TRemoveAction);
begin
  v := (v and not $1800) or ((Ord(Value) and $3) shl 11);
end;

{$EndRegion}

{$Region 'TsgItemMeta'}

procedure TsgItemMeta.Init<T>(OnFree: TFreeProc);
begin
  TypeInfo := System.TypeInfo(T);
  h := hMeta.From(System.GetTypeKind(T), System.IsManagedType(T), System.HasWeakRef(T));
  ItemSize := sizeof(T);
  Self.OnFree := OnFree;
  InitMethods;
end;

procedure TsgItemMeta.Init<T>(Flags: TRegionFlagSet; RemoveAction: TRemoveAction;
  OnFree: TFreeProc);
begin
  TypeInfo := System.TypeInfo(T);
  h := hMeta.From(System.GetTypeKind(T), System.IsManagedType(T), System.HasWeakRef(T));
  ItemSize := sizeof(T);
  Self.OnFree := OnFree;
  if rfSegmented in Flags then
    h.SetSegmented(True);
  if rfRangeCheck in Flags then
    h.SetRangeCheck(True);
  if rfNotification in Flags then
    h.SetNotification(True);
  if rfOwnedObject in Flags then
    h.SetOwnedObject(True);
  h.RemoveAction := RemoveAction;
  InitMethods;
end;

procedure Assign1(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  PByte(Dest)^ := PByte(Value)^;
end;

procedure Assign2(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  PWord(Dest)^ := PWord(Value)^;
end;

procedure Assign4(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  PCardinal(Dest)^ := PCardinal(Value)^;
end;

procedure Assign8(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  PUInt64(Dest)^ := PUInt64(Value)^;
end;

procedure AssignItemValue(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  Move(Value^, Dest^, meta.ItemSize);
end;

procedure AssignManaged(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  System.CopyRecord(Dest, Value, meta.TypeInfo);
end;

procedure AssignVariant(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  PVariant(Dest)^ := PVariant(Value)^;
end;

procedure AssignMRef(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  case meta.h.TypeKind of
    TTypeKind.tkUString: PString(Dest)^ := PString(Value)^;
    TTypeKind.tkDynArray: PBytes(Dest)^ := PBytes(Value)^;
    TTypeKind.tkInterface: PInterface(Dest)^ := PInterface(Value)^;
{$IF Defined(AUTOREFCOUNT)}
    TTypeKind.tkClass: PObject(Dest)^ := PObject(Value)^;
{$ENDIF}
    TTypeKind.tkLString: PRawByteString(Dest)^ := PRawByteString(Value)^;
{$IF not Defined(NEXTGEN)}
    TTypeKind.tkWString: PWideString(Dest)^ := PWideString(Value)^;
{$ENDIF}
  end;
end;

procedure UDFree(meta: PsgItemMeta; p: Pointer);
begin
  meta.OnFree(p);
end;

procedure FreeManaged(meta: PsgItemMeta; p: Pointer);
begin
  FinalizeRecord(p, meta.TypeInfo);
end;

procedure FreeMRef(meta: PsgItemMeta; p: Pointer);
begin
  case meta.h.TypeKind of
    TTypeKind.tkUString: PString(p)^ := '';
    TTypeKind.tkDynArray: PBytes(p)^ := nil;
    TTypeKind.tkInterface: PInterface(p)^ := nil;
{$IF Defined(AUTOREFCOUNT)}
    TTypeKind.tkClass: PObject(p)^ := nil;
{$ENDIF}
    TTypeKind.tkLString: PRawByteString(p)^ := '';
{$IF not Defined(NEXTGEN)}
    TTypeKind.tkWString: PWideString(p)^ := '';
{$ENDIF}
  end;
end;

procedure FreeVariant(meta: PsgItemMeta; p: Pointer);
begin
  PVariant(p)^ := 0;
end;

procedure TsgItemMeta.InitMethods;
begin
  if h.ManagedType then
  begin
    if (ItemSize = SizeOf(Pointer)) and not h.HasWeakRef and not (h.TypeKind in [tkRecord, tkMRecord]) then
    begin
      Self.AssignItem := AssignMRef;
      if Assigned(OnFree) then
        Self.FreeItem := UDFree
      else
        Self.FreeItem := FreeMRef;
    end
    else if h.TypeKind = TTypeKind.tkVariant then
    begin
      Self.AssignItem := AssignVariant;
      if Assigned(OnFree) then
        Self.FreeItem := UDFree
      else
        Self.FreeItem := FreeVariant;
    end
    else
    begin
      Self.AssignItem := AssignManaged;
      if Assigned(OnFree) then
        Self.FreeItem := UDFree
      else
        Self.FreeItem := FreeManaged;
    end;
  end
  else
  begin
    case ItemSize of
      0: raise EsgError.Create(EsgError.InvalidParameters);
      1: Self.AssignItem := Assign1;
      2: Self.AssignItem := Assign2;
      4: Self.AssignItem := Assign4;
      8: Self.AssignItem := Assign8;
      else Self.AssignItem := AssignItemValue;
    end;
    if Assigned(OnFree) then
      Self.FreeItem := UDFree
    else
      Self.FreeItem := nil;
  end;
end;

{$EndRegion}

{$Region 'TsgMeta'}

procedure TsgMeta.Init<T>(FreeItem: TFreeItem);
begin
end;

procedure TsgMeta.InitMethods;
begin
end;

{$EndRegion}

{$Region 'TsgItem'}

procedure TsgItem.Init<T>(const Region: TMemoryRegion; var Value: T);
begin
  Self.Region := @Region;
  if System.TypeInfo(T) <> Region.FMeta.TypeInfo then
    raise EsgError.Create(EsgError.IncompatibleDataType);
  Ptr := @Value;
end;

procedure TsgItem.Assign(const Value);
begin
  Region.AssignItem(Region.Meta, Ptr, @Value);
end;

procedure TsgItem.Free;
begin
  Region.FreeItem(Region.Meta, Ptr);
end;

{$EndRegion}

{$Region 'TMemSegment'}

class procedure TMemSegment.NewSegment(var s: PMemSegment; HeapSize: Cardinal);
begin
  // Create a new memory segment
  GetMem(s, HeapSize);
  s.Init(HeapSize);
end;

class procedure TMemSegment.IncreaseHeapSize(var s: PMemSegment; NewHeapSize: Cardinal);
var
  Offset: NativeUInt;
begin
  Check((s <> nil) and (NewHeapSize > s.GetHeapSize), 'IncreaseHeapSize error');
  Offset := NativeUInt(s.FreePtr) - NativeUInt(s);
  ReallocMem(s, NewHeapSize);
  // Increase memory segment size
  s.FreePtr := PByte(s) + Offset;
  s.TopMem := PByte(s) + NewHeapSize;
  FillChar(s.FreePtr^, s.GetFreeSize, 0);
end;

procedure TMemSegment.Init(HeapSize: Cardinal);
begin
  Next := nil;
  FreePtr := PByte(@Self) + sizeof(TMemSegment);
  TopMem := PByte(@Self) + HeapSize;
  FillChar(FreePtr^, GetFreeSize, 0);
end;

function TMemSegment.Occupy(Size: Cardinal): Pointer;
var
  p: PByte;
begin
  p := FreePtr + Size;
  // if there is not enough memory
  if p > TopMem then
    exit(nil);
  Result := FreePtr;
  FreePtr := p;
{$IFDEF DEBUG}
  CheckPointer(Result, Size);
{$ENDIF}
end;

procedure TMemSegment.CheckPointer(Ptr: Pointer; Size: Cardinal);
var
  lo: NativeUInt;
begin
  lo := NativeUInt(@Self) + sizeof(TMemSegment);
  Check(InRange(NativeUInt(Ptr), lo, NativeUInt(TopMem)));
end;

function TMemSegment.GetFreeSize: Cardinal;
begin
{$IFDEF DEBUG}
  Assert(NativeUInt(TopMem) >= NativeUInt(FreePtr));
{$ENDIF}
  Result := TopMem - FreePtr;
end;

function TMemSegment.GetHeapRef: PByte;
begin
  Result := PByte(@Self) + sizeof(TMemSegment);
end;

function TMemSegment.GetHeapSize: Cardinal;
begin
{$IFDEF DEBUG}
  Assert(NativeUInt(TopMem) >= NativeUInt(@Self));
{$ENDIF}
  Result := TopMem - PByte(@Self);
end;

function TMemSegment.GetOccupiedSize: Cardinal;
begin
{$IFDEF DEBUG}
  Assert(NativeUInt(FreePtr) >= NativeUInt(GetHeapRef));
{$ENDIF}
  Result := FreePtr - GetHeapRef;
end;

{$EndRegion}

{$Region 'TMemoryRegion'}

procedure TMemoryRegion.Init(Meta: PsgItemMeta; BlockSize: Cardinal);
begin
  FillChar(Self, sizeof(TMemoryRegion), 0);
  Self.FMeta := Meta^;
  Self.BlockSize := BlockSize;
  Self.FCapacity := 0;
  Self.FCount := 0;
  Self.Heap := nil;
end;

procedure TMemoryRegion.Free;
begin
  if FTemporary <> nil then
  begin
    FreeMem(FTemporary);
    FTemporary := nil;
  end;
  FreeHeap(Heap);
  FCount := 0;
end;

procedure TMemoryRegion.Clear;
begin
  // Clear managed fields
  if Assigned(FreeItem) then
    FreeItems(Heap);
  // Clear the memory of all segments.
  // Return to the heap memory of all segments except the first.
  FreeHeap(Heap.Next);
  // Determine the size of free memory
  Heap.Init(Heap.GetHeapSize);
  FCount := 0;
end;

function TMemoryRegion.Valid: Boolean;
begin
  Result := FMeta.h.Valid;
end;

procedure TMemoryRegion.FreeHeap(var Heap: PMemSegment);
var
  p, q: PMemSegment;
begin
  p := Heap;
  while p <> nil do
  begin
    q := p.Next;
    if Assigned(FreeItem) then
      FreeItems(p);
    FreeMem(p);
    p := q;
  end;
  Heap := nil;
end;

procedure TMemoryRegion.FreeItems(p: PMemSegment);
var
  N: Integer;
  Ptr: Pointer;
  a, b: NativeUInt;
begin
  Ptr := p.GetHeapRef;
  N := GetOccupiedCount(p);
  while N > 0 do
  begin
    FreeItem(@FMeta, Ptr);
    a := NativeUInt(Ptr);
    Ptr := Pointer(NativeUInt(Ptr) + FMeta.ItemSize);
    b := NativeUInt(Ptr);
    Check(a + FMeta.ItemSize = b);
    Dec(N);
  end;
end;

function TMemoryRegion.IncreaseCapacity(NewCount: Integer): Pointer;
begin
  GrowHeap(NewCount);
  Result := Heap.GetHeapRef;
end;

function TMemoryRegion.IncreaseAndAlloc(NewCount: Integer): Pointer;
var
  Old, Size: Integer;
begin
  Old := Capacity;
  Result := IncreaseCapacity(NewCount);
  Size := (Capacity - Old) * Integer(FMeta.ItemSize);
  Alloc(Size);
end;

function TMemoryRegion.GetOccupiedCount(p: PMemSegment): Integer;
begin
  Result := p.GetOccupiedSize div FMeta.ItemSize;
end;

procedure TMemoryRegion.GrowHeap(NewCount: Integer);
var
  BlockCount, Size, NewHeapSize: Cardinal;
  s: PMemSegment;
begin
  Size := Grow(NewCount) * Integer(FMeta.ItemSize);
  BlockCount := (Size + sizeof(TMemoryRegion)) div BlockSize + 1;
  NewHeapSize := BlockCount * BlockSize;
  if Heap = nil then
    // create a new segment
    TMemSegment.NewSegment(Heap, NewHeapSize)
  else if not FMeta.h.Segmented then
    // increase the size of the memory segment
    TMemSegment.IncreaseHeapSize(Heap, NewHeapSize)
  else
  begin
    // create a new segment and place it at the beginning of the list
    TMemSegment.NewSegment(s, NewHeapSize);
    s.Next := Heap;
    Heap := s;
  end;
  FCapacity := (Heap.GetHeapSize - sizeof(TMemSegment)) div FMeta.ItemSize;
end;

function TMemoryRegion.Grow(NewCount: Integer): Integer;
begin
  Result := Capacity;
  repeat
    if Result > 64 then
      Result := (Result * 3) div 2
    else
      Result := Result + 16;
    if Result < 0 then
      OutOfMemoryError;
  until Result >= NewCount;
end;

function TMemoryRegion.Alloc(Size: Cardinal): Pointer;
begin
{$IFDEF DEBUG}
  Check(Valid and (Size mod FMeta.ItemSize = 0));
{$ENDIF}
  if (Heap = nil) or (Heap.GetFreeSize < Size) then
    GrowHeap(Size div FMeta.ItemSize + 1);
  Result := Heap.Occupy(Size);
  if Result = nil then
    OutOfMemoryError;
end;

function TMemoryRegion.GetItemPtr(Index: Cardinal): Pointer;
begin
  Result := Heap.GetHeapRef + Index * FMeta.ItemSize;
end;

function TMemoryRegion.GetMeta: PsgItemMeta;
begin
  Result := @FMeta;
end;

function TMemoryRegion.GetTemporary: Pointer;
begin
  if FTemporary = nil then
  begin
    FTemporary := AllocMem(ItemSize);
    FillChar(FTemporary^, ItemSize, 0);
  end;
  Result := FTemporary;
end;

{$EndRegion}

{$Region 'TUnbrokenRegion'}

procedure TUnbrokenRegion.Init(Meta: PsgItemMeta; BlockSize: Cardinal);
begin
  FRegion.Init(Meta, BlockSize);
end;

procedure TUnbrokenRegion.InitWithCount(Meta: PsgItemMeta; Count: Cardinal);
var
  BlockSize: Cardinal;
begin
  BlockSize := Meta.ItemSize * Count + 1;
  FRegion.Init(Meta, BlockSize);
  SetCount(Count);
end;

procedure TUnbrokenRegion.Free;
begin
  FRegion.Free;
end;

procedure TUnbrokenRegion.Clear;
begin
  FRegion.Clear;
end;

function TUnbrokenRegion.GetRegion: PMemoryRegion;
begin
  Result := @FRegion;
end;

procedure TUnbrokenRegion.SetCount(NewCount: Integer);
begin
  if NewCount <> FRegion.FCount then
  begin
    if Capacity <= NewCount then
      FRegion.GrowHeap(NewCount);
    FRegion.FCount := NewCount;
  end;
end;

function TUnbrokenRegion.AddItem: PByte;
begin
  Inc(FRegion.FCount);
  if Capacity <= FRegion.FCount then
    FRegion.GrowHeap(FRegion.FCount);
  Result := FRegion.Heap.Occupy(ItemSize);
end;

procedure TUnbrokenRegion.Insert(Index: Integer; const Value);
var
  MemSize: Integer;
  Dest, Source: PByte;
begin
  CheckIndex(Index, Count + 1);
  AddItem;
  Source := GetItems + Cardinal(Index) * ItemSize;
  if Index <> Count - 1 then
  begin
    MemSize := Cardinal(Count - Index - 1) * ItemSize;
    Dest := Source + ItemSize;
    System.Move(Source^, Dest^, MemSize);
  end;
  FRegion.AssignItem(@FRegion.FMeta, Source, @Value);
end;

procedure TUnbrokenRegion.Delete(Index: Integer);
var
  MemSize: Integer;
  Dest, Source: PByte;
begin
  CheckIndex(Index, FRegion.FCount);
  Dec(FRegion.FCount);
  if Index < FRegion.FCount then
  begin
    MemSize := Cardinal(FRegion.FCount - Index) * ItemSize;
    Dest := GetItems + Cardinal(Index) * ItemSize;
    Source := Dest + ItemSize;
    System.Move(Source^, Dest^, MemSize);
  end;
end;

procedure TUnbrokenRegion.Exchange(Index1, Index2: Integer);
var
  ItemSize: Integer;
  DTemp: PByte;
  PTemp: PByte;
  Items: Pointer;
  STemp: array [0..255] of Byte;
begin
  ItemSize := FRegion.Meta.ItemSize;
  DTemp := nil;
  PTemp := @STemp[0];
  Items := FRegion.GetItemPtr(0);
  try
    if ItemSize > sizeof(STemp) then
    begin
      GetMem(DTemp, ItemSize);
      PTemp := DTemp;
    end;
    Move(PByte(Items)[Index1 * ItemSize], PTemp[0], ItemSize);
    Move(PByte(Items)[Index2 * ItemSize], PByte(Items)[Index1 * ItemSize], ItemSize);
    Move(PTemp[0], PByte(Items)[Index2 * ItemSize], ItemSize);
  finally
    FreeMem(DTemp);
  end;
end;

procedure TUnbrokenRegion.CopyFrom(const Region: PUnbrokenRegion;
  Index, Cnt: Integer);
var
  Dest, Value: PByte;
begin
  while Cnt > 0 do
  begin
    Value := Region.GetItemPtr(Index);
    Dest := AddItem;
    FRegion.AssignItem(@FRegion.FMeta, Dest, Value);
    Dec(Cnt);
  end;
end;

function TUnbrokenRegion.GetCount: Integer;
begin
  Result := FRegion.FCount;
end;

function TUnbrokenRegion.GetItems: PByte;
begin
  Result := FRegion.Heap.GetHeapRef;
end;

function TUnbrokenRegion.GetItemPtr(Index: Cardinal): Pointer;
begin
  CheckIndex(Index, FRegion.FCount);
  Result := FRegion.GetItemPtr(Index);
end;

function TUnbrokenRegion.NextItem(Item: Pointer): Pointer;
var
  p: NativeUInt;
begin
  p := NativeUInt(Item) - NativeUInt(FRegion.Heap.GetHeapRef);
  if p < FRegion.Heap.GetHeapSize then
    Result := Pointer(NativeUInt(Item) + FRegion.FMeta.ItemSize)
  else
    Result := nil;
end;

function TUnbrokenRegion.GetMeta: PsgItemMeta;
begin
  Result := @FRegion.FMeta;
end;

procedure TUnbrokenRegion.AssignItem(Dest, Src: Pointer);
begin
  FRegion.AssignItem(@FRegion.FMeta, Dest, Src);
end;

{$EndRegion}

{$Region 'TSegmentedRegion'}

procedure TSegmentedRegion.Init(Meta: PsgItemMeta; BlockSize: Cardinal);
begin
  FRegion.Init(Meta, BlockSize);
end;

procedure TSegmentedRegion.Free;
begin
  FRegion.Free;
end;

function TSegmentedRegion.AddItem: Pointer;
begin
  Inc(FRegion.FCount);
  if FRegion.Capacity <= FRegion.FCount then
    FRegion.GrowHeap(FRegion.FCount);
  Result := FRegion.Heap.Occupy(ItemSize);
end;

function TSegmentedRegion.GetRegion: PMemoryRegion;
begin
  Result := @FRegion;
end;

function TSegmentedRegion.GetCapacity: Integer;
begin
  Result := FRegion.Capacity;
end;

function TSegmentedRegion.GetCount: Integer;
begin
  Result := FRegion.FCount;
end;

procedure TSegmentedRegion.Dispose(Items: Pointer; Count: Cardinal);
begin

end;

function TSegmentedRegion.GetItemPtr(Index: Cardinal): Pointer;
var
  n: Cardinal;
  s: PMemSegment;
begin
  s := FRegion.Heap;
  while s <> nil do
  begin
    n := FRegion.GetOccupiedCount(s);
    if Index < n then
      exit(s.GetHeapRef + Index * FRegion.FMeta.ItemSize);
    Dec(Index, n);
    s := s.Next;
  end;
  Result := nil;
end;

function TSegmentedRegion.GetMeta: PsgItemMeta;
begin
  Result := @FRegion.FMeta;
end;

procedure TSegmentedRegion.AssignItem(Dest, Src: Pointer);
begin
  FRegion.AssignItem(@FRegion.FMeta, Dest, Src);
end;

{$EndRegion}

{$Region 'TRegionItems'}

procedure TRegionItems.Init;
begin
  root := nil;
end;

procedure TRegionItems.Add(p: PRegionItem);
begin
  p.Next := root;
  root := p;
end;

function TRegionItems.Remove: PRegionItem;
var
  p: PRegionItem;
begin
  p := root;
  root := p.Next;
  p.Next := nil;
  Result := p;
end;

function TRegionItems.Empty: Boolean;
begin
  Result := root = nil;
end;

{$EndRegion}

{$Region 'THeapPool'}

procedure FreeRegion(Ptr: Pointer);
var
  Item: PRegionItem;
begin
  Item := PRegionItem(Ptr);
  Item.r.FreeHeap(Item.r.Heap);
end;

constructor THeapPool.Create(BlockSize: Word);
begin
  inherited Create;
  FBlockSize := BlockSize;
  New(FRegions);
  FRegions.Init(@TsgContext.MemoryRegionMeta, BlockSize);
  FRealesed.Init;
end;

destructor THeapPool.Destroy;
begin
  if FRegions <> nil then
  begin
    FRegions.Free;
    Dispose(FRegions);
    FRegions := nil;
  end;
  inherited;
end;

function THeapPool.CreateUnbrokenRegion(Meta: PsgItemMeta): PUnbrokenRegion;
begin
  Meta.h.SetSegmented(False);
{$IFDEF DEBUG}
  Assert(sizeof(TUnbrokenRegion) = sizeof(TMemoryRegion));
{$ENDIF}
  Result := PUnbrokenRegion(FindOrCreateRegion(Meta));
end;

function THeapPool.CreateRegion(Meta: PsgItemMeta): PSegmentedRegion;
begin
  Meta.h.SetSegmented(True);
{$IFDEF DEBUG}
  Assert(sizeof(TSegmentedRegion) = sizeof(TMemoryRegion), '');
{$ENDIF}
  Result := PSegmentedRegion(FindOrCreateRegion(Meta));
end;

function THeapPool.FindOrCreateRegion(Meta: PsgItemMeta): PMemoryRegion;
var
  p: PRegionItem;
begin
  if not FRealesed.Empty then
    p := FRealesed.Remove
  else
    p := FRegions.Alloc(sizeof(TMemoryRegion));
  p.r.Init(Meta, FBlockSize);
  Result := @p.r;
end;

procedure THeapPool.Release(r: PMemoryRegion);
begin
  try
    r.FreeHeap(r.Heap);
  except
  end;
end;

{$EndRegion}

{$Region 'TsgMemoryManager'}

procedure TsgMemoryManager.Init(Heap: Pointer; HeapSize: Cardinal);
begin
  if Heap = nil then
    raise EsgError.Create(EsgError.InvalidParameters);
  Avail := Heap;
  Avail^.Next := nil;
  HeapSize := (HeapSize + 3) and not 3;
  Avail^.Size := HeapSize;
  Self.Heap := Heap;
  TopMemory := Pointer(NativeUInt(Heap) + HeapSize);
end;

function TsgMemoryManager.Alloc(Size: Cardinal): Pointer;
var
  p, q: PsgFreeBlock;
begin
{$IFDEF DEBUG}
  Assert(Avail <> nil);
{$ENDIF}
  // Align block size to 4 bytes.
  Size := (Size + 3) and not 3;
  if Size = 0 then
    raise EsgError.Create(EsgError.InvalidSize);
  p := PsgFreeBlock(@Avail);
  repeat
    q := p^.Next;
    if q = nil then exit(nil);
    if Size = q.Size then
    begin
      p^.Next := q^.Next;
      exit(q);
    end;
    if Size < q.Size then
    begin
      p^.Next := PsgFreeBlock(PByte(p^.Next) + Size);
      p := p^.Next;
      p^.Next := q^.Next;
      p^.Size := q^.Size - Size;
      exit(q);
    end;
    p := q;
  until False;
end;

function TsgMemoryManager.Realloc(Ptr: Pointer;
  OldSize, Size: Cardinal): Pointer;
var
  delta: Cardinal;
  p, q, r, n: PsgFreeBlock;
begin
  OldSize := (OldSize + 3) and not 3;
  Size := (Size + 3) and not 3;
  if (OldSize >= Size) or (Size mod MinSize <> 0) then
    raise EsgError.Create(EsgError.InvalidSize);
  n := nil;
  // look for a block with the address Ptr + OldSize in the free memory list
  r := PsgFreeBlock(NativeUInt(Ptr) + OldSize);
  p := PsgFreeBlock(@Avail);
  repeat
    q := p^.Next;
    if q = nil then
    begin
      // If we were unable to increase the transferred block of memory,
      // then we take a new block of memory
      if n <> nil then
      begin
        q := n.Next;
        if Size = q.Size then
          n^.Next := q^.Next
        else if Size < q.Size then
        begin
          n^.Next := PsgFreeBlock(PByte(n^.Next) + Size);
          n := n^.Next;
          n^.Next := q^.Next;
          n^.Size := q^.Size - Size;
        end;
        // and copy the values into it and delete the old block
        Move(Ptr^, q^, OldSize);
        FreeMem(Ptr, OldSize);
      end;
      Result := q;
      exit;
    end;
    if q = r then
    begin
      // is there the desired piece of memory
      delta := Size - OldSize;
      if delta = q.Size then
      begin
        Result := Ptr;
        p^.Next := q^.Next;
        break;
      end;
      if delta < q.Size then
      begin
        Result := Ptr;
        p^.Next := PsgFreeBlock(PByte(p^.Next) + delta);
        p := p^.Next;
        p^.Next := q^.Next;
        p^.Size := q^.Size - delta;
        break;
      end;
    end
    // If it is a suitable block, remember the pointer preceding it.
    else if (n = nil) and (Size <= q.Size) then
      n := p;
    p := q;
  until False;
  FillChar(r^, delta, 0);
end;

procedure TsgMemoryManager.FreeMem(Ptr: Pointer; Size: Cardinal);
var
  q, p, x: PsgFreeBlock;
begin
  p := Ptr;
  // Align block size to 4 bytes.
  p^.Size := (Size + 3) and not 3;
  q := PsgFreeBlock(@Avail);
  repeat
    x := q^.Next;
    if (x = nil) or (NativeUInt(p) <= NativeUInt(x)) then
    begin
      p^.Next := x;
      break;
    end;
    q := q^.Next;
  until False;
  q^.Next := p;
  // Combine two blocks into one.
  x := p^.Next;
  if p^.Size + NativeUInt(p) = NativeUInt(x) then
  begin
    p^.Next := x^.Next;
    Inc(p^.Size, x^.Size);
  end;
  p := q;
  // Combine two blocks into one.
  x := p^.Next;
  if p^.Size + NativeUInt(p) = NativeUInt(x) then
  begin
    p^.Next := x^.Next;
    Inc(p^.Size, x^.Size);
  end;
end;

procedure TsgMemoryManager.Dealloc(Ptr: Pointer; Size: Cardinal);
begin
  if (Ptr = nil) or
     (NativeUInt(Ptr) < NativeUInt(Heap)) or
     (NativeUInt(Ptr) > NativeUInt(TopMemory)) then
    raise EsgError.Create(EsgError.InvalidPointer);
  FreeMem(Ptr, Size);
end;

{$EndRegion}

{$Region 'TsgContext'}

constructor TsgContext.Create;
begin
  inherited;
  FHeapPool := THeapPool.Create;
end;

destructor TsgContext.Destroy;
begin
  ClearHeapPool;
  inherited;
end;

class procedure TsgContext.InitMetadata;
begin
  if FPointerMeta.TypeInfo <> nil then exit;
  PointerMeta.Init<Pointer>;
  MemoryRegionMeta.Init<TMemoryRegion>([rfSegmented],
    TRemoveAction.HoldValue, FreeRegion);
end;

function TsgContext.CreateRegion(Meta: PsgItemMeta): PSegmentedRegion;
begin
  Result := FHeapPool.CreateRegion(Meta);
end;

function TsgContext.CreateUnbrokenRegion(Meta: PsgItemMeta): PUnbrokenRegion;
begin
  Result := FHeapPool.CreateUnbrokenRegion(Meta);
end;

procedure TsgContext.ClearHeapPool;
begin
  FreeAndNil(FHeapPool);
end;

function TsgContext.GetHeapPool: THeapPool;
begin
  if FHeapPool = nil then
    FHeapPool := THeapPool.Create;
  Result := FHeapPool;
end;

procedure TsgContext.Release(r: PMemoryRegion);
begin

end;

{$EndRegion}

end.

