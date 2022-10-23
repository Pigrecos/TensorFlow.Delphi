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

unit Oz.SGL.Collections;

interface

uses
  System.Classes, System.SysUtils, System.Math,
  Oz.SGL.Heap, Oz.SGL.HandleManager, Oz.SGL.Hash;

{$T+}

{$Region 'Forward declaration'}
type
  PSharedRegion = ^TSharedRegion;

{$EndRegion}

{$Region 'TObjectHeap'}

  TObjectHeap<T> = record
  type
    PItem = ^T;
  var
    region: TSegmentedRegion;
  public
    constructor From(var meta: PsgItemMeta; BlockSize: Cardinal);
    function Get(Clear: Boolean): PItem;
  end;

{$EndRegion}

{$Region 'TsgArray<T>: Generic Array with memory allocation from a shared memory region'}

  TMemoryDescriptor = record
    h: hCollection;
    Count: Cardinal;
    Items: PByte;
    procedure Clear;
  end;

  PsgArrayHelper = ^TsgArrayHelper;
  TsgArrayHelper = record
  private
    FDesc: TMemoryDescriptor;
    FRegion: PSharedRegion;
    FCount: Cardinal;
  public
    procedure Init(Region: PSharedRegion; Capacity: Cardinal);
    procedure Free;
    procedure Grow;
    procedure SetCapacity(NewCapacity: Cardinal);
    procedure SetCount(NewCount: Cardinal);
    function Add: PByte;
    function Insert(Index: Cardinal): PByte;
    function GetItem(Index: Cardinal): PByte; inline;
  end;

  TsgArray<T> = record
  type
    PItem = ^T;
  private
    FList: TsgArrayHelper;
    function GetItem(Index: Cardinal): PItem; inline;
    procedure SetCount(NewCount: Cardinal); inline;
  public
    procedure Init(Region: PSharedRegion; Capacity: Cardinal);
    procedure Free; inline;
    function Insert(Index: Cardinal): PItem; inline;
    function Add: PItem; inline;
    property Count: Cardinal read FList.FCount write SetCount;
    property Items[Index: Cardinal]: PItem read GetItem;
  end;

{$EndRegion}

{$Region 'TsgTupleElementMeta: Meta for tuple element'}

  PsgTupleElementMeta = ^TsgTupleElementMeta;
  TsgTupleElementMeta = record
  const
    // Align tuple element to the word boundary
    AllignTuple = sizeof(Pointer);
  private
    FOffset: Cardinal;
    FMeta: TsgItemMeta;
    function GetMeta: PsgItemMeta;
    function GetFreeItem: TFreeItem;
    function GetAssignItem: TAssignProc;
    function GetItemSize: Cardinal;
  public
    procedure Init<T>;
    // Determine the offset to the start of the next tuple element.
    function NextTupleOffset(Allign: Boolean): Cardinal;
    // The offset of an element in a tuple
    property Offset: Cardinal read FOffset write FOffset;
    // Tuple element metadata
    property Meta: PsgItemMeta read GetMeta;
    // Free tuple element
    property Free: TFreeItem read GetFreeItem;
    // Dest^ := Value^; - Assign tuple element
    property Assign: TAssignProc read GetAssignItem;
    property Size: Cardinal read GetItemSize;
  end;

{$EndRegion}

{$Region 'TsgTupleMeta: Tuple metadata'}

  PsgTupleMeta = ^TsgTupleMeta;
  TsgTupleMeta = record
  type
    TsgTeMetaList = TsgArray<TsgTupleElementMeta>;
    PsgTeMetaList = ^TsgTeMetaList;
    TAssignTuple = procedure(Dest, Value: Pointer) of object;
  private
    FSize: Cardinal;
    FElements: TsgTeMetaList;
    FOnFree: TFreeProc;
    FAssignTuple: TAssignTuple;
    procedure AssignTuple(Dest, Value: Pointer);
    procedure MoveTuple(Dest, Value: Pointer);
    procedure CheckManaged;
    // Add tuple element
    procedure AddElement(meta: PsgTupleElementMeta; Allign: Boolean);
    procedure AddTe<T>(Allign: Boolean);
    procedure Init(Count: Cardinal; OnFree: TFreeProc);
  public
    procedure MakePair<T1, T2>(OnFree: TFreeProc = nil;
      Allign: Boolean = True);
    procedure MakeTrio<T1, T2, T3>(OnFree: TFreeProc = nil;
      Allign: Boolean = True);
    procedure MakeQuad<T1, T2, T3, T4>(OnFree: TFreeProc = nil;
      Allign: Boolean = True);
    // Creates a tuple by concatenating
    procedure Cat(const Tuple: TsgTupleMeta; OnFree: TFreeProc = nil;
      Allign: Boolean = True);
    // Add metadata for a tuple element to the end of tuple.
    procedure Add<T>(OnFree: TFreeProc = nil; Allign: Boolean = True);
    // Insert metadata for the tuple element at the start of tuple.
    procedure Insert<T>(OnFree: TFreeProc = nil; Allign: Boolean = True);
    // Return a reference to the meta element of the tuple
    function Get(Index: Cardinal): PsgTupleElementMeta; inline;
    // Assign tuple
    property Assign: TAssignTuple read FAssignTuple;
    // Memory size of all elements
    property Size: Cardinal read FSize;
    // Number of elements
    property Count: Cardinal read FElements.FList.FCount;
    // meta elements of the tuple
    property Elements[Index: Cardinal]: PsgTupleElementMeta read Get;
  end;

{$EndRegion}

{$Region 'TsgTupleElement: Tuple element'}

  TsgTupleElement = record
  private
    Ptr: Pointer;
    TeMeta: PsgTupleElementMeta;
  public
    // Assign value
    procedure Assign(pvalue: Pointer); inline;
    // Return a reference to the element value of the tuple
    function GetPvalue: Pointer; inline;
  end;
  TsgTupleElements = TsgArray<TsgTupleElement>;

{$EndRegion}

{$Region 'TsgTuple: Tuple of the type defined by generic types'}

  PsgTuple = ^TsgTuple;
  TsgTuple = record
  private
    FPtr: Pointer;
    FTupleMeta: TsgTupleMeta;
  public
    // creates a proxy for working with a tuple
    procedure MakeTuple(Ptr: Pointer; const TupleMeta: TsgTupleMeta);
    // swap the contents of two tuples
    procedure Swap(Pvalue: Pointer);
    // return a reference to the element of the tuple
    function Get(Index: Integer): TsgTupleElement;
    // return a reference to the element of the tuple
    function Tie(Index: Integer): TsgTupleElements;
  end;

{$EndRegion}

{$Region 'TsgTuples: Tuple memory region'}

  PsgTuples = ^TsgTuples;
  TsgTuples = record
  private
    FRegion: PUnbrokenRegion;
    FCount: Cardinal;
    FTupleMeta: PsgTupleMeta;
    function GetItem(Index: Cardinal): PsgTuple;
    procedure CheckCapacity(NewCapacity: Integer); inline;
    procedure SetCount(NewCount: Cardinal);
  public
    procedure Init(const tupleMeta: TsgTupleMeta; Capacity: Cardinal;
      Flags: TRegionFlagSet = []);
    procedure Free;
    function Add: PsgTuple;
    property Count: Cardinal read FCount write SetCount;
    property Items[Index: Cardinal]: PsgTuple read GetItem; default;
  end;

{$EndRegion}

{$Region 'TsgList<T>: Generic List of Values'}

  PsgListHelper = ^TsgListHelper;
  TsgListHelper = record
  strict private
    function Compare(const Left, Right): Boolean;
  public type
    TEnumerator = record
    private
      FIndex: Integer;
      FList: PsgListHelper;
      function GetCurrent: Pointer; inline;
    public
      constructor From(Value: PsgListHelper);
      function MoveNext: Boolean;
      property Current: Pointer read GetCurrent;
    end;
  private
    FRegion: PUnbrokenRegion;
    function GetItems: PByte; inline;
    function GetCount: Integer; inline;
    procedure SetCount(NewCount: Integer);
    procedure QuickSort(Compare: TListSortCompareFunc; L, R: Integer);
  public
    procedure Init(Meta: PsgItemMeta);
    procedure Free;
    procedure Clear;
    function GetPtr(Index: Integer): Pointer;
    function Add: Pointer;
    procedure Delete(Index: Integer);
    procedure Insert(Index: Integer; const Value);
    function Remove(const Value): Integer;
    procedure Sort(Compare: TListSortCompare);
    procedure Exchange(Index1, Index2: Integer);
    procedure Reverse;
    procedure Assign(const Source: TsgListHelper);
  end;

  TsgList<T> = record
  private type
    TItems = array [0 .. High(Word)] of T;
    PItems = ^TItems;
    PItem = ^T;
  private
    FListHelper: TsgListHelper;
    function GetItems: PItems; inline;
    function GetCount: Integer; inline;
    function GetItem(Index: Integer): T;
    procedure SetItem(Index: Integer; const Value: T);
    procedure SetCount(Value: Integer); inline;
  public
    constructor From(Meta: PsgItemMeta; Capacity: Integer);
    procedure Free; inline;
    procedure Clear; inline;
    function Add(const Value: T): Integer; overload;
    function Add: PItem; overload; inline;
    procedure Delete(Index: Integer); inline;
    procedure Insert(Index: Integer; const Value: T); inline;
    function Remove(const Value: T): Integer; inline;
    procedure Exchange(Index1, Index2: Integer); inline;
    procedure Reverse; inline;
    procedure Sort(Compare: TListSortCompare);
    procedure Assign(Source: TsgList<T>); inline;
    function GetPtr(Index: Integer): PItem; inline;
    function IsEmpty: Boolean; inline;
    function GetEnumerator: TsgListHelper.TEnumerator; inline;
    property Count: Integer read GetCount write SetCount;
    property Items[Index: Integer]: T read GetItem write SetItem; default;
    property List: PItems read GetItems;
  end;

{$EndRegion}

{$Region 'TsgStack<T: record>'}

  TsgStack<T: record> = record
  strict private
    FList: TsgList<T>;
    function GetItem(Index: Integer): T; inline;
    function GetCount: Integer; inline;
  public
    constructor From(var Meta: PsgItemMeta; Capacity: Integer);
    procedure Free; inline;
    procedure Clear; inline;
    function Peek: T;
    procedure Push(Item: T);
    function Pop: T;
    function Empty: Boolean;
    property Count: Integer read GetCount;
    property Items[Index: Integer]: T read GetItem;
  end;

{$EndRegion}

{$Region 'TsgPointerArray: Untyped List of Pointers'}

  TsgPointersArrayRange = 0..$7FFFFFFF div (sizeof(Pointer) * 2) - 1;
  TsgPointers = array [TsgPointersArrayRange] of Pointer;
  PsgPointers = ^TsgPointers;

  // An array of pointers for quick sorting and searching.
  PsgPointerArray = ^TsgPointerArray;
  TsgPointerArray = record
  public type
    TEnumerator = record
    private
      FPointers: PsgPointerArray;
      FIndex: Integer;
      function GetCurrent: Pointer;
    public
      constructor From(const Pointers: TsgPointerArray);
      function MoveNext: Boolean;
      property Current: Pointer read GetCurrent;
    end;
  private
    // region for pointers
    FList: PsgPointers;
    FListRegion: PUnbrokenRegion;
    FCount: Integer;
    function Get(Index: Integer): Pointer;
    procedure Put(Index: Integer; Item: Pointer);
  public
    constructor From(Capacity: Integer);
    procedure Free;
    procedure Add(ptr: Pointer);
    procedure Sort(Compare: TListSortCompare);
    function GetEnumerator: TEnumerator;
    property Count: Integer read FCount;
    property Items[Index: Integer]: Pointer read Get write Put;
  end;

{$EndRegion}

{$Region 'TsgPointerList: Untyped List of Values accessed by pointer'}

  TItemFunc = reference to function(Item: Pointer): Boolean;

  PsgPointerList = ^TsgPointerList;
  TsgPointerList = record
  public type
    TEnumerator = record
    private
      FPointers: PsgPointerList;
      FIndex: Integer;
      function GetCurrent: Pointer;
    public
      constructor From(const Pointers: TsgPointerList);
      function MoveNext: Boolean;
      property Current: Pointer read GetCurrent;
    end;
  private
    // region for pointers
    FListRegion: PUnbrokenRegion;
    // region for items
    FItemsRegion: PSegmentedRegion;
    function Get(Index: Integer): Pointer;
    procedure Put(Index: Integer; Item: Pointer);
    procedure SetCount(NewCount: Integer);
    function GetCount: Integer;
  public
    constructor From(Meta: PsgItemMeta);
    procedure Free;
    procedure Clear;
    function First: Pointer; inline;
    function Last: Pointer; inline;
    function NextAfter(prev: Pointer): Pointer; inline;
    procedure Assign(const Source: TsgPointerList);
    function Add(Item: Pointer): Integer; overload;
    // Add an empty record and return its pointer
    function Add: Pointer; overload;
    // Insert element
    procedure Insert(Index: Integer; Item: Pointer);
    // Delete element
    procedure Delete(Index: Integer);
    // Exchange elements
    procedure Exchange(Index1, Index2: Integer);
    function IndexOf(Item: Pointer): Integer;
    procedure Sort(Compare: TListSortCompare);
    procedure Reverse;
    function TraverseBy(F: TItemFunc): Pointer;
    procedure RemoveBy(F: TItemFunc);
    function IsEmpty: Boolean; inline;
    function GetEnumerator: TEnumerator;
    property Count: Integer read GetCount write SetCount;
    property Items[Index: Integer]: Pointer read Get write Put;
  end;

{$EndRegion}

{$Region 'TsgRecordList<T>: Generic List of Values accessed by pointer'}

  TsgRecordList<T> = record
  type
    PItem = ^T;
    TEnumerator = record
    private
      FEnumerator: TsgPointerList.TEnumerator;
      function GetCurrent: PItem; inline;
    public
      constructor From(const Pointers: TsgPointerList);
      function MoveNext: Boolean; inline;
      property Current: PItem read GetCurrent;
    end;
  private
    FList: TsgPointerList;
    function Get(Index: Integer): PItem; inline;
    procedure Put(Index: Integer; Item: PItem);
    function GetCount: Integer; inline;
    procedure SetCount(Value: Integer);

  public
    constructor From(Meta: PsgItemMeta);
    procedure Free;
    procedure Clear;
    procedure AddRange(const collection: TArray<T>);
    function Add(Item: PItem): Integer; overload; inline;
    // Add an empty record and return its pointer
    function Add: PItem; overload; inline;
    procedure Delete(Index: Integer); inline;
    procedure Exchange(Index1, Index2: Integer); inline;
    function IndexOf(Item: PItem): Integer; inline;
    procedure Assign(const Source: TsgRecordList<T>);
    procedure Sort(Compare: TListSortCompare); inline;
    procedure Reverse; inline;
    function IsEmpty: Boolean; inline;
    function GetEnumerator: TEnumerator;
    function ToArray: TArray<T>;
    property Count: Integer read GetCount write SetCount;
    property Items[Index: Integer]: PItem read Get write Put; default;
    property List: TsgPointerList read FList;
  end;

{$EndRegion}

{$Region 'TCustomForwardList: Untyped Forward List'}

  PCustomForwardList = ^TCustomForwardList;
  TCustomForwardList = record
  type
    PItem = ^TItem;
    TItem = record
      next: PItem;
    end;

    TEnumerator = record
    private
      FItem: PItem;
      function GetCurrent: PItem;
    public
      constructor From(List: PCustomForwardList);
      function MoveNext: Boolean;
      property Current: PItem read GetCurrent;
    end;

  private
    FHead: PItem;
    FLast: PItem;
    FRegion: PSegmentedRegion;
  public
    procedure Init(Meta: PsgItemMeta);
    procedure Free;
    // Checks if the container has no elements.
    function Empty: Boolean; inline;
    // Returns the number of elements in the container.
    function GetCount: Integer;
    // Erases all elements from the container.
    procedure Clear;
    // Returns a reference to the first element in the container.
    // Calling front on an empty container is undefined.
    function Front: PItem;
    // Prepends the the empty value to the beginning of the container.
    // No iterators or references are invalidated.
    function PushFront: PItem;
    // Removes the first element of the container.
    procedure PopFront;
    // Inserts value after pos
    function InsertAfter(const Pos: PItem): PItem;
    // Reverses the order of the elements in the container.
    // No references or iterators become invalidated.
    procedure Reverse;
    // Sorts the elements in ascending order. The order of equal elements is preserved.
    procedure Sort(Compare: TListSortCompare);
  end;

{$EndRegion}

{$Region 'TsgForwardList<T>: Generic Unidirectional Linked List'}

  TsgForwardList<T> = record
  type
    PItem = ^TItem;
    TItem = record
      Link: TCustomForwardList.TItem;
      Value: T;
    end;
    PValue = ^T;

    TEnumerator = record
    private
      FEnumerator: TCustomForwardList.TEnumerator;
      function GetCurrent: PValue; inline;
    public
      constructor From(List: PCustomForwardList);
      function MoveNext: Boolean; inline;
      property Current: PValue read GetCurrent;
    end;

    TIterator = record
    private
      function GetValue: PValue;
    public
      Item: PItem;
      // Go to the next node
      procedure Next;
      // Iterator at the end of the list.
      function Eol: Boolean;
      // Pointer to Value
      property Value: PValue read GetValue;
    end;

  private
    FList: TCustomForwardList;
    FNodeMeta: TsgItemMeta;
    function GetRegion: PSegmentedRegion; inline;
    function GetCount: Integer; inline;
  public
    procedure Init(Meta: PsgItemMeta);
    procedure Free; inline;
    // Erases all elements from the container. After this call, Count returns zero.
    // Invalidates any references, pointers, or iterators referring to contained
    // elements. Any past-the-end iterator remains valid.
    procedure Clear; inline;
    // Checks if the container has no elements, i.e. whether begin() == end().
    function Empty: Boolean; inline;
    // Returns a reference to the first element in the container.
    function Front: TIterator; inline;
    // Prepends the the empty value to the beginning of the container.
    // No iterators or references are invalidated.
    function PushFront: TIterator; overload; inline;
    // Prepends the given element value to the beginning of the container.
    // No iterators or references are invalidated.
    procedure PushFront(const Value: T); overload; inline;
    // Inserts empty value after Pos
    function InsertAfter(Pos: TIterator): TIterator; overload; inline;
    // Inserts value after Pos
    function InsertAfter(Pos: TIterator; const Value: T): TIterator; overload; inline;
    // Removes the first element of the container.
    // If there are no elements in the container, the behavior is undefined.
    // References and iterators to the erased element are invalidated.
    procedure PopFront; inline;
    // Reverses the order of the elements in the container.
    // No references or iterators become invalidated.
    procedure Reverse; inline;
    // Sorts the elements in ascending order. The order of equal elements is preserved.
    procedure Sort(Compare: TListSortCompare); inline;
    // Get Delphi Enumerator
    function GetEnumerator: TEnumerator;
    // The number of elements in the container.
    property Count: Integer read GetCount;
    // Memory region
    property Region: PSegmentedRegion read GetRegion;
  end;

{$EndRegion}

{$Region 'TCustomLinkedList: Untyped Bidirectional Linked List'}

  PCustomLinkedList = ^TCustomLinkedList;
  TCustomLinkedList = record
  type
    PItem = ^TItem;
    TItem = record
      next: PItem;
      prev: PItem;
    end;
    TEnumerator = record
    private
      FItem: PItem;
      function GetCurrent: PItem;
    public
      constructor From(const List: TCustomLinkedList);
      function MoveNext: Boolean;
      property Current: PItem read GetCurrent;
    end;
  private
    FRegion: PSegmentedRegion;
    FHead: PItem;
    FLast: PItem;
  public
    procedure Init(Meta: PsgItemMeta);
    procedure Free;
    // Erases all elements from the container. After this call, Count returns zero.
    // Invalidates any references, pointers, or iterators referring to contained
    // elements. Any past-the-end iterator remains valid.
    procedure Clear;
    // Checks if the container has no elements, i.e. whether begin() == end().
    function Empty: Boolean; inline;
    // Returns the number of elements in the container,
    function Count: Integer;
    // Returns a reference to the first element in the container.
    // Calling front on an empty container is undefined.
    function Front: PItem;
    // Returns reference to the last element in the container.
    // Calling back on an empty container causes undefined behavior.
    function Back: PItem;
    // Prepends the the empty value to the beginning of the container.
    // No iterators or references are invalidated.
    function PushFront: PItem;
    // Appends the empty value to the end of list and return a pointer to it
    function PushBack: PItem;
    // Inserts value before pos
    function Insert(const Pos: PItem): PItem;
    // Removes the first element of the container.
    // If there are no elements in the container, the behavior is undefined.
    // References and iterators to the erased element are invalidated.
    procedure PopFront;
    // Removes the last element of the list.
    // Calling pop_back on an empty container results in undefined behavior.
    // References and iterators to the erased element are invalidated.
    procedure PopBack;
    // Reverses the order of the elements in the container.
    // No references or iterators become invalidated.
    procedure Reverse;
    // Sorts the elements in ascending order. The order of equal elements is preserved.
    procedure Sort(Compare: TListSortCompare);
    // Get Delphi Enumerator
    function GetEnumerator: TEnumerator;
  end;

{$EndRegion}

{$Region 'TsgLinkedList<T>: Generic Bidirectional Linked List'}

  TsgLinkedList<T> = record
  type
    PItem = ^TItem;
    TItem = record
      Link: TCustomLinkedList.TItem;
      Value: T;
    end;
    PValue = ^T;
    TEnumerator = record
    private
      FEnumerator: TCustomLinkedList.TEnumerator;
      function GetCurrent: PValue; inline;
    public
      constructor From(const List: TCustomLinkedList);
      function MoveNext: Boolean; inline;
      property Current: PValue read GetCurrent;
    end;
    TIterator = record
    private
      Item: PItem;
      function GetValue: PValue;
    public
      // Go to the next node
      procedure Next;
      // Go to the previous node
      procedure Prev;
      // Iterator at the end of the list.
      function Eol: Boolean;
      // Iterator before the beginning of the list.
      function Bol: Boolean;
      // Pointer to Value
      property Value: PValue read GetValue;
    end;
  private
    FList: TCustomLinkedList;
    function GetRegion: PSegmentedRegion; inline;
    function GetCount: Integer; inline;
  public
    procedure Init(Meta: PsgItemMeta);
    procedure Free; inline;
    // Erases all elements from the container. After this call, Count returns zero.
    // Invalidates any references, pointers, or iterators referring to contained
    // elements. Any past-the-end iterator remains valid.
    procedure Clear; inline;
    // Checks if the container has no elements, i.e. whether begin() == end().
    function Empty: Boolean; inline;
    // Returns the number of elements in the container.
    // Returns a reference to the first element in the container.
    // Calling front on an empty container is undefined.
    function Front: TIterator; inline;
    // Returns reference to the last element in the container.
    // Calling back on an empty container causes undefined behavior.
    function Back: TIterator; inline;
    // Prepends the the empty value to the beginning of the container.
    // No iterators or references are invalidated.
    function PushFront: TIterator; overload; inline;
    // Prepends the given element value to the beginning of the container.
    // No iterators or references are invalidated.
    procedure PushFront(const Value: T); overload; inline;
    // Appends the empty value to the end of list and return a pointer to it
    function PushBack: TIterator; overload; inline;
    // Appends the given element value to the end of list
    procedure PushBack(const Value: T); overload; inline;
    // Inserts value after Pos
    function Insert(Pos: TIterator; const Value: T): TIterator;
    // Removes the first element of the container.
    // If there are no elements in the container, the behavior is undefined.
    // References and iterators to the erased element are invalidated.
    procedure PopFront; inline;
    // Removes the last element of the list.
    // Calling pop_back on an empty container results in undefined behavior.
    // References and iterators to the erased element are invalidated.
    procedure PopBack; inline;
    // Reverses the order of the elements in the container.
    // No references or iterators become invalidated.
    procedure Reverse; inline;
    // Sorts the elements in ascending order. The order of equal elements is preserved.
    procedure Sort(Compare: TListSortCompare); inline;
    // Get Delphi Enumerator
    function GetEnumerator: TEnumerator;
    // The number of elements in the container.
    property Count: Integer read GetCount;
    // Memory region
    property Region: PSegmentedRegion read GetRegion;
  end;

{$EndRegion}

{$Region 'TsgCustomHashMap: Untyped Unordered dictionary'}

  PsgCustomHashMap = ^TsgCustomHashMap;

  TsgCustomHashMapIterator = record
  private
    index: Integer;
    map: PsgCustomHashMap;
    procedure Init(map: PsgCustomHashMap; idx: Integer);
  public
    class operator Equal(const a, b: TsgCustomHashMapIterator): Boolean; inline;
    class operator NotEqual(const a, b: TsgCustomHashMapIterator): Boolean; inline;
    procedure Next; inline;
    function GetKey: Pointer; inline;
    function GetValue: Pointer; inline;
  end;

  TsgCustomHashMap = record
  type
    TAssignPair = reference to procedure(pair: Pointer);
    // Collision list element
    PCollision = ^TCollision;
    TCollision = record
      Next: PCollision;
      function GetPairRef: Pointer;
    end;
    // Hash table element (entry)
    pEntry = ^TEntry;
    TEntry = record
      root: PCollision;
    end;
  private
    FEntries: PUnbrokenRegion;
    FCollisions: PSegmentedRegion;
    FPair: PsgTupleMeta;
    FHasher: TsgHasher;
    function GetCount: Integer;
    // Get a prime number for the expected number of items
    function GetEntries(ExpectedSize: Integer): Integer;
  public
    constructor From(PairMeta: PsgTupleMeta; ExpectedSize: Integer; Hasher: PsgHasher);
    procedure Free;
    // Already initialized
    function Valid: Boolean; inline;
    // Find item by key
    function Find(key: Pointer): Pointer;
    // Inserts element into the container, if the container doesn't already
    // contain an element with an equivalent key.
    function Insert(pair: Pointer): Pointer;
    // Insert an element or assigns to the current element if the key already exists
    function InsertOrAssign(pair: Pointer): Pointer;
    // Return a temporary variable
    function GetTemporaryPair: Pointer; inline;
    // Return the iterator to the beginning
    function Begins: TsgCustomHashMapIterator;
    // Next to the last one.
    function Ends: TsgCustomHashMapIterator;
  end;

{$EndRegion}

{$Region 'TsgHashMap<Key, T>: Generic Unordered dictionary'}

  TsgPair<TKey, TValue> = record
  var
    Key: TKey;
    Value: TValue;
  public
    constructor From(const Key: TKey; const Value: TValue);
  end;

  TsgHashMapIterator<Key, T> = record
  type
    PItem = ^T;
    PKey = ^Key;
    TPair = TsgPair<Key, T>;
    PPair = ^TPair;
  private
    it: TsgCustomHashMapIterator;
  public
    class operator Equal(const a, b: TsgHashMapIterator<Key, T>): Boolean; //inline;
    class operator NotEqual(const a, b: TsgHashMapIterator<Key, T>): Boolean; //inline;
    procedure Next; inline;
    function GetPair: PPair; inline;
    function GetKey: PKey; inline;
    function GetValue: PItem; inline;
  end;

  // Has constant lookup time using memory pool
  TsgHashMap<Key, T> = record
  type
    TPair = TsgPair<Key, T>;
    PPair = ^TPair;
  private
    FMap: TsgCustomHashMap;
    function GetCount: Integer; inline;
  public
    constructor From(ExpectedSize: Integer; Hasher: PsgHasher; FreePair: TFreeProc = nil);
    procedure Free; inline;
    // Finds an element with key equivalent to key.
    function Find(const k: Key): PPair; inline;
    // Inserts element into the container, if the container doesn't already
    // contain an element with an equivalent key.
    function Insert(const pair: TsgPair<Key, T>): PPair; inline;
    // Insert an element or assigns to the current element if the key already exists
    function InsertOrAssign(const pair: TsgPair<Key, T>): PPair; inline;
    // Return a temporary variable
    function GetTemporaryPair: PPair; inline;
    // Return the iterator to the beginning
    function Begins: TsgHashMapIterator<Key, T>; inline;
    // Next to the last one.
    function Ends: TsgHashMapIterator<Key, T>; inline;
    // Return the number of items
    property Count: Integer read GetCount;
  end;

{$EndRegion}

{$Region 'TsgTreeIterator: Iterator for 2-3 tree'}

  TsgTreeAction = (taFind, taInsert, taInsertEmpty, taInsertOrAssign, taCount);

  TsgTreeIterator = record
  type
    PNode = ^TNode;
    PPNode = ^PNode;
    TNode = record
      left, right: PNode;
      case Integer of
        0: (lh, rh: Boolean);
        1: (forAlignment: Int64);   // field for memory alignment
                                    // in different memory models (32 or 64 bit)
    end;
  private
    Stack: array of PNode;
    function Sentinel: PNode;       // 0 - element (always on the stack)
    function Current: PPNode;       // 1 - element (always on the stack)
    function Res: PPNode;           // 2 - element (always on the stack)
    procedure Push(Item: Pointer);
    function Pop: Pointer;
    function Empty: Boolean; inline;
  public
    constructor Init(Root, Sentinel: PNode);
    procedure Next;
    function GetItem: Pointer;
  end;

{$EndRegion}

{$Region 'TsgCustomTree: Untyped Dictionary based on 2-3 trees'}

  TsgCustomTree = record
  private type
    PNode = TsgTreeIterator.PNode;
    TParams = record
      action: TsgTreeAction;
      h, cnt: Integer;
      node: PNode;
      pval: Pointer;
      procedure Init(ta: TsgTreeAction; pval: Pointer);
    end;
    TNodeProc = procedure(p: PNode) of object;
    TUpdateProc = procedure(p: PNode; pval: Pointer) of object;
  private
    Compare: TListSortCompare;
    Update: TUpdateProc;
    Visit: TNodeProc;
    Region: PSegmentedRegion;
    Root, Sentinel: PNode;
    procedure Visiter(node: PNode);
    procedure Search(var p: PNode; var prm: TParams);
    procedure CreateNode(var p: PNode);
  public
    procedure Init(Meta: PsgItemMeta; Compare: TListSortCompare;
      Update: TUpdateProc);
    procedure Free;
    procedure Clear;
    procedure Find(pval: Pointer; var iter: TsgTreeIterator);
    function Get(pval: Pointer): PNode;
    // Return the number of items with key
    function Count(pval: Pointer): Integer;
    procedure Insert(pval: Pointer);
    procedure InsertOrAssign(pval: Pointer);
    procedure Begins(var iter: TsgTreeIterator);
    // Next to the last (guard element)
    function Ends: PNode;
    // Tree bypass
    procedure Inorder(Visit: TNodeProc);
  end;

{$EndRegion}

{$Region 'TsgMap<Key, T>: Generic Ordered Dictionary based on 2-3 tree'}

  {$Region 'TsgMapIterator<Key, T>: Generic Iterator for 2-3 tree'}

  TsgMapIterator<Key: record; T: record> = record
  type
    PItem = ^T;
    PKey = ^Key;
    PNode = ^TNode;
    PPNode = ^PNode;
    TNode = record
      Dt: TsgTreeIterator.TNode;
      case Integer of
        0: (k: Key; v: T);
        1: (pair: TsgPair<Key, T>);
    end;
  private
    Iter: TsgTreeIterator;
  public
    constructor Init(Root, Sentinel: TsgTreeIterator.PNode);
    class operator Equal(const a: TsgMapIterator<Key, T>; b: PNode): Boolean;
    class operator NotEqual(const a: TsgMapIterator<Key, T>; b: PNode): Boolean;
    procedure Next; inline;
    function GetKey: PKey;
    function GetValue: PItem;
  end;

  {$EndRegion}

  TsgMap<Key: record; T: record> = record
  public type
    PItem = ^T;
    PNode = TsgMapIterator<Key, T>.PNode;
    TNodeProc = procedure(p: PNode) of object;
  private
    tree: TsgCustomTree;
    procedure UpdateValue(pnd: TsgCustomTree.PNode; pval: Pointer);
  public
    constructor From(Compare: TListSortCompare; OnFreeNode: TFreeProc = nil);
    procedure Free; inline;
    procedure Clear;
    function Find(const k: Key): TsgMapIterator<Key, T>; inline;
    function Count(const k: Key): Integer; inline;
    procedure Insert(const pair: TsgPair<Key, T>); inline;
    function Emplace(const k: Key): PNode;
    procedure InsertOrAssign(const pair: TsgPair<Key, T>);
    function Begins: TsgMapIterator<Key, T>; inline;
    function Ends: PNode;
    // Bypass the tree in order
    procedure Inorder(Visit: TNodeProc); inline;
    function Get(index: Key): PItem;
    procedure Put(index: Key; const Value: PItem);
    property Items[index: Key]: PItem read Get write Put; default;
  end;

{$EndRegion}

{$Region 'TsgSet<Key>: Set based on 2-3 trees'}

  {$Region 'TsgSetIterator<Key, T>: Iterator for 2-3 trees'}

  TsgSetIterator<Key: record> = record
  type
    PKey = ^Key;
    PNode = ^TNode;
    PPNode = ^PNode;
    TNode = record
      Dt: TsgTreeIterator.TNode;
      k: Key;
    end;
  private
    Iter: TsgTreeIterator;
  public
    constructor Init(Root, Sentinel: PNode);
    class operator Equal(const a: TsgSetIterator<Key>; b: PNode): Boolean;
    class operator NotEqual(const a: TsgSetIterator<Key>; b: PNode): Boolean;
    procedure Next; inline;
    function GetKey: PKey;
  end;

  {$EndRegion}

  TsgSet<Key: record> = record
  private type
    PNode = TsgSetIterator<Key>.PNode;
    TNodeProc = procedure(p: PNode) of object;
  private
    tree: TsgCustomTree;
    procedure UpdateValue(pnd: TsgCustomTree.PNode; pval: Pointer);
  public
    procedure Init(Compare: TListSortCompare; OnFreeNode: TFreeProc = nil);
    procedure Free; inline;
    procedure Clear(Compare: TListSortCompare; OnFree: TFreeProc = nil);
    procedure Insert(const k: Key); inline;
    function Find(const k: Key): TsgSetIterator<Key>; inline;
    // Count the number of elements with key
    function Count(const k: Key): Integer; inline;
    // Bypass the tree in order
    procedure Inorder(Visit: TNodeProc); inline;
    function Begins: TsgSetIterator<Key>; inline;
    function Ends: PNode;
  end;

{$EndRegion}

{$Region 'TSharedRegion: Shared typed memory region'}

  TSharedRegion = record
  const
    RegionHandle: hRegion = (v: 1);
  strict private
    FRegion: TMemoryRegion;
    FMemoryManager: TsgMemoryManager;
    FHandleManager: TsgHandleManager;
    FSizes: array [TsgHandleManager.TIndex] of Cardinal;
    function GetMeta: PsgItemMeta; inline;
    function GetItemSize: Cardinal; inline;
    procedure ClearManagedTypes(const descr: TMemoryDescriptor);
    procedure FreeUsed(h: hCollection);
  public
    // Initialize shared memory region for collections
    procedure Init(Meta: PsgItemMeta; Capacity: Cardinal);
    // Free the region
    procedure Free;
    // Allocate memory for collection items
    procedure Alloc(var descr: TMemoryDescriptor);
    // Return memory to heap
    procedure FreeMem(var descr: TMemoryDescriptor);
    // Reallocate memory for collection items
    procedure Realloc(var descr: TMemoryDescriptor; Count: Cardinal);
    property ItemSize: Cardinal read GetItemSize;
    property Meta: PsgItemMeta read GetMeta;
  end;

{$EndRegion}

{$Region 'TsgSystemContext: System processing context'}

  TsgSystemContext = class(TsgContext)
  private type
    TRegionId = (rItemMeta, rTeMeta, rTupleMeta, rSharedRegion);
    PSharedData = ^TSharedData;
    TSharedData = record
      Meta: TsgItemMeta;
      Region: TSharedRegion;
      List: TsgArrayHelper;
    end;
    TMetaList = array [TRegionId] of TSharedData;
  private
    FMetaList: TMetaList;
    // Init metadata for Tuple meta region
    procedure InitTupleMeta(var meta: TsgItemMeta;
      ItemSize: Cardinal; Flags: TRegionFlagSet);
    // ?reate ArrayHelper from shared regions
    function CreateArrayHelper(Capacity: Cardinal): PsgArrayHelper;
  public
    constructor Create;
    destructor Destroy; override;

    // ?reate metadata for type T
    function CreateMeta<T>(OnFree: TFreeProc = nil): PsgItemMeta; overload;
    function CreateMeta<T>(Flags: TRegionFlagSet; RemoveAction: TRemoveAction;
      OnFree: TFreeProc = nil): PsgItemMeta; overload;

    // ?reate metadata for tuple
    function CreateTupleMeta: PsgTupleMeta;

    // ?reate TArray<TsgTupleElementMeta>
    procedure CreateTeMetas(Count: Cardinal;
      var List: TsgTupleMeta.TsgTeMetaList);

    // Return shared region
    function GetShareRegion(id: TRegionId): PSharedRegion;

    // Factory methods to create array from shared regions
    procedure CreateArray<T>(Capacity: Cardinal; var Value: TsgArray<T>);
  end;

{$EndRegion}

{$Region 'TsgLog'}

  TsgLog = record
  private
    FLocalDebug: Boolean;
    FLog: TStringList;
    procedure AddLine(const Msg: string);
  public
    procedure Init;
    procedure Free;
    // Save to file
    procedure SaveToFile(const filename: string);
    // Logging when the FLocalDebug flag is set
    procedure print(const Msg: string); overload;
    procedure print(const Msg: string;
      const Args: array of const); overload;
    // Displaying an explanatory message to the user
    procedure Msg(const Msg: string); overload; inline;
    procedure Msg(const Fmt: string;
      const Args: array of const); overload;
    property LocalDebug: Boolean read FLocalDebug write FLocalDebug;
  end;

{$EndRegion}

{$Region 'Subroutines'}

// Quick sort
procedure QuickSort(List: PsgPointers; L, R: Integer; SCompare: TListSortCompareFunc);

{$EndRegion}

var
  SysCtx: TsgSystemContext;
  log: TsgLog;

implementation

{$Region 'Subroutines'}

procedure Swap(var i, j: Double);
var
  temp: Double;
begin
  temp := i;
  i := j;
  j := temp;
end;

procedure Exchange(pointers: PsgPointers; i, j: Integer); inline;
var
  temp: Pointer;
begin
  temp := pointers[i];
  pointers[i] := pointers[j];
  pointers[j] := temp;
end;

procedure CheckCount(Count: Integer); inline;
begin
  if Count < 0 then
    raise EsgError.Create(EsgError.ListCountError, Count);
end;

procedure QuickSort(List: PsgPointers; L, R: Integer; SCompare: TListSortCompareFunc);

  procedure Sort(L, R: Integer);
  var
    i, j: Integer;
    x: Pointer;
  begin
    i := L;
    j := R;
    x := List[(L + R) div 2];
    repeat
      while SCompare(List[i], x) < 0 do
      begin
        if i >= R then break;
        Inc(i);
      end;
      while SCompare(List[j], x) > 0 do
      begin
        if j <= L then break;
        Dec(j);
      end;
      if i <= j then
      begin
        Exchange(List, i, j);
        Inc(i); Dec(j);
      end;
    until i > j;
    if L < j then QuickSort(List, L, j, SCompare);
    if i < R then QuickSort(List, i, R, SCompare);
  end;

  procedure ShortSort(L, R: Integer);
  var
    i, max: Integer;
  begin
    while R > L do
    begin
      max := L;
      for i := L + 1 to R do
        if SCompare(List[i], List[max]) > 0 then
          max := i;
      Exchange(List, max, R);
      Dec(R);
    end;
  end;

begin
  // Below a certain size, it is faster to use the O(n^2) sort method
  if (R - L) <= 8 then
    ShortSort(L, R)
  else
    Sort(L, R);
end;

{$EndRegion}

{$Region 'TObjectHeap'}

constructor TObjectHeap<T>.From(var meta: PsgItemMeta; BlockSize: Cardinal);
begin
  meta := SysCtx.CreateMeta<T>;
  region.Init(meta, BlockSize);
end;

function TObjectHeap<T>.Get(Clear: Boolean): PItem;
begin
  Result := region.AddItem;
end;

{$EndRegion}

{$Region 'TMemoryDescriptor'}

procedure TMemoryDescriptor.Clear;
begin
  h.v := 0;
  Count := 0;
  Items := nil;
end;

{$EndRegion}

{$Region 'TsgArrayHelper: Generic Array'}

procedure TsgArrayHelper.Init(Region: PSharedRegion; Capacity: Cardinal);
begin
  FRegion := Region;
  FCount := 0;
  FDesc.Count := Capacity;
  Region.Alloc(FDesc);
end;

procedure TsgArrayHelper.Free;
begin
  FRegion.FreeMem(FDesc);
end;

procedure TsgArrayHelper.Grow;
var
  NewCapacity: Cardinal;
begin
  NewCapacity := GrowCollection(FDesc.Count, FCount + 1);
  SetCapacity(NewCapacity);
end;

procedure TsgArrayHelper.SetCapacity(NewCapacity: Cardinal);
begin
  if NewCapacity < FDesc.Count then
    raise EsgError.Create(EsgError.CapacityError, NewCapacity);
  if NewCapacity <> FDesc.Count then
    FRegion.Realloc(FDesc, NewCapacity);
end;

procedure TsgArrayHelper.SetCount(NewCount: Cardinal);
begin
  if NewCount <> FCount then
  begin
    if NewCount > FDesc.Count then
      SetCapacity(NewCount);
    FCount := NewCount;
  end;
end;

function TsgArrayHelper.Add: PByte;
begin
  if FCount = FDesc.Count then
    Grow;
  Result := GetItem(FCount);
  Inc(FCount);
end;

function TsgArrayHelper.Insert(Index: Cardinal): PByte;
begin
  if FCount = FDesc.Count then
    Grow;
  Result := GetItem(Index);
  if Index < FCount then
    System.Move(Result^, GetItem(Index + 1)^, FRegion.ItemSize * (FCount - Index));
  Inc(FCount);
end;

function TsgArrayHelper.GetItem(Index: Cardinal): PByte;
begin
  Result := FDesc.Items + FRegion.ItemSize * Index;
end;

{$EndRegion}

{$Region 'TsgArray<T>: Generic Array'}

procedure TsgArray<T>.Init(Region: PSharedRegion; Capacity: Cardinal);
begin
  Check(Region.Meta.TypeInfo = System.TypeInfo(T));
  FList.Init(Region, Capacity);
end;

procedure TsgArray<T>.Free;
begin
  FList.Free;
end;

function TsgArray<T>.GetItem(Index: Cardinal): PItem;
begin
  Result := PItem(FList.GetItem(Index));
end;

procedure TsgArray<T>.SetCount(NewCount: Cardinal);
begin
  FList.SetCount(NewCount);
end;

function TsgArray<T>.Insert(Index: Cardinal): PItem;
begin
  Result := PItem(FList.Insert(Index));
end;

function TsgArray<T>.Add: PItem;
begin
  Result := PItem(FList.Add);
end;

{$EndRegion}

{$Region 'TsgTupleElementMeta'}

procedure TsgTupleElementMeta.Init<T>;
begin
  FOffset := 0;
  Meta.Init<T>;
end;

function TsgTupleElementMeta.NextTupleOffset(Allign: Boolean): Cardinal;
var
  n: Cardinal;
begin
  n := FMeta.ItemSize;
  if Allign then
    n := ((n + AllignTuple - 1) div AllignTuple) * AllignTuple;
  Result := FOffset + n;
end;

function TsgTupleElementMeta.GetFreeItem: TFreeItem;
begin
  Result := FMeta.FreeItem;
end;

function TsgTupleElementMeta.GetAssignItem: TAssignProc;
begin
  Result := FMeta.AssignItem;
end;

function TsgTupleElementMeta.GetItemSize: Cardinal;
begin
  Result := FMeta.ItemSize;
end;

function TsgTupleElementMeta.GetMeta: PsgItemMeta;
begin
  Result := @FMeta;
end;

{$EndRegion}

{$Region 'TsgTupleMeta'}

procedure TsgTupleMeta.Init(Count: Cardinal; OnFree: TFreeProc);
begin
  FillChar(Self, sizeof(TsgTupleMeta), 0);
  SysCtx.CreateTeMetas(Count, FElements);
  FOnFree := OnFree;
end;

function TsgTupleMeta.Get(Index: Cardinal): PsgTupleElementMeta;
begin
  Result := FElements.GetItem(Index);
end;

procedure TsgTupleMeta.AssignTuple(Dest, Value: Pointer);
var
  i: Integer;
  e: PsgTupleElementMeta;
begin
  for i := 0 to FElements.Count - 1 do
  begin
    e := FElements.GetItem(i);
    e.Assign(e.Meta, PByte(Dest) + e.Offset, PByte(Value) + e.Offset);
  end;
end;

procedure TsgTupleMeta.MoveTuple(Dest, Value: Pointer);
begin
  Move(Value^, Dest^, Size);
end;

procedure TsgTupleMeta.CheckManaged;
var
  i: Integer;
  e: PsgTupleElementMeta;
begin
  FAssignTuple := MoveTuple;
  for i := 0 to FElements.Count - 1 do
  begin
    e := FElements.GetItem(i);
    if e.meta.h.ManagedType or e.meta.h.HasWeakRef then
    begin
      FAssignTuple := AssignTuple;
      exit;
    end;
  end;
end;

procedure TsgTupleMeta.AddElement(meta: PsgTupleElementMeta; Allign: Boolean);
var
  te: PsgTupleElementMeta;
begin
  te := FElements.Add;
  te^ := meta^;
  te.FOffset := FSize;
  FSize := te.NextTupleOffset(Allign);
end;

procedure TsgTupleMeta.AddTe<T>(Allign: Boolean);
var
  te: PsgTupleElementMeta;
begin
  te := FElements.Add;
  te.Init<T>;
  te.FOffset := FSize;
  FSize := te.NextTupleOffset(Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.MakePair<T1, T2>(OnFree: TFreeProc; Allign: Boolean);
begin
  Init(2, OnFree);
  AddTe<T1>(Allign);
  AddTe<T2>(Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.MakeTrio<T1, T2, T3>(OnFree: TFreeProc; Allign: Boolean);
begin
  Init(3, OnFree);
  AddTe<T1>(Allign);
  AddTe<T2>(Allign);
  AddTe<T3>(Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.MakeQuad<T1, T2, T3, T4>(OnFree: TFreeProc; Allign: Boolean);
begin
  Init(4, OnFree);
  AddTe<T1>(Allign);
  AddTe<T2>(Allign);
  AddTe<T3>(Allign);
  AddTe<T4>(Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.Cat(const Tuple: TsgTupleMeta; OnFree: TFreeProc;
  Allign: Boolean);
var
  i: Cardinal;
begin
  FOnFree := OnFree;
  for i := 0 to Tuple.Count - 1 do
    AddElement(Tuple.Get(i), Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.Add<T>(OnFree: TFreeProc; Allign: Boolean);
begin
  AddTe<T>(Allign);
  CheckManaged;
end;

procedure TsgTupleMeta.Insert<T>(OnFree: TFreeProc; Allign: Boolean);
var
  i: Integer;
  te: PsgTupleElementMeta;
begin
  FSize := 0;
  te := FElements.Insert(0);
  te.Init<T>;
  for i := 0 to FElements.Count - 1 do
  begin
    te := FElements.Items[i];
    te.Offset := FSize;
    FSize := te.NextTupleOffset(Allign);
  end;
  CheckManaged;
end;

{$EndRegion}

{$Region 'TsgTupleElement'}

procedure TsgTupleElement.Assign(pvalue: Pointer);
begin
  TeMeta.Assign(TeMeta.Meta, Ptr, pvalue);
end;

function TsgTupleElement.GetPvalue: Pointer;
begin
  Result := nil;
end;

{$EndRegion}

{$Region 'TsgTuple'}

procedure TsgTuple.MakeTuple(Ptr: Pointer; const TupleMeta: TsgTupleMeta);
begin
  FPtr := Ptr;
  FTupleMeta := TupleMeta;
end;

procedure TsgTuple.Swap(Pvalue: Pointer);
begin
  // todo:
end;

function TsgTuple.Get(Index: Integer): TsgTupleElement;
begin
  Result.TeMeta := FTupleMeta.FElements.GetItem(Index);
  Result.Ptr := PByte(FPtr) + Result.TeMeta.Offset;
end;

function TsgTuple.Tie(Index: Integer): TsgTupleElements;
begin

end;

{$EndRegion}

{$Region 'TsgTuples'}

procedure MoveTuple(meta: PsgItemMeta; Dest, Value: Pointer);
begin
  Move(Value^, Dest^, PsgTuples(meta).FTupleMeta.Size);
end;

procedure AssignTuple(meta: PsgItemMeta; Dest, Value: Pointer);
var
  i: Integer;
  te: PsgTupleElementMeta;
begin
  for i := 0 to PsgTuples(meta).Count - 1 do
  begin
    te := PsgTuples(meta).FTupleMeta.Elements[i];
    te.Assign(te.Meta, PByte(Dest) + te.Offset, PByte(Value) + te.Offset);
  end;
end;

procedure FreeTuple(meta: PsgItemMeta; p: Pointer);
var
  i: Integer;
  te: PsgTupleElementMeta;
begin
  for i := 0 to PsgTuples(meta).Count - 1 do
  begin
    te := PsgTuples(meta).FTupleMeta.Elements[i];
    if Assigned(te.Free) then
      te.Free(te.Meta, PByte(p) + te.Offset);
  end;
end;

procedure TsgTuples.Init(const tupleMeta: TsgTupleMeta; Capacity: Cardinal;
  Flags: TRegionFlagSet);
var
  i: Integer;
  te: PsgTupleElementMeta;
  managedType: Boolean;
  meta: TsgItemMeta;
begin
  SysCtx.InitTupleMeta(meta, tupleMeta.FSize, Flags);
  managedType := False;
  for i := 0 to Count - 1 do
  begin
    te := FTupleMeta.Elements[i];
    if te.Meta.h.ManagedType then
    begin
      managedType := True;
      break;
    end;
  end;
  if not managedType then
    meta.AssignItem := MoveTuple
  else
  begin
    meta.FreeItem := FreeTuple;
    meta.AssignItem := AssignTuple;
  end;
  FRegion := SysCtx.Pool.CreateUnbrokenRegion(@meta);
end;

procedure TsgTuples.Free;
begin
  FRegion.Free;
end;

procedure TsgTuples.SetCount(NewCount: Cardinal);
begin
  if NewCount <> FCount then
  begin
    CheckCapacity(NewCount);
    FCount := NewCount;
  end;
end;

procedure TsgTuples.CheckCapacity(NewCapacity: Integer);
begin

end;

function TsgTuples.GetItem(Index: Cardinal): PsgTuple;
begin
  Result := FRegion.GetItemPtr(Index);
end;

function TsgTuples.Add: PsgTuple;
begin
  Result := nil;
end;

{$EndRegion}

{$Region 'TsgListHelper.TEnumerator'}

var ListHelper: PsgListHelper = nil;

constructor TsgListHelper.TEnumerator.From(Value: PsgListHelper);
begin
  FList := Value;
  FIndex := -1;
end;

function TsgListHelper.TEnumerator.GetCurrent: Pointer;
begin
  Result := FList.GetPtr(FIndex);
end;

function TsgListHelper.TEnumerator.MoveNext: Boolean;
begin
  Inc(FIndex);
  Result := Cardinal(FIndex) < Cardinal(FList.GetCount);
end;

{$EndRegion}

{$Region 'TsgListHelper'}

procedure TsgListHelper.Init(Meta: PsgItemMeta);
begin
  FRegion := SysCtx.Pool.CreateUnbrokenRegion(Meta);
end;

procedure TsgListHelper.Free;
begin
  FRegion.Free;
end;

procedure TsgListHelper.Clear;
begin
  FRegion.Clear;
end;

function TsgListHelper.GetPtr(Index: Integer): Pointer;
begin
  Result := FRegion.GetItemPtr(Index);
end;

function TsgListHelper.Add: Pointer;
begin
  Result := FRegion.AddItem;
end;

procedure TsgListHelper.SetCount(NewCount: Integer);
begin
  FRegion.SetCount(NewCount);
end;

procedure TsgListHelper.Delete(Index: Integer);
begin
  FRegion.Delete(Index);
end;

procedure TsgListHelper.QuickSort(Compare: TListSortCompareFunc; L, R: Integer);
var
  I, J: Integer;
  pivot: Pointer;
begin
  if L < R then
  begin
    repeat
      if (R - L) = 1 then
      begin
        if Compare(GetPtr(L), GetPtr(R)) > 0 then
          Exchange(L, R);
        break;
      end;
      I := L;
      J := R;
      pivot := GetPtr(L + (R - L) shr 1);
      repeat
        while Compare(GetPtr(I), pivot) < 0 do
          Inc(I);
        while Compare(GetPtr(J), pivot) > 0 do
          Dec(J);
        if I <= J then
        begin
          if I <> J then
            Exchange(I, J);
          Inc(I);
          Dec(J);
        end;
      until I > J;
      if (J - L) > (R - I) then
      begin
        if I < R then
          QuickSort(Compare, I, R);
        R := J;
      end
      else
      begin
        if L < J then
          QuickSort(Compare, L, J);
        L := I;
      end;
    until L >= R;
  end;
end;

procedure TsgListHelper.Sort(Compare: TListSortCompare);
begin
  if GetCount > 1 then
    QuickSort(Compare, 0, GetCount - 1);
end;

procedure TsgListHelper.Exchange(Index1, Index2: Integer);
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

procedure TsgListHelper.Insert(Index: Integer; const Value);
begin
  FRegion.Insert(Index, Value);
end;

function TsgListHelper.Remove(const Value): Integer;
var
  i: Integer;
begin
  for i := 0 to GetCount - 1 do
    if Compare(FRegion.GetItemPtr(i)^, Byte(Value)) then
      exit(i);
  Result := -1;
end;

procedure TsgListHelper.Reverse;
var
  b, e: Integer;
begin
  b := 0;
  e := GetCount - 1;
  while b < e do
  begin
    Exchange(b, e);
    Inc(b);
    Dec(e);
  end;
end;

procedure TsgListHelper.Assign(const Source: TsgListHelper);
var
  Cnt: Integer;
begin
  Cnt := Source.GetCount;
  FRegion.CopyFrom(Source.FRegion, 0, Cnt);
end;

function TsgListHelper.GetCount: Integer;
begin
  Result := FRegion.GetCount;
end;

function TsgListHelper.GetItems: PByte;
begin
  Result := FRegion.GetItems;
end;

function TsgListHelper.Compare(const Left, Right): Boolean;
begin
  Result := CompareMem(@Left, @Right, FRegion.Meta.ItemSize)
end;

{$EndRegion}

{$Region 'TsgList<T>'}

constructor TsgList<T>.From(Meta: PsgItemMeta; Capacity: Integer);
begin
  FListHelper.Init(Meta);
end;

procedure TsgList<T>.Free;
begin
  FListHelper.Free;
end;

procedure TsgList<T>.Clear;
begin
  FListHelper.Clear;
end;

function TsgList<T>.Add: PItem;
begin
  Result := FListHelper.Add;
end;

function TsgList<T>.Add(const Value: T): Integer;
var
  p: PItem;
begin
  Result := FListHelper.GetCount;
  p := FListHelper.Add;
  p^ := Value;
end;

procedure TsgList<T>.Delete(Index: Integer);
begin
  FListHelper.Delete(Index);
end;

procedure TsgList<T>.Insert(Index: Integer; const Value: T);
begin
  FListHelper.Insert(Index, Value);
end;

function TsgList<T>.Remove(const Value: T): Integer;
begin
  Result := FListHelper.Remove(Value);
end;

procedure TsgList<T>.Exchange(Index1, Index2: Integer);
begin
  FListHelper.Exchange(Index1, Index2);
end;

procedure TsgList<T>.Reverse;
begin
  FListHelper.Reverse;
end;

procedure TsgList<T>.Sort(Compare: TListSortCompare);
begin
  FListHelper.Sort(Compare);
end;

procedure TsgList<T>.Assign(Source: TsgList<T>);
begin
  FListHelper.Assign(Source.FListHelper);
end;

function TsgList<T>.GetPtr(Index: Integer): PItem;
begin
  Result := FListHelper.GetPtr(Index);
end;

function TsgList<T>.GetCount: Integer;
begin
  Result := FListHelper.GetCount;
end;

function TsgList<T>.GetEnumerator: TsgListHelper.TEnumerator;
begin
  Result := TsgListHelper.TEnumerator.From(@FListHelper);
end;

function TsgList<T>.GetItem(Index: Integer): T;
begin
  CheckIndex(Index, FListHelper.GetCount);
  Result := GetItems[Index];
end;

function TsgList<T>.GetItems: PItems;
begin
  Result := PItems(FListHelper.GetItems);
end;

function TsgList<T>.IsEmpty: Boolean;
begin
  Result := FListHelper.GetCount = 0;
end;

procedure TsgList<T>.SetCount(Value: Integer);
begin
  FListHelper.SetCount(Value);
end;

procedure TsgList<T>.SetItem(Index: Integer; const Value: T);
begin
  PItem(FListHelper.GetPtr(Index))^ := Value;
end;

{$EndRegion}

{$Region 'TsgStack<T>'}

constructor TsgStack<T>.From(var Meta: PsgItemMeta; Capacity: Integer);
begin
  FList := TsgList<T>.From(Meta, Capacity);
end;

procedure TsgStack<T>.Free;
begin
  FList.Free;
end;

procedure TsgStack<T>.Clear;
begin
  FList.Clear;
end;

function TsgStack<T>.Peek: T;
begin
  Result := FList.GetItem(Count - 1);
end;

procedure TsgStack<T>.Push(Item: T);
begin
  FList.Add(Item);
end;

function TsgStack<T>.Pop: T;
begin
  Result := GetItem(Count - 1);
  FList.Count := FList.Count - 1;
end;

function TsgStack<T>.Empty: Boolean;
begin
  Result := Count = 0;
end;

function TsgStack<T>.GetItem(Index: Integer): T;
begin
  Result := FList.GetItem(Index);
end;

function TsgStack<T>.GetCount: Integer;
begin
  Result := FList.GetCount;
end;

{$EndRegion}

{$Region 'TsgPointerArray.TEnumerator'}

constructor TsgPointerArray.TEnumerator.From(const Pointers: TsgPointerArray);
begin
  FPointers := @Pointers;
  FIndex := -1;
end;

function TsgPointerArray.TEnumerator.GetCurrent: Pointer;
begin
  Result := FPointers.Get(FIndex);
end;

function TsgPointerArray.TEnumerator.MoveNext: Boolean;
begin
  Inc(FIndex);
  Result := FIndex < FPointers.FCount;
end;

{$EndRegion}

{$Region 'TsgPointerArray: Array of pointers'}

constructor TsgPointerArray.From(Capacity: Integer);
begin
  FListRegion := SysCtx.CreateUnbrokenRegion(@TsgContext.PointerMeta);
  FList := FListRegion.Region.IncreaseCapacity(Capacity);
  FCount := 0;
end;

procedure TsgPointerArray.Free;
begin
  FListRegion.Free;
end;

function TsgPointerArray.Get(Index: Integer): Pointer;
begin
  CheckIndex(Index, FCount);
  Result := FList[Index];
end;

function TsgPointerArray.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(Self);
end;

procedure TsgPointerArray.Put(Index: Integer; Item: Pointer);
begin
  CheckIndex(Index, FCount);
  if Item <> FList[Index] then
    FList[Index] := Item;
end;

procedure TsgPointerArray.Add(ptr: Pointer);
var
  idx: Integer;
begin
  Check(ptr <> nil);
  idx := FCount;
  if FListRegion.Capacity <= idx then
    FList := FListRegion.Region.IncreaseAndAlloc(idx);
  Inc(FCount);
  FList[idx] := ptr;
end;

procedure TsgPointerArray.Sort(Compare: TListSortCompare);
begin
  if Count > 1 then
    QuickSort(FList, 0, Count - 1,
      function(Item1, Item2: Pointer): Integer
      begin
        Result := Compare(Item1, Item2);
      end);
end;

{$EndRegion}

{$Region 'TsgPointerList.TEnumerator'}

constructor TsgPointerList.TEnumerator.From(const Pointers: TsgPointerList);
begin
  FPointers := @Pointers;
  FIndex := -1;
end;

function TsgPointerList.TEnumerator.GetCurrent: Pointer;
begin
  Result := PPointer(FPointers.Get(FIndex))^;
end;

function TsgPointerList.TEnumerator.MoveNext: Boolean;
begin
  Inc(FIndex);
  Result := FIndex < FPointers.Count;
end;

{$EndRegion}

{$Region 'TsgPointerList'}

constructor TsgPointerList.From(Meta: PsgItemMeta);
begin
  FListRegion := SysCtx.CreateUnbrokenRegion(@SysCtx.PointerMeta);
  FItemsRegion := SysCtx.CreateRegion(Meta);
end;

procedure TsgPointerList.Free;
begin
  FItemsRegion.Free;
  FListRegion.Free;
end;

procedure TsgPointerList.Clear;
begin
  FItemsRegion.Region.Clear;
  FListRegion.Region.Clear;
end;

function TsgPointerList.First: Pointer;
begin
  if Count > 0 then
    Result := Get(0)
  else
    Result := nil;
end;

function TsgPointerList.Last: Pointer;
begin
  if Count > 0 then
    Result := Get(Count - 1)
  else
    Result := nil;
end;

function TsgPointerList.NextAfter(prev: Pointer): Pointer;
begin
  if prev = nil then
    Result := nil
  else if prev = Get(Count - 1) then
    Result := nil
  else
    Result := Pointer(NativeUInt(prev) + NativeUInt(FItemsRegion.ItemSize));
end;

procedure TsgPointerList.Assign(const Source: TsgPointerList);
var
  i: Integer;
begin
  Count := 0;
  for i := 0 to Source.Count - 1 do
    Add(Source.Get(i));
end;

function TsgPointerList.Add: Pointer;
begin
  Result := FItemsRegion.AddItem;
  PPointer(FListRegion.AddItem)^ := Result;
end;

function TsgPointerList.Add(Item: Pointer): Integer;
var
  p: Pointer;
begin
  Result := FListRegion.Count;
  p := Add;
  FItemsRegion.AssignItem(p, Item);
end;

procedure TsgPointerList.Insert(Index: Integer; Item: Pointer);
var
  dest: Pointer;
begin
  if Index = Count then
    Add(Item)
  else
  begin
    dest := FItemsRegion.AddItem;
    FItemsRegion.AssignItem(dest, Item);
    FListRegion.Insert(Index, dest);
  end;
end;

procedure TsgPointerList.Delete(Index: Integer);
begin
  FItemsRegion.Dispose(Get(Index), 1);
  FListRegion.Delete(Index);
end;

procedure TsgPointerList.Exchange(Index1, Index2: Integer);
begin
  FListRegion.Exchange(Index1, Index2);
end;

function TsgPointerList.IndexOf(Item: Pointer): Integer;
var
  i: Integer;
  p: PPointer;
begin
  p := PPointer(FListRegion.GetItems);
  for i := 0 to Count - 1 do
  begin
    if p^ = Item then
      exit(i);
    Inc(p);
  end;
  Result := -1;
end;

procedure TsgPointerList.Sort(Compare: TListSortCompare);
begin
  if Count > 1 then
    QuickSort(PsgPointers(FListRegion.GetItems), 0, Count - 1,
      function(Item1, Item2: Pointer): Integer
      begin
        Result := Compare(Item1, Item2);
      end);
end;

procedure TsgPointerList.Reverse;
var
  b, e: Integer;
begin
  b := 0;
  e := Count - 1;
  while b < e do
  begin
    Exchange(b, e);
    Inc(b);
    Dec(e);
  end;
end;

function TsgPointerList.Get(Index: Integer): Pointer;
begin
  Result := FListRegion.GetItemPtr(Index);
end;

procedure TsgPointerList.Put(Index: Integer; Item: Pointer);
begin
  FItemsRegion.AssignItem(Get(Index), Item);
end;

function TsgPointerList.GetCount: Integer;
begin
 if FListRegion = nil then Exit(0);

  Result := FListRegion.GetCount;
end;

function TsgPointerList.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(Self);
end;

procedure TsgPointerList.SetCount(NewCount: Integer);
begin
  FListRegion.SetCount(NewCount);
end;

function TsgPointerList.TraverseBy(F: TItemFunc): Pointer;
var
  i: Integer;
begin
  for i := 0 to Count - 1 do
  begin
    Result := Get(i);
    if F(Result) then exit;
  end;
  Result := nil;
end;

procedure TsgPointerList.RemoveBy(F: TItemFunc);
var
  dest, src: Integer;
  item: Pointer;
begin
  dest := 0;
  for src := 0 to Count - 1 do
  begin
    item := Get(src);
    if F(item) then
      // this item will be removed
    else
    begin
      if src <> dest then
        Put(dest, item);
      Inc(dest);
    end;
  end;
  Count := dest;
end;

function TsgPointerList.IsEmpty: Boolean;
begin
  Result := Count = 0;
end;

{$EndRegion}

{$Region 'TsgRecordList<T>.TEnumerator'}

constructor TsgRecordList<T>.TEnumerator.From(const Pointers: TsgPointerList);
begin
  FEnumerator := TsgPointerList.TEnumerator.From(Pointers);
end;

function TsgRecordList<T>.TEnumerator.GetCurrent: PItem;
begin
  Result := FEnumerator.GetCurrent;
end;

function TsgRecordList<T>.TEnumerator.MoveNext: Boolean;
begin
  Result := FEnumerator.MoveNext;
end;

{$EndRegion}

{$Region 'TsgRecordList<T>'}

constructor TsgRecordList<T>.From(Meta: PsgItemMeta);
begin
  //Assert(Meta <> nil);
  var m : TsgItemMeta := default(TsgItemMeta);
  if Meta = nil then
    Meta := @m;

  FList := TsgPointerList.From(Meta);
end;

procedure TsgRecordList<T>.Free;
begin
  FList.Free;
end;

procedure TsgRecordList<T>.Clear;
begin
  FList.Clear;
end;

function TsgRecordList<T>.Add(Item: PItem): Integer;
begin
  Result := FList.Add(Item);
end;

function TsgRecordList<T>.Add: PItem;
begin
  Result := FList.Add;
end;

procedure TsgRecordList<T>.AddRange(const collection: TArray<T>);
var
  item: T;
begin
  for item in collection do
    Add(@item);
end;

procedure TsgRecordList<T>.Delete(Index: Integer);
begin
  FList.Delete(Index);
end;

procedure TsgRecordList<T>.Exchange(Index1, Index2: Integer);
begin
  FList.Exchange(Index1, Index2);
end;

function TsgRecordList<T>.IndexOf(Item: PItem): Integer;
begin
  Result := FList.IndexOf(Item);
end;

procedure TsgRecordList<T>.Assign(const Source: TsgRecordList<T>);
var
  i: Integer;
begin
  Count := 0;
  for i := 0 to Source.Count - 1 do
    Add(Source.Items[i]);
end;

procedure TsgRecordList<T>.Sort(Compare: TListSortCompare);
begin
  FList.Sort(Compare);
end;

function TsgRecordList<T>.ToArray: TArray<T>;
var
  i: Integer;
begin
   Result := [];
   for i := 0 to FList.Count - 1 do
     Result :=  Result + [ PItem(FList.Get(i)^)^ ];
end;

procedure TsgRecordList<T>.Reverse;
begin
  FList.Reverse;
end;

function TsgRecordList<T>.Get(Index: Integer): PItem;
begin
  Result := PItem(FList.Get(Index)^);
end;

function TsgRecordList<T>.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(FList);
end;

procedure TsgRecordList<T>.Put(Index: Integer; Item: PItem);
begin
  FList.Put(Index, Item);
end;

function TsgRecordList<T>.GetCount: Integer;
begin
    Result := FList.GetCount;
end;

procedure TsgRecordList<T>.SetCount(Value: Integer);
begin
  FList.SetCount(Value);
end;

function TsgRecordList<T>.IsEmpty: Boolean;
begin
  Result := FList.Count = 0;
end;

{$EndRegion}

{$Region 'TCustomForwardList.TEnumerator'}

constructor TCustomForwardList.TEnumerator.From(List: PCustomForwardList);
begin
  FItem := PItem(@List.FHead);
end;

function TCustomForwardList.TEnumerator.GetCurrent: PItem;
begin
  Result := FItem;
end;

function TCustomForwardList.TEnumerator.MoveNext: Boolean;
begin
  Result := (FItem <> nil) and (FItem.next <> nil) and (FItem.next.next <> nil);
  if Result then FItem := FItem.next;
end;

{$EndRegion}

{$Region 'TCustomForwardList'}

procedure TCustomForwardList.Init(Meta: PsgItemMeta);
begin
  FRegion := SysCtx.CreateRegion(Meta);
  FHead := FRegion.Region.Alloc(FRegion.ItemSize);
  FLast := FHead;
end;

procedure TCustomForwardList.Free;
begin
  FRegion.Free;
end;

function TCustomForwardList.Empty: Boolean;
begin
  Result := FLast = FHead;
end;

function TCustomForwardList.GetCount: Integer;
var
  p: PItem;
begin
  p := FHead;
  Result := 0;
  while p <> FLast do
  begin
    Inc(Result);
    p := p.next;
  end;
end;

procedure TCustomForwardList.Clear;
begin
  FRegion.Region.Clear;
  FHead := FRegion.Region.Alloc(FRegion.ItemSize);
  FLast := FHead;
end;

function TCustomForwardList.Front: PItem;
begin
  Result := FHead;
end;

function TCustomForwardList.PushFront: PItem;
var
  new: PItem;
begin
  new := FRegion.Region.Alloc(FRegion.ItemSize);
  new.next := FHead;
  FHead := new;
  Result := new;
end;

procedure TCustomForwardList.PopFront;
begin
  Check(not Empty, 'PopFront: list empty');
  FHead := FHead.next;
end;

function TCustomForwardList.InsertAfter(const Pos: PItem): PItem;
var
  new: PItem;
begin
  new := FRegion.Region.Alloc(FRegion.ItemSize);
  new.next := Pos.next;
  Pos.next := new;
  Result := new;
end;

procedure TCustomForwardList.Reverse;
var
  q, p, n: PItem;
begin
  if Empty then exit;
  q := FLast;
  p := FHead;
  n := FHead;
  while p <> FLast do
  begin
    n := n.next;
    p.next := q;
    q := p;
    p := n;
  end;
  FHead.next := nil;
  FHead := q;
end;

procedure TCustomForwardList.Sort(Compare: TListSortCompare);
var
  i, n: Integer;
  pa: TsgPointerArray;
  p, q: PItem;
begin
  n := GetCount;
  if n <= 1 then exit;
  pa := TsgPointerArray.From(n);
  try
    p := FHead;
    while p <> FLast do
    begin
      pa.Add(p);
      p := p.next;
    end;
    pa.Sort(Compare);
    q := nil;
    for i := 0 to pa.Count - 1 do
    begin
      p := pa.Items[i];
      if i = 0 then
        FHead := p
      else
        q.next := p;
      q := p;
    end;
    p.next := FLast;
  finally
    pa.Free
  end;
end;

{$EndRegion}

{$Region 'TsgForwardList<T>.TEnumerator'}

constructor TsgForwardList<T>.TEnumerator.From(List: PCustomForwardList);
begin
  FEnumerator := TCustomForwardList.TEnumerator.From(List);
end;

function TsgForwardList<T>.TEnumerator.GetCurrent: PValue;
begin
  Result := @(PItem(FEnumerator.GetCurrent).Value);
end;

function TsgForwardList<T>.TEnumerator.MoveNext: Boolean;
begin
  Result := FEnumerator.MoveNext;
end;

{$EndRegion}

{$Region 'TsgForwardList<T>.TIterator.'}

function TsgForwardList<T>.TIterator.GetValue: PValue;
begin
  Result := @Item.Value;
end;

procedure TsgForwardList<T>.TIterator.Next;
begin
  Item := PItem(Item.Link.next);
end;

function TsgForwardList<T>.TIterator.Eol: Boolean;
begin
  Result := (Item = nil) or (Item.Link.next = nil);
end;

{$EndRegion}

{$Region 'TsgForwardList<T>'}

procedure TsgForwardList<T>.Init(Meta: PsgItemMeta);
begin
  FList.Init(Meta);
end;

procedure TsgForwardList<T>.Free;
begin
  FList.Free;
end;

procedure TsgForwardList<T>.Clear;
begin
  FList.Clear;
end;

function TsgForwardList<T>.Empty: Boolean;
begin
  Result := FList.Empty;
end;

function TsgForwardList<T>.Front: TIterator;
begin
  Result.Item := PItem(FList.Front);
end;

function TsgForwardList<T>.PushFront: TIterator;
begin
  Result.Item := PItem(FList.PushFront);
end;

procedure TsgForwardList<T>.PushFront(const Value: T);
var
  it: TIterator;
begin
  it := PushFront;
  it.Value^ := Value;
end;

function TsgForwardList<T>.InsertAfter(Pos: TIterator): TIterator;
begin
  Result.Item := PItem(FList.InsertAfter(TCustomForwardList.PItem(Pos)));
end;

function TsgForwardList<T>.InsertAfter(Pos: TIterator; const Value: T): TIterator;
begin
  Result.Item := PItem(FList.InsertAfter(TCustomForwardList.PItem(Pos)));
  Result.Item.Value := Value;
end;

procedure TsgForwardList<T>.PopFront;
begin
  FList.PopFront;
end;

procedure TsgForwardList<T>.Reverse;
begin
  FList.Reverse;
end;

procedure TsgForwardList<T>.Sort(Compare: TListSortCompare);
begin
  FList.Sort(Compare);
end;

function TsgForwardList<T>.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(@FList);
end;

function TsgForwardList<T>.GetRegion: PSegmentedRegion;
begin
  Result := FList.FRegion;
end;

function TsgForwardList<T>.GetCount: Integer;
begin
  Result := FList.GetCount;
end;

{$EndRegion}

{$Region 'TCustomLinkedList.TEnumerator'}

constructor TCustomLinkedList.TEnumerator.From(const List: TCustomLinkedList);
begin
  FItem := PItem(@List.FHead);
end;

function TCustomLinkedList.TEnumerator.GetCurrent: PItem;
begin
  Result := FItem;
end;

function TCustomLinkedList.TEnumerator.MoveNext: Boolean;
begin
  Result := (FItem <> nil) and (FItem.next <> nil) and (FItem.next.next <> nil);
  if Result then FItem := FItem.next;
end;

{$EndRegion}

{$Region 'TCustomLinkedList'}

procedure TCustomLinkedList.Init(Meta: PsgItemMeta);
begin
  FRegion := SysCtx.CreateRegion(Meta);
  FHead := FRegion.Region.Alloc(FRegion.ItemSize);
  FLast := FHead;
end;

procedure TCustomLinkedList.Free;
begin
  FRegion.Free;
end;

function TCustomLinkedList.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(Self);
end;

procedure TCustomLinkedList.Clear;
begin
  FRegion.Region.Clear;
  FHead := FRegion.Region.Alloc(FRegion.ItemSize);
  FLast := FHead;
end;

function TCustomLinkedList.Empty: Boolean;
begin
  Result := FLast.prev = nil;
end;

function TCustomLinkedList.Count: Integer;
var
  p: PItem;
begin
  p := FHead;
  Result := 0;
  while p <> FLast do
  begin
    Inc(Result);
    p := p.next;
  end;
end;

function TCustomLinkedList.Front: PItem;
begin
  Result := FHead;
end;

function TCustomLinkedList.Back: PItem;
begin
  Result := FLast.prev;
end;

function TCustomLinkedList.PushFront: PItem;
var
  new: PItem;
begin
  new := FRegion.Region.Alloc(FRegion.ItemSize);
  new.next := FHead;
  FHead.prev := new;
  FHead := new;
  Result := new;
end;

function TCustomLinkedList.PushBack: PItem;
var
  p, new: PItem;
begin
  if FLast.prev = nil then
    Result := PushFront
  else
  begin
    p := FLast.prev;
    new := FRegion.Region.Alloc(FRegion.ItemSize);
    new.next := FLast;
    new.prev := p;
    FLast.prev := new;
    p.next := new;
    Result := new;
  end;
end;

function TCustomLinkedList.Insert(const Pos: PItem): PItem;
var
  new: PItem;
begin
  new := FRegion.Region.Alloc(FRegion.ItemSize);
  new.next := Pos.next;
  new.prev := Pos;
  Pos.next.prev := new;
  Pos.next := new;
  Result := new;
end;

procedure TCustomLinkedList.PopFront;
begin
  Check(not Empty, 'PopFront: list empty');
  FHead := FHead.next;
  FHead.prev := nil;
end;

procedure TCustomLinkedList.PopBack;
var
  p, q: PItem;
begin
  Check(not Empty, 'PopBack: list empty');
  p := FLast.prev;
  q := p.prev;
  if q = nil then
  begin
    FHead := FLast;
    FLast.prev := nil;
  end
  else
  begin
    q.next := FLast;
    FLast.prev := q;
  end;
end;

procedure TCustomLinkedList.Reverse;
var
  q, p, n: PItem;
begin
  if Empty then exit;
  q := FLast;
  p := FHead;
  n := FHead;
  while p <> FLast do
  begin
    n := n.next;
    p.prev := n;
    p.next := q;
    q := p;
    p := n;
  end;
  FLast.prev := FHead;
  FHead.next := nil;
  FHead := q;
  FHead.prev := nil;
end;

procedure TCustomLinkedList.Sort(Compare: TListSortCompare);
var
  i, n: Integer;
  pa: TsgPointerArray;
  p, q: PItem;
begin
  n := Count;
  if n <= 1 then exit;
  pa := TsgPointerArray.From(n);
  try
    p := FHead;
    while p <> FLast do
    begin
      pa.Add(p);
      p := p.next;
    end;
    pa.Sort(Compare);
    q := nil;
    for i := 0 to pa.Count - 1 do
    begin
      p := pa.Items[i];
      if i = 0 then
        FHead := p
      else
        q.next := p;
      p.prev := q;
      q := p;
    end;
    p.next := FLast;
  finally
    pa.Free
  end;
end;

{$EndRegion}

{$Region 'TsgLinkedList<T>.TEnumerator'}

constructor TsgLinkedList<T>.TEnumerator.From(const List: TCustomLinkedList);
begin
  FEnumerator := TCustomLinkedList.TEnumerator.From(List);
end;

function TsgLinkedList<T>.TEnumerator.GetCurrent: PValue;
begin
  Result := @(PItem(FEnumerator.GetCurrent).Value);
end;

function TsgLinkedList<T>.TEnumerator.MoveNext: Boolean;
begin
  Result := FEnumerator.MoveNext;
end;

{$EndRegion}

{$Region 'TsgLinkedList<T>.TIterator.'}

function TsgLinkedList<T>.TIterator.GetValue: PValue;
begin
  Result := @Item.Value;
end;

procedure TsgLinkedList<T>.TIterator.Next;
begin
  Item := PItem(Item.Link.next);
end;

procedure TsgLinkedList<T>.TIterator.Prev;
begin
  Item := PItem(Item.Link.prev);
end;

function TsgLinkedList<T>.TIterator.Eol: Boolean;
begin
  Result := (Item = nil) or (Item.Link.next = nil);
end;

function TsgLinkedList<T>.TIterator.Bol: Boolean;
begin
  Result := (Item = nil) or (Item.Link.next = nil);
end;

{$EndRegion}

{$Region 'TsgLinkedList<T>'}

procedure TsgLinkedList<T>.Init(Meta: PsgItemMeta);
begin
  FList.Init(Meta);
end;

procedure TsgLinkedList<T>.Free;
begin
  FList.Free;
end;

procedure TsgLinkedList<T>.Clear;
begin
  FList.Clear;
end;

function TsgLinkedList<T>.Empty: Boolean;
begin
  Result := FList.Empty;
end;

function TsgLinkedList<T>.GetCount: Integer;
begin
  Result := FList.Count;
end;

function TsgLinkedList<T>.GetRegion: PSegmentedRegion;
begin
  Result := FList.FRegion;
end;

function TsgLinkedList<T>.Front: TIterator;
begin
  Result.Item := PItem(FList.Front);
end;

function TsgLinkedList<T>.GetEnumerator: TEnumerator;
begin
  Result := TEnumerator.From(FList);
end;

function TsgLinkedList<T>.Back: TIterator;
begin
  Result.Item := PItem(FList.Back);
end;

function TsgLinkedList<T>.PushFront: TIterator;
begin
  Result.Item := PItem(FList.PushFront);
end;

procedure TsgLinkedList<T>.PushFront(const Value: T);
begin
  PushFront.Value^ := Value;
end;

function TsgLinkedList<T>.PushBack: TIterator;
begin
  Result.Item := PItem(FList.PushBack);
end;

procedure TsgLinkedList<T>.PushBack(const Value: T);
begin
  PushBack.Value^ := Value;
end;

procedure TsgLinkedList<T>.PopFront;
begin
  FList.PopFront;
end;

procedure TsgLinkedList<T>.PopBack;
begin
  FList.PopBack;
end;

function TsgLinkedList<T>.Insert(Pos: TIterator; const Value: T): TIterator;
begin
  Result.Item := PItem(FList.Insert(TCustomLinkedList.PItem(Pos)));
  Result.Item.Value := Value;
end;

procedure TsgLinkedList<T>.Reverse;
begin
  FList.Reverse;
end;

procedure TsgLinkedList<T>.Sort(Compare: TListSortCompare);
begin
  FList.Sort(Compare);
end;

{$EndRegion}

{$Region 'TsgPair<TKey, TValue>'}

constructor TsgPair<TKey, TValue>.From(const Key: TKey; const Value: TValue);
begin
  Self.Key := Key;
  Self.Value := Value;
end;

{$EndRegion}

{$Region 'TsgHashMapIterator<Key, T>'}

class operator TsgHashMapIterator<Key, T>.Equal(
  const a, b: TsgHashMapIterator<Key, T>): Boolean;
begin
  Result := a.it.index = b.it.index;
end;

class operator TsgHashMapIterator<Key, T>.NotEqual(
  const a, b: TsgHashMapIterator<Key, T>): Boolean;
begin
  Result := a.it.index <> b.it.index;
end;

procedure TsgHashMapIterator<Key, T>.Next;
begin
  it.Next;
end;

function TsgHashMapIterator<Key, T>.GetPair: PPair;
begin
  Result := PPair(it.GetKey);
end;

function TsgHashMapIterator<Key, T>.GetKey: PKey;
begin
  Result := PKey(it.GetKey);
end;

function TsgHashMapIterator<Key, T>.GetValue: PItem;
begin
  Result := PItem(it.GetValue);
end;

{$EndRegion}

{$Region 'TsgCustomHashMap.TCollision'}

function TsgCustomHashMap.TCollision.GetPairRef: Pointer;
begin
  Result := PByte(@Self) + sizeof(Pointer);
end;

{$EndRegion}

{$Region 'TsgCustomHashMap.TIterator'}

procedure TsgCustomHashMapIterator.Init(map: PsgCustomHashMap; idx: Integer);
begin
  Self.index := idx;
  Self.map := map;
end;

class operator TsgCustomHashMapIterator.Equal(const a, b: TsgCustomHashMapIterator): Boolean;
begin
  Result := a.index = b.index;
end;

class operator TsgCustomHashMapIterator.NotEqual(const a, b: TsgCustomHashMapIterator): Boolean;
begin
  Result := a.index <> b.index;
end;

function TsgCustomHashMapIterator.GetKey: Pointer;
begin
  Result := PByte(map.FCollisions.GetItemPtr(index)) + sizeof(Pointer);
end;

function TsgCustomHashMapIterator.GetValue: Pointer;
begin
  Result := PByte(GetKey) + map.FPair.Get(1).Offset;
end;

procedure TsgCustomHashMapIterator.Next;
begin
  Inc(index);
end;

{$EndRegion}

{$Region 'TsgCustomHashMap'}

constructor TsgCustomHashMap.From(PairMeta: PsgTupleMeta; ExpectedSize: Integer;
  Hasher: PsgHasher);
var
  EntryMeta, CollisionMeta: PsgItemMeta;
  TabSize: Integer;
begin
  EntryMeta := SysCtx.CreateMeta<TEntry>;
  FEntries := SysCtx.CreateUnbrokenRegion(EntryMeta);
  TabSize := GetEntries(ExpectedSize);
  FEntries.Count := TabSize;
  FPair := PairMeta;
  if Hasher = nil then
    FHasher := TsgHasher.From(PairMeta.Get(0).Meta)
  else
    FHasher := Hasher^;
  CollisionMeta := SysCtx.CreateMeta<TCollision>;
  CollisionMeta.ItemSize := sizeof(TCollision) + FPair.Size;
  FCollisions := SysCtx.CreateRegion(CollisionMeta);
end;

function TsgCustomHashMap.GetCount: Integer;
begin
  Result := FCollisions.Count;
end;

function TsgCustomHashMap.GetEntries(ExpectedSize: Integer): Integer;
begin
  // the size of the entry table must be a prime number
  if ExpectedSize < 1000 then
    Result := 307
  else if ExpectedSize < 3000 then
    Result := 1103
  else if ExpectedSize < 10000 then
    Result := 2903
  else if ExpectedSize < 30000 then
    Result := 19477
  else
    Result := 32469;
end;

procedure TsgCustomHashMap.Free;
begin
  Check(Valid);
  FCollisions.Free;
  FEntries.Free;
  FillChar(Self, sizeof(Self), 0);
end;

function TsgCustomHashMap.Valid: Boolean;
begin
  Result := FEntries.Meta.h.Valid and FCollisions.Meta.h.Valid;
end;

function TsgCustomHashMap.Find(key: Pointer): Pointer;
var
  eidx: Integer;
  p: PCollision;
begin
  eidx := FHasher.GetHash(key) mod FEntries.Count;
  p := PCollision(FEntries.GetItemPtr(eidx));
  while p <> nil do
  begin
    if FHasher.Equals(key, p.GetPairRef) then
    begin
      Result := p.GetPairRef;
      exit;
    end;
    p := p.Next;
  end;
  Result := nil;
end;

function TsgCustomHashMap.Insert(pair: Pointer): Pointer;
var
  eidx: Integer;
  entry: pEntry;
  p, n: PCollision;
begin
  eidx := FHasher.GetHash(pair) mod FEntries.Count;
  entry := FEntries.GetItemPtr(eidx);
  p := entry.root;
  while p <> nil do
  begin
    if FHasher.Equals(@pair, p.GetPairRef) then
      exit(p.GetPairRef);
    p := p.Next;
  end;
  // Insert collision at the beginning of the list
  n := FCollisions.AddItem;
  n.Next := entry.root;
  FPair.Assign(n.GetPairRef, pair);
  entry.root := n;
  Result := n.GetPairRef;
end;

function TsgCustomHashMap.InsertOrAssign(pair: Pointer): Pointer;
var
  eidx: Integer;
  entry: pEntry;
  p, n: PCollision;
begin
  eidx := FHasher.GetHash(pair) mod FEntries.Count;
  if eidx < 0 then
        eidx := 0;

  entry := FEntries.GetItemPtr(eidx);
  p := entry.root;
  while p <> nil do
  begin
    if FHasher.Equals(@pair, p.GetPairRef) then
    begin
      FPair.Assign(p.GetPairRef, pair);
      exit(p.GetPairRef);
    end;
    p := p.Next;
  end;
  // Insert collision at the beginning of the list
  n := FCollisions.AddItem;
  n.Next := entry.root;
  FPair.Assign(n.GetPairRef, pair);
  entry.root := n;
  Result := n.GetPairRef;
end;

function TsgCustomHashMap.GetTemporaryPair: Pointer;
begin
  Result := FCollisions.Region.GetTemporary;
end;

function TsgCustomHashMap.Begins: TsgCustomHashMapIterator;
begin
  Result.Init(@Self, 0);
end;

function TsgCustomHashMap.Ends: TsgCustomHashMapIterator;
begin
  Result.Init(@Self, FCollisions.Count);
end;

{$EndRegion}

{$Region 'TsgHashMap<Key, T>'}

constructor TsgHashMap<Key, T>.From(ExpectedSize: Integer;
  Hasher: PsgHasher; FreePair: TFreeProc);
var
  meta: PsgTupleMeta;
begin
  meta := SysCtx.CreateTupleMeta;
  meta.MakePair<Key, T>(FreePair);
  FMap := TsgCustomHashMap.From(meta, ExpectedSize, Hasher);
end;

procedure TsgHashMap<Key, T>.Free;
begin
  FMap.Free;
end;

function TsgHashMap<Key, T>.GetTemporaryPair: PPair;
begin
  Result := PPair(FMap.GetTemporaryPair);
end;

function TsgHashMap<Key, T>.GetCount: Integer;
begin
  Result := FMap.GetCount;
end;

function TsgHashMap<Key, T>.Find(const k: Key): PPair;
begin
  Result := PPair(FMap.Find(@k));
end;

function TsgHashMap<Key, T>.Insert(const pair: TsgPair<Key, T>): PPair;
begin
  Result := PPair(FMap.Insert(@pair));
end;

function TsgHashMap<Key, T>.InsertOrAssign(const pair: TsgPair<Key, T>): PPair;
begin
  Result := PPair(FMap.InsertOrAssign(@pair));
end;

function TsgHashMap<Key, T>.Begins: TsgHashMapIterator<Key, T>;
begin
  Result.it := FMap.Begins;
end;

function TsgHashMap<Key, T>.Ends: TsgHashMapIterator<Key, T>;
begin
  Result.it := FMap.Ends;
end;

{$EndRegion}

{$Region 'TsgTreeIterator'}

constructor TsgTreeIterator.Init(Root, Sentinel: PNode);
begin
  SetLength(Stack, 3);
  Stack[0] := Sentinel;
  Stack[1] := Root;
  Stack[2] := Root;
end;

function TsgTreeIterator.GetItem: Pointer;
begin
  Result := @Res^;
end;

function TsgTreeIterator.Sentinel: PNode;
begin
  Result := Stack[0];
end;

function TsgTreeIterator.Current: PPNode;
begin
  Result := @Stack[1];
end;

function TsgTreeIterator.Res: PPNode;
begin
  Result := @Stack[2];
end;

procedure TsgTreeIterator.Next;
begin
  while not Empty or (Current^ <> Sentinel) do
  begin
    if Current^ <> Sentinel then
    begin
      Push(Current^);
      Current^ := Current^.left;
    end
    else
    begin
      Current^ := Pop;
      Res^ := Current^;
      Current^ := Current^.right;
      break;
    end;
  end;
end;

function TsgTreeIterator.Pop: Pointer;
var
  Idx: Integer;
begin
  Check(not Empty, 'Stack empty');
  Idx := High(Stack);
  Result := Stack[Idx];
  SetLength(Stack, Idx);
end;

procedure TsgTreeIterator.Push(Item: Pointer);
var
  Idx: Integer;
begin
  Idx := Length(Stack);
  SetLength(Stack, Idx + 1);
  Stack[Idx] := Item;
end;

function TsgTreeIterator.Empty: Boolean;
begin
  Result := Length(Stack) <= 3;
end;

{$EndRegion}

{$Region 'TsgCustomTree'}

procedure TsgCustomTree.TParams.Init(ta: TsgTreeAction; pval: Pointer);
begin
  FillChar(Self, sizeof(TParams), 0);
  Self.action := ta;
  Self.pval := pval;
end;

procedure TsgCustomTree.Init(Meta: PsgItemMeta; Compare: TListSortCompare;
  Update: TUpdateProc);
begin
  Self := Default(TsgCustomTree);
  Region := SysCtx.CreateRegion(Meta);
  Self.Compare := Compare;
  Self.Update := Update;
  CreateNode(Sentinel);
  Root := Sentinel;
end;

procedure TsgCustomTree.Free;
begin
  if Region <> nil then
    Region.Free;
  Self := Default(TsgCustomTree);
end;

procedure TsgCustomTree.Clear;
var
  Compare: TListSortCompare;
  Update: TUpdateProc;
  Meta: PsgItemMeta;
begin
  Meta := Region.Meta;
  Compare := Self.Compare;
  Update := Self.Update;
  Free;
  Init(Meta, Compare, Update);
end;

procedure TsgCustomTree.Find(pval: Pointer; var iter: TsgTreeIterator);
var
  prm: TParams;
begin
  prm.Init(TsgTreeAction.taFind, pval);
  Search(root, prm);
  iter.Init(prm.node, Sentinel);
end;

function TsgCustomTree.Get(pval: Pointer): PNode;
var
  prm: TParams;
begin
  prm.Init(TsgTreeAction.taFind, pval);
  Search(root, prm);
  Result := prm.node;
end;

function TsgCustomTree.Count(pval: Pointer): Integer;
var
  prm: TParams;
begin
  prm.Init(TsgTreeAction.taCount, pval);
  Search(root, prm);
  Result := prm.cnt;
end;

procedure TsgCustomTree.Insert(pval: Pointer);
var
  prm: TParams;
begin
  prm.Init(TsgTreeAction.taInsert, pval);
  Search(root, prm);
end;

procedure TsgCustomTree.InsertOrAssign(pval: Pointer);
var
  prm: TParams;
begin
  prm.Init(TsgTreeAction.taInsertOrAssign, pval);
  Search(root, prm);
end;

procedure TsgCustomTree.Begins(var iter: TsgTreeIterator);
begin
  iter.Init(Root, Sentinel);
end;

function TsgCustomTree.Ends: PNode;
begin
  Result := Sentinel;
end;

procedure TsgCustomTree.Search(var p: PNode; var prm: TParams);
const
  NodeSize = sizeof(TsgTreeIterator.TNode);
var
  q, r: PNode;
  pval: Pointer;
  cmp: Integer;
begin
  if p = Sentinel then
  begin
    // not found
    if prm.action = TsgTreeAction.taFind then
      prm.node := Sentinel
    else
    begin
      CreateNode(p);
      prm.h := 2;
      prm.node := p;
      if prm.action in [TsgTreeAction.taInsert, taInsertOrAssign] then
        Update(p, prm.pval);
    end
  end
  else
  begin
    pval := Pointer(NativeUInt(p) + NodeSize);
    cmp := Compare(prm.pval, pval);
    if cmp < 0 then
    begin
      Search(p.left, prm);
      if prm.h > 0 then
        if p.lh then
        begin
          q := p.left; prm.h := 2; p.lh := False;
          if q.lh then // LL
          begin
            p.left := q.right; q.lh := False;
            q.right := p; p := q;
          end
          else if q.rh then
          begin // LR
            r := q.right; q.rh := False;
            q.right := r.left; r.left := q;
            p.left := r.right; r.right := p; p := r;
          end;
        end
        else
        begin
          Dec(prm.h);
          if prm.h > 0 then p.lh := True;
        end;
    end
    else if cmp > 0 then
    begin
      Search(p.right, prm);
      if prm.h > 0 then
        if p.rh then
        begin
          q := p.right; prm.h := 2; p.rh := False;
          if q.rh then  // RR
          begin
            p.right := q.left;
            q.left := p; q.rh := False; p := q;
          end
          else
          begin  // RL
            r := q.left; q.lh := False;
            q.left := r.right; r.right := q;
            p.right := r.left; r.left := p; p := r;
          end;
        end
        else
        begin
          Dec(prm.h);
          if prm.h > 0 then p.rh := True;
        end;
    end
    else
    begin
      // found
      prm.node := p;
      Inc(prm.cnt);
      prm.h := 0;
      if prm.action = TsgTreeAction.taInsertOrAssign then
        Update(p, prm.pval);
    end;
  end;
end;

procedure TsgCustomTree.CreateNode(var p: PNode);
begin
  p := Region.Region.Alloc(Region.ItemSize);
  p.left := Sentinel;
  p.right := Sentinel;
  p.lh := False;
  p.rh := False;
end;

procedure TsgCustomTree.Inorder(Visit: TNodeProc);
begin
  Self.Visit := Visit;
  Visiter(Root);
end;

procedure TsgCustomTree.Visiter(node: PNode);
begin
  if node <> Sentinel then
  begin
    Visiter(node.left);
    Visit(node);
    Visiter(node.right);
  end;
end;

{$EndRegion}

{$Region 'TsgMapIterator<Key, T>'}

constructor TsgMapIterator<Key, T>.Init(Root, Sentinel: TsgTreeIterator.PNode);
begin
  Iter.Init(Root, Sentinel);
end;

class operator TsgMapIterator<Key, T>.Equal(
  const a: TsgMapIterator<Key, T>; b: PNode): Boolean;
begin
  Result := a.Iter.Res^ = TsgTreeIterator.PNode(b);
end;

class operator TsgMapIterator<Key, T>.NotEqual(
  const a: TsgMapIterator<Key, T>; b: PNode): Boolean;
begin
  Result := a.Iter.Res^ <> TsgTreeIterator.PNode(b);
end;

function TsgMapIterator<Key, T>.GetKey: PKey;
begin
  Result := @PNode(Iter.Res^).k;
end;

function TsgMapIterator<Key, T>.GetValue: PItem;
begin
  Result := @PNode(Iter.Res^).v;
end;

procedure TsgMapIterator<Key, T>.Next;
begin
  Iter.Next;
end;

{$EndRegion}

{$Region 'TsgMap<Key, T>'}

constructor TsgMap<Key, T>.From(Compare: TListSortCompare;
  OnFreeNode: TFreeProc);
var
  Meta: PsgItemMeta;
begin
  Meta := SysCtx.CreateMeta<TsgMapIterator<Key, T>.TNode>(OnFreeNode);
  tree.Init(Meta, Compare, UpdateValue);
end;

procedure TsgMap<Key, T>.Free;
begin
  tree.Free;
end;

procedure TsgMap<Key, T>.Clear;
begin
  tree.Clear;
end;

function TsgMap<Key, T>.Find(const k: Key): TsgMapIterator<Key, T>;
begin
  tree.Find(@k, Result.Iter);
end;

function TsgMap<Key, T>.Count(const k: Key): Integer;
begin
  Result := tree.Count(@k);
end;

function TsgMap<Key, T>.Begins: TsgMapIterator<Key, T>;
begin
  tree.Begins(Result.Iter);
end;

procedure TsgMap<Key, T>.Insert(const pair: TsgPair<Key, T>);
begin
  tree.Insert(@pair);
end;

function TsgMap<Key, T>.Emplace(const k: Key): PNode;
var
  prm: TsgCustomTree.TParams;
begin
  prm.Init(TsgTreeAction.taInsertEmpty, @k);
  tree.Search(tree.root, prm);
  Result := PNode(prm.node);
  Result.k := k;
end;

procedure TsgMap<Key, T>.InsertOrAssign(const pair: TsgPair<Key, T>);
begin
  tree.Insert(@pair);
end;

function TsgMap<Key, T>.Ends: PNode;
begin
  Result := PNode(tree.Ends);
end;

procedure TsgMap<Key, T>.Inorder(Visit: TNodeProc);
begin
  tree.Inorder(TsgCustomTree.TNodeProc(Visit));
end;

function TsgMap<Key, T>.Get(index: Key): PItem;
var
  p: TsgTreeIterator.PNode;
begin
  p := tree.Get(@index);
  if p = tree.Ends then
    Result := nil
  else
    Result := @(PNode(p)^).v;
end;

procedure TsgMap<Key, T>.Put(index: Key; const Value: PItem);
var
  pair: TsgPair<Key, T>;
begin
  pair.Key := index;
  pair.Value := Value^;
  tree.Insert(@pair);
end;

procedure TsgMap<Key, T>.UpdateValue(pnd: TsgCustomTree.PNode; pval: Pointer);
type
  TPr = TsgPair<Key, T>;
  PT = ^Tpr;
begin
  PNode(pnd).pair := PT(pval)^;
end;

{$EndRegion}

{$Region 'TsgSetIterator<Key, T>'}

constructor TsgSetIterator<Key>.Init(Root, Sentinel: PNode);
begin
  Iter.Init(TsgTreeIterator.PNode(Root), TsgTreeIterator.PNode(Sentinel));
end;

class operator TsgSetIterator<Key>.Equal(
  const a: TsgSetIterator<Key>; b: PNode): Boolean;
begin
  Result := a.Iter.Res^ = TsgTreeIterator.PNode(b);
end;

class operator TsgSetIterator<Key>.NotEqual(
  const a: TsgSetIterator<Key>; b: PNode): Boolean;
begin
  Result := a.Iter.Res^ <> TsgTreeIterator.PNode(b);
end;

function TsgSetIterator<Key>.GetKey: PKey;
begin
  Result := @PNode(Iter.Res^).k;
end;

procedure TsgSetIterator<Key>.Next;
begin
  Iter.Next;
end;

{$EndRegion}

{$Region 'TsgSet<Key>'}

procedure TsgSet<Key>.Init(Compare: TListSortCompare; OnFreeNode: TFreeProc);
var
  Meta: PsgItemMeta;
begin
  Meta := SysCtx.CreateMeta<TsgSetIterator<Key>.TNode>(OnFreeNode);
  tree.Init(Meta, Compare, UpdateValue);
end;

procedure TsgSet<Key>.Free;
begin
  tree.Free;
end;

procedure TsgSet<Key>.Clear;
begin
  tree.Clear;
end;

function TsgSet<Key>.Find(const k: Key): TsgSetIterator<Key>;
begin
  tree.Find(@k, Result.Iter);
end;

function TsgSet<Key>.Count(const k: Key): Integer;
begin
  Result := tree.Count(@k);
end;

procedure TsgSet<Key>.Insert(const k: Key);
begin
  tree.Insert(@k);
end;

procedure TsgSet<Key>.Inorder(Visit: TNodeProc);
begin
  tree.Inorder(TsgCustomTree.TNodeProc(Visit));
end;

function TsgSet<Key>.Begins: TsgSetIterator<Key>;
begin
  tree.Begins(Result.Iter);
end;

function TsgSet<Key>.Ends: PNode;
begin
  Result := PNode(tree.Ends);
end;

procedure TsgSet<Key>.UpdateValue(pnd: TsgCustomTree.PNode; pval: Pointer);
type
  PK = ^Key;
begin
  PNode(pnd).k := PK(pval)^;
end;

{$EndRegion}

{$Region 'TSharedRegion'}

procedure TSharedRegion.Init(Meta: PsgItemMeta; Capacity: Cardinal);
var
  Heap: Pointer;
begin
  FRegion.Init(Meta, 4096);
  Heap := FRegion.IncreaseCapacity(Capacity);
  FMemoryManager.Init(Heap, Capacity * FRegion.ItemSize);
  FHandleManager.Init(RegionHandle);
end;

procedure TSharedRegion.Free;
begin
  if FRegion.Meta.h.ManagedType then
    FHandleManager.Traversal(FreeUsed);
  FRegion.Free;
end;

procedure TSharedRegion.FreeUsed(h: hCollection);
var
  descr: TMemoryDescriptor;
begin
  descr.Items := FHandleManager.Get(h);
  descr.Count := FSizes[h.Index];
  ClearManagedTypes(descr);
end;

procedure TSharedRegion.Alloc(var descr: TMemoryDescriptor);
begin
  descr.Items := FMemoryManager.Alloc(descr.Count * ItemSize);
  if descr.Items = nil then
    raise EsgError.Create(EsgError.NotEnoughMemory);
  descr.h := FHandleManager.Add(descr.Items);
  FSizes[descr.h.Index] := descr.Count;
end;

procedure TSharedRegion.FreeMem(var descr: TMemoryDescriptor);
begin
  FHandleManager.Remove(descr.h);
  if FRegion.Meta.h.ManagedType then
    ClearManagedTypes(descr);
  FMemoryManager.FreeMem(descr.Items, descr.Count * ItemSize);
  descr.Clear;
end;

procedure TSharedRegion.ClearManagedTypes(const descr: TMemoryDescriptor);
var
  p: PByte;
  n: Cardinal;
begin
  n := descr.Count;
  p := descr.Items;
  while n > 0 do
  begin
    FRegion.Meta.FreeItem(FRegion.Meta, p);
    p := p + ItemSize;
    Dec(n);
  end;
end;

procedure TSharedRegion.Realloc(var descr: TMemoryDescriptor; Count: Cardinal);
var
  p: Pointer;
begin
  FHandleManager.Get(descr.h);
  p := FMemoryManager.Realloc(descr.Items, descr.Count * ItemSize, Count * ItemSize);
  if p = nil then
    raise EsgError.Create(EsgError.NotEnoughMemory);
  descr.Items := p;
  descr.Count := Count;
end;

function TSharedRegion.GetItemSize: Cardinal;
begin
  Result := FRegion.Meta.ItemSize;
end;

function TSharedRegion.GetMeta: PsgItemMeta;
begin
  Result := FRegion.Meta;
end;

{$EndRegion}

{$Region 'TsgLog'}

procedure TsgLog.Init;
begin
  FLog := TStringList.Create;
end;

procedure TsgLog.Free;
begin
  FreeAndNil(FLog);
end;

procedure TsgLog.SaveToFile(const filename: string);
begin
  FLog.SaveToFile(filename);
  FLog.Clear;
end;

procedure TsgLog.AddLine(const Msg: string);
begin
  FLog.Add(Msg);
end;

procedure TsgLog.print(const Msg: string);
begin
  AddLine(Msg);
end;

procedure TsgLog.print(const Msg: string; const Args: array of const);
var
  i: Integer;
  s, v: string;
  Arg: TVarRec;
begin
  s := Msg;
  for i := 0 to High(Args) do
  begin
    Arg := Args[i];
    case Arg.VType of
      vtInteger:
        v := IntToStr(Arg.VInteger);
      vtInt64:
        v := IntToStr(Arg.VInt64^);
      vtExtended, vtCurrency:
        v := Format('%.4f', [Arg.VExtended^]);
      vtUnicodeString:
        v := string(Arg.VUnicodeString);
      vtChar, vtWideChar:
        v := Char(Arg.VChar);
      vtAnsiString:
        v := string(Arg.VString);
      else
        raise EsgError.Create(EsgError.InvalidParameters);
    end;
    s := s + v;
  end;
  AddLine(s);
end;

procedure TsgLog.Msg(const Msg: string);
begin
  AddLine(Msg);
end;

procedure TsgLog.Msg(const Fmt: string; const Args: array of const);
begin
  AddLine(Format(Fmt, Args));
end;

{$EndRegion}

{$Region 'TsgSystemContext'}

constructor TsgSystemContext.Create;
var
  i: TRegionId;
  dt: PSharedData;
begin
  inherited;
  for i := Low(TRegionId) to High(TRegionId) do
  begin
    dt := @FMetaList[i];
    case i of
      rItemMeta: dt.Meta.Init<TsgItemMeta>;
      rTeMeta: dt.Meta.Init<TsgTupleElementMeta>;
      rTupleMeta: InitTupleMeta(dt.Meta, sizeof(TsgTupleMeta), []);
      rSharedRegion: dt.Meta.Init<TsgArrayHelper>;
    end;
    dt.Region.Init(@dt.Meta, 65536);
    dt.List.Init(@dt.Region, 4096);
  end;
end;

destructor TsgSystemContext.Destroy;
var
  i: TRegionId;
  dt: PSharedData;
begin
  for i := Low(TRegionId) to High(TRegionId) do
  begin
    dt := @FMetaList[i];
    dt.List.Free;
    dt.Region.Free;
  end;
  inherited;
end;

procedure TsgSystemContext.InitTupleMeta(var meta: TsgItemMeta;
  ItemSize: Cardinal; Flags: TRegionFlagSet);
begin
  if rfSegmented in Flags then
    meta.h.Segmented := True;
  if rfRangeCheck in Flags then
    meta.h.RangeCheck := True;
  if rfNotification in Flags then
    meta.h.Notification := True;
  if rfOwnedObject in Flags then
    meta.h.OwnedObject := True;
  meta.ItemSize := ItemSize;
end;

function TsgSystemContext.GetShareRegion(id: TRegionId): PSharedRegion;
begin
  Result := @FMetaList[id].Region;
end;

function TsgSystemContext.CreateMeta<T>(OnFree: TFreeProc = nil): PsgItemMeta;
begin
  Result := PsgItemMeta(FMetaList[rItemMeta].List.Add);
  Result.Init<T>(OnFree);
end;

function TsgSystemContext.CreateMeta<T>(Flags: TRegionFlagSet;
  RemoveAction: TRemoveAction; OnFree: TFreeProc): PsgItemMeta;
begin
  Result := PsgItemMeta(FMetaList[rItemMeta].List.Add);
  Result.Init<T>(Flags, RemoveAction, OnFree);
end;

function TsgSystemContext.CreateTupleMeta: PsgTupleMeta;
begin
  Result := PsgTupleMeta(FMetaList[rTupleMeta].List.Add);
end;

procedure TsgSystemContext.CreateTeMetas(Count: Cardinal;
  var List: TsgTupleMeta.TsgTeMetaList);
begin
  List.Init(@FMetaList[rTeMeta].Region, Count);
end;

function TsgSystemContext.CreateArrayHelper(Capacity: Cardinal): PsgArrayHelper;
begin
  Result := PsgArrayHelper(FMetaList[rSharedRegion].List.Add);
  Result.Init(@FMetaList[rSharedRegion].Region, Capacity);
end;

procedure TsgSystemContext.CreateArray<T>(Capacity: Cardinal; var Value: TsgArray<T>);
begin
  Value.FList := CreateArrayHelper(Capacity)^;
end;

{$EndRegion}

procedure InitSysCtx;
begin
  TsgSystemContext.InitMetadata;
  SysCtx := TsgSystemContext.Create;
end;

procedure ClearSysCtx;
begin
  FreeAndNil(SysCtx);
end;

initialization
  InitSysCtx;

finalization
  ClearSysCtx;

end.

