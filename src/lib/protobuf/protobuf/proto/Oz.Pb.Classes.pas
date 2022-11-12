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
unit Oz.Pb.Classes;
interface
uses
  System.Classes, System.SysUtils, System.Math, System.Rtti, System.TypInfo,
  Oz.Pb.StrBuffer, Oz.SGL.Heap, Oz.SGL.Collections;
{$T+}
{$HINTS OFF}

const
  TAG_TYPE_BITS = 3;
  TAG_TYPE_MASK = (1 shl TAG_TYPE_BITS) - 1;
  RecursionLimit = 64;
type
{$Region 'EProtobufError'}
  EProtobufError = class(Exception)
  const
    NotImplemented = 0;
    InvalidEndTag = 1;
    InvalidWireType = 2;
    InvalidSize = 3;
    RecursionLimitExceeded = 4;
    MalformedVarint = 5;
    EofEncounterd = 6;
    NegativeSize = 7;
    TruncatedMessage = 8;
    Impossible = 99;
  public
    constructor Create(ErrNo: Integer); overload;
  end;
{$EndRegion}
{$Region 'TWire'}
  TWireType = 0..7;
  TWire = record
  const
    VARINT = 0;
    FIXED64 = 1;
    LENGTH_DELIMITED = 2;
    START_GROUP = 3;
    END_GROUP = 4;
    FIXED32 = 5;
    Names: array [VARINT .. FIXED32] of string = (
      'VARINT',
      'FIXED64',
      'LENGTH_DELIMITED',
      'START_GROUP',
      'END_GROUP',
      'FIXED32');
  end;
{$EndRegion}
{$Region 'TpbTag: Proto field tag'}
  TpbTag = record
  var
    v: Integer;
  public
    // Given a tag value, determines the field number (the upper 29 bits).
    function FieldNumber: Integer; inline;
    // Given a tag value, determines the wire type (the lower 3 bits).
    function WireType: TWireType; inline;
    // Makes a tag value given a field number and wire type.
    procedure MakeTag(FieldNo: Integer; WireType: TWireType); inline;
  end;
{$EndRegion}
{$Region 'TpbOneof: Variant field'}
  TpbOneof = record
    tag: Integer;
    value: TValue;
  end;
{$EndRegion}
{$Region 'TpbInput: Decode data from the buffer and place them to object fields'}
  PpbInput = ^TpbInput;
  TpbInput = record
  const
    RECURSION_LIMIT = 64;
    SIZE_LIMIT = 64 shl 20;  // 64 mb
  private var
    FBuf: PByte;
    FLast: PByte;
    FCurrent: PByte;
    FLastTag: TpbTag;
    FOwnsData: Boolean;
    FRecursionDepth: ShortInt;
    FStack: array [0 .. RECURSION_LIMIT - 1] of PByte;
  public
    procedure Init; overload;
    procedure Init(const pb: TpbInput); overload;
    procedure Init(Buf: PByte; BufSize: Integer; OwnsData: Boolean); overload;
    class function From(const Buf: TBytes): TpbInput; overload; static;
    class function From(Buf: PByte; BufSize: Integer;
      OwnsData: Boolean = False): TpbInput; overload; static;
    procedure Free;
    // I/O routines to file and stream
    procedure SaveToStream(Stream: TStream);
    procedure SaveToFile(const FileName: string);
    procedure LoadFromFile(const FileName: string);
    procedure LoadFromStream(Stream: TStream);
    // Merge messages
    procedure mergeFrom(const builder: TpbInput);
    // Read message length, push current FLast to stack, and calc new FLast
    procedure Push;
    // Restore FLast
    procedure Pop;
    // No more data
    function Eof: Boolean;
    /// <summary>
    /// Peeks at the next tag in the stream. If it matches <paramref name="tag"/>,
    /// the tag is consumed and the method returns <c>true</c>; otherwise, the
    /// stream is left in the original position and the method returns <c>false</c>.
    /// </summary>
    function ConsumeTag(tag : Integer): Boolean;
    // Attempt to read a field tag, returning zero if we have reached EOF
    function readTag: TpbTag;
    // Check whether the latter match the value read tag
    // Used to test for nested groups
    procedure checkLastTagWas(value: Integer);
    // Reads and discards a Single field, given its tag value
    function skipField(tag: TpbTag): Boolean;
    // Reads and discards an entire message
    procedure skipMessage;
    // Read a Double field value
    function readDouble: Double;
    // Read a float field value
    function readFloat: Single;
    // Read an Int64 field value
    function readInt64: Int64;
    // Read an Int32 field value
    function readInt32: Integer;
    // Read a fixed64 field value
    function readFixed64: Int64;
    // Read a fixed32 field value
    function readFixed32: Integer;
    // Read a Boolean field value
    function readBoolean: Boolean;
    // Read a string field value
    function readString: string;
    // Read nested message
    procedure readMessage(builder: PpbInput);
    // Read a uint32 field value
    function readUInt32: Integer;
    // Read a enum field value
    function readEnum: Integer;
    // Read an sfixed32 field value
    function readSFixed32: Integer;
    // Read an sfixed64 field value
    function readSFixed64: Int64;
    // Read an sint32 field value
    function readSInt32: Integer;
    // Read an sint64 field value
    function readSInt64: Int64;
    // Read a raw Varint from the stream,
    // if larger than 32 bits, discard the upper bits
    function readRawVarint32: Integer;
    // Read a raw Varint
    function readRawVarint64: Int64;
    // Read a 32-bit little-endian Integer
    function readRawLittleEndian32: Integer;
    // Read a 64-bit little-endian Integer
    function readRawLittleEndian64: Int64;
    // Read one byte
    function readRawByte: ShortInt;
    // Read "size" bytes
    procedure readRawBytes(var data; size: Integer);
    function readBytes: TBytes;
    // Skip "size" bytes
    procedure skipRawBytes(size: Integer);
    // Return the content as a string
    function ToString(lo: Integer = 0): string;
  end;
{$EndRegion}
{$Region 'TpbOutput: Encode the object fields and and write them to buffer'}
  PpbOutput = ^TpbOutput;
  TpbOutput = record
  private
    FBuffer: TSegmentBuffer;
  public
    class function From: TpbOutput; static;
    procedure Free;
    procedure Clear;
    procedure SaveToStream(Stream: TStream); inline;
    procedure SaveToFile(const FileName: string); inline;
    // Encode and write varint
    procedure writeRawVarint32(value: Integer);
    // Encode and write varint
    procedure writeRawVarint64(value: Int64);
    // Encode and write tag
    procedure writeTag(fieldNo: Integer; wireType: Integer);
    // Encode and write single byte
    procedure writeRawByte(value: ShortInt); inline;
    // Encode and write bytes
    procedure writeRawBytes(const value: TBytes);
    // Encode and write string
    procedure writeRawString(const value: string);
    // Write the data with specified size
    procedure writeRawData(const p: Pointer; size: Integer); inline;
    // Write a Double field, including tag
    procedure writeDouble(fieldNo: Integer; value: Double);
    // Write a Single field, including tag
    procedure writeFloat(fieldNo: Integer; value: Single);
    // Write a Int64 field, including tag
    procedure writeInt64(fieldNo: Integer; value: Int64);
    // Write a Int64 field, including tag
    procedure writeInt32(fieldNo: Integer; value: Integer);
    // Write a fixed64 field, including tag
    procedure writeFixed64(fieldNo: Integer; value: Int64);
    // Write a fixed32 field, including tag
    procedure writeFixed32(fieldNo: Integer; value: Integer);
    // Write a Boolean field, including tag
    procedure writeBoolean(fieldNo: Integer; value: Boolean);
    // Write a string field, including tag
    procedure writeString(fieldNo: Integer; const value: string);
    // Write a bytes field, including tag
    procedure writeBytes(fieldNo: Integer; const value: TBytes);
    // Write a message field, including tag
    procedure writeMessage(fieldNo: Integer; const msg: TpbOutput);
    //  Write a unsigned Int32 field, including tag
    procedure writeUInt32(fieldNo: Integer; value: Cardinal);
    // Get the result as a bytes
    function GetBytes: TBytes; inline;
    // Get serialized size
    function getSerializedSize: Integer;
    // Write to buffer
    procedure writeTo(buffer: TpbOutput);
    // Return the content as a string
    function ToString(lo: Integer = 0): string;
  end;
{$EndRegion}
{$Region 'TpbLoader: Load object'}
  TpbLoader = record
  private
    Fpb: TpbInput;
    function GetPb: PpbInput; inline;
  public
    procedure Init; inline;
    procedure Free; inline;
    property Pb: PpbInput read GetPb;
  end;
{$EndRegion}
{$Region 'TpbSaver: Save a object'}
  TpbSaver = record
  private
    Fpb: TpbOutput;
    function GetPb: PpbOutput; inline;
  public
    procedure Init; inline;
    procedure Free; inline;
    procedure Clear; inline;
    property Pb: PpbOutput read GetPb;
  end;
{$EndRegion}
{$Region 'TpbOps: Save and Load procedures for type'}
  TSaveProc = procedure(const S: TpbSaver; const [Ref] Value);
  TLoadProc = procedure(const L: TpbLoader; var Value);
  PObjMeta = ^TObjMeta;
  PPropMeta = ^TPropMeta;
  TSaveObj = procedure(om: PObjMeta; const S: TpbSaver; const [Ref] Value);
  TLoadObj = procedure(om: PObjMeta; const L: TpbLoader; var Value);
  TpbFieldKind = (
    fkSingleProp, // Single property
    fkObj,        // record or object
    fkList,       // TsgRecordList<T>
    fkObjList,
    fkMap,        // TsgHashMap<Key, T>
    fkObjMap);
  PpbOps = ^TpbOps;
  TpbOps = record
  public
    // Find the right read/save procedure for the specified type
    class function From<T>: TpbOps; overload; static;
    class function From(info: PTypeInfo; size: Integer): TpbOps; overload; static;
    // Create for user defined type
    class function From(om: PObjMeta): TpbOps; overload; static;
    // Save property to pb
    procedure SaveTo(const S: TpbSaver; const [Ref] Value); inline;
    // Load property from pb
    procedure LoadFrom(const L: TpbLoader; var Value); inline;
    // Returns wire
    function GetWire: Integer;
  private
    info: PTypeInfo;
    om: PObjMeta;
    case TpbFieldKind of
      fkSingleProp: (
        Save: TSaveProc;
        Load: TLoadProc);
      fkObj, fkList, fkMap: (
        SaveObj: TSaveObj;
        LoadObj: TLoadObj);
  end;
{$EndRegion}
{$Region 'TFieldParam: Field paramater'}
  TFieldParam = record
    name: AnsiString;
    fieldNumber: Integer;
    offset: Integer;
    constructor From(const name: AnsiString; fno, offset: Integer);
  end;
{$EndRegion}
{$Region 'TPropMeta: Metadata for serializing the property'}
  TPropMeta = record
  type
    PValue = ^TValue;
  private
    name: AnsiString; // name for xml/json
    offset: Word;
    kind: TpbFieldKind;
    defValue: PValue;
    tag: TpbTag;
    ops: TpbOps;
    // Get pointer to field of object
    function GetField(const [Ref] Obj): Pointer;
  public
    procedure Init(kind: TpbFieldKind; const name: AnsiString;
      fno, offset: Integer; const ops: TpbOps);
    function EqualToDefault(var field): Boolean;
    function ToString: string;
    procedure SetDefValue<T>(const Value: T);
  end;
{$EndRegion}
{$Region 'TObjMeta: Metadata for serializing object'}
  TObjMeta = record
  type
    TGetProp = function(om: PObjMeta; fno: Integer): PPropMeta;
    TGetPropBy = (getByBinary, getByFind, getByIndex);
    TObjectMethod = procedure(var obj);
  var
    info: PTypeInfo;
    props: TArray<TPropMeta>;
  private
    FGetProp: TGetProp;
    FInit: TObjectMethod;
    class function PropByBinary(om: PObjMeta; fno: Integer): PPropMeta; static;
    class function PropByFind(om: PObjMeta; fno: Integer): PPropMeta; static;
    class function PropByIndex(om: PObjMeta; fno: Integer): PPropMeta; static;
    procedure SaveList(pm: PPropMeta; const S: TpbSaver; const [Ref] obj);
    procedure SaveMap(pm: PPropMeta; const S: TpbSaver; const [Ref] obj);
    procedure LoadList(const tag: TpbTag; pm: PPropMeta; const L: TpbLoader; var obj);
    procedure LoadMap(const tag: TpbTag; pm: PPropMeta; const L: TpbLoader; var obj);
    procedure SaveProps(const S: TpbSaver; const [Ref] obj);
    procedure LoadProps(const L: TpbLoader; var obj);
  public
    class function From<T>(Init: TObjectMethod; get: TGetPropBy = getByBinary): TObjMeta; static;
    // Save instance to pb
    class procedure SaveTo(om: PObjMeta; const S: TpbSaver; const [Ref] obj); static;
    // Load instance from pb
    class procedure LoadFrom(om: PObjMeta; const L: TpbLoader; var obj); static;
    // Add metadata for standard type
    procedure Add<T>(const name: AnsiString; fno, offset: Integer);
    // Add metadata for map collection
    procedure AddMap<Key>(kind: TpbFieldKind; const name: AnsiString;
      fno, offset: Integer; const ops: TpbOps);
    // Add metadata for user defined type
    procedure AddObj(kind: TpbFieldKind; const name: AnsiString;
      fno, offset: Integer; const ops: TpbOps);
    function ToString: string;
    // Get property
    property GetProp: TGetProp read FGetProp;
    // Init instance
    property Init: TObjectMethod read FInit;
  end;
{$EndRegion}
{$Region 'Procedures'}
function decodeZigZag32(n: Integer): Integer;
function decodeZigZag64(n: Int64): Int64;
{$EndRegion}
implementation
{$Region 'Procedures'}
function decodeZigZag32(n: Integer): Integer;
begin
  Result := (n shr 1) xor -(n and 1);
end;
function decodeZigZag64(n: Int64): Int64;
begin
  Result := (n shr 1) xor -(n and 1);
end;
{$EndRegion}
{$Region 'EProtobufError'}
constructor EProtobufError.Create(ErrNo: Integer);
var Msg: string;
begin
  case ErrNo of
    NotImplemented: Msg := 'Not implemented';
    InvalidEndTag: Msg := 'Pb: invalid end tag';
    InvalidWireType: Msg := 'Pb: invalid wire type';
    InvalidSize: Msg := 'Pb: readString (size <= 0)';
    RecursionLimitExceeded: Msg := 'Pb: recursion Limit Exceeded';
    MalformedVarint: Msg := 'Pb: malformed Varint';
    EofEncounterd: Msg := 'Pb: eof encounterd';
    NegativeSize: Msg := 'Pb: negative Size';
    TruncatedMessage: Msg := 'Pb: truncated Message';
    Impossible: Msg := 'Impossible';
    else Msg := 'Error: ' + IntToStr(ErrNo);
  end;
  Create(Msg);
end;
{$EndRegion}
{$Region 'TpbTag'}
function TpbTag.FieldNumber: Integer;
begin
  Result := v shr TAG_TYPE_BITS;
end;
function TpbTag.WireType: TWireType;
begin
  result := v and TAG_TYPE_MASK;
end;
procedure TpbTag.MakeTag(FieldNo: Integer; WireType: TWireType);
begin
  v := (FieldNo shl TAG_TYPE_BITS) or wireType;
end;
{$EndRegion}
{$Region 'TpbInput'}
procedure TpbInput.Init;
begin
  Self := Default(TpbInput);
end;
procedure TpbInput.Init(Buf: PByte; BufSize: Integer; OwnsData: Boolean);
begin
  FOwnsData := OwnsData;
  FRecursionDepth := 0;
  if not OwnsData then
    FBuf := Buf
  else
  begin
    // allocate a buffer and copy the data
    GetMem(FBuf, BufSize);
    Move(Buf^, FBuf^, BufSize);
  end;
  FCurrent := FBuf;
  FLast := FBuf + BufSize;
end;
procedure TpbInput.Init(const pb: TpbInput);
begin
  FBuf := pb.FBuf;
  FCurrent := FBuf;
  FLast := pb.FLast;
  Self.FOwnsData := False;
end;
class function TpbInput.From(const Buf: TBytes): TpbInput;
begin
  Result.Init(@Buf[0], Length(Buf), False);
end;
class function TpbInput.From(Buf: PByte; BufSize: Integer;
  OwnsData: Boolean = False): TpbInput;
begin
  Result.Init(Buf, BufSize, OwnsData);
end;
function TpbInput.Eof: Boolean;
begin
  Result := FCurrent >= FLast;
end;
function TpbInput.ConsumeTag(tag : Integer): Boolean;
begin
    Result := False;
    var tTag := readTag;
    if tTag.v = tag then
       Result := True;
end;
procedure TpbInput.Free;
begin
  if FOwnsData then
    FreeMem(FBuf);
  Self := Default(TpbInput);
end;
function TpbInput.readTag: TpbTag;
begin
  if FCurrent < FLast then
    FLastTag.v := readRawVarint32
  else
    FLastTag.v := 0;
  Result := FLastTag;
end;
procedure TpbInput.checkLastTagWas(value: Integer);
begin
  if FLastTag.v <> value then
    raise EProtobufError.Create(EProtobufError.InvalidEndTag);
end;
function TpbInput.skipField(tag: TpbTag): Boolean;
begin
  Result := True;
  case tag.WireType of
    TWire.VARINT:
      readInt32;
    TWire.FIXED64:
      readRawLittleEndian64;
    TWire.LENGTH_DELIMITED:
      skipRawBytes(readRawVarint32);
    TWire.FIXED32:
      readRawLittleEndian32;
    else
      raise EProtobufError.Create('Protocol buffer: invalid WireType');
  end;
end;
procedure TpbInput.skipMessage;
var tag: TpbTag;
begin
  repeat
    tag := readTag;
  until (tag.v = 0) or (not skipField(tag));
end;
function TpbInput.readDouble: Double;
begin
  readRawBytes(Result, SizeOf(Double));
end;
function TpbInput.readFloat: Single;
begin
  readRawBytes(Result, SizeOf(Single));
end;
function TpbInput.readInt64: Int64;
begin
  Result := readRawVarint64;
end;
function TpbInput.readInt32: Integer;
begin
  Result := readRawVarint32;
end;
function TpbInput.readFixed64: Int64;
begin
  Result := readRawLittleEndian64;
end;
function TpbInput.readFixed32: Integer;
begin
  Result := readRawLittleEndian32;
end;
function TpbInput.readBoolean: Boolean;
begin
  Result := readRawVarint32 <> 0;
end;
function TpbInput.readString: string;
var
  buf, text: TBytes;
begin
  // Decode utf8 to string
  buf := readBytes;
  text := TEncoding.UTF8.Convert(TEncoding.UTF8, TEncoding.Unicode, buf);
  Result := TEncoding.Unicode.GetString(text);
end;
procedure TpbInput.readMessage(builder: PpbInput);
begin
  readRawVarint32;
  if FRecursionDepth >= RECURSION_LIMIT then
    raise EProtobufError.Create(EProtobufError.RecursionLimitExceeded);
  Inc(FRecursionDepth);
  builder.mergeFrom(Self);
  checkLastTagWas(0);
  dec(FRecursionDepth);
end;
function TpbInput.readUInt32: Integer;
begin
  Result := readRawVarint32;
end;
function TpbInput.readEnum: Integer;
begin
  Result := readRawVarint32;
end;
function TpbInput.readSFixed32: Integer;
begin
  Result := readRawLittleEndian32;
end;
function TpbInput.readSFixed64: Int64;
begin
  Result := readRawLittleEndian64;
end;
function TpbInput.readSInt32: Integer;
begin
  Result := decodeZigZag32(readRawVarint32);
end;
function TpbInput.readSInt64: Int64;
begin
  Result := decodeZigZag64(readRawVarint64);
end;
function TpbInput.readRawVarint32: Integer;
var
  tmp: ShortInt;
  shift: Integer;
begin
  shift := 0;
  Result := 0;
  repeat
    // for negative numbers number value may be to 10 byte
    if shift >= 64 then
      raise EProtobufError.Create(EProtobufError.MalformedVarint);
    tmp := readRawByte;
    Result := Result or ((tmp and $7f) shl shift);
    Inc(shift, 7);
  until tmp >= 0;
end;
function TpbInput.readRawVarint64: Int64;
var
  tmp: ShortInt;
  shift: Integer;
  i64: Int64;
begin
  shift := -7;
  Result := 0;
  repeat
    Inc(shift, 7);
    if shift >= 64 then
      raise EProtobufError.Create(EProtobufError.MalformedVarint);
    tmp := readRawByte;
    i64 := tmp and $7f;
    i64 := i64 shl shift;
    Result := Result or i64;
  until tmp >= 0;
end;
function TpbInput.readRawLittleEndian32: Integer;
begin
  readRawBytes(Result, SizeOf(Result));
end;
function TpbInput.readRawLittleEndian64: Int64;
begin
  readRawBytes(Result, SizeOf(Result));
end;
function TpbInput.readRawByte: ShortInt;
begin
  if FCurrent > FLast then
    raise EProtobufError.Create(EProtobufError.EofEncounterd);
  Result := ShortInt(FCurrent^);
  Inc(FCurrent);
end;
procedure TpbInput.readRawBytes(var data; size: Integer);
begin
  if FCurrent > FLast then
    raise EProtobufError.Create(EProtobufError.EofEncounterd);
  Move(FCurrent^, data, size);
  Inc(FCurrent, size);
end;
function TpbInput.readBytes: TBytes;
var
  size: Integer;
begin
  size := readRawVarint32;
  if size <= 0 then
     size := size ;//raise EProtobufError.Create(EProtobufError.InvalidSize);
  if FCurrent > FLast then
    raise EProtobufError.Create(EProtobufError.EofEncounterd);
  SetLength(Result, size);
  Move(FCurrent^, Pointer(Result)^, size);
  Inc(FCurrent, size);
end;
procedure TpbInput.skipRawBytes(size: Integer);
begin
  if size < 0 then
    raise EProtobufError.Create(EProtobufError.NegativeSize);
  if FCurrent > FLast then
    raise EProtobufError.Create(EProtobufError.TruncatedMessage);
  Inc(FCurrent, size);
end;
function TpbInput.ToString(lo: Integer): string;
var
  p, hi: PByte;
  s: string;
begin
  Result := '';
  hi := FCurrent;
  p := FBuf + lo;
  while p < hi do
  begin
    s := IntToHex(p^, 2);
    Inc(p);
    if Result = '' then
      Result := s
    else
      Result := Result + ' ' + s;
  end;
end;
procedure TpbInput.SaveToFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToStream(Stream);
  finally
    Stream.Free;
  end;
end;
procedure TpbInput.SaveToStream(Stream: TStream);
begin
  Stream.WriteBuffer(Pointer(FBuf)^, Cardinal(FLast) - Cardinal(FBuf) + 1);
end;
procedure TpbInput.LoadFromFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromStream(Stream);
  finally
    Stream.Free;
  end;
end;
procedure TpbInput.LoadFromStream(Stream: TStream);
var
  Size: Integer;
begin
  if FOwnsData then begin
    FreeMem(FBuf);
    FBuf := nil;
  end;
  FOwnsData := True;
  Size := Stream.Size;
  GetMem(FBuf, Size);
  FCurrent := FBuf;
  FLast := FBuf + Size;
  Stream.Position := 0;
  Stream.Read(Pointer(FBuf)^, Size);
end;
procedure TpbInput.mergeFrom(const builder: TpbInput);
begin
  raise EProtobufError.Create(EProtobufError.NotImplemented);
end;
procedure TpbInput.Push;
var
  Size: Integer;
  Last: PByte;
begin
  var position := FCurrent - FBuf ;
  if position >= $8c69 then
       position := position;

  FStack[FRecursionDepth] := FLast;
  Inc(FRecursionDepth);
  Size := readInt32;
  Last := FCurrent + Size;
  Assert(Last <= FLast);
  FLast := Last;
end;
procedure TpbInput.Pop;
begin
   if FCurrent <> FLast then
       var position := FCurrent - FBuf ;

  Assert(FCurrent = FLast);
  dec(FRecursionDepth);
  FLast := FStack[FRecursionDepth];
end;
{$EndRegion}
{$Region 'TpbOutput'}
class function TpbOutput.From: TpbOutput;
begin
  Result.FBuffer := TSegmentBuffer.Create;
end;
procedure TpbOutput.Free;
begin
  FreeAndNil(FBuffer);
end;
procedure TpbOutput.Clear;
begin
  FBuffer.Clear;
end;
procedure TpbOutput.writeRawByte(value: ShortInt);
begin
  // -128..127
  FBuffer.Add(Byte(value));
end;
procedure TpbOutput.writeRawBytes(const value: TBytes);
begin
  writeRawVarint32(Length(value));
  FBuffer.Add(value);
end;
procedure TpbOutput.writeRawString(const value: string);
var
  bytes, text: TBytes;
begin
  bytes := TEncoding.Unicode.GetBytes(value);
  text := TEncoding.Unicode.Convert(TEncoding.Unicode, TEncoding.UTF8, bytes);
  writeRawVarint32(Length(text));
  FBuffer.Add(text);
end;
procedure TpbOutput.writeRawData(const p: Pointer; size: Integer);
begin
  FBuffer.Add(p, size);
end;
procedure TpbOutput.writeTag(fieldNo, wireType: Integer);
var
  tag: TpbTag;
begin
  tag.MakeTag(fieldNo, wireType);
  writeRawVarint32(tag.v);
end;
procedure TpbOutput.writeRawVarint32(value: Integer);
var b: ShortInt;
begin
  repeat
    b := value and $7F;
    value := value shr 7;
    if value <> 0 then
      b := b + $80;
    writeRawByte(b);
  until value = 0;
end;
procedure TpbOutput.writeRawVarint64(value: Int64);
var b: ShortInt;
begin
  repeat
    b := value and $7F;
    value := value shr 7;
    if value <> 0 then
      b := b + $80;
    writeRawByte(b);
  until value = 0;
end;
procedure TpbOutput.writeBoolean(fieldNo: Integer; value: Boolean);
begin
  writeTag(fieldNo, TWire.VARINT);
  writeRawByte(ord(value));
end;
procedure TpbOutput.writeDouble(fieldNo: Integer; value: Double);
begin
  writeTag(fieldNo, TWire.FIXED64);
  writeRawData(@value, SizeOf(value));
end;
procedure TpbOutput.writeFloat(fieldNo: Integer; value: Single);
begin
  writeTag(fieldNo, TWire.FIXED32);
  writeRawData(@value, SizeOf(value));
end;
procedure TpbOutput.writeFixed32(fieldNo, value: Integer);
begin
  writeTag(fieldNo, TWire.FIXED32);
  writeRawData(@value, SizeOf(value));
end;
procedure TpbOutput.writeFixed64(fieldNo: Integer; value: Int64);
begin
  writeTag(fieldNo, TWire.FIXED64);
  writeRawData(@value, SizeOf(value));
end;
procedure TpbOutput.writeInt32(fieldNo, value: Integer);
begin
  writeTag(fieldNo, TWire.VARINT);
  writeRawVarint32(value);
end;
procedure TpbOutput.writeInt64(fieldNo: Integer; value: Int64);
begin
  writeTag(fieldNo, TWire.VARINT);
  writeRawVarint64(value);
end;
procedure TpbOutput.writeString(fieldNo: Integer; const value: string);
begin
  writeTag(fieldNo, TWire.LENGTH_DELIMITED);
  writeRawString(value);
end;
procedure TpbOutput.writeBytes(fieldNo: Integer; const value: TBytes);
begin
  writeTag(fieldNo, TWire.LENGTH_DELIMITED);
  writeRawBytes(value);
end;
procedure TpbOutput.writeUInt32(fieldNo: Integer; value: Cardinal);
begin
  writeTag(fieldNo, TWire.VARINT);
  writeRawVarint32(value);
end;
procedure TpbOutput.writeMessage(fieldNo: Integer; const msg: TpbOutput);
var sz: Integer;
begin
  writeTag(fieldNo, TWire.LENGTH_DELIMITED);
  sz := msg.getSerializedSize;
  writeRawVarint32(sz);
  msg.writeTo(Self);
end;
function TpbOutput.GetBytes: TBytes;
begin
  result := FBuffer.GetBytes;
end;
procedure TpbOutput.SaveToFile(const FileName: string);
begin
  FBuffer.SaveToFile(FileName);
end;
procedure TpbOutput.SaveToStream(Stream: TStream);
begin
  FBuffer.SaveToStream(Stream);
end;
function TpbOutput.getSerializedSize: Integer;
begin
  result := FBuffer.GetCount;
end;
procedure TpbOutput.writeTo(buffer: TpbOutput);
begin
  buffer.FBuffer.Add(GetBytes);
end;
function TpbOutput.ToString(lo: Integer): string;
var
  i, hi: Integer;
  bytes: TBytes;
  s: string;
begin
  hi := FBuffer.GetCount - 1;
  bytes := FBuffer.GetBytes;
  for i := lo to hi do
  begin
    s := IntToHex(bytes[i], 2);
    if i = lo then
      Result := s
    else
      Result := Result + ' ' + s;
  end;
end;
{$EndRegion}
{$Region 'TpbCustomLoader: Base class for a load object'}
procedure TpbLoader.Init;
begin
  FPb.Init;
end;
procedure TpbLoader.Free;
begin
  FPb.Free;
end;
function TpbLoader.GetPb: PpbInput;
begin
  Result := @FPb;
end;
{$EndRegion}
{$Region 'TpbSaver: Base class save a object'}
procedure TpbSaver.Init;
begin
  FPb := TpbOutput.From;
end;
procedure TpbSaver.Free;
begin
  FPb.Free;
end;
procedure TpbSaver.Clear;
begin
  FPb.Clear;
end;
function TpbSaver.GetPb: PpbOutput;
begin
  Result := @FPb;
end;
{$EndRegion}
{$Region 'TpbOps'}
procedure WriteByte(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawByte(Shortint(value));
end;
procedure WriteInt16(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawVarint32(Word(value));
end;
procedure WriteInt32(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawVarint32(Int32(value));
end;
procedure WriteInt64(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawVarint64(Int64(value));
end;
procedure WriteString(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawString(string(value));
end;
procedure WriteSingle(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawData(@value, sizeof(Single));
end;
procedure WriteDouble(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawData(@value, sizeof(Double));
end;
procedure WriteExtended(const S: TpbSaver; const [Ref] value);
var
  v: Double;
begin
  v := Extended(value);
  S.Pb.writeRawData(@v, sizeof(Double));
end;
procedure WriteCurrency(const S: TpbSaver; const [Ref] value);
begin
  S.Pb.writeRawData(@value, sizeof(Currency));
end;
procedure ReadByte(const L: TpbLoader; var value);
begin
  Shortint(value) := L.Pb.readRawByte;
end;
procedure ReadInt16(const L: TpbLoader; var value);
begin
  Word(value) := L.Pb.readRawVarint32;
end;
procedure ReadInt32(const L: TpbLoader; var value);
begin
  Int32(value) := L.Pb.readRawVarint32;
end;
procedure ReadInt64(const L: TpbLoader; var value);
begin
  Int64(value) := L.Pb.readRawVarint64;
end;
procedure ReadString(const L: TpbLoader; var value);
begin
  string(value) := L.Pb.readString;
end;
procedure ReadSingle(const L: TpbLoader; var value);
begin
  L.Pb.readRawBytes(value, SizeOf(Single));
end;
procedure ReadDouble(const L: TpbLoader; var value);
begin
  L.Pb.readRawBytes(value, sizeof(Double));
end;
procedure ReadExtended(const L: TpbLoader; var value);
var
  v: Double;
begin
  L.Pb.readRawBytes(v, sizeof(Double));
  Extended(value) := v;
end;
procedure ReadCurrency(const L: TpbLoader; var value);
begin
  L.Pb.readRawBytes(value, sizeof(Currency));
end;
const
  // Integer
  IoProcByte: TpbOps = (Save: WriteByte; Load: ReadByte);
  IoProcInt16: TpbOps = (Save: WriteInt16; Load: ReadInt16);
  IoProcInt32: TpbOps = (Save: WriteInt32; Load: ReadInt32);
  IoProcInt64: TpbOps = (Save: WriteInt64; Load: ReadInt64);
  // Real
  IoProcR4: TpbOps = (Save: WriteSingle; Load: ReadSingle);
  IoProcR8: TpbOps = (Save: WriteDouble; Load: ReadDouble);
  IoProcR10: TpbOps = (Save: WriteExtended; Load: ReadExtended);
  IoProcRC8: TpbOps = (Save: WriteCurrency; Load: ReadCurrency);
  // String
  IoProcString: TpbOps = (Save: WriteString; Load: ReadString);
function SelectBinary(info: PTypeInfo; size: Integer): PpbOps;
begin
  case size of
    1: Result := @IoProcByte;
    2: Result := @IoProcInt16;
    4: Result := @IoProcInt32;
    8: Result := @IoProcInt64;
    else
    begin
      System.Error(reRangeError);
      exit(nil);
    end;
  end;
end;
function SelectInteger(info: PTypeInfo; size: Integer): PpbOps;
begin
  case GetTypeData(info)^.OrdType of
    otSByte, otUByte: Result := @IoProcByte;
    otSWord, otUWord: Result := @IoProcInt16;
    otSLong, otULong: Result := @IoProcInt32;
  else
    System.Error(reRangeError);
    exit(nil);
  end;
end;
function SelectFloat(info: PTypeInfo; size: Integer): PpbOps;
begin
  case GetTypeData(info)^.FloatType of
    ftSingle: Result := @IoProcR4;
    ftDouble: Result := @IoProcR8;
    ftExtended: Result := @IoProcR10;
    ftCurr: Result := @IoProcRC8;
  else
    System.Error(reRangeError);
    exit(nil);
  end;
end;
type
  TSelectProc = function(info: PTypeInfo; size: Integer): PpbOps;
  PIoInfo = ^TIoInfo;
  TIoInfo = record
    Selector: Boolean;
    Data: Pointer;
  end;
const
  VtabIo: array[TTypeKind] of TIoInfo = (
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
    (Selector: False; Data: @IoProcString),
    // tkSet
    (Selector: True; Data: @SelectBinary),
    // tkClass
    (Selector: False; Data: nil),
    // tkMethod
    (Selector: False; Data: nil),
    // tkWChar
    (Selector: False; Data: nil),
    // tkLString
    (Selector: False; Data: nil),
    // tkWString
    (Selector: False; Data: nil),
    // tkVariant
    (Selector: False; Data: nil),
    // tkArray
    (Selector: False; Data: nil),
    // tkRecord
    (Selector: False; Data: nil),
    // tkInterface
    (Selector: False; Data: nil),
    // tkInt64
    (Selector: False; Data: @IoProcInt64),
    // tkDynArray
    (Selector: False; Data: nil),
    // tkUString
    (Selector: False; Data: @IoProcString),
    // tkClassRef
    (Selector: False; Data: nil),
    // tkPointer
    (Selector: False; Data: nil),
    // tkProcedure
    (Selector: False; Data: nil),
    // tkMRecord
    (Selector: False; Data: nil)
  );
class function TpbOps.From<T>: TpbOps;
begin
  Result := TpbOps.From(System.TypeInfo(T), SizeOf(T));
end;
function TpbOps.GetWire: Integer;
begin
  case info.Kind of
    tkInteger, tkInt64, tkChar, tkEnumeration, tkSet:
      Result := TWire.VARINT;
    tkFloat:
      case GetTypeData(info)^.FloatType of
        ftSingle: Result := TWire.FIXED32;
        ftDouble: Result := TWire.FIXED64;
        else
          raise EProtobufError.Create('Invalid parameter');
      end;
    else
      Result := TWire.LENGTH_DELIMITED;
  end;
end;
class function TpbOps.From(info: PTypeInfo; size: Integer): TpbOps;
var
  pio: PIoInfo;
begin
  if info = nil then
    raise EProtobufError.Create('Invalid parameter');
  pio := @VtabIo[info^.Kind];
  if pio^.Selector then
    Result := TSelectProc(pio^.Data)(info, size)^
  else if pio^.Data <> nil then
    Result := PpbOps(pio^.Data)^
  else
    raise EProtobufError.Create('Type serialization is not supported');
  Result.info := info;
  Result.om := nil;
end;
class function TpbOps.From(om: PObjMeta): TpbOps;
begin
  Result.info := om.info;
  Result.om := om;
  Result.SaveObj := om.SaveTo;
  Result.LoadObj := om.LoadFrom;
end;
procedure TpbOps.LoadFrom(const L: TpbLoader; var Value);
begin
  Load(L, Value);
end;
procedure TpbOps.SaveTo(const S: TpbSaver; const [Ref] Value);
begin
  Save(S, Value);
end;
{$EndRegion}
{$Region 'TPropMeta}
procedure TPropMeta.Init(kind: TpbFieldKind; const name: AnsiString;
  fno, offset: Integer; const ops: TpbOps);
begin
  Self.name := name;
  Self.offset := offset;
  defValue := nil;
  Self.ops := ops;
  Self.tag.MakeTag(fno, ops.GetWire);
end;
function TPropMeta.ToString: string;
const
  Kinds: array [TpbFieldKind] of string =
    ('Single', 'Obj', 'List', 'ObjList', 'Map', 'ObjMap');
begin
  Result := Format('%s offset=%d kind=%s', [name, offset, Kinds[kind]]);
end;
function TPropMeta.GetField(const [Ref] Obj): Pointer;
begin
  Result := PByte(@Obj) + offset;
end;
procedure TPropMeta.SetDefValue<T>(const Value: T);
begin
  Assert(TypeInfo(T) = ops.info);
  if defValue = nil then
    New(defValue);
  defValue^ := TValue.From<T>(Value);
end;
function TPropMeta.EqualToDefault(var field): Boolean;
var
  v: TValue;
begin
  Result := False;
  if defValue = nil then
  begin
    case ops.info.Kind of
      tkInteger, tkChar, tkWChar, tkEnumeration:
        case GetTypeData(ops.info)^.OrdType of
          otSByte, otUByte: Result := Byte(field) = 0;
          otSWord, otUWord: Result := Word(field)= 0;
          otSLong, otULong: Result := Cardinal(field) = 0;
        end;
      tkInt64:
        Result := Int64(field) = 0;
      tkFloat:
        case GetTypeData(ops.info)^.FloatType of
          ftSingle: Result := IsZero(Single(field));
          ftDouble: Result := IsZero(Double(field));
          ftExtended: Result := IsZero(Extended(field));
        end;
      tkString:
        Result := Length(string(field)) = 0;
      tkLString, tkWString, tkUString:
        begin
          TValue.Make(@field, ops.info, v);
          Result := Length(v.AsString) = 0;
        end
    end;
  end
  else
  begin
    TValue.Make(@field, ops.info, v);
    case ops.info.Kind of
      tkInteger, tkChar, tkWChar, tkEnumeration, tkInt64:
        Result := v.AsOrdinal = defValue.AsOrdinal;
      tkFloat:
        Result := SameValue(v.AsType<Extended>, defValue.AsType<Extended>);
      tkString, tkLString, tkWString, tkUString:
        Result := v.AsString = defValue.AsString;
    end;
  end;
end;
{$EndRegion}
{$Region 'TFieldParam}
constructor TFieldParam.From(const name: AnsiString; fno, offset: Integer);
begin
  Self.name := name;
  Self.fieldNumber := fno;
  Self.offset := offset;
end;
{$EndRegion}
{$Region 'TObjMeta}
class function TObjMeta.From<T>(Init: TObjectMethod;
  get: TGetPropBy = getByBinary): TObjMeta;
begin
  Result.info := TypeInfo(T);
  Result.props := [];
  Result.FInit := Init;
  case get of
    getByBinary: Result.FGetProp := Result.PropByBinary;
    getByFind: Result.FGetProp := Result.PropByFind;
    getByIndex: Result.FGetProp := Result.PropByIndex;
  end;
end;
function TObjMeta.ToString: string;
begin
  Result := string(info.Name);
end;
procedure TObjMeta.Add<T>(const name: AnsiString; fno, offset: Integer);
var
  meta: TPropMeta;
begin
  meta.Init(fkSingleProp, name, fno, offset, TpbOps.From<T>);
  props := props + [meta];
end;
procedure TObjMeta.AddMap<Key>(kind: TpbFieldKind; const name: AnsiString;
  fno, offset: Integer; const ops: TpbOps);
var
  meta: TPropMeta;
begin
  meta.Init(kind, name, fno, offset, ops);
  props := props + [meta];
end;
procedure TObjMeta.AddObj(kind: TpbFieldKind; const name: AnsiString;
  fno, offset: Integer; const ops: TpbOps);
var
  meta: TPropMeta;
begin
  meta.Init(kind, name, fno, offset, ops);
  props := props + [meta];
end;
class function TObjMeta.PropByBinary(om: PObjMeta; fno: Integer): PPropMeta;
var
  L, R, M, n: Integer;
begin
  L := 0;
  R := High(om.props);
  while L <> R do
  begin
    M := (L + R) div 2;
    Result := @om.props[M];
    n := Result.tag.FieldNumber;
    if n < fno then
      L := M + 1
    else if n > fno then
      R := M - 1
    else
      exit;
  end;
  Result := @om.props[L];
end;
class function TObjMeta.PropByIndex(om: PObjMeta; fno: Integer): PPropMeta;
begin
  Result := @om.props[fno - 1];
end;
class function TObjMeta.PropByFind(om: PObjMeta; fno: Integer): PPropMeta;
var
  i: Integer;
begin
  for i := 0 to High(om.props) do
  begin
    Result := @om.props[i];
    if Result.tag.FieldNumber = fno then
      exit;
  end;
  Result := nil;
end;
procedure TObjMeta.SaveList(pm: PPropMeta; const S: TpbSaver; const [Ref] obj);
var
  i: Integer;
  h: TpbSaver;
  List: PsgPointerList;
  value: Pointer;
begin
  h.Init;
  try
    List := PsgPointerList(@obj);
    for i := 0 to List.Count - 1 do
    begin
      h.Clear;
      value := PPointer(List.Items[i])^;
      if pm.kind = fkList then
        pm.ops.Save(h, value^)
      else
        pm.ops.SaveObj(pm.ops.om, h, value^);
      S.Pb.writeMessage(pm.tag.FieldNumber, h.Pb^);
    end;
  finally
    h.Free;
  end;
end;
procedure TObjMeta.LoadList(const tag: TpbTag; pm: PPropMeta;
  const L: TpbLoader; var obj);
var
  List: PsgPointerList;
  value: Pointer;
begin
  L.Pb.Push;
  try
    List := PsgPointerList(@obj);
    repeat
      value := List.Add;
      if pm.kind = fkList then
        pm.ops.Load(L, value^)
      else
        pm.ops.LoadObj(pm.ops.om, L, value^);
    until tag.v <> L.Pb.readTag.v;
  finally
    L.Pb.Pop;
  end;
end;
procedure TObjMeta.SaveMap(pm: PPropMeta; const S: TpbSaver; const [Ref] obj);
var
  Map: PsgCustomHashMap;
  it: TsgCustomHashMapIterator;
  h: TpbSaver;
begin
  Map := PsgCustomHashMap(@obj);
  h.Init;
  try
    it := Map.Begins;
    while it <> Map.Ends do
    begin
      h.Clear;
      if pm.kind = fkMap then
        pm.ops.Save(h, it.GetKey^)
      else
        pm.ops.SaveObj(pm.ops.om, h, it.GetKey^);
      S.Pb.writeMessage(pm.tag.FieldNumber, h.Pb^);
      it.Next;
    end;
  finally
    h.Free;
  end;
end;
procedure TObjMeta.LoadMap(const tag: TpbTag; pm: PPropMeta;
  const L: TpbLoader; var obj);
var
  Map: PsgCustomHashMap;
  pair: Pointer;
begin
  L.Pb.Push;
  try
    Map := PsgCustomHashMap(@obj);
    repeat
      pair := Map.GetTemporaryPair;
      if pm.kind = fkMap then
        pm.ops.Load(L, pair^)
      else
        pm.ops.LoadObj(pm.ops.om, L, pair^);
      Map.InsertOrAssign(pair);
    until tag.v <> L.Pb.readTag.v;
  finally
    L.Pb.Pop;
  end;
end;
class procedure TObjMeta.SaveTo(om: PObjMeta; const S: TpbSaver; const [Ref] obj);
begin
  log.print(om.ToString);
  om.SaveProps(S, obj);
end;
class procedure TObjMeta.LoadFrom(om: PObjMeta; const L: TpbLoader; var obj);
begin
  log.print(om.ToString);
  om.LoadProps(L, obj);
end;
procedure TObjMeta.SaveProps(const S: TpbSaver; const [Ref] obj);
var
  i, lo, hi: Integer;
  pm: PPropMeta;
  field: Pointer;
  h: TpbSaver;
begin
  log.print('SaveProps');
  for i := 0 to High(props) do
  begin
    pm := @props[i];
    log.print(Format('  [%d] %s', [i, pm.ToString]));
    field := pm.GetField(obj);
    lo := S.Pb.FBuffer.GetCount;
    case pm.kind of
      fkSingleProp:
        if not pm.EqualToDefault(field^) then
        begin
          S.Pb.writeRawVarint32(pm.tag.v);
          pm.ops.Save(S, field^);
        end;
      fkObj:
        begin
          h.Init;
          try
            pm.ops.SaveObj(pm.ops.om, h, field^);
            S.Pb.writeMessage(pm.tag.FieldNumber, h.Pb^);
          finally
            h.Free;
          end;
        end;
      fkList, fkObjList:
        SaveList(pm, S, field^);
      fkMap, fkObjMap:
        SaveMap(pm, S, field^);
    end;
    hi := S.Pb.FBuffer.GetCount;
    log.print(Format('  prop %s, lo=%d hi=%d Count=$%x',
      [pm.name, lo, hi, hi - lo + 1]));
    log.print('  ', [S.Pb.ToString(lo)]);
  end;
end;
procedure TObjMeta.LoadProps(const L: TpbLoader; var obj);
var
  fieldNo, lo, hi: Integer;
  tag: TpbTag;
  pm: PPropMeta;
  field: Pointer;
begin
  Init(obj);
  log.print('LoadProps');
  tag := L.Pb.readTag;
  while tag.v <> 0 do
  begin
    fieldNo := tag.FieldNumber;
    pm := GetProp(@Self, fieldNo);
    log.print(Format('  %s', [pm.ToString]));
    field := pm.GetField(obj);
    lo := L.Pb.FCurrent - L.Pb.FBuf - 1;
    case pm.kind of
      fkSingleProp:
        pm.ops.Load(L, field^);
      fkObj:
        begin
          L.Pb.Push;
          try
            pm.ops.om.LoadProps(L, field^);
          finally
            L.Pb.Pop;
          end;
        end;
      fkList, fkObjList:
        LoadList(tag, pm, L, field^);
      fkMap, fkObjMap:
        LoadMap(tag, pm, L, field^);
    end;
    hi := L.Pb.FLast - L.Pb.FBuf;
    log.print(Format('  prop %s, lo=%d hi=%d Count=$%x',
      [pm.name, hi - lo, hi, hi - lo + 1]));
    log.print('  ', [L.Pb.ToString(lo)]);
    tag := L.Pb.readTag;
  end;
end;
{$EndRegion}
end.
