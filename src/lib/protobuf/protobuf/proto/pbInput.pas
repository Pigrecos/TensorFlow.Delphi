// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// http://code.google.com/p/protobuf/
//
// Author this port to delphi - Marat Shaimardanov, Tomsk (2007..2020)
//
// Send any postcards with postage stamp to my address:
// Frunze 131/1, 56, Russia, Tomsk, 634021
// then you can use this code in self project.

unit pbInput;

interface

uses Classes, SysUtils, pbPublic;

type

  // Reads and decodes protocol message fields.
  PProtoBufInput = ^TProtoBufInput;
  TProtoBufInput = record
  const
    RECURSION_LIMIT = 64;
    SIZE_LIMIT = 64 shl 20;  // 64 mb  
  private var
    FLen: Integer;
    FBuffer: PByte;
    FPos: Integer;
    FLastTag: Integer;
    FOwnsData: Boolean;
    FRecursionDepth: ShortInt;
  public
    procedure Init; overload;
    procedure Init(const pb: TProtoBufInput); overload;
    procedure Init(Buf: PByte; BufSize: Integer; OwnsData: Boolean); overload;
    class function From(const Buf: TBytes): TProtoBufInput; overload; static;
    class function From(Buf: PByte; BufSize: Integer;
      OwnsData: Boolean = False): TProtoBufInput; overload; static;
    procedure Free;

    // I/O routines to file and stream
    procedure SaveToStream(Stream: TStream);
    procedure SaveToFile(const FileName: string);
    procedure LoadFromFile(const FileName: string);
    procedure LoadFromStream(Stream: TStream);
    // Merge messages
    procedure mergeFrom(const builder: TProtoBufInput);
    // Set buffer posititon
    procedure setPos(Pos: Integer);
    // Get buffer posititon
    function getPos: Integer;
    // Attempt to read a field tag, returning zero if we have reached EOF
    function readTag: Integer;
    // Check whether the latter match the value read tag
    // Used to test for nested groups
    procedure checkLastTagWas(value: Integer);
    // Reads and discards a Single field, given its tag value
    function skipField(tag: Integer): Boolean;
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
    procedure readMessage(builder: PProtoBufInput);
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
    function readBytes(size: Integer): TBytes;
    // Skip "size" bytes
    procedure skipRawBytes(size: Integer);
  end;

function decodeZigZag32(n: Integer): Integer;
function decodeZigZag64(n: Int64): Int64;

implementation

const
  ProtoBufException = 'Protocol buffer exception: ';

function decodeZigZag32(n: Integer): Integer;
begin
  Result := (n shr 1) xor -(n and 1);
end;

function decodeZigZag64(n: Int64): Int64;
begin
  Result := (n shr 1) xor -(n and 1);
end;

{ TProtoBufInput }

procedure TProtoBufInput.Init;
begin
  Self := Default(TProtoBufInput);
end;

procedure TProtoBufInput.Init(Buf: PByte; BufSize: Integer; OwnsData: Boolean);
begin
  FLen := BufSize;
  FPos := 0;
  FOwnsData := OwnsData;
  FRecursionDepth := 0;
  if not OwnsData then
    FBuffer := Buf
  else
  begin
    // allocate a buffer and copy the data
    GetMem(FBuffer, FLen);
    Move(Buf^, FBuffer^, FLen);
  end;
end;

procedure TProtoBufInput.Init(const pb: TProtoBufInput);
begin
  Self.FBuffer := pb.FBuffer;
  Self.FPos := 0;
  Self.FOwnsData := False;
end;

class function TProtoBufInput.From(const Buf: TBytes): TProtoBufInput;
begin
  Result.Init(@Buf[0], Length(Buf), False);
end;

class function TProtoBufInput.From(Buf: PByte; BufSize: Integer;
  OwnsData: Boolean = False): TProtoBufInput; 
begin
  Result.Init(Buf, BufSize, OwnsData);
end;

procedure TProtoBufInput.Free;
begin
  if FOwnsData then
    FreeMem(FBuffer, FLen);
  Self := Default(TProtoBufInput);
end;

function TProtoBufInput.readTag: Integer;
begin
  if FPos < FLen then
    FLastTag := readRawVarint32
  else
    FLastTag := 0;
  Result := FLastTag;
end;

procedure TProtoBufInput.checkLastTagWas(value: Integer);
begin
  Assert(FLastTag = value, ProtoBufException + 'invalid end tag');
end;

function TProtoBufInput.skipField(tag: Integer): Boolean;
begin
  Result := True;
  case getTagWireType(tag) of
    TWire.VARINT:
      readInt32;
    TWire.FIXED64:
      readRawLittleEndian64;
    TWire.LENGTH_DELIMITED:
      skipRawBytes(readRawVarint32);
    TWire.FIXED32:
      readRawLittleEndian32;
    else
      raise Exception.Create('InvalidProtocolBufferException.invalidWireType');
  end;
end;

procedure TProtoBufInput.skipMessage;
var tag: Integer;
begin
  repeat
    tag := readTag;
  until (tag = 0) or (not skipField(tag));
end;

function TProtoBufInput.readDouble: Double;
begin
  readRawBytes(Result, SizeOf(Double));
end;

function TProtoBufInput.readFloat: Single;
begin
  readRawBytes(Result, SizeOf(Single));
end;

function TProtoBufInput.readInt64: Int64;
begin
  Result := readRawVarint64;
end;

function TProtoBufInput.readInt32: Integer;
begin
  Result := readRawVarint32;
end;

function TProtoBufInput.readFixed64: Int64;
begin
  Result := readRawLittleEndian64;
end;

function TProtoBufInput.readFixed32: Integer;
begin
  Result := readRawLittleEndian32;
end;

function TProtoBufInput.readBoolean: Boolean;
begin
  Result := readRawVarint32 <> 0;
end;

function TProtoBufInput.readString: string;
var
  size: Integer;
  buf, text: TBytes;
begin
  size := readRawVarint32;
  Assert(size > 0, ProtoBufException + 'readString (size <= 0)');
  // Decode utf8 to string
  buf := readBytes(size);
  text := TEncoding.UTF8.Convert(TEncoding.UTF8, TEncoding.Unicode, buf);
  Result := TEncoding.Unicode.GetString(text);
end;

procedure TProtoBufInput.readMessage(builder: PProtoBufInput);
begin
  readRawVarint32;
  Assert(FRecursionDepth < RECURSION_LIMIT,
    ProtoBufException + 'recursion Limit Exceeded');
  Inc(FRecursionDepth);
  builder.mergeFrom(Self);
  checkLastTagWas(0);
  dec(FRecursionDepth);
end;

function TProtoBufInput.readUInt32: Integer;
begin
  Result := readRawVarint32;
end;

function TProtoBufInput.readEnum: Integer;
begin
  Result := readRawVarint32;
end;

function TProtoBufInput.readSFixed32: Integer;
begin
  Result := readRawLittleEndian32;
end;

function TProtoBufInput.readSFixed64: Int64;
begin
  Result := readRawLittleEndian64;
end;

function TProtoBufInput.readSInt32: Integer;
begin
  Result := decodeZigZag32(readRawVarint32);
end;

function TProtoBufInput.readSInt64: Int64;
begin
  Result := decodeZigZag64(readRawVarint64);
end;

function TProtoBufInput.readRawVarint32: Integer;
var
  tmp: ShortInt;
  shift: Integer;
begin
  shift := 0;
  Result := 0;
  repeat
    // for negative numbers number value may be to 10 byte
    Assert(shift < 64, ProtoBufException + 'malformed Varint');
    tmp := readRawByte;
    Result := Result or ((tmp and $7f) shl shift);
    Inc(shift, 7);
  until tmp >= 0;
end;

function TProtoBufInput.readRawVarint64: Int64;
var
  tmp: ShortInt;
  shift: Integer;
  i64: Int64;
begin
  shift := -7;
  Result := 0;
  repeat
    Inc(shift, 7);
    Assert(shift < 64, ProtoBufException + 'malformed Varint');
    tmp := readRawByte;
    i64 := tmp and $7f;
    i64 := i64 shl shift;
    Result := Result or i64;
  until tmp >= 0;
end;

function TProtoBufInput.readRawLittleEndian32: Integer;
begin
  readRawBytes(Result, SizeOf(Result));
end;

function TProtoBufInput.readRawLittleEndian64: Int64;
begin
  readRawBytes(Result, SizeOf(Result));
end;

function TProtoBufInput.readRawByte: ShortInt;
begin
  Assert(FPos < FLen, ProtoBufException + 'eof encounterd');
  Result := ShortInt(FBuffer[FPos]);
  Inc(FPos);
end;

procedure TProtoBufInput.readRawBytes(var data; size: Integer);
begin
  Assert(FPos + size <= FLen, ProtoBufException + 'eof encounterd');
  Move(FBuffer[FPos], data, size);
  Inc(FPos, size);
end;

function TProtoBufInput.readBytes(size: Integer): TBytes;
begin
  Assert(FPos + size <= FLen, ProtoBufException + 'eof encounterd');
  SetLength(Result, size);
  Move(FBuffer[FPos], Pointer(Result)^, size);
  Inc(FPos, size);
end;

procedure TProtoBufInput.skipRawBytes(size: Integer);
begin
  Assert(size >= 0, ProtoBufException + 'negative Size');
  Assert(FPos + size <= FLen, ProtoBufException + 'truncated Message');
  Inc(FPos, size);
end;

procedure TProtoBufInput.SaveToFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TProtoBufInput.SaveToStream(Stream: TStream);
begin
  Stream.WriteBuffer(Pointer(FBuffer)^, FLen);
end;

procedure TProtoBufInput.LoadFromFile(const FileName: string);
var Stream: TStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TProtoBufInput.LoadFromStream(Stream: TStream);
begin
  if FOwnsData then begin
    FreeMem(FBuffer, FLen);
    FBuffer := nil;
  end;
  FOwnsData := True;
  FLen := Stream.Size;
  GetMem(FBuffer, FLen);
  Stream.Position := 0;
  Stream.Read(Pointer(FBuffer)^, FLen);
end;

procedure TProtoBufInput.mergeFrom(const builder: TProtoBufInput);
begin
  Assert(False, 'under conctruction');
end;

procedure TProtoBufInput.setPos(Pos: Integer);
begin
  FPos := Pos;
end;

function TProtoBufInput.getPos: Integer;
begin
  Result := FPos;
end;

end.
