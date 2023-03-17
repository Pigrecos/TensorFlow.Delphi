// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// http://code.google.com/p/protobuf/
//
// Author this port to delphi - Marat Shaimardanov, Tomsk (2007..2020)
//
// Send any postcards with postage stamp to my address:
// Frunze 131/1, 56, Russia, Tomsk, 634021
// then you can use this code in self project.

unit pbOutput;

interface

uses
  Classes, SysUtils, Oz.Pb.StrBuffer, pbPublic;

type

  TProtoBufOutput = record
  private
    FBuffer: TSegmentBuffer;
  public
    class function From: TProtoBufOutput; static;
    procedure Free;
    procedure Clear;

    procedure SaveToStream(Stream: TStream); inline;
    procedure SaveToFile(const FileName: string); inline;

    // Encode and write varint
    procedure writeRawVarint32(value: Integer);
    // Encode and write varint
    procedure writeRawVarint64(value: Int64);
    // Encode and write tag
    procedure writeTag(fieldNumber: Integer; wireType: Integer); inline;
    // Encode and write single byte
    procedure writeRawByte(value: ShortInt); inline;
    // Write the data with specified size
    procedure writeRawData(const p: Pointer; size: Integer); inline;

    // Get the result as a bytes
    function GetBytes: TBytes; inline;
    // Write a Double field, including tag
    procedure writeDouble(fieldNumber: Integer; value: Double);
    // Write a Single field, including tag
    procedure writeFloat(fieldNumber: Integer; value: Single);
    // Write a Int64 field, including tag
    procedure writeInt64(fieldNumber: Integer; value: Int64);
    // Write a Int64 field, including tag
    procedure writeInt32(fieldNumber: Integer; value: Integer);
    // Write a fixed64 field, including tag
    procedure writeFixed64(fieldNumber: Integer; value: Int64);
    // Write a fixed32 field, including tag
    procedure writeFixed32(fieldNumber: Integer; value: Integer);
    // Write a Boolean field, including tag
    procedure writeBoolean(fieldNumber: Integer; value: Boolean);
    // Write a string field, including tag
    procedure writeString(fieldNumber: Integer; const value: string);
    // Write a message field, including tag
    procedure writeMessage(fieldNumber: Integer; const value: TProtoBufOutput);
    //  Write a unsigned Int32 field, including tag
    procedure writeUInt32(fieldNumber: Integer; value: Cardinal);
    // Get serialized size
    function getSerializedSize: Integer;
    // Write to buffer
    procedure writeTo(buffer: TProtoBufOutput);
  end;

implementation

{$r-}

{ TProtoBuf }

class function TProtoBufOutput.From: TProtoBufOutput;
begin
  Result.FBuffer := TSegmentBuffer.Create;
end;

procedure TProtoBufOutput.Free;
begin
  FreeAndNil(FBuffer);
end;

procedure TProtoBufOutput.Clear;
begin
  FBuffer.Clear;
end;

procedure TProtoBufOutput.writeRawByte(value: ShortInt);
begin
  // -128..127
  FBuffer.Add(Byte(value));
end;

procedure TProtoBufOutput.writeRawData(const p: Pointer; size: Integer);
begin
  FBuffer.Add(p, size);
end;

procedure TProtoBufOutput.writeTag(fieldNumber, wireType: Integer);
begin
  writeRawVarint32(makeTag(fieldNumber, wireType));
end;

procedure TProtoBufOutput.writeRawVarint32(value: Integer);
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

procedure TProtoBufOutput.writeRawVarint64(value: Int64);
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

procedure TProtoBufOutput.writeBoolean(fieldNumber: Integer; value: Boolean);
begin
  writeTag(fieldNumber, TWire.VARINT);
  writeRawByte(ord(value));
end;

procedure TProtoBufOutput.writeDouble(fieldNumber: Integer; value: Double);
begin
  writeTag(fieldNumber, TWire.FIXED64);
  writeRawData(@value, SizeOf(value));
end;

procedure TProtoBufOutput.writeFloat(fieldNumber: Integer; value: Single);
begin
  writeTag(fieldNumber, TWire.FIXED32);
  writeRawData(@value, SizeOf(value));
end;

procedure TProtoBufOutput.writeFixed32(fieldNumber, value: Integer);
begin
  writeTag(fieldNumber, TWire.FIXED32);
  writeRawData(@value, SizeOf(value));
end;

procedure TProtoBufOutput.writeFixed64(fieldNumber: Integer; value: Int64);
begin
  writeTag(fieldNumber, TWire.FIXED64);
  writeRawData(@value, SizeOf(value));
end;

procedure TProtoBufOutput.writeInt32(fieldNumber, value: Integer);
begin
  writeTag(fieldNumber, TWire.VARINT);
  writeRawVarint32(value);
end;

procedure TProtoBufOutput.writeInt64(fieldNumber: Integer; value: Int64);
begin
  writeTag(fieldNumber, TWire.VARINT);
  writeRawVarint64(value);
end;

procedure TProtoBufOutput.writeString(fieldNumber: Integer;
  const value: string);
var
  bytes, text: TBytes;
begin
  writeTag(fieldNumber, TWire.LENGTH_DELIMITED);
  bytes := TEncoding.Unicode.GetBytes(value);
  text := TEncoding.Unicode.Convert(TEncoding.Unicode, TEncoding.UTF8, bytes);
  writeRawVarint32(Length(text));
  FBuffer.Add(text);
end;

procedure TProtoBufOutput.writeUInt32(fieldNumber: Integer; value: Cardinal);
begin
  writeTag(fieldNumber, TWire.VARINT);
  writeRawVarint32(value);
end;

procedure TProtoBufOutput.writeMessage(fieldNumber: Integer;
  const value: TProtoBufOutput);
begin
  writeTag(fieldNumber, TWire.LENGTH_DELIMITED);
  writeRawVarint32(value.getSerializedSize);
  value.writeTo(self);
end;

function TProtoBufOutput.GetBytes: TBytes;
begin
  result := FBuffer.GetBytes;
end;

procedure TProtoBufOutput.SaveToFile(const FileName: string);
begin
  FBuffer.SaveToFile(FileName);
end;

procedure TProtoBufOutput.SaveToStream(Stream: TStream);
begin
  FBuffer.SaveToStream(Stream);
end;

function TProtoBufOutput.getSerializedSize: Integer;
begin
  result := FBuffer.GetCount;
end;

procedure TProtoBufOutput.writeTo(buffer: TProtoBufOutput);
begin
  buffer.FBuffer.Add(GetBytes);
end;

end.
