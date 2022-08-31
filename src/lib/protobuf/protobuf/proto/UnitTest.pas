// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// http://code.google.com/p/protobuf/
//
// Author this port to delphi - Marat Shaimardanov, Tomsk (2007..2020)
//
// Send any postcards with postage stamp to my address:
// Frunze 131/1, 56, Russia, Tomsk, 634021
// then you can use this code in self project.

unit UnitTest;

interface

uses SysUtils, pbPublic, pbInput, pbOutput;

procedure TestAll;

implementation

procedure TestVarint;
type
  TVarintCase = record
    bytes: array [0..9] of Byte;  // Encoded bytes.
    size: Integer;                // Encoded size, in bytes.
    value: Int64;                 // Parsed value.
  end;
const
  VarintCases: array [0..7] of TVarintCase = (
    // 32-bit values
    (bytes: ($00, $00, $00, $00, $00, $00, $00, $00, $00, $00);
     size: 1; value: 0),
    (bytes: ($01, $00, $00, $00, $00, $00, $00, $00, $00, $00);
     size: 1; value: 1),
    (bytes: ($7f, $00, $00, $00, $00, $00, $00, $00, $00, $00);
     size: 1; value: 127),
    (bytes: ($a2, $74, $00, $00, $00, $00, $00, $00, $00, $00);
     size: 2; value: 14882),
    (bytes: ($ff, $ff, $ff, $ff, $0f, $00, $00, $00, $00, $00);
     size: 5; value: -1),
    // 64-bit
    (bytes: ($be, $f7, $92, $84, $0b, $00, $00, $00, $00, $00);
     size: 5; value: 2961488830),
    (bytes: ($be, $f7, $92, $84, $1b, $00, $00, $00, $00, $00);
     size: 5; value: 7256456126),
    (bytes: ($80, $e6, $eb, $9c, $c3, $c9, $a4, $49, $00, $00);
     size: 8; value: 41256202580718336)
  );
var
  i, j: Integer;
  test: TVarintCase;
  pb: TProtoBufInput;
  buf: TBytes;
  i64: Int64;
  int: Integer;
begin
  for i := 0 to 7 do
  begin
    test := VarintCases[i];
    // создать тестовый буфер
    buf := [];
    for j := Low(test.bytes) to test.size - 1 do
      buf := buf + [test.bytes[j]];
    pb := TProtoBufInput.From(@buf[0], test.size);
    try
      if i < 5 then begin
        int := pb.readRawVarint32;
        Assert(test.value = int, 'Test Varint fails');
      end
      else
      begin
        i64 := pb.readRawVarint64;
        Assert(test.value = i64, 'Test Varint fails');
      end;
    finally
      pb.Free;
    end;
  end;
end;

procedure TestReadLittleEndian32;
type
  TLittleEndianCase = record
    bytes: array [0..3] of byte;      // Encoded bytes.
    value: Integer;                   // Parsed value.
  end;
const
  LittleEndianCases: array [0..5] of TLittleEndianCase = (
    (bytes: ($78, $56, $34, $12); value: $12345678),
    (bytes: ($f0, $de, $bc, $9a); value: Integer($9abcdef0)),
    (bytes: ($ff, $00, $00, $00); value: 255),
    (bytes: ($ff, $ff, $00, $00); value: 65535),
    (bytes: ($4e, $61, $bc, $00); value: 12345678),
    (bytes: ($b2, $9e, $43, $ff); value: -12345678)
  );
var
  i, j: Integer;
  test: TLittleEndianCase;
  pb: TProtoBufInput;
  buf: TBytes;
  int: Integer;
begin
  for i := 0 to 5 do
  begin
    test := LittleEndianCases[i];
    buf := [];
    for j := Low(test.bytes) to High(test.bytes) do
      buf := buf + [test.bytes[j]];
    pb := TProtoBufInput.From(buf);
    try
      int := pb.readRawLittleEndian32;
      Assert(test.value = int, 'Test readRawLittleEndian32 fails');
    finally
      pb.Free;
    end;
  end;
end;

procedure TestReadLittleEndian64;
type
  TLittleEndianCase = record
    bytes: array [0..7] of byte;      // Encoded bytes.
    value: Int64;                     // Parsed value.
  end;
const
  LittleEndianCases: array [0..3] of TLittleEndianCase = (
    (bytes: ($67, $45, $23, $01, $78, $56, $34, $12);
     value: $1234567801234567),
    (bytes: ($f0, $de, $bc, $9a, $78, $56, $34, $12);
     value: $123456789abcdef0),
    (bytes: ($79, $df, $0d, $86, $48, $70, $00, $00);
     value: 123456789012345),
    (bytes: ($87, $20, $F2, $79, $B7, $8F, $FF, $FF);
     value: -123456789012345)
  );
var
  i, j: Integer;
  test: TLittleEndianCase;
  pb: TProtoBufInput;
  buf: TBytes;
  int: Int64;
begin
  for i := 0 to 3 do
  begin
    test := LittleEndianCases[i];
    SetLength(buf, 8);
    for j := 0 to High(test.bytes) do
      buf[j] := test.bytes[j];
    pb := TProtoBufInput.From(@buf[0], 8);
    try
      int := pb.readRawLittleEndian64;
      Assert(test.value = int, 'Test readRawLittleEndian64 fails');
    finally
      pb.Free;
    end;
  end;
end;

procedure TestDecodeZigZag;
type
  TInt32Case = record
    r: Int32; // Expected Result
    i: Int32; // Input parameter
  end;
  TInt64Case = record
    r: Int64; // Expected Result
    i: Int64; // Input parameter
  end;
const
  Int32Cases: array [0..7] of TInt32Case = (
    (r: 0; i: 0),
    (r: -1; i: 1),
    (r: 1; i: 2),
    (r: -2; i: 3),
    (r: $3FFFFFFF; i: $7FFFFFFE),
    (r: $C0000000; i: $7FFFFFFF),
    (r: $7FFFFFFF; i: $FFFFFFFE),
    (r: $80000000; i: $FFFFFFFF));
  Int64Cases: array [0..9] of TInt64Case = (
    (r: 0; i: 0),
    (r: -1; i: 1),
    (r: 1; i: 2),
    (r: -2; i: 3),
    (r: $000000003FFFFFFF; i: $000000007FFFFFFE),
    (r: $FFFFFFFFC0000000; i: $000000007FFFFFFF),
    (r: $000000007FFFFFFF; i: $00000000FFFFFFFE),
    (r: $FFFFFFFF80000000; i: $00000000FFFFFFFF),
    (r: $7FFFFFFFFFFFFFFF; i: $FFFFFFFFFFFFFFFE),
    (r: $8000000000000000; i: $FFFFFFFFFFFFFFFF));
var
  i: Integer;
  a: TInt32Case;
  b: TInt64Case;
begin
  (* 32 *)
  for i := Low(Int32Cases) to High(Int32Cases) do
  begin
    a := Int32Cases[i];
    Assert(a.r = decodeZigZag32(a.i));
  end;
  (* 64 *)
  for i := Low(Int64Cases) to High(Int64Cases) do
  begin
    b := Int64Cases[i];
    Assert(b.r = decodeZigZag64(b.i));
  end;
end;

procedure TestReadString;
const
  TEST_string  = 'ABC '#$0410#$0411#$0412' '#$0E01#$0E02#$0E03;
  TEST_integer = 12345678;
  TEST_single  = 12345.123;
  TEST_double  = 1234567890.123;
var
  out_pb: TProtoBufOutput;
  in_pb: TProtoBufInput;
  tag, t: Integer;
  text: string;
  int: Integer;
  dbl: Double;
  flt: Single;
  delta: Extended;
begin
  out_pb := TProtoBufOutput.From;
  try
    out_pb.writeString(1, TEST_string);
    out_pb.writeFixed32(2, TEST_integer);
    out_pb.writeFloat(3, TEST_single);
    out_pb.writeDouble(4, TEST_double);
    out_pb.SaveToFile('test.dmp');
  finally
    out_pb.Free;
  end;
  in_pb.Init;
  try
   in_pb. LoadFromFile('test.dmp');
    // TEST_string
    tag := makeTag(1, TWire.LENGTH_DELIMITED);
    t := in_pb.readTag;
    Assert(tag = t);
    text := in_pb.readString;
    Assert(TEST_string = text);
    // TEST_integer
    tag := makeTag(2, TWire.FIXED32);
    t := in_pb.readTag;
    Assert(tag = t);
    int := in_pb.readFixed32;
    Assert(TEST_integer = int);
    // TEST_single
    tag := makeTag(3, TWire.FIXED32);
    t := in_pb.readTag;
    Assert(tag = t);
    flt := in_pb.readFloat;
    delta := TEST_single - flt;
    Assert(abs(delta) < 0.001);
    // TEST_double
    tag := makeTag(4, TWire.FIXED64);
    t := in_pb.readTag;
    Assert(tag = t);
    dbl := in_pb.readDouble;
    {$OVERFLOWCHECKS ON}
    delta := dbl - TEST_double;
    Assert(abs(delta) < 0.000001);
  finally
    in_pb.Free;
  end;
end;

procedure TestMemoryLeak;
const
  Mb = 1024 * 1024;
var
  in_pb: TProtoBufInput;
  buf_size: Integer;
  buf: TArray<Byte>;
  i: Integer;
begin
  buf_size := 64 * Mb;
  SetLength(buf, buf_size);
  for i := 0 to 200 do
  begin
    in_pb := TProtoBufInput.From(@buf[0], Length(buf), false);
    in_pb.Free;
  end;
end;

procedure TestAll;
begin
  TestVarint;
  TestReadLittleEndian32;
  TestReadLittleEndian64;
  TestDecodeZigZag;
  TestReadString;
  TestMemoryLeak;
end;

end.

