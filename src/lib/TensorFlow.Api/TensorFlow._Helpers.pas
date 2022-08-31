{$REGION 'Licence'}
{ Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Org. Source:  https://github.com/tensorflow/tensorflow
Org. Docu:    https://www.tensorflow.org
The structure of this Delphi porting was oriented to C# porting by Miguel Deicaza.
C# Source:    https://github.com/migueldeicaza/TensorFlowSharp
Delphi porting version: 1.2
==============================================================================}
{$ENDREGION}
unit TensorFlow._Helpers;
interface
uses System.Types, Winapi.Windows, System.AnsiStrings,
     TensorFlow.LowLevelAPI;
// String-Helpers: Conversion to ...
// -----------------------------------------------------------------------------
function _PTFChar(const str: TFString): PTFChar; overload;
function _PTFChar(const str: String; var strBuf: TFString): PTFChar; overload;
// Helper functions for Arrays
// -----------------------------------------------------------------------------
function _AllocMem(values: TArray<Integer>; var dim, byte_size: TF_int64_t):    Pointer; overload;
function _AllocMem(values: TArray<TArray<Integer>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMem(values: TArray<Single>;   var dim, byte_size: TF_int64_t):   Pointer; overload;
function _AllocMem(values: TArray<TArray<Single>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMem(values: TArray<TFString>; var dim, byte_size: TF_int64_t):   Pointer; overload;
function _AllocMem(values: TArray<TArray<TFString>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemAByte(values: TArray<Byte>;    var dim, byte_size: TF_int64_t):    Pointer; overload;
function _AllocMemAByte(values: TArray<TArray<Byte>>;  var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemABool(values: TArray<Boolean>;    var dim, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemABool(values: TArray<TArray<Boolean>>;  var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemAInt16(values: TArray<Int16>;   var dim, byte_size: TF_int64_t):    Pointer; overload;
function _AllocMemAInt16(values: TArray<TArray<Int16>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemAInt64(values: TArray<Int64>;   var dim, byte_size: TF_int64_t):    Pointer; overload;
function _AllocMemAInt64(values: TArray<TArray<Int64>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _AllocMemADouble(values: TArray<Double>;   var dim, byte_size: TF_int64_t):   Pointer; overload;
function _AllocMemADouble(values: TArray<TArray<Double>>; var dim1, dim2, byte_size: TF_int64_t): Pointer; overload;
function _GetArray(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Integer>):   Boolean; overload;
function _GetArray(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Integer>>): Boolean; overload;
function _GetArray(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Single>):   Boolean; overload;
function _GetArray(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Single>>): Boolean; overload;
function _GetArray(data: Pointer; const dim, total_byte_size: TF_int64_t;
                              var values: TArray<TFString>):  Boolean; overload;
function _GetArray(data: Pointer; const dim1, dim2, total_byte_size: TF_int64_t;
                              var values: TArray<TArray<TFString>>): Boolean; overload;
function _GetArrayAByte(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Byte>):    Boolean; overload;
function _GetArrayAByte(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Byte>>):  Boolean; overload;
function _GetArrayABool(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Boolean>):    Boolean; overload;
function _GetArrayABool(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Boolean>>):  Boolean; overload;
function _GetArrayAInt16(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Int16>):   Boolean; overload;
function _GetArrayAInt16(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Int16>>): Boolean; overload;
function _GetArrayAInt64(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Int64>):   Boolean; overload;
function _GetArrayAInt64(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Int64>>): Boolean; overload;
function _GetArrayADouble(data: Pointer; const dim: TF_int64_t;
                              var values: TArray<Double>):   Boolean; overload;
function _GetArrayADouble(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Double>>): Boolean; overload;
/// <summary>Get a pointer array for a given array with fix data size for every element</summary>
function _GetPointerArray(data: Pointer; const dim, one_byte_size: Integer;
                              var parray: TArray<Pointer>):  Boolean;
/// <summary>Get a pointer array for a given string array with variable size of elements</summary>
function _GetPTFCharArray(values: TArray<TFString>;
                              var lengths: TArray<TF_size_t>;
                              var parray: TArray<PTFChar>): Boolean;
// Encode-/Decode-Helpers:
function _EncodeStr(const str: TFString): TFString;
function _DecodeStr(const str: TFString): TFString;
function _EncodeStrings(srcArray: TArray<TFString>; var dstArray: TArray<TFString>): Boolean; overload;
function _EncodeStrings(srcArray: TArray<TArray<TFString>>; var dstArray: TArray<TArray<TFString>>): Boolean; overload;
function _DecodeStrings(srcArray: TArray<TFString>; var dstArray: TArray<TFString>): Boolean;  overload;
function _DecodeStrings(srcArray: TArray<TArray<TFString>>; var dstArray: TArray<TArray<TFString>>): Boolean; overload;

implementation
// String-Helpers: Conversion to ...
// -----------------------------------------------------------------------------
function _PTFChar(const str: TFString): PTFChar;
begin
 if Length(str) > 0 then begin
   Result := PAnsiChar(str);
 end
 else
   Result := Nil;
end;
function _PTFChar(const str: String; var strBuf: TFString): PTFChar;
begin
 if Length(str) > 0 then begin
   strBuf := TFString(str);
   Result := PAnsiChar(strBuf);
 end
 else begin
   strBuf := '';
   Result := Nil;
 end;
end;
//------------------------------------------------------------------------------

function _AllocMem(values: TArray<Integer>; 
                   var dim, byte_size: TF_int64_t):  Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PInt32;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int32) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PInt32(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMem(values: TArray<TArray<Integer>>; 
                   var dim1, dim2, byte_size: TF_int64_t):  Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PInt32;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Int32) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PInt32(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMem(values: TArray<Single>; 
                   var dim, byte_size: TF_int64_t):  Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PSingle;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Single) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PSingle(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMem(values: TArray<TArray<Single>>; 
                   var dim1, dim2, byte_size: TF_int64_t):  Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PSingle;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Single) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PSingle(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMem(values: TArray<TFString>; 
                   var dim, byte_size: TF_int64_t): Pointer;
var
 i, l_iByteSize: Integer;
 l_iStrLng, l_iOffset: UInt64;
 l_iFullByteSize: TF_int64_t;
 l_pOffset: PUInt64;
 l_pData, l_pBase, l_pSrc, l_pDataStart, l_pLimit:  PTFChar;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := 0;
   for i := 0 to dim-1 do
     l_iFullByteSize := l_iFullByteSize + sizeof(UInt64) + sizeof(TFChar) * (Length(values[i])+1);   // <- Mit \0 am Schluss
   GetMem(l_pBase, l_iFullByteSize);
   l_pData      := l_pBase;
   l_pOffset    := PUInt64(l_pBase);
   l_pDataStart := l_pBase + sizeof(UInt64) * dim;
   l_pLimit     := l_pBase + l_iFullByteSize;
   l_iOffset    := 0;
   for i := 0 to dim-1 do begin
     l_iStrLng   := UInt64(Length(values[i])+1);
     l_iByteSize := sizeof(TFChar) * l_iStrLng;
     l_pOffset^  := l_iOffset;
     Inc(l_pOffset);
     l_pSrc  := PTFChar(@(values[i][1]));
     l_pData := l_pDataStart + l_iOffset;
     Move(l_pSrc^, l_pData^, l_iByteSize);
     l_pData  := l_pData + l_iByteSize-1;
     l_pData^ := #0;     // <-
     l_iOffset := l_iOffset + l_iByteSize;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMem(values: TArray<TArray<TFString>>; 
                   var dim1, dim2, byte_size: TF_int64_t): Pointer;
var
 i, j, l_iByteSize: Integer;
 l_iStrLng, l_iOffset: UInt64;
 l_iFullByteSize: TF_int64_t;
 l_pOffset: PUInt64;
 l_pData, l_pBase, l_pSrc, l_pDataStart, l_pLimit:  PTFChar;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2 := Length(values[0]);
   l_iFullByteSize := 0;
   for i := 0 to dim1-1 do begin
     for j := 0 to dim2-1 do begin
       l_iFullByteSize := l_iFullByteSize + sizeof(UInt64) + sizeof(TFChar) * (Length(values[i][j])+1);  // <- Mit \0 am Schluss
     end;
   end;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData      := l_pBase;
   l_pOffset    := PUInt64(l_pBase);
   l_pDataStart := l_pBase + sizeof(UInt64) * dim1;
   l_pLimit     := l_pBase + l_iFullByteSize;
   l_iOffset    := 0;
   for i := 0 to dim1-1 do begin
     for j := 0 to dim2-1 do begin
       l_iStrLng   := UInt64(Length(values[i][j])+1);
       l_iByteSize := sizeof(TFChar) * l_iStrLng;
       l_pOffset^  := l_iOffset;
       Inc(l_pOffset);
       l_pSrc  := PTFChar(@(values[i][j][1]));
       l_pData := l_pDataStart + l_iOffset;
       Move(l_pSrc^, l_pData^, l_iByteSize);
       l_pData  := l_pData + l_iByteSize-1;
       l_pData^ := #0;       // <- \0
       l_iOffset := l_iOffset + l_iByteSize;
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _GetArray(data: Pointer; const dim: TF_int64_t;
                   var values: TArray<Integer>):  Boolean;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PInt32;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int32) * dim;
   l_pData := PInt32(data);
   l_pVal  := PInt32(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArray(data: Pointer; const dim1, dim2: TF_int64_t;
                   var values: TArray<TArray<Integer>>):  Boolean;
var
 i, j: Integer;
 l_pVal, l_pData: PInt32;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PInt32(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PInt32(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArray(data: Pointer; const dim: TF_int64_t;
                   var values: TArray<Single>):  Boolean; overload;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PSingle;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Single) * dim;
   l_pData := PSingle(data);
   l_pVal  := PSingle(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArray(data: Pointer; const dim1, dim2: TF_int64_t;
                   var values: TArray<TArray<Single>>):  Boolean; overload;
var
 i, j: Integer;
 l_pVal, l_pData: PSingle;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PSingle(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PSingle(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArray(data: Pointer; const dim, total_byte_size: TF_int64_t;
                   var values: TArray<TFString>): Boolean; overload;
var
 i, l_iByteSize: Integer;
 l_iStrLng, l_iOffset, l_iNextOffset, l_iLimitOffset: UInt64;
 l_iFullByteSize: TF_int64_t;
 l_pLng: PUInt64;
 l_pOffset: PUInt64;
 l_pData, l_pBase, l_pDst, l_pDataStart, l_pLimit:  PTFChar;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   SetLength(values, dim);
   l_pBase   := PTFChar(data);
   l_pLimit  := l_pBase + total_byte_size;
   l_pDataStart:= l_pBase + sizeof(UInt64) * dim;
   l_pOffset := PUInt64(l_pBase);
   l_iLimitOffset := UInt64(l_pLimit - l_pDataStart);
   for i := 0 to dim-1 do begin
     l_iOffset := l_pOffset^;
     l_pData := l_pDataStart + l_iOffset;
     Inc(l_pOffset);
     if i < dim-1 then
       l_iNextOffset := l_pOffset^
     else
       l_iNextOffset := l_iLimitOffset;
     Assert(l_iNextOffset <= l_iLimitOffset, '_GetArray for TFStringArray1d: l_iNextOffset <= l_iLimitOffset');
     Assert(l_iOffset < l_iLimitOffset, '_GetArray for TFStringArray1d: l_iOffset < l_iLimitOffset');
     l_iStrLng  := l_iNextOffset - l_iOffset;
     l_iByteSize:= sizeof(TFChar) * l_iStrLng;
     SetLength(values[i],l_iStrLng);
     l_pDst     := PTFChar(@(values[i][1]));
     Move(l_pData^, l_pDst^, l_iByteSize);
     if values[i][l_iStrLng] = #0 then
       SetLength(values[i],l_iStrLng-1);
   end;
   Result := True;
 end;
end;
function _GetArray(data: Pointer; const dim1, dim2, total_byte_size: TF_int64_t;
                   var values: TArray<TArray<TFString>>): Boolean; overload;
var
 i, j, l_iByteSize: Integer;
 l_iStrLng, l_iOffset, l_iNextOffset, l_iLimitOffset: UInt64;
 l_iFullByteSize: TF_int64_t;
 l_pLng: PUInt64;
 l_pOffset: PUInt64;
 l_pData, l_pBase, l_pDst, l_pDataStart, l_pLimit:  PTFChar;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   SetLength(values, dim1);
   l_pBase   := PTFChar(data);
   l_pLimit  := l_pBase + total_byte_size;
   l_pDataStart:= l_pBase + sizeof(UInt64) * dim1;
   l_pOffset := PUInt64(l_pBase);
   l_iLimitOffset := UInt64(l_pLimit - l_pDataStart);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     for j := 0 to dim2-1 do begin
       l_iOffset := l_pOffset^;
       l_pData := l_pDataStart + l_iOffset;
       Inc(l_pOffset);
       if i < dim1-1 then
         l_iNextOffset := l_pOffset^
       else
         l_iNextOffset := l_iLimitOffset;
       Assert(l_iNextOffset <= l_iLimitOffset, '_GetArray for TFStringArray2d: l_iNextOffset <= l_iLimitOffset');
       Assert(l_iOffset < l_iLimitOffset, '_GetArray for TFStringArray2d: l_iOffset < l_iLimitOffset');
       l_iStrLng  := l_iNextOffset - l_iOffset;
       l_iByteSize:= sizeof(TFChar) * l_iStrLng;
       SetLength(values[i][j],l_iStrLng);
       l_pDst     := PTFChar(@(values[i][j][1]));
       Move(l_pData^, l_pDst^, l_iByteSize);
       if values[i][j][l_iStrLng] = #0 then
         SetLength(values[i][j],l_iStrLng-1);
     end;
   end;
   Result := True;
 end;
end;
function _AllocMemAByte(values: TArray<Byte>; 
                        var dim, byte_size: TF_int64_t): Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PInt8;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int8) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PInt8(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemAByte(values: TArray<TArray<Byte>>;  
                        var dim1, dim2, byte_size: TF_int64_t): Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PInt8;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Int8) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PInt8(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemABool(values: TArray<Boolean>; 
                        var dim, byte_size: TF_int64_t): Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PBoolean;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Boolean) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PBoolean(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemABool(values: TArray<TArray<Boolean>>; 
                        var dim1, dim2, byte_size: TF_int64_t): Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PInt8;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Int8) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PInt8(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemAInt16(values: TArray<Int16>;
                         var dim, byte_size: TF_int64_t): Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PInt16;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int16) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal := PInt16(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemAInt16(values: TArray<TArray<Int16>>;
                         var dim1, dim2, byte_size: TF_int64_t): Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PInt16;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(TF_int64_t) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PInt16(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemAInt64(values: TArray<Int64>; 
                         var dim, byte_size: TF_int64_t): Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PInt64;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int64) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PInt64(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemAInt64(values: TArray<TArray<Int64>>; 
                         var dim1, dim2, byte_size: TF_int64_t): Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData: PInt64;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Int64) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PInt64(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemADouble(values: TArray<Double>; 
                          var dim, byte_size: TF_int64_t):  Pointer;
var
 l_iFullByteSize: Integer;
 l_pBase:   Pointer;
 l_pVal:    PDouble;
begin
 dim := Length(values);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Double) * dim;
   GetMem(l_pBase, l_iFullByteSize);
   l_pVal       := PDouble(@values[0]);
   Move(l_pVal^, l_pBase^, l_iFullByteSize);
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _AllocMemADouble(values: TArray<TArray<Double>>; 
                          var dim1, dim2, byte_size: TF_int64_t):  Pointer;
var
 i, j, l_iFullByteSize: Integer;
 l_pBase:  Pointer;
 l_pVal, l_pData:  PDouble;
begin
 dim1 := Length(values);
 if dim1 > 0 then begin
   dim2       := Length(values[0]);
   l_iFullByteSize := sizeof(Double) * dim1 * dim2;
   GetMem(l_pBase, l_iFullByteSize);
   l_pData := l_pBase;
   for i := 0 to dim1-1 do begin
     l_pVal := PDouble(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pData^ := l_pVal^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   byte_size := TF_int64_t(l_iFullByteSize);
   Result := l_pBase;
 end
 else begin
   byte_size := 0;
   Result := Nil;
 end;
end;
function _GetArrayAByte(data: Pointer; const dim: TF_int64_t;
                        var values: TArray<Byte>): Boolean;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PInt8;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int8) * dim;
   l_pData := PInt8(data);
   l_pVal  := PInt8(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArrayAByte(data: Pointer; const dim1, dim2: TF_int64_t;
                        var values: TArray<TArray<Byte>>): Boolean;
var
 i, j: Integer;
 l_pVal, l_pData: PInt8;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PInt8(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PInt8(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArrayABool(data: Pointer; const dim: TF_int64_t;
                        var values: TArray<Boolean>): Boolean;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PBoolean;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Boolean) * dim;
   l_pData := PBoolean(data);
   l_pVal  := PBoolean(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArrayABool(data: Pointer; const dim1, dim2: TF_int64_t;
                        var values: TArray<TArray<Boolean>>): Boolean;
var
 i, j: Integer;
 l_pVal, l_pData: PBoolean;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PBoolean(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PBoolean(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArrayAInt16(data: Pointer; const dim: TF_int64_t;
                         var values: TArray<Int16>):   Boolean;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PInt16;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int16) * dim;
   l_pData := PInt16(data);
   l_pVal  := PInt16(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArrayAInt16(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Int16>>): Boolean; overload;
var
 i, j: Integer;
 l_pVal, l_pData: PInt16;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PInt16(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PInt16(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArrayAInt64(data: Pointer; const dim: TF_int64_t;
                         var values: TArray<Int64>):   Boolean;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PInt64;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Int64) * dim;
   l_pData := PInt64(data);
   l_pVal  := PInt64(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArrayAInt64(data: Pointer; const dim1, dim2: TF_int64_t;
                              var values: TArray<TArray<Int64>>): Boolean; overload;
var
 i, j: Integer;
 l_pVal, l_pData: PInt64;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PInt64(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PInt64(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_pVal^ := l_pData^;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetArrayADouble(data: Pointer; const dim: TF_int64_t;
                          var values: TArray<Double>):   Boolean; overload;
var
 l_iFullByteSize: Integer;
 l_pData, l_pVal: PDouble;
begin
 Result := False;
 SetLength(values,dim);
 if dim > 0 then begin
   l_iFullByteSize := sizeof(Double) * dim;
   l_pData := PDouble(data);
   l_pVal  := PDouble(@values[0]);
   Move(l_pData^, l_pVal^, l_iFullByteSize);
   Result := True;
 end;
end;
function _GetArrayADouble(data: Pointer; const dim1, dim2: TF_int64_t;
                          var values: TArray<TArray<Double>>): Boolean; overload;
var
 i, j: Integer;
 l_dVal: Double;
 l_pVal, l_pData: PDouble;
begin
 Result := False;
 SetLength(values,dim1);
 if dim1 > 0 then begin
   l_pData := PDouble(data);
   for i := 0 to dim1-1 do begin
     SetLength(values[i],dim2);
     l_pVal := PDouble(@values[i][0]);
     for j := 0 to dim2-1 do begin
       l_dVal  := l_pData^;
       l_pVal^ := l_dVal;
       Inc(l_pData);
       Inc(l_pVal);
     end;
   end;
   Result := True;
 end;
end;
function _GetPointerArray(data: Pointer; const dim, one_byte_size: Integer;
                          var parray: TArray<Pointer>):  Boolean;
var
 i: Integer;
 l_iOffset: TF_int64_t;
 l_pVal: PTFChar;
begin
 Result := False;
 SetLength(parray,dim);
 if dim > 0 then begin
   l_pVal := PTFChar(data);
   l_iOffset := 0;
   for i := 0 to dim-1 do begin
     parray[i] := l_pVal + l_iOffset;
     l_iOffset := l_iOffset + one_byte_size;
   end;
   Result := True;
 end;
end;
function _GetPTFCharArray(values: TArray<TFString>;
                          var lengths: TArray<TF_size_t>;
                          var parray: TArray<PTFChar>): Boolean;
var
 i, n, l: Integer;
 l_pVal: PTFChar;
begin
 Result := False;
 n := Length(values);
 SetLength(lengths,n);
 SetLength(parray,n);
 if n > 0 then begin
   for i := 0 to n-1 do begin
     l := Length(values[i]);
     lengths[i] := l;
     parray[i]  := @(values[i][1]);
   end;
   Result := True;
 end;
end;
// Encode-/Decode-Helpers:
// -----------------------------------------------------------------------------
function _EncodeStr(const str: TFString): TFString;
var
 lng: Integer;
 l_iSize, l_iLen, l_iEncodeLen: TF_size_t;
 l_sEncodeStr: TFString;
 l_pStr, l_pEncodeStr: PTFChar;
 l_pStatus : PTF_Status;
begin
 Result := '';
 lng := Length(str);
 if lng > 0 then begin
   l_iLen       := lng * SizeOf(TFChar);
   l_iEncodeLen := l_iLen + 8;
   l_pStatus    := TF_NewStatus();
   SetLength(l_sEncodeStr,lng+8);     // 8-byte offset
   l_pStr := _PTFChar(str);
   l_pEncodeStr := _PTFChar(l_sEncodeStr);
   var t := TF_AllocateTensor(Integer(TF_STRING), nil, 0, TF_TSRING_SIZE);
   var dst : PTFString := TF_TensorData(t);
   TF_StringInit(dst);
   TF_StringCopy(dst, l_pStr, l_iEncodeLen);


   //l_iSize    := TF_StringEncode(l_pStr, l_iLen, l_pEncodeStr, l_iEncodeLen, l_pStatus);
   if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
     (l_pEncodeStr+l_iSize)^ := #0;
     Result := l_pEncodeStr;
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
function _EncodeStrings(srcArray: TArray<TFString>; var dstArray: TArray<TFString>): Boolean;
var
 i, l_iDim, l_iOrgLen, l_iEncodeLen, l_iSize: Integer;
 l_sOrgStr, l_sEncodeStr: TFString;
 l_pStatus : PTF_Status;
 l_pEncodeStr : PTFString;
 l_pOrgStr: PTFChar;
 l_sMsg: TFString;
begin
 Result := False;
 l_iDim := Length(srcArray);
 if l_iDim > 0 then begin
   Result := True;
   l_pStatus := TF_NewStatus();
   SetLength(dstArray,l_iDim);
   for i := 0 to l_iDim-1 do begin
     l_iOrgLen    := Length(srcArray[i]);
     l_iEncodeLen := l_iOrgLen+2;
     SetLength(l_sEncodeStr,l_iEncodeLen);
     l_sOrgStr := srcArray[i];
     l_pOrgStr := @(l_sOrgStr[1]);
     l_pEncodeStr := @(l_sEncodeStr[1]);
     var t := TF_AllocateTensor(Integer(TF_STRING), nil, 0, TF_TSRING_SIZE);
     var dst : PTFString := TF_TensorData(t);
     TF_StringInit(dst);
     TF_StringCopy(dst, l_pOrgStr, l_iEncodeLen);
     //l_iSize := TF_StringEncode(l_pOrgStr, l_iOrgLen,      l_pEncodeStr, l_iEncodeLen, l_pStatus);
     (*if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
       (l_pEncodeStr+l_iSize)^ := #0;
       dstArray[i] := l_sEncodeStr;
     end
     else begin
       l_sMsg := TF_Message(l_pStatus);
       Result := False;
     end;*)
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
function _EncodeStrings(srcArray: TArray<TArray<TFString>>; var dstArray: TArray<TArray<TFString>>): Boolean;
var
 i, j, l_iDim1, l_iDim2, l_iOrgLen, l_iEncodeLen, l_iSize: Integer;
 l_sOrgStr, l_sEncodeStr: TFString;
 l_pEncodeStr: PTFChar;
 l_pOrgStr   : PTFString;
 l_pStatus : PTF_Status;
 l_sMsg: TFString;
begin
 Result := False;
 l_iDim1 := Length(srcArray);
 if l_iDim1 > 0 then begin
   Result := True;
   l_pStatus := TF_NewStatus();
   SetLength(dstArray,l_iDim1);
   for i := 0 to l_iDim1-1 do begin
     l_iDim2 := Length(srcArray[i]);
     SetLength(dstArray[i], l_iDim2);
     for j := 0 to l_iDim2-1 do begin
       l_iOrgLen    := Length(srcArray[i][j]);
       l_iEncodeLen := l_iOrgLen+2;
       SetLength(l_sEncodeStr,l_iEncodeLen);
       l_sOrgStr := srcArray[i][j];
       l_pOrgStr := @(l_sOrgStr[1]);
       l_pEncodeStr := @(l_sEncodeStr[1]);
       //l_iSize := TF_StringEncode(l_pOrgStr, l_iOrgLen, l_pEncodeStr, l_iEncodeLen, l_pStatus);
       //l_iSize := TF_StringEncode(l_pOrgStr, l_iOrgLen, l_pEncodeStr, l_iEncodeLen, l_pStatus);
       if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
         (l_pEncodeStr+l_iSize)^ := #0;
         dstArray[i][j] := l_sEncodeStr;
       end
       else
         Result := False;
     end;
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
function _DecodeStr(const str: TFString): TFString;
var
 lng: Integer;
 l_iSize, l_iLen, l_iDecodeLen: TF_size_t;
 l_sDecodeStr, l_sDecodeStrDest: TFString;
 l_pStr : PTFString;
 l_pDecodeStr, l_pDecodeStrDest: PTFChar;
 l_pStatus : PTF_Status;
begin
 Result := '';
 lng := Length(str);
 if lng > 0 then begin
   l_iLen       := lng * SizeOf(TFChar);
   l_iDecodeLen := l_iLen;
   l_pStatus    := TF_NewStatus();
   l_pStr := @(str[1]);
   SetLength(l_sDecodeStr,lng);
   l_pDecodeStr := _PTFChar(l_sDecodeStr);
   l_pDecodeStr := TF_StringGetDataPointer(l_pStr);
   l_iDecodeLen := TF_StringGetSize(l_pStr);
   //l_iSize :=  TF_StringDecode(l_pStr, l_iLen, l_pDecodeStr, @l_iDecodeLen, l_pStatus);
   if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
     (l_pDecodeStr+l_iDecodeLen)^ := #0;
     Result := l_pDecodeStr;
     { SetLength(l_sDecodeStrDest,l_iDecodeLen);
       l_pDecodeStrDest := _PTFChar(l_sDecodeStrDest);
       System.AnsiStrings.StrPCopy(l_pDecodeStrDest,l_pDecodeStr);
       Result := l_pDecodeStrDest;  }
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
function _DecodeStrings(srcArray: TArray<TFString>; var dstArray: TArray<TFString>): Boolean;
var
 l_iSize, l_iOrgLen, l_iDecodeLen: TF_size_t;
 i, l_iDim: Integer;
 l_sOrgStr, l_sDecodeStr, l_sDecodeStrDest: TFString;
 l_pStatus : PTF_Status;
 l_pOrgStr : PTFString;
 l_pDecodeStr: PTFChar;
begin
 Result := False;
 l_iDim := Length(srcArray);
 if l_iDim > 0 then begin
   Result := True;
   l_pStatus := TF_NewStatus();
   SetLength(dstArray,l_iDim);
   for i := 0 to l_iDim-1 do begin
     l_sOrgStr    := srcArray[i];
     l_iOrgLen    := Length(l_sOrgStr);
     l_iDecodeLen := l_iOrgLen+2;
     SetLength(l_sDecodeStr,l_iDecodeLen);
     l_pOrgStr := @(l_sOrgStr[1]);
     l_pDecodeStr := @(l_sDecodeStr[1]);
     l_pDecodeStr := TF_StringGetDataPointer(l_pOrgStr);
     l_iDecodeLen := TF_StringGetSize(@(l_sOrgStr[1]));
     //l_iSize :=  TF_StringDecode(l_pOrgStr, l_iOrgLen, l_pDecodeStr, @l_iDecodeLen, l_pStatus);
     if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
       (l_pDecodeStr+l_iDecodeLen)^ := #0;
       SetLength(l_sDecodeStrDest,l_iDecodeLen);
       System.AnsiStrings.StrPCopy(PTFChar(l_sDecodeStrDest),l_pDecodeStr);
       dstArray[i] := l_sDecodeStrDest;
     end
     else
       Result := False;
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
function _DecodeStrings(srcArray: TArray<TArray<TFString>>; var dstArray: TArray<TArray<TFString>>): Boolean;
var
 l_iSize, l_iOrgLen, l_iDecodeLen: TF_size_t;
 i, j, l_iDim1, l_iDim2: Integer;
 l_sOrgStr, l_sDecodeStr: TFString;
 l_pStatus : PTF_Status;
 l_pOrgStr : PTFString;
 l_pDecodeStr: PTFChar;
begin
 Result := False;
 l_iDim1 := Length(srcArray);
 if l_iDim1 > 0 then begin
   Result := True;
   l_pStatus := TF_NewStatus();
   SetLength(dstArray,l_iDim1);
   for i := 0 to l_iDim1-1 do begin
     l_iDim2 := Length(srcArray);
     SetLength(dstArray[i],l_iDim2);
     for j := 0 to l_iDim2-1 do begin
       l_sOrgStr    := srcArray[i][j];
       l_iOrgLen    := Length(l_sOrgStr);
       l_iDecodeLen := l_iOrgLen+2;
       SetLength(l_sDecodeStr,l_iDecodeLen);
       l_pOrgStr := @(l_sOrgStr[1]);
       l_pDecodeStr := @(l_sDecodeStr[1]);
       l_pDecodeStr := TF_StringGetDataPointer(l_pOrgStr);
       l_iDecodeLen := TF_StringGetSize(@(l_sOrgStr[1]));
       //l_iSize :=  TF_StringDecode(l_pOrgStr, l_iOrgLen, l_pDecodeStr, @l_iDecodeLen, l_pStatus);
       if TF_GetCode(l_pStatus) = TF_Code.TF_OK then begin
         (l_pDecodeStr+l_iDecodeLen)^ := #0;
         dstArray[i][j] := l_sDecodeStr;
       end
       else
         Result := False;
     end;
   end;
   TF_DeleteStatus(l_pStatus);
 end;
end;
end.
