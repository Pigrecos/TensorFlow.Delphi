(* Protocol buffer code generator, for Delphi
 * Copyright (c) 2020 Marat Shaimardanov
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

unit Oz.Pb.Scanner;

interface

uses
  System.SysUtils, Oz.Cocor.Utils, Oz.Cocor.Lib;

type

  TpbScanner = class(TBaseScanner)
  private
    function Comment0: Boolean;
    function Comment1: Boolean;
    procedure CheckLiteral;
  protected
    procedure NextCh; override;
    procedure AddCh; override;
    function NextToken: TToken; override;
  public
    constructor Create(const src: string);
  end;

implementation

constructor TpbScanner.Create(const src: string);
var
  i: Integer;
begin
  inherited;
  MaxToken := 62;
  NoSym := 62;
  for i := 65 to 90 do start.Add(i, 1);
  for i := 97 to 122 do start.Add(i, 1);
  for i := 49 to 57 do start.Add(i, 14);
  start.Add(48, 15);
  start.Add(34, 16);
  start.Add(39, 17);
  start.Add(123, 26);
  start.Add(125, 27);
  start.Add(61, 28);
  start.Add(59, 29);
  start.Add(40, 30);
  start.Add(41, 31);
  start.Add(46, 32);
  start.Add(45, 33);
  start.Add(43, 34);
  start.Add(91, 35);
  start.Add(44, 36);
  start.Add(93, 37);
  start.Add(60, 38);
  start.Add(62, 39);
  start.Add(Ord(TBuffer.EF), -1);
end;

procedure TpbScanner.NextCh;
begin
  if oldEols > 0 then
  begin
    ch := LF; Dec(oldEols);
  end
  else
  begin
    pos := buffer.Pos; ch := Chr(buffer.Read); Inc(col);
    // replace isolated CR by LF in order to make
    // eol handling uniform across Windows, Unix and Mac
    if (ch = CR) and (buffer.Peek <> Ord(LF)) then
      ch := LF;
    if ch = LF then
    begin
      Inc(line); col := 0;
    end;
  end;
end;

procedure TpbScanner.AddCh;
begin
  if ch <> TBuffer.EF then
  begin
    tval := tval + ch; Inc(tlen);
    NextCh;
  end;
end;

function TpbScanner.Comment0: Boolean;
var
  level, pos0, line0, col0: Integer;
begin
  level := 1; pos0 := pos; line0 := line; col0 := col;
  NextCh;
  if ch = '/' then
  begin
    NextCh;
    repeat
      if ch = #13 then
      begin
        Dec(level);
        if level = 0 then
        begin
          oldEols := line - line0; NextCh;
          exit(True);
        end;
        NextCh;
      end
      else if ch = TBuffer.EF then
        exit(False)
      else
        NextCh;
    until False;
  end
  else
  begin
    buffer.Pos := pos0; NextCh;
    line := line0; col := col0;
  end;
  Result := False;
end;

function TpbScanner.Comment1: Boolean;
var
  level, pos0, line0, col0: Integer;
begin
  level := 1; pos0 := pos; line0 := line; col0 := col;
  NextCh;
  if ch = '*' then
  begin
    NextCh;
    repeat
      if ch = '*' then
      begin
        NextCh;
        if ch = '/' then
        begin
          Dec(level);
          if level = 0 then
          begin
            oldEols := line - line0; NextCh;
            exit(True);
          end;
          NextCh;
        end;
      end
      else if ch = '/' then
      begin
        NextCh;
        if ch = '*' then
        begin
          Inc(level); NextCh;
        end;
      end
      else if ch = TBuffer.EF then
        exit(False)
      else
        NextCh;
    until False;
  end
  else
  begin
    buffer.Pos := pos0; NextCh;
    line := line0; col := col0;
  end;
  Result := False;
end;

procedure TpbScanner.CheckLiteral;
begin
  if t.val = 'message' then
    t.kind := 9
  else if t.val = 'syntax' then
    t.kind := 12
  else if t.val = 'import' then
    t.kind := 15
  else if t.val = 'weak' then
    t.kind := 16
  else if t.val = 'public' then
    t.kind := 17
  else if t.val = 'package' then
    t.kind := 18
  else if t.val = 'option' then
    t.kind := 19
  else if t.val = 'service' then
    t.kind := 23
  else if t.val = 'rpc' then
    t.kind := 24
  else if t.val = 'stream' then
    t.kind := 25
  else if t.val = 'returns' then
    t.kind := 26
  else if t.val = 'inf' then
    t.kind := 27
  else if t.val = 'nan' then
    t.kind := 28
  else if t.val = 'true' then
    t.kind := 29
  else if t.val = 'false' then
    t.kind := 30
  else if t.val = 'repeated' then
    t.kind := 33
  else if t.val = 'optional' then
    t.kind := 34
  else if t.val = 'required' then
    t.kind := 35
  else if t.val = 'map' then
    t.kind := 39
  else if t.val = 'oneof' then
    t.kind := 42
  else if t.val = 'double' then
    t.kind := 43
  else if t.val = 'float' then
    t.kind := 44
  else if t.val = 'bytes' then
    t.kind := 45
  else if t.val = 'int32' then
    t.kind := 46
  else if t.val = 'int64' then
    t.kind := 47
  else if t.val = 'uint32' then
    t.kind := 48
  else if t.val = 'uint64' then
    t.kind := 49
  else if t.val = 'sint32' then
    t.kind := 50
  else if t.val = 'sint64' then
    t.kind := 51
  else if t.val = 'fixed32' then
    t.kind := 52
  else if t.val = 'fixed64' then
    t.kind := 53
  else if t.val = 'sfixed32' then
    t.kind := 54
  else if t.val = 'sfixed64' then
    t.kind := 55
  else if t.val = 'bool' then
    t.kind := 56
  else if t.val = 'string' then
    t.kind := 57
  else if t.val = 'reserved' then
    t.kind := 58
  else if t.val = 'to' then
    t.kind := 59
  else if t.val = 'max' then
    t.kind := 60
  else if t.val = 'enum' then
    t.kind := 61
end;

function TpbScanner.NextToken: TToken;
var
  recKind, recEnd, state: Integer;
begin
  while (ch = ' ') or Between(ch, #9, #13) do
    NextCh;
  if ((ch = '/') and Comment0) or
     ((ch = '/') and Comment1) then exit(NextToken);
  recKind := NoSym;
  recEnd := pos;
  t := NewToken;
  t.pos := pos; t.col := col; t.line := line;
  if start.ContainsKey(Ord(ch)) then
    state := start[Ord(ch)]
  else
    state := 0;
  tval := ''; tlen := 0;
  AddCh;
  repeat
    case state of
      -1:
      begin
        t.kind := eofSym;
        break; // NextCh already done
      end;
      0:
      begin
        if recKind <> NoSym then
        begin
          tlen := recEnd - t.pos;
          SetScannerBehindT;
        end;
        t.kind := recKind;
        break; // NextCh already done
      end;
      1:
      begin
        recEnd := pos; recKind := 1;
        if Between(ch, '0', '9') or Between(ch, 'A', 'Z') or (ch = '_') or
           Between(ch, 'a', 'z') then
        begin
          AddCh; state := 1;
        end
        else
        begin
          t.kind := 1; t.val := tval; CheckLiteral;
          exit(t);
        end;
      end;
      2:
      if Between(ch, '0', '9') or Between(ch, 'A', 'F') or Between(ch, 'a', 'f') then
      begin
        AddCh; state := 3;
      end
      else
      begin
        state := 0;
      end;
      3:
      begin
        recEnd := pos; recKind := 4;
        if Between(ch, '0', '9') or Between(ch, 'A', 'F') or Between(ch, 'a', 'f') then
        begin
          AddCh; state := 3;
        end
        else
        begin
          t.kind := 4; break;
        end;
      end;
      4:
      if Between(ch, '0', '9') then
      begin
        AddCh; state := 4;
      end
      else if ch = '.' then
      begin
        AddCh; state := 5;
      end
      else
      begin
        state := 0;
      end;
      5:
      begin
        recEnd := pos; recKind := 5;
        if Between(ch, '0', '9') then
        begin
          AddCh; state := 5;
        end
        else if (ch = 'E') or (ch = 'e') then
        begin
          AddCh; state := 6;
        end
        else
        begin
          t.kind := 5; break;
        end;
      end;
      6:
      if Between(ch, '0', '9') then
      begin
        AddCh; state := 8;
      end
      else if (ch = '+') or (ch = '-') then
      begin
        AddCh; state := 7;
      end
      else
      begin
        state := 0;
      end;
      7:
      if Between(ch, '0', '9') then
      begin
        AddCh; state := 8;
      end
      else
      begin
        state := 0;
      end;
      8:
      begin
        recEnd := pos; recKind := 5;
        if Between(ch, '0', '9') then
        begin
          AddCh; state := 8;
        end
        else
        begin
          t.kind := 5; break;
        end;
      end;
      9:
      if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
         Between(ch, '#', '[') or Between(ch, ']', #65535) then
      begin
        AddCh; state := 9;
      end
      else if ch = '"' then
      begin
        AddCh; state := 11;
      end
      else if ch = '\' then
      begin
        AddCh; state := 10;
      end
      else
      begin
        state := 0;
      end;
      10:
      if Between(ch, ' ', '~') then
      begin
        AddCh; state := 9;
      end
      else
      begin
        state := 0;
      end;
      11:
      begin
        t.kind := 6; break;
      end;
      12:
      begin
        t.kind := 7; break;
      end;
      13:
      begin
        t.kind := 8; break;
      end;
      14:
      begin
        recEnd := pos; recKind := 2;
        if Between(ch, '0', '9') then
        begin
          AddCh; state := 14;
        end
        else if ch = '.' then
        begin
          AddCh; state := 5;
        end
        else
        begin
          t.kind := 2; break;
        end;
      end;
      15:
      begin
        recEnd := pos; recKind := 3;
        if Between(ch, '0', '7') then
        begin
          AddCh; state := 18;
        end
        else if Between(ch, '8', '9') then
        begin
          AddCh; state := 4;
        end
        else if (ch = 'X') or (ch = 'x') then
        begin
          AddCh; state := 2;
        end
        else if ch = '.' then
        begin
          AddCh; state := 5;
        end
        else
        begin
          t.kind := 3; break;
        end;
      end;
      16:
      if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
         Between(ch, '#', '[') or Between(ch, ']', #65535) then
      begin
        AddCh; state := 16;
      end
      else if (ch = #10) or (ch = #13) then
      begin
        AddCh; state := 12;
      end
      else if ch = '"' then
      begin
        AddCh; state := 11;
      end
      else if ch = '\' then
      begin
        AddCh; state := 19;
      end
      else
      begin
        state := 0;
      end;
      17:
      if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
         Between(ch, '#', '&') or Between(ch, '(', '[') or Between(ch, ']', #65535) then
      begin
        AddCh; state := 20;
      end
      else if ch = #39 then
      begin
        AddCh; state := 9;
      end
      else if ch = '"' then
      begin
        AddCh; state := 21;
      end
      else if ch = '\' then
      begin
        AddCh; state := 22;
      end
      else
      begin
        state := 0;
      end;
      18:
      begin
        recEnd := pos; recKind := 3;
        if Between(ch, '0', '7') then
        begin
          AddCh; state := 18;
        end
        else if Between(ch, '8', '9') then
        begin
          AddCh; state := 4;
        end
        else if ch = '.' then
        begin
          AddCh; state := 5;
        end
        else
        begin
          t.kind := 3; break;
        end;
      end;
      19:
      if Between(ch, ' ', '~') then
      begin
        AddCh; state := 16;
      end
      else
      begin
        state := 0;
      end;
      20:
      if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
         Between(ch, '#', '&') or Between(ch, '(', '[') or Between(ch, ']', #65535) then
      begin
        AddCh; state := 9;
      end
      else if ch = '"' then
      begin
        AddCh; state := 11;
      end
      else if ch = '\' then
      begin
        AddCh; state := 10;
      end
      else if ch = #39 then
      begin
        AddCh; state := 23;
      end
      else
      begin
        state := 0;
      end;
      21:
      begin
        recEnd := pos; recKind := 6;
        if ch = #39 then
        begin
          AddCh; state := 13;
        end
        else
        begin
          t.kind := 6; break;
        end;
      end;
      22:
      if Between(ch, ' ', '~') then
      begin
        AddCh; state := 24;
      end
      else
      begin
        state := 0;
      end;
      23:
      begin
        recEnd := pos; recKind := 8;
        if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
           Between(ch, '#', '[') or Between(ch, ']', #65535) then
        begin
          AddCh; state := 9;
        end
        else if ch = '"' then
        begin
          AddCh; state := 11;
        end
        else if ch = '\' then
        begin
          AddCh; state := 10;
        end
        else
        begin
          t.kind := 8; break;
        end;
      end;
      24:
      if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
         Between(ch, '#', '&') or Between(ch, '(', '/') or Between(ch, ':', '@') or
         Between(ch, 'G', '[') or Between(ch, ']', '`') or Between(ch, 'g', #65535) then
      begin
        AddCh; state := 9;
      end
      else if Between(ch, '0', '9') or Between(ch, 'A', 'F') or Between(ch, 'a', 'f') then
      begin
        AddCh; state := 24;
      end
      else if ch = '"' then
      begin
        AddCh; state := 11;
      end
      else if ch = '\' then
      begin
        AddCh; state := 10;
      end
      else if ch = #39 then
      begin
        AddCh; state := 25;
      end
      else
      begin
        state := 0;
      end;
      25:
      begin
        recEnd := pos; recKind := 8;
        if (ch <= #9) or Between(ch, #11, #12) or Between(ch, #14, '!') or
           Between(ch, '#', '[') or Between(ch, ']', #65535) then
        begin
          AddCh; state := 9;
        end
        else if ch = '"' then
        begin
          AddCh; state := 11;
        end
        else if ch = '\' then
        begin
          AddCh; state := 10;
        end
        else
        begin
          t.kind := 8; break;
        end;
      end;
      26:
      begin
        t.kind := 10; break;
      end;
      27:
      begin
        t.kind := 11; break;
      end;
      28:
      begin
        t.kind := 13; break;
      end;
      29:
      begin
        t.kind := 14; break;
      end;
      30:
      begin
        t.kind := 20; break;
      end;
      31:
      begin
        t.kind := 21; break;
      end;
      32:
      begin
        t.kind := 22; break;
      end;
      33:
      begin
        t.kind := 31; break;
      end;
      34:
      begin
        t.kind := 32; break;
      end;
      35:
      begin
        t.kind := 36; break;
      end;
      36:
      begin
        t.kind := 37; break;
      end;
      37:
      begin
        t.kind := 38; break;
      end;
      38:
      begin
        t.kind := 40; break;
      end;
      39:
      begin
        t.kind := 41; break;
      end;
    end;
  until false;
  t.val := tval;
  Result := t;
end;

end.

