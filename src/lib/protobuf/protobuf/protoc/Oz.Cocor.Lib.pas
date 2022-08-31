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

unit Oz.Cocor.Lib;

interface

uses
  System.Classes, System.SysUtils, System.Contnrs, System.Math,
  System.Generics.Defaults, System.Generics.Collections;

type
  FatalError = class(Exception);

{$Region 'TOwnedList'}

  TOwnedList<T: class> = class(TList<T>)
  private
    procedure Changed(Sender: TObject; const Item: T; Action: TCollectionNotification);
  public
    constructor Create;
  end;

{$EndRegion}

{$Region 'TOwnedDictionary'}

  TOwnedDictionary<TKey; TValue: class> = class(TDictionary<TKey, TValue>)
  protected
    procedure ValueNotify(const Value: TValue; Action: TCollectionNotification); override;
  end;

{$EndRegion}

{$Region 'TBuffer'}

  PBuffer = ^TBuffer;
  TBuffer = record
  const
    LF = #10;
    CR = #13;
    EF = #26;
  strict private
    FBuf: string;
    FPos: Integer;
  public
    constructor From(const src: string);
    procedure Open(const filename: string);
    function CharAt(pos: Integer): Char;
    function GetLine(var line: string): Boolean;
    function Eof: Boolean;
    function Read: Integer;
    function Peek: Integer;
    property Pos: Integer read FPos write FPos;
    property Buf: string read FBuf;
  end;

{$EndRegion}

{$Region 'TToken'}

  TToken = class
    next: TToken;     // Tokens are kept in linked list
    kind: Integer;    // token kind
    pos: Integer;     // token position in bytes in the source text (starting at 0)
    col: Integer;     // token column (starting at 1)
    line: Integer;    // token line (starting at 1)
    val: string;      // token value
  end;

{$EndRegion}

{$Region 'TPosition: position of source code stretch'}

  // e.g. semantic action, resolver expressions
  PPosition = ^TPosition;
  TPosition = record
    beg: Integer;        // start relative to the beginning of the file
    ends: Integer;       // end of stretch
    col: Integer;        // column number of start position
    line: Integer;       // line number of start position
    // empty content
    function Empty: Boolean;
    procedure SetEmpty;
    // init by token
    procedure Start(t: TToken);
    // update by lookahead token
    procedure Update(la: TToken); overload;
    procedure Update(la: TToken; col: Integer); overload;
  end;

{$EndRegion}

{$Region 'TBaseScanner'}

  TBaseScanner = class
  const
    LF = #10;
    CR = #13;
  type
    TStartMap = TDictionary<Integer, Integer>;
  class var
    MaxToken: Integer;
    NoSym: Integer;
  const
    eofSym = 0;
  private
    FTokens: TObjectList;
    FBuffer: TBuffer;  // scanner buffer
    tokens: TToken;    // list of tokens already peeked (first token is a dummy)
    pt: TToken;        // current peek token
    function GetBuffer: PBuffer;
  protected
    t: TToken;         // current token
    ch: Char;          // current input character
    pos: Integer;      // byte position of current character
    col: Integer;      // column number of current character
    line: Integer;     // line number of current character
    oldEols: Integer;  // EOLs that appeared in a comment;
    start: TStartMap;  // maps first token character to start state
    tval: string;      // text of current token
    tlen: Integer;     // length of current token
    procedure SetScannerBehindT;
    function NextToken: TToken; virtual; abstract;
    // peek for the next token, ignore pragmas
    function Peek: TToken;
    // make sure that peeking starts at the current scan position
    procedure ResetPeek;
    procedure NextCh; virtual; abstract;
    procedure AddCh; virtual; abstract;
  public
    constructor Create(const src: string);
    destructor Destroy; override;
    function NewToken: TToken;
    // get the next token (possibly a token already seen during peeking)
    function Scan: TToken;
    // source
    property buffer: PBuffer read GetBuffer;
  end;

{$EndRegion}

{$Region 'TErrors'}

  TErrLevel = (elSyntax, elSemantic, elWarning);
  TErrDesc = class
    lvl: TErrLevel;
    s: string;
    line, col: Integer;
    constructor Create(lvl: TErrLevel; const s: string; line, col: Integer);
    function Print(const srcLine: string): string;
  end;

  TBaseParser = class;
  TErrors = class
  type
    TErrorList = TOwnedList<TErrDesc>;
  const
    CR = #13;
    LF = #10;
  strict private
    FParser: TBaseParser;
    FErrors: TErrorList;
    FInternalErrors: TErrorList;
    function GetCount: Integer;
  public
    constructor Create(parser: TBaseParser);
    destructor Destroy; override;
    procedure SynErr(nr, line, col: Integer);
    procedure SemErr(const s: string; line, col: Integer); overload;
    procedure SemErr(const s: string); overload;
    procedure Warning(const s: string; line, col: Integer); overload;
    procedure Warning(const s: string); overload;
    procedure Print(const src: string; lst: TStrings);
    // number of errors detected
    property Count: Integer read GetCount;
  end;

{$EndRegion}

{$Region 'TBaseParser'}

  TBaseParser = class
  const
    minErrDist = 2;
  private
    FScanner: TBaseScanner;
    FErrors: TErrors;
    FListing: TStrings;
  protected
    errDist: Integer;
    noString: string;      // used in declarations of literal tokens
    t: TToken;             // last recognized token
    la: TToken;            // lookahead token
    procedure SynErr(n: Integer);
    procedure Expect(n: Integer);
    procedure ExpectWeak(n, follow: Integer);
    function WeakSeparator(n, syFol, repFol: Integer): Boolean;
    function StartOf(s: Integer): Boolean;
    function Starts(s, kind: Integer): Boolean; virtual; abstract;
    procedure Get; virtual; abstract;
  public
    constructor Create(scanner: TBaseScanner; listing: TStrings);
    destructor Destroy; override;
    function ErrorMsg(nr: Integer): string; virtual; abstract;
    procedure SemErr(const s: string);
    procedure Parse; virtual; abstract;
    procedure PrintErrors;
    property scanner: TBaseScanner read FScanner;
    property errors: TErrors read FErrors;
  end;

{$EndRegion}

{$Region 'TCocoPart'}

  TCocoPart = class
  protected
    FParser: TBaseParser;
  public
    constructor Create(parser: TBaseParser);
  end;

{$EndRegion}

{$Region 'Subroutines'}

function ToPascalString(const s: string): string;
function RTrim(const s: string): string;
function Blank(n: Integer): string;
function AsName(const name: string): string;

{$EndRegion}

implementation

{$Region 'Subroutines'}

function ToPascalString(const s: string): string;
var n: Integer;
begin
  n := Length(s);
  Assert((n >= 2) and (s[1] = '"' ) and (s[n] = '"' ));
  Result := s;
  Result[1] := '''';
  Result[n] := '''';
end;

function RTrim(const s: string): string;
var
  i: Integer;
  ch: Char;
begin
  i := Length(s);
  repeat
    Dec(i);
    if i < 0 then
      break;
    ch := s.Chars[i];
  until (ch <> #13) and (ch <> #10) and (ch <> ' ') and (ch <> #9);
  Result := s.SubString(0, i + 1);
end;

function Blank(n: Integer): string;
var i: Integer;
begin
  SetLength(Result, Max(0, n));
  for i := 1 to n do Result[i] := ' ';
end;

function AsName(const name: string): string;
begin
  Result := Copy(name, 1, 12) +  Blank(12 - Length(name));
end;

{$EndRegion}

{$Region 'TOwnedList: '}

procedure TOwnedList<T>.Changed(Sender: TObject; const Item: T;
  Action: TCollectionNotification);
begin
  if Action = cnRemoved then
    Item.Free;
end;

constructor TOwnedList<T>.Create;
begin
  inherited;
  OnNotify := Changed;
end;

{$EndRegion}

{$Region 'TOwnedDictionary'}

procedure TOwnedDictionary<TKey, TValue>.ValueNotify(const Value: TValue;
  Action: TCollectionNotification);
begin
  if Action = cnRemoved then
    Value.Free;
end;

{$EndRegion}

{$Region 'TBuffer'}

constructor TBuffer.From(const src: string);
begin
  FBuf := src;
  FPos := 1;
end;

procedure TBuffer.Open(const filename: string);
var
  list: TStringList;
begin
  list := TStringList.Create;
  try
    try
      list.LoadFromFile(filename);
    except
      on EFileStreamError do
        raise FatalError.Create('Cannot open file: ' + filename);
    end;
    FBuf := list.Text;
    FPos := 1;
  finally
    list.Free;
  end;
end;

function TBuffer.Eof: Boolean;
begin
  Result := FPos > Length(FBuf);
end;

function TBuffer.Read: Integer;
begin
  if Eof then
    Result := Ord(EF)
  else
  begin
    Result := Ord(FBuf[FPos]);
    Inc(FPos);
  end;
end;

function TBuffer.Peek: Integer;
begin
  Result := Ord(FBuf[FPos]);
end;

function TBuffer.CharAt(pos: Integer): Char;
begin
  if pos > Length(FBuf) then
    Result := EF
  else
    Result := FBuf[pos];
end;

function TBuffer.GetLine(var line: string): Boolean;
var
  len, i: Integer;
  ch: Char;
begin
  len := 0;
  repeat
    ch := CharAt(FPos + len);
    if (ch = CR) or (ch = LF) or (ch = EF) then break;
    Inc(len);
  until false;
  SetLength(line, len);
  for i := 1 to len do
  begin
    line[i] := CharAt(FPos);
    Inc(FPos);
  end;
  Result := (CharAt(FPos) <> EF);
  if CharAt(FPos) = CR then
    Inc(FPos);
  if CharAt(FPos) = LF then
    Inc(FPos);
end;

{$EndRegion}

{$Region 'TCocoPart'}

constructor TCocoPart.Create(parser: TBaseParser);
begin
  FParser := parser;
end;

{$EndRegion}

{$Region 'TPosition'}

function TPosition.Empty: Boolean;
begin
  Result := beg >= ends;
end;

procedure TPosition.SetEmpty;
begin
  beg := 0;
  ends := 0;
  col := 0;
  line := 0;
end;

procedure TPosition.Start(t: TToken);
begin
  beg := t.pos;
  col := t.col;
  line := t.line;
end;

procedure TPosition.Update(la: TToken);
begin
  ends := la.pos;
end;

procedure TPosition.Update(la: TToken; col: Integer);
begin
  ends := la.pos;
  Self.col := col;
end;

{$EndRegion}

{$Region 'TBaseScanner'}

constructor TBaseScanner.Create(const src: string);
begin
  inherited Create;
  FTokens := TObjectList.Create;
  start := TDictionary<Integer, Integer>.Create(128);
  FBuffer := TBuffer.From(src);
  pos := -1;
  line := 1;
  col := 0;
  oldEols := 0;
  NextCh;
  pt := NewToken;  // first token is a dummy
  tokens := pt;
end;

destructor TBaseScanner.Destroy;
begin
  start.Free;
  FTokens.Free;
  inherited;
end;

function TBaseScanner.GetBuffer: PBuffer;
begin
  Result := @FBuffer;
end;

function TBaseScanner.NewToken: TToken;
begin
  Result := TToken.Create;
  FTokens.Add(Result);
end;

procedure TBaseScanner.SetScannerBehindT;
var
  i: Integer;
begin
  FBuffer.Pos := t.pos;
  NextCh;
  line := t.line; col := t.col;
  for i := 0 to tlen - 1 do
    NextCh;
end;

function TBaseScanner.Scan: TToken;
begin
  if tokens.next = nil then
    Result := NextToken
  else
  begin
    tokens := tokens.next;
    pt := tokens;
    Result := tokens;
  end;
end;

function TBaseScanner.Peek: TToken;
begin
  repeat
    if pt.next = nil then
      pt.next := NextToken;
    pt := pt.next;
  until pt.kind <= MaxToken; // skip pragmas
  Result := pt;
end;

procedure TBaseScanner.ResetPeek;
begin
  pt := tokens;
end;

{$EndRegion}

{$Region 'TErrDesc'}

constructor TErrDesc.Create(lvl: TErrLevel; const s: string; line, col: Integer);
begin
  Self.lvl := lvl;
  Self.line := line;
  Self.col := col;
  Self.s := s;
end;

function TErrDesc.Print(const srcLine: string): string;
const
  ErrLevel: array [TErrLevel] of string = (
    '*****   ',
    'error   ',
    'warning ');
var
  offset: string;
  i, n: Integer;
  c: Char;
begin
  offset := ErrLevel[lvl];
  if line > 0 then
  begin
    i := 1;
    n := Length(srcLine);
    while i < col - 1 do
    begin
      if (i <= n) and (srcLine[i] = #9) then
        c := #9
      else
        c := ' ';
      offset := offset + c;
      Inc(i)
    end;
    offset := offset + '^';
  end;
  Result := Format('%s %s', [offset, s]);
end;

{$EndRegion}

{$Region 'TErrors'}

constructor TErrors.Create(parser: TBaseParser);
begin
  inherited Create;
  FParser := parser;
  FErrors := TErrorList.Create;
  FInternalErrors := TErrorList.Create;
end;

destructor TErrors.Destroy;
begin
  FErrors.Free;
  FInternalErrors.Free;
  inherited;
end;

function TErrors.GetCount: Integer;
begin
  Result := FErrors.Count + FInternalErrors.Count;
end;

procedure TErrors.SynErr(nr, line, col: Integer);
var s: string;
begin
  s := FParser.ErrorMsg(nr);
  FErrors.Add(TErrDesc.Create(elSyntax, s, line, col));
end;

procedure TErrors.SemErr(const s: string; line, col: Integer);
begin
  FErrors.Add(TErrDesc.Create(elSemantic, s, line, col));
end;

procedure TErrors.SemErr(const s: string);
begin
  FInternalErrors.Add(TErrDesc.Create(elSemantic, s, 0, 0));
end;

procedure TErrors.Warning(const s: string; line, col: Integer);
begin
  FErrors.Add(TErrDesc.Create(elWarning, s, line, col));
end;

procedure TErrors.Warning(const s: string);
begin
  FInternalErrors.Add(TErrDesc.Create(elWarning, s, 0, 0));
end;

procedure TErrors.Print(const src: string; lst: TStrings);
var
  buf: TBuffer;
  Comparer: IComparer<TErrDesc>;
  err: TErrDesc;
  idx, lnr: Integer;
  line: string;

  function NextError: TErrDesc;
  begin
    if idx > FErrors.Count - 1 then
      Result := nil
    else
      Result := FErrors[idx];
    Inc(idx);
  end;

begin
  lst.Clear;
  buf := TBuffer.From(src);
  if FErrors.Count > 1 then
  begin
    Comparer := TDelegatedComparer<TErrDesc>.Create(
      function(const a, b: TErrDesc): Integer
      begin
        Result := a.line - b.line;
        if Result = 0 then
           Result := a.col - b.col;
      end);
    FErrors.Sort(Comparer);
  end;
  idx := 0;
  err := NextError;
  lnr := 1;
  while buf.GetLine(line) do
  begin
    lst.Add(Format('%5d  %s', [lnr, line]));
    while (err <> nil) and (err.line = lnr) do
    begin
      lst.Add(err.Print(line));
      err := NextError
    end;
    Inc(lnr);
  end;
  if err <> nil then
  begin
    lst.Add(Format('%5d', [lnr]));
    while err <> nil do
    begin
      err.Print(line);
      err := NextError
    end
  end;
  lst.Add('');
  for idx := 0 to FInternalErrors.Count - 1 do
    lst.Add(FInternalErrors[idx].Print(''));
  lst.Add('');
  line := Format('%5d error', [Count]);
  if Count <> 1 then
    line := line + 's';
  lst.Add(line);
end;

{$EndRegion}

{$Region 'TBaseParser'}

constructor TBaseParser.Create(scanner: TBaseScanner; listing: TStrings);
begin
  inherited Create;
  FScanner := scanner;
  FListing :=  listing;
  FErrors := TErrors.Create(Self);
  errDist := minErrDist;
  noString := '-none-';
end;

destructor TBaseParser.Destroy;
begin
  errors.Free;
  scanner.Free;
  inherited;
end;

procedure TBaseParser.SynErr(n: Integer);
begin
  if errDist >= minErrDist then
    errors.SynErr(n, la.line, la.col);
  errDist := 0;
end;

procedure TBaseParser.SemErr(const s: string);
begin
  if errDist >= minErrDist then
    errors.SemErr(s, t.line, t.col);
  errDist := 0;
end;

function TBaseParser.StartOf(s: Integer): Boolean;
begin
  Result := Starts(s, la.kind);
end;

procedure TBaseParser.Expect(n: Integer);
begin
  if la.kind = n then
    Get
  else
    SynErr(n);
end;

procedure TBaseParser.ExpectWeak(n, follow: Integer);
begin
  if la.kind = n then
    Get
  else
  begin
    SynErr(n);
    while not StartOf(follow) do Get;
  end;
end;

procedure TBaseParser.PrintErrors;
begin
  FListing.BeginUpdate;
  try
    Errors.Print(scanner.FBuffer.Buf, FListing);
  finally
    FListing.EndUpdate;
  end;
end;

function TBaseParser.WeakSeparator(n, syFol, repFol: Integer): Boolean;
var
  kind: Integer;
begin
  kind := la.kind;
  if kind = n then
  begin
    Get;
    Result := true;
  end
  else if StartOf(repFol) then
  begin
    Result := false;
  end
  else
  begin
    SynErr(n);
    while not (Starts(syFol, kind) or Starts(repFol, kind) or Starts(0, kind)) do
    begin
      Get;
      kind := la.kind;
    end;
    Result := StartOf(syFol);
  end;
end;

{$EndRegion}

end.

