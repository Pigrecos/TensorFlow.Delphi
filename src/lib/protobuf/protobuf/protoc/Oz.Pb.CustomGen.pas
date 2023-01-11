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
unit Oz.Pb.CustomGen;
interface
uses
  System.Classes, System.SysUtils, Generics.Collections,
  Oz.Cocor.Utils, Oz.Cocor.Lib, Oz.Pb.Tab, Oz.Pb.Classes, Oz.Pb.Gen;
{$Region 'TCustomGen: A base class for the code generator'}
type
  TCustomGen = class(TGen)
  protected
    FImportProcess : Boolean;
    FUnionProcess  : Boolean;
    sb: TStringBuilder;
    IndentLevel: Integer;
    maps: TMapTypes;
    mapvars: TMapTypes;
    pairMessage: TObjDesc;
    pairType: TTypeDesc;
    // Wrappers for TStringBuilder
    procedure Wr(const s: string); overload;
    procedure Wr(const f: string; const Args: array of const); overload;
    procedure Wrln; overload;
    procedure Wrln(const s: string); overload;
    procedure Wrln(const f: string; const Args: array of const); overload;
    // Indent control
    procedure Indent;
    procedure Dedent;
    function GetCode: string; override;
    // Enum code
    procedure EnumDecl(obj: PObj);
    // Map code
    procedure MapDecl(obj: PObj);
    // Field tag ident
    function FieldTag(obj: PObj): string;
    // Field tag declaration
    procedure FieldTagDecl(obj: PObj);
    // Union declaration
    procedure UnionDecl(u:PObj);
    // Field declaration
    procedure FieldDecl(obj: PObj);
    // field property
    procedure FieldProperty(obj: PObj);
    // Initialize field value
    procedure FieldInit(obj: PObj);
    // Free field
    procedure FieldFree(obj: PObj);
    // Get read statement
    function GetRead(obj: PObj;isUnion:Boolean=False): string;
    // Field read from buffer
    procedure FieldRead(obj: PObj);
    // Write field to buffer
    procedure FieldWrite(obj: PObj); virtual;
    // Field reflection
    procedure FieldReflection(obj: PObj);
    // unused
    procedure GenComment(const comment: string);
    // Get Union Field
    function GetWriteUnion(obj: PObj): string;
    // Message code
    function GetPair(maptypes: TMapTypes; typ: PType; mas: TGetMap): string;
    procedure MessageDecl(msg: PObj);
    procedure MessageImpl(msg: PObj);
    procedure LoadDecl(msg: PObj);
    procedure SaveDecl(msg: PObj);
    procedure LoadImpl(msg: PObj);
    procedure SaveImpl(msg: PObj);
    procedure SaveMaps;
    // Top level code
    procedure ModelDecl;
    procedure ModelImpl;
    procedure BuilderDecl(Load: Boolean);
    procedure BuilderImpl(Load: Boolean);
    function GetBuilderName(Load: Boolean): string;
    // abstract
    function TestNil: string; virtual; abstract;
    function RepeatedCollection: string; virtual; abstract;
    function AddItemMap(obj: PObj): string; virtual; abstract;
    function MapCollection: string; virtual; abstract;
    function CreateName: string; virtual; abstract;
    function CreateMapName: string; virtual; abstract;
    procedure GenUses; virtual; abstract;
    function GenPairStr:string; virtual; abstract;
    procedure GenDecl(Load: Boolean); virtual; abstract;
    procedure GenEntityType(msg: PObj); virtual; abstract;
    procedure GenEntityDecl; virtual; abstract;
    procedure GenEntityImpl(msg: PObj); virtual; abstract;
    procedure GenLoadDecl(msg: PObj); virtual; abstract;
    procedure GenSaveDecl(msg: PObj); virtual; abstract;
    procedure GenLoadImpl; virtual; abstract;
    procedure GenSaveProc; virtual; abstract;
    procedure GenInitLoaded; virtual;
    procedure GenLoadMethod(msg: PObj); virtual; abstract;
    function  GenRead(msg: PObj): string; virtual; abstract;
    procedure GenFieldRead(msg: PObj); virtual; abstract;
    procedure GenSaveImpl(msg: PObj); virtual; abstract;
  public
    constructor Create(Parser: TBaseParser);
    destructor Destroy; override;
    procedure GenerateCode; override;
  end;
{$EndRegion}
{$Region 'TFieldGen: Auxiliary class for the field generator'}
  TFieldGen = record
  var
    g: TCustomGen;
    obj: PObj;
    o: TFieldOptions;
    ft: string;
    m, mn, mt, n, t: string;
    checkNil: Boolean;
  private
    function GetTag: string;
    // Embedded types
    procedure GenType;
    procedure GenEnum;
    procedure GenMap(const pair: string);
    procedure GenMessage;
    procedure GenUnion;
  public
    procedure Init(g: TCustomGen; obj: PObj; o: TFieldOptions; const ft: string);
    procedure Gen;
  end;
{$EndRegion}
implementation
uses
  Oz.Pb.Parser;
{$Region 'TCustomGen'}
constructor TCustomGen.Create(Parser: TBaseParser);
begin
  inherited;
  FImportProcess := False;
  FUnionProcess  := False;
  sb := TStringBuilder.Create;
  maps := TList<PType>.Create;
  mapvars := TList<PType>.Create;
end;
destructor TCustomGen.Destroy;
begin
  mapvars.Free;
  maps.Free;
  sb.Free;
  inherited;
end;
function TCustomGen.GetCode: string;
begin
  Result := sb.ToString;
end;
procedure TCustomGen.GenerateCode;
var
  ns: string;
begin
  ns := 'ProtoGen.'+AsCamel(Tab.Module.Name);
  Wrln('unit %s;', [ns]);
  Wrln;
  Wrln('interface');
  Wrln;
  GenUses;
  Wrln('type');
  Wrln;
  Indent;

  // forword declaration
  //
  for var Item in tab.fwd_decl do
      Wrln('T'+Item.Key+' = class;');
  Wrln;

  try
    ModelDecl;
    BuilderDecl({Load=}True);
    BuilderDecl({Load=}False);
  finally
    Dedent;
  end;
  Wrln('implementation');
  Wrln;
  ModelImpl;
  BuilderImpl(True);
  BuilderImpl(False);
  Wrln('end.');
end;
procedure TCustomGen.GenInitLoaded;
begin
end;
procedure TCustomGen.ModelDecl;
var
  obj, x: PObj;
begin
  obj := tab.Module.Obj; // root proto file
  x := obj.dsc;
  while x <> nil do
  begin
    if x.cls = TMode.mType then
      case x.typ.form of
        TTypeMode.tmEnum: EnumDecl(x);
        TTypeMode.tmMessage: MessageDecl(x);
        TTypeMode.tmMap: MapDecl(x);
      end;
    x := x.next;
  end;
end;
procedure TCustomGen.ModelImpl;
var
  obj, x: PObj;
begin
  obj := tab.Module.Obj; // root proto file
  x := obj.dsc;
  while x <> tab.Guard do
  begin
    if x.cls = TMode.mType then
      if x.typ.form = TTypeMode.tmMessage then
        MessageImpl(x);
    x := x.next;
  end;
end;
procedure TCustomGen.EnumDecl(obj: PObj);
var
  x: PObj;
  n: Integer;
begin
  Wrln('%s = (', [obj.AsType]);
  x := obj.typ.dsc;
  while x <> tab.Guard do
  begin
    n := x.val.AsInt64;
    Wr('  %s = %d', [x.Name, n]);
    x := x.next;
    if x <> tab.Guard then
      Wr(',')
    else
      Wr(');');
    Wrln;
  end;
  Wrln;
end;
procedure TCustomGen.MapDecl(obj: PObj);
var
  x: PObj;
  key, value: PType;
begin
  x := obj.typ.dsc;
  key := gen.tab.UnknownType;
  value := gen.tab.UnknownType;
  while x <> tab.Guard do
  begin
    if x.name = 'key' then
      key := x.typ
    else if x.name = 'value' then
      value := x.typ;
    x := x.next;
  end;
  Wrln('%s = ' + MapCollection + ';',
    [obj.AsType, key.declaration.AsType, Value.declaration.AsType]);
  Wrln;
end;
procedure TCustomGen.MessageDecl(msg: PObj);
var
  x: PObj;
  typ: PType;
begin
  if tab.fwd_decl.ContainsKey(msg.name) then
    FImportProcess := True;

  // generate nested messages
  x := msg.dsc;
  while x <> tab.Guard do
  begin
    if x.cls = TMode.mType then
      case x.typ.form of
        TTypeMode.tmEnum: EnumDecl(x);
        TTypeMode.tmMessage: MessageDecl(x);
        TTypeMode.tmMap: MapDecl(x);
      end;
    x := x.next;
  end;
  GenEntityType(msg);
  // generate field tag definitions
  Wrln('const');
  Indent;
  typ := msg.typ;
  Assert(typ.form = TTypeMode.tmMessage);
  try
    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldTagDecl(x);
      x := x.next;
    end;
  finally
    Dedent;
  end;
  // generate field declarations
  Wrln('private');
  Indent;
  try
    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldDecl(x);
      x := x.next;
    end;
  finally
    Dedent;
  end;
  Wrln('public');
  Indent;
  try
    GenEntityDecl;
    Wrln('// properties');
    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldProperty(x);
      x := x.next;
    end;
  finally
    Dedent;
    FImportProcess := False;
  end;
  Wrln('end;'); // class
  Wrln;
end;
procedure TCustomGen.MessageImpl(msg: PObj);
var
  x: PObj;
begin
  if tab.fwd_decl.ContainsKey(msg.name) then
    FImportProcess := True;

  // generate nested messages
  x := msg.dsc;
  while x <> tab.Guard do
  begin
    if x.cls = TMode.mType then
      if x.typ.form = TTypeMode.tmMessage then
        MessageImpl(x);
    x := x.next;
  end;
  Wrln('{ %s }', [msg.AsType]);
  Wrln;
  GenEntityImpl(msg);
  Wrln('end;');
  Wrln;

  FImportProcess := False;
end;

procedure TCustomGen.LoadDecl(msg: PObj);
var
  typ: PType;
  x: PObj;
begin
  typ := msg.typ;
  if msg.cls <> TMode.mType then exit;
  if typ.form <> TTypeMode.tmMessage then exit;
  GenLoadDecl(msg);
  x := msg.dsc;
  while x <> tab.Guard do
  begin
    typ := x.typ;
    if (x.cls = TMode.mType) and (typ.form = TTypeMode.tmMessage) then
      LoadDecl(x);
    x := x.next;
  end;
end;
procedure TCustomGen.SaveDecl(msg: PObj);
var
  typ: PType;
  x: PObj;
begin
  typ := msg.typ;
  if msg.cls <> TMode.mType then exit;
  case typ.form of
    TTypeMode.tmMap:
      if maps.IndexOf(typ) < 0 then
        maps.Add(typ);
    TTypeMode.tmMessage:
      begin
        GenSaveDecl(msg);
        x := msg.dsc;
        while x <> tab.Guard do
        begin
          if x.cls = TMode.mType then
            SaveDecl(x);
          x := x.next;
        end;
      end;
  end;
end;

procedure TCustomGen.LoadImpl(msg: PObj);
var
  x: PObj;
  typ: PType;
  s, t: string;

  procedure ReadUnion(x: PObj);
  var
    o: TFieldOptions;
    w, f, tag: string;
  begin
      if x = nil then exit;
      Indent;
      while x <> tab.Guard do
      begin
          o := x.aux as TFieldOptions;
          w := TWire.Names[GetWireType(x.typ.form)];
          f := o.Msg.name;
          tag := t + '.' + FieldTag(x);
          Wrln('%s:', [tag]);
          Indent;
          Wrln('begin');
          Wrln('  Assert(wireType = TWire.%s);', [w]);
          Wrln('  var v : %s;', ['TpbOneof']);
          Wrln('  v.tag := %s;', [tag]);
          if x.typ.form = TTypeMode.tmMessage then
          begin
              Indent;
              Wrln('Pb.Push;');
              Wrln('try');
              Indent;
          end;

          Wrln('  v.value := %s;', [GetRead(x,True)]);

          Wrln('  Value.%s := v;', [f]);
          //Wrln('  %s.value := v;', [f]);
          if x.typ.form = TTypeMode.tmMessage then
          begin
              Dedent;
              Wrln('finally');
              Wrln(' Pb.Pop');
              Wrln('end;');
              Dedent;
          end;
          Wrln('end;');
          Dedent;
          x := x.next;
      end;
      Dedent;
  end;

begin
    // generate nested messages
    x := msg.dsc;
    while x <> nil do
    begin
        if x.cls = TMode.mType then
          if x.typ.form = TTypeMode.tmMessage then
            LoadImpl(x);
        x := x.next;
    end;
    typ := msg.typ;
    if msg.cls <> TMode.mType then exit;
    if typ.form <> TTypeMode.tmMessage then exit;
    s := msg.DelphiName;
    t := msg.AsType;
    GenLoadMethod(msg);
    Wrln('var');
    Wrln('  fieldNumber, wireType: integer;');
    Wrln('  tag: TpbTag;');
    Wrln('begin');
    Indent;
    Wrln('Value := %s.Create;', [t]);
    //GenInitLoaded;
    Wrln('tag := Pb.readTag;');
    Wrln('while tag.v <> 0 do');
    Wrln('begin');
    Indent;
    Wrln('wireType := tag.WireType;');
    Wrln('fieldNumber := tag.FieldNumber;');
    x := typ.dsc;
    if x = tab.Guard then
      // empty message
      Wrln('Pb.skipField(tag);')
    else
    begin
        Wrln('case fieldNumber of');
        while x <> tab.Guard do
        begin
            if x.typ.form = TTypeMode.tmUnion then
              ReadUnion(x.typ.dsc)
            else
              FieldRead(x);
            x := x.next;
        end;
        Wrln('  else');
        Wrln('    Pb.skipField(tag);');
        Wrln('end;');
    end;
    Wrln('tag := Pb.readTag;');
    Dedent;
    Wrln('end;');
    Dedent;
    Wrln('end;');
    Wrln('');
end;

function TCustomGen.GetWriteUnion(obj: PObj): string;
var
  msg: PObj;
  m, n: string;
  key, value: PObj;
begin
  msg := obj.typ.declaration;
  m := msg.DelphiName;
  n := obj.DelphiName;
  case obj.typ.form of
    TTypeMode.tmMessage:
      begin
          Result := GenRead(msg);
          Wrln('');
          Indent;
          Wrln('var v1 : T%s;', [m]);
          Wrln(result+'(v1);');
          Dedent;
          Result := Format('TValue.From<T%s>(v1)', [m]);
      end;
    TTypeMode.tmEnum:
      begin
          Result := Format('TValue.From<T%s>(T%s(S.Pb.writeInt32))', [m,m])
      end;
    TTypeMode.tmMap:
      begin
        key := obj.typ.dsc;
        value := key.next;
        Result := Format('%s, %s', [GetRead(key), GetRead(value)]);
      end;
    TTypeMode.tmBool:
      Result := 'S.Pb.writeBoolean';
    TTypeMode.tmInt64, TTypeMode.tmUint64:
      Result := 'S.Pb.writeInt64';
    else  begin
      Result := Format('S.Pb.write%s', [AsCamel(msg.name)]);

      if msg.name = 'bytes' then
        Result := Format('TValue.From<TBytes>(S.Pb.write%s)', [AsCamel(msg.name)])
    end;
  end;
end;

procedure TCustomGen.SaveImpl(msg: PObj);
var
  typ: PType;
  x: PObj;
  t : string;

  procedure WriteUnion(x: PObj);
  var
    o: TFieldOptions;
    f, tag: string;
  begin
      if x = nil then exit;
      FUnionProcess := True;
      Indent;
      while x <> tab.Guard do
      begin
          o := x.aux as TFieldOptions;
          f := o.Msg.name;
          t := msg.AsType;
          tag := t + '.' + FieldTag(x);
          Wrln('%s:', [tag]);
          Indent;
          Wrln('begin');
          Indent;
          FieldWrite(x);
          Dedent;
          Wrln('end;');
          Dedent;
          x := x.next;
      end;
      Dedent;
      Wrln('end;');
      FUnionProcess := False;
  end;

  function HasRepeatedVars(x: PObj): Boolean;
  begin
      while x <> tab.Guard do
      begin
          if TFieldOptions(x.aux).Rule = TFieldRule.Repeated then
            exit(True);
          x := x.next;
      end;
      Result := False;
  end;

  procedure SaveMessage;
  var
    x: PObj;
    typ: PType;
  begin
      GenSaveImpl(msg);
      mapvars.Clear;
      x := msg.dsc;
      while x <> tab.Guard do
      begin
          if x.cls = TMode.mType then
          begin
              typ := x.typ;
              if (typ.form = TTypeMode.tmMap) and (mapvars.IndexOf(typ) < 0) then
                mapvars.Add(typ);
          end;
          x := x.next;
      end;
      Wrln('var ');
      Indent;
      Wrln('i : Integer;');
      Wrln('h : TpbSaver;');
      Dedent;
      Wrln('');

      Wrln('begin');
      Indent;
      try
          typ := msg.typ;
          x := typ.dsc;
          while x <> tab.Guard do
          begin
              if x.typ.form = TTypeMode.tmUnion then
              begin
                 Wrln('case Value.%s.tag of',[x.name]);
                 WriteUnion(x.typ.dsc) ;
              end
              else
                 FieldWrite(x);
              x := x.next;
          end;
      finally
        Dedent;
      end;
      Wrln('end;');
      Wrln('');
  end;

begin
    // generate nested messages
    x := msg.dsc;
    while x <> nil {tab.Guard} do
    begin
        typ := x.typ;
        if (x.cls = TMode.mType) and (typ.form = TTypeMode.tmMessage) then
          SaveImpl(x);
        x := x.next;
    end;
    typ := msg.typ;
    if (msg.cls = TMode.mType) and (typ.form = TTypeMode.tmMessage) then
      SaveMessage;
end;

procedure TFieldGen.Init(g: TCustomGen; obj: PObj; o: TFieldOptions; const ft: string);
begin
  Self.g := g;
  Self.obj := obj;
  Self.m := AsCamel(obj.typ.declaration.name);
  Self.o := o;
  Self.ft := ft;
  Self.checkNil := True;
  mn := o.Msg.DelphiName;
  mt := o.Msg.AsType;
  n := obj.DelphiName;
  t := obj.AsType;
end;
procedure TFieldGen.Gen;
begin
  if o.Default <> '' then
  begin
    g.Wrln('if %s.F%s <> %s then', [m, t, o.Default]);
    g.Indent;
  end;
  case obj.typ.form of
    TTypeMode.tmDouble .. TTypeMode.tmSint64:
      GenType;
    TTypeMode.tmEnum:
      GenEnum;
    TTypeMode.tmMessage:
      GenMessage;
    TTypeMode.tmMap:
      GenMap(g.GetPair(g.mapvars, obj.typ, TGetMap.asVarUsing));
    TTypeMode.tmUnion:
      GenUnion;
    else
      raise Exception.Create('unsupported field type');
  end;
  if o.Default <> '' then
    g.Dedent;
end;
function TFieldGen.GetTag: string;
begin
  if (ft = '1') or (ft = '2') then
    Result := ft
  else
    Result := mt + '.' + ft;
end;
procedure TFieldGen.GenType;
var
  m: string;
  FFieldPrefix : string;
begin
  if g.FImportProcess then  FFieldPrefix := ''
  else                      FFieldPrefix := 'F' ;

  // Pb.writeString(TPerson.ftName, Person.Name);
  if o.rule <> TFieldRule.Repeated then
  begin
    m := DelphiRwMethods[obj.typ.form];
    if g.FUnionProcess then
        g.Wrln('S.Pb.write%s(Value.%s, Value.%s.value.AsType<%s>);', [AsCamel(m), ft{GetTag}, mn,t])
    else
        g.Wrln('S.Pb.write%s(%s, Value.%s);', [AsCamel(m), GetTag, n]);
  end
  else
  begin
    g.Wrln('h.Init;');
    g.Wrln('try');
    n := Plural(n);
    if TObjDesc.Keywords.IndexOf(n) >= 0 then
    begin
        if FFieldPrefix = '' then
           n := '&'+ n;
    end;

    g.Wrln('  for i := 0 to Value.%s.Count - 1 do', [n]);
    case obj.typ.form of
      TTypeMode.tmInt32, TTypeMode.tmUint32, TTypeMode.tmSint32,
      TTypeMode.tmBool, TTypeMode.tmEnum:
        begin
            if obj.typ.form = TTypeMode.tmBool then
               g.Wrln('    h.Pb.writeRawVarint32(Integer(Value.%s%s[i]));', [FFieldPrefix, n])
            else
               g.Wrln('    h.Pb.writeRawVarint32(Value.%s%s[i]);', [FFieldPrefix, n]);
        end;
      TTypeMode.tmInt64, TTypeMode.tmUint64, TTypeMode.tmSint64:
        g.Wrln('    h.Pb.writeRawVarint64(Value.%s%s[i]);', [FFieldPrefix, n]);
      TTypeMode.tmFixed64, TTypeMode.tmSfixed64, TTypeMode.tmDouble,
      TTypeMode.tmSfixed32, TTypeMode.tmFixed32, TTypeMode.tmFloat:
        begin
            g.Wrln(' begin');
            g.Wrln('    var vVar : %s := Value.%s%s[i];', [t, FFieldPrefix, n]);
            g.Wrln('    h.Pb.writeRawData(@vVar, sizeof(%s));', [t]);
            g.Wrln(' end;');
        end;
      TTypeMode.tmString:
        g.Wrln('    h.Pb.writeRawString(Value.%s%s[i]);', [FFieldPrefix, n]);
      TTypeMode.tmBytes:
        g.Wrln('    h.Pb.writeRawBytes(Value.%s%s[i]);', [FFieldPrefix,n]);
    end;
    g.Wrln('  S.Pb.writeMessage(%s, h.Pb^);', [GetTag]);
    g.Wrln('finally');
    g.Wrln('  h.Free;');
    g.Wrln('end;');
  end;
end;
procedure TFieldGen.GenEnum;
begin
  if o.rule <> TFieldRule.Repeated then
    if g.FUnionProcess then
        g.Wrln('S.Pb.writeInt32(%s, Ord(Value.%s.value.AsType<%s>));', [GetTag, mn,t])
    else
        g.Wrln('S.Pb.writeInt32(%s, Ord(Value.%s));', [GetTag, AsCamel(n)])
  else
  begin
    g.Wrln('h.Init;');
    g.Wrln('try');
    n := Plural(n);
    g.Wrln('  for i := 0 to Value.%s.Count - 1 do', [n]);
    g.Wrln('    h.Pb.writeRawVarint32(Ord(Value.%s[i]));', [n]);
    g.Wrln('  S.Pb.writeMessage(%s, h.Pb^);', [GetTag]);
    g.Wrln('finally');
    g.Wrln('  h.Free;');
    g.Wrln('end;');
  end;
end;
procedure TFieldGen.GenMessage;
var
  FFieldPrefix : string;
begin
  if g.FImportProcess then  FFieldPrefix := ''
  else                      FFieldPrefix := 'F' ;

  if o.rule <> TFieldRule.Repeated then
  begin
    if (ft <> '2') and checkNil then
    begin
      if g.FUnionProcess then
         g.Wrln('if Value.%s.value.AsType<%s> <> nil then', [mn,t])
      else
         g.Wrln('if Value.%s%s <> nil then', [FFieldPrefix,n]);
      g.Indent;
    end;
    if g.FUnionProcess then
        g.Wrln('S.SaveObj<%s>(Value.%s.value.AsType<%s>, Save%s, %s);', [t, mn, t, m, GetTag])
    else
        g.Wrln('S.SaveObj<%s>(Value.%s%s, Save%s, %s);', [t, FFieldPrefix,n, m, GetTag]);
    if (ft <> '2') and checkNil then
      g.Dedent;
  end
  else
  begin
    n := AsCamel(Plural(n));
    g.Wrln('if Value.%s%s.Count > 0 then', [FFieldPrefix,n]);
    g.Wrln('  S.SaveList<%s>(Value.%s%s, Save%s, %s);', [t, FFieldPrefix,n, m, GetTag]);
  end;
end;
procedure TFieldGen.GenMap(const pair: string);
var
  s            : string;
  FFieldPrefix : string;
begin
  if g.FImportProcess then  FFieldPrefix := ''
  else                      FFieldPrefix := 'F' ;

  if o.rule <> TFieldRule.Repeated then
  begin
    if ft <> '2' then
    begin
      g.Wrln('if Value.F%s%s then', [n,g.TestNil]);
      g.Wrln('begin');
      g.Indent;
    end;
    g.Wrln('h.Init;');
    g.Wrln('try');
    g.Wrln('  for var it in Value.%s%s do',[FFieldPrefix,n]);
    g.Wrln('  begin');
    g.Wrln('      h.clear;');
    g.Wrln('      h.Save%s(it);', [m]);
    s := Format('      S.Pb.writeMessage(%s, h.Pb^);', [GetTag]);
    g.Wrln(s);
    g.Wrln('  end;');
    g.Wrln('finally');
    g.Wrln('  h.Free;');
    g.Wrln('end;');
    if ft <> '2' then
    begin
      g.Dedent;
      g.Wrln('end;');
    end;
  end
  else
  begin
    g.Wrln('h.Init;');
    g.Wrln('try');
    g.Wrln('  for %s in %s.F%s do', [pair, mn, n]);
    g.Wrln('  begin');
    g.Wrln('    h.Clear;');
    g.Wrln('    h.Save%s(Value);', [AsCamel(m)]);
    g.Wrln('    S.Pb.writeMessage(%s, h.Pb^);', [GetTag]);
    g.Wrln('  end;');
    g.Wrln('finally');
    g.Wrln('  h.Free;');
    g.Wrln('end;');
  end;
end;
procedure TFieldGen.GenUnion;
var
  fg: TFieldGen;
  x: PObj;
begin
  x := obj.typ.dsc;
  if x = nil then exit;
  while x <> g.tab.Guard do
  begin
    fg.Init(g, x, x.aux as TFieldOptions, g.FieldTag(x));
    fg.Gen;
    x := x.next;
  end;
end;
procedure TCustomGen.SaveMaps;
var
  typ: PType;
  map, key, value: PObj;
  s, t: string;
  ko, vo: TFieldOptions;
  procedure G(obj: PObj; o: TFieldOptions; const ft: string);
  var fg: TFieldGen;
  begin
    fg.Init(Self, obj, o, ft);
    fg.Gen;
  end;
begin
  pairMessage.cls := TMode.mType;
  pairMessage.name := 'Value';
  pairMessage.typ := @pairType;
  pairType.form := TTypeMode.tmMessage;
  pairType.declaration := @pairMessage;
  for typ in maps do
  begin
    map := typ.declaration;
    s := map.DelphiName;
    t := map.AsType;
    Wrln('procedure %s.Save%s(%s);', [GetBuilderName(False), map.DelphiName, GetPair(maps, typ, TGetMap.asParam)]);
    Wrln('var');
    Wrln('  S: TpbSaver;');
    Wrln('begin');
    Indent;
    try
      key := typ.dsc;
      ko := TFieldOptions.Create(key, @pairMessage, 1, TFieldRule.Singular);
      G(key, ko, '1');
      value := key.next;
      vo := TFieldOptions.Create(value, @pairMessage, 2, TFieldRule.Singular);
      G(value, vo, '2');
    finally
      Dedent;
    end;
    Wrln('end;');
    Wrln('');
  end;
end;
function TCustomGen.FieldTag(obj: PObj): string;
var
  n: string;
  o: TFieldOptions;
begin
(*
   ftId | ftPhones
*)
  o := obj.aux as TFieldOptions;
  n := AsCamel(obj.name);
  if o.Rule = TFieldRule.Repeated then
    n := Plural(n);
  Result := 'ft' + n;
end;

procedure TCustomGen.UnionDecl(u:PObj);
var

  x: PObj;
begin
  x := u.typ.dsc;
  if x = nil then exit;
  while x <> tab.Guard do
  begin
    var o := x.aux as TFieldOptions;

    Wrln('%s = %d;', [FieldTag(x), o.Tag]);
    x := x.next;
  end;
end;

procedure TCustomGen.FieldTagDecl(obj: PObj);
var o: TFieldOptions;
begin
(*
   ftId = 1;
*)
  o := obj.aux as TFieldOptions;

  if obj.typ.form = TTypeMode.tmUnion then
     UnionDecl(obj)
  else
     Wrln('%s = %d;', [FieldTag(obj), o.Tag]);
end;
procedure TCustomGen.FieldDecl(obj: PObj);
var
  n, t: string;
  o: TFieldOptions;
begin
(*
   FId: Integer;
*)
  o := obj.aux as TFieldOptions;
  n := obj.AsField;
  if obj.typ.form = TTypeMode.tmUnion then
    t := 'TpbOneof'
  else
    t := obj.AsType;
  if o.Rule = TFieldRule.Repeated then
  begin
    n := Plural(n);
    t := Format(RepeatedCollection, [t]);
  end;
  Wrln('%s: %s;', [n, t]);
end;
procedure TCustomGen.FieldProperty(obj: PObj);
var
  n, f, t, s: string;
  ro: Boolean;
  o: TFieldOptions;
begin
(*
  // here can be field comment
  Id: Integer read FId write FId;
*)
  o := obj.aux as TFieldOptions;
  ro := o.ReadOnly;
  n := obj.DelphiName;
  f := obj.AsField;
  if obj.typ.form = TTypeMode.tmUnion then
    t := 'TpbOneof'
  else
    t := obj.AsType;

  if o.Rule = TFieldRule.Repeated then
  begin
    ro := True;
    n := Plural(obj.name);
    t := Format(RepeatedCollection, [t]);
    f := 'F' + n;
    if TObjDesc.Keywords.IndexOf(f) >= 0 then
      f := '&' + f;
    if TObjDesc.Keywords.IndexOf(n) >= 0 then
      n := '&' + n;
  end;
  s := Format('property %s: %s read %s', [n, t, f]);
  if ro then
    s := s + ';'
  else
    s := s + Format(' write %s;', [f]);
  Wrln(s);
end;
procedure TCustomGen.FieldInit(obj: PObj);
var
  f, t, coll: string;
  o: TFieldOptions;
  key, value: PObj;
begin
  o := obj.aux as TFieldOptions;
  f := obj.AsField;
  if o.Default <> '' then
    Wrln('%s := %s;', [f, o.Default])
  else if o.Rule = TFieldRule.Repeated then
  begin
    t := obj.AsType;
    coll := Format(RepeatedCollection, [t]);
    Wrln('');
    //Wrln('m := System.Default(TsgItemMeta);');
    //Wrln('m.Init<%s>;',[t]);
    Wrln('%s := %s.%s;', [Plural(f), coll, CreateName])
  end
  else if obj.typ.form = TTypeMode.tmMap then
  begin
    key := obj.typ.dsc;
    value := key.next;
    coll := Format(MapCollection, [key.AsType, value.AsType]);
    Wrln('%s := %s.%s;', [f, coll, CreateMapName]);
  end;
end;
procedure TCustomGen.FieldFree(obj: PObj);
var
  f: string;
  o: TFieldOptions;
begin
  o := obj.aux as TFieldOptions;
  f := obj.AsField;
  if o.Rule = TFieldRule.Repeated then
    Wrln('%s.Free;', [Plural(f)])
  else if obj.typ.form = TTypeMode.tmMap then
    Wrln('%s.Free;', [f])
end;

function TCustomGen.GetRead(obj: PObj;isUnion:Boolean): string;
var
  msg: PObj;
  m, n: string;
  key, value: PObj;
begin
  msg := obj.typ.declaration;
  m := msg.DelphiName;
  n := obj.DelphiName;
  case obj.typ.form of
    TTypeMode.tmMessage:
      begin
         Result := GenRead(msg);
         if isUnion then
         begin
            Wrln('');
            Indent;
            Wrln('var v1 : T%s;', [m]);
            Wrln(result+'(v1);');
            Dedent;
            Result := Format('TValue.From<T%s>(v1)', [m]);
         end;
      end;
    TTypeMode.tmEnum:
      begin
         Result := Format('T%s(Pb.readInt32)', [m]);
         if isUnion then
            Result := Format('TValue.From<T%s>(T%s(Pb.readInt32))', [m,m])
      end;
    TTypeMode.tmMap:
      begin
        key   := obj.typ.dsc;
        value := key.next;
        Result := Format('%s, %s', [GetRead(key), GetRead(value)]);
        if value.typ.form =  TTypeMode.tmMessage then
        begin
            Wrln('var v1 : T%s;', [value.typ.declaration.name]);
            Wrln('%s(v1);',[GetRead(value)]);
            Result := Format('%s, v1', [GetRead(key)]);
        end;

      end;
    TTypeMode.tmBool:
      Result := 'Pb.readBoolean';
    TTypeMode.tmInt64, TTypeMode.tmUint64:
      Result := 'Pb.readInt64';
    else  begin
      Result := Format('Pb.read%s', [AsCamel(msg.name)]);

      if isUnion then
        if msg.name = 'bytes' then
           Result := Format('TValue.From<TBytes>(Pb.read%s)', [AsCamel(msg.name)])
    end;
  end;
end;
procedure TCustomGen.FieldRead(obj: PObj);
var
  o              : TFieldOptions;
  msg            : PObj;
  w, mn, mt, m, n: string;
  procedure GenType;
  begin
      var tType := msg.typ.declaration.name.Replace('float','single');
      tType := tType.Replace('bytes','TBytes');
      tType := tType.Replace('bool','boolean');
      tType := tType.Replace('DataType','Integer');
      if o.Rule <> TFieldRule.Repeated then
      begin
          Wrln('begin');
          Wrln('  Assert(wireType = TWire.%s);', [w]);
          Wrln('  Value.%s := %s;', [n, GetRead(obj)]);
          Wrln('end;');
      end  else
      begin
          Wrln('begin');
          Indent;
          Wrln('  var vTipo : '+ tType +';');
          if tType = 'TBytes' then
              Wrln('  if IsPackedRepeatedField(tag, TValue.From<TBytes>(vTipo)) then')
          else
              Wrln('  if IsPackedRepeatedField(tag, vTipo) then');
          Wrln('  begin');
          Indent;
          Wrln('  Pb.Push;');
          Wrln('  try');
          Wrln('    while not Pb.Eof do');
          Wrln('    begin');
          Indent;
          Indent;
          Indent;
          GenFieldRead(obj);
          Dedent;
          Dedent;
          Dedent;
          Wrln('    end');
          Wrln('  finally');
          Wrln('    Pb.Pop;');
          Wrln('  end;');
          Dedent;
          Wrln('  end');
          Wrln('  else begin');
          Indent;
          Wrln('  repeat');
          Indent;
          Indent;
          GenFieldRead(obj);
          Dedent;
          Dedent;
          Wrln('  until not Pb.ConsumeTag(tag.v);');
          Dedent;
          Wrln('  end;');
          Dedent;
          Wrln('end;');
      end;
  end;
  procedure GenEnum;
  begin
    var tType := msg.typ.declaration.name.Replace('float','single');
    tType := tType.Replace('bytes','TBytes');
    tType := tType.Replace('bool','boolean');
    tType := tType.Replace('DataType','Integer');
    if o.Rule <> TFieldRule.Repeated then
    begin
        Wrln('begin');
        Wrln('  Assert(wireType = TWire.%s);', [w]);
        Wrln('  Value.%s := %s;', [n, GetRead(obj)]);
        Wrln('end;');
    end else
    begin
        Wrln('begin');
        Indent;
        Wrln('  var vTipo : '+ tType +';');
        if tType = 'TBytes' then
              Wrln('  if IsPackedRepeatedField(tag, TValue.From<TBytes>(vTipo)) then')
        else
              Wrln('  if IsPackedRepeatedField(tag, vTipo) then');
        Wrln('  begin');
        Indent;
        Wrln('  Pb.Push;');
        Wrln('  try');
        Wrln('    while not Pb.Eof do');
        Wrln('    begin');
        Indent;
        Indent;
        Indent;
        GenFieldRead(obj);
        Dedent;
        Dedent;
        Dedent;
        Wrln('    end');
        Wrln('  finally');
        Wrln('    Pb.Pop;');
        Wrln('  end;');
        Dedent;
        Wrln('  end');
        Wrln('  else begin');
        Indent;
        Wrln('  repeat');
        Indent;
        Indent;
        GenFieldRead(obj);
        Dedent;
        Dedent;
        Wrln('  until not Pb.ConsumeTag(tag.v);');
        Dedent;
        Wrln('  end;');
        Dedent;
        Wrln('end;');
      end;
  end;
  procedure GenMessage;
  begin
      Wrln('begin');
      Indent;
      Wrln('Assert(wireType = TWire.LENGTH_DELIMITED);');
      GenFieldRead(obj);
      Dedent;
      Wrln('end;');
  end;
  procedure GenMap;
  begin
       Wrln('begin');
       Indent;
       Wrln('%s.%s.%s(%s);', ['Value', n, AddItemMap(obj),GetRead(obj)]);
       Dedent;
       Wrln('end;');
  end;
begin
  o := obj.aux as TFieldOptions;
  mn := o.Msg.DelphiName;
  mt := o.Msg.AsType;
  msg := obj.typ.declaration;
  w := TWire.Names[GetWireType(obj.typ.form)];
  m := msg.DelphiName;
  n := obj.DelphiName;
  Indent;
  try
    Wrln('%s.%s:', [mt, FieldTag(obj)]);
    Indent;
    try
      case obj.typ.form of
        TTypeMode.tmMessage: GenMessage;
        TTypeMode.tmEnum: GenEnum;
        TTypeMode.tmMap: GenMap;
        else GenType;
      end;
    finally
      Dedent;
    end;
  finally
    Dedent;
  end;
end;
procedure TCustomGen.FieldWrite(obj: PObj);
var
  fg: TFieldGen;
begin
  if obj.cls <> TMode.mField then exit;
  fg.Init(Self, obj, obj.aux as TFieldOptions, FieldTag(obj));
  fg.Gen;
end;
procedure TCustomGen.FieldReflection(obj: PObj);
begin
  raise Exception.Create('under consruction');
end;
procedure TCustomGen.GenComment(const comment: string);
var
  s: string;
begin
  for s in comment.Split([#13#10], TStringSplitOptions.None) do
    Wrln('// ' + s)
end;
function TCustomGen.GetPair(maptypes: TMapTypes; typ: PType; mas: TGetMap): string;
var
  msg, key, value: PObj;
  i: Integer;
  s: string;
begin
  msg := typ.declaration;
  Assert((msg.cls = TMode.mType) and (typ.form = TTypeMode.tmMap));
  i := maptypes.IndexOf(typ);
  if (i > 0) and (mas = TGetMap.asVarUsing) then
    s := Format('Item%d', [i])
  else
    s := 'Value';
  if mas = TGetMap.asVarUsing then exit(s);
  key := typ.dsc;
  value := key.next;
  Result := Format('%s: %s<%s, %s>', [s, GenPairStr, key.AsType, value.AsType]);
end;
function TCustomGen.GetBuilderName(Load: Boolean): string;
begin
  if Load then
    Result := 'TLoadHelper'
  else
    Result := 'TSaveHelper';
end;
procedure TCustomGen.BuilderDecl(Load: Boolean);
const
  Names: array [Boolean] of string = ('TpbSaver', 'TpbLoader');
var
  obj, x, m,importType: PObj;
  typ: PType;
  s: string;
  Done : TList<String>;
begin
  Done := TList<String>.Create;
  Wrln('%s = record helper for %s', [GetBuilderName(Load), Names[Load]]);
  GenDecl(Load);
  Wrln('public');
  Indent;
  try
    maps.Clear;
    obj := tab.Module.Obj; // root proto file
    x := obj.dsc;
    while x <> nil do
    begin
        if x.cls = TMode.mType then
        begin
            if   Load then  LoadDecl(x)
            else            SaveDecl(x);
        end;
        x := x.next;
    end;

    FImportProcess := True;
    importType := tab.Module.Import;
    while (importType <> nil) and (importType <> tab.Guard)  do
    begin
        x := importType.dsc;
        while x <> nil do
        begin
            if not Done.Contains(x.name) then
            begin
                if x.cls = TMode.mType then
                begin
                    if   Load then LoadDecl(x)
                    else           SaveDecl(x);
                end;
                Done.Add(x.name) ;
            end;
            x := x.next;
        end;
        importType := importType.next;
    end;

    if not Load then
    begin
        for typ in maps do
        begin
            m := typ.declaration;
            s := GetPair(maps, typ, TGetMap.asParam);
            Wrln('procedure Save%s(%s);', [m.DelphiName, s]);
        end;
    end;
  finally
     Dedent;
     FImportProcess := False;
     Done.Free;
  end;
  Wrln('end;');
  Wrln;
end;
procedure TCustomGen.BuilderImpl;
var
  obj, x, importType: PObj;
  Done : TList<String>;
begin
  FImportProcess := False;

  Done := TList<String>.Create;

  if Load then  GenLoadImpl
  else          GenSaveProc;

  obj := tab.Module.Obj; // root proto file
  x := obj.dsc;
  try
    while x <> nil  do
    begin
        if x.cls = TMode.mType then
        begin
          if   Load then  LoadImpl(x)
          else            SaveImpl(x);
        end;
        x := x.next;
    end;

    FImportProcess := True;
    importType := tab.Module.Import;
    while (importType <> nil) and (importType <> tab.Guard)  do
    begin
        x := importType.dsc;
        while x <> nil do
        begin
            if not Done.Contains(x.name) then
            begin
                if x.cls = TMode.mType then
                begin
                    if   Load then  LoadImpl(x)
                    else            SaveImpl(x);
                end;
                Done.Add(x.name) ;
            end;
            x := x.next;
        end;
        importType := importType.next;
    end;

    if not Load then
      SaveMaps;

  finally
    Done.Free;
    FImportProcess := False;
  end;
end;
procedure TCustomGen.Wr(const s: string);
begin
  sb.Append(s);
end;
procedure TCustomGen.Wr(const f: string; const Args: array of const);
begin
  sb.AppendFormat(Blank(IndentLevel * 2) + f, Args);
end;
procedure TCustomGen.Wrln;
begin
  sb.AppendLine;
end;
procedure TCustomGen.Wrln(const s: string);
begin
  sb.AppendLine(Blank(IndentLevel * 2) + s);
end;
procedure TCustomGen.Wrln(const f: string; const Args: array of const);
begin
  if System.SysUtils.Format(f, Args) = 'TTensorProto.ftIntVals:' then
     var g := f;
  sb.AppendFormat(Blank(IndentLevel * 2) + f, Args);
  sb.AppendLine;
end;
procedure TCustomGen.Indent;
begin
  Inc(IndentLevel);
end;
procedure TCustomGen.Dedent;
begin
  Dec(IndentLevel);
  if IndentLevel < 0 then
    IndentLevel := 0;
end;
{$EndRegion}
end.
