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
unit Oz.Pb.GenSGL;
interface
uses
  System.SysUtils, System.Generics.Collections, Oz.Cocor.Utils, Oz.Pb.Tab, Oz.Pb.CustomGen;
{$Region 'TGenSGL: code generator for Oz.SGL.Collections'}
type
  TGenSGL = class(TCustomGen)
  protected
    function  TestNil: string; override;
    procedure FieldWrite(obj: PObj); override;
    function  AddItemMap(obj: PObj): string; override;
    function  MapCollection: string; override;
    function  RepeatedCollection: string; override;
    function  CreateName: string; override;
    function  CreateMapName: string; override;
    procedure GenEntityType(msg: PObj); override;
    procedure GenUses; override;
    procedure GenDecl(Load: Boolean); override;
    procedure GenEntityDecl; override;
    procedure GenEntityImpl(msg: PObj); override;
    procedure GenLoadDecl(msg: PObj); override;
    procedure GenSaveDecl(msg: PObj); override;
    procedure GenLoadImpl; override;
    procedure GenSaveProc; override;
    procedure GenInitLoaded; override;
    procedure GenLoadMethod(msg: PObj); override;
    function  GenRead(msg: PObj): string; override;
    procedure GenFieldRead(msg: PObj); override;
    procedure GenSaveImpl(msg: PObj); override;
    function  GenPairStr:string; override;
  end;
{$EndRegion}
implementation
uses
  Oz.Pb.Parser;
{$Region 'TGenSGL'}
function TGenSGL.AddItemMap(obj: PObj): string;
begin
    var key := obj.typ.dsc;
    var value := key.next;

    Result := Format('InsertOrAssign(TsgPair<%s, %s>.From',[key.AsType, value.AsType]);
end;
function TGenSGL.MapCollection: string;
begin
  Result := 'TsgHashMap<%s, %s>';
end;
function TGenSGL.RepeatedCollection: string;
begin
  Result := 'TsgRecordList<%s>';
end;
function TGenSGL.TestNil: string;
begin
    Result := '.Count > 0'
end;

function TGenSGL.CreateName: string;
begin
  Result := 'From(@m)';
end;
function TGenSGL.CreateMapName: string;
begin
  Result := 'From(0,nil)';
end;
procedure TGenSGL.GenEntityType(msg: PObj);
var
  s: string;
begin
  s := AsCamel(msg.typ.declaration.name);
  if not FImportProcess then
  begin
     Wrln('P%s = ^T%s;', [s, s]);
     Wrln('T%s = record', [s]);
  end
  else
     Wrln('T%s = Class', [s])
end;
procedure TGenSGL.GenUses;
var
  x : PObj;
  Done : TList<String>;
begin

  Done := TList<String>.Create;
  try
    Wrln('uses');
    if tab.Module.Import <> nil then
    begin
        Wrln('  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,Oz.SGL.Heap,');
        x := tab.Module.Import;
        while x <> tab.Guard do
        begin
            if not Done.Contains(x.name) then
            begin
                var sUserUnit : string := '  ProtoGen.'+x.name.Replace('_','');
                if x.next <> tab.Guard then  sUserUnit := sUserUnit + ','
                else                         sUserUnit := sUserUnit + ';' ;
                Wrln(sUserUnit);

                Done.Add(x.name) ;
            end;

            x := x.next;
        end;
    end else
    begin
        Wrln('  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,Oz.SGL.Heap;');
    end;
    Wrln;
    Wrln('{$T+}');
    Wrln;
  finally
    Done.Free;
  end;
end;
procedure TGenSGL.GenDecl(Load: Boolean);
begin
  if not Load then
  begin
    Wrln('type');
    Wrln('  TSave<T> = procedure(const S: TpbSaver; const Value: T);');
    Wrln('  TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);');
    Wrln('private');
    Wrln('  procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);');
    Wrln('  procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);');
    Wrln('  procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>;');
    Wrln('    Save: TSavePair<Key, Value>; Tag: Integer);');
  end;
end;
procedure TGenSGL.FieldWrite(obj: PObj);
var
  fg: TFieldGen;
begin
  if obj.cls <> TMode.mField then exit;
  fg.Init(Self, obj, obj.aux as TFieldOptions, FieldTag(obj));
  fg.checkNil := False;
  fg.Gen;
end;
procedure TGenSGL.GenEntityDecl;
begin
  if not FImportProcess then
  begin
      Wrln('procedure Init;');
      Wrln('procedure Free;');
  end else
  begin
      Wrln('Constructor Init;');
      Wrln('destructor Free;');
  end;
end;
procedure TGenSGL.GenEntityImpl(msg: PObj);
var
  typ: PType;
  t: string;
  x: PObj;
begin
  typ := msg.typ;
  // parameterless Init;
  t := msg.AsType;
  if not FImportProcess then
    Wrln('procedure %s.Init;', [t])
  else
    Wrln('Constructor %s.Init;', [t]);
  Wrln('begin');
  Indent;
  try
    if not FImportProcess then
    begin
      Wrln('Self := Default(%s);', [t]);
      Wrln('var m : TsgItemMeta;');
    end else
    begin
      Wrln('inherited Create;');
      Wrln('var m : TsgItemMeta;');
    end;

    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldInit(x);
      x := x.next;
    end;
  finally
    Dedent;
  end;
  Wrln('end;');
  Wrln;
  if not FImportProcess then
     Wrln('procedure %s.Free;', [t])
  else
     Wrln('destructor %s.Free;', [t]);

  Wrln('begin');
  Indent;
  try
    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldFree(x);
      x := x.next;
    end;
    if FImportProcess then
      Wrln('inherited Destroy;');
  finally
    Dedent;
  end;
end;
procedure TGenSGL.GenLoadDecl(msg: PObj);
begin
  Wrln('procedure Load%s(var Value: %s);', [msg.DelphiName, msg.AsType]);
end;
procedure TGenSGL.GenSaveDecl(msg: PObj);
begin
  Wrln('class procedure Save%s(const S: TpbSaver; const Value: %s); static;',
    [msg.DelphiName, msg.AsType]);
end;
procedure TGenSGL.GenLoadMethod(msg: PObj);
var
  s, t: string;
begin
  s := msg.DelphiName;
  t := msg.AsType;
  Wrln('procedure %s.Load%s(var Value: %s);', [GetBuilderName(True), s, t]);
end;

function TGenSGL.GenPairStr: string;
begin
    Result := 'TsgPair' ;
end;

function TGenSGL.GenRead(msg: PObj): string;
begin
  Result := Format('Load%s', [msg.DelphiName]);
end;
procedure TGenSGL.GenFieldRead(msg: PObj);
var
  o: TFieldOptions;
  n: string;
  m,e: Boolean;
begin

  m := msg.typ.form = TTypeMode.tmMessage;
  e := msg.typ.form = TTypeMode.tmEnum;
  if m then
  begin
    Wrln('Pb.Push;');
    Wrln('try');
    Indent;
  end;
  o := msg.aux as TFieldOptions;
  if FImportProcess then n := AsCamel(msg.name)
  else                   n := 'F' + AsCamel(msg.name);
  if o.Rule <> TFieldRule.Repeated then
  begin
    if msg.typ.form <> TTypeMode.tmMessage then
       Wrln('%s(Value.%s);', [GetRead(msg), n])
    else begin
        var tType := 'T'+msg.typ.declaration.name;
        Wrln('var v : %s := Value.%s;', [tType,n]);
        Wrln('%s(v);', [GetRead(msg)]);
        Wrln('Value.%s := v;', [n]);
    end;
  end else
  begin
    n := Plural(n);
    var tType := msg.typ.declaration.name.Replace('float','single');
    tType := tType.Replace('bytes','TBytes');
    tType := tType.Replace('bool','boolean');
    if m  then
    begin
        tType := 'T'+ tType;
        Wrln('var v : %s;', [tType]);
        Wrln('%s(v);', [GetRead(msg)])
    end
    else if e then
    begin
        tType := 'T'+ tType;
        Wrln('var v : %s := %s;', [tType, GetRead(msg)]);
    end else
    begin
        Wrln('var v : %s := %s;', [tType, GetRead(msg)]);
    end;
    Wrln('Value.%s.Add(@v);', [n]);
  end;
  if m then
  begin
    Dedent;
    Wrln('finally');
    Wrln('  Pb.Pop;');
    Wrln('end;');
  end;
end;
procedure TGenSGL.GenInitLoaded;
begin
  Wrln('Value.Init;');
end;
procedure TGenSGL.GenLoadImpl;
begin
  // empty
end;
procedure TGenSGL.GenSaveProc;
begin
  Wrln('{ TSaveHelper }');
  Wrln;
  Wrln('procedure TSaveHelper.SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);');
  Wrln('var');
  Wrln('  h: TpbSaver;');
  Wrln('begin');
  Wrln('  h.Init;');
  Wrln('  try');
  Wrln('    Save(h, obj);');
  Wrln('    Pb.writeMessage(tag, h.Pb^);');
  Wrln('  finally');
  Wrln('    h.Free;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
  Wrln('procedure TSaveHelper.SaveList<T>(const List: TsgRecordList<T>;');
  Wrln('  Save: TSave<T>; Tag: Integer);');
  Wrln('var');
  Wrln('  i: Integer;');
  Wrln('  h: TpbSaver;');
  Wrln('begin');
  Wrln('  h.Init;');
  Wrln('  try');
  Wrln('    for i := 0 to List.Count - 1 do');
  Wrln('    begin');
  Wrln('      h.Clear;');
  Wrln('      Save(h, List[i]^);');
  Wrln('      Pb.writeMessage(tag, h.Pb^);');
  Wrln('    end;');
  Wrln('  finally');
  Wrln('    h.Free;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
  Wrln('procedure TSaveHelper.SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>;');
  Wrln('  Save: TSavePair<Key, Value>; Tag: Integer);');
  Wrln('var');
  Wrln('  h: TpbSaver;');
  Wrln('  Pair: TsgHashMapIterator<Key, Value>.PPair;');
  Wrln('  it: TsgHashMapIterator<Key, Value>;');
  Wrln('begin');
  Wrln('  h.Init;');
  Wrln('  try');
  Wrln('    it := Map.Begins;');
  Wrln('    while it <> Map.Ends do');
  Wrln('    begin');
  Wrln('      h.Clear;');
  Wrln('      Save(h, it.GetPair^);');
  Wrln('      Pb.writeMessage(tag, h.Pb^);');
  Wrln('      it.Next;');
  Wrln('    end;');
  Wrln('  finally');
  Wrln('    h.Free;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
end;
procedure TGenSGL.GenSaveImpl(msg: PObj);
var
  s, t: string;
begin
  s := msg.DelphiName;
  t := msg.AsType;
  Wrln('class procedure %s.Save%s(const S: TpbSaver; const Value: %s);',
    [GetBuilderName(False), s, t]);
end;
{$EndRegion}
end.
