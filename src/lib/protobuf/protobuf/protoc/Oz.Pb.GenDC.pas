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
unit Oz.Pb.GenDC;
interface
uses
  System.SysUtils, System.Generics.Collections, Oz.Cocor.Utils, Oz.Pb.Tab, Oz.Pb.CustomGen;
{$Region 'TGenDC: code generator for delphi'}
type
  TGenDC = class(TCustomGen)
  protected
    function  TestNil: string; override;
    function  AddItemMap(obj: PObj): string; override;
    function  MapCollection: string; override;
    function  RepeatedCollection: string; override;
    function  CreateName: string; override;
    function  CreateMapName: string; override;
    procedure GenUses; override;
    procedure GenDecl(Load: Boolean); override;
    procedure FieldWrite(obj: PObj);
    procedure GenEntityType(msg: PObj); override;
    procedure GenEntityDecl; override;
    procedure GenEntityImpl(msg: PObj); override;
    procedure GenLoadDecl(msg: PObj); override;
    procedure GenSaveDecl(msg: PObj); override;
    procedure GenLoadImpl; override;
    procedure GenSaveProc; override;
    procedure GenLoadMethod(msg: PObj); override;
    function  GenRead(msg: PObj): string; override;
    procedure GenFieldRead(msg: PObj); override;
    procedure GenInitLoaded; override;
    procedure GenSaveImpl(msg: PObj); override;
    function  GenPairStr:string; override;
  end;
{$EndRegion}
implementation
uses
  Oz.Pb.Parser;
{$Region 'TGenDC'}
function TGenDC.AddItemMap(obj: PObj): string;
begin
    Result := 'AddOrSetValue';
end;
function TGenDC.MapCollection: string;
begin
  Result := 'TDictionary<%s, %s>';
end;
function TGenDC.RepeatedCollection: string;
begin
  Result := 'TList<%s>';
end;
function TGenDC.TestNil: string;
begin
    Result :=  ' <> nil'
end;

function TGenDC.CreateName: string;
begin
  Result := 'Create';
end;

function TGenDC.CreateMapName: string;
begin
    Result := 'Create';
end;

procedure TGenDC.GenEntityType(msg: PObj);
var
  s: string;
begin
  s := AsCamel(msg.typ.declaration.name);
  //if not FImportProcess then
  //begin
  //   Wrln('P%s = ^T%s;', [s, s]);
  //   Wrln('T%s = record', [s]);
  //end
  //else
     Wrln('T%s = Class', [s])
end;

procedure TGenDC.GenUses;
var
  x    : PObj;
  Done : TList<String>;
begin

  Done := TList<String>.Create;
  try
    Wrln('uses');
    if tab.Module.Import <> nil then
    begin
        Wrln('  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,');
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
        Wrln('  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;');
    end;
    Wrln;
    Wrln('{$T+}');
    Wrln;
  finally
    Done.Free;
  end;
end;

procedure TGenDC.GenDecl(Load: Boolean);
begin
  if Load then
  begin
    Wrln('type');
    Wrln('  TLoad<T: constructor> = procedure(var Value: T) of object;');
    Wrln('  TLoadPair<Key, Value> = procedure(var Pair: TPair<Key, Value>) of object;');
    Wrln('private');
    Wrln('  procedure LoadObj<T: constructor>(var obj: T; Load: TLoad<T>);');
    Wrln('  procedure LoadList<T: constructor>(const List: TList<T>; Load: TLoad<T>);');
  end
  else
  begin
    Wrln('type');
    Wrln('  TSave<T>              = procedure(const S: TpbSaver; const Value: T);');
    Wrln('  TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TPair<Key, Value>);');
    Wrln('private');
    Wrln('  procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);');
    Wrln('  procedure SaveList<T>(const List: TList<T>; Save: TSave<T>; Tag: Integer);');
    Wrln('  procedure SaveMap<Key, Value>(const Map: TDictionary<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);');
  end;
end;

procedure TGenDC.FieldWrite(obj: PObj);
var
  fg: TFieldGen;
begin
  if obj.cls <> TMode.mField then exit;
  fg.Init(Self, obj, obj.aux as TFieldOptions, FieldTag(obj));
  fg.checkNil := False;
  fg.Gen;
end;

procedure TGenDC.GenEntityDecl;
begin
    Wrln('Constructor Create;');
    Wrln('destructor  Destroy; Override;');
end;

procedure TGenDC.GenEntityImpl(msg: PObj);
var
  typ: PType;
  t: string;
  x: PObj;
begin
  typ := msg.typ;
  // parameterless Init;
  t := msg.AsType;
  Wrln('Constructor %s.Create;', [t]);
  Wrln('begin');
  Indent;
  try
    Wrln('inherited Create;');

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
  Wrln('destructor %s.Destroy;', [t]);

  Wrln('begin');
  Indent;
  try
    x := typ.dsc;
    while x <> tab.Guard do
    begin
      FieldFree(x);
      x := x.next;
    end;
    Wrln('inherited Destroy;');
  finally
    Dedent;
  end;
end;

procedure TGenDC.GenLoadDecl(msg: PObj);
var
  t: string;
begin
  t := msg.AsType;
  Wrln('procedure Load%s(var Value: %s);', [msg.DelphiName, t]);
end;

procedure TGenDC.GenSaveDecl(msg: PObj);
begin
  Wrln('class procedure Save%s(const S: TpbSaver; const Value: %s); static;', [msg.DelphiName, msg.AsType]);
end;

procedure TGenDC.GenLoadMethod(msg: PObj);
var
  s, t: string;
begin
  s := msg.DelphiName;
  t := msg.AsType;
  Wrln('procedure %s.Load%s(var Value: %s);', [GetBuilderName(True), s, t]);
end;

function TGenDC.GenPairStr: string;
begin
    Result := 'TPair' ;
end;

function TGenDC.GenRead(msg: PObj): string;
begin
  Result := Format('Load%s', [msg.DelphiName]);
  //Result := Format('Load%s(%s.Create', [msg.DelphiName, msg.AsType]);
end;

procedure TGenDC.GenFieldRead(msg: PObj);
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
  else                   n := {'F' +} AsCamel(msg.name);
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
    Wrln('Value.%s.Add(v);', [n]);
  end;
  if m then
  begin
    Dedent;
    Wrln('finally');
    Wrln('  Pb.Pop;');
    Wrln('end;');
  end;
end;

procedure TGenDC.GenInitLoaded;
begin
  Wrln('Value.Create;');
end;

procedure TGenDC.GenLoadImpl;
begin
  Wrln('{ TLoadHelper }');
  Wrln;
  Wrln('procedure TLoadHelper.LoadObj<T>(var obj: T; Load: TLoad<T>);');
  Wrln('begin');
  Wrln('  Pb.Push;');
  Wrln('  try');
  Wrln('    var v : TValue := TValue.From<T>(obj);');
  Wrln('    var vObj       := v.AsObject;');
  Wrln('    obj := vObj.Create;');
  Wrln('    Load(obj);');
  Wrln('  finally');
  Wrln('    Pb.Pop;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
  Wrln('procedure TLoadHelper.LoadList<T>(const List: TList<T>; Load: TLoad<T>);');
  Wrln('var');
  Wrln('  obj: T;');
  Wrln('begin');
  Wrln('  Pb.Push;');
  Wrln('  try');
  Wrln('    var v : TValue := TValue.From<T>(obj);');
  Wrln('    var vObj       := v.AsObject;');
  Wrln('    obj := vObj.Create;');
  Wrln('    Load(obj);');
  Wrln('    List.Add(obj);');
  Wrln('  finally');
  Wrln('    Pb.Pop;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
end;

procedure TGenDC.GenSaveProc;
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
  Wrln('procedure TSaveHelper.SaveList<T>(const List: TList<T>; Save: TSave<T>; Tag: Integer);');
  Wrln('var');
  Wrln('  i: Integer;');
  Wrln('  h: TpbSaver;');
  Wrln('  Item: T;');
  Wrln('begin');
  Wrln('  h.Init;');
  Wrln('  try');
  Wrln('    for i := 0 to List.Count - 1 do');
  Wrln('    begin');
  Wrln('      h.Clear;');
  Wrln('      Item := List[i];');
  Wrln('      Save(h, Item);');
  Wrln('      Pb.writeMessage(tag, h.Pb^);');
  Wrln('    end;');
  Wrln('  finally');
  Wrln('    h.Free;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
  Wrln('procedure TSaveHelper.SaveMap<Key, Value>(const Map: TDictionary<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);');
  Wrln('var');
  Wrln('  h: TpbSaver;');
  Wrln('  Pair: TPair<Key, Value>;');
  Wrln('begin');
  Wrln('  h.Init;');
  Wrln('  try');
  Wrln('    for Pair in Map do');
  Wrln('    begin');
  Wrln('      h.Clear;');
  Wrln('      Save(h, Pair);');
  Wrln('      Pb.writeMessage(tag, h.Pb^);');
  Wrln('    end;');
  Wrln('  finally');
  Wrln('    h.Free;');
  Wrln('  end;');
  Wrln('end;');
  Wrln;
end;

procedure TGenDC.GenSaveImpl(msg: PObj);
var
  s, t: string;
begin
  s := msg.DelphiName;
  t := msg.AsType;
  Wrln('class procedure %s.Save%s(const S: TpbSaver; const Value: %s);', [GetBuilderName(False), s, t]);
end;
{$EndRegion}
end.
