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

unit Oz.Pb.Gen;

interface

uses
  System.Classes, System.SysUtils, System.Math, Generics.Collections,
  Oz.Cocor.Utils, Oz.Cocor.Lib, Oz.Pb.Tab, Oz.Pb.Classes;

{$Region 'TGen: abstract code generator'}

type
  TMapTypes = TList<PType>;

  TGetMap = (
    asVarDecl,
    asParam,
    asVarUsing);

  TGen = class(TCocoPart)
  protected
    function GetCode: string; virtual; abstract;
  public
    procedure GenerateCode; virtual; abstract;
    // Generated code
    property Code: string read GetCode;
  end;

{$EndRegion}

function GetCodeGen(Parser: TBaseParser): TGen;

implementation

uses
  Oz.Pb.Parser, Oz.Pb.Options, Oz.Pb.GenSGL, Oz.Pb.GenDC;

function GetCodeGen(Parser: TBaseParser): TGen;
begin
  case TpbParser(Parser).options.GenLib of
    TOptions.TCollectionLibrary.clSGL:
      Result := TGenSGL.Create(Parser);
    else
      // TOptions.TCollectionLibrary.clDelphi
      Result := TGenDC.Create(Parser);
  end;
end;

end.
