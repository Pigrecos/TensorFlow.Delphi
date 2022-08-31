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

unit Oz.Pb.Options;

interface

uses
  System.SysUtils, System.Classes;

type

{$Region 'TOptions: Compilation settings'}

  TOptions = class
  const
    Version = '1.0 (for Delphi)';
    ReleaseDate = '11 August 2020';
  type
    TCollectionLibrary = (clDelphi, clSGL);
  private
    FSrcName: string;
    FSrcDir: string;
    FOutPath: string;
    FCodeGen: string;
    FListing: TStrings;
    FGenLib: TCollectionLibrary;
  public
    constructor Create;
    function GetVersion: string;
    procedure Help;
    property Listing: TStrings read FListing write FListing;
    // Get options from the command line
    procedure ParseCommandLine;
    // Set option
    procedure SetOption(const s: string);
    // Name of the proto file (including path)
    property SrcName: string read FSrcName;
    // Specify the directory in which to search for imports.
    // May be specified multiple times; directories will be searched in order.
    // If not given, the current working directory is used.
    property SrcDir: string read FSrcDir write FSrcDir;
    // Path for generated delphi files
    property OutPath: string read FOutPath write FOutPath;
    // The Collections library used for the generated code
    property GenLib: TCollectionLibrary read FGenLib;
  end;

{$EndRegion}

// Return current settings (sigleton)
function GetOptions: TOptions;

implementation

var
  FOptions: TOptions = nil;

function GetOptions: TOptions;
begin
  if FOptions = nil then
    FOptions := TOptions.Create;
  result := FOptions;
end;

procedure FreeOptions;
begin
  FreeAndNil(FOptions);
end;

{$Region 'TOptions'}

constructor TOptions.Create;
begin

end;

function TOptions.GetVersion: string;
begin
  Result := Format(
    'Protoc - Protocîl buffer code generator, V%s'#13#10 +
    'Delphi version by Marat Shaimardanov %s'#13#10,
    [Version, ReleaseDate]);
end;

procedure TOptions.Help;
begin
  WriteLn('Usage: Protoc file.proto {Option}');
  WriteLn('Options:');
  WriteLn('  -proto <protoFilesDirectory>');
  WriteLn('  -o <outputDirectory>');
  WriteLn('  -c <code generation kind>');
  WriteLn('     list of code generation kind:');
  WriteLn('       • s - standard code');
  WriteLn('       • m - using metadata');
  WriteLn('       • a - using attributes');
  WriteLn('  -genDelphi');
  WriteLn('  -genSGL');
end;

procedure TOptions.ParseCommandLine;
var
  i: Integer;
  p: string;

  function GetParam: Boolean;
  begin
    Result := i < ParamCount;
    if Result then
    begin
      Inc(i);
      p := LowerCase(ParamStr(i).Trim);
    end;
  end;

begin
  i := 0;
  while GetParam do
  begin
    if (p = '-proto') and GetParam then
      FSrcDir := p
    else if (p = '-o') and GetParam then
      FOutPath := p
    else if (p = '-c') and GetParam then
      FCodeGen := p
    else if (p = '-gendelphi') then
      FGenLib := clDelphi
    else if (p = '-gensgl') then
      FGenLib := clSGL
    else
      FSrcName := p;
  end;
  if FOutPath = '' then
    FOutPath := FSrcDir;
end;

procedure TOptions.SetOption(const s: string);
var
  name, value: string;
  option: TArray<string>;
begin
  option := s.Split(['=', ' '], 2);
  name := option[0];
  value := option[1];
end;

{$EndRegion}

initialization

finalization
  FreeOptions;

end.
