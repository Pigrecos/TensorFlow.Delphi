unit ProtoGen.Graph;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.&function,
  ProtoGen.fulltype,
  ProtoGen.nodedef,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types,
  ProtoGen.versions,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.opdef;

{$T+}

type
  TGraphDef = Class
  const
    ftNodes = 1;
    ftVersions = 4;
    ftVersion = 3;
    ftLibrary = 2;
  private
    FNodes: TList<TNodeDef>;
    FVersions: TVersionDef;
    FVersion: Integer;
    FLibrary: TFunctionDefLibrary;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Nodes: TList<TNodeDef> read FNodes;
    property Versions: TVersionDef read FVersions write FVersions;
    property Version: Integer read FVersion write FVersion;
    property &Library: TFunctionDefLibrary read FLibrary write FLibrary;
  end;

implementation

{ TGraphDef }

Constructor TGraphDef.Create;
begin
  inherited Create;
  
  FNodes := TList<TNodeDef>.Create;
end;

destructor TGraphDef.Destroy;
begin
  FNodes.Free;
  inherited Destroy;
end;

end.
