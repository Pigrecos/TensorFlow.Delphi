unit ProtoGen.NodeDef;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.fulltype,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  TExperimentalDebugInfo = Class
  const
    ftOriginalNodeNamess = 1;
    ftOriginalFuncNamess = 2;
  private
    FOriginalNodeNamess: TList<string>;
    FOriginalFuncNamess: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property OriginalNodeNamess: TList<string> read FOriginalNodeNamess;
    property OriginalFuncNamess: TList<string> read FOriginalFuncNamess;
  end;

  TNodeDef = Class
  const
    ftName = 1;
    ftOp = 2;
    ftInputs = 3;
    ftDevice = 4;
    ftAttr = 5;
    ftExperimentalDebugInfo = 6;
    ftExperimentalType = 7;
  private
    FName: string;
    FOp: string;
    FInputs: TList<string>;
    FDevice: string;
    FAttr: TStringAttrValue;
    FExperimentalDebugInfo: TExperimentalDebugInfo;
    FExperimentalType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Op: string read FOp write FOp;
    property Inputs: TList<string> read FInputs;
    property Device: string read FDevice write FDevice;
    property Attr: TStringAttrValue read FAttr write FAttr;
    property ExperimentalDebugInfo: TExperimentalDebugInfo read FExperimentalDebugInfo write FExperimentalDebugInfo;
    property ExperimentalType: TFullTypeDef read FExperimentalType write FExperimentalType;
  end;

implementation

{ TExperimentalDebugInfo }

Constructor TExperimentalDebugInfo.Create;
begin
  inherited Create;
  
  FOriginalNodeNamess := TList<string>.Create;
  
  FOriginalFuncNamess := TList<string>.Create;
end;

destructor TExperimentalDebugInfo.Destroy;
begin
  FOriginalNodeNamess.Free;
  FOriginalFuncNamess.Free;
  inherited Destroy;
end;

{ TNodeDef }

Constructor TNodeDef.Create;
begin
  inherited Create;
  
  FInputs := TList<string>.Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TNodeDef.Destroy;
begin
  FInputs.Free;
  FAttr.Free;
  inherited Destroy;
end;

end.
