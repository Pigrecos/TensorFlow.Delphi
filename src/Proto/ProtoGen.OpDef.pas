unit ProtoGen.OpDef;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.fulltype,
  ProtoGen.types,
  ProtoGen.tensor,
  ProtoGen.tensorshape;

{$T+}

type

  TOpDeprecation = class;

  TArgDef = Class
  const
    ftName = 1;
    ftDescription = 2;
    ftType = 3;
    ftTypeAttr = 4;
    ftNumberAttr = 5;
    ftTypeListAttr = 6;
    ftHandleDatas = 7;
    ftIsRef = 16;
    ftExperimentalFullType = 17;
  private
    FName: string;
    FDescription: string;
    FType: TDataType;
    FTypeAttr: string;
    FNumberAttr: string;
    FTypeListAttr: string;
    FHandleDatas: TList<TResourceHandleProto>;
    FIsRef: Boolean;
    FExperimentalFullType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Description: string read FDescription write FDescription;
    property &Type: TDataType read FType write FType;
    property TypeAttr: string read FTypeAttr write FTypeAttr;
    property NumberAttr: string read FNumberAttr write FNumberAttr;
    property TypeListAttr: string read FTypeListAttr write FTypeListAttr;
    property HandleDatas: TList<TResourceHandleProto> read FHandleDatas;
    property IsRef: Boolean read FIsRef write FIsRef;
    property ExperimentalFullType: TFullTypeDef read FExperimentalFullType write FExperimentalFullType;
  end;

  TAttrDef = Class
  const
    ftName = 1;
    ftType = 2;
    ftDefaultValue = 3;
    ftDescription = 4;
    ftHasMinimum = 5;
    ftMinimum = 6;
    ftAllowedValues = 7;
  private
    FName: string;
    FType: string;
    FDefaultValue: TAttrValue;
    FDescription: string;
    FHasMinimum: Boolean;
    FMinimum: Int64;
    FAllowedValues: TAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property &Type: string read FType write FType;
    property DefaultValue: TAttrValue read FDefaultValue write FDefaultValue;
    property Description: string read FDescription write FDescription;
    property HasMinimum: Boolean read FHasMinimum write FHasMinimum;
    property Minimum: Int64 read FMinimum write FMinimum;
    property AllowedValues: TAttrValue read FAllowedValues write FAllowedValues;
  end;

  TOpDef = Class
  const
    ftName = 1;
    ftInputArgs = 2;
    ftOutputArgs = 3;
    ftControlOutputs = 20;
    ftAttrs = 4;
    ftDeprecation = 8;
    ftSummary = 5;
    ftDescription = 6;
    ftIsCommutative = 18;
    ftIsAggregate = 16;
    ftIsStateful = 17;
    ftAllowsUninitializedInput = 19;
    ftIsDistributedCommunication = 21;
  private
    FName: string;
    FInputArgs: TList<TArgDef>;
    FOutputArgs: TList<TArgDef>;
    FControlOutputs: TList<string>;
    FAttrs: TList<TAttrDef>;
    FDeprecation: TOpDeprecation;
    FSummary: string;
    FDescription: string;
    FIsCommutative: Boolean;
    FIsAggregate: Boolean;
    FIsStateful: Boolean;
    FAllowsUninitializedInput: Boolean;
    FIsDistributedCommunication: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property InputArgs: TList<TArgDef> read FInputArgs;
    property OutputArgs: TList<TArgDef> read FOutputArgs;
    property ControlOutputs: TList<string> read FControlOutputs;
    property Attrs: TList<TAttrDef> read FAttrs;
    property Deprecation: TOpDeprecation read FDeprecation write FDeprecation;
    property Summary: string read FSummary write FSummary;
    property Description: string read FDescription write FDescription;
    property IsCommutative: Boolean read FIsCommutative write FIsCommutative;
    property IsAggregate: Boolean read FIsAggregate write FIsAggregate;
    property IsStateful: Boolean read FIsStateful write FIsStateful;
    property AllowsUninitializedInput: Boolean read FAllowsUninitializedInput write FAllowsUninitializedInput;
    property IsDistributedCommunication: Boolean read FIsDistributedCommunication write FIsDistributedCommunication;
  end;

  TOpDeprecation = Class
  const
    ftVersion = 1;
    ftExplanation = 2;
  private
    FVersion: Integer;
    FExplanation: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Version: Integer read FVersion write FVersion;
    property Explanation: string read FExplanation write FExplanation;
  end;

  TOpList = Class
  const
    ftOps = 1;
  private
    FOps: TList<TOpDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ops: TList<TOpDef> read FOps;
  end;

implementation

{ TArgDef }

Constructor TArgDef.Create;
begin
  inherited Create;
  
  FHandleDatas := TList<TResourceHandleProto>.Create;
end;

destructor TArgDef.Destroy;
begin
  FHandleDatas.Free;
  inherited Destroy;
end;

{ TAttrDef }

Constructor TAttrDef.Create;
begin
  inherited Create;
end;

destructor TAttrDef.Destroy;
begin
  inherited Destroy;
end;

{ TOpDef }

Constructor TOpDef.Create;
begin
  inherited Create;
  
  FInputArgs := TList<TArgDef>.Create;
  
  FOutputArgs := TList<TArgDef>.Create;
  
  FControlOutputs := TList<string>.Create;
  
  FAttrs := TList<TAttrDef>.Create;
end;

destructor TOpDef.Destroy;
begin
  FInputArgs.Free;
  FOutputArgs.Free;
  FControlOutputs.Free;
  FAttrs.Free;
  inherited Destroy;
end;

{ TOpDeprecation }

Constructor TOpDeprecation.Create;
begin
  inherited Create;
end;

destructor TOpDeprecation.Destroy;
begin
  inherited Destroy;
end;

{ TOpList }

Constructor TOpList.Create;
begin
  inherited Create;
  
  FOps := TList<TOpDef>.Create;
end;

destructor TOpList.Destroy;
begin
  FOps.Free;
  inherited Destroy;
end;

end.
