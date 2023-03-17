unit ProtoGen.&Function;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.nodedef,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types,
  ProtoGen.opdef,
  ProtoGen.fulltype;

{$T+}

type

  TFunctionDef = class;
  TRegisteredGradient = class;
  TGradientDef = class;

  TFunctionDefLibrary = Class
  const
    ftFunctions = 1;
    ftGradients = 2;
    ftRegisteredGradientss = 3;
  private
    FFunctions: TList<TFunctionDef>;
    FGradients: TList<TGradientDef>;
    FRegisteredGradientss: TList<TRegisteredGradient>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Functions: TList<TFunctionDef> read FFunctions;
    property Gradients: TList<TGradientDef> read FGradients;
    property RegisteredGradientss: TList<TRegisteredGradient> read FRegisteredGradientss;
  end;

  TArgAttrs = Class
  const
    ftAttr = 1;
  private
    FAttr: TStringAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Attr: TStringAttrValue read FAttr write FAttr;
  end;

  TUint32ArgAttrs  = TDictionary<UInt32, TArgAttrs>;

  TFunctionDef = Class
  const
    ftSignature = 1;
    ftAttr = 5;
    ftArgAttr = 7;
    ftResourceArgUniqueId = 8;
    ftNodeDefs = 3;
    ftRet = 4;
    ftControlRet = 6;
  private
    FSignature: TOpDef;
    FAttr: TStringAttrValue;
    FArgAttr: TUint32ArgAttrs;
    FResourceArgUniqueId: TUint32Uint32;
    FNodeDefs: TList<TNodeDef>;
    FRet: TStringString;
    FControlRet: TStringString;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Signature: TOpDef read FSignature write FSignature;
    property Attr: TStringAttrValue read FAttr write FAttr;
    property ArgAttr: TUint32ArgAttrs read FArgAttr write FArgAttr;
    property ResourceArgUniqueId: TUint32Uint32 read FResourceArgUniqueId write FResourceArgUniqueId;
    property NodeDefs: TList<TNodeDef> read FNodeDefs;
    property Ret: TStringString read FRet write FRet;
    property ControlRet: TStringString read FControlRet write FControlRet;
  end;

  TGradientDef = Class
  const
    ftFunctionName = 1;
    ftGradientFunc = 2;
  private
    FFunctionName: string;
    FGradientFunc: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property FunctionName: string read FFunctionName write FFunctionName;
    property GradientFunc: string read FGradientFunc write FGradientFunc;
  end;

  TRegisteredGradient = Class
  const
    ftGradientFunc = 1;
    ftRegisteredOpType = 2;
  private
    FGradientFunc: string;
    FRegisteredOpType: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property GradientFunc: string read FGradientFunc write FGradientFunc;
    property RegisteredOpType: string read FRegisteredOpType write FRegisteredOpType;
  end;

implementation

{ TFunctionDefLibrary }

Constructor TFunctionDefLibrary.Create;
begin
  inherited Create;
  
  FFunctions := TList<TFunctionDef>.Create;
  
  FGradients := TList<TGradientDef>.Create;
  
  FRegisteredGradientss := TList<TRegisteredGradient>.Create;
end;

destructor TFunctionDefLibrary.Destroy;
begin
  FFunctions.Free;
  FGradients.Free;
  FRegisteredGradientss.Free;
  inherited Destroy;
end;

{ TArgAttrs }

Constructor TArgAttrs.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TArgAttrs.Destroy;
begin
  FAttr.Free;
  inherited Destroy;
end;

{ TFunctionDef }

Constructor TFunctionDef.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
  FArgAttr := TDictionary<UInt32, TArgAttrs>.Create;
  FResourceArgUniqueId := TDictionary<UInt32, UInt32>.Create;
  
  FNodeDefs := TList<TNodeDef>.Create;
  FRet := TDictionary<string, string>.Create;
  FControlRet := TDictionary<string, string>.Create;
end;

destructor TFunctionDef.Destroy;
begin
  FAttr.Free;
  FArgAttr.Free;
  FResourceArgUniqueId.Free;
  FNodeDefs.Free;
  FRet.Free;
  FControlRet.Free;
  inherited Destroy;
end;

{ TGradientDef }

Constructor TGradientDef.Create;
begin
  inherited Create;
end;

destructor TGradientDef.Destroy;
begin
  inherited Destroy;
end;

{ TRegisteredGradient }

Constructor TRegisteredGradient.Create;
begin
  inherited Create;
end;

destructor TRegisteredGradient.Destroy;
begin
  inherited Destroy;
end;

end.
