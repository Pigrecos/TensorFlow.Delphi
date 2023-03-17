unit ProtoGen.ControlFlow;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes,

  ProtoGen.types;

{$T+}

type

  TCondContextDef = class;
  TWhileContextDef = class;

  TValuesDef = Class
  const
    ftValuess = 1;
    ftExternalValues = 2;
  private
    FValuess: TList<string>;
    FExternalValues: TStringString;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Valuess: TList<string> read FValuess;
    property ExternalValues: TStringString read FExternalValues write FExternalValues;
  end;

  TControlFlowContextDef = Class
  const
    ftCondCtxt = 1;
    ftWhileCtxt = 2;
  private
    FCtxt: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ctxt: TpbOneof read FCtxt write FCtxt;
  end;

  TCondContextDef = Class
  const
    ftContextName = 1;
    ftPredName = 2;
    ftPivotName = 3;
    ftBranch = 4;
    ftValuesDef = 5;
    ftNestedContextss = 6;
  private
    FContextName: string;
    FPredName: string;
    FPivotName: string;
    FBranch: Integer;
    FValuesDef: TValuesDef;
    FNestedContextss: TList<TControlFlowContextDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ContextName: string read FContextName write FContextName;
    property PredName: string read FPredName write FPredName;
    property PivotName: string read FPivotName write FPivotName;
    property Branch: Integer read FBranch write FBranch;
    property ValuesDef: TValuesDef read FValuesDef write FValuesDef;
    property NestedContextss: TList<TControlFlowContextDef> read FNestedContextss;
  end;

  TWhileContextDef = Class
  const
    ftContextName = 1;
    ftParallelIterations = 2;
    ftBackProp = 3;
    ftSwapMemory = 4;
    ftPivotName = 5;
    ftPivotForPredName = 6;
    ftPivotForBodyName = 7;
    ftLoopExitNamess = 8;
    ftLoopEnterNamess = 10;
    ftValuesDef = 9;
    ftMaximumIterationsName = 11;
    ftNestedContextss = 12;
  private
    FContextName: string;
    FParallelIterations: Integer;
    FBackProp: Boolean;
    FSwapMemory: Boolean;
    FPivotName: string;
    FPivotForPredName: string;
    FPivotForBodyName: string;
    FLoopExitNamess: TList<string>;
    FLoopEnterNamess: TList<string>;
    FValuesDef: TValuesDef;
    FMaximumIterationsName: string;
    FNestedContextss: TList<TControlFlowContextDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ContextName: string read FContextName write FContextName;
    property ParallelIterations: Integer read FParallelIterations write FParallelIterations;
    property BackProp: Boolean read FBackProp write FBackProp;
    property SwapMemory: Boolean read FSwapMemory write FSwapMemory;
    property PivotName: string read FPivotName write FPivotName;
    property PivotForPredName: string read FPivotForPredName write FPivotForPredName;
    property PivotForBodyName: string read FPivotForBodyName write FPivotForBodyName;
    property LoopExitNamess: TList<string> read FLoopExitNamess;
    property LoopEnterNamess: TList<string> read FLoopEnterNamess;
    property ValuesDef: TValuesDef read FValuesDef write FValuesDef;
    property MaximumIterationsName: string read FMaximumIterationsName write FMaximumIterationsName;
    property NestedContextss: TList<TControlFlowContextDef> read FNestedContextss;
  end;

implementation

{ TValuesDef }

Constructor TValuesDef.Create;
begin
  inherited Create;
  
  FValuess := TList<string>.Create;
  FExternalValues := TDictionary<string, string>.Create;
end;

destructor TValuesDef.Destroy;
begin
  FValuess.Free;
  FExternalValues.Free;
  inherited Destroy;
end;

{ TControlFlowContextDef }

Constructor TControlFlowContextDef.Create;
begin
  inherited Create;
end;

destructor TControlFlowContextDef.Destroy;
begin
  inherited Destroy;
end;

{ TCondContextDef }

Constructor TCondContextDef.Create;
begin
  inherited Create;
  
  FNestedContextss := TList<TControlFlowContextDef>.Create;
end;

destructor TCondContextDef.Destroy;
begin
  FNestedContextss.Free;
  inherited Destroy;
end;

{ TWhileContextDef }

Constructor TWhileContextDef.Create;
begin
  inherited Create;
  
  FLoopExitNamess := TList<string>.Create;
  
  FLoopEnterNamess := TList<string>.Create;
  
  FNestedContextss := TList<TControlFlowContextDef>.Create;
end;

destructor TWhileContextDef.Destroy;
begin
  FLoopExitNamess.Free;
  FLoopEnterNamess.Free;
  FNestedContextss.Free;
  inherited Destroy;
end;

end.
