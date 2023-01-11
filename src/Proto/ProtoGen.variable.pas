unit ProtoGen.Variable;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type

  TSaveSliceInfoDef = class;

  TVariableSynchronization = (
    VARIABLE_SYNCHRONIZATION_AUTO = 0,
    VARIABLE_SYNCHRONIZATION_NONE = 1,
    VARIABLE_SYNCHRONIZATION_ON_WRITE = 2,
    VARIABLE_SYNCHRONIZATION_ON_READ = 3);

  TVariableAggregation = (
    VARIABLE_AGGREGATION_NONE = 0,
    VARIABLE_AGGREGATION_SUM = 1,
    VARIABLE_AGGREGATION_MEAN = 2,
    VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA = 3);

  TVariableDef = Class
  const
    ftVariableName = 1;
    ftInitialValueName = 6;
    ftInitializerName = 2;
    ftSnapshotName = 3;
    ftSaveSliceInfoDef = 4;
    ftIsResource = 5;
    ftTrainable = 7;
    ftSynchronization = 8;
    ftAggregation = 9;
  private
    FVariableName: string;
    FInitialValueName: string;
    FInitializerName: string;
    FSnapshotName: string;
    FSaveSliceInfoDef: TSaveSliceInfoDef;
    FIsResource: Boolean;
    FTrainable: Boolean;
    FSynchronization: TVariableSynchronization;
    FAggregation: TVariableAggregation;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property VariableName: string read FVariableName write FVariableName;
    property InitialValueName: string read FInitialValueName write FInitialValueName;
    property InitializerName: string read FInitializerName write FInitializerName;
    property SnapshotName: string read FSnapshotName write FSnapshotName;
    property SaveSliceInfoDef: TSaveSliceInfoDef read FSaveSliceInfoDef write FSaveSliceInfoDef;
    property IsResource: Boolean read FIsResource write FIsResource;
    property Trainable: Boolean read FTrainable write FTrainable;
    property Synchronization: TVariableSynchronization read FSynchronization write FSynchronization;
    property Aggregation: TVariableAggregation read FAggregation write FAggregation;
  end;

  TSaveSliceInfoDef = Class
  const
    ftFullName = 1;
    ftFullShapes = 2;
    ftVarOffsets = 3;
    ftVarShapes = 4;
  private
    FFullName: string;
    FFullShapes: TList<Int64>;
    FVarOffsets: TList<Int64>;
    FVarShapes: TList<Int64>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property FullName: string read FFullName write FFullName;
    property FullShapes: TList<Int64> read FFullShapes;
    property VarOffsets: TList<Int64> read FVarOffsets;
    property VarShapes: TList<Int64> read FVarShapes;
  end;

implementation

{ TVariableDef }

Constructor TVariableDef.Create;
begin
  inherited Create;
end;

destructor TVariableDef.Destroy;
begin
  inherited Destroy;
end;

{ TSaveSliceInfoDef }

Constructor TSaveSliceInfoDef.Create;
begin
  inherited Create;
  
  FFullShapes := TList<Int64>.Create;
  
  FVarOffsets := TList<Int64>.Create;
  
  FVarShapes := TList<Int64>.Create;
end;

destructor TSaveSliceInfoDef.Destroy;
begin
  FFullShapes.Free;
  FVarOffsets.Free;
  FVarShapes.Free;
  inherited Destroy;
end;

end.
