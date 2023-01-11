unit ProtoGen.Cluster;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.Types;

{$T+}

type
  TJobDef = Class
  const
    ftName = 1;
    ftTasks = 2;
  private
    FName: string;
    FTasks: TInt32String;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Tasks: TInt32String read FTasks write FTasks;
  end;

  TClusterDef = Class
  const
    ftJobs = 1;
  private
    FJobs: TList<TJobDef>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Jobs: TList<TJobDef> read FJobs;
  end;

implementation

{ TJobDef }

Constructor TJobDef.Create;
begin
  inherited Create;
  FTasks := TDictionary<Integer, string>.Create;
end;

destructor TJobDef.Destroy;
begin
  FTasks.Free;
  inherited Destroy;
end;

{ TClusterDef }

Constructor TClusterDef.Create;
begin
  inherited Create;
  
  FJobs := TList<TJobDef>.Create;
end;

destructor TClusterDef.Destroy;
begin
  FJobs.Free;
  inherited Destroy;
end;

end.
