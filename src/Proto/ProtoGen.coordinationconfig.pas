unit ProtoGen.CoordinationConfig;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type
  TCoordinationServiceConfig = Class
  const
    ftServiceType = 1;
    ftServiceLeader = 2;
    ftEnableHealthCheck = 3;
    ftClusterRegisterTimeoutInMs = 4;
    ftHeartbeatTimeoutInMs = 5;
    ftCoordinatedJobss = 6;
  private
    FServiceType: string;
    FServiceLeader: string;
    FEnableHealthCheck: Boolean;
    FClusterRegisterTimeoutInMs: Int64;
    FHeartbeatTimeoutInMs: Int64;
    FCoordinatedJobss: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property ServiceType: string read FServiceType write FServiceType;
    property ServiceLeader: string read FServiceLeader write FServiceLeader;
    property EnableHealthCheck: Boolean read FEnableHealthCheck write FEnableHealthCheck;
    property ClusterRegisterTimeoutInMs: Int64 read FClusterRegisterTimeoutInMs write FClusterRegisterTimeoutInMs;
    property HeartbeatTimeoutInMs: Int64 read FHeartbeatTimeoutInMs write FHeartbeatTimeoutInMs;
    property CoordinatedJobss: TList<string> read FCoordinatedJobss;
  end;

implementation

{ TCoordinationServiceConfig }

Constructor TCoordinationServiceConfig.Create;
begin
  inherited Create;
  
  FCoordinatedJobss := TList<string>.Create;
end;

destructor TCoordinationServiceConfig.Destroy;
begin
  FCoordinatedJobss.Free;
  inherited Destroy;
end;

end.
