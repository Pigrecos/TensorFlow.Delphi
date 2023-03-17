unit ProtoGen.VerifierConfig;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.types;

{$T+}

type

  TVerifierConfig = Class
  const
    ftVerificationTimeoutInMs = 1;
    ftStructureVerifier = 2;
  private
    FVerificationTimeoutInMs: Int64;
    FStructureVerifier: TToggle;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property VerificationTimeoutInMs: Int64 read FVerificationTimeoutInMs write FVerificationTimeoutInMs;
    property StructureVerifier: TToggle read FStructureVerifier write FStructureVerifier;
  end;

implementation

{ TVerifierConfig }

Constructor TVerifierConfig.Create;
begin
  inherited Create;
end;

destructor TVerifierConfig.Destroy;
begin
  inherited Destroy;
end;

end.
