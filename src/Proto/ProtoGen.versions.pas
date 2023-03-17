unit ProtoGen.Versions;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type
  TVersionDef = Class
  const
    ftProducer = 1;
    ftMinConsumer = 2;
    ftBadConsumerss = 3;
  private
    FProducer: Integer;
    FMinConsumer: Integer;
    FBadConsumerss: TList<Integer>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Producer: Integer read FProducer write FProducer;
    property MinConsumer: Integer read FMinConsumer write FMinConsumer;
    property BadConsumerss: TList<Integer> read FBadConsumerss;
  end;

implementation

{ TVersionDef }

Constructor TVersionDef.Create;
begin
  inherited Create;
  
  FBadConsumerss := TList<Integer>.Create;
end;

destructor TVersionDef.Destroy;
begin
  FBadConsumerss.Free;
  inherited Destroy;
end;

end.
