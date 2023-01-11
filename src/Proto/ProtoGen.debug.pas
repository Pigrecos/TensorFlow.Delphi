unit ProtoGen.Debug;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type


  TDebugTensorWatch = Class
  const
    ftNodeName = 1;
    ftOutputSlot = 2;
    ftDebugOpss = 3;
    ftDebugUrlss = 4;
    ftTolerateDebugOpCreationFailures = 5;
  private
    FNodeName: string;
    FOutputSlot: Integer;
    FDebugOpss: TList<string>;
    FDebugUrlss: TList<string>;
    FTolerateDebugOpCreationFailures: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property NodeName: string read FNodeName write FNodeName;
    property OutputSlot: Integer read FOutputSlot write FOutputSlot;
    property DebugOpss: TList<string> read FDebugOpss;
    property DebugUrlss: TList<string> read FDebugUrlss;
    property TolerateDebugOpCreationFailures: Boolean read FTolerateDebugOpCreationFailures write FTolerateDebugOpCreationFailures;
  end;

  TDebugOptions = Class
  const
    ftDebugTensorWatchOptss = 4;
    ftGlobalStep = 10;
    ftResetDiskByteUsage = 11;
  private
    FDebugTensorWatchOptss: TList<TDebugTensorWatch>;
    FGlobalStep: Int64;
    FResetDiskByteUsage: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property DebugTensorWatchOptss: TList<TDebugTensorWatch> read FDebugTensorWatchOptss;
    property GlobalStep: Int64 read FGlobalStep write FGlobalStep;
    property ResetDiskByteUsage: Boolean read FResetDiskByteUsage write FResetDiskByteUsage;
  end;

  TDebuggedSourceFile = Class
  const
    ftHost = 1;
    ftFilePath = 2;
    ftLastModified = 3;
    ftBytes = 4;
    ftLiness = 5;
  private
    FHost: string;
    FFilePath: string;
    FLastModified: Int64;
    FBytes: Int64;
    FLiness: TList<string>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Host: string read FHost write FHost;
    property FilePath: string read FFilePath write FFilePath;
    property LastModified: Int64 read FLastModified write FLastModified;
    property Bytes: Int64 read FBytes write FBytes;
    property Liness: TList<string> read FLiness;
  end;

  TDebuggedSourceFiles = Class
  const
    ftSourceFiless = 1;
  private
    FSourceFiless: TList<TDebuggedSourceFile>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property SourceFiless: TList<TDebuggedSourceFile> read FSourceFiless;
  end;

implementation

{ TDebugTensorWatch }

Constructor TDebugTensorWatch.Create;
begin
  inherited Create;
  
  FDebugOpss := TList<string>.Create;
  
  FDebugUrlss := TList<string>.Create;
end;

destructor TDebugTensorWatch.Destroy;
begin
  FDebugOpss.Free;
  FDebugUrlss.Free;
  inherited Destroy;
end;

{ TDebugOptions }

Constructor TDebugOptions.Create;
begin
  inherited Create;
  
  FDebugTensorWatchOptss := TList<TDebugTensorWatch>.Create;
end;

destructor TDebugOptions.Destroy;
begin
  FDebugTensorWatchOptss.Free;
  inherited Destroy;
end;

{ TDebuggedSourceFile }

Constructor TDebuggedSourceFile.Create;
begin
  inherited Create;
  
  FLiness := TList<string>.Create;
end;

destructor TDebuggedSourceFile.Destroy;
begin
  FLiness.Free;
  inherited Destroy;
end;

{ TDebuggedSourceFiles }

Constructor TDebuggedSourceFiles.Create;
begin
  inherited Create;
  
  FSourceFiless := TList<TDebuggedSourceFile>.Create;
end;

destructor TDebuggedSourceFiles.Destroy;
begin
  FSourceFiless.Free;
  inherited Destroy;
end;

end.
