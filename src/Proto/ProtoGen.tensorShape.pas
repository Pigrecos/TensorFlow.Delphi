unit ProtoGen.TensorShape;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type


  TDim = Class
  const
    ftSize = 1;
    ftName = 2;
  private
    FSize: Int64;
    FName: string;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Size: Int64 read FSize write FSize;
    property Name: string read FName write FName;
  end;

  TTensorShapeProto = Class
  const
    ftDims = 2;
    ftUnknownRank = 3;
  private
    FDims: TList<TDim>;
    FUnknownRank: Boolean;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dims: TList<TDim> read FDims;
    property UnknownRank: Boolean read FUnknownRank write FUnknownRank;
  end;

implementation

{ TDim }

Constructor TDim.Create;
begin
  inherited Create;
  FSize := 0;
  FName:= '' ;
end;

destructor TDim.Destroy;
begin
  inherited Destroy;
end;

{ TTensorShapeProto }

Constructor TTensorShapeProto.Create;
begin
  inherited Create;
  
  FDims := TList<TDim>.Create;
end;

destructor TTensorShapeProto.Destroy;
begin
  FDims.Free;
  inherited Destroy;
end;

end.
