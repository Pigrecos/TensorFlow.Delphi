unit ProtoGen.ResourceHandle;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  TDtypeAndShape = Class
  const
    ftDtype = 1;
    ftShape = 2;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
  end;

  TResourceHandleProto = Class
  const
    ftDevice = 1;
    ftContainer = 2;
    ftName = 3;
    ftHashCode = 4;
    ftMaybeTypeName = 5;
    ftDtypesAndShapess = 6;
  private
    FDevice: string;
    FContainer: string;
    FName: string;
    FHashCode: Int64;
    FMaybeTypeName: string;
    FDtypesAndShapess: TList<TDtypeAndShape>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Device: string read FDevice write FDevice;
    property Container: string read FContainer write FContainer;
    property Name: string read FName write FName;
    property HashCode: Int64 read FHashCode write FHashCode;
    property MaybeTypeName: string read FMaybeTypeName write FMaybeTypeName;
    property DtypesAndShapess: TList<TDtypeAndShape> read FDtypesAndShapess;
  end;

implementation

{ TDtypeAndShape }

Constructor TDtypeAndShape.Create;
begin
  inherited Create;
end;

destructor TDtypeAndShape.Destroy;
begin
  inherited Destroy;
end;

{ TResourceHandleProto }

Constructor TResourceHandleProto.Create;
begin
  inherited Create;
  
  FDtypesAndShapess := TList<TDtypeAndShape>.Create;
end;

destructor TResourceHandleProto.Destroy;
begin
  FDtypesAndShapess.Free;
  inherited Destroy;
end;

end.
