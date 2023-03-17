unit ProtoGen.CppShapeInference;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.fulltype,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  THandleShapeAndType = Class
  const
    ftShape = 1;
    ftDtype = 2;
    ftType = 4;
  private
    FShape: TTensorShapeProto;
    FDtype: TDataType;
    FType: TFullTypeDef;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property Dtype: TDataType read FDtype write FDtype;
    property &Type: TFullTypeDef read FType write FType;
  end;

  THandleData = Class
  const
    ftIsSet = 1;
    ftShapeAndTypes = 2;
  private
    FIsSet: Boolean;
    FShapeAndTypes: TList<THandleShapeAndType>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property IsSet: Boolean read FIsSet write FIsSet;
    property ShapeAndTypes: TList<THandleShapeAndType> read FShapeAndTypes;
  end;

  TCppShapeInferenceResult = Class
  const
    ftShape = 1;
    ftHandleData = 4;
  private
    FShape: TTensorShapeProto;
    FHandleData: THandleData;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Shape: TTensorShapeProto read FShape write FShape;
    property HandleData: THandleData read FHandleData write FHandleData;
  end;

  TCppShapeInferenceInputsNeeded = Class
  const
    ftInputTensorsNeededs = 1;
    ftInputTensorsAsShapesNeededs = 2;
  private
    FInputTensorsNeededs: TList<Integer>;
    FInputTensorsAsShapesNeededs: TList<Integer>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property InputTensorsNeededs: TList<Integer> read FInputTensorsNeededs;
    property InputTensorsAsShapesNeededs: TList<Integer> read FInputTensorsAsShapesNeededs;
  end;

implementation

{ THandleShapeAndType }

Constructor THandleShapeAndType.Create;
begin
  inherited Create;
end;

destructor THandleShapeAndType.Destroy;
begin
  inherited Destroy;
end;

{ THandleData }

Constructor THandleData.Create;
begin
  inherited Create;
  
  FShapeAndTypes := TList<THandleShapeAndType>.Create;
end;

destructor THandleData.Destroy;
begin
  FShapeAndTypes.Free;
  inherited Destroy;
end;

{ TCppShapeInferenceResult }

Constructor TCppShapeInferenceResult.Create;
begin
  inherited Create;
end;

destructor TCppShapeInferenceResult.Destroy;
begin
  inherited Destroy;
end;

{ TCppShapeInferenceInputsNeeded }

Constructor TCppShapeInferenceInputsNeeded.Create;
begin
  inherited Create;
  
  FInputTensorsNeededs := TList<Integer>.Create;
  
  FInputTensorsAsShapesNeededs := TList<Integer>.Create;
end;

destructor TCppShapeInferenceInputsNeeded.Destroy;
begin
  FInputTensorsNeededs.Free;
  FInputTensorsAsShapesNeededs.Free;
  inherited Destroy;
end;

end.
