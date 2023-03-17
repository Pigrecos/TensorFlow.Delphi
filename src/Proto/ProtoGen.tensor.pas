unit ProtoGen.Tensor;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.resourcehandle,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type

  TVariantTensorDataProto = class;

  TTensorProto = Class
  const
    ftDtype = 1;
    ftTensorShape = 2;
    ftVersionNumber = 3;
    ftTensorContent = 4;
    ftHalfVals = 13;
    ftFloatVals = 5;
    ftDoubleVals = 6;
    ftIntVals = 7;
    ftStringVals = 8;
    ftScomplexVals = 9;
    ftInt64Vals = 10;
    ftBoolVals = 11;
    ftDcomplexVals = 12;
    ftResourceHandleVals = 14;
    ftVariantVals = 15;
    ftUint32Vals = 16;
    ftUint64Vals = 17;
  private
    FDtype: TDataType;
    FTensorShape: TTensorShapeProto;
    FVersionNumber: Integer;
    FTensorContent: TBytes;
    FHalfVals: TList<Integer>;
    FFloatVals: TList<Single>;
    FDoubleVals: TList<Double>;
    FIntVals: TList<Integer>;
    FStringVals: TList<TBytes>;
    FScomplexVals: TList<Single>;
    FInt64Vals: TList<Int64>;
    FBoolVals: TList<Boolean>;
    FDcomplexVals: TList<Double>;
    FResourceHandleVals: TObjectList<TResourceHandleProto>;
    FVariantVals: TObjectList<TVariantTensorDataProto>;
    FUint32Vals: TList<UInt32>;
    FUint64Vals: TList<Int64>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property TensorShape: TTensorShapeProto read FTensorShape write FTensorShape;
    property VersionNumber: Integer read FVersionNumber write FVersionNumber;
    property TensorContent: TBytes read FTensorContent write FTensorContent;
    property HalfVals: TList<Integer> read FHalfVals;
    property FloatVals: TList<Single> read FFloatVals;
    property DoubleVals: TList<Double> read FDoubleVals;
    property IntVals: TList<Integer> read FIntVals;
    property StringVals: TList<TBytes> read FStringVals;
    property ScomplexVals: TList<Single> read FScomplexVals;
    property Int64Vals: TList<Int64> read FInt64Vals;
    property BoolVals: TList<Boolean> read FBoolVals;
    property DcomplexVals: TList<Double> read FDcomplexVals;
    property ResourceHandleVals: TObjectList<TResourceHandleProto> read FResourceHandleVals;
    property VariantVals: TObjectList<TVariantTensorDataProto> read FVariantVals;
    property Uint32Vals: TList<UInt32> read FUint32Vals;
    property Uint64Vals: TList<Int64> read FUint64Vals;
  end;

  TVariantTensorDataProto = Class
  const
    ftTypeName = 1;
    ftMetadata = 2;
    ftTensorss = 3;
  private
    FTypeName: string;
    FMetadata: TBytes;
    FTensorss: TObjectList<TTensorProto>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TypeName: string read FTypeName write FTypeName;
    property Metadata: TBytes read FMetadata write FMetadata;
    property Tensorss: TObjectList<TTensorProto> read FTensorss;
  end;

implementation

{ TTensorProto }

Constructor TTensorProto.Create;
begin
  inherited Create;
  
  FHalfVals := TList<Integer>.Create;
  
  FFloatVals := TList<Single>.Create;
  
  FDoubleVals := TList<Double>.Create;
  
  FIntVals := TList<Integer>.Create;
  
  FStringVals := TList<TBytes>.Create;
  
  FScomplexVals := TList<Single>.Create;
  
  FInt64Vals := TList<Int64>.Create;
  
  FBoolVals := TList<Boolean>.Create;
  
  FDcomplexVals := TList<Double>.Create;
  
  FResourceHandleVals := TObjectList<TResourceHandleProto>.Create;
  FVariantVals        := TObjectList<TVariantTensorDataProto>.Create;
  
  FUint32Vals := TList<UInt32>.Create;
  
  FUint64Vals := TList<Int64>.Create;
end;

destructor TTensorProto.Destroy;
begin
  FHalfVals.Free;
  FFloatVals.Free;
  FDoubleVals.Free;
  FIntVals.Free;
  FStringVals.Free;
  FScomplexVals.Free;
  FInt64Vals.Free;
  FBoolVals.Free;
  FDcomplexVals.Free;
  FResourceHandleVals.Free;
  FVariantVals.Free;
  FUint32Vals.Free;
  FUint64Vals.Free;
  inherited Destroy;
end;

{ TVariantTensorDataProto }

Constructor TVariantTensorDataProto.Create;
begin
  inherited Create;
  
  FTensorss := TObjectList<TTensorProto>.Create;
end;

destructor TVariantTensorDataProto.Destroy;
begin
  FTensorss.Free;
  inherited Destroy;
end;

end.
