unit ProtoGen.FullType;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type


  TFullTypeId = (
    TFT_UNSET = 0,
    TFT_VAR = 1,
    TFT_ANY = 2,
    TFT_PRODUCT = 3,
    TFT_NAMED = 4,
    TFT_FOR_EACH = 20,
    TFT_CALLABLE = 100,
    TFT_TENSOR = 1000,
    TFT_ARRAY = 1001,
    TFT_OPTIONAL = 1002,
    TFT_LITERAL = 1003,
    TFT_BOOL = 200,
    TFT_UINT8 = 201,
    TFT_UINT16 = 202,
    TFT_UINT32 = 203,
    TFT_UINT64 = 204,
    TFT_INT8 = 205,
    TFT_INT16 = 206,
    TFT_INT32 = 207,
    TFT_INT64 = 208,
    TFT_HALF = 209,
    TFT_FLOAT = 210,
    TFT_DOUBLE = 211,
    TFT_BFLOAT16 = 215,
    TFT_COMPLEX64 = 212,
    TFT_COMPLEX128 = 213,
    TFT_STRING = 214,
    TFT_DATASET = 10102,
    TFT_RAGGED = 10103,
    TFT_MUTEX_LOCK = 10202,
    TFT_LEGACY_VARIANT = 10203);

  TFullTypeDef = Class
  const
    ftTypeId = 1;
    ftArgss = 2;
    ftS = 3;
    ftI = 4;
  private
    FTypeId: TFullTypeId;
    FArgss: TObjectList<TFullTypeDef>;
    FAttr: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property TypeId: TFullTypeId read FTypeId write FTypeId;
    property Argss: TObjectList<TFullTypeDef> read FArgss;
    property Attr: TpbOneof read FAttr write FAttr;
  end;

implementation

{ TFullTypeDef }

Constructor TFullTypeDef.Create;
begin
  inherited Create;
  
  FArgss := TObjectList<TFullTypeDef>.Create;
end;

destructor TFullTypeDef.Destroy;
begin
  FArgss.Free;
  inherited Destroy;
end;

end.
