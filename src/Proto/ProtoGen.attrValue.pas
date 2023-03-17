unit ProtoGen.AttrValue;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types,
  ProtoGen.resourcehandle;

{$T+}

type
  TAttrValue = Class;


  TStringAttrValue = TDictionary<string, TAttrValue>;

  TNameAttrList = class;

  TListValue = Class
  const
    ftSs = 2;
    ftIs = 3;
    ftFs = 4;
    ftBs = 5;
    ftTypes = 6;
    ftShapes = 7;
    ftTensors = 8;
    ftFuncs = 9;
  private
    FSs     : TList<TBytes>;
    FIs     : TList<Int64>;
    FFs     : TList<Single>;
    FBs     : TList<Boolean>;
    FTypes  : TList<TDataType>;
    FShapes : TList<TTensorShapeProto>;
    FTensors: TList<TTensorProto>;
    FFuncs  : TList<TNameAttrList>;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Ss: TList<TBytes> read FSs;
    property &Is: TList<Int64> read FIs;
    property Fs: TList<Single> read FFs;
    property Bs: TList<Boolean> read FBs;
    property Types: TList<TDataType> read FTypes;
    property Shapes: TList<TTensorShapeProto> read FShapes;
    property Tensors: TList<TTensorProto> read FTensors;
    property Funcs: TList<TNameAttrList> read FFuncs;
  end;

  TAttrValue = Class
  const
    ftS = 2;
    ftI = 3;
    ftF = 4;
    ftB = 5;
    ftType = 6;
    ftShape = 7;
    ftTensor = 8;
    ftList = 1;
    ftFunc = 10;
    ftPlaceholder = 9;
  private
    FValue: TpbOneof;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Value: TpbOneof read FValue write FValue;
  end;

  TNameAttrList = Class
  const
    ftName = 1;
    ftAttr = 2;
  private
    FName: string;
    FAttr: TStringAttrValue;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Name: string read FName write FName;
    property Attr: TStringAttrValue read FAttr write FAttr;
  end;

implementation

{ TListValue }

Constructor TListValue.Create;
begin
  inherited Create;
  
  FSs := TList<TBytes>.Create;
  
  FIs := TList<Int64>.Create;
  
  FFs := TList<Single>.Create;
  
  FBs := TList<Boolean>.Create;
  
  FTypes := TList<TDataType>.Create;
  
  FShapes := TList<TTensorShapeProto>.Create;
  
  FTensors := TList<TTensorProto>.Create;
  
  FFuncs := TList<TNameAttrList>.Create;
end;

destructor TListValue.Destroy;
begin
  FSs.Free;
  FIs.Free;
  FFs.Free;
  FBs.Free;
  FTypes.Free;
  FShapes.Free;
  FTensors.Free;
  FFuncs.Free;
  inherited Destroy;
end;

{ TAttrValue }

Constructor TAttrValue.Create;
begin
  inherited Create;
end;

destructor TAttrValue.Destroy;
begin
  inherited Destroy;
end;

{ TNameAttrList }

Constructor TNameAttrList.Create;
begin
  inherited Create;
  FAttr := TDictionary<string, TAttrValue>.Create;
end;

destructor TNameAttrList.Destroy;
begin
  FAttr.Free;
  inherited Destroy;
end;

end.
