unit ProtoGen.OpDef;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,Oz.SGL.Heap,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.fulltype,
  ProtoGen.types,
  ProtoGen.tensor,
  ProtoGen.tensorshape;

{$T+}

type

  TOpDeprecation = class;

  PArgDef = ^TArgDef;
  TArgDef = record
  const
    ftName = 1;
    ftDescription = 2;
    ftType = 3;
    ftTypeAttr = 4;
    ftNumberAttr = 5;
    ftTypeListAttr = 6;
    ftHandleDatas = 7;
    ftIsRef = 16;
    ftExperimentalFullType = 17;
  private
    FName: string;
    FDescription: string;
    FType: TDataType;
    FTypeAttr: string;
    FNumberAttr: string;
    FTypeListAttr: string;
    FHandleDatas: TsgRecordList<TResourceHandleProto>;
    FIsRef: Boolean;
    FExperimentalFullType: TFullTypeDef;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property Description: string read FDescription write FDescription;
    property &Type: TDataType read FType write FType;
    property TypeAttr: string read FTypeAttr write FTypeAttr;
    property NumberAttr: string read FNumberAttr write FNumberAttr;
    property TypeListAttr: string read FTypeListAttr write FTypeListAttr;
    property HandleDatas: TsgRecordList<TResourceHandleProto> read FHandleDatas;
    property IsRef: Boolean read FIsRef write FIsRef;
    property ExperimentalFullType: TFullTypeDef read FExperimentalFullType write FExperimentalFullType;
  end;

  PAttrDef = ^TAttrDef;
  TAttrDef = record
  const
    ftName = 1;
    ftType = 2;
    ftDefaultValue = 3;
    ftDescription = 4;
    ftHasMinimum = 5;
    ftMinimum = 6;
    ftAllowedValues = 7;
  private
    FName: string;
    FType: string;
    FDefaultValue: TAttrValue;
    FDescription: string;
    FHasMinimum: Boolean;
    FMinimum: Int64;
    FAllowedValues: TAttrValue;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property &Type: string read FType write FType;
    property DefaultValue: TAttrValue read FDefaultValue write FDefaultValue;
    property Description: string read FDescription write FDescription;
    property HasMinimum: Boolean read FHasMinimum write FHasMinimum;
    property Minimum: Int64 read FMinimum write FMinimum;
    property AllowedValues: TAttrValue read FAllowedValues write FAllowedValues;
  end;

  POpDef = ^TOpDef;
  TOpDef = record
  const
    ftName = 1;
    ftInputArgs = 2;
    ftOutputArgs = 3;
    ftControlOutputs = 20;
    ftAttrs = 4;
    ftDeprecation = 8;
    ftSummary = 5;
    ftDescription = 6;
    ftIsCommutative = 18;
    ftIsAggregate = 16;
    ftIsStateful = 17;
    ftAllowsUninitializedInput = 19;
    ftIsDistributedCommunication = 21;
  private
    FName: string;
    FInputArgs: TsgRecordList<TArgDef>;
    FOutputArgs: TsgRecordList<TArgDef>;
    FControlOutputs: TsgRecordList<string>;
    FAttrs: TsgRecordList<TAttrDef>;
    FDeprecation: TOpDeprecation;
    FSummary: string;
    FDescription: string;
    FIsCommutative: Boolean;
    FIsAggregate: Boolean;
    FIsStateful: Boolean;
    FAllowsUninitializedInput: Boolean;
    FIsDistributedCommunication: Boolean;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property InputArgs: TsgRecordList<TArgDef> read FInputArgs;
    property OutputArgs: TsgRecordList<TArgDef> read FOutputArgs;
    property ControlOutputs: TsgRecordList<string> read FControlOutputs;
    property Attrs: TsgRecordList<TAttrDef> read FAttrs;
    property Deprecation: TOpDeprecation read FDeprecation write FDeprecation;
    property Summary: string read FSummary write FSummary;
    property Description: string read FDescription write FDescription;
    property IsCommutative: Boolean read FIsCommutative write FIsCommutative;
    property IsAggregate: Boolean read FIsAggregate write FIsAggregate;
    property IsStateful: Boolean read FIsStateful write FIsStateful;
    property AllowsUninitializedInput: Boolean read FAllowsUninitializedInput write FAllowsUninitializedInput;
    property IsDistributedCommunication: Boolean read FIsDistributedCommunication write FIsDistributedCommunication;
  end;

  TOpDeprecation = Class
  const
    ftVersion = 1;
    ftExplanation = 2;
  private
    FVersion: Integer;
    FExplanation: string;
  public
    Constructor Init;
    destructor Free;
    // properties
    property Version: Integer read FVersion write FVersion;
    property Explanation: string read FExplanation write FExplanation;
  end;

  POpList = ^TOpList;
  TOpList = record
  const
    ftOps = 1;
  private
    FOps: TsgRecordList<TOpDef>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Ops: TsgRecordList<TOpDef> read FOps;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadOpDef(var Value: TOpDef);
    procedure LoadArgDef(var Value: TArgDef);
    procedure LoadAttrDef(var Value: TAttrDef);
    procedure LoadOpDeprecation(var Value: TOpDeprecation);
    procedure LoadOpList(var Value: TOpList);
    procedure LoadAttrValue(var Value: TAttrValue);
    procedure LoadListValue(var Value: TListValue);
    procedure LoadNameAttrList(var Value: TNameAttrList);
    procedure LoadResourceHandleProto(var Value: TResourceHandleProto);
    procedure LoadDtypeAndShape(var Value: TDtypeAndShape);
    procedure LoadFullTypeDef(var Value: TFullTypeDef);
    procedure LoadTensorProto(var Value: TTensorProto);
    procedure LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
    procedure LoadTensorShapeProto(var Value: TTensorShapeProto);
    procedure LoadDim(var Value: TDim);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);

  public
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveOpDef(const S: TpbSaver; const Value: TOpDef); static;
    class procedure SaveArgDef(const S: TpbSaver; const Value: TArgDef); static;
    class procedure SaveAttrDef(const S: TpbSaver; const Value: TAttrDef); static;
    class procedure SaveOpDeprecation(const S: TpbSaver; const Value: TOpDeprecation); static;
    class procedure SaveOpList(const S: TpbSaver; const Value: TOpList); static;
    class procedure SaveAttrValue(const S: TpbSaver; const Value: TAttrValue); static;
    class procedure SaveListValue(const S: TpbSaver; const Value: TListValue); static;
    class procedure SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList); static;
    class procedure SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto); static;
    class procedure SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape); static;
    class procedure SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef); static;
    class procedure SaveTensorProto(const S: TpbSaver; const Value: TTensorProto); static;
    class procedure SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
    procedure SaveStringAttrValue(Item: TPair<string, TAttrValue>);
  end;

implementation
       uses Oz.Pb.StrBuffer;

{ TArgDef }

procedure TArgDef.Init;
begin
  Self := Default(TArgDef);

  var m : TsgItemMeta;
  m.Init<TResourceHandleProto>;
  FHandleDatas := TsgRecordList<TResourceHandleProto>.From(@m);
end;

procedure TArgDef.Free;
begin
  FHandleDatas.Free;
end;

{ TAttrDef }

procedure TAttrDef.Init;
begin
  Self := Default(TAttrDef);
end;

procedure TAttrDef.Free;
begin
end;

{ TOpDef }

procedure TOpDef.Init;
begin
  Self := Default(TOpDef);

  var m : TsgItemMeta;
  m.Init<TArgDef>;
  FInputArgs  := TsgRecordList<TArgDef>.From( @m );
  FOutputArgs := TsgRecordList<TArgDef>.From( @m );

  var s : TsgItemMeta;
  s.Init<string>;
  FControlOutputs := TsgRecordList<string>.From(@s);

  var n : TsgItemMeta;
  n.Init<TAttrDef>;
  FAttrs := TsgRecordList<TAttrDef>.From(@n);

  FDeprecation:= TOpDeprecation.Init;
end;

procedure TOpDef.Free;
begin
  FInputArgs.Free;
  FOutputArgs.Free;
  FControlOutputs.Free;
  FAttrs.Free;
end;

{ TOpDeprecation }

Constructor TOpDeprecation.Init;
begin
  inherited Create;
end;

destructor TOpDeprecation.Free;
begin
  inherited Destroy;
end;

{ TOpList }

procedure TOpList.Init;
begin
  Self := Default(TOpList);

  var n : TsgItemMeta;
  n.Init<TOpDef>;
  FOps := TsgRecordList<TOpDef>.From(@n);
end;

procedure TOpList.Free;
begin
  FOps.Free;
end;

procedure TLoadHelper.LoadArgDef(var Value: TArgDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TArgDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TArgDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TArgDef.ftType:
        begin
          Assert(wireType = TWire.VARINT);
          Value.&Type := TDataType(Pb.readInt32);
        end;
      TArgDef.ftTypeAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeAttr := Pb.readString;
        end;
      TArgDef.ftNumberAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NumberAttr := Pb.readString;
        end;
      TArgDef.ftTypeListAttr:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeListAttr := Pb.readString;
        end;
      TArgDef.ftHandleDatas:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TResourceHandleProto;
            LoadResourceHandleProto(v);
            Value.FHandleDatas.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TArgDef.ftIsRef:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsRef := Pb.readBoolean;
        end;
      TArgDef.ftExperimentalFullType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.FExperimentalFullType;
            LoadFullTypeDef(v);
            Value.FExperimentalFullType := v;
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadAttrDef(var Value: TAttrDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TAttrDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TAttrDef.ftType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.&Type := Pb.readString;
        end;
      TAttrDef.ftDefaultValue:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrValue := Value.FDefaultValue;
            LoadAttrValue(v);
            Value.FDefaultValue := v;
          finally
            Pb.Pop;
          end;
        end;
      TAttrDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TAttrDef.ftHasMinimum:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HasMinimum := Pb.readBoolean;
        end;
      TAttrDef.ftMinimum:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Minimum := Pb.readInt64;
        end;
      TAttrDef.ftAllowedValues:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrValue := Value.FAllowedValues;
            LoadAttrValue(v);
            Value.FAllowedValues := v;
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOpDef(var Value: TOpDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TOpDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TOpDef.ftInputArgs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TArgDef;
            LoadArgDef(v);
            Value.FInputArgs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftOutputArgs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TArgDef;
            LoadArgDef(v);
            Value.FOutputArgs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftControlOutputs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FControlOutputs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftAttrs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAttrDef;
            LoadAttrDef(v);
            Value.FAttrs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftDeprecation:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOpDeprecation := Value.FDeprecation;
            LoadOpDeprecation(v);
            Value.FDeprecation := v;
          finally
            Pb.Pop;
          end;
        end;
      TOpDef.ftSummary:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Summary := Pb.readString;
        end;
      TOpDef.ftDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Description := Pb.readString;
        end;
      TOpDef.ftIsCommutative:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsCommutative := Pb.readBoolean;
        end;
      TOpDef.ftIsAggregate:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsAggregate := Pb.readBoolean;
        end;
      TOpDef.ftIsStateful:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsStateful := Pb.readBoolean;
        end;
      TOpDef.ftAllowsUninitializedInput:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllowsUninitializedInput := Pb.readBoolean;
        end;
      TOpDef.ftIsDistributedCommunication:
        begin
          Assert(wireType = TWire.VARINT);
          Value.IsDistributedCommunication := Pb.readBoolean;
        end;
      else   begin
        Pb.skipField(tag);
        Pb.Push;
             end;
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOpDeprecation(var Value: TOpDeprecation);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TOpDeprecation.ftVersion:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Version := Pb.readInt32;
        end;
      TOpDeprecation.ftExplanation:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Explanation := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadOpList(var Value: TOpList);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TOpList.ftOps:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TOpDef;
            LoadOpDef(v);
            Value.FOps.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadListValue(var Value: TListValue);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TListValue.ftSs:
        begin
          //Pb.Push;
          try
            repeat

              var v : TBytes := Pb.readBytes;
              Value.Ss.Add(@v);
            until not Pb.ConsumeTag(tag.v);
          finally
            //Pb.Pop;
          end;
        end;
      TListValue.ftIs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.&Is.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftFs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.Fs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftBs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : boolean := Pb.readBoolean;
              Value.Bs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftTypes:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TDataType := TDataType(Pb.readInt32);
              Value.Types.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftShapes:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto;
            LoadTensorShapeProto(v);
            Value.Shapes.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftTensors:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorProto;
            LoadTensorProto(v);
            Value.Tensors.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TListValue.ftFuncs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNameAttrList;
            LoadNameAttrList(v);
            Value.Funcs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadAttrValue(var Value: TAttrValue);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TAttrValue.ftS:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftS;
          v.value := TValue.From<TBytes>(Pb.readBytes);
          Value.value := v;
        end;
      TAttrValue.ftI:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftI;
          v.value := Pb.readInt64;
          Value.value := v;
        end;
      TAttrValue.ftF:
        begin
          Assert(wireType = TWire.FIXED32);
          var v : TpbOneof;
          v.tag := TAttrValue.ftF;
          v.value := Pb.readFloat;
          Value.value := v;
        end;
      TAttrValue.ftB:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftB;
          v.value := Pb.readBoolean;
          Value.value := v;
        end;
      TAttrValue.ftType:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TAttrValue.ftType;
          v.value := TValue.From<Integer>(Pb.readInt32);
          Value.value := v;
        end;
      TAttrValue.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftShape;

          Pb.Push;
          try
            var v1 : TTensorShapeProto;
            LoadTensorShapeProto(v1);
            v.value := TValue.From<TTensorShapeProto>(v1);
            Value.value := v;
          finally
            Pb.Pop;
          end;
        end;
      TAttrValue.ftTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftTensor;
        
          Pb.Push;
          try
            var v1 : TTensorProto;
            LoadTensorProto(v1);
            v.value := TValue.From<TTensorProto>(v1);
            Value.value := v;
          finally
            Pb.Pop
          end;
        end;
      TAttrValue.ftList:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TpbOneof;
            v.tag := TAttrValue.ftList;

            var v1 : TListValue;
            LoadListValue(v1);
            v.value := TValue.From<TListValue>(v1);
            Value.value := v;
          finally
            Pb.Pop;
          end;
        end;
      TAttrValue.ftFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftFunc;

          Pb.Push;
          try
            var v1 : TNameAttrList;
            LoadNameAttrList(v1);
            v.value := TValue.From<TNameAttrList>(v1);
            Value.value := v;
          finally
            Pb.Pop;
          end;
        end;
      TAttrValue.ftPlaceholder:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftPlaceholder;
          v.value := Pb.readString;
          Value.value := v;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadNameAttrList(var Value: TNameAttrList);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TNameAttrList.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TNameAttrList.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDtypeAndShape(var Value: TDtypeAndShape);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TDtypeAndShape.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TDtypeAndShape.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.Shape;
            LoadTensorShapeProto(v);
            Value.Shape := v;
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadResourceHandleProto(var Value: TResourceHandleProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TResourceHandleProto.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TResourceHandleProto.ftContainer:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Container := Pb.readString;
        end;
      TResourceHandleProto.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TResourceHandleProto.ftHashCode:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HashCode := Pb.readInt64;
        end;
      TResourceHandleProto.ftMaybeTypeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.MaybeTypeName := Pb.readString;
        end;
      TResourceHandleProto.ftDtypesAndShapess:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDtypeAndShape;
            LoadDtypeAndShape(v);
            Value.DtypesAndShapess.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadFullTypeDef(var Value: TFullTypeDef);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TFullTypeDef.ftTypeId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TypeId := TFullTypeId(Pb.readInt32);
        end;
      TFullTypeDef.ftArgss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef;
            LoadFullTypeDef(v);
            Value.Argss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TFullTypeDef.ftS:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TFullTypeDef.ftS;
          v.value := Pb.readString;
          Value.attr := v;
        end;
      TFullTypeDef.ftI:
        begin
          Assert(wireType = TWire.VARINT);
          var v : TpbOneof;
          v.tag := TFullTypeDef.ftI;
          v.value := Pb.readInt64;
          Value.attr := v;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadTensorProto(var Value: TTensorProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TTensorProto.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TTensorProto.ftTensorShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorShapeProto := Value.TensorShape;
            LoadTensorShapeProto(v);
            Value.TensorShape := v;
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftVersionNumber:
        begin
          Assert(wireType = TWire.VARINT);
          Value.VersionNumber := Pb.readInt32;
        end;
      TTensorProto.ftTensorContent:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TensorContent := Pb.readBytes;
        end;
      TTensorProto.ftHalfVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.HalfVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftFloatVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.FloatVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftDoubleVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : double := Pb.readDouble;
              Value.DoubleVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftIntVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int32 := Pb.readInt32;
              Value.IntVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftStringVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TBytes := Pb.readBytes;
              Value.StringVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftScomplexVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : single := Pb.readFloat;
              Value.ScomplexVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftInt64Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.Int64Vals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftBoolVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : boolean := Pb.readBoolean;
              Value.BoolVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftDcomplexVals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : double := Pb.readDouble;
              Value.DcomplexVals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftResourceHandleVals:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TResourceHandleProto;
            LoadResourceHandleProto(v);
            Value.ResourceHandleVals.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftVariantVals:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TVariantTensorDataProto;
            LoadVariantTensorDataProto(v);
            Value.VariantVals.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftUint32Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : uint32 := Pb.readUint32;
              Value.Uint32Vals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TTensorProto.ftUint64Vals:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.Uint64Vals.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadVariantTensorDataProto(var Value: TVariantTensorDataProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TVariantTensorDataProto.ftTypeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TypeName := Pb.readString;
        end;
      TVariantTensorDataProto.ftMetadata:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Metadata := Pb.readBytes;
        end;
      TVariantTensorDataProto.ftTensorss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorProto;
            LoadTensorProto(v);
            Value.Tensorss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDim(var Value: TDim);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TDim.ftSize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Size := Pb.readInt64;
        end;
      TDim.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadTensorShapeProto(var Value: TTensorShapeProto);
var
  fieldNumber, wireType: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin
    wireType := tag.WireType;
    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TTensorShapeProto.ftDims:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDim;
            LoadDim(v);
            Value.Dims.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TTensorShapeProto.ftUnknownRank:
        begin
          Assert(wireType = TWire.VARINT);
          Value.UnknownRank := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

{ TSaveHelper }

procedure TSaveHelper.SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
var
  h: TpbSaver;
begin
  h.Init;
  try
    Save(h, obj);
    Pb.writeMessage(tag, h.Pb^);
  finally
    h.Free;
  end;
end;

procedure TSaveHelper.SaveList<T>(const List: TsgRecordList<T>;
  Save: TSave<T>; Tag: Integer);
var
  i: Integer;
  h: TpbSaver;
begin
  h.Init;
  try
    for i := 0 to List.Count - 1 do
    begin
      h.Clear;
      Save(h, List[i]^);
      Pb.writeMessage(tag, h.Pb^);
    end;
  finally
    h.Free;
  end;
end;

procedure TSaveHelper.SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>;
  Save: TSavePair<Key, Value>; Tag: Integer);
var
  h: TpbSaver;
  Pair: TsgHashMapIterator<Key, Value>.PPair;
  it: TsgHashMapIterator<Key, Value>;
begin
  h.Init;
  try
    it := Map.Begins;
    while it <> Map.Ends do
    begin
      h.Clear;
      Save(h, it.GetPair^);
      Pb.writeMessage(tag, h.Pb^);
      it.Next;
    end;
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveArgDef(const S: TpbSaver; const Value: TArgDef);
begin
  S.Pb.writeString(TArgDef.ftName, Value.Name);
  S.Pb.writeString(TArgDef.ftDescription, Value.Description);
  S.Pb.writeInt32(TArgDef.ftType, Ord(Value.&Type));
  S.Pb.writeString(TArgDef.ftTypeAttr, Value.TypeAttr);
  S.Pb.writeString(TArgDef.ftNumberAttr, Value.NumberAttr);
  S.Pb.writeString(TArgDef.ftTypeListAttr, Value.TypeListAttr);
  if Value.FHandleDatas.Count > 0 then
    S.SaveList<TResourceHandleProto>(Value.FHandleDatas, SaveResourceHandleProto, TArgDef.ftHandleDatas);
  S.Pb.writeBoolean(TArgDef.ftIsRef, Value.IsRef);
  S.SaveObj<TFullTypeDef>(Value.FExperimentalFullType, SaveFullTypeDef, TArgDef.ftExperimentalFullType);
end;

class procedure TSaveHelper.SaveAttrDef(const S: TpbSaver; const Value: TAttrDef);
begin
  S.Pb.writeString(TAttrDef.ftName, Value.Name);
  S.Pb.writeString(TAttrDef.ftType, Value.&Type);
  S.SaveObj<TAttrValue>(Value.FDefaultValue, SaveAttrValue, TAttrDef.ftDefaultValue);
  S.Pb.writeString(TAttrDef.ftDescription, Value.Description);
  S.Pb.writeBoolean(TAttrDef.ftHasMinimum, Value.HasMinimum);
  S.Pb.writeInt64(TAttrDef.ftMinimum, Value.Minimum);
  S.SaveObj<TAttrValue>(Value.FAllowedValues, SaveAttrValue, TAttrDef.ftAllowedValues);
end;

class procedure TSaveHelper.SaveOpDef(const S: TpbSaver; const Value: TOpDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TOpDef.ftName, Value.Name);
  if Value.FInputArgs.Count > 0 then
    S.SaveList<TArgDef>(Value.FInputArgs, SaveArgDef, TOpDef.ftInputArgs);
  if Value.FOutputArgs.Count > 0 then
    S.SaveList<TArgDef>(Value.FOutputArgs, SaveArgDef, TOpDef.ftOutputArgs);
  h.Init;
  try
    for i := 0 to Value.ControlOutputs.Count - 1 do
      h.Pb.writeRawString(Value.FControlOutputs[i]^);
    S.Pb.writeMessage(TOpDef.ftControlOutputs, h.Pb^);
  finally
    h.Free;
  end;
  if Value.FAttrs.Count > 0 then
    S.SaveList<TAttrDef>(Value.FAttrs, SaveAttrDef, TOpDef.ftAttrs);
  S.SaveObj<TOpDeprecation>(Value.FDeprecation, SaveOpDeprecation, TOpDef.ftDeprecation);
  S.Pb.writeString(TOpDef.ftSummary, Value.Summary);
  S.Pb.writeString(TOpDef.ftDescription, Value.Description);
  S.Pb.writeBoolean(TOpDef.ftIsCommutative, Value.IsCommutative);
  S.Pb.writeBoolean(TOpDef.ftIsAggregate, Value.IsAggregate);
  S.Pb.writeBoolean(TOpDef.ftIsStateful, Value.IsStateful);
  S.Pb.writeBoolean(TOpDef.ftAllowsUninitializedInput, Value.AllowsUninitializedInput);
  S.Pb.writeBoolean(TOpDef.ftIsDistributedCommunication, Value.IsDistributedCommunication);
end;

class procedure TSaveHelper.SaveOpDeprecation(const S: TpbSaver; const Value: TOpDeprecation);
begin
  S.Pb.writeInt32(TOpDeprecation.ftVersion, Value.Version);
  S.Pb.writeString(TOpDeprecation.ftExplanation, Value.Explanation);
end;

class procedure TSaveHelper.SaveOpList(const S: TpbSaver; const Value: TOpList);
begin
  if Value.FOps.Count > 0 then
    S.SaveList<TOpDef>(Value.FOps, SaveOpDef, TOpList.ftOps);
end;

class procedure TSaveHelper.SaveListValue(const S: TpbSaver; const Value: TListValue);
var
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.Ss.Count - 1 do
      h.Pb.writeRawBytes(Value.Ss[i]^);
    S.Pb.writeMessage(TListValue.ftSs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.&Is.Count - 1 do
      h.Pb.writeRawVarint64(Value.&Is[i]^);
    S.Pb.writeMessage(TListValue.ftIs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Fs.Count - 1 do
      h.Pb.writeRawData(Value.Fs[i], sizeof(Single));
    S.Pb.writeMessage(TListValue.ftFs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Bs.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.Bs[i]^));
    S.Pb.writeMessage(TListValue.ftBs, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.&Types.Count - 1 do
      h.Pb.writeRawVarint32(Ord(Value.&Types[i]^));
    S.Pb.writeMessage(TListValue.ftTypes, h.Pb^);
  finally
    h.Free;
  end;
  if Value.Shapes.Count > 0 then
    S.SaveList<TTensorShapeProto>(Value.Shapes, SaveTensorShapeProto, TListValue.ftShapes);
  if Value.Tensors.Count > 0 then
    S.SaveList<TTensorProto>(Value.Tensors, SaveTensorProto, TListValue.ftTensors);
  if Value.Funcs.Count > 0 then
    S.SaveList<TNameAttrList>(Value.Funcs, SaveNameAttrList, TListValue.ftFuncs);
end;

class procedure TSaveHelper.SaveAttrValue(const S: TpbSaver; const Value: TAttrValue);
begin
  case Value.value.tag of
    TAttrValue.ftS:
      begin
        S.Pb.writeBytes(Value.ftS, Value.Value.value.AsType<TBytes>);
      end;
    TAttrValue.ftI:
      begin
        S.Pb.writeInt64(Value.ftI, Value.Value.value.AsType<Int64>);
      end;
    TAttrValue.ftF:
      begin
        S.Pb.writeFloat(Value.ftF, Value.Value.value.AsType<Single>);
      end;
    TAttrValue.ftB:
      begin
        S.Pb.writeBoolean(Value.ftB, Value.Value.value.AsType<Boolean>);
      end;
    TAttrValue.ftType:
      begin
        S.Pb.writeInt32(Value.ftType, Value.Value.value.AsType<Integer>);
      end;
    TAttrValue.ftShape:
      begin
        S.SaveObj<TTensorShapeProto>(Value.Value.value.AsType<TTensorShapeProto>, SaveTensorShapeProto, Value.ftShape);
      end;
    TAttrValue.ftTensor:
      begin
        S.SaveObj<TTensorProto>(Value.Value.value.AsType<TTensorProto>, SaveTensorProto, Value.ftTensor);
      end;
    TAttrValue.ftList:
      begin
        S.SaveObj<TListValue>(Value.Value.value.AsType<TListValue>, SaveListValue, Value.ftList);
      end;
    TAttrValue.ftFunc:
      begin
        S.SaveObj<TNameAttrList>(Value.Value.value.AsType<TNameAttrList>, SaveNameAttrList, Value.ftFunc);
      end;
    TAttrValue.ftPlaceholder:
      begin
        S.Pb.writeString(Value.ftPlaceholder, Value.Value.value.AsType<string>);
      end;
  end;
end;

class procedure TSaveHelper.SaveNameAttrList(const S: TpbSaver; const Value: TNameAttrList);
var
  h : TpbSaver;

begin
  S.Pb.writeString(TNameAttrList.ftName, Value.Name);
  if Value.Attr.Count > 0 then
  begin
    h.Init;
    try
      for var p in Value.Attr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TNameAttrList.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveDtypeAndShape(const S: TpbSaver; const Value: TDtypeAndShape);
begin
  S.Pb.writeInt32(TDtypeAndShape.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TDtypeAndShape.ftShape);
end;

class procedure TSaveHelper.SaveResourceHandleProto(const S: TpbSaver; const Value: TResourceHandleProto);
begin
  S.Pb.writeString(TResourceHandleProto.ftDevice, Value.Device);
  S.Pb.writeString(TResourceHandleProto.ftContainer, Value.Container);
  S.Pb.writeString(TResourceHandleProto.ftName, Value.Name);
  S.Pb.writeInt64(TResourceHandleProto.ftHashCode, Value.HashCode);
  S.Pb.writeString(TResourceHandleProto.ftMaybeTypeName, Value.MaybeTypeName);
  if Value.DtypesAndShapess.Count > 0 then
    S.SaveList<TDtypeAndShape>(Value.DtypesAndShapess, SaveDtypeAndShape, TResourceHandleProto.ftDtypesAndShapess);
end;

class procedure TSaveHelper.SaveFullTypeDef(const S: TpbSaver; const Value: TFullTypeDef);
begin
  S.Pb.writeInt32(TFullTypeDef.ftTypeId, Ord(Value.TypeId));
  if Value.Argss.Count > 0 then
    S.SaveList<TFullTypeDef>(Value.Argss, SaveFullTypeDef, TFullTypeDef.ftArgss);
  case Value.attr.tag of
    TFullTypeDef.ftS:
      begin
        S.Pb.writeString(Value.ftS, Value.Attr.value.AsType<string>);
      end;
    TFullTypeDef.ftI:
      begin
        S.Pb.writeInt64(Value.ftI, Value.Attr.value.AsType<Int64>);
      end;
  end;
end;

class procedure TSaveHelper.SaveTensorProto(const S: TpbSaver; const Value: TTensorProto);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt32(TTensorProto.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.TensorShape, SaveTensorShapeProto, TTensorProto.ftTensorShape);
  S.Pb.writeInt32(TTensorProto.ftVersionNumber, Value.VersionNumber);
  S.Pb.writeBytes(TTensorProto.ftTensorContent, Value.TensorContent);
  h.Init;
  try
    for i := 0 to Value.HalfVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.HalfVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftHalfVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.FloatVals.Count - 1 do
      h.Pb.writeRawData(Value.FloatVals[i], sizeof(Single));
    S.Pb.writeMessage(TTensorProto.ftFloatVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DoubleVals.Count - 1 do
      h.Pb.writeRawData(Value.DoubleVals[i], sizeof(Double));
    S.Pb.writeMessage(TTensorProto.ftDoubleVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.IntVals.Count - 1 do
      h.Pb.writeRawVarint32(Value.IntVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftIntVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.StringVals.Count - 1 do
      h.Pb.writeRawBytes(Value.StringVals[i]^);
    S.Pb.writeMessage(TTensorProto.ftStringVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.ScomplexVals.Count - 1 do
      h.Pb.writeRawData(Value.ScomplexVals[i], sizeof(Single));
    S.Pb.writeMessage(TTensorProto.ftScomplexVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Int64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Int64Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftInt64Vals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.BoolVals.Count - 1 do
      h.Pb.writeRawVarint32(Integer(Value.BoolVals[i]^));
    S.Pb.writeMessage(TTensorProto.ftBoolVals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DcomplexVals.Count - 1 do
      h.Pb.writeRawData(Value.DcomplexVals[i], sizeof(Double));
    S.Pb.writeMessage(TTensorProto.ftDcomplexVals, h.Pb^);
  finally
    h.Free;
  end;
  if Value.ResourceHandleVals.Count > 0 then
    S.SaveList<TResourceHandleProto>(Value.ResourceHandleVals, SaveResourceHandleProto, TTensorProto.ftResourceHandleVals);
  if Value.VariantVals.Count > 0 then
    S.SaveList<TVariantTensorDataProto>(Value.VariantVals, SaveVariantTensorDataProto, TTensorProto.ftVariantVals);
  h.Init;
  try
    for i := 0 to Value.Uint32Vals.Count - 1 do
      h.Pb.writeRawVarint32(Value.Uint32Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftUint32Vals, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.Uint64Vals.Count - 1 do
      h.Pb.writeRawVarint64(Value.Uint64Vals[i]^);
    S.Pb.writeMessage(TTensorProto.ftUint64Vals, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveVariantTensorDataProto(const S: TpbSaver; const Value: TVariantTensorDataProto);
begin
  S.Pb.writeString(TVariantTensorDataProto.ftTypeName, Value.TypeName);
  S.Pb.writeBytes(TVariantTensorDataProto.ftMetadata, Value.Metadata);
  if Value.Tensorss.Count > 0 then
    S.SaveList<TTensorProto>(Value.Tensorss, SaveTensorProto, TVariantTensorDataProto.ftTensorss);
end;

class procedure TSaveHelper.SaveDim(const S: TpbSaver; const Value: TDim);
begin
  S.Pb.writeInt64(TDim.ftSize, Value.Size);
  S.Pb.writeString(TDim.ftName, Value.Name);
end;

class procedure TSaveHelper.SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto);
begin
  if Value.Dims.Count > 0 then
    S.SaveList<TDim>(Value.Dims, SaveDim, TTensorShapeProto.ftDims);
  S.Pb.writeBoolean(TTensorShapeProto.ftUnknownRank, Value.UnknownRank);
end;

procedure TSaveHelper.SaveStringAttrValue(Item: TPair<string, TAttrValue>);
begin
  Pb.writeString(1, Item.Key);
  SaveObj<TAttrValue>(Item.Value, SaveAttrValue, 2);
end;

end.
