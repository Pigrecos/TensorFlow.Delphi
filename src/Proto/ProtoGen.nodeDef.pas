unit ProtoGen.NodeDef;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes, oz.SGL.Heap, oz.SGL.Hash,
  Spring.Collections, spring.Collections.MultiMaps,
  ProtoGen.attrvalue,
  ProtoGen.resourcehandle,
  ProtoGen.fulltype,
  ProtoGen.tensor,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  TStringAttrValue = TMultiMap<string, TAttrValue>;

  PExperimentalDebugInfo = ^TExperimentalDebugInfo;
  TExperimentalDebugInfo = record
  const
    ftOriginalNodeNamess = 1;
    ftOriginalFuncNamess = 2;
  private
    FOriginalNodeNamess: TsgRecordList<string>;
    FOriginalFuncNamess: TsgRecordList<string>;
  public
    procedure Init;
    procedure Free;
    // properties
    property OriginalNodeNamess: TsgRecordList<string> read FOriginalNodeNamess;
    property OriginalFuncNamess: TsgRecordList<string> read FOriginalFuncNamess;
  end;

  PNodeDef = ^TNodeDef;
  TNodeDef = record
  const
    ftName = 1;
    ftOp = 2;
    ftInputs = 3;
    ftDevice = 4;
    ftAttr = 5;
    ftExperimentalDebugInfo = 6;
    ftExperimentalType = 7;
  private
    FName: string;
    FOp: string;
    FInputs: TsgRecordList<string>;
    FDevice: string;
    FAttr: TStringAttrValue;
    FExperimentalDebugInfo: TExperimentalDebugInfo;
    FExperimentalType: TFullTypeDef;
  public
    procedure Init;
    procedure Free;
    // properties
    property Name: string read FName write FName;
    property Op: string read FOp write FOp;
    property Inputs: TsgRecordList<string> read FInputs;
    property Device: string read FDevice write FDevice;
    property Attr: TStringAttrValue read FAttr write FAttr;
    property ExperimentalDebugInfo: TExperimentalDebugInfo read FExperimentalDebugInfo write FExperimentalDebugInfo;
    property ExperimentalType: TFullTypeDef read FExperimentalType write FExperimentalType;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadNodeDef(var Value: TNodeDef);
    procedure LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
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
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TPair<Key, Value>);
  private
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);

  public
    procedure SaveMap<Key, Value>(const Map: TMultiMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveNodeDef(const S: TpbSaver; const Value: TNodeDef); static;
    class procedure SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo); static;
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

{ TExperimentalDebugInfo }

procedure TExperimentalDebugInfo.Init;
begin
  Self := Default(TExperimentalDebugInfo);
  FOriginalNodeNamess := TsgRecordList<string>.From(nil);
  FOriginalFuncNamess := TsgRecordList<string>.From(nil);
end;

procedure TExperimentalDebugInfo.Free;
begin
  FOriginalNodeNamess.Free;
  FOriginalFuncNamess.Free;
end;

{ TNodeDef }

procedure TNodeDef.Init;
begin
  Self := Default(TNodeDef);

  var m := default(TsgItemMeta);
  m.Init<string>;

  FInputs := TsgRecordList<string>.From(@m);
  FAttr := TMultiMap<string, TAttrValue>.Create;
end;

procedure TNodeDef.Free;
begin
  FInputs.Free;
  FAttr.Free;
end;

procedure TLoadHelper.LoadExperimentalDebugInfo(var Value: TExperimentalDebugInfo);
var
  fieldNumber: integer;
  tag: TpbTag;
begin
  Value.Init;
  tag := Pb.readTag;
  while tag.v <> 0 do
  begin

    fieldNumber := tag.FieldNumber;
    case fieldNumber of
      TExperimentalDebugInfo.ftOriginalNodeNamess:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FOriginalNodeNamess.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TExperimentalDebugInfo.ftOriginalFuncNamess:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FOriginalFuncNamess.Add(@v);
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

procedure TLoadHelper.LoadNodeDef(var Value: TNodeDef);
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
      TNodeDef.ftName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Name := Pb.readString;
        end;
      TNodeDef.ftOp:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Op := Pb.readString;
        end;
      TNodeDef.ftInputs:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FInputs.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TNodeDef.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TNodeDef.ftAttr:
        begin
          var v1 : TAttrValue;
          LoadAttrValue(v1);
          Value.Attr.Add(Pb.readString, v1);
        end;
      TNodeDef.ftExperimentalDebugInfo:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TExperimentalDebugInfo := Value.FExperimentalDebugInfo;
            LoadExperimentalDebugInfo(v);
            Value.FExperimentalDebugInfo := v;
          finally
            Pb.Pop;
          end;
        end;
      TNodeDef.ftExperimentalType:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TFullTypeDef := Value.FExperimentalType;
            LoadFullTypeDef(v);
            Value.FExperimentalType := v;
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
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : TBytes := Pb.readBytes;
              Value.Ss.Add(@v);
            end
          finally
            Pb.Pop;
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
          v.value := TValue.From<TDataType>(TDataType(Pb.readInt32));
          Value.value := v;
        end;
      TAttrValue.ftShape:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftShape;
        
          var v1 : TTensorShapeProto;
          LoadTensorShapeProto(v1);
          v.value := TValue.From<TTensorShapeProto>(v1);
          Value.value := v;
        end;
      TAttrValue.ftTensor:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftTensor;
        
          var v1 : TTensorProto;
          LoadTensorProto(v1);
          v.value := TValue.From<TTensorProto>(v1);
          Value.value := v;
        end;
      TAttrValue.ftList:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftList;
        
          var v1 : TListValue;
          LoadListValue(v1);
          v.value := TValue.From<TListValue>(v1);
          Value.value := v;
        end;
      TAttrValue.ftFunc:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          var v : TpbOneof;
          v.tag := TAttrValue.ftFunc;
        
          var v1 : TNameAttrList;
          LoadNameAttrList(v1);
          v.value := TValue.From<TNameAttrList>(v1);
          Value.value := v;
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

procedure TSaveHelper.SaveMap<Key, Value>(const Map: TMultiMap<Key, Value>;
  Save: TSavePair<Key, Value>; Tag: Integer);
var
  h: TpbSaver;
  p: TPair<Key, Value>;
begin
  h.Init;
  try

    for p in  Map do
    begin

      h.Clear;
      Save(h, p);
      Pb.writeMessage(tag, h.Pb^);

    end;
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveExperimentalDebugInfo(const S: TpbSaver; const Value: TExperimentalDebugInfo);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    for i := 0 to Value.OriginalNodeNamess.Count - 1 do
      h.Pb.writeRawString(Value.FOriginalNodeNamess[i]^);
    S.Pb.writeMessage(TExperimentalDebugInfo.ftOriginalNodeNamess, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.OriginalFuncNamess.Count - 1 do
      h.Pb.writeRawString(Value.FOriginalFuncNamess[i]^);
    S.Pb.writeMessage(TExperimentalDebugInfo.ftOriginalFuncNamess, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveNodeDef(const S: TpbSaver; const Value: TNodeDef);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TNodeDef.ftName, Value.Name);
  S.Pb.writeString(TNodeDef.ftOp, Value.Op);
  h.Init;
  try
    for i := 0 to Value.Inputs.Count - 1 do
      h.Pb.writeRawString(Value.FInputs[i]^);
    S.Pb.writeMessage(TNodeDef.ftInputs, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeString(TNodeDef.ftDevice, Value.Device);
  if Value.FAttr.Count > 0 then
  begin
    h.Init;
    try

      for var p in Value.FAttr do
      begin
          h.clear;
          h.SaveStringAttrValue(p);
          S.Pb.writeMessage(TNodeDef.ftAttr, h.Pb^);
      end;
    finally
      h.Free;
    end;
  end;
  S.SaveObj<TExperimentalDebugInfo>(Value.FExperimentalDebugInfo, SaveExperimentalDebugInfo, TNodeDef.ftExperimentalDebugInfo);
  S.SaveObj<TFullTypeDef>(Value.FExperimentalType, SaveFullTypeDef, TNodeDef.ftExperimentalType);
end;

class procedure TSaveHelper.SaveListValue(const S: TpbSaver; const Value: TListValue);
var 
  i : Integer;
  h : TpbSaver;

begin
  h.Init;
  try
    if Value.Ss.Count > 0 then
    begin
        for i := 0 to Value.Ss.Count - 1 do
          h.Pb.writeRawBytes(Value.Ss[i]^);
        S.Pb.writeMessage(TListValue.ftSs, h.Pb^);
    end;
  finally
    h.Free;
  end;
  h.Init;
  try
    if Value.&Is.Count > 0 then
    begin
        for i := 0 to Value.&Is.Count - 1 do
          h.Pb.writeRawVarint64(Value.&Is[i]^);
        S.Pb.writeMessage(TListValue.ftIs, h.Pb^);
    end;
  finally
    h.Free;
  end;
  h.Init;
  try
    if Value.Fs.Count > 0 then
    begin
        for i := 0 to Value.Fs.Count - 1 do
          h.Pb.writeRawData(Value.Fs[i], sizeof(Single));
        S.Pb.writeMessage(TListValue.ftFs, h.Pb^);
    end;
  finally
    h.Free;
  end;
  h.Init;
  try
    if Value.Bs.Count > 0 then
    begin
        for i := 0 to Value.Bs.Count - 1 do
          h.Pb.writeRawVarint32(Integer(Value.Bs[i]^));
        S.Pb.writeMessage(TListValue.ftBs, h.Pb^);
    end;
  finally
    h.Free;
  end;
  h.Init;
  try
    if Value.&Types.Count > 0 then
    begin
        for i := 0 to Value.&Types.Count - 1 do
          h.Pb.writeRawVarint32(Ord(Value.&Types[i]^));
        S.Pb.writeMessage(TListValue.ftTypes, h.Pb^);
    end;
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
      h.Pb.writeRawData(Value.StringVals[i]^, Length(Value.StringVals[i]^));
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
