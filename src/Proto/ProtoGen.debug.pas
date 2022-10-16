unit ProtoGen.Debug;

interface

uses
  System.Classes, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes;

{$T+}

type


  PDebugTensorWatch = ^TDebugTensorWatch;
  TDebugTensorWatch = record
  const
    ftNodeName = 1;
    ftOutputSlot = 2;
    ftDebugOpss = 3;
    ftDebugUrlss = 4;
    ftTolerateDebugOpCreationFailures = 5;
  private
    FNodeName: string;
    FOutputSlot: Integer;
    FDebugOpss: TsgRecordList<string>;
    FDebugUrlss: TsgRecordList<string>;
    FTolerateDebugOpCreationFailures: Boolean;
  public
    procedure Init;
    procedure Free;
    // properties
    property NodeName: string read FNodeName write FNodeName;
    property OutputSlot: Integer read FOutputSlot write FOutputSlot;
    property DebugOpss: TsgRecordList<string> read FDebugOpss;
    property DebugUrlss: TsgRecordList<string> read FDebugUrlss;
    property TolerateDebugOpCreationFailures: Boolean read FTolerateDebugOpCreationFailures write FTolerateDebugOpCreationFailures;
  end;

  PDebugOptions = ^TDebugOptions;
  TDebugOptions = record
  const
    ftDebugTensorWatchOptss = 4;
    ftGlobalStep = 10;
    ftResetDiskByteUsage = 11;
  private
    FDebugTensorWatchOptss: TsgRecordList<TDebugTensorWatch>;
    FGlobalStep: Int64;
    FResetDiskByteUsage: Boolean;
  public
    procedure Init;
    procedure Free;
    // properties
    property DebugTensorWatchOptss: TsgRecordList<TDebugTensorWatch> read FDebugTensorWatchOptss;
    property GlobalStep: Int64 read FGlobalStep write FGlobalStep;
    property ResetDiskByteUsage: Boolean read FResetDiskByteUsage write FResetDiskByteUsage;
  end;

  PDebuggedSourceFile = ^TDebuggedSourceFile;
  TDebuggedSourceFile = record
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
    FLiness: TsgRecordList<string>;
  public
    procedure Init;
    procedure Free;
    // properties
    property Host: string read FHost write FHost;
    property FilePath: string read FFilePath write FFilePath;
    property LastModified: Int64 read FLastModified write FLastModified;
    property Bytes: Int64 read FBytes write FBytes;
    property Liness: TsgRecordList<string> read FLiness;
  end;

  PDebuggedSourceFiles = ^TDebuggedSourceFiles;
  TDebuggedSourceFiles = record
  const
    ftSourceFiless = 1;
  private
    FSourceFiless: TsgRecordList<TDebuggedSourceFile>;
  public
    procedure Init;
    procedure Free;
    // properties
    property SourceFiless: TsgRecordList<TDebuggedSourceFile> read FSourceFiless;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadDebugTensorWatch(var Value: TDebugTensorWatch);
    procedure LoadDebugOptions(var Value: TDebugOptions);
    procedure LoadDebuggedSourceFile(var Value: TDebuggedSourceFile);
    procedure LoadDebuggedSourceFiles(var Value: TDebuggedSourceFiles);
  end;

  TSaveHelper = record helper for TpbSaver
  type
    TSave<T> = procedure(const S: TpbSaver; const Value: T);
    TSavePair<Key, Value> = procedure(const S: TpbSaver; const Pair: TsgPair<Key, Value>);
  private

    procedure SaveList<T>(const List: TsgRecordList<T>; Save: TSave<T>; Tag: Integer);
  public
    procedure SaveObj<T>(const obj: T; Save: TSave<T>; Tag: Integer);
    procedure SaveMap<Key, Value>(const Map: TsgHashMap<Key, Value>; Save: TSavePair<Key, Value>; Tag: Integer);

    class procedure SaveDebugTensorWatch(const S: TpbSaver; const Value: TDebugTensorWatch); static;
    class procedure SaveDebugOptions(const S: TpbSaver; const Value: TDebugOptions); static;
    class procedure SaveDebuggedSourceFile(const S: TpbSaver; const Value: TDebuggedSourceFile); static;
    class procedure SaveDebuggedSourceFiles(const S: TpbSaver; const Value: TDebuggedSourceFiles); static;
  end;

implementation

{ TDebugTensorWatch }

procedure TDebugTensorWatch.Init;
begin
  Self := Default(TDebugTensorWatch);
  FDebugOpss := TsgRecordList<string>.From(nil);
  FDebugUrlss := TsgRecordList<string>.From(nil);
end;

procedure TDebugTensorWatch.Free;
begin
  FDebugOpss.Free;
  FDebugUrlss.Free;
end;

{ TDebugOptions }

procedure TDebugOptions.Init;
begin
  Self := Default(TDebugOptions);
  FDebugTensorWatchOptss := TsgRecordList<TDebugTensorWatch>.From(nil);
end;

procedure TDebugOptions.Free;
begin
  FDebugTensorWatchOptss.Free;
end;

{ TDebuggedSourceFile }

procedure TDebuggedSourceFile.Init;
begin
  Self := Default(TDebuggedSourceFile);
  FLiness := TsgRecordList<string>.From(nil);
end;

procedure TDebuggedSourceFile.Free;
begin
  FLiness.Free;
end;

{ TDebuggedSourceFiles }

procedure TDebuggedSourceFiles.Init;
begin
  Self := Default(TDebuggedSourceFiles);
  FSourceFiless := TsgRecordList<TDebuggedSourceFile>.From(nil);
end;

procedure TDebuggedSourceFiles.Free;
begin
  FSourceFiless.Free;
end;

procedure TLoadHelper.LoadDebugTensorWatch(var Value: TDebugTensorWatch);
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
      TDebugTensorWatch.ftNodeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NodeName := Pb.readString;
        end;
      TDebugTensorWatch.ftOutputSlot:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OutputSlot := Pb.readInt32;
        end;
      TDebugTensorWatch.ftDebugOpss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FDebugOpss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TDebugTensorWatch.ftDebugUrlss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FDebugUrlss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TDebugTensorWatch.ftTolerateDebugOpCreationFailures:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TolerateDebugOpCreationFailures := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDebugOptions(var Value: TDebugOptions);
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
      TDebugOptions.ftDebugTensorWatchOptss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDebugTensorWatch;
            LoadDebugTensorWatch(v);
            Value.FDebugTensorWatchOptss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TDebugOptions.ftGlobalStep:
        begin
          Assert(wireType = TWire.VARINT);
          Value.GlobalStep := Pb.readInt64;
        end;
      TDebugOptions.ftResetDiskByteUsage:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ResetDiskByteUsage := Pb.readBoolean;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDebuggedSourceFile(var Value: TDebuggedSourceFile);
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
      TDebuggedSourceFile.ftHost:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Host := Pb.readString;
        end;
      TDebuggedSourceFile.ftFilePath:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.FilePath := Pb.readString;
        end;
      TDebuggedSourceFile.ftLastModified:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LastModified := Pb.readInt64;
        end;
      TDebuggedSourceFile.ftBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Bytes := Pb.readInt64;
        end;
      TDebuggedSourceFile.ftLiness:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : string := Pb.readString;
              Value.FLiness.Add(@v);
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

procedure TLoadHelper.LoadDebuggedSourceFiles(var Value: TDebuggedSourceFiles);
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
      TDebuggedSourceFiles.ftSourceFiless:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDebuggedSourceFile;
            LoadDebuggedSourceFile(v);
            Value.FSourceFiless.Add(@v);
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

class procedure TSaveHelper.SaveDebugTensorWatch(const S: TpbSaver; const Value: TDebugTensorWatch);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TDebugTensorWatch.ftNodeName, Value.NodeName);
  S.Pb.writeInt32(TDebugTensorWatch.ftOutputSlot, Value.OutputSlot);
  h.Init;
  try
    for i := 0 to Value.DebugOpss.Count - 1 do
      h.Pb.writeRawString(Value.FDebugOpss[i]^);
    S.Pb.writeMessage(TDebugTensorWatch.ftDebugOpss, h.Pb^);
  finally
    h.Free;
  end;
  h.Init;
  try
    for i := 0 to Value.DebugUrlss.Count - 1 do
      h.Pb.writeRawString(Value.FDebugUrlss[i]^);
    S.Pb.writeMessage(TDebugTensorWatch.ftDebugUrlss, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeBoolean(TDebugTensorWatch.ftTolerateDebugOpCreationFailures, Value.TolerateDebugOpCreationFailures);
end;

class procedure TSaveHelper.SaveDebugOptions(const S: TpbSaver; const Value: TDebugOptions);
begin
  if Value.FDebugTensorWatchOptss.Count > 0 then
    S.SaveList<TDebugTensorWatch>(Value.FDebugTensorWatchOptss, SaveDebugTensorWatch, TDebugOptions.ftDebugTensorWatchOptss);
  S.Pb.writeInt64(TDebugOptions.ftGlobalStep, Value.GlobalStep);
  S.Pb.writeBoolean(TDebugOptions.ftResetDiskByteUsage, Value.ResetDiskByteUsage);
end;

class procedure TSaveHelper.SaveDebuggedSourceFile(const S: TpbSaver; const Value: TDebuggedSourceFile);
var
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeString(TDebuggedSourceFile.ftHost, Value.Host);
  S.Pb.writeString(TDebuggedSourceFile.ftFilePath, Value.FilePath);
  S.Pb.writeInt64(TDebuggedSourceFile.ftLastModified, Value.LastModified);
  S.Pb.writeInt64(TDebuggedSourceFile.ftBytes, Value.Bytes);
  h.Init;
  try
    for i := 0 to Value.Liness.Count - 1 do
      h.Pb.writeRawString(Value.FLiness[i]^);
    S.Pb.writeMessage(TDebuggedSourceFile.ftLiness, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveDebuggedSourceFiles(const S: TpbSaver; const Value: TDebuggedSourceFiles);
begin
  if Value.FSourceFiless.Count > 0 then
    S.SaveList<TDebuggedSourceFile>(Value.FSourceFiless, SaveDebuggedSourceFile, TDebuggedSourceFiles.ftSourceFiless);
end;

end.
