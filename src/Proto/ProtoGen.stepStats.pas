unit ProtoGen.StepStats;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Oz.SGL.Collections, Oz.Pb.Classes,
  ProtoGen.allocationdescription,
  ProtoGen.tensordescription,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type


  PAllocationRecord = ^TAllocationRecord;
  TAllocationRecord = record
  const
    ftAllocMicros = 1;
    ftAllocBytes = 2;
  private
    FAllocMicros: Int64;
    FAllocBytes: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property AllocMicros: Int64 read FAllocMicros write FAllocMicros;
    property AllocBytes: Int64 read FAllocBytes write FAllocBytes;
  end;

  PAllocatorMemoryUsed = ^TAllocatorMemoryUsed;
  TAllocatorMemoryUsed = record
  const
    ftAllocatorName = 1;
    ftTotalBytes = 2;
    ftPeakBytes = 3;
    ftLiveBytes = 4;
    ftAllocationRecordss = 6;
    ftAllocatorBytesInUse = 5;
  private
    FAllocatorName: string;
    FTotalBytes: Int64;
    FPeakBytes: Int64;
    FLiveBytes: Int64;
    FAllocationRecordss: TsgRecordList<TAllocationRecord>;
    FAllocatorBytesInUse: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property AllocatorName: string read FAllocatorName write FAllocatorName;
    property TotalBytes: Int64 read FTotalBytes write FTotalBytes;
    property PeakBytes: Int64 read FPeakBytes write FPeakBytes;
    property LiveBytes: Int64 read FLiveBytes write FLiveBytes;
    property AllocationRecordss: TsgRecordList<TAllocationRecord> read FAllocationRecordss;
    property AllocatorBytesInUse: Int64 read FAllocatorBytesInUse write FAllocatorBytesInUse;
  end;

  PNodeOutput = ^TNodeOutput;
  TNodeOutput = record
  const
    ftSlot = 1;
    ftTensorDescription = 3;
  private
    FSlot: Integer;
    FTensorDescription: TTensorDescription;
  public
    procedure Init;
    procedure Free;
    // properties
    property Slot: Integer read FSlot write FSlot;
    property TensorDescription: TTensorDescription read FTensorDescription write FTensorDescription;
  end;

  PMemoryStats = ^TMemoryStats;
  TMemoryStats = record
  const
    ftTempMemorySize = 1;
    ftPersistentMemorySize = 3;
    ftPersistentTensorAllocIdss = 5;
    ftDeviceTempMemorySize = 2;
    ftDevicePersistentMemorySize = 4;
    ftDevicePersistentTensorAllocIdss = 6;
  private
    FTempMemorySize: Int64;
    FPersistentMemorySize: Int64;
    FPersistentTensorAllocIdss: TsgRecordList<Int64>;
    FDeviceTempMemorySize: Int64;
    FDevicePersistentMemorySize: Int64;
    FDevicePersistentTensorAllocIdss: TsgRecordList<Int64>;
  public
    procedure Init;
    procedure Free;
    // properties
    property TempMemorySize: Int64 read FTempMemorySize write FTempMemorySize;
    property PersistentMemorySize: Int64 read FPersistentMemorySize write FPersistentMemorySize;
    property PersistentTensorAllocIdss: TsgRecordList<Int64> read FPersistentTensorAllocIdss;
    property DeviceTempMemorySize: Int64 read FDeviceTempMemorySize write FDeviceTempMemorySize;
    property DevicePersistentMemorySize: Int64 read FDevicePersistentMemorySize write FDevicePersistentMemorySize;
    property DevicePersistentTensorAllocIdss: TsgRecordList<Int64> read FDevicePersistentTensorAllocIdss;
  end;

  PNodeExecStats = ^TNodeExecStats;
  TNodeExecStats = record
  const
    ftNodeName = 1;
    ftAllStartMicros = 2;
    ftOpStartRelMicros = 3;
    ftOpEndRelMicros = 4;
    ftAllEndRelMicros = 5;
    ftMemorys = 6;
    ftOutputs = 7;
    ftTimelineLabel = 8;
    ftScheduledMicros = 9;
    ftThreadId = 10;
    ftReferencedTensors = 11;
    ftMemoryStats = 12;
    ftAllStartNanos = 13;
    ftOpStartRelNanos = 14;
    ftOpEndRelNanos = 15;
    ftAllEndRelNanos = 16;
    ftScheduledNanos = 17;
  private
    FNodeName: string;
    FAllStartMicros: Int64;
    FOpStartRelMicros: Int64;
    FOpEndRelMicros: Int64;
    FAllEndRelMicros: Int64;
    FMemorys: TsgRecordList<TAllocatorMemoryUsed>;
    FOutputs: TsgRecordList<TNodeOutput>;
    FTimelineLabel: string;
    FScheduledMicros: Int64;
    FThreadId: UInt32;
    FReferencedTensors: TsgRecordList<TAllocationDescription>;
    FMemoryStats: TMemoryStats;
    FAllStartNanos: Int64;
    FOpStartRelNanos: Int64;
    FOpEndRelNanos: Int64;
    FAllEndRelNanos: Int64;
    FScheduledNanos: Int64;
  public
    procedure Init;
    procedure Free;
    // properties
    property NodeName: string read FNodeName write FNodeName;
    property AllStartMicros: Int64 read FAllStartMicros write FAllStartMicros;
    property OpStartRelMicros: Int64 read FOpStartRelMicros write FOpStartRelMicros;
    property OpEndRelMicros: Int64 read FOpEndRelMicros write FOpEndRelMicros;
    property AllEndRelMicros: Int64 read FAllEndRelMicros write FAllEndRelMicros;
    property Memorys: TsgRecordList<TAllocatorMemoryUsed> read FMemorys;
    property Outputs: TsgRecordList<TNodeOutput> read FOutputs;
    property TimelineLabel: string read FTimelineLabel write FTimelineLabel;
    property ScheduledMicros: Int64 read FScheduledMicros write FScheduledMicros;
    property ThreadId: UInt32 read FThreadId write FThreadId;
    property ReferencedTensors: TsgRecordList<TAllocationDescription> read FReferencedTensors;
    property MemoryStats: TMemoryStats read FMemoryStats write FMemoryStats;
    property AllStartNanos: Int64 read FAllStartNanos write FAllStartNanos;
    property OpStartRelNanos: Int64 read FOpStartRelNanos write FOpStartRelNanos;
    property OpEndRelNanos: Int64 read FOpEndRelNanos write FOpEndRelNanos;
    property AllEndRelNanos: Int64 read FAllEndRelNanos write FAllEndRelNanos;
    property ScheduledNanos: Int64 read FScheduledNanos write FScheduledNanos;
  end;

  TUint32String = TsgHashMap<UInt32, string>;

  PDeviceStepStats = ^TDeviceStepStats;
  TDeviceStepStats = record
  const
    ftDevice = 1;
    ftNodeStatss = 2;
    ftThreadNames = 3;
  private
    FDevice: string;
    FNodeStatss: TsgRecordList<TNodeExecStats>;
    FThreadNames: TUint32String;
  public
    procedure Init;
    procedure Free;
    // properties
    property Device: string read FDevice write FDevice;
    property NodeStatss: TsgRecordList<TNodeExecStats> read FNodeStatss;
    property ThreadNames: TUint32String read FThreadNames write FThreadNames;
  end;

  PStepStats = ^TStepStats;
  TStepStats = record
  const
    ftDevStatss = 1;
  private
    FDevStatss: TsgRecordList<TDeviceStepStats>;
  public
    procedure Init;
    procedure Free;
    // properties
    property DevStatss: TsgRecordList<TDeviceStepStats> read FDevStatss;
  end;

  TLoadHelper = record helper for TpbLoader
  public
    procedure LoadAllocationRecord(var Value: TAllocationRecord);
    procedure LoadAllocatorMemoryUsed(var Value: TAllocatorMemoryUsed);
    procedure LoadNodeOutput(var Value: TNodeOutput);
    procedure LoadMemoryStats(var Value: TMemoryStats);
    procedure LoadNodeExecStats(var Value: TNodeExecStats);
    procedure LoadDeviceStepStats(var Value: TDeviceStepStats);
    procedure LoadStepStats(var Value: TStepStats);
    procedure LoadAllocationDescription(var Value: TAllocationDescription);
    procedure LoadTensorDescription(var Value: TTensorDescription);
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

    class procedure SaveAllocationRecord(const S: TpbSaver; const Value: TAllocationRecord); static;
    class procedure SaveAllocatorMemoryUsed(const S: TpbSaver; const Value: TAllocatorMemoryUsed); static;
    class procedure SaveNodeOutput(const S: TpbSaver; const Value: TNodeOutput); static;
    class procedure SaveMemoryStats(const S: TpbSaver; const Value: TMemoryStats); static;
    class procedure SaveNodeExecStats(const S: TpbSaver; const Value: TNodeExecStats); static;
    class procedure SaveDeviceStepStats(const S: TpbSaver; const Value: TDeviceStepStats); static;
    class procedure SaveStepStats(const S: TpbSaver; const Value: TStepStats); static;
    class procedure SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription); static;
    class procedure SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription); static;
    class procedure SaveTensorShapeProto(const S: TpbSaver; const Value: TTensorShapeProto); static;
    class procedure SaveDim(const S: TpbSaver; const Value: TDim); static;
    procedure SaveUint32String(Item: TsgPair<UInt32, string>);
  end;

implementation

{ TAllocationRecord }

procedure TAllocationRecord.Init;
begin
  Self := Default(TAllocationRecord);
end;

procedure TAllocationRecord.Free;
begin
end;

{ TAllocatorMemoryUsed }

procedure TAllocatorMemoryUsed.Init;
begin
  Self := Default(TAllocatorMemoryUsed);
  FAllocationRecordss := TsgRecordList<TAllocationRecord>.From(nil);
end;

procedure TAllocatorMemoryUsed.Free;
begin
  FAllocationRecordss.Free;
end;

{ TNodeOutput }

procedure TNodeOutput.Init;
begin
  Self := Default(TNodeOutput);
end;

procedure TNodeOutput.Free;
begin
end;

{ TMemoryStats }

procedure TMemoryStats.Init;
begin
  Self := Default(TMemoryStats);
  FPersistentTensorAllocIdss := TsgRecordList<Int64>.From(nil);
  FDevicePersistentTensorAllocIdss := TsgRecordList<Int64>.From(nil);
end;

procedure TMemoryStats.Free;
begin
  FPersistentTensorAllocIdss.Free;
  FDevicePersistentTensorAllocIdss.Free;
end;

{ TNodeExecStats }

procedure TNodeExecStats.Init;
begin
  Self := Default(TNodeExecStats);
  FMemorys := TsgRecordList<TAllocatorMemoryUsed>.From(nil);
  FOutputs := TsgRecordList<TNodeOutput>.From(nil);
  FReferencedTensors := TsgRecordList<TAllocationDescription>.From(nil);
end;

procedure TNodeExecStats.Free;
begin
  FMemorys.Free;
  FOutputs.Free;
  FReferencedTensors.Free;
end;

{ TDeviceStepStats }

procedure TDeviceStepStats.Init;
begin
  Self := Default(TDeviceStepStats);
  FNodeStatss := TsgRecordList<TNodeExecStats>.From(nil);
  FThreadNames := TsgHashMap<UInt32, string>.From(0,nil);
end;

procedure TDeviceStepStats.Free;
begin
  FNodeStatss.Free;
  FThreadNames.Free;
end;

{ TStepStats }

procedure TStepStats.Init;
begin
  Self := Default(TStepStats);
  FDevStatss := TsgRecordList<TDeviceStepStats>.From(nil);
end;

procedure TStepStats.Free;
begin
  FDevStatss.Free;
end;

procedure TLoadHelper.LoadAllocationRecord(var Value: TAllocationRecord);
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
      TAllocationRecord.ftAllocMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocMicros := Pb.readInt64;
        end;
      TAllocationRecord.ftAllocBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocBytes := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadAllocatorMemoryUsed(var Value: TAllocatorMemoryUsed);
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
      TAllocatorMemoryUsed.ftAllocatorName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorName := Pb.readString;
        end;
      TAllocatorMemoryUsed.ftTotalBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TotalBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftPeakBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PeakBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftLiveBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.LiveBytes := Pb.readInt64;
        end;
      TAllocatorMemoryUsed.ftAllocationRecordss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationRecord;
            LoadAllocationRecord(v);
            Value.FAllocationRecordss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TAllocatorMemoryUsed.ftAllocatorBytesInUse:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocatorBytesInUse := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadNodeOutput(var Value: TNodeOutput);
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
      TNodeOutput.ftSlot:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Slot := Pb.readInt32;
        end;
      TNodeOutput.ftTensorDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TTensorDescription := Value.FTensorDescription;
            LoadTensorDescription(v);
            Value.FTensorDescription := v;
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

procedure TLoadHelper.LoadMemoryStats(var Value: TMemoryStats);
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
      TMemoryStats.ftTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.TempMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftPersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.PersistentMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftPersistentTensorAllocIdss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.FPersistentTensorAllocIdss.Add(@v);
            end
          finally
            Pb.Pop;
          end;
        end;
      TMemoryStats.ftDeviceTempMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DeviceTempMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftDevicePersistentMemorySize:
        begin
          Assert(wireType = TWire.VARINT);
          Value.DevicePersistentMemorySize := Pb.readInt64;
        end;
      TMemoryStats.ftDevicePersistentTensorAllocIdss:
        begin
          Pb.Push;
          try
            while not Pb.Eof do
            begin
              var v : int64 := Pb.readInt64;
              Value.FDevicePersistentTensorAllocIdss.Add(@v);
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

procedure TLoadHelper.LoadNodeExecStats(var Value: TNodeExecStats);
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
      TNodeExecStats.ftNodeName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.NodeName := Pb.readString;
        end;
      TNodeExecStats.ftAllStartMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllStartMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftOpStartRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpStartRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftOpEndRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpEndRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftAllEndRelMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllEndRelMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftMemorys:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocatorMemoryUsed;
            LoadAllocatorMemoryUsed(v);
            Value.FMemorys.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftOutputs:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeOutput;
            LoadNodeOutput(v);
            Value.FOutputs.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftTimelineLabel:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.TimelineLabel := Pb.readString;
        end;
      TNodeExecStats.ftScheduledMicros:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScheduledMicros := Pb.readInt64;
        end;
      TNodeExecStats.ftThreadId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ThreadId := Pb.readUint32;
        end;
      TNodeExecStats.ftReferencedTensors:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationDescription;
            LoadAllocationDescription(v);
            Value.FReferencedTensors.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftMemoryStats:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TMemoryStats := Value.FMemoryStats;
            LoadMemoryStats(v);
            Value.FMemoryStats := v;
          finally
            Pb.Pop;
          end;
        end;
      TNodeExecStats.ftAllStartNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllStartNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftOpStartRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpStartRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftOpEndRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.OpEndRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftAllEndRelNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllEndRelNanos := Pb.readInt64;
        end;
      TNodeExecStats.ftScheduledNanos:
        begin
          Assert(wireType = TWire.VARINT);
          Value.ScheduledNanos := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadDeviceStepStats(var Value: TDeviceStepStats);
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
      TDeviceStepStats.ftDevice:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.Device := Pb.readString;
        end;
      TDeviceStepStats.ftNodeStatss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TNodeExecStats;
            LoadNodeExecStats(v);
            Value.FNodeStatss.Add(@v);
          finally
            Pb.Pop;
          end;
        end;
      TDeviceStepStats.ftThreadNames:
        begin
          Value.ThreadNames.InsertOrAssign(TsgPair<UInt32, string>.From(Pb.readUint32, Pb.readString));
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadStepStats(var Value: TStepStats);
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
      TStepStats.ftDevStatss:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TDeviceStepStats;
            LoadDeviceStepStats(v);
            Value.FDevStatss.Add(@v);
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

procedure TLoadHelper.LoadAllocationDescription(var Value: TAllocationDescription);
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
      TAllocationDescription.ftRequestedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.RequestedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatedBytes:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocatedBytes := Pb.readInt64;
        end;
      TAllocationDescription.ftAllocatorName:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Value.AllocatorName := Pb.readString;
        end;
      TAllocationDescription.ftAllocationId:
        begin
          Assert(wireType = TWire.VARINT);
          Value.AllocationId := Pb.readInt64;
        end;
      TAllocationDescription.ftHasSingleReference:
        begin
          Assert(wireType = TWire.VARINT);
          Value.HasSingleReference := Pb.readBoolean;
        end;
      TAllocationDescription.ftPtr:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Ptr := Pb.readInt64;
        end;
      else
        Pb.skipField(tag);
    end;
    tag := Pb.readTag;
  end;
end;

procedure TLoadHelper.LoadTensorDescription(var Value: TTensorDescription);
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
      TTensorDescription.ftDtype:
        begin
          Assert(wireType = TWire.VARINT);
          Value.Dtype := TDataType(Pb.readInt32);
        end;
      TTensorDescription.ftShape:
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
      TTensorDescription.ftAllocationDescription:
        begin
          Assert(wireType = TWire.LENGTH_DELIMITED);
          Pb.Push;
          try
            var v : TAllocationDescription := Value.AllocationDescription;
            LoadAllocationDescription(v);
            Value.AllocationDescription := v;
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

class procedure TSaveHelper.SaveAllocationRecord(const S: TpbSaver; const Value: TAllocationRecord);
begin
  S.Pb.writeInt64(TAllocationRecord.ftAllocMicros, Value.AllocMicros);
  S.Pb.writeInt64(TAllocationRecord.ftAllocBytes, Value.AllocBytes);
end;

class procedure TSaveHelper.SaveAllocatorMemoryUsed(const S: TpbSaver; const Value: TAllocatorMemoryUsed);
begin
  S.Pb.writeString(TAllocatorMemoryUsed.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftTotalBytes, Value.TotalBytes);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftPeakBytes, Value.PeakBytes);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftLiveBytes, Value.LiveBytes);
  if Value.FAllocationRecordss.Count > 0 then
    S.SaveList<TAllocationRecord>(Value.FAllocationRecordss, SaveAllocationRecord, TAllocatorMemoryUsed.ftAllocationRecordss);
  S.Pb.writeInt64(TAllocatorMemoryUsed.ftAllocatorBytesInUse, Value.AllocatorBytesInUse);
end;

class procedure TSaveHelper.SaveNodeOutput(const S: TpbSaver; const Value: TNodeOutput);
begin
  S.Pb.writeInt32(TNodeOutput.ftSlot, Value.Slot);
  S.SaveObj<TTensorDescription>(Value.FTensorDescription, SaveTensorDescription, TNodeOutput.ftTensorDescription);
end;

class procedure TSaveHelper.SaveMemoryStats(const S: TpbSaver; const Value: TMemoryStats);
var 
  i : Integer;
  h : TpbSaver;

begin
  S.Pb.writeInt64(TMemoryStats.ftTempMemorySize, Value.TempMemorySize);
  S.Pb.writeInt64(TMemoryStats.ftPersistentMemorySize, Value.PersistentMemorySize);
  h.Init;
  try
    for i := 0 to Value.PersistentTensorAllocIdss.Count - 1 do
      h.Pb.writeRawVarint64(Value.FPersistentTensorAllocIdss[i]^);
    S.Pb.writeMessage(TMemoryStats.ftPersistentTensorAllocIdss, h.Pb^);
  finally
    h.Free;
  end;
  S.Pb.writeInt64(TMemoryStats.ftDeviceTempMemorySize, Value.DeviceTempMemorySize);
  S.Pb.writeInt64(TMemoryStats.ftDevicePersistentMemorySize, Value.DevicePersistentMemorySize);
  h.Init;
  try
    for i := 0 to Value.DevicePersistentTensorAllocIdss.Count - 1 do
      h.Pb.writeRawVarint64(Value.FDevicePersistentTensorAllocIdss[i]^);
    S.Pb.writeMessage(TMemoryStats.ftDevicePersistentTensorAllocIdss, h.Pb^);
  finally
    h.Free;
  end;
end;

class procedure TSaveHelper.SaveNodeExecStats(const S: TpbSaver; const Value: TNodeExecStats);
begin
  S.Pb.writeString(TNodeExecStats.ftNodeName, Value.NodeName);
  S.Pb.writeInt64(TNodeExecStats.ftAllStartMicros, Value.AllStartMicros);
  S.Pb.writeInt64(TNodeExecStats.ftOpStartRelMicros, Value.OpStartRelMicros);
  S.Pb.writeInt64(TNodeExecStats.ftOpEndRelMicros, Value.OpEndRelMicros);
  S.Pb.writeInt64(TNodeExecStats.ftAllEndRelMicros, Value.AllEndRelMicros);
  if Value.FMemorys.Count > 0 then
    S.SaveList<TAllocatorMemoryUsed>(Value.FMemorys, SaveAllocatorMemoryUsed, TNodeExecStats.ftMemorys);
  if Value.FOutputs.Count > 0 then
    S.SaveList<TNodeOutput>(Value.FOutputs, SaveNodeOutput, TNodeExecStats.ftOutputs);
  S.Pb.writeString(TNodeExecStats.ftTimelineLabel, Value.TimelineLabel);
  S.Pb.writeInt64(TNodeExecStats.ftScheduledMicros, Value.ScheduledMicros);
  S.Pb.writeInt32(TNodeExecStats.ftThreadId, Value.ThreadId);
  if Value.FReferencedTensors.Count > 0 then
    S.SaveList<TAllocationDescription>(Value.FReferencedTensors, SaveAllocationDescription, TNodeExecStats.ftReferencedTensors);
  S.SaveObj<TMemoryStats>(Value.FMemoryStats, SaveMemoryStats, TNodeExecStats.ftMemoryStats);
  S.Pb.writeInt64(TNodeExecStats.ftAllStartNanos, Value.AllStartNanos);
  S.Pb.writeInt64(TNodeExecStats.ftOpStartRelNanos, Value.OpStartRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftOpEndRelNanos, Value.OpEndRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftAllEndRelNanos, Value.AllEndRelNanos);
  S.Pb.writeInt64(TNodeExecStats.ftScheduledNanos, Value.ScheduledNanos);
end;

class procedure TSaveHelper.SaveDeviceStepStats(const S: TpbSaver; const Value: TDeviceStepStats);
var 
  h : TpbSaver;

begin
  S.Pb.writeString(TDeviceStepStats.ftDevice, Value.Device);
  if Value.FNodeStatss.Count > 0 then
    S.SaveList<TNodeExecStats>(Value.FNodeStatss, SaveNodeExecStats, TDeviceStepStats.ftNodeStatss);
  if Value.FThreadNames.Count > 0 then
  begin
    h.Init;
    try
      var it := Value.FThreadNames.Begins;
      while it <> Value.FThreadNames.Ends do
      begin
          h.clear;
          h.SaveUint32String(it.GetPair^);
          S.Pb.writeMessage(TDeviceStepStats.ftThreadNames, h.Pb^);
          it.Next;
      end;
    finally
      h.Free;
    end;
  end;
end;

class procedure TSaveHelper.SaveStepStats(const S: TpbSaver; const Value: TStepStats);
begin
  if Value.FDevStatss.Count > 0 then
    S.SaveList<TDeviceStepStats>(Value.FDevStatss, SaveDeviceStepStats, TStepStats.ftDevStatss);
end;

class procedure TSaveHelper.SaveAllocationDescription(const S: TpbSaver; const Value: TAllocationDescription);
begin
  S.Pb.writeInt64(TAllocationDescription.ftRequestedBytes, Value.RequestedBytes);
  S.Pb.writeInt64(TAllocationDescription.ftAllocatedBytes, Value.AllocatedBytes);
  S.Pb.writeString(TAllocationDescription.ftAllocatorName, Value.AllocatorName);
  S.Pb.writeInt64(TAllocationDescription.ftAllocationId, Value.AllocationId);
  S.Pb.writeBoolean(TAllocationDescription.ftHasSingleReference, Value.HasSingleReference);
  S.Pb.writeInt64(TAllocationDescription.ftPtr, Value.Ptr);
end;

class procedure TSaveHelper.SaveTensorDescription(const S: TpbSaver; const Value: TTensorDescription);
begin
  S.Pb.writeInt32(TTensorDescription.ftDtype, Ord(Value.Dtype));
  S.SaveObj<TTensorShapeProto>(Value.Shape, SaveTensorShapeProto, TTensorDescription.ftShape);
  S.SaveObj<TAllocationDescription>(Value.AllocationDescription, SaveAllocationDescription, TTensorDescription.ftAllocationDescription);
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

procedure TSaveHelper.SaveUint32String(Item: TsgPair<UInt32, string>);
begin
  Pb.writeInt32(1, Item.Key);
  Pb.writeString(2, Item.Value);
end;

end.
