unit ProtoGen.AllocationDescription;

interface

uses
  System.Classes, System.SysUtils,  System.Rtti, Generics.Collections, Oz.Pb.Classes;

{$T+}

type
  TAllocationDescription = Class
  const
    ftRequestedBytes = 1;
    ftAllocatedBytes = 2;
    ftAllocatorName = 3;
    ftAllocationId = 4;
    ftHasSingleReference = 5;
    ftPtr = 6;
  private
    FRequestedBytes: Int64;
    FAllocatedBytes: Int64;
    FAllocatorName: string;
    FAllocationId: Int64;
    FHasSingleReference: Boolean;
    FPtr: Int64;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property RequestedBytes: Int64 read FRequestedBytes write FRequestedBytes;
    property AllocatedBytes: Int64 read FAllocatedBytes write FAllocatedBytes;
    property AllocatorName: string read FAllocatorName write FAllocatorName;
    property AllocationId: Int64 read FAllocationId write FAllocationId;
    property HasSingleReference: Boolean read FHasSingleReference write FHasSingleReference;
    property Ptr: Int64 read FPtr write FPtr;
  end;

implementation

{ TAllocationDescription }

Constructor TAllocationDescription.Create;
begin
  inherited Create;
end;

destructor TAllocationDescription.Destroy;
begin
  inherited Destroy;
end;

end.
