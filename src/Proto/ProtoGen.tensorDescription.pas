unit ProtoGen.TensorDescription;

interface

uses
  System.Classes, System.Generics.Collections, System.Rtti, System.SysUtils, Generics.Collections, Oz.Pb.Classes,
  ProtoGen.allocationdescription,
  ProtoGen.tensorshape,
  ProtoGen.types;

{$T+}

type
  TTensorDescription = Class
  const
    ftDtype = 1;
    ftShape = 2;
    ftAllocationDescription = 4;
  private
    FDtype: TDataType;
    FShape: TTensorShapeProto;
    FAllocationDescription: TAllocationDescription;
  public
    Constructor Create;
    destructor  Destroy; Override;
    // properties
    property Dtype: TDataType read FDtype write FDtype;
    property Shape: TTensorShapeProto read FShape write FShape;
    property AllocationDescription: TAllocationDescription read FAllocationDescription write FAllocationDescription;
  end;

implementation

{ TTensorDescription }

Constructor TTensorDescription.Create;
begin
  inherited Create;
end;

destructor TTensorDescription.Destroy;
begin
  inherited Destroy;
end;

end.
