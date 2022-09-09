unit Numpy.Axis;

interface
     uses System.SysUtils, Spring;

type
  TAxis = Record
    private
     function GetSize: Integer;
    public
      isScalar : Boolean;
      axis     : Nullable< TArray<Integer> >;

      property size : Integer read GetSize;
  End;

implementation

{ TAxis }

function TAxis.GetSize: Integer;
begin
   if axis = nil then Result := -1
   else               Result := Length(axis.Value);

end;

end.
