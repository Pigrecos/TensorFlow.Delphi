unit Numpy.Axis;

interface
     uses System.SysUtils, Spring;

type
  TAxis = Record
    isScalar : Boolean;
    axis     : Nullable< TArray<Integer> >;
  End;

implementation

end.
