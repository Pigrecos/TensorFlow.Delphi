unit TensorFlow.Slice;
{$REGION 'Licence'}
(*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************)
{$ENDREGION}

interface
         uses System.SysUtils,
              system.Rtti,
              System.RegularExpressionsCore,
              System.RegularExpressions,

              Spring,
              spring.Collections,

              TensorFlow.DApiBase;

type

  TRegExHelper = record helper for TRegEx
  private
     class var FRegExOffset: Integer;
  public
   function NamedInGroup(const m: TMatch; const Name: string): Boolean;
  end;

  /// <summary>                                                                                                                                         <br></br>
  /// NDArray can be indexed using slicing                                                                                                              <br></br>
  /// A slice is constructed by start:stop:step notation                                                                                                <br></br>
  ///                                                                                                                                                   <br></br>
  /// Examples:                                                                                                                                         <br></br>
  ///                                                                                                                                                   <br></br>
  /// a[start:stop]  # items start through stop-1                                                                                                       <br></br>
  /// a[start:]      # items start through the rest of the array                                                                                        <br></br>
  /// a[:stop]       # items from the beginning through stop-1                                                                                          <br></br>
  ///                                                                                                                                                   <br></br>
  /// The key point to remember is that the :stop value represents the first value that is not                                                          <br></br>
  /// in the selected slice. So, the difference between stop and start is the number of elements                                                        <br></br>
  /// selected (if step is 1, the default).                                                                                                             <br></br>
  ///                                                                                                                                                   <br></br>
  /// There is also the step value, which can be used with any of the above:                                                                            <br></br>
  /// a[:]           # a copy of the whole array                                                                                                        <br></br>
  /// a[start:stop:step] # start through not past stop, by step                                                                                         <br></br>
  ///                                                                                                                                                   <br></br>
  /// The other feature is that start or stop may be a negative number, which means it counts                                                           <br></br>
  /// from the end of the array instead of the beginning. So:                                                                                           <br></br>
  /// a[-1]    # last item in the array                                                                                                                 <br></br>
  /// a[-2:]   # last two items in the array                                                                                                            <br></br>
  /// a[:-2]   # everything except the last two items                                                                                                   <br></br>
  /// Similarly, step may be a negative number:                                                                                                         <br></br>
  ///                                                                                                                                                   <br></br>
  /// a[::- 1]    # all items in the array, reversed                                                                                                    <br></br>
  /// a[1::- 1]   # the first two items, reversed                                                                                                       <br></br>
  /// a[:-3:-1]  # the last two items, reversed                                                                                                         <br></br>
  /// a[-3::- 1]  # everything except the last two items, reversed                                                                                      <br></br>
  ///                                                                                                                                                   <br></br>
  /// NumSharp is kind to the programmer if there are fewer items than                                                                                  <br></br>
  /// you ask for. For example, if you  ask for a[:-2] and a only contains one element, you get an                                                      <br></br>
  /// empty list instead of an error.Sometimes you would prefer the error, so you have to be aware                                                      <br></br>
  /// that this may happen.                                                                                                                             <br></br>
  ///                                                                                                                                                   <br></br>
  /// Adapted from Greg Hewgill's answer on Stackoverflow: https://stackoverflow.com/questions/509211/understanding-slice-notation                      <br></br>
  ///                                                                                                                                                   <br></br>
  /// Note: special IsIndex == true                                                                                                                     <br></br>
  /// It will pick only a single value at Start in this dimension effectively reducing the Shape of the sliced matrix by 1 dimension.                   <br></br>
  /// It can be used to reduce an N-dimensional array/matrix to a (N-1)-dimensional array/matrix                                                        <br></br>
  ///                                                                                                                                                   <br></br>
  /// Example:                                                                                                                                          <br></br>
  /// a=[[1, 2], [3, 4]]                                                                                                                                <br></br>
  /// a[:, 1] returns the second column of that 2x2 matrix as a 1-D vector                                                                              <br></br>
  /// </summary>
  Slice = record
    private
      hAssigned : Cardinal;
      procedure Parse(slice_notation: string);
    public
      Start      : TNullableInteger;
      Stop       : TNullableInteger;
      Step       : Integer;
      IsIndex    : Boolean;
      IsEllipsis : Boolean;
      IsNewAxis  : Boolean;

      /// <summary>
      /// ndarray can be indexed using slicing
      /// slice is constructed by start:stop:step notation
      /// </summary>
      /// <param name="start">Start index of the slice, null means from the start of the array</param>
      /// <param name="stop">Stop index (first index after end of slice), null means to the end of the array</param>
      /// <param name="step">Optional step to select every n-th element, defaults to 1</param>
      constructor Create(istart: TNullableInteger ; istop: TNullableInteger ; istep: Integer = 1; bisIndex: Boolean = false);overload;
      constructor Create(slice_notation: string);overload;
      /// <summary>
      /// return exactly one element at this dimension and reduce the shape from n-dim to (n-1)-dim
      /// </summary>
      /// <param name="index"></param>
      /// <returns></returns>
      class function Index(index: Integer): slice;static;
      /// <summary>
      /// Parses Python array slice notation and returns an array of Slice objects
      /// </summary>
      class function  ParseSlices(multi_slice_notation: string): TArray<Slice>; static;
      class function  AreAllIndex(slices: TArray<Slice>; var indices: TArray<Integer>): Boolean;  static;
      class function  IsContinuousBlock(slices: TArray<Slice>; ndim: Integer): Boolean;  static;
      function ToString: string;

      class Operator Equal(a,b : Slice): Boolean;
      class Operator NotEqual(a,b : Slice): Boolean;
      class Operator Implicit(index: Integer): Slice;
      class Operator Implicit(sSlice: string): Slice;

      /// <summary>
      /// Length of the slice.
      /// <remarks>
      /// The length is not guaranteed to be known for i.e. a slice like ":". Make sure to check Start and Stop
      /// for null before using it</remarks>
      /// </summary>
      function Len : TNullableInteger;
      /// <summary>
      /// return : for this dimension
      /// </summary>
      class function All : Slice; static;
      /// <summary>
      /// return 0:0 for this dimension
      /// </summary>
      class function None : Slice; static;
      /// <summary>
      /// fill up the missing dimensions with : at this point, corresponds to ...
      /// </summary>
      class function Ellipsis : Slice; static;
      /// <summary>
      /// insert a new dimension at this point
      /// </summary>
      class function NewAxis  : Slice; static;
  end;


implementation

{ Slice }

constructor Slice.Create(istart, istop: TNullableInteger; istep: Integer; bisIndex: Boolean);
begin
    hAssigned  := $44444444;
    Start   := istart;
    Stop    := istop;
    Step    := istep;
    IsIndex := bisIndex;
end;

class function Slice.AreAllIndex(slices: TArray<Slice>; var indices: TArray<Integer>): Boolean;
begin
    SetLength(indices, Length(slices)) ;
    for var i := 0 to Length(slices) - 1 do
    begin
        if slices[i].Start = nil then indices[i] := 0
        else                          indices[i] :=slices[i].Start;
        if not slices[i].IsIndex then
            Exit(false);
    end;
    Result := true;
end;

class function Slice.IsContinuousBlock(slices: TArray<Slice>; ndim: Integer): Boolean;
begin
    for var i := ndim + 1 to  Length(slices) - 1 do
    begin
        if slices[i] = Slice.All then
            continue;
        Exit(false);
    end;
    Result := true;
end;

constructor Slice.Create(slice_notation: string);
begin
    hAssigned  := $44444444;
    Parse(slice_notation);
end;

class function Slice.Ellipsis: Slice;
begin
    Result := Slice.Create(0, 0,1);
    Result.IsEllipsis := True;
end;

class operator Slice.Equal(a, b: Slice): Boolean;
begin
    if (a.hAssigned <> $44444444) or (b.hAssigned <> $44444444) then
       Exit(false);

    Result := (a.Start = b.Start) and (a.Stop = b.Stop) and (a.Step = b.Step);
end;

class function Slice.All: Slice;
begin
    Result := Slice.Create(nil, nil);
end;

function Slice.Len: TNullableInteger;
begin
    Result := Stop.Value - Start.Value;
end;

class function Slice.NewAxis: Slice;
begin
    Result := Slice.Create(0, 0,1);
    Result.IsNewAxis := True;
end;

class function Slice.None: Slice;
begin
    Result := Slice.Create(0, 0,1);
end;

class operator Slice.NotEqual(a, b: Slice): Boolean;
begin
    Result := not (a = b);
end;

class operator Slice.Implicit(sSlice: string): Slice;
begin
   Result :=  Slice.Create(sSlice)
end;

class operator Slice.Implicit(index: Integer): Slice;
begin
   Result :=  Slice.Index(index)
end;

class function Slice.Index(index: Integer): slice;
begin
    Result := Slice.Create(index, index + 1);
    Result.IsIndex := True;
end;

procedure Slice.Parse(slice_notation: string);
begin
    if string.IsNullOrEmpty(slice_notation) then
        Raise TFException.Create('Slice notation expected, got empty string or null');

    var t := TRegex.Create('^\s*((?<start>[+-]?\s*\d+)?\s*:\s*(?<stop>[+-]?\s*\d+)?\s*(:\s*(?<step>[+-]?\s*\d+)?)?|(?<index>[+-]?\s*\d+)|(?<ellipsis>\.\.\.)|(?<newaxis>(np\.)?newaxis))\s*$');
    var match :=  t.Match(slice_notation) ;

    if  not match.Success then
        raise  TFException.Create('Invalid slice notation: '+ ''+slice_notation+'');

    if t.NamedInGroup(match,'ellipsis') then
    begin
        Start := 0;
        Stop := 0;
        Step := 1;
        IsEllipsis := true;
        Exit;
    end;
    if t.NamedInGroup(match,'newaxis') then
    begin
        Start := 0;
        Stop  := 0;
        Step := 1;
        IsNewAxis := true;
        Exit;
    end;
    if t.NamedInGroup(match,'index')  then
    begin
        var i : Integer;
        if not Integer.TryParse( TRegex.Replace(match.Groups['index'].Value , '\s+', ''), i ) then
           raise  TFException.Create('Invalid value for index: ' + match.Groups['index'].Value);
        Start := i;
        Stop  := start.Value + 1;
        Step := 1; // special case for dimensionality reduction by picking a single element
        IsIndex := true;
        Exit;
    end;
    var start_string : string;
    var stop_string  : string;
    var step_string  : string;
    if t.NamedInGroup(match,'start')  then  start_string := TRegex.Replace(match.Groups['start'].Value , '\s+', '')
    else                                    start_string := TRegex.Replace('' , '\s+', '') ;

    if t.NamedInGroup(match,'stop')  then   stop_string := TRegex.Replace(match.Groups['stop'].Value , '\s+', '')
    else                                    stop_string := TRegex.Replace('' , '\s+', '') ;

    if t.NamedInGroup(match,'step')  then   step_string := TRegex.Replace(match.Groups['step'].Value , '\s+', '')
    else                                    step_string := TRegex.Replace('' , '\s+', '') ;

    if string.IsNullOrWhiteSpace(start_string) then
        Start := nil
    else begin
        var i : Integer;
        if not Integer.TryParse(start_string, i) then
            raise  TFException.Create('Invalid value for start: '+start_string);
        Start := i;
    end;
    if string.IsNullOrWhiteSpace(stop_string) then
        Stop := nil
    else begin
        var i : Integer;
        if not Integer.TryParse(stop_string, i) then
            raise  TFException.Create('Invalid value for stop: '+stop_string);
        Stop := i;
    end;
    if string.IsNullOrWhiteSpace(step_string) then
        Step := 1
    else begin
        var i : Integer;
        if not Integer.TryParse(step_string, i) then
            raise  TFException.Create('Invalid value for step: '+step_string);
        Step := i;
    end;
end;

class function Slice.ParseSlices(multi_slice_notation: string): TArray<Slice>;
begin
    Result := [];
    var sRes := TRegex.Split(multi_slice_notation, ',\s*');
    for var i := 0 to Length(sRes) - 1 do
    begin
        if not string.IsNullOrWhiteSpace(sres[i]) then
        begin
            Result := Result + [ Slice.Create(sres[i]) ]
        end;
    end;
end;

function Slice.ToString: string;
begin
    if IsIndex then
    begin
        if Start = nil then Exit('0')
        else                Exit(IntToStr(Start));
    end
    else if IsNewAxis then
        Exit('np.newaxis')
    else if IsEllipsis then
        Exit('...');
    var optional_step : string;
    if Step = 1 then optional_step := ''
    else             optional_step := ':'+ IntToStr(Step);
    var sStart : string;
    if Start = 0 then sStart := ''
    else              sStart := IntToStr(Start);
    var sStop : string;
    if Stop = nil then sStop := ''
    else               sStop := IntToStr(Stop);
    Result := Format('%s:%%',[sStart, sStop, optional_step]) ;

end;

{ TRegExHelper }

function TRegExHelper.NamedInGroup(const m: TMatch; const Name: string): Boolean;
var
  ctx      : TRTTIContext;
  FRecord  : TRttiRecordType ;
begin
    // Hack record Private Property
    //
    var t        := TypeInfo(TRegEx);
    FRecord      := ctx.GetType(t).AsRecord ;
    FRegExOffset := FRecord.GetField('FRegEx').Offset;

    var PerlReg := TPerlRegEx(Pointer(NativeInt(@Self) + FRegExOffset)^) ;

    var idx  := PerlReg.NamedGroup(Name);
    if (idx >= 0) and  (idx < m.Groups.Count) then
          result :=  true
    else
      result := False;
end;

end.
