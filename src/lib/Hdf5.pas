unit Hdf5;
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

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils, Winapi.Windows,
         System.Variants,
         System.Generics.Collections,
         System.Rtti,
         System.TypInfo,

         Spring,

         hdf5dll;

const NomeHdf5Dll : string ='hdf5.dll';


type
  Hdf5ElementType = (
        Unknown = 0,
        Group,
        Dataset,
        Attribute,
        CommitedDatatype);
  TDateTimeType = (
        Ticks,
        UnixTimeSeconds,
        UnixTimeMilliseconds);
  /// <summary>
  /// Character set to use for text strings.
  /// </summary>
  TCharacterSetType = (
      /// <summary>
      /// US ASCII [value = 0].
      /// </summary>
      ASCII = 0,
      /// <summary>
      /// UTF-8 Unicode encoding [value = 1].
      /// </summary>
      UTF8 = 1);
  /// <summary>
  /// Type of padding to use in character strings.
  /// </summary>
  TCharacterPaddingType = (
      /// <summary>
      /// null terminate like in C
      /// </summary>
      NULLTERM = 0,
      /// <summary>
      /// pad with nulls
      /// </summary>
      NULLPAD = 1,
      /// <summary>
      /// pad with spaces like in Fortran
      /// </summary>
      SPACEPAD = 2);

  THdf5LogLevel = (
        Debug,
        Info,
        Warning,
        Error);

  TSettings = class
    public
       DateTimeType                   : TDateTimeType;
       LowerCaseNaming                : Boolean;
       H5InternalErrorLoggingEnabled  : Boolean;
       ThrowOnError                   : Boolean;
       ThrowOnNonExistNameWhenReading : Boolean;
       OverrideExistingData           : Boolean;
       Version                        : Single;
       GlobalLoggingEnabled           : boolean;
       /// <summary>
       /// Character set to use for text strings.
       /// </summary>
       CharacterSetType               : TCharacterSetType;
       /// <summary>
       /// Type of padding to use in character strings.
       /// </summary>
       CharacterPaddingType           : TCharacterPaddingType;

       constructor Create;
  end;

  THdf5Utils = class
    private
      class function GetId(parentId: Int64; name: string; dataType: Int64; spaceId: Int64; tipo: Hdf5ElementType): int64;
    public
       class var LogError   : TProc<string>;
       class var LogInfo    : TProc<string>;
       class var LogDebug   : TProc<string>;
       class var LogWarning : TProc<string>;

       class function  StringToByte(str: string): TArray<Byte>;
       class function  GetCharacterSet(characterSetType: TCharacterSetType): H5T_cset_t;
       class function  GetCharacterPadding(characterPaddingType: TCharacterPaddingType):H5T_str_t;
       class function  GetDatasetId(parentId: Int64; name: string; dataType: Int64; spaceId: Int64): int64;
       class function  GetAttributeId(parentId: Int64; name: string; dataType, spaceId: Int64): int64;
       class Function  ReadStringBuffer(buffer: TArray<Byte>) : string;
       class function  NormalizedName(name: string): string;
       class function  ItemExists(groupId: Int64; groupName: string; tipo: Hdf5ElementType): boolean;
       class function  GetRealAttributeName(id: Int64; name: string; alternativeName: string): Tuple<boolean, string> ;
       class function  GetRealName(id: Int64; name: string; alternativeName: string): Tuple<boolean, string> ;
       class procedure LogMessage(msg: string; level: THdf5LogLevel);
    end;

  THdf5 = class
    public
       class var FH5            : THDF5Dll;
       class var FIsInizialized : Boolean;
       class var FSetting       : TSettings;

       class function GetDatatype(tipo: TValue): Int64; overload;
       class function GetDatatype(pTipo: PTypeInfo): Int64; overload;

    public
       /// <summary>
       /// Opens a Hdf-5 file
       /// </summary>
       /// <param name="filename"></param>
       /// <param name="readOnly"></param>
       /// <returns></returns>
       class function OpenFile(filename: string; readOnly: Boolean = false): Int64;
       class function CreateFile(filename: string): Int64;
       class function CloseFile(fileId: Int64): Int64;
       class function GroupExists(groupId: Int64; groupName: string) : Boolean;
       class function ReadStringAttributes(groupId: Int64; name: string; alternativeName: string; mandatory: Boolean):Tuple<Boolean, TArray<string>> ;
       class function OpenAttributeIfExists(fileOrGroupId: Int64; name: string; alternativeName: string): Int64;
       class function ReadDataset<T>(groupId: Int64; name: string; alternativeName: string = ''; mandatory : Boolean= false): Tuple<Boolean, TArray<T>>;
       class function ReadPrimitiveAttributes<T>(groupId: Int64; name: string; alternativeName: string): Tuple<Boolean, TArray<T>>; 
       class function WriteAttribute<T>(groupId: Int64; name: string; attribute: T): Tuple<Integer, Int64>;
       class function WriteAttributes<T>(groupId: Int64; name: string; attributes: TArray<T>): Tuple<Integer, Int64>;
       class function WritePrimitiveAttribute<T>(groupId: Int64; name: string; attributes: TArray<T>) : Tuple<Integer, Int64>;
       class function WriteStringAttributes(groupId: Int64; name: string; values: TArray<string>; groupOrDatasetName: string = '') : Tuple<Integer, Int64>;
       /// <summary>
       /// Reads an n-dimensional dataset.
       /// </summary>
       /// <typeparam name="T">Generic parameter strings or primitive type</typeparam>
       /// <param name="groupId">id of the group. Can also be a file Id</param>
       /// <param name="name">name of the dataset</param>
       /// <param name="alternativeName">Alternative name</param>
       /// <returns>The n-dimensional dataset</returns>
       class function ReadDatasetToArray<T>(groupId: Int64; name: string; alternativeName: string =''): Tuple<Boolean, TArray<T>>;
       class function WriteDatasetFromArray<T>(groupId: Int64; name: string; dset: TArray<T>; Shape: TArray<Integer>= []): Tuple<Integer, Int64>;
       class function CreateOrOpenGroup(fileOrGroupId: Int64; groupName: string): Int64;
       class function CloseGroup(groupId: Int64): Integer;
  end;


implementation

{ TSettings }

constructor TSettings.Create;
begin
    DateTimeType := TDateTimeType.Ticks;
    ThrowOnError := true;
    OverrideExistingData := true;
    CharacterPaddingType := TCharacterPaddingType.SPACEPAD;
    CharacterSetType := TCharacterSetType.UTF8;
    Version := 2.0;
    GlobalLoggingEnabled := true;
end;

{ THdf5Utils }

class function THdf5Utils.GetAttributeId(parentId: Int64; name: string; dataType, spaceId: Int64): int64;
begin
    Result := GetId(parentId, name, dataType, spaceId, Hdf5ElementType.Attribute);
end;

class function THdf5Utils.GetCharacterSet(characterSetType: TCharacterSetType): H5T_cset_t;
begin
    case characterSetType of
      ASCII: Result := H5T_CSET_ASCII;
      UTF8:  Result := H5T_CSET_UTF8;
    else
      raise Exception.Create('THdf5Utils.GetCharacterSet Error');
    end;
end;

class function THdf5Utils.GetCharacterPadding(characterPaddingType: TCharacterPaddingType): H5T_str_t;
begin
    case characterPaddingType of
      NULLTERM: Result := H5T_STR_NULLTERM;
      NULLPAD: Result := H5T_STR_NULLPAD;
      SPACEPAD: Result := H5T_STR_SPACEPAD;
    else
      raise Exception.Create('THdf5Utils.GetCharacterPadding Error');
    end;
end;

class function THdf5Utils.GetDatasetId(parentId: Int64; name: string; dataType, spaceId: Int64): int64;
begin
    Result := GetId(parentId, name, dataType, spaceId, Hdf5ElementType.Dataset);
end;

class function THdf5Utils.GetId(parentId: Int64; name: string; dataType, spaceId : Int64; tipo: Hdf5ElementType): int64;
begin
    var normalizedName := NormalizedName(name);
    var exists := ItemExists(parentId, normalizedName, tipo);
    if exists then
    begin
        LogMessage(normalizedName+ ' already exists', THdf5LogLevel.Debug);
        if not THdf5.FSetting.OverrideExistingData then
        begin
            if THdf5.FSetting.ThrowOnError then
               raise Exception.Create(normalizedName +' already exists');

            Exit(-1);
        end;
    end;
    var datasetId : Int64 := -1;
    case tipo of
        Hdf5ElementType.Unknown: begin end;
        Hdf5ElementType.Group,
        Hdf5ElementType.Dataset:
          begin
            if exists then
               THdf5.FH5.H5Ldelete(parentId, PAnsiChar(AnsiString(normalizedName)),H5P_DEFAULT);
            datasetId := THdf5.FH5.H5Dcreate2(parentId, PAnsiChar(AnsiString(normalizedName)), dataType, spaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          end;
        Hdf5ElementType.Attribute:
          begin
            if exists then
               THdf5.FH5.H5Adelete(parentId, PAnsiChar(AnsiString(normalizedName)));
            datasetId := THdf5.FH5.H5Acreate2(parentId, PAnsiChar(AnsiString(normalizedName)), dataType, spaceId,H5P_DEFAULT,H5P_DEFAULT);
          end
    else
       raise Exception.Create('ArgumentOutOfRangeException(nameof(type)');
    end;
    if datasetId = -1 then
    begin
        var error : string := 'Unable to create data set for '+normalizedName;
        LogMessage(normalizedName+' already exists', THdf5LogLevel.Error);
        if THdf5.FSetting.ThrowOnError then
          raise Exception.Create(error);
    end;
    Result := datasetId;
end;

class function THdf5Utils.GetRealAttributeName(id: Int64; name, alternativeName: string): Tuple<boolean, string>;
var
  normalized : AnsiString;
begin
    normalized := NormalizedName(name);
    if (not string.IsNullOrEmpty(normalized)) and (THdf5.FH5.H5Aexists(id, PAnsiChar(normalized)) > 0) then
    begin
        Exit( Tuple.Create(true, string(normalized)) );
    end;
    normalized := NormalizedName(alternativeName);
    if (not string.IsNullOrEmpty(normalized)) and (THdf5.FH5.H5Aexists(id, PAnsiChar(normalized)) > 0) then
    begin
        Exit( Tuple.Create(true, string(normalized)) );
    end;
    Result := Tuple.Create(false, '');
end;

class function THdf5Utils.GetRealName(id: Int64; name, alternativeName: string): Tuple<boolean, string>;
var
  normalized : AnsiString;
begin
    normalized := NormalizedName(name);
    if (not string.IsNullOrEmpty(normalized)) and (THdf5.FH5.H5Lexists(id, PAnsiChar(normalized), H5P_DEFAULT) > 0) then
    begin
        Exit( Tuple.Create(true, string(normalized)) );
    end;
    normalized := NormalizedName(alternativeName);
    if (not string.IsNullOrEmpty(normalized)) and (THdf5.FH5.H5Lexists(id, PAnsiChar(normalized), H5P_DEFAULT) > 0) then
    begin
        Exit( Tuple.Create(true, string(normalized)) );
    end;
    Result := Tuple.Create(false, '');
end;

class function THdf5Utils.ItemExists(groupId: Int64; groupName: string; tipo: Hdf5ElementType): boolean;
var
  NormName : AnsiString;
begin
    NormName := NormalizedName(groupName);
    case tipo of
      Hdf5ElementType.Group,
      Hdf5ElementType.Dataset:    Result := THdf5.FH5.H5Lexists(groupId, PAnsiChar(NormName), H5P_DEFAULT) > 0;
      Hdf5ElementType.Attribute:  Result := THdf5.FH5.H5Aexists(groupId,  PAnsiChar(NormName)) > 0;
    else
      raise Exception.Create('ArgumentOutOfRangeException(nameof(type), type, null');
    end;
end;

class procedure THdf5Utils.LogMessage(msg: string; level: THdf5LogLevel);
begin
    if THdf5.FSetting.GlobalLoggingEnabled then
      Exit;

    case level of
      Debug   : LogDebug(msg);
      Info    : LogInfo(msg);
      Warning : LogWarning(msg);
      Error   : LogError(msg);
    end;
end;

class function THdf5Utils.NormalizedName(name: string): string;
begin
   if string.IsNullOrEmpty(name) then  Exit('');

   Result := name;

   if THdf5.FSetting.LowerCaseNaming then
      Result := name.ToLower;
end;

class function THdf5Utils.ReadStringBuffer(buffer: TArray<Byte>): string;
begin
    case THdf5.FSetting.CharacterSetType of
      TCharacterSetType.ASCII:  Result := TEncoding.ASCII.GetString(buffer);
      TCharacterSetType.UTF8:   Result := TEncoding.UTF8.GetString(buffer);
    else
      raise Exception.Create('ArgumentOutOfRangeException');
    end;
end;

class function THdf5Utils.StringToByte(str: string): TArray<Byte>;
begin
    case THdf5.FSetting.CharacterSetType of
      TCharacterSetType.ASCII:  Result := TEncoding.ASCII.GetBytes(str);
      TCharacterSetType.UTF8:   Result := TEncoding.UTF8.GetBytes(str);
    else
      raise Exception.Create('ArgumentOutOfRangeException');
    end;
end;

{ THdf5 }

class function THdf5.OpenFile(filename: string; readOnly: Boolean): Int64;
var
  access : UInt32;
begin
    if readOnly then access := H5F_ACC_RDONLY
    else             access := H5F_ACC_RDWR;

    Result := THdf5.FH5.H5Fopen(PAnsiChar(AnsiString(filename)), access, H5P_DEFAULT);
end;

class function THdf5.CloseGroup(groupId: Int64): Integer;
begin
   Result := FH5.H5Gclose(groupId);
end;

class function THdf5.CreateFile(filename: string): Int64;
begin
    Result:= THdf5.FH5.H5FCreate(PAnsiChar(AnsiString(filename)), H5F_ACC_TRUNC,H5P_DEFAULT, H5P_DEFAULT);
end;

class function THdf5.CreateOrOpenGroup(fileOrGroupId: Int64; groupName: string): Int64;
var
  normalizedName : string ;
begin
    normalizedName := THdf5Utils.NormalizedName(groupName);

    if THdf5Utils.ItemExists(fileOrGroupId, groupName, Hdf5ElementType.Group) then
       Result := THdf5.FH5.H5Gopen2(fileOrGroupId, PAnsiChar(AnsiString(normalizedName)), H5P_DEFAULT)
    else
       Result := THdf5.FH5.H5Gcreate2(fileOrGroupId, PAnsiChar(AnsiString(normalizedName)), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
end;

class function THdf5.CloseFile(fileId: Int64): Int64;
begin
   Result := THdf5.FH5.H5Fclose(fileId);
end;

class function THdf5.GetDatatype(tipo: TValue): Int64;
var
 pTipo  : PTypeInfo;
begin
    pTipo := tipo.TypeInfo;

    if      pTipo = TypeInfo(Byte)   then  Result := THdf5.FH5.H5T_NATIVE_UINT8
    else if pTipo = TypeInfo(Int8)   then  Result := THdf5.FH5.H5T_NATIVE_INT8
    else if pTipo = TypeInfo(UInt16) then  Result := THdf5.FH5.H5T_NATIVE_UINT16
    else if pTipo = TypeInfo(Int16)  then  Result := THdf5.FH5.H5T_NATIVE_INT16
    else if pTipo = TypeInfo(UInt32) then  Result := THdf5.FH5.H5T_NATIVE_UINT32
    else if pTipo = TypeInfo(Int32)  then  Result := THdf5.FH5.H5T_NATIVE_INT32
    else if pTipo = TypeInfo(UInt64) then  Result := THdf5.FH5.H5T_NATIVE_UINT64
    else if pTipo = TypeInfo(Int64)  then  Result := THdf5.FH5.H5T_NATIVE_INT64
    else if pTipo = TypeInfo(Single) then  Result := THdf5.FH5.H5T_NATIVE_FLOAT
    else if pTipo = TypeInfo(Double) then  Result := THdf5.FH5.H5T_NATIVE_DOUBLE
    else if pTipo = TypeInfo(Char)   then  Result := THdf5.FH5.H5T_C_S1
    else if pTipo = TypeInfo(AnsiChar) then  Result := THdf5.FH5.H5T_C_S1
    else
      raise Exception.Create(tipo.TypeInfo.Name + ' Data Type not supported');
end;

class function THdf5.GetDatatype(pTipo: PTypeInfo): Int64;
begin
    if      pTipo = TypeInfo(Byte)   then  Result := THdf5.FH5.H5T_NATIVE_UINT8
    else if pTipo = TypeInfo(Int8)   then  Result := THdf5.FH5.H5T_NATIVE_INT8
    else if pTipo = TypeInfo(UInt16) then  Result := THdf5.FH5.H5T_NATIVE_UINT16
    else if pTipo = TypeInfo(Int16)  then  Result := THdf5.FH5.H5T_NATIVE_INT16
    else if pTipo = TypeInfo(UInt32) then  Result := THdf5.FH5.H5T_NATIVE_UINT32
    else if pTipo = TypeInfo(Int32)  then  Result := THdf5.FH5.H5T_NATIVE_INT32
    else if pTipo = TypeInfo(UInt64) then  Result := THdf5.FH5.H5T_NATIVE_UINT64
    else if pTipo = TypeInfo(Int64)  then  Result := THdf5.FH5.H5T_NATIVE_INT64
    else if pTipo = TypeInfo(Single) then  Result := THdf5.FH5.H5T_NATIVE_FLOAT
    else if pTipo = TypeInfo(Double) then  Result := THdf5.FH5.H5T_NATIVE_DOUBLE
    else if pTipo = TypeInfo(Char)   then  Result := THdf5.FH5.H5T_C_S1
    else if pTipo = TypeInfo(AnsiChar) then  Result := THdf5.FH5.H5T_C_S1
    else
      raise Exception.Create(pTipo^.Name + ' Data Type not supported');
end;

class function THdf5.GroupExists(groupId: Int64; groupName: string): Boolean;
begin
    Result := THdf5Utils.ItemExists(groupId, groupName, Hdf5ElementType.Group);
end;

class function THdf5.ReadDataset<T>(groupId: Int64; name, alternativeName: string; mandatory: Boolean): Tuple<Boolean, TArray<T>>;
begin
   Result := ReadDatasetToArray<T>(groupId, name, alternativeName);
end;

class function THdf5.ReadDatasetToArray<T>(groupId: Int64; name, alternativeName: string): Tuple<Boolean, TArray<T>>;
var
  Valid       : Boolean;
  datasetName : string;
  res,dset    : TArray<T>;
  tdataS      : Tuple<Boolean, string>;
  Tipoinfo    : PTypeInfo ;
  rank        : Integer;
  datasetId,
  datatype,
  spaceId,
  count,
  memId       : Int64;
  rankChunk   : Integer;
  maxDims,
  dims,
  chunkDims   : TArray<UInt64>;
  lengths     : TArray<Int64>;
begin
    res        := [];
    tdataS     := THdf5Utils.GetRealName(groupId, name, alternativeName);
    Valid      := tdataS.Value1;
    datasetName:= tdataS.Value2;
    if  not Valid then
    begin
        THdf5Utils.LogError('Error reading '+ groupId.ToString+'.'+'Name:'+name+'. AlternativeName:'+alternativeName);
        Result := Tuple.Create(false, res);
        Exit;
    end;
    Tipoinfo  := TypeInfo(T);
    datasetId := THdf5.FH5.H5Dopen2(groupId, PAnsiChar(AnsiString(datasetName)),H5P_DEFAULT);
    datatype  := GetDatatype(Tipoinfo);
    spaceId   := THdf5.FH5.H5Dget_space(datasetId);
    rank      := THdf5.FH5.H5Sget_simple_extent_ndims(spaceId);
    count     := THdf5.FH5.H5Sget_simple_extent_npoints(spaceId);

    if (rank >= 0) and (count >= 0) then
    begin
        SetLength(maxDims,rank) ;
        SetLength(dims,rank) ;
        SetLength(chunkDims,rank) ;

        memId := THdf5.FH5.H5Sget_simple_extent_dims(spaceId, @dims[0], @maxDims[0]);
        lengths := [];
        {TODO -oMax -cHdf5 : Fix for reading array for layer weight}
        for var i := 0 to Length(dims)-1 do
            lengths := lengths + [ dims[i] ];
        SetLength(dset,lengths[0]);
        //dset = Array.CreateInstance(type, lengths);
        //var typeId = H5D.get_type(datasetId);
        //var mem_type = H5T.copy(datatype);
        if datatype = THdf5.FH5.H5T_C_S1 then
        begin
            THdf5.FH5.H5Tset_size(datatype, 2);
        end;
        var propId := THdf5.FH5.H5Dget_create_plist(datasetId);
        if H5D_layout_t.H5D_CHUNKED = THdf5.FH5.H5Pget_layout(propId) then
        begin
            rankChunk := THdf5.FH5.H5Pget_chunk(propId, rank, @chunkDims[0]);
        end;
        memId := THdf5.FH5.H5Screate_simple(rank, @dims[0], @maxDims[0]);
        THdf5.FH5.H5Dread(datasetId, datatype, memId, spaceId, H5P_DEFAULT, @dset[0]);
    end else
    begin
        dset := [ ];
    end;
    THdf5.FH5.H5Dclose(datasetId);
    THdf5.FH5.H5Sclose(spaceId);
    Result := Tuple.Create(true, dset);
end;

class function THdf5.ReadPrimitiveAttributes<T>(groupId: Int64; name, alternativeName: string): Tuple<Boolean, TArray<T>>;
var
  normName    : string;
  res,
  attributes  : TArray<T>;
  Tipoinfo    : PTypeInfo ;
  datatype,
  attributeId : Int64;
  rank        : Integer;
  spaceId,
  typeId,
  memId       : Int64;
  maxDims,
  dims,
  lengths     : TArray<Int64>;
begin
    res      := [];
    Tipoinfo := TypeInfo(T);
    datatype := GetDatatype(Tipoinfo);

    normName    := THdf5Utils.NormalizedName(name);
    attributeId := THdf5.FH5.H5Aopen(groupId, PAnsiChar(AnsiString(normName)),H5P_DEFAULT);

    normName := THdf5Utils.NormalizedName(alternativeName);
    if attributeId <= 0 then
       attributeId := THdf5.FH5.H5Aopen(groupId, PAnsiChar(AnsiString(normName)),H5P_DEFAULT);

    if attributeId <= 0 then
    begin
        THdf5Utils.LogError('Error reading '+ groupId.ToString+'. Name:'+name+ '. AlternativeName:'+alternativeName);
        result := Tuple.Create(false, res);
    end;

    spaceId := THdf5.FH5.H5Aget_space(attributeId);
    rank    := THdf5.FH5.H5Sget_simple_extent_ndims(spaceId);
    SetLength(maxDims,rank) ;
    SetLength(dims,rank) ;

    memId := THdf5.FH5.H5Sget_simple_extent_dims(spaceId, @dims[0], @maxDims[0]);
    lengths := [];
    for var i := 0 to Length(dims)-1 do
        lengths := lengths + [ dims[i] ];
    SetLength(attributes,lengths[0]);
    //Array attributes = Array.CreateInstance(type, lengths);

    typeId := THdf5.FH5.H5Aget_type(attributeId);
    //var mem_type = H5T.copy(datatype);
    if datatype = THdf5.FH5.H5T_C_S1 then
        THdf5.FH5.H5Tset_size(datatype,2);

    //var propId = H5A.get_create_plist(attributeId);
    //memId = H5S.create_simple(rank, dims, maxDims);
    THdf5.FH5.H5Aread(attributeId, datatype, @attributes[0]);
    THdf5.FH5.H5Tclose(typeId);
    THdf5.FH5.H5Aclose(attributeId);
    THdf5.FH5.H5Sclose(spaceId);
    Result := Tuple.Create(true, attributes);

end;

class function THdf5.ReadStringAttributes(groupId: Int64; name, alternativeName: string; mandatory: Boolean): Tuple<Boolean, TArray<string>>;
var
  nameToUse : Tuple<boolean, string>;
  strs      : TList<string>;
  datasetId,
  typeId,
  spaceId,
  count     : Int64;
  rdata     : TArray<pByte>;
  i         : Integer;
begin
    strs := TList<string>.Create;
    try
      nameToUse := THdf5Utils.GetRealAttributeName(groupId, name, alternativeName);
      if not nameToUse.Value1 then
      begin
          THdf5Utils.LogMessage( Format('Error reading %d. Name:%s. AlternativeName:%s',[groupId,name,alternativeName]), THdf5LogLevel.Warning) ;
          if (mandatory) or (FSetting.ThrowOnNonExistNameWhenReading) then
          begin
              THdf5Utils.LogMessage( Format('Error reading %d. Name:%s. AlternativeName:%s',[groupId,name,alternativeName]), THdf5LogLevel.Error);
              raise Exception.Create('unable to read '+name +' or '+ alternativeName);
          end;
          Result := Tuple.Create(false, strs.ToArray);
      end;

      datasetId := THdf5.FH5.H5AOpen(groupId, PAnsiChar(AnsiString(nameToUse.Value2)),H5P_DEFAULT);
      typeId    := THdf5.FH5.H5AGet_type(datasetId);
      spaceId   := THdf5.FH5.H5AGet_space(datasetId);
      count     := THdf5.FH5.H5SGet_simple_extent_npoints(spaceId);
      THdf5.FH5.H5Sclose(spaceId);
      SetLength(rdata,count);
      THdf5.FH5.H5Aread(datasetId, typeId, @rdata[0]);
      for i := 0 to Length(rdata)-1 do
      begin
          if rdata[i] = nil then
             continue;
          var len : Integer := 0;
          var Data : PByte := rdata[i];
          while Data[i] <> 0 do
           inc(len);
          var buffer : Tarray<Byte> ; SetLength(buffer,len);
          CopyMemory(@buffer[0],Data,len);
          var s : string := THdf5Utils.ReadStringBuffer(buffer);
          strs.Add(s);
           THdf5.FH5.H5free_memory(rdata[i]);
      end;
    finally
      strs.free;
    end;
    THdf5.FH5.H5Tclose(typeId);
    THdf5.FH5.H5Aclose(datasetId);
    Result := Tuple.Create(true, strs.ToArray);
end;

class function THdf5.WriteAttribute<T>(groupId: Int64; name: string; attribute: T): Tuple<Integer, Int64>;
begin
    Result:= WriteAttributes<T>(groupId, name, [ attribute ]);
end;

class function THdf5.WriteAttributes<T>(groupId: Int64; name: string; attributes: TArray<T>): Tuple<Integer, Int64>;
var
  typeCode : PTypeInfo;
  StringAttrs : TArray<string>;
begin

    typeCode :=  TypeInfo(T);
    if typeCode = TypeInfo(Boolean) then
    begin
         //var bls = collection.ConvertArray<bool, ushort>(Convert.ToUInt16);
        // result = rw.WriteFromArray<ushort>(groupId, name, bls);
    end
   else if typeCode = TypeInfo(char) then
    begin
       // var chrs = collection.ConvertArray<char, ushort>(Convert.ToUInt16);
       // result = rw.WriteFromArray<ushort>(groupId, name, chrs);
    end
    else if typeCode = TypeInfo(ansichar) then
    begin
        //var chrs = collection.ConvertArray<char, ushort>(Convert.ToUInt16);
        //result = rw.WriteFromArray<ushort>(groupId, name, chrs);
    end
    else if typeCode = TypeInfo(Byte) then
    begin
        var a : TArray<Byte>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Byte> ];
        end;
        result := WritePrimitiveAttribute<Byte>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Int8) then
    begin
        var a : TArray<Int8>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Int8> ];
        end;
        result := WritePrimitiveAttribute<Int8>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Single) then
    begin
        var a : TArray<Single>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Single> ];
        end;
        result := WritePrimitiveAttribute<Single>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Double) then
    begin
        var a : TArray<Double>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Double> ];
        end;
        result := WritePrimitiveAttribute<Double>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Int16) then
    begin
        var a : TArray<Int16>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Int16> ];
        end;
        result := WritePrimitiveAttribute<Int16>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Int32) then
    begin
        var a : TArray<Int32>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Int32> ];
        end;
        result := WritePrimitiveAttribute<Int32>(groupId, name, a);
    end
    else if typeCode = TypeInfo(Int64) then
    begin
        var a : TArray<Int64>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<Int64> ];
        end;
        result := WritePrimitiveAttribute<Int64>(groupId, name, a);
    end
    else if typeCode = TypeInfo(UInt16) then
    begin
        var a : TArray<UInt16>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<UInt16> ];
        end;
        result := WritePrimitiveAttribute<UInt16>(groupId, name, a);
    end
    else if typeCode = TypeInfo(UInt32) then
    begin
        var a : TArray<UInt32>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<UInt32> ];
        end;
       result := WritePrimitiveAttribute<UInt32>(groupId, name, a);
    end
    else if typeCode = TypeInfo(UInt64) then
    begin
        var a : TArray<UInt64>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<UInt64> ];
        end;
        result := WritePrimitiveAttribute<UInt64>(groupId, name, a);
    end
    else if typeCode = TypeInfo(string)  then
    begin
        //var a : TArray<string>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            StringAttrs := StringAttrs + [ v.AsType<string> ];
        end;
        result := WriteStringAttributes(groupId, name, StringAttrs)
    end
    else if typeCode = TypeInfo(AnsiString)  then
    begin
        var a : TArray<string>;
        for var i := 0 to Length(attributes)-1  do
        begin
            var v := TValue.From<T>(attributes[i]);
            a := a + [ v.AsType<AnsiString> ];
        end;
        result := WriteStringAttributes(groupId, name, a);
    end

end;

class function THdf5.WriteDatasetFromArray<T>(groupId: Int64; name: string; dset: TArray<T>; Shape: TArray<Integer>): Tuple<Integer, Int64>;
 var
   rank,iRes : Integer;
   dims      : TArray<UInt64>;
   maxDims   : TArray<UInt64>;
   spaceId,
   datatype,
   typeId,
   datasetId : Int64;
   normalizedName : string;
begin

    rank    := 1;
    dims    := [Length(dset)];
    if Length(Shape) > 0 then
    begin
        rank := Length(Shape);
        dims := [];
        for var i := 0 to Length(Shape) - 1 do
          dims := dims + [ Shape[i] ];
    end;

    maxDims := [];
    spaceId := FH5.H5Screate_simple(rank, @dims[0], nil);
    datatype:= GetDatatype(TypeInfo(T));
    typeId  := FH5.H5Tcopy(datatype);
    if datatype = FH5.H5T_C_S1 then
       FH5.H5Tset_size(datatype, 2);

    normalizedName := THdf5Utils.NormalizedName(name);
    datasetId  := THdf5Utils.GetDatasetId(groupId, normalizedName, datatype, spaceId);
    if datasetId = -1 then
      Exit( Tuple<Integer, Int64>.Create(-1, -1) );

    iRes := FH5.H5Dwrite(datasetId, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, @dset[0]);

    FH5.H5Dclose(datasetId);
    FH5.H5Sclose(spaceId);
    FH5.H5Tclose(typeId);
    Result:= Tuple.Create(iRes, datasetId);
end;

class function THdf5.WritePrimitiveAttribute<T>(groupId: Int64; name: string; attributes: TArray<T>): Tuple<Integer, Int64>;
 var
   rank,iRes  : Integer;
   dims       : TArray<UInt64>;
   maxDims    : TArray<UInt64>;
   spaceId,
   datatype,
   typeId,
   tmpId,
   attributeId: Int64;
   nameToUse  : string;
begin
    tmpId := groupId;
    rank  := 1;
    dims := [Length(attributes)];
    maxDims := [];
    spaceId := FH5.H5Screate_simple(rank, @dims[0], nil);
    datatype := GetDatatype(TypeInfo(T));
    typeId   := FH5.H5Tcopy(datatype);
    nameToUse := THdf5Utils.NormalizedName(name);
    //var attributeId = H5A.create(groupId, nameToUse, datatype, spaceId);
    attributeId := THdf5Utils.GetAttributeId(groupId, nameToUse, datatype, spaceId);
    iRes        := FH5.H5Awrite(attributeId, datatype, @attributes[0]);
    FH5.H5Aclose(attributeId);
    FH5.H5Sclose(spaceId);
    FH5.H5Tclose(typeId);
    if tmpId <> groupId then
       FH5.H5Dclose(groupId);
    Result:= Tuple.Create(iRes, attributeId);
end;

class function THdf5.WriteStringAttributes(groupId: Int64; name: string; values: TArray<string>; groupOrDatasetName: string): Tuple<Integer, Int64>;
var
  tmpId,
  datasetId,
  datatype,
  spaceId,
  attributeId    : Int64;
  strSz,cntr,res : Integer;
  normalizedName : string;
  wdata          : TArray<PByte>;
  buff           : TArray<Byte>;
begin
    tmpId := groupId;
    if not string.IsNullOrWhiteSpace(groupOrDatasetName) then
    begin
        var normName : AnsiString := AnsiString( THdf5Utils.NormalizedName(groupOrDatasetName) );
        datasetId := FH5.H5Dopen2(groupId, PAnsiChar(normName), H5P_DEFAULT);
        if datasetId > 0 then
          groupId := datasetId;
    end else
    begin
    end;
    // create UTF-8 encoded attributes
    datatype := FH5.H5Tcreate(H5T_class_t.H5T_STRING, H5T_VARIABLE);
    FH5.H5Tset_cset(datatype, THdf5Utils.GetCharacterSet(FSetting.CharacterSetType));
    FH5.H5Tset_strpad(datatype, THdf5Utils.GetCharacterPadding(FSetting.CharacterPaddingType));
    strSz   := Length(values);
    var aDims : TArray<uint64>:= [strSz];
    spaceId := FH5.H5Screate_simple(1, @aDims[0], nil);
    normalizedName := THdf5Utils.NormalizedName(name);
    attributeId := THdf5Utils.GetAttributeId(groupId, normalizedName, datatype, spaceId);
    SetLength(wdata,strSz) ;
    cntr := 0;
    for var str in values do
    begin
        buff := THdf5Utils.StringToByte(str);
        wdata[cntr] := @buff[0];
        Inc(cntr);
    end;
    res := FH5.H5Awrite(attributeId, datatype, @wdata[0]);
    FH5.H5Aclose(attributeId);
    FH5.H5Sclose(spaceId);
    FH5.H5Tclose(datatype);
    if tmpId <> groupId then
       FH5.H5Dclose(groupId);
    Result := Tuple.Create(res, attributeId);

end;

class function THdf5.OpenAttributeIfExists(fileOrGroupId: Int64; name, alternativeName: string): Int64;
begin
    if THdf5Utils.ItemExists(fileOrGroupId, name, Hdf5ElementType.Attribute) then
    begin
        Result := THdf5.FH5.H5Aopen(fileOrGroupId, PAnsiChar(AnsiString(name)), H5P_DEFAULT);
        Exit;
    end;
    if THdf5Utils.ItemExists(fileOrGroupId, alternativeName, Hdf5ElementType.Attribute) then
    begin
        Result := THdf5.FH5.H5Aopen(fileOrGroupId, PAnsiChar(AnsiString(alternativeName)), H5P_DEFAULT);
        Exit;
    end;
    Result := -1;
end;

initialization
begin
    THdf5.FIsInizialized  := False;
    if FileExists(NomeHdf5Dll) then
    begin
       THdf5.FH5 := THDF5Dll.Create(NomeHdf5Dll);
       if  THdf5.FH5.IsValid then
       begin
           THdf5.FIsInizialized := True;
           THdf5.FSetting       := TSettings.Create;
       end;
    end;
end;

finalization
begin
    if  THdf5.FIsInizialized then
    begin
        THdf5.FSetting.Free;
        THdf5.FH5.Free;
    end;
end;

end.
