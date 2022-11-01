unit TensorFlow.EagerTensor;
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
   uses System.SysUtils,
        System.Rtti,

        TF4D.Core.CApi,
        TensorFlow.DApiEager,
        TensorFlow.DApi;

type

  TEagerTensor = class(TFTensor)
    protected
       procedure NewEagerTensorHandle(h:Pointer);
       procedure NativeDispose(hnd: Pointer); override;
    private
       m_Device : string;
       procedure Resolve;
       function GetDeviceName: string;

    public
       constructor Create(h:Pointer);overload;
       constructor Create(h: Pointer;NewEagerTensor: Boolean);overload; override;
       constructor Create(shape: TFShape;dType: TF_DataType);overload;
       constructor Create(value: TValue; shape: PTFShape); overload;

       constructor Create(bytes: TArray<TF_TString>;shape: TFShape);overload;

       constructor Create(bytes: TArray<Boolean>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Boolean>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Boolean>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Byte>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Byte>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Byte>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int16>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int16>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int16>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int32>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int32>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int32>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Int64>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Int64>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Int64>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<UInt64>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<UInt64>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<UInt64>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Single>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Single>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Single>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Single>>>>;shape: TFShape; dtype: TF_DataType);overload;

       constructor Create(bytes: TArray<Double>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<Double>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<Double>>>;shape: TFShape; dtype: TF_DataType);overload;
       constructor Create(bytes: TArray<TArray<TArray<TArray<Double>>>>;shape: TFShape; dtype: TF_DataType);overload;
       destructor Destroy; override;

       class function GetRank(eTensor:TEagerTensor): Integer;
       class function GetDims(eTensor:TEagerTensor): TArray<Integer>;

       procedure Setshape(const Value: TFShape);
       /// <summary>
       /// _create_substitute_placeholder
       /// </summary>
       /// <returns></returns>
       function  AsPlaceholder(name: string = ''): TFTensor;
       function  AsConstant(name: string = ''): TFTensor;
       procedure copy_handle_data(target_t: TFTensor);

       property Device : string read GetDeviceName;

  end;

implementation
      uses System.TypInfo,
           Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops;

{ TEagerTensor }

constructor TEagerTensor.Create(h: Pointer);
begin
    EagerTensorHandle := h;
    Resolve;
end;

constructor TEagerTensor.Create(h: Pointer;NewEagerTensor: Boolean);
begin
     NewEagerTensorHandle(h);
end;

constructor TEagerTensor.Create(shape: TFShape;dType: TF_DataType);
begin
    inherited Create(shape,dType);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Boolean>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Boolean>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Byte>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int16>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int16>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int32>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int32>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int64>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int64>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<UInt64>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<UInt64>>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<UInt64>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Single>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(TFTensor.InitTensor<Single>(bytes,shape,dtype));
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Single>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Single>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Double>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype) );
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Double>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype) );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Double>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(value: TValue; shape: PTFShape);
var
  aDim  : Integer;
  dtype : TF_DataType;
  vValue: TValue;
  lIsArray : Boolean;
begin
    aDim := 0;
    vValue := value;
    lIsArray := False;
    while vValue.IsArray do
    begin
        lIsArray := True;
        vValue := vValue.GetArrayElement(0);
        inc(aDim)
    end;

    var v : TFShape;
    if (shape = nil) or (shape.IsNil) then
    begin
        v := TUtils.GetShape(value);
        shape := @v;
    end;

    dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
        Exit;
    end;

    if (lIsArray) and (dtype = TF_DOUBLE) and (string.LowerCase(vValue.TypeInfo.Name).Contains('extended')) then
    begin
        case aDim of
          1: begin
             var aValue : TArray<Double> := [];
             var a := value.AsType< TArray<Extended> > ;
             for var i := 0 to Length(a)- 1 do
               aValue := aValue + [ Double(a[i]) ];

             Value := TValue.From< TArray<Double> >(aValue);
          end;
          2:begin
             var aValue : TArray< TArray<Double> > ;
             var a := value.AsType< TArray< TArray<Extended>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                   aValue[i] := aValue[i] + [ Double(a[i][j]) ];
             end;

             Value := TValue.From< TArray<TArray<Double>> >(aValue);
          end;
          3: begin
             var aValue : TArray<TArray<TArray<Single>>> ;
             var a := value.AsType< TArray<TArray<TArray<Single>>> > ;
             for var i := 0 to Length(a) -1 do
             begin
                 SetLength(aValue,Length(aValue)+1);
                 for var j := 0 to Length(a[i])- 1 do
                 begin
                      SetLength(aValue[i],Length(aValue[i])+1);
                      for var k := 0 to Length(a[i][j])- 1 do
                        aValue[i][j] := aValue[i][j] + [ Double(a[i][j][k]) ];
                 end;
             end;
          end;
        end;
    end;


    case aDim of
       0 : begin
         case dtype of
           TF_FLOAT:  Create( Single(value.AsExtended) );
           TF_DOUBLE: Create( double(value.AsExtended) );
           TF_INT32:  Create( value.AsInteger );
           TF_UINT8:  Create( Byte(value.AsOrdinal) );
           TF_INT16:  Create( int16(value.AsOrdinal) );
           TF_INT8:   Create( int8(value.AsOrdinal) ) ;
           TF_STRING: Create( AnsiString(value.AsString) );
           TF_INT64:  Create( value.AsInt64 );
           TF_BOOL:   Create( value.AsBoolean );
           TF_UINT16: Create( word(value.AsOrdinal) );
           TF_UINT32: Create( Cardinal(value.AsOrdinal) );
           TF_UINT64: Create( value.AsUInt64 );
         end;
       end;
       1 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<Single> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<Double> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<Int32> >,    shape,dtype) ;
           TF_UINT8:  Create( value.AsType< TArray<UInt8> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<Int16> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<Int8> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<string> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<Int64> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<Boolean> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<UInt16> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<UInt32> >,  shape,dtype);
           TF_UINT64: Create( value.AsType< TArray<UInt64> >,  shape,dtype) ;
         end;
       end;
       2 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<TArray<Single>> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<TArray<Double>> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<TArray<Int32>> >,    shape,dtype);
           TF_UINT8:  Create( value.AsType< TArray<TArray<UInt8>> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<TArray<Int16>> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<TArray<Int8>> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<TArray<string>> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<TArray<Int64>> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<TArray<Boolean>> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<TArray<UInt16>> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<TArray<UInt32>> >,  shape,dtype) ;
           TF_UINT64: Create( value.AsType< TArray<TArray<UInt64>> >,  shape,dtype) ;
         end;
       end;
       3 : begin
         case dtype of
           TF_FLOAT:  Create( value.AsType< TArray<TArray<TArray<Single>>> >,  shape,dtype) ;
           TF_DOUBLE: Create( value.AsType< TArray<TArray<TArray<Double>>> >,  shape,dtype) ;
           TF_INT32:  Create( value.AsType< TArray<TArray<TArray<Int32>>> >,    shape,dtype) ;
           TF_UINT8:  Create( value.AsType< TArray<TArray<TArray<UInt8>>> >,    shape,dtype) ;
           TF_INT16:  Create( value.AsType< TArray<TArray<TArray<Int16>>> >,    shape,dtype) ;
           TF_INT8:   Create( value.AsType< TArray<TArray<TArray<Int8>>> >,      shape,dtype) ;
           TF_STRING: Create( value.AsType< TArray<TArray<TArray<string>>> >,  shape,dtype) ;
           TF_INT64:  Create( value.AsType< TArray<TArray<TArray<Int64>>> >,    shape,dtype) ;
           TF_BOOL:   Create( value.AsType< TArray<TArray<TArray<Boolean>>> >,shape,dtype) ;
           TF_UINT16: Create( value.AsType< TArray<TArray<TArray<UInt16>>> >,  shape,dtype) ;
           TF_UINT32: Create( value.AsType< TArray<TArray<TArray<UInt32>>> >,  shape,dtype);
           TF_UINT64: Create( value.AsType< TArray<TArray<TArray<UInt64>>> >,  shape,dtype) ;
         end;

       end;

    end;

end;


procedure TEagerTensor.copy_handle_data(target_t: TFTensor);
begin
    if (target_t.dtype = TF_DataType.TF_RESOURCE) or (target_t.dtype = TF_DataType.TF_VARIANT) then
    begin
        // need to export
        //(target_t.graph, target_t._as_tf_output(), 0, new IntPtr[0], new int[0], new DataType[0], tf.Status.Handle);
    end;
end;

function TEagerTensor.AsConstant(name: string): TFTensor;
begin
   Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(nil),
                                          function(v1: TControlDependenciesController): TFTensor
                                            begin
                                                Result := tf.constant(numpy, DtInvalid, nil, name)
                                            end );
end;

function TEagerTensor.AsPlaceholder(name: string): TFTensor;
begin
    var placeholder := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(nil),
                                          function(v1: TControlDependenciesController): TFTensor
                                            begin
                                                Result := tf.placeholder(dtype, nil, name)
                                            end );
    copy_handle_data(placeholder);
    Result := placeholder;
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype));
    NewEagerTensorHandle(Handle);
end;

destructor TEagerTensor.Destroy;
begin
   inherited Destroy;
end;

procedure TEagerTensor.NativeDispose(hnd: Pointer);
begin
 if Assigned(hnd) then
   TFE_DeleteTensorHandle(hnd);
end;

function TEagerTensor.GetDeviceName: string;
begin
    m_Device :=  string(AnsiString( TFE_TensorHandleDeviceName(EagerTensorHandle, tf.Status.Handle)));
    Result := m_Device;
end;

class function TEagerTensor.GetDims(eTensor: TEagerTensor): TArray<Integer>;
var
  dims : TArray<Integer>;

begin
    var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
    SetLength(dims, TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle));
    for var i := 0 to Length(dims)-1 do
        dims[i] := TFE_TensorHandleDim(tfe_tensor_handle, i, tf.Status.Handle);
    Result := dims;
end;

class function TEagerTensor.GetRank(eTensor: TEagerTensor): Integer;
begin
     var tfe_tensor_handle := TFE_NewTensorHandle(eTensor.handle,tf.Status.Handle);
     Result := TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TF_TString>; shape: TFShape);
begin
    inherited Create( StringTensor(bytes,shape) );
    NewEagerTensorHandle(Handle);
end;

procedure TEagerTensor.NewEagerTensorHandle(h: Pointer);
begin
    EagerTensorHandle := TFE_NewTensorHandle(h,tf.Status.Handle) ;
    tf.Status.RaiseEx;
end;

procedure TEagerTensor.Resolve;
begin
    Handle := TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

procedure TEagerTensor.Setshape(const Value: TFShape);
begin
    if not Shape.is_compatible_with(Value) then
        raise Exception.Create('Tensor''s shape is not compatible.');

    if value.IsNil  then
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, nil, -1, tf.Status.Handle)
    else
      TF_GraphSetTensorShape(graph.Handle, _as_tf_output, @value.dims, value.ndim, tf.Status.Handle);
    tf.Status.RaiseEx;
end;

end.
