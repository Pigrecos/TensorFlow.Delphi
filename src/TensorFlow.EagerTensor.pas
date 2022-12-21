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
        System.Generics.Collections,
        System.Rtti,

        TF4D.Core.CApi,
        TensorFlow.DApiEager,
        TensorFlow.DApi,
        TensorFlow.Tensors.Ragged;

type

  TEagerTensor = class(TFTensor)
    protected
       procedure NewEagerTensorHandle(h:Pointer);
       procedure NativeDispose(hnd: Pointer); override;
    private
       m_Device : string;
       procedure Resolve;
       function GetDeviceName: string;
    protected
       function GetShape: TFShape;  override;
       procedure Setshape(const Value: TFShape);  override;

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

       /// <summary>
       /// _create_substitute_placeholder
       /// </summary>
       /// <returns></returns>
       function  AsPlaceholder(name: string = ''): TFTensor;
       function  AsConstant(name: string = ''): TFTensor;
       procedure copy_handle_data(target_t: TFTensor);
       function  ToString: string;override;

       property Device : string read GetDeviceName;

  end;

  TEagerTensorArray = class(TTensorArray)
    private
      Fdynamic_size     : Boolean;
      Felement_shape    : TFShape;
      Fcolocate_with    : TList<TFTensor>;
      Fclear_after_read : Boolean;
      Ftensor_array     : TList<TFTensor>;

      function size(name: string = ''): TFTensor;
    protected

    public
      constructor Create(_dtype: TF_DataType; size: TFTensor; dynamic_size: Boolean = false;
                         clear_after_read : Boolean= true; tensor_array_name : string = ''; _handle : TFTensor= nil;
                         _flow : TFTensor = nil; _infer_shape: Boolean = true; _element_shape : PTFShape= nil;
                         _colocate_with_first_write_call : Boolean= true; _name: string = '');
      function  scatter(indices: TFTensor; value: TFTensor; name: string = ''): TTensorArray;
      procedure _merge_element_shape(shape: TFShape);
      procedure _maybe_colocate_with(value: TFTensor);
      function read<T>(index: T; name: string = ''): TFTensor; reintroduce;
      function unstack(value: TFTensor; name: string = ''): TTensorArray; override;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; override;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; reintroduce; overload;
      function stack(name: string = ''): TFTensor; override;
      function gather(indices: TFTensor; name: string = ''): TFTensor; override;
  end;

implementation
      uses System.TypInfo,

           Tensorflow,
           Tensorflow.NameScope,
           Tensorflow.Utils,
           TensorFlow.Ops,
           TensorFlow.Constant_op,
           TensorFlow.gen_data_flow_ops,
           Tensorflow.math_ops,
           Tensorflow.array_ops,
           NumPy.NDArray,
           Numpy;

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
     inherited Create( TFTensor.InitTensor(shape,bytes,dtype) );
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
    aDim     := 0;
    vValue   := value;
    lIsArray := False;
    dtype    := dtInvalid;
    while vValue.IsArray do
    begin
        if value.GetArrayLength < 1 then
        begin
           var ttt := value.TypeData^.DynArrElType^ ;
           dtype := TDTypes.as_tf_dtype(ttt);
           Break;
        end;
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

    if value.GetArrayLength > 0 then
      dtype:= TUtils.GetDataType(value);

    if (shape.Size = 0) and (dtype <> TF_DataType.TF_STRING ) then
    begin
        inherited Create(shape, dtype);
        NewEagerTensorHandle(Handle);
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

function TEagerTensor.ToString: string;
begin
    var nd  := TNDArray.Create(Self);
    var str := NDArrayRender.ToString(nd);
    Result  := Format('tf.Tensor: shape=%s, dtype=%s, numpy=%s',[Shape.ToString, Tdtypes.ToString(dtype), str]);
end;

procedure TEagerTensor.Setshape(const Value: TFShape);
begin
    if not Shape.is_compatible_with(Value) then
        raise Exception.Create('Tensor''s shape is not compatible.');
end;

function TEagerTensor.GetShape: TFShape;
begin
    var sShape := System.default(TFShape);

    if rank < 0 then
        Exit(sShape);

    var numDims := TFE_TensorHandleNumDims(eagerTensorHandle, tf.Status.Handle);
    var dims : TArray<Integer> ;
    if numDims > 0 then
    begin
        SetLength(dims,numDims) ;

        for var i := 0 to Length(dims)- 1 do
          dims[i] := TFE_TensorHandleDim(eagerTensorHandle, i, tf.Status.Handle);
    end;
    Result := dims;
end;

{ TEagerTensorArray }

constructor TEagerTensorArray.Create(_dtype: TF_DataType; size: TFTensor; dynamic_size, clear_after_read: Boolean; tensor_array_name: string; _handle, _flow: TFTensor;
  _infer_shape: Boolean; _element_shape: PTFShape; _colocate_with_first_write_call: Boolean; _name: string);
begin
    Fflow             := constant_op.constant(Integer(0));
    Finfer_shape      := _infer_shape;
    if Assigned(_element_shape) then Felement_shape := _element_shape^
    else                             Felement_shape := TFShape.null;

    Fdtype            := Tdtypes.as_base_dtype(_dtype);
    Fdynamic_size     := dynamic_size;
    Fclear_after_read := clear_after_read;
    Ftensor_array     := TList<TFTensor>.Create;
    Fcolocate_with_first_write_call := colocate_with_first_write_call;
end;

function TEagerTensorArray.gather(indices: TFTensor; name: string): TFTensor;
var
  element_shape : TFShape;
  value         : TFTensor;
begin
    element_shape := TFShape.Null;

    value := gen_data_flow_ops.tensor_array_gather_v3(Fhandle, indices, Fflow, Fdtype, @element_shape, name) ;

    //if (element_shape != null)
    //value.set_shape(-1, element_shape.dims);

    Result := value;
end;

function TEagerTensorArray.read<T>(index: T; name: string): TFTensor;
var
  index_int : Integer;
begin
    index_int := -1;
    var v : TValue := TValue.From<T>(index);
    if v.TypeInfo = TypeInfo(Integer) then
    begin
        index_int := v.AsInteger
    end
    else if v.TypeInfo = TypeInfo(TFTensor) then
    begin
        var nd : NDArray := v.AsType<TFTensor>.numpy;
        index_int := nd;
    end else
        raise Exception.Create('read<T> Error');

    if Fclear_after_read then
      Ftensor_array[index_int] := nil;

    Result := Ftensor_array[index_int];
end;

function TEagerTensorArray.scatter(indices, value: TFTensor; name: string): TTensorArray;
begin
    raise Exception.Create('Error Not Implemented scatter');
end;

function TEagerTensorArray.size(name: string): TFTensor;
begin
    Result := gen_data_flow_ops.tensor_array_size_v3(Fhandle, Fflow, name);
end;

function TEagerTensorArray.stack(name: string): TFTensor;
begin
    Tops.colocate_with(Fhandle);

    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'TensorArrayStack', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              var Limit : TValue :=  size;
                              Result := gather(math_ops.range(0, @Limit), name);
                          end);
end;

function TEagerTensorArray.unstack(value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayUnstack', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              var num_elements : TFTensor := array_ops.shape(value)[0];
                              var Limit : TValue :=  num_elements;
                              Result := scatter(math_ops.range(0, @Limit), value, name);
                          end);
end;

function TEagerTensorArray.write(index, value: TFTensor; name: string): TTensorArray;
begin
    if Finfer_shape then
        Felement_shape := Felement_shape.merge_with(value.shape);

    Ftensor_array.add(value);

    Result := self;
end;

function TEagerTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    var value_tensor := Tops.convert_to_tensor(TValue.From<T>(value), DtInvalid, 'value', False, Fdtype );
    var index_tensor := Tops.convert_to_tensor(index, DtInvalid, 'index');
    Result := write(index_tensor, value_tensor, name);
end;

procedure TEagerTensorArray._maybe_colocate_with(value: TFTensor);
begin
    Fcolocate_with.Add(value);
end;

procedure TEagerTensorArray._merge_element_shape(shape: TFShape);
begin
    Felement_shape.concatenate(shape);
end;

end.
