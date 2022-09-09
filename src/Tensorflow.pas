unit Tensorflow;

interface
  uses System.SysUtils, System.Rtti,  System.TypInfo,
       quick.Logger,
       System.Generics.Collections,
       Spring.Collections.Dictionaries,
       Spring.Collections.Extensions,
       Spring.Collections.Stacks,
       Spring,
       Quick.Logger.Provider.Files,
       TensorFlow.LowLevelAPI,
       TensorFlow.DApiBase,
       TensorFlow.DApi,
       TensorFlow.DApiOperations,
       TensorFlow.Context,
       TensorFlow.EagareRunner,
       TensorFlow.DApiEager,
       Tensorflow.Utils,

       ProtoGen.Tensor,
       Protogen.tensorShape,
       ProtoGen.attrValue,
       ProtoGen.types,
       ProtoGen.opDef,
       protogen.config;



const
  C_GRAPH_MODE : Integer = 0;
  C_EAGER_MODE : Integer = 1;

type
  TTensors = class (TEmptyEnumerable<TFTensor>)
  private
    Fitems : TList<TFTensor> ;
    Fdtype : TF_DataType;
    Fshape : TFShape;
    Frank  : Integer;
    Fgraph : TFGraph;
    FIsList: Boolean;
    FLength: Integer;
    FIsCreatedInGraphMode : Boolean;
    function  GetItem(index: Integer): TFTensor;
    procedure SetItem(index: Integer; const Value: TFTensor);
    function Getdtype: TF_DataType;
    function Getshape: TFShape;
    function GetRank: Integer;
    function GetGraph: TFGraph;
    function GetLen: Integer;

  public
     constructor Create(tensors: TArray<TFTensor>);
     procedure   Add(tensor: TFTensor);
     procedure   AddRange(tensors: TArray<TFTensor>);
     procedure   Insert(index: Integer; tensor: TFTensor);

     property IsCreatedInGraphMode: Boolean  read FIsCreatedInGraphMode write FIsCreatedInGraphMode;
     property IsList: Boolean      read FIsList write FIsList;
     property Length: Integer      read GetLen;
     property graph : TFGraph      read GetGraph;
     property rank  : Integer      read GetRank;
     property shape : TFShape      read Getshape;
     property dtype : TF_DataType  read Getdtype;
     property item[index: Integer]: TFTensor  read GetItem write SetItem; default;
  end;


  TEagerTensor = class(TFTensor)
    protected
       procedure NewEagerTensorHandle(h:Pointer);
    private
       m_Device : string;
       procedure Resolve;
       function GetDeviceName: string;

    public
       constructor Create(h:Pointer);overload;
       constructor Create(h: Pointer;NewEagerTensor: Boolean);overload;
       constructor Create(shape: TFShape;dType: TF_DataType);overload;

       constructor Create(bytes: TArray<TFString>;shape: TFShape);overload;

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

       class function GetRank(eTensor:TEagerTensor): Integer;
       class function GetDims(eTensor:TEagerTensor): TArray<Integer>;

       procedure Setshape(const Value: TFShape);

       property Device : string read GetDeviceName;

  end;

  OpDefLibrary = class
    private
      class function SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
      class function _IsListParameter(arg: TArgDef): Boolean;
      class function _IsListValue(v: TValue): Boolean;
      class procedure SetAttrs(op_type_name : string;
                               input_arg    : TArgDef;
                               op_def       : TOpDef;
                               attrs        : TDictionary<string, TValue>;
                               inferred_from: TDictionary<string, TValue>;
                               types        : TList<TF_DataType>;
                               base_types   : TList<TF_DataType>;
                               input_types  : TList<TF_DataType>;
                               values       : TValue);
    public
      class function _MakeType(v: TF_DataType; attr_def: TAttrDef): TDataType;
      class function _MakeShape(shape: TFShape; attr_def: TAttrDef): TTensorShapeProto;
      class function _apply_op_helper(op_type_name: string;name: string = ''; args : TArray<TParameter> = nil): TFOperation; overload;
      class function _apply_op_helperDict(op_type_name: string; name: string = ''; keywords: TDictionary<string, TValue> = nil): TFOperation;overload;

  end;


  TTensorflow = class(TFDisposable)
    private
      function GetVersion: string;
    protected
		  procedure NativeDispose(hnd: Pointer); override;

    public
      byte8_t   : TF_DataType;
      int8_t    : TF_DataType;
      int16_t   : TF_DataType;
      int32_t   : TF_DataType;
      int64_t   : TF_DataType;
      float16_t : TF_DataType;
      float32_  : TF_DataType;
      float64_t : TF_DataType;
      bool_t    : TF_DataType;
      chars_t   : TF_DataType;
      string_t  : TF_DataType;

      Status  : TFStatus;
      Context : TContext;
      OpDefLib: OpDefLibrary;
      Runner  : TEagerRunner;

      constructor Create;
      procedure   enable_eager_execution;
      function    executing_eagerly:Boolean;
      function    get_default_graph: TFgraph;
      procedure   reset_default_graph;
      function    peak_default_graph: TFgraph;

      function convert_to_tensor(value: TValue; dtype: TF_DataType= TF_DATATYPE_UNKNOWN; name: string= ''; preferred_dtype: TF_DataType=TF_DATATYPE_UNKNOWN): TFTensor;
      /// <summary>
      ///
      /// </summary>
      /// <param name="value"></param>
      /// <param name="dtype"></param>
      /// <param name="shape"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function constant(value: TValue; dtype : TF_DataType = TF_DATATYPE_UNKNOWN; shape : PTFShape= nil; name : AnsiString = 'Const'): TFTensor;
      /// <summary>
      ///     Creates a new graph.
      /// </summary>
      ///<remarks>Has no interaction with graph defaulting. Equivalent to new Graph();</remarks>
      function Graph: TFGraph;

      property Version : string read GetVersion;
  end;

  TConstant_op = class
    private
      class function convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType=TF_DATATYPE_UNKNOWN): TFTensor; overload;
      class function _eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
      class function _eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
    public
      class function convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;
      class function convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape: Boolean; allow_broadcast: Boolean) : TFTensor;overload;
      /// <summary>
      /// Creates a constant tensor.
      ///
      /// The resulting tensor is populated with values of type `dtype`, as
      /// specified by arguments `value` and (optionally) `shape`
      /// </summary>
      /// <param name="value">A constant value (or list) of output type `dtype`.</param>
      /// <param name="dtype">The type of the elements of the resulting tensor.</param>
      /// <param name="shape">Optional dimensions of resulting tensor.</param>
      /// <param name="name">Optional name for the tensor.</param>
      /// <returns></returns>
      class function constant(value: PValue; dtype : TF_DataType= TF_DATATYPE_UNKNOWN;
                              shape : PTFShape = nil; verify_shape : Boolean = false;
                              allow_broadcast : Boolean = true; name : AnsiString = 'Const'): TFTensor;

  end;

  var
   tf : TTensorflow;

implementation
   uses Oz.Pb.Classes, Oz.SGL.Collections,oz.Pb.StrBuffer, pbPublic, pbInput, pbOutput,
        NDArray,
        TensorFlow.Ops , Numpy.Axis;

{ TTensorflow }

function TTensorflow.convert_to_tensor(value: TValue; dtype: TF_DataType; name: string; preferred_dtype: TF_DataType): TFTensor;
begin
    Result := TOps.convert_to_tensor(value,dtype, name,False,preferred_dtype);
end;


function TTensorflow.constant(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString): TFTensor;
begin
    Result :=TConstant_op.constant(@value,
                                    dtype,
                                    shape,
                                    False,
                                    True,
                                    name);
end;

constructor TTensorflow.Create;
begin
    byte8_t   := TF_DataType.TF_UINT8;
    int8_t    := TF_DataType.TF_INT8;
    int16_t   := TF_DataType.TF_INT16;
    int32_t   := TF_DataType.TF_INT32;
    int64_t   := TF_DataType.TF_INT64;
    float16_t := TF_DataType.TF_HALF;
    float32_  := TF_DataType.TF_FLOAT;
    float64_t := TF_DataType.TF_DOUBLE;
    bool_t    := TF_DataType.TF_BOOL;
    chars_t   := TF_DataType.TF_STRING;
    string_t  := TF_DataType.TF_STRING;

    Context   := TContext.Create;
    Status    := TFStatus.Create;
    OpDefLib  := OpDefLibrary.Create;
    runner    := TEagerRunner.Create ;

    Logger.Providers.Add(GlobalLogFileProvider);
    with GlobalLogFileProvider do
    begin
      FileName := '.\Logs.log';
      LogLevel := LOG_ALL;
      TimePrecission := True;
      MaxRotateFiles := 3;
      MaxFileSizeInMB := 5;
      RotatedFilesPath := '.\RotatedLogs';
      CompressRotatedFiles := False;
      Enabled := True;
    end;
end;

procedure TTensorflow.enable_eager_execution;
begin
    Context.eager_mode;
end;

function TTensorflow.executing_eagerly: Boolean;
begin
    Result := Context.executing_eagerly;
end;

function TTensorflow.GetVersion: string;
begin
     Result := string(AnsiString(TF_Version));
end;

procedure TTensorflow.NativeDispose(hnd: Pointer);
begin
  inherited;

  Context.Free;
  Status.Free;
  OpDefLib.Free;

end;

function TTensorflow.get_default_graph: TFgraph;
begin
    Result := TOps.get_default_graph;
end;

function TTensorflow.Graph: TFGraph;
begin
    Result := TFGraph.Create;
end;

function TTensorflow.peak_default_graph: TFgraph;
begin
    Result := TOps.peak_default_graph;
end;

procedure TTensorflow.reset_default_graph;
begin
    TOps.reset_default_graph
end;

{ TEagerTensor }

constructor TEagerTensor.Create(h: Pointer);
begin
    EagerTensorHandle := h;
    Resolve;

    self.MMLck();
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
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Boolean>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Boolean>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Boolean>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Boolean>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Byte>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Byte>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Byte>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Byte>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Byte>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int16>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int16>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int16>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype).Handle );
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int16>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int16>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int32>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int32>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int32>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int32>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int32>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Int64>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Int64>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Int64>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Int64>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Int64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<UInt64>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<UInt64>>;shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<UInt64>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<UInt64>>>>;shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<UInt64>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Single>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Single>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Single>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Single>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Single>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<Double>; shape: TFShape; dtype: TF_DataType);
begin
     inherited Create(bytes);
     NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<Double>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create(bytes);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<Double>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
end;

constructor TEagerTensor.Create(bytes: TArray<TArray<TArray<TArray<Double>>>>; shape: TFShape; dtype: TF_DataType);
begin
    inherited Create( TFTensor.InitTensor<Double>(bytes,shape,dtype).Handle);
    NewEagerTensorHandle(Handle);
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

constructor TEagerTensor.Create(bytes: TArray<TFString>; shape: TFShape);
begin
    inherited Create( StringTensor(bytes,shape).Handle );
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

{ OpDefLibrary }

class function OpDefLibrary._IsListParameter(arg: TArgDef): Boolean;
begin
     if not string.IsNullOrEmpty(arg.NumberAttr)  then
       Exit(True)
     else if not string.IsNullOrEmpty(arg.TypeListAttr) then
       Exit(True)
     else
       Result := False;
end;

class function OpDefLibrary._IsListValue(v: TValue): Boolean;
begin
    Result := v.IsArray;
end;

class procedure OpDefLibrary.SetAttrs(op_type_name : string;input_arg: TArgDef; op_def: TOpDef; attrs, inferred_from: TDictionary<string, TValue>; types,
                                      base_types, input_types: TList<TF_DataType>; values: TValue);
begin
    var input_name := input_arg.Name;

    if  not string.IsNullOrEmpty(input_arg.NumberAttr) then
    begin
        if attrs.ContainsKey(input_arg.NumberAttr) then
        begin

        end else
        begin
            if(values.IsArray) and (values.GetArrayElement(0).TypeInfo = TypeInfo(TFTensor)) then
            begin
                var num_attr : TAttrDef;
                for var i := 0 to op_def.Attrs.Count -1 do
                begin
                    if op_def.Attrs.Items[i]^.Name = input_arg.NumberAttr then
                    begin
                        num_attr := op_def.Attrs.Items[i]^;
                        Break;
                    end;
                end;
                if (num_attr.HasMinimum) and (values.GetArrayLength < num_attr.Minimum) then
                    raise Exception.Create(Format('"%s" to "%s" Op with length %d shorter than minimum length %d',[input_name,op_type_name,values.GetArrayLength,num_attr.Minimum]));

                attrs.AddOrSetValue(input_arg.NumberAttr,TObject(values.GetArrayLength));
                inferred_from.AddOrSetValue(input_arg.NumberAttr, TObject(input_name));
            end;
        end;
        // All tensors must have the same base type.
        if input_arg.&Type <> TDataType.DT_INVALID then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr, TObject(base_types[0]));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TObject(input_name));
            //var type_attr = op_def.Attr.First(x => x.Name == input_arg.TypeAttr);
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeAttr) then
    begin
        var attr_value := base_types[0];
        if attrs.ContainsKey(input_arg.TypeAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr,TObject(attr_value));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TObject(input_name));
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
    begin
        var attr_value := base_types;
        if attrs.ContainsKey(input_arg.TypeListAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeListAttr, attr_value);
            inferred_from.AddOrSetValue(input_arg.TypeListAttr,TObject(input_name));
        end;
    end;
    if input_arg.IsRef then
        input_types.AddRange(types)
    else
        input_types.AddRange(base_types);
end;

class function OpDefLibrary._apply_op_helper(op_type_name: string; name: string; args: TArray<TParameter>): TFOperation;
begin
    if args = nil then
       Result := _apply_op_helperDict(op_type_name, name)
    else
      Result := _apply_op_helperDict(op_type_name, name, TUtils.ConvertToDict(args));
end;

class function OpDefLibrary._apply_op_helperDict(op_type_name: string; name: string; keywords: TDictionary<string, TValue>): TFOperation;
var
  aObj        : TArray<TValue>;
  g           : TFGraph;
  attrs       : TDictionary<string, TValue>;
  attr_protos : System.Generics.Collections.TDictionary<string, TAttrValue>;
  inputs      : TList<TFTensor>;
  input_types : TList<TF_DataType>;
  values      : TValue;

  op_def      : TOpDef;
begin
    if keywords = nil then  aObj := []
    else                    aObj := keywords.Values.ToArray;

    g := TOps._get_graph_from_inputs(aObj);
    op_def := g.GetOpDef(op_type_name);

    // Default name if not specified.
    if String.IsNullOrEmpty(name) then
        name := op_type_name;

    (*// Check for deprecation
    if (op_def.Deprecation != null && op_def.Deprecation.Version > 0)
    {
    }*)

    var default_type_attr_map := TDictionary<string, TValue>.Create;
    for var attr_def in op_def.Attrs do
    begin
        if attr_def.&Type <> 'type' then continue;
        var key := attr_def.Name;
        if attr_def.DefaultValue.Value.tag =  attr_def.DefaultValue.ftType then
        begin
            default_type_attr_map.AddOrSetValue(key, attr_def.DefaultValue.Value.value);
        end;
    end;

    attrs       := TDictionary<string, TValue>.Create;
    inputs      := TList<TFTensor>.Create;
    input_types := TList<TF_DataType>.Create;
    values      := nil;

    g.as_default;

    var scope := TOps.name_scope(name);
    scope._Enter_;

    var inferred_from := TDictionary<string, TValue>.Create;
    var base_types    := TList<TF_DataType>.Create;
    var types         := TList<TF_DataType>.CReate;
    var _scope_name   := scope.ToString;

    // Perform input type inference
    for var i := 0 to op_def.InputArgs.Count - 1 do
    begin
        var input_arg : TArgDef := op_def.InputArgs[i]^;
        var input_name:= input_arg.Name;

        if keywords.ContainsKey(input_name) then
            values := keywords[input_name]
        else if keywords.ContainsKey(input_name + '_') then
        begin
            input_name := input_name + '_';
            values     := keywords[input_name];
        end
        else if keywords.ContainsKey('input_'+ IntTostr(i)) then
        begin
            values := keywords['input_'+ IntTostr(i)];
        end
        else
            raise Exception.Create('No argument for input ' + input_name);
        // Goals:
        // * Convert values to Tensors if it contains constants.
        // * Verify that values is a list if that matches the input_arg's
        // type.
        // * If the input_arg's type is determined by attrs, either set
        // those attrs and validate those attr values are legal (if
        // they have not yet been set) or validate the input matches
        // the type indicated by the attrs (if they have already been
        // inferred via an earlier input).
        // * If the input_arg has an explicit type, make sure the input
        // conforms.

        var dtype        : TDataType := TDataType.DT_INVALID;
        var default_dtype: TDataType := TDataType.DT_INVALID;

        if _IsListParameter(input_arg) then
        begin
            if not _IsListValue(values) then
                raise Exception.Create('Expected list for {input_name} argument to {op_type_name} Op, not {values}.');

            if input_arg.&Type <> TDataType.DT_INVALID then
                dtype := TDataType(input_arg.&Type)
            else if not String.IsNullOrEmpty(input_arg.NumberAttr) then
            begin
                if attrs.ContainsKey(input_arg.TypeAttr) then
                    dtype := TDataType( attrs[input_arg.TypeAttr].AsInteger )
                else begin
                   var aEle := values.GetArrayElement(0);
                   if aEle.IsType<TFTensor> then
                       dtype := TDtypes.as_datatype_enum(values.GetArrayElement(0).asType<TFTensor>.Dtype)
                   else if aEle.IsObject then
                   begin
                       for var t := 0 to values.GetArrayLength - 1 do
                       begin
                          var item := values.GetArrayElement(t);
                          if item.IsType<TFTensor> then
                          begin
                              dtype := TDtypes.as_datatype_enum(item.AsType<TFTensor>.Dtype);
                          end;
                       end;
                   end else
                       raise Exception.Create('can''t infer the dtype for {values.GetType()}');
                end;
                if (dtype = TDataType.DT_INVALID) and (default_type_attr_map.ContainsKey(input_arg.TypeAttr)) then
                    default_dtype := TDataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);
            end;

            if ( not input_arg.IsRef) and (dtype <> TDataType.DT_INVALID) then
                dtype := Tdtypes.as_base_dtype(dtype);

            var RetVal := TOps.internal_convert_n_to_tensor(values.AsType< TArray<TValue> >,
                                                            Tdtypes.as_tf_dtype(dtype),
                                                            input_arg.Name,
                                                            Tdtypes.as_tf_dtype(default_dtype),
                                                            input_arg.IsRef);
            values := TValue.From< TArray<TFTensor> >(RetVal);
        end else
        begin
            if input_arg.&Type <> TDataType.DT_INVALID then
                dtype := TDataType(input_arg.&Type)
            else if attrs.ContainsKey(input_arg.TypeAttr) then
                dtype := TDataType(attrs[input_arg.TypeAttr].AsInteger)
            else if (TUtils.isinstance(values, TypeInfo(string))) and (dtype = TDataType.DT_INVALID) then
                dtype := TDataType.DT_STRING
            else if default_type_attr_map.ContainsKey(input_arg.TypeAttr) then
                default_dtype := TDataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);

            var value := TOps.convert_to_tensor(values.AsType<TObject>,
                                                Tdtypes.as_tf_dtype(dtype),
                                                input_arg.Name,
                                                input_arg.IsRef,
                                                Tdtypes.as_tf_dtype(default_dtype));

            values := TValue.From< TArray<TFTensor> >([ value ] );
        end;

        if (values.IsArray) and ( values.GetArrayElement(0).IsType<TFTensor> ) then
        begin
            var values2 : TArray<TFTensor> := values.AsType< TArray<TFTensor> >;
            inputs.AddRange(values2);
            for var j := 0 to Length(values2) -1 do
            begin
                types.Add(values2[j].TensorDataType);
                base_types.Add( Tdtypes.as_base_dtype(values2[j].TensorDataType) ) ;
            end;
        end
        else
           raise Exception.Create('NotImplementedException("_IsListParameter")');

        SetAttrs(op_type_name,
                 input_arg,
                 op_def,
                 attrs,
                 inferred_from,
                 types,
                 base_types,
                 input_types,
                 values);
    end;

    // Process remaining attrs
    for var attr in op_def.Attrs do
    begin
        if keywords.ContainsKey(attr.Name) then
        begin
            attrs.AddOrSetValue(attr.Name, keywords[attr.Name] );
        end
    end;
    // Convert attr values to AttrValue protos.
    attr_protos := System.Generics.Collections.TDictionary<string, TAttrValue>.Create;
    for var  attr_def in op_def.Attrs do
    begin
        var key := attr_def.Name;
        if attrs.ContainsKey(key) then
        begin
            attr_protos.AddOrSetValue(key, SetAttrValue(op_def, attr_def^, attrs[key] ) );
        end else
        begin
            if attr_def.DefaultValue.Value.value.AsObject = nil then
            begin
                raise Exception.Create('Missing required positional argument ' + key);
            end;
        end;
    end;
    attrs.Clear();

    // Determine output types (possibly using attrs)
    var output_types := TList<TF_DataType>.Create;

    for var arg in op_def.OutputArgs do
    begin
        types := TList<TF_DataType>.Create;
        if not string.IsNullOrEmpty(arg.NumberAttr) then
        begin
        end
        else if not string.IsNullOrEmpty(arg.TypeAttr) then
        begin
            types := TList<TF_DataType>.Create;
            types.Add( TF_DataType(attr_protos[arg.TypeAttr].Value.value.AsInteger) );
        end;
        if arg.IsRef then
        begin
            var aTemp : TArray<TF_DataType> := [];
            for var i := 0 to types.Count - 1 do
            begin
                aTemp := aTemp + [ Tdtypes.as_ref(types[i]) ];
            end;
            types.Clear;
            types.Free;
            types := TList<TF_DataType>.Create(aTemp);
        end;
        output_types.AddRange(types);
    end;

    // We add an explicit colocation constraint between
    // the newly created op and any of its reference-typed inputs.
   (* var must_colocate_inputs = zip(op_def.InputArg, inputs)
        .Where(x => x.Item1.IsRef)
        .Select(x => x.Item2)
        .ToArray();
    _MaybeColocateWith(must_colocate_inputs);
    *)

    // Add Op to graph
    var ret_op := g.create_op(op_type_name,
                              inputs.ToArray,
                              output_types.ToArray,
                              input_types.ToArray,
                              name{_scope_name},
                              attr_protos,
                              @op_def);

    scope._exit_;

    g.gExit;

    Result := ret_op;
end;

class function OpDefLibrary._MakeShape(shape: TFShape; attr_def:TAttrDef): TTensorShapeProto;
begin
    Result := TUtils.as_shape_proto(shape);
end;

class function OpDefLibrary._MakeType(v: TF_DataType; attr_def:TAttrDef): TDataType;
begin
    Result :=  Tdtypes.as_datatype_enum( Tdtypes.as_base_dtype(v) );
end;

class function OpDefLibrary.SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
var
   v          : TpbOneof;
   attr_value : TAttrValue;

begin
    attr_value.Init;

    if attr_def.&Type.StartsWith('list(') then
    begin
        if attr_def.HasMinimum then
        begin
            v.tag := TAttrValue.ftList;
            var v1 : TListValue; v1.Init;
            v.value := TValue.From<TListValue>(v1);

            attr_value.Value := v;
        end;
    end;

    if attr_def.&Type = 'string' then
    begin
         v.tag   := TAttrValue.ftS;
         var b   := TEncoding.UTF8.GetBytes( value.AsString );
         v.value := TValue.From< TBytes >(b);

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'type' then
    begin
        v.tag   := TAttrValue.ftType;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(type)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TList<TF_DataType> >;
        for var i := 0 to l.Count - 1 do
        begin
            var d : TDataType := _MakeType(l[i],attr_def);
            v1.Types.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(int)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TArray<Integer> >;
        for var i := 0 to Length(l) - 1 do
        begin
            var d : Int64 := l[i];
            v1.&Is.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        v.tag   := TAttrValue.ftList;
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'bool' then
    begin
        v.tag   := TAttrValue.ftB;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'float' then
    begin
        v.tag   := TAttrValue.ftF;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'int' then
    begin
        v.tag   := TAttrValue.ftI;
        v.value := value;

        attr_value.Value := v;
        if (attr_def.HasMinimum)and ( v.value.AsInt64 < attr_def.Minimum) then
           raise Exception.Create(Format('Attr %s of %s Op passed %d less than minimum $d.',[attr_def.Name,op_def.Name,v.value.AsInt64,attr_def.Minimum]));
    end
    else if attr_def.&Type = 'shape' then
    begin
         if (value.IsEmpty) and ( not attr_def.DefaultValue.Value.value.IsEmpty) then
             attr_value.Value := attr_def.DefaultValue.Value;

         if (value.IsEmpty=False)  then
         begin
             if value.IsType<TFShape> then
             begin
                 var v1 := TUtils.as_shape_proto(value.AsType<TFShape>) ;
                 v.tag  := TAttrValue.ftShape;
                 v.value:= TValue.From<TTensorShapeProto>(v1) ;
                 attr_value.Value := v;
             end
             else if value.IsType< TArray<Int64> > then
             begin
                 var v1 := TUtils.as_shape<Int64>(value.AsType<TArray<Int64>>) ;
                 v.tag  := TAttrValue.ftShape;
                 v.value:= TValue.From<TTensorShapeProto>(v1) ;
                 attr_value.Value := v;
             end
             else if value.IsType< TArray<Integer> > then
             begin
                 var v1 := TUtils.as_shape<Integer>(value.AsType< TArray<Integer>>) ;
                 v.tag  := TAttrValue.ftShape;
                 v.value:= TValue.From<TTensorShapeProto>(v1) ;
                 attr_value.Value := v;
             end;

         end;
    end
    else if attr_def.&Type = 'list(shape)' then
    begin
         raise Exception.Create('TODO List');
         //attr_value.List.Shape.AddRange((value as Shape[]).Select(x => _MakeShape(x, attr_def)));
          var v1 : TListValue;
          v1 := attr_value.Value.value.AsType<TListValue>;

          var l := value.AsType< TArray<TFShape> >;
          for var i := 0 to Length(l) - 1 do
          begin
              var d : TTensorShapeProto := _MakeShape(l[i],attr_def);
              v1.Shapes.Add(@d)
          end;
          v.value := TValue.From<TListValue>(v1);
          v.tag   := TAttrValue.ftList;
          attr_value.Value := v;
    end else
    begin
        raise Exception.Create('SetAttrValue: can''t not convert attr_def.Type {attr_def.Type} to protos.');
    end;


    Result := attr_value;
end;

{ TTensors }

procedure TTensors.Add(tensor: TFTensor);
begin
    Fitems.Add(tensor)
end;

procedure TTensors.AddRange(tensors: TArray<TFTensor>);
begin
    Fitems.AddRange(tensors)
end;

constructor TTensors.Create(tensors: TArray<TFTensor>);
begin
    Fitems.AddRange(tensors);
end;

function TTensors.Getdtype: TF_DataType;
begin
   Result := Fitems.First.dtype
end;

function TTensors.GetGraph: TFGraph;
begin
    Result := Fitems.First.graph;
end;

function TTensors.GetItem(index: Integer): TFTensor;
begin
    Result := Fitems[index]
end;

function TTensors.GetLen: Integer;
begin
    Result := Fitems.Count;
end;

function TTensors.GetRank: Integer;
begin
    Result := Fitems.First.rank;
end;

function TTensors.Getshape: TFShape;
begin
    Result := Fitems.First.Shape;
end;

procedure TTensors.Insert(index: Integer; tensor: TFTensor);
begin
    Fitems.Insert(index,tensor)
end;

procedure TTensors.SetItem(index: Integer; const Value: TFTensor);
begin
    Fitems[index] :=  Value;
end;

{ TConstant_op }

class function TConstant_op.constant(value: PValue; dtype: TF_DataType; shape: PTFShape; verify_shape, allow_broadcast: Boolean;
  name: AnsiString): TFTensor;
begin
    if value = nil then
        Exit(nil);

    if tf.executing_eagerly then
        Result := convert_to_eager_tensor(value^, dtype, shape, name, verify_shape, allow_broadcast)
    else
        Result := convert_to_graph_tensor(value^, dtype, shape, name, verify_shape, allow_broadcast);
end;

class function TConstant_op.convert_to_graph_tensor(value: TValue; dtype: TF_DataType; shape: TFShape; name: AnsiString; verify_shape,
  allow_broadcast: Boolean): TFTensor;
var
  v : TpbOneof;
begin
    var g : TFGraph := TOps.get_default_graph;

    var tp := TUtils.make_tensor_proto(value, dtype,@shape, verify_shape, allow_broadcast);

    var tensor_value : TAttrValue;
    tensor_value.Init;
    v.tag   := TAttrValue.ftTensor;
    v.value := TValue.From<TTensorProto>(tp);
    tensor_value.Value := v;

    var dtype_value : TAttrValue;
    dtype_value.Init;
    v.tag   := TAttrValue.ftType;
    v.value := TValue.From<Integer>( Ord(dtype)  );
    dtype_value.Value := v;

    var attrs := System.Generics.Collections.TDictionary<string, TAttrValue>.Create;

    attrs.Add('value',tensor_value);
    attrs.Add('dtype',dtype_value);

    var oper := g.create_op(
        'Const',
        [],
        [TF_DataType(Ord(dtype_value.Value.value.AsType<Integer>))],
        [],
        name,
        attrs);

    Result := oper.outputs[0];
end;

class function TConstant_op._eager_reshape(tensor: TFTensor; shape: TArray<Integer>; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(tensor.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(shape), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ tensor, dims_t ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'Tshape', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Reshape', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function TConstant_op._eager_fill(dims: TArray<Integer>; value: TFTensor; ctx: TContext): TFTensor;
begin
    var attr_t := Tdtypes.as_datatype_enum(value.dtype);
    var dims_t := convert_to_eager_tensor(TValue.From< TArray<Integer> >(dims), ctx, Tdtypes.cint32);
    var inputs_flat : TArray<TFTensor> := [ dims_t, value ];
    var attrs : TArray<TValue> := [ 'T', TValue.From<Integer>(ord(attr_t)), 'index_type', TValue.From<Integer>(Ord(TF_DataType.TF_INT32)) ];
    var res   := tf.Runner.Execute(ctx, 'Fill', 1, inputs_flat, attrs);
    Result := res[0];
end;

class function TConstant_op.convert_to_eager_tensor(value: TValue; dtype: TF_DataType; shape: PTFShape; name: AnsiString; verify_shape, allow_broadcast: Boolean): TFTensor;
begin
    var t := convert_to_eager_tensor(value, tf.Context, dtype);

    if ( PTFShape(shape) = nil) or (shape.IsNull) then
        Exit(t);

    if t.shape.Equals( TValue.From<TFShape>(shape)) then
        Exit(t);

    if verify_shape then
        raise Exception.Create( Format('Expected Tensor''s shape: %s, got %s.',[shape.ToString,t.Shape.ToString]));

    var num_t := t.shape.size;
    if num_t = shape.size then
        Exit(_eager_reshape(t, shape^, tf.Context) );
    if num_t = 1 then
    begin
        if t.dtype = Tdtypes.cbool then
            raise Exception.Create('Not Implemented')
        else
            Exit( _eager_fill(shape^, t, tf.Context) );
    end;

    raise Exception.Create('Not Implemented')
end;

class function TConstant_op.convert_to_eager_tensor(value: TValue; ctx: TContext; dtype: TF_DataType): TFTensor;
begin
    ctx.ensure_initialized;
    var tipo : PTypeInfo;
    tipo:= value.TypeInfo;
    // convert data type
    if (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) and
       (string.LowerCase(tipo.Name) <> 'tndarray') and
       (value.IsArray = False) and
       (dtype <> TUtils.GetDataType(value))  then
    begin
        case dtype of
            TF_DataType.TF_DOUBLE: value := value.AsType<Double>;
            TF_DataType.TF_FLOAT:  value := value.AsType<Single>;
            TF_DataType.TF_INT64:  value := value.AsType<Int64>;
            TF_DataType.TF_INT32:  value := value.AsType<Int32>;
        end;
    end
    else if (dtype <> TF_DataType.TF_DATATYPE_UNKNOWN) and (value.IsType<TNDArray>) and  ( value.AsType<TNDArray>.Dtype = dtype ) then
    begin
        var nd := value.AsType<TNDArray>;
        value := T_Math_Ops.cast(nd, dtype);
    end;
    // non ascii char
    if (dtype = TF_DataType.TF_STRING) and (value.IsArray) and (value.GetArrayElement(0).IsType<Byte> ) then
    begin
        Result := TEagerTensor.Create(Value.AsType< TArray<Byte> >, TFShape.Scalar, TF_DataType.TF_STRING);
        Exit;
    end;
    if value.IsType<TEagerTensor>      then  Result := value.AsType<TEagerTensor>
    else if value.IsType<TNDArray>     then  Result := value.AsType<TNDArray>
    else if value.IsType<TFShape>      then
    begin
         var vval := Value.AsType<TFShape>;
         Result := TEagerTensor.Create(vval.dims, TFShape.Create([vval.ndim]),TUtils.GetDataType(Value) );
    end
    else if value.IsType<TAxis>      then
    begin
         var vval := Value.AsType<TAxis>;
         var shape : TFShape;
         if vval.IsScalar then shape := TFShape.Scalar
         else                  shape := TFShape.Create([vval.size]);
         Result := TEagerTensor.Create(vval.axis, shape,TUtils.GetDataType(Value) );
    end
    else if (value.IsType<string>) or (value.IsType<AnsiString>) then
    begin
        var vval := Value.AsType<string>;
        Result := TEagerTensor.Create([vval], TFShape.scalar );
    end
    else if (value.IsType<TArray<String>>) or (value.IsType<TArray<AnsiString>>) then
    begin
        var vval : TArray<TFString> := Value.AsType<TArray<TFString>>;
        Result := TEagerTensor.Create(vval, TFShape.Create( [ Length(vval) ] ) );
    end
    else if value.IsType<Boolean> then
    begin
        var vval := Value.AsType<Boolean>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_BOOL);
    end
    else if value.IsType<Boolean> then
    begin
        var vval := Value.AsType<Byte>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT8);
    end
    else if value.IsType<Integer> then
    begin
        var vval := Value.AsType<Integer>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT32);
    end
    else if value.IsType<Int64> then
    begin
        var vval := Value.AsType<Int64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_INT64);
    end
    else if value.IsType<UInt64> then
    begin
        var vval := Value.AsType<UInt64>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_UINT64);
    end
    else if value.IsType<Single> then
    begin
        var vval : Single := Value.AsType<Single>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_FLOAT);
    end
    else if value.IsType<Single> then
    begin
        var vval : Double := Value.AsType<Double>;
        Result := TEagerTensor.Create([vval], TFShape.scalar, TF_DataType.TF_DOUBLE);
    end else
    begin
       raise Exception.Create('NotImplemented convert_to_eager_tensor Type: '+ Value.TypeInfo.Name);
    end;
end;

initialization
begin
    tf := TTensorflow.Create;
end;

end.


