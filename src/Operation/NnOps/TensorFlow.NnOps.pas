unit TensorFlow.NnOps;
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
         system.TypInfo,

         Spring,
         Spring.Collections,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         Tensorflow.Utils,
         TensorFlow.Context,
         Tensorflow.NameScope,
         TensorFlow.Variable,
         TensorFlow.Interfaces,
         TensorFlow.Initializer,

         Keras.Engine,
         Keras.ArgsDefinition,

         ProtoGen.variable;

type
  LSTMStateTuple = class;

  Conv2dParams = class
    private

    public
        Name: string;
        /// <summary>
        /// An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        /// Specify the data format of the input and output data. With the
        /// default format "NHWC", the data is stored in the order of:
        /// [batch, height, width, channels].
        /// </summary>
        DataFormat : string;
        /// <summary>
        /// Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
        /// A 4-D tensor. The dimension order is interpreted according to the value
        /// </summary>
        Input : TFTensor;
        /// <summary>
        /// An integer vector representing the shape of `input`
        /// </summary>
        InputSizes : TFTensor;
        /// <summary>
        /// A 4-D tensor of shape
        /// </summary>
        Filter : TFTensor;
        /// <summary>
        /// An integer vector representing the tensor shape of `filter`
        /// </summary>
        FilterSizes : TFTensor;
        /// <summary>
        /// A `Tensor`. Must have the same type as `filter`.
        /// 4-D with shape `[batch, out_height, out_width, out_channels]`.
        /// </summary>
        OutBackProp : TFTensor;
        /// <summary>
        /// The stride of the sliding window for each
        /// dimension of `input`. The dimension order is determined by the value of
        /// `data_format`, see below for details.
        /// </summary>
        Strides : TArray<Integer>;
        /// <summary>
        /// A `string` from: `"SAME", "VALID", "EXPLICIT"`.
        /// </summary>
        Padding : string;
        ExplicitPaddings : TArray<Integer>;
        UseCudnnOnGpu: Boolean;
        Dilations : TArray<Integer>;

        constructor Create;
  end;

  ConvolutionInternal = class
    private
       Fargs       : ConvolutionalArgs;
       Fname       : string;

       function _get_sequence(value: TArray<Integer>; n, channel_index: Integer): IList<integer>;
    function Get_data_format: string;
    function Get_Padding: string;
    public

       constructor Create(args: ConvolutionalArgs);
       function Apply(input: TFTensor; filters: TFTensor): TFTensor;

       property data_format: string read Get_data_format;
       property padding    : string read Get_Padding;
  end;

  FusedBatchNormParams = class
    private

    public
       Name          : string;
       YBackprop     : TFTensor;
       X             : TFTensor;
       Scale         : TFTensor;
       ReserveSpace1 : TFTensor;
       ReserveSpace2 : TFTensor;
       ReserveSpace3 : TFTensor;
       Epsilon       : Single;
       DataFormat    : string;
       IsTraining    : Boolean;

       constructor Create;
  end;

  /// <summary>
  /// Abstract object representing an RNN cell.
  ///
  /// Every `RNNCell` must have the properties below and implement `call` with
  /// the signature `(output, next_state) = call(input, state)`.  The optional
  /// third input argument, `scope`, is allowed for backwards compatibility
  /// purposes; but should be left off for new subclasses.
  ///
  /// This definition of cell differs from the definition used in the literature.
  /// In the literature, 'cell' refers to an object with a single scalar output.
  /// This definition refers to a horizontal array of such units.
  ///
  /// An RNN cell, in the most abstract setting, is anything that has
  /// a state and performs some operation that takes a matrix of inputs.
  /// This operation results in an output matrix with `self.output_size` columns.
  /// If `self.state_size` is an integer, this operation also results in a new
  /// state matrix with `self.state_size` columns.  If `self.state_size` is a
  /// (possibly nested tuple of) Shape object(s), then it should return a
  /// matching structure of Tensors having shape `[batch_size].concatenate(s)`
  /// for each `s` in `self.batch_size`.
  /// </summary>
  RnnCell = class(TInterfacedObject, ILayer, RNNArgs.IRnnArgCell)
     private
        /// <summary>
        /// Attribute that indicates whether the cell is a TF RNN cell, due the slight
        /// difference between TF and Keras RNN cell.
        /// </summary>
        Fis_tf_rnn_cell: Boolean;
        Fbuilt         : Boolean;

        /// <summary>
        /// Return zero-filled state tensor(s).
        /// </summary>
        /// <param name="batch_size"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        function zero_state(batch_size: TFTensor; dtype: TF_DataType) : TFTensor;
        function _zero_state_tensors(_state_size: TValue; batch_size: TFTensor; dtype: TF_DataType) : TFTensor ;
        {$REGION 'Property Accessors'}
        function GetState_size : TValue;  virtual;
        function GetOutputSize: Integer;  virtual;
        {$ENDREGION}
     public


        constructor Create(trainable: Boolean = true; name: string = ''; dtype: TF_DataType = DtInvalid; _reuse: pBoolean = nil);
        function get_initial_state(inputs: TFTensor = nil; batch_size : TFTensor= nil; dtype: TF_DataType = DtInvalid): TValue;
        {$REGION 'Func/Proc Accessors'}
        function GetName      :string;
        function GetTrainable : Boolean;
        function GetBuilt     : Boolean;
        function GetLayers    : TList<ILayer>;
        function GetInNodes   : TList<INode>;
        function GetOutNodes  : TList<INode>;
        function GetTrainVars : TList<IVariableV1>;
        function GetTrainW    : TList<IVariableV1>;
        function GetNotTrainW : TList<IVariableV1>;
        function GetOutShape  : TFShape;
        function GetBatchShape: TFShape;
        function GetDtype     : TF_DataType;
        function Apply(inputs: TFTensors; state: TFTensor = nil; is_training: Boolean = false): TFTensors;
        procedure build(input_shape: TFShape);
        function count_params: Integer;
        function get_config: LayerArgs;
        {$ENDREGION}

        property Built       : Boolean read Fbuilt;
        property state_size  : TValue  read GetState_size;
        property output_size : Integer read GetOutputSize;
  end;

  LayerRnnCell = class(RnnCell)
    private

    protected
        FinputSpec : TInputSpec;
        Fbuilt     : Boolean;
        Fgraph     : TFGraph;

        Fscope         : VariableScope;
        Fcurrent_scope : VariableScope;

        Freuse                  : pBoolean;
        Fuse_resource_variables : Boolean;
        Fkeras_style            : Boolean;

        function _name_scope: string;
        procedure _set_scope(scope: VariableScope = nil);
        procedure build(inputs_shape: TFShape); virtual;
        function apply(inputs: TFTensor; training : TFTensor = nil): Tuple<TFTensor,TFTensor>; virtual;
        procedure _add_elements_to_collection(elements: TArray<TFOperation>; collection_list: TArray<String>) ; virtual;
        /// <summary>
        /// Adds a new variable to the layer, or gets an existing one; returns it.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="initializer"></param>
        /// <param name="trainable"></param>
        /// <param name="synchronization"></param>
        /// <param name="aggregation"></param>
        /// <returns></returns>
        function add_weight(name            : string;
                            shape           : TArray<Integer>;
                            dtype           : TF_DataType = DtInvalid;
                            initializer     : IInitializer = nil;
                            trainable       : Boolean= true;
                            synchronization : TVariableSynchronization = VARIABLE_SYNCHRONIZATION_AUTO;
                            aggregation     : TVariableAggregation     = VARIABLE_AGGREGATION_NONE): IVariableV1; virtual;
    public
        constructor Create(trainable : Boolean  = true; name: string = '';  dtype : TF_DataType = DtInvalid;  _reuse: PBoolean = nil);
        function  __call__(inputs: TFTensors; state : TFTensor = nil; training : TFTensor= nil;  scope : VariableScope= nil): TFTensors;
  end;

  BasicRnnCell = class(LayerRnnCell)
    private
        Fnum_units             : Integer;
        Factivation            : TFunc<TFTensor, string, TFTensor>;
        FWEIGHTS_VARIABLE_NAME : string ;
        FBIAS_VARIABLE_NAME    : string ;

        function GetState_size : TValue;  override;
        function GetOutputSize : Integer;  override;
    protected
        procedure build(inputs_shape: TFShape) ; override;
        function Call(inputs: TFTensors; state: TFTensor = nil; is_training : boolean= false): TFTensors;
    public
        _kernel : IVariableV1;
        _bias   : IVariableV1;

        constructor Create(num_units: Integer; activation :  TFunc<TFTensor, string, TFTensor> = nil; reuse: PBoolean = nil; name: string = ''; dtype: TF_DataType = DtInvalid);
  end;

  rnn_cell_impl = record
    private

    public
        function BasicRNNCell(num_units: Integer): BasicRnnCell;
        class function _concat(prefix: TFTensor; suffix: Integer; &static : Boolean = false):TFTensor; overload; static;
        class function _concat(prefix: TArray<Integer>; suffix: Integer; &static: Boolean = false): TFShape; overload; static;
  end;

  StackedRNNCellsArgs = class(LayerArgs)
    public
        Cells : IList<RnnCell> ;
        Kwargs: TDictionary<string, TValue>;

        Constructor Create;
  end;

  /// <summary>
  /// Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  ///
  /// Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  /// and `h` is the output.
  ///
  /// Only used when `state_is_tuple=True`.
  /// </summary>
  LSTMStateTuple = class(TInterfacedObject, ICanBeFlattened)
    c : TValue;
    h : TValue;

    constructor Create(_c,_h: TValue); overload;
    constructor Create(_c,_h: TFTensor); overload;
    function Flatten: TArray<TValue>;
  end;

  /// <summary>
  /// Performs the max pooling on the input.
  /// </summary>
  MaxPoolFunction = class(TInterfacedObject, IPoolFunction)

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
  end;

  /// <summary>
  /// Performs the max pooling on the input.
  /// </summary>
  AveragePoolFunction = class(TInterfacedObject, IPoolFunction)

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
  end;

implementation
      uses TensorFlow.DApiBase,
           TensorFlow.Ops,
           TensorFlow.gen_nn_ops,
           Tensorflow.array_ops,
           Tensorflow.math_ops,
           Tensorflow.nn_ops,
           TensorFlow.Constant_op,
           TensorFlow,

           NumPy.NDArray;

{ ConvolutionInternal }

constructor ConvolutionInternal.Create(args: ConvolutionalArgs);
begin
    Fargs := args;
    Fname := args.Name;
end;

function ConvolutionInternal.Get_data_format: string;
begin
   Result := Fargs.DataFormat;
end;

function ConvolutionInternal.Get_Padding: string;
begin
     Result := Fargs.Padding;
end;

function ConvolutionInternal.Apply(input, filters: TFTensor): TFTensor;
begin
    var filters_rank     := filters.shape.ndim;
    var inputs_rank      := input.shape.ndim;
    var num_spatial_dims := Fargs.NumSpatialDims;
    if Fargs.Rank = 1 then
    begin
        // Special case: Conv1D
        num_spatial_dims := 1;
    end
    else if num_spatial_dims = -1 then
    begin
        num_spatial_dims := filters_rank - 2;
    end;

    // Channel dimension.
    var num_batch_dims := inputs_rank - num_spatial_dims - 1;
    var a : TArray<Integer> := [1,2,3];
    if not TArray.Contains<Integer>(a,num_spatial_dims ) then
      raise Exception.Create('num_spatial_dims (input.shape.ndims - num_batch_dims - 1) must be one of 1, 2 or 3 but saw'+IntToStr(num_spatial_dims) +'. num_batch_dims: '+ IntToStr(num_batch_dims)+'.');

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(Fname, '', nil),
                            function(v1: TNameScope): TFTensor
                              begin
                                  Fname := v1.ToString;
                                  if num_spatial_dims = 2 then
                                  begin
                                      var channel_index := num_batch_dims + num_spatial_dims;
                                      var dilations     := _get_sequence(Fargs.DilationRate, num_spatial_dims, channel_index).ToArray;
                                      var strides       := _get_sequence(Fargs.Strides, num_spatial_dims, channel_index).ToArray;
                                      var Conv2dP := Conv2dParams.Create;
                                      Conv2dP.Input      := input;
                                      Conv2dP.Filter     := filters;
                                      Conv2dP.Strides    := strides;
                                      Conv2dP.Padding    := padding;
                                      Conv2dP.DataFormat := data_format;
                                      Conv2dP.Dilations  := dilations;
                                      Conv2dP.Name       := Fname;
                                      result := gen_nn_ops.conv2d(Conv2dP);
                                  end else
                                  begin
                                      var channel_first  := data_format = 'NCW';
                                      var spatial_start_dim : Integer ;
                                      var channel_index     : Integer;
                                      if  channel_first  then spatial_start_dim := -2
                                      else                    spatial_start_dim := -3;
                                      if  channel_first  then channel_index := 1
                                      else                    channel_index := 2;
                                      var dilations := _get_sequence(Fargs.DilationRate, 1, channel_index);
                                      var strides   := _get_sequence(Fargs.Strides, 1, channel_index);
                                      strides.Insert(0, 1);
                                      dilations.Insert(0, 1);
                                      input   := array_ops.expand_dims(input, spatial_start_dim);
                                      filters := array_ops.expand_dims(filters, 0);
                                      var Conv2dP := Conv2dParams.Create;
                                      Conv2dP.Input      := input;
                                      Conv2dP.Filter     := filters;
                                      Conv2dP.Strides    := strides.ToArray;
                                      Conv2dP.Padding    := padding;
                                      if  channel_first  then Conv2dP.DataFormat := 'NCHW'
                                      else                    Conv2dP.DataFormat := 'NHWC';
                                      Conv2dP.Dilations  := dilations.ToArray;
                                      Conv2dP.Name       := Fname;
                                      result := gen_nn_ops.conv2d(Conv2dP);
                                      result := array_ops.squeeze(result, [ spatial_start_dim ]);
                                  end;
                              end );;
end;

function ConvolutionInternal._get_sequence(value: TArray<Integer>; n: Integer; channel_index: Integer): IList<integer>;
begin
    var seq := TCollections.CreateList<Integer>;

    if channel_index = 1 then
    begin
        seq.Add(1);
        seq.Add(1);
        seq.AddRange(value);
    end else
    begin
        seq.Add(1);
        seq.AddRange(value);
        seq.Add(1);
    end;
    Result := seq;
end;

{ Conv2dParams }

constructor Conv2dParams.Create;
begin
    DataFormat       := 'NHWC';
    ExplicitPaddings := [];
    UseCudnnOnGpu    := True;
    Dilations        := [ 1, 1, 1, 1 ];
end;

{ FusedBatchNormParams }

constructor FusedBatchNormParams.Create;
begin
    Epsilon    := 0.0001;
    DataFormat := 'NHWC';
    IsTraining := true;
end;

{ RnnCell }

constructor RnnCell.Create(trainable: Boolean; name: string; dtype: TF_DataType; _reuse: pBoolean);
begin
    inherited Create;

    Fis_tf_rnn_cell := true;
end;

function RnnCell.Apply(inputs: TFTensors; state: TFTensor; is_training: Boolean): TFTensors;
begin
   raise TFException.Create('Not Implemented - Apply()');
end;

procedure RnnCell.build(input_shape: TFShape);
begin
    raise TFException.Create('Not Implemented - build()');
end;

function RnnCell.count_params: Integer;
begin
    raise TFException.Create('Not Implemented - count_params()');
end;

function RnnCell.get_config: LayerArgs;
begin
   raise TFException.Create('Not Implemented - get_config()');
end;


function RnnCell.GetBatchShape: TFShape;
begin
    raise TFException.Create('Not Implemented - GetBatchShape()');
end;

function RnnCell.GetBuilt: Boolean;
begin
   Result := Fbuilt;
end;

function RnnCell.GetDtype: TF_DataType;
begin
    raise TFException.Create('Not Implemented - GetDtype()');
end;

function RnnCell.GetInNodes: TList<INode>;
begin
   raise TFException.Create('Not Implemented - GetInNodes()');
end;

function RnnCell.GetLayers: TList<ILayer>;
begin
    raise TFException.Create('Not Implemented - GetLayers()');
end;

function RnnCell.GetName: string;
begin
    raise TFException.Create('Not Implemented - GetName()');
end;

function RnnCell.GetNotTrainW: TList<IVariableV1>;
begin
    raise TFException.Create('Not Implemented - GetNotTrainW()');
end;

function RnnCell.GetOutNodes: TList<INode>;
begin
    raise TFException.Create('Not Implemented - GetOutNodes()');
end;

function RnnCell.GetOutputSize: Integer;
begin
    raise TFException.Create('Not Implemented - GetOutputSize()');
end;

function RnnCell.GetOutShape: TFShape;
begin
    raise TFException.Create('Not Implemented - GetOutShape()');
end;

function RnnCell.GetState_size: TValue;
begin
    raise TFException.Create('Not Implemented - GetState_size()');
end;

function RnnCell.GetTrainable: Boolean;
begin
   raise TFException.Create('Not Implemented - GetTrainable()');
end;

function RnnCell.GetTrainVars: TList<IVariableV1>;
begin
    raise TFException.Create('Not Implemented - GetTrainVars()');
end;

function RnnCell.GetTrainW: TList<IVariableV1>;
begin
    raise TFException.Create('Not Implemented - GetTrainW()');
end;

function RnnCell.get_initial_state(inputs, batch_size: TFTensor; dtype: TF_DataType): TValue;
begin
    if inputs <> nil then
      raise Exception.Create('Not Implemented. get_initial_state input is not null');
    Result := zero_state(batch_size, dtype);
end;

function RnnCell.zero_state(batch_size: TFTensor; dtype: TF_DataType): TFTensor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([ batch_size ]);
    Var tName :=  PTypeInfo(TypeInfo(RnnCell))^.Name+'ZeroState';

    var output : TFTensor := nil;
    TUtils.tf_with<TNameScope>( TOps.name_scope( tName, '', @newVal),
                          procedure(v1: TNameScope)
                            begin
                                output := _zero_state_tensors( state_size, batch_size, dtype);
                            end );
    Result := output;
end;

function RnnCell._zero_state_tensors(_state_size: TValue; batch_size: TFTensor; dtype: TF_DataType): TFTensor;
begin
    if state_size.TypeInfo = TypeInfo(Integer)  then
    begin
        var state_size_int := state_size.AsType<Integer>;
        var mapFun : TFunc<Integer, TFTensor> := function(s : Integer): TFTensor
                       begin
                          var c        := rnn_cell_impl._concat(batch_size, s);
                          var size     := array_ops.zeros(c, dtype);
                          var c_static := rnn_cell_impl._concat(batch_size, s, true);
                          size.set_shape(c_static);
                          Result := size;
                       end ;
        var output := nest.map_structure<Integer>(mapFun, state_size_int);
        Result := output;
        exit;
    end;
    raise Exception.Create('Not Implemented _zero_state_tensors');
end;

{ LSTMStateTuple }

constructor LSTMStateTuple.Create(_c, _h: TFTensor);
begin
    c := _c;
    h := _h;
end;

constructor LSTMStateTuple.Create(_c, _h: TValue);
begin
    c := _c;
    h := _h;
end;

function LSTMStateTuple.Flatten: TArray<TValue>;
begin
    Result := [c,h]
end;

{ MaxPoolFunction }

function MaxPoolFunction.Apply(value: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    var vValues : TValue := value;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MaxPool', @vValues),
                  function(v1: TNameScope): TFTensor
                    begin
                        name :=  v1.ToString;
                        value := Tops.convert_to_tensor(value, DtInvalid, 'input');
                        Result := gen_nn_ops.max_pool(value, ksize, strides, padding, data_format, name)
                    end );
end;

{ AveragePoolFunction }

function AveragePoolFunction.Apply(value: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    var vValues : TValue := value;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'AveragePool', @vValues),
                  function(v1: TNameScope): TFTensor
                    begin
                        name :=  v1.ToString;
                        value := Tops.convert_to_tensor(value, DtInvalid, 'input');
                        Result := gen_nn_ops.average_pool(value, ksize, strides, padding, data_format, name)
                    end );
end;

{ StackedRNNCellsArgs }

constructor StackedRNNCellsArgs.Create;
begin
    inherited Create;

    Kwargs := nil;
end;

{ LayerRnnCell }

constructor LayerRnnCell.Create(trainable: Boolean; name: string; dtype: TF_DataType; _reuse: PBoolean);
begin
    inherited Create(True,name, dtype, _reuse);
    // For backwards compatibility, legacy layers do not use `ResourceVariable`
    // by default.
    Fuse_resource_variables := false;
    Freuse                  := _reuse;

    // Avoid an incorrect lint error
    Fbuilt       := false;
    Fkeras_style := false;
end;

procedure LayerRnnCell.build(inputs_shape: TFShape);
begin

end;

function LayerRnnCell.apply(inputs, training: TFTensor): Tuple<TFTensor, TFTensor>;
begin
    var results := __call__(TFTensors.Create(inputs), training);
    Result := Tuple.Create(results[0], results[1]);
end;

function LayerRnnCell.__call__(inputs: TFTensors; state, training: TFTensor; scope: VariableScope): TFTensors;
begin
    _set_scope(scope);
    Fgraph := Tops._get_graph_from_inputs(inputs.First, Fgraph);

    var scope_context_manager : variable_scope;
    if Fbuilt then
    begin
        var lReuse : Boolean  := True;
        scope_context_manager := tf.variable_scope(Fscope,'',nil , @lReuse, false);
    end else
    begin
        scope_context_manager := tf.variable_scope(Fscope, '', nil, Freuse, false);
    end;

    var outputs : TFTensors := nil;
    TUtils.tf_with<variable_scope>(scope_context_manager, procedure(scope2 : variable_scope)
              begin
                  Fcurrent_scope := scope2.scope;
                  // Actually call layer
              end);
    // Update global default collections.
    Result := outputs;
end;

procedure LayerRnnCell._add_elements_to_collection(elements: TArray<TFOperation>; collection_list: TArray<String>);
begin
    for var name in collection_list do
    begin
        var collection := Tops.get_collection_ref<TFOperation>(name);

        for var element in elements do
            if not collection.Contains(element) then
                collection.Add(element);
    end;
end;

function LayerRnnCell.add_weight(name: string; shape: TArray<Integer>; dtype: TF_DataType; initializer: IInitializer; trainable: Boolean; synchronization: TVariableSynchronization;
  aggregation: TVariableAggregation): IVariableV1;
begin
    var default_graph := Tops.get_default_graph;
    var init_graph : TFGraph := nil;
    var existing_variables : TArray<IVariableV1> := [];

    if synchronization = VARIABLE_SYNCHRONIZATION_ON_READ then
        trainable := false;

    if default_graph.building_function then
    begin
        raise Exception.Create('Not Implemented');
    end else
    begin
        init_graph         := default_graph;
        existing_variables := variables.global_variables.ToArray;
    end;

    if dtype = TF_DataType.DtInvalid then
        dtype := TF_DataType.TF_FLOAT;

    _set_scope;
    var reuse := (Fbuilt) or ( (Freuse <> nil) and (Freuse^) );
    Result := tf.Variable(Integer(0));
end;

function LayerRnnCell._name_scope: string;
begin
    Result  := Fcurrent_scope.original_name_scope;
end;

procedure LayerRnnCell._set_scope(scope: VariableScope);
begin
    if Fscope = nil then
    begin
        if (Freuse <> nil) and  (Freuse^) then
        begin
            raise Exception.Create('Not Implemented _set_scope _reuse.HasValue');
            (*with(tf.variable_scope(scope == null ? _base_name : scope),
                captured_scope => _scope = captured_scope);*)
        end else
        begin

        end;
    end;
end;

{ BasicRnnCell }

constructor BasicRnnCell.Create(num_units: Integer; activation: TFunc<TFTensor, string, TFTensor>; reuse: PBoolean; name: string; dtype: TF_DataType);
begin
     FWEIGHTS_VARIABLE_NAME := 'kernel';
     FBIAS_VARIABLE_NAME    := 'bias';

     inherited Create(True,  name, dtype, reuse);

     // Inputs must be 2-dimensional.
     FinputSpec := TInputSpec.Create(dtInvalid, 2);

     Fnum_units := num_units;
     if not Assigned(activation)  then
        Factivation := math_ops.tanh
     else
        Factivation := activation;
end;

procedure BasicRnnCell.build(inputs_shape: TFShape);
begin
    var input_depth := inputs_shape.dims[inputs_shape.ndim - 1];

    _kernel := add_weight( FWEIGHTS_VARIABLE_NAME,  [ (input_depth + Fnum_units), Fnum_units ]);
    _bias   := add_weight( FBIAS_VARIABLE_NAME, [ Fnum_units ], DtInvalid, tf.zeros_initializer);

    Fbuilt := true;
end;

function BasicRnnCell.Call(inputs: TFTensors; state: TFTensor; is_training: boolean): TFTensors;
begin
    // Most basic RNN: output = new_state = act(W * input + U * state + B).
    var tInput : TArray<TFTensor> := [ inputs.First, state ];
    var concat := array_ops.concat(tInput, 1);

    var gate_inputs := math_ops.matmul(concat, _kernel.AsTensor);
    gate_inputs     := nn_ops.bias_add(gate_inputs, _bias);
    var output      := Factivation(gate_inputs, '');

    Result := TFTensors.Create([output, output]);
end;

function BasicRnnCell.GetOutputSize: Integer;
begin
   Result := Fnum_units;
end;

function BasicRnnCell.GetState_size: TValue;
begin
   Result := Fnum_units;
end;

{ rnn_cell_impl }

function rnn_cell_impl.BasicRNNCell(num_units: Integer): BasicRnnCell;
begin
   Result := TensorFlow.NnOps.BasicRnnCell.Create(num_units);
end;

class function rnn_cell_impl._concat(prefix: TFTensor; suffix: Integer; static: Boolean): TFTensor;
begin
    var p := prefix;
    var p_static := TUtils.constant_value(prefix);

    if      p.ndim = 0  then  p := array_ops.expand_dims(p, 0)
    else if p.ndim <> 1 then  raise Exception.Create('prefix tensor must be either a scalar or vector, but saw tensor: '+ p.ToString);

    var s_tensor_shape := TFShape.Create([suffix]);

    var s_static : TArray<Int64> := [];
    if s_tensor_shape.ndim > -1  then s_static := s_tensor_shape.dims ;

    var s : TFTensor := nil;
    if s_tensor_shape.IsFullyDefined  then s := constant_op.constant(s_tensor_shape.dims, Tdtypes.cint32, 'Const') ;

    if &static then
    begin
        if p_static = nil then Exit(nil);
        var nd : NDArray := p_static;
        var iDim : Int64 := nd;

        var shape := TFShape.Create([iDim]).concatenate(s_static);
        raise Exception.Create('Not Implemented RNNCell _concat');
    end else
    begin
        if (p = nil) or (s = nil) then
          raise Exception.Create('Provided a prefix or suffix of None: '+ prefix.ToString + ' and '+ suffix.ToString);
        Result := array_ops.concat([ p, s ], 0);
    end;
end;

class function rnn_cell_impl._concat(prefix: TArray<Integer>; suffix: Integer; static: Boolean): TFShape;
begin
    var p := TFShape.Create(prefix);
    var p_static := prefix;

    var p_tensor : TFTensor := nil;
    if p.IsFullyDefined  then p_tensor := constant_op.constant(p.dims, Tdtypes.cint32, 'Const') ;

    var s_tensor_shape := TFShape.Create([suffix]);

    var s_static : TArray<Int64> := [];
    if s_tensor_shape.ndim > -1  then s_static := s_tensor_shape.dims ;

    var s_tensor : TFTensor := nil;
    if s_tensor_shape.IsFullyDefined  then s_tensor := constant_op.constant(s_tensor_shape.dims, Tdtypes.cint32, 'Const') ;

    if &static then
    begin
        if p_static = nil then Exit(nil);
        var shape := TFShape.Create(p_static).concatenate(s_static);
        Exit(shape);
    end else
    begin
        if (p = nil) or (s_tensor = nil) then
          raise Exception.Create('Provided a prefix or suffix of None: {prefix} and {suffix}');
        // return array_ops.concat(new[] { p_tensor, s_tensor }, 0);
        raise Exception.Create('Not Implemented');
    end
end;

end.
