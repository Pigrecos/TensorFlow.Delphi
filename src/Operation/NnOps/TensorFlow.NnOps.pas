unit TensorFlow.NnOps;
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
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils,
         Spring,
         Spring.Collections,
         Spring.Collections.Lists,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Tensorflow.Utils,
         TensorFlow.Context,
         Tensorflow.NameScope,
         TensorFlow.Ops,
         TensorFlow.Variable,
         TensorFlow.Interfaces,

         Keras.Layer,
         Keras.ArgsDefinition;

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
       Fdata_format: string;
       Fname       : string;
       Fpadding    : string;

       function _get_sequence(value: TArray<Integer>; n, channel_index: Integer): IList<integer>;
    public

       constructor Create(args: ConvolutionalArgs);
       function Apply(input: TFTensor; filters: TFTensor): TFTensor;
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
     public
        output_size          : Integer;

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
        function count_params: Integer;
        function get_config: LayerArgs;
        {$ENDREGION}
        {$REGION 'Property Accessors'}
        function GetState_size : TValue;
        {$ENDREGION}

        property Built       : Boolean read Fbuilt;
        property state_size  : TValue  read GetState_size;
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

implementation
      uses TensorFlow.DApiBase,
           TensorFlow.gen_nn_ops,
           Tensorflow.array_ops;

{ ConvolutionInternal }

constructor ConvolutionInternal.Create(args: ConvolutionalArgs);
begin
    Fargs := args;
    Fname := args.Name;
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
    var a : TArray<Integer>;
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
                                      Conv2dP.Padding    := Fpadding;
                                      Conv2dP.DataFormat := Fdata_format;
                                      Conv2dP.Dilations  := dilations;
                                      Conv2dP.Name       := Fname;
                                      result := gen_nn_ops.conv2d(Conv2dP);
                                  end else
                                  begin
                                      var channel_first  := Fdata_format = 'NCW';
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
                                      Conv2dP.Padding    := Fpadding;
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

function RnnCell.GetOutShape: TFShape;
begin
    raise TFException.Create('Not Implemented - GetOutShape()');
end;

function RnnCell.GetState_size: TValue;
begin
    result := state_size;
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
    (*
    if (state_size is int state_size_int)
    {
        var output = nest.map_structure(s =>
        {
            var c = rnn_cell_impl._concat(batch_size, s);
            var size = array_ops.zeros(c, dtype: dtype);
            var c_static = rnn_cell_impl._concat(batch_size, s, @static: true);
            size.set_shape(c_static);
            return size;
        }, state_size_int);
        return output;
    }
    throw new NotImplementedException("_zero_state_tensors");
    *)
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

end.
