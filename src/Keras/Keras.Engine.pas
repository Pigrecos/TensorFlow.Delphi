unit Keras.Engine;
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
          System.Generics.Collections,

          Spring,
          Spring.Collections.Enumerable,
          Spring.Container.Common,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.Variable,

          Keras.Regularizers,
          Keras.Saving;

const null = $11111111;

type
  AutotuneAlgorithm =( HILL_CLIMB = 0, GRADIENT_DESCENT = 1 );
  ILayer     = interface;

  LayerArgs = class
     public
        /// <summary>
        /// Indicates whether the layer's weights are updated during training
        /// and whether the layer's updates are run during training.
        /// </summary>
        Trainable : Boolean;
        Name      : string;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        DType     : TF_DataType;
        /// <summary>
        /// Whether the `call` method can be used to build a TF graph without issues.
        /// This attribute has no effect if the model is created using the Functional
        /// API. Instead, `model.dynamic` is determined based on the internal layers.
        /// </summary>
        bDynamic  : boolean;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        InputShape : TFShape;
        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        BatchInputShape : TFShape;
        BatchSize       : Integer;
        /// <summary>
        /// Initial weight values.
        /// </summary>
        Weights         : TArray<Single>;
        /// <summary>
        /// Regularizer function applied to the output of the layer(its "activation").
        /// </summary>
        ActivityRegularizer : IRegularizer;
        Autocast            : Boolean;
        IsFromConfig        : Boolean;
        constructor Create;
  end;

  NodeConfig = class
     public
        Name        : string;
        NodeIndex   : integer;
        TensorIndex : Integer;

        function ToString: string; override;
  end;

  LayerConfig = class
     public
        Name        : string;
        ClassName   : string;
        Config      : LayerArgs;
        InboundNodes: TList<NodeConfig>;

        constructor Create;
  end;

  INode = interface
    ['{F604E1FF-BF7E-4A41-967E-FEB9C4A2B52F}']
    function  GetInputTensor: TFTensors;
    function  GetOutptus: TFTensors;
    function  GetILayer: ILayer;
    function  GetlstLayer:  TList<TFTensor>;
    procedure SetlstLayer(const lst: TList<TFTensor>);
    function  GetINodes: TArray<INode>;
    function  GetInLayers: TArray<ILayer>;
    function  GetIsInput: Boolean;

    function iterate_inbound :  Enumerable< Tuple<ILayer, Integer, Integer, TFTensor> >;
    function serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;

    property input_tensors: TFTensors      read GetInputTensor;
    property Outputs      : TFTensors      read GetOutptus;
    property Layer        : ILayer         read GetILayer;
    property KerasInputs  : TList<TFTensor>read GetlstLayer write SetlstLayer;
    property ParentNodes  : TArray<INode>  read GetINodes;
    property InboundLayers: TArray<ILayer> read GetInLayers;
    property is_input     : Boolean        read GetIsInput;
  end;

   NodeArgs = class
    InboundLayers : TArray<ILayer>;
    NodeIndices   : TArray<Integer>;
    TensorIndices : TArray<Integer>;
    InputTensors  : TFTensors;
    Outputs       : TFTensors;
  end;

  IPoolFunction = interface
    ['{17352CCA-5C47-49AD-9265-C9762F7F3FB6}']

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
  end;

  IAccumulator = interface
    ['{353EB77F-F529-43A0-8667-D513E6B5A45C}']
  end;

  /// <summary>
  /// Functional object that defines a shardable computation.
  /// </summary>
  ICombiner = interface
    ['{0C519749-3A1D-4545-813F-A81FEC5C3B69}']

    procedure Compute(values: TFTensor; accumulator : IAccumulator= nil);
    procedure Merge;
    procedure Extract;
    function  Restore: IAccumulator;
    procedure Serialize;
    procedure Deserialize;
  end;

  /// <summary>
  /// A `Node` describes the connectivity between two layers.
  ///
  /// Each time a layer is connected to some new input,
  /// a node is added to `layer._inbound_nodes`.
  /// Each time the output of a layer is used by another layer,
  /// a node is added to `layer._outbound_nodes`.
  /// </summary>
  Node = class(TInterfacedObject,INode)
    private
       Fargs        : NodeArgs;
       Flayer       : ILayer;
       FKerasInputs :  TList<TFTensor>;

       function  GetInputTensor: TFTensors;
       function  GetOutptus: TFTensors;
       function  GetlstLayer:  TList<TFTensor>;
       procedure SetlstLayer(const lst: TList<TFTensor>);
       function  GetILayer: ILayer;
       function  GetIsInput: Boolean;
       function  GetINodes: TArray<INode>;
       function  GetInLayers: TArray<ILayer>;
    protected

    public
     constructor Create(_args:NodeArgs);
     procedure  Connect(_layer: ILayer);
     /// <summary>
     /// Maps Keras Tensors to computed Tensors using `tensor_dict`.
     /// </summary>
     /// <param name="tensor_dict"></param>
     /// <returns></returns>
     function MapArguments(tensor_dict: TDictionary< Int64, TQueue<TFTensor> >): TFTensors;
     function iterate_inbound :  Enumerable< Tuple<ILayer, Integer, Integer, TFTensor> >;
     function serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;


      property input_tensors: TFTensors      read GetInputTensor;
      property Outputs      : TFTensors      read GetOutptus;
      property KerasInputs  : TList<TFTensor>read GetlstLayer write SetlstLayer;
      property Layer        : ILayer         read GetILayer;
      property is_input     : Boolean        read GetIsInput;
      property ParentNodes  : TArray<INode>  read GetINodes;
      property InboundLayers: TArray<ILayer> read GetInLayers;
  end;

  ILayer = interface
    ['{8B096EA0-F97B-47DA-BA05-49674F9282C5}']
    function GetName:string;
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

    function Apply(inputs: TFTensors; state: TFTensor = nil; training: Boolean = false): TFTensors;
    function count_params: Integer;
    function get_config: LayerArgs;

    property Name                    : string              read GetName;
    property Trainable               : Boolean             read GetTrainable;
    property Built                   : Boolean             read GetBuilt;
    property Layers                  : TList<ILayer>       read GetLayers;
    property InboundNodes            : TList<INode>        read GetInNodes;
    property OutboundNodes           : TList<INode>        read GetOutNodes;
    property trainable_variables     : TList<IVariableV1>  read GetTrainVars;
    property trainable_weights       : TList<IVariableV1>  read GetTrainW;
    property non_trainable_weights   : TList<IVariableV1>  read GetNotTrainW;
    property output_shape            : TFShape             read GetOutShape;
    property BatchInputShape         : TFShape             read GetBatchShape;
    property DType                   : TF_DataType         read GetDtype;
  end;

  IModel = interface
    ['{2B165250-B224-4DF2-B0B1-D1577D054A70}']
  end;

  CallContextManager = class(TInterfacedObject, IDisposable)
     private
        Fbuild_graph : Boolean;
     protected
        procedure Dispose;
     public
        constructor Create(build_graph: Boolean);
  end;

  TCallContext = class(CallContextManager)
     constructor Create;
     function Enter(build_graph: Boolean): CallContextManager;
  end;

  /// <summary>
  /// Specifies the ndim, dtype and shape of every input to a layer.
  /// </summary>
  TInputSpec = class
     private
        FnDim      : Nullable<Integer>;
        Fmin_ndim  : Nullable<Integer>;
        Faxes      : TDictionary<Integer,Integer>;
        FAllAxisDim: TArray<Integer>;
        Fshape     : TFShape;
     public
        constructor Create(dtype: TF_DataType = DtInvalid; _ndim : Integer= null; _min_ndim: Integer = null; _axes: TDictionary<Integer,Integer> = nil; shape : PTFShape= nil);

        property nDim       : Nullable<Integer> read FnDim;
        property min_ndim   : Nullable<Integer> read Fmin_ndim;
        property axes       : TDictionary<Integer,Integer> read Faxes;
        property AllAxisDim : TArray<Integer>   read FAllAxisDim;
  end;

  /// <summary>
  /// Tracks the Layer call that created a Tensor, for Keras Graph Networks.
  /// </summary>
  TKerasHistory = class(TObject)
    private
      FLayer        : ILayer;
      Fnode_index   : Integer;
      Ftensor_index : Integer;
    public
      constructor Create(_layer: ILayer; _node_index: Integer; _tensor_index: Integer);
      function ToTuple: Tuple<ILayer,Integer,Integer>;

      property Layer        : ILayer  read FLayer;
      property node_index   : Integer read Fnode_index;
      property tensor_index : Integer read Ftensor_index;
  end;

  /// <summary>
  /// Handles iterating over epoch-level `tf.data.Iterator` objects.
  /// </summary>
  DataHandler = class
    private

    public

  end;

  Container  = class
    protected
       Foutput_names  : TArray<string>;
       Fbuilt         : Boolean;
    public

      constructor Create(output_names: TArray<String>);
  end;

implementation
          uses Tensorflow.Utils;

{ TInputSpec }

constructor TInputSpec.Create(dtype: TF_DataType; _ndim, _min_ndim: Integer; _axes: TDictionary<Integer, Integer>; shape: PTFShape);
begin
    if _ndim = null then FnDim := nil
    else                FnDim := _ndim ;

   if _axes = nil then
      _axes := TDictionary<Integer, Integer>.Create;
  Faxes := _axes;

  if _min_ndim = null then Fmin_ndim := nil
  else                     Fmin_ndim := _min_ndim ;

  if shape <> nil then Fshape := shape^
  else                 Fshape := default(TFShape);

  if (Fndim = nil) and (shape <> nil) then
      Fndim := Fshape.ndim;

  FAllAxisDim := [];
  if Faxes <> nil then
    FAllAxisDim := Faxes.Values.ToArray;
end;

{ CallContextManager }

constructor CallContextManager.Create(build_graph: Boolean);
begin
    Fbuild_graph := build_graph;
end;

procedure CallContextManager.Dispose;
begin

end;

{ CallContext }

constructor TCallContext.Create;
begin

end;

function TCallContext.Enter(build_graph: Boolean): CallContextManager;
begin
   Result := CallContextManager.Create(build_graph);
end;

{ LayerArgs }

constructor LayerArgs.Create;
begin
    Trainable := true;
    Name      := '';
    DType     := TF_DataType.TF_FLOAT;
    bDynamic  := false;
    BatchSize := -1;
end;

{ KerasHistory }

constructor TKerasHistory.Create(_layer: ILayer; _node_index, _tensor_index: Integer);
begin
    Flayer        := _layer;
    Fnode_index   := _node_index;
    Ftensor_index := _tensor_index;
end;

function TKerasHistory.ToTuple: Tuple<ILayer, Integer, Integer>;
begin
    Result := Tuple<ILayer, Integer, Integer>.Create(Flayer,Fnode_index,Ftensor_index)
end;


{ Node }

constructor Node.Create(_args: NodeArgs);
begin
    FKerasInputs := TList<TFTensor>.Create;
    Fargs := _args;
end;

procedure Node.Connect(_layer: ILayer);
begin
    Flayer := _layer;

    if Fargs.InputTensors <> nil then
        FKerasInputs.AddRange(Fargs.InputTensors);

    // Wire up Node to Layers.
    Flayer.InboundNodes.Add(self);

    for var kt in FKerasInputs do
    begin
        if kt.KerasHistory = nil then
            continue;
        var kKt := kt.KerasHistory as TKerasHistory;
        var t   := kKt.ToTuple;
        var inbound_layer := t.Value1;
        if inbound_layer <> nil then
            inbound_layer.OutboundNodes.Add(Self);
    end;

    // Set metadata on outputs.
    var node_index := Flayer.InboundNodes.Count - 1;
    for var i := 0 to Outputs.Count - 1 do
    begin
        var tensor          := Outputs[i];
        tensor.KerasHistory :=  TKerasHistory.Create(layer, node_index, i);
    end;
end;

function Node.serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;
begin
    Result := TList<NodeConfig>.Create;

    for var i := 0 to FKerasInputs.Count-1 do
    begin
        var kh  := FKerasInputs[i].KerasHistory;
        var kkh := kh as TKerasHistory;

        var node_key       := make_node_key(kkh.Layer.Name, kkh.Node_Index);
        var new_node_index := TUtils.Get<string, Integer>(node_conversion_map, node_key, 0);

        var resNode := NodeConfig.Create;
        resNode.Name        := kkh.Layer.Name;
        resNode.NodeIndex   := new_node_index;
        resNode.TensorIndex := kkh.tensor_index;

        Result.Add(resNode);
    end;
end;

function Node.GetILayer: ILayer;
begin
    Result := Flayer;
end;

function Node.GetInLayers: TArray<ILayer>;
begin
    Result := [];
    var t := iterate_inbound.ToArray;
    for var i := 0 to Length(t)-1 do
      Result := Result + [ t[i].Value1 ] ;
end;

function Node.GetINodes: TArray<INode>;
begin
    var node_deps := TList<INode>.Create;
    for var kt in FKerasInputs do
    begin
        var kkh := kt.KerasHistory as TKerasHistory;
        var t   := kkh.ToTuple;
        var llayer     := t.Value1;
        var node_index := t.Value2;
        if llayer <> nil then
            node_deps.Add(llayer.InboundNodes[node_index]);
    end;
    Result := node_deps.ToArray;
end;

function Node.GetInputTensor: TFTensors;
begin
   if is_input then Result := Outputs
   else             Result := Fargs.InputTensors;
end;

function Node.GetIsInput: Boolean;
begin
    Result := Fargs.InputTensors = nil;
end;

function Node.GetlstLayer: TList<TFTensor>;
begin
    Result := FKerasInputs;
end;

procedure Node.SetlstLayer(const lst: TList<TFTensor>);
begin
    FKerasInputs := lst;
end;

function Node.GetOutptus: TFTensors;
begin
    Result := Fargs.Outputs
end;

function Node.iterate_inbound: Enumerable<Tuple<ILayer, Integer, Integer, TFTensor>>;
begin
    var res : TArray< Tuple<ILayer, Integer, Integer, TFTensor> > := [];
    for var kt in FKerasInputs do
    begin
        var kKt := kt.KerasHistory as TKerasHistory;
        var t   := kKt.ToTuple;
        var layer        := t.Value1 ;
        var node_index   := t.Value2;
        var tensor_index := t.Value3;
        res := res + [ tuple.Create(layer, node_index, tensor_index, kt) ] ;
    end;
    Result := Enumerable<Tuple<ILayer, Integer, Integer, TFTensor>>.Create(res);
end;

function Node.MapArguments(tensor_dict: TDictionary<Int64, TQueue<TFTensor>>): TFTensors;
begin
    if FKerasInputs.Count = 1 then
    begin
        var kt_id := FKerasInputs[0].Id;
        Result    := TFTensors.Create( tensor_dict[kt_id].Dequeue );
    end else
    begin
        var flat_arguments := FKerasInputs.ToArray;
        for var kt_index := 0 to  FKerasInputs.Count-1 do
        begin
            var kt                   := FKerasInputs[kt_index] ;
            flat_arguments[kt_index] := tensor_dict[kt.Id].Dequeue;
        end;
        Result := TFTensors.Create( flat_arguments );
    end;
end;

{ NodeConfig }

function NodeConfig.ToString: string;
begin
    Result := Format('%s, %d, %d',[Name, NodeIndex, TensorIndex])
end;

{ LayerConfig }

constructor LayerConfig.Create;
begin

end;

{ Container }

constructor Container.Create(output_names: TArray<String>);
begin
    Foutput_names := output_names;
end;

end.
