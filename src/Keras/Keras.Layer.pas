unit Keras.Layer;
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

interface
       uses System.SysUtils,
            Spring,
            Spring.Collections,
            Spring.Collections.Lists,
            Spring.Collections.Dictionaries,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable;

type

  ILayer          = interface;
  IRegularizer    = interface;
  RegularizerArgs = class;


  RegularizerArgs  = class
    public
      X : TFTensor;

     constructor Create(t: TFTensor);
  end;

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

  IRegularizer = interface
    ['{F08A34E9-848E-4BF7-915F-6216D1C4E078}']

    function Apply(args: RegularizerArgs): TFTensor;
  end;

  IPoolFunction = interface
    ['{17352CCA-5C47-49AD-9265-C9762F7F3FB6}']

    function Apply(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor;
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

    function iterate_inbound :  IEnumerable< Tuple<ILayer, Integer, Integer, TFTensor> >;
    function serialize(make_node_key: TFunc<string, Integer, string>; node_conversion_map: TDictionary<string, Integer>): TList<NodeConfig>;

    property input_tensors: TFTensors      read GetInputTensor;
    property Outputs      : TFTensors      read GetOutptus;
    property Layer        : ILayer         read GetILayer;
    property KerasInputs  : TList<TFTensor>read GetlstLayer write SetlstLayer;
    property ParentNodes  : TArray<INode>  read GetINodes;
    property InboundLayers: TArray<ILayer> read GetInLayers;
    property is_input     : Boolean        read GetIsInput;
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

    function Apply(inputs: TFTensors; state: TFTensor = nil; is_training: Boolean = false): TFTensors;
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

implementation

{ RegularizerArgs }

constructor RegularizerArgs.Create(t: TFTensor);
begin
    X := t;
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

{ NodeConfig }

function NodeConfig.ToString: string;
begin
    Result := Format('"%s, %d, %d";',[Name, NodeIndex, TensorIndex])
end;

end.
