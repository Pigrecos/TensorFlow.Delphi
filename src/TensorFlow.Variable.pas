unit TensorFlow.Variable;

interface
     uses TensorFlow.LowLevelAPI,TensorFlow.DApiBase,TensorFlow.Tensor;

type
  /// <summary>
  /// A variable maintains state in the graph across calls to `run()`. You add a
  /// variable to the graph by constructing an instance of the class `Variable`.
  ///
  /// The `Variable()` constructor requires an initial value for the variable,
  /// which can be a `Tensor` of any type and shape. The initial value defines the
  /// type and shape of the variable. After construction, the type and shape of
  /// the variable are fixed. The value can be changed using one of the assign methods.
  /// https://tensorflow.org/guide/variables
  /// </summary>
  IVariableV1 = interface
  ['{DEBD12E5-E613-4F9A-AEDC-99579EFA9798}']

  end;

  RefVariable = class(TInterfacedObject, IVariableV1)
     private
        Fdtype : TF_DataType;
    function GetTipo: TF_DataType;
     public
        _Variable : TFTensor;

     property Handle : TFTensor    read _variable;
     property dtype  : TF_DataType read GetTipo;
  end;

  BaseResourceVariable = class(TFDisposable)
     private
        Fdtype :  TF_DataType;
     public

     property dtype :  TF_DataType read Fdtype;
  end;

  /// <summary>
  /// Variable based on resource handles.
  /// </summary>
  ResourceVariable = class(BaseResourceVariable,IVariableV1)

  end;


implementation

{ RefVariable }

function RefVariable.GetTipo: TF_DataType;
begin
    Fdtype := _variable.dtype;
    Result := Fdtype;

end;

end.
