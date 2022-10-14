unit TensorFlow.resource_variable_ops;

interface
        uses System.SysUtils,
             Spring,
             spring.Collections.Dictionaries,
             Spring.Collections.Lists,

             TF4D.Core.CApi,
             TensorFlow.DApiBase,
             TensorFlow.DApi,
             TensorFlow.Variable;

type
  resource_variable_ops = record
    private

    public
      /// <summary>
      /// Creates a variable handle with information to do shape inference.
      /// </summary>
      /// <param name="initial_value"></param>
      /// <param name="shape"></param>
      /// <param name="shared_name"></param>
      /// <param name="name"></param>
      /// <param name="graph_mode"></param>
      /// <returns></returns>
      class function eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name: string; name: string; graph_mode: Boolean): TFTensor; Static;
      class function shape_safe_assign_variable_handle(tHandle: TFTensor;  shape: TArray<Integer>; value: TFTensor; name: string = '') : TFOperation; static;
      class function is_resource_variable(vVar :IVariableV1): Boolean; static;
  end;

implementation
        uses TensorFlow.Ops,
             TensorFlow.gen_resource_variable_ops;

{ resorce_variable_ops }

class function resource_variable_ops.eager_safe_variable_handle(initial_value: TFTensor; shape: TFShape; shared_name, name: string; graph_mode: Boolean): TFTensor;
begin

end;

class function resource_variable_ops.is_resource_variable(vVar: IVariableV1): Boolean;
begin
    Result := vVar is ResourceVariable;
end;

class function resource_variable_ops.shape_safe_assign_variable_handle(tHandle: TFTensor; shape: TArray<Integer>; value: TFTensor; name: string): TFOperation;
begin
    var value_tensor := Tops.convert_to_tensor(value);
    Result           := gen_resource_variable_ops.assign_variable_op(tHandle, value_tensor, name)
end;

end.
