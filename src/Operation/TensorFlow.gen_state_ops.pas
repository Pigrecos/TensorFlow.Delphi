unit TensorFlow.gen_state_ops;

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

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
  gen_state_ops = record
    private

    public
      /// <summary>
      /// Holds state in the form of a tensor that persists across steps.
      /// Outputs a ref to the tensor state so it may be read or modified.
      /// </summary>
      /// <param name="shape">The shape of the variable tensor.</param>
      /// <param name="dtype">The type of elements in the variable tensor.</param>
      /// <param name="name"></param>
      /// <param name="container"></param>
      /// <param name="shared_name"></param>
      /// <returns></returns>
      class function variable_v2(shape: TArray<Integer>; dtype: TF_DataType; name: string = ''; container : string= ''; shared_name: string = '') : TFTensor; static;
      /// <summary>
      /// Update 'ref' by assigning 'value' to it
      /// </summary>
      /// <param name="ref"></param>
      /// <param name="value"></param>
      /// <param name="validate_shape"></param>
      /// <param name="use_locking"></param>
      /// <param name="name"></param>
      class function assign<T>(ref: T; value: TValue; validate_shape : Boolean= true; use_locking : Boolean= true;  name: string = '') : TFTensor; static;
      class function assign_add<T>(ref: IVariableV1; value: T; use_locking : Boolean= false; name: string = '') : TFTensor; static;
      class function assign_sub(ref: IVariableV1; value: TFTensor; use_locking : Boolean= false; name : string = '')  : TFTensor; static;
      /// <summary>
      /// Adds sparse updates to a variable reference.
      /// </summary>
      /// <param name="ref"></param>
      /// <param name="indices"></param>
      /// <param name="updates"></param>
      /// <param name="use_locking"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function scatter_add(ref: IVariableV1; indices: TFTensor; updates: TFTensor; use_locking: Boolean = false; name : string= '') : TFTensor; static;
      class function is_variable_initialized(ref: RefVariable; name: string = '') : TFTensor; static;
  end;

implementation
      uses Tensorflow,
           Numpy,
           TensorFlow.Context,
           TensorFlow.Ops,
           Tensorflow.gen_array_ops,
           TensorFlow.gen_math_ops,
           Tensorflow.NameScope,
           Tensorflow.Utils,
           TensorFlow.Constant_op,
           TensorFlow.EagerTensor;


{ gen_state_ops }

class function gen_state_ops.variable_v2(shape: TArray<Integer>; dtype: TF_DataType; name, container, shared_name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('VariableV2', name, [ GetArg('dtype',dtype),GetArg('shape',TValue.From< TArray<Integer> >(shape)),GetArg('container', container),GetArg('shared_name', shared_name)]);
    Result := _op.outputs[0];

    // test
    //
    var _attrs := TDictionary<string, TValue>.Create;
    try
      _attrs.Add('dtype', _op.get_attr('dtype') );
      _attrs.Add('shape', _op.get_attr('shape') );
      _attrs.Add('container', _op.get_attr('container') );
       _attrs.Add('shared_name', _op.get_attr('shared_name') );
    finally
      _attrs.free;
    end;
end;

class function gen_state_ops.assign<T>(ref: T; value: TValue; validate_shape, use_locking: Boolean; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('Assign', name, ExecuteOpArgs.Create([TValue.From<T>(ref),value])
                            .SetAttributes(['validate_shape', validate_shape,'use_locking',use_locking ])).FirstOrDefault(nil);
end;

class function gen_state_ops.assign_add<T>(ref: IVariableV1; value: T; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('AssignAdd', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('value',TValue.From<T>(value)),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.assign_sub(ref: IVariableV1; value: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('AssignSub', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('value',value),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.scatter_add(ref: IVariableV1; indices, updates: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ScatterAdd', name, [ GetArg('ref',TValue.From<IVariableV1>(ref)),GetArg('indices',indices),GetArg('updates',updates),GetArg('use_locking', use_locking)]);
    Result := _op.outputs[0];
end;

class function gen_state_ops.is_variable_initialized(ref: RefVariable; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('IsVariableInitialized', name, [ GetArg('ref',TValue.From<RefVariable>(ref))]);
    Result := _op.outputs[0];
end;

end.
