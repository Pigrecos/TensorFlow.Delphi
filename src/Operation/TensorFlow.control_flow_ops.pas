unit TensorFlow.control_flow_ops;

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
   uses  System.SysUtils,
         Spring,
         Spring.Collections.Dictionaries,
         Spring.Collections.Lists,

         TensorFlow.DApiBase,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context ;

type
  control_flow_ops = record
     private
       class function _GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string = ''): TFOperation;static;
     public
       class function group<T:  ITensorOrOperation>(inputs: TArray<T>; name : string= '') : TFOperation; static;
       /// <summary>
       /// Does nothing. Only useful as a placeholder for control edges.
       /// </summary>
       /// <param name="name"></param>
       /// <returns></returns>
       class function no_op(name : string= ''): TFOperation;static;
  end;

implementation
      uses Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops,
           Tensorflow.NameScope,
           TensorFlow.gen_control_flow_ops ;

{ control_flow_ops }

class function control_flow_ops.group<T>(inputs: TArray<T>; name: string): TFOperation;
begin
    var vInputs := TValue.From< TArray<T> >(inputs) ;

    Result := TUtils.tf_with<TNameScope,TFOperation>( TOps.name_scope(name, 'group_deps', @vInputs),
                                          function(v1: TNameScope): TFOperation
                                            begin
                                                name := v1.ToString;

                                                // Sorts *inputs according to their devices.
                                                var ops_on_device := TDictionary<string, TList<T>>.Create;
                                                for var inp in inputs do
                                                begin
                                                    if ops_on_device.ContainsKey(inp.Device) then
                                                        ops_on_device[inp.Device].Add(inp)
                                                    else
                                                        ops_on_device.add( inp.Device, TList<T>.Create([ inp ]) );
                                                end;
                                                // 1-level tree. The root node is the returned NoOp node.
                                                if ops_on_device.Count = 1 then
                                                begin
                                                    var dev  := ops_on_device.Keys.First;
                                                    var deps := ops_on_device.Values.First;
                                                    var aOp : TArray<TFOperation> := [];
                                                    for var i := 0 to deps.Count - 1 do
                                                       aOp := aOp + [ deps[i].op ];
                                                    Result := _GroupControlDeps(dev, aOp, name);
                                                    Exit;
                                                end;
                                                // 2-level tree. The root node is the returned NoOp node.
                                                // deps contains 1 NoOp node for each device.
                                                raise TFException.Create('control_flow_ops.group');

                                            end );

end;

class function control_flow_ops.no_op(name: string): TFOperation;
begin
    Result := gen_control_flow_ops.no_op(name)
end;

class function control_flow_ops._GroupControlDeps(dev: string; deps: TArray<TFOperation>; name: string): TFOperation;
begin
   var aValue : TArray<TValue> := [];
   for var i := 0 to Length(deps) -1  do
     aValue := aValue + [ TValue.From<TFOperation>(deps[i]) ] ;

   Result := TUtils.tf_with<TControlDependenciesController,TFOperation>( Tops.control_dependencies(aValue),
                                          function(v1: TControlDependenciesController): TFOperation
                                            begin
                                                if dev = '' then
                                                  Result := gen_control_flow_ops.no_op(name)
                                                else
                                                   Result := gen_control_flow_ops.no_op(name);
                                            end );
end;

end.
