unit TensorFlow.control_flow_ops;
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
   uses  System.SysUtils,
         Spring,
         Spring.Collections.Enumerable,
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
       class function tuple(tensors: TArray<TFTensor>; name: string = ''; control_inputs : TArray<TFOperation> = nil) : TArray<TFTensor>; static;
       class function group<T:  ITensorOrOperation>(inputs: TArray<T>; name : string= '') : TFOperation; static;
       /// <summary>
       /// Produces the content of `output_tensor` only after `dependencies`.
       ///
       /// In some cases, a user may want the output of an operation to be
       /// consumed externally only after some other dependencies have run
       /// first.This function ensures returns `output_tensor`, but only after all
       /// operations in `dependencies` have run.Note that this means that there is
       /// no guarantee that `output_tensor` will be evaluated after any `dependencies`
       /// have run.
       ///
       /// See also `tf.tuple` and `tf.group`.
       /// </summary>
       /// <param name="dependencies">Iterable of operations to run before this op finishes.</param>
       /// <param name="output_tensor">A `Tensor` or `IndexedSlices` that will be returned.</param>
       /// <param name="name">(Optional) A name for this operation.</param>
       /// <returns>Same as `output_tensor`.</returns>
       class function with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Does nothing. Only useful as a placeholder for control edges.
       /// </summary>
       /// <param name="name"></param>
       /// <returns></returns>
       class function no_op(name : string= ''): TFOperation; static;
       class function _Identity(data: TFTensor;  name : string = ''): TFTensor; static;
  end;

implementation
      uses Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops,
           Tensorflow.NameScope,
           Tensorflow.gen_array_ops,
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

class function control_flow_ops.tuple(tensors: TArray<TFTensor>; name: string; control_inputs: TArray<TFOperation>): TArray<TFTensor>;
begin
    var vInputs := TValue.From< TArray<TFTensor> >(tensors) ;

    Result := TUtils.tf_with<TNameScope,TArray<TFTensor>>( TOps.name_scope(name, 'tuple', @vInputs),
                function(v1: TNameScope): TArray<TFTensor>
                  begin
                      name := v1.ToString;


                      var gating_ops : TArray<TFOperation>:= [];
                      for var i := 0 to Length(tensors)-1 do
                      begin
                          if tensors[i] <> nil then
                             gating_ops := gating_ops + [ tensors[i].Op ];
                      end;

                      if control_inputs <> nil  then
                      begin
                          for var c in control_inputs do
                             gating_ops := gating_ops + [ c ];
                      end;
                      // Note that in order to ensure ordering in the pbtxt, we must take care to
                      // ensure the order here.
                      var l_gating_ops := Enumerable<TFOperation>.create(gating_ops);
                      l_gating_ops := l_gating_ops.OrderBy<Integer>(function (o: TFOperation): Integer
                                                                      begin
                                                                        Result := o.id;
                                                                      end);
                      var gate := group<TFOperation>(l_gating_ops.ToArray);
                      var tpl := TList<TFTensor>.Create ;
                      try
                        for var t in tensors do
                        begin
                            if t <> nil then tpl.Add( with_dependencies([ gate ], t) )
                            else             tpl.Add(nil);
                        end;
                        Result := tpl.ToArray;
                      finally
                        tpl.Free;
                      end;

                  end );
end;

class function control_flow_ops.with_dependencies(dependencies: TArray<TFOperation>; output_tensor: TFTensor; name: string): TFTensor;
begin
   var adeps : TArray<TValue> := [];
   var aValue : TArray<TValue> := [];

   for var i := 0 to Length(dependencies) -1  do
     adeps := adeps + [ TValue.From<TFOperation>(dependencies[i]) ] ;

   aValue := adeps + [ output_tensor ];

   //TODO: missing original code
   //if context.executing_eagerly():
   //    return output_tensor
   Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'control_dependency', @aValue),
                function(v1: TNameScope): TFTensor
                  begin
                      name := v1.ToString;

                      Tops.colocate_with(output_tensor);
                      Result := TUtils.tf_with<TControlDependenciesController,TFTensor>( Tops.control_dependencies(adeps),
                                  function(v1: TControlDependenciesController): TFTensor
                                    begin
                                        output_tensor := Tops.convert_to_tensor_or_composite(output_tensor);
                                        Result := _Identity(output_tensor, name);
                                    end );
                  end );
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

class function control_flow_ops._Identity(data: TFTensor; name: string): TFTensor;
begin
    data := Tops.internal_convert_to_tensor_or_composite(data, DtInvalid, '', true);
    if Ord(data.dtype) > 100 then
       raise TFException.Create('Not Implemented "_Identity"')
    else
        Result := gen_array_ops.identity(data, name);
end;

end.
