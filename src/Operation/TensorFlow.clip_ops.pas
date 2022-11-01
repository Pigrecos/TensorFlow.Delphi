unit TensorFlow.clip_ops;
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
         Spring.Collections.Enumerable,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context;

type
  clip_ops = record
    private

    public
      class function clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor = nil; name: string = ''): Tuple<TFTensors,TFTensor> ; static;
      class function clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string = ''): TFTensor ; static;
      /// <summary>
      /// Computes the global norm of multiple tensors.
      /// </summary>
      /// <param name="t_list"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function global_norm(t_list:  TArray<TFTensor>; name: string = '') : TFTensor ; static;
  end;

implementation
         uses TensorFlow.Tensor,
              Tensorflow.Utils,
              Tensorflow.NameScope,
              TensorFlow.Ops,
              Tensorflow.math_ops,
              Tensorflow.array_ops,
              TensorFlow.Constant_op,
              TensorFlow.nn_ops;

{ clip_ops }

class function clip_ops.clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor; name: string): Tuple<TFTensors, TFTensor>;
begin
    var vValues : TArray<TValue> := [];
    for var i := 0 to Length(t_list)-1 do
       vValues := vValues + [ t_list[i] ];

    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    use_norm := global_norm(t_list, name);
    Result := TUtils.tf_with<TNameScope,Tuple<TFTensors, TFTensor>>( TOps.name_scope(name, 'clip_by_global_norm', @newVal),
                                          function(v1: TNameScope): Tuple<TFTensors, TFTensor>
                                            begin
                                                // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
                                                var scale_for_finite := clip_norm * TTensor(math_ops.minimum(
                                                    Single(1.0) / TTensor(use_norm),
                                                    TTensor(constant_op.constant(1.0, use_norm.dtype, 'Const')) / clip_norm));
                                                // If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
                                                // this will make scale NaN.
                                                var scale := scale_for_finite + (TTensor(use_norm) - use_norm);
                                                var values_clipped : TFTensors := TFTensors.Create;
                                                var i : Integer := 0;
                                                for var v in t_list do
                                                begin
                                                    values_clipped.Add(array_ops.identity(v * scale, 'name_'+ IntToStr(i)) );
                                                    Inc(i);
                                                end;
                                                Result := Tuple<TFTensors, TFTensor>.Create(values_clipped, use_norm);
                                            end );
end;

class function clip_ops.clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string): TFTensor;
begin
    var vValues : TArray<TValue> := [t, TValue.From<T1>(clip_value_min), TValue.From<T2>(clip_value_max)];
    var newVal  : TValue         := TValue.From<TArray<TValue>>(vValues);;

    Result := TUtils.tf_with<TNameScope, TFTensor>( TOps.name_scope(name, 'clip_by_value', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var values := Tops.convert_to_tensor(t, DtInvalid, 't');
                                                // Go through list of tensors, for each value in each tensor clip
                                                var t_min := math_ops.minimum(values, clip_value_max);
                                                // Assert that the shape is compatible with the initial shape,
                                                // to prevent unintentional broadcasting.
                                                var _ := values.shape.merge_with(t_min.shape);
                                                var t_max := math_ops.maximum<TFTensor, T1>(t_min, clip_value_min, name);
                                                _ := values.shape.merge_with(t_max.shape);
                                                Result := t_max;
                                            end );
end;

class function clip_ops.global_norm(t_list: TArray<TFTensor>; name: string): TFTensor;
var
  Selfun   : TFunc<TFTensor,TFTensor>;
begin
    Selfun   := Function(x: TFTensor): TFTensor
                 begin
                     Result := nn_ops.l2_loss(x);
                 end ;
    var vValues : TArray<TValue> := [];
    for var i := 0 to Length(t_list)-1 do
       vValues := vValues + [ t_list[i] ];

    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'global_norm', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                var half_squared_norms := Enumerable<TFTensor>.Create(t_list).Select(Selfun).ToArray;
                                                var half_squared_norm := math_ops.reduce_sum(array_ops.stack(half_squared_norms));
                                                var norm := math_ops.sqrt(TTensor(half_squared_norm) *
                                                    constant_op.constant(2.0, half_squared_norm.dtype,'Const'),
                                                    'global_norm');
                                                Result := norm;
                                            end );
end;

end.

