unit TensorFlow.stateless_random_ops;
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
           rtti,

           Spring,

           TF4D.Core.CApi,
           TensorFlow.DApi,
           TensorFlow.Core ;

type
   stateless_random_ops  = record
      private
        class function _ShapeTensor(shape: TArray<Integer>): TFTensor; static;
        class function _get_key_counter(seed: TArray<Integer>; alg: Integer) : Tuple<TFTensor, TFTensor>; static;
      public
        class function  stateless_random_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed : TArray<Integer> = []; name: string = '') : TFTensor; static;

   end;


implementation
      uses Tensorflow.Utils,
           Tensorflow.Ops,
           TensorFlow.gen_random_ops,
           Tensorflow.math_ops,
           TensorFlow.Tensor ;

{ stateless_random_ops }

class function stateless_random_ops.stateless_random_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: TArray<Integer>; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([TValue.From<TFShape>(shape), TValue.From<TArray<Integer>>(seed), mean, stddev]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'stateless_random_normal', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                name := v1.ToString;
                                var shape_tensor := _ShapeTensor(shape);
                                var mean_tensor  := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor:= Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                if seed = nil then
                                begin
                                     var f_rnd : Integer := Random(Integer.MaxValue);
                                     seed := [ f_rnd, 0 ];
                                end;

                                var key_count := _get_key_counter(seed, 3);
                                var key     := key_count.Value1;
                                var counter := key_count.Value2;

                                var rnd := gen_random_ops.stateless_random_normal_v2(shape_tensor, key, counter, 3, dtype);
                                var value := math_ops.add(TTensor(rnd) * stddev, mean_tensor, name);

                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function stateless_random_ops._get_key_counter(seed: TArray<Integer>; alg: Integer): Tuple<TFTensor, TFTensor>;
begin
    var results := gen_random_ops.stateless_random_get_key_counter(seed);
    Result := Tuple.Create(results[0], results[1]);
end;

class function stateless_random_ops._ShapeTensor(shape: TArray<Integer>): TFTensor;
begin
    Result := Tops.convert_to_tensor(shape, dtInvalid, 'shape');
end;

end.
