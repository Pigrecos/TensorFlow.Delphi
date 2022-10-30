unit TensorFlow.random_ops;
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
           rtti,
           Spring,
           Spring.Collections.Enumerable,
           Spring.Collections.Lists,
           TF4D.Core.CApi,
           TensorFlow.DApiBase,
           TensorFlow.DApi,
           Numpy.Axis,

           TensorFlow.Context ;

type
  random_ops = record
    private
        class function _ShapeTensor(shape: TArray<Integer>) : TFTEnsor; static;
        /// <summary>
        /// Implementation for random.categorical (v1) and random.categorical (v2).
        /// </summary>
        /// <param name="logits"></param>
        /// <param name="num_samples"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        class function multinomial_categorical_impl(logits: TFTensor; num_samples: Integer; dtype: TF_DataType = DtInvalid; seed : pInteger= nil) : TFTEnsor; static;
    public
        /// <summary>
        ///
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function random_normal(shape: TFShape; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        class function random_uniform(shape: TArray<Integer>; minval: Single = 0; maxval: Single = 1; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; overload; static;
        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        class function random_uniform_int(shape: TArray<Integer>; minval: Integer = 0; maxval: Integer = 1; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        class function random_uniform(shape: TFTensor; minval: Integer = 0; maxval: TFTensor = nil; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; overload; static;
        /// <summary>
        /// Randomly shuffles a tensor along its first dimension.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        class function random_shuffle(value: TFTensor; seed: Integer = 0; name: string = '') : TFTEnsor; static;
        class function truncated_normal(shape: TArray<Integer>; mean: Single = 0.0; stddev: Single = 1.0; dtype: TF_DataType = TF_FLOAT; seed: pInteger = nil; name: string = '') : TFTEnsor; static;
        class function multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger = nil; name: string = ''; output_dtype: TF_DataType = DtInvalid) : TFTEnsor; static;
  end;

implementation
      uses Tensorflow,
           TensorFlow.Ops,
           Tensorflow.Utils,
           Tensorflow.NameScope,
           Tensorflow.math_ops,
           TensorFlow.Framework,
           TensorFlow.gen_random_ops,
           TensorFlow.Tensor;

{ random_ops }

class function random_ops.random_normal(shape: TFShape; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From<TFShape>(shape), mean, stddev]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_normal', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                name := string(v1.ToString);
                                var shape_tensor  := _ShapeTensor(shape);
                                var mean_tensor   := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor := Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);
                                var seed1         := tSeed.Value1;
                                var seed2         := tSeed.Value1;
                                var rnd           := gen_random_ops.random_standard_normal(shape_tensor, dtype, seed1, seed2);
                                var mul           := TTensor(rnd) * TTensor(stddev_tensor);
                                var value         := math_ops.add(mul, mean_tensor, name);
                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function random_ops.random_uniform(shape: TArray<Integer>; minval, maxval: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                var name := string(v1.ToString);

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);

                                var seed1         := tSeed.Value1;
                                var seed2         := tSeed.Value1;
                                var tensorShape   := TUtils.shape_tensor(shape);
                                var minTensor : TTensor := Tops.convert_to_tensor(minval, dtype, 'min');
                                var maxTensor : TTensor := Tops.convert_to_tensor(maxval, dtype, 'max');
                                var rnd       :TTensor  := gen_random_ops.random_uniform(tensorShape, dtype, seed1, seed2);
                                Result := math_ops.add(rnd * (maxTensor - minTensor), minTensor, name);
                            end );
end;

class function random_ops.random_uniform_int(shape: TArray<Integer>; minval: Integer; maxval: Integer; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform_int', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                var name := string(v1.ToString);

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);

                                var seed1         := tSeed.Value1;
                                var seed2         := tSeed.Value1;
                                var tensorShape   := TUtils.shape_tensor(shape);
                                var minTensor := Tops.convert_to_tensor(minval, DtInvalid, 'min');
                                var maxTensor := Tops.convert_to_tensor(maxval, DtInvalid, 'max');
                                Result        := gen_random_ops.random_uniform_int(tensorShape, minTensor, maxTensor, seed1, seed2);
                            end );
end;

class function random_ops.random_uniform(shape: TFTensor; minval: Integer; maxval: TFTensor; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([shape, minval, maxval]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'random_uniform', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                var name      := string(v1.ToString);
                                var minTensor : TTensor := Tops.convert_to_tensor(minval, dtype, 'min');
                                var maxTensor : TTensor;
                                if maxval = nil then maxTensor := Tops.convert_to_tensor(1 , dtype, 'max')
                                else                 maxTensor := Tops.convert_to_tensor(integer(TTensor(maxval)) , dtype, 'max');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);

                                var seed1         := tSeed.Value1;
                                var seed2         := tSeed.Value1;
                                if Tdtypes.is_integer(dType) then
                                begin
                                    Result := gen_random_ops.random_uniform_int(shape, minTensor, maxTensor, seed1, seed2, name);
                                end else
                                begin
                                    var rnd := gen_random_ops.random_uniform(shape, dtype);
                                    Result  := math_ops.add(rnd * (maxTensor - minTensor), minTensor, name);
                                end;
                            end );
end;

class function random_ops.random_shuffle(value: TFTensor; seed: Integer; name: string): TFTEnsor;
var
  tSeed : Tuple<TNullableInteger, TNullableInteger>;
begin
    tSeed      := random_seed.get_seed(seed);
    var seed1  := tSeed.Value1;
    var seed2  := tSeed.Value1;

    Result := gen_random_ops.random_shuffle(value, seed1, seed2, name);
end;

class function random_ops.truncated_normal(shape: TArray<Integer>; mean, stddev: Single; dtype: TF_DataType; seed: pInteger; name: string): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([TValue.From< TArray<Integer> >(shape), mean, stddev]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'truncated_normal', @newVal),
                          function(v1: TNameScope): TFTensor
                          var
                            tSeed : Tuple<TNullableInteger, TNullableInteger>;
                            begin
                                name              := string(v1.ToString);
                                var shape_tensor  := _ShapeTensor(shape);
                                var mean_tensor   := Tops.convert_to_tensor(mean, dtype, 'mean');
                                var stddev_tensor := Tops.convert_to_tensor(stddev, dtype, 'stddev');

                                var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
                                if Assigned(seed) then nSeed := seed^;

                                tSeed             := random_seed.get_seed(nSeed);
                                var seed1         := tSeed.Value1;
                                var seed2         := tSeed.Value1;
                                var rnd           := gen_random_ops.truncated_normal(shape_tensor, dtype, seed1, seed2);
                                var mul           := TTensor(rnd) * TTensor(stddev_tensor);
                                var value         := math_ops.add(mul, mean_tensor, name);
                                // tensor_util.maybe_set_static_shape(value, shape)
                                Result := value;
                            end );
end;

class function random_ops._ShapeTensor(shape: TArray<Integer>): TFTEnsor;
begin
   Result := Tops.convert_to_tensor(TValue.From< TArray<Integer> >(shape), DtInvalid, 'shape');
end;

class function random_ops.multinomial(logits: TFTensor; num_samples: Integer; seed: pInteger; name: string; output_dtype: TF_DataType): TFTEnsor;
begin
    var newVal : TValue := TValue.From<TArray<TValue>>([ logits ]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'multinomial', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                Result := multinomial_categorical_impl(logits, num_samples, output_dtype, seed);
                            end );
end;

class function random_ops.multinomial_categorical_impl(logits: TFTensor; num_samples: Integer; dtype: TF_DataType; seed: pInteger): TFTEnsor;
var
  tSeed : Tuple<TNullableInteger, TNullableInteger>;
begin
    logits := Tops.convert_to_tensor(logits, DtInvalid, 'logits');

    var nSeed : Nullable<Integer> := System.Default(Nullable<Integer>);
    if Assigned(seed) then nSeed := seed^;

    tSeed             := random_seed.get_seed(nSeed);
    var seed1         := tSeed.Value1;
    var seed2         := tSeed.Value1;
    Result :=  gen_random_ops.multinomial(logits, num_samples, seed1, seed2, dtype);
end;

end.

