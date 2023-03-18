unit TensorFlow.gen_random_ops;
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
           rtti,

           Spring,

           TF4D.Core.CApi,
           TensorFlow.DApi,
           TensorFlow.Core ;

type
  gen_random_ops = record
    private

    public
      /// <summary>
      /// Outputs random values from a normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_standard_normal(shape: TFTensor; dtype: TF_DataType = DtInvalid; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random integers from a uniform distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="minval"></param>
      /// <param name="maxval"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_uniform_int(shape: TFTensor; minval: TFTensor; maxval: TFTensor; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random values from a uniform distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_uniform(shape: TFTensor; dtype: TF_DataType; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      ///
      /// </summary>
      /// <param name="value"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function random_shuffle(value: TFTensor; seed: Integer = 0; seed2: Integer = 0; name: string = ''): TFTensor; static;
      /// <summary>
      /// Outputs random values from a truncated normal distribution.
      /// </summary>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <param name="seed"></param>
      /// <param name="seed2"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function truncated_normal(shape: TFTensor; dtype: TF_DataType; seed: Integer = 0; seed2: Integer = 0; name: string = '') : TFTensor; static;
      Class function multinomial(logits: TFTensor; num_samples: Integer; seed: Integer = 0; seed2: Integer = 0; output_dtype: TF_DataType = TF_INT64; name: string = '') : TFTensor; static;
      Class function stateless_random_normal_v2(shape: TFTensor; key: TFTensor; counter: TFTensor; alg: Integer; dtype: TF_DataType; name: string = '') : TFTensor; static;
      Class function stateless_random_get_key_counter(seed: TArray<Integer>; name: string = '') : TFTensors; static;
  end;

implementation
        uses Tensorflow,

             Tensorflow.Utils;

{ gen_random_ops }

class function gen_random_ops.random_standard_normal(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomStandardNormal', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.random_uniform_int(shape, minval, maxval: TFTensor; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomUniformInt', name, ExecuteOpArgs.Create([ shape, minval, maxval ])
                                                 .SetAttributes(['seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.stateless_random_get_key_counter(seed: TArray<Integer>; name: string): TFTensors;
begin
     Result := tf.Context.ExecuteOp('StatelessRandomGetKeyCounter', name, ExecuteOpArgs.Create([ seed ]))
end;

class function gen_random_ops.stateless_random_normal_v2(shape, key, counter: TFTensor; alg: Integer; dtype: TF_DataType; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('StatelessRandomNormalV2', name, ExecuteOpArgs.Create([ shape, key, counter, alg ])
                                                              .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)) ]) ).First;
end;

class function gen_random_ops.random_uniform(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomUniform', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue.From<Integer>(Ord(dtype)),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.random_shuffle(value: TFTensor; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('RandomShuffle', name, ExecuteOpArgs.Create([ value ])
                                                 .SetAttributes(['seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.truncated_normal(shape: TFTensor; dtype: TF_DataType; seed, seed2: Integer; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('TruncatedNormal', name, ExecuteOpArgs.Create([ shape ])
                                                 .SetAttributes(['dtype', TValue(dtype),'seed', seed, 'seed2',seed2 ]) ).First;
end;

class function gen_random_ops.multinomial(logits: TFTensor; num_samples, seed, seed2: Integer; output_dtype: TF_DataType; name: string): TFTensor;
begin
   var _op := tf.OpDefLib._apply_op_helper('Multinomial', name, [ GetArg('logits',logits), GetArg('seed',seed), GetArg('seed2',seed2), GetArg('output_dtype',TValue(output_dtype)) ]);
   Result := _op.Output;
end;

end.
