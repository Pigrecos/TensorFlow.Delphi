unit TensorFlow.gen_sparse_ops;
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
   uses  System.SysUtils,
         Spring,

         TensorFlow.DApi ;

type
  gen_sparse_ops = Record
    private

    public
      /// <summary>
      /// Converts a sparse representation into a dense tensor.
      /// </summary>
      /// <param name="sparse_indices"></param>
      /// <param name="output_shape"></param>
      /// <param name="sparse_values"></param>
      /// <param name="default_value"></param>
      /// <param name="validate_indices"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      Class function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TArray<Integer>;sparse_values: T; default_value: T; validate_indices: boolean = true; name: string = ''): TFTensor; overload; static;
      Class function sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TFTensor; sparse_values: TFTensor; default_value: T; validate_indices: boolean = true; name: string = ''): TFTensor; overload; static;
  End;


implementation
      uses Tensorflow,

               Tensorflow.Utils;
{ gen_sperse_ops }

class function gen_sparse_ops.sparse_to_dense<T>(sparse_indices: TFTensor; output_shape: TArray<Integer>; sparse_values, default_value: T; validate_indices: boolean;
  name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('SparseToDense', name,[ GetArg('sparse_indices',sparse_indices),
                                                                    GetArg('output_shape', TValue.From< TArray<Integer> >(output_shape)),
                                                                    GetArg('sparse_values',TValue.From<T>(sparse_values)),
                                                                    GetArg('default_value',TValue.From<T>(default_value)),
                                                                    GetArg('validate_indices',validate_indices ) ] );
    Result := _op.output;
end;


class function gen_sparse_ops.sparse_to_dense<T>(sparse_indices, output_shape, sparse_values: TFTensor; default_value: T; validate_indices: boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('SparseToDense', name,[ GetArg('sparse_indices',sparse_indices),
                                                                    GetArg('output_shape', output_shape),
                                                                    GetArg('sparse_values',sparse_values),
                                                                    GetArg('default_value',TValue.From<T>(default_value)),
                                                                    GetArg('validate_indices',validate_indices ) ] );
    Result := _op.output;
end;

end.
