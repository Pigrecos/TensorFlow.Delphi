unit TensorFlow.bitwise_ops;
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

interface
    uses System.SysUtils,

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApiBase,
         TensorFlow.DApi,
         TensorFlow.Context ;

type
  /// <summary>
  /// Operations for bitwise manipulation of integers.
  /// https://www.tensorflow.org/api_docs/python/tf/bitwise
  /// </summary>
  bitwise_ops = class
     private
        /// <summary>
        /// Helper method to invoke unary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function unary_op(x: TFTensor; opName: string; name: string) : TFTensor;
        /// <summary>
        /// Helper method to invoke binary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function binary_op(x: TFTensor; y: TFTensor; opName: string; name: string)  : TFTensor;
     public
        /// <summary>
        /// Elementwise computes the bitwise left-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/left_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function left_shift(x: TFTensor; y: TFTensor; name: string = '') : TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise right-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/right_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function right_shift(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise inversion of `x`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/invert
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function invert(x: TFTensor; name: string = '') : TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise AND of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_and
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_and(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise OR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_or
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_or(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
        /// <summary>
        /// Elementwise computes the bitwise XOR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_xor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function bitwise_xor(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;
  end;

implementation
        uses Tensorflow;

{ bitwise_ops }

function bitwise_ops.binary_op(x, y: TFTensor; opName, name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp(opName, name, ExecuteOpArgs.Create([x,y])).FirstOrDefault(nil);
end;

function bitwise_ops.unary_op(x: TFTensor; opName, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp(opName, name, ExecuteOpArgs.Create([x])).FirstOrDefault(nil);
end;

function bitwise_ops.bitwise_and(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseAnd', name);
end;

function bitwise_ops.bitwise_or(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseOr', name);
end;

function bitwise_ops.bitwise_xor(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'BitwiseXor', name);
end;

function bitwise_ops.invert(x: TFTensor; name: string): TFTensor;
begin
   Result := unary_op(x, 'Invert', name);
end;

function bitwise_ops.left_shift(x, y: TFTensor; name: string): TFTensor;
begin
    Result := binary_op(x, y, 'LeftShift', name);
end;

function bitwise_ops.right_shift(x, y: TFTensor; name: string): TFTensor;
begin
     Result := binary_op(x, y, 'RightShift', name);
end;

end.
