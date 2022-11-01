unit TensorFlow.gen_ops;
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
         System.Math,
         Spring,
         Spring.Collections.Dictionaries,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         TensorFlow.OpDefLibrary,
         Tensorflow.Utils,
         TensorFlow.Context;

type
  gen_ops = record
    private

    public
      /// <summary>
      ///    Computes exponential linear: <c>exp(features) - 1</c> if &amp;lt; 0, <c>features</c> otherwise.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Elu'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      /// <remarks>
      ///    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
      ///    ](http://arxiv.org/abs/1511.07289)
      /// </remarks>
      class function elu(features: TFTensor; name: string = 'Elu'): TFTensor; static;
      /// <summary>
      ///    Computes softplus: <c>log(exp(features) + 1)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Softplus'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function softplus(features: TFTensor; name: string = 'Softplus'): TFTensor; static;
      /// <summary>
      ///    Computes softsign: <c>features / (abs(features) + 1)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Softsign'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function softsign(features: TFTensor; name: string = 'Softsign'): TFTensor; static;
      /// <summary>
      ///    Computes rectified linear: <c>max(features, 0)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Relu'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function relu(features: TFTensor; name: string = 'Relu') : TFTensor; static;
      /// <summary>
      ///    Computes rectified linear 6: <c>min(max(features, 0), 6)</c>.
      /// </summary>
      /// <param name="features">
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Relu6'.
      /// </param>
      /// <returns>
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      class function relu6(features: TFTensor; name: string = 'Relu6') : TFTensor; static;
      /// <summary>
      ///    Clips tensor values to a specified min and max.
      /// </summary>
      /// <param name="t">
      ///    A <c>Tensor</c>.
      /// </param>
      /// <param name="clip_value_min">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The minimum value to clip by.
      /// </param>
      /// <param name="clip_value_max">
      ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
      ///    as <c>t</c>. The maximum value to clip by.
      /// </param>
      /// <param name="name">
      /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ClipByValue'.
      /// </param>
      /// <returns>
      ///    A clipped <c>Tensor</c> with the same shape as input 't'.
      ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
      /// </returns>
      /// <remarks>
      ///    Given a tensor <c>t</c>, this operation returns a tensor of the same type and
      ///    shape as <c>t</c> with its values clipped to <c>clip_value_min</c> and <c>clip_value_max</c>.
      ///    Any values less than <c>clip_value_min</c> are set to <c>clip_value_min</c>. Any values
      ///    greater than <c>clip_value_max</c> are set to <c>clip_value_max</c>.
      /// </remarks>
      class function clip_by_value(t: TFTensor; clip_value_min: TFTensor; clip_value_max: TFTensor; name: string = 'ClipByValue') : TFTensor; static;
  end;


implementation
       uses Tensorflow;

{ gen_ops }

class function gen_ops.clip_by_value(t, clip_value_min, clip_value_max: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('t', t);
      dict.add('clip_value_min', clip_value_min);
      dict.add('clip_value_max', clip_value_max);
      var op := tf.OpDefLib._apply_op_helperDict('ClipByValue', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.elu(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Elu', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.relu(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Relu', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.relu6(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Relu6', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.softplus(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Softplus', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

class function gen_ops.softsign(features: TFTensor; name: string): TFTensor;
begin
    var dict := TDictionary<string, TValue>.Create;
    try
      dict.add('features', features);
      var op := tf.OpDefLib._apply_op_helperDict('Softsign', name, dict);
      Result := op.output;
    finally
      dict.free;
    end;
end;

end.
