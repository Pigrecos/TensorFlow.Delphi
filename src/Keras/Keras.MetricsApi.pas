unit Keras.MetricsApi;
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

          TF4D.Core.CApi,
          TensorFlow.DApi;

type
  MetricsApi = class
    public
      function binary_accuracy(y_true: TFTensor; y_pred: TFTensor): TFTensor;
      function categorical_accuracy(y_true: TFTensor; y_pred: TFTensor): TFTensor;
      /// <summary>
      /// Calculates how often predictions matches integer labels.
      /// </summary>
      /// <param name="y_true">Integer ground truth values.</param>
      /// <param name="y_pred">The prediction values.</param>
      /// <returns>Sparse categorical accuracy values.</returns>
      function sparse_categorical_accuracy(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
      function mean_absolute_error(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
      function mean_absolute_percentage_error(y_true: TFTensor; y_pred: TFTensor) : TFTensor;
  end;

implementation
         uses Tensorflow,
              TensorFlow.Tensor,
              Tensorflow.math_ops,
              Tensorflow.array_ops;
{ MetricsApi }

function MetricsApi.binary_accuracy(y_true, y_pred: TFTensor): TFTensor;
var
  threshold : Single;
begin
    threshold := 0.5;
    y_pred := math_ops.cast(TTensor(y_pred) > threshold, y_pred.dtype);
    Result := tf.keras.backend.mean(math_ops.equal(y_true, y_pred), -1);
end;

function MetricsApi.categorical_accuracy(y_true, y_pred: TFTensor): TFTensor;
var
  eql : TFTensor;
begin
    eql := math_ops.equal(math_ops.argmax(y_true, -1), math_ops.argmax(y_pred, -1));
    Result := math_ops.cast(eql, TF_DataType.TF_FLOAT);
end;

function MetricsApi.sparse_categorical_accuracy(y_true, y_pred: TFTensor): TFTensor;
var
  y_pred_rank,
  y_true_rank  : Integer;
begin
    y_pred_rank := y_pred.shape.ndim;
    y_true_rank := y_true.shape.ndim;
    // If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank <>-1) and  (y_pred_rank <> -1) and (y_true.shape.ndim = y_pred.shape.ndim) then
        y_true := array_ops.squeeze(y_true, [ -1 ]);
    y_pred := math_ops.argmax(y_pred, -1);

    // If the predicted output and actual output types don't match, force cast them
    // to match.
    if y_pred.dtype <> y_true.dtype then
        y_pred := math_ops.cast(y_pred, y_true.dtype);

    Result := math_ops.cast(math_ops.equal(y_true, y_pred), TF_DataType.TF_FLOAT);
end;

function MetricsApi.mean_absolute_error(y_true, y_pred: TFTensor): TFTensor;
begin
    y_true := math_ops.cast(y_true, y_pred.dtype);
    Result := tf.keras.backend.mean(math_ops.abs(TTensor(y_pred) - y_true), -1);
end;

function MetricsApi.mean_absolute_percentage_error(y_true, y_pred: TFTensor): TFTensor;
begin
    y_true := math_ops.cast(y_true, y_pred.dtype);
    var diff := (TTensor(y_true) - y_pred) / math_ops.maximum(math_ops.abs(y_true), tf.keras.backend.epsilon);
    Result   := Single(100) * TTensor(tf.keras.backend.mean(math_ops.abs(diff), -1));
end;

end.

