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
          System.Rtti,
          System.Generics.Collections,

          Spring,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.Core,

          Keras.Core,

          Numpy.Axis;

type
  metrics_utils = record
  private
       class function _filter_top_k(x: TFTensor; k: Integer): TFTensor; static;
    public
       class function hamming_loss_fn(y_true: TFTensor; y_pred: TFTensor; threshold: TFTensor; mode: string): TFTensor; static;
       class function accuracy(y_true: TFTensor; y_pred: TFTensor): TFTensor; static;
       class function cosine_similarity(y_true: TFTensor; y_pred: TFTensor; axis: PAxis = nil): TFTensor; static;
       /// <summary>
       /// Creates float Tensor, 1.0 for label-prediction match, 0.0 for mismatch.
       /// </summary>
       /// <param name="y_true"></param>
       /// <param name="y_pred"></param>
       /// <returns></returns>
       class function sparse_categorical_matches(y_true: TFTensor; y_pred: TFTensor): TFTensor; static;
       class function binary_matches(y_true: TFTensor; y_pred: TFTensor; threshold : Single= 0.5): TFTensor; static;
       class function sparse_top_k_categorical_matches(y_true: TFTensor; y_pred: TFTensor; k: Integer = 5): TFTensor; static;
       class function update_confusion_matrix_variables(variables_to_update : TDictionary<string, IVariableV1>;
                                                        var y_true        : TFTensor;
                                                        var y_pred        : TFTensor;
                                                        thresholds    : TFTensor;
                                                        top_k         : Integer;
                                                        class_id      : Integer;
                                                        sample_weight : TFTensor = nil;
                                                        multi_label   : Boolean  = false;
                                                        label_weights : TFTensor = nil;
                                                        thresholds_distributed_evenly : Boolean= false) : TFTensor; static;
  end;

  MetricsApi = class(TInterfacedObject,IMetricsApi)
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
      function top_k_categorical_accuracy(y_true: TFTensor; y_pred: TFTensor; k : Integer= 5) : TFTensor;
      function TopKCategoricalAccuracy(k: Integer = 5; name: string = 'top_k_categorical_accuracy'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      function Recall(thresholds: Single = 0.5; top_k: Integer = 0; class_id : Integer = 0; name: string = 'recall'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
      function Precision(thresholds: Single = 0.5; top_k: Integer = 0; class_id : Integer= 0; name: string = 'recall'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      function BinaryAccuracy(name: string = 'binary_accuracy'; dtype: TF_DataType = TF_FLOAT; threshold : Single= 0.5): IMetricFunc;
      function CategoricalCrossentropy(name: string = 'categorical_crossentropy'; dtype: TF_DataType = TF_FLOAT; from_logits: Boolean = false; label_smoothing : Single= 0; axis: PAxis = nil): IMetricFunc;
      function CategoricalAccuracy(name: string = 'categorical_accuracy';dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      function categorical_crossentropy(y_true: TFTensor; y_pred: TFTensor; from_logits: Boolean = false; label_smoothing :Single= 0; axis : PAxis= nil): TFTensor;
      function Accuracy(name: string = 'accuracy'; dtype : TF_DataType =TF_FLOAT): IMetricFunc;
      function CosineSimilarity(name: string = 'cosine_similarity'; dtype : TF_DataType = TF_FLOAT; axis : PAxis= nil): IMetricFunc;
      function F1Score(num_classes: Integer; average: string = ''; threshold : PSingle= nil; name : string= 'f1_score'; dtype : TF_DataType = TF_FLOAT): IMetricFunc;
      function FBetaScore(num_classes: Integer; average: string = ''; beta: Single = 0.1; threshold: PSingle = nil; name: string = 'fbeta_score'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
      function HammingLoss(mode: string; threshold: PSingle = nil; name: string = 'hamming_loss'; dtype : TF_DataType=TF_FLOAT): IMetricFunc;
      function sparse_categorical_crossentropy(y_true: TFTensor; y_pred: TFTensor; from_logits : Boolean= false; ignore_class : PInteger= nil; axis : PAxis= nil): TFTensor;
      function SparseCategoricalCrossentropy(name: string = 'sparse_categorical_crossentropy'; dtype: TF_DataType = TF_FLOAT; from_logits: Boolean = false; ignore_class: PInteger = nil; axis: PAxis = nil): IMetricFunc;
      function SparseCategoricalAccuracy(name: string = 'sparse_categorical_accuracy'; dtype : TF_DataType= TF_FLOAT): IMetricFunc;
      function SparseTopKCategoricalAccuracy(k: Integer = 5; name: string = 'sparse_top_k_categorical_accuracy'; dtype: TF_DataType = TF_FLOAT): IMetricFunc;
  end;

implementation
         uses Tensorflow,
              tensorFlow.Tensor,
              TensorFlow.Operations,
              TensorFlow.Slice,

              keras.Container,
              keras.Utils,


              Numpy;
{ MetricsApi }

function MetricsApi.binary_accuracy(y_true, y_pred: TFTensor): TFTensor;
var
  threshold : Single;
begin
    threshold := 0.5;
    y_pred := math_ops.cast(TTensor(y_pred) > threshold, y_pred.dtype);
    Result := TKerasApi.keras.backend.mean(math_ops.equal(y_true, y_pred), -1);
end;

function MetricsApi.Accuracy(name: string; dtype: TF_DataType): IMetricFunc;
begin
   Result := Keras.Container.Accuracy.Create(name, dtype)
end;

function MetricsApi.CosineSimilarity(name: string; dtype: TF_DataType; axis: PAxis): IMetricFunc;
begin
   var assi : TAxis := -1;
   if axis <> nil then  assi := axis^;
   Result := Keras.Container.CosineSimilarity.Create(name, dtype, @assi);
end;

function MetricsApi.F1Score(num_classes: Integer; average: string; threshold: PSingle; name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.F1Score.Create(num_classes, average, threshold, name, dtype);
end;

function MetricsApi.FBetaScore(num_classes: Integer; average: string; beta: Single; threshold: PSingle; name: string; dtype: TF_DataType): IMetricFunc;
begin
   Result := Keras.Container.FBetaScore.Create(num_classes, average, beta, threshold, name, dtype);
end;

function MetricsApi.HammingLoss(mode: string; threshold: PSingle; name: string; dtype: TF_DataType): IMetricFunc;
begin
    var s : Single := threshold^;
    var nd_threshold : TNDArray :=  TNDArray.Create(s);
    Result := Keras.Container.HammingLoss.Create(mode, nd_threshold, name, dtype);
end;

function MetricsApi.BinaryAccuracy(name: string; dtype: TF_DataType; threshold: Single): IMetricFunc;
begin
    Result := Keras.Container.BinaryAccuracy.Create;
end;

function MetricsApi.CategoricalAccuracy(name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.CategoricalAccuracy.Create(name, dtype);
end;

function MetricsApi.CategoricalCrossentropy(name: string; dtype: TF_DataType; from_logits: Boolean; label_smoothing: Single; axis: PAxis): IMetricFunc;
begin
    Result := Keras.Container.CategoricalCrossentropy.Create;
end;

function MetricsApi.categorical_accuracy(y_true, y_pred: TFTensor): TFTensor;
var
  eql : TFTensor;
begin
    eql := math_ops.equal(math_ops.argmax(y_true, -1), math_ops.argmax(y_pred, -1));
    Result := math_ops.cast(eql, TF_DataType.TF_FLOAT);
end;

function MetricsApi.categorical_crossentropy(y_true, y_pred: TFTensor; from_logits: Boolean; label_smoothing: Single; axis: PAxis): TFTensor;
begin
    y_true := tf.cast(y_true, y_pred.dtype);
    // var label_smoothing_tensor = tf.convert_to_tensor(label_smoothing, dtype: y_pred.dtype);
    if label_smoothing > 0 then
    begin
        var num_classes := tf.cast(tf.shape(y_true)[-1], y_pred.dtype);
        y_true          := TTensor(y_true) * (1.0 - label_smoothing) + (label_smoothing / TTensor(num_classes));
    end;
    var a : TAxis := -1;
    if axis <> nil then
       a := axis^;
    Result := TKerasApi.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits, a);
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

function MetricsApi.TopKCategoricalAccuracy(k: Integer; name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.TopKCategoricalAccuracy.Create(k, name, dtype);
end;

function MetricsApi.top_k_categorical_accuracy(y_true, y_pred: TFTensor; k: Integer): TFTensor;
begin
    Result := metrics_utils.sparse_top_k_categorical_matches(tf.math.argmax(y_true, -1), y_pred, k );
end;

function MetricsApi.mean_absolute_error(y_true, y_pred: TFTensor): TFTensor;
begin
    y_true := math_ops.cast(y_true, y_pred.dtype);
    Result := TKerasApi.keras.backend.mean(math_ops.abs(TTensor(y_pred) - y_true), -1);
end;

function MetricsApi.mean_absolute_percentage_error(y_true, y_pred: TFTensor): TFTensor;
begin
    y_true := math_ops.cast(y_true, y_pred.dtype);
    var diff := (TTensor(y_true) - y_pred) / math_ops.maximum(math_ops.abs(y_true), TKerasApi.keras.backend.epsilon);
    Result   := Single(100) * TTensor(TKerasApi.keras.backend.mean(math_ops.abs(diff), -1));
end;

function MetricsApi.Precision(thresholds: Single; top_k, class_id: Integer; name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.Precision.Create(thresholds, top_k, class_id, name, dtype)
end;

function MetricsApi.Recall(thresholds: Single; top_k, class_id: Integer; name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.Recall.Create(thresholds, top_k, class_id, name, dtype);
end;

function MetricsApi.sparse_categorical_crossentropy(y_true, y_pred: TFTensor; from_logits: Boolean; ignore_class: PInteger; axis: PAxis): TFTensor;
begin
    var assi : TAxis := -1;
    if Assigned(axis) then assi := axis^;

    Result := TKerasApi.keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits, assi, ignore_class);
end;

function MetricsApi.SparseCategoricalAccuracy(name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.SparseCategoricalAccuracy.Create(name, dtype)
end;

function MetricsApi.SparseCategoricalCrossentropy(name: string; dtype: TF_DataType; from_logits: Boolean; ignore_class: PInteger; axis: PAxis): IMetricFunc;
begin
    var assi : TAxis := -1;
    if Assigned(axis) then assi := axis^;

    Result := Keras.Container.SparseCategoricalCrossentropy.Create(name, dtype, from_logits, ignore_class, @assi)
end;

function MetricsApi.SparseTopKCategoricalAccuracy(k: Integer; name: string; dtype: TF_DataType): IMetricFunc;
begin
    Result := Keras.Container.SparseTopKCategoricalAccuracy.Create(k, name, dtype)
end;

{ metrics_utils }

class function metrics_utils.accuracy(y_true, y_pred: TFTensor): TFTensor;
begin
    if y_true.dtype <> y_pred.dtype then
        y_pred := tf.cast(y_pred, y_true.dtype);
    Result := tf.cast(tf.equal(y_true, y_pred), TKerasApi.keras.backend.floatx);
end;

class function metrics_utils.cosine_similarity(y_true, y_pred: TFTensor; axis: PAxis): TFTensor;
begin
    var assi : TAxis;
    assi := -1;
    if Assigned(axis) then assi := axis^;

    y_true := tf.linalg.l2_normalize(y_true, assi);
    y_pred := tf.linalg.l2_normalize(y_pred, assi);
    Result := tf.reduce_sum(TTensor(y_true) * y_pred, assi);
end;

class function metrics_utils.hamming_loss_fn(y_true, y_pred, threshold: TFTensor; mode: string): TFTensor;
begin
    if threshold = nil then
    begin
        var assi : TAxis := - 1;
        threshold := tf.reduce_max(y_pred, @assi, true);
        // make sure [0, 0, 0] doesn't become [1, 1, 1]
        // Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred := tf.logical_and(TTEnsor(y_pred) >= threshold, TTensor(tf.abs(y_pred)) > Single(1e-12));
    end else
    begin
        y_pred := TTensor(y_pred) > threshold;
    end;
    y_true := tf.cast(y_true, tf.int32_t);
    y_pred := tf.cast(y_pred, tf.int32_t);
    if mode = 'multiclass' then
    begin
        var nonzero : TTensor := tf.cast(tf.math.count_nonzero(TTensor(y_true) * y_pred, -1), tf.float32_t);
        Result := Double(1.0) - nonzero;
    end else
    begin
        var nonzero : TTensor  := tf.cast(tf.math.count_nonzero(TTensor(y_true) - y_pred, -1), tf.float32_t);
        Result := nonzero / y_true.shape[-1];
    end;
end;

class function metrics_utils.binary_matches(y_true, y_pred: TFTensor; threshold: Single): TFTensor;
begin
    y_pred := tf.cast(TTensor(y_pred) > threshold, y_pred.dtype);
    Result := tf.cast(tf.equal(y_true, y_pred), TKerasApi.keras.backend.floatx);
end;

class function metrics_utils.sparse_categorical_matches(y_true, y_pred: TFTensor): TFTensor;
begin
    var reshape_matches  := false;
    var y_true_rank      := y_true.shape.ndim;
    var y_pred_rank      := y_pred.shape.ndim;
    var y_true_org_shape := tf.shape(y_true);
    if (y_true_rank > -1) and (y_pred_rank > -1) and (y_true.ndim = y_pred.ndim )  then
    begin
        reshape_matches := true;
        y_true := tf.squeeze(y_true, TFShape.Create([-1]));
    end;
    y_pred := tf.math.argmax(y_pred, -1);
    y_pred := tf.cast(y_pred, y_true.dtype);
    var matches := tf.cast(tf.equal(y_true, y_pred),  TKerasApi.keras.backend.floatx );
    if reshape_matches then
    begin
        Result := tf.reshape(matches, y_true_org_shape);
        Exit;
    end;
    Result := matches;
end;

class function metrics_utils.sparse_top_k_categorical_matches(y_true, y_pred: TFTensor; k: Integer): TFTensor;
begin
    var reshape_matches  := false;
    var y_true_rank      := y_true.shape.ndim;
    var y_pred_rank      := y_pred.shape.ndim;
    var y_true_org_shape := tf.shape(y_true);
    if y_pred_rank > 2 then
       y_pred := tf.reshape(y_pred, TFShape.Create([-1, y_pred.shape[-1]]));
    if y_true_rank > 1 then
    begin
        reshape_matches := true;
        y_true          := tf.reshape(y_true, -1);
    end;
    var x := tf.math.in_top_k( y_pred, tf.cast(y_true, np.np_int32), k);
    var matches := tf.cast(x, TKerasApi.keras.backend.floatx );
    if reshape_matches then
    begin
        Result := tf.reshape(matches, y_true_org_shape);
        Exit;
    end;
    Result := matches;
end;

class function metrics_utils._filter_top_k(x: TFTensor; k: Integer): TFTensor;
begin
    var NEG_INF : Double := -1e10;
    var t := tf.math.top_k(x, k, false);
    var top_k_idx := t[1];
    var pA : TAxis := -2;
    var top_k_mask : TTensor := tf.reduce_sum( tf.one_hot(top_k_idx, x.shape[-1], nil, nil, DtInvalid, -1), @pA );
    Result := x * top_k_mask + NEG_INF * (Integer(1) - top_k_mask);
end;

class function metrics_utils.update_confusion_matrix_variables(variables_to_update: TDictionary<string, IVariableV1>; var y_true, y_pred: TFTensor; thresholds: TFTensor; top_k,
  class_id: Integer; sample_weight: TFTensor; multi_label: Boolean; label_weights: TFTensor; thresholds_distributed_evenly: Boolean): TFTensor;
begin
    var variable_dtype := variables_to_update.ToArray[0].Value.dtype;
    y_true := tf.cast(y_true, variable_dtype);
    y_pred := tf.cast(y_pred, variable_dtype);
    var num_thresholds := thresholds.shape.dims[0];

    var one_thresh : TFTensor;
    if multi_label then
    begin
        one_thresh := tf.equal(tf.cast(constant_op.constant(Integer(1)), tf.int32_t), tf.rank(thresholds), 'one_set_of_thresholds_cond');
    end else
    begin
        one_thresh := tf.cast(constant_op.constant(true), tf.bool_t);
    end;

    if sample_weight = nil then
    begin
        var y_pred_y_true_ := losses_utils.squeeze_or_expand_dimensions(y_pred, y_true);
        y_pred := y_pred_y_true_.Value1;
        y_true := y_pred_y_true_.Value2;
    end else
    begin
        sample_weight := tf.cast(sample_weight, variable_dtype);
        var tDim := losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight);
        y_pred        := tDim.Value1;
        y_true        := tDim.Value2;
        sample_weight := tDim.Value3;
    end;

    if top_k > 0 then
       y_pred := _filter_top_k(y_pred, top_k);


    if class_id > 0 then
    begin
        y_true := y_true[ [Slice.All, class_id] ];
        y_pred := y_pred[ [Slice.All, class_id] ];
    end;

    if thresholds_distributed_evenly then
       raise Exception.Create('Not Implemented');

    var pred_shape      := tf.shape(y_pred);
    var num_predictions := pred_shape[0];

    var num_labels: TFTensor;
    if y_pred.shape.ndim = 1 then
    begin
        num_labels := constant_op.constant(Integer(1));
    end else
    begin
        var pA : TAxis := 0;
        num_labels := tf.reduce_prod(pred_shape['1:'], @pA);
    end;
    var a : TArray<Integer> := [];
    var thresh_label_tile := tf.where(one_thresh, num_labels, tf.ones(a, tf.int32_t));

    // Reshape predictions and labels, adding a dim for thresholding.
    var predictions_extra_dim, labels_extra_dim : TFTensor;
    if multi_label then
    begin
        predictions_extra_dim := tf.expand_dims(y_pred, 0);
        labels_extra_dim      := tf.expand_dims(tf.cast(y_true, tf.bool_t), 0);
    end else
    begin
        // Flatten predictions and labels when not multilabel.
        predictions_extra_dim := tf.reshape(y_pred, TFShape.Create([1, -1]));
        labels_extra_dim      := tf.reshape(tf.cast(y_true, tf.bool_t), TFShape.Create([1, -1]));
    end;

    // Tile the thresholds for every prediction.
    var thresh_pretile_shape, thresh_tiles, data_tiles : TArray<TValue> ;

    if multi_label then
    begin
        thresh_pretile_shape := [ num_thresholds, 1, -1 ];
        thresh_tiles         := [ 1, num_predictions, thresh_label_tile ];
        data_tiles           := [ num_thresholds, 1, 1 ];
    end else
    begin
        thresh_pretile_shape := [ num_thresholds, -1 ];
        thresh_tiles         := [ 1, TFTensor(TTensor(num_predictions) * num_labels) ];
        data_tiles           := [ num_thresholds, 1 ];
    end;
    var thresh_tiled := tf.tile(tf.reshape(thresholds, thresh_pretile_shape), tf.stack( TValue.From< TArray<TValue> >(thresh_tiles) ));

    // Tile the predictions for every threshold.
    var preds_tiled := tf.tile(predictions_extra_dim, data_tiles);

    // Compare predictions and threshold.
    var pred_is_pos := tf.greater(preds_tiled, thresh_tiled);

    // Tile labels by number of thresholds
    var label_is_pos := tf.tile(labels_extra_dim, data_tiles);

    var weights_tiled : TFTensor := nil;

    if sample_weight <> nil then
    begin
        (*sample_weight = broadcast_weights(tf.cast(sample_weight, dtype: variable_dtype), y_pred);*)
        weights_tiled := tf.tile(tf.reshape(sample_weight, thresh_tiles), data_tiles);
    end;

    if (label_weights <> nil) and (not multi_label) then
       raise Exception.Create('Not Implemented') ;


    var weighted_assign_add : TFunc<TFTensor, TFTensor, TFTensor, IVariableV1, ITensorOrOperation> :=
                       function(l_label:TFTensor; pred:TFTensor; weights : TFTensor; v_var : IVariableV1): ITensorOrOperation
                            begin
                                var label_and_pred : TTensor := tf.cast(tf.logical_and(l_label, pred), v_var.dtype);
                                if weights <> nil then
                                   label_and_pred := label_and_pred * tf.cast(weights, v_var.dtype);

                                var aP: TAxis := 1;

                                if      v_var is RefVariable          then Result := (v_var as RefVariable)         .assign_add(tf.reduce_sum(label_and_pred, @aP))
                                else if v_var is BaseResourceVariable then Result := (v_var as BaseResourceVariable).assign_add(tf.reduce_sum(label_and_pred, @aP))
                                else raise Exception.Create('weighted_assign_add Error!');
                            end;


    var loop_vars := TDictionary< string, Tuple<TFTensor, TFTensor> >.Create;
    loop_vars.Add('tp',Tuple.Create(label_is_pos, pred_is_pos) ) ;

    var update_tn := variables_to_update.ContainsKey('tn');
    var update_fp := variables_to_update.ContainsKey('fp');
    var update_fn := variables_to_update.ContainsKey('fn');

    var pred_is_neg : TFTensor := nil;
    if (update_fn) or (update_tn) then
    begin
        pred_is_neg     := tf.logical_not(pred_is_pos);
        loop_vars.AddOrSetValue('fn', Tuple.Create(label_is_pos, pred_is_neg));
    end;

    if (update_fp) or (update_tn) then
    begin
        var label_is_neg := tf.logical_not(label_is_pos);
        loop_vars.AddOrSetValue('fp', Tuple.Create(label_is_neg, pred_is_pos));
        if update_tn then
            loop_vars.AddOrSetValue('tn', Tuple.Create(label_is_neg, pred_is_neg));
    end;

    var update_ops := TList<ITensorOrOperation>.Create;
    for var matrix_cond in loop_vars.Keys do
    begin
        var tM := loop_vars[matrix_cond];
        var llabel := tM.Value1;
        var pPred  := tM.Value2;
        if variables_to_update.ContainsKey(matrix_cond) then
        begin
            var op := weighted_assign_add(llabel, pPred, weights_tiled, variables_to_update[matrix_cond]);
            update_ops.Add(op);
        end;
    end;

    tf.group<ITensorOrOperation>(update_ops.ToArray);
    Result := nil ;
end;

end.


