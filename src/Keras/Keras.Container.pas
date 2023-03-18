unit Keras.Container;
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
           System.Generics.Collections,

           Spring,

           TF4D.Core.CApi,
           TensorFlow.DApi,
           TensorFlow.Core,

           keras.Core,
           Keras.Layer,
           Keras.LossFunc,
           Keras.MetricsApi,

           Numpy.Axis;

type
  /// <summary>
  /// Encapsulates metrics that perform a reduce operation on the values.
  /// </summary>
  Reduce = class(Metric)
    public
      constructor Create(_reduction: string; name: string; dtype : TF_DataType = DtInvalid);

      function    update_state(values: TFTensor; sample_weight : TFTensor= nil): TFTensor; reintroduce; overload ;
      function    R_result: TFTensor; override;
  end;

  Recall = class(Metric)
    private
      Fthresholds     : TFTensor;
      Ftop_k          : Integer;
      Fclass_id       : Integer;
      Ftrue_positives : IVariableV1;
      Ffalse_negatives: IVariableV1;
      Fthresholds_distributed_evenly : Boolean;
    public
      constructor  Create(thresholds : Single= 0.5; top_k : Integer= 1; class_id : Integer= 0; name: string = 'recall'; dtype: TF_DataType = TF_FLOAT);
      function     update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
      procedure    reset_states; override;
      function     R_result: TFTensor; override;
  end;

  Precision = class(Metric)
    private
      Fthresholds     : TFTensor;
      Ftop_k          : Integer;
      Fclass_id       : Integer;
      Ftrue_positives : IVariableV1;
      Ffalse_positives: IVariableV1;
      Fthresholds_distributed_evenly : Boolean;
    public
      constructor  Create(thresholds : Single= 0.5; top_k : Integer= 0; class_id : Integer= 0; name: string = 'Precision'; dtype: TF_DataType = TF_FLOAT);
      function     update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
      procedure    reset_states; override;
      function     R_result: TFTensor; override;
  end;

  FBetaScore = class(Metric)
    private
      Fnum_classes    : Integer;
      Faverage        : string;
      Fbeta           : TFTensor;
      Fthreshold      : TFTensor;
      FAxis           : TAxis;
      Finit_shape     : TArray<Integer>;

      Ftrue_positives : IVariableV1;
      Ffalse_positives: IVariableV1;
      Ffalse_negatives: IVariableV1;
      Fweights_intermediate : IVariableV1;

      function _weighted_sum(VVal: TFTensor; sample_weight: TFtensor= nil): TFTensor;
    public
      constructor  Create(num_classes: Integer; average: string = ''; beta: Single = 0.1; threshold: PSingle = nil; name: string = 'fbeta_score'; dtype: TF_DataType=TF_FLOAT);
      function     update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
      procedure    reset_states; override;
      function     R_result: TFTensor; override;
  end;

  F1Score = class(FBetaScore)
    public
       constructor  Create(num_classes: Integer; average: string = ''; threshold: PSingle = nil; name: string = 'f1_score'; dtype: TF_DataType=TF_FLOAT);
  end;

  /// <summary>
  /// Computes the (weighted) mean of the given values.
  /// </summary>
  Mean = class(Reduce)
    public
      constructor Create(name: string = 'mean'; dtype: TF_DataType = TF_FLOAT);
  end;

  MeanMetricWrapper = class(Mean)
    private
       F_fn : TFunc<TFTensor, TFTensor, TFTensor>;
    public
       constructor Create(fn: TFunc<TFTensor, TFTensor, TFTensor>; name: string; dtype : TF_DataType = TF_FLOAT);
       function    update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
  end;

  TopKCategoricalAccuracy = class(MeanMetricWrapper)
    public
       constructor Create(k : Integer = 5 ; name: string= 'top_k_categorical_accuracy'; dtype : TF_DataType = TF_FLOAT);
  end;

  BinaryAccuracy = class(MeanMetricWrapper)
    public
       constructor Create(name: string= 'binary_accuracy'; dtype : TF_DataType = TF_FLOAT; threshold : Single = 0.5);
  end;

  CategoricalAccuracy = class(MeanMetricWrapper)
    public
       constructor Create(name: string= 'categorical_accuracy'; dtype : TF_DataType = TF_FLOAT);
  end;

  CategoricalCrossentropy = class(MeanMetricWrapper)
    public
       constructor Create(name: string = 'categorical_crossentropy'; dtype : TF_DataType = TF_FLOAT; from_logits: Boolean = false; label_smoothing: Single = 0; axis: PAxis = nil);
  end;

  CosineSimilarity = class(MeanMetricWrapper)
    public
       constructor Create(name: string = 'cosine_similarity'; dtype : TF_DataType = TF_FLOAT; axis: PAxis = nil);
  end;

  Accuracy = class(MeanMetricWrapper)
    public
       constructor Create(name: string = 'accuracy'; dtype : TF_DataType = TF_FLOAT);
  end;

  HammingLoss = class(MeanMetricWrapper)
    public
       constructor Create(mode: string; threshold : TNDArray= nil; name: string = 'hamming_loss'; dtype: TF_DataType = TF_FLOAT);
  end;

  SparseCategoricalAccuracy = class(MeanMetricWrapper)
    public
       constructor Create(name: string = 'sparse_categorical_accuracy'; dtype: TF_DataType = TF_FLOAT);
  end;

  SparseCategoricalCrossentropy = class(MeanMetricWrapper)
    public
       constructor Create(name: string = 'sparse_categorical_crossentropy'; dtype: TF_DataType = TF_FLOAT; from_logits: Boolean = false; ignore_class: PInteger = nil; axis: PAxis = nil);
  end;

  SparseTopKCategoricalAccuracy = class(MeanMetricWrapper)
    public
       constructor Create(k: Integer = 5; name: string = 'sparse_top_k_categorical_accuracy'; dtype: TF_DataType = TF_FLOAT);
  end;

  Container  = class
    protected
       Foutput_names  : TArray<string>;
       Fbuilt         : Boolean;
    public

      constructor Create(output_names: TArray<String>);
  end;

  MetricsContainer = class(Container)
    private
       Fuser_metrics    : TArray<IMetricFunc>;
       Fmetric_names    : TArray<string>;
       Fmetrics         : TArray<Metric>;
       Fmetrics_in_order: TList<IMetricFunc> ;

       function GetMetrics: TList<IMetricFunc>;
    public
       constructor Create(_metrics: TArray<string>; output_names : TArray<string>= []);overload;
       constructor Create(_metrics: TArray<IMetricFunc>; output_names : TArray<string>= []);overload;
       procedure   update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight:TFTensor = nil);
       procedure   Build(y_true: TFTensor; y_pred: TFTensor);
       procedure   _set_metric_names;
       procedure   _create_ordered_metrics;
       function    _get_metric_objects(_metrics: TArray<string>; y_t: TFTensor; y_p: TFTensor): TArray<Metric>; overload;
       function    _get_metric_object(metric: string; y_t: TFTensor; y_p: TFTensor): Metric; overload;

       property metrics : TList<IMetricFunc> read GetMetrics;
  end;

  LossesContainer = class(Container)
    private
       Fuser_losses        : ILossFunc;
       Flosses             : ILossFunc;
       Floss_metric        : Mean;
       Fbuilt              : Boolean;
      // Fper_output_metrics : TArray<TFTensor>;

       function GetMetrics: TList<Metric>;
    public
       constructor Create(losses: ILossFunc; output_names : TArray<string>= nil);
       /// <summary>
       /// Computes the overall loss.
       /// </summary>
       /// <param name="y_true"></param>
       /// <param name="y_pred"></param>
       function Call(y_true: TFTensor; y_pred: TfTensor) : TFTensor;
       procedure Build(y_pred: TFTensor);
       procedure _create_metrics;

       property metrics : TList<Metric> read GetMetrics;
  end;


implementation
        uses Tensorflow,
             TensorFlow.Tensor,
             TensorFlow.Ops,
             Tensorflow.Utils,
             Tensorflow.math_ops,
             Tensorflow.array_ops,

             Numpy,


             Keras.Utils;

{ Reduce }

function AssignAdd(vVariable: IVariableV1; t_tensor: TFTensor): TFTensor;
begin
  if      vVariable is RefVariable          then Result := (vVariable as RefVariable)         .assign_add(t_tensor)
  else if vVariable is BaseResourceVariable then Result := (vVariable as BaseResourceVariable).assign_add(t_tensor)
  else raise Exception.Create('AssignAdd Error!');
end;

constructor Reduce.Create(_reduction, name: string; dtype: TF_DataType);
begin
    inherited Create(name, dtype);

    Freduction := _reduction;
    Fdtype     := dtype;

    total      := add_weight('total', default(TFShape), TF_FLOAT, tf.zeros_initializer);

    if (_reduction = Reduction.WEIGHTED_MEAN) or (_reduction = Reduction.SUM_OVER_BATCH_SIZE) then
          count := add_weight('count', default(TFShape), TF_FLOAT, tf.zeros_initializer);
end;

function Reduce.R_result: TFTensor;
begin
    if Freduction = Reduction.SUM then
        Exit( array_ops.identity(total.AsTensor) )
    else if (Freduction = Reduction.WEIGHTED_MEAN) or (Freduction = Reduction.SUM_OVER_BATCH_SIZE) then
        Exit( math_ops.div_no_nan(total.AsTensor, count.AsTensor) );

    Result := inherited R_result;
end;

function Reduce.update_state(values, sample_weight: TFTensor): TFTensor;
begin
    if sample_weight <> nil then
    begin
        var values_sample_weight := losses_utils.squeeze_or_expand_dimensions( values, nil, sample_weight);
        values        := values_sample_weight.Value1;
        sample_weight := values_sample_weight.Value3;

        sample_weight := math_ops.cast(sample_weight, values.dtype);
        values        := math_ops.multiply(values, sample_weight);
    end;

    var update_total_op : TFTensor := nil;
    var value_sum := math_ops.reduce_sum(values);
    Tutils.tf_with<TControlDependenciesController>(Tops.control_dependencies([ value_sum ]), procedure(ctrl: TControlDependenciesController)
                    begin
                        if      total is RefVariable          then update_total_op := (total as RefVariable)         .assign_add(value_sum)
                        else if total is BaseResourceVariable then update_total_op := (total as BaseResourceVariable).assign_add(value_sum)
                    end);

    // Exit early if the reduction doesn't have a denominator.
    if Freduction = Reduction.SUM then
        Exit( update_total_op );

    // Update `count` for reductions that require a denominator.
    var num_values : TFTensor := nil;
    if Freduction = Reduction.SUM_OVER_BATCH_SIZE then
        num_values := math_ops.cast(array_ops.size(values), Fdtype)
    else if Freduction = ReductionV2.WEIGHTED_MEAN then
    begin
        if sample_weight = nil then
            num_values := math_ops.cast(array_ops.size(values), Fdtype)
        else
            num_values := math_ops.reduce_sum(sample_weight);
    end;
    Result := TUtils.tf_with<TControlDependenciesController,TFTensor>(Tops.control_dependencies([ update_total_op ]),
            function(d : TControlDependenciesController ): TFtensor
              begin
                  if      count is RefVariable          then Result := (count as RefVariable)         .assign_add(num_values)
                  else if count is BaseResourceVariable then Result := (count as BaseResourceVariable).assign_add(num_values)
                  else raise Exception.Create('Reduce.update_state Error!');
              end);
end;

{ Container }

constructor Container.Create(output_names: TArray<String>);
begin
    Foutput_names := output_names;
end;

{ MetricsContainer }

constructor MetricsContainer.Create(_metrics: TArray<IMetricFunc>; output_names: TArray<string>);
begin
    inherited Create(output_names);

    Fuser_metrics := _metrics;
    Fbuilt        := false;
end;

constructor MetricsContainer.Create(_metrics, output_names: TArray<string>);
begin
    inherited Create(output_names);

    Fmetric_names := _metrics;
    Fbuilt        := false;
end;

procedure MetricsContainer.Build(y_true, y_pred: TFTensor);
begin
    Fmetrics := _get_metric_objects(Fmetric_names, y_true, y_pred);
    _set_metric_names;
    _create_ordered_metrics;
    Fbuilt := true;
end;

function MetricsContainer.GetMetrics: TList<IMetricFunc>;
begin
    if not Fbuilt then
        Exit( TList<IMetricFunc>.Create);

    Result := Fmetrics_in_order;
end;

procedure MetricsContainer.update_state(y_true, y_pred, sample_weight: TFTensor);
begin
    if not Fbuilt then
        Build(y_true, y_pred);

    for var metric_obj in Fmetrics_in_order do
        metric_obj.update_state(y_true, y_pred);
end;

procedure MetricsContainer._create_ordered_metrics;
begin
    Fmetrics_in_order := TList<IMetricFunc>.Create;
    for var m in Fuser_metrics do
        Fmetrics_in_order.Add(m);
end;

function MetricsContainer._get_metric_object(metric: string; y_t, y_p: TFTensor): Metric;
var
  metric_obj : TFunc<TFTensor, TFTensor, TFTensor>;
begin
    metric_obj := nil;
    if (metric = 'accuracy') or (metric = 'acc') then
    begin
        var y_t_rank := y_t.rank;
        var y_p_rank := y_p.rank;
        var y_t_last_dim := y_t.shape[y_t.shape.ndim - 1];
        var y_p_last_dim := y_p.shape[y_p.shape.ndim - 1];

        var is_binary: Boolean              := y_p_last_dim = 1;
        var is_sparse_categorical : Boolean := ((y_t_rank < y_p_rank) or (y_t_last_dim = 1)) and (y_p_last_dim > 1);

        if is_binary then
            metric_obj := MetricsApi(tf.keras.metrics).binary_accuracy
        else if is_sparse_categorical then
            metric_obj := MetricsApi(tf.keras.metrics).sparse_categorical_accuracy
        else
            metric_obj := MetricsApi(tf.keras.metrics).categorical_accuracy;

        metric := 'accuracy';
    end
    else if (metric = 'mean_absolute_error') or (metric = 'mae')  then
    begin
        metric_obj := MetricsApi(tf.keras.metrics).mean_absolute_error;
        metric     := 'mean_absolute_error';
    end
    else if (metric = 'mean_absolute_percentage_error') or (metric = 'mape') then
    begin
        metric_obj := MetricsApi(tf.keras.metrics).mean_absolute_percentage_error;
        metric     := 'mean_absolute_percentage_error';
    end
    else
       raise Exception.Create('Not Implemented');

    Result := MeanMetricWrapper.Create(metric_obj, metric);
end;

function MetricsContainer._get_metric_objects(_metrics: TArray<string>; y_t, y_p: TFTensor): TArray<Metric>;
var
  res : TArray<Metric>;
begin
    res := [];
    for var i := 0 to Length(_metrics)-1 do
    begin
       var m := _get_metric_object(_metrics[i], y_t, y_p);
       res := res + [ m ];
    end;
    Result := res;
end;

procedure MetricsContainer._set_metric_names;
begin

end;

{ Mean }

constructor Mean.Create(name: string; dtype: TF_DataType);
begin
    inherited Create(Reduction.WEIGHTED_MEAN, name, dtype)
end;

{ MeanMetricWrapper }

constructor MeanMetricWrapper.Create(fn: TFunc<TFTensor, TFTensor, TFTensor>; name: string; dtype: TF_DataType);
begin
    inherited Create(name, dtype);

    F_fn :=  fn
end;

function MeanMetricWrapper.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    y_true := math_ops.cast(y_true, Fdtype);
    y_pred := math_ops.cast(y_pred, Fdtype);

    var y_pred_y_true_ := losses_utils.squeeze_or_expand_dimensions(y_pred, y_true);
    y_pred := y_pred_y_true_.Value1;
    y_true := y_pred_y_true_.Value2;

    var matches := F_fn(y_true, y_pred);
    Result := inherited update_state(matches, sample_weight);
end;

{ LossesContainer }

constructor LossesContainer.Create(losses: ILossFunc; output_names: TArray<string>);
begin
    inherited Create(output_names) ;

    Fuser_losses := losses;
    Flosses      := losses;
    Floss_metric := Mean.Create('loss');
    Fbuilt       := false;
end;

function LossesContainer.Call(y_true, y_pred: TfTensor): TFTensor;
var
  loss_value, batch_dim,
  loss_metric_value     : TFTensor;
  loss_values           : TList<TFTensor>;
  loss_metric_values    : TList<TFTensor>;
begin
    if not Fbuilt then
        Build(y_pred);

    loss_value        := Flosses.Call(y_true, y_pred);
    loss_metric_value := loss_value;
    batch_dim         := array_ops.shape(y_true)[0];

    loss_values        := TList<TFTensor>.Create;
    loss_metric_values := TList<TFTensor>.Create;

    (*if (_losses.Reduction == ReductionV2.SUM_OVER_BATCH_SIZE
        || _losses.Reduction == ReductionV2.AUTO)
        loss_value = losses_utils.scale_loss_for_distribution(loss_value);*)
    loss_values.Add(loss_value);
    loss_metric_values.Add(loss_metric_value);

    if loss_values.Count > 0 then
    begin
        var total_loss_metric_value := math_ops.add_n(loss_metric_values.ToArray);
        Floss_metric.update_state(total_loss_metric_value, batch_dim);
        // loss_values = losses_utils.cast_losses_to_common_dtype(loss_values);
        var total_loss := math_ops.add_n(loss_values.ToArray);
        Result := total_loss;
    end else
    begin
        // Ok for a model to have no compiled loss.
        Result := array_ops.zeros(TFShape.Null);
    end;
end;

procedure LossesContainer.Build(y_pred: TFTensor);
begin
   _create_metrics;
   Fbuilt := true;
end;

function LossesContainer.GetMetrics: TList<Metric>;
begin
    if not Fbuilt then
        Exit( TList<Metric>.Create);

    Result := TList<Metric>.Create([Floss_metric]);
end;

procedure LossesContainer._create_metrics;
begin
   // _per_output_metrics = _output_names.Select(x => null);
end;

{ TopKCategoricalAccuracy }

constructor TopKCategoricalAccuracy.Create(k: Integer; name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.sparse_top_k_categorical_matches( tf.math.argmax(yt, -1), yp, k) ;
                        end, name, dtype);
end;

{ BinaryAccuracy }

constructor BinaryAccuracy.Create(name: string; dtype: TF_DataType; threshold: Single);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.binary_matches(yt, yp) ;
                        end, name, dtype);
end;

{ CategoricalAccuracy }

constructor CategoricalAccuracy.Create(name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.sparse_categorical_matches(tf.math.argmax(yt, -1), yp) ;
                        end, name, dtype);
end;

{ CategoricalCrossentropy }

constructor CategoricalCrossentropy.Create(name: string; dtype: TF_DataType; from_logits: Boolean; label_smoothing: Single; axis: PAxis);
begin
     var a : TAxis := -1;
     if axis <> nil then a := axis^;
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := tf.keras.metrics.categorical_crossentropy(yt, yp, from_logits, label_smoothing, @a) ;
                        end, name, dtype);
end;

{ CosineSimilarity }

constructor CosineSimilarity.Create(name: string; dtype: TF_DataType; axis: PAxis);
begin
     var a : TAxis := -1;
     if axis <> nil then  a := axis^;
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.cosine_similarity(yt, yp, @a) ;
                        end, name, dtype);
end;

{ Accuracy }

constructor Accuracy.Create(name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.accuracy(yt, yp) ;
                        end, name, dtype);
    Fdtype := dtype;
end;

{ HammingLoss }

constructor HammingLoss.Create(mode: string; threshold: TNDArray; name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result := metrics_utils.hamming_loss_fn(yt, yp, threshold, mode) ;
                        end, name, dtype);
end;

{ SparseCategoricalAccuracy }

constructor SparseCategoricalAccuracy.Create(name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result :=  metrics_utils.sparse_categorical_matches(yt, yp) ;
                        end, name, dtype);
end;

{ SparseCategoricalCrossentropy }

constructor SparseCategoricalCrossentropy.Create(name: string; dtype: TF_DataType; from_logits: Boolean; ignore_class: PInteger; axis: PAxis);
begin
     var a : TAxis := -1;
     if axis <> nil then  a := axis^;
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result :=  tf.keras.metrics.sparse_categorical_crossentropy(yt, yp, from_logits, ignore_class, @a) ;
                        end, name, dtype);
end;

{ SparseTopKCategoricalAccuracy }

constructor SparseTopKCategoricalAccuracy.Create(k: Integer; name: string; dtype: TF_DataType);
begin
     inherited Create( function(yt: TFTensor; yp: TFTensor): TFTensor
                        begin
                            Result :=  metrics_utils.sparse_top_k_categorical_matches(yt, yp, k) ;
                        end, name, dtype);
end;

{ Recall }

constructor Recall.Create(thresholds: Single; top_k, class_id: Integer; name: string; dtype: TF_DataType);
begin
    inherited Create(name, dtype);

    Fthresholds      := constant_op.constant( TArray<Single>.Create(thresholds) );
    Ftrue_positives  := add_weight('true_positives',  1, TF_FLOAT, tf.initializers.zeros_initializer);
    Ffalse_negatives := add_weight('false_negatives', 1, TF_FLOAT, tf.initializers.zeros_initializer);
end;

procedure Recall.reset_states;
begin
    var num_thresholds : Integer := Fthresholds.size;

    var l := TList<Tuple<IVariableV1, TNDArray>>.Create;
    l.Add( Tuple.Create(Ftrue_positives, np.zeros(num_thresholds)) );
    l.Add( Tuple.Create(Ffalse_negatives, np.zeros(num_thresholds)) );

    tf.keras.backend.batch_set_value(l);
end;

function Recall.R_result: TFTensor;
begin
    var res := tf.divide(Ftrue_positives.AsTensor, tf.add(Ftrue_positives, Ffalse_negatives));

    if Fthresholds.size = 1 then
       Result := res[0]
    else
       Result := res;
end;

function Recall.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    var d := TDictionary<string, IVariableV1>.Create;
    d.Add('tp', Ftrue_positives);
    d.Add('fn', Ffalse_negatives);
    Result := metrics_utils.update_confusion_matrix_variables(d, y_true, y_pred, Fthresholds, Ftop_k, Fclass_id, sample_weight, False, nil, Fthresholds_distributed_evenly);
end;

{ Precision }

constructor Precision.Create(thresholds: Single; top_k, class_id: Integer; name: string; dtype: TF_DataType);
begin
    inherited Create(name, dtype);

    Fthresholds      := constant_op.constant( TArray<Single>.Create(thresholds) );
    Ftop_k           := top_k;
    Fclass_id        := class_id;
    Ftrue_positives  := add_weight('true_positives',  1, TF_FLOAT, tf.initializers.zeros_initializer);
    Ffalse_positives := add_weight('false_negatives', 1, TF_FLOAT, tf.initializers.zeros_initializer);
end;

procedure Precision.reset_states;
begin
    var num_thresholds : Integer := Fthresholds.size;

    var l := TList<Tuple<IVariableV1, TNDArray>>.Create;
    l.Add( Tuple.Create(Ftrue_positives, np.zeros(num_thresholds)) );
    l.Add( Tuple.Create(Ffalse_positives, np.zeros(num_thresholds)) );

    tf.keras.backend.batch_set_value(l);
end;

function Precision.R_result: TFTensor;
begin
    var res := tf.divide(Ftrue_positives.AsTensor, tf.add(Ftrue_positives, Ffalse_positives));

    if Fthresholds.size = 1 then
       Result := res[0]
    else
       Result := res;
end;

function Precision.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    var d := TDictionary<string, IVariableV1>.Create;
    d.Add('tp', Ftrue_positives);
    d.Add('fp', Ffalse_positives);
    Result := metrics_utils.update_confusion_matrix_variables(d, y_true, y_pred, Fthresholds, Ftop_k, Fclass_id, sample_weight, False, nil, Fthresholds_distributed_evenly);
end;

{ FBetaScore }

constructor FBetaScore.Create(num_classes: Integer; average: string; beta: Single; threshold: PSingle; name: string; dtype: TF_DataType);
begin
    inherited Create(name, dtype);

    Fnum_classes := num_classes;
    Faverage := average;
    Fbeta := constant_op.constant(beta);
    Fdtype := dtype;

    if Assigned(threshold) then
      Fthreshold := constant_op.constant(threshold^);


    Finit_shape := [];

    if average <> 'micro' then
    begin
        Faxis := 0;
        Finit_shape := [ num_classes ];
    end;

    Ftrue_positives  := add_weight('true_positives', Finit_shape, TF_FLOAT, tf.initializers.zeros_initializer);
    Ffalse_positives := add_weight('false_positives', Finit_shape, TF_FLOAT, tf.initializers.zeros_initializer);
    Ffalse_negatives := add_weight('false_negatives', Finit_shape, TF_FLOAT, tf.initializers.zeros_initializer);
    Fweights_intermediate := add_weight('weights_intermediate', Finit_shape, TF_FLOAT, tf.initializers.zeros_initializer);
end;

function FBetaScore._weighted_sum(vVal: TFTensor; sample_weight : TFtensor): TFTensor;
begin
    if sample_weight <> nil then
      vVal := tf.multiply(vVal, tf.expand_dims(sample_weight, 1));

    Result := tf.reduce_sum(vVal, @Faxis);
end;

function FBetaScore.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    if Fthreshold = nil then
    begin
        var assi : TAxis := -1;
        Fthreshold := tf.reduce_max(y_pred, @assi, true);
        // make sure [0, 0, 0] doesn't become [1, 1, 1]
        // Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred := tf.logical_and(TTensor(y_pred) >= Fthreshold, TTensor(tf.abs(y_pred)) > Single(1e-12));
    end else
    begin
        y_pred := TTensor(y_pred) > Fthreshold;
    end;

    y_true := tf.cast(y_true, Fdtype);
    y_pred := tf.cast(y_pred, Fdtype);

    AssignAdd(Ftrue_positives, _weighted_sum(TTEnsor(y_pred) * y_true, sample_weight));
    AssignAdd(Ffalse_positives,_weighted_sum(TTensor(y_pred) * (Integer(1) - TTensor(y_true)), sample_weight));
    AssignAdd(Ffalse_negatives,_weighted_sum((Integer(1) - TTensor(y_pred)) * TTensor(y_true), sample_weight));
    AssignAdd(Fweights_intermediate,_weighted_sum(y_true, sample_weight));

    Result := Fweights_intermediate.AsTensor;
end;

function FBetaScore.R_result: TFTensor;
begin
    var precision : TTensor := tf.math.divide_no_nan(Ftrue_positives.AsTensor, TTensor(Ftrue_positives.AsTensor) + Ffalse_positives.AsTensor);
    var recall    : TTensor := tf.math.divide_no_nan(Ftrue_positives.AsTensor, TTensor(Ftrue_positives.AsTensor) + Ffalse_negatives.AsTensor);

    var mul_value := precision * recall;
    var add_value := (tf.math.square(Fbeta) * precision) + recall;
    var mean      : TTensor := tf.math.divide_no_nan(mul_value, add_value);
    var f1_score := mean * (Integer(1) + TTensor(tf.math.square(Fbeta)));

    var weights : TFTensor ;
    if Faverage = 'weighted' then
    begin
        weights  := tf.math.divide_no_nan(Fweights_intermediate.AsTensor, tf.reduce_sum(Fweights_intermediate.AsTensor));
        f1_score := tf.reduce_sum(f1_score * weights);
    end
    // micro, macro
    else if Faverage <> '' then
    begin
        f1_score := tf.reduce_mean(f1_score);
    end;

    Result := f1_score;
end;

procedure FBetaScore.reset_states;
begin
    var reset_value := np.zeros(Finit_shape, Fdtype);

    var l := TList<Tuple<IVariableV1, TNDArray>>.Create;
    l.Add( Tuple.Create(Ftrue_positives, reset_value) );
    l.Add( Tuple.Create(Ffalse_positives, reset_value) );
    l.Add( Tuple.Create(Ffalse_negatives, reset_value) );
    l.Add( Tuple.Create(Fweights_intermediate, reset_value) );

    tf.keras.backend.batch_set_value(l);
end;

{ F1Score }

constructor F1Score.Create(num_classes: Integer; average: string; threshold: PSingle; name: string; dtype: TF_DataType);
begin
    inherited Create(num_classes, average, 1, threshold, name, dtype)
end;

end.



