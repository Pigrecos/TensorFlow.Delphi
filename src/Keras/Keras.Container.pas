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

           Keras.Layer,
           Keras.LossFunc,
           Keras.Engine;

type
  /// <summary>
  /// Encapsulates metrics that perform a reduce operation on the values.
  /// </summary>
  Reduce = class(Metric)
    public
      constructor Create(_reduction: string; name: string; dtype : TF_DataType = DtInvalid);

      function    update_state(y_true: TFTensor; y_pred : TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
      function    R_result: TFTensor; override;
  end;

  /// <summary>
  /// Computes the (weighted) mean of the given values.
  /// </summary>
  Mean = class(Reduce)
    public
      constructor Create(name: string = 'mean'; dtype: TF_DataType = TF_FLOAT);
      function    update_state(values: TFTensor; sample_weight : TFTensor= nil): TFTensor; reintroduce; overload;
  end;

  MeanMetricWrapper = class(Mean)
    private
       F_fn : TFunc<TFTensor, TFTensor, TFTensor>;
    public
       constructor Create(fn: TFunc<TFTensor, TFTensor, TFTensor>; name: string; dtype : TF_DataType = TF_FLOAT);
       function    update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor; override;
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
       Fuser_metrics    : TArray<string>;
       Fmetric_names    : TArray<string>;
       Fmetrics         : TArray<Metric>;
       Fmetrics_in_order: TList<Metric> ;

       function GetMetrics: TList<Metric>;
    public
       constructor Create(_metrics: TArray<string>; output_names : TArray<string>= []);
       procedure   update_state(y_true: TFTensor; y_pred: TFTensor; sample_weight:TFTensor = nil);
       procedure   Build(y_true: TFTensor; y_pred: TFTensor);
       procedure   _set_metric_names;
       procedure   _create_ordered_metrics;
       function    _get_metric_objects(_metrics: TArray<string>; y_t: TFTensor; y_p: TFTensor): TArray<Metric>; overload;
       function    _get_metric_object(metric: string; y_t: TFTensor; y_p: TFTensor): Metric; overload;

       property metrics : TList<Metric> read GetMetrics;
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
             TensorFlow.Variable,
             Tensorflow.NameScope,
             TensorFlow.Ops,
             Tensorflow.Utils,
             Tensorflow.math_ops,
             Tensorflow.array_ops,

             Keras.Utils;

{ Reduce }

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

function Reduce.update_state(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
    raise Exception.Create('update_state');
end;

{ Container }

constructor Container.Create(output_names: TArray<String>);
begin
    Foutput_names := output_names;
end;

{ MetricsContainer }

constructor MetricsContainer.Create(_metrics, output_names: TArray<string>);
begin
    inherited Create(output_names);

    Fuser_metrics := _metrics;
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

function MetricsContainer.GetMetrics: TList<Metric>;
begin
    if not Fbuilt then
        Exit( TList<Metric>.Create);

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
    Fmetrics_in_order := TList<Metric>.Create;
    for var m in Fmetrics do
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
            metric_obj := tf.keras.metrics.binary_accuracy
        else if is_sparse_categorical then
            metric_obj := tf.keras.metrics.sparse_categorical_accuracy
        else
            metric_obj := tf.keras.metrics.categorical_accuracy;

        metric := 'accuracy';
    end
    else if (metric = 'mean_absolute_error') or (metric = 'mae')  then
    begin
        metric_obj := tf.keras.metrics.mean_absolute_error;
        metric     := 'mean_absolute_error';
    end
    else if (metric = 'mean_absolute_percentage_error') or (metric = 'mape') then
    begin
        metric_obj := tf.keras.metrics.mean_absolute_percentage_error;
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

function Mean.update_state(values, sample_weight: TFTensor): TFTensor;
var
 tLoss           : Tuple<TFTensor,TFTensor>;
 update_total_op,
 value_sum,
 num_values      : TFTensor;
begin
    if sample_weight <> nil then
    begin
        tLoss := losses_utils.squeeze_or_expand_dimensions(values, sample_weight);
        values       := tLoss.Value1;
        sample_weight:= tLoss.Value2;

        sample_weight := math_ops.cast(sample_weight, values.dtype);
        values        := math_ops.multiply(values, sample_weight);
    end;

    update_total_op := nil;
    value_sum       := math_ops.reduce_sum(values);
    TUtils.tf_with<TControlDependenciesController>(Tops.control_dependencies([ value_sum ]),
            procedure(d : TControlDependenciesController )
              begin
                  if      total is RefVariable          then update_total_op := (total as RefVariable)         .assign_add(value_sum)
                  else if total is BaseResourceVariable then update_total_op := (total as BaseResourceVariable).assign_add(value_sum)
              end);

    // Exit early if the reduction doesn't have a denominator.
    if Freduction = Reduction.SUM then
        Exit( update_total_op );

    // Update `count` for reductions that require a denominator.
    num_values := nil;
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

end.



