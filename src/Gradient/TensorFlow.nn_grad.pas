unit TensorFlow.nn_grad;
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
         System.Generics.Collections,
         Generics.Defaults,
         System.Math,

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         Tensorflow.Gradient;

type
  nn_grad = class
      private
        FGradFunction     : TArray<TGradFunc>;
      public
        constructor Create;
        destructor Destroy;  override;

        property GradFunction  : TArray<TGradFunc> read FGradFunction;
  end;

implementation
        uses Tensorflow,
             TensorFlow.Constant_op,
             Tensorflow.Utils,
             TensorFlow.Ops,
             TensorFlow.Tensor,
             Tensorflow.math_ops,
             Tensorflow.gen_array_ops,
             TensorFlow.gen_math_ops,
             Tensorflow.array_ops,
             TensorFlow.gen_nn_ops,
             TensorFlow.nn_ops,

             TensorFlow.NnOps,

             Numpy,
             NumPy.NDArray;

function  IsZero(g: TFTensor): Boolean;
var
  aZerosNames : TArray<string>;
begin
    aZerosNames := ['ZerosLike', 'Zeros'];
    if TArray.contains(aZerosNames, g.op.tipo) then
        Exit(True);

     raise Exception.Create( 'IsZero');
end;

function _BroadcastMul(vec: TFTensor; mat: TFTensor): TFTensor;
begin
    vec    := array_ops.expand_dims(vec, -1);
    Result := vec * TTensor(mat);
end;

/// <summary>
/// Return the gradients for the 2 inputs of bias_op.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
function _BiasAddGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  grad         : TFTensor;
  data_format  : string;
  bias_add_grad: TFTensor;
begin
    grad := grads[0];
    data_format := op.get_attr('data_format').ToString;
    bias_add_grad := gen_nn_ops.bias_add_grad(grad, data_format);
    Result        := [grad, bias_add_grad];
end;

function _ReluGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [gen_nn_ops.relu_grad(grads[0], op.outputs[0])];
end;

function _LeakyReluGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  grad : TFTensor;
  x    : TFTensor;
  alpha: Single;
begin
    grad  := grads[0];
    x     := op.inputs[0];
    alpha := op.get_attr<Single>('alpha');
    Result := [gen_nn_ops.leaky_relu_grad(grad, x, alpha)];
end;

/// <summary>
/// The derivative of the softmax nonlinearity.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
function _SoftmaxGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  grad_softmax : TFTensor;
  softmax      : TFTensor;
  mul          : TFTensor;
  sum_channels : TFTensor;
  sub          : TFTensor;
begin
    grad_softmax := grads[0];
    softmax      := op.outputs[0];
    mul          := grad_softmax * TTensor(softmax);
    sum_channels := math_ops.reduce_sum(mul, constant_op.constant(-1), true);
    sub := grad_softmax - TTensor(sum_channels);
    Result := [sub * TTensor(softmax)];
end;

/// <summary>
/// Gradient function for SoftmaxCrossEntropyWithLogits.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
function _SoftmaxCrossEntropyWithLogitsGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  grad_loss, grad_grad, softmax_grad, grad, logits: TFTensor;
begin
    grad_loss := grads[0];
    grad_grad := grads[1];
    softmax_grad := op.outputs[1];
    grad := _BroadcastMul(grad_loss, softmax_grad);

    logits := op.inputs[0];
    if (grad_grad <> nil) and (not IsZero(grad_grad)) then
    raise Exception.Create('_SoftmaxCrossEntropyWithLogitsGrad');

    Result := TArray<TFTensor>.Create(grad, _BroadcastMul(grad_loss, -TTensor(nn_ops.log_softmax(logits))) );
end;

function _SparseSoftmaxCrossEntropyWithLogitsGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  sparse_softmax_grad_without_gradient, grad_0: TFTensor;
begin
    sparse_softmax_grad_without_gradient := array_ops.prevent_gradient(op.outputs[1], 'Currently there is no way to take the second '+
                                                                                      'derivative of sparse_softmax_cross_entropy_with_logits '+
                                                                                      'due to the fused implementation''s interaction with '+
                                                                                      'tf.gradients()');

    grad_0 := grads[0];

    Result := TArray<TFTensor>.Create(_BroadcastMul(grad_0, sparse_softmax_grad_without_gradient), nil) ;
end;

function _SquaredDifferenceGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
var
  x, y  : TFTensor;
  scale : TFTensor;
  x_grad: TFTensor;
begin
    x := op.inputs[0];
    y := op.inputs[1];

    scale  := Tops.convert_to_tensor(Single(2.0), x.dtype);
    x_grad := math_ops.scalar_mul(scale, grads[0]) * (x - TTensor(y));
    Result := [x_grad, -TTensor(x_grad)];
end;

/// <summary>
/// The derivatives for deconvolution.
/// </summary>
/// <param name="op">The Deconvolution op.</param>
/// <param name="grads">The tensor representing the gradient w.r.t. the output</param>
/// <returns>The gradients w.r.t. the input and the filter</returns>
function _Conv2DBackpropInputGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
var
  grad             : TFTensor;
  dilations        : TArray<integer>;
  strides          : TArray<integer>;
  padding          : string;
  explicit_paddings: TArray<integer>;
  use_cudnn_on_gpu : boolean;
  data_format      : string;
  params           : Conv2dParams;
begin
    grad              := grads[0];
    dilations         := op.get_attr_list<integer>('dilations');
    strides           := op.get_attr_list<integer>('strides');
    padding           := op.get_attr<string>('padding');
    explicit_paddings := op.get_attr_list<integer>('explicit_paddings');
    use_cudnn_on_gpu  := op.get_attr<boolean>('use_cudnn_on_gpu');
    data_format       := op.get_attr<string>('data_format');

    params := Conv2dParams.Create;
    params.Input            := grad;
    params.Filter           := op.inputs[1];
    params.Strides          := strides;
    params.Padding          := padding;
    params.DataFormat       := data_format;
    params.Dilations        := dilations;
    params.ExplicitPaddings := explicit_paddings;
    params.UseCudnnOnGpu    := use_cudnn_on_gpu;

    Result := [gen_nn_ops.conv2d_backprop_filter(grad, array_ops.shape(op.inputs[1]), op.inputs[2], strides, padding, use_cudnn_on_gpu, explicit_paddings, data_format, dilations ) ,
               gen_nn_ops.conv2d(params) ];
end;

/// <summary>
/// Gradient function for Conv2D.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
function _Conv2DGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
var
  dilations        : TArray<integer>;
  strides          : TArray<integer>;
  padding          : string;
  explicit_paddings: TArray<integer>;
  use_cudnn_on_gpu : boolean;
  data_format      : string;
begin
    dilations         := op.get_attr_list<Integer>('dilations');
    strides           := op.get_attr_list<Integer>('strides');
    padding           := op.get_attr<string>('padding');
    explicit_paddings := op.get_attr_list<Integer>('explicit_paddings');
    use_cudnn_on_gpu  := op.get_attr<Boolean>('use_cudnn_on_gpu');
    data_format       := op.get_attr<string>('data_format');
    var shape         := gen_array_ops.shape_n([ op.inputs[0], op.inputs[1] ]);

     Result := [ gen_nn_ops.conv2d_backprop_input(shape[0], op.inputs[1],  grads[0], strides, padding, use_cudnn_on_gpu, explicit_paddings, data_format, dilations),
                 gen_nn_ops.conv2d_backprop_filter(op.inputs[0], shape[1], grads[0], strides, padding, use_cudnn_on_gpu, explicit_paddings, data_format, dilations)];
end;

function _BaseFusedBatchNormGrad(op: TFOperation; version: integer; grads: TArray<TFTensor>): TArray<TFTensor>;
var
  x            : TFTensor;
  grad_y       : TFTensor;
  scale        : TFTensor;
  epsilon      : single;
  data_format  : string;
  is_training  : boolean;
  grad_fun     : TFunc<FusedBatchNormParams, TArray<TFTensor>>;
  params       : FusedBatchNormParams;
  pop_mean     : TFTensor;
  pop_var      : TFTensor;
  results      : TArray<TFTensor>;
  dx           : TFTensor;
  dscale       : TFTensor;
  doffset      : TFTensor;
begin
    x           := op.inputs[0];
    grad_y      := grads[0];
    scale       := op.inputs[1];
    epsilon     := op.get_attr<single>('epsilon');
    data_format := op.get_attr<string>('data_format');
    is_training := op.get_attr<boolean>('is_training');
    grad_fun    := nil;

    case version of
      2: grad_fun := gen_nn_ops.fused_batch_norm_grad_v3;
      1: raise Exception.Create('Not implemented');
    else
      grad_fun := gen_nn_ops.fused_batch_norm_grad;
    end;

    if is_training then
    begin
        params := FusedBatchNormParams.Create;

        params.YBackprop     := grad_y;
        params.X             := x;
        params.Scale         := scale;
        params.ReserveSpace1 := op.outputs[3];
        params.ReserveSpace2 := op.outputs[4];
        params.ReserveSpace3 := nil;
        if version = 2  then  params.ReserveSpace3 := op.outputs[5];
        params.Epsilon       := epsilon;
        params.DataFormat    := data_format;
        params.IsTraining    := is_training;

        Result := grad_fun(params);
    end else
    begin
        pop_mean := op.inputs[3];
        pop_var := op.inputs[4];
        if data_format = 'NCHW' then
           raise Exception.Create('Not implemented');

        params := FusedBatchNormParams.Create;

        params.YBackprop     := grad_y;
        params.X             := x;
        params.Scale         := scale;
        params.ReserveSpace1 := pop_mean;
        params.ReserveSpace2 := pop_var;
        params.ReserveSpace3 := nil;
        if version = 2  then  params.ReserveSpace3 := op.outputs[5];
        params.Epsilon       := epsilon;
        params.DataFormat    := data_format;
        params.IsTraining    := is_training;

        results              := grad_fun(params);

        dx      := results[0];
        dscale  := results[1];
        doffset := results[2];
        if data_format = 'NCHW' then
          raise Exception.Create('Not implemented');

        Result := TArray<TFTensor>.Create(dx, dscale, doffset, nil, nil);
    end;
end;

function _FusedBatchNormGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    Result := _BaseFusedBatchNormGrad(op, 0, grads);
end;

function _FusedBatchNormV2Grad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    Result :=_BaseFusedBatchNormGrad(op, 1, grads);
end;

function _FusedBatchNormV3Grad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    Result :=_BaseFusedBatchNormGrad(op, 2, grads);
end;

function _BatchNormWithGlobalNormalizationGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    raise Exception.Create('BatchNormWithGlobalNormalization');
end ;

function _MaxPoolGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    var grad := grads[0];
    Result := [ gen_nn_ops.max_pool_grad(op.inputs[0], op.outputs[0],
                                         grad,
                                         op.get_attr_list<Integer>('ksize'),
                                         op.get_attr_list<Integer>('strides'),
                                         op.get_attr('padding').ToString,
                                         op.get_attr('data_format').ToString) ]
end;

/// <summary>
/// Return the gradients for TopK.
/// </summary>
/// <param name="op"></param>
/// <param name="grads"></param>
/// <returns></returns>
function _TopKGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
var
  grad                             : TFTensor;
  in_shape, ind_shape, cast, size,
  ind_lastdim, stack, ind_2d,
  in_lastdim, outerdim, cast1,
  range2, dim2, cast2, ind, scatter: TFTensor;
begin
    grad      := grads[0];
    in_shape  := array_ops.shape(op.inputs[0]);
    ind_shape := array_ops.shape(op.outputs[1]);

    // int32 is not supported on GPU hence up-casting
    cast        := math_ops.cast(ind_shape, TF_DataType.TF_INT64);
    size        := TTensor(array_ops.size(ind_shape)) - 1;
    ind_lastdim := array_ops.gather(cast, size);

    // Flatten indices to 2D.
    var vValue : TArray<TValue> := [Int64(-1), ind_lastdim];
    stack      := array_ops.stack(TValue.From< TArray<TValue> >(vValue));
    ind_2d     := array_ops.reshape(op.outputs[1], stack);

    in_lastdim := array_ops.gather(math_ops.cast(in_shape, TF_DataType.TF_INT64), TTensor(array_ops.size(in_shape)) - 1);
    outerdim   := array_ops.shape(ind_2d)._slice(0);

    // Compute linear indices(flattened to 1D).
    cast1  := math_ops.cast(outerdim, TF_DataType.TF_INT64);

    var vDelta : TValue  := in_lastdim;
    var tLimit : TFTensor:= TTensor(cast1) * in_lastdim;
    var vLimit : TValue  := tLimit;
    range2 := math_ops.range(tf.constant(Int64(0)), @vLimit, @vDelta);

    dim2   := array_ops.expand_dims(range2, -1);
    cast2  := math_ops.cast(dim2, TF_DataType.TF_INT32);
    ind    := array_ops.reshape(ind_2d + TTensor(cast2), TArray<Integer>.Create( -1 ));

    // Substitute grad to appropriate locations and fill the rest with zeros,
    // finally reshaping it to the original input shape.
    scatter := gen_array_ops.scatter_nd(array_ops.expand_dims(ind, -1),
                                        array_ops.reshape(grad, TArray<Integer>.Create( -1 )), [ math_ops.reduce_prod(in_shape) ]);

    var aR : TArray<Integer> := [];
    Result := [ array_ops.reshape(scatter, in_shape), array_ops.zeros(aR, TF_DataType.TF_INT32)];
end;

{ nn_grad }

constructor nn_grad.Create;
begin
    FGradFunction := [ TGradFunc.Create('BiasAdd',                             _BiasAddGrad),
                       TGradFunc.Create('Relu',                                _ReluGrad),
                       TGradFunc.Create('LeakyRelu',                           _LeakyReluGrad),
                       TGradFunc.Create('Softmax',                             _SoftmaxGrad),
                       TGradFunc.Create('SoftmaxCrossEntropyWithLogits',       _SoftmaxCrossEntropyWithLogitsGrad),
                       TGradFunc.Create('SparseSoftmaxCrossEntropyWithLogits', _SparseSoftmaxCrossEntropyWithLogitsGrad),
                       TGradFunc.Create('SquaredDifference',                   _SquaredDifferenceGrad),
                       TGradFunc.Create('Conv2DBackpropInput',                 _Conv2DBackpropInputGrad ),
                       TGradFunc.Create('Conv2D',                              _Conv2DGrad),
                       TGradFunc.Create('FusedBatchNorm',                      _FusedBatchNormGrad),
                       TGradFunc.Create('FusedBatchNormV2',                    _FusedBatchNormV2Grad),
                       TGradFunc.Create('FusedBatchNormV3',                    _FusedBatchNormV3Grad),
                       TGradFunc.Create('BatchNormWithGlobalNormalization',    _BatchNormWithGlobalNormalizationGrad),
                       TGradFunc.Create('MaxPool',                             _MaxPoolGrad),
                       TGradFunc.Create('TopK',                                _TopKGrad) ] ;
end;

destructor nn_grad.Destroy;
begin
    FGradFunction := [];
end;

end.
