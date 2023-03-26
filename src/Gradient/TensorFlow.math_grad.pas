unit TensorFlow.math_grad;
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
         TensorFlow.Core,
         TensorFlow.DApiBase,
         Tensorflow.Gradient;

type
    math_grad = class
      private
        FGradFunction     : TArray<TGradFunc>;
      public
        constructor Create;
        destructor Destroy;  override;

        property GradFunction  : TArray<TGradFunc> read FGradFunction;
    end;
    function _ShapesFullySpecifiedAndEqual(x: TFTensor; y: TFTensor; grad: TFTensor): Boolean;
    function SmartBroadcastGradientArgs(x: TFTensor; y: TFTensor; grad: TFTensor): TArray< Tuple<TFTensor, TFTensor, boolean>>;

implementation
      uses Tensorflow,
           Tensorflow.Utils,
           TensorFlow.Ops,
           TensorFlow.Tensor,
           TensorFlow.Operations,

           Numpy,
           NumPy.NDArray;



function _ShapesFullySpecifiedAndEqual(x: TFTensor; y: TFTensor; grad: TFTensor): Boolean;
begin
    var x_shape := x._shape_tuple;
    var y_shape := y._shape_tuple;
    var grad_shape := grad._shape_tuple;
    Result := (Length(x_shape) > 0) and (Length(y_shape) > 0);
    Result := (Result) and (TUtils.SequenceEqual<Integer>(x_shape, y_shape)) and (TUtils.SequenceEqual<Integer>(y_shape, grad_shape));
    Result := (Result) and (not TArray.Contains<Integer>(x_shape, - 1) )
end;

/// <summary>
/// Optimized version of `broadcast_gradient_args` that caches results.
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
function SmartBroadcastGradientArgs(x: TFTensor; y: TFTensor; grad: TFTensor): TArray< Tuple<TFTensor, TFTensor, boolean>>;
var
   sx, sy: TFTensor;
begin
    if (x.shape.IsFullyDefined) and (y.shape.IsFullyDefined) then
    begin
        sx := array_ops.shape(x);
        sy := array_ops.shape(y);
    end else
    begin
        sx := array_ops.shape_internal(x, '', false);
        sy := array_ops.shape_internal(y, '', false);
    end;
    var rx_ry := gen_array_ops.broadcast_gradient_args(sx, sy);
    var rx    := rx_ry.Value1;
    var ry    := rx_ry.Value2;
    Result := [
               Tuple<TFTensor, TFTensor, boolean>.Create(sx,rx, not x.shape.Equals(TValue.From<TFShape>(grad.shape)) ),
               Tuple<TFTensor, TFTensor, boolean>.Create(sy,ry, not y.shape.Equals(TValue.From<TFShape>(grad.shape)) )
              ];
    
end;

// [RegisterGradient('Abs')]
function _AbsGrad(op: TFOperation; grads: TArray<TFTensor>) : TArray<TFTensor>;
begin
    var x    := op.inputs[0];
    var grad := grads[0];
    Result := [ TTensor(grad) * math_ops.sign(x) ];
end ;

// [RegisterGradient('Add')]
function _AddGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var x := op.inputs[0];
    var y := op.inputs[1];
    var grad := grads[0];
    if (grad is TFTensor) and (_ShapesFullySpecifiedAndEqual(x, y, grad)) then
    begin
       Result := [ grad, grad ];
       Exit;
    end;
    var sx    := array_ops.shape(x);
    var sy    := array_ops.shape(y);
    var rx_ry := gen_array_ops.broadcast_gradient_args(sx, sy);
    var rx    := rx_ry.Value1;
    var ry    := rx_ry.Value2;
    var sum1 := math_ops.reduce_sum(grad, rx);
    var r1   := gen_array_ops.reshape(sum1, sx);
    var sum2 := math_ops.reduce_sum(grad, ry);
    var r2   := gen_array_ops.reshape(sum2, sy);
    Result := [ r1, r2 ];
end ;

// [RegisterGradient('AddV2')]
function _AddV2Grad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := _AddGrad(op, grads);
end;

/// <summary>
/// Copies the gradient to all inputs.
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('AddN')]
function _AddNGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    for var i := 0 to op.inputs.Count - 1 do
      Result := Result + [ grad ];
end ;

// [RegisterGradient('Cumsum')]
function _CumsumGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad      := grads[0];
    var axis      := op.inputs[1];
    var exclusive := op.get_attr<Boolean>('exclusive');
    var reverse   := op.get_attr<Boolean>('reverse');
    Result := [ math_ops.cumsum(grad, axis,  exclusive, not reverse), nil ]
end;

// [RegisterGradient('DivNoNan')]
function _DivNoNanGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var y    := op.inputs[1];
    var sx   := array_ops.shape(x);
    var sy   := array_ops.shape(y);
    var rx_ry:= gen_array_ops.broadcast_gradient_args(sx, sy);
    var rx   := rx_ry.Value1;
    var ry   := rx_ry.Value2;
    x := math_ops.conj(x);
    y := math_ops.conj(y);
    var reduce_sum1 := math_ops.reduce_sum(math_ops.div_no_nan(grad, y), rx);
    var reduce_sum2 := math_ops.reduce_sum(TTensor(grad) * math_ops.div_no_nan( math_ops.div_no_nan(-(TTensor(x)), y), y ), ry);
    Result := [array_ops.reshape(reduce_sum1, sx), array_ops.reshape(reduce_sum2, sy)];
end;

/// <summary>
/// Returns grad * exp(x).
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('Exp')]
function _ExpGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var y    := op.outputs[0];  // y = e^x
    var adeps : TArray<TValue> := [ grad ];
    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        y := math_ops.conj(y);
                        // forward_compatible(2019, 9, 14)
                        // return new Tensor[] { math_ops.mul_no_nan(y, grad) };
                        Result := [ TTensor(grad) * y ];
                    end );
end;

// [RegisterGradient("Rsqrt")]
function _RsqrtGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var y    := op.outputs[0];
    var adeps : TArray<TValue> := [ grad ];
    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        y := math_ops.conj(y);
                        var factor := constant_op.constant(Single(-0.5), y.dtype,'Const') ;

                        Result := [ TTensor(grad) * (factor * TTensor(math_ops.square(y)) * y) ];
                    end );
end;

// [RegisterNoGradient('GreaterEqual')]
function _GreaterEqualGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := nil
end;

// [RegisterNoGradient('OnesLike')]
function _OnesLike(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := nil
end;

// [RegisterNoGradient('ZerosLike')]
function _ZerosLike(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
   Result := nil
end;

// [RegisterGradient('Identity')]
function _IdGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ grads[0] ];
end;

// [RegisterGradient('Lgamma')]
function _LgammaGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var adeps : TArray<TValue> := [ grad ];
    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                       x := math_ops.conj(x);
                       Result := [ TTensor(grad) * math_ops.digamma(x) ];
                    end );
end;

// [RegisterGradient('Log')]
function _LogGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var adeps : TArray<TValue> := [ grad ];
    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                       x := math_ops.conj(x);
                       Result := [ TTensor(grad) * math_ops.reciprocal(x) ];
                    end );
end;

// [RegisterGradient('Log1p')]
function _Log1pGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var adeps : TArray<TValue> := [ grad ];
    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                       x := math_ops.conj(x);
                       Result := [ TTensor(grad) * math_ops.reciprocal(1 + TTensor(x)) ];
                    end );
end;

// [RegisterGradient('Mul')]
function _MulGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var x    := op.inputs[0];
    var y    := op.inputs[1];
    var grad := grads[0];
    if (op is EagerOperation) and (TArray.Contains<Int64>((op as EagerOperation).SkipInputIndices,1)) and (y.ndim = 0)  then
    begin
        Result := [ gen_math_ops.mul(grad, math_ops.conj(y)), nil ];
        Exit;
    end;
    if (grad is TFTensor) and (_ShapesFullySpecifiedAndEqual(x, y, grad)) and (TArray.Contains<TF_DataType>( [tf.int32_t, tf.float32_t], grad.dtype) ) then
    begin
        Result := [ gen_math_ops.mul(grad, y), gen_math_ops.mul(grad, x) ];
        Exit;
    end;

    var broads := SmartBroadcastGradientArgs(x, y, grad);

    var sx_rx_must_reduce_x := broads[0];
    var sx            := sx_rx_must_reduce_x.Value1;
    var rx            := sx_rx_must_reduce_x.Value2;
    var must_reduce_x := sx_rx_must_reduce_x.Value3;

    var sy_ry_must_reduce_y := broads[1];
    var sy            := sy_ry_must_reduce_y.Value1;
    var ry            := sy_ry_must_reduce_y.Value2;
    var must_reduce_y := sy_ry_must_reduce_y.Value3;

    x := math_ops.conj(x);
    y := math_ops.conj(y);
    var gx : TFTensor;
    var gy : TFTensor;

    if (op is EagerOperation) and  (TArray.Contains<Int64>( (op as EagerOperation).SkipInputIndices,0  ) ) then
        gx := nil
    else if not must_reduce_x then
        gx := gen_math_ops.mul(grad, y)
    else
        gx := array_ops.reshape( math_ops.reduce_sum(gen_math_ops.mul(grad, y), rx), sx);

    if (op is EagerOperation) and  (TArray.Contains<Int64>( (op as EagerOperation).SkipInputIndices,1  ) ) then
        gy := nil
    else if not must_reduce_y then
        gy := gen_math_ops.mul(x, grad)
    else
        gy := array_ops.reshape( math_ops.reduce_sum(gen_math_ops.mul(x, grad), ry), sy);

    Result := [ gx, gy ];
end;

// [RegisterGradient('MatMul')]
function _MatMulGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var grad_a : TFTensor := nil;
    var grad_b : TFTensor := nil;
    var t_a := op.get_attr<Boolean>('transpose_a');
    var t_b := op.get_attr<Boolean>('transpose_b');
    var a := math_ops.conj(op.inputs[0]);
    var b := math_ops.conj(op.inputs[1]);
    if (not t_a) and (not t_b) then
    begin    //lass function gen_math_ops.mat_mul(a: TFTensor; b: TFTensor; transpose_a : Boolean; transpose_b : Boolean; name : string) : TFTensor;
        grad_a := gen_math_ops.mat_mul(grad, b,   False, true);
        grad_b := gen_math_ops.mat_mul(a,   grad, true);
    end
    else if (not t_a) and (t_b) then
    begin
        grad_a := gen_math_ops.mat_mul(grad, b);
        grad_b := gen_math_ops.mat_mul(grad, a, true);
    end
    else if (t_a) and (not t_b) then
    begin
        grad_a := gen_math_ops.mat_mul(b, grad, False, True);
        grad_b := gen_math_ops.mat_mul(a, grad)
    end
    else if t_a and t_b then
    begin
        grad_a := gen_math_ops.mat_mul(b, grad, true, true);
        grad_b := gen_math_ops.mat_mul(grad, a, true, true);
    end;
    Result := [ grad_a, grad_b ];
end;

// [RegisterGradient('BatchMatMul')]
function _BatchMatMul(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var grad_a : TFTensor := nil;
    var grad_b : TFTensor := nil;
    var t_a := op.get_attr<Boolean>('adj_x');
    var t_b := op.get_attr<Boolean>('adj_y');
    var a := math_ops.conj(op.inputs[0]);
    var b := math_ops.conj(op.inputs[1]);
    if (not t_a) and (not t_b) then
    begin
        grad_a := math_ops.batch_matmul(grad, b,    False, true);
        grad_b := math_ops.batch_matmul(a,    grad, true);
    end
    else if (not t_a) and (t_b)  then
    begin
        grad_a := math_ops.batch_matmul(grad, b);
        grad_b := math_ops.batch_matmul(grad, a, true);
    end
    else if (t_a) and (not t_b) then
    begin
        grad_a := math_ops.batch_matmul(b, grad, False,true);
        grad_b := math_ops.batch_matmul(a, grad );
    end
    else if t_a and t_b then
    begin
        grad_a := math_ops.batch_matmul(b, grad, true, true);
        grad_b := math_ops.batch_matmul(grad, a, true, true);
    end;
    Result := [ grad_a, grad_b ];
end;

function _safe_shape_div(x: TFTensor; y: TFTensor): TFTensor;
begin
    Result := math_ops.floordiv( x, gen_math_ops.maximum(y, 1) );
end;

// [RegisterGradient('Sum')]
function _SumGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var input_0_shape := op.inputs[0]._shape_tuple;
    var input_shape : TFTensor;
    if Length(input_0_shape) > 0 then
    begin
        var axes := TUtils.constant_value(op.inputs[1]);
        if not (axes = nil) then
        begin
            var rank := Length(input_0_shape);
            if TUtils.SequenceEqual<Integer>( TUtils.Range(0, rank).ToArray, axes.ToArray<Integer>) then
            begin
                if tf.Context.executing_eagerly then
                begin
                    // should add ones_rank_cache
                    var new_shape := constant_op.constant(TUtils.Range(0, rank).Select(function(x: Integer): Integer
                                                                                        begin
                                                                                           Result := 1;
                                                                                        end ).ToArray, TF_INT32,'');
                    grad := array_ops.reshape(grad, new_shape);
                end else
                begin
                    var new_shape := TUtils.Range(rank).Select(function(x: Integer): Integer
                                                                begin
                                                                   Result := 1;
                                                                end ).ToArray;
                    grad := array_ops.reshape(grad, new_shape);
                end;
                // If shape is not fully defined (but rank is), we use Shape.
                if not TArray.Contains<Integer>(input_0_shape,-1) then
                    input_shape := constant_op.constant(input_0_shape)
                else
                    input_shape := array_ops.shape(op.inputs[0]);
                Result :=[ gen_array_ops.tile(grad, input_shape), nil ];
                Exit;
            end
            else if ( not TArray.Contains<Integer>(input_0_shape,-1) ) and ( not tf.Context.executing_eagerly ) then
            begin
                axes                       := axes.reshape( TFShape.Create([-1]) );
                var shape_tensor           := tf.constant(op.inputs[0].shape.as_int_list);
                var output_shape_kept_dims := math_ops.reduced_shape(shape_tensor, axes);
                var tile_scaling           := _safe_shape_div(shape_tensor, output_shape_kept_dims);
                grad                       := array_ops.reshape(grad, output_shape_kept_dims);
                Result := [ array_ops.tile(grad, tile_scaling), nil ];
                exit;
            end;
        end;
    end;

    input_shape := array_ops.shape(op.inputs[0]);
    if tf.executing_eagerly then
    begin
        if  not op.get_attr<boolean>('keep_dims') then
        begin
            Tops.colocate_with(input_shape);
            var output_shape_kept_dims := math_ops.reduced_shape(input_shape, op.inputs[1]);
            // var tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims);
            grad := gen_array_ops.reshape(grad, output_shape_kept_dims);
        end;
        Result := [ gen_array_ops.broadcast_to(grad, input_shape), nil ];
        exit;
    end else
    begin
        Tops.colocate_with(input_shape);
        var output_shape_kept_dims := math_ops.reduced_shape(input_shape, op.inputs[1]);
        var tile_scaling := _safe_shape_div(input_shape, output_shape_kept_dims);
        grad := gen_array_ops.reshape(grad, output_shape_kept_dims);
        Result := [ gen_array_ops.tile(grad, tile_scaling), nil ];
        exit;
    end;
end;

// [RegisterGradient('Mean')]
function _MeanGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var sum_grad := _SumGrad(op, grads)[0];
    var input_shape  := op.inputs[0]._shape_tuple;
    var output_shape := op.outputs[0]._shape_tuple;
    var res           : TFTensor;
    var factor_tensor : TFTensor;
    if (tf.executing_eagerly) and (Length(input_shape) > 0) and (Length(output_shape) > 0) then
    begin
        var input_size : NDArray := np.prod<Integer>(input_shape);
        var output_size: NDArray := np.prod<Integer>(output_shape);
        var factor      := Integer(input_size) / Max( Integer(output_size), 1);
        factor_tensor := constant_op.constant(factor, sum_grad.dtype,'');
    end else
    begin
        var input_shape_tensor  := array_ops.shape(op.inputs[0]);
        var output_shape_tensor := array_ops.shape(op.outputs[0]);
        factor_tensor := _safe_shape_div(math_ops.reduce_prod(input_shape_tensor), math_ops.reduce_prod(output_shape_tensor));
    end;
    res := math_ops.truediv(sum_grad, math_ops.cast(factor_tensor, sum_grad.dtype));
    Result := [ res, nil ];
end;

function _MinOrMaxGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var input_shape := array_ops.shape(op.inputs[0]);
    var output_shape_kept_dims := math_ops.reduced_shape(input_shape, op.inputs[1]);
    var y := op.outputs[0];
    y := array_ops.reshape(y, output_shape_kept_dims);
    grad := array_ops.reshape(grad, output_shape_kept_dims);
    // Compute the number of selected (maximum or minimum) elements in each
    // reduction dimension. If there are multiple minimum or maximum elements
    // then the gradient will be divided between them.
    var indicators := math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype);
    var num_selected := array_ops.reshape(math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims);
    Result := [ math_ops.div(indicators, num_selected) * TTensor(grad), nil ];
end;

/// <summary>
/// Gradient for Max.
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('Max')]
function _MaxGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := _MinOrMaxGrad(op, grads);
end;

/// <summary>
/// Gradient for Min.
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('Min')]
function _MinGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := _MinOrMaxGrad(op, grads);
end;

/// <summary>
/// Factor out the code for the gradient of Maximum or Minimum.
/// </summary>
/// <param name='op'></param>
/// <param name='grad'></param>
/// <returns></returns>
function _MaximumMinimumGrad(isMaximum: Boolean; op: TFOperation; grad: TFTensor): TArray<TFTensor>;
begin
    var x := op.inputs[0];
    var y := op.inputs[1];
    var gdtype := grad.dtype;
    var sx := array_ops.shape(x);
    var sy := array_ops.shape(y);
    var gradshape := array_ops.shape(grad);
    var zeros := array_ops.zeros(gradshape, gdtype);
    var xmask : TFTensor;
    if isMaximum then xmask := gen_math_ops.greater_equal(x, y)
    else              xmask := gen_math_ops.less_equal(x, y);

    var rx_ry := gen_array_ops.broadcast_gradient_args(sx, sy);
    var rx := rx_ry.Value1;
    var ry := rx_ry.Value2;

    var xgrad := array_ops.where(xmask, grad, zeros);
    var gx    := array_ops.reshape(math_ops.reduce_sum(xgrad, rx), sx);
    var ygrad := array_ops.where(xmask, zeros, grad);
    var gy    := array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy);
    Result := [ gx, gy ];
end;

/// <summary>
/// Returns grad*(x > y, x &lt;= y) with type of grad.
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('Maximum')]
function _MaximumGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := _MaximumMinimumGrad(true, op, grads[0]);
end;

/// <summary>
/// Returns grad*(x &lt; y, x >= y) with type of grad.
/// </summary>
/// <param name='op'></param>
/// <param name='grads'></param>
/// <returns></returns>
// [RegisterGradient('Minimum')]
function _MinimumGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := _MaximumMinimumGrad(false, op, grads[0]);
end;

// [RegisterGradient('Neg')]
function _NegGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := [ - TTensor( grads[0] )];
end;

// [RegisterGradient('Select')]
function _SelectGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var c := op.inputs[0];
    var x := op.inputs[1];
    var zeros := array_ops.zeros_like(x);
    Result := [ nil,
                array_ops.where(c, grad, zeros),
                array_ops.where(c, zeros, grad)
               ];
end;

// [RegisterGradient('Sub')]
function _SubGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x := op.inputs[0];
    var y := op.inputs[1];
    if (grad is TFTensor) and (_ShapesFullySpecifiedAndEqual(x, y, grad) ) then
    begin
        Result := [ grad, - TTensor(grad) ];
        Exit;
    end;
    var broads := SmartBroadcastGradientArgs(x, y, grad);
    var sx            := broads[0].Value1;
    var rx            := broads[0].Value2;
    //var must_reduce_x := broads[0].Value3;

    var sy            := broads[1].Value1;
    var ry            := broads[1].Value2;
    //var must_reduce_y := broads[1].Value3;

    var gx := array_ops.reshape(math_ops.reduce_sum(grad, rx), sx);
    var gy := array_ops.reshape(math_ops.reduce_sum(- TTensor(grad), ry), sy);
    Result := [ gx, gy ];
end;

// [RegisterGradient('RealDiv')]
function _RealDivGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x  := op.inputs[0];
    var y  := op.inputs[1];
    var sx := array_ops.shape(x);
    var sy := array_ops.shape(y);
    var rx_ry := gen_array_ops.broadcast_gradient_args(sx, sy);
    var rx := rx_ry.Value1;
    var ry := rx_ry.Value2;
    x := math_ops.conj(x);
    y := math_ops.conj(y);
    var reshape1 := array_ops.reshape(
        math_ops.reduce_sum(
            math_ops.realdiv(grad, y), rx),
        sx);
    var reshape2 := array_ops.reshape(
        math_ops.reduce_sum(
            TTensor(grad) * math_ops.realdiv(math_ops.realdiv(-TTensor(x), y), y), ry),
        sy);
    Result :=  [ reshape1, reshape2 ];
end;

// [RegisterGradient('Sigmoid')]
function _SigmoidGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var y    := op.outputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                       y      := math_ops.conj(y);
                       Result := [ gen_math_ops.sigmoid_grad(y, grad) ];
                    end );
end;

// [RegisterGradient('Sign')]
function _SignGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var x    := op.inputs[0];
    var s := x.shape;
    var zero := constant_op.constant(0.0, x.dtype, @s);
    Result   := [ zero ];
end;

// [RegisterGradient('Square')]
function _SquareGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x     := math_ops.conj(x);
                        var y := constant_op.constant(2.0, x.dtype,'');
                        Result := [ math_ops.multiply(grad, math_ops.multiply(x, y)) ];
                    end );
end;

// [RegisterGradient('Sqrt')]
function _SqrtGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var y    := op.outputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        y          := math_ops.conj(y);
                        var factor := constant_op.constant(0.5, y.dtype,'');
                        Result := [ TTensor(grad) * ( TTensor(factor) * math_ops.reciprocal(y) ) ];
                    end );
end;

// [RegisterGradient('Asin')]
function _ASinGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x := math_ops.conj(x);
                        // the derivative of
                        // y = asin(x)
                        // is
                        // d/dx asin(x) = 1 / sqrt(1-x*x)
                        Result := [ math_ops.multiply( grad, 1 /  TTensor(gen_math_ops.sqrt(1 - TTensor(gen_math_ops.square(x)) )) ) ];
                    end );
end;

// [RegisterGradient('Sin')]
function _SinGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply(grad, gen_math_ops.cos(x)) ];
                    end );
end;

// [RegisterGradient('Sinh')]
function _SinhGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply(grad, gen_math_ops.cosh(x)) ];
                    end );
 end;

// [RegisterGradient('Acos')]
function _ACosGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        // the derivative of
                        // y = acos(x)
                        // is
                        // d/dx acos(x) = -1 / sqrt(1-x*x) = -d/dx asin(x)
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply( grad, -1 / TTensor(gen_math_ops.sqrt(1 - TTensor(gen_math_ops.square(x)))) ) ];
                    end );
end;

// [RegisterGradient('Cast')]
function _CastGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var src_type := TDtypes.as_base_dtype(x.dtype);
    var dst_type := TDtypes.as_base_dtype(grad.dtype);

    if (TDtypes.is_value_dtype(src_type)) and (TDtypes.is_value_dtype(dst_type)) then Result := [ math_ops.cast(grad, src_type) ]
    else                                                                              Result :=[];
end;

// [RegisterGradient('Cos')]
function _CosGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply(grad, - TTensor(gen_math_ops.sin(x))) ];
                    end );
end;

// [RegisterGradient('Cosh')]
function _CoshGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply(grad, gen_math_ops.sinh(x)) ];
                    end );
end;

// [RegisterGradient('Atan')]
function _ATanGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        // the derivative of
                        // y = atan(x)
                        // is
                        // d/dx atan(x) = 1 / (1 + x*x)
                        x := math_ops.conj(x);
                        Result := [ math_ops.multiply(grad, 1 / (1 + TTensor(gen_math_ops.square(x)))) ];
                    end );
end;

// [RegisterGradient('Tanh')]
function _TanhGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var y    := op.outputs[0];

    var adeps : TArray<TValue> ;
    for var i := 0 to Length(grads)-1 do
       adeps := adeps + [ grads[i] ] ;

    Result := TUtils.tf_with<TControlDependenciesController,TArray<TFTensor>>( Tops.control_dependencies(adeps),
                  function(v1: TControlDependenciesController): TArray<TFTensor>
                    begin
                        y := math_ops.conj(y);
                        Result := [ gen_math_ops.tanh_grad(y, grad) ];
                    end );
end;

// [RegisterGradient('Pow')]
function _PowGrad(op: TFOperation; grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    var grad := grads[0];
    var x    := op.inputs[0];
    var y    := op.inputs[1];
    if (op is EagerOperation) and  (TArray.Contains<Int64>( (op as EagerOperation).SkipInputIndices,1 ) ) and (y.ndim = 0) then
    begin
        x := math_ops.conj(x);
        y := math_ops.conj(y);

        Result := [TTensor(grad) * y * math_ops.pow(x, TTensor(y) - 1),
                   nil];
        Exit;
    end;
    var z := op.outputs[0];
    var broads := SmartBroadcastGradientArgs(x, y, grad);
    var sx            := broads[0].Value1;
    var rx            := broads[0].Value2;
    //var must_reduce_x := broads[0].Value3;

    var sy            := broads[1].Value1;
    var ry            := broads[1].Value2;
    //var must_reduce_y := broads[1].Value3;

    x := math_ops.conj(x);
    y := math_ops.conj(y);
    z := math_ops.conj(z);

    var mul        := TTEnsor(grad) * y * math_ops.pow(x, TTensor(y) - 1.0);
    var reduce_sum := math_ops.reduce_sum(mul, rx);
    var gx         := gen_array_ops.reshape(reduce_sum, sx);
    // Avoid false singularity at x = 0
    var mask : TFTensor;
    if TDtypes.is_complex(x.dtype) then  raise TFException.Create('x.dtype.is_complex()')
    else                                 mask := TTensor(x) > 0.0;
    var ones   := array_ops.ones_like(x);
    var safe_x := array_ops.where(mask, x, ones);
    var x1     := gen_array_ops.log(safe_x);
    var y1     := array_ops.zeros_like(x);
    var log_x  := array_ops.where(mask, x1, y1);
    var mul1   := TTensor(grad) * z * log_x;

    var reduce_sum1 := math_ops.reduce_sum(mul1, ry);
    var gy          := gen_array_ops.reshape(reduce_sum1, sy);
    Result := [ gx, gy ];
end;

constructor math_grad.Create;
begin
    FGradFunction := [ TGradFunc.Create('Abs',         _AbsGrad),
                       TGradFunc.Create('Add',         _AddGrad),
                       TGradFunc.Create('AddV2',       _AddV2Grad),
                       TGradFunc.Create('AddN',        _AddNGrad),
                       TGradFunc.Create('Cumsum',      _CumsumGrad),
                       TGradFunc.Create('DivNoNan',    _DivNoNanGrad),
                       TGradFunc.Create('Exp',         _ExpGrad),
                       TGradFunc.Create('Rsqrt',       _RsqrtGrad),
                       TGradFunc.Create('GreaterEqual',_GreaterEqualGrad ),
                       TGradFunc.Create('OnesLike',    _OnesLike),
                       TGradFunc.Create('ZerosLike',   _ZerosLike),
                       TGradFunc.Create('Identity',    _IdGrad),
                       TGradFunc.Create('Lgamma',      _LgammaGrad),
                       TGradFunc.Create('Log',         _LogGrad),
                       TGradFunc.Create('Log1p',       _Log1pGrad),
                       TGradFunc.Create('Mul',         _MulGrad),
                       TGradFunc.Create('MatMul',      _MatMulGrad),
                       TGradFunc.Create('BatchMatMul', _BatchMatMul),
                       TGradFunc.Create('Mean',        _MeanGrad),
                       TGradFunc.Create('Max',         _MaxGrad),
                       TGradFunc.Create('Min',         _MinGrad),
                       TGradFunc.Create('Maximum',     _MaximumGrad),
                       TGradFunc.Create('Minimum',     _MinimumGrad),
                       TGradFunc.Create('Neg',         _NegGrad),
                       TGradFunc.Create('Sub',         _SubGrad),
                       TGradFunc.Create('Sum',         _SumGrad),
                       TGradFunc.Create('RealDiv',     _RealDivGrad),
                       TGradFunc.Create('Sigmoid',     _SigmoidGrad),
                       TGradFunc.Create('Sign',        _SignGrad),
                       TGradFunc.Create('Square',      _SquareGrad),
                       TGradFunc.Create('Sqrt',        _SqrtGrad),
                       TGradFunc.Create('Asin',        _ASinGrad),
                       TGradFunc.Create('Sin',         _SinGrad),
                       TGradFunc.Create('Sinh',        _SinhGrad),
                       TGradFunc.Create('Acos',        _ACosGrad),
                       TGradFunc.Create('Cast',        _CastGrad),
                       TGradFunc.Create('Cos',         _CosGrad),
                       TGradFunc.Create('Cosh',        _CoshGrad),
                       TGradFunc.Create('Atan',        _ATanGrad),
                       TGradFunc.Create('Tanh',        _TanhGrad),
                       TGradFunc.Create('Pow',         _PowGrad)

                     ] ;
end;

destructor math_grad.Destroy;
begin
    FGradFunction := [];
end;

end.
