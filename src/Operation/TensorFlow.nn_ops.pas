unit TensorFlow.nn_ops;
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
{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses System.SysUtils,
         Spring,
         Spring.Collections.Enumerable,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.DApiBase,
         Numpy.Axis,
         TensorFlow.Context,
         TensorFlow.NnOps,
         TensorFlow.Variable,
         Keras.ArgsDefinition;

type
  nn_ops = record
    private
      class function _get_noise_shape(x: TFTensor; noise_shape: TFTensor): TFTensor; static;
      /// <summary>
      /// Flattens logits' outer dimensions and keep its last dimension.
      /// </summary>
      /// <param name="logits"></param>
      /// <returns></returns>
      class function _flatten_outer_dims(logits: TFTensor) : TFTensor; static;
    public
      class function convolution_internal(padding: string; strides: TArray<Integer>; dilation_rate: TArray<Integer>; rank: Integer; name: string = ''; data_format: string = ''):  ConvolutionInternal; static;
      class function l2_loss(t: TFTensor; name: string = ''): TFTensor; static;
      /// <summary>
      /// Adds `bias` to `value`.
      /// </summary>
      /// <param name="value"></param>
      /// <param name="bias"></param>
      /// <param name="data_format"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function bias_add(value: TFTensor; bias: IVariableV1; data_format: string = ''; name: string = ''): TFTensor; static;
      /// <summary>
      /// Computes dropout.
      /// </summary>
      /// <param name="x"></param>
      /// <param name="rate"></param>
      /// <param name="noise_shape"></param>
      /// <param name="seed"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function dropout_v2(x: TFTensor; rate: TFTensor; noise_shape: TFTensor = nil; seed: pInteger = nil; name: string = '') : TFTensor; static;
      class function in_top_k(predictions: TFTensor; targets: TFTensor; k: Integer; name: string = '') : TFTensor; static;
      class function log_softmax(logits: TFTensor; axis: Integer = -1; name:string = ''): TFTensor; static;
      class function softmax(logits: TFTensor; axis: Integer = -1; name: string = ''): TFTensor; static;
      class function leaky_relu(features: TFTensor; alpha: Single = 0.2; name: string = ''): TFTensor; static;
      class function max_pool(value: TFTensor; ksize: TArray<Integer>; strides: TArray<Integer>; padding: string; data_format: string = 'NHWC'; name: string = ''): TFTensor; static;
      class function _softmax(logits: TFTensor; compute_op: TFunc<TFTensor, string, TFTensor>; dim: Integer = -1; name: string = '') : TFTensor; static;
      /// <summary>
      /// Computes sparse softmax cross entropy between `logits` and `labels`.
      /// </summary>
      /// <param name="labels"></param>
      /// <param name="logits"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function sparse_softmax_cross_entropy_with_logits(labels: TFTensor = nil; logits: TFTensor = nil; name: string = '') : TFTensor; static;
      class function softmax_cross_entropy_with_logits_v2_helper(labels: TFTensor; logits: TFTensor; axis : Integer= -1; name : string= ''): TFTensor; static;
  end;

implementation
     uses Tensorflow,
          TensorFlow.Tensor,
          Tensorflow.Utils,
          Tensorflow.NameScope,
          TensorFlow.Ops,
          Tensorflow.math_ops,
          Tensorflow.array_ops,
          TensorFlow.Constant_op,
          TensorFlow.gen_nn_ops,
          TensorFlow.gen_math_ops,
          TensorFlow.random_ops;

{ nn_ops }

class function nn_ops.convolution_internal(padding: string; strides, dilation_rate: TArray<Integer>; rank: Integer; name, data_format: string): ConvolutionInternal;
var
  args: ConvolutionalArgs;
begin
    args := ConvolutionalArgs.Create;

    args.Rank         := rank;
    args.Padding      := padding;
    args.Strides      := strides;
    args.DilationRate := dilation_rate;
    args.DataFormat   := data_format;
    args.Name         := name;

    Result := ConvolutionInternal.Create(args)
end;

class function nn_ops.bias_add(value: TFTensor; bias: IVariableV1; data_format, name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([value, TValue.From<IVariableV1>(bias)]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'BiasAdd', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                 name   := v1.ToString;
                                 Result := gen_nn_ops.bias_add(value, bias, data_format, name);
                            end );
end;

class function nn_ops.dropout_v2(x, rate, noise_shape: TFTensor; seed: pInteger; name: string): TFTensor;
begin
    var newVal : TValue := x;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'dropout', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                                if not Tdtypes.is_floating(x.dtype) then
                                   raise TFException.Create('Not Implemented, x has to be a floating point tensor since it''s going to be scaled. Got a {x.dtype} tensor instead.');
                                var keep_prob := 1 - TTEnsor(rate);
                                var scale     := 1 / keep_prob;
                                var scale_tensor := Tops.convert_to_tensor(scale, x.dtype);
                                var ret          := gen_math_ops.mul(x, scale_tensor);
                                noise_shape := _get_noise_shape(x, noise_shape);
                                // Sample a uniform distribution on [0.0, 1.0) and select values larger than
                                // rate.
                                //
                                // NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
                                // and subtract 1.0.
                                var random_tensor := random_ops.random_uniform(noise_shape, 0,nil, x.dtype ,seed);
                                // NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
                                // float to be selected, hence we use a >= comparison.
                                var keep_mask := TTEnsor(random_tensor) >= rate;
                                ret           := x * scale * math_ops.cast(keep_mask, x.dtype);
                                if  not tf.executing_eagerly then
                                    ret.shape := x.shape;
                                Result := ret;
                            end );
end;

class function nn_ops.in_top_k(predictions, targets: TFTensor; k: Integer; name: string): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'in_top_k', nil),
                          function(v1: TNameScope): TFTensor
                            begin
                                Result := gen_nn_ops.in_top_kv2(predictions, targets, k, name);
                            end );
end;

class function nn_ops.l2_loss(t: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('L2Loss', name, ExecuteOpArgs.Create([ t ])).FirstOrDefault(nil);
end;

class function nn_ops.leaky_relu(features: TFTensor; alpha: Single; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([features, alpha]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'LeakyRelu', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                features := Tops.convert_to_tensor(features, DtInvalid, 'features');
                                if TDTypes.is_integer(features.dtype) then
                                    features := math_ops.cast(features, Tdtypes.cfloat32);
                                Result := gen_nn_ops.leaky_relu(features, alpha, name);
                                //return math_ops.maximum(alpha * features, features, name: name);
                            end );
end;

class function nn_ops.log_softmax(logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
     Result :=  _softmax(logits, gen_nn_ops.log_softmax, axis, name);
end;

class function nn_ops.softmax(logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
    Result := _softmax(logits, gen_nn_ops.softmax, axis, name);
end;


class function nn_ops.max_pool(value: TFTensor; ksize, strides: TArray<Integer>; padding, data_format, name: string): TFTensor;
begin
    var newVal : TValue := value;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MaxPool', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;
                                value := Tops.convert_to_tensor(value, DtInvalid, 'input');
                                Result := gen_nn_ops.max_pool(value,
                                                              ksize,
                                                              strides,
                                                              padding,
                                                              data_format,
                                                              name);
                            end );
end;

class function nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits: TFTensor; axis: Integer; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([logits, labels]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'softmax_cross_entropy_with_logits', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                name   := v1.ToString;

                                var precise_logits := logits;
                                var input_rank     := array_ops.rank(precise_logits);
                                var shape          := logits.shape;
                                if axis <> -1 then
                                   raise TFException.Create('Not Implemented softmax_cross_entropy_with_logits_v2_helper axis <> -1');
                                var input_shape := array_ops.shape(precise_logits);
                                // Make precise_logits and labels into matrices.
                                precise_logits := _flatten_outer_dims(precise_logits);
                                labels         := _flatten_outer_dims(labels);
                                // Do the actual op computation.
                                // The second output tensor contains the gradients.  We use it in
                                // _CrossEntropyGrad() in nn_grad but not here.
                                var tt := gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name);
                                var cost            := tt.Value1;
                                var unused_backprop := tt.Value2;
                                // The output cost shape should be the input minus axis.
                                var output_shape := array_ops.slice(input_shape, TArray<TFTensor>.Create(constant_op.constant(0)),
                                                                                 TArray<TFTensor>.Create( math_ops.subtract(input_rank, 1) ) );
                                cost := array_ops.reshape(cost, output_shape);
                                Result := cost;
                            end );
end;

class function nn_ops.sparse_softmax_cross_entropy_with_logits(labels, logits: TFTensor; name: string): TFTensor;
begin
    var newVal : TValue := TValue.From< TArray<TValue> >([labels, logits]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'SparseSoftmaxCrossEntropyWithLogits', @newVal),
                          function(v1: TNameScope): TFTensor
                            begin
                                labels := tops.convert_to_tensor(labels);
                                logits := tops.convert_to_tensor(logits);
                                var precise_logits : TFTensor;
                                if logits.dtype = TF_HALF then precise_logits := math_ops.cast(logits, tdtypes.Cfloat32)
                                else                           precise_logits := logits;
                                // Store label shape for result later.
                                var labels_static_shape := labels.shape;
                                var labels_shape        := array_ops.shape(labels);
                                (*bool static_shapes_fully_defined = (
                                    labels_static_shape.is_fully_defined() &&
                                        logits.get_shape()[:-1].is_fully_defined());*)
                                // Check if no reshapes are required.
                                if logits.shape.ndim = 2 THEN
                                begin
                                    var tCost := gen_nn_ops.sparse_softmax_cross_entropy_with_logits(precise_logits, labels, name);
                                    var cost := tCost.Value1;
                                    var _    := tCost.Value2;
                                    if logits.dtype = Tdtypes.cfloat16 then
                                        Result := math_ops.cast(cost, Tdtypes.cfloat32)
                                    else
                                        Result := cost;
                                    Exit;
                                end;
                                // Perform a check of the dynamic shapes if the static shapes are not fully
                                // defined.
                                raise TFException.Create('Not Implemented sparse_softmax_cross_entropy_with_logits');
                            end );
end;

class function nn_ops._flatten_outer_dims(logits: TFTensor): TFTensor;
begin
    var rank := array_ops.rank(logits);

    var last_dim_size := array_ops.slice(array_ops.shape(logits), TArray<TFTensor>.Create(math_ops.subtract(rank, 1)),
                                                                  TArray<TFTensor>.Create(constant_op.constant(1) ) );

    var ops    := array_ops.concat([ [-1] , last_dim_size ], 0);
    var output := array_ops.reshape(logits, ops);
    // Set output shape if known.
    if not tf.Context.executing_eagerly then
    begin
        var shape := logits.shape;
        if (not shape.IsNil) and (shape.ndim > 0)  then
        begin
            var product: Int64 := 1;
            var product_valid  := true;
            var eDim := Enumerable<Int64>.create(shape.dims) ;
            for var d in eDim.Take(shape.ndim - 1) do
            begin
                if d = -1 then
                begin
                    product_valid := false;
                    break;
                end else
                begin
                    product := product * d;
                end
            end;
            if product_valid then
            begin
                var output_shape := [ product ];
                raise TFException.Create('Not Implemented _flatten_outer_dims product_valid');
            end;
        end;
    end;
    Result := output;
end;

class function nn_ops._get_noise_shape(x, noise_shape: TFTensor): TFTensor;
begin
    if noise_shape = nil then  Result := array_ops.shape(x)
    else                       Result := noise_shape;
end;

class function nn_ops._softmax(logits: TFTensor; compute_op: TFunc<TFTensor, string, TFTensor>; dim: Integer; name: string): TFTensor;
begin
    logits := Tops.convert_to_tensor(logits);

    var shape := logits.shape;
    var is_last_dim : Boolean := (dim = -1) or (dim = shape.ndim - 1);
    if is_last_dim then
        Exit( compute_op(logits, name) );
    raise TFException.Create('Not Implemented _softmax helper');
end;

end.
