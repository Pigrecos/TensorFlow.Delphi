unit Tensorflow.math_ops;
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
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context ;

type
    math_ops = record
     private
       class function _truediv_python3(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
       class function _ReductionDims(x, axis: TFTensor): TFTensor; static;
       class function _may_reduce_to_scalar(keepdims: Boolean; axis: PAxis; _output: TFTensor): TFTensor; overload; static;
       class function _may_reduce_to_scalar(keepdims: Boolean; axis: TFTensor; _output: TFTensor) : TFTensor;  overload; static;
     public
       class function cast(x: TFTensor; dtype: TF_DataType = DtInvalid; name: string = ''): TFTensor;static;
       class function add<Tx, Ty>(x: Tx; y: Ty; name: string = '') : TFTensor; static;
       class function add_v2(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
       /// <summary>
       /// Divide two values using Python 2 semantics. Used for Tensor.__div__.
       /// </summary>
       /// <param name="x">`Tensor` numerator of real numeric type.</param>
       /// <param name="y">`Tensor` denominator of real numeric type.</param>
       /// <param name="name">A name for the operation</param>
       /// <returns>`x / y` returns the quotient of x and y.</returns>
       class function &div(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; static;
       class function truediv(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
       class function multiply(x: TFTensor; y: TFTensor; name: string = ''): TFTensor;static;
       /// <summary>
       /// Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
       /// </summary>
       /// <param name="a"></param>
       /// <param name="b"></param>
       /// <param name="transpose_a">If `True`, `a` is transposed before multiplication.</param>
       /// <param name="transpose_b">If `True`, `b` is transposed before multiplication.</param>
       /// <param name="adjoint_a">If `True`, `a` is conjugated and transposed before multiplication.</param>
       /// <param name="adjoint_b">If `True`, `b` is conjugated and transposed before multiplication.</param>
       /// <param name="a_is_sparse">If `True`, `a` is treated as a sparse matrix.</param>
       /// <param name="b_is_sparse">If `True`, `b` is treated as a sparse matrix.</param>
       /// <param name="name">Name for the operation (optional).</param>
       /// <returns>
       /// A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
       /// the product of the corresponding matrices in `a` and `b`, e.g. if all
       /// transpose or adjoint attributes are `False`:
       /// </returns>
       class function matmul(a: TFTensor; b: TFTensor;
                             transpose_a : Boolean = false; transpose_b : Boolean= false;
                             adjoint_a   : Boolean = false; adjoint_b   : Boolean= false;
                             a_is_sparse : Boolean = false; b_is_sparse : Boolean= false;
                             name: string = ''): TFTensor; overload;static;
       class function matmul(a: TFTensor; b: TFTensor; name: string): TFTensor; overload;static;
       /// <summary>
       /// Returns the complex conjugate of a complex number.
       /// </summary>
       /// <param name="x">`Tensor` to conjugate.  Must have numeric or variant type.</param>
       /// <param name="name">A name for the operation (optional).</param>
       /// <returns>A `Tensor` that is the conjugate of `x` (with the same type).</returns>
       class function conj(x: TFTensor; name: string = ''): TFTensor; static;
       class function equal<Tx, Ty>(x: Tx; y: Ty; name : string= ''): TFTensor; static;
       class function not_equal<Tx, Ty>(x: Tx; y: Ty; name : string= '') : TFTensor; static;
       class function range(start: TValue; limit: TValue; delta: TValue; dtype: Nullable<TF_DataType>; name: string = 'range'): TFTensor; static;
       class function reduce_sum(input_tensor: TFTensor; axis : TFTensor = nil; keepdims: Boolean = false; name: string = ''): TFTensor; static;
       class function pow<Tx, Ty>(x: Tx; y: Ty; name: string = '') : TFTensor; static;
       class function logical_and(x: TFTensor; y: TFTensor; name: string = ''): TFTensor; static;
  end;

implementation
         uses Tensorflow,
              TensorFlow.Ops,
              TensorFlow.gen_math_ops,
              Tensorflow.array_ops,
              Tensorflow.NameScope,
              Tensorflow.Utils,
              TensorFlow.Framework;

{ math_ops }

class function math_ops.&div(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ x ];
    vValues := vValues + [ y ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'div', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                x := Tops.convert_to_tensor(x, DtInvalid, 'x');
                                                y := Tops.convert_to_tensor(y, Tdtypes.as_base_dtype(x.Dtype), 'y');
                                                var x_dtype := Tdtypes.as_base_dtype(x.Dtype);
                                                var y_dtype := Tdtypes.as_base_dtype(y.Dtype);
                                                if x_dtype <> y_dtype then
                                                   raise Exception.Create('x and y must have the same dtype, got {x_dtype} != {y_dtype}');
                                                if Tdtypes.is_floating(x_dtype) then
                                                    Result := gen_math_ops.real_div(x, y, name)
                                                else
                                                    Result := gen_math_ops.floor_div(x, y, name);
                                            end );
end;

class function math_ops.equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.equal(x, y, True, name);
end;

class function math_ops.logical_and(x, y: TFTensor; name: string): TFTensor;
begin
    Result := gen_math_ops.logical_and(x, y, name);
end;

class function math_ops.not_equal<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.not_equal(x, y, name)
end;

class function math_ops.pow<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    var vX := TValue.From<Tx>(x);
    var vY := TValue.From<Ty>(y);
    var newVal : TValue := TValue.From<TArray<TValue>>([vX,vY]);

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Pow', @newVal),
                  function(v1: TNameScope): TFTensor
                    begin
                        name := v1.ToString;

                        var x_tensor := Tops.convert_to_tensor(vX, DtInvalid, 'x');
                        var y_tensor := Tops.convert_to_tensor(vY, Tdtypes.as_base_dtype(x_tensor.dtype), 'y');

                        Result := tf.Context.ExecuteOp('Pow', name, ExecuteOpArgs.Create([x_tensor, y_tensor])).FirstOrDefault(nil)
                    end );
end;

class function math_ops.range(start, limit, delta: TValue; dtype: Nullable<TF_DataType>; name: string): TFTensor;
begin
    if limit.IsEmpty then
    begin
        limit := start;
        start := 0;
    end;
    var dtype1 : TF_DataType;
    if not (dtype = nil) then
        dtype1 := dtype
    else
        dtype1 := TUtils.GetdataType(limit);
    var newVal : TValue := TValue.From<TArray<TValue>>([start, limit,delta]);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Range', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;

                                                var start1 := Tops.convert_to_tensor(start, dtype1, 'start');
                                                var limit1 := Tops.convert_to_tensor(limit, dtype1, 'limit');
                                                var v : TValue;
                                                if delta.isEmpty then v := 1
                                                else                  v := delta;
                                                var delta1 := Tops.convert_to_tensor(v, dtype1, 'delta');
                                                Result := gen_math_ops.range(start1, limit1, delta1, name);
                                            end );
end;

class function math_ops.reduce_sum(input_tensor, axis: TFTensor; keepdims: Boolean; name: string): TFTensor;
begin
    var r  := _ReductionDims(input_tensor, axis);
    var m  := gen_math_ops._sum(input_tensor, r, keepdims, name);
    Result := _may_reduce_to_scalar(keepdims, axis, m);
end;

class function math_ops._may_reduce_to_scalar(keepdims: Boolean; axis: PAxis; _output: TFTensor) : TFTensor;
begin
    Result := nil;
    var dims: TArray<TF_int64_t> := [];
    if ( not common_shapes.has_fully_defined_shape(_output) ) and ( not keepdims) and (axis = nil) and ( _output.shape = TFShape.Create(dims) ) then
            Result := _output;
end;

class function math_ops._may_reduce_to_scalar(keepdims: Boolean; axis: TFTensor; _output: TFTensor) : TFTensor;
begin
    Result := nil;
    var dims: TArray<TF_int64_t> := [];
    if ( not common_shapes.has_fully_defined_shape(_output) ) and ( not keepdims) and (axis = nil) and ( _output.shape = TFShape.Create(dims) ) then
            Result := _output;
end;

class function math_ops._ReductionDims(x, axis: TFTensor): TFTensor;
begin
    if axis <> nil then
    begin
        Result := axis;
    end else
    begin
        var rank := array_ops.rank(x);
        Result := range(0, rank, 1, nil);
    end;
end;


class function math_ops.matmul(a, b: TFTensor; transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse: Boolean;
  name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ a ];
    vValues := vValues + [ b ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'MatMul', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;

                                                if (transpose_a) and (adjoint_a) then
                                                    raise Exception.Create('Only one of transpose_a and adjoint_a can be True.');
                                                if (transpose_b) and (adjoint_b) then
                                                    raise Exception.Create('Only one of transpose_b and adjoint_b can be True.');
                                                if adjoint_a then
                                                begin
                                                    a := conj(a);
                                                    transpose_a := true;
                                                end;
                                                if adjoint_b then
                                                begin
                                                    b := conj(b);
                                                    transpose_b := true;
                                                end;
                                                result := gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
                                            end );
end;

class function math_ops.matmul(a, b: TFTensor; name: string): TFTensor;
begin
    Result := matmul(a,b,False,False,False,False,False,False,name);
end;

class function math_ops.multiply(x, y: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Mul', name, ExecuteOpArgs.Create([x, y])).FirstOrDefault(nil)
end;

class function math_ops.truediv(x, y: TFTensor; name: string): TFTensor;
begin
    Result := _truediv_python3(x, y, name);
end;

class function math_ops._truediv_python3(x, y: TFTensor; name: string): TFTensor;
begin
    var vValues : TArray<TValue>;
    vValues := vValues + [ x ];
    vValues := vValues + [ y ];
    var newVal : TValue := TValue.From<TArray<TValue>>(vValues);;
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'truediv', @newVal),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var x_dtype := Tdtypes.as_base_dtype(x.Dtype);
                                                var y_dtype := Tdtypes.as_base_dtype(y.Dtype);
                                                if x_dtype <> y_dtype then
                                                   raise Exception.Create('x and y must have the same dtype, got {x_dtype} != {y_dtype}');
                                                var dtype : TF_DataType;
                                                case x_dtype of
                                                  TF_DataType.TF_UINT8  : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT8   : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT16  : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_UINT16 : dtype := TF_DataType.TF_FLOAT;
                                                  TF_DataType.TF_INT32  : dtype := TF_DataType.TF_DOUBLE;
                                                  TF_DataType.TF_INT64  : dtype := TF_DataType.TF_DOUBLE;
                                                else
                                                  dtype := x_dtype;
                                                end;
                                                x := cast(x, dtype);
                                                y := cast(y, dtype);
                                                Result := gen_math_ops.real_div(x, y, name);
                                            end );
end;

class function math_ops.add<Tx, Ty>(x: Tx; y: Ty; name: string): TFTensor;
begin
    Result := gen_math_ops.add(x, y, name);
end;

class function math_ops.add_v2(x, y: TFTensor; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('AddV2', name, ExecuteOpArgs.Create([x, y])).FirstOrDefault(nil)
end;

class function math_ops.cast(x: TFTensor; dtype: TF_DataType; name: string): TFTensor;
begin
    var base_type := Tdtypes.as_base_dtype(dtype);
    if base_type = x.dtype then
        Exit(x);

    var vvalue := TValue.From<TFTensor>(x);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Cast', @vvalue),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := string(v1.ToString);
                                                if Tdtypes.as_base_dtype(x.dtype) <> base_type then
                                                    x := gen_math_ops.cast(x, base_type, name);
                                                Result := x;
                                            end );
end;

class function math_ops.conj(x: TFTensor; name: string): TFTensor;
begin
    var dt := x.dtype;
    if (Tdtypes.is_floating(dt)) or (Tdtypes.is_integer(dt))then
        Exit( x );

    var vValues : TValue := x;

    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'Conj', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                Result :=  v1._values.AsType<TFTensor>;
                                            end );
end;

end.
