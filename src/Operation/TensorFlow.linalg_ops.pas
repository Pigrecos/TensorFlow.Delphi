unit TensorFlow.linalg_ops;
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
{$WARN SYMBOL_DEPRECATED OFF}

interface
    uses
      SysUtils,
      System.Math,
      System.Rtti,

      Spring,
      Spring.Collections.Enumerable,

      TF4D.Core.CApi,
      TensorFlow.DApi,
      TensorFlow.DApiBase,
      TensorFlow.Core,
      Numpy.Axis;

type
  linalg_ops = class
    private

    public
      function eye(num_rows: Integer; num_columns: Integer = -1; batch_shape : PTFShape= nil; dtype: TF_DataType = TF_DOUBLE; name: string = ''): TFTensor;
      function matrix_inverse(input: TFTensor; adjoint: Boolean = false; name : string= ''): TFTensor;
      function matrix_solve_ls(matrix: TFTensor; rhs: TFTensor; l2_regularizer: TFTensor = nil; fast: Boolean = true; name: string = ''): TFTensor;
      function norm(tensor: TFTensor; _ord: string = 'euclidean'; axis: PAxis = nil; name: string = ''; keepdims: Boolean = true): TFTensor;
      function _composite_impl(matrix: TFTensor; rhs: TfTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _overdetermined(matrix: TFTensor; rhs: TFTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _underdetermined(matrix: TFTensor; rhs: TFTensor; l2_regularizer : TFTensor = nil): TFTensor;
      function _RegularizedGramianCholesky(matrix: TFTensor; l2_regularizer: TFTensor; first_kind: Boolean): TFTensor;
      function cholesky(input: TFTensor; name: string = '') : TFTensor;
      function cholesky_solve(chol: TFTensor; rhs: TFTensor; name: string = '') : TFTensor;
      function matrix_triangular_solve(matrix: TFTensor; rhs: TFTensor; lower: Boolean = true; adjoint: Boolean = false; name: string = ''): TFTensor;
      function qr(input: TFTensor; full_matrices: Boolean = false; name: string = ''): TFTensors;
  end;


implementation
      uses Tensorflow,
           TensorFlow.Tensor,
           TensorFlow.Ops,
           Tensorflow.Utils,
           Tensorflow.array_ops,
           Tensorflow.math_ops,
           NumPy.NDArray;

{ linalg_ops }

function linalg_ops.cholesky(input: TFTensor; name: string): TFTensor;
begin
     Result := tf.Context.ExecuteOp('Cholesky', name, ExecuteOpArgs.Create([ input])).First;
end;

function linalg_ops.cholesky_solve(chol, rhs: TFTensor; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([chol, rhs]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'cholesky_solve', @vvalue),
                          function(v1: TNameScope): TFTensor
                            begin
                                var y := matrix_triangular_solve(chol, rhs, true, false);
                                var x := matrix_triangular_solve(chol, y,  true,  true);
                                Result := x;
                            end );
end;

function linalg_ops.eye(num_rows, num_columns: Integer; batch_shape: PTFShape; dtype: TF_DataType; name: string): TFTensor;
begin
    var vvalue := TValue.From< TArray<TValue> >([num_rows, num_columns, TValue.From<PTFShape>(batch_shape)]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'eye', @vvalue),
                  function(v1: TNameScope): TFTensor
                    begin
                        if num_columns = -1 then
                            num_columns := num_rows;
                        var is_square: Boolean := num_columns = num_rows;
                        var diag_size := Min(num_rows, num_columns);
                        var sShape : TFShape;
                        if batch_shape = nil  then
                            sShape := TFShape.Create(TArray<Integer>.Create())
                        else
                            sShape := batch_shape^;
                        var batch_shape_tensor := Tops.convert_to_tensor(TValue.From<TFShape>(sShape), tf.int32_t, 'shape');
                        var diag_shape := array_ops.concat([ batch_shape_tensor, tf.constant(TArray<Integer>.Create(diag_size )) ], 0);
                        var shape : TFTensor := nil;
                        if not is_square then
                            shape := array_ops.concat([ batch_shape_tensor, tf.constant(TArray<Integer>.Create( num_rows, num_columns )) ], 0);
                        var diag_ones := array_ops.ones(diag_shape, dtype);
                        if is_square then
                            Exit ( array_ops.matrix_diag(diag_ones) )
                        else begin
                            var zero_matrix := array_ops.zeros(shape, dtype);
                            Result := array_ops.matrix_set_diag(zero_matrix, diag_ones);
                        end;
                    end );
end;

function linalg_ops.matrix_inverse(input: TFTensor; adjoint: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatrixInverse', name, ExecuteOpArgs.Create([ input ])
                                                 .SetAttributes(['adjoint', adjoint ]) ).First;
end;

function linalg_ops.matrix_solve_ls(matrix, rhs, l2_regularizer: TFTensor; fast: Boolean; name: string): TFTensor;
begin
   Result := _composite_impl(matrix, rhs, l2_regularizer);
end;

function linalg_ops.matrix_triangular_solve(matrix, rhs: TFTensor; lower, adjoint: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('MatrixTriangularSolve', name, ExecuteOpArgs.Create([ matrix, rhs ])
                                                 .SetAttributes(['lower',lower,'adjoint', adjoint ]) ).First;
end;

function linalg_ops.norm(tensor: TFTensor; _ord: string; axis: PAxis; name: string; keepdims: Boolean): TFTensor;
begin
    var vvalue : TValue := tensor;
    var is_matrix_norm := (axis <> nil) and (Length(axis^.axis) = 2);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'eye', @vvalue),
                  function(v1: TNameScope): TFTensor
                    begin
                        if is_matrix_norm  then
                           raise TFException.Create('Not Implemented');
                        var res := math_ops.sqrt( math_ops.reduce_sum(TTensor(tensor) * math_ops.conj(tensor), axis^, true) );
                        if not keepdims then
                            res := array_ops.squeeze(res, axis^.axis);
                        Result := res;
                    end );
end;

function linalg_ops.qr(input: TFTensor; full_matrices: Boolean; name: string): TFTensors;
begin
    Result := tf.Context.ExecuteOp('Qr', name, ExecuteOpArgs.Create([ input ])
                                                 .SetAttributes(['full_matrices',full_matrices ]) );
end;

function linalg_ops._composite_impl(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var matrix_shape : TFShape := Enumerable<int64>.Create(matrix.shape.dims).Skip(matrix.shape.ndim - 2).ToArray;
    if matrix_shape.IsFullyDefined then
    begin
        if matrix_shape[-2] >= matrix_shape[-1] then
            Exit( _overdetermined(matrix, rhs, l2_regularizer) )
        else
            Exit( _underdetermined(matrix, rhs, l2_regularizer) );
    end;
    raise TFException.Create('Not Implemented');
end;

function linalg_ops._overdetermined(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var chol := _RegularizedGramianCholesky(matrix, l2_regularizer, true);
    Result   := cholesky_solve(chol, math_ops.matmul(matrix, rhs, False, False, true));
end;

function linalg_ops._RegularizedGramianCholesky(matrix, l2_regularizer: TFTensor; first_kind: Boolean): TFTensor;
begin
    var gramian := math_ops.matmul(matrix, matrix, False, False, first_kind, not first_kind);

    if l2_regularizer <> nil then
    begin
        var matrix_shape := array_ops.shape(matrix);
        var batch_shape  := matrix_shape[':-2'];
        var sShape := batch_shape.shape;
        var small_dim : TFTensor;
        if first_kind then small_dim := matrix_shape[-1]
        else               small_dim := matrix_shape[-2];
        var npy : NDArray := small_dim.numpy;
        var identity      := eye(npy, -1, @sShape, matrix.dtype);
        var small_dim_static : Int64;
        if first_kind then small_dim_static := matrix.shape[-1]
        else               small_dim_static := matrix.shape[-2];
        var a := Enumerable<int64>.Create(matrix.shape.dims).Take(matrix.shape.ndim - 2).ToArray;
        TArray.Concat<Int64>([a, TArray<Int64>.Create(small_dim_static, small_dim_static)] );
        identity.shape := TArray.Concat<Int64>([a, TArray<Int64>.Create(small_dim_static, small_dim_static)] );
        gramian := gramian + (TTensor(l2_regularizer) * identity);
    end;
    Result := cholesky(gramian);
end;

function linalg_ops._underdetermined(matrix, rhs, l2_regularizer: TFTensor): TFTensor;
begin
    var chol := _RegularizedGramianCholesky(matrix, l2_regularizer, false);
    Result := math_ops.matmul(matrix, cholesky_solve(chol, rhs), False, False, true);
end;

end.
