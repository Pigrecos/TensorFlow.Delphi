unit TensorFlow.embedding_ops;
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

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.Context,
         TensorFlow.Variable ;

type
  embedding_ops = record
    private

    public
        /// <summary>
        /// Helper function for embedding_lookup and _compute_sampled_logits.
        /// </summary>
        /// <param name="params"></param>
        /// <param name="ids"></param>
        /// <param name="partition_strategy"></param>
        /// <param name="name"></param>
        /// <param name="max_norm"></param>
        /// <returns></returns>
        class function _embedding_lookup_and_transform(params: IVariableV1;      ids: TFTensor; partition_strategy : string= 'mod'; name : string= ''; max_norm: string = ''): TFTensor; overload; static;
        class function _embedding_lookup_and_transform(params: TArray<TFTensor>; ids: TFTensor; partition_strategy : string= 'mod'; name : string= ''; max_norm: string = ''): TFTensor; overload; static;
        class function _clip(params: TFTensor; ids: TFTensor; max_norm : string= '') : TFTensor; static;
        class function embedding_lookup(params: TArray<TFTensor>; ids: TFTensor; partition_strategy: string = 'mod'; name: string = ''; validate_indices : Boolean= true; max_norm: string = ''): TFTensor; overload; static;
        class function embedding_lookup(params: IVariableV1;      ids: TFTensor; partition_strategy: string = 'mod'; name: string = ''; validate_indices : Boolean= true; max_norm : string= ''): TFTensor; overload; static;
  end;

implementation
         uses TensorFlow.Constant_op,
              TensorFlow.Ops,
              Tensorflow.array_ops,
              Tensorflow.NameScope,
              Tensorflow.Utils;

{ embedding_ops }

class function embedding_ops._embedding_lookup_and_transform(params: IVariableV1; ids: TFTensor; partition_strategy, name, max_norm: string): TFTensor;
begin
    var vValues : TArray<TValue> := [TValue.From<IVariableV1>(params), ids];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'embedding_lookup', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var np : Integer := 1;
                                                ids := Tops.convert_to_tensor(ids, DtInvalid, 'ids');
                                                if np = 1 then
                                                begin
                                                    var gather := array_ops.gather(params.AsTensor, ids, name);
                                                    var res    := _clip(gather, ids, max_norm);

                                                    Result := array_ops.identity(res);
                                                    Exit;
                                                end;

                                                raise Exception.Create('_embedding_lookup_and_transform');
                                            end );
end;

class function embedding_ops._embedding_lookup_and_transform(params: TArray<TFTensor>; ids: TFTensor; partition_strategy, name, max_norm: string): TFTensor;
begin
    var vValues : TArray<TValue> := [TValue.From<TArray<TFTensor>>(params), ids];
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'embedding_lookup', @vValues),
                                          function(v1: TNameScope): TFTensor
                                            begin
                                                name := v1.ToString;
                                                var np : Integer := Length(params);
                                                params := Tops.convert_n_to_tensor_or_indexed_slices(params, DtInvalid, 'params');
                                                ids := Tops.convert_to_tensor(ids, DtInvalid, 'ids');
                                                if np = 1 then
                                                begin
                                                    Tops.colocate_with(params[0]);
                                                    var res := _clip(array_ops.gather(@params[0], ids, name), ids, max_norm);
                                                    Result := array_ops.identity(res);
                                                end else
                                                begin
                                                    // Flatten the ids. There are two cases where we need to do this.
                                                    raise Exception.Create('_embedding_lookup_and_transform');
                                                end;
                                            end );
end;

class function embedding_ops.embedding_lookup(params: TArray<TFTensor>; ids: TFTensor; partition_strategy, name: string; validate_indices: Boolean; max_norm: string): TFTensor;
begin
    Result := _embedding_lookup_and_transform(params, ids, partition_strategy, name, max_norm);
end;

class function embedding_ops.embedding_lookup(params: IVariableV1; ids: TFTensor; partition_strategy, name: string; validate_indices: Boolean; max_norm: string): TFTensor;
begin
    Result := _embedding_lookup_and_transform(params, ids, partition_strategy, name, max_norm)
end;

class function embedding_ops._clip(params, ids: TFTensor; max_norm: string): TFTensor;
begin
    if max_norm = '' then
       Exit(params);
    raise Exception.Create('_clip');
end;

end.
