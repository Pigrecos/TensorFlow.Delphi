unit TensorFlow.dataset_ops;
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

         Keras.Core;

type
  dataset_ops = record
    private
      function anonymous_iterator_v3_eager_fallback(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string ; ctx: TContext): TFTensor;
    public
      function tensor_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string = ''): TFTensor;
      /// <summary>
      /// Creates a dataset that emits each dim-0 slice of `components` once.
      /// </summary>
      /// <param name="components"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function tensor_slice_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string = ''): TFTensor;
      function range_dataset(start: TFTensor; stop: TFTensor; step: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function repeat_dataset(input_dataset: TFTensor; count: TFTensor; output_types : TArray<TF_DataType>; output_shapes: TArray<TFShape>; name : string= '') : TFTensor;
      function shard_dataset(input_dataset: TFTensor; num_shards: TFTensor; index: TFTensor; output_types: TArray<TF_DataType> ; output_shapes: TArray<TFShape>; require_non_empty: Boolean = false; name: string = '') : TFTensor;
      function zip_dataset(input_datasets: TArray<TFTEnsor>; output_types: TArray<TF_DataType>; output_shapes:TArray<TFShape>; name: string = '') : TFTensor;
      function shuffle_dataset_v3(input_dataset: TFTensor; buffer_size: TFTensor; seed: TFTensor; seed2: TFTensor; seed_generator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; reshuffle_each_iteration: Boolean = true; name: string = ''): TFTEnsor;
      function skip_dataset(input_dataset : TFTensor; count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function dummy_seed_generator(name: string = '') : TFTensor;
      function concatenate_dataset(input_dataset: TFTensor; another_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      function cache_dataset_v2(input_dataset: TFTensor; filename: TFTensor; cache: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that batches `batch_size` elements from `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="buffer_size"></param>
      /// <param name="drop_remainder"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="parallel_copy"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function batch_dataset_v2(input_dataset: TFTensor; buffer_size: TFTensor; drop_remainder: TFTensor; output_types: TArray<TF_DataType>; output_shapes:TArray<TFShape>; parallel_copy: Boolean = false; name: string = '') : TFTensor;
      /// <summary>
      ///
      /// </summary>
      /// <param name="name"></param>
      /// <returns></returns>
      function dummy_memory_cache(name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that asynchronously prefetches elements from `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="buffer_size"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="slack_period"></param>
      /// <param name="legacy_autotune"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function prefetch_dataset(input_dataset: TFTensor; buffer_size: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; slack_period: Integer = 0; legacy_autotune: Boolean= true; name: string = ''): TFTensor;
      /// <summary>
      /// Creates a dataset that contains `count` elements from the `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="count"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function take_dataset(input_dataset: TFTensor; count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset by applying optimizations to `input_dataset`.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="optimizations"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="optimization_configs"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function optimize_dataset(input_dataset: TFTensor; optimizations: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; optimization_configs : TArray<string>= []; name: string = '') : TFTensor;
      function optimize_dataset_v2(input_dataset: TFTensor; optimizations_enabled: TFTensor; optimizations_disabled: TFTensor; optimizations_default: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; optimization_configs: TArray<string> = []; name: string = '') : TFTensor;
      /// <summary>
      /// Identity transformation that models performance.
      /// </summary>
      /// <param name="input_dataset"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="algorithm"></param>
      /// <param name="cpu_budget"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function model_dataset(input_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64; name: string = '') : TFTensor;
      /// <summary>
      /// A container for an iterator resource.
      /// </summary>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns>A tuple of `Tensor` objects (handle, deleter).</returns>
      function anonymous_iterator_v2(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : Tuple<TFTensor, TFTensor>;
      function anonymous_iterator_v3(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Makes a new iterator from the given `dataset` and stores it in `iterator`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="iterator"></param>
      /// <param name="name"></param>
      /// <returns>The created Operation.</returns>
      procedure make_iterator(dataset: TFTensor; iterator: TFTensor; name: string = '') ;
      /// <summary>
      ///
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="iterator"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism : Boolean= true; preserve_cardinality: Boolean = false; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset containing elements of `input_dataset` matching `predicate`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="predicate"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function filter_dataset(dataset: TFTensor; predicate: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '')  : TFTensor;
      /// <summary>
      /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="f"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function flat_map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
      /// <summary>
      /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
      /// </summary>
      /// <param name="dataset"></param>
      /// <param name="num_parallel_calls"></param>
      /// <param name="f"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="use_inter_op_parallelism"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function parallel_map_dataset_v2(dataset: TFTensor; num_parallel_calls: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism: Boolean = true; deterministic: string = 'default'; preserve_cardinality : Boolean= false; name : string= '') : TFTensor;
      /// <summary>
      /// A container for an iterator resource.
      /// </summary>
      /// <param name="handle"></param>
      /// <param name="deleter"></param>
      /// <param name="name"></param>
      /// <returns>The created Operation.</returns>
      procedure delete_iterator(handle: TFTensor; deleter: TFTensor; name: string = '') ;
      /// <summary>
      /// Gets the next output from the given iterator .
      /// </summary>
      /// <param name="iterator"></param>
      /// <param name="output_types"></param>
      /// <param name="output_shapes"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      function iterator_get_next(iterator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = ''): TArray<TFTensor>;
  end;

implementation
        uses Tensorflow,
             Tensorflow.Utils;


{ dataset_ops }

function dataset_ops.anonymous_iterator_v3(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string = '') : TFTensor;
var
 iInfo : TFastPathOpExecInfo;
begin
    var ctx := tf.Context;
    var attrs := TDictionary<string, TValue>.Create;
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])];

    attrs.Add('output_types', TValue.From<TArray<Integer>>(typeIntArray));
    attrs.Add('output_shapes', TValue.From<TArray<TFShape>>(output_shapes));
    if ctx.executing_eagerly then
    begin
        try
            iInfo := TFastPathOpExecInfo.Create('AnonymousIteratorV3', name,[]);
            iInfo.attrs := attrs;
            var Res := tf.Runner.TFE_FastPathExecute(iInfo);
            Result :=  Res[0];
            Exit;
        except
            Result := anonymous_iterator_v3_eager_fallback(output_types, output_shapes, name, ctx);
            Exit;
        end
    end;
    Result := tf.OpDefLib._apply_op_helper('AnonymousIteratorV3', name, [ GetArg('output_types',  attrs['output_types']),
                                                                          GetArg('output_shapes', attrs['output_shapes']) ]).outputs[0];
end;

function dataset_ops.anonymous_iterator_v3_eager_fallback(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string ; ctx: TContext): TFTensor;
begin
    var dictAttrs := TDictionary<string, TValue>.Create;
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])];

    dictAttrs.Add('output_types', TValue.From<TArray<Integer>>(typeIntArray));
    dictAttrs.Add('output_shapes', TValue.From<TArray<TFShape>>(output_shapes));

    var attrs : TArray<TValue> := [ dictAttrs['output_types'], dictAttrs['output_shapes'] ];

    var Res := TExecute.quick_execute('AnonymousIteratorV3', 1, [], attrs, ctx, name);
    Result  := Res[0];
end;

function dataset_ops.anonymous_iterator_v2(output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): Tuple<TFTensor, TFTensor>;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var Res := tf.Context.ExecuteOp('AnonymousIteratorV2', name, ExecuteOpArgs.Create([])
          .SetAttributes(['output_types', TValue.From<TArray<Integer>>(typeIntArray), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )]));

    Result := Tuple.Create(Res[0],Res[0])
end;

function dataset_ops.batch_dataset_v2(input_dataset, buffer_size, drop_remainder: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  parallel_copy: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('BatchDatasetV2', name, ExecuteOpArgs.Create([input_dataset, buffer_size, drop_remainder])
                 .SetAttributes(['parallel_copy', parallel_copy, 'output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.cache_dataset_v2(input_dataset, filename, cache: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('CacheDatasetV2', name, ExecuteOpArgs.Create([input_dataset, filename, cache])
                 .SetAttributes(['output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.concatenate_dataset(input_dataset, another_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('ConcatenateDataset', name, ExecuteOpArgs.Create([input_dataset, another_dataset])
                 .SetAttributes(['output_types', TValue.From<TArray<Integer>>( typeIntArray ), 'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

procedure dataset_ops.delete_iterator(handle, deleter: TFTensor; name: string);
begin
    tf.Context.ExecuteOp('DeleteIterator', name, ExecuteOpArgs.Create([handle, deleter]));
end;

function dataset_ops.dummy_memory_cache(name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DummyMemoryCache', name, ExecuteOpArgs.Create([])).First;
end;

function dataset_ops.dummy_seed_generator(name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('DummySeedGenerator', name, ExecuteOpArgs.Create([])).First;
end;

function dataset_ops.filter_dataset(dataset: TFTensor; predicate: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('FilterDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['predicate',     TValue.From<ConcreteFunction>(predicate),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.flat_map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('FlatMapDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['f',     TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.iterator_get_next(iterator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TArray<TFTensor>;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    try
    Result := tf.Context.ExecuteOp('IteratorGetNext', name, ExecuteOpArgs.Create([iterator])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).toArray;
    except
      result := [];
      Exit;
    end;
end;

procedure dataset_ops.make_iterator(dataset, iterator: TFTensor; name: string);
begin
    tf.Context.ExecuteOp('MakeIterator', name, ExecuteOpArgs.Create([dataset, iterator]));
end;

function dataset_ops.map_dataset(dataset: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; use_inter_op_parallelism,
  preserve_cardinality: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('MapDataset', name, ExecuteOpArgs.Create([dataset, a])
                       .SetAttributes(['f',             TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'use_inter_op_parallelism',  use_inter_op_parallelism,
                                       'preserve_cardinality', preserve_cardinality])).First;
end;

function dataset_ops.model_dataset(input_dataset: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; algorithm: AutotuneAlgorithm; cpu_budget,
  ram_budget: Int64; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('ModelDataset', name, ExecuteOpArgs.Create([input_dataset])
                       .SetAttributes(['algorithm',     TValue.From<Integer>(Ord(algorithm)),
                                       'cpu_budget',    cpu_budget,
                                       'ram_budget',    ram_budget,
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.optimize_dataset(input_dataset, optimizations: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  optimization_configs: TArray<string>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('OptimizeDataset', name, ExecuteOpArgs.Create([input_dataset, optimizations])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'optimization_configs', TValue.From<TArray<string>>( optimization_configs )])).First;
end;

function dataset_ops.optimize_dataset_v2(input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default: TFTensor; output_types: TArray<TF_DataType>;
  output_shapes: TArray<TFShape>; optimization_configs: TArray<string>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('OptimizeDatasetV2', name, ExecuteOpArgs.Create([input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default])
                       .SetAttributes(['output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'optimization_configs', TValue.From<TArray<string>>( optimization_configs )])).First;
end;

function dataset_ops.parallel_map_dataset_v2(dataset, num_parallel_calls: TFTensor; f: ConcreteFunction; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  use_inter_op_parallelism: Boolean; deterministic: string; preserve_cardinality: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    var a : TArray<TFTensor>:= [];
    Result := tf.Context.ExecuteOp('ParallelMapDatasetV2', name, ExecuteOpArgs.Create([dataset, a, num_parallel_calls])
                       .SetAttributes(['f',             TValue.From<ConcreteFunction>(f),
                                       'output_types',  TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes', TValue.From<TArray<TFShape>>( output_shapes ),
                                       'use_inter_op_parallelism',  use_inter_op_parallelism,
                                       'deterministic',             deterministic,
                                       'preserve_cardinality', preserve_cardinality])).First;
end;

function dataset_ops.prefetch_dataset(input_dataset, buffer_size: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; slack_period: Integer;
  legacy_autotune: Boolean; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

    Result := tf.Context.ExecuteOp('PrefetchDataset', name, ExecuteOpArgs.Create([input_dataset, buffer_size])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes ),
                                       'slack_period',   slack_period,
                                       'legacy_autotune',legacy_autotune])).First;
end;

function dataset_ops.range_dataset(start, stop, step: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   var typeIntArray : TArray<Integer> := [];
   for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('RangeDataset', name, ExecuteOpArgs.Create([start, stop, step])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.repeat_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('RepeatDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.shard_dataset(input_dataset, num_shards, index: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; require_non_empty: Boolean;
  name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ShardDataset', name, ExecuteOpArgs.Create([input_dataset, num_shards, index])
                       .SetAttributes(['require_non_empty', require_non_empty,
                                       'output_types',      TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',     TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.shuffle_dataset_v3(input_dataset, buffer_size, seed, seed2, seed_generator: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>;
  reshuffle_each_iteration: Boolean; name: string): TFTEnsor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ShuffleDatasetV3', name, ExecuteOpArgs.Create([input_dataset, buffer_size, seed, seed2, seed_generator])
                       .SetAttributes(['reshuffle_each_iteration', reshuffle_each_iteration,
                                       'output_types',             TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',            TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.skip_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('SkipDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.take_dataset(input_dataset, count: TFTensor; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('TakeDataset', name, ExecuteOpArgs.Create([input_dataset, count])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.tensor_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('TensorDataset', name, ExecuteOpArgs.Create([components])
                       .SetAttributes(['output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.tensor_slice_dataset(components: TArray<TFTensor>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('TensorSliceDataset', name, ExecuteOpArgs.Create([components])
                       .SetAttributes(['output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

function dataset_ops.zip_dataset(input_datasets: TArray<TFTEnsor>; output_types: TArray<TF_DataType>; output_shapes: TArray<TFShape>; name: string): TFTensor;
begin
    var typeIntArray : TArray<Integer> := [];
    for var i := 0 to Length(output_types)-1 do
       typeIntArray := typeIntArray + [ Ord(output_types[i])] ;

   Result := tf.Context.ExecuteOp('ZipDataset', name, ExecuteOpArgs.Create([input_datasets])
                       .SetAttributes(['output_types',   TValue.From<TArray<Integer>>( typeIntArray ),
                                       'output_shapes',  TValue.From<TArray<TFShape>>( output_shapes )])).First;
end;

end.
