unit Keras.Data;
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

          Spring,
          Spring.Container.Common,

          TF4D.Core.CApi,
          TensorFlow.DApi,
          TensorFlow.dataset_ops,
          TensorFlow.Framework,

          Keras.Models;

type

  DatasetOptions = class
  end;

  IDatasetV2 = interface;

  /// <summary>
  /// An iterator producing tf.Tensor objects from a tf.data.Dataset.
  /// </summary>
  OwnedIterator = class(TInterfacedObject, IDisposable)
     public
       _dataset : IDatasetV2;
       _element_spec      : TArray<TensorSpec>;
       ops                : dataset_ops;
       _deleter           : TFTensor;
       _iterator_resource : TFTensor;

       constructor Create(dataset: IDatasetV2);
       procedure   _create_iterator(dataset: IDatasetV2);
       function    next: TArray<TFTensor>;
       procedure Dispose;
  end;

  IDatasetV2 = interface(IEnumerable< Tuple<TFTensor, TFTensor> >)
     ['{DFD57016-C441-4BD5-8D1C-E6A26CF9739A}']
      function  Get_class_name: TArray<String>;
      procedure Set_class_name(const Value: TArray<String>);
      function  Get_variant_tensor: TFTensor;
      procedure Set_variant_tensor(const Value: TFTensor);
      function  Get_output_shapes:TArray<TFShape>;
      function  Get_output_types:TArray<TF_DataType>;
      function  Get_element_spec:TArray<TensorSpec>;
      function  Get_structure: TArray<TensorSpec>;
      procedure Set_structure(const Value: TArray<TensorSpec>);

      /// <summary>
      /// Caches the elements in this dataset.
      /// </summary>
      /// <param name="filename"></param>
      /// <returns></returns>
      function cache(filename : string= ''): IDatasetV2;

      /// <summary>
      /// Creates a `Dataset` by concatenating the given dataset with this dataset.
      /// </summary>
      /// <param name="dataset"></param>
      /// <returns></returns>
      function concatenate(dataset: IDatasetV2): IDatasetV2;

      /// <summary>
      ///
      /// </summary>
      /// <param name="count"></param>
      /// <returns></returns>
      function &repeat(count: Integer = -1): IDatasetV2;

      /// <summary>
      /// Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
      /// </summary>
      /// <param name="num_shards">The number of shards operating in parallel</param>
      /// <param name="index">The worker index</param>
      /// <returns></returns>
      function shard(num_shards: Integer; index: Integer): IDatasetV2;

      function shuffle(buffer_size: Integer; seed : pInteger = nil; reshuffle_each_iteration: Boolean = true): IDatasetV2;

      /// <summary>
      /// Creates a `Dataset` that skips `count` elements from this dataset.
      /// </summary>
      /// <param name="count"></param>
      /// <returns></returns>
      function skip(count: Integer): IDatasetV2;

      function batch(batch_size: Integer; drop_remainder: Boolean = false): IDatasetV2;

      function prefetch(buffer_size : Integer= -1; slack_period: PInteger = nil): IDatasetV2;

      function take(count: Integer): IDatasetV2;

      function optimize(optimizations: TArray<string>; optimization_configs: TArray<string>): IDatasetV2;

      function map(map_func: TFunc<TFTensors, TFTensors>; use_inter_op_parallelism : Boolean= true; preserve_cardinality: Boolean = true; use_legacy_function: Boolean = false): IDatasetV2; overload;
      function map(map_func: TFunc<TFTensors, TFTensors>; num_parallel_calls: Integer): IDatasetV2; overload;

      function filter(map_func: TFunc<TFTensors, TFTensors>): IDatasetV2; overload;
      function filter(map_func: TFunc<TFTensor, Boolean>): IDatasetV2; overload;

      function make_one_shot_iterator: OwnedIterator;

      function flat_map(map_func: TFunc<TFTensor, IDatasetV2>): IDatasetV2;

      function model(algorithm: AutotuneAlgorithm; cpu_budget: Int64; ram_budget: Int64): IDatasetV2;

      function with_options(options: DatasetOptions): IDatasetV2;

      /// <summary>
      /// Apply options, such as optimization configuration, to the dataset.
      /// </summary>
      /// <returns></returns>
      function apply_options: IDatasetV2;

      /// <summary>
      /// Returns the cardinality of `dataset`, if known.
      /// </summary>
      /// <param name="name"></param>
      /// <returns></returns>
      function cardinality(name: string = ''): TFTensor;

      property class_names    : TArray<String>      read Get_class_name     write Set_class_name;
      property variant_tensor : TFTensor            read Get_variant_tensor write Set_variant_tensor;
      property output_shapes  : TArray<TFShape>     read Get_output_shapes ;
      property output_types   : TArray<TF_DataType> read Get_output_types;
      property element_spec   : TArray<TensorSpec>  read Get_element_spec;
      property structure      : TArray<TensorSpec>  read Get_structure      write Set_structure;
  end;

implementation
          uses  Winapi.Windows,
                Tensorflow;

 { OwnedIterator }
constructor OwnedIterator.Create(dataset: IDatasetV2);
begin

    _create_iterator(dataset);
end;

procedure OwnedIterator.Dispose;
begin
   tf.Runner.Execute(tf.Context, 'DeleteIterator', 0, [ _iterator_resource, _deleter ], []);
end;

function OwnedIterator.next: TArray<TFTensor>;
begin
    try
      var res := ops.iterator_get_next(_iterator_resource, _dataset.output_types, _dataset.output_shapes);
      for var i := 0 to Length(res) - 1 do
      begin
          res[i].shape := _element_spec[i].shape;
      end;
      Result := res;

    except
       on e : ERangeError do
        MessageBoxA(0,'Error',PAnsiChar(AnsiString(e.Message)),mb_Ok)
    end;

end;

procedure OwnedIterator._create_iterator(dataset: IDatasetV2);
begin
    dataset := dataset.apply_options();
    _dataset := dataset;
    _element_spec := dataset.element_spec;
    // _flat_output_types =
    var tI := ops.anonymous_iterator_v2(_dataset.output_types, _dataset.output_shapes);
    _iterator_resource := tI.Value1;
    _deleter           := tI.Value2;
    ops.make_iterator(dataset.variant_tensor, _iterator_resource);
end;

end.
