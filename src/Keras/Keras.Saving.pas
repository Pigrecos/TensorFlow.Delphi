unit Keras.Saving;
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
         System.Rtti,

         Spring,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         Tensorflow.Variable,
         Keras.Core,
         Numpy,

         hdf5dll,
         Hdf5;

type

  hdf5_format = class
     private
       class procedure save_attributes_to_hdf5_group<T>(f: Int64; name: string; data: TArray<T>); static;
       class procedure WriteDataset(f: Int64; name: string; data: TFTensor); static;
       class function  Split<T>(list: TArray<T>; chunkSize: Integer): TList<TList<T>>;
       class procedure WriteAttrs<T>(f: Int64; typename, name: string; data: TArray<T>); static;

     public
       const HDF5_OBJECT_HEADER_LIMIT : Integer = 64512;
     public
       /// <summary>
       /// Preprocess layer weights between different Keras formats.
       /// </summary>
       /// <param name="layer"></param>
       /// <param name="weights"></param>
       /// <param name="original_keras_version"></param>
       /// <param name="original_backend"></param>
       class function  preprocess_weights_for_loading(layer: ILayer; weights: TList<TNDArray>; original_keras_version : string= ''; original_backend : string= ''): TList<TNDArray>;
       /// <summary>
       /// Converts weights for RNN layers between native and CuDNN format.
       /// </summary>
       /// <param name="layer"></param>
       /// <param name="weights"></param>
       class  function _convert_rnn_weights(layer: ILayer; weights: TList<TNDArray>): TList<TNDArray>;
       class procedure load_weights_from_hdf5_group(f: Int64; layers: TList<ILayer>);
       class procedure save_weights_to_hdf5_group(f: Int64; layers: TList<ILayer>);
       class function  load_attributes_from_hdf5_group(group: Int64; name: string): TArray<string>;
       class function  _legacy_weights(layer: ILayer): TList<IVariableV1>;
  end;

implementation
       uses  System.Math,
             Tensorflow;

{ hdf5_format }

class function hdf5_format.load_attributes_from_hdf5_group(group: Int64; name: string): TArray<string>;
var
  tAttrs  : Tuple<Boolean, TArray<String>>;
  success : Boolean;
  attr    : TArray<String>;
begin
    tAttrs := THdf5.ReadStringAttributes(group, name, '', true);
    success := tAttrs.Value1;
    attr    := tAttrs.Value2;

    Result := [];
    if success then
       Result := attr;
end;

class procedure hdf5_format.load_weights_from_hdf5_group(f: Int64; layers: TList<ILayer>);
var
  original_keras_version : string;
  original_backend,name  : string;
  tAttrs                 : Tuple<Boolean, TArray<String>>;
  success                : Boolean;
  attr,layer_names       : TArray<String>;
  ver_major,ver_minor,k  : Integer;
  filtered_layers        : TList<ILayer>;
  filtered_layer_names,
  lstLayerName           : TList<string>;
  weight_value_tuples    : TList<Tuple<IVariableV1, TNDArray>>;
  weight_values          : TList<TNDArray>;
begin
    original_keras_version := '2.5.0';
    original_backend       := '';

    tAttrs := THdf5.ReadStringAttributes(f, 'keras_version', '', true);
    success := tAttrs.Value1;
    attr    := tAttrs.Value2;
    if success then
        original_keras_version := attr[0];
    // keras version should be 2.5.0+
    ver_major := Integer.Parse(original_keras_version.Split(['.'])[0]);
    ver_minor := integer.Parse(original_keras_version.Split(['.'])[1]);
    if (ver_major < 2) or ((ver_major = 2) and (ver_minor < 5)) then
       raise Exception.Create('keras version should be 2.5.0 or later.');

    tAttrs := THdf5.ReadStringAttributes(f, 'backend', '', true);
    success := tAttrs.Value1;
    attr    := tAttrs.Value2;
    if success then
        original_backend := attr[0];

    filtered_layers := TList<ILayer>.Create;
    try
      for var layer in layers do
      begin
          var weights := _legacy_weights(layer);
          if weights.Count > 0 then
              filtered_layers.Add(layer);
      end;

      lstLayerName := TList<string>.Create;
      try
        for var it in filtered_layers do
           lstLayerName.Add(it.Name);

        layer_names := load_attributes_from_hdf5_group(f, 'layer_names');
        filtered_layer_names := TList<string>.Create;
        try
          for name in layer_names do
          begin
              if not lstLayerName.Contains(name) then
                  continue;

              var g : Int64 := THdf5.FH5.H5Gopen2(f, PAnsiChar(AnsiString(name)), H5P_DEFAULT);
              var weight_names := load_attributes_from_hdf5_group(g, 'weight_names');
              if Length(weight_names) > 0 then
                  filtered_layer_names.Add(name);
              THdf5.FH5.H5Gclose(g);
          end;

          layer_names := filtered_layer_names.ToArray;
          if Length(layer_names) <> filtered_layers.Count then
             raise Exception.Create('You are trying to load a weight file containing layer_names layers into a model with filtered_layers.Count layers.');

          weight_value_tuples := TList<Tuple<IVariableV1, TNDArray>>.Create;
          for k := 0 to Length(layer_names) -1 do
          begin
              name := layer_names[k];
              weight_values := TList<TNDArray>.Create;
              var g : Int64 := THdf5.FH5.H5Gopen2(f, PAnsiChar(AnsiString(name)), H5P_DEFAULT);
              var weight_names := load_attributes_from_hdf5_group(g, 'weight_names');
              for var i_ in weight_names do
              begin
                  var tDataset := THdf5.ReadDataset<Single>(g, i_);
                  success  := tDataset.Value1;
                  var res  := tDataset.value2;
                  if success then
                      weight_values.Add( np.np_array(res) );
              end;
              THdf5.FH5.H5Gclose(g);
              var layer            := filtered_layers[k];
              var symbolic_weights := _legacy_weights(layer);
              preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend);
              if weight_values.Count <> symbolic_weights.Count then
                  raise Exception.Create('Layer #' + k.ToString + ' (named ' + layer.Name + 'in the current model) was found to ' +
                                         'correspond to layer ' + name + ' in the save file. However the new layer ' + layer.Name +
                                         ' expects ' + symbolic_weights.Count.ToString + ' weights, but the saved weights have ' +
                                         weight_values.Count.ToString + ' elements.');

              var tupleArray : TArray<Tuple<IVariableV1, TNDArray>> := [];
              for var i := 0 to symbolic_weights.Count - 1 do
                  tupleArray := tupleArray + [ Tuple.Create(symbolic_weights[i], weight_values[i]) ];

              weight_value_tuples.AddRange(tupleArray);
          end;
        finally
          filtered_layer_names.free;
        end;
      finally
        lstLayerName.free;
      end;
    finally
      filtered_layers.free;
    end;

    tf.keras.backend.batch_set_value(weight_value_tuples);
end;

class function hdf5_format.Split<T>(list: TArray<T>; chunkSize: Integer): TList<TList<T>>;
var
  splitList  : TList<TList<T>>;
  chunkCount : Integer;
begin
    splitList  := TList<TList<T>>.Create;
    chunkCount := Ceil(double(Length(list)) / double(chunkSize));

    for var c := 0 to chunkCount - 1 do
    begin
        var skip := c * chunkSize;
        var take := skip + chunkSize;
        var chunk := TList<T>.Create;

        for var e := skip to take - 1 do
        begin
            if e >= Length(list) then Break;

            chunk.Add( list[e] );
        end;
        splitList.Add(chunk);
    end;

    Result := splitList;
end;

class procedure hdf5_format.WriteAttrs<T>(f: Int64; typename: string; name: string; data: TArray<T>);
begin
    THdf5.WriteAttributes<T>(f, name, data)
end;

class procedure hdf5_format.save_attributes_to_hdf5_group<T>(f: Int64; name: string; data: TArray<T>);
var
  num_chunks,
  getSize,
  getCount,
  chunk_id     : Integer;
  getType      : string;
  chunk_data   : TList<T>;
  chunked_data : TList<TList<T>>;
begin
    num_chunks := 1;

    chunked_data := Split<T>(data, num_chunks);
    getSize := 0;

    getType := 'string';
    if Length(data) > 0 then
    begin
        getType := string(PTypeInfo(TypeInfo(T))^.Name).ToLower;
    end;

    if      getType = 'single'     then getSize := sizeof(Single)
    else if getType = 'double'     then getSize := sizeof(double)
    else if getType = 'string'     then getSize := -1
    else if getType = 'ansistring' then getSize := -1
    else if getType = 'int32'      then getSize := sizeof(Integer)
    else if getType = 'integer'    then getSize := sizeof(Integer)
    else if getType = 'int64'      then getSize := sizeof(Int64)
    else                                getSize := -1;

    getCount := chunked_data.Count;

    if getSize <> -1 then
    begin
        num_chunks := Ceil(double(getCount * getSize) / HDF5_OBJECT_HEADER_LIMIT);
        if num_chunks > 1 then chunked_data := Split<T>(data, num_chunks);
    end;

    if num_chunks > 1 then
    begin
        for chunk_id := 0 to chunked_data.Count - 1 do
        begin
            chunk_data := chunked_data[chunk_id] ;
            WriteAttrs<T>(f, getType, name + chunk_id.ToString, chunk_data.ToArray);
        end;
    end else
    begin
        WriteAttrs<T>(f, getType, name, data);
    end;
end;

class procedure hdf5_format.WriteDataset(f: Int64; name: string; data: TFTensor);
begin
    case data.dtype of
      TF_DataType.TF_FLOAT: THdf5.WriteDatasetFromArray<single>(f, name, data.ToArray<single>,Data.shape.as_int_list);
      TF_DataType.TF_DOUBLE:THdf5.WriteDatasetFromArray<double>(f, name, data.ToArray<double>,Data.shape.as_int_list);
      TF_DataType.TF_INT32: THdf5.WriteDatasetFromArray<integer>(f, name, data.ToArray<Integer>,Data.shape.as_int_list);
      TF_DataType.TF_INT64: THdf5.WriteDatasetFromArray<int64>(f, name, data.ToArray<int64>,Data.shape.as_int_list);
    else
      THdf5.WriteDatasetFromArray<single>(f, name, data.ToArray<single>);
    end;
end;

class procedure hdf5_format.save_weights_to_hdf5_group(f: Int64; layers: TList<ILayer>);
var
  layerName : TList<string>;
begin
    layerName := TList<string>.Create;
    try
      for var layer in layers do
         layerName.Add(layer.Name);

      if layerName.Count < 1  then  layerName.Add(' ');

      save_attributes_to_hdf5_group<string>(f, 'layer_names', layerName.ToArray);
      THdf5.WriteAttribute(f, 'backend', 'tensorflow');
      THdf5.WriteAttribute(f, 'keras_version', '"2.5.0');

      for var layer in layers do
      begin
          var weights := _legacy_weights(layer);
          if weights.Count = 0 then
              continue;

          var weight_names := TList<string>.Create;
          try
            // weight_values= keras.backend.batch_get_value(weights);
            for var weight in weights do
                weight_names.Add(weight.Name);

            var g := THdf5.CreateOrOpenGroup(f, THdf5Utils.NormalizedName(layer.Name));
            save_attributes_to_hdf5_group<string>(g, 'weight_names', weight_names.ToArray);
            for var i := 0 to weight_names.Count - 1 do
            begin
                var name := weight_names[i];
                var val  := weights[i];
                var tensor := val.AsTensor;
                if name.IndexOf('/') > 1 then
                begin
                    var crDataGroup := THdf5.CreateOrOpenGroup(g, THdf5Utils.NormalizedName(name.Split(['/'])[0]));
                    WriteDataset(crDataGroup, name.Split(['/'])[1], tensor);
                    THdf5.CloseGroup(crDataGroup);
                end else
                begin
                    WriteDataset(g, name, tensor);
                end;
            end;
            THdf5.CloseGroup(g);
          finally
            weight_names.free;
          end;
      end;
    finally
      layerName.Free;
    end;
end;

class function hdf5_format._legacy_weights(layer: ILayer): TList<IVariableV1>;
begin
    var weights := layer.TrainableWeights;
    weights.AddRange(layer.NonTrainableWeights);
    Result := weights;
end;

class function hdf5_format.preprocess_weights_for_loading(layer: ILayer; weights: TList<TNDArray>; original_keras_version, original_backend: string): TList<TNDArray>;
begin
    // convert CuDNN layers
    Result := _convert_rnn_weights(layer, weights);
end;

class function hdf5_format._convert_rnn_weights(layer: ILayer; weights: TList<TNDArray>): TList<TNDArray>;
begin
    var ClassLayer   :=  TObject(layer)  ;
    var target_class :=  ClassLayer.ClassName;
    Result := weights;
end;

end.
