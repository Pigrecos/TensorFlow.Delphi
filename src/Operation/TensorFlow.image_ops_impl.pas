unit TensorFlow.image_ops_impl;
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
         rtti,

         Spring.Collections.Enumerable,
         Spring.Collections.Lists,

         TensorFlow.DApi,
         TensorFlow.Context;

type
  ResizeMethod= record
    const
      BILINEAR        : string = 'bilinear';
      NEAREST_NEIGHBOR: string = 'nearest';
      BICUBIC         : string = 'bicubic';
      AREA            : string = 'area';
      LANCZOS3        : string = 'lanczos3';
      LANCZOS5        : string = 'lanczos5';
      GAUSSIAN        : string = 'gaussian';
      MITCHELLCUBIC   : string = 'mitchellcubic';
  end;

  image_ops_impl = record
    private
       class function _resize_images_common(images: TFTensor; resizer_fn : TFunc<TFTensor, TFTensor, TFTensor>; size: TFTensor; preserve_aspect_ratio: Boolean; name: string; skip_resize_if_same: Boolean): TFTensor; static;
       class function _ImageDimensions(image: TFTensor; rank: Integer): TArray<Int64>; static;
    public
       /// <summary>
       /// Resize `images` to `size` using the specified `method`.
       /// </summary>
       /// <param name="images"></param>
       /// <param name="size"></param>
       /// <param name="method"></param>
       /// <param name="preserve_aspect_ratio"></param>
       /// <param name="antialias"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function resize_images_v2<T>(images: TFTensor; size: T; method: string = 'bilinear'; preserve_aspect_ratio : Boolean= false; antialias : Boolean= false; name : string= ''): TFTensor; static;
  end;

implementation
         uses Tensorflow,
              TensorFlow.Tensor,
              TensorFlow.Tensors.Ragged,
              Tensorflow.NameScope,
              Tensorflow.Utils,
              TensorFlow.Ops,
              Tensorflow.array_ops,
              Tensorflow.math_ops,
              TensorFlow.gen_image_ops;

{ image_ops_impl }

class function image_ops_impl.resize_images_v2<T>(images: TFTensor; size: T; method: string; preserve_aspect_ratio, antialias: Boolean; name: string): TFTensor;
begin
    var resize_fn : TFunc<TFTensor, TFTensor, TfTensor> := function(Img : TFTensor; _size: TFTensor): TFTensor
                                        begin
                                            if method = ResizeMethod.BILINEAR then
                                                Exit( gen_image_ops.resize_bilinear(Img, _size, false, true) )
                                            else if method = ResizeMethod.NEAREST_NEIGHBOR then
                                                Exit(  gen_image_ops.resize_nearest_neighbor(Img, _size, false, true) );
                                            raise Exception.Create('resize_images_v2');
                                        end;
    var size_tensor := Tops.convert_to_tensor(TValue.From<T>(size), tf.int32_t);
    Result          := _resize_images_common(images, resize_fn, size_tensor, preserve_aspect_ratio, name, false);
end;


class function image_ops_impl._ImageDimensions(image: TFTensor; rank: Integer): TArray<Int64>;
begin
    if image.shape.IsFullyDefined then
        Exit( image.shape.dims)
    else begin
        var static_shape  := image.shape.with_rank(rank).dims;
        var dynamic_shape := array_ops.unstack(array_ops.shape(image), @rank);

        var ss_storage : TArray<Int64> := [];
        var ds_storage : TArray<Int64> := [];
        // var sd = static_shape.Zip(dynamic_shape, (first, second) => storage[storage.Length] = first;
        for var i := 0 to Length(static_shape)-1 do
        begin
            var ss := static_shape[i];
            var ds := dynamic_shape[i];

            ss_storage := ss_storage + [ ss ] ;
            ds_storage := ds_storage + [ Int64(TTensor(ds)) ] ;
        end;
        var sd := Enumerable<Int64>.Create(static_shape);
        var sdd:= TList<TFTensor>.Create(dynamic_shape);
        sd.Zip<TFTensor,Boolean>(sdd, function(ss: Int64; ds: TFTensor): Boolean
                                       begin
                                           ss_storage := ss_storage + [ ss ] ;
                                           ds_storage := ds_storage + [ Int64(TTensor(ds)) ] ;
                                           Result := true;
                                       end);


        if Length(ss_storage) > 0 then Result := ss_storage
        else                           Result := ds_storage;
    end;
end;

class function image_ops_impl._resize_images_common(images: TFTensor; resizer_fn: TFunc<TFTensor, TFTensor, TFTensor>; size: TFTensor; preserve_aspect_ratio: Boolean;
  name: string; skip_resize_if_same: Boolean): TFTensor;
begin
     var vValues : TArray<TValue> := [images, size];
     Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'resize', @vValues),
                        function(v1: TNameScope): TFTensor
                          begin
                              if images.shape.ndim = -1 then
                                 raise Exception.Create('"images" contains no shape.');
                              var is_batch : Boolean := true;
                              if images.shape.ndim = 3 then
                              begin
                                  is_batch := false;
                                  images := array_ops.expand_dims(images, 0);
                              end
                              else if images.shape.ndim <> 4 then
                                 raise Exception.Create('"images" must have either 3 or 4 dimensions.');

                              var height := images.dims[1];
                              var width  := images.dims[2];

                              if not size.shape.is_compatible_with(TFShape.Create([ 2 ])) then
                                 raise Exception.Create('"size" must be a 1-D Tensor of 2 elements: new_height, new_width');

                              if preserve_aspect_ratio then
                              begin
                                  var _chcw_ := _ImageDimensions(images, 4);

                                  var scale_factor_height := math_ops.cast(size[0], Tdtypes.cfloat32) / TTEnsor(_chcw_[1]);
                                  var scale_factor_width  := math_ops.cast(size[1], Tdtypes.cfloat32) / TTEnsor(_chcw_[2]);

                                  var scale_factor        := math_ops.minimum(scale_factor_height, scale_factor_width);

                                  var scaled_height_const := math_ops.cast(math_ops.round(TTEnsor(scale_factor) * _chcw_[1]), Tdtypes.cint32);
                                  var scaled_width_const  := math_ops.cast(math_ops.round(TTEnsor(scale_factor) * _chcw_[2]), Tdtypes.cint32);

                                  var v : TValue := [scaled_height_const, scaled_width_const];
                                  size := Tops.convert_to_tensor( v, Tdtypes.cint32, 'size');
                              end;

                              var size_const_as_shape := TUtils.constant_value_as_shape(size);
                              var new_height_const    := tensor_shape.dimension_at_index(size_const_as_shape, 0).value;
                              var new_width_const     := tensor_shape.dimension_at_index(size_const_as_shape, 1).value;

                              var x_null : Boolean := true;
                              if skip_resize_if_same then
                              begin
                                  for var x: Integer in [ new_width_const, width, new_height_const, height ] do
                                  begin
                                      if (width <> new_width_const) and (height = new_height_const) then
                                          break;

                                      if x <> 0 then
                                        x_null := false;

                                  end;
                                  if  not x_null then
                                      images := array_ops.squeeze(images, [ 0 ]);
                                  Result := images;
                                  Exit;
                              end;

                              images := resizer_fn(images, size);

                              images.shape := TFShape.Create([-1, new_height_const, new_width_const, -1]);

                              if not is_batch then
                                  images := array_ops.squeeze(images, [ 0 ]);
                              Result := images;
                          end);

end;

end.


