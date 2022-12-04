unit TensorFlow.gen_image_ops;
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
         rtti,

         TensorFlow.DApi,
         TensorFlow.Context;

type
  gen_image_ops = record
    class function resize_nearest_neighbor<Tsize>(images: TFTensor; size: Tsize; align_corners: Boolean = false; half_pixel_centers : Boolean= false; name: string = ''): TFTensor; static;
    class function resize_bilinear(images: TFTensor; size: TFTensor; align_corners : Boolean= false; half_pixel_centers : Boolean = false; name : string= ''): TFTensor; static;
  end;

implementation
        uses Tensorflow;

{ gen_image_ops }

class function gen_image_ops.resize_bilinear(images, size: TFTensor; align_corners, half_pixel_centers: Boolean; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('ResizeBilinear', name, ExecuteOpArgs.Create([images, size])
              .SetAttributes(['align_corners', align_corners, 'half_pixel_centers',  half_pixel_centers ])).First;
end;

class function gen_image_ops.resize_nearest_neighbor<Tsize>(images: TFTensor; size: Tsize; align_corners, half_pixel_centers: Boolean; name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('ResizeNearestNeighbor', name, ExecuteOpArgs.Create([images, TValue.From<Tsize>(size)])
              .SetAttributes(['align_corners', align_corners, 'half_pixel_centers',  half_pixel_centers ])).First;
end;

end.
