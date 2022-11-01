unit Keras.Activations;
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

interface
       uses System.SysUtils,
            Spring,
            Spring.Collections,
            Spring.Collections.Lists,
            Spring.Collections.Dictionaries,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            Numpy.Axis,
            TensorFlow.Context,
            TensorFlow.Variable,

            TensorFlow.Initializer,

            Keras.Layer;

type
  TActivation = Reference To function(features: TFTensor; name: string = ''): TFTensor;

implementation

end.
