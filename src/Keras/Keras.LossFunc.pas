unit Keras.LossFunc;
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

          TensorFlow.DApi;

type

   ILossFunc = interface
   ['{AE7EE5F7-1243-45C6-86B1-C43DB3146C9F}']
     function GetReduction : string;
     function GetName : string;

     function Call(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor;

     property Reduction : string read GetReduction;
     property Name      : string read GetName;
   end;

implementation

end.
