unit TensorFlow.Interfaces;
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
         System.Rtti;

type
  ICanBeFlattened = interface
    ['{46A5FDDF-CB1F-40A8-8E4F-883DDDBED7EF}']

      function Flatten: TArray<TValue>;
  end;

  /// <summary>
  /// in order to limit function return value
  /// is Tensor or TensorArray
  /// </summary>
  ITensorOrTensorArray = interface
    ['{C2D5D17A-38F4-4DE5-B299-24A24C4A64A7}']

  end;

  IFromMergeVars<T> = interface
    ['{9C2F771D-EF50-4731-934E-4B5DAAC6400F}']

    function FromMergeVars(mergeVars: TArray<ITensorOrTensorArray>): T ;
  end;

  IPackable<T> = interface
    ['{9C2F771D-EF50-4731-934E-4B5DAAC6400F}']

    function Pack(sequences: TArray<TValue>): T ;
  end;


implementation

end.
