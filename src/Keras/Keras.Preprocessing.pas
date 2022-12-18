unit Keras.Preprocessing;
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

          TensorFlow.DApi,

          Keras.Engine,
          Keras.ILayersApi,
          Keras.ArgsDefinition,
          Keras.Layer;

type
  TSequence = class

  end;

  DatasetUtils = class

  end;

  TextApi  = class

  end;

  /// <summary>
  /// Text tokenization API.
  /// This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
  /// (each integer being the index of a token in a dictionary) or into a vector where the coefficient for
  /// each token could be binary, based on word count, based on tf-idf...
  /// </summary>
  /// <remarks>
  /// This code is a fairly straight port of the Python code for Keras text preprocessing found at:
  /// https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
  /// </remarks>
  Tokenizer  = class

  end;

  TPreprocessing = class(TInterfacedObject, IPreprocessing)
     protected
       Ftext : TextApi;
     public
        sequence     : TSequence;
        dataset_utils: DatasetUtils;

        function Resizing(height: Integer; width: Integer; interpolation: string = 'bilinear'): ILayer;
        function TextVectorization(standardize: TFunc<TFTensor, TFTensor> = nil; split : string= 'whitespace'; max_tokens: Integer = -1; output_mode: string = 'int'; output_sequence_length: Integer = -1): ILayer;

        constructor Create;

        property text : TextApi read Ftext;
  end;

implementation

{ Preprocessing }

constructor TPreprocessing.Create;
begin
   sequence      := TSequence.Create;
   dataset_utils := DatasetUtils.Create;
   Ftext         := TextApi.Create;
end;

function TPreprocessing.Resizing(height, width: Integer; interpolation: string): ILayer;
begin

end;

function TPreprocessing.TextVectorization(standardize: TFunc<TFTensor, TFTensor>; split: string; max_tokens: Integer; output_mode: string;
  output_sequence_length: Integer): ILayer;
var
  args: TextVectorizationArgs;
begin
    args := TextVectorizationArgs.Create;

    args.Standardize := standardize;
    args.Split       := split;
    args.MaxTokens   := max_tokens;
    args.OutputMode  := output_mode;
    args.OutputSequenceLength := output_sequence_length;

    Result := Keras.Layer.TextVectorization.create(args)
end;

end.
