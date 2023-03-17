unit TensorFlow.String_ops;
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
    uses
      SysUtils,
      System.Rtti,

      Spring,

      TF4D.Core.CApi,
      TensorFlow.DApi,
      TensorFlow.Context,
      TensorFlow.Tensors.Ragged;

type
  string_ops = Class
     private

     public
        function lower(input: TFTensor; encoding : string = ''; name: String = ''): TFTensor;
        function regex_replace(input: TFTensor; pattern: string; rewrite: string; replace_global: Boolean = true; name: string = ''): TFTensor;
        /// <summary>
        /// Return substrings from `Tensor` of strings.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="pos"></param>
        /// <param name="len"></param>
        /// <param name="name"></param>
        /// <param name="uint"></param>
        /// <returns></returns>
        function substr<T>(input: T; pos: Integer; len: Integer; &uint: string = 'BYTE'; name: string = ''): TFTensor;
        /// <summary>
        /// Computes the length of each string given in the input tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <param name="unit"></param>
        /// <returns></returns>
        function string_length(input: TFTensor; name: string = ''; &unit: string = 'BYTE'): TFTensor;
        function string_format(inputs: TArray<TFTensor>; template: string = '%s'; placeholder: string = '%s'; summarize: Integer = 3; name: string = ''): TFTensor;
        function string_split_v2(input: TFTensor; sep: string = ' '; maxsplit: Integer = -1; name: string = ''):  RaggedTensor;
        function unicode_decode_with_offsets(input: TFTensor; input_encoding: string; errors: string; replacement_char: Integer = $FFFD; replace_control_characters: Boolean = false; name: string = ''): Tuple<RaggedTensor, RaggedTensor>;
        function _unicode_decode(input: TFTensor; input_encoding: string; errors: string; replacement_char: Integer; replace_control_characters: Boolean; with_offsets: Boolean; name: string = ''): Tuple<RaggedTensor, RaggedTensor>;
  end;

implementation
    uses Tensorflow,
         Tensorflow.NameScope,
         Tensorflow.Utils,
         TensorFlow.Ops,
         Tensorflow.array_ops,TensorFlow.Slice ;

{ string_ops }

function string_ops.lower(input: TFTensor; encoding, name: String): TFTensor;
begin
     Result := tf.Context.ExecuteOp('StringLower', name, ExecuteOpArgs.Create([ input, encoding])).First;
end;

function string_ops.regex_replace(input: TFTensor; pattern, rewrite: string; replace_global: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StaticRegexReplace', name, ExecuteOpArgs.Create([ input ])
                         .SetAttributes(['pattern',pattern,'rewrite',rewrite,'replace_global',replace_global])).First;
end;

function string_ops.string_length(input: TFTensor; name, &unit: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ input ]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;
                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr<string>('unit');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp('StringLength', name, Args
                           .SetAttributes(['unit', &unit ])).First;
end;

function string_ops.substr<T>(input: T; pos, len: Integer; uint, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Substr', name, ExecuteOpArgs.Create([ TValue.From<T>(input), Pos, len]).SetAttributes(['unit',uint])).First;
end;

function string_ops.string_format(inputs: TArray<TFTensor>; template, placeholder: string; summarize: Integer; name: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                   Result := [];
                                   var pParam : TParameter;

                                   pParam.sNome := 'T' ;
                                   pParam.vValue:= op.get_attr('T');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'template' ;
                                   pParam.vValue:= op.get_attr<string>('template');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'placeholder' ;
                                   pParam.vValue:= op.get_attr<string>('placeholder');
                                   Result := Result + [ pParam ] ;

                                   pParam.sNome := 'summarize' ;
                                   pParam.vValue:= op.get_attr<Integer>('summarize');
                                   Result := Result + [ pParam ] ;
                               end;

    Result := tf.Context.ExecuteOp('StringFormat', name, Args
                           .SetAttributes(['template',template, 'placeholder',placeholder, 'summarize', summarize ])).First;
end;

function string_ops.string_split_v2(input: TFTensor; sep: string; maxsplit: Integer; name: string): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'StringSplit'),
                          function(v1: TNameScope): RaggedTensor
                            begin
                                Tops.convert_to_tensor(sep, TF_DataType.TF_STRING);
                                if input.rank = 0 then
                                begin
                                    var parts := string_split_v2(array_ops.stack([ input ]), sep, maxsplit, name);
                                    Result := parts;
                                    exit;
                                end;

                                var Args := ExecuteOpArgs.Create([input, sep]) ;

                                Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                         begin
                                             Result := [];
                                             var pParam : TParameter;

                                             pParam.sNome := 'maxsplit' ;
                                             pParam.vValue:= op.get_attr<Integer>('maxsplit');
                                             Result := Result + [ pParam ] ;
                                         end;

                                var Res := tf.Context.ExecuteOp('StringSplitV2', name, Args.SetAttributes(['maxsplit',maxsplit ]));
                                var indices := res[0];
                                var values  := res[1];
                                var shape   := res[2];

                                indices.shape := TFShape.Create([-1, 2]);
                                values.shape  := TFShape.Create([-1]);
                                shape.shape   := TFShape.Create([2]);
                                var sparse_res := TSparseTensor.Create(indices, values, shape);
                                Result := RaggedTensor.from_value_rowids(sparse_res.values, sparse_res.indices[[Slice.All, 0]], sparse_res.dense_shape[0], '', false);
                            end );
end;

function string_ops.unicode_decode_with_offsets(input: TFTensor; input_encoding, errors: string; replacement_char: Integer; replace_control_characters: Boolean;
  name: string): Tuple<RaggedTensor, RaggedTensor>;
begin
     Result := TUtils.tf_with<TNameScope,Tuple<RaggedTensor, RaggedTensor>>( TOps.name_scope(name, 'UnicodeDecodeWithOffsets'),
                          function(v1: TNameScope): Tuple<RaggedTensor, RaggedTensor>
                            begin
                                var t := _unicode_decode(input, input_encoding, errors, replacement_char, replace_control_characters, true, name);
                                var codepoints := t.value1;
                                var byte_start_offsets := t.value2;
                                Result := Tuple<RaggedTensor, RaggedTensor>.Create(codepoints, byte_start_offsets);
                            end);
end;

function string_ops._unicode_decode(input: TFTensor; input_encoding, errors: string; replacement_char: Integer; replace_control_characters, with_offsets: Boolean;
  name: string): Tuple<RaggedTensor, RaggedTensor>;
begin
    if with_offsets then
    begin
        var Args := ExecuteOpArgs.Create([input]);
        Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                     begin
                                         Result := [];
                                         var pParam : TParameter;

                                         pParam.sNome := 'input_encoding' ;
                                         pParam.vValue:= op.get_attr<String>('input_encoding');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'errors' ;
                                         pParam.vValue:= op.get_attr<String>('errors');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'replacement_char' ;
                                         pParam.vValue:= op.get_attr<Integer>('replacement_char');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'replace_control_characters' ;
                                         pParam.vValue:= op.get_attr<Boolean>('replace_control_characters');
                                         Result := Result + [ pParam ] ;

                                         pParam.sNome := 'Tsplits' ;
                                         pParam.vValue:= op.get_attr('Tsplits');
                                         Result := Result + [ pParam ] ;
                                     end;

        var flat_result := tf.Context.ExecuteOp('RaggedTensorToVariant', name, Args
                      .SetAttributes(['input_encoding',input_encoding,'errors',errors,'replacement_char',replacement_char,'replace_control_characters',replace_control_characters]));

        var codepoints := RaggedTensor.from_row_splits(flat_result[1], flat_result[0],'', false);
        var offsets    := RaggedTensor.from_row_splits(flat_result[2], flat_result[0], '',false);
        Result         :=  Tuple<RaggedTensor, RaggedTensor>.Create(codepoints, offsets);
    end;
    Result :=  Tuple<RaggedTensor, RaggedTensor>.Create(nil,nil);
end;

end.
