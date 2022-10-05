unit TensorFlow.String_ops;

interface
    uses
      SysUtils,
      System.Rtti,
      TF4D.Core.CApi,
      TensorFlow.DApi,
      TensorFlow.Context,
      Numpy.Axis;

type
  string_ops = record
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
  end;

implementation
    uses Tensorflow;

{ string_ops }

function string_ops.lower(input: TFTensor; encoding, name: String): TFTensor;
begin
     Result := tf.Context.ExecuteOp('StringLower', name, ExecuteOpArgs.Create([ input, encoding])).FirstOrDefault(nil);
end;

function string_ops.regex_replace(input: TFTensor; pattern, rewrite: string; replace_global: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('StaticRegexReplace', name, ExecuteOpArgs.Create([ input ])
                         .SetAttributes(['pattern',pattern,'rewrite',rewrite,'replace_global',replace_global])).FirstOrDefault(nil);
end;

function string_ops.string_length(input: TFTensor; name, &unit: string): TFTensor;
begin
    var Args := ExecuteOpArgs.Create([ input ]) ;

    Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                               begin
                                    //unit = op.get_attr<string>("unit")
                               end;

    Result := tf.Context.ExecuteOp('StringLength', name, Args
                           .SetAttributes(['unit', &unit ])).FirstOrDefault(nil);
end;

function string_ops.substr<T>(input: T; pos, len: Integer; uint, name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('Substr', name, ExecuteOpArgs.Create([ TValue.From<T>(input), Pos, len]).SetAttributes(['unit',uint])).FirstOrDefault(nil);
end;

end.
