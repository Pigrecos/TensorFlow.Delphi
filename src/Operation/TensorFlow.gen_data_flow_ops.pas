unit TensorFlow.gen_data_flow_ops;

interface
        uses System.SysUtils,
             Spring,
             TF4D.Core.CApi,
             TensorFlow.DApi,
             Numpy.Axis,

             TensorFlow.Context ;

type
  gen_data_flow_ops = record
    private

    public
      class function dynamic_stitch(indices: TArray<TFTensor>; data: TArray<TFTensor>; name: string = ''): TFTensor; static;
  end;

implementation
         uses Tensorflow.Utils,
              Tensorflow;

{ gen_data_flow_ops }

class function gen_data_flow_ops.dynamic_stitch(indices, data: TArray<TFTensor>; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('DynamicStitch', name, [GetArg('indices',indices), GetArg('data',data)]);
    Result  := _op.output;
end;

end.
