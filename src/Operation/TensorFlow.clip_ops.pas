unit TensorFlow.clip_ops;

interface
    uses System.SysUtils,
         Spring,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context;

type
  clip_ops = record
    private

    public
      class function clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor = nil; name: string = ''): Tuple<TFTensor,TFTensor> ; static;
      class function clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string = ''): TFTensor ; static;
      /// <summary>
      /// Computes the global norm of multiple tensors.
      /// </summary>
      /// <param name="t_list"></param>
      /// <param name="name"></param>
      /// <returns></returns>
      class function global_norm(t_list:  TArray<TFTensor>; name: string = '') : TFTensor ; static;
  end;

implementation

{ clip_ops }

class function clip_ops.clip_by_global_norm(t_list: TArray<TFTensor>; clip_norm: Single; use_norm: TFTensor; name: string): Tuple<TFTensor, TFTensor>;
begin

end;

class function clip_ops.clip_by_value<T1, T2>(t: TFTensor; clip_value_min: T1; clip_value_max: T2; name: string): TFTensor;
begin

end;

class function clip_ops.global_norm(t_list: TArray<TFTensor>; name: string): TFTensor;
begin

end;

end.
