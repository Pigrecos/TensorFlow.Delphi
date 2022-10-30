unit TensorFlow.Interfaces;

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
