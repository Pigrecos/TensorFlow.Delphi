unit Keras.Activations;

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
