unit Tensorflow.Session;

{$WARN IMPLICIT_STRING_CAST OFF}
{$WARN IMPLICIT_STRING_CAST_LOSS OFF}

interface
    uses Winapi.Windows,
         System.SysUtils,
         System.Generics.Collections,

         Spring,
         Spring.Collections,
         Spring.Collections.Lists,
         Spring.Collections.Dictionaries,

         TF4D.Core.CApi,
         TensorFlow.DApiBase,
         TensorFlow.DApi,

         Protogen.config ;


implementation
           uses Tensorflow,
                TensorFlow.Ops,
                Tensorflow.Utils,

                Oz.Pb.Classes,
                Oz.Pb.StrBuffer;


end.
