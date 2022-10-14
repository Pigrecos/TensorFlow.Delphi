unit TensorFlow.Tensors.Ragged;

interface
    uses System.SysUtils,

         Spring.Collections,
         rtti,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.Framework;

type
  Dimension = record
    private
      Fvalue : Int64;
    public
      constructor Create(value_   : Int64);
      function merge_with(other: Dimension):Dimension;
      function ToString: string;

      class operator implicit(value_:Int64): Dimension;
      class operator implicit(value_:Dimension): Int64;

      property value  : Int64 read Fvalue;
  end;

  /// <summary>
  /// Represents a sparse tensor.
  /// </summary>
  TSparseTensor = record
    private

    public
      indices    : TFTensor;
      values     : TFTensor;
      dense_shape: TFTensor;
      constructor Create(indices_: TFTensor; values_: TFTensor;dense_shape_ : TFTensor);overload;
      constructor Create(indices_: TArray< TArray<Int64> >; values_: TValue;dense_shape_ : TArray<Int64>); overload;
      procedure _init;
  end;



implementation
       uses Tensorflow.Utils,
            Tensorflow.NameScope,
            TensorFlow.Ops ;

{ SparseTensor }

constructor TSparseTensor.Create(indices_, values_, dense_shape_: TFTensor);
begin
    self.indices    := indices_;
    self.values     := values_;
    self.dense_shape:= dense_shape_;
    _init();
end;

constructor TSparseTensor.Create(indices_: TArray<TArray<Int64>>; values_: TValue; dense_shape_: TArray<Int64>);
begin
   Self := TUtils.tf_with<TNameScope,TSparseTensor>( TOps.name_scope('', 'SparseTensor'),
                                          function(v1: TNameScope): TSparseTensor
                                            begin
                                                Result.indices    := Tops.convert_to_tensor(TValue.From< TArray<TArray<Int64>> >( indices_), TDtypes.cint64, 'indices');
                                                Result.values     := Tops.convert_to_tensor(values_, TF_DataType.DtInvalid, 'values');
                                                Result.dense_shape:= Tops.convert_to_tensor(TValue.From< TArray<Int64> >(dense_shape_), TDtypes.cint64, 'dense_shape');
                                            end );
    _init;
end;

procedure TSparseTensor._init;
begin
    var indices_shape    := indices.shape.with_rank(2);
    var values_shape     := values.shape.with_rank(1);
    var dense_shape_shape:= dense_shape.shape.with_rank(1);
    indices_shape['0'].merge_with(TFShape.Create( [ values_shape[0] ]));
    indices_shape['1'].merge_with(TFShape.Create( [ dense_shape_shape[0] ] ));
end;

{ Dimension }

constructor Dimension.Create(value_: Int64);
begin
    Fvalue := value_;
end;

class operator Dimension.implicit(value_: Int64): Dimension;
begin
    Result := Dimension.Create(value_)
end;

class operator Dimension.implicit(value_: Dimension): Int64;
begin
    Result := value_.value
end;

function Dimension.merge_with(other: Dimension): Dimension;
begin
    if Fvalue = -1 then Result := Dimension.Create(other.value)
    else                Result := Dimension(Fvalue);
end;

function Dimension.ToString: string;
begin
    Result := 'Dimension('+ IntToStr(Fvalue) +')';
end;

end.
