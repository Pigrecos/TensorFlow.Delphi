unit TensorFlow.Tensors.Ragged;
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
    uses System.SysUtils,
         System.Generics.Collections,

         Spring,
         Spring.Collections,
         rtti,

         TF4D.Core.CApi,
         TensorFlow.DApi,
         TensorFlow.Slice,
         TensorFlow.Framework,
         TensorFlow.Interfaces;

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

  tensor_shape = class
     public
      class function dimension_at_index(shape: TFShape; index: Integer): Dimension;
      class function dimension_value(dimension: Dimension) : Integer;
  end;

  /// <summary>
  /// Partitioning of a sequence of values into contiguous subsequences ("rows").
  /// </summary>
  RowPartition = class(CompositeTensor)
    private
       Frow_splits  : TFTensor;
       Frow_lengths : TFTensor;
       Fvalue_rowids: TFTensor;
       Fnrows       : TFTensor;

       function GetStaticRow: Integer;
       function GetSt_uni_row_l: Integer;
    public
      constructor Create(row_splits: TFTensor; row_lengths : TFTensor = nil; value_rowids: TFTensor = nil; nrows: TFTensor = nil; uniform_row_length : TFTensor= nil);
      /// <summary>
      /// Creates a `RowPartition` with rows partitioned by `value_rowids`.
      /// </summary>
      /// <param name="value_rowids"></param>
      /// <param name="nrows"></param>
      /// <param name="validate"></param>
      /// <param name="preferred_dtype"></param>
      /// <returns></returns>
      class function from_value_rowids(value_rowids: TFTensor; nrows: TFTensor = nil; validate : Boolean = true; preferred_dtype: TF_DataType = DtInvalid): RowPartition;
      class function from_row_splits(row_splits: TFTensor; validate : Boolean= true; preferred_dtype : TF_DataType = DtInvalid): RowPartition;

      property row_splits   : TFTensor read Frow_splits;
      property static_nrows : Integer  read GetStaticRow;
      property static_uniform_row_length : Integer read GetSt_uni_row_l;

  end;

  /// <summary>
  /// Represents a ragged tensor.
  /// </summary>
  RaggedTensor = class (CompositeTensor)
     private
        Fvalues        : TFTensor;
        Frow_partition : RowPartition;
        Frow_splits    : TFTensor;

        function GetDtype: TF_DataType;
        function GetRow_splits: TFTensor;
        function getShape: TFShape;
        function GetItem(index: Integer): TFTensor; overload;
        function GetItem(slices: TArray<Slice>): RaggedTensor; overload;
        function _ragged_getitem(row_key: Integer) : TFTensor;
        function _ragged_getitem_inner_dimensions(input: RaggedTensor; slices: TArray<Slice>): RaggedTensor;
        function getNest_row_splits: TArray<TFTensor>;

     public
        constructor Create(values: TFTensor; internal : Boolean = true; row_partition : RowPartition= nil);
        class function from_row_partition(values: TFTensor; row_partition: RowPartition; validate: Boolean = true): RaggedTensor;
        /// <summary>
        /// Creates a `RaggedTensor` with rows partitioned by `value_rowids`.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="value_rowids"></param>
        /// <param name="nrows"></param>
        /// <param name="name"></param>
        /// <param name="validate"></param>
        /// <returns></returns>
        class function from_value_rowids(values: TFTensor; value_rowids: TFTensor; nrows: TFTensor = nil; name: string = ''; validate: Boolean = true): RaggedTensor;
        class function from_row_splits(values: TFTensor; row_splits: TFTensor; name : string = ''; validate: Boolean = true) : RaggedTensor;
        function _to_variant(batched_input : Boolean= false; name : string= '') : TFTensor;
        class function FromTensor(t: TFTensor): RaggedTensor;
        function ToTensor: TFTensor;

        property flat_values                : TFTensor         read Fvalues;
        property row_splits                 : TFTensor         read GetRow_splits;
        property dtype                      : TF_DataType      read GetDtype;
        property shape                      : TFShape          read getShape;
        property nested_row_splits          : TArray<TFTensor> read getNest_row_splits;
        property Item[i: Integer]           : TFTensor         read GetItem; default;
        property Item[slices: TArray<Slice>]: RaggedTensor     read GetItem; default;
  end;

  /// <summary>
  /// TensorArray is designed to hide an underlying implementation object
  /// and as such accesses many of that object's hidden fields.
  ///
  /// "Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.
  /// This class is meant to be used with dynamic iteration primitives such as
  /// `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  /// "flow" control flow dependencies.
  /// </summary>
  TTensorArray = class( TInterfacedObject, ITensorOrTensorArray)
    protected
      Fdtype : TF_DataType;
      Fhandle: TFTensor;
      Fflow  : TFTensor;
      Finfer_shape : Boolean;
      Fcolocate_with_first_write_call : Boolean;
    public
      function unstack(value: TFTensor; name: string = ''): TTensorArray; virtual; abstract;
      function stack(name: string = ''): TFTensor; virtual; abstract;
      function gather(indices: TFTensor; name: string = ''): TFTensor; virtual; abstract;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; virtual; abstract;

      function read<T>(index: T; name: string = ''): TFTensor;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; overload;

      property  dtype                           : TF_DataType read Fdtype;
      property  handle                          : TFTensor    read FHandle;
      property  flow                            : TFTensor    read Fflow;
      property  infer_shape                     : Boolean     read Finfer_shape;
      property  colocate_with_first_write_call  : Boolean     read Fcolocate_with_first_write_call;
  end;

  BodyItem = class(TInterfacedObject, ICanBeFlattened, IPackable<BodyItem>, IFromMergeVars<BodyItem>)
    private
      FI      : TFTensor;
      FAccs_ta: TArray<TTensorArray>;
    public
      constructor Create; overload;
      constructor Create(v_I : TFTensor; v_accs_ta: TArray<TTensorArray>); overload;

      function Flatten: TArray<TValue>;
      function FromMergeVars(mergeVars: TArray<ITensorOrTensorArray>): BodyItem ;
      function Pack(sequences: TArray<TValue>): BodyItem ;

      property I       : TFTensor             read FI;
      property Accs_ta : TArray<TTensorArray> read FAccs_ta;
  end;

  TGraphTensorArray = class(TTensorArray)
    private
      Fdynamic_size     : Boolean;
      Felement_shape    : TList<TFShape>;
      Fcolocate_with    : TList<TFTensor>;
      Fclear_after_read : Boolean;
      Ftensor_array     : TList<TFTensor>;

      function size(name: string = ''): TFTensor;
    protected

    public
      constructor Create(_dtype: TF_DataType; size: TFTensor; dynamic_size: Boolean = false;
                         clear_after_read : Boolean= true; tensor_array_name : string = ''; _handle : TFTensor= nil;
                         _flow : TFTensor = nil; _infer_shape: Boolean = true; _element_shape : PTFShape= nil;
                         _colocate_with_first_write_call : Boolean= true; _name: string = '');
      function  scatter(indices: TFTensor; value: TFTensor; name: string = ''): TTensorArray;
      procedure _merge_element_shape(shape: TFShape);
      procedure _maybe_colocate_with(value: TFTensor);
      function read<T>(index: T; name: string = ''): TFTensor; reintroduce;
      function unstack(value: TFTensor; name: string = ''): TTensorArray; override;
      function write(index: TFTensor; value: TFTensor; name: string = ''): TTensorArray; overload; override;
      function write<T>(index: Integer; value: T; name: string = ''): TTensorArray; reintroduce; overload;
      function stack(name: string = ''): TFTensor; override;
      function gather(indices: TFTensor; name: string = ''): TFTensor; override;
  end;


implementation
       uses Tensorflow,
            TensorFlow.Context,
            Tensorflow.Utils,
            Tensorflow.NameScope,
            TensorFlow.Ops,
            Tensorflow.math_ops,
            Tensorflow.array_ops,
            TensorFlow.Constant_op,
            TensorFlow.gen_data_flow_ops,
            TensorFlow.tensor_array_ops,
            NumPy.NDArray ;

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

{ RowPartition }

constructor RowPartition.Create(row_splits, row_lengths, value_rowids, nrows, uniform_row_length: TFTensor);
begin
    Frow_splits   := row_splits;
    Frow_lengths  := row_lengths;
    Fvalue_rowids := value_rowids;
    Fnrows        := nrows;
end;

class function RowPartition.from_value_rowids(value_rowids, nrows: TFTensor; validate: Boolean; preferred_dtype: TF_DataType): RowPartition;
begin
    Result := TUtils.tf_with<TNameScope,RowPartition>( TOps.name_scope('', 'RowPartitionFromValueRowIds'),
                  function(v1: TNameScope): RowPartition
                    begin
                        var value_rowids_int32 := math_ops.cast(value_rowids, Tdtypes.cint32);
                        var nrows_int32        := math_ops.cast(nrows, Tdtypes.cint32);
                        var row_lengths        := tf.math.bincount(value_rowids_int32, nil, nrows_int32, nrows_int32,  value_rowids.dtype);
                        var a1 := Tops.convert_to_tensor( TArray<Int64>.create( 0 ) );
                        var a2 := tf.cumsum(row_lengths);
                        var row_splits := array_ops.concat([ a1,a2 ], 0);
                        Result := RowPartition.Create(row_splits, row_lengths, value_rowids, nrows);
                    end );
end;

class function RowPartition.from_row_splits(row_splits: TFTensor; validate: Boolean; preferred_dtype: TF_DataType): RowPartition;
begin
    Result := TUtils.tf_with<TNameScope,RowPartition>( TOps.name_scope('', 'RowPartitionFromRowSplits'),
                  function(v1: TNameScope): RowPartition
                    begin
                        Result :=  RowPartition.Create(row_splits);
                    end );
end;

function RowPartition.GetStaticRow: Integer;
begin
    Result := Frow_splits.shape[0] - 1;
end;

function RowPartition.GetSt_uni_row_l: Integer;
begin
    Result := -1
end;

{ RaggedTensor }

constructor RaggedTensor.Create(values: TFTensor; internal: Boolean; row_partition: RowPartition);
begin
     Fvalues        := values;
     Frow_partition := row_partition;
end;

class function RaggedTensor.FromTensor(t: TFTensor): RaggedTensor;
begin
    Result := t.Tag.AsType<RaggedTensor>;
end;

function RaggedTensor.ToTensor: TFTensor;
begin
    Result := Self._to_variant;
end;

class function RaggedTensor.from_row_partition(values: TFTensor; row_partition: RowPartition; validate: Boolean): RaggedTensor;
begin
    Result := RaggedTensor.Create(values, true, row_partition);
end;

function RaggedTensor.GetDtype: TF_DataType;
begin
    Result := Fvalues.Dtype;
end;

function RaggedTensor.GetRow_splits: TFTensor;
begin
    Result := Frow_partition.row_splits
end;

function RaggedTensor.getShape: TFShape;
begin
    var nrows := Frow_partition.static_nrows;
    var ncols := Frow_partition.static_uniform_row_length;
    Result := TFShape.Create([nrows, ncols]);
end;

function RaggedTensor._ragged_getitem(row_key: Integer): TFTensor;
begin
    var starts := Frow_splits[':-1'];
    var limits := Frow_splits['1:'];
    var row    := Fvalues[ starts[row_key], limits[row_key] ];
    Result := row;
end;

function RaggedTensor._ragged_getitem_inner_dimensions(input: RaggedTensor; slices: TArray<Slice>): RaggedTensor;
begin
    Result := input;
end;

function RaggedTensor._to_variant(batched_input: Boolean; name: string): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'RaggedToVariant'),
                  function(v1: TNameScope): TFTensor
                    begin
                        var Args := ExecuteOpArgs.Create([nested_row_splits, flat_values]);
                        Args.GetGradientAttrs :=  function(op: TFOperation): TArray<TParameter>
                                       begin
                                           Result := [];
                                           var pParam : TParameter;

                                           pParam.sNome := 'RAGGED_RANK' ;
                                           pParam.vValue:= op.get_attr<Integer>('RAGGED_RANK');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'Tvalues' ;
                                           pParam.vValue:= op.get_attr('Tvalues');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'Tsplits' ;
                                           pParam.vValue:= op.get_attr('Tsplits');
                                           Result := Result + [ pParam ] ;

                                           pParam.sNome := 'batched_input' ;
                                           pParam.vValue:= op.get_attr<Boolean>('batched_input');
                                           Result := Result + [ pParam ] ;
                                       end;

                        Result := tf.Context.ExecuteOp('RaggedTensorToVariant', name, Args.SetAttributes(['batched_input',batched_input])).First;
                    end );
end;

class function RaggedTensor.from_value_rowids(values: TFTensor; value_rowids: TFTensor; nrows: TFTensor; name: string; validate: Boolean): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'RaggedFromValueRowIds'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        var row_partition := RowPartition.from_value_rowids(value_rowids, nrows, validate);
                        Result := from_row_partition(values, row_partition, validate);
                    end );
end;

class function RaggedTensor.from_row_splits(values, row_splits: TFTensor; name: string; validate: Boolean): RaggedTensor;
begin
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope(name, 'RaggedFromRowSplits'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        var row_partition := RowPartition.from_row_splits(row_splits, validate);
                        Result := from_row_partition(values, row_partition, validate);
                    end );
end;

function RaggedTensor.GetItem(index: Integer): TFTensor;
begin
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('', 'RaggedGetItem'),
                  function(v1: TNameScope): TFTensor
                    begin
                        Result   := _ragged_getitem(index);
                    end );
end;

function RaggedTensor.GetItem(slices: TArray<Slice>): RaggedTensor;
begin
    var row_key := slices[0];
    var inner_keys := slices;
    Delete(inner_keys,0,1);
    var args := TUtils.ParseSlices(slices);
    Result := TUtils.tf_with<TNameScope,RaggedTensor>( TOps.name_scope('', 'RaggedGetItem'),
                  function(v1: TNameScope): RaggedTensor
                    begin
                        Result :=_ragged_getitem_inner_dimensions(self, inner_keys);
                    end );
end;

function RaggedTensor.getNest_row_splits: TArray<TFTensor>;
begin
    Result := [ Frow_splits ]
end;

{ tensor_shape }

class function tensor_shape.dimension_at_index(shape: TFShape; index: Integer): Dimension;
begin
    if   shape.ndim < 0  then Result := Dimension.Create(-1)
    else                      Result := Dimension.Create(shape.dims[index])
end;

class function tensor_shape.dimension_value(dimension: Dimension): Integer;
begin
    Result := dimension.value
end;

{ BodyItem }

constructor BodyItem.Create;
begin

end;

constructor BodyItem.Create(v_I: TFTensor; v_accs_ta: TArray<TTensorArray>);
begin
    FI       := v_I;
    FAccs_ta := v_accs_ta;
end;

function BodyItem.Flatten: TArray<TValue>;
begin
    var elements := TList<TValue>.Create([ FI ]);
    var a : TArray<TValue> := [];
    for var i := 0 to Length(FAccs_ta)-1 do
        a := a + [ TValue.From<TTensorArray>(FAccs_ta[i]) ];

    elements.AddRange(a);
    Result := elements.ToArray;
end;

function BodyItem.FromMergeVars(mergeVars: TArray<ITensorOrTensorArray>): BodyItem;
begin
    FI       := mergeVars[1] as TFTensor;
    FAccs_ta := [ mergeVars[2]  as TTensorArray ];
    Result := self;
end;

function BodyItem.Pack(sequences: TArray<TValue>): BodyItem;
begin
    FI       := sequences[0].AsType<TFTensor>;
    FAccs_ta := [ sequences[1].AsType<TTensorArray> ];

    Result := BodyItem.Create(FI, FAccs_ta);
end;

{ TTensorArray }

function TTensorArray.read<T>(index: T; name: string): TFTensor;
begin
 Result := nil;
end;

function TTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    Result := nil;
end;

{ TGraphTensorArray }

constructor TGraphTensorArray.Create(_dtype: TF_DataType; size: TFTensor; dynamic_size, clear_after_read: Boolean; tensor_array_name: string; _handle, _flow: TFTensor;
  _infer_shape: Boolean; _element_shape: PTFShape; _colocate_with_first_write_call: Boolean; _name: string);
begin
    Fclear_after_read := clear_after_read;
    Fdynamic_size     := dynamic_size;
    Fdtype            := _dtype;

    Fcolocate_with_first_write_call := _colocate_with_first_write_call;
    if Fcolocate_with_first_write_call then
        Fcolocate_with := TList<TFTensor>.Create;

    // Record the current static shape for the array elements. The element
    // shape is defined either by `element_shape` or the shape of the tensor
    // of the first write. If `infer_shape` is true, all writes checks for
    // shape equality.
    if _element_shape = nil then
    begin
        Finfer_shape   := _infer_shape;
        Felement_shape := TList<TFShape>.Create;
    end else
    begin
        Finfer_shape   := true;
        Felement_shape := TList<TFShape>.Create([ _element_shape ]);
    end;

    var vvalue := TValue.From< TArray<TFTensor> >([_handle, size, _flow]);
    TUtils.tf_with<TNameScope>( TOps.name_scope(_name, 'TensorArray', @vvalue),
        procedure(v1: TNameScope)
          begin
              var scope : string := v1.toString;
              if _handle <> nil then
              begin
                  Fhandle := _handle;
                  Fflow   := _flow;
              end else
              begin
                  var create : TFunc< Tuple<TFTensor, TFTensor> > := function:Tuple<TFTensor, TFTensor>
                        begin
                            Result :=  gen_data_flow_ops.tensor_array_v3(size, _dtype, _element_shape, dynamic_size, clear_after_read, _infer_shape, tensor_array_name, scope )
                        end;

                  // Construct the TensorArray with an empty device.  The first
                  // write into the TensorArray from a Tensor with a set device
                  // will retroactively set the device value of this op.
                  if _colocate_with_first_write_call then
                  begin
                      Tops.colocate_with(true);
                      var t1 := create;
                      Fhandle := t1.Value1;
                      Fflow   := t1.Value2;
                  end else
                  begin
                      var t1 := create;
                      Fhandle := t1.Value1;
                      Fflow   := t1.Value2;
                  end;
              end;
          end);
end;

function TGraphTensorArray.gather(indices: TFTensor; name: string): TFTensor;
var
  element_shape : TFShape;
  value         : TFTensor;
begin
    element_shape := TFShape.Null;

    if Felement_shape.Count > 0 then
       element_shape := Felement_shape[0];

    value := gen_data_flow_ops.tensor_array_gather_v3(Fhandle, indices, Fflow, Fdtype, @element_shape, name) ;

    //if (element_shape != null)
    //value.set_shape(-1, element_shape.dims);

    Result := value;
end;

function TGraphTensorArray.read<T>(index: T; name: string): TFTensor;
begin
    var value := gen_data_flow_ops.tensor_array_read_v3(Fhandle, constant_op.constant(TValue.From<T>(index)), Fflow, Fdtype, name);

    if Felement_shape <> nil then
        value.shape := Felement_shape[0].dims;

    Result := value;
end;

function TGraphTensorArray.scatter(indices, value: TFTensor; name: string): TTensorArray;
begin
    raise Exception.Create('Error Not Implemented scatter');
end;

function TGraphTensorArray.size(name: string): TFTensor;
begin
    Result := gen_data_flow_ops.tensor_array_size_v3(Fhandle, Fflow, name);
end;

function TGraphTensorArray.stack(name: string): TFTensor;
begin
    Tops.colocate_with(Fhandle);

    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle]);
    Result := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope(name, 'TensorArrayStack', @vvalue),
                        function(v1: TNameScope): TFTensor
                          begin
                              var Limit : TValue :=  size;
                              Result := gather(math_ops.range(0, @Limit), name);
                          end);
end;

function TGraphTensorArray.unstack(value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayUnstack', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              var num_elements : TFTensor := array_ops.shape(value)[0];
                              var Limit : TValue :=  num_elements;
                              Result := scatter(math_ops.range(0, @Limit), value, name);
                          end);
end;

function TGraphTensorArray.write(index, value: TFTensor; name: string): TTensorArray;
begin
    var vvalue := TValue.From< TArray<TFTensor> >([Fhandle,index, value]);
    Result := TUtils.tf_with<TNameScope,TTensorArray>( TOps.name_scope(name, 'TensorArrayWrite', @vvalue),
                        function(v1: TNameScope): TTensorArray
                          begin
                              _maybe_colocate_with(value);
                              var flow_out := gen_data_flow_ops.tensor_array_write_v3(Fhandle, index, value, Fflow, name);

                              Result := tensor_array_ops.build_ta_with_new_flow(Self, flow_out);
                          end);
end;

function TGraphTensorArray.write<T>(index: Integer; value: T; name: string): TTensorArray;
begin
    var value_tensor := Tops.convert_to_tensor(TValue.From<T>(value), DtInvalid, 'value', False, Fdtype );
    var index_tensor := Tops.convert_to_tensor(index, DtInvalid, 'index');
    Result := write(index_tensor, value_tensor, name);
end;

procedure TGraphTensorArray._maybe_colocate_with(value: TFTensor);
begin
    Fcolocate_with.Add(value);
end;

procedure TGraphTensorArray._merge_element_shape(shape: TFShape);
begin
    Felement_shape.Add(shape);
end;

end.


