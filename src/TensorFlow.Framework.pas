unit TensorFlow.Framework;
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

           Spring,
           Spring.Collections.Dictionaries,

           TF4D.Core.CApi,
           TensorFlow.DApiBase,
           TensorFlow.DApi,

           Oz.Pb.Classes,
           ProtoGen.opDef;

type
  /// <summary>
  /// Specifies a TensorFlow value type.
  /// </summary>
  TypeSpec = class

  end;

  /// <summary>
  /// Describes a dense object with shape, dtype, and name.
  /// </summary>
  DenseSpec = class(TypeSpec)
     private

     protected
        Fshape : TFShape;
        Fdtype : TF_DataType;
        Fname  : string;
     public

       constructor Create(_shape: TFShape; _dtype: TF_DataType = TF_FLOAT; _name: string = '');
       function ToString: string; override;

       property shape : TFShape     read Fshape;
       property dtype : TF_DataType read Fdtype;
       property name  : string      read Fname;
  end;

  TensorSpec  = class(DenseSpec)
     public
       constructor Create(shape: TFShape; dtype: TF_DataType = TF_FLOAT; name: string = '');
       function _unbatch: TensorSpec;
       function _batch(dim: Integer = -1):  TensorSpec;
  end;

  smart_module = class
    class function smart_cond(_pred: TFTensor; true_fn : TFunc< TArray<TFTensor> > = nil; false_fn : TFunc< TArray<TFTensor> > = nil; name: string = ''): TArray<TFTensor>; overload;
    class function smart_cond(_pred: Boolean; true_fn : TFunc<TFTensor> = nil; false_fn : TFunc<TFTensor> = nil; name: string = ''): TFTensor; overload;

    class function smart_constant_value(_pred: TFTensor) : Nullable<Boolean>;
  end;

  common_shapes = class
     public
       class function has_fully_defined_shape(tensor: TFTensor): Boolean;
       class function rank(tensor: TFTensor): Integer;
       /// <summary>
       /// Returns the broadcasted shape between `shape_x` and `shape_y
       /// </summary>
       /// <param name="shape_x"></param>
       /// <param name="shape_y"></param>
       class function broadcast_shape(shape_x: TFTensor; shape_y: TFTEnsor): TFtensor;
       /// <summary>
       /// Helper functions for is_broadcast_compatible and broadcast_shape.
       /// </summary>
       /// <param name="shape_x"> A `Shape`</param>
       /// <param name="shape_y"> A `Shape`</param>
       /// <return> Returns None if the shapes are not broadcast compatible,
       /// a list of the broadcast dimensions otherwise.
       /// </return>
       class function _broadcast_shape_helper(shape_x: TFTensor; shape_y: TFTEnsor): TFtensor;
  end;

  // tensor_shape move to Tensorflow.tensor.Ragged  for Circular reference

  /// <summary>
  /// Abstract base class for Tensor-like objects that are composed from Tensors.
  /// </summary>
  CompositeTensor = class abstract
  end;

  /// <summary>
  /// A sparse representation of a set of tensor slices at given indices.
  /// </summary>
  IndexedSlices = {class(CompositeTensor)} record
    private
       Fvalues     : TFTensor;
       Findices    : TFTensor;
       Fdense_shape: TFTensor;

       function GetDevice: string;
       function GetDtype: TF_DataType;
       function GetGraph: TFGraph;
       function GetName: string;
       function GetOp: TFOperation;
    public
        constructor Create(_values: TFTensor; _indices: TFTensor; _dense_shape: TFTensor = nil);
        class operator implicit(iSlices: IndexedSlices): TFTensor;
        class operator implicit(tTEnsor: TFTensor): IndexedSlices;

        property values      : TFTensor    read Fvalues;
        property indices     : TFTensor    read Findices;
        property dense_shape : TFTensor    read Fdense_shape;
        property name        : string      read GetName;
        property device      : string      read GetDevice;
        property op          : TFOperation read GetOp ;
        property dtype       : TF_DataType read GetDtype;
        property graph       : TFGraph     read GetGraph;
  end;

  op_def_registry = class
     private
       class var registered_ops  : TDictionary<string,TOpDef>;
     public

     class function get_registered_ops: TDictionary<string,TOpDef> ;
     class function GetOpDef(tipo : string): TOpDef;
   end;

   random_seed = record
     private
        const DEFAULT_GRAPH_SEED = 87654321;
        class var Fgraph_to_seed_dict : TDictionary<string,Integer> ;
     public
       class function get_seed(op_seed: TNullableInteger) : Tuple<TNullableInteger,TNullableInteger>; static;
       class function get_seed_tensor(op_seed: TNullableInteger) : Tuple<TFTensor,TFTensor>; static;
   end;

implementation
   uses System.Classes,
        System.Math,
        TensorFlow.Constant_op,
        Tensorflow,
        TensorFlow.Ops,
        Tensorflow.Utils,
        Tensorflow.NameScope,
        Tensorflow.array_ops,
        Tensorflow.math_ops,
        TensorFlow.control_flow_ops,

        NumPy.NDArray;

{ op_def_registry }

class function op_def_registry.GetOpDef(tipo: string): TOpDef;
begin
    var ops := get_registered_ops;
    Result  := ops[tipo];
end;

class function op_def_registry.get_registered_ops: TDictionary<string, TOpDef>;
var
  Loader: TpbLoader;

begin
    if not Assigned(registered_ops)  then
       registered_ops := TDictionary<string, TOpDef>.Create;

    // double validation to avoid multi-thread executing
    if registered_ops.Count > 0 then
        Exit(registered_ops);

    var buffer := TFBuffer.Create( TF_GetAllOpList );
    var op_list : TOpList;

    var aBuf := buffer.toArray;
    Loader.Init;
    Loader.Pb.Init(@aBuf[0],Length(aBuf),false);

    Loader.LoadOpList(op_list);

    for var i := 0 to op_list.Ops.Count - 1 do
    begin
       var op_def : TOpDef := op_list.Ops[i]^;
       registered_ops.AddOrSetValue(op_def.Name,op_def);
    end;

    Result := registered_ops
end;

{ common_shapes }

class function common_shapes.broadcast_shape(shape_x, shape_y: TFTEnsor): TFtensor;
begin

    var return_dims := _broadcast_shape_helper(shape_x, shape_y);
    // return tensor_shape(return_dims);
    raise TFException.Create('Not Finite NumberException');
end;

class function common_shapes._broadcast_shape_helper(shape_x, shape_y: TFTEnsor): TFtensor;
begin
    raise TFException.Create('Not Finite NumberException');
end;

class function common_shapes.has_fully_defined_shape(tensor: TFTensor): Boolean;
begin
   Result := tensor.shape.IsFullyDefined;
end;

class function common_shapes.rank(tensor: TFTensor): Integer;
begin
   Result := tensor.rank;
end;

{ random_seed }

class function random_seed.get_seed(op_seed: TNullableInteger): Tuple<TNullableInteger, TNullableInteger>;
var
 seed: Integer;
begin
    var global_seed: Nullable<Integer>;

    if tf.executing_eagerly then
        global_seed := tf.Context.global_seed
    else
        global_seed := Tops.get_default_graph.seed;
    if global_seed.HasValue then
    begin
        if  not op_seed.HasValue then
        begin
            if tf.executing_eagerly then
            begin
                op_seed := tf.Context.internal_operation_seed;
            end
            else begin
                 if  not Fgraph_to_seed_dict.TryGetValue(Tops.get_default_graph.graph_key, seed) then
                    seed := 0;
                 Fgraph_to_seed_dict.AddOrSetValue(Tops.get_default_graph.graph_key, seed + 1);
                 op_seed := seed;
            end;
        end;
        Result := Tuple<TNullableInteger, TNullableInteger>.Create(global_seed, op_seed);
        Exit;
    end;
    if op_seed <> nil then  Result :=  Tuple<TNullableInteger, TNullableInteger>.Create(DEFAULT_GRAPH_SEED, op_seed)
    else                    Result :=  Tuple<TNullableInteger, TNullableInteger>.Create(0, 0)
end;

class function random_seed.get_seed_tensor(op_seed: TNullableInteger): Tuple<TFTensor, TFTensor>;
begin
    var tseed := get_seed(op_seed);
    var seed := tseed.Value1;
    var seed2:= tseed.Value2;

    var _seed, _seed2 : TFTensor;
    if seed = nil then  _seed := constant_op.constant(Int64(0), DtInvalid, 'seed')
    else                _seed := constant_op.constant(Int64(seed.Value), DtInvalid, 'seed');
    if seed2 = nil then
        _seed2 := constant_op.constant(Int64(0), DtInvalid, 'seed2')
    else begin
        _seed2 := TUtils.tf_with<TNameScope,TFTensor>( TOps.name_scope('seed2'),
                          function(v1: TNameScope): TFTensor
                            begin
                                _seed2 := constant_op.constant(Int64(seed2.Value));
                                Result :=  array_ops.where_v2(
                                  math_ops.logical_and(
                                      math_ops.equal(_seed, Int64(0)),
                                      math_ops.equal(_seed2, Int64(0))),
                                  constant_op.constant( Power(2,31) - 1),
                                  _seed2,
                                  v1.ToString);
                            end );
    end;
    Result := Tuple<TFTensor, TFTensor>.Create(_seed, _seed2);
end;

{ IndexedSlices }

constructor IndexedSlices.Create(_values, _indices, _dense_shape: TFTensor);
begin
    Fvalues     := _values;
    Findices    := _indices;
    Fdense_shape:= _dense_shape;
    Fvalues.Tag := TValue.From<IndexedSlices>(Self);
end;

function IndexedSlices.GetDevice: string;
begin
    Result := Fvalues.Device;
end;

function IndexedSlices.GetDtype: TF_DataType;
begin
    Result := Fvalues.Dtype;
end;

function IndexedSlices.GetGraph: TFGraph;
begin
    Result := Fvalues.graph;
end;

function IndexedSlices.GetName: string;
begin
    Result := Fvalues.Name;
end;

function IndexedSlices.GetOp: TFOperation;
begin
    Result := Fvalues.Op;
end;

class operator IndexedSlices.implicit(iSlices: IndexedSlices): TFTensor;
begin
     Result := iSlices.values;
end;

class operator IndexedSlices.implicit(tTEnsor: TFTensor): IndexedSlices;
begin
    Result := tTEnsor.Tag.AsType<IndexedSlices>;
end;

{ smart_module }

class function smart_module.smart_cond(_pred: Boolean; true_fn, false_fn: TFunc<TFTensor>; name: string): TFTensor;
begin
    if _pred then Result := true_fn
    else          Result := false_fn;
end;

class function smart_module.smart_cond(_pred: TFTensor; true_fn, false_fn: TFunc<TArray<TFTensor>>; name: string): TArray<TFTensor>;
var
  pred_value :Nullable<Boolean>;
begin
    pred_value := smart_module.smart_constant_value(_pred);
    if pred_value.HasValue then
    begin
        var res : TArray<TFTensor>;
        if pred_value.Value then res := true_fn
        else                     res := false_fn;
        Result := res;
    end else
    begin
        Result := control_flow_ops.cond<TFTensor>(_pred, true_fn, false_fn, name);
    end;
end;

class function smart_module.smart_constant_value(_pred: TFTensor): Nullable<Boolean>;
begin
    var pred_value := TUtils.constant_value(_pred);
    if pred_value = nil then
    begin
        var res : TArray<Pointer> ;
        SetLength(res, _pred.op.NumOutputs) ;

        var evaluated := TF_TryEvaluateConstant(_pred.graph.handle, _pred._as_tf_output, @res[0], tf.Status.Handle);
        if (evaluated= 0) or (TF_GetCode(tf.Status.Handle) <> TF_Code.TF_OK)  then
            Result := nil
        else
            raise TFException.Create('Not Implemented');
    end;

    Result := Boolean(NDArray(pred_value));
end;

{ DenseSpec }

constructor DenseSpec.Create(_shape: TFShape; _dtype: TF_DataType; _name: string);
begin
    Fshape := _shape;
    Fdtype := _dtype;
    Fname  := _name;
end;

function DenseSpec.ToString: string;
begin
   Result := Format('shape=%s, dtype=%s, name=%s',[Fshape.ToString, Tdtypes.ToString(Fdtype), Fname] )
end;

{ TensorSpec }

constructor TensorSpec.Create(shape: TFShape; dtype: TF_DataType; name: string);
begin
    inherited Create(shape, dtype, name);
end;

function TensorSpec._batch(dim: Integer): TensorSpec;
begin
    var shapes := shape.dims;
    shapes := [dim] + shapes ;
    Result := TensorSpec.Create(shapes, Fdtype);
end;

function TensorSpec._unbatch: TensorSpec;
begin
    if Fshape.ndim = 0 then
       raise Exception.Create('Unbatching a tensor is only supported for rank >= 1');
    var a  := Fshape.dims;
    Delete(a,0,1) ;
    Result := TensorSpec.Create(a, Fdtype);
end;

initialization
begin
    random_seed.Fgraph_to_seed_dict := TDictionary<string,Integer>.Create;
end;

finalization
begin
    random_seed.Fgraph_to_seed_dict.Free;
end;

end.






