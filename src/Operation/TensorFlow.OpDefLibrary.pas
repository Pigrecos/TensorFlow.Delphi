unit TensorFlow.OpDefLibrary;
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
    uses   System.SysUtils, System.Generics.Collections,System.Rtti, System.TypInfo,
           Spring.Collections.Dictionaries,

           Spring.Collections.Stacks,
           spring.Collections.Enumerable,
           Spring,

           TF4D.Core.CApi,
           TensorFlow.DApiBase,
           TensorFlow.DApi,
           TensorFlow.Context,
           Tensorflow.Utils,

           Protogen.tensorShape,
           ProtoGen.attrValue,
           ProtoGen.types,
           ProtoGen.opDef;

type

OpDefLibrary = class
    private
      class function SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
      class function _IsListParameter(arg: TArgDef): Boolean;
      class function _IsListValue(v: TValue): Boolean;
      class procedure SetAttrs(op_type_name : string;
                               input_arg    : TArgDef;
                               op_def       : TOpDef;
                               var attrs        : TDictionary<string, TValue>;
                               var inferred_from: TDictionary<string, TValue>;
                               types        : TList<TF_DataType>;
                               base_types   : TList<TF_DataType>;
                               var input_types  : TList<TF_DataType>;
                               values       : TValue);
    public
      class function _MakeType(v: TF_DataType; attr_def: TAttrDef): TDataType;
      class function _MakeShape(shape: TFShape; attr_def: TAttrDef): TTensorShapeProto;
      class function _apply_op_helper(op_type_name: string;name: string = ''; args : TArray<TParameter> = nil): TFOperation; overload;
      class function _apply_op_helperDict(op_type_name: string; name: string = ''; keywords: TDictionary<string, TValue> = nil): TFOperation;overload;

  end;



implementation
     uses TensorFlow.Ops,  Oz.Pb.Classes,oz.Pb.StrBuffer;

{ OpDefLibrary }

class function OpDefLibrary._IsListParameter(arg: TArgDef): Boolean;
begin
     if not string.IsNullOrEmpty(arg.NumberAttr)  then
       Exit(True)
     else if not string.IsNullOrEmpty(arg.TypeListAttr) then
       Exit(True)
     else
       Result := False;
end;

class function OpDefLibrary._IsListValue(v: TValue): Boolean;
begin
    Result := v.IsArray;
end;

class procedure OpDefLibrary.SetAttrs(op_type_name : string;input_arg: TArgDef; op_def: TOpDef; var attrs, inferred_from: TDictionary<string, TValue>; types,
                                      base_types: TList<TF_DataType>;var input_types: TList<TF_DataType>; values: TValue);
begin
    var input_name := input_arg.Name;

    if  not string.IsNullOrEmpty(input_arg.NumberAttr) then
    begin
        if attrs.ContainsKey(input_arg.NumberAttr) then
        begin

        end else
        begin
            if(values.IsArray) and (values.GetArrayElement(0).TypeInfo = TypeInfo(TFTensor)) then
            begin
                var num_attr : TAttrDef;
                for var i := 0 to op_def.Attrs.Count -1 do
                begin
                    if op_def.Attrs.Items[i]^.Name = input_arg.NumberAttr then
                    begin
                        num_attr := op_def.Attrs.Items[i]^;
                        Break;
                    end;
                end;
                if (num_attr.HasMinimum) and (values.GetArrayLength < num_attr.Minimum) then
                    raise Exception.Create(Format('"%s" to "%s" Op with length %d shorter than minimum length %d',[input_name,op_type_name,values.GetArrayLength,num_attr.Minimum]));

                attrs.AddOrSetValue(input_arg.NumberAttr,TValue.From<Int64>(values.GetArrayLength));
                inferred_from.AddOrSetValue(input_arg.NumberAttr, TValue.From<string>(input_name));
            end;
        end;
        // All tensors must have the same base type.
        if input_arg.&Type <> TDataType.DT_INVALID then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr, TValue.From<Integer>( Ord(base_types[0]) ));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TValue.From<string>(input_name));
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeAttr) then
    begin
        var attr_value := base_types[0];
        if attrs.ContainsKey(input_arg.TypeAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeAttr,TValue.From<Integer>( Ord(attr_value) ));
            inferred_from.AddOrSetValue(input_arg.TypeAttr,TValue.From<string>(input_name));
        end;
    end
    else if not string.IsNullOrEmpty(input_arg.TypeListAttr) then
    begin
        var attr_value := base_types;
        if attrs.ContainsKey(input_arg.TypeListAttr) then
        begin

        end else
        begin
            attrs.AddOrSetValue(input_arg.TypeListAttr, TValue.From< TList<TF_DataType> >(attr_value) );
            inferred_from.AddOrSetValue(input_arg.TypeListAttr,TValue.From<string>(input_name));
        end;
    end;
    if input_arg.IsRef then
        input_types.AddRange(types)
    else
        input_types.AddRange(base_types);
end;

class function OpDefLibrary._apply_op_helper(op_type_name: string; name: string; args: TArray<TParameter>): TFOperation;
begin
    if args = nil then
       Result := _apply_op_helperDict(op_type_name, name)
    else
      Result := _apply_op_helperDict(op_type_name, name, TUtils.ConvertToDict(args));
end;

class function OpDefLibrary._apply_op_helperDict(op_type_name: string; name: string; keywords: TDictionary<string, TValue>): TFOperation;
var
  aObj        : TArray<TValue>;
  g           : TFGraph;
  attrs       : TDictionary<string, TValue>;
  attr_protos : TDictionary<string, TAttrValue>;
  inputs      : TList<TFTensor>;
  input_types : TList<TF_DataType>;
  values      : TValue;

  op_def      : TOpDef;
begin
    if keywords = nil then  aObj := []
    else                    aObj := keywords.Values.ToArray;

    g := TOps._get_graph_from_inputs(aObj);
    op_def := g.GetOpDef(op_type_name);

    // Default name if not specified.
    if String.IsNullOrEmpty(name) then
        name := op_type_name;

    (*// Check for deprecation
    if (op_def.Deprecation != null && op_def.Deprecation.Version > 0)
    {
    }*)

    var default_type_attr_map := TDictionary<string, TValue>.Create;
    for var attr_def in op_def.Attrs do
    begin
        if attr_def.&Type <> 'type' then continue;
        var key := attr_def.Name;
        if attr_def.DefaultValue.Value.tag =  attr_def.DefaultValue.ftType then
        begin
            default_type_attr_map.AddOrSetValue(key, attr_def.DefaultValue.Value.value);
        end;
    end;

    attrs       := TDictionary<string, TValue>.Create;
    inputs      := TList<TFTensor>.Create;
    input_types := TList<TF_DataType>.Create;
    values      := nil;

    g.as_default;

    var scope := TOps.name_scope(name);
    scope._Enter_;

    var inferred_from := TDictionary<string, TValue>.Create;
    var base_types    := TList<TF_DataType>.Create;
    var types         := TList<TF_DataType>.Create;
    var _scope_name   := scope.ToString;

    // Perform input type inference
    for var i := 0 to op_def.InputArgs.Count - 1 do
    begin
        var input_arg : TArgDef := op_def.InputArgs[i]^;
        var input_name:= input_arg.Name;

        if keywords.ContainsKey(input_name) then
            values := keywords[input_name]
        else if keywords.ContainsKey(input_name + '_') then
        begin
            input_name := input_name + '_';
            values     := keywords[input_name];
        end
        else if keywords.ContainsKey('input_'+ IntTostr(i)) then
        begin
            values := keywords['input_'+ IntTostr(i)];
        end
        else
            raise Exception.Create('No argument for input ' + input_name);
        // Goals:
        // * Convert values to Tensors if it contains constants.
        // * Verify that values is a list if that matches the input_arg's
        // type.
        // * If the input_arg's type is determined by attrs, either set
        // those attrs and validate those attr values are legal (if
        // they have not yet been set) or validate the input matches
        // the type indicated by the attrs (if they have already been
        // inferred via an earlier input).
        // * If the input_arg has an explicit type, make sure the input
        // conforms.

        var dtype        : TDataType := TDataType.DT_INVALID;
        var default_dtype: TDataType := TDataType.DT_INVALID;

        if _IsListParameter(input_arg) then
        begin
            if not _IsListValue(values) then
                raise Exception.Create('Expected list for {input_name} argument to {op_type_name} Op, not {values}.');

            if input_arg.&Type <> TDataType.DT_INVALID then
                dtype := TDataType(input_arg.&Type)
            else if not String.IsNullOrEmpty(input_arg.NumberAttr) then
            begin
                if attrs.ContainsKey(input_arg.TypeAttr) then
                    dtype := TDataType( attrs[input_arg.TypeAttr].AsInteger )
                else begin
                   var aEle := values.GetArrayElement(0);
                   if aEle.IsType<TFTensor> then
                       dtype := TDtypes.as_datatype_enum(values.GetArrayElement(0).asType<TFTensor>.Dtype)
                   else if aEle.IsObject then
                   begin
                       for var t := 0 to values.GetArrayLength - 1 do
                       begin
                          var item := values.GetArrayElement(t);
                          if item.IsType<TFTensor> then
                          begin
                              dtype := TDtypes.as_datatype_enum(item.AsType<TFTensor>.Dtype);
                          end;
                       end;
                   end else
                       raise Exception.Create('can''t infer the dtype for {values.GetType()}');
                end;
                if (dtype = TDataType.DT_INVALID) and (default_type_attr_map.ContainsKey(input_arg.TypeAttr)) then
                    default_dtype := TDataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);
            end;

            if ( not input_arg.IsRef) and (dtype <> TDataType.DT_INVALID) then
                dtype := Tdtypes.as_base_dtype(dtype);

            var RetVal := TOps.internal_convert_n_to_tensor(values.AsType< TArray<TFTensor> >,
                                                            Tdtypes.as_tf_dtype(dtype),
                                                            input_arg.Name,
                                                            Tdtypes.as_tf_dtype(default_dtype),
                                                            input_arg.IsRef);
            values := TValue.From< TArray<TFTensor> >(RetVal);
        end else
        begin
            if input_arg.&Type <> TDataType.DT_INVALID then
                dtype := TDataType(input_arg.&Type)
            else if attrs.ContainsKey(input_arg.TypeAttr) then
                dtype := TDataType(attrs[input_arg.TypeAttr].AsInteger)
            else if (TUtils.isinstance(values, TypeInfo(string))) and (dtype = TDataType.DT_INVALID) then
                dtype := TDataType.DT_STRING
            else if default_type_attr_map.ContainsKey(input_arg.TypeAttr) then
                default_dtype := TDataType(default_type_attr_map[input_arg.TypeAttr].AsType<Integer>);

            var value := TOps.convert_to_tensor(values,
                                                Tdtypes.as_tf_dtype(dtype),
                                                input_name,
                                                input_arg.IsRef,
                                                Tdtypes.as_tf_dtype(default_dtype));

            values := TValue.From< TArray<TFTensor> >([ value ] );
        end;

        if (values.IsArray) and ( values.GetArrayElement(0).IsType<TFTensor> ) then
        begin
            var values2 : TArray<TFTensor> := values.AsType< TArray<TFTensor> >;
            inputs.AddRange(values2);
            base_types.Clear;
            for var j := 0 to Length(values2) -1 do
            begin
                types.Add(values2[j].dtype);
                base_types.Add( Tdtypes.as_base_dtype(values2[j].dtype) ) ;
            end;
        end
        else
           raise Exception.Create('NotImplementedException("_IsListParameter")');

        SetAttrs(op_type_name,
                 input_arg,
                 op_def,
                 attrs,
                 inferred_from,
                 types,
                 base_types,
                 input_types,
                 values);
    end;

    // Process remaining attrs
    for var attr in op_def.Attrs do
    begin
        if keywords.ContainsKey(attr.Name) then
        begin
            attrs.AddOrSetValue(attr.Name, keywords[attr.Name] );
        end
    end;
    // Convert attr values to AttrValue protos.
    attr_protos := TDictionary<string, TAttrValue>.Create;
    for var  attr_def in op_def.Attrs do
    begin
        var key := attr_def.Name;
        if attrs.ContainsKey(key) then
        begin
            attr_protos.AddOrSetValue(key, SetAttrValue(op_def, attr_def^, attrs[key] ) );
        end else
        begin
            if attr_def.DefaultValue.Value.value.AsObject = nil then
            begin
                raise Exception.Create('Missing required positional argument ' + key);
            end;
        end;
    end;
    attrs.Clear();

    // Determine output types (possibly using attrs)
    var output_types := TList<TF_DataType>.Create;

    for var arg in op_def.OutputArgs do
    begin
        types := TList<TF_DataType>.Create;
        if not string.IsNullOrEmpty(arg.NumberAttr) then
        begin
        end
        else if not string.IsNullOrEmpty(arg.TypeAttr) then
        begin
            types := TList<TF_DataType>.Create;
            types.Add( TF_DataType(attr_protos[arg.TypeAttr].Value.value.AsInteger) );
        end;
        if arg.IsRef then
        begin
            var aTemp : TArray<TF_DataType> := [];
            for var i := 0 to types.Count - 1 do
            begin
                aTemp := aTemp + [ Tdtypes.as_ref(types[i]) ];
            end;
            types.Clear;
            types.Free;
            types := TList<TF_DataType>.Create(aTemp);
        end;
        output_types.AddRange(types);
    end;

    // We add an explicit colocation constraint between
    // the newly created op and any of its reference-typed inputs.
   (* var must_colocate_inputs = zip(op_def.InputArg, inputs)
        .Where(x => x.Item1.IsRef)
        .Select(x => x.Item2)
        .ToArray();
    _MaybeColocateWith(must_colocate_inputs);
    *)

    // Add Op to graph
    var ret_op := g.create_op(op_type_name,
                              inputs.ToArray,
                              output_types.ToArray,
                              input_types.ToArray,
                              _scope_name,
                              attr_protos,
                              @op_def);

    scope._exit_;

    g.gExit;

    Result := ret_op;
end;

class function OpDefLibrary._MakeShape(shape: TFShape; attr_def:TAttrDef): TTensorShapeProto;
begin
    Result := TUtils.as_shape_proto(shape);
end;

class function OpDefLibrary._MakeType(v: TF_DataType; attr_def:TAttrDef): TDataType;
begin
    Result :=  Tdtypes.as_datatype_enum( Tdtypes.as_base_dtype(v) );
end;

class function OpDefLibrary.SetAttrValue(op_def: TOpDef; attr_def: TAttrDef; value: TValue): TAttrValue;
var
   v          : TpbOneof;
   attr_value : TAttrValue;

begin
    attr_value.Init;

    if attr_def.&Type.StartsWith('list(') then
    begin
        if attr_def.HasMinimum then
        begin
            v.tag := TAttrValue.ftList;
            var v1 : TListValue; v1.Init;
            v.value := TValue.From<TListValue>(v1);

            attr_value.Value := v;
        end;
    end;

    if attr_def.&Type = 'string' then
    begin
         v.tag   := TAttrValue.ftS;
         var b   := TEncoding.UTF8.GetBytes( value.AsString );
         v.value := TValue.From< TBytes >(b);

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'type' then
    begin
        var tipo := _MakeType( TF_DataType(value.AsType<Integer>), attr_def);
        v.tag   := TAttrValue.ftType;
        v.value :=  TValue.From<Integer>( Ord(tipo));

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(type)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TList<TF_DataType> >;
        for var i := 0 to l.Count - 1 do
        begin
            var d : TDataType := _MakeType(l[i],attr_def);
            v1.Types.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'list(int)' then
    begin
        var v1 : TListValue;
        v1 := attr_value.Value.value.AsType<TListValue>;

        var l := value.AsType< TArray<Integer> >;
        for var i := 0 to Length(l) - 1 do
        begin
            var d : Int64 := l[i];
            v1.&Is.Add(@d)
        end;
        v.value := TValue.From<TListValue>(v1);
        v.tag   := TAttrValue.ftList;
        attr_value.Value := v;
    end
    else if attr_def.&Type = 'bool' then
    begin
        v.tag   := TAttrValue.ftB;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'float' then
    begin
        v.tag   := TAttrValue.ftF;
        v.value := value;

        attr_value.Value := v;
    end
    else if attr_def.&Type = 'int' then
    begin
        v.tag   := TAttrValue.ftI;
        v.value := value;

        attr_value.Value := v;
        if (attr_def.HasMinimum)and ( v.value.AsInt64 < attr_def.Minimum) then
           raise Exception.Create(Format('Attr %s of %s Op passed %d less than minimum $d.',[attr_def.Name,op_def.Name,v.value.AsInt64,attr_def.Minimum]));
    end
    else if attr_def.&Type = 'shape' then
    begin
         if (value.IsEmpty) and ( not attr_def.DefaultValue.Value.value.IsEmpty) then
             attr_value.Value := attr_def.DefaultValue.Value;

         if value.IsType<TArray<Integer> > then
         begin
             var v1 := TUtils.as_shape_proto(value.AsType<TArray<Integer>>) ;
             v.tag  := TAttrValue.ftShape;
             v.value:= TValue.From<TTensorShapeProto>(v1) ;
             attr_value.Value := v;
         end
         else if value.IsType< TArray<Int64> > then
         begin
             var v1 := TUtils.as_shape<Int64>(value.AsType<TArray<Int64>>) ;
             v.tag  := TAttrValue.ftShape;
             v.value:= TValue.From<TTensorShapeProto>(v1) ;
             attr_value.Value := v;
         end
         else if value.IsType< TFShape > then
         begin
             var v1 := TUtils.as_shape<Integer>(value.AsType<  TFShape >) ;
             v.tag  := TAttrValue.ftShape;
             v.value:= TValue.From<TTensorShapeProto>(v1) ;
             attr_value.Value := v;
         end;
    end
    else if attr_def.&Type = 'list(shape)' then
    begin
          var v1 : TListValue;
          v1 := attr_value.Value.value.AsType<TListValue>;

          var l := value.AsType< TArray<TFShape> >;
          for var i := 0 to Length(l) - 1 do
          begin
              var d : TTensorShapeProto := _MakeShape(l[i],attr_def);
              v1.Shapes.Add(@d)
          end;
          v.value := TValue.From<TListValue>(v1);
          v.tag   := TAttrValue.ftList;
          attr_value.Value := v;
    end else
    begin
        raise Exception.Create('SetAttrValue: can''t not convert attr_def.Type {attr_def.Type} to protos.');
    end;


    Result := attr_value;
end;

end.
