unit TensorFlow.Framework;

interface
      uses System.SysUtils, spring.Collections.Dictionaries,

           TensorFlow.LowLevelAPI,
           TensorFlow.DApiBase,

           Oz.Pb.Classes,
           ProtoGen.opDef;

type
   op_def_registry = class
     private
       class var registered_ops  : TDictionary<string,TOpDef>;
     public

     class function get_registered_ops: TDictionary<string,TOpDef> ;
     class function GetOpDef(tipo : string): TOpDef;
   end;

implementation
   uses System.Classes;

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
    //Loader.Pb.SaveToFile('testpb1.pb') ;
    //loaderpb := Loader.pb.From( buffer.toArray );

    Loader.LoadOpList(op_list);


    var l : TStringList := TStringList.Create;
    var cc := op_list.Ops.Count;
    for var i := 0 to op_list.Ops.Count - 1 do
    begin
       var op_def : TOpDef := op_list.Ops[i]^;
       registered_ops.AddOrSetValue(op_def.Name,op_def);
       l.Add(op_def.Name)
    end;
    //l.SaveToFile('Oplist.txt');

    Result := registered_ops
end;

end.
