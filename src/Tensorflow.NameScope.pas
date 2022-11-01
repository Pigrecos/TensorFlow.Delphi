unit Tensorflow.NameScope;
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
           Spring, spring.Collections.Lists,
           TF4D.Core.CApi,
           TensorFlow.DApiBase,
           TensorFlow.Context;
type
  /// <summary>
  /// Returns a context manager that creates hierarchical names for operations.
  /// </summary>
  TNameScope = class(TInterfacedObject,ITensorFlowObject)
    private
      function enter_eager_name_scope(ctx: TContext; name:TF_TString): Tuple<TF_TString,TF_TString>;
    public
      _name          : TF_TString;
      _default_name  : TF_TString;
      _values        : TValue;
      scope_name     : TF_TString;
      old_scope_name : TF_TString;
      _skip_on_eager : boolean;

      constructor Create(name: TF_TString; default_name : TF_TString = ''; values : PValue = nil; skip_on_eager : Boolean = True);
      function ToString: TF_TString; reintroduce;
      procedure _Enter_;
      procedure _Exit_;
  end;

implementation
     uses Tensorflow, TensorFlow.DApi,TensorFlow.Ops;

{ TNameScope }

constructor TNameScope.Create(name, default_name: TF_TString; values: PValue; skip_on_eager: Boolean);
begin
    _name := name;
    _default_name := default_name;
    if values <> nil then
      _values := values^
    else
      _values := default(TValue);
    _skip_on_eager := skip_on_eager;
end;

function TNameScope.enter_eager_name_scope(ctx: TContext; name: TF_TString): Tuple<TF_TString, TF_TString>;
begin
    if _skip_on_eager then
        Exit(Tuple<TF_TString, TF_TString>.Create('',''));
    if name = '' then
        name := _default_name;
    var scope_name := name;
    var old_name   := ctx.ScopeName;
    // A trailing slash breaks out of nested name scopes, indicating a
    // fully specified scope name, for compatibility with Graph.name_scope.
    if not string(name).EndsWith('/') then
    begin
        scope_name := name + '/';
        if not string.IsNullOrEmpty(old_name) then
            scope_name := AnsiString(old_name) + scope_name;
    end;
    ctx.ScopeName := string(scope_name);

    Result  := Tuple<TF_TString, TF_TString>.Create(scope_name, old_name);
end;

function TNameScope.ToString: TF_TString;
begin
    Result := scope_name;
end;

procedure TNameScope._Enter_;
var
  tRes : Tuple<TF_TString,TF_TString>;
begin
    if tf.Context.executing_eagerly then
    begin
        tRes := enter_eager_name_scope(tf.Context, _name);
        scope_name     := tRes.Value1;
        old_scope_name := tRes.Value2;
    end else
    begin
        if _name = '' then
          _name := _default_name;
        var g : TFGraph := nil;
        if (not _values.IsEmpty) and (_values.IsType< TList<TFTensor> >) then
        begin
            var vList : TList<TFTensor> := _values.AsType< TList<TFTensor> > ;
            g := TOps._get_graph_from_inputs(vList.ToArray);
        end
        else if (not _values.IsEmpty) and (_values.IsType< TArray<TFTensor> >) then
        begin
            var vArray : TArray<TFTensor> := _values.AsType<  TArray<TFTensor> >;
            g := TOps._get_graph_from_inputs(vArray);
        end;
        if g = nil then
            g := TOps.get_default_graph;
        old_scope_name := g._name_stack;
        scope_name     := g.name_scope(_name);
    end;
end;

procedure TNameScope._Exit_;
begin
    if tf.Context.executing_eagerly then
        tf.Context.ScopeName := string(old_scope_name)
    else
        TOps.get_default_graph._name_stack := old_scope_name;
end;

end.
