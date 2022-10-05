unit Tensorflow.Graph;

interface
     uses System.SysUtils,
          TensorFlow.DApi,
          Spring.Collections.Stacks;

const
  EAGER_CONST_THRESHOLD : Integer = 128;

type

  /// <summary>
  ///     Serves as a stack for determining current default graph.
  /// </summary>
  DefaultGraphStack = class
     private
      F_stack : TStack<TFGraph>;
      F_global_default_graph : TFGraph;
     public
      constructor Create;
      function  get_default: TFGraph;
      function  get_controller(g: TFGraph): TFGraph;
      function  peak_controller: TFGraph;
      procedure pop;
      procedure reset;

      property global_default_graph : TFGraph         read F_global_default_graph;
      property stack                : TStack<TFGraph> read F_stack;
  end;

  /// <summary>
  /// Graph representing a function body.
  /// </summary>
  TFuncGraph = class(TFGraph)
     private

     protected

     public

       function capture(tensor: TFTensor; name: string = ''; shape: PTFShape = nil): TFTensor;
  end;




implementation

{ DefaultGraphStack }

constructor DefaultGraphStack.Create;
begin
    inherited Create;
    F_stack := TStack<TFGraph>.Create;
end;

function DefaultGraphStack.get_controller(g: TFGraph): TFGraph;
begin
    F_stack.Push(g);
    Result := g;
end;

function DefaultGraphStack.get_default: TFGraph;
begin
    if      F_stack.Count > 0             then  Exit(F_stack.Peek)
    else if F_global_default_graph <> nil then  Exit(F_global_default_graph)
    else                                        F_global_default_graph := TFGraph.Create;
    Result := F_global_default_graph;
end;

function DefaultGraphStack.peak_controller: TFGraph;
begin
    if F_stack.Count = 0 then
        Exit(nil);
    Result := F_stack.Peek;
end;

procedure DefaultGraphStack.pop;
begin
    F_stack.Pop
end;

procedure DefaultGraphStack.reset;
begin
    F_stack.Clear;
    F_global_default_graph := nil;
end;

{ TFuncGraph }

function TFuncGraph.capture(tensor: TFTensor; name: string; shape: PTFShape): TFTensor;
begin

end;

end.
