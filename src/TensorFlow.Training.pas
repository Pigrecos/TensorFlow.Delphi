unit TensorFlow.Training;
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
          Spring.Collections.Enumerable,

          TF4D.Core.CApi,
          TensorFlow.DApiBase,
          TensorFlow.DApi,
          TensorFlow.Core,
          TensorFlow.Variable,
          TensorFlow.Initializer,
          Tensorflow.Utils;

type
 Optimizer = class ;

 GateGradientType = (GATE_NONE = 0, GATE_OP = 1, GATE_GRAPH = 2);

 gen_training_ops  = record
   private

   public
     class function resource_apply_adam(_var: TFTensor; m: TFTensor; v: TFTensor; beta1_power: TFTensor; beta2_power: TFTensor; lr: TFTensor; beta1: TFTensor; beta2: TFTensor; epsilon: TFTensor; grad: TFTensor; use_locking : Boolean = false; use_nesterov : Boolean = false; name : string = '') : TFTensor; static;
     class function apply_adam         (_var: TFTensor; m: TFTensor; v: TFTensor; beta1_power: TFTensor; beta2_power: TFTensor; lr: TFTensor; beta1: TFTensor; beta2: TFTensor; epsilon: TFTensor; grad: TFTensor; use_locking : Boolean = false; use_nesterov : Boolean = false; name : string = '') : TFTensor; static;
     class function apply_gradient_descent         (_var: IVariableV1; alpha: TFTensor; delta: TFTensor; use_locking : Boolean = false; name : string = ''): TFTensor; static;
     class function resource_apply_gradient_descent(_var: TFTensor;    alpha: TFTensor; delta: TFTensor; use_locking : Boolean = false; name : string = ''): TFTensor; static;
 end;

 SlotCreator = class
    private
      /// <summary>
      /// Helper function for creating a slot variable.
      /// </summary>
      /// <param name="primary"></param>
      /// <param name="val"></param>
      /// <param name="scope"></param>
      /// <param name="validate_shape"></param>
      /// <param name="shape"></param>
      /// <param name="dtype"></param>
      /// <returns></returns>
      function _create_slot_var(primary: IVariableV1; val: TObject; scope: string; validate_shape: Boolean; shape: TFShape; dtype: TF_DataType): IVariableV1;
    public
      /// <summary>
      /// Create a slot initialized to the given value.
      /// </summary>
      /// <param name="primary"></param>
      /// <param name="val"></param>
      /// <param name="name"></param>
      /// <param name="colocate_with_primary"></param>
      /// <returns></returns>
      function create_slot(primary: RefVariable; val: TFTensor; name: string; colocate_with_primary: Boolean = true): IVariableV1;
      /// <summary>
      /// Create a slot initialized to 0 with same shape as the primary object.
      /// </summary>
      /// <param name="primary"></param>
      /// <param name="name"></param>
      /// <param name="dtype"></param>
      /// <param name="colocate_with_primary"></param>
      /// <returns></returns>
      function create_zeros_slot(primary: IVariableV1; name: string; dtype: TF_DataType = DtInvalid; colocate_with_primary : Boolean= true): IVariableV1;
      /// <summary>
      /// Creates a slot initialized using an `Initializer`.
      /// </summary>
      /// <returns></returns>
      function create_slot_with_initializer(primary: IVariableV1; initializer: IInitializer; shape: TFShape; dtype: TF_DataType; name: string; colocate_with_primary : Boolean = true): IVariableV1;
 end;

  _OptimizableVariable  = class abstract
      function target : TFTensor; virtual; abstract;
      function update_op(_optimizer: Optimizer; g: TFTensor): TFOperation; virtual; abstract;
  end;

  T_optimizer = class
    private

    public
       class function _get_processor(v:RefVariable): _OptimizableVariable; overload;
       class function _get_processor(v: ResourceVariable): _OptimizableVariable; overload;
  end;

  _RefVariableProcessor = class(_OptimizableVariable)
      private
        Fv : RefVariable;
      public
        constructor Create(v: RefVariable);
        function target : TFTensor; override;
        function update_op(_optimizer: Optimizer; g: TFTensor): TFOperation; override;
  end;

  _DenseResourceVariableProcessor = class(_OptimizableVariable)
      private
        Fv : ResourceVariable;
      public
        constructor Create(v: ResourceVariable);
        function target : TFTensor; override;
        function update_op(_optimizer: Optimizer; g: TFTensor): TFOperation; override;
  end;


 Trackable = class abstract
    private
       Fself_update_uid : Integer;
    public
       /// <summary>
       /// Restore-on-create for a variable be saved with this `Checkpointable`.
       /// </summary>
       /// <returns></returns>
       function _add_variable_with_custom_getter(args: VariableArgs): IVariableV1 ; virtual;
       /// <summary>
       /// Pop and load any deferred checkpoint restores into `trackable`.
       /// </summary>
       /// <param name="name"></param>
       /// <param name="trackable"></param>
       procedure _handle_deferred_dependencies(name: string; trackable: IVariableV1);  virtual;
       function _track_checkpointable(checkpointable: IVariableV1; name: string; overwrite: Boolean = false) : IVariableV1;  virtual;
       /// <summary>
       /// Initialize dependency management.
       /// </summary>
       procedure _maybe_initialize_trackable; virtual;
    public
 end;

 AutoTrackable = class abstract(Trackable)

 end;

 /// <summary>
 /// Base class for optimizers.
 /// This class defines the API to add Ops to train a model.  You never use this
 /// class directly, but instead instantiate one of its subclasses such as
 /// `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
 /// </summary>
 Optimizer = class(Trackable)
    private
       Fname : string;
       Flr   : Single;
       Flr_t : TFTensor;

       /// <summary>
       /// Create the beta1 and beta2 accumulators on the same device as the first
       /// variable. Sort the var_list to make sure this device is consistent across
       /// workers (these need to go on the same PS, otherwise some updates are
       /// silently ignored).
       /// </summary>
       /// <param name="var_list"></param>
       procedure _create_slots(var_list: TArray<IVariableV1>); virtual;
       /// <summary>
       /// Add an extra variable, not associated with a slot.
       /// </summary>
       /// <param name="initial_value"></param>
       /// <param name="name"></param>
       /// <param name="colocate_with"></param>
       function _create_non_slot_variable(initial_value: Single; name: string; colocate_with: IVariableV1): IVariableV1;
       /// <summary>
       /// Return a slot named `name` created for `var` by the Optimizer.
       /// </summary>
       /// <param name="var"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       function get_slot(_var: IVariableV1; name: string): IVariableV1;
       function _var_key(_var: IVariableV1): string;
       function _get_non_slot_variable(name: string; graph : TFGraph= nil): IVariableV1;
       function _scale_loss(loss_value: TFTensor) : TFTensor;
       function _call_if_callable<T>(param: T): T;
       /// <summary>
       /// Find or create a slot initialized with 0.0.
       /// </summary>
       /// <param name="var"></param>
       /// <param name="slot_name"></param>
       /// <param name="op_name"></param>
       /// <returns></returns>
       function _zeros_slot(_var: IVariableV1; slot_name: string; op_name: string): IVariableV1;
       /// <summary>
       /// Restore a newly created slot variable's value.
       /// </summary>
       procedure _restore_slot_variable(slot_name: string; variable: IVariableV1; slot_variable: IVariableV1) ;
       function  _slot_dict(slot_name: string): TDictionary<string, IVariableV1>;
    public
      // Values for gate_gradients.
      const GATE_NONE : Integer = 0;
      const GATE_OP   : Integer = 1;
      const GATE_GRAPH: Integer = 2;
    public
      Fuse_locking               : Boolean;
      Fslots                     : TDictionary< string, TDictionary<string, IVariableV1> > ;
      Fnon_slot_dict             : TDictionary< string, IVariableV1>;
      Fdeferred_slot_restorations: TDictionary<string, TValue>;
      slot_creator               : SlotCreator;
      constructor Create(_learning_rate: Single;   _use_locking: Boolean; _name: string = '');overload;
      constructor Create(_learning_rate: TFTensor; _use_locking: Boolean; _name: string = '');overload;
      destructor Destroy; override;
      /// <summary>
      /// Add operations to minimize `loss` by updating `var_list`
      ///
      ///  This method simply combines calls `compute_gradients()` and
      ///  `apply_gradients()`. If you want to process the gradient before applying
      ///  them call `compute_gradients()` and `apply_gradients()` explicitly instead
      ///  of using this function.
      /// </summary>
      /// <param name="loss">A `Tensor` containing the value to minimize.</param>
      /// <param name="global_step">Optional `Variable` to increment by one after the
      /// variables have been updated.</param>
      /// <param name="var_list">Optional list or tuple of `Variable` objects to update to
      /// minimize `loss`.  Defaults to the list of variables collected in
      /// the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.</param>
      /// <param name="gate_gradients">
      /// How to gate the computation of gradients.  Can be
      /// `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      /// </param>
      /// <param name="aggregation_method">
      /// Specifies the method used to combine gradient terms.
      /// Valid values are defined in the class `AggregationMethod`.
      /// </param>
      /// <param name="colocate_gradients_with_ops"></param>
      /// <param name="name">Optional name for the returned operation.</param>
      /// <param name="grad_loss">Optional. A `Tensor` holding the gradient computed for `loss`.</param>
      /// <returns>
      /// An Operation that updates the variables in `var_list`.  If `global_step`
      /// was not `None`, that operation also increments `global_step`.
      /// </returns>
      function minimize(loss                        : TFTensor;
                        global_step                 : IVariableV1 = nil;
                        var_list                    : TList<IVariableV1> = nil;
                        gate_gradients              : GateGradientType = GateGradientType.GATE_OP;
                        aggregation_method          : PInteger = nil;
                        colocate_gradients_with_ops : Boolean= false;
                        name                        : string = '';
                        grad_loss                   : TFTensor = nil): TFOperation;
      /// <summary>
      /// Apply gradients to variables.
      ///
      /// This is the second part of `minimize()`. It returns an `Operation` that
      /// applies gradients.
      /// </summary>
      /// <param name="grads_and_vars">List of (gradient, variable) pairs as returned by
      /// `compute_gradients()`.</param>
      /// <param name="global_step">Optional `Variable` to increment by one after the
      /// variables have been updated.</param>
      /// <param name="name">Optional name for the returned operation.  Default to the
      /// name passed to the `Optimizer` constructor.</param>
      /// <returns>
      /// An `Operation` that applies the specified gradients. If `global_step`
      /// was not None, that operation also increments `global_step`.</returns>
      function apply_gradients(grads_and_vars: TArray< Tuple<TFTensor, IVariableV1> >; global_step: IVariableV1 = nil; name : string= '') : TFOperation;
      function _finish(update_ops: TArray<TFOperation>; name_scope: string): TFOperation; virtual;
      function _apply_dense(grad: TFTensor; _var: ResourceVariable): TFOperation ; overload; virtual;
      function _apply_dense(grad: TFTensor; _var: RefVariable):TFOperation; overload;  virtual;
      /// <summary>
      /// Add ops to apply sparse gradients to `var`, with repeated sparse indices.
      /// </summary>
      /// <param name="grad"></param>
      /// <param name="var"></param>
      /// <returns></returns>
      function _apply_sparse_duplicate_indices(grad: IndexedSlices; _var: RefVariable):TFOperation; overload;
      function _apply_sparse_duplicate_indices(grad: IndexedSlices; _var: ResourceVariable):TFOperation; overload;
      function _apply_sparse(grad: IndexedSlices; _var: ResourceVariable) :TFOperation; overload; virtual;
      function _apply_sparse(grad: IndexedSlices; _var: RefVariable) :TFOperation; overload; virtual;
      function _deduplicate_indexed_slices(values: TFTensor; indices: TFTensor): Tuple<TFTensor, TFTensor>;
      procedure _prepare; virtual;
      /// <summary>
      /// Compute gradients of `loss` for the variables in `var_list`.
      /// </summary>
      /// <param name="loss"></param>
      /// <param name="gate_gradients"></param>
      /// <returns>
      /// A list of (gradient, variable) pairs. Variable is always present, but
      /// gradient can be `None`.
      /// </returns>
      function  compute_gradients(loss                        : TFTensor;
                                  var_list                    : TList<IVariableV1> = nil;
                                  aggregation_method          : PInteger = nil;
                                  gate_gradients              : GateGradientType = GATE_OP;
                                  colocate_gradients_with_ops : Boolean = false;
                                  grad_loss : TFTensor = nil): TArray< Tuple<TFTensor, IVariableV1> >;

      property Name              : string   read Fname;
      property LearningRate      : Single   read Flr;
      property LearningRateTensor: TFTensor read Flr_t;

 end;

 /// <summary>
 /// Optimizer that implements the gradient descent algorithm.
 /// </summary>
 GradientDescentOptimizer = class(Optimizer)
    private
      FuseTensor : Boolean;
    public
      /// <summary>
      /// Construct a new gradient descent optimizer.
      /// </summary>
      /// <param name="learning_rate">A Tensor or a floating point value.  The learning
      /// rate to use.</param>
      /// <param name="use_locking">If true use locks for update operations.</param>
      /// <param name="name">Optional name prefix for the operations created when applying
      /// gradients.Defaults to "GradientDescent".</param>
      /// <remarks>
      /// When eager execution is enabled, `learning_rate` can be a callable that
      /// takes no arguments and returns the actual value to use.This can be useful
      /// for changing these values across different invocations of optimizer
      /// functions.
      /// </remarks>
      constructor Create(_learning_rate: Single;   _use_locking: Boolean = false; _name: string = 'GradientDescent'); overload;
      constructor Create(_learning_rate: TFTensor; _use_locking: Boolean = false; _name: string = 'GradientDescent'); overload;
      procedure _prepare; override;
 end;

 /// <summary>
 /// Optimizer that implements the Adam algorithm.
 /// http://arxiv.org/abs/1412.6980
 /// </summary>
 AdamOptimizer = class(Optimizer)
    private
       Fbeta1    : Single;
       Fbeta2    : Single;
       Fepsilon  : Single;
       Fbeta1_t  : TFTensor;
       Fbeta2_t  : TFTensor;
       Fepsilon_t: TFTensor;
       Fdtype    : TF_DataType;

       function _apply_sparse_shared(grad: TFTensor; _var: IVariableV1; indices: TFTensor; scatter_add: TFunc<IVariableV1, TFTensor, TFTensor, TFTensor>) : TFOperation;
       procedure _create_slots(var_list: TArray<IVariableV1>); override;
       function _get_beta_accumulators: Tuple<IVariableV1, IVariableV1>;
    public
       constructor Create(_learning_rate: Single;   beta1: Single = 0.9; beta2: Single = 0.999; epsilon: Single = 1e-8; _use_locking: Boolean = false; dtype : TF_DataType = TF_FLOAT; _name : string = 'Adam');overload;
       constructor Create(_learning_rate: TFTensor; beta1: Single = 0.9; beta2: Single = 0.999; epsilon: Single = 1e-8; _use_locking: Boolean = false; dtype : TF_DataType = TF_FLOAT; _name : string = 'Adam');overload;
       function _apply_sparse(grad: IndexedSlices; _var: ResourceVariable) : TFOperation; overload; override;
       function _apply_sparse(grad: IndexedSlices; _var: RefVariable) : TFOperation; overload; override;
       function _apply_dense(grad: TFTensor; _var: ResourceVariable) : TFOperation; override;
       function _finish(update_ops: TArray<TFOperation>; name_scope: string): TFOperation; override;
       procedure _prepare; override;
 end;

implementation
       uses Tensorflow,
            TensorFlow.Ops,
            TensorFlow.Tensor,
            Tensorflow.Gradient,
            TensorFlow.Operations ;

{ Trackable }

function Trackable._add_variable_with_custom_getter(args: VariableArgs): IVariableV1;
begin
    {$HINTS OFF}
    TUtils.tf_with<TNameScope>( TOps.init_scope, procedure(v1: TNameScope)
                                                  begin
                                                      var checkpoint_initializer : IInitializer := nil;
                                                      if tf.Context.executing_eagerly then
                                                      begin
                                                      end else
                                                      begin
                                                          checkpoint_initializer := nil;
                                                      end;
                                                  end );
    var new_variable := args.Getter(args);

    // If we set an initializer and the variable processed it, tracking will not
    // assign again. It will add this variable to our dependencies, and if there
    // is a non-trivial restoration queued, it will handle that. This also
    // handles slot variables.
    if (not args.Overwrite) or (new_variable is RefVariable) then
    begin
        Result := _track_checkpointable(new_variable, args.Name, args.Overwrite);
    end else
    begin
        Result := new_variable;
    end;
end;

procedure Trackable._handle_deferred_dependencies(name: string; trackable: IVariableV1);
begin
    _maybe_initialize_trackable;
    { TODO -oMax -c :  30/10/2022 21:15:15 }
end;

procedure Trackable._maybe_initialize_trackable;
begin
    Fself_update_uid := -1;
end;

function Trackable._track_checkpointable(checkpointable: IVariableV1; name: string; overwrite: Boolean): IVariableV1;
begin
    Result := checkpointable;
end;

{ Optimizer }

constructor Optimizer.Create(_learning_rate: Single; _use_locking: Boolean; _name: string);
begin
    if String.IsNullOrEmpty(_name) then
       raise  TFException.Create('Must specify the optimizer name');
    Fname        := _name;
    Fuse_locking := _use_locking;
    Flr          := _learning_rate;
    // Dictionary of slots.
    Fslots                      := TDictionary<string, TDictionary<string, IVariableV1> >.Create;
    Fnon_slot_dict              := TDictionary<string, IVariableV1>.Create;
    Fdeferred_slot_restorations := TDictionary<string, TValue>.create;

    slot_creator                := SlotCreator.Create;
end;

constructor Optimizer.Create(_learning_rate: TFTensor; _use_locking: Boolean; _name: string);
begin
    if String.IsNullOrEmpty(_name) then
        raise  TFException.Create('Must specify the optimizer name');

    Fname        := _name;
    Fuse_locking := _use_locking;
    Flr_t        := _learning_rate;
    // Dictionary of slots.
    Fslots                      := TDictionary<string, TDictionary<string, IVariableV1> >.Create;
    Fnon_slot_dict              := TDictionary<string, IVariableV1>.Create;
    Fdeferred_slot_restorations := TDictionary<string, TValue>.create;
    slot_creator                := SlotCreator.Create;
end;

destructor Optimizer.Destroy;
begin
    if Assigned(Flr_t) then
      Flr_t.free;
    if Assigned(Fslots) then
      Fslots.Free;
    if Assigned(Fnon_slot_dict) then
      Fnon_slot_dict.Free;
    if Assigned(Fdeferred_slot_restorations) then
      Fdeferred_slot_restorations.Free;
    if Assigned(slot_creator) then
      slot_creator.Free;
end;

function Optimizer.apply_gradients(grads_and_vars: TArray<Tuple<TFTensor, IVariableV1>>; global_step: IVariableV1; name: string): TFOperation;
var
  item                     : Tuple<TFTensor, IVariableV1>;
  converted_grads_and_vars : TList< Tuple<TFTensor, IVariableV1, _OptimizableVariable> >;
  Grad,tGrad               : TFTensor;
  vVar                     : IVariableV1;
  oProcessor               : _OptimizableVariable;
  var_list                 : TArray<IVariableV1>;
  update_ops               : TList<TFOperation>;
begin
    var wFunc:TPredicate<Tuple<TFTensor, IVariableV1, _OptimizableVariable>> := function(const x: Tuple<TFTensor, IVariableV1, _OptimizableVariable> ) : Boolean
                  begin
                      Result := x.Value1 <> nil
                  end;
    var selFunc : TFunc<Tuple<TFTensor, IVariableV1, _OptimizableVariable>,IVariableV1> := function (Arg1: Tuple<TFTensor, IVariableV1, _OptimizableVariable> ):IVariableV1
                   begin
                      Result := Arg1.Value2;
                   end;
    // No DistributionStrategy case.
    converted_grads_and_vars := TList< Tuple<TFTensor, IVariableV1, _OptimizableVariable> >.Create;
    try
      for item in grads_and_vars do
      begin
          Grad := item.Value1;
          vVar := item.Value2;
          if Grad <> nil then
          begin
              // Convert the grad to Tensor or IndexedSlices if necessary.
              tGrad       := Tops.convert_to_tensor_or_indexed_slices(Grad);
              oProcessor  := T_optimizer._get_processor(vVar as ResourceVariable);
              converted_grads_and_vars.Add( Tuple.Create (tGrad, vVar, oProcessor) );
          end;
      end;
      var eConverted_grads_and_vars := Enumerable< Tuple<TFTensor, IVariableV1, _OptimizableVariable> >.Create(converted_grads_and_vars.ToArray);
      var e1                        := Enumerable< Tuple<TFTensor, IVariableV1, _OptimizableVariable> >( eConverted_grads_and_vars.Where( wFunc ) );
      var_list := e1.Select<IVariableV1>( selFunc ).ToArray;

      if Length(var_list) = 0 then
         raise TFException.Create('No gradients provided for any variable');

      Tops.init_scope;

      _create_slots(var_list);

      update_ops := TList<TFOperation>.Create;
      try
        Result := TUtils.tf_with<TNameScope,TFOperation>( TOps.name_scope(name, Self.Name, nil),
                    function(v1: TNameScope): TFOperation
                    var
                      t3            : Tuple<TFTensor, IVariableV1, _OptimizableVariable>;
                      gGrad         : TFTensor;
                      _var          : IVariableV1;
                      processor     : _OptimizableVariable;
                      apply_updates : TFOperation ;
                      cd            : TArray<TValue>;
                      begin
                          name := v1.ToString;
                          _prepare;
                          for t3 in converted_grads_and_vars do
                          begin
                              gGrad     := t3.Value1;
                              _var      := t3.Value2;
                              processor := t3.Value3;
                              if gGrad = nil then
                                  continue;
                              var scope_name := _var.Op.name;
                              TUtils.tf_with<TNameScope>( Tops.name_scope('update_' + scope_name),
                                Procedure(v1: TNameScope)
                                var
                                  op : TFOperation;
                                   begin
                                      op := processor.update_op(Self, gGrad);
                                      update_ops.Add(op);
                                   end);
                          end;
                          apply_updates := nil;
                          if global_step = nil then
                          begin
                              apply_updates := _finish(update_ops.ToArray, name);
                          end else
                          begin
                              for var i := 0 to update_ops.Count - 1 do
                                 cd := cd + [ TValue.From<TFOperation>(update_ops[i]) ];
                              cd := cd + [ TValue.From<string>('update')  ] ;

                              TUtils.tf_with<TControlDependenciesController>( Tops.control_dependencies(cd),
                                Procedure(dep: TControlDependenciesController)
                                   begin
                                      // ops.colocate_with(global_step);
                                      // TODO: port this if branch once ResourceVariable has been ported!
                                      //if (global_step is ResourceVariable)
                                      //{
                                      //        # TODO(apassos): the implicit read in assign_add is slow; consider
                                      //        # making it less so.
                                      //        apply_updates = resource_variable_ops.assign_add_variable_op(
                                      //            global_step.handle,
                                      //            ops.convert_to_tensor(1, dtype = global_step.dtype),
                                      //            name = name)
                                      //}
                                      //else
                                      begin
                                          apply_updates := state_ops.assign_add(global_step, Tops.convert_to_tensor(1, global_step.dtype), False, name).Op;
                                      end
                                   end);
                          end;
                          if  not tf.Context.executing_eagerly then
                          begin
                              var train_op := Tops.get_collection_ref<TFOperation>(tf.GraphKeys.TRAIN_OP);
                              if (train_op <> nil) and (train_op.Contains(apply_updates) ) then
                                  train_op.Add(apply_updates);
                          end;
                          Result := apply_updates;
                      end );

      finally
        update_ops.Free;
      end;
    finally
     converted_grads_and_vars.free ;
    end;
end;

function Optimizer.compute_gradients(loss: TFTensor; var_list: TList<IVariableV1>; aggregation_method: PInteger; gate_gradients: GateGradientType;
  colocate_gradients_with_ops: Boolean; grad_loss: TFTensor): TArray<Tuple<TFTensor, IVariableV1>>;
begin
    var selFunc : TFunc<IVariableV1,_OptimizableVariable> := function (Arg1: IVariableV1 ):_OptimizableVariable
               begin
                  Result := T_optimizer._get_processor(Arg1 as ResourceVariable)
               end;
    var selFunc1 : TFunc<_OptimizableVariable,TFTensor> := function (Arg1: _OptimizableVariable ):TFTensor
               begin
                  Result := Arg1.target;
               end;

    // Scale loss if using a "mean" loss reduction and multiple replicas.
    loss := _scale_loss(loss);

    if var_list = nil then
    begin
        var vars := Tops.get_collection<IVariableV1>(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES).ToArray;
        var tmp  := variables.trainable_variables;
        var_list := TList<IVariableV1>.Create( (tmp.AsType< TList<IVariableV1> >).toArray + vars );
    end;

    var e_var_list := Enumerable<IVariableV1>.Create(var_list.ToArray + Tops.get_collection<IVariableV1>(tf.GraphKeys._STREAMING_MODEL_PORTS).ToArray  );
    var processors := Enumerable<_OptimizableVariable>( e_var_list.Select<_OptimizableVariable>(selFunc) );
    var var_refs   := processors.Select<TFTensor>(selFunc1).ToArray;

    var grad_ys : TArray<TFTensor>:= [];
    if grad_loss <> nil then
      grad_ys := [ grad_loss ];

    var grads : TArray<TFTensor> := gradients_impl.gradients([loss ], var_refs, grad_ys, 'gradients', colocate_gradients_with_ops, gate_gradients = GateGradientType.GATE_OP, aggregation_method);

    if Ord(gate_gradients) = Optimizer.GATE_GRAPH then
        grads := control_flow_ops.tuple(grads);

    var grads_and_vars : TArray<Tuple<TFTensor, IVariableV1>> := [];
    var a_var_list     : TArray<IVariableV1> := e_var_list.ToArray;
    for var i := 0 to Length(grads) -1 do
    begin
        var tTuple     := Tuple.Create<TFTensor, IVariableV1>( grads[i], a_var_list[i]);
        grads_and_vars := grads_and_vars + [ tTuple ];
    end;
    Result := grads_and_vars;
end;

function Optimizer.minimize(loss: TFTensor; global_step: IVariableV1; var_list: TList<IVariableV1>; gate_gradients: GateGradientType; aggregation_method: PInteger;
  colocate_gradients_with_ops: Boolean; name: string; grad_loss: TFTensor): TFOperation;
 var
   grads_and_vars : TArray<Tuple<TFTensor, IVariableV1>>;
   vars_with_grad : TArray<IVariableV1>;
begin
    // TODO: strongly type aggregation_method
    grads_and_vars := compute_gradients(loss, var_list, aggregation_method, gate_gradients, colocate_gradients_with_ops, grad_loss);

    vars_with_grad := [];
    for var  i := 0 to Length(grads_and_vars) - 1 do
    begin
        if grads_and_vars[i].Value1 <> nil then
            vars_with_grad := vars_with_grad + [ grads_and_vars[i].Value2 ];
    end;

    if Length(vars_with_grad) = 0 then
       raise TFException.Create('No gradients provided for any variable, check your graph for ops that do not support gradients');
    Result := apply_gradients(grads_and_vars, global_step, name);
end;

function Optimizer.get_slot(_var: IVariableV1; name: string): IVariableV1;
begin
    var named_slots : TDictionary<string, IVariableV1> := nil;
    if Fslots.ContainsKey(name) then
       named_slots := Fslots[name];

    if named_slots = nil then
        Exit(nil);
    if named_slots.ContainsKey( _var_key(_var) ) then  Result := named_slots[_var_key(_var)]
    else                                               Result := nil;
end;

function Optimizer._apply_dense(grad: TFTensor; _var: ResourceVariable): TFOperation;
begin
    if tf.executing_eagerly then
    begin
        var alpha := math_ops.cast(LearningRateTensor, TDTypes.as_base_dtype(_var.dtype));
        Result    := gen_training_ops.resource_apply_gradient_descent(_var.ToTensor, alpha, grad, Fuse_locking).op;
    end else
    begin
        var alpha := math_ops.cast(LearningRateTensor, TDTypes.as_base_dtype(_var.dtype));
        Result    := gen_training_ops.apply_gradient_descent(_var, alpha, grad, Fuse_locking).op;
    end
end;

function Optimizer._apply_dense(grad: TFTensor; _var: RefVariable): TFOperation;
begin
    var alpha := math_ops.cast(LearningRateTensor, TDTypes.as_base_dtype(_var.dtype));
    Result := gen_training_ops.apply_gradient_descent(_var, alpha, grad, Fuse_locking).op;
end;

function Optimizer._apply_sparse(grad: IndexedSlices; _var: RefVariable): TFOperation;
begin
   raise TFException.Create('Not Implemented "_apply_sparse"');
end;

function Optimizer._apply_sparse(grad: IndexedSlices; _var: ResourceVariable): TFOperation;
begin
    raise TFException.Create('Not Implemented "_apply_sparse"');
end;

function Optimizer._apply_sparse_duplicate_indices(grad: IndexedSlices; _var: RefVariable): TFOperation;
begin
    var tS := _deduplicate_indexed_slices(grad.values, grad.indices);
    var summed_values := tS.Value1;
    var unique_indices:= tS.Value2;
    var gradient_no_duplicate_indices := IndexedSlices.Create(unique_indices, summed_values, grad.dense_shape);

    Result := _apply_sparse(gradient_no_duplicate_indices, _var);
end;

function Optimizer._apply_sparse_duplicate_indices(grad: IndexedSlices; _var: ResourceVariable): TFOperation;
begin
    var tS := _deduplicate_indexed_slices(grad.values, grad.indices);
    var summed_values := tS.Value1;
    var unique_indices:= tS.Value2;
    var gradient_no_duplicate_indices := IndexedSlices.Create(unique_indices, summed_values, grad.dense_shape);

    Result := _apply_sparse(gradient_no_duplicate_indices, _var);
end;

function Optimizer._call_if_callable<T>(param: T): T;
begin
   Result := param;
end;

function Optimizer._create_non_slot_variable(initial_value: Single; name: string; colocate_with: IVariableV1): IVariableV1;
begin
    // Recommendation: Use OptimizerV2 if your optimizer uses non-slot variables.
    var graph := colocate_with.Graph;
    var key := name+'.'+ graph.graph_key;
    var v : IVariableV1 := nil;
    if Fnon_slot_dict.ContainsKey(key) then
      v :=  Fnon_slot_dict[key] ;
    if v = nil then
    begin
        _maybe_initialize_trackable();
        var bTrain : Boolean := False;
        var bResource :=  resource_variable_ops.is_resource_variable(colocate_with) ;
        v := variable_scope.default_variable_creator(initial_value, name, @bTrain, nil, TDTypes.as_base_dtype(colocate_with.dtype), nil, False, @bResource);
        // Restore this variable by name if necessary, but don't add a
        // Trackable dependency. Optimizers return the current graph's
        // non-slot variables from _checkpoint_dependencies explicitly rather
        // than unconditionally adding dependencies (since there may be multiple
        // non-slot variables with the same name in different graphs, trying to
        // save all of them would result in errors).
        _handle_deferred_dependencies(name, v);
        Fnon_slot_dict.AddOrSetValue(key, v);
    end;
    Result := v;
end;

procedure Optimizer._create_slots(var_list: TArray<IVariableV1>);
begin

end;

function Optimizer._deduplicate_indexed_slices(values, indices: TFTensor): Tuple<TFTensor, TFTensor>;
begin
    var tU := array_ops.unique(indices);
    var unique_indices      := tU.Value1;
    var new_index_positions := tU.Value2;

    var shape         := array_ops.shape(unique_indices)._slice(0);
    var summed_values := math_ops.unsorted_segment_sum(values, new_index_positions, shape);
    Result := Tuple.Create(summed_values, unique_indices);
end;

function Optimizer._finish(update_ops: TArray<TFOperation>; name_scope: string): TFOperation;
begin
    Result := control_flow_ops.group<TFOperation>(update_ops, name_scope);
end;

function Optimizer._get_non_slot_variable(name: string; graph: TFGraph): IVariableV1;
begin
    var key      := name+'.'+graph.graph_key;
    var non_slot : IVariableV1 := nil ;
    if Fnon_slot_dict.ContainsKey(key) then
         non_slot :=  Fnon_slot_dict[key] ;

    Result := non_slot;
end;

procedure Optimizer._prepare;
begin

end;

procedure Optimizer._restore_slot_variable(slot_name: string; variable, slot_variable: IVariableV1);
begin
    var variable_key := _var_key(variable);
    // TODO
end;

function Optimizer._scale_loss(loss_value: TFTensor): TFTensor;
begin
    Tops.get_default_graph.is_loss_scaled_by_optimizer := false;
    // TODO
    // if distribute_lib.get_loss_reduction() == ds_reduce_util.ReduceOp.MEAN:
    Result := loss_value;
end;

function Optimizer._slot_dict(slot_name: string): TDictionary<string, IVariableV1>;
begin
    var named_slots : TDictionary<string, IVariableV1> := nil;
    if Fslots.ContainsKey(slot_name) then
     named_slots := Fslots[slot_name];

    if named_slots = nil then
    begin
        named_slots := TDictionary<string, IVariableV1>.Create;
        Fslots.AddOrSetValue(slot_name, named_slots);
    end ;
    Result := named_slots;
end;

function Optimizer._var_key(_var: IVariableV1): string;
begin
   Result := _var.Op.graph.graph_key+'.'+ _var.Op.name;
end;

function Optimizer._zeros_slot(_var: IVariableV1; slot_name, op_name: string): IVariableV1;
begin
    var named_slots := _slot_dict(slot_name);
    if not named_slots.ContainsKey(_var_key(_var)) then
    begin
        var new_slot_variable := slot_creator.create_zeros_slot(_var, op_name);
        _restore_slot_variable(slot_name, _var, new_slot_variable);
        named_slots.AddOrSetValue(_var_key(_var), new_slot_variable);
    end;
    Result := named_slots[_var_key(_var)];
end;

{ T_optimizer }

class function T_optimizer._get_processor(v: ResourceVariable): _OptimizableVariable;
begin
    Result := _DenseResourceVariableProcessor.Create(v);
end;

class function T_optimizer._get_processor(v: RefVariable): _OptimizableVariable;
begin
    Result := _RefVariableProcessor.Create(v);
end;

{ _RefVariableProcessor }

constructor _RefVariableProcessor.Create(v: RefVariable);
begin
    Fv := v;
end;

function _RefVariableProcessor.target: TFTensor;
begin
    Result := Fv._ref;
end;

function _RefVariableProcessor.update_op(_optimizer: Optimizer; g: TFTensor): TFOperation;
var
  upd_op : TFOperation;
begin

    if g.Tag.IsType<IndexedSlices> then
    begin
        Result := _optimizer._apply_sparse_duplicate_indices(g, Fv);
        Exit;
    end else
    begin
        upd_op := _optimizer._apply_dense(g, Fv);
    end;
    Result := upd_op;
end;

{ _DenseResourceVariableProcessor }

constructor _DenseResourceVariableProcessor.Create(v: ResourceVariable);
begin
    inherited Create;
    Fv := v;
end;

function _DenseResourceVariableProcessor.target: TFTensor;
begin
    Result := Fv.tHandle;
end;

function _DenseResourceVariableProcessor.update_op(_optimizer: Optimizer; g: TFTensor): TFOperation;
var
  upd_op : TFOperation;
begin
    if (g.Tag.TypeInfo<> nil) and (g.Tag.IsType<IndexedSlices>) then
    begin
        Result := _optimizer._apply_sparse_duplicate_indices(g, Fv);
        Exit;
    end else
    begin
        upd_op := _optimizer._apply_dense(g, Fv);
    end;
    Result := upd_op;
end;

{ gen_training_ops }

class function gen_training_ops.resource_apply_adam(_var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad: TFTensor; use_locking, use_nesterov: Boolean;
  name: string): TFTensor;
begin
   Result := tf.Context.ExecuteOp('ResourceApplyAdam', name, ExecuteOpArgs.Create([ _var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad ])
              .SetAttributes(['use_locking', use_locking, 'use_nesterov',  use_nesterov ])).First;
end;

class function gen_training_ops.apply_adam(_var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad: TFTensor; use_locking, use_nesterov: Boolean;
  name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ApplyAdam', name, ExecuteOpArgs.Create([ _var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad ])
              .SetAttributes(['use_locking', use_locking, 'use_nesterov',  use_nesterov ])).First;
end;

class function gen_training_ops.apply_gradient_descent(_var: IVariableV1; alpha, delta: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('ApplyGradientDescent', name,  [ GetArg('var', TValue.From<IVariableV1>(_var)),
                                                                             GetArg('alpha',alpha),
                                                                             GetArg('delta',delta),
                                                                             GetArg('use_locking',use_locking) ]);
    Result := _op.output;
end;

class function gen_training_ops.resource_apply_gradient_descent(_var, alpha, delta: TFTensor; use_locking: Boolean; name: string): TFTensor;
begin
    Result := tf.Context.ExecuteOp('ResourceApplyGradientDescent', name, ExecuteOpArgs.Create([ _var, alpha, delta ])
              .SetAttributes(['use_locking', use_locking ])).First;
end;

{ SlotCreator }

function SlotCreator.create_slot(primary: RefVariable; val: TFTensor; name: string; colocate_with_primary: Boolean): IVariableV1;
begin
    var validate_shape := val.shape.IsFullyDefined;
    var prefix := primary.Op.name;
    Result := TUtils.tf_with<TNameScope,IVariableV1>( TOps.name_scope('',  prefix + '/' + name),
                  function(v1: TNameScope): IVariableV1
                    begin
                        Result := _create_slot_var(primary, val, '', validate_shape, nil, TF_DataType.DtInvalid);
                    end );

end;

function SlotCreator.create_slot_with_initializer(primary: IVariableV1; initializer: IInitializer; shape: TFShape; dtype: TF_DataType; name: string;
  colocate_with_primary: Boolean): IVariableV1;
begin
    var validate_shape := shape.IsFullyDefined;
    var prefix := primary.Op.name;
    Result := TUtils.tf_with<TNameScope,IVariableV1>( TOps.name_scope('',  prefix + '/' + name),
                  function(v1: TNameScope): IVariableV1
                    begin
                        Result := _create_slot_var(primary,TObject(initializer), '', validate_shape, shape, dtype);
                    end );
end;

function SlotCreator.create_zeros_slot(primary: IVariableV1; name: string; dtype: TF_DataType; colocate_with_primary: Boolean): IVariableV1;
begin
    if dtype = TF_DataType.DtInvalid then
        dtype := primary.dtype;
    var slot_shape := primary.shape;
    if slot_shape.IsFullyDefined then
    begin
        var initializer := Zeros.Create;
        Result := create_slot_with_initializer(primary, initializer, slot_shape, dtype, name, colocate_with_primary);
    end else
    begin
       raise TFException.Create('Not Implemented create_zeros_slot is not fully defined.');
    end;
end;

function SlotCreator._create_slot_var(primary: IVariableV1; val: TObject; scope: string; validate_shape: Boolean; shape: TFShape; dtype: TF_DataType): IVariableV1;
begin
    var use_resource :Boolean  := primary is ResourceVariable;
    if resource_variable_ops.is_resource_variable(primary) then
        use_resource := true;
    var bTrainable : Boolean := False;
    var slot := tf.compat.v1.get_variable(scope, @shape, dtype, val, @bTrainable, nil, @use_resource, validate_shape);
    Result := slot;
end;

{ GradientDescentOptimizer }

constructor GradientDescentOptimizer.Create(_learning_rate: Single; _use_locking: Boolean; _name: string);
begin
    inherited Create(_learning_rate, _use_locking, _name);
    Flr        := _learning_rate;
    FuseTensor := false;
end;

constructor GradientDescentOptimizer.Create(_learning_rate: TFTensor; _use_locking: Boolean; _name: string);
begin
    inherited Create(_learning_rate, _use_locking, _name);
    Flr_t      := _learning_rate;
    FuseTensor := True;
end;

procedure GradientDescentOptimizer._prepare;
begin
    if not FuseTensor then
    Begin
        var lr := _call_if_callable(Flr);
        Flr_t  := Tops.convert_to_tensor(single(lr), DtInvalid, 'learning_rate');
    end;
end;

{ AdamOptimizer }

constructor AdamOptimizer.Create(_learning_rate, beta1, beta2, epsilon: Single; _use_locking: Boolean; dtype: TF_DataType; _name: string);
begin
    inherited Create(_learning_rate, _use_locking, _name);
    Fbeta1   := beta1;
    Fbeta2   := beta2;
    Fepsilon := epsilon;
    Fdtype   := dtype;
end;

constructor AdamOptimizer.Create(_learning_rate: TFTensor; beta1, beta2, epsilon: Single; _use_locking: Boolean; dtype: TF_DataType; _name: string);
begin
    inherited Create(_learning_rate, _use_locking, _name);
    Fbeta1   := beta1;
    Fbeta2   := beta2;
    Fepsilon := epsilon;
    Fdtype   := dtype;
end;

function AdamOptimizer._apply_dense(grad: TFTensor; _var: ResourceVariable): TFOperation;
begin
    var m := get_slot(_var, 'm');
            var v := get_slot(_var, 'v');
            var tAcc := _get_beta_accumulators;
            var beta1_power := tAcc.Value1;
            var beta2_power := tAcc.Value2;
            Result := gen_training_ops.apply_adam(
                                          _var.tHandle,
                                          m.tHandle,
                                          v.tHandle,
                                          math_ops.cast(beta1_power.tHandle, TDtypes.as_base_dtype(_var.dtype)),
                                          math_ops.cast(beta2_power.tHandle, TDtypes.as_base_dtype(_var.dtype)),
                                          math_ops.cast(Flr_t,              TDtypes.as_base_dtype(_var.dtype)),
                                          math_ops.cast(Fbeta1_t,           TDtypes.as_base_dtype(_var.dtype)),
                                          math_ops.cast(Fbeta2_t,           TDtypes.as_base_dtype(_var.dtype)),
                                          math_ops.cast(Fepsilon_t,         TDtypes.as_base_dtype(_var.dtype)),
                                          grad,
                                          Fuse_locking).op;
end;

function AdamOptimizer._apply_sparse(grad: IndexedSlices; _var: ResourceVariable): TFOperation;
begin
    Result := _apply_sparse_shared(grad.values, _var, grad.indices,
                   function (x: IVariableV1; i: TFTensor; v: TFTensor)  : TFTensor
                    begin
                        Result := state_ops.scatter_add(x, i, v, Fuse_locking);
                    end);
end;

function AdamOptimizer._apply_sparse(grad: IndexedSlices; _var: RefVariable): TFOperation;
begin
    Result := _apply_sparse_shared(grad.values, _var, grad.indices,
                   function (x: IVariableV1; i: TFTensor; v: TFTensor)  : TFTensor
                    begin
                        Result := state_ops.scatter_add(x, i, v, Fuse_locking);
                    end);
end;

function AdamOptimizer._apply_sparse_shared(grad: TFTensor; _var: IVariableV1; indices: TFTensor; scatter_add: TFunc<IVariableV1, TFTensor, TFTensor, TFTensor>): TFOperation;
begin
    var tAcc := _get_beta_accumulators;
    var beta1_power_v := tAcc.Value1;
    var beta2_power_v := tAcc.Value2;
    var beta1_power : TFTensor := math_ops.cast(beta1_power_v, TDtypes.as_base_dtype(_var.dtype));
    var beta2_power : TFTensor := math_ops.cast(beta2_power_v, TDtypes.as_base_dtype(_var.dtype));
    var lr_t                   := math_ops.cast(Flr_t,         TDtypes.as_base_dtype(_var.dtype));
    var beta1_t                := math_ops.cast(Fbeta1_t,      TDtypes.as_base_dtype(_var.dtype));
    var beta2_t                := math_ops.cast(Fbeta2_t,      TDtypes.as_base_dtype(_var.dtype));
    var epsilon_t              := math_ops.cast(Fepsilon_t,    TDtypes.as_base_dtype(_var.dtype));
    var lr := (TTEnsor(lr_t) * math_ops.sqrt(1 - TTensor(beta2_power)) / (1 - TTensor(beta1_power)));
    var m := get_slot(_var, 'm');
    var m_scaled_g_values := TTensor(grad) * (1 - TTensor(beta1_t));
    var m_t := state_ops.assign(m, m.AsTensor * TTensor(beta1_t), Fuse_locking);
    TUtils.tf_with<TControlDependenciesController>( Tops.control_dependencies([m_t]) ,
                                Procedure(dep: TControlDependenciesController)
                                 begin
                                     m_t := scatter_add(m, indices, m_scaled_g_values);
                                 end);
    var v                 := get_slot(_var, 'v');
    var v_scaled_g_values := (TTensor(grad) * grad) * (1 - TTEnsor(beta2_t));
    var v_t               := state_ops.assign(v, v.AsTensor * TTensor(beta2_t), Fuse_locking);
    TUtils.tf_with<TControlDependenciesController>( Tops.control_dependencies([v_t]) ,
                                Procedure(dep: TControlDependenciesController)
                                 begin
                                     v_t := scatter_add(v, indices, v_scaled_g_values);
                                 end);
    var v_sqrt     := math_ops.sqrt(v_t);
    var var_update := state_ops.assign_sub(_var, lr * m_t / (TTensor(v_sqrt) + epsilon_t), Fuse_locking);
    Result := control_flow_ops.group<TFTensor>( [ var_update, m_t, v_t ]);
end;

procedure AdamOptimizer._create_slots(var_list: TArray<IVariableV1>);
begin
    var first_var := Enumerable<IVariableV1>.Create(var_list)
                                              .OrderBy<string>( function(x: IVariableV1) : string
                                                         begin
                                                             Result := x.Name;
                                                         end).First;

    _create_non_slot_variable(Fbeta1, 'beta1_power', first_var);
    _create_non_slot_variable(fbeta2, 'beta2_power', first_var);
    // Create slots for the first and second moments.
    for var v in var_list DO
    BEGIN
        _zeros_slot(v, 'm', Name);
        _zeros_slot(v, 'v', Name);
    end;
end;

function AdamOptimizer._finish(update_ops: TArray<TFOperation>; name_scope: string): TFOperation;
begin
    var operations := TList<ITensorOrOperation>.Create;
    try
      for var i := 0 to Length(update_ops)-1 do
         operations.add(update_ops[i]);
      var value : TArray<TValue> := [];
      for var i := 0 to Length(update_ops)-1 do
        value := value + [ TValue.From<TFOperation>(update_ops[i]) ] ;
      TUtils.tf_with<TControlDependenciesController>( Tops.control_dependencies(value) ,
                                Procedure(dep: TControlDependenciesController)
                                 begin
                                    var tAcc := _get_beta_accumulators;
                                    var beta1_power := tAcc.Value1;
                                    var beta2_power := tAcc.Value2;
                                    Tops.colocate_with(beta1_power);
                                    var update_beta1 : TFTensor := nil;
                                    var update_beta2 : TFTensor := nil;
                                    if      beta1_power is ResourceVariable then  update_beta1 := (beta1_power as ResourceVariable).assign(beta1_power.AsTensor * TTensor(Fbeta1_t), Fuse_locking)
                                    else if beta1_power is RefVariable then       update_beta1 := (beta1_power as RefVariable).     assign(beta1_power.AsTensor * TTensor(Fbeta1_t), Fuse_locking);

                                    if      beta2_power is ResourceVariable then  update_beta2 := (beta2_power as ResourceVariable).assign(beta2_power.AsTensor * TTensor(Fbeta1_t), Fuse_locking)
                                    else if beta2_power is RefVariable then       update_beta2 := (beta2_power as RefVariable).     assign(beta2_power.AsTensor * TTensor(Fbeta1_t), Fuse_locking);

                                    operations.Add(update_beta1);
                                    operations.Add(update_beta2);
                                 end);
      Result := control_flow_ops.group<ITensorOrOperation>(operations.ToArray, name_scope);
    finally
      operations.Free;
    end;
end;

function AdamOptimizer._get_beta_accumulators: Tuple<IVariableV1, IVariableV1>;
begin
    Tops.init_scope;
    var graph := Tops.get_default_graph;
    var t1 := _get_non_slot_variable('beta1_power', graph) ;
    var t2 := _get_non_slot_variable('beta2_power', graph) ;
    Result := Tuple.Create(t1,t2);
end;

procedure AdamOptimizer._prepare;
begin
    var lr      := _call_if_callable(Flr);
    var beta1   := _call_if_callable(Fbeta1);
    var beta2   := _call_if_callable(Fbeta2);
    var epsilon := _call_if_callable(Fepsilon);
    if Assigned(Flr_t)      then  Flr_t      := Tops.convert_to_tensor(lr,    Fdtype,  'learning_rate');
    if Assigned(Fbeta1_t)   then  Fbeta1_t   := Tops.convert_to_tensor(beta1, Fdtype,  'beta1');
    if Assigned(Fbeta2_t)   then  Fbeta2_t   := Tops.convert_to_tensor(beta2, Fdtype,  'beta2');
    if Assigned(Fepsilon_t) then  Fepsilon_t := Tops.convert_to_tensor(epsilon, Fdtype,'epsilon');
end;

end.
