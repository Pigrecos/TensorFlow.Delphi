unit Keras.Optimizer;
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
            System.Generics.Defaults,
            System.Generics.Collections,
            System.TypInfo,

            Spring,
            Spring.Collections.Enumerable,

            TF4D.Core.CApi,
            TensorFlow.DApi,
            TensorFlow.DApiBase,

            TensorFlow.Context,
            TensorFlow.Variable,
            TensorFlow.Training,
            TensorFlow.Initializer,

            ProtoGen.variable,

            Keras.Core;

type
   DeviceDType = class( TEqualityComparer<DeviceDType>)
     public
       Device : string;
       DType  : TF_DataType;

       function ToString: string; override;
       function Equals(const Left, Right: DeviceDType): Boolean; override;
       function GetHashCode(const Value: DeviceDType): Integer;  override;

   end;

   /// <summary>
   /// Updated base class for optimizers.
   /// </summary>
   OptimizerV2 = class(TInterfacedObjectEx, IOptimizer)
      private
         // delphi class inherithed limit!!
         FIternalTrackable : Trackable;
         Fargs           : OptimizerV2Args;
         Fhypers_created : Boolean;
         Fname           : string;
         Fiterations     : IVariableV1;
         Fweights        : TList<IVariableV1>;
         Fhyper          : TDictionary<string, Single>;
         Fhyper_variables: TDictionary<string, IVariableV1>;
         Fmomentum       : Boolean;
         Finitial_decay  : Single;
         Fuse_locking    : Boolean;
         Fslots          : TDictionary< string, TDictionary<string, IVariableV1> >;
         Fslot_names     : TList<string>;
         function  Getiter: ResourceVariable;
         function  GetLr: IVariableV1;
         procedure _set_hyper(name: string; value: Single);
         function  _resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ): TFOperation; virtual;
         procedure _prepare_local(device_dtype: DeviceDType; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ); virtual;
         function  _get_hyper(name: string; dtype: TF_DataType = TF_DataType.DtInvalid): TFTensor;
         procedure _create_slots(var_list: TArray<IVariableV1>); virtual;
      protected
         function  get_slot(_var: IVariableV1; slot_name: string): IVariableV1;
         function  _fallback_apply_state(var_device: string; var_dtype: TF_DataType): TDictionary<string, TFTensor>;
         function  add_slot(_var: IVariableV1; slot_name: string; initializer : IInitializer= nil): IVariableV1;
      public
         constructor Create(_args: OptimizerV2Args);
         destructor  Destroy; override;
         /// <summary>
         /// Apply gradients to variables.
         /// </summary>
         /// <param name="grads_and_vars"></param>
         /// <param name="name"></param>
         /// <param name="experimental_aggregate_gradients"></param>
         function  apply_gradients(grads_and_vars: TArray< Tuple<TFTensor, ResourceVariable> >; name : string= ''; experimental_aggregate_gradients : Boolean = True) : TFOperation; overload;
         function  apply_gradients(grads_and_vars: Tuple<TFTensor, ResourceVariable>;           name : string= ''; experimental_aggregate_gradients : Boolean = True) : TFOperation; overload;
         procedure apply_grad_to_update_var(_var: ResourceVariable; grad: TFTensor; apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> );
         procedure _distributed_apply(grads_and_vars : TArray< Tuple<TFTensor, ResourceVariable> >; name: string; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>> );
         function  aggregate_gradients(grads_and_vars : TArray< Tuple<TFTensor, IVariableV1> > ) : TArray<TFTensor>;
         function  clip_gradients(grads: TArray<TFTensor>): TArray<TFTensor>;
         function  _prepare(var_list: TArray<IVariableV1>): TDictionary<DeviceDType, TDictionary<string, TFTensor>>;
         function  _decayed_lr(var_dtype: TF_DataType): TFTensor;
         procedure _create_all_weights(var_list: TArray<IVariableV1>);
         procedure _create_hypers;
         function add_weight(name            : string;
                             shape           : TFShape;
                             dtype           : TF_DataType = TF_DataType.TF_FLOAT;
                             initializer     : IInitializer = nil;
                             trainable       : Boolean = false;
                             synchronization : TVariableSynchronization = VARIABLE_SYNCHRONIZATION_AUTO;
                             aggregation     : TVariableAggregation = VARIABLE_AGGREGATION_NONE) : ResourceVariable;
         property args      : OptimizerV2Args   read Fargs;
         property name      : string            read Fname;
         property iterations: ResourceVariable  read Getiter ;
         property lr        : IVariableV1       read GetLr;
    end;
    /// <summary>
    /// Optimizer that implements the Adam algorithm.
    /// Adam optimization is a stochastic gradient descent method that is based on
    /// adaptive estimation of first-order and second-order moments.
    /// </summary>
    TAdam = class(OptimizerV2)
      private
         Fepsilon : Single;
         Famsgrad : Boolean;

         procedure _create_slots(var_list: TArray<IVariableV1>); override;
         procedure _prepare_local(device_dtype: DeviceDType; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ); override;
         function  _resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ): TFOperation; override;
      public
         constructor Create(learning_rate : Single = 0.001; beta_1 : Single = 0.9; beta_2 : Single = 0.999; epsilon: Single = 1e-7; amsgrad : Boolean= false; name : string= 'Adam');
    end;
    TRMSprop = class(OptimizerV2)
      private
         Fargs : RMSpropArgs;

         procedure _create_slots(var_list: TArray<IVariableV1>); override;
         procedure _prepare_local(device_dtype: DeviceDType; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ); override;
         function  _resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ): TFOperation; override;
    function GetCent: Boolean;
      public
         constructor Create(args: RMSpropArgs);

         property centered : Boolean read GetCent;
   end;
   TSGD = class(OptimizerV2)
      private
         procedure _prepare_local(device_dtype: DeviceDType; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ); override;
         function  _resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>> ): TFOperation; override;
      public
         constructor Create(learning_rate: Single; momentum : Single = 0.0; nesterov : Boolean = false; decay: Single = 0.0);
   end;

   OptimizerApi = class
      private

      public
        destructor Destroy ; override;
        /// <summary>
        /// Adam optimization is a stochastic gradient descent method that is based on
        /// adaptive estimation of first-order and second-order moments.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="beta_1"></param>
        /// <param name="beta_2"></param>
        /// <param name="epsilon"></param>
        /// <param name="amsgrad"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function  Adam(learning_rate : Single = 0.001; beta_1 : Single = 0.9; beta_2 : Single = 0.999; epsilon: Single = 1e-7; amsgrad : Boolean= false; name : string= 'Adam') : OptimizerV2;
        /// <summary>
        /// Construct a new RMSprop optimizer.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="rho"></param>
        /// <param name="momentum"></param>
        /// <param name="epsilon"></param>
        /// <param name="centered"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        function RMSprop(learning_rate : Single = 0.001; rho : Single = 0.9; momentum : Single = 0.0; epsilon : Single = 1e-7; centered : Boolean = false; name: string = 'RMSprop'): OptimizerV2;
        function SGD(learning_rate: Single): TSGD;
   end;

implementation
        uses Tensorflow,
             TensorFlow.Tensor,
             TensorFlow.Ops,
             Tensorflow.NameScope,
             Tensorflow.Utils,
             TensorFlow.control_flow_ops,
             Tensorflow.array_ops,
             Tensorflow.math_ops,

             Keras.Utils;


{ DeviceDType }

function DeviceDType.Equals(const Left, Right: DeviceDType): Boolean;
begin
    Result :=  Left.ToString = Right.ToString;
end;

function DeviceDType.GetHashCode(const Value: DeviceDType): Integer;
begin
   Result := 0;
end;

function DeviceDType.ToString: string;
begin
    Result := Device +', '+ Tdtypes.ToString(DType);
end;

{ OptimizerV2 }

constructor OptimizerV2.Create(_args: OptimizerV2Args);
begin
     FIternalTrackable := Trackable.Create;

     Fargs            := _args;
     Fweights         := TList<IVariableV1>.Create;
     Fhyper           := TDictionary<string, Single>.Create;
     Fhyper_variables := TDictionary<string, IVariableV1>.Create;
     Fslots           := TDictionary<string, TDictionary<string, IVariableV1>>.Create;
     Fslot_names      := TList<string>.Create;
     Finitial_decay   := 0.0;
     Fuse_locking     := True;
     _set_hyper('learning_rate', args.LearningRate);
     _set_hyper('decay', args.InitialDecay);
end;

destructor OptimizerV2.Destroy;
begin
     Fargs.Free;
     if Assigned(Fweights) then
       Fweights.Free;
     if Assigned(Fhyper) then
       Fhyper.Free;
     if Assigned(Fhyper_variables) then
       Fhyper_variables.Free;
     if Assigned(Fslots) then
       Fslots.Free;
     if Assigned(Fslot_names) then
       Fslot_names.Free;
end;

function OptimizerV2.apply_gradients(grads_and_vars: Tuple<TFTensor, ResourceVariable>; name: string; experimental_aggregate_gradients: Boolean): TFOperation;
begin
    Result := apply_gradients([grads_and_vars], name, experimental_aggregate_gradients)
end;

function OptimizerV2.apply_gradients(grads_and_vars: TArray<Tuple<TFTensor, ResourceVariable>>; name: string; experimental_aggregate_gradients: Boolean): TFOperation;
var
  var_list : TArray<IVariableV1>;
begin
    var_list := [];
    for var i := 0 to Length(grads_and_vars)- 1 do
      var_list := var_list + [ grads_and_vars[i].Value2 ];

    Result := TUtils.tf_with<TNameScope,TFOperation>( TOps.name_scope(Fname),
                    function(v1: TNameScope): TFOperation
                      begin
                          Tops.init_scope;
                          _create_all_weights(var_list);
                          if Length(grads_and_vars) = 0 then
                          begin
                              Result := control_flow_ops.no_op;
                              Exit;
                          end;
                          var apply_state := _prepare(var_list);
                          // if(experimental_aggregate_gradients)
                          begin
                              // var reduced_grads = _aggregate_gradients(grads_and_vars);
                              _distributed_apply(grads_and_vars, name, apply_state);
                          end;
                          Result := nil;

                      end );
end;
procedure OptimizerV2.apply_grad_to_update_var(_var: ResourceVariable; grad: TFTensor; apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
    _resource_apply_dense(_var, grad, apply_state);
    // if var.constraint is not None:
    //     with ops.control_dependencies([update_op]):
    //         return var.assign(var.constraint(var))
end;
function OptimizerV2._resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>): TFOperation;
begin
    raise TFException.Create('_resource_apply_dense');
end;
procedure OptimizerV2._distributed_apply(grads_and_vars: TArray<Tuple<TFTensor, ResourceVariable>>; name: string;
  _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
    TUtils.tf_with<TNameScope>( TOps.name_scope(name,'',nil, true),
        procedure(v1: TNameScope)
          begin
              for var grad_var in grads_and_vars do
              begin
                  var grad := grad_var.Value1;
                  var _var := grad_var.Value2;
                  TUtils.tf_with<TNameScope>( TOps.name_scope('update'),
                    procedure(v1: TNameScope)
                      begin
                           apply_grad_to_update_var(_var, grad, _apply_state);
                      end);
              end;
              if      Fiterations is RefVariable          then (Fiterations as RefVariable)         .assign_add(Tops.convert_to_tensor(1, Fiterations.dtype))
              else if Fiterations is BaseResourceVariable then (Fiterations as BaseResourceVariable).assign_add(Tops.convert_to_tensor(1, Fiterations.dtype))
          end );
end;

function OptimizerV2.aggregate_gradients(grads_and_vars: TArray<Tuple<TFTensor, IVariableV1>>): TArray<TFTensor>;
begin
    Result := [];
    for var i := 0 to Length(grads_and_vars)-1 do
         Result := Result + [ grads_and_vars[i].Value1 ];
end;

function OptimizerV2.clip_gradients(grads: TArray<TFTensor>): TArray<TFTensor>;
begin
    Result := grads;
end;

function OptimizerV2.get_slot(_var: IVariableV1; slot_name: string): IVariableV1;
begin
    var slot_dict := Fslots[_var.UniqueId];
    Result :=  slot_dict[slot_name];
end;

function OptimizerV2._prepare(var_list: TArray<IVariableV1>): TDictionary<DeviceDType, TDictionary<string, TFTensor>>;
var
 _apply_state : TDictionary<DeviceDType, TDictionary<string, TFTensor>>;
 aDev         : TArray<DeviceDType>;
begin
    _apply_state := TDictionary<DeviceDType, TDictionary<string, TFTensor>>.Create;
    aDev := [];
    for var i := 0 to Length(var_list)-1  do
    begin
        var dt := DeviceDType.Create ;
        dt.Device := var_list[i].Device;
        var n := var_list[i].name;
        dt.DType  := TDtypes.as_base_dtype(var_list[i].dtype);
        aDev := aDev + [ dt ] ;
    end;
    var keys := Enumerable<DeviceDType>.Create(aDev).Distinct(DeviceDType.Create).ToArray;
    for var device_dtype in keys do
    begin
        _apply_state.AddOrSetValue(device_dtype, TDictionary<string, TFTensor>.Create);
        _prepare_local(device_dtype, _apply_state);
    end;
    Result := _apply_state;
end;

function OptimizerV2._fallback_apply_state(var_device: string; var_dtype: TF_DataType): TDictionary<string, TFTensor>;
begin
     raise TFException.Create('_fallback_apply_state');
end;

procedure OptimizerV2._prepare_local(device_dtype: DeviceDType; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
    if Fhyper.ContainsKey('learning_rate') then
    begin
        var lr_t := array_ops.identity(_decayed_lr(device_dtype.DType));
        _apply_state[device_dtype].AddOrSetValue('lr_t', lr_t);
    end;
end;

function OptimizerV2._decayed_lr(var_dtype: TF_DataType): TFTensor;
begin
    var lr_t := _get_hyper('learning_rate', var_dtype);
    if Finitial_decay > 0.0 then
    begin
        raise TFException.Create('Not Implemented');
    end;
    Result := lr_t;
end;

function OptimizerV2._get_hyper(name: string; dtype: TF_DataType): TFTensor;
begin
    var value := Fhyper_variables[name];
    Result := math_ops.cast(value, dtype);
end;

procedure OptimizerV2._create_all_weights(var_list: TArray<IVariableV1>);
begin
    if Fiterations = nil then
    begin
        Fiterations := add_weight('iter', TArray<Integer>.create(), TF_DataType.TF_INT64, nil, False, VARIABLE_SYNCHRONIZATION_AUTO, VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA);
        Fweights.Add(Fiterations);
    end;
    _create_hypers;
    _create_slots(var_list);
end;

procedure OptimizerV2._set_hyper(name: string; value: Single);
begin
    Fhyper.AddOrSetValue(name,value);
end;

procedure OptimizerV2._create_hypers;
begin
    if Fhypers_created then
        Exit;
    for var dict in Fhyper do
    begin
        var name  := dict.Key;
        var value := dict.Value;
        Fhyper_variables.AddOrSetValue(name, add_weight(name, TArray<Integer>.create(), TF_DataType.TF_FLOAT, tf.constant_initializer(value), false, VARIABLE_SYNCHRONIZATION_AUTO,VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA));
    end;
    Fhypers_created := true;
end;

procedure OptimizerV2._create_slots(var_list: TArray<IVariableV1>);
begin
    if Fmomentum then
    begin
        (*for var in var_list:
            self.add_slot(var, "momentum")*)
    end;
end;

function OptimizerV2.add_slot(_var: IVariableV1; slot_name: string; initializer: IInitializer): IVariableV1;
begin
    if initializer = nil then
        initializer := tf.zeros_initializer;
    if not Fslot_names.Contains(slot_name) then
        Fslot_names.Add(slot_name);
    if not Fslots.ContainsKey(_var.UniqueId) then
        Fslots.AddOrSetValue(_var.UniqueId, TDictionary<string, IVariableV1>.Create);
    var slot_dict := Fslots[_var.UniqueId];
    if not slot_dict.ContainsKey(slot_name) then
    begin
        var pSh : PTFShape := nil;
        var s   : TFShape  := _var.shape;
        if not s.isNil then
            pSh := @s;
        var weight := tf.Variable(initializer, false, true, true, _var.Name+'/'+slot_name, _var.dtype, VARIABLE_AGGREGATION_NONE, pSh);
        slot_dict.AddOrSetValue(slot_name, weight);
        Fweights.Add(weight);
        Result := weight;
    end else
    begin
        Result := slot_dict[slot_name];
    end;
end;

function OptimizerV2.add_weight(name: string; shape: TFShape; dtype: TF_DataType; initializer: IInitializer; trainable: Boolean; synchronization: TVariableSynchronization;
  aggregation: TVariableAggregation): ResourceVariable;
begin
    if initializer = nil then
        initializer := tf.zeros_initializer;
    if dtype = TF_DataType.DtInvalid then
        dtype := TF_DataType.TF_FLOAT;
    var varArg : VariableArgs;
    varArg.Name            := name;
    varArg.Shape           := shape;
    varArg.Getter          := base_layer_utils.make_variable;
    varArg.DType           := dtype;
    varArg.Overwrite       := true;
    varArg.Initializer     := initializer;
    varArg.Trainable       := trainable;
    varArg.UseResource     := true;
    varArg.Synchronization := synchronization;
    varArg.Aggregation     := aggregation;
    var variable := FIternalTrackable._add_variable_with_custom_getter(varArg);
    Result := variable as ResourceVariable;
end;

function OptimizerV2.Getiter: ResourceVariable;
begin
   Result := Fiterations as ResourceVariable
end;

function OptimizerV2.GetLr: IVariableV1;
begin
    Result := nil;
    if Fhyper_variables.ContainsKey('learning_rate') then
         Result := Fhyper_variables['learning_rate'];
end;

{ TSGD }

constructor TSGD.Create(learning_rate, momentum: Single; nesterov: Boolean; decay: Single);
begin
    Fname := 'SGD';

    inherited Create( OptimizerV2Args.Create );

    _set_hyper('learning_rate', learning_rate);
    _set_hyper('decay', decay);
    Fmomentum := momentum > 0;
   _set_hyper('momentum', momentum);
end;

procedure TSGD._prepare_local(device_dtype: DeviceDType; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
  inherited _prepare_local(device_dtype, _apply_state);

  _apply_state[device_dtype].AddOrSetValue('momentum', array_ops.identity( _get_hyper('momentum', device_dtype.DType) ) );

end;

function TSGD._resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>): TFOperation;
begin
    if Fmomentum then
    begin
        raise Exception.Create('_resource_apply_dense');
    end;
    var device_dtype : DeviceDType := nil;
    for var Key in _apply_state.Keys do
    begin
        if (key.Device = _var.Device) and (key.DType = Tdtypes.as_base_dtype(_var.dtype))  then
        begin
            device_dtype := key;
            Break;
        end;
    end;
    Result := gen_training_ops.resource_apply_gradient_descent(_var.tHandle, _apply_state[device_dtype]['lr_t'], grad, Fuse_locking).op;
end;

{ OptimizerApi }

function OptimizerApi.Adam(learning_rate, beta_1, beta_2, epsilon: Single; amsgrad: Boolean; name: string): OptimizerV2;
begin
    Result := TAdam.Create(learning_rate, beta_1, beta_2, epsilon, amsgrad, name)
end;

destructor OptimizerApi.Destroy;
begin
  inherited Destroy;
end;

function OptimizerApi.RMSprop(learning_rate, rho, momentum, epsilon: Single; centered: Boolean; name: string): OptimizerV2;
begin
    var rmsArg := RMSpropArgs.Create;

    rmsArg.LearningRate := learning_rate;
    rmsArg.RHO          := rho;
    rmsArg.Momentum     := momentum;
    rmsArg.Epsilon      := epsilon;
    rmsArg.Centered     := centered;
    rmsArg.Name         := name;

    Result := TRMSprop.Create(rmsArg);
end;

function OptimizerApi.SGD(learning_rate: Single): TSGD;
begin
    Result := TSGD.Create(learning_rate);
end;

{ TAdam }

constructor TAdam.Create(learning_rate, beta_1, beta_2, epsilon: Single; amsgrad: Boolean; name: string);
begin
    Fname    := 'Adam';

    inherited Create( OptimizerV2Args.Create );

    _set_hyper('learning_rate', learning_rate);
    // _set_hyper("decay", _initial_decay);
    _set_hyper('beta_1', beta_1);
    _set_hyper('beta_2', beta_2);
    Fepsilon := epsilon;
    Famsgrad := amsgrad;
end;

procedure TAdam._create_slots(var_list: TArray<IVariableV1>);
begin
    for var _var in var_list do
      add_slot(_var, 'm');
    for var _var in var_list do
      add_slot(_var, 'v');
    if Famsgrad then
    begin
      for  var _var in var_list do
          add_slot(_var, 'vhat');
    end;

end;

procedure TAdam._prepare_local(device_dtype: DeviceDType; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
    inherited _prepare_local(device_dtype, _apply_state);

    var var_dtype  := device_dtype.DType;
    var var_device := device_dtype.Device;
    var local_step := math_ops.cast( TResourceVariable(iterations) + 1, var_dtype);
    var beta_1_t   := array_ops.identity(_get_hyper('beta_1', var_dtype));
    var beta_2_t   := array_ops.identity(_get_hyper('beta_2', var_dtype));
    var beta_1_power := math_ops.pow(beta_1_t, local_step);
    var beta_2_power := math_ops.pow(beta_2_t, local_step);
    var lr := _apply_state[device_dtype]['lr_t'] * (math_ops.sqrt(1 - TTEnsor(beta_2_power)) / (1 - TTEnsor(beta_1_power)));
    // update state
    _apply_state[device_dtype].AddOrSetValue('lr',                  lr);
    _apply_state[device_dtype].AddOrSetValue('epsilon',             Tops.convert_to_tensor(Fepsilon));
    _apply_state[device_dtype].AddOrSetValue('beta_1_t',            beta_1_t);
    _apply_state[device_dtype].AddOrSetValue('beta_1_power',        beta_1_power);
    _apply_state[device_dtype].AddOrSetValue('one_minus_beta_1_t',  1 - TTEnsor(beta_1_t));
    _apply_state[device_dtype].AddOrSetValue('beta_2_t',            beta_2_t);
    _apply_state[device_dtype].AddOrSetValue('beta_2_power',        beta_2_power);
    _apply_state[device_dtype].AddOrSetValue('one_minus_beta_2_t',  1 - TTEnsor(beta_2_t));

end;

function TAdam._resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>): TFOperation;
begin
    var var_device := _var.Device;
    var var_dtype  := TDtypes.as_base_dtype(_var.dtype);

    var coefficients : TDictionary<string, TFTensor> := nil;
    for var Key in _apply_state do
    begin
        if (key.key.Device = _var.Device) and (key.key.DType = Tdtypes.as_base_dtype(_var.dtype))  then
        begin
            coefficients := key.value;
            Break;
        end;
    end;
    if coefficients = nil then
      coefficients := _fallback_apply_state(var_device, var_dtype) ;

    var m := get_slot(_var, 'm');
    var v := get_slot(_var, 'v');
    if not Famsgrad then
        Result := gen_training_ops.resource_apply_adam(_var.tHandle,
            m.tHandle,
            v.tHandle,
            coefficients['beta_1_power'],
            coefficients['beta_2_power'],
            coefficients['lr_t'],
            coefficients['beta_1_t'],
            coefficients['beta_2_t'],
            coefficients['epsilon'],
            grad,
            Fuse_locking).op
    else
       raise TFException.Create('Not Implemented');
end;

{ TRMSprop }

constructor TRMSprop.Create(args: RMSpropArgs);
begin
    Fname := 'RMSprop';
    Fargs := args;

    inherited Create( args );
    _set_hyper('rho', args.RHO);
    _set_hyper('momentum', args.Momentum);
end;

function TRMSprop.GetCent: Boolean;
begin
    Result := Fargs.Centered
end;

procedure TRMSprop._create_slots(var_list: TArray<IVariableV1>);
begin
    for var _var in var_list do
      add_slot(_var, 'rms');
    for var _var in var_list do
      add_slot(_var, 'momentum');
    if centered then
    begin
      for  var _var in var_list do
          add_slot(_var, 'mg');
    end;

end;

procedure TRMSprop._prepare_local(device_dtype: DeviceDType; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>);
begin
  inherited _prepare_local(device_dtype, _apply_state);

  var rho := array_ops.identity(_get_hyper('rho', device_dtype.DType));
  _apply_state[device_dtype].AddOrSetValue('neg_lr_t', - TTEnsor(_apply_state[device_dtype]['lr_t']));
  _apply_state[device_dtype].AddOrSetValue('epsilon', Tops.convert_to_tensor(Fargs.Epsilon, device_dtype.DType));
  _apply_state[device_dtype].AddOrSetValue('rho', rho);
  _apply_state[device_dtype].AddOrSetValue('momentum', array_ops.identity(_get_hyper('momentum', device_dtype.DType)));
  _apply_state[device_dtype].AddOrSetValue('one_minus_rho', Single(1.0) - TTensor(rho));
end;

function TRMSprop._resource_apply_dense(_var: IVariableV1; grad: TFTensor; _apply_state: TDictionary<DeviceDType, TDictionary<string, TFTensor>>): TFOperation;
var
  rms                   : IVariableV1;
  rms_t, denom_t, var_t : TFTensor;
  coefficients          : TDictionary<string, TFTensor>;
begin
    coefficients := nil;
    for var state in _apply_state do
    begin
        if (state.Key.DType = Tdtypes.as_base_dtype(_var.dtype) ) and (state.Key.Device = _var.Device) then
        begin
            coefficients := state.Value;
            break;
        end;
    end;
    rms := get_slot(_var, 'rms');
    if Fmomentum then
    begin
        raise TFException.Create('NotImplemented');
    end else
    begin
        rms_t   := TTensor(coefficients['rho']) * rms.AsTensor + TTensor(coefficients['one_minus_rho']) * math_ops.square(grad);
        rms_t   := state_ops.assign(rms, rms_t, Fuse_locking);
        denom_t := rms_t;
        if centered then
        begin
             raise TFException.Create('NotImplemented');
        end;
        var_t := TTensor(_var.AsTensor) - coefficients['lr_t'] * TTensor(grad) / ( TTensor(math_ops.sqrt(denom_t)) + coefficients['epsilon'] );
        Result := state_ops.assign(_var, var_t, Fuse_locking).op;
    end;
end;

end.


