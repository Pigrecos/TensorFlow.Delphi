unit Keras.Callbacks;
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

interface
     uses System.SysUtils,
          System.Generics.Collections,
          System.Diagnostics,
          System.Math,
          System.Rtti,

          Spring,

          Keras.Core,

          TensorFlow.Core,
          TensorFlow.DApi;

type

  CallbackParams = class  abstract
     public
       mModel : IModel;
       Verbose: integer;
       Epochs : integer;
       Steps  : INt64;

       constructor Create;
  end;

  History =  class(TInterfacedObject,  ICallback)
    private
       Fepochs      : TList<Integer>;
       Fparameters  : CallbackParams;
       Fhistory     : TDictionary<string, TList<Single>> ;

       function GetLog: string;
       function Get_history: TDictionary<string, TList<Single>>;
       procedure Set_history(Value: TDictionary<string, TList<Single>>);
    public
       procedure on_train_begin;
       procedure on_train_end;
       procedure on_epoch_begin(epoch: Integer);
       procedure on_train_batch_begin(step: Int64);
       procedure on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
       procedure on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
       //
       procedure on_predict_begin;
       procedure on_predict_batch_begin(step: Int64);
       procedure on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
       procedure on_predict_end;
       //
       procedure on_test_begin;
       procedure on_test_batch_begin(step: Int64);
       procedure on_test_batch_end(end_step: Int64; logs : TDictionary<string, Single>);
      //
       //
       constructor Create(_parameters: CallbackParams);

       property epochs      : TList<Integer>  read Fepochs;
       property parameters  : CallbackParams  read Fparameters;
  end;

  CallbackList = class
    private
       Fcallbacks : TList<ICallback>;
       FMSg       : string;

       function GetHistory: History;
    public
       constructor Create(_parameters: CallbackParams);
       procedure on_train_begin;
       procedure on_epoch_begin(epoch: Integer);
       procedure on_train_batch_begin(step: Int64);
       procedure on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
       procedure on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
       //
       procedure on_predict_begin;
       procedure on_predict_batch_begin(step: Int64);
       procedure on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
       procedure on_predict_end;
       //
       procedure on_test_begin;
       procedure on_test_batch_begin(step: Int64);
       procedure on_test_batch_end(end_step: Int64; logs : TDictionary<string, Single>);
       //
       property hHistory  : History read GetHistory;
       property callbacks : TList<ICallback>  read  Fcallbacks;
       property sLog      : string  read FMSg;
  end;

  ProgbarLogger = class(TInterfacedObject,  ICallback)
    private
       Fcalled_in_fit : Boolean ;
       Fseen          : Integer;
       Fparameters    : CallbackParams;
       Fsw            : TStopwatch  ;
       FMSg           : string;
       Fhistory       : TDictionary<string, TList<Single>> ;

       function  Get_history: TDictionary<string, TList<Single>>;
       procedure Set_history(Value: TDictionary<string, TList<Single>>);
       function  GetLog: string;
    public
      procedure on_train_begin;
      procedure on_train_end;
      procedure on_epoch_begin(epoch: Integer);
      procedure on_train_batch_begin(step: Int64);
      procedure on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
      procedure on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
      procedure on_test_begin;
      procedure on_test_batch_begin(step: Int64);
      procedure on_test_batch_end(end_step: Int64; logs : TDictionary<string, Single>);
      //
      procedure on_predict_begin;
      procedure on_predict_batch_begin(step: Int64);
      procedure on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
      procedure on_predict_end;
      //
      constructor Create(_parameters: CallbackParams);
      procedure _reset_progbar;
      procedure _maybe_init_progbar;
  end;

  /// <summary>
  /// Stop training when a monitored metric has stopped improving.
  /// </summary>
  /// <param name="parameters"></param>
  /// <param name="monitor"></param>
  EarlyStopping = class(TInterfacedObject,  ICallback)
    private
        Fpaitence        : Integer;
        Fmin_delta       : Integer;
        Fverbose         : Integer;
        Fstopped_epoch   : Integer;
        Fwait            : Integer;
        Fbest_epoch      : Integer;
        Fstart_from_epoch: Integer;
        Fbest            : Single;
        Fbaseline        : Single;
        Fmonitor         : string;
        Fmode            : string;
        Fbest_weights    : TList<IVariableV1>;
        Fparameters      : CallbackParams;
        Frestore_best_weights : Boolean;
        FMSg             : string;
        Fhistory         : TDictionary<string, TList<Single>> ;

        function  GetLog: string;
        function  Get_history: TDictionary<string, TList<Single>>;
        procedure Set_history(Value: TDictionary<string, TList<Single>>);
        function  get_monitor_value(logs : TDictionary<string, Single>) : Single;
        function  _is_improvement(monitor_value: Single; reference_value: Single): Boolean;
    public
        procedure on_train_begin;
        procedure on_train_end;
        procedure on_epoch_begin(epoch: Integer);
        procedure on_train_batch_begin(step: Int64);
        procedure on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
        procedure on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
        //
        procedure on_predict_begin;
        procedure on_predict_batch_begin(step: Int64);
        procedure on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
        procedure on_predict_end;
        //
        procedure on_test_begin;
        procedure on_test_batch_begin(step: Int64);
        procedure on_test_batch_end(end_step: Int64; logs : TDictionary<string, Single>);
        //
        // user need to pass a CallbackParams to EarlyStopping, CallbackParams at least need the model
        constructor Create(parameters: CallbackParams; monitor : string= 'val_loss'; min_delta : Integer= 0; patience: Integer = 0;  verbose: Integer = 1;
                           mode : string= 'auto'; baseline : Single= 0; restore_best_weights : Boolean= false; start_from_epoch: Integer = 0);
  end;

implementation
         uses Numpy;

{ CallbackParams }

constructor CallbackParams.Create;
begin
   inherited Create;
end;

{ History }

constructor History.Create(_parameters: CallbackParams);
begin
   Fparameters := _parameters;
end;

procedure History.on_train_begin;
begin
    Fepochs := TList<Integer>.Create;
    Fhistory:= TDictionary<string, TList<Single>>.Create;
end;

procedure History.on_train_end;
begin

end;

function History.GetLog: string;
begin

end;

function History.Get_history: TDictionary<string, TList<Single>>;
begin
    Result := Fhistory
end;

procedure History.Set_history(Value: TDictionary<string, TList<Single>>);
begin
    Fhistory := Value
end;

procedure History.on_epoch_begin(epoch: Integer);
begin

end;

procedure History.on_test_begin;
begin
    Fepochs := TList<Integer>.Create;
    Fhistory:= TDictionary<string, TList<Single>>.Create;
end;

procedure History.on_test_batch_begin(step: Int64);
begin

end;

procedure History.on_test_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin

end;

procedure History.on_train_batch_begin(step: Int64);
begin

end;

procedure History.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin

end;

procedure History.on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
begin
    Fepochs.Add(epoch);

    for var log in epoch_logs do
    begin
        if not Fhistory.ContainsKey(log.Key) then
           Fhistory.Add(log.Key, TList<Single>.Create);
        Fhistory[log.Key].Add(log.Value);
    end;
end;

procedure History.on_predict_begin;
begin
    Fepochs := TList<Integer>.Create;
    Fhistory:= TDictionary<string, TList<Single>>.Create;
end;

procedure History.on_predict_batch_begin(step: Int64);
begin

end;

procedure History.on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
begin

end;

procedure History.on_predict_end;
begin

end;

{ CallbackList }

constructor CallbackList.Create(_parameters: CallbackParams);
begin
    Fcallbacks := TList<ICallback>.Create;

    Fcallbacks.Add(History.Create(_parameters));
    Fcallbacks.Add(ProgbarLogger.Create(_parameters));
end;

function CallbackList.GetHistory: History;
begin
    Result := Fcallbacks[0] as History;
end;

procedure CallbackList.on_train_begin;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_train_begin;
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_epoch_begin(epoch: Integer);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_epoch_begin(epoch);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_test_begin;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_test_begin;
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_test_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_test_batch_end(end_step, logs);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_test_batch_begin(step: Int64);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_test_batch_begin(step);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_train_batch_begin(step: Int64);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_train_batch_begin(step);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_train_batch_end(end_step, logs);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_epoch_end(epoch, epoch_logs);
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_predict_begin;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_begin;
         FMSg := FMSg + Fcallbacks[i].sLog;
    end;
end;

procedure CallbackList.on_predict_batch_begin(step: Int64);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_batch_begin(step);
         FMSg := FMSg + Fcallbacks[i].sLog ;
    end;
end;

procedure CallbackList.on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_batch_end(end_step, logs);
         FMSg := FMSg + Fcallbacks[i].sLog ;
    end;
end;

procedure CallbackList.on_predict_end;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_end;
         FMSg := FMSg + Fcallbacks[i].sLog ;
    end;
end;

{ ProgbarLogger }

constructor ProgbarLogger.Create(_parameters: CallbackParams);
begin
    Fcalled_in_fit := False ;
    Fseen          := 0;
    Fparameters    := _parameters;
end;

procedure ProgbarLogger.on_train_begin;
begin
    Fcalled_in_fit := true;
    Fsw            := TStopwatch.StartNew;
end;

procedure ProgbarLogger.on_train_end;
begin

end;

function ProgbarLogger.GetLog: string;
begin
   Result := FMSg;
end;

function ProgbarLogger.Get_history: TDictionary<string, TList<Single>>;
begin
    Result := Fhistory
end;

procedure ProgbarLogger.Set_history(Value: TDictionary<string, TList<Single>>);
begin
    Fhistory := Value;
end;

procedure ProgbarLogger.on_epoch_begin(epoch: Integer);
begin
    _reset_progbar;
    _maybe_init_progbar;
    FMSg := Format('Epoch: %.3d/%.3d',[epoch + 1,Fparameters.Epochs]);
end;

procedure ProgbarLogger.on_test_begin;
begin
    Fsw  := TStopwatch.StartNew;
end;

procedure ProgbarLogger.on_test_batch_begin(step: Int64);
begin
    Fsw.Reset;
    Fsw.Start
end;

procedure ProgbarLogger.on_test_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    Fsw.Stop;

    var elapse := Fsw.ElapsedMilliseconds;
    var results : string := '';
    for var it in logs do
       results := results + ' - ' + Format('%s: %.6f',[ it.Key, it.Value]);

    FMSg := Format('%.4d/%.4d - %dms/step - %s',[end_step + 1,Fparameters.Steps,elapse, results]);
end;

procedure ProgbarLogger.on_train_batch_begin(step: Int64);
begin
    Fsw.Reset;
    Fsw.Start;
end;

procedure ProgbarLogger.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    Fsw.Stop;

    var elapse := Fsw.ElapsedMilliseconds/MSecsPerDay;
    var results : string := '';
    for var it in logs do
     results := results + ' - ' + Format('%s: %.6f',[ it.key, it.Value]);

    var progress := '';
    var length := 30.0 / Fparameters.Steps;
    for var i := 0 to Floor(end_step * length - 1) -1 do
      progress := progress + '=';

    if progress.Length < 28 then progress := progress + '>'
    else                         progress := progress + '=';

    var remaining := '';
    for var i := 1 to 30 - progress.Length do
        remaining := remaining + '.';

    FMSg := Format('Epoch: %.4d/%.4d [%s%s] - %s - %s',[end_step + 1,Fparameters.Steps,progress,remaining,FormatDateTime('hh:nn:ss:zzz',elapse), results]);
end;

procedure ProgbarLogger.on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
begin
    FMSg := '';
end;

procedure ProgbarLogger.on_predict_batch_begin(step: Int64);
begin

end;

procedure ProgbarLogger.on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
begin

end;

procedure ProgbarLogger.on_predict_begin;
begin
    _reset_progbar;
    _maybe_init_progbar;
end;

procedure ProgbarLogger.on_predict_end;
begin

end;

procedure ProgbarLogger._reset_progbar;
begin
    Fseen := 0;
end;

procedure ProgbarLogger._maybe_init_progbar;
begin

end;

{ EarlyStopping }

constructor EarlyStopping.Create(parameters: CallbackParams; monitor: string; min_delta, patience, verbose: Integer; mode: string; baseline: Single;
  restore_best_weights: Boolean; start_from_epoch: Integer);
begin
    Fparameters    := parameters;
    Fstopped_epoch := 0;
    Fwait          := 0;
    Fmonitor       := monitor;
    Fpaitence      := patience;
    Fverbose       := verbose;
    Fbaseline      := baseline;
    Fstart_from_epoch := start_from_epoch;
    Fmin_delta     := Abs(min_delta);
    Frestore_best_weights := restore_best_weights;
    Fmode := mode;

    if (mode <> 'auto') and (mode <> 'min') and (mode <> 'max') then
      if IsConsole then
         WriteLn('EarlyStopping mode '+mode+' is unknown, fallback to auto mode.');

end;

function EarlyStopping.GetLog: string;
begin
   Result := FMSg;
end;

function EarlyStopping.Get_history: TDictionary<string, TList<Single>>;
begin
    Result := Fhistory
end;

procedure EarlyStopping.Set_history(Value: TDictionary<string, TList<Single>>);
begin
    Fhistory := Value;
end;

procedure EarlyStopping.on_train_begin;
begin
    Fwait          := 0;
    Fstopped_epoch := 0;
    Fbest_epoch    := 0;
    Fbest          := double.PositiveInfinity;
end;

procedure EarlyStopping.on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
begin
    var current := get_monitor_value(epoch_logs);
    // If no monitor value exists or still in initial warm-up stage.
    if (current = 0) or (epoch < Fstart_from_epoch) then
        Exit;;
    // Restore the weights after first epoch if no progress is ever made.
    if (Frestore_best_weights) and (Fbest_weights = nil) then
      Fbest_weights := Fparameters.mModel.TrainableWeights;

    Fwait := Fwait+1;

    if _is_improvement(current, Fbest) then
    begin
        Fbest := current;
        Fbest_epoch := epoch;
        if Frestore_best_weights then
            Fbest_weights := Fparameters.mModel.TrainableWeights;
        // Only restart wait if we beat both the baseline and our previous best.
        if (Fbaseline = 0) or (_is_improvement(current, Fbaseline)) then
            Fwait := 0;
    end;
    // Only check after the first epoch.
    if (Fwait >= Fpaitence) and (epoch > 0) then
    begin
        Fstopped_epoch := epoch;
        Fparameters.mModel.Stop_training := true;
        if (Frestore_best_weights) and (Fbest_weights <> nil) then
        begin
            if (Fverbose > 0) and (IsConsole) then
             WriteLn('Restoring model weights from the end of the best epoch: '+IntTostr(Fbest_epoch + 1));

        end;
        // Because loading the weight variable into the model has not yet been implemented, so Earlystopping can't load best_weight yet.
        // TODO(Wanglongzhi2001): implement it.
        // _parameters.Model.load_weights(best_weights);
    end;
end;

procedure EarlyStopping.on_train_end;
begin
    if (Fstopped_epoch > 0) and (Fverbose > 0) then
     if IsConsole then
       WriteLn('Epoch '+ IntToStr(Fstopped_epoch + 1)+': early stopping');

end;

function EarlyStopping.get_monitor_value(logs: TDictionary<string, Single>): Single;
begin
    if logs = nil then
       logs:= TDictionary<string, Single>.Create;

    if not logs.ContainsKey(Fmonitor) then
        logs.Add(Fmonitor,0);

    var monitor_value := logs[Fmonitor];
    if monitor_value = 0 then
      if IsConsole then
        WriteLn('Early stopping conditioned on metric '+ Fmonitor +' which is not available. Available metrics are: '+string.Join(', ', logs.Keys.ToArray));

    Result := monitor_value;
end;

function EarlyStopping._is_improvement(monitor_value, reference_value: Single): Boolean;
begin
    var less_op    := (monitor_value - Fmin_delta) < reference_value;
    var greater_op := (monitor_value - Fmin_delta) >= reference_value;
    if Fmode = 'min' then
        Exit(less_op)
    else if Fmode = 'max' then
        Exit(greater_op)
    else begin
        if (Fmonitor.EndsWith('acc')) or (Fmonitor.EndsWith('accuracy')) or (Fmonitor.EndsWith('auc')) then
           Exit(greater_op)
        else
            Exit(less_op);
    end;
end;

procedure EarlyStopping.on_epoch_begin(epoch: Integer);
begin

end;

procedure EarlyStopping.on_predict_batch_begin(step: Int64);
begin

end;

procedure EarlyStopping.on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
begin

end;

procedure EarlyStopping.on_predict_begin;
begin

end;

procedure EarlyStopping.on_predict_end;
begin

end;

procedure EarlyStopping.on_test_batch_begin(step: Int64);
begin

end;

procedure EarlyStopping.on_test_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin

end;

procedure EarlyStopping.on_test_begin;
begin

end;

procedure EarlyStopping.on_train_batch_begin(step: Int64);
begin

end;

procedure EarlyStopping.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin

end;

end.
