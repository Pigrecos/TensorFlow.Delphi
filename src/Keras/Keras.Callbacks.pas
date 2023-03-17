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
       property hHistory  : History read GetHistory;
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
       function GetLog: string;
    public
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
      constructor Create(_parameters: CallbackParams);
      procedure _reset_progbar;
      procedure _maybe_init_progbar;
  end;


implementation

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
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_epoch_begin(epoch: Integer);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_epoch_begin(epoch);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_train_batch_begin(step: Int64);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_train_batch_begin(step);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_train_batch_end(end_step, logs);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_epoch_end(epoch: Integer; epoch_logs: TDictionary<string, Single>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_epoch_end(epoch, epoch_logs);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_predict_begin;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_begin;
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_predict_batch_begin(step: Int64);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_batch_begin(step);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_predict_batch_end(end_step: Int64; logs: TDictionary<string, TFTensors>);
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_batch_end(end_step, logs);
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
    end;
end;

procedure CallbackList.on_predict_end;
begin
    FMSg := '';
    for var i := 0 to Fcallbacks.Count - 1 do
    begin
         Fcallbacks[i].on_predict_end;
         FMSg := FMSg + Fcallbacks[i].sLog + sLineBreak;
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

procedure ProgbarLogger.on_train_batch_begin(step: Int64);
begin
    Fsw.Reset;
    Fsw.Start;
end;

procedure ProgbarLogger.on_train_batch_end(end_step: Int64; logs: TDictionary<string, Single>);
begin
    Fsw.Stop;

    var elapse := Fsw.ElapsedMilliseconds;
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

    FMSg := Format('Epoch: %.4d/%.4d [%s%s] - %d - %s',[end_step + 1,Fparameters.Steps,progress,remaining,elapse, results]);
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


end.
