unit TensorFlow.gen_control_flow_ops;
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
         Spring,
         TF4D.Core.CApi,
         TensorFlow.DApi,
         Numpy.Axis,

         TensorFlow.Context ;

type
  gen_control_flow_ops = record
    private

    public
       class function control_trigger(name: string = ''): TFOperation;static;
       class function no_op(name: string = ''): TFOperation; static;
       /// <summary>
       /// Creates or finds a child frame, and makes `data` available to the child frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="frame_name"></param>
       /// <param name="is_constant"></param>
       /// <param name="parallel_iterations"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function enter(data: TFTensor; frame_name: string = 'frame_name'; is_constant:Boolean = false; parallel_iterations : Integer= 10; name: string = ''): TFTensor;static;
       /// <summary>
       /// Forwards the input to the output.
       /// </summary>
       /// <param name="input"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function loop_cond(input: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Makes its input available to the next iteration.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function ref_next_iteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Makes its input available to the next iteration.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function next_iteration(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Exits the current frame to its parent frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function ref_exit(data: TFTensor; name: string = ''): TFTensor; static;
       /// <summary>
       /// Exits the current frame to its parent frame.
       /// </summary>
       /// <param name="data"></param>
       /// <param name="name"></param>
       /// <returns></returns>
       class function _exit(data: TFTensor; name: string = ''): TFTensor; static;
       class function ref_switch(data, pred: TFTensor; name: string = '') : TArray<TFTensor>; static;
       /// <summary>
       /// Forwards `data` to the output port determined by `pred`.
       ///
       /// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
       /// the data goes to `output_false`.
       ///
       /// See also `RefSwitch` and `Merge`.
       /// </summary>
       /// <param name="data">A `Tensor`. The tensor to be forwarded to the appropriate output.</param>
       /// <param name="pred">A `Tensor` of type `bool`.
       /// A scalar that specifies which output port will receive data.
       /// </param>
       /// <param name="name"> A name for the operation (optional).</param>
       /// <returns>A tuple of `Tensor` objects (output_false, output_true).
       ///
       /// output_false: A `Tensor`. Has the same type as `data`.
       /// output_true: A `Tensor`. Has the same type as `data`.
       /// </returns>
       class function switch(data, pred: TFTensor; name: string = '') : TArray<TFTensor>; static;
  end;

implementation
       uses Tensorflow,
            Tensorflow.Utils ;

{ gen_control_flow_ops }

class function gen_control_flow_ops.control_trigger(name: string): TFOperation;
begin
    var _op := tf.OpDefLib._apply_op_helper('ControlTrigger', name, []);
    Result := _op;
end;

class function gen_control_flow_ops.enter(data: TFTensor; frame_name: string; is_constant: Boolean; parallel_iterations: Integer; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Enter', name, [GetArg('data',data),GetArg('frame_name',frame_name),GetArg('is_constant',is_constant),GetArg('parallel_iterations',parallel_iterations)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.loop_cond(input: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('NoOp', name, [GetArg('input',input)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.next_iteration(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('NextIteration', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.no_op(name: string): TFOperation;
begin
    var _op := tf.OpDefLib._apply_op_helper('NoOp', name, []);
    Result := _op;
end;

class function gen_control_flow_ops.ref_next_iteration(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefNextIteration', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.ref_exit(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefExit', name, [GetArg('data',data)]);
    Result := _op.output;
end;

class function gen_control_flow_ops.ref_switch(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('RefSwitch', name, [GetArg('data',data)]);
    Result := _op.outputs;
end;

class function gen_control_flow_ops.switch(data, pred: TFTensor; name: string): TArray<TFTensor>;
begin
    var _op := tf.OpDefLib._apply_op_helper('Switch', name, [GetArg('data',data),GetArg('pred',pred)]);
    Result := [ _op.outputs[0], _op.outputs[1] ];
end;

class function gen_control_flow_ops._exit(data: TFTensor; name: string): TFTensor;
begin
    var _op := tf.OpDefLib._apply_op_helper('Exit', name, [GetArg('data',data)]);
    Result := _op.output;
end;

end.
