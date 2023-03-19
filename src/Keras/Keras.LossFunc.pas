unit Keras.LossFunc;
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

          TF4D.Core.CApi,
          TensorFlow.DApi,
          Tensorflow.Core,

          Keras.Core;


type
   Reduction = record
       const
        NONE                      : string = 'none';
        SUM                       : string = 'sum';
        WEIGHTED_SUM              : string = 'weighted_sum';
        SUM_OVER_BATCH_SIZE       : string = 'weighted_sum_over_batch_size';
        WEIGHTED_MEAN             : string = 'weighted_mean';
        SUM_BY_NONZERO_WEIGHTS    : string = 'weighted_sum_by_nonzero_weights';
        SUM_OVER_NONZERO_WEIGHTS  : string = 'weighted_sum_by_nonzero_weights';
   end;

   ReductionV2 = record
       const
         NONE                : string = 'none';
         AUTO                : string = 'auto';
         SUM                 : string = 'sum';
         SUM_OVER_BATCH_SIZE : string = 'sum_over_batch_size';
         WEIGHTED_MEAN       : string = 'weighted_mean';
   end;

   /// <summary>
   /// Loss base class.
   /// </summary>
   Loss =  class abstract
     private
       Fallow_sum_over_batch_size : Boolean;
      // Fname_scope                : string;
     protected
       Freduction   : string;
       Fname        : string;
       Ffrom_logits : boolean;
     public

       constructor Create(_reduction : string = 'auto'; _name : string = ''; _from_logits : boolean= false) ;
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; virtual;
       function    Call(y_true: TFTensor; y_pred: TFTensor; sample_weight: TFTensor = nil): TFTensor;

       property Reduction : string read Freduction;
       property Name      : string read FName;
   end;

   LossFunctionWrapper = class(Loss)
     public
       constructor Create(_reduction : string = 'auto'; _name : string = ''; _from_logits : boolean= false) ;
   end;

   TBinaryCrossentropy = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public
       label_smoothing : Single;

       constructor Create(_from_logits : Boolean= false; _label_smoothing : Single= 0; _reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TCategoricalCrossentropy = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public
       label_smoothing : Boolean;

       constructor Create(_from_logits : Boolean= false; _label_smoothing : Single= 0; _reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TCosineSimilarity = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       axis : Integer;
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _axis: Integer = -1; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   THuber = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       delta : TFTensor;
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _delta: TFTensor = nil; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TLogCosh = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TMeanAbsoluteError = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TMeanAbsolutePercentageError = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TMeanSquaredError = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TMeanSquaredLogarithmicError = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = '');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TSparseCategoricalCrossentropy = class(Loss, ILossFunc)
     private
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(_reduction : string = ''; _name: string = ''; from_logits: Boolean = False);
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   TSigmoidFocalCrossEntropy = class(Loss, ILossFunc)
     private
       FAlpha : Single;
       FGamma : Single;
       {$IFNDEF AUTOREFCOUNT}
         private const
           objDestroyingFlag = Integer($80000000);
           function GetRefCount: Integer; inline;
       {$ENDIF}
     protected
       {$IFNDEF AUTOREFCOUNT}
         [Volatile] FRefCount: Integer;
         class procedure __MarkDestroying(const Obj); static; inline;
       {$ENDIF}
       function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
       function _AddRef: Integer; stdcall;
       function _Release: Integer; stdcall;
     public

       constructor Create(from_logits: Boolean = false; alpha: Single = 0.25; gamma: Single = 2.0; reduction : string= 'none'; name: string = 'sigmoid_focal_crossentropy');
       function    Apply(y_true: TFTensor; y_pred: TFTensor; from_logits: boolean = false; axis: Integer = -1): TFTensor; override;
       {$IFNDEF AUTOREFCOUNT}
          procedure AfterConstruction; override;
          procedure BeforeDestruction; override;
          class function NewInstance: TObject; override;
          property RefCount: Integer read GetRefCount;
        {$ENDIF}
   end;

   LossesApi = class(TInterfacedObject, ILossesApi)
     public
       constructor Create;

       function  BinaryCrossentropy(from_logits : Boolean= false; label_smoothing: Single = 0; axis: Integer = -1; reduction : string= 'auto'; name: string = 'binary_crossentropy'): ILossFunc;
       function  SparseCategoricalCrossentropy(reduction : string= ''; name : string= '';from_logits: Boolean = false) : ILossFunc;
       function  CategoricalCrossentropy(reduction : string= ''; name : string= ''; from_logits: Boolean = false): ILossFunc;
       function  MeanSquaredError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanSquaredLogarithmicError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanAbsolutePercentageError(reduction : string= ''; name : string= '') : ILossFunc;
       function  MeanAbsoluteError(reduction : string= ''; name : string= '') : ILossFunc;
       function  CosineSimilarity(reduction : string= ''; name : string= '';axis: Integer=-1): ILossFunc;
       function  Huber(reduction : string= ''; name : string= ''; delta: TFTensor=nil): ILossFunc;
       function  LogCosh(reduction : string= ''; name : string= ''): ILossFunc;
       function  SigmoidFocalCrossEntropy(from_logits: Boolean = false; alpha: Single = 0.25; gamma: Single = 2.0; reduction: string = 'none'; name: string = 'sigmoid_focal_crossentropy'): ILossFunc;
   end;

implementation
         uses Tensorflow,
              TensorFlow.Ops,
              TensorFlow.Tensor,
              TensorFlow.Operations,

              NumPy.Axis,
              NumPy.NDArray,

              Keras.Utils;
{ Loss }

constructor Loss.Create(_reduction, _name: string; _from_logits: boolean);
begin
    if _reduction = '' then Freduction := ReductionV2.SUM_OVER_BATCH_SIZE
    else                    Freduction := _reduction;

    Fname                      := _name;
    Ffrom_logits               := _from_logits;
    Fallow_sum_over_batch_size := false;
end;

function Loss.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
begin
    Result := nil;
end;

function Loss.Call(y_true, y_pred, sample_weight: TFTensor): TFTensor;
begin
   var losses := Apply(y_true, y_pred, Ffrom_logits);

   var red : string := Freduction;
   if Freduction = ReductionV2.AUTO then red := ReductionV2.SUM_OVER_BATCH_SIZE;

   Result := losses_utils.compute_weighted_loss(losses, sample_weight, red);
end;

{ LossFunctionWrapper }

constructor LossFunctionWrapper.Create(_reduction, _name: string; _from_logits: boolean);
begin
   inherited Create(_reduction, _name, _from_logits);
end;

{ CategoricalCrossentropy }

constructor TCategoricalCrossentropy.Create(_from_logits: Boolean; _label_smoothing: Single; _reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'categorical_crossentropy'
    else               sName := _name;

    inherited Create(_reduction, sName, _from_logits)
end;


function TCategoricalCrossentropy.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
begin
    // Try to adjust the shape so that rank of labels = rank of logits - 1.
    Result := TKerasApi.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits);
end;

{$IFNDEF AUTOREFCOUNT}
function TCategoricalCrossentropy.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TCategoricalCrossentropy.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TCategoricalCrossentropy(Obj).FRefCount;
  until AtomicCmpExchange(TCategoricalCrossentropy(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TCategoricalCrossentropy.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TCategoricalCrossentropy.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TCategoricalCrossentropy.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TCategoricalCrossentropy(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TCategoricalCrossentropy.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TCategoricalCrossentropy._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TCategoricalCrossentropy._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ CosineSimilarity }

constructor TCosineSimilarity.Create(_reduction: string; _axis: Integer; _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'cosine_similarity'
    else               sName := _name;

    inherited Create(_reduction, sName);
    axis := _axis;
end;

function TCosineSimilarity.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_true_normalize,
  y_pred_normalize : TFTensor;
begin
    y_true_normalize := nn_impl.l2_normalize(y_true, self.axis);
    y_pred_normalize := nn_impl.l2_normalize(y_pred, self.axis);
    Result := - TTensor(math_ops.reduce_sum(TTensor(y_true_normalize) * y_pred_normalize, constant_op.constant(Self.axis) ));
end;

{$IFNDEF AUTOREFCOUNT}
function TCosineSimilarity.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TCosineSimilarity.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TCosineSimilarity(Obj).FRefCount;
  until AtomicCmpExchange(TCosineSimilarity(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TCosineSimilarity.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TCosineSimilarity.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TCosineSimilarity.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TCosineSimilarity(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TCosineSimilarity.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TCosineSimilarity._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TCosineSimilarity._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ Huber }

constructor THuber.Create(_reduction: string; _delta: TFTensor; _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'THuber'
    else               sName := _name;

    inherited Create(_reduction, sName);
    if _delta = nil then delta := tf.Variable(Double(1.0)).ToTensor
    else                 delta := _delta;
end;

function THuber.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_pred_cast, y_true_cast,
  delta, error, abs_error, half : TFTensor;
begin
    y_pred_cast := math_ops.cast(y_pred, TF_DataType.TF_FLOAT);
    y_true_cast := math_ops.cast(y_true, TF_DataType.TF_FLOAT);
    delta       := math_ops.cast(self.delta, TF_DataType.TF_FLOAT);
    error       := math_ops.subtract(y_pred_cast, y_true_cast);
    abs_error   := math_ops.abs(error);
    half        := Tops.convert_to_tensor(0.5, abs_error.dtype);
    Result := gen_math_ops.mean(array_ops.where_v2(TTensor(abs_error) <= delta,
                                                TFTensor(TTensor(half) * math_ops.pow(error, 2)),
                                                TFTensor(TTensor(half) * math_ops.pow(delta, 2) + delta * (TTensor(abs_error) - delta))), -1);
end;

{$IFNDEF AUTOREFCOUNT}
function THuber.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure THuber.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := THuber(Obj).FRefCount;
  until AtomicCmpExchange(THuber(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure THuber.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure THuber.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function THuber.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  THuber(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function THuber.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function THuber._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function THuber._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ LogCosh }

constructor TLogCosh.Create(_reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'log_cosh'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TLogCosh.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
 y_pred_dispatch,
 y_true_cast, x,s_x  : TFTensor;
begin
    y_pred_dispatch := Tops.convert_to_tensor(y_pred);
    y_true_cast     := gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
    x               := TTensor(y_pred_dispatch) - y_true_cast;

    s_x := gen_math_ops.softplus(Double(-2.0) * TTensor(x));
    Result := gen_math_ops.mean(TTensor(x) + s_x - math_ops.cast(math_ops.log(tf.Variable(Double(2.0)).ToTensor), x.dtype), -1);
end;

{$IFNDEF AUTOREFCOUNT}
function TLogCosh.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TLogCosh.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TLogCosh(Obj).FRefCount;
  until AtomicCmpExchange(TLogCosh(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TLogCosh.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TLogCosh.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TLogCosh.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TLogCosh(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TLogCosh.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TLogCosh._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TLogCosh._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ MeanAbsoluteError }

constructor TMeanAbsoluteError.Create(_reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'mean_absolute_error'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TMeanAbsoluteError.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_pred_dispatch, y_true_cast : TFTensor;
begin
    y_pred_dispatch := Tops.convert_to_tensor(y_pred);
    y_true_cast     := gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
    Result          := gen_math_ops.mean(math_ops.abs(TTensor(y_pred_dispatch) - y_true_cast), -1);
end;

{$IFNDEF AUTOREFCOUNT}
function TMeanAbsoluteError.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TMeanAbsoluteError.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TMeanAbsoluteError(Obj).FRefCount;
  until AtomicCmpExchange(TMeanAbsoluteError(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TMeanAbsoluteError.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TMeanAbsoluteError.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TMeanAbsoluteError.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TMeanAbsoluteError(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TMeanAbsoluteError.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TMeanAbsoluteError._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TMeanAbsoluteError._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ MeanAbsolutePercentageError }

constructor TMeanAbsolutePercentageError.Create(_reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'mean_absolute_percentage_error'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TMeanAbsolutePercentageError.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_pred_dispatch, y_true_cast, diff : TFTensor;
begin
    y_pred_dispatch := Tops.convert_to_tensor(y_pred);
    y_true_cast     := gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
    diff            := TTensor(math_ops.abs(TTensor(y_true_cast) - y_pred_dispatch)) / gen_math_ops.maximum(math_ops.abs(y_true_cast), gen_math_ops.cast(tf.constant(Double(1e-7)), y_pred_dispatch.dtype));
    Result          := gen_math_ops.cast(tf.constant(Integer(100)), y_pred_dispatch.dtype) * TTensor(gen_math_ops.mean(diff, -1));
end;

{$IFNDEF AUTOREFCOUNT}
function TMeanAbsolutePercentageError.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TMeanAbsolutePercentageError.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TMeanAbsolutePercentageError(Obj).FRefCount;
  until AtomicCmpExchange(TMeanAbsolutePercentageError(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TMeanAbsolutePercentageError.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TMeanAbsolutePercentageError.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TMeanAbsolutePercentageError.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TMeanAbsolutePercentageError(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TMeanAbsolutePercentageError.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TMeanAbsolutePercentageError._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TMeanAbsolutePercentageError._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ MeanSquaredError }

constructor TMeanSquaredError.Create(_reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'mean_squared_error'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TMeanSquaredError.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_pred_dispatch, y_true_cast : TFTensor;
begin
    y_pred_dispatch := Tops.convert_to_tensor(y_pred);
    y_true_cast     := gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
    REsult          := gen_math_ops.mean(gen_math_ops.squared_difference(y_pred_dispatch, y_true_cast), -1);
end;

{$IFNDEF AUTOREFCOUNT}
function TMeanSquaredError.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TMeanSquaredError.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TMeanSquaredError(Obj).FRefCount;
  until AtomicCmpExchange(TMeanSquaredError(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TMeanSquaredError.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TMeanSquaredError.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TMeanSquaredError.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TMeanSquaredError(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TMeanSquaredError.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TMeanSquaredError._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TMeanSquaredError._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ MeanSquaredLogarithmicError }

constructor TMeanSquaredLogarithmicError.Create(_reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'mean_squared_logarithmic_error'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TMeanSquaredLogarithmicError.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
var
  y_pred_dispatch,y_true_cast : TFTensor;
  first_log, second_log       : TFTensor;
begin
    y_pred_dispatch := Tops.convert_to_tensor(y_pred);
    y_true_cast     := gen_math_ops.cast(y_true, y_pred_dispatch.dtype);

    if y_pred_dispatch.dtype = TF_DataType.TF_DOUBLE then
    begin
        first_log  := math_ops.log(TTensor(gen_math_ops.maximum(y_pred_dispatch, Double(1e-7))) + Double(1.0));
        second_log := math_ops.log(TTensor(gen_math_ops.maximum(y_true_cast, Double(1e-7)))     + Double(1.0));
    end else
    begin
        first_log  := math_ops.log(TTensor(gen_math_ops.maximum(y_pred_dispatch, Single(1e-7))) + Single(1.0));
        second_log := math_ops.log(TTensor(gen_math_ops.maximum(y_true_cast, Single(1e-7)))     + Single(1.0));
    end;
    Result := gen_math_ops.mean(gen_math_ops.squared_difference(first_log, second_log), -1);
end;

{$IFNDEF AUTOREFCOUNT}
function TMeanSquaredLogarithmicError.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TMeanSquaredLogarithmicError.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TMeanSquaredLogarithmicError(Obj).FRefCount;
  until AtomicCmpExchange(TMeanSquaredLogarithmicError(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TMeanSquaredLogarithmicError.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TMeanSquaredLogarithmicError.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TMeanSquaredLogarithmicError.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TMeanSquaredLogarithmicError(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TMeanSquaredLogarithmicError.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TMeanSquaredLogarithmicError._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TMeanSquaredLogarithmicError._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ SparseCategoricalCrossentropy }

constructor TSparseCategoricalCrossentropy.Create(_reduction, _name: string; from_logits: Boolean);
var
  sName : string;
begin
    if _name = '' then sName := 'sparse_categorical_crossentropy'
    else               sName := _name;

    inherited Create(_reduction, sName);
end;

function TSparseCategoricalCrossentropy.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
begin
    y_true := tf.cast(y_true, TF_DataType.TF_INT64);

    if not from_logits then
    begin
        var epsilon := tf.constant(TKerasApi.Keras.backend.epsilon, y_pred.dtype);
        y_pred      := tf.clip_by_value(y_pred, epsilon, 1 - TTEnsor(epsilon));
        y_pred      := tf.log(y_pred);
    end;

    // Try to adjust the shape so that rank of labels = rank of logits - 1.
    var output_shape := array_ops.shape_v2(y_pred);
    var output_rank  := y_pred.shape.ndim;
    var target_rank  := y_true.shape.ndim;
    var update_shape := target_rank <> output_rank - 1;
    if update_shape then
    begin
        var sA  : TArray<Integer> := [-1];
        var n : NDArray :=  output_shape[-1].numpy;
        var sA1 : TArray<Integer> := [-1, n ];
        y_true := array_ops.reshape(y_true, sA);
        y_pred := array_ops.reshape(y_pred, sA1);
    end;
    Result := tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred);
end;

{$IFNDEF AUTOREFCOUNT}
function TSparseCategoricalCrossentropy.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TSparseCategoricalCrossentropy.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TSparseCategoricalCrossentropy(Obj).FRefCount;
  until AtomicCmpExchange(TSparseCategoricalCrossentropy(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TSparseCategoricalCrossentropy.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TSparseCategoricalCrossentropy.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TSparseCategoricalCrossentropy.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TSparseCategoricalCrossentropy(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TSparseCategoricalCrossentropy.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TSparseCategoricalCrossentropy._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TSparseCategoricalCrossentropy._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ TBinaryCrossentropy }

constructor TBinaryCrossentropy.Create(_from_logits: Boolean; _label_smoothing: Single; _reduction, _name: string);
var
  sName : string;
begin
    if _name = '' then sName := 'binary_crossentropy'
    else               sName := _name;

    inherited Create(_reduction, sName, _from_logits);

    label_smoothing := _label_smoothing;
end;

function TBinaryCrossentropy.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
begin
    var sum := TKerasApi.keras.backend.binary_crossentropy(y_true, y_pred, from_logits);
    Result  := TKerasApi.keras.backend.mean(sum, axis);
end;

{$IFNDEF AUTOREFCOUNT}
function TBinaryCrossentropy.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TBinaryCrossentropy.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TBinaryCrossentropy(Obj).FRefCount;
  until AtomicCmpExchange(TBinaryCrossentropy(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TBinaryCrossentropy.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TBinaryCrossentropy.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TBinaryCrossentropy.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TBinaryCrossentropy(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TBinaryCrossentropy.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TBinaryCrossentropy._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TBinaryCrossentropy._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ TSigmoidFocalCrossEntropy }

constructor TSigmoidFocalCrossEntropy.Create(from_logits: Boolean; alpha, gamma: Single; reduction, name: string);
begin
    inherited Create(reduction, Name, from_logits);
    Falpha  := alpha;
    Fgamma  := gamma;
end;

function TSigmoidFocalCrossEntropy.Apply(y_true, y_pred: TFTensor; from_logits: boolean; axis: Integer): TFTensor;
begin
    y_true := tf.cast(y_true, y_pred.dtype);
    var ce := TKerasApi.keras.backend.binary_crossentropy(y_true, y_pred, from_logits);
    var pred_prob : TTensor ;
    if from_logits then
     pred_prob := tf.sigmoid(y_pred)
    else
      pred_prob := y_pred;

    var p_t               : TTensor := (y_true * pred_prob) + ((Single(1) - TTensor(y_true)) * (Single(1) - pred_prob));
    var alpha_factor      : TTensor := constant_op.constant(Single(1.0));
    var modulating_factor : TTensor := constant_op.constant(Single(1.0));

    if Falpha > 0 then
    begin
        var alpha : TTensor := tf.cast(constant_op.constant(Falpha), y_true.dtype);
        alpha_factor        := y_true * alpha + (Single(1) - TTensor(y_true)) * (Single(1) - alpha);
    end;

    if Fgamma > 0 then
    begin
        var gamma         := tf.cast(constant_op.constant(Fgamma), y_true.dtype);
        modulating_factor := tf.pow(Single(1) - p_t, gamma);
    end;

    var assi : Taxis := -1;
    Result := tf.reduce_sum(alpha_factor * modulating_factor * ce, @assi);
end;

{$IFNDEF AUTOREFCOUNT}
function TSigmoidFocalCrossEntropy.GetRefCount: Integer;
begin
  Result := FRefCount and not objDestroyingFlag;
end;

class procedure TSigmoidFocalCrossEntropy.__MarkDestroying(const Obj);
var
  LRef: Integer;
begin
  repeat
    LRef := TSigmoidFocalCrossEntropy(Obj).FRefCount;
  until AtomicCmpExchange(TSigmoidFocalCrossEntropy(Obj).FRefCount, LRef or objDestroyingFlag, LRef) = LRef;
end;

procedure TSigmoidFocalCrossEntropy.AfterConstruction;
begin
// Release the constructor's implicit refcount
  AtomicDecrement(FRefCount);
end;

procedure TSigmoidFocalCrossEntropy.BeforeDestruction;
begin
  if RefCount <> 0 then
    Error(reInvalidPtr);
end;

// Set an implicit refcount so that refcounting during construction won't destroy the object.
class function TSigmoidFocalCrossEntropy.NewInstance: TObject;
begin
  Result := inherited NewInstance;
  TSigmoidFocalCrossEntropy(Result).FRefCount := 1;
end;

{$ENDIF AUTOREFCOUNT}

function TSigmoidFocalCrossEntropy.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TSigmoidFocalCrossEntropy._AddRef: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicIncrement(FRefCount);
{$ELSE}
  Result := __ObjAddRef;
{$ENDIF}
end;

function TSigmoidFocalCrossEntropy._Release: Integer;
begin
{$IFNDEF AUTOREFCOUNT}
  Result := AtomicDecrement(FRefCount);
  if Result = 0 then
  begin
    // Mark the refcount field so that any refcounting during destruction doesn't infinitely recurse.
    __MarkDestroying(Self);
    Destroy;
  end;
{$ELSE}
  Result := __ObjRelease;
{$ENDIF}
end;
{$ENDREGION}

{ LossesApi }

constructor LossesApi.Create;
begin
  inherited;

end;

function LossesApi.BinaryCrossentropy(from_logits: Boolean; label_smoothing: Single; axis: Integer; reduction, name: string): ILossFunc;
begin
    Result := TBinaryCrossentropy.Create(from_logits, label_smoothing, reduction, name)
end;

function LossesApi.CategoricalCrossentropy(reduction, name: string; from_logits: Boolean): ILossFunc;
begin
    Result := TCategoricalCrossentropy.Create(from_logits, 0,reduction, name);
end;

function LossesApi.CosineSimilarity(reduction, name: string; axis: Integer): ILossFunc;
begin
    Result := TCosineSimilarity.Create(reduction, axis, name );
end;

function LossesApi.Huber(reduction, name: string; delta: TFTensor): ILossFunc;
begin
    Result := THuber.Create(reduction, delta, name );
end;

function LossesApi.LogCosh(reduction, name: string): ILossFunc;
begin
    Result := TLogCosh.Create(reduction, name);
end;

function LossesApi.MeanAbsoluteError(reduction, name: string): ILossFunc;
begin
    Result := TMeanAbsoluteError.Create(reduction, name);
end;

function LossesApi.MeanAbsolutePercentageError(reduction, name: string): ILossFunc;
begin
    Result := TMeanAbsolutePercentageError.Create(reduction, name);
end;

function LossesApi.MeanSquaredError(reduction, name: string): ILossFunc;
begin
     Result := TMeanSquaredError.Create(reduction, name);
end;

function LossesApi.MeanSquaredLogarithmicError(reduction, name: string): ILossFunc;
begin
    Result := TMeanSquaredLogarithmicError.Create(reduction, name);
end;

function LossesApi.SigmoidFocalCrossEntropy(from_logits: Boolean; alpha, gamma: Single; reduction, name: string): ILossFunc;
begin
    Result := TSigmoidFocalCrossEntropy.Create(from_logits, alpha, gamma, reduction, name)
end;

function LossesApi.SparseCategoricalCrossentropy(reduction, name: string; from_logits: Boolean): ILossFunc;
begin
    Result := TSparseCategoricalCrossentropy.Create(reduction, name, from_logits);
end;

end.
