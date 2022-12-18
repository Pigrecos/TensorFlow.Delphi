object frmMain: TfrmMain
  Left = 0
  Top = 0
  Caption = 'TensorFlow for Delphi'
  ClientHeight = 299
  ClientWidth = 671
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OnShow = FormShow
  TextHeight = 13
  object btnTest: TBitBtn
    Left = 495
    Top = 266
    Width = 168
    Height = 25
    Caption = 'Test'
    TabOrder = 0
    OnClick = btnTestClick
  end
  object mmo1: TMemo
    Left = 8
    Top = 8
    Width = 481
    Height = 283
    Lines.Strings = (
      'mmo1')
    ScrollBars = ssVertical
    TabOrder = 1
  end
  object btnLinReg: TBitBtn
    Left = 495
    Top = 224
    Width = 168
    Height = 25
    Caption = 'Linear Regression -Graph mode-'
    TabOrder = 2
    OnClick = btnLinRegClick
  end
  object btnLinReg1: TBitBtn
    Left = 495
    Top = 184
    Width = 168
    Height = 25
    Caption = 'Linear Regression -Eager mode-'
    TabOrder = 3
    OnClick = btnLinReg1Click
  end
  object btnKerasLayers: TBitBtn
    Left = 495
    Top = 8
    Width = 168
    Height = 25
    Caption = 'Keras Layers'
    TabOrder = 4
  end
end
