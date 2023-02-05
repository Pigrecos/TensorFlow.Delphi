object frmMain: TfrmMain
  Left = 0
  Top = 0
  Caption = 'TensorFlow for Delphi'
  ClientHeight = 396
  ClientWidth = 824
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OnShow = FormShow
  TextHeight = 13
  object btnTest: TBitBtn
    Left = 8
    Top = 363
    Width = 89
    Height = 25
    Caption = 'Test'
    TabOrder = 0
    OnClick = btnTestClick
  end
  object mmo1: TMemo
    Left = 8
    Top = 8
    Width = 801
    Height = 349
    Lines.Strings = (
      'mmo1')
    ScrollBars = ssVertical
    TabOrder = 1
  end
  object btnLinReg: TBitBtn
    Left = 103
    Top = 363
    Width = 162
    Height = 25
    Caption = 'Linear Regression -Graph mode-'
    TabOrder = 2
    OnClick = btnLinRegClick
  end
  object btnLinReg1: TBitBtn
    Left = 271
    Top = 363
    Width = 162
    Height = 25
    Caption = 'Linear Regression -Eager mode-'
    TabOrder = 3
    OnClick = btnLinReg1Click
  end
  object btnKerasLayers: TBitBtn
    Left = 439
    Top = 363
    Width = 106
    Height = 25
    Caption = 'Keras Layers'
    TabOrder = 4
    OnClick = btnKerasLayersClick
  end
  object btnModels: TBitBtn
    Left = 734
    Top = 363
    Width = 75
    Height = 25
    Caption = 'Test Models'
    TabOrder = 5
    OnClick = btnModelsClick
  end
  object btnPreProcess: TBitBtn
    Left = 551
    Top = 363
    Width = 113
    Height = 25
    Caption = 'PreProcessing'
    TabOrder = 6
    OnClick = btnPreProcessClick
  end
end
