object frmMain: TfrmMain
  Left = 0
  Top = 0
  Caption = 'TensorFlow for Delphi'
  ClientHeight = 568
  ClientWidth = 768
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OnShow = FormShow
  TextHeight = 13
  object pnl1: TPanel
    Left = 0
    Top = 527
    Width = 768
    Height = 41
    Align = alBottom
    TabOrder = 0
    ExplicitTop = 526
    ExplicitWidth = 764
    object btnModels: TBitBtn
      Left = 678
      Top = 8
      Width = 75
      Height = 25
      Caption = 'Test Models'
      TabOrder = 0
      OnClick = btnModelsClick
    end
    object btnPreProcess: TBitBtn
      Left = 551
      Top = 8
      Width = 113
      Height = 25
      Caption = 'PreProcessing'
      TabOrder = 1
      OnClick = btnPreProcessClick
    end
    object btnKerasLayers: TBitBtn
      Left = 439
      Top = 6
      Width = 106
      Height = 25
      Caption = 'Keras Layers'
      TabOrder = 2
      OnClick = btnKerasLayersClick
    end
    object btnLinReg1: TBitBtn
      Left = 271
      Top = 8
      Width = 162
      Height = 25
      Caption = 'Linear Regression -Eager mode-'
      TabOrder = 3
      OnClick = btnLinReg1Click
    end
    object btnLinReg: TBitBtn
      Left = 103
      Top = 8
      Width = 162
      Height = 25
      Caption = 'Linear Regression -Graph mode-'
      TabOrder = 4
      OnClick = btnLinRegClick
    end
    object btnTest: TBitBtn
      Left = 8
      Top = 8
      Width = 89
      Height = 25
      Caption = 'Test'
      TabOrder = 5
      OnClick = btnTestClick
    end
  end
  object pnl2: TPanel
    Left = 0
    Top = 0
    Width = 768
    Height = 527
    Align = alClient
    TabOrder = 1
    ExplicitWidth = 764
    ExplicitHeight = 526
    object mmo1: TMemo
      Left = 1
      Top = 1
      Width = 766
      Height = 525
      Align = alClient
      Lines.Strings = (
        'mmo1')
      ScrollBars = ssVertical
      TabOrder = 0
      ExplicitWidth = 762
      ExplicitHeight = 524
    end
  end
end
