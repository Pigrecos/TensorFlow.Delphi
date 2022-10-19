object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'TensorFlow for Delphi'
  ClientHeight = 299
  ClientWidth = 635
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
    Width = 132
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
    TabOrder = 1
  end
  object btnLinReg: TBitBtn
    Left = 495
    Top = 224
    Width = 132
    Height = 25
    Caption = 'Linear Regression'
    TabOrder = 2
    OnClick = btnLinRegClick
  end
end
