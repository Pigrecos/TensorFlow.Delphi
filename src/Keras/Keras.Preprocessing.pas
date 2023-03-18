unit Keras.Preprocessing;
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
     uses System.SysUtils, System.IOUtils,
          System.Rtti,
          System.Generics.Defaults,
          System.Generics.Collections,

          Spring,
          Spring.Collections.Sets,
          Spring.Collections.Enumerable,

          TensorFlow.DApi,

          Keras.Core,
          Keras.Layer,
          Keras.Data;

const
   DefaultFilter = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'+ #9+#13#10;

type
  Tokenizer  = class;

  TSequence = class
    public
      /// <summary>
      /// Pads sequences to the same length.
      /// https://keras.io/preprocessing/sequence/
      /// https://faroit.github.io/keras-docs/1.2.0/preprocessing/sequence/
      /// </summary>
      /// <param name="sequences">List of lists, where each element is a sequence.</param>
      /// <param name="maxlen">Int, maximum length of all sequences.</param>
      /// <param name="dtype">Type of the output sequences.</param>
      /// <param name="padding">String, 'pre' or 'post':</param>
      /// <param name="truncating">String, 'pre' or 'post'</param>
      /// <param name="value">Float or String, padding value.</param>
      /// <returns></returns>
      function  pad_sequences(sequences: TList< TArray<Integer> >; maxlen : Integer= 0; dtype : string = 'int32'; padding : string = 'pre'; truncating: string = 'pre'; value : TObject= nil): TNDArray;
  end;

  DatasetUtils = class
    public
      function  labels_to_dataset(labels: TArray<Integer>; label_mode: string; num_classes: Integer) : IDatasetV2;
      /// <summary>
      /// Potentially restict samples & labels to a training or validation split.
      /// </summary>
      /// <param name="samples"></param>
      /// <param name="labels"></param>
      /// <param name="validation_split"></param>
      /// <param name="subset"></param>
      /// <returns></returns>
      function get_training_or_validation_split<T1, T2>(samples: TArray<T1>; labels: TArray<T2>; validation_split: Single; subset: string) : Tuple< TArray<T1>, TArray<T2> >;
      /// <summary>
      /// Make list of all files in the subdirs of `directory`, with their labels.
      /// </summary>
      /// <param name="directory"></param>
      /// <param name="labels"></param>
      /// <param name="formats"></param>
      /// <param name="class_names"></param>
      /// <param name="shuffle"></param>
      /// <param name="seed"></param>
      /// <param name="follow_links"></param>
      /// <returns>
      /// file_paths, labels, class_names
      /// </returns>
      function index_directory(directory: string; labels: string; formats: TArray<string> = []; class_names: TArray<string> = []; shuffle: Boolean = true; seed : PInteger= nil; follow_links : Boolean= false): Tuple<TArray<string>, TArray<Integer>, TArray<string>>;
  end;

  TextApi  = class
    public
      function Tokenizer(num_words: Integer = -1; filters: string = DefaultFilter; lower: Boolean= true; split : Char = ' '; char_level: Boolean = false; oov_token: string = ''; analyzer : TFunc<string, TArray<string>> = nil): Keras.Preprocessing.Tokenizer;
      class function text_to_word_sequence(text: string; filters: string = DefaultFilter; lower : Boolean= true; split : Char= ' ') : TArray<string>; static;
  end;

  /// <summary>
  /// Text tokenization API.
  /// This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
  /// (each integer being the index of a token in a dictionary) or into a vector where the coefficient for
  /// each token could be binary, based on word count, based on tf-idf...
  /// </summary>
  /// <remarks>
  /// This code is a fairly straight port of the Python code for Keras text preprocessing found at:
  /// https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
  /// </remarks>
  Tokenizer  = class
     private
       const modes : array[0..3] of string = ( 'binary', 'count', 'tfidf', 'freq' );
     private
       Fnum_words  : Integer;
       Ffilters    : string;
       Flower      : Boolean;
       Fsplit      : Char;
       Fchar_level : Boolean;
       Foov_token  : string;
       Fanalyzer   : TFunc<string, TArray<string>> ;

       Fdocument_count : Integer;

       Fword_docs  : TDictionary<string, Integer>;
       Fword_counts: TDictionary<string, Integer>;
       Findex_docs : TDictionary<Integer, Integer>;

       function ConvertToSequence(oov_index: Integer; seq: TArray<string>): TList<integer>;
     protected

     public
       word_index  : TDictionary<string, Integer>;
       index_word  : TDictionary<Integer, string>;

       /// <summary>
       /// Updates internal vocabulary based on a list of texts.
       /// </summary>
       /// <param name="texts">A list of strings, each containing one or more tokens.</param>
       /// <remarks>Required before using texts_to_sequences or texts_to_matrix.</remarks>
       procedure fit_on_texts(texts: TArray<string>); overload;
       /// <summary>
       /// Updates internal vocabulary based on a list of texts.
       /// </summary>
       /// <param name="texts">A list of list of strings, each containing one token.</param>
       /// <remarks>Required before using texts_to_sequences or texts_to_matrix.</remarks>
       procedure fit_on_texts(texts: TArray<TArray<string>>); overload;
       /// <summary>
       /// Updates internal vocabulary based on a list of sequences.
       /// </summary>
       /// <param name="sequences"></param>
       /// <remarks>Required before using sequences_to_matrix (if fit_on_texts was never called).</remarks>
       procedure fit_on_sequences(sequences: TList<TArray<Integer>>);
       /// <summary>
       /// Transforms each string in texts to a sequence of integers.
       /// </summary>
       /// <param name="texts"></param>
       /// <returns></returns>
       /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
       function texts_to_sequences(texts: TArray<string>): TList< TArray<Integer> >; overload;
       /// <summary>
       /// Transforms each token in texts to a sequence of integers.
       /// </summary>
       /// <param name="texts"></param>
       /// <returns></returns>
       /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
       function texts_to_sequences(texts: TArray<TArray<string>>): TList< TArray<Integer> >; overload;
       function texts_to_sequences_generator(texts: TArray<string>): TList< TArray<Integer> >; overload;
       function texts_to_sequences_generator(texts: TArray<TArray<string>>): TList< TArray<Integer> >; overload;
       /// <summary>
       /// Transforms each sequence into a list of text.
       /// </summary>
       /// <param name="sequences"></param>
       /// <returns>A list of texts(strings)</returns>
       /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
       function sequences_to_texts(sequences: TList<TArray<Integer>>): TList<string>;
       function sequences_to_texts_generator(sequences: TList<TArray<Integer>>): TList<string>;
       /// <summary>
       /// Convert a list of texts to a Numpy matrix.
       /// </summary>
       /// <param name="texts">A sequence of strings containing one or more tokens.</param>
       /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
       /// <returns></returns>
       function texts_to_matrix(texts: TArray<string>; mode : string= 'binary'): TNDArray;  overload;
       /// <summary>
       /// Convert a list of texts to a Numpy matrix.
       /// </summary>
       /// <param name="texts">A sequence of lists of strings, each containing one token.</param>
       /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
       /// <returns></returns>
       function texts_to_matrix(texts: TArray<TArray<string>>; mode: string = 'binary'): TNDArray;  overload;
       /// <summary>
       /// Converts a list of sequences into a Numpy matrix.
       /// </summary>
       /// <param name="sequences">A sequence of lists of integers, encoding tokens.</param>
       /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
       /// <returns></returns>
       function sequences_to_matrix(sequences: TList<TArray<integer>>; mode: string= 'binary'): TNDArray;

       constructor Create(num_words : Integer= -1; filters : string= DefaultFilter; lower : Boolean= true; split : Char= ' '; char_level : Boolean= false; oov_token : string = ''; analyzer : TFunc<string, TArray<string>> = nil) ;overload;
       constructor Create(oov_token : string) ;overload;
       destructor Destroy; override;
  end;

  TPreprocessing = class(TInterfacedObject, IPreprocessing)
     protected
       Ftext : TextApi;
     public
        sequence     : TSequence;
        dataset_utils: DatasetUtils;

        function Resizing(height: Integer; width: Integer; interpolation: string = 'bilinear'): ILayer;
        function TextVectorization(standardize: TFunc<TFTensor, TFTensor> = nil; split : string= 'whitespace'; max_tokens: Integer = -1; output_mode: string = 'int'; output_sequence_length: Integer = -1): ILayer;

        constructor Create;
        destructor  Destroy; override;

        property text : TextApi read Ftext;
  end;

implementation
         uses Tensorflow.Utils,
              tensorflow.Slice,
              Tensorflow,
              Numpy,
              NumPy.NDArray;

{ Preprocessing }

constructor TPreprocessing.Create;
begin
   sequence      := TSequence.Create;
   dataset_utils := DatasetUtils.Create;
   Ftext         := TextApi.Create;
end;

destructor TPreprocessing.Destroy;
begin
  sequence.Free;
  dataset_utils.Free;
  Ftext.Free;
  inherited;
end;

function TPreprocessing.Resizing(height, width: Integer; interpolation: string): ILayer;
var
  args: ResizingArgs;
begin
    args := ResizingArgs.Create;

    args.Height          := height;
    args.Width           := width;
    args.Interpolation   := interpolation;

    Result := Keras.Layer.Resizing.create(args)

end;

function TPreprocessing.TextVectorization(standardize: TFunc<TFTensor, TFTensor>; split: string; max_tokens: Integer; output_mode: string;
  output_sequence_length: Integer): ILayer;
var
  args: TextVectorizationArgs;
begin
    args := TextVectorizationArgs.Create;

    args.Standardize := standardize;
    args.Split       := split;
    args.MaxTokens   := max_tokens;
    args.OutputMode  := output_mode;
    args.OutputSequenceLength := output_sequence_length;

    Result := Keras.Layer.TextVectorization.create(args)
end;

{ TSequence }

function TSequence.pad_sequences(sequences: TList<TArray<Integer>>; maxlen: Integer; dtype, padding, truncating: string; value: TObject): TNDArray;
var
  ilengths: TArray<Integer>;
  vValue  : TValue;
begin
  if value <> nil then
    raise Exception.Create('padding with a specific value is not supported');
  if (padding <> 'pre') and (padding <> 'post') then
    raise Exception.Create('padding must be ''pre'' or ''post''.');
  if (truncating <> 'pre') and (truncating <> 'post') then
    raise Exception.Create('truncating must be ''pre'' or ''post''.');

  var eSeq : Enumerable<TArray<Integer> > := Enumerable<TArray<Integer> >.Create(sequences.ToArray);
  ilengths := eSeq.Select<Integer>( function( e:TArray<Integer>): Integer
                            begin
                                 Result := Length(e);
                            end).ToArray;

  if maxlen = 0 then
    maxlen :=  Enumerable<Integer>.Create(ilengths).Max;

  if value = nil then
       vValue := Single(0);

    var tipo := Tdtypes.as_tf_dtype_fromName(dtype);
    var nd   := np.zeros( TFShape.Create([Length(ilengths), maxlen]), tipo );

    for var i := 0 to nd.dims[0] - 1 do
    begin
        var s := eSeq.ElementAt(i);
        var eS := Enumerable<Integer>.Create(s);
        if Length(s) > maxlen then
        begin
            if truncating = 'pre' then s := eS.Skip(eS.Count- maxlen).ToArray
            else                       s := eS.Take(maxlen).ToArray
        end;
        var sliceString : string ;
        if padding = 'pre' then   sliceString := Format('%d,%d:',[i,maxlen- Length(s)])
        else                      sliceString := Format('%d,:%d',[i,Length(s)]);

        var slices : TArray<Slice> := [];
        var splitStr := sliceString.Split([',']) ;
        for var j := 0 to Length( splitStr ) - 1  do
           slices := slices + [ Slice.Create(splitStr[j]) ];

        nd[slices] := np.np_array(s);
    end;

    Result := nd;
end;

{ DatasetUtils }

function DatasetUtils.get_training_or_validation_split<T1, T2>(samples: TArray<T1>; labels: TArray<T2>; validation_split: Single; subset: string): Tuple<TArray<T1>, TArray<T2>>;
var
  num_val_samples : Integer;
  samplesEnum     : TArray<T1>;
  labelsEnum      : TArray<T2>;
  eSamples        : Enumerable<T1> ;
  elabels         : Enumerable<T2> ;
begin
    if subset.IsEmpty then
    Exit(Tuple<TArray<T1>, TArray<T2>>.Create(samples, labels));

    eSamples := Enumerable<T1>.Create(samples);
    elabels  := Enumerable<T2>.Create(labels);

    num_val_samples := Trunc(Length(samples) * validation_split);
    if subset = 'training' then
    begin
        tf.LogMsg('Using ' + (Length(samples) - num_val_samples).ToString + ' files for training.');
        samplesEnum := eSamples.Take(Length(samples) - num_val_samples).ToArray;
        labelsEnum  := elabels.Take(Length(labels) - num_val_samples).ToArray;
    end
    else if subset = 'validation' then
    begin
      tf.LogMsg('Using ' + num_val_samples.ToString + ' files for validation.');
      samplesEnum := eSamples.Skip(Length(samples) - num_val_samples).ToArray;
      labelsEnum  := elabels.Skip(Length(labels) - num_val_samples).ToArray;
    end
    else
      raise Exception.Create('subset must be ''training'' or  ''validation''. ');

    Result := Tuple<TArray<T1>, TArray<T2>>.Create(samplesEnum, labelsEnum)
end;

function DatasetUtils.index_directory(directory, labels: string; formats, class_names: TArray<string>; shuffle: Boolean; seed: PInteger;
  follow_links: Boolean): Tuple<TArray<string>, TArray<Integer>, TArray<string>>;
var
  label_list : TList<Integer>;
  file_paths : TList<string>;
begin
    label_list := TList<integer>.Create;
    file_paths := TList<string>.Create;
    try
      var class_dirs := TDirectory.GetDirectories(directory);
      for var i := 0 to Length(class_dirs) - 1 do
      begin
          var dirs    := class_dirs[i].Split(TPath.DirectorySeparatorChar);
          class_names := class_names + [ dirs[High(dirs)] ];
      end;

      for var l_Label := 0 to Length(class_dirs) - 1 do
      begin
          var files := TDirectory.GetFiles(class_dirs[l_Label]);
          file_paths.AddRange(files);
          for var i := 0 to Length(files) -1 do
            label_list.Add(l_Label) ;
      end;

      var return_labels     := label_list.ToArray;
      var return_file_paths := file_paths.ToArray;

      if shuffle then
      begin
          var iSeed : Integer;
          if not Assigned(seed) then
          begin
              var nd : NDArray := np.random.randint(Trunc(1e6));
              iSeed := nd;
          end else
          begin
              iSeed := seed^;
          end;
          var random_index := np.arange(label_list.Count);
          tf.set_random_seed(iSeed);
          np.random.shuffle(random_index);
          var index := random_index.ToArray<Integer>;

          for var i := 0 to label_list.Count -1 do
          begin
              return_labels[i]     := label_list[index[i]];
              return_file_paths[i] := file_paths[index[i]];
          end;
      end;

      tf.LogMsg( 'Found ' + Length(return_file_paths).ToString + ' files belonging to '+ Length(class_names).ToString+ ' classes.');
      Result := Tuple.Create(return_file_paths, return_labels, class_names);
    finally
      label_list.free;
      file_paths.free;
    end;
end;

function DatasetUtils.labels_to_dataset(labels: TArray<Integer>; label_mode: string; num_classes: Integer): IDatasetV2;
begin
    var label_ds := tf.data.Dataset.from_tensor_slices( TNDArray.Create(labels) );

    if label_mode = 'binary' then
       raise Exception.Create('Not Implemented')
    else if label_mode = 'categorical' then
        raise Exception.Create('Not Implemented');

    Result := label_ds;
end;

{ TextApi }

class function TextApi.text_to_word_sequence(text, filters: string; lower: Boolean; split: Char): TArray<string>;
begin
    if lower then
      text := text.ToLower;

    var newText : string := '';
    for var i := 1 to text.Length do
      if not filters.Contains( text[i] ) then
         newText := newText + text[i];

    Result := newText.Split([split]) ;
end;

function TextApi.Tokenizer(num_words: Integer; filters: string; lower: Boolean; split: Char; char_level: Boolean; oov_token: string;
  analyzer: TFunc<string, TArray<string>>): Keras.Preprocessing.Tokenizer;
begin
    Result := Keras.Preprocessing.Tokenizer.Create(num_words, filters, lower, split, char_level, oov_token, analyzer);
end;

{ Tokenizer }

constructor Tokenizer.Create(num_words: Integer; filters: string; lower: Boolean; split: Char; char_level: Boolean; oov_token: string; analyzer: TFunc<string, TArray<string>>);
begin
    Fdocument_count  := 0;
    word_index  := nil;
    index_word  := nil;

    Fword_docs  := TDictionary<string, Integer>.Create;
    Fword_counts:= TDictionary<string, Integer>.Create;
    Findex_docs := nil;

    Fnum_words := num_words;
    Ffilters   := filters;
    Flower     := lower;
    Fsplit     := split;
    Fchar_level:= char_level;
    Foov_token := oov_token;
    if Assigned(analyzer) then Fanalyzer := analyzer
    else  Fanalyzer := function(text: string): TArray<string>
                        begin
                            Result := TextApi.text_to_word_sequence(text, filters, lower, split);
                        end;
end;

constructor Tokenizer.Create(oov_token: string);
begin
    Create(-1, DefaultFilter, true, ' ', false, oov_token, nil);
end;

destructor Tokenizer.Destroy;
begin
  Fword_docs.Free;
  Fword_counts.Free;

  if Assigned(word_index) then
      word_index.Free;
  if Assigned(index_word) then
     index_word.Free;

  if Assigned(Findex_docs) then
     Findex_docs.Free;

  inherited Destroy;
end;

function Tokenizer.ConvertToSequence(oov_index: Integer; seq: TArray<string>): TList<integer>;
var
   vect : TList<Integer>;
   w,s  : string;
   i    : Integer;
begin
    vect := TList<Integer>.Create;
    for  w in seq  do
    begin
        s := w;
        if Flower then   s := s.ToLower;
        i := -1;
        if word_index.TryGetValue(w, i) then
        begin
            if (Fnum_words <> -1) and (i >= Fnum_words) then
            begin
                if oov_index <> -1 then
                begin
                    vect.Add(oov_index);
                end;
            end else
            begin
                vect.Add(i);
            end;
        end
        else if oov_index <> -1 then
        begin
            vect.Add(oov_index);
        end;
    end;

    Result := vect;
end;

procedure Tokenizer.fit_on_sequences(sequences: TList<TArray<Integer>>);
begin
   raise Exception.Create('fit_on_sequences - Not Implemented');
end;

procedure Tokenizer.fit_on_texts(texts: TArray<string>);
var
  text,w    : string;
  seq       : TArray<string>;
  hSet      : THashSet<string>;
  sorted_voc: Enumerable<string>;
  i,count   : Integer;
  wcounts   : TArray<TPair<string,Integer>>;
  AwordSort : TArray<string>;
begin
    for i := 0 to Length(texts) -1 do
    begin
        seq := nil;
        text := texts[i];
        if Flower then  text := text.ToLower;

        Fdocument_count := Fdocument_count + 1;

        if Fchar_level then  raise Exception.Create('Not Implemented "char_level == true"')
        else                 seq := Fanalyzer(text);

        for w in seq do
        begin
            count := 0;
            Fword_counts.TryGetValue(w, count);
            Fword_counts.AddOrSetValue(w, count + 1);
        end;

        hSet := THashSet<string>.Create(seq);
        try
          for w in hSet do
          begin
              count := 0;
              Fword_docs.TryGetValue(w, count);
              Fword_docs.AddOrSetValue(w, count + 1);
          end;
        finally
          hSet.Free;
        end;
    end;

    wcounts := Fword_counts.ToArray;
    TArray.Sort<TPair<string,Integer>>(wcounts, TDelegatedComparer<TPair<string,Integer>>.Construct(function(const Left, Right: TPair<string,Integer>): Integer
                                                                                                      begin
                                                                                                        Result := -TComparer<Integer>.Default.Compare(Left.Value, Right.Value);
                                                                                                      end));

    AwordSort := [];
    for i := 0 to Length(wcounts) - 1 do
      AwordSort := AwordSort + [wcounts[i].Key];

    if foov_token <> '' then
       AwordSort := [foov_token] + AwordSort;

    sorted_voc := Enumerable<string>.Create(AwordSort);

    if Fnum_words > 0 - 1 then
    begin
        var pickWord : Integer := Fnum_words;
        if foov_token <> ''  then Inc(pickWord);

        sorted_voc := sorted_voc.Take(pickWord);
    end;

    word_index := TDictionary<string, Integer>.Create(sorted_voc.Count);
    index_word := TDictionary<Integer, string>.Create(sorted_voc.Count);
    Findex_docs:= TDictionary<Integer, Integer>.Create(Fword_docs.Count);

    for i := 0 to sorted_voc.Count-1 do
    begin
        word_index.AddOrSetValue(sorted_voc.ToList[i], i + 1);
        index_word.AddOrSetValue(i + 1, sorted_voc.ToList[i]);
    end;

    for var kv in Fword_docs do
    begin
        var idx := -1;
        if (word_index.TryGetValue(kv.Key, idx)) then
            Findex_docs.AddOrSetValue(idx, kv.Value);
    end;
end;

procedure Tokenizer.fit_on_texts(texts: TArray<TArray<string>>);
var
  w,text    : string;
  seq       : TArray<string>;
  hSet      : THashSet<string>;
  sorted_voc: Enumerable<string>;
  i,count   : Integer;
  wcounts   : TArray<TPair<string,Integer>>;
  AwordSort : TArray<string>;
begin
    for seq in texts do
    begin
        for w in seq do
        begin
            text := w;
            if Flower then  text := text.ToLower;

            count := 0;
            Fword_counts.TryGetValue(text, count);
            Fword_counts.AddOrSetValue(text, count + 1);
        end;

        hSet := THashSet<string>.Create(Fword_counts.Keys.ToArray);
        try
          for w in hSet do
          begin
              count := 0;
              Fword_docs.TryGetValue(w, count);
              Fword_docs.AddOrSetValue(w, count + 1);
          end;
        finally
          hSet.Free;
        end;
    end;

    wcounts := Fword_counts.ToArray;
    TArray.Sort<TPair<string,Integer>>(wcounts, TDelegatedComparer<TPair<string,Integer>>.Construct(function(const Left, Right: TPair<string,Integer>): Integer
                                                                                                      begin
                                                                                                        Result := -TComparer<Integer>.Default.Compare(Left.Value, Right.Value);
                                                                                                      end));
    AwordSort := [];
    for i := 0 to Length(wcounts) - 1 do
      AwordSort := AwordSort + [wcounts[i].Key];

    if foov_token <> '' then
       AwordSort := [foov_token] + AwordSort;

    sorted_voc := Enumerable<string>.Create(AwordSort);

    if Fnum_words > 0 - 1 then
    begin
        var pickWord : Integer := Fnum_words;
        if foov_token <> ''  then Inc(pickWord);

        sorted_voc := sorted_voc.Take(pickWord);
    end;

    word_index := TDictionary<string, Integer>.Create(sorted_voc.Count);
    index_word := TDictionary<Integer, string>.Create(sorted_voc.Count);
    Findex_docs:= TDictionary<Integer, Integer>.Create(Fword_docs.Count);

    for i := 0 to sorted_voc.Count-1 do
    begin
        word_index.AddOrSetValue(sorted_voc.ToList[i], i + 1);
        index_word.AddOrSetValue(i + 1, sorted_voc.ToList[i]);
    end;

    for var kv in Fword_docs do
    begin
        var idx := -1;
        if (word_index.TryGetValue(kv.Key, idx)) then
            Findex_docs.AddOrSetValue(idx, kv.Value);
    end;
end;

function Tokenizer.sequences_to_matrix(sequences: TList<TArray<integer>>; mode: string): TNDArray;
var
  word_count,i,
  seq_length,
  j,id         : Integer;
  c,tf_         : Double;
  x,idf        : TNDArray;
  seq          : TArray<integer>;
  counts       : TDictionary<integer, integer>;
begin
    if not TArray.Contains<string>( modes,mode) then
     raise Exception.Create('Unknown vectorization mode: ' +mode);

    if Fnum_words = -1 then
    begin
        if word_index <> nil then
        begin
            word_count := word_index.Count + 1;
        end else
        begin
            raise Exception.Create('Specifya dimension (''num_words'' arugment), or fit on some text data first.');
        end;
    end else
    begin
        word_count := Fnum_words;
    end;

    if (mode = 'tfidf') and (Fdocument_count = 0) then
    begin
       raise Exception.Create('Fit the Tokenizer on some text data before using the ''tfidf'' mode.');
    end;

    x := np.zeros(TFShape.Create([sequences.Count, word_count]));

    for i := 0 to  sequences.Count -1 do
    begin
        seq := sequences[i];
        if Length(seq) = 0 then
            continue;

        counts := TDictionary<integer, integer>.Create;
        try
          seq_length := Length(seq);

          for j in seq do
          begin
              if j >= word_count then
                  continue;

              var count := 0;
              counts.TryGetValue(j, count);
              counts.AddOrSetValue(j, count + 1);
          end;

          if mode = 'count' then
          begin
              for var kv in counts do
              begin
                  j         := kv.Key;
                  c         := kv.Value + 0.0;
                  x[[i, j]] := TNDArray.Create(Double(c));
              end;
          end
          else if mode = 'freq' then
          begin
              for var kv in counts do
              begin
                  j         := kv.Key;
                  c         := kv.Value + 0.0;
                  x[[i, j]] := TNDArray.Create( Double(c / seq_length));
              end;
          end
          else if mode = 'binary' then
          begin
              for var kv in counts do
              begin
                  j := kv.Key;
                  // var c = kv.Value + 0.0;
                  x[[i, j]] := TNDArray.Create( Double(1.0) );
              end;
          end
          else if mode = 'tfidf' then
          begin
              for var kv in counts do
              begin
                  j := kv.Key;
                  c := kv.Value + 0.0;
                  id := 0;
                  Findex_docs.TryGetValue(j, id);
                  tf_ := 1.0 + NDArray(np.log( TNDArray.Create(c) ));
                  idf := np.log( TNDArray.Create( Double(1.0 + Fdocument_count / (1 + id)) ) );
                  x[[i, j]] := tf_ * NDArray( idf );
              end;
          end;
        finally
          counts.Free;
        end;
    end;

    Result := x;
end;

function Tokenizer.sequences_to_texts(sequences: TList<TArray<Integer>>): TList<string>;
begin
     Result := TList<string>.Create( sequences_to_texts_generator(sequences).ToArray );
end;

function Tokenizer.sequences_to_texts_generator(sequences: TList<TArray<Integer>>): TList<string>;
var
  oov_index,i,j: Integer;
  seq          : TArray<Integer>;
  bldr         : TStringBuilder;
  word         : string;
begin
    oov_index := -1;
    word_index.TryGetValue(Foov_token, oov_index);

    Result := TList<string>.Create;

    for i := 0 to sequences.Count - 1 do
    begin
        seq :=  sequences[i];

        bldr := TStringBuilder.Create;
        try
          for j := 0 to Length(seq) -1 do
          begin
              if i > 0 then  bldr.Append(' ');

              word := '';
              if index_word.TryGetValue(seq[i], word) then
              begin
                  if (Fnum_words <> -1) and (i >= Fnum_words) then
                  begin
                      if oov_index <> -1 then
                      begin
                          bldr.Append(Foov_token);
                      end;
                  end else
                  begin
                      bldr.Append(word);
                  end;
              end
              else if oov_index <> -1 then
              begin
                  bldr.Append(Foov_token);
              end;
          end;
          Result.Add(bldr.ToString) ;
        finally
          bldr.Free;
        end;
    end;
end;

function Tokenizer.texts_to_matrix(texts: TArray<string>; mode: string): TNDArray;
begin
    Result := sequences_to_matrix( texts_to_sequences(texts), mode );
end;

function Tokenizer.texts_to_matrix(texts: TArray<TArray<string>>; mode: string): TNDArray;
begin
    Result := sequences_to_matrix( texts_to_sequences(texts), mode );
end;

function Tokenizer.texts_to_sequences(texts: TArray<string>): TList<TArray<Integer>>;
begin
     Result := texts_to_sequences_generator(texts);
end;

function Tokenizer.texts_to_sequences(texts: TArray<TArray<string>>): TList<TArray<Integer>>;
begin
    Result := texts_to_sequences_generator(texts);
end;

function Tokenizer.texts_to_sequences_generator(texts: TArray<string>): TList<TArray<Integer>>;
var
  oov_index : Integer ;
  i         : Integer;
  text      : string;
  seq       : TArray<String>;
begin
    oov_index := -1;
    word_index.TryGetValue(Foov_token, oov_index);

    Result := TList<TArray<Integer>>.Create;

    for i := 0 to Length(texts)-1 do
    begin
        text :=  texts[i];
        if flower then  text := text.ToLower;

        seq  := [];

        if Fchar_level then
        begin
            raise Exception.Create('Not Implemented "char_level = true"');
        end else
        begin
            seq := Fanalyzer(text);
        end;

       Result.Add( ConvertToSequence(oov_index, seq).ToArray );
    end;
end;

function Tokenizer.texts_to_sequences_generator(texts: TArray<TArray<string>>): TList<TArray<Integer>>;
var
  oov_index : Integer ;
  i         : Integer;
  seq       : TArray<String>;
begin
    oov_index := -1;
    word_index.TryGetValue(Foov_token, oov_index);

    Result := TList<TArray<Integer>>.Create;

    for i := 0 to Length(texts) - 1 do
    begin
        seq  := texts[i];
        Result.Add( ConvertToSequence(oov_index, seq).ToArray );
    end;
end;

end.
