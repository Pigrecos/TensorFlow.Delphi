**TensorFlow.Delphi** (TF.Delphi) provides a Delphi(Pascal)Standard binding for [TensorFlow](https://www.tensorflow.org/). It aims to implement the complete Tensorflow API in Delphi which allows Pascal developers to develop, train and deploy Machine Learning models with the Pascal Delphi(porting to free pascal in the future).

*master branch is based on tensorflow v2.11.* ***No python engine or installation required*.**



#### Note: This is a work-in-progress. please treat it as such.Pull request are welcome

### Why TensorFlow in Delphi-Pascal ?

The *intent* is to bring popular data science technology into the Delphi world and to provide .Delphi/Pascal developers with a powerful Machine Learning tool set without reinventing the wheel. Since the APIs are kept as similar as possible you can immediately adapt any existing TensorFlow code in Delphi with a zero learning curve. Take a look at a comparison picture and see how comfortably a TensorFlow/Python script translates into a Delphi program with TensorFlow.Delphi.

![csharp vs pacal](https://github.com/Pigrecos/TensorFlow.Delphi/blob/main/src/lib/img/Gather.png)

![](https://github.com/Pigrecos/TensorFlow.Delphi/blob/main/src/lib/img/Slice.png)

philosophy allows a large number of machine learning code written in Python to be quickly migrated to Delphi, enabling Delphi/Pascal developers to use cutting edge machine learning models and access a vast number of TensorFlow resources which would not be possible without this project.

Tensorflow.Delphi also implements TensorFlow's high level API where all the magic happens. This computation graph building layer is still under active development. Once it is completely implemented you can build new Machine Learning models in Delphi-Pascal.

As a reference code I used 

[TesorFlow.NET](https://github.com/SciSharp/TensorFlow.NET)

[TensorFlow Python](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)

#### Sample code



Linear Regression in `Graph` mode:

```pascal
function LinearRegression.Run(mmo1: TMemo): Boolean;
begin
    tf.compat.v1.disable_eager_execution;

    PrepareData;

    // tf Graph Input
    var X : TTensor := tf.placeholder(tf.float32_t);
    var Y : TTensor := tf.placeholder(tf.float32_t);

    // Set model weights
    // We can set a fixed init value in order to debug
    // var rnd1 = rng.randn<float>();
    // var rnd2 = rng.randn<float>();
    var W  := tf.Variable(Single(-0.06), 'weight');
    var b  := tf.Variable(Single(-0.73), 'bias');

    // Construct a linear model
    var pred : TTensor := tf.add(tf.multiply(X, W), b);
    //var pred1 := (X * W) + b;  OK

    // Mean squared error
    var cost := TTensor(tf.reduce_sum(tf.pow(pred - Y, 2.0)))  / (2.0 * n_samples) ;

    // Gradient descent
    // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    var optimizer := tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

    // Initialize the variables (i.e. assign their default value)
    var init := tf.global_variables_initializer;

    // Start training
    var sess := tf.Session;
    // Run the initializer
    sess.run(init);

    // Fit all training data
    var epoch: Integer ;
    for epoch := 0 to training_epochs -1 do
    begin
        for var zItem in TUtils.zip<Single>(train_X, train_Y) do
        begin
            var v_x : Single := zItem.Value1 ;
            var v_y : Single := zItem.Value2 ;
            sess.run(optimizer, [ Tuple<TValue,TValue>.Create(X, v_x), Tuple<TValue,TValue>.Create(Y, v_y) ]);
        end;

        // Display logs per epoch step
        if ((epoch + 1) mod display_step) = 0 then
        begin
            var fc : Single := NDArray(sess.run(cost, [ Tuple<TValue,TValue>.Create(X, train_X), Tuple<TValue,TValue>.Create(Y, train_Y) ]));
            var fW : Single := NDArray(sess.run( TResourceVariable(W) ));
            var fb : Single := NDArray(sess.run( TResourceVariable(b) ));
            mmo1.Lines.Add( Format('Epoch: %d cost=%.9f + "W=%.9f b=%.9f"',[epoch + 1,fc, fW,fb]) );
        end;
    end;

    mmo1.Lines.Add('Optimization Finished!');
    var training_cost : Single := NDArray(sess.run(cost, [ Tuple<TValue,TValue>.Create(X, train_X), Tuple<TValue,TValue>.Create(Y, train_Y) ]));
    var fW            : Single := NDArray(sess.run(  TResourceVariable(W) ));
    var fb            : Single := NDArray(sess.run(  TResourceVariable(b) ));

    mmo1.Lines.Add('');
    mmo1.Lines.Add(Format('Training cost=%.9f W=%.9f b=%.9f',[training_cost, fW, fb]));

    // Testing example
    var test_X : NDArray := np.np_array( TArray<Single>.Create(6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1) );
    var test_Y : NDArray := np.np_array( TArray<Single>.Create(1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03) );

    mmo1.Lines.Add('Testing... (Mean square loss Comparison)');

    var t_cost                 := TTensor(tf.reduce_sum(tf.pow(pred - Y, 2.0)))  / (2.0 * test_X.shape[0]) ;
    var testing_cost : Single  := NDArray(sess.run(t_cost, [ Tuple<TValue,TValue>.Create(X, test_X), Tuple<TValue,TValue>.Create(Y, test_Y) ]));

    mmo1.Lines.Add('');
    mmo1.Lines.Add( Format('Testing cost=%.9f',[testing_cost]) );
    var diff := Abs( training_cost - testing_cost);
    mmo1.Lines.Add( Format('Absolute mean square loss difference: %.9f',[diff]) );
    mmo1.Lines.Add('');

    Result := diff < 0.01;

end;
```

#### Compiling:

Compatible with all delphi versions that support inline variable(10.3 and higher).

To compile you need some external libraries:

-  [Spring4d](https://bitbucket.org/sglienke/spring4d/src/master/) 
-  [Neon](https://github.com/paolo-rossi/delphi-neon) 
-  [TurboPack Abbrevia](https://github.com/TurboPack/Abbrevia) 
-  [Download libTensorflow](https://www.tensorflow.org/install/lang_c?hl=en) and rename to 'tensorflow-cpu-win64-2.11.0.dll'

### Contribute:

Feel like contributing to one of the hottest projects in the Machine Learning field? Want to know how Tensorflow magically creates the computational graph? We appreciate every contribution however small. There are tasks for novices to experts alike, if everyone tackles only a small task the sum of contributions will be huge.

You can:
* Let everyone know about this project
* Port Tensorflow unit tests from Python to Delphi or Pascal
* Port missing Tensorflow code from Python to  Delphi or Pascal
* Port Tensorflow examples to Delphi or Pascal and raise issues if you come accross missing parts of the API
* Debug one of the unit tests that is marked as Ignored to get it to work
* Debug one of the not yet working examples and get it to work

### Todo List:

- Add functionality
- Testing
- Memory Leak
- and More..... 

### Contact

Follow us on [Twitter](https://twitter.com/Marte0016)

