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



Linear Regression in `Eager` mode:

```pascal
var train_X := np.np_array<Single>               ([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]);
var train_Y := np.np_array<Single>([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]);
var n_samples := train_X.shape[0];
```

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

