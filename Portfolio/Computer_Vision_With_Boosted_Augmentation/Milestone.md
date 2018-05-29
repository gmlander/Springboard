## Computer Vision With Boosted Augmentation - Milestone Report

This project is a practical study conducted with a real world dataset, from the
[iMaterialist](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018) competition on kaggle. My intention is to create a more well thought and sensible methodology for applying Test Time Augmentation (TTA) to an image classifier.

The intended audience is anyone looking to apply state-of-the-art methods to improve their image classifier, and in particular, core contributers / maintainers of open source ML libraries.

The data was acquired from the competition already. At ~ 30GB, it's a medium dataset (at least where images are concerned). While examining the data I found that class distribution in the training data is very uneven, whereas the test data is fairly uniform. So I added label wts to my model fits. I also discovered most of the images are around 500x500. Ideally I would train the classifier up near that resolution, but I don't have the hardware to handle that growth in params, and have limited myself to 224x224.

Other potential data sets to use for this project could of course be ImageNet or CIFAR. I also think this augmentaion technique could be extensible to NLP by transforming n-grams, but that's a long way off.

Initial findings have been making improvements to the speed of data input to the model through testing different file formats and building a custom generator class to handle pooling / multiprocessing. I also discovered that Keras' default predict method is incredibly slow and that I will have to freeze the graph any time I perform inference.

My Xception model has made some progress, but is seriously plateaued around 80% accuracy.

While the notebook I've worked in is hosted on github, it is much more easily browsed from Google Colab where they were built (and where dependent files are hosted).

[Part 1](https://drive.google.com/file/d/1gXWExvvcdyf9EC4YZ3S-xZBQbOBEMqWB/view?usp=sharing)
