## Computer Vision With Boosted Augmentation

This project is a practical study conducted with a real world dataset, from the
[iMaterialist](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018) competition on kaggle. My intention is to create a more well thought and sensible methodology for applying Test Time Augmentation (TTA) to an image classifier.

The intended audience is anyone looking to apply state-of-the-art methods to improve their image classifier, and in particular, core contributers / maintainers of open source ML libraries.

The data was acquired from the competition already. At ~ 30GB, it's a medium dataset (at least where images are concerned) so instead of cleaning, I will instead be focused on chunk testing and wrangling to make the most efficient pipeline for training.

My deliverables will be a series of notebooks covering -- my initial exploration of the data; model building / training / tuning /pleading; AWS modeling with fast.ai; and finally tying multiple models together with my proposed TTA CNN 'hat'.
