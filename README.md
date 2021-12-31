# Applied Machine learning course

My work for Applied ML course during WS 2021/2022 on University of Economics, Prague. Comments/description in the notebooks are mostly in Czech.

We started with ML basics - EDA, feature engineering, classification problems and regression problems. Then we moved to text processing, sentiment analysis, recommendation engines, image processing (style transfer, augmentation, face detection, image generation), deep learning (CNN, RNN, GAN), timeseries forecasting, (tensorflow) model serving, text generation & reinforcement learning.

We mostly worked with:
* pandas
* numpy
* scikit-learn
* matplotlib
* tensorflow/keras

*Note: all notebooks would need a significant dose of refactoring*

<hr>

## Content

04_titanic_my.ipynb
* evergreen task - do EDA, feature engineering, apply few models, tune the models
* binary classification

05_spam.ipynb
* classify text as a spam/ham
* text processing - embeddings, TF-IDF
* binary classification

06_movie_recommendation_my.ipynb
* content filtering movie recommender 
* KDTree, Cosine similarity

07_imdb_my.ipynb
* deep learning sentiment analysis

07_mnist_my.ipynb
* simple deep learning multiclass classification with data augmentation

08_fashion_mnist_my.ipynb
* deep learning multiclass classification

08_misc_predict_my.ipynb
* image object detection using Keras

08_mnist_my.ipynb
* deep learning multiclass classification with data augmentation

09_timeseries_my.ipynb
* weather timeseries forecasting using RNN

09_timeseries_my_colab.ipynb
* weather timeseries forecasting using RNN - Google Colab version

10_faces_my.ipynb
* face detection with DeepFace and MTCNN

10_functional_my.ipynb
* weather timeseries forecasting using RNN
* Keras functional API used
* multi-output model

10_serving.ipynb
* testing sample tensorflow serving

11_shakespeare_my.ipynb
* next word (character) prediction using RNN

11_shakespeare_my_colab.ipynb
* next word (character) prediction using RNN - Google Colab version

11_style_transfer_my.ipynb
* simple image style transfer using Tensorflow pretrained model

11_vae_my.ipynb
* "new" image generation using variational autoencoder