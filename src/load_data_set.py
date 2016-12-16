from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

print len(X_train), len(X_test)