# imports
import pandas as pd 
import tensorflow as tf 


def tf_pseudo_sample_fit(model_method, X_train, y_train, X_test, num_epochs=50, log=True):
    ''' method to pseudo label test data and retrain network with combined dataset '''
    # get the model
    model = model_method()
    if log:
        print("got model summary \n{}".format(model.summary()))

    # convert to pandas for fitting
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # train the model
    model.fit(X_train_np, y_train_np, epochs=num_epochs)

    # predict on test data
    y_pred = model.predict(X_test)
    y_pred = tf.squeeze(y_pred)

    # combine train and test datasets
    X_combined = pd.concat([X_train, X_test], axis=0).to_numpy()
    y_combined = pd.concat([y_train, pd.Series(y_pred, name='target')], axis=0).to_numpy()

    # retrain the model
    model = model_method()
    model.fit(X_combined, y_combined, epochs=num_epochs)

    # return the model
    return model

