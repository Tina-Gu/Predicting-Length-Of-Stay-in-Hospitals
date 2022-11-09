import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.utils import class_weight
from data_preprocessing import gendata

def get_dataset(doPCA = False): #set to True when doing backpropagation function
	train_X, test_X, train_Y, test_Y = gendata(doPCA)
	train_Y = train_Y - 1
	test_Y = test_Y - 1
	output_size = train_Y.max() - train_Y.min()+1
	return train_X, test_X, train_Y, test_Y, output_size


def mlp(train_X, test_X, train_Y, test_Y, f,s,t,a,lr):
    model_sklearn = MLPClassifier(max_iter=100000, hidden_layer_sizes=(f, s, t), activation=a,
                                  learning_rate_init=lr, )
    model_sklearn.fit(train_X, train_Y)
    # print("Training")
    # pred_X_train_sklearn = model_sklearn.predict(train_X)
    # cm = classification_report(train_Y, pred_X_train_sklearn)
    # print(cm)
    print("Test")
    pred_y_test_sklearn = model_sklearn.predict(test_X)
    cm1 = classification_report(test_Y, pred_y_test_sklearn)
    print(cm1)

    
def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

  
#Backpropagation model
def backpropagation(train_X, test_X, train_Y, test_Y):
    model = Sequential()
    #Input node 34, hidden node 68, activation function is sigmoid
    model.add(Dense(units = 68,  input_dim =34 ,  kernel_initializer =  'normal' ,  activation = 'sigmoid'))
    #Hidden node 17, output node 1, activation function is relu
    model.add(Dense(units = 17, kernel_initializer =  'normal',activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer =  'normal'))
    #Learning rate is 0.001 the optimizer is adam, the loss function is mse
    adam=optimizers.Adam(lr=0.001,  epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[soft_acc])
    model.fit(train_X, train_Y, epochs=15000, verbose=2)
    loss = model.evaluate(test_X,  test_Y,verbose=2)
    print("The mean square error is:", loss)

    
#It needs at least tensorflow(1.14) and Sklearn(0.22).
def cnn(train_X, test_X, train_Y, test_Y, output_size, class_weights):
        model = tf.keras.Sequential()
        #34input, 512hidden
        model.add(layers.Dense(512, input_shape=(34,), activation='relu'))
        model.add(layers.Dropout(0.5))
        #512input,1024 hidden
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        #32*32=1024
        model.add(tf.keras.layers.Reshape((32, 32, 1)))
        #64 kernel，64 feature map
        model.add(layers.Conv2D(64, 3, activation='relu'))
        model.add(layers.Conv2D(32, 3, activation='relu'))
        model.add(tf.keras.layers.Flatten())
        #17 output
        model.add(layers.Dense(output_size, kernel_initializer='glorot_uniform'))
        model.summary()

        lr=0.01
        epoch = 10000
        sgd = optimizers.SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss = keras.losses.SparseCategoricalCrossentropy
                      (from_logits=True), optimizer=sgd, metrics=['accuracy'])

        # here we use the test_set as the validation_set    
        model.fit(train_X,train_Y,batch_size=64,epochs=epoch,verbose=2,
                  validation_data=(test_X,test_Y))
        score = model.evaluate(test_X,test_Y,verbose=0)
        print(score)
        
        
def lstm(train_X, test_X, train_Y, test_Y):

    # reshape the dataset from 2 ndims to 3 ndims.
    train_X= np.reshape(train_X,(train_X.shape[0], 1, train_X.shape[1]))
    test_X= np.reshape(test_X,(test_X.shape[0], 1, test_X.shape[1]))

    # build bidirectional RNN model.
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"), input_shape=(1, 34,)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.1), input_shape=(1, 34)))
    model.add(layers.Dense(units = 18))
    model.summary()

    #start training the model.
    model.compile( 
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"])
    model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=30, batch_size = 50, verbose=0)
    vectors = model.predict(test_X) #get the output of centroids.
    print("Finish RNN training ...")
    #reshape the output back to 2 ndims.
    vectors= np.reshape(vectors,(vectors.shape[0], vectors.shape[2]))
    # build MLP classifier
    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, solver='adam', activation='logistic', hidden_layer_sizes=(32, 36, 17), batch_size=100)).fit(vectors, test_Y)
    pred = clf.predict(vectors)

    #print the results of model.
    print("")
    print("===================RNN and MLP classifier==================")
    print("Accuracy of mlp", accuracy_score(pred, test_Y))
    print("MSE", mean_squared_error(pred, test_Y))
    print("Confusion Matrix: ")
    print(confusion_matrix(pred, test_Y))
    print("======================== END ===========================")
            
      
def main():
        train_X, test_X, train_Y, test_Y, output_size = get_dataset()
        class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(train_Y),
                                                    train_Y)
        #baseline model
        mlp(train_X, test_X, train_Y, test_Y, 128, 64, 32, 'relu', 0.01)
        backpropagation(Train_X, Test_X, Train_Y, Test_Y) #with PCA
        cnn(train_X, test_X, train_Y, test_Y, output_size, class_weights)
        lstm(Train_X, Test_X, Train_Y, Test_Y)
        
        
main()



