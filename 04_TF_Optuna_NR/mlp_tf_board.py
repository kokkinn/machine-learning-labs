# Imports
import datetime

import optuna
import tensorflow
from keras import activations
from keras.layers import Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

activation_functions = ('relu', 'sigmoid', 'softmax', 'tanh', 'selu', 'exponential')


def create_model(activation_function, dropout: float, optimizer: str):
    feature_vector_length = 784
    num_classes = 10
    input_shape = (feature_vector_length,)

    model = Sequential()
    model.add(Dense(350, input_shape=input_shape, activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))  # last layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def train_model(model, batch_size, learning_rate, num_epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=learning_rate,
              callbacks=[tensorboard_callback])
    return None


# Configuration options
feature_vector_length = 784
num_classes = 10

# Load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape the data - MLPs do not understand such things as '2D'.
# Reshape to 28 x 28 pixels = 784 features
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

# Convert into greyscale
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert target classes to categorical ones
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Create the model
# params = {'activation_function': 'relu', 'dropout': 0.2, 'optimizer': 'Adam'}
# model = create_model(**params)
# train_model(model)
#
# # Test the model after training
# test_results = model.evaluate(X_test, Y_test, verbose=1, callbacks=[tensorboard_callback])
# print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
# print(test_results)
#
for _ in range(50):
    model = create_model(
        optimizer='RMSprop',
        activation_function='relu',
        dropout=0.2
    )
    train_model(
        model=model,
        learning_rate=0.1,
        batch_size=250,
        num_epochs=10
    )
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(test_results[1])


def objective(trial):
    params = {
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        # 'num_epochs': trial.suggest_int('num_epochs', 3, 15),
        # 'batch_size': trial.suggest_int('batch_size', 20, 300),
        # 'activation_function': trial.suggest_categorical('activation_function', activation_functions),
        # 'dropout': trial.suggest_float('dropout', 0.0, 0.4)

    }

    # model = build_model(params)
    model = create_model(
        optimizer=params['optimizer'],
        # optimizer='Adam',
        activation_function=params['activation_function'],
        # activation_function='relu',
        dropout=params['dropout']
        # dropout=0.1
    )
    train_model(
        model=model,
        # learning_rate=params['learning_rate'],
        learning_rate=0.2,
        # batch_size=params['batch_size'],
        batch_size=250,
        # num_epochs=params['num_epochs']
        num_epochs=5
    )

    # accuracy = train_and_evaluate(params, model)
    test_results = model.evaluate(X_test, Y_test, verbose=0)
    accuracy = test_results[1]
    return accuracy

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
#
# best_trial = study.best_params
# print(best_trial)
# for key, value in best_trial.params.items():
#     print("{}: {}".format(key, value))
