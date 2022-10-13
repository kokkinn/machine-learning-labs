# Imports
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow
from keras import activations
from keras.layers import Dropout
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

activation_functions = ('relu', 'sigmoid', 'softmax', 'tanh', 'selu', 'exponential')


def numbers_comparison_image(dataset, pca_num):
    pca_image = PCA(pca_num)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].matshow(dataset[9].reshape(28, 28))
    axarr[0, 1].matshow(dataset[10].reshape(28, 28))
    dataset = pca_image.fit_transform(dataset)
    axarr[1, 0].matshow(pca_image.inverse_transform(dataset[9]).reshape(28, 28) * 255)
    axarr[1, 1].matshow(pca_image.inverse_transform(dataset[10]).reshape(28, 28) * 255)
    plt.show()


def create_model(activation_function, dropout: float, optimizer: str):
    feature_vector_length = 400
    num_classes = 10
    input_shape = (feature_vector_length,)
    model = Sequential()
    model.add(Dense(200, input_shape=input_shape, activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Dense(30, activation=activation_function))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))  # last layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def train_model(model_to_train, batch_size, learning_rate, num_epochs, train_images_trfu, train_labels_trfu):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model_to_train.fit(train_images_trfu, train_labels_trfu, epochs=num_epochs, batch_size=batch_size, verbose=0,
                                 validation_split=learning_rate,
                                 callbacks=[tensorboard_callback]
                                 )
    return history


def test_model(model_to_test, test_images_tefu, test_labels_tefu):
    test_results = model_to_test.evaluate(test_images_tefu, test_labels_tefu, verbose=0)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    return test_results


# Configuration options
num_classes = 10

# Load the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert target classes to categorical ones
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Reshape an array of 2d arrays 28x28 to an array of vectors 1x784
train_images = train_images.reshape(-1, 784)
test_images = test_images.reshape(-1, 784)

# # Normalize
# scaler = StandardScaler()
# train_images = scaler.fit_transform(train_images)
# test_images = scaler.fit_transform(test_images)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Use PCA to reduce dimension of input data
N_COMPONENTS = 400
pca = PCA(n_components=N_COMPONENTS)
pca.fit(train_images)
train_images_pca = pca.transform(train_images)
test_images_pca = pca.transform(test_images)

# # show comparison images grid
# numbers_comparison_image(test_images, pca_num=100)

# # return to initial shape
# test_images_inv = pca.inverse_transform(test_images)
# test_images_inv = scaler.inverse_transform(test_images_inv)

# create a dataset with accuracies
df_fineval = pd.DataFrame(columns=['Num', 'Accuracy', 'Elapsed_time'], dtype=object)

for i in range(1, 31):
    # Create the model
    model = create_model(activation_function='relu', dropout=0.179, optimizer='RMSprop')

    # train
    start_time = time.time()
    history = train_model(model_to_train=model,
                          learning_rate=0.1,
                          batch_size=250,
                          num_epochs=10,
                          train_images_trfu=train_images_pca,
                          train_labels_trfu=train_labels)

    # test
    eval_results = test_model(model_to_test=model, test_images_tefu=test_images_pca, test_labels_tefu=test_labels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    df_fineval.loc[len(df_fineval)] = [int(i), eval_results[1], elapsed_time]
    df_fineval['Num'] = df_fineval['Num'].apply(np.int64)
    print(i)
df_fineval.to_csv('accuracy_after_pca_400.csv', index=False)

# # analyze
df_fineval_before_pca = pd.read_csv('accuracy_after_pca_400.csv')
print(df_fineval_before_pca[:].mean())

# plt.plot(history.history['accuracy'], label='Train')
# plt.plot(history.history['val_accuracy'], label='Validation')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.title('Model Accuracy - Before PCA', pad=15)
# plt.legend(loc='lower right')
# plt.show()


# for _ in range(50):
#     model = create_model(
#         optimizer='RMSprop',
#         activation_function='relu',
#         dropout=0.2
#     )
#     train_model(
#         model=model,
#         learning_rate=0.1,
#         batch_size=250,
#         num_epochs=10
#     )
#     test_results = model.evaluate(test_images_pca, test_labels, verbose=1)
#     print(test_results[1])


# def objective(trial):
#     params = {
#         # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
#         # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
#         # 'num_epochs': trial.suggest_int('num_epochs', 3, 15),
#         # 'batch_size': trial.suggest_int('batch_size', 20, 300),
#         # 'activation_function': trial.suggest_categorical('activation_function', activation_functions),
#         # 'dropout': trial.suggest_float('dropout', 0.0, 0.4)
#
#     }
#
#     # model = build_model(params)
#     model = create_model(
#         optimizer=params['optimizer'],
#         # optimizer='Adam',
#         activation_function=params['activation_function'],
#         # activation_function='relu',
#         dropout=params['dropout']
#         # dropout=0.1
#     )
#     train_model(
#         model=model,
#         # learning_rate=params['learning_rate'],
#         learning_rate=0.2,
#         # batch_size=params['batch_size'],
#         batch_size=250,
#         # num_epochs=params['num_epochs']
#         num_epochs=5
#     )
#
#     # accuracy = train_and_evaluate(params, model)
#     test_results = model.evaluate(test_images, test_labels, verbose=0)
#     accuracy = test_results[1]
#     return accuracy

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
#
# best_trial = study.best_params
# print(best_trial)
# for key, value in best_trial.params.items():
#     print("{}: {}".format(key, value))
