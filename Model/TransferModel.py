from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model
from keras.applications import ResNet50, resnet50,  InceptionV3, inception_v3,  Xception, xception,  VGG19, vgg19
from keras.preprocessing import image
import h5py
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import re
np.random.seed(2018)

epochs = 8
batch_size = 64
dropout = 0.5

def export_vectors(MODEL, image_size, func=None):
    width, height = image_size
    x = Input((width, height, 3))
    if func:
        x = func(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(fc_size, activation='relu')(x)
    #https://jizhi.im/community/discuss/2017-06-03-12-4-30-pm GlobalAveragePooling 把一个整个featuremap 转换为一个点
    # print(x.shape)
    model = Model(input=base_model.input, output=x)# Build new model except the origininal output fc layer
    generator = image.ImageDataGenerator()
    train_generator = generator.flow_from_directory(r'G:\Kaggle\Cat-Dogs\data\train1', image_size, shuffle=False, batch_size=16)
    # train_filenames = train_generator.filenames
    # train_nb_sample = len(train_filenames)
    test_generator = generator.flow_from_directory(r'G:\Kaggle\Cat-Dogs\data\test1', image_size, shuffle=False, batch_size=16)
    train = model.predict_generator(train_generator)
    test = model.predict_generator(test_generator)
    with h5py.File("augmented{}.h5".format(MODEL.__name__)) as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset('label', data=train_generator.classes)


def prepare_dataset():
    X_train = []
    X_test = []
    for model_weights in ["G:\Kaggle\Cat-Dogs\kernel\median_ResNet50.h5", "G:\Kaggle\Cat-Dogs\kernel\median_InceptionV3.h5", "G:\Kaggle\Cat-Dogs\kernel\median_VGG19.h5", "G:\Kaggle\Cat-Dogs\kernel\median_Xception.h5"]:
        with h5py.File(model_weights, 'r') as f:
            X_train.append(np.array(f['train']))
            X_test.append(np.array(f['test']))
            y_train = np.array(f['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    X_train, y_train = shuffle(X_train, y_train)
    return X_train, X_test, y_train


def build_model(X_train, y_train):
    fig_size = X_train.shape[1:]
    tensor_input = Input(shape=fig_size)
    x = Dropout(dropout)(tensor_input)
    x = Dense(1)(x)
    y = Activation('sigmoid')(x)

    model = Model(inputs=tensor_input, outputs=y)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.2)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test, verbose=1)
    y_pred = y_pred.clip(min=0.005, max=0.995)
    # df_res = pd.read_csv(r'G:\Kaggle\Cat-Dogs\sample_submission.csv')
    gen = image.ImageDataGenerator()
    test_gen = gen.flow_from_directory(r'G:\Kaggle\Cat-Dogs\data\test1', (224, 224), shuffle=False, batch_size=32, class_mode=None)
    df_res = pd.read_csv(r'G:\Kaggle\Cat-Dogs\sample_submission.csv')
    for i, filename in enumerate(test_gen.filenames):
        # print(filename)
        index = re.search(r'\\(\d+).', filename).group(1)
        # print(index)
        index = int(index)
        # index = int(filename[(filename.rfind(r'\\') + 1):filename.rfind('.')])
        # print(y_pred[i])
        df_res.set_value(index - 1, 'label', y_pred[i])
    df_res.to_csv('predicted-5.csv', index=None)
    # print(df_res.head(20))
    return df_res, y_pred

if __name__ == "__main__":
    # export_vectors(ResNet50, (224, 224))
    # export_vectors(InceptionV3, (299, 299), inception_v3.preprocess_input)
    # export_vectors(Xception, (299, 299), xception.preprocess_input)
    # export_vectors(VGG19, vgg19.preprocess_input)
    X_train, X_test, y_train = prepare_dataset()
    fitted_model = build_model(X_train, y_train)
    df_res, y_pred = predict(fitted_model, X_test)
