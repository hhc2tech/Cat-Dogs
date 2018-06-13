from keras.layers import Input, GlobalAveragePooling2D
from keras.models import *
from keras.applications import ResNet50, resnet50,  InceptionV3, inception_v3,  Xception, xception,  VGG19, vgg19
from keras.preprocessing import image
import h5py

def export_vectors(MODEL, image_size, func=None):
    width, height = image_size
    x = Input((width, height, 3))
    if func:
        x = lambda(func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = MODEL(base_model.input, GlobalAveragePooling2D()(base_model.output))
    generator = image.ImageDataGenerator()
    train_generator = generator.flow_from_directory(r'G:\Kaggle\Cat-Dogs\data\train1', image_size, shuffle=False, batch_size=16)
    test_generator = generator.flow_from_directory(r'G:\Kaggle\Cat-Dogs\data\test1', image_size, shuffle=False, batch_size=16)
    train = model.predict_generator(train_generator, train_generator.nb_sample)
    test = model.predict_generator(test_generator, test_generator.nb_sample)
    with h5py.File("middle{}.h5".format(MODEL.func_name)) as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset('label', data=train_generator.classes)

if __name__ == "__main__":
    export_vectors(ResNet50, (224, 224))
    export_vectors(InceptionV3, (299, 299), inception_v3.preprocess_input)
    export_vectors(Xception, (299, 299), xception.preprocess_input)
    export_vectors(VGG19, vgg19.preprocess_input)
