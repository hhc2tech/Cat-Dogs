import os
import shutil


def split_train_dataset(path):
    """
    Search through the directory of train data and split all files into cat and train set
    :return: Cat set Dog set
    """
    cat_train = None
    dog_train = None
    for root, dirs, files in os.walk(path):
        cat_train = filter(lambda x: x[:3] == 'cat', files)
        dog_train = filter(lambda x: x[:3] == 'dog', files)
    return cat_train, dog_train


def rm_mkdir(dirname):
    """
    Create directory for
    :param dirname:
    :return:
    """
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


if __name__ == "__main__":
    cat_train, dog_train = split_train_dataset(r'G:\Kaggle\Cat-Dogs\train')
    rm_mkdir('data/train1')
    os.mkdir('data/train1/cat')
    os.mkdir('data/train1/dog')
    rm_mkdir('data/test1')
    os.symlink('data/test', 'data/test1')
    [os.symlink('data/train' + file, 'data/train1/cat' + file) for file in cat_train]
    [os.symlink('data/train' + file, 'data/train1/dog' + file) for file in dog_train]
