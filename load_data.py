import os
import shutil


def split_train_dataset(path):
    """
    Search through the directory of train data and split all files into cat and train set
    :return: Cat set Dog set
    """
    files = os.listdir(path)
    cat_train = filter(lambda x: x[:3] == "cat", files)
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
    cat_train, dog_train = split_train_dataset(r'G:\Kaggle\Cat-Dogs\data\train')
    rm_mkdir('G:\Kaggle\Cat-Dogs\data/train1')
    os.mkdir('G:\Kaggle\Cat-Dogs\data/train1/cat')
    os.mkdir('G:\Kaggle\Cat-Dogs\data/train1/dog')
    rm_mkdir('G:\Kaggle\Cat-Dogs\data/test1')
    os.symlink('G:/Kaggle/Cat-Dogs/data/test', 'G:/Kaggle/Cat-Dogs/data/test1/test')
    [os.symlink('G:\Kaggle/Cat-Dogs/data/train/' + file, 'G:/Kaggle/Cat-Dogs/data/train1/cat/' + file) for file in cat_train]
    [os.symlink('G:/Kaggle/Cat-Dogs/data/train/' + file, 'G:/Kaggle/Cat-Dogs/data/train1/dog/' + file) for file in dog_train]
    # IF OSError: symbolic link privilege not held occurred, use windows ipython under the previlidge of administer
    # https: // github.com / explosion / spaCy / issues / 895
