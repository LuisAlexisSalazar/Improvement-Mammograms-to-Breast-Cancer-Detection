import os

from src.settings.config import ALL_MODE_DOWNLOAD, PATH_DATA_RAW, PATH_DATA_PREPROCESSING


def create_folder_if_not_exist(path_folder):
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)


def create_all_folder(folder_to_classification, list_sub_folders):
    create_folder_if_not_exist(folder_to_classification)
    for sub_folder in list_sub_folders:
        create_folder_if_not_exist(sub_folder)

    there_are_not_data = []
    for sub_folder in list_sub_folders:
        there_are_not_data.append(len(os.listdir(sub_folder)) == 0)
    # ? operation AND en list
    return all(there_are_not_data)


def is_mode_valid(mode):
    if mode in ALL_MODE_DOWNLOAD:
        return True
    return False


def get_path_folder_to_read(mode, dataset):
    if is_mode_valid(mode):
        return PATH_DATA_RAW + dataset + "/" + mode + "/"
    else:
        return None


def get_path_folder_to_result_preprocessing(mode, dataset):
    if is_mode_valid(mode):
        return PATH_DATA_PREPROCESSING + dataset + "/" + mode + "/"
    else:
        return None


def get_paths_sub_folders(path):
    list_sub_folders = os.listdir(path)
    list_path_subfolder = []
    for subfolder in list_sub_folders:
        list_path_subfolder.append(path + subfolder + "/")
    return list_path_subfolder


# ["BinaryNM", "BinaryBM", "BinaryBN", "Binary(BM)N", "ClassBMN"]
def create_folders_to_mode(path, mode):
    if mode == "BinaryNM":
        create_folder_if_not_exist(path + "normal/")
        create_folder_if_not_exist(path + "benigno/")
    elif mode == "BinaryBM":
        create_folder_if_not_exist(path + "benigno/")
        create_folder_if_not_exist(path + "maligno/")
    elif mode == "BinaryBN":
        create_folder_if_not_exist(path + "benigno/")
        create_folder_if_not_exist(path + "normal/")
    elif mode == "Binary(BM)N":
        create_folder_if_not_exist(path + "tumoral/")
        create_folder_if_not_exist(path + "normal/")
    elif mode == "ClassBMN":
        create_folder_if_not_exist(path + "benigno/")
        create_folder_if_not_exist(path + "normal/")
        create_folder_if_not_exist(path + "maligno/")
