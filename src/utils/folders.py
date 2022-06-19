import os


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
