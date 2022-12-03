import cv2


def read_image_separate_MIAS(df_dataset_MIAS,
                             list_path_folder,
                             path_to__classification,
                             mode_classification):
    folder_base_images = "/MINI-RAW"
    # print(path_to__classification)
    # df_dataset_MIAS = folder_base_images + folder_base_images
    if mode_classification == "BinaryNM":
        # df_dataset_MIAS = df_dataset_MIAS.dropna(subset=['abn_class'])
        df_dataset_MIAS = df_dataset_MIAS.drop(df_dataset_MIAS[df_dataset_MIAS['abn_class'] == "B"].index)
        # Reemplazar Estado vacio a etiqueta N
        df_dataset_MIAS["abn_class"].fillna(value='N', inplace=True)

        df_dataset_MIAS = df_dataset_MIAS[["name_file", "abn_class"]]
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_to__classification + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            if row['abn_class'] == 'N':
                cv2.imwrite(list_path_folder[0] + "/" + row['name_file'] + ".jpg", img)

            else:  # M
                cv2.imwrite(list_path_folder[1] + "/" + row['name_file'] + ".jpg", img)
    elif mode_classification == "BinaryBM":
        # ?Eliminar el estado vacio porque son los N;normales
        df_dataset_MIAS = df_dataset_MIAS.dropna(subset=['abn_class'])
        df_dataset_MIAS = df_dataset_MIAS[["name_file", "abn_class"]]
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_to__classification + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            if row['abn_class'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['name_file'] + ".jpg", img)

            else:  # M
                cv2.imwrite(list_path_folder[1] + "/" + row['name_file'] + ".jpg", img)
    elif mode_classification == "BinaryBN":
        df_dataset_MIAS["abn_class"].fillna(value='N', inplace=True)
        df_dataset_MIAS.drop(df_dataset_MIAS[df_dataset_MIAS.abn_class == "M"].index, inplace=True)
        df_dataset_MIAS = df_dataset_MIAS[["name_file", "abn_class"]]

        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_to__classification + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            if row['abn_class'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['name_file'] + ".jpg", img)

            else:  # N
                cv2.imwrite(list_path_folder[1] + "/" + row['name_file'] + ".jpg", img)
    elif mode_classification == "Binary(BM)N":
        df_dataset_MIAS["abn_class"].fillna(value='N', inplace=True)
        df_dataset_MIAS['abn_class'].mask(df_dataset_MIAS['abn_class'].isin(['B', 'M']), "T", inplace=True)
        df_dataset_MIAS = df_dataset_MIAS[["name_file", "abn_class"]]
        # print(df_dataset_MIAS["abn_class"].unique())

        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_to__classification + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            if row['abn_class'] == 'T':
                cv2.imwrite(list_path_folder[0] + "/" + row['name_file'] + ".jpg", img)

            else:  # N
                cv2.imwrite(list_path_folder[1] + "/" + row['name_file'] + ".jpg", img)
    elif mode_classification == "ClassBMN":
        df_dataset_MIAS["abn_class"].fillna(value='N', inplace=True)
        df_dataset_MIAS = df_dataset_MIAS[["name_file", "abn_class"]]

        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_to__classification + row['name_file'] + ".pgm", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            if row['abn_class'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['name_file'] + ".jpg", img)
            elif row['abn_class'] == 'M':
                cv2.imwrite(list_path_folder[1] + "/" + row['name_file'] + ".jpg", img)
            else:  # N
                cv2.imwrite(list_path_folder[2] + "/" + row['name_file'] + ".jpg", img)


# toDo: Para pf2 hacer completo el preprocesamiento
def read_image_separate_MINI_DDSM(df_dataset_MINI_DDSM,
                                  list_path_folder,
                                  path_to__classification,
                                  mode_classification):
    if mode_classification == "BinaryNM":
        df_dataset_MINI_DDSM.drop(df_dataset_MINI_DDSM.loc[df_dataset_MINI_DDSM['Status'] == 'Benign'].index,
                                  inplace=True)

        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Normal'], 'N')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Cancer'], 'M')

        df_dataset_MIAS = df_dataset_MINI_DDSM[["fullPath", "Status", "fileName"]]
        folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
        path_folder_to_read_images = path_to__classification + folder_base_images
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            if row['Status'] == 'N':
                cv2.imwrite(list_path_folder[0] + "/" + row['fileName'], img)

            else:  # M
                cv2.imwrite(list_path_folder[1] + "/" + row['fileName'], img)

    elif mode_classification == "BinaryBM":
        df_dataset_MINI_DDSM.drop(df_dataset_MINI_DDSM.loc[df_dataset_MINI_DDSM['Status'] == 'Normal'].index,
                                  inplace=True)

        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Benign'], 'B')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Cancer'], 'M')

        df_dataset_MIAS = df_dataset_MINI_DDSM[["fullPath", "Status", "fileName"]]
        folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
        path_folder_to_read_images = path_to__classification + folder_base_images
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            if row['Status'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['fileName'], img)
            else:  # M
                cv2.imwrite(list_path_folder[1] + "/" + row['fileName'], img)
    elif mode_classification == "BinaryBN":
        df_dataset_MINI_DDSM.drop(df_dataset_MINI_DDSM.loc[df_dataset_MINI_DDSM['Status'] == 'Cancer'].index,
                                  inplace=True)

        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Benign'], 'B')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Normal'], 'N')

        df_dataset_MIAS = df_dataset_MINI_DDSM[["fullPath", "Status", "fileName"]]
        folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
        path_folder_to_read_images = path_to__classification + folder_base_images
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            if row['Status'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['fileName'], img)

            else:  # M
                cv2.imwrite(list_path_folder[1] + "/" + row['fileName'], img)

    elif mode_classification == "Binary(BM)N":

        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Benign'], 'T')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Cancer'], 'T')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Normal'], 'N')

        df_dataset_MIAS = df_dataset_MINI_DDSM[["fullPath", "Status", "fileName"]]
        folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
        path_folder_to_read_images = path_to__classification + folder_base_images
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            if row['Status'] == 'T':
                cv2.imwrite(list_path_folder[0] + "/" + row['fileName'], img)

            else:  # N
                cv2.imwrite(list_path_folder[1] + "/" + row['fileName'], img)

    elif mode_classification == "ClassBMN":
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Benign'], 'B')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Cancer'], 'M')
        df_dataset_MINI_DDSM["Status"] = df_dataset_MINI_DDSM['Status'].replace(['Normal'], 'N')

        df_dataset_MIAS = df_dataset_MINI_DDSM[["fullPath", "Status", "fileName"]]
        folder_base_images = "MINI-DDSM-Complete-JPEG-8/"
        path_folder_to_read_images = path_to__classification + folder_base_images
        for index, row in df_dataset_MIAS.iterrows():
            img = cv2.imread(path_folder_to_read_images + row['fullPath'], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            if row['Status'] == 'B':
                cv2.imwrite(list_path_folder[0] + "/" + row['fileName'], img)

            elif row['Status'] == 'M':
                cv2.imwrite(list_path_folder[1] + "/" + row['fileName'], img)

            else:  # N
                cv2.imwrite(list_path_folder[2] + "/" + row['fileName'], img)
