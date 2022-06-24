import tensorflow as tf


def print_num_gpus_available() -> None:
    """
    Prints the number of GPUs available on the current machine.
    :return: None
    """
    print("Numero de GPU disponibles: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if tf.test.gpu_device_name():
        print('Dispositivo de GPU por defecto:', tf.test.gpu_device_name())
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     print("Name:", gpu.name, "  Type:", gpu.device_type)
    else:
        print("Deberia instalar una versión de Tensorflow para su GPU")
