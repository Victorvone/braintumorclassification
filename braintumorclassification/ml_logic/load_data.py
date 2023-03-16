from tensorflow.keras.utils import image_dataset_from_directory


def load_train_data(directory):
    """
    loads train data from a local directory
    """
    train_ds, val_ds = image_dataset_from_directory(
        f"{directory}/Training",
        labels="inferred",
        label_mode="categorical",
        seed=123,
        image_size=(255, 255),
        validation_split=0.2,
        subset="both",
        color_mode="rgb",
        batch_size=64,
    )
    return train_ds, val_ds


def load_test_data(directory):
    """
    loads test data from a local directory
    """

    test_ds = image_dataset_from_directory(
        f"{directory}/Testing",
        labels="inferred",
        label_mode="categorical",
        seed=123,
        image_size=(255, 255),
        color_mode="rgb",
        batch_size=64,
    )

    return test_ds
