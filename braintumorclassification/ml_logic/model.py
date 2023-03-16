from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras import layers, Sequential, optimizers, callbacks, Model


def initialize_model(zoom: bool,
                     flip: bool):
    """
    Initialize the Neural Network with pretrained weights on EfficientNetV2B3
    """
    base_model = EfficientNetV2B3(include_top=False,
                                  weights='imagenet',
                                  input_shape=(255, 255, 3),
                                  pooling='max',
                                  include_preprocessing=True)

    model = Sequential()
    if zoom is True:
        model.add(layers.RandomFlip(mode="vertical"))
    if flip is True:
        model. add(layers.RandomZoom(0.1))
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(4, activation='softmax'))

    base_model.trainable = False
    print("\n✅ model initialized")
    return model


def compile_model(model: Model, learning_rate: float, metric):
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=metric)

    print("\n✅ model compiled")
    return model


def train_model(model: Model,
                train_ds,
                val_ds,
                batch_size: int,
                epochs: int):
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    print("Train model...")

    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es],
                        verbose=1)

    print("\n✅ model trained on a shitload of data")
    return model, history


def evaluate_model(model: Model,
                   test_ds,
                   metric):
    """
    Evaluate trained model performance on dataset
    """
    metrics = model.evaluate(test_ds)
    loss = metrics[0]
    metric_result = metrics[1]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} {metric} {round(metric_result, 2)}")

    return metrics
