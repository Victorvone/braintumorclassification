from braintumorclassification.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from braintumorclassification.ml_logic.registry import load_model, save_model
from braintumorclassification.ml_logic.predict_and_explain import predict_and_gradcam
from braintumorclassification.ml_logic.load_data import load_train_data, load_test_data
from braintumorclassification.ml_logic.params import LEARNING_RATE, METRIC, BATCH_SIZE, EPOCHS, FLIP, ZOOM


def train():
    """
    Train a new model
    Save final model once it has seen all data
    Compute validation metrics
    """
    # Load train data
    train_ds, val_ds = load_train_data('braintumorclassification/raw_data')

    # Initialize model
    model = initialize_model(flip=bool(FLIP), zoom=bool(ZOOM))

    # Compile model
    model = compile_model(model, learning_rate=LEARNING_RATE, metric=METRIC)

    # Train model
    model, history = train_model(model, train_ds, val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)

    params = {'batch_size': BATCH_SIZE}
    metrics = {'accuracy': history[1]}

    # Save model
    save_model(model, params, metrics)

    return None


def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    # Load test data
    test_ds = load_test_data('braintumorclassification/raw_data')

    # Load model
    model = load_model()
    metrics = evaluate_model(model, test_ds, METRIC)

    return metrics


# def pred(image: np.array()):
#     """
#     Make a prediction using the latest trained model
#     """
#     # Load model
#     model = load_model()

#     # Predict and explain
#     prediction , grid_gradcam = predict_and_gradcam(model, image)
#     return prediction , grid_gradcam


if __name__ == "__main__":
    train()
    # pred()
    evaluate()
