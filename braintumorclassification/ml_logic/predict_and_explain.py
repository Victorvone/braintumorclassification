from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.vanilla_gradients import VanillaGradients
import numpy as np
from tf_explain.core.activations import ExtractActivations


def predict_and_explain(model,
                        image,
                        layers_name=None,
                        act_viz=True,
                        grad_cam=True,
                        occlusion_sensitivity=True,
                        vanilla_gradient=True):

    '''Returns prediction of tumor classification, Activation visualizations, Grad-CAM, Occlusion sensitivity
    and Vanilla gradient. It also saves pictures in ../visualizations folder. Image needs to follow the shape: none, 255, 255, 3'''

    # Prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    # Format prediction to required format for explanations
    def format_for_expl(image, class_index):
        X = image
        y = np.expand_dims(np.array([class_index]), axis=0)
        return (X, y)

    image_tuple = format_for_expl(image, class_index)

    # Activation visualizations
    if act_viz is True and layers_name is not None:
        actviz = ExtractActivations()
        grid_actviz = actviz.explain(image_tuple, model, layers_name=layers_name)
        actviz.save(grid_actviz, "../Visualizations/", "ActViz.png")
    else:
        grid_actviz = None

    # Grad-CAM
    if grad_cam is True:
        gradcam = GradCAM()
        grid_gradcam = gradcam.explain(image_tuple, model, class_index=class_index)
        # maybe change class_index to variable +
        gradcam.save(grid_gradcam, "../Visualizations/", "GradCam.png")
    else:
        grid_gradcam = None

    # Occlusion sensitivity
    if occlusion_sensitivity is True:
        occsens = OcclusionSensitivity()
        grid_occsens = occsens.explain(image_tuple, model, class_index=class_index, patch_size=8)
        occsens.save(grid_occsens, "../Visualizations/", "OcclusionSensitivity.png")
    else:
        grid_occsens = None

    # Vanilla gradient
    if vanilla_gradient is True:
        Vanillagrad = VanillaGradients()
        grid_vanillagrad = Vanillagrad.explain(image_tuple, model, class_index=class_index)
        Vanillagrad.save(grid_vanillagrad, "../Visualizations/", "VanillaGradient.png")
    else:
        grid_vanillagrad = None

    return prediction, grid_actviz, grid_gradcam, grid_occsens, grid_vanillagrad


def predict_and_gradcam(model,
                        image,
                        layers_name=None):

    '''Returns prediction of tumor classification, Activation visualizations, Grad-CAM, Occlusion sensitivity
    and Vanilla gradient. It also saves pictures in ../visualizations folder. Image needs to follow the shape: none, 255, 255, 3'''

    # Prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    # Format prediction to required format for explanations
    def format_for_expl(image, class_index):
        X = image
        y = np.expand_dims(np.array([class_index]), axis=0)
        return (X, y)

    image_tuple = format_for_expl(image, class_index)

    # Grad-CAM
    gradcam = GradCAM()
    grid_gradcam = gradcam.explain(image_tuple, model, class_index=class_index)
    gradcam.save(grid_gradcam, "../Visualizations/", "GradCam.png")

    return prediction, grid_gradcam
