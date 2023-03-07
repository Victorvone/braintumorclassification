from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.vanilla_gradients import VanillaGradients


def predict_and_explain(model,
                        image_tuple,
                        grad_cam: True,
                        occlusion_sensitivity: True,
                        vanilla_gradient: True):
    '''Returns prediction of tumor classification and saves Grad-CAM, Occlusion sensitivity
    and Vanilla gradient pictures in ../visualizations folder. Image tuple needs
    to contain two arrays (X, y) of the following shape: none, 255, 255, 3 and none, 4'''

    # prediction
    prediction = model.predict(image_tuple)

    # Grad-CAM
    if grad_cam is True:
        gradcam = GradCAM()
        grid = gradcam.explain(image_tuple, model, class_index=0, layer_name='conv2d_7')
        # maybe change class_index to variable +
        gradcam.save(grid, "../Visualizations/", "grad_cam.png")

    # Occlusion sensitivity
    if occlusion_sensitivity is True:
        occsens = OcclusionSensitivity()
        grid = occsens.explain(image_tuple, model, class_index=0, patch_size=3)
        occsens.save(grid, "../Visualizations/", "OcclusionSensitivity.png")

    # Vanilla gradient
    if vanilla_gradient is True:
        Vanillagrad = VanillaGradients()
        grid = Vanillagrad.explain(image_tuple, model, class_index=0)
        Vanillagrad.save(grid, "../Visualizations/", "VanillaGradient.png")

    return prediction
