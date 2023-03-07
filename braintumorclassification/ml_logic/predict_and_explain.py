from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.vanilla_gradients import VanillaGradients


def predict_and_explain(model, image_array):
    '''Returns prediction of tumor classification and saves Grad-CAM, Occlusion sensitivity
    and Vanilla gradient pictures in ../visualizations folder'''

    # prediction
    prediction = model.predict(image_array)

    # Grad-CAM
    gradcam = GradCAM()
    grid = gradcam.explain(image_array, model, class_index=0, layer_name='conv2d_7')
    # maybe change class_index to variable +
    gradcam.save(grid, "../Visualizations/", "grad_cam.png")

    # Occlusion sensitivity
    occsens = OcclusionSensitivity()
    grid = occsens.explain(image_array, model, class_index=0, patch_size=3)
    occsens.save(grid, "../Visualizations/", "OcclusionSensitivity.png")

    # Vanilla gradient
    Vanillagrad = VanillaGradients()
    grid = Vanillagrad.explain(image_array, model, class_index=0)
    Vanillagrad.save(grid, "../Visualizations/", "VanillaGradient.png")

    return prediction
