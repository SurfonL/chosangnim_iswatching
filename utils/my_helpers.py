from os.path import join
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input as efficientnet_preprocess_input
from skimage.transform import rescale
from skimage.util import pad as padding
import cv2
import math

def get_model(framework, model_variant):
    # TensorFlow
    if framework in ['tensorflow', 'tf']:
        from tensorflow.python.platform.gfile import FastGFile
        from tensorflow.compat.v1 import GraphDef
        from tensorflow.compat.v1.keras.backend import get_session
        from tensorflow import import_graph_def
        f = FastGFile(join('models', 'tensorflow', 'EfficientPose{0}.pb'.format(model_variant.upper())), 'rb')
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
        model = get_session()
        model.graph.as_default()
        import_graph_def(graph_def)

    # TensorFlow Lite
    elif framework in ['tensorflowlite', 'tflite']:
        from tensorflow import lite
        model = lite.Interpreter(
            model_path=join('models', 'tflite', 'EfficientPose{0}.tflite'.format(model_variant.upper())))
        model.allocate_tensors()

    return model, \
           {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[
               model_variant]

def infer(batch, model, lite, framework):
    """
    Perform inference on supplied image batch.

    Args:
        batch: ndarray
            Stack of preprocessed images
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        lite: boolean
            Defines if EfficientPose Lite model is used
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)

    Returns:
        EfficientPose model outputs for the supplied batch.
    """

    # Keras
    if framework in ['keras', 'k']:
        if lite:
            batch_outputs = model.predict(batch)
        else:
            batch_outputs = model.predict(batch)[-1]

    # TensorFlow
    elif framework in ['tensorflow', 'tf']:
        output_tensor = model.graph.get_tensor_by_name('upscaled_confs/BiasAdd:0')
        if lite:
            batch_outputs = model.run(output_tensor, {'input_1_0:0': batch})
        else:
            batch_outputs = model.run(output_tensor, {'input_res1:0': batch})

    # TensorFlow Lite
    elif framework in ['tensorflowlite', 'tflite']:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], batch)
        model.invoke()
        batch_outputs = model.get_tensor(output_details[-1]['index'])

    return batch_outputs


def resize(source_array, target_height, target_width):
    """
    Resizes an image or image-like Numpy array to be no larger than (target_height, target_width) or (target_height, target_width, c).

    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Desired maximum height
        target_width: int
            Desired maximum width

    Returns:
        Resized Numpy array.
    """

    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]

    # Compute correct scale for resizing operation
    target_ratio = target_height / target_width
    source_ratio = source_height / source_width
    if target_ratio > source_ratio:
        scale = target_width / source_width
    else:
        scale = target_height / source_height

    # Perform rescaling
    resized_array = rescale(source_array, scale, multichannel=True)

    return resized_array


def pad(source_array, target_height, target_width):
    """
    Pads an image or image-like Numpy array with zeros to fit the target-size.

    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Height of padded image
        target_width: int
            Width of padded image

    Returns:
        Zero-padded Numpy array of shape (target_height, target_width) or (target_height, target_width, c).
    """

    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]

    # Ensure array is resized properly
    if (source_height > target_height) or (source_width > target_width):
        source_array = resize(source_array, target_height, target_width)
        source_height, source_width = source_array.shape[:2]

    # Compute padding variables
    pad_left = int((target_width - source_width) / 2)
    pad_top = int((target_height - source_height) / 2)
    pad_right = int(target_width - source_width - pad_left)
    pad_bottom = int(target_height - source_height - pad_top)
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    has_channels_dim = len(source_array.shape) == 3
    if has_channels_dim:
        paddings.append([0, 0])

    # Perform padding
    target_array = padding(source_array, paddings, 'constant')

    return target_array


def preprocess(batch, resolution, lite=False):
    """
    Preprocess Numpy array according to model preferences.

    Args:
        batch: ndarray
            Numpy array of shape (n, h, w, 3)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used

    Returns:
        Preprocessed Numpy array of shape (n, resolution, resolution, 3).
    """

    # Resize frames according to side
    batch = [resize(frame, resolution, resolution) for frame in batch]

    # Pad frames in batch to form quadratic input
    batch = [pad(frame, resolution, resolution) for frame in batch]

    # Convert from normalized pixels to RGB absolute values
    batch = [np.uint8(255 * frame) for frame in batch]

    # Construct Numpy array from batch
    batch = np.asarray(batch)

    # Preprocess images in batch
    if lite:
        batch = efficientnet_preprocess_input(batch, mode='tf')
    else:
        batch = efficientnet_preprocess_input(batch, mode='torch')

    return batch


def extract_coordinates(frame_output, frame_width, frame_height,  real_time=False):
    """
    Extract coordinates from supplied confidence maps.

    Args:
        frame_output: ndarray
            Numpy array of shape (h, w, c)
        frame_height: int
            Height of relevant frame
        frame_width: int
            Width of relevant frame
        real-time: boolean
            Defines if processing is performed in real-time

    Returns:
        List of predicted coordinates for all c body parts in the frame the outputs are computed from.
    """

    # Define body parts
    body_parts = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder',
                  'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip',
                  'left_knee', 'left_ankle']

    # Define confidence level
    confidence = 0.3

    # Fetch output resolution
    output_height, output_width = frame_output.shape[0:2]

    # Initialize coordinates
    frame_coords = []

    # Iterate over body parts
    for i in range(frame_output.shape[-1]):

        # Find peak point
        conf = frame_output[..., i]
        max_index = np.argmax(conf)
        peak_y = float(math.floor(max_index / output_width))
        peak_x = max_index % output_width

        # Verify confidence
        if real_time and conf[int(peak_y), int(peak_x)] < confidence:
            peak_x = -0.5
            peak_y = -0.5
        else:
            peak_x += 0.5
            peak_y += 0.5

        # Normalize coordinates
        peak_x /= output_width
        peak_y /= output_height

        # Convert to original aspect ratio
        if frame_width > frame_height:
            norm_padding = (frame_width - frame_height) / (2 * frame_width)
            peak_y = (peak_y - norm_padding) / (1.0 - (2 * norm_padding))
            peak_y = -0.5 / output_height if peak_y < 0.0 else peak_y
            peak_y = 1.0 if peak_y > 1.0 else peak_y
        elif frame_width < frame_height:
            norm_padding = (frame_height - frame_width) / (2 * frame_height)
            peak_x = (peak_x - norm_padding) / (1.0 - (2 * norm_padding))
            peak_x = -0.5 / output_width if peak_x < 0.0 else peak_x
            peak_x = 1.0 if peak_x > 1.0 else peak_x

        frame_coords.append((body_parts[i], peak_x, peak_y))

    return frame_coords

def std_process(frame, model_variant, model, resolution,framework):
    # (480 640 3)
    frame = frame.to_ndarray(format="bgr24")
    # batch인데 어차피 1임
    frame = frame[:,::-1]
    batch = [frame[..., ::-1]]
    lite = True if model_variant.endswith('_lite') else False
    frame_height, frame_width = frame.shape[:2]
    # Preprocess batch
    batch = preprocess(batch, resolution, lite)
    batch_outputs = infer(batch, model, lite, framework)
    # Extract coordinates for frame
    frame_coordinates = extract_coordinates(batch_outputs[0, ...], frame_width, frame_height,
                                                       real_time=True)

    return frame, frame_coordinates, frame_width, frame_height