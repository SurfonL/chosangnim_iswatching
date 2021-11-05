import cv2
from streamlit_webrtc import webrtc_streamer
import av
from utils.ShoulderP import ShoulderP
from os.path import join, normpath
# from utils import helpers


#resolution



class VideoProcessor:
    def recv(self, frame):

        #(480 640 3)
        frame = frame.to_ndarray(format="bgr24")

        # #batch인데 어차피 1임
        # batch = [frame[...,::-1]]
        #
        # framework_name = 'tflite'
        # model_name = 'I_lite'
        # framework = framework_name.lower()
        # model_variant = model_name.lower()
        # lite = True if model_variant.endswith('_lite') else False
        # frame_height, frame_width = frame.shape[:2]
        #
        # model, resolution = self.get_model(framework, model_variant)
        #
        # # Preprocess batch
        # batch = helpers.preprocess(batch, resolution, lite)
        # batch_outputs = self.infer(batch, model, lite, framework)
        #
        # # Extract coordinates for frame
        # frame_coordinates = helpers.extract_coordinates(batch_outputs[0, ...], frame_width, frame_height, real_time=True)
        #
        # frame = cv2.line(frame, 1)
        # frame = ShoulderP.draw_circle(frame, frame_coordinates, frame_width, frame_height)
        # print(frame)


        return av.VideoFrame.from_ndarray(frame, format="bgr24")


    def get_model(self, framework, model_variant):
        """
        Load the desired EfficientPose model variant using the requested deep learning framework.

        Args:
            framework: string
                Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
            model_variant: string
                EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)

        Returns:
            Initialized EfficientPose model and corresponding resolution.
        """

        # Keras
        if framework in ['keras', 'k']:
            from tensorflow.keras.backend import set_learning_phase
            from tensorflow.keras.models import load_model
            set_learning_phase(0)
            model = load_model(join('models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())), custom_objects={'BilinearWeights': helpers.keras_BilinearWeights, 'Swish': helpers.Swish(helpers.eswish), 'eswish': helpers.eswish, 'swish1': helpers.swish1})

        # TensorFlow
        elif framework in ['tensorflow', 'tf']:
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
            model = lite.Interpreter(model_path=join('models', 'tflite', 'EfficientPose{0}.tflite'.format(model_variant.upper())))
            model.allocate_tensors()

        # PyTorch
        elif framework in ['pytorch', 'torch']:
            from imp import load_source
            try:
                MainModel = load_source('MainModel', join('models', 'pytorch', 'EfficientPose{0}.py'.format(model_variant.upper())))
            except:
                print('\n##########################################################################################################')
                print('Desired model "EfficientPose{0}Lite" not available in PyTorch. Please select among "RT", "I", "II", "III" or "IV".'.format(model_variant.split('lite')[0].upper()))
                print('##########################################################################################################\n')
                return False, False
            model = load(join('models', 'pytorch', 'EfficientPose{0}'.format(model_variant.upper())))
            model.eval()
            qconfig = quantization.get_default_qconfig('qnnpack')
            backends.quantized.engine = 'qnnpack'

        return model, {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[model_variant]

    def infer(self, batch, model, lite, framework):
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


webrtc_streamer(key="example", video_processor_factory=VideoProcessor)