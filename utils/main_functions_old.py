from utils.track import Track
from utils.ShoulderP import ShoulderP
from utils import helpers
import time
import sys
from getopt import getopt, error
from os.path import join, normpath

class MainFunctions(Track):

    def analyze_camera(self, model, framework, resolution, lite):
        sp_state = False
        # Load video
        import cv2
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        coordinates = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('\n##########################################################################################################')
        while (True):
            # Read frame
            _, frame = cap.read()
            # Construct batch
            batch = [frame[..., ::-1]]
            # Preprocess batch
            batch = helpers.preprocess(batch, resolution, lite)
            # Perform inference
            batch_outputs = self.infer(batch, model, lite, framework)
            # Extract coordinates for frame
            frame_coordinates = helpers.extract_coordinates(batch_outputs[0, ...], frame_height, frame_width,real_time=True)
            #Todo: add discriminator
            #Show Counts
            text = "Counts : " + str(ShoulderP.times)
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, text, (0, 100), font, 1, (255, 255, 255), 2)
            frame = cv2.flip(frame,1)

            #ADD FUNCTIONS HERE
            ShoulderP.draw_circle(frame, frame_coordinates, frame_width, frame_height)
            sp_state = ShoulderP.sp_count(frame_coordinates, sp_state)
            # print(ShoulderP.times)
            # print(ShoulderP.rate_r, ShoulderP.rate_l)

            coordinates += [frame_coordinates]
            # Draw and display predictions

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print total operation time
        print('Camera operated in {0} seconds'.format(time.time() - start_time))
        print(
            '##########################################################################################################\n')
        # print(temp)
        return coordinates




##########실행
if __name__ == '__main__':
    # Fetch arguments
    args = sys.argv[1:]

    # Define options
    short_options = 'p:m:f:vs'
    long_options = ['path=', 'model=', 'framework=', 'visualize', 'store']
    try:
        arguments, values = getopt(args, short_options, long_options)
    except error as err:
        print(
            '\n##########################################################################################################')
        print(str(err))
        print(
            '##########################################################################################################\n')
        sys.exit(2)

    # Define default choices
    file_path = None
    # './shoulder_p_train_u/7.jpg'
    model_name = 'I_lite'
    framework_name = 'tflite'
    visualize = False
    store = False
    mode = 'shoulderp'

    # Set custom choices
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--path'):
            file_path = current_value if len(current_value) > 0 else None
        elif current_argument in ('-m', '--model'):
            model_name = current_value
        elif current_argument in ('-f', '--framework'):
            framework_name = current_value
        elif current_argument in ('-v', '--visualize'):
            visualize = True
        elif current_argument in ('-s', '--store'):
            store = True
    print(
        '\n##########################################################################################################')
    print(
        'The program will attempt to analyze {0} using the "{1}" framework with model "{2}", and the user did{3} like to store the predictions and wanted{4} to visualize the result.'.format(
            '"' + file_path + '"' if file_path is not None else 'the camera', framework_name, model_name,
            '' if store else ' not', '' if visualize or file_path is None else ' not'))
    print(
        '##########################################################################################################\n')
    track = MainFunctions()
    track.main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize,
               store=store, mode=mode)