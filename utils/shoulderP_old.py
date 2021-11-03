import glob
from utils.track import *


def train():
    mod_name = 'IV'

    up_path = './shoulder_p_train_u/'
    for filename in glob.glob(up_path+'/*.jpg'):
        file_path = filename
        model_name = mod_name
        framework_name = 'TFLite'
        visualize = True
        store = True
        track=Track()
        track.main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize,
             store=store)

    down_path=os.getcwd() + '/shoulder_p_train_d/'
    for filename in glob.glob(down_path+'/*.jpg'):
        file_path = filename
        model_name = mod_name
        framework_name = 'TFLite'
        visualize = True
        store = True
        track = Track()
        track.main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize,
             store=store)
    return 0


def run():
    file_path = None
    model_name = 'II_Lite'
    framework_name = 'TFLite'
    visualize = False
    store = True
    main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)

train()
