from src.BaseModel import BaseModel
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time 

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='top-down view for football field')
parser.add_argument('--source', type=str, help='path to source "img or video"')
parser.add_argument('--output', type=str, help='path to output')


def main():
    args = parser.parse_args()
    image_format = False
    video_format = False
    # List of common image formats
    image_formats = ['jpg', 'jpeg', 'png']
    # List of common video formats
    video_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']

    source_format = args.source.split('.')[-1]
    if (source_format in image_formats):
        image_format = True
    elif (source_format in video_formats):
        video_format = True
    else:
        raise('Invalid input format.')
    
    output_format = args.output.split('.')[-1]
    if image_format:
        if output_format not in image_formats:
            raise('Output format isn\'t like input format.')
    elif video_format:
        if output_format not in video_formats:
            raise('Output format isn\'t like input format.')
    
    
    # load models
    corners_model = YOLO('models/corners_new.pt')
    basemodel = BaseModel(corners_model)


    if image_format:
        img = cv2.imread(args.source)

        #wake up the model
        print('loading model....')
        basemodel.detect_pitch_corners(img)
        print('model loaded')

        start = time.time()
        corners = basemodel.detect_pitch_corners(img)
        print(f'time taken to detect corners: {round(time.time()-start,2)} S')

        img = basemodel.draw_corners(img,corners)
        top_view, _ = basemodel.get_top_view(img, corners)
        mrg_view,_ = basemodel.get_merge_view(img,corners)

        img = cv2.resize(img,(640,640))
        top_view = cv2.resize(top_view,(640,640))
        mrg_view = cv2.resize(mrg_view,(640,640))

        comp = np.concatenate([img,top_view,mrg_view],axis=1)

        cv2.imwrite(args.output, comp)
        print('output saved to ',args.output)
    
    elif video_format:
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (640*3, 640))

        #wake up the model
        ret, frame = cap.read()
        print('loading model....')
        basemodel.detect_pitch_corners(frame)
        print('model loaded')

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_counter += 1
                
                start = time.time()
                corners = basemodel.detect_pitch_corners(frame)
                print(f'time taken to detect corners for frame {frame_counter} is: {round(time.time()-start,2)} S')

                frame = basemodel.draw_corners(frame,corners)
                try:
                    top_view, _ = basemodel.get_top_view(frame, corners,thresh=150)
                    mrg_view,_ = basemodel.get_merge_view(frame,corners)
                except:
                    top_view = basemodel.est
                    mrg_view = frame

                frame = cv2.resize(frame,(640,640))
                top_view = cv2.resize(top_view,(640,640))
                mrg_view = cv2.resize(mrg_view,(640,640))

                comp = np.concatenate([frame,top_view,mrg_view],axis=1)
                out.write(comp)

                cv2.imshow('frame', comp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print('output saved to ',args.output)
    


if __name__ == '__main__':
    main()


