from src.BaseModel import BaseModel
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time 
from scipy.signal import savgol_filter
from src.utils.drawing import draw_player_point,draw_player_rect,draw_point
from src.homography_package import get_transformed_point
from src.utils.utils import get_player_point,calculate_distance
from src.cluster_package import cluster_players_team,predict_team
from deep_sort_realtime.deepsort_tracker import DeepSort
import keyboard



# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='top-down view for football field')
parser.add_argument('--source', type=str,default='clibs/possession.mp4', help='path to source "img or video"')
parser.add_argument('--output', type=str,default='output/possession_test.mp4', help='path to output')

tracker= DeepSort(max_age=10)


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
    corners_model = YOLO('models/corners_new4.pt')
    game_objects_model = YOLO('models/game_objects2.pt')
    # model = YOLO('models/all_game_objects1.pt')
    basemodel = BaseModel(corners_model,game_objects_model)


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
        top_view_H = basemodel.get_top_view_homography(corners)
        top_view = basemodel.warp_top_view(img,top_view_H)
        mrg_view_H = basemodel.get_merge_view_homography(corners)
        mrg_view = basemodel.warp_merge_view(img,mrg_view_H)

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
        # out = cv2.VideoWriter(args.output, fourcc, fps, (640*3, 640))
        out = cv2.VideoWriter(args.output, fourcc, fps, (1280,720))

        #wake up the model
        ret, frame = cap.read()
        print('loading models....')
        # basemodel.detect_all_game_objects(frame)
        corners = basemodel.detect_pitch_corners(frame)
        objects_output = basemodel.detect_game_objects(frame,conf=.5)
        print('models loaded')

        topview_homo_history = []
        mrgview_homo_history = []

        frame_counter = 0
        players_for_cluster = []

        last_loc = None
        
        ball_history = []

        top_view = basemodel.est
        mrg_view = frame
        top_view_analysis = basemodel.est.copy()

        
        team1_possession = 1
        team2_possession = 1
        teams_list = []
        
        while cap.isOpened():

            if not basemodel.clustered:
                # corners,objects_output = basemodel.detect_all_game_objects(frame,0.5)
                corners = basemodel.detect_pitch_corners(frame,conf=0.25)
                objects_output = basemodel.detect_game_objects(frame,conf=.8)
                for i in objects_output:
                    players_for_cluster.append([i[0][0],i[0][1],i[0][0]+i[0][2],i[0][1]+i[0][3]])
                sucess,basemodel.colors = cluster_players_team(frame,players_for_cluster,basemodel.cluster_model)
                if sucess:
                    cap = cv2.VideoCapture(args.source)
                    basemodel.clustered = True
                    print('team clustered')
                    continue
                else:
                    continue

            ret, frame = cap.read()
            
            if ret == True:
                frame = cv2.resize(frame,(1280,720))
                start = time.time()
                frame_copy = frame.copy()

                frame_counter += 1
                


                if frame_counter%1 == 0 and not(keyboard.is_pressed('v')):

                    top_view_analysis = basemodel.est.copy()
                    
                    corners = basemodel.detect_pitch_corners(frame,conf=0.15)
                    objects_output = basemodel.detect_game_objects(frame,conf=.5)
                    
                    # corners,objects_output = basemodel.detect_all_game_objects(frame,0.5)
                    objects_output = tracker.update_tracks(objects_output,frame=frame)
                            
                    # frame = basemodel.draw_corners(frame,corners)

                    try:
                        # top_view_H = basemodel.get_top_view_homography(corners,thresh=80)
                        mrg_view_H = basemodel.get_merge_view_homography(corners,thresh=80)
                        top_view_H = np.linalg.inv(mrg_view_H)

                        topview_homo_history.append(top_view_H)
                        mrgview_homo_history.append(mrg_view_H)

                        if len(topview_homo_history) > 30:
                            topview_homo_history = savgol_filter(topview_homo_history, 30, 1, axis=0).tolist()
                            top_view_H = topview_homo_history[-1]
                            topview_homo_history = topview_homo_history[-30:]
                        
                        if len(mrgview_homo_history) > 30:
                            mrgview_homo_history = savgol_filter(mrgview_homo_history, 30, 1, axis=0).tolist()
                            mrg_view_H = mrgview_homo_history[-1]
                            mrgview_homo_history = mrgview_homo_history[-30:]

                        top_view = basemodel.warp_top_view(frame,np.array(top_view_H))
                        mrg_view = basemodel.warp_merge_view(frame,np.array(mrg_view_H))

                        
                        #detect ball
                        mask = basemodel.est.copy()
                        cv2.rectangle(mask,basemodel.est_points['23'],basemodel.est_points['6'],(255,255,255),-1)
                        cv2.rectangle(mask,(26,268),(55,340),(255,255,255),-1)
                        cv2.rectangle(mask,(585,268),(609,340),(255,255,255),-1)


                        cv2.circle(mask,(531,312),3,(0,0,0),-1)
                        cv2.circle(mask,(106,312),3,(0,0,0),-1)

                        mask = cv2.warpPerspective(
                            mask, np.array(mrg_view_H), (frame.shape[1], frame.shape[0])
                        )
                        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

                        frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

                        ball,last_loc = basemodel.detect_ball(frame_copy,last_loc)
                                    
                        if ball:
                            ball_history.append(ball)
                            if len(ball_history)>10:
                                ball_history = ball_history[-10:]

                        else:
                            if len(ball_history)>1:
                                last_ball = np.array(ball_history[-1])
                                last_last_ball = np.array(ball_history[-2])
                                ball = list(last_ball + (last_ball-last_last_ball))
                                
                        

                        closthest_player = None
                        closthest_player_point = None
                        clothest_dist = float('inf')
                        closthest_player_team = None

                        for track in objects_output:
                            if not track.is_confirmed():
                                continue
                            det_id = track.det_class
                            class_name = basemodel.game_objects_model.names[det_id]
                            ltrb = track.to_ltrb()
                            if class_name == 'player':
                                player = [int(i) for i in ltrb]
                                try:
                                    team = predict_team(player,frame,basemodel.cluster_model)
                                    player_point = get_player_point(player)
                                    if ball:
                                        dist = calculate_distance(ball[0],ball[1],player_point[0],player_point[1])
                                        
                                        if dist < clothest_dist and dist < 25:
                                            clothest_dist = dist
                                            closthest_player = player
                                            closthest_player_team = team
                                            closthest_player_point = player_point
                                            teams_list.append(team)

                                    transformed_player = get_transformed_point(player_point,np.array(top_view_H))
                                    if basemodel.est_points['28'][1]>transformed_player[1]>basemodel.est_points['1'][1] and basemodel.est_points['1'][0]>transformed_player[0]>basemodel.est_points['23'][0]:
                                        top_view_analysis = draw_player_point(top_view_analysis,transformed_player,basemodel.colors[team],size=10)
                                except:
                                    pass
                        
                        
                        # for track in objects_output:
                        #     det_id = track[2]
                        #     class_name = basemodel.game_objects_model.names[det_id]
                        #     ltrb = [track[0][0],track[0][1],track[0][0]+track[0][2],track[0][1]+track[0][3]]
                        #     if class_name == 'player':
                        #         player = [int(i) for i in ltrb]
                        #         try:
                        #             team = predict_team(player,frame,basemodel.cluster_model)
                        #             player_point = get_player_point(player)
                        #             if ball:
                        #                 dist = calculate_distance(ball[0],ball[1],player_point[0],player_point[1])
                                        
                        #                 if dist < clothest_dist and dist < 25:
                        #                     clothest_dist = dist
                        #                     closthest_player = player
                        #                     closthest_player_team = team
                        #                     closthest_player_point = player_point
                        #                     teams_list.append(team)

                        #             transformed_player = get_transformed_point(player_point,np.array(top_view_H))
                        #             if basemodel.est_points['28'][1]>transformed_player[1]>basemodel.est_points['1'][1] and basemodel.est_points['1'][0]>transformed_player[0]>basemodel.est_points['23'][0]:
                        #                 top_view_analysis = draw_player_point(top_view_analysis,transformed_player,basemodel.colors[team],size=10)
                        #         except:
                        #             pass


                        if ball:
                            if closthest_player_point:
                                upper_player_point = [closthest_player_point[0],closthest_player[1]-10]
                                cv2.circle(frame,upper_player_point,5,basemodel.colors[closthest_player_team],-1)
                        
                        if ball:
                            transformed_ball = list(get_transformed_point(ball,np.array(top_view_H)))
                            transformed_ball[1] += 5

                            if basemodel.est_points['28'][1]>transformed_ball[1]>basemodel.est_points['1'][1] and basemodel.est_points['1'][0]>transformed_ball[0]>basemodel.est_points['23'][0]:
                                top_view_analysis = draw_player_point(top_view_analysis,transformed_ball,(0,255,255),size=7)
                                mrg_view = draw_player_point(mrg_view,ball,(0,0,255),size=10)

                        if len(teams_list)>=15:
                            t1 = teams_list.count('team1')
                            t2 = teams_list.count('team2')
                            if t1>t2:
                                team1_possession += 1
                            elif t2>t1:
                                team2_possession += 1
                            teams_list = []

                    except Exception as e:
                        print(e)
                        top_view = basemodel.est
                        mrg_view = frame
                        topview_homo_history = []
                        mrgview_homo_history = []

                top_view_analysis_copy = top_view_analysis[basemodel.est_points['23'][1]:basemodel.est_points['28'][1]+5,basemodel.est_points['23'][0]:basemodel.est_points['1'][0],:]
                frame[620-90:620+90,640-160:640+160,:] = cv2.resize(top_view_analysis_copy,(320,180))
                cv2.rectangle(frame,(640-160,620-130),(640+160,620-90),(0,0,0),-1)
                try:
                    cv2.putText(
                        frame,
                        f' possession',
                        (640-95,620-105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,255,255),
                        3
                    )
                    cv2.putText(
                        frame,
                        f'{int((team1_possession/(team1_possession+team2_possession))*100)}%',
                        (640-150,620-101),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        basemodel.colors['team1'],
                        2
                    )
                    cv2.putText(
                        frame,
                        f'{int((team2_possession/(team1_possession+team2_possession))*100)}%',
                        (640+95,620-101),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        basemodel.colors['team2'],
                        2
                    )
                except:
                    pass

                
                
                cv2.imshow('img',frame)
                out.write(frame)
                # frame = cv2.resize(frame,(640,640))
                # top_view = cv2.resize(top_view,(640,640))
                # mrg_view = cv2.resize(mrg_view,(640,640))
                # top_view_analysis = cv2.resize(top_view_analysis,(640,640))

                # try:
                #     cv2.putText(frame,f'team1 possession: {int((team1_possession/(team1_possession+team2_possession))*100)}%',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,basemodel.colors['team1'],2)
                #     cv2.putText(frame,f'team2 possession: {int((team2_possession/(team1_possession+team2_possession))*100)}%',(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,basemodel.colors['team2'],2)
                # except Exception as e:
                #     print(e)

                # comp = np.concatenate([frame,top_view,top_view_analysis],axis=1)
                # out.write(comp)

                print(f'time taken for frame {frame_counter} is: {round(time.time()-start,2)} S')

                
                # cv2.imshow('frame', comp)
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


