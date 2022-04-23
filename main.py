print("-----Initializing-----")

import argparse
import time
from SDD_FIQA import *
from animations.animations import *
from face_reg.detection import *
from SmileScore.smileScore import *
from animations.make_video import *
import datetime
from misc.log import *

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection and Recognition',
                                     usage='A module to detect and recognize faces in pictures')

    parser.add_argument('--anchor_dataset_path',
                        help='Path to your folder containing anchor images',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--original_dataset_path',
                        help='Path to your folder containing input images',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--output_path',
                        help='Output video path',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--number_of_images',
                        help='Number of images will be presented in the video',
                        type=int,
                        required=False,
                        default=6)

    parser.add_argument('--effect_speed',
                        help='Video args',
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument('--duration',
                        help='Video args',
                        type=int,
                        required=False,
                        default=3)

    parser.add_argument('--fps',
                        help='Video args',
                        type=int,
                        required=False,
                        default=75)
                        
    parser.add_argument('--fraction',
                        help='Resize images',
                        type=float,
                        required=False,
                        default=1)

    parser.add_argument('--find_person',
                        help='Find the person',
                        type = str,
                        required=False,
                        default=None)
                        
    parser.add_argument('--log',
                        help='Find the person',
                        type=bool,
                        required=False,
                        default=False)
                        
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    start = time.time()
    
    print("-----Starting face detection module-----")
    log = log_init()
    log = write_log(old_log=log, 
                    new_message="-----Starting face detection module-----", 
                    type="string + enter")

    df, input_img, new_log = face_detection(args.original_dataset_path, args.anchor_dataset_path, args.find_person)
    log = write_log(old_log=log, 
                    new_message=new_log, 
                    type="string + enter")
    
    log = write_log(old_log=log, 
                    new_message=df, 
                    type="dataframe + enter")
                    
    end = time.time()
    print(f"-----Done face detection. Time since start {end - start}s-----")
    log = write_log(old_log=log, 
                    new_message=f"-----Done face detection. Time since start {end - start}s-----", 
                    type="string + enter")
                    
    print("-----Starting face image quality assessment module-----")
    log = write_log(old_log=log, 
                    new_message="-----Starting face image quality assessment module-----", 
                    type="string + enter")

    df, input_img = FIQA(df, input_img)

    log = write_log(old_log=log, 
                    new_message=df, 
                    type="dataframe + enter")
                    
    end = time.time()
    print(f"-----Done face image quality assessment. Time since start {end - start}s-----")
    log = write_log(old_log=log, 
                    new_message=f"-----Done face image quality assessment. Time since start {end - start}s-----", 
                    type="string + enter")
    print("-----Starting smile score assessment module-----")
    log = write_log(old_log=log, 
                    new_message="-----Starting smile score assessment module-----", 
                    type="string + enter")

    smile_model = load_smile_model(r"model/smile_score.h5")
    df, input_img = get_smile_score(df, input_img, smile_model)
    
    log = write_log(old_log=log, 
                    new_message=df, 
                    type="dataframe + enter")
    
    end = time.time()
    print(f"-----Done smile score assessment. Time since start {end - start}s-----")
    log = write_log(old_log=log, 
                    new_message=f"-----Done smile score assessment. Time since start {end - start}s-----", 
                    type="string + enter")
                    
    print("-----Starting create video-----")
    log = write_log(old_log=log, 
                    new_message="-----Starting create video-----", 
                    type="string + enter")

    make_video(img_list=input_img[:args.number_of_images],
               output_path=args.output_path,
               effect_speed=args.effect_speed, 
               duration=args.duration, 
               fps=args.fps, 
               fraction=args.fraction)

    end = time.time()
    print(f"-----Done create video. Time since start {end - start}s-----")
    log = write_log(old_log=log, 
                    new_message=f"-----Done create video. Time since start {end - start}s-----", 
                    type="string + enter")
                    
    print("-----DONE-----")
    log = write_log(old_log=log, 
                    new_message="-----DONE-----", 
                    type="string + enter")
    
    if args.log:
        log_final(log)


if __name__ == '__main__':
    main()
