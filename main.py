print("-----Initializing-----")

import argparse
import time
from SDD_FIQA import *
from animations.animations import *
from face_reg.detection import *
from SmileScore.smileScore import *
from animations.make_video import *
import datetime
import warnings

warnings.filters('ignore')

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
                        type=str,
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
    if args.log:
        now = datetime.datetime.now()
        f = open(f"{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_log.txt", "w+")
        
        start = time.time()
        
        print("-----Starting face detection module-----")
        f.write("-----Starting face detection module-----\n")
        
        df = face_detection(args.original_dataset_path, args.anchor_dataset_path, args.find_person)
        
        f.write(str(list(df["filename"])))
        f.write("\n")
        f.write(str(list(df["bboxes"])))
        f.write("\n")
        f.write(str(list(df["ids"])))
        f.write("\n")
        
        end = time.time()
        
        print(f"-----Done face detection. Time since start {end-start}s-----")
        f.write(f"-----Done face detection. Time since start {end-start}s-----")
        f.write("\n")
        print("-----Starting face image quality assessment module-----")
        f.write("-----Starting face image quality assessment module-----")
        f.write("\n")
        
        df = FIQA(df=df, path=args.original_dataset_path)
        
        f.write(str(list(df["filename"])))
        f.write("\n")
        f.write(str(list(df["fiqa_score"])))
        f.write("\n")
        
        end = time.time()
        
        print(f"-----Done face image quality assessment. Time since start {end-start}s-----")
        f.write(f"-----Done face image quality assessment. Time since start {end-start}s-----")
        f.write("\n")
        print("-----Starting smile score assessment module-----")
        f.write("-----Starting smile score assessment module-----")
        f.write("\n")
        
        smile_model = load_smile_model(r"model/smile_score.h5")
        df = get_smile_score(path=args.original_dataset_path, df=df, model=smile_model)
        
        f.write(str(list(df["filename"])))
        f.write("\n")
        f.write(str(list(df["score"])))
        f.write("\n")
        
        end = time.time()
        
        print(f"-----Done smile score assessment. Time since start {end-start}s-----")
        f.write(f"-----Done smile score assessment. Time since start {end-start}s-----")
        f.write("\n")
        print("-----Starting create video-----")
        f.write("-----Starting create video-----")
        f.write("\n")
        
        # img_list = process_images_for_vid(list(df["filename"])[0:args.number_of_images], effect_speed=args.effect_speed, duration=args.duration,
                                          # fps=args.fps, fraction=args.fraction)
        print(list(df["filename"])[0:args.number_of_images])
        f.write(str(list(df["filename"])[0:args.number_of_images]))
        f.write("\n")
        
        make_video(img_list=list(df["filename"])[0:args.number_of_images], output_path=args.output_path, 
                   effect_speed=args.effect_speed, duration=args.duration, fps=args.fps, fraction=args.fraction)
                   
        end = time.time()
        
        print(f"-----Done create video. Time since start {end-start}s-----")
        f.write(f"-----Done create video. Time since start {end-start}s-----")
        f.write("\n")
        print("-----DONE-----")
        f.write("-----DONE-----")
        f.close()
        
    else:
        start = time.time()
        print("-----Starting face detection module-----")
        
        df = face_detection(args.original_dataset_path, args.anchor_dataset_path, args.find_person)
        
        end = time.time()
        print(f"-----Done face detection. Time since start {end-start}s-----")
        print("-----Starting face image quality assessment module-----")
        
        df = FIQA(df=df, path=args.original_dataset_path)
        
        end = time.time()
        print(f"-----Done face image quality assessment. Time since start {end-start}s-----")
        print("-----Starting smile score assessment module-----")
        
        smile_model = load_smile_model(r"model/smile_score.h5")
        df = get_smile_score(path=args.original_dataset_path, df=df, model=smile_model)
        
        end = time.time()
        print(f"-----Done smile score assessment. Time since start {end-start}s-----")
        print("-----Starting create video-----")
        
        # img_list = process_images_for_vid(list(df["filename"])[0:args.number_of_images], effect_speed=args.effect_speed, duration=args.duration,
                                          # fps=args.fps, fraction=args.fraction)
                                          
        print(list(df["filename"])[0:args.number_of_images])
        
        make_video(img_list=list(df["filename"])[0:args.number_of_images], output_path=args.output_path, 
                   effect_speed=args.effect_speed, duration=args.duration, fps=args.fps, fraction=args.fraction)
                   
        end = time.time()
        print(f"-----Done create video. Time since start {end-start}s-----")
        print("-----DONE-----")


if __name__ == '__main__':
    main()
