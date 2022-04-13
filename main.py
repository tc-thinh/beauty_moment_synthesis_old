import argparse

from SDD_FIQA import *
from animations.animations import *
from face_reg.detection import *
from SmileScore.smileScore import *


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
                        default=10)

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

    args = parser.parse_args()
    return args


def load_models():
    smile_model = load_model(r"")
    return smile_model


def main():
    args = parse_args()
    df = face_detection(args.original_dataset_path, args.anchor_dataset_path)
    df = FIQA(df)
    print(df.head())
    smile_model = load_model(r"")
    filename_list = get_smile_score(df, smile_model)  # return ordered image name
    print(filename_list)
    # img_list = process_images_for_vid(df, k=number_of_images, effect_speed=args.effect_speed, duration=args.duration,
    #                                   fps=args.fps)
    # make_video(img_list=img_list, output_path=output_path)


if __name__ == '__main__':
    main()
