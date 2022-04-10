import argparse

from SDD_FIQA import *
from animations.animations import *


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
                        required=True,
                        default=10)

    parser.add_argument('--effect_speed',
                        help='Video args',
                        type=int,
                        required=True,
                        default=1)

    parser.add_argument('--duration',
                        help='Video args',
                        type=int,
                        required=True,
                        default=3)

    parser.add_argument('--fps',
                        help='Video args',
                        type=int,
                        required=True,
                        default=75)

    args = parser.parse_args()
    return args


def load_models():
    smile_model = smileScore.load_model("")
    return smile_model


def main():
    args = parse_args()
    df = face_reg(args.original_dataset_path, args.anchor_dataset_path)
    df = FIQA(df)
    smile_model = load_models()
    df = get_smile_scores(df)  # return ordered image name
    img_list = process_images_for_vid(df, k=number_of_images, effect_speed=args.effect_speed, duration=args.duration,
                                      fps=args.fps)
    make_video(img_list=img_list, output_path=output_path)


if __name__ == '__main__':
    main()
