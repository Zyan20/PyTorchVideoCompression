from os import path
import os

import argparse

import csv

"""
pipline:
    1. convert raw yuv to image (uncompressed)

    2. convert raw yuv to compressed mkv
    3. convert compressed mkv to compressed images

folder:
    sequences:
        - [seq]:
            - raw           (uncompressed images)
                - im0001.png
                - im0002.png


            - H265L[crf]    (compressed images)
                - im0001.png
                - im0002.png


            - out           (intermediate products)
                - out.mkv
                - report.csv
"""


def _form_cmd(opitions):
    cmd = ""
    for opt in opitions:
        cmd += (opt + " ")
    
    return cmd

def extract_imgs_from_yuv(input, output_folder, im_width, im_height):
    options = [
        "ffmpeg",
        "-pix_fmt", "yuv420p",
        "-s:v", f"{im_width}x{im_height}",
        "-i", input,
        "-f", "image2",
        f"{output_folder}/im%04d.png"
    ]
    cmd = _form_cmd(options)
    print(cmd)
    os.system(cmd)

def extract_imgs_from_video(input, output_folder):
    options = [
        "ffmpeg",
        "-i", input,
        "-f", "image2",
        f"{output_folder}/im%04d.png"
    ]
    cmd = _form_cmd(options)
    print(cmd)
    os.system(cmd)


def encode_video(input, output_folder, codec, crf, GOP):
    options = [
        "ffmpeg",

        "-pix_fmt", "yuv420p",
        "-s:v", f"{im_width}x{im_height}",
        "-i", input,

        "-c:v", codec,
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-x265-params", f"crf={crf}:keyint={GOP}:verbose=1:csv-log-level=1:csv={output_folder}/report.txt",
        f"{output_folder}/out.mkv"
    ]
    cmd = _form_cmd(options)
    print(cmd)
    os.system(cmd)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yuv_seq_folder", type = str, help = "raw yuv sequences")
    parser.add_argument("--yuv_width", type = str)
    parser.add_argument("--yuv_height", type = str)


    parser.add_argument("--save_folder", type = str, help = "sequences folder to save", default = "./sequences")

    parser.add_argument("--codec", type = str, help = "codec to enode raw sequenes", default = "libx265")
    parser.add_argument("--crf", type = str, default = "20")
    parser.add_argument("--GOP", type = str, default = "12")


    args = parser.parse_args()

    yuv_folder, save_folder = args.yuv_seq_folder, args.save_folder
    im_width, im_height = args.yuv_width, args.yuv_height

    codec, crf, GOP = args.codec, args.crf, args.GOP


    yuv_videos = []
    for doc in os.listdir(yuv_folder):
        if doc.endswith(".yuv"):
            yuv_videos.append(doc)

    for yuv in yuv_videos:
        sub_folder = yuv[:-4]

        # extract uncompresssd image for raw yuv
        output_folder = path.join(save_folder, sub_folder, "uncompressed")

        if not path.exists(output_folder):
            os.makedirs(output_folder)

            input = path.join(yuv_folder, yuv)
            extract_imgs_from_yuv(input, output_folder, im_width, im_height)

        # encode yuv videos into out.mkv
        output_folder = path.join(save_folder, sub_folder, "out")

        if not path.exists(output_folder):
            os.makedirs(output_folder)

            input = path.join(yuv_folder, yuv)
            encode_video(input, output_folder, codec, crf, GOP)

        # extract compresssd image for encoded out.mkv
        output_folder = path.join(save_folder, sub_folder, "compressed")

        if not path.exists(output_folder):
            os.makedirs(output_folder)

            input = path.join(save_folder, sub_folder, "out", "out.mkv")
            extract_imgs_from_video(input, output_folder)

