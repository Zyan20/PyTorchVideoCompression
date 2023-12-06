import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--seq_folder", type = str)
argparser.add_argument("-sw", "--seq_width", type = str)
argparser.add_argument("-sh", "--seq_height", type = str)

argparser.add_argument("--save_folder", type = str)
argparser.add_argument("-tw", "--target_weight", type = str)
argparser.add_argument("-th", "--target_height", type = str)


args = argparser.parse_args()

seq_folder, save_folder = args.seq_folder, args.save_folder
seq_width, seq_height = args.seq_width, args.seq_height
tar_width, tar_height = args.target_weight, args.target_height

docs = os.listdir(seq_folder)

for d in docs:
    if d.endswith(".yuv"):  # yuv文件

        fileName = os.path.splitext(os.path.basename(d))[0]     # short filename
        newFileName = fileName.replace(f"{seq_width}x{seq_height}", f"{tar_width}x{tar_height}")

        cmds = [
            "ffmpeg",
            "-pix_fmt", "yuv420p",
            "-s", f"{seq_width}x{seq_height}",
            "-i", seq_folder + d,
            "-vf", f"crop={tar_width}:{tar_height}:0:0",
            os.path.join(save_folder, newFileName) + ".yuv"
        ]

        cmd = ""
        for c in cmds:
            cmd += (c + " ")

        print(cmd)
        os.system(cmd)