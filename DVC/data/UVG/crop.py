import subprocess
import os.path

path = "D:/VideoCompression/Dataset/UVG/"
docs = os.listdir(path)

for d in docs:
    if d.endswith(".yuv"):
        fileName = os.path.splitext(os.path.basename(d))[0]
        newFileName = fileName.replace("1920x1080", "1920x1024")
        cmd = [
            "ffmpeg",
            "-pix_fmt", "yuv420p",
            "-s", "1920x1080",
            "-i", path + d,
            "-vf", "crop=1920:1024:0:0",
            f"./videos_crop/{newFileName}.yuv"
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        # 获取输出流
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break

            print(output)