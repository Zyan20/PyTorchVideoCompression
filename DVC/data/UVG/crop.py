import os.path


datasetFolder = "D:/VideoCompression/Dataset/UVG/"
docs = os.listdir(datasetFolder)

outputFolder = "./videos_crop/"

for d in docs:
    if d.endswith(".yuv"):  # yuv文件

        fileName = os.path.splitext(os.path.basename(d))[0]     # short filename
        newFileName = fileName.replace("1920x1080", "1920x1024")

        cmds = [
            "ffmpeg",
            "-pix_fmt", "yuv420p",
            "-s", "1920x1080",
            "-i", datasetFolder + d,
            "-vf", "crop=1920:1024:0:0",
            outputFolder + newFileName + ".yuv"
        ]

        cmd = ""
        for c in cmds:
            cmd += c + " "

        print(cmd)
        os.system(cmd)
