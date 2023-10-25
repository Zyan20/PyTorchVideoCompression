import argparse
import os
import shutil

def getVideoNameShort(videoPath):
    return os.path.basename(videoPath).split("_")[0]

def makeDirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convertYUV2img(inputVideoPath, sw, sh, folder = "source"):
    videoNameShort = getVideoNameShort(inputVideoPath)

    cmd = f"ffmpeg \
    -pix_fmt yuv420p \
    -s:v {sw}x{sh} \
    -i {inputVideoPath} \
    -f image2 \
    out/{videoNameShort}/{folder}/img%04d.png"

    print(f"====== converting {inputVideoPath} to images =====")
    print(cmd)
    os.system(cmd)

    print("\n\n\n\n")


def convertVideo2img(inputVideoPath, outputFolder):

    cmd = f"ffmpeg \
    -i {inputVideoPath} \
    -f image2 \
    {outputFolder}/img%04d.png"

    print(f"====== converting {inputVideoPath} to images =====")
    print(cmd)
    os.system(cmd)

    print("\n\n\n\n")


def h265encode(inputVideoPath, crf, sw, sh):
    videoNameShort = getVideoNameShort(inputVideoPath)

    # frist 100 frames
    # keyint: interval of I frames
    cmd = f'ffmpeg \
    -pix_fmt yuv420p \
    -s:v {sw}x{sh} -i {inputVideoPath} \
    -c:v libx265 \
    -preset veryfast \
    -tune zerolatency \
    -x265-params \
    "crf={crf}:keyint=12:verbose=1:csv-log-level=1:csv=out/{videoNameShort}/report_{videoNameShort}.txt" \
    out/{videoNameShort}/h265/out.mkv'


    print(f"====== encoding {inputVideoPath} by h265 =====")
    print(cmd)
    os.system(cmd)
    print("\n\n\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("crf", type = str, default = "20")
    parser.add_argument("sw", type = str, help = "image size width", default = 1920)
    parser.add_argument("sh", type = str, help = "image size height", default = 1024)

    args = parser.parse_args()
    crf, sw, sh = args.crf, args.sw, args.sh


    # root folder for cropped uvg videos
    datasetFolder = "../videos_crop/"
    imageFolder = "../images/"
    videos = []

    docs = os.listdir(datasetFolder)
    for d in docs:
        if d.endswith(".yuv"):
            videos.append(d)        # only contains video.yuv

    
    for yuv in videos:
        videoNameShort = getVideoNameShort(yuv)

        # extract img.png from orignal yuv video
        if not os.path.exists(f"out/{videoNameShort}/source"):
            os.makedirs(f"out/{videoNameShort}/source")
            convertYUV2img(datasetFolder + yuv, sw, sh, "source")

        # encode yuv videos into out.mkv by h265
        if not os.path.exists(f"out/{videoNameShort}/h265"):
            os.makedirs(f"out/{videoNameShort}/h265")
            h265encode(datasetFolder + yuv, crf, sw, sh)

        # extract images from h265 encoded videos
        outputFolder = os.path.join(imageFolder, videoNameShort, f"H265L{crf}")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
            convertVideo2img(f"out/{videoNameShort}/h265/out.mkv", outputFolder)
        

        # dstPath = os.path.join(imageFolder, videoNameShort, f"H265L{crf}")
        # shutil.copytree(outputFolder, dstPath)