import numpy as np
import cv2
import math



def yuv420_to_rgb(yuv_file, width, height):
    # bytes of each frame
    frame_size = (width * height * 3) // 2

    with open(yuv_file, 'rb') as f:
        yuv_data = f.read()

    yuv_array = np.frombuffer(yuv_data, dtype = np.uint8)


    for i in range(0, len(yuv_array), frame_size):
        # for each frame
        yuv = yuv_array[i: i + frame_size].reshape((height * 3 // 2, width))
        RGB = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

        yield RGB



if __name__ == "__main__":
    yuv_file = "/media/zyan/data/HEVC_Sequence/416x240_50/BasketballPass_416x240_50.yuv"
    width = 416
    height = 240
    
    for frame in yuv420_to_rgb(yuv_file, width, height):
        cv2.imshow("RGB", frame)
        cv2.waitKey()


