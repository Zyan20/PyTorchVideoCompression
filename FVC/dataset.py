import os
import torch
import imageio
import numpy as np
import torch.utils.data as data
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels

from PIL import Image
from torchvision import transforms


class UVGDataSet(data.Dataset):
    def __init__(self, root="/data/dataset/UVG/images/", filelist="/data/dataset/UVG/originalv.txt", refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        print("Testing UVG dataset using I frame : " + refdir)
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = self.getbpp(refdir)
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1

            if testfull:
                framerange = cnt // 12
                if cnt % 12 > 0:
                    framerange += 1
                gop = 12
            else:
                framerange = 1
                gop = 12

            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, 'im'+str(i * gop + 1).zfill(4)+'.png')
                inputpath = []
                for j in range(gop):
                    if os.path.exists(os.path.join(root, seq, 'im' + str(i * gop + j + 1).zfill(3)+'.png')):
                        inputpath.append(os.path.join(root, seq, 'im' + str(i * gop + j + 1).zfill(3)+'.png'))
                    else:
                        break
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)


    def getbpp(self, ref_i_folder):
        Ibpp = dict()
        Ibpp['H265L20'] = []
        Ibpp['H265L23'] = []
        Ibpp['H265L26'] = []
        Ibpp['H265L29'] = []
        Ibpp['BPGQ20'] = []
        Ibpp['BPGQ22'] = []
        Ibpp['BPGQ24'] = []
        Ibpp['BPGQ26'] = []
        Ibpp['BPGQ28'] = []
        Ibpp['BPGQ30'] = []
        return np.mean(Ibpp[ref_i_folder])


    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refout = dict()
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if not "psnr" in refout:
                refout["bpp"] = self.refbpp[index]
                refout["psnr"] = CalcuPSNR(input_image, ref_image)
                refout["msssim"] = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image)

        input_images = np.array(input_images)

        return input_images, ref_image, refout



class DataSet(data.Dataset):
    def __init__(self, path="/data/dataset/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.path = path
        self.image_input_list, self.image_ref_list = self.get_single_vimeo(filefolderlist=self.path)
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.image_input_list))

    def singleframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_single_vimeo(filefolderlist=self.path)

    def multiframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_multi_vimeo(filefolderlist=self.path)

    def get_single_vimeo(self, rootdir="/data/dataset/vimeo_septuplet/sequences/", filefolderlist="/data/dataset/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 2
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
            input_list += [y]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref

    def get_multi_vimeo(self, rootdir="/data/dataset/vimeo_septuplet/sequences/", filefolderlist="/data/dataset/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 6
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

            for number in [-5, -4, -3, -2, -1, 0]:
                refnumber = curnumber + number
                refname = y[0:-5] + str(refnumber) + '.png'
                input_list += [refname]

            # for refnumber in range(curnumber - 1, curnumber - 7, -1):
            #     refname = y[0:-5] + str(refnumber) + '.png'
            #     input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.image_ref_list[index])
        ref_image = ref_image.astype(np.float32) / 255.0
        ref_image = ref_image.transpose(2, 0, 1)
        ref_image = torch.from_numpy(ref_image).float()
        input_images = []

        for input_name in self.image_input_list[index]:
            input_image = imageio.imread(input_name)
            input_image = input_image.astype(np.float32) / 255.0
            input_image = input_image.transpose(2, 0, 1)
            input_image = torch.from_numpy(input_image).float()
            input_images.append(input_image)
        input_images = torch.cat(input_images, 0)

        ref_image, input_images = random_crop_and_pad_image_and_labels(ref_image, input_images, [self.im_height, self.im_width])
        ref_image, input_images = random_flip(ref_image, input_images)

        return ref_image, input_images


class HEVCDataSet(data.Dataset):

    def __init__(self,
        seqs_folder,
        crf = 20,
        GOP = 12            
    ) -> None:
        super().__init__()

        self.GOP = GOP

        self.refs = []
        self.inputs = []

        for seq in os.listdir(seqs_folder):
            ref_folder = os.path.join(seqs_folder, seq, f"H265L{crf}")

            img_folder = os.path.join(seqs_folder, seq)

            imgs = os.listdir(img_folder)

            frame_group = len(imgs) // GOP    # num of GOP

            for i in range(frame_group):
                ref_frame_path = os.path.join(ref_folder, "im{:04}.png".format(i * frame_group + 1))

                input_imgs = []
                for j in range(GOP):
                    img_path = os.path.join(img_folder, "im{:04}.png".format(i * GOP + j + 2))
                    if os.path.exists(img_path):
                        input_imgs.append(img_path)

                        
                self.refs.append(ref_frame_path)
                self.inputs.append(input_imgs)
                


    def __getitem__(self, index):
        ref_image = imageio.imread(self.refs[index]).transpose(2, 0, 1).astype(np.float32) / 255.0

        input_images = []
        for filename in self.inputs[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
            input_image, _ = self.pad(torch.tensor(input_image))
            input_images.append(input_image)



        ref, _ = self.pad(torch.Tensor(ref_image))
        inputs = torch.Tensor(input_images)

        return ref, inputs
        
            


    def pad(self, x, p = 64):
        h, w = x.size(1), x.size(2)

        new_h = (h + p - 1) // p * p     
        new_w = (w + p - 1) // p * p

        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left     
        padding_top = (new_h - h) // 2     
        padding_bottom = new_h - h - padding_top     
        padding = (padding_left, padding_right, padding_top, padding_bottom)     

        x = F.pad(
            x,         
            padding,         
            mode="constant",         
            value=0,     
        )
        return x, padding


    def crop(self, x, padding):
        return F.pad(x, tuple(-p for p in padding))

if __name__ == "__main__":
    data = HEVCDataSet(
        seqs_folder = "./data/HEVC_D/images"
    )

    (ref, inputs) = data[1]
    print(ref.size(), inputs.size())
