"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np
from PIL import Image
import os
import shutil
from tifffile import imwrite, imread

def unprocess(img, scale, std=0.5, mean=0.5):
    #print("init img", img)
    # img = img[0:2].cpu().float().numpy()
    temp = np.zeros((1,512,512))
    temp = img.cpu().float().numpy()
    # temp[1] = img[1].cpu().float().numpy()
    # temp[2] = img[2].cpu().float().numpy()
    # print(temp.max())
    #print("img max", np.max(img))
    temp = ((temp * std) + mean) * scale
    # temp = temp[0,:,:]
    return temp.astype(float)
    #print("img3",img)
    # return Image.fromarray(img.astype(np.uint16))
    # return Image.fromarray(img.reshape(3,512,512).astype(np.uint16))

def current(img):
    temp = np.zeros((1,376,344))
    temp = img.cpu().float().numpy()

    return temp.astype(float)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # dirnames = ["FakeB", "RealA", "RealB"]
    dirnames = ["FakeA", "FakeB", "RealA", "RealB", "RecA", "RecB"]
    rtname = "results\\"+opt.dataroot.split("\\")[-1]+"_results_"+opt.epoch
    if os.path.isdir(rtname):
        shutil.rmtree(rtname)
    os.mkdir(rtname)
    for dirname in dirnames:
        os.mkdir(rtname+"\\"+dirname)

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # opt.norm = batchnorm
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        if (i+1) % 200 == 0:
            print("Processed", i+1, "images")
            
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        # fakeB = model.fake_B
        fakeB, recA, fakeA, recB = model.fake_B, model.rec_A, model.fake_A, model.rec_B


        ImFakeB = current(fakeB.data[0])
        ImRecA  = current(recA.data[0])
        ImRecB  = current(recB.data[0])
        ImFakeA = current(fakeA.data[0])
        
        ImRealA = current(data["A"].data[0])
        ImRealB = current(data["B"].data[0])

        # print(data)
        img_ind = data["A_paths"][0].split("\\")[-1].split("_")[:-1]

        # ImFakeB.save(rtname+"\\FakeB\\"+"_".join(img_ind)+"_FakeB.tiff")
        np.save(rtname+"\\FakeB\\"+"_".join(img_ind)+"_FakeB.npy",ImFakeB)

        # ImFakeA.save(rtname+"\\FakeA\\"+"_".join(img_ind)+"_FakeA.tiff")
        np.save(rtname+"\\FakeA\\"+"_".join(img_ind)+"_FakeA.npy",ImFakeA)

        # ImRecB.save(rtname+"\\RecB\\"+"_".join(img_ind)+"_RecB.tiff")
        np.save(rtname+"\\RecB\\"+"_".join(img_ind)+"_RecB.npy",ImRecB)

        # ImRecA.save(rtname+"\\RecA\\"+"_".join(img_ind)+"_RecA.tiff")
        np.save(rtname+"\\RecA\\"+"_".join(img_ind)+"_RecA.npy",ImRecA)

        # ImRealA.save(rtname+"\\RealA\\"+"_".join(img_ind)+"_RealA.tiff")
        np.save(rtname+"\\RealA\\"+"_".join(img_ind)+"_RealA.npy",ImRealA)

        # ImRealB.save(rtname+"\\RealB\\"+"_".join(img_ind)+"_RealB.tiff")
        np.save(rtname+"\\RealB\\"+"_".join(img_ind)+"_RealB.npy",ImRealB)
        

        ####################
        #if i % 20 == 0:
        #    print(data)
        #    print(data["A_paths"][0].split("/")[-1])
               
        #make directories for these four as well as the real data

