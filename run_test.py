import optparse

import numpy as np
import cv2

from awb import AWB_GrayWorld
from convert import Convert
from demosaicing import Demosaic, Demosaic_NN
from filters import Filter_WM
from noise import Noise

if __name__ == "__main__":
    parser = optparse.OptionParser()
    group1 = optparse.OptionGroup(parser, "General")
    group1.add_option("--i", help="image file path", 
        default=".\\assets\\edinburgh.png", action="store", type=str)
    group1.add_option("--v", help="visualise?", action="store_true")
    group1.add_option("--s", help="save?", action="store_true")
    parser.add_option_group(group1)

    group2 = optparse.OptionGroup(parser, "Demosaicing")
    group1.add_option("--d", help="do demosaicing?", action="store_true")
    group2.add_option("--dp", help="bayer pattern", default='rggb', 
        action="store", type=str)
    group2.add_option("--drw", help="R channel weight", default=0.3, 
        action="store", type=float)
    group2.add_option("--dgw", help="G channel weight", default=0.59, 
        action="store", type=float)
    group2.add_option("--dbw", help="B channel weight", default=0.11, 
        action="store", type=float)
    parser.add_option_group(group2)

    group2 = optparse.OptionGroup(parser, "Auto white balance")
    group2.add_option("--a", help="do awb?", action="store_true")
    group2.add_option("--aws", help="input working space", default="srgb", 
        action="store", type=str)
    group2.add_option("--ail", help="input illuminant", default="D65", 
        action="store", type=str)
    group2.add_option("--aat", help="adaptive transform", default="vonKries", 
        action="store", type=str)

    group4 = optparse.OptionGroup(parser, "Filtering")   
    group4.add_option("--f", help="do filtering?", action="store_true")
    group4.add_option("--fk", help="kernel file path (.ini)", 
        default=".\\etc\\kernel_CWM3x3_1.ini", action="store", type=str)
    parser.add_option_group(group4)

    group5 = optparse.OptionGroup(parser, "Noise")
    group5.add_option("--n", help="add noise?", action="store_true")   
    group5.add_option("--nspq", help="Quantity of salt and pepper noise " + 
        " to apply to image as a fraction", default=0.001, action="store", 
        type=float)
    group5.add_option("--nspf", help="Fraction of salt to pepper noise", 
        default=0.5, action="store", type=float)
    parser.add_option_group(group5)

    options, args = parser.parse_args()

    outputs = []
    outputs_titles = []

    # Read image and kernel.
    #
    im_in = cv2.imread(options.i, cv2.IMREAD_GRAYSCALE)
    outputs.append(np.copy(im_in))
    outputs_titles.append("input")

    # Add some noise, if requesed.
    #
    if options.n:
        im_in = Noise.add_snp(im_in, options.nspf, options.nspq)
        outputs.append(np.copy(im_in))
        outputs_titles.append("added noise")

    # Demosaic image, if requested.
    #
    if options.d:
        dem = Demosaic_NN(options.dp)
        im_BGR = dem.run(im_in)
        outputs.append(np.copy(im_BGR))
        outputs_titles.append("demosaiced")
    else:
        im_BGR = im_in

    # Perform AWB, if requested.
    #
    if options.a:
        awb = AWB_GrayWorld(options.aws, options.ail, options.aat)
        im_BGR = awb.run(im_BGR)
        outputs.append(np.copy(im_BGR))
        outputs_titles.append("auto white balanced")  

    # Convert to mono by taking a weighting of the channels.
    #
    im_mono = Convert.bgr2mono(
        im_BGR, R_weight=options.drw, G_weight=options.dgw, B_weight=options.dbw)
    outputs.append(np.copy(im_mono))
    outputs_titles.append("monochromed")

    # Remove s&p noise from image, if requested.
    #
    if options.f:
        kernel = np.loadtxt(options.fk, delimiter=',')
        wm = Filter_WM(kernel)
        im_filtered = wm.run(im_mono)
        outputs.append(np.copy(im_filtered))
        outputs_titles.append("filtered")

    # Display?
    #
    if options.v:
        for output, title in zip(outputs, outputs_titles):
            cv2.imshow(title, output)
        cv2.waitKey(0)

    # Save?
    #
    if options.s:
        for output, title in zip(outputs, outputs_titles):
            cv2.imwrite(".\\tmp\\" + title + ".png", output)



