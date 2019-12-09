import argparse
import json 

import numpy as np
import cv2

class AWB():
    def __init__(self, inputWorkingSpace, inputIlluminant, 
    inputAdaptationTransform):
        self.inputWorkingSpace = inputWorkingSpace
        self.inputIlluminant = inputIlluminant
        self.inputAdaptationTransform = inputAdaptationTransform    

    def _xy2XYZ(self, xy, Y):
        ''' Convert from xy chromaticities to XYZ working space. '''
        X = (Y/xy[1])*xy[0]
        Z = (Y/xy[1])*(1-xy[0]-xy[1])
        return (X, Y, Z)

    def _XYZ2xy(self, XYZ):
        ''' Convert from an XYZ working space to xy chromaticities. '''
        x = XYZ[0] / np.sum(XYZ)
        y = XYZ[1] / np.sum(XYZ)
        return (x, y)
    
    def run(self):
        pass

class AWB_GrayWorld(AWB):
    ''' Perform auto white balance using the Gray world assumption. '''
    def __init__(self, inputWorkingSpace, inputIlluminant, 
    inputAdaptationTransform):
        super().__init__(inputWorkingSpace, inputIlluminant, 
        inputAdaptationTransform)

    def run(self, data8):
        # Sanity check for data.
        #
        try:
            assert len(data8.shape) == 3
        except AssertionError:
            raise Exception("Image does not have correct number of channels. ")
            exit(0)


        if self.inputWorkingSpace == 'srgb':
            # Define transform to convert from RGB to XYZ working space.
            # (http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html)
            #
            m_sRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], 
                                 [0.2126729, 0.7151522, 0.0721750], 
                                 [0.0193339, 0.1191920, 0.9503041]])

            # Retrieve XYZ for reference illuminant.
            #
            XYZ_illuminant = []
            with open("etc\\illuminants.json", 'r') as f:
                illuminants = json.load(f)
                for i in illuminants:
                    if i['title'] == self.inputIlluminant:
                        XYZ_illuminant = np.array(eval(i['M']))
                        break
            if len(XYZ_illuminant) == 0:
                raise Exception("Unable to find illuminant in file. ")

            # Retrieve XYZ to LMS colour adapatation matrix.
            #
            m_XYS2LMS= []
            with open("etc\\CAM.json", 'r') as f:
                CAMs = json.load(f)
                for c in CAMs:
                    if c['title'] == self.inputAdaptationTransform:
                        m_XYS2LMS = np.array(eval(c['M']))
                        break
            if len(m_XYS2LMS) == 0:
                raise Exception("Unable to find CAM in file. ")

            # Tranpose and switch axes so format changes from [y, x, channel] 
            # to [channel, y, x]. 
            #
            RGB = np.swapaxes(data8.T, 1, 2)
            RGB = np.reshape(RGB, newshape=(3, data8.T.shape[1]*data8.T.shape[2], 1))

            # Find the "grayest" reference colour in the sRGB image. This can 
            # be done by averaging over the first axis of the reshaped matrix.
            #
            RGB_gray = np.mean(RGB, axis=1).flatten()

            # Convert the reference gray RGB value to the XYZ colour space, then 
            # find the corresponding xy chromaticities.
            #
            xy_gray = self._XYZ2xy(np.dot(m_sRGB2XYZ, RGB_gray))

            # We're using D65 illuminant, where Y has been normalised to 100.
            #
            XYZ_gray = self._xy2XYZ(xy_gray, 100)

            # Convert gray value from XYZ to LMS colour space using the von 
            # Kries transform method.
            #
            LMS_gray = np.dot(m_XYS2LMS, XYZ_gray)

            # Convert reference white from XYZ to LMS colour space using the von 
            # Kries transform method.
            #
            LMS_illuminant = np.dot(m_XYS2LMS, XYZ_illuminant)

            # Compute the diagonal gain matrix in tristimulus LMS colour space and 
            # the corresponding transform [transform] to convert a source XYZ to a
            # destination XYZ in the adapted colour space.
            #
            gains = np.diag(LMS_illuminant/LMS_gray)
            transform = np.dot(np.dot(gains, m_XYS2LMS), np.linalg.inv(m_XYS2LMS))

            # Apply this transform to the image XYZ and convert back to RGB.
            #
            XYZ = np.dot(data8, m_sRGB2XYZ)
            XYZ_transformed = np.dot(XYZ, transform)
            XYZ_transformed_norm = XYZ_transformed
            RGB_transformed_norm = np.dot(XYZ_transformed, np.linalg.inv(m_sRGB2XYZ))
            RGB_transformed = RGB_transformed_norm
            RGB_transformed_saturated = \
                RGB_transformed_norm[RGB_transformed_norm < 0] = 0
            RGB_transformed_saturated = \
                RGB_transformed_norm[RGB_transformed_norm > 254] = 254
                
            return RGB_transformed_norm.astype(np.uint8)
        else:
            raise Exception("unknown input working space.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="image file path", 
        default=".\\assets\\edinburgh_BGR.png", action="store", type=str)
    parser.add_argument("-ws", help="input working space", default="srgb", 
        action="store", type=str)
    parser.add_argument("-il", help="input illuminant", default="D65", 
        action="store", type=str)
    parser.add_argument("-at", help="adaptive transform", default="vonKries", 
        action="store", type=str)
    parser.add_argument("-v", help="visualise?", action="store_true")
    parser.add_argument("-s", help="save?", action="store_true")
    args = parser.parse_args()

    # Read image and kernel.
    #
    im = cv2.imread(args.i, cv2.IMREAD_UNCHANGED)

    # Run AWB on image.
    #
    awb = AWB_GrayWorld(args.ws, args.il, args.at)
    im_awb = awb.run(im)

    # Display?
    #
    if args.v:
        cv2.imshow("input", im)
        cv2.imshow("balanced", im_awb)
        cv2.waitKey(0)

    # Save?
    #
    if args.s:
        cv2.imwrite("balanced.png", im_awb)
