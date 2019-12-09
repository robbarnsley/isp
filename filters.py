import argparse

import numpy as np
import cv2

class Filter():
    def __init__(self, kernel):
        try:
            assert kernel.shape[1] % 2 == 1
            assert kernel.shape[0] % 2 == 1
        except AssertionError:
            raise Exception("Kernel dimensions must be odd. ")
        self.kernel = np.array(kernel)

    def run(self):
        pass        

class Filter_WM(Filter):
    ''' A weighted median filter for monochrome images. '''
    def __init__(self, kernel):
        super().__init__(kernel)

    def run(self, data8):
        ''' Apply filter to 8-bit image data, [data8]. '''
        # Sanity check for data.
        #
        try:
            assert len(data8.shape) == 2
        except AssertionError:
            raise Exception("Image does not have 2 channels. ")

        # Convert data to 16bit datatype, to allow sign.
        #
        data16 = data8.astype(np.int16)

        # Zero pad array so filter can work to edges. 
        #  
        # Assign as -1 so padding values can be distinguished from true 0s in 
        # image later.
        #
        paddingValue_x = (self.kernel.shape[1]-1)//2
        paddingValue_y = (self.kernel.shape[0]-1)//2 
        data16_padded = np.pad(
            data16, ((paddingValue_y,),(paddingValue_x,)), 
            mode='constant', constant_values=((-1,),(-1,)))
        
        # Consider each pixel in image and create an aperture of size 
        # [self.kernel] centred on this pixel.
        #
        data16_filtered = np.zeros(shape=data8.shape)
        for idx_y in range(paddingValue_y, data8.shape[0]-paddingValue_y):
            for idx_x in range(paddingValue_x, data8.shape[1]-paddingValue_x):
                window_x_lo = idx_x - paddingValue_x 
                window_x_hi = idx_x + paddingValue_x + 1
                window_y_lo = idx_y - paddingValue_y
                window_y_hi = idx_y + paddingValue_y + 1
                data16_window = data16_padded[
                    window_y_lo:window_y_hi, window_x_lo:window_x_hi]

                # Truncate padding, if applicable.
                #
                data16_window_noPad = data16_window[data16_window >= 0]
                kernel_noPad = self.kernel[data16_window >= 0]

                # Generate the list of values to use for the weighted median.
                #
                valuesToMedian = []
                for idx_k, k in enumerate(kernel_noPad):
                    for i in range(int(k)):
                        valuesToMedian.append(data16_window_noPad[idx_k])

                # And assign.
                #
                data16_filtered[idx_y, idx_x] = int(
                    np.median(valuesToMedian))

        # Convert back to native uint8.
        #
        data8_filtered = data16_filtered.astype(np.uint8)
        
        return data8_filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="image file path", 
        default=".\\assets\\edinburgh_mono_s&p.png", action="store", type=str)
    parser.add_argument("-k", help="kernel file path (.ini)", 
        default=".\\etc\\kernel_CWM3x3_1.ini", action="store", type=str)
    parser.add_argument("-v", help="visualise?", action="store_true")
    parser.add_argument("-s", help="save?", action="store_true")
    args = parser.parse_args()

    # Read image and kernel.
    #
    data8 = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
    kernel = np.loadtxt(args.k, delimiter=',')

    wm = Filter_WM(kernel)
    im_filtered = wm.run(data8)

    # Display?
    #
    if args.v:
        cv2.imshow("input", data8)
        cv2.imshow("filtered", im_filtered)
        cv2.waitKey(0)

    # Save?
    #
    if args.s:
        cv2.imwrite("filtered.png", im_filtered)