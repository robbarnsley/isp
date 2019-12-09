import argparse

import numpy as np
import cv2

from convert import Convert

class Demosaic():
    def __init__(self, patternCode):
        self.patternCode = patternCode

        # Sanity check for Bayer pattern code.
        #
        try:
            assert len(self.patternCode) == 4
            assert 'r' in self.patternCode
            assert 'b' in self.patternCode
            assert self.patternCode.count('g') == 2
        except AssertionError:
            raise Exception("Not a valid Bayer pattern code. ")

        # Uniquify the G components.
        #
        # Required to separate interpolation logic between the two green mask 
        # components.
        #
        self.bayerPattern = []
        for idx, char in enumerate(patternCode):
            totalcount = patternCode.count(char)
            count = patternCode[:idx].count(char)
            self.bayerPattern.append(
                char + str(count + 1) if totalcount > 1 else char)
        self.bayerPattern = np.reshape(self.bayerPattern, newshape=(2,2))

    def run(self):
        pass

class Demosaic_NN(Demosaic):
    ''' Simple NN interpolated demosaicing. '''
    def __init__(self, patternCode):
        super().__init__(patternCode)

    def run(self, data8, display=True):
        ''' Apply demosaicing with bayer pattern [self.bayerPattern] to 
        data [data8].

        Returns 8-bit BGR array.
        '''
        # Sanity check for data.
        #
        try:
            assert len(data8.shape) == 2
        except AssertionError:
            raise Exception("Image does not have 2 channels. ")

        # Create a tiled array of the same size as [data8] with the repeated 
        # Bayer pattern, [self.bayerPattern].
        #
        pattern_size_x = data8.shape[1] if data8.shape[1] % 2 == 0 else \
            data8.shape[1] + 1
        pattern_size_y = data8.shape[0] if data8.shape[0] % 2 == 0 else \
            data8.shape[0] + 1
        bayerPatternTiled = np.tile(
            self.bayerPattern, (pattern_size_y//2, pattern_size_x//2))
        bayerPatternTiled = bayerPatternTiled if data8.shape[1] % 2 == 0 else \
            bayerPatternTiled[:,:-1]
        bayerPatternTiled = bayerPatternTiled if data8.shape[1] % 2 == 0 else \
            bayerPatternTiled[:-1,:]

        # Consider each pixel in turn, and interpolate for the two missing 
        # channels.
        #
        BGR = np.zeros(shape=(data8.shape[0], data8.shape[1], 3), 
            dtype=np.uint8)
        for idx_j, row in enumerate(data8):
            for idx_i, val in enumerate(row):
                char = bayerPatternTiled[idx_j][idx_i]
                if self.patternCode == 'rggb':
                    if char == 'r':
                        R = val

                        # .X.
                        # X.X
                        # .X.
                        G_arr = []
                        if idx_j > 0:
                            G_arr.append(data8[idx_j-1][idx_i])
                        if idx_i > 0:
                            G_arr.append(data8[idx_j][idx_i-1])
                        if idx_j < data8.shape[0]-1:
                            G_arr.append(data8[idx_j+1][idx_i])
                        if idx_i < data8.shape[1]-1:
                            G_arr.append(data8[idx_j][idx_i+1])
                        G = int(np.mean(G_arr))

                        # X.X
                        # ...
                        # X.X
                        B_arr = []
                        if idx_j > 0:
                            if idx_i > 0:
                                B_arr.append(data8[idx_j-1][idx_i-1])
                            if idx_i < data8.shape[1]-1:
                                B_arr.append(data8[idx_j-1][idx_i+1])  
                        if idx_j < data8.shape[0]-1:
                            if idx_i > 0:
                                B_arr.append(data8[idx_j+1][idx_i-1])
                            if idx_i < data8.shape[1]-1:
                                B_arr.append(data8[idx_j+1][idx_i+1])  
                        B = int(np.mean(B_arr))
                    elif char == 'g1':
                        G = val

                        # ...  
                        # X.X
                        # ...
                        R_arr = []
                        if idx_i > 0:
                            R_arr.append(data8[idx_j][idx_i-1])
                        if idx_i < data8.shape[1]-1:
                            R_arr.append(data8[idx_j][idx_i+1])
                        R = int(np.mean(R_arr))

                        # .X.  
                        # ...
                        # .X.
                        B_arr = []
                        if idx_j > 0:
                            B_arr.append(data8[idx_j-1][idx_i])
                        if idx_j < data8.shape[0]-1:
                            B_arr.append(data8[idx_j+1][idx_i])
                        B = int(np.mean(B_arr))
                    elif char == 'g2':
                        G = val

                        # ...  
                        # X.X
                        # ...
                        B_arr = []
                        if idx_i > 0:
                            B_arr.append(data8[idx_j][idx_i-1])
                        if idx_i < data8.shape[1]-1:
                            B_arr.append(data8[idx_j][idx_i+1])
                        B = int(np.mean(B_arr))

                        # .X.  
                        # ...
                        # .X.
                        R_arr = []
                        if idx_j > 0:
                            R_arr.append(data8[idx_j-1][idx_i])
                        if idx_j < data8.shape[0]-1:
                            R_arr.append(data8[idx_j+1][idx_i])
                        R = int(np.mean(R_arr))
                    elif char == 'b':
                        B = val

                        # .X.
                        # X.X
                        # .X.
                        G_arr = []
                        if idx_j > 0:
                            G_arr.append(data8[idx_j-1][idx_i])
                        if idx_i > 0:
                            G_arr.append(data8[idx_j][idx_i-1])
                        if idx_j < data8.shape[0]-1:
                            G_arr.append(data8[idx_j+1][idx_i])
                        if idx_i < data8.shape[1]-1:
                            G_arr.append(data8[idx_j][idx_i+1])
                        G = int(np.mean(G_arr))

                        # X.X
                        # ...
                        # X.X
                        R_arr = []
                        if idx_j > 0:
                            if idx_i > 0:
                                R_arr.append(data8[idx_j-1][idx_i-1])
                            if idx_i < data8.shape[1]-1:
                                R_arr.append(data8[idx_j-1][idx_i+1])  
                        if idx_j < data8.shape[0]-1:
                            if idx_i > 0:
                                R_arr.append(data8[idx_j+1][idx_i-1])
                            if idx_i < data8.shape[1]-1:
                                R_arr.append(data8[idx_j+1][idx_i+1])  
                        R = int(np.mean(R_arr))
                    BGR[idx_j,idx_i,0] = B
                    BGR[idx_j,idx_i,1] = G
                    BGR[idx_j,idx_i,2] = R
        return BGR
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="image file path", 
        default=".\\assets\\edinburgh.png", action="store")
    parser.add_argument("-p", help="bayer pattern", default='rggb', 
        action="store", type=str)
    parser.add_argument("-m", help="make mono?", action="store_true")
    parser.add_argument("-rw", help="R channel weight", default=0.3, 
        action="store", type=float)
    parser.add_argument("-gw", help="G channel weight", default=0.59, 
        action="store", type=float)
    parser.add_argument("-bw", help="B channel weight", default=0.11, 
        action="store", type=float)
    parser.add_argument("-v", help="visualise?", action="store_true")
    parser.add_argument("-s", help="save?", action="store_true")
    args = parser.parse_args()

    # Read image and kernel.
    #
    im = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)

    # Instantiate Demosaic_NN with the given bayer pattern code [args.p], 
    # and run on the input bayer image. 
    #
    dem = Demosaic_NN(args.p)
    im_BGR = dem.run(im)

    if args.m:
        im_mono = Convert.bgr2mono(
            im_BGR, R_weight=args.rw, G_weight=args.gw, B_weight=args.bw)

    # Display?
    #
    if args.v:
        cv2.imshow("input", im)
        cv2.imshow("demosaiced", im_BGR)
        if args.m:
            cv2.imshow("mono", im_mono)
        cv2.waitKey(0)

    # Save?
    #
    if args.s:
        cv2.imwrite("demosaiced.png", im_BGR)

