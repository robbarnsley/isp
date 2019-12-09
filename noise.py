import numpy as np

class Noise():
    def __init__(self):
        pass

    @staticmethod
    def add_snp(im, fraction_s_v_p, quantity, upperValue=255, 
        lowerValue=0):
        # Add salt.
        #
        num_s = np.ceil(quantity * im.size * fraction_s_v_p)
        coords = [np.random.randint(0, i - 1, int(num_s)) for i in im.shape]
        im[tuple(coords)] = upperValue

        # Add pepper.
        #
        num_p = np.ceil(quantity * im.size * fraction_s_v_p)
        coords = [np.random.randint(0, i - 1, int(num_p)) for i in im.shape]
        im[tuple(coords)] = lowerValue

        return im

