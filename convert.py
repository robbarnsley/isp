import numpy as np

class Convert():
    def __init__(self):
        pass

    @staticmethod
    def bgr2mono(data8, R_weight=1/3, G_weight=1/3, B_weight=1/3):
        ''' Convert BGR to mono with component channel weighting. '''
        # Sanity check for data.
        #
        try:
            assert len(data8.shape) == 3
        except AssertionError:
            raise Exception("Image does not have 3 channels. ")
        B_weighted = (data8[:,:,0]*B_weight)
        G_weighted = (data8[:,:,1]*G_weight)
        R_weighted = (data8[:,:,2]*R_weight) 
        
        return (B_weighted + G_weighted + R_weighted).astype(np.uint8)


