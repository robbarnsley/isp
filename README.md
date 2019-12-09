# tasks
 
## Overview
 
**2. Describe a simple algorithm to remove salt and pepper noise from a monochrome image.**`

The work for this can be found in `filters.py`. The included routines perform a simple convolution of a median or weighted median kernel over the input.

**3. Describe a simple method to perform auto white balance (AWB) on an RGB image.**

The work for this can be found in `awb.py`. The included routines perform white balance assuming the Gray world approximation. The von Kries transform method is used to scale the LMS response to a specified illuminant (a CIE D65 illuminant is used as an example).

**4. Describe a simple method to convert a Bayer image directly into monochrome.**

The work for this can be found in both `demosaicing.py` and `convert.py`. The included routines perform simple NN interpolation and weighted averaging.

## Dependencies

- numpy
- cv2

## Execution

Each class (`filters.py`, `awb.py` and `demosaicing.py`) has its own main function. Arguments can be found by specifying the `--help` flag on the command line.

There is also a script `run_test.py`, which can execute each processing stage sequentially, e.g.

`$ python run_test.py --n --d --a --f --v`

Input (credit: https://github.com/codeplaysoftware/visioncpp/):

![input](/img/input.png)

Adding artificial s&p noise:
 
![added_noise](/img/added_noise.png)

Demosaiced to RGB:
 
![demosaiced](/img/demosaiced.png)

Auto-white balanced:

![awb](/img/auto_white_balanced.png)

Converted from RGB to monochrome:
 
![monochromed](/img/monochromed.png)

Median filtered:
 
![filtered](/img/filtered.png)







