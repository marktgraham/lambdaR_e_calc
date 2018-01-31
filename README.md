# lambdaR_e_calc

There are two functions provided:

lambdaR_e_calc: A function that calculates lambdaR_e from 2D stellar kinematic data. The beam correction is automatically performed "in house" provided the dispersion of the point source function (PSF) and the Sersic index are given. 

lambdaR_e_correct: A function that calculates the beam-corrected values from observed values of lambdaR_e. This function is useful for correcting a catalogue of values as arrays can be given.

Both these functions have been tested and give identical results within the numerical precision.

# Dependencies

* matplotlib
* numpy
* display_pixels, available [here](http://purl.org/cappellari/software)

# Example

There are two test functions with usage examples. A text file for a MaNGA galaxy is provided for test_lambdaR_e_calc.

# Comments

Any questions or comments can be addressed here or to mark.graham@physics.ox.ac.uk
