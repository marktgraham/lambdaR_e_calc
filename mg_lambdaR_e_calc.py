__author__ = 'graham'

import numpy as np
import matplotlib.pyplot as plt
from cap_display_pixels import display_pixels
import matplotlib.patches as patches

'''
    Copyright (C) 2018, Mark T. Graham
    E-mail: mark.graham@physics.ox.ac.uk
    
    If you have found this software useful for your research,
    I would appreciate a citation for Graham et al (2018).

    See example at the bottom for usage instructions.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''


def lambdaR_e_calc(xpix, ypix, flux, vel_pix, disp_pix, eff_rad, ellip, phot_ang,
                  sig_psf=0., n=2., sigma_e=False, plot=False, vmin=None, vmax=None, dmin=None, dmax=None):
    '''
    Given 2D kinematic data, this routine calculates a value for the luminosity-weighted
    stellar angular momentum parameter lambdaR within one effective radius
    (see Equation (2) from Graham et al. 2018 = G18). Plotting requires the routine display_pixels available
    from Michele Cappellari's webpage: http://purl.org/cappellari/software

    The required quanitites are:
    *   xpix:       Array of pixel x-coordinates, spatial scale should be arcsec.
    *   ypix:       Array of pixel y-coordinates.
    *   flux:       Array of pixel flux.
    *   vel_pix:    Array of pixel velocity (should be corrected to the systemic velocity).
    *   disp_pix:   Array of pixel velocity dispersion (must be corrected for the instrumental dispersion).
    *   eff_rad:    Circular effective radius in arcsec.
    *   ellip:      Ellipticity of the half-light ellipse = 1 - axis ratio.
    *   phot_ang:   Photometric major axis East of North (E = 90 degrees = 9 o'clock on the sky).

    Optional:
    *   sigma_e:    If True, will calculate the effective velocity dispersion within the half-light ellipse
                    (see Equation (3) from G18).

    #################################################################################################

    Optional PSF Correction

    lambdaR_e is affected by smearing of the velocity field due to the finite PSF. In Graham et al. 2018,
    we presented an analytic correction to account for this effect (see Subsection 3.8 and Appendix C).
    The correction should be applied for regular rotators where the semi major axis is larger than the
    dispersion of the PSF (sig_PSF) and the Sersic index falls within the range 0.5 < n < 6.5.

    *   sig_psf:    Dispersion of the PSF (= FWHM/2.355, assumed to be a Gaussian).
                    Default = 0 (i.e. no correction)
    *   n:          Sersic index.
                    Default = 2

    #################################################################################################

    Plotting commands:
    *   plot:       If true, the routine will plot maps of the velocity, velocity dispersion and flux
                    indicating which pixels were included in the calculation.
                    Default = False
    *   vmin:       Minimum velocity for plotting.
    *   vmax:       Maximum velocity for plotting.
    *   dmin:       Minimum velocity dispersion for plotting.
    *   dmax:       Maximum velocity dispersion for plotting.

    #################################################################################################

    Returns:
    *   lambdaR:    Returns lambdaR_e within the half-light ellipse.
    *   error:      Returns the error calculated using the equations given in Figure C4 of G18.
    *   frac:       Returns the fraction of pixels within the half-light ellipse with disp_pix = 0
                    (See Subsection 3.6 and Appendix B of G18).
    *   sigma_e:    If sigma_e option is True, then sigma_e is returned.

    '''

    xpix, ypix, flux, vel_pix, disp_pix = np.ravel(xpix), np.ravel(ypix), np.ravel(flux), np.ravel(vel_pix), np.ravel(disp_pix)

    # Rotate kinematic data to lie along the major axis
    theta = np.radians(90-phot_ang)
    xpix_rot = xpix * np.cos(theta) - ypix * np.sin(theta)
    ypix_rot = xpix * np.sin(theta) + ypix * np.cos(theta)

    # Select pixels contained wth half-light ellipse
    w = (xpix_rot ** 2 * (1-ellip) + ypix_rot ** 2 / (1-ellip) < eff_rad ** 2)

    # Calculate fraction of half-light ellipse covered by pixels
    try:
        frac = (vel_pix.size - sum(np.isnan(vel_pix)))/sum(w)

    except ZeroDivisionError:
        frac = 1

    if frac < 0.85:
        print('Warning: data covers less than 85% of the half-light ellipse')

    r = np.sqrt(xpix ** 2 + ypix ** 2)                                      # radius vector

    # Calculate lambdaR_e, Equation 2, G18
    try:
        num = flux[w] * r[w] * abs(vel_pix[w])                                  # numerator of lambdaR
        denom = flux[w] * r[w] * np.sqrt(vel_pix[w] ** 2 + disp_pix[w] ** 2)    # denominator of lambdaR
    except IndexError:
        print('xpix:', xpix.shape, 'ypix:', ypix.shape, 'flux:', flux.shape, 'vel_pix:', vel_pix.shape, 'disp_pix:', disp_pix.shape)
        print('Check that input sizes match: all quantities should have sizes equal to the number of pixels')

    try:
        lambdaR = np.around(sum(num)/sum(denom), 3)
    except ZeroDivisionError:
        print('Denominator == 0')
        lambdaR = -999.

    # Measure fraction of pixels with sigma=0
    try:
        w1 = disp_pix[w] == 0
        frac = round(sum(w1) / len(w1), 3)
    except ZeroDivisionError:
        frac = 0

    # Calculate beam correction
    semi_maj_axis = eff_rad/np.sqrt(1-ellip)            # Semi-major axis
    sig_psf_re_ratio = sig_psf/semi_maj_axis            # Ratio between sig_PSF and semi-major axis

    if sig_psf_re_ratio > 1:
        print('Semi-major axis is smaller than sig_PSF: not correcting')
    if (n < 0.5) | (n > 6.5):
        print('Sersic index is outside the range 0.5 < n < 6.5: not correcting')

    # Equation 5, G18
    lambdaR_true = np.around(lambdaR * (1 + (n - 2) * (0.26 * sig_psf_re_ratio)) * (1 + (sig_psf_re_ratio / 0.47) ** 1.76) ** 0.84, 3)

    if lambdaR_true > 1:
        print('Warning: corrected value of lambdaR_e is greater than 1')

    error = np.around([-0.08*n*sig_psf_re_ratio, 0.03*sig_psf_re_ratio], 3)           # Figure C4, G18

    if lambdaR_true + error[1] < lambdaR:
        error[1] = lambdaR - lambdaR_true

    lambdaR = "{0:.4f}".format(lambdaR_true)

    if sigma_e:
        try:
            num = flux[w] * (vel_pix[w] ** 2 + disp_pix[w] ** 2)  # numerator of sigma_e
            denom = flux[w]    # denominator of sigma_e
        except IndexError:
            print('xpix:', xpix.shape, 'ypix:', ypix.shape, 'flux:', flux.shape, 'vel_pix:', vel_pix.shape, 'disp_pix:', disp_pix.shape)
            print('Check that input sizes match: all quantities should have sizes equal to the number of pixels')

        try:
            sigma_e = "{0:.1f}".format(np.sqrt(sum(num)/sum(denom)))
        except ZeroDivisionError:
            print('Denominator == 0')
            sigma_e = -999

    if plot:
        # Plot velocity, velocity dispersion and flux
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        ax = plt.gca()

        if not vmax:
            vmax = max(vel_pix)
        if not vmin:
            vmin = min(vel_pix)

        display_pixels(xpix, ypix, vel_pix, vmin=vmin, vmax=vmax, alpha=0.2)
        im = display_pixels(xpix[w], ypix[w], vel_pix[w], vmin=vmin, vmax=vmax)

        ellipse1 = patches.Ellipse(xy=(0, 0), width=2 * eff_rad * np.sqrt(1 - ellip), fill=False,
                                   height=2 * eff_rad / np.sqrt(1 - ellip), angle=phot_ang,
                                   color='red', linewidth=3)
        ax.add_patch(ellipse1)

        ellipse2 = patches.Ellipse(xy=(0, 0), width=2 * sig_psf, fill=False,
                                   height=2 * sig_psf, angle=0,
                                   color='grey', linewidth=3)
        ax.add_patch(ellipse2)

        ax.set_xlim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        ax.set_ylim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        cbar = plt.colorbar(im[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('km/s', rotation=270, fontsize=18)
        ax.set_title('Velocity')

        plt.subplot(1, 3, 2)
        ax = plt.gca()

        if not dmax:
            dmax = max(disp_pix)
        if not dmin:
            dmin = min(disp_pix)

        display_pixels(xpix, ypix, disp_pix, vmin=dmin, vmax=dmax, alpha=0.2)
        im = display_pixels(xpix[w], ypix[w], disp_pix[w], vmin=dmin, vmax=dmax)

        ellipse1 = patches.Ellipse(xy=(0, 0), width=2 * eff_rad * np.sqrt(1 - ellip), fill=False,
                                   height=2 * eff_rad / np.sqrt(1 - ellip), angle=phot_ang,
                                   color='red', linewidth=3)
        ax.add_patch(ellipse1)

        ellipse2 = patches.Ellipse(xy=(0, 0), width=2 * sig_psf, fill=False,
                                   height=2 * sig_psf, angle=0,
                                   color='grey', linewidth=3)
        ax.add_patch(ellipse2)

        ax.set_xlim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        ax.set_ylim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        cbar = plt.colorbar(im[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('km/s', rotation=270, fontsize=18)
        ax.set_title('Velocity Dispersion')

        plt.subplot(1, 3, 3)
        ax = plt.gca()

        display_pixels(xpix, ypix, flux, alpha=0.2)
        im = display_pixels(xpix[w], ypix[w], flux[w])

        ellipse1 = patches.Ellipse(xy=(0, 0), width=2 * eff_rad * np.sqrt(1 - ellip), fill=False,
                                   height=2 * eff_rad / np.sqrt(1 - ellip), angle=phot_ang,
                                   color='red', linewidth=3)
        ax.add_patch(ellipse1)

        ellipse2 = patches.Ellipse(xy=(0, 0), width=2 * sig_psf, fill=False,
                                   height=2 * sig_psf, angle=0,
                                   color='grey', linewidth=3)
        ax.add_patch(ellipse2)

        ax.set_xlim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        ax.set_ylim(-1.1 * eff_rad / np.sqrt(1 - ellip), 1.1 * eff_rad / np.sqrt(1 - ellip))
        plt.colorbar(im[0], ax=ax, fraction=0.046, pad=0.04)
        plt.title('Flux')
        plt.suptitle('$\lambda_{R_e}$ = ' + str(lambdaR), fontsize=24)
        plt.tight_layout(h_pad=1.0)
        plt.show()

    if sigma_e:
        return lambdaR, error, frac, sigma_e
    else:
        return lambdaR, error, frac


def lambdaR_e_correct(lam_obs, semi_maj_axis, sig_psf, n, plot=False):
    '''
    Given observed values of lambda_r_e and the ratio between sigma_PSF and the semi-major axis,
    the function returns the corrected values of lambdaR_e. Accepts single values or arrays.

    The required quanitites are:
    *   lam_obs:        Array/single value of observed lambdaR_e.
    *   semi_maj_axis:  Array/single value of semi-major axis.
    *   sig_psf:        Array/single value of the dispersion of the PSF (= FWHM/2.355)
    *   n:              Array/single value for the Sersic index

    Returns:
    *   lam_true:       Array/single value of corrected lambdaR_e.
    *   error_low:      The error in the negative direction.
    *   error_high:     The error in the positive direction.

    Plotting:
    If plot=True, the routine plots a graphic visualising the shift in lambdaR_e due to the correction.
    Indicates galaxies where sigma_PSF > semi_maj_axis (not corrected) and where lam_true > 1 (corrected)

    '''

    if (np.min(n) < 0.5) | (np.max(n) > 6.5):
        print('Sersic index contains values outside the range 0.5 < n < 6.5')

    sig_psf_re_ratio = sig_psf/semi_maj_axis

    lam_true = np.around(lam_obs * (1 + (n - 2) * (0.26 * sig_psf_re_ratio)) * (1 + (sig_psf_re_ratio / 0.47) ** 1.76) ** 0.84, 3)

    w, w1, w2 = (sig_psf_re_ratio <= 1) & (lam_true <= 1), sig_psf_re_ratio > 1, lam_true > 1

    if (isinstance(lam_obs, type(np.array([])))) & plot:
        plt.figure(figsize=(4, 4))

        plt.plot(np.linspace(0, 1, 2), np.linspace(1, 0, 2), color='r')
        for i in np.linspace(0.2, 0.8, 4):
            plt.plot(np.linspace(0, i, 2), np.linspace(i, 0, 2), color='k', linestyle='--')

        blue_point = plt.scatter(lam_obs[w], lam_true[w]-lam_obs[w], s=3, color='grey')
        red_star = plt.scatter(lam_obs[w1], lam_true[w1]-lam_obs[w1], s=30, facecolor='', edgecolor='r', marker='*')
        plt.fill([0, 1, 1, 0], [1, 1, 0, 1], color='0.7')
        purple_triangle = plt.scatter(lam_obs[w2], lam_true[w2]-lam_obs[w2], s=30, facecolor='', edgecolor='purple', marker='^')
        plt.xlabel('$\lambda_{obs}$', fontsize=18)
        plt.ylabel('$\Delta \lambda$', fontsize=18)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        line = plt.axhline(np.mean(lam_true[w]-lam_obs[w]))

        p1 = patches.Rectangle((0, 0), 1, 1, fc='0.7', ec='r')

        plt.legend((blue_point, red_star, purple_triangle, line),
           ('Corrected galaxies', 'Uncorrected as $\sigma/R_e>1$)',
            '$\lambda_{true}>1$',
            'Mean shift for corrected galaxies'),
           scatterpoints=1, bbox_to_anchor=(0.5, 0.95), loc='center', ncol=2, fontsize=12, frameon=False)

        plt.savefig('beam_correction', bbox_inches='tight', dpi=300)
        plt.close()
        # plt.show()

    if isinstance(lam_obs, type(np.array([]))):
        # Don't correct for lam_true > 1
        lam_true[w1] = lam_obs[w1]
        error_low = np.zeros_like(lam_true)
        error_high = np.zeros_like(lam_true)
        error_low[~w1] = -0.08*n[~w1]*sig_psf_re_ratio[~w1]
        error_high[~w1] = 0.03*sig_psf_re_ratio[~w1]

        w = lam_true + error_low < lam_obs
        error_low[w] = lam_obs[w] - lam_true[w]

        return np.around(lam_true, 3), np.around(error_low, 3), np.around(error_high, 3)

    if isinstance(lam_obs, type(np.float(0))):
        if w1:
            # Don't correct for lam_true > 1
            lam_true = lam_obs
            error_low, error_high = 0, 0
        else:
            error_low = -0.08 * n * sig_psf_re_ratio
            error_high = 0.03 * sig_psf_re_ratio

            if lam_true + error_low < lam_obs:
                error_low = lam_obs - lam_true

        return float('%.4f' % lam_true), float('%.4f' % error_low), float('%.4f' % error_high),


def test_lambdaR_e_calc():
    xpix, ypix, flux, vel_pix, disp_pix = np.genfromtxt('../../Publications (LaTeX)/MG Paper I/7957-6103.txt').T

    reff_final, eps, astro_pa, n, psf = 8.33, 0.124, 94.7, 4.812, 1.058     # Taken from Table 2. of G18 for 7957-6103

    lambdaR_e_calc(xpix, ypix, flux, vel_pix, disp_pix, reff_final, eps, astro_pa, plot=True)
    lambdaR_e_calc(xpix, ypix, flux, vel_pix, disp_pix, reff_final, eps, astro_pa, n=n, sig_psf=psf, plot=True)
    plt.show()


def test_lambdaR_e_correct(sma=10, lambda_reg=0.5):
    lam_0 = lambdaR_e_correct(0.1, sma, 0, 2)

    sersic_array = np.linspace(1, 6, 6)

    lam_1 = lambdaR_e_correct(lam_obs=np.full_like(sersic_array, lambda_reg),
                                         semi_maj_axis=np.full_like(sersic_array, sma),
                                         sig_psf=np.full_like(sersic_array, 2.5 / 2.355),
                                         n=sersic_array)
    lam_1_value = lam_1[0]
    lam_1_error_low = lam_1[1]
    lam_1_error_high = lam_1[2]

    lam_2 = lambdaR_e_correct(lambda_reg, 1, 2.5 / 2.355, 3)

    print('Semi-major axis = %.1f"' % sma)
    print('Non-Regular Slow Rotator (no correction required): lambda_obs=0.1, SMA=%.1f"' % sma)
    print('lambda_true = %s, error = [%s, %s]' % (lam_0[0], lam_0[1], lam_0[2]))
    print('###########################################################################')
    print('Effect of Sersic Index:')
    print('Regular Fast Rotator: lambda_obs = %s, SMA=%s", FWHM=2.5"' % (lambda_reg, sma))
    print('###########################################################################')
    print('n             #       1 #       2 #       3 #       4 #       5 #       6 #')
    print('lambda_true   #  %.4f #  %.4f #  %.4f #  %.4f #  %.4f #  %.4f #' %
          (lam_1[0][0], lam_1[0][1], lam_1[0][2], lam_1[0][3], lam_1[0][4], lam_1[0][5]))
    print('error_lower   # %.4f # %.4f # %.4f # %.4f # %.4f # %.4f #' %
          (lam_1[1][0], lam_1[1][1], lam_1[1][2], lam_1[1][3], lam_1[1][4], lam_1[1][5]))
    print('error_high    #  %.4f #  %.4f #  %.4f #  %.4f #  %.4f #  %.4f #' %
          (lam_1[2][0], lam_1[2][1], lam_1[2][2], lam_1[2][3], lam_1[2][4], lam_1[2][5]))
    print('###########################################################################')

    plt.errorbar(sersic_array,          # Sersic index
                 lam_1_value,                   # lambda_true
                 yerr=[-lam_1_error_low,        # lower limit
                       lam_1_error_high],       # upper limit
                 fmt='o', ecolor='grey', capsize=2, capthick=2)
    plt.axhline(y=lambda_reg, linestyle='--')
    plt.xlabel('Sersic Index, $n$')
    plt.ylabel('$\lambda_{R_e}$', fontsize=16)
    plt.ylim(0, 1)
    print('Regular Fast Rotator where SMA < sig_PSF: lambda_obs = %s, SMA=1", FWHM=2.5", n=3' % lambda_reg)
    print('lambda_true = %s, error = [%s, %s]' % (lam_2[0], lam_2[1], lam_2[2]))
    plt.show()


if __name__ == '__main__':

    test_lambdaR_e_calc()

    test_lambdaR_e_correct()
