#!/bin/python
#copyright Zheng, Jiongzhi @2020
"""
calculate nanostructure thermal conductivity using cumulative thermal
conductivity and mfp relation

minimal requirement: python-3.5, numpy-1.17 and scipy-1.4

example:
 calculate the thermal conductivity of a nanowire of diameter from 1 to 100
 micrometer, and then plot them. "cum_kappa.txt" stores the cumulative
 kappa (from 0 - 1) w.r.t mfp. the bulk thermal conductivity is 147 W/m.K::

 python nanokappa.py cum_kappa.txt --scale-factor 147 --structure nanowire --lmin 1 --lmax 100 --show

 calculate the thermal conductivity of a thin film of thickness from 1 to 10 micrometer, and then plot them.
 "cum_kappa.txt" stores the cumulative kappa (from 0 -147) VS. mfp.

 python nanokapa.py cum_kappa.txt --structure thin_film -- lmin 1 --lmax 10 --show

  cross plane usage:
  python nanokappa_cross_plane.py cumulative_100K.txt --lmin 0.01 --lmax 0.05 --structure=thin_film_cross_plane --show
  or
  python nanokappa_cross_plane.py zhi_test.csv --lmin 0.01 --lmax 0.05 --structure=thin_film_cross_plane --show

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Argument for nanokappa.py to calculate thermal '
                    'conductivity of nanostructures from cum_kappa & mfp'
                    'relation')
    parser.add_argument(
        'cum_kappa_file',
        type=str,
        help='The path of file that stores the cumulative thermal '
             'conductivity contributiuion (1) w.r.t mean free path (um)')
    parser.add_argument(
        '--scale-factor',
         type=float,
         default=1,
         help='The scale factor to multiply with cumulative thermal '
              'conductivtiy. If cum_kappa is normalized, this value is the total'
              'thermal conductivity.')
    parser.add_argument(
         '--structure',
          type=str,
          default='thin_film',
          help='The nano structure that determines the relationship between '
               'actual phonon mfp with the bulk mfp. Options are "nanowire",'
               '"thin_film," and "custum", "cross_plane_thin_film".')
    parser.add_argument(
          '--lmin',
           dest='min_general_length',
           type=float,
           default=1e-3,
           help='Minimun general length (um) to print the resukts. In case of'
                'nanowire, it is diameter for nanowire, thickness for thin film'
                'or the pre-defined length for the custom structure.')
    parser.add_argument(
           '--lmax',
            dest='max_general_length',
            type=float,
            default=1e2,
            help='Maximum general length (um) to print the results. In case of '
                 'nanowire, it is diameter for nanowire, thickness for thin film'
                 'or the pre-defined length for the custom structure.')
    parser.add_argument(
             '--num',
              type=int,
              default=50,
              help='The interval number of length points to interpolate between '
                   'lmin and lmax. ')
    parser.add_argument('--show', action='store_true',
                        help='Draw the kappa-length curve. ')
    parser.add_argument('--log', action='store_true',
                        help='Interpolate & draw the kappa-length curve '
                             ' with log scale.')
    args = parser.parse_args()
    return args


def diffuse_nanowire(kn):
    """
    The mfp shrinkage calculation for diffuse nanowire.

    .. math::
       B(Kn)=(1+Kn)^{-1}
    Args:
       Kn (float): The Knudsen number.
    :param kn:
    :return:
    """
    return 1 / (1 + kn)


def in_plane_thin_film(kn):
    """
    The mfp shrinkage calculation for in-plane thin film.

    .. math:
       B(Kn) = 1 - (3/8) Kn(1-4E_3 Kn^{-1}+4E_5 Kn^{-1})
       , where E_3 and E_5 are exponential integral polynomials.

    Args:
        kn (float): The knudsen number.
    :param kn:
    :return:
    """

    e3 = sc.expn(3, 1 / kn) # exponential integral E_3(1/kn)
    e5 = sc.expn(5, 1 / kn) # exponential integral E_5{1/kn}
    return 1 - 3. / 8. * kn * (1 - 4 * e3 + 4 * e5)

def cross_plane_thin_film(kn):
    """The mfp shrinkage calculation for cross-plane thin film.

    .. math::
       B(Kn)=1 + 3. * Kn(E_5 Kn^{-1} - 1. / 4.)
       where E_5 are exponential integral polynomials.

       Args:
           kn (float): The knudsen number.
    """
    e5 = sc.expn(5, 1 / kn)
    return 1 + 3. * kn * (e5 - 1. / 4.)

def custom_structure(kn):
    """
    The mfp shrinkage calculation for your own structure.

    you should implement this function on your own.

    Args:
      kn (float): The Knudsen number.
    :param kn:
    :return:
    """
    e5 = sc.expn(5, 1 / kn)
    return 1 + 3. * kn * (e5 - 1. / 4.)

def nano_thermal_conductivity(length, cum_kappa, structure_func):
    """
    Thermal conductivity calculation with the cumulative kappa w.r.t. mfp.

    This equation is given as Eq. (11) in 'F. Yang & C. Dames, PRB (2013)'

    .. math::
       \kappa = - \int_0 ^ \infty \kappa(MFP) dB

    Args:
        length (float): The general length (unit um) of the structure.
        cum_kappa (np.ndarray): the cumulative kappa w.r.t. mfp relation,
          It has shape (N, 2), where the first & second columns are mfp and cumulative kappa respectively.

        structure_func (func): The structure function that determines the mfp
        shrinkage function (a.k.a the value 'B' in the paper)
    :param length:
    :param cum_kappa:
    :param structure_func:
    :return: float: The calculated thermal conductivity of the given structure.
    """
    mfps = cum_kappa[:,0]
    kappas = cum_kappa[:,1]

    knudsen = mfps / length
    mfp_shrinkage_factor = structure_func(knudsen) # B_nano number
    # mfp_shrinkage is 1 when mfp = 0 and 0 when mfp=oo
    mfp_shrinkage_factor = np.insert(mfp_shrinkage_factor, 0, 1)
    mfp_shrinkage_factor = np.insert(
        mfp_shrinkage_factor, len(mfp_shrinkage_factor), 0)
    # cum_kappa is 0 when mfp = 0 and max(cum_kappa) when mfp=oo
    kappas = np.insert(kappas, 0, 0)
    kappas = np.insert(kappas, len(kappas), kappas.max())
    kappa = -np.trapz(kappas, x=mfp_shrinkage_factor)
    return kappa


def main():
    args = parse_args()
    structure = args.structure
    if structure == 'nanowire':
        structure_func = diffuse_nanowire
    elif structure == 'thin_film':
        structure_func = in_plane_thin_film
    elif structure == 'thin_film_cross_plane':
        structure_func = cross_plane_thin_film
    elif structure == 'custom':
        structure_func = custom_structure
    else:
        raise ValueError(f'Structure {structure} is unsupported')
    lmin = args.min_general_length
    lmax = args.max_general_length
    assert lmin > 0 and lmax > 0, (
        f'The length values can only be positive, but got lmin={lmin} and'
        f'lmax={lmax}')
    scale_factor = args.scale_factor
    assert scale_factor > 0, (
        f'scale_factor can only be positive, but got {scale_factor}')
    num_interpolate = args.num
    assert num_interpolate > 0, ('Number of interpolate points has to be'
                                 f'positive, but got {num_interpolate}')
    cum_kappa_file = args.cum_kappa_file
    assert os.path.isfile(cum_kappa_file), f'File {cum_kappa_file} dose not exist'
    #cum_kappa = np.loadtxt(cum_kappa_file, delimiter=None)
    cum_kappa = np.loadtxt(cum_kappa_file, delimiter=',')
    print (np.shape(cum_kappa))
    kappas = np.zeros(num_interpolate)

    if args.log:
        # logarithm scale
       lengths = np.logspace(
           np.log10(lmin), np.log10(lmax), num=num_interpolate)
    else:
        lengths=np.linspace(lmin, lmax, num=num_interpolate)

    for i, length in enumerate(lengths):
        kappa = nano_thermal_conductivity(length, cum_kappa, structure_func)
        kappas[i] = kappa * scale_factor

    print(f'{"Length(um)":15s}\t{"Kappa":10s}')
    for length, kappa in zip(lengths, kappas):
        print(f'{length: 10.3e}\t{kappa:10.3f}')

    if args.show:
        plt.plot(lengths, kappas)
        if args.log:
            plt.xscale('log')
        plt.xlabel('Length (um)')
        plt.ylabel('Thermal conductivitiy (W/m.K)')
        plt.show()


if __name__ == '__main__':
    main()
