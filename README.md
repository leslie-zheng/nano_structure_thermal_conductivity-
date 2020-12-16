# nano_structure_thermal_conductivity-

this simple script allows you to calculate the in plane and cross plane thermal conductivity in thin film based on the cumulative thermal conductivity VS. mean free path curve
of bulk materials, here the thickness of thin film is thick enough not the single layer or double layer thin film.

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
