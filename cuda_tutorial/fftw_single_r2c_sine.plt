# gnuplot> load "fftw_single_r2c_sine.plt"
# written by JK, 2016-06-18

# to generate a postscript file
# set term postscript
# set output "fftw_single_r2c_sine.ps"
# load "fftw_single_r2c_sine.plt"

set multiplot
set size 1.0,0.5

set origin 0.0,0.5
set xrange [0:63]
set yrange [-1.0:1.0]
plot 'fftw_single_r2c_sine_in.dat' using 1:2 with points pointtype 5

set origin 0.0,0.0
set xrange [0:32]
set yrange [-35:35]
plot 'fftw_single_r2c_sine_out.dat' using 1:3 with points pointtype 5

unset multiplot
