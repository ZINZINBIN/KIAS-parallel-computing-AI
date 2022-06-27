# gnuplot> load "cufft_c2c_rect.plt"
# written by JK, 2016-06-18

# to generate a postscript file
# set term postscript
# set output "cufft_c2c_rect.ps"
# load "cufft_c2c_rect.plt"

set multiplot
set size 1.0,0.5

set origin 0.0,0.5
set xrange [0:63]
set yrange [0:1.2]
plot 'cufft_c2c_rect_in.dat' using 1:2 with points pointtype 5

set origin 0.0,0.0
set xrange [0:63]
set yrange [-5:35]
plot 'cufft_c2c_rect_out.dat' using 1:2 with points pointtype 5

unset multiplot
