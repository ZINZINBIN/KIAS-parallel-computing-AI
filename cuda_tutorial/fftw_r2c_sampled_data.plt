# gnuplot> load "fftw_r2c_sampled_data.plt"
# written by JK, 2016-06-18

# to generate a postscript file
# set term postscript
# set output "fftw_r2c_sampled_data.ps"
# load "fftw_r2c_sampled_data.plt"

set logscale y
set xlabel 'Frequency [kHz]'
set ylabel 'Linear Spectrum'
plot 'fftw_r2c_sampled_data_out.dat' using 1:2
