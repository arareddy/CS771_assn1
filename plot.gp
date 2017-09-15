set yrange [0.75 to 0.9]
set xrange [0:20]
plot 'accuracies.dat' using 1:2 with lines notitle
set term png
set output "plot.png"
replot
