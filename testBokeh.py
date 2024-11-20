import bokeh.io as bio
import bokeh.plotting as bpl

p = bpl.figure()
p.circle([1,2,3],[4,5,6], color="orange")

bio.export_png(p,filename="ema.png")

#bpl.output_file("ema.html") 

#bpl.show(p)
