#!/usr/bin/env python

__version__ = "Time-stamp: <2010-12-27 17:42 yannick@lyopc469>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
"""

import numpy as NP

class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refsample):
        """refsample is the reference (data) sample to be compared to."""

        self.ref = NP.asarray(refsample)

    def setup_axes(self, fig, rect=111):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes.

        Wouldn't the ideal be to define its own non-linear
        transformation, so that coordinates are directly r=stddev and
        theta=correlation? I guess it would allow 
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.concatenate((NP.arange(10)/10.,[0.95,0.99]))
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,NP.pi/2, # 1st quadrant
                                                     0,1.5*self.ref.std()),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)         # Useless
        
        # Grid
        ax.grid()

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        print "Reference std:", self.ref.std()
        self.ax.plot([0],self.ref.std(),'ko', label='_')
        t = NP.linspace(0,NP.pi/2)
        r = NP.zeros_like(t) + self.ref.std()
        self.ax.plot(t,r,'k--', label='_')

        return self.ax

    def get_coords(self, sample):
        """Computes theta=arccos(correlation),rad=stddev of sample
        wrt. reference sample."""

        std = NP.std(sample)
        corr = NP.corrcoef(self.ref, sample) # [[1,rho],[rho,1]]
        theta = NP.arccos(corr[0,1])

        print "Sample std,rho:",std,corr[0,1]

        return theta,std

    def plot_sample(self, sample, *args, **kwargs):
        """Add sample to the Taylor diagram. args and kwargs are
        directly propagated to the plot command."""

        t,r = self.get_coords(sample)
        l, = self.ax.plot(t,r, *args, **kwargs) # (theta,radius)

        return l


if __name__=='__main__':

    import matplotlib.pyplot as PLT

    x = NP.linspace(0,4*NP.pi,100)
    data = NP.sin(x)                           # Data
    m1 = data + 0.2*NP.random.randn(len(x))    # Model 1
    m2 = 0.8*data + .1*NP.random.randn(len(x)) # Model 2
    m3 = NP.sin(x-NP.pi/10)                    # Model 3

    dia = TaylorDiagram(data)
    
    fig = PLT.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1, xlabel='X', ylabel='Y')
    ax2 = dia.setup_axes(fig, 122)

    ax1.plot(x,data,'ko', label='Data')
    ax1.plot(x,m1,'b-', label='Model 1')
    ax1.plot(x,m2,'r-', label='Model 2')
    ax1.plot(x,m3,'g-', label='Model 3')

    dia.plot_sample(m1, 'bo')
    dia.plot_sample(m2, 'ro')
    dia.plot_sample(m3, 'go')
    
    ax1.legend(numpoints=1, prop=dict(size='small'), loc='best')

    PLT.show()
