
class Rectangle1D:
    def __init__(self, xlim, tlim = None):
        self.x_ub = xlim[0]
        self.x_lb = xlim[1]
        if tlim is not None:
            self.t0 = tlim[0]
            self.tmax = tlim[1]

class Rectangle2D(Rectangle1D):
    def __init__(self, xlim, ylim, tlim = None):
        super().__init__(self, xlim, tlim = tlim)
        self.y_ub = ylim[0]
        self.y_lb = ylim[1]


class Rectangle3D(Rectangle2D):
    def __init__(self, xlim, ylim, zlim, tlim = None):
        super().__init__(self, xlim, ylim, tlim = tlim)
        self.z_ub = zlim[0]
        self.z_lb = zlim[1]

class DomainND:
    def __init__(self, vals, fidel):
        self.bounds = vals
        self.fidel = fidel
    
