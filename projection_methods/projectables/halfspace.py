class Halfspace(Projectable):
    """A projectable halfspace

     Defines a halfpsace of the form
        \{ x | <a, x> \leq b \}.
     """
    def __init__(self, x, a, b):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
            a (array-like): normal vector
            b (array-like): offset vector
        """
        self.a = a
        self.b = b
        constr = [a.T * x <= b]
        super(Halfspace, self).__init__(x, constr)


    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += "a: " + str(self.a) + "\n"
        string += "b: " + str(self.b)
        return string
        

