import abc


class Oracle:
    """A set oracle interface

    An Oracle encapsulates a set of points that live in some
    subset of Euclidean space. It provides access to its
    underlying set via two methods: a user may query it to
    see whether a point resides in the set, or she may
    ask to obtain an outer approximation of it.

    Note that an Oracle is an abstract interface.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def query(self, x_0):
        """At minimum, returns x_0 if x_0 \in set"""
        pass


    @abc.abstractmethod
    def outer(self, kind):
        """Returns an outer approximation of the set"""
        pass
