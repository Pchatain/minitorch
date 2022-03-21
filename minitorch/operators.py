"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    """
    Multiplies two numbers.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float : product of x and y
        :math:`f(x, y) = x * y`
    """
    return x * y


def id(x):
    """
    Identity function.

    Args:
        x (float): the input value

    Returns:
        float: the input values
        :math:`f(x) = x`
    """
    return x


def add(x, y):
    """
    Adds two numbers

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        The sum of both numbers
        :math:`f(x, y) = x + y`
    """
    return x + y


def neg(x):
    """
    Takes the negation of a number, otherwise known as the additive inverse of a number.

    Args:
        x (float): The number to negate

    Returns:
        The additive inverse of x
        :math:`f(x) = -x`
    """
    return -x


def lt(x, y):
    """
    Indicates whether the first number is less than the second number.

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        float : 1.0 if the first number is strictly smaller, and 0.0 otherwise
        :math:`f(x) =` 1.0 if x is less than y else 0.0

    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    """
    Indicates whether two numbers are equal.

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        float : 1.0 if the numbers are equal, and 0.0 otherwise
        :math:`f(x) =` 1.0 if x is equal to y else 0.0

    """
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    """
    Returns the maximum of two numbers.

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        float : The maximum of x and y
        :math:`f(x) = x` if x is greater than y else y

    """
    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    """
    Indicates whether two numbers are close enough to be considered equal.

    Args:
        x (float): The first number
        y (float): The second number

    Returns:
        bool : True if the numbers are close enough to be considered equal, False otherwise
        :math:`f(x) = |x - y| < 1e-2`

    """
    return abs(x - y) < 1e-2


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x > 0:
        return x
    else:
        return 0


EPS = 1e-6


def log(x):
    r"""
    :math:`f(x) = log(x)`

    Args:
        x (float): input

    Returns:
        float : log value
    """
    return math.log(x + EPS)


def exp(x):
    r"""
    :math:`f(x) = e^x`

    Args:
        x (float): input

    Returns:
        float : exp value
    """
    return math.exp(x)


def log_back(x, d):
    r"""
    Calculate the derivative of f(x) and multiply by d.

    :math:`d \times log'(x)`

    """
    return d / x


def inv(x):
    r""":math:`f(x) = 1/x`

    Args:
        x (float): input

    Returns:
        float : inverse value
    """
    return 1.0 / x


def inv_back(x, d):
    r"""If :math:`f(x) = 1/x` compute :math:`d \times f'(x)`

    Args:
        x (float): input
        d (float): constant multiplier

    Returns:
        float : derivative of d/x
    """
    return -d * x**-2


def relu_back(x, d):
    r"""
    If :math:`f = relu` compute :math:`d \times f'(x)`

    Args:
        x (float): input
        d (float): constant multiplier

    Returns:
        float : derivative of d/x
    """
    if x > 0:
        return d
    else:
        return 0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    return lambda ls: [fn(x) for x in ls]


def negList(ls):
    r"""Use :func:`map` and :func:`neg` to negate each element in `ls`
    Args:
        ls (list): list of numbers

    Returns:
        list : list of numbers with each element negated
    """
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def map2(ls1, ls2):
        zipped = zip(ls1, ls2)
        return [fn(*x) for x in zipped]

    return map2


def addLists(ls1, ls2):
    r"""Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`

    Args:
        ls1 (list): first list
        ls2 (list): second list

    Returns:
        list : the sum of the two lists
    """
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """

    def reduce_list(ls):
        curr = start
        for elem in ls:
            curr = fn(elem, curr)
        return curr

    return reduce_list


def sum(ls):
    r"""Sum up a list using :func:`reduce` and :func:`add`.

    Args:
        ls (list): list of numbers

    Returns:
        float : sum of the list
    """
    return reduce(add, 0)(ls)


def prod(ls):
    r"""Product of a list using :func:`reduce` and :func:`mul`.
    Args:
        ls (list): list of numbers

    Returns:
        float : product of all elements in the list
    """
    return reduce(mul, 1)(ls)
