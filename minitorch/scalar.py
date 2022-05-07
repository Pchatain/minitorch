from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np


# ## Task 1.1
# Central Difference calculation


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`
        arg (int): the number :math:`i` of the arg to compute the derivative
        epsilon (float): a small constant

    Returns:
        float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    f_x = f(*vals)
    plus_one_vals = [x for x in vals]
    plus_one_vals[arg] += epsilon
    f_x1 = f(*plus_one_vals)
    return (f_x1 - f_x) / epsilon


# ## Task 1.2 and 1.4
# Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.
    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        return LT.apply(self, b)

    def __gt__(self, b):
        return LT.apply(b, self)

    def __eq__(self, b):
        return EQ.apply(self, b)

    def __sub__(self, b):
        return Add.apply(self, Neg.apply(b))

    def __neg__(self):
        return Neg.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def get_data(self):
        "Returns the raw float value"
        return self.data


class ScalarFunction(FunctionBase):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx, *inputs):
        r"""
        Forward call, compute :math:`f(x_0 \ldots x_{n-1})`.

        Args:
            ctx (:class:`Context`): A container object to save
                                    any information that may be needed
                                    for the call to backward.
            *inputs (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`.

        Should return float the computation of the function :math:`f`.
        """
        ctx.save_for_backward(inputs)
        raise NotImplementedError("i put this in")
        return ctx.f(inputs)
        # return -1.0000
        # not sure how we are supposed to access the functiion f.

    @staticmethod
    def backward(ctx, d_out):
        r"""
        Backward call, computes :math:`f'_{x_i}(x_0 \ldots x_{n-1}) \times d_{out}`.

        Args:
            ctx (Context): A container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in the chain rule.

        Should return the computation of the derivative function
        :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.

        """
        raise NotImplementedError("i put this ini")
        # f, vals, args.
        inputs = ctx.saved_values
        grad = []
        for i in range(len(inputs)):
            differential = central_difference(FunctionBase, inputs, arg=i)
            grad.append(differential)
        # all_inputs = []
        # for i in range(len(inputs)):
        #     tmp_tuple = (inputs, i)
        #     all_inputs.append(tmp_tuple)
        # grad = map(central_difference, all_inputs)
        return d_out * grad

    # Checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Addition function :math:`f(x, y) = x + y`"

    @staticmethod
    def forward(ctx, a, b):
        ctx.f = operators.add
        return operators.add(a, b)

    @staticmethod
    def backward(ctx, d_output):
        return d_output, d_output


class Log(ScalarFunction):
    "Log function :math:`f(x) = log(x)`"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(
            (a, b)
        )  # derivative with respec to var 1 is b, and var 2 is a.
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx, d_output):
        a, b = ctx.saved_values
        return (d_output * b, d_output * a)


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx, d_output):
        x = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.neg(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return -1.0 * d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return d_output * operators.sigmoid(a) * operators.sigmoid(1 - a)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx, d_output):
        x = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return d_output * a * operators.exp(a)


class LT(ScalarFunction):
    "Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError()
        a, b = ctx.saved_values
        diff = 0
        if a == b:
            diff = 1.0
        return diff * d_output


class EQ(ScalarFunction):
    "Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError()
        out = 0
        a, b = ctx.saved_values
        return out * d_output


def derivative_check(f, *scalars):
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f (function) : function from n-scalars to 1-scalar.
        *scalars (list of :class:`Scalar`) : n input scalar values.
    """
    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]
    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
    