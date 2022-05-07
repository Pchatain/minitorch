variable_count = 1


# ## Module 1

# Variable is the main class for autodifferentiation logic for scalars
# and tensors.


class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """

    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0

    def requires_grad_(self, val):
        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            val (bool): whether to require grad
        """
        self.history = History()

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    @property
    def derivative(self):
        return self._derivative

    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def accumulate_derivative(self, val):
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            val (number): value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.zero_derivative_()

    def expand(self, x):
        "Placeholder for tensor variables"
        return x

    # Helper functions for children classes.

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0


# Some helper functions for handling optional tuples.


def wrap_tuple(x):
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


# Classes for Functions.


class Context:
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):  # pragma: no cover
        return self.saved_values


class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        if self is None:
            return []
        var_and_derivs = self.last_fn.chain_rule(self.ctx, self.inputs, d_output)
        derivatives = []
        for var, deriv in var_and_derivs:
            derivatives.append(deriv)
        return derivatives

        # derivatives = []
        # grads = self.chain_rule(self.cls, self.ctx, self.inputs, d_output)
        # for grad in grads:
        #     var, deriv = grad
        #     derivatives.append(deriv)

        # # for input in self.inputs:
        # #     print(f"derivative with respect to {input.name} is {input.derivative}")
        # #     derivatives.append(d_output * input.derivative)
        # return derivatives


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Implement by children class.
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # Tip: Note when implementing this function that
        # cls.backward may return either a value or a tuple.
        # this should be good
        derivatives = cls.backward(ctx, d_output)
        if not isinstance(derivatives, list) and not isinstance(derivatives, tuple):
            derivatives = [derivatives]
            # needed for when derivatives returns just a number

        output = []
        for i, input in enumerate(inputs):
            if isinstance(input, Variable):
                if is_constant(input):
                    continue
                name = input.name
                output.append((Variable(None, name=name), derivatives[i]))
        return output


# Algorithms for backpropagation


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def topological_sort(variable):
    """
    Computes the topological order of the computation graph.

    Args:
        variable (:class:`Variable`): The right-most variable

    Returns:
        list of Variables : Non-constant Variables in topological order
                            starting from the right.
    """

    def visit(node, nodeQueue, finishedNodes, L):
        try:
            tetst = node.name
        except:
            # print(f"Node is not correct type, has type {type(node)} and value {node}")
            return
        if node.unique_id in finishedNodes:
            return
        if node.unique_id in nodeQueue:
            print("Cycle detected")
            return ValueError("not a DAG, uh oh.")
        nodeQueue[node.unique_id] = node
        if node.history is not None:
            previous_variables = node.history.inputs
            # print(f"Previous variables are {previous_variables} from variable {node}")
            if previous_variables is not None:
                for previous_variable in previous_variables:
                    visit(previous_variable, nodeQueue, finishedNodes, L)

        nodeQueue.pop(node.unique_id)
        finishedNodes[node.unique_id] = node
        L.insert(0, node)

    L = []
    nodeQueue = {}  # variable.unique_id: (variable, None)
    finishedNodes = {}
    node = variable
    while nodeQueue is not None:
        visit(node, nodeQueue, finishedNodes, L)
        if len(nodeQueue) == 0:
            nodeQueue = None
    # print("---------found-L--------")
    # for i in L:
    #     print(i.unique_id)
    # print("----------------------")
    return L


def backpropagate(variable, deriv):
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    See :doc:`backpropagate` for details on the algorithm.

    Args:
        variable (:class:`Variable`): The right-most variable
        deriv (number) : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # step 0 get ordered queue
    # print(f"We initialize the funcitoin at {variable.name}")
    top_sort = topological_sort(variable)

    # step 1 create dict of variables and curr derivatives
    curr_derivatives = {v.unique_id: 0.0 for v in top_sort}
    curr_derivatives[variable.unique_id] = deriv
    # step 2, for each node in backward order pull var and eriv from queue
    for var in top_sort:
        # a. if the variabel is a leaf, accumulate deriv and loop to (1)
        # print(f"We are processing {var.name}")
        if var.history is None:
            # var.accumulate_derivative(curr_derivatives[var.unique_id])
            continue
        # print(f"The past functitons are {var.history.last_fn}")
        if var.is_leaf():
            # print(f"We identified leaf {var.name}")
            var.accumulate_derivative(curr_derivatives[var.unique_id])
            continue
            # accumulate_derivative(var, curr_derivatives[var.unique_id])
        # b. if var is not a leaf
        # 1. call .backprop_step on the last function that created it wtih derivative as d_out
        derivatives = var.history.backprop_step(curr_derivatives[var.unique_id])
        index = 0
        if not isinstance(derivatives, list):
            derivatives = [derivatives]
        # print(
        #     f"We have len(var.history.inputs) = {len(var.history.inputs)}, and derivatives = {derivatives}"
        # )
        for i in range(len(var.history.inputs)):
            input = var.history.inputs[i]
            if isinstance(input, Variable):
                if is_constant(input):
                    continue  # because in chain_rule we skip over constants (deriv=0)
                # print(f"i is {i} and index is {index}")
                curr_derivatives[input.unique_id] += derivatives[index]
                # print(f", and derivatives is {derivatives[index]}")
                index += 1
        # print("----------------")
        # 2. loop through all the variables + derivatives produced by the chain rule
        # 3. accumulate the derivatives for the variable in a dictionary (check .unique_id)

    #     if is_constant(var):
    #         continue
    #     var.backprop_step(deriv)
    # variable.derivative = deriv


"""
I am stuck becasue I can't get it to work even in the most basic case. 
I don't understand how to pass the derivatiives. It doesn't make sesne, because once you process the derivative
for one variable, you should be passing it along to the variables that depend on it,
but so far it's just processing everything in place and the variables are not communicating in any way.

I guess there should be access to the 
"""

"""
So i figured it out, derivatives have to add and they ahve to come from the nodes before.
I figured this problem out by writing down on paper what the functions do, writing down 
exactly why the code made no sense, and then actually trying to figure
out how this could in principle work. From there, I was able to narrow down to only like
2 possible ways to implement the code, and was able to select from there. I debugged by
printing output, and paying attention to error codes. CUrrent errors:

relu is ggiving int error. int not subscriptible.
divconstant, exp, subconstant = list index out of range -- Fixed these bugs

inv, add, cube, log, multconstant, neg, sig, square,  = good

complex = overflow error

"""
