import torch
from stlcgpp.utils import *


# Expressions
class Expression(torch.nn.Module):
    """
    Class representing an expression with a name and value.
    """

    def __init__(self, name, value):
        """
        Initialize an Expression.

        Args:
            name (str): The name of the expression.
            value (float): The value of the expression.
        """
        super(Expression, self).__init__()
        self.value = value
        self.name = name

    def set_name(self, new_name):
        """
        Set a new name for the expression.

        Args:
            new_name (str): The new name of the expression.
        """
        self.name = new_name

    def set_value(self, new_value):
        """
        Set a new value for the expression.

        Args:
            new_value (float): The new value of the expression.
        """
        self.value = new_value

    def get_name(self):
        """
        Get the name of the expression.

        Returns:
            str: The name of the expression.
        """
        return self.name

    def forward(self):
        """
        Forward pass to get the value of the expression.

        Returns:
            float: The value of the expression.
        """
        return self.value

    def __neg__(self):
        """
        Negate the expression.

        Returns:
            Expression: A new expression with negated value.
        """
        return Expression("-" + self.name, -self.value)

    def __add__(self, other):
        """
        Add another expression or value to this expression.

        Args:
            other (Expression or float): The other expression or value to add.

        Returns:
            Expression: A new expression representing the sum.
        """
        if isinstance(other, Expression):
            return Expression(self.name + " + " + other.name, self.value + other.value)
        else:
            return Expression(self.name + " + other", self.value + other)

    def __radd__(self, other):
        """
        Add this expression to another value (right-hand side).

        Args:
            other (float): The other value to add.

        Returns:
            Expression: A new expression representing the sum.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract another expression or value from this expression.

        Args:
            other (Expression or float): The other expression or value to subtract.

        Returns:
            Expression: A new expression representing the difference.
        """
        if isinstance(other, Expression):
            return Expression(self.name + " - " + other.name, self.value - other.value)
        else:
            return Expression(self.name + " - other", self.value - other)

    def __rsub__(self, other):
        """
        Subtract this expression from another value (right-hand side).

        Args:
            other (float): The other value to subtract from.

        Returns:
            Expression: A new expression representing the difference.
        """
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        """
        Multiply this expression by another expression or value.

        Args:
            other (Expression or float): The other expression or value to multiply.

        Returns:
            Expression: A new expression representing the product.
        """
        if isinstance(other, Expression):
            return Expression(self.name + " × " + other.name, self.value * other.value)
        else:
            return Expression(self.name + " * other", self.value * other)

    def __rmul__(self, other):
        """
        Multiply another value by this expression (right-hand side).

        Args:
            other (float): The other value to multiply.

        Returns:
            Expression: A new expression representing the product.
        """
        return self.__mul__(other)

    def __truediv__(a, b):
        """
        Divide this expression by another expression or value.

        Args:
            a (Expression): The numerator expression.
            b (Expression or float): The denominator expression or value.

        Returns:
            Expression: A new expression representing the quotient.
        """
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        return Expression(
            numerator.name + "/" + denominator.name, numerator.value / denominator.value
        )

    # Comparators
    def __lt__(lhs, rhs):
        """
        Less than comparator for expressions.

        Args:
            lhs (Expression or str): The left-hand side expression or string.
            rhs (Expression or float): The right-hand side expression or value.

        Returns:
            LessThan: A LessThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __le__(lhs, rhs):
        """
        Less than or equal to comparator for expressions.

        Args:
            lhs (Expression or str): The left-hand side expression or string.
            rhs (Expression or float): The right-hand side expression or value.

        Returns:
            LessThan: A LessThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __gt__(lhs, rhs):
        """
        Greater than comparator for expressions.

        Args:
            lhs (Expression or str): The left-hand side expression or string.
            rhs (Expression or float): The right-hand side expression or value.

        Returns:
            GreaterThan: A GreaterThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __ge__(lhs, rhs):
        """
        Greater than or equal to comparator for expressions.

        Args:
            lhs (Expression or str): The left-hand side expression or string.
            rhs (Expression or float): The right-hand side expression or value.

        Returns:
            GreaterThan: A GreaterThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        """
        Equal to comparator for expressions.

        Args:
            lhs (Expression or str): The left-hand side expression or string.
            rhs (Expression or float): The right-hand side expression or value.

        Returns:
            Equal: An Equal formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of Equal needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    def __str__(self):
        """
        String representation of the expression.

        Returns:
            str: The name of the expression.
        """
        return str(self.name)


# Predicates
class Predicate(torch.nn.Module):
    """
    Class representing a predicate with a name and a predicate function.
    """

    def __init__(self, name, predicate_function: Callable):
        """
        Initialize a Predicate.

        Args:
            name (str): The name of the predicate.
            predicate_function (Callable): The function representing the predicate.
        """
        super(Predicate, self).__init__()
        self.name = name
        self.predicate_function = predicate_function

    def forward(self, signal: torch.Tensor):
        """
        Forward pass to evaluate the predicate function on a signal.

        Args:
            signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The result of the predicate function.
        """
        return self.predicate_function(signal)

    def set_name(self, new_name):
        """
        Set a new name for the predicate.

        Args:
            new_name (str): The new name of the predicate.
        """
        self.name = new_name

    def __neg__(self):
        """
        Negate the predicate.

        Returns:
            Predicate: A new predicate with negated function.
        """
        return Predicate("- " + self.name, lambda x: -self.predicate_function(x))

    def __add__(self, other):
        """
        Add another predicate to this predicate.

        Args:
            other (Predicate): The other predicate to add.

        Returns:
            Predicate: A new predicate representing the sum.
        """
        if isinstance(other, Predicate):
            return Predicate(
                self.name + " + " + other.name,
                lambda x: self.predicate_function(x) + other.predicate_function(x),
            )
        else:
            raise ValueError("Type error. Must be Predicate")

    def __radd__(self, other):
        """
        Add this predicate to another predicate (right-hand side).

        Args:
            other (Predicate): The other predicate to add.

        Returns:
            Predicate: A new predicate representing the sum.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract another predicate from this predicate.

        Args:
            other (Predicate): The other predicate to subtract.

        Returns:
            Predicate: A new predicate representing the difference.
        """
        if isinstance(other, Predicate):
            return Predicate(
                self.name + " - " + other.name,
                lambda x: self.predicate_function(x) - other.predicate_function(x),
            )
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rsub__(self, other):
        """
        Subtract this predicate from another predicate (right-hand side).

        Args:
            other (Predicate): The other predicate to subtract from.

        Returns:
            Predicate: A new predicate representing the difference.
        """
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        """
        Multiply this predicate by another predicate.

        Args:
            other (Predicate): The other predicate to multiply.

        Returns:
            Predicate: A new predicate representing the product.
        """
        if isinstance(other, Predicate):
            return Predicate(
                self.name + " x " + other.name,
                lambda x: self.predicate_function(x) * other.predicate_function(x),
            )
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rmul__(self, other):
        """
        Multiply another predicate by this predicate (right-hand side).

        Args:
            other (Predicate): The other predicate to multiply.

        Returns:
            Predicate: A new predicate representing the product.
        """
        return self.__mul__(other)

    def __truediv__(a, b):
        """
        Divide this predicate by another predicate.

        Args:
            a (Predicate): The numerator predicate.
            b (Predicate): The denominator predicate.

        Returns:
            Predicate: A new predicate representing the quotient.
        """
        if isinstance(a, Predicate) and isinstance(b, Predicate):
            return Predicate(
                a.name + " / " + b.name,
                lambda x: a.predicate_function(x) / b.predicate_function(x),
            )
        else:
            raise ValueError("Type error. Must be Predicate")

    # Comparators
    def __lt__(lhs, rhs):
        """
        Less than comparator for predicates.

        Args:
            lhs (Predicate or str): The left-hand side predicate or string.
            rhs (Predicate or float): The right-hand side predicate or value.

        Returns:
            LessThan: A LessThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Predicate
        ), "LHS of LessThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __le__(lhs, rhs):
        """
        Less than or equal to comparator for predicates.

        Args:
            lhs (Predicate or str): The left-hand side predicate or string.
            rhs (Predicate or float): The right-hand side predicate or value.

        Returns:
            LessThan: A LessThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Predicate
        ), "LHS of LessThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __gt__(lhs, rhs):
        """
        Greater than comparator for predicates.

        Args:
            lhs (Predicate or str): The left-hand side predicate or string.
            rhs (Predicate or float): The right-hand side predicate or value.

        Returns:
            GreaterThan: A GreaterThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Predicate
        ), "LHS of GreaterThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __ge__(lhs, rhs):
        """
        Greater than or equal to comparator for predicates.

        Args:
            lhs (Predicate or str): The left-hand side predicate or string.
            rhs (Predicate or float): The right-hand side predicate or value.

        Returns:
            GreaterThan: A GreaterThan formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Predicate
        ), "LHS of GreaterThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        """
        Equal to comparator for predicates.

        Args:
            lhs (Predicate or str): The left-hand side predicate or string.
            rhs (Predicate or float): The right-hand side predicate or value.

        Returns:
            Equal: An Equal formula.
        """
        assert isinstance(lhs, str) | isinstance(
            lhs, Predicate
        ), "LHS of Equal needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    def __hash__(self):
        return hash((self.name, self.predicate_function))

    def __str__(self):
        """
        String representation of the predicate.

        Returns:
            str: The name of the predicate.
        """
        return str(self.name)


def convert_to_input_values(inputs):
    """
    Convert inputs to their numerical values.

    Args:
        inputs (Union[Expression, torch.Tensor, tuple]): The inputs to convert.

    Returns:
        Union[float, torch.Tensor, tuple]: The numerical values of the inputs.
    """
    if not isinstance(inputs, tuple):
        if isinstance(inputs, Expression):
            assert (
                inputs.value is not None
            ), "Input Expression does not have numerical values"
            # if Expression is not time reversed
            return inputs.value
        elif isinstance(inputs, torch.Tensor):
            return inputs
        else:
            raise ValueError("Not a invalid input trace")
    else:
        return (convert_to_input_values(inputs[0]), convert_to_input_values(inputs[1]))


# STL formula
class STLFormula(torch.nn.Module):
    """
    Class for an STL formula.
    NOTE: If Expressions and Predicates are used, then the signals will be reversed if needed. Otherwise, user is responsible for keeping track.
    """

    def __init__(self):
        """
        Initialize an STLFormula.
        """
        super(STLFormula, self).__init__()

    def robustness_trace(self, signal: torch.Tensor, **kwargs):
        """
        Computes the robustness trace of the formula given an input signal.

        Args:
            signal (torch.Tensor): Expected size [time_dim, state_dim] for Predicate-based formulas or [time_dim] for Expression-based formulas
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: The robustness trace of the input signal. Expected size [time_dim]
        """
        # return signal
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal: torch.Tensor, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal (torch.Tensor): Expected size [time_dim, state_dim] for Predicate-based formulas or [time_dim] for Expression-based formulas
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: Robustness value of the input signal. Expected size [1]
        """
        return self.forward(signal, **kwargs)[0]

    def eval_trace(self, signal: torch.Tensor, **kwargs):
        """
        Boolean of robustness_trace.

        Args:
            signal (torch.Tensor): Expected size [time_dim, state_dim] for Predicate-based formulas or [time_dim] for Expression-based formulas
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: Boolean evaluation of the robustness trace. Expected size [time_dim]
        """
        return self.forward(signal, **kwargs) > 0

    def eval(self, signal: torch.Tensor, **kwargs):
        """
        Boolean of robustness.

        Args:
            signal (torch.Tensor): Expected size [time_dim, state_dim] for Predicate-based formulas or [time_dim] for Expression-based formulas
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: Boolean evaluation of the robustness. Expected size [1]
        """
        return self.robustness(signal, **kwargs) > 0

    def forward(self, signal: torch.Tensor, **kwargs):
        """
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.

        Args:
            signal (torch.Tensor): Expected size [time_dim, state_dim] for Predicate-based formulas or [time_dim] for Expression-based formulas
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: The robustness trace. . Expected size [time_dim]
        """
        inputs = convert_to_input_values(signal)
        return self.robustness_trace(inputs, **kwargs)

    def _next_function(self):
        """
        Function to keep track of the subformulas. For visualization purposes.
        """
        raise NotImplementedError("_next_function not year implemented")

    """ Overwriting some built-in functions for notational simplicity """

    def __and__(self, psi):
        """
        Logical AND operation for STL formulas.

        Args:
            psi (STLFormula): The other STL formula.

        Returns:
            And: An And formula.
        """
        return And(self, psi)

    def __or__(self, psi):
        """
        Logical OR operation for STL formulas.

        Args:
            psi (STLFormula): The other STL formula.

        Returns:
            Or: An Or formula.
        """
        return Or(self, psi)

    def __invert__(self):
        """
        Logical NOT operation for STL formulas.

        Returns:
            Negation: A Negation formula.
        """
        return Negation(self)


class Identity(STLFormula):
    """
    The identity formula. Use in UntilRecurrent.
    """

    def __init__(self, name="x"):
        """
        Initialize an Identity formula.

        Args:
            name (str): The name of the identity formula.
        """
        super().__init__()
        self.name = name

    def robustness_trace(self, signal: torch.Tensor, **kwargs):
        """
        Return the input signal as the robustness trace.

        Args:
            signal (torch.Tensor): The input signal.
            kwargs: Other arguments.

        Returns:
            torch.Tensor: The input signal.
        """
        return signal

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return []

    def __str__(self):
        """
        String representation of the identity formula.

        Returns:
            str: The name of the identity formula.
        """
        return self.name


class LessThan(STLFormula):
    """
    The LessThan STL formula.
    """

    def __init__(
        self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]
    ):
        """
        Initialize a LessThan formula.

        Args:
            lhs (Union[Predicate, Expression, str]): The left-hand side of the comparison.
            rhs (Union[float, torch.Tensor]): The right-hand side of the comparison.
        """
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        """
        Compute the robustness trace for the LessThan formula.

        Args:
            signal (Union[torch.Tensor, Expression]): The input signal.
            kwargs: Other arguments.

        Returns:
            torch.Tensor: The robustness trace.
        """
        if isinstance(self.lhs, Predicate):
            return self.rhs - self.lhs(signal)
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return self.rhs - signal.value
        else:
            return self.rhs - signal

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.lhs, self.rhs]

    def __str__(self):
        """
        String representation of the LessThan formula.

        Returns:
            str: The string representation of the formula.
        """
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " < " + str(self.rhs)


class GreaterThan(STLFormula):
    """
    The GreaterThan STL formula.
    """

    def __init__(
        self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]
    ):
        """
        Initialize a GreaterThan formula.

        Args:
            lhs (Union[Predicate, Expression, str]): The left-hand side of the comparison.
            rhs (Union[float, torch.Tensor]): The right-hand side of the comparison.
        """
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        """
        Compute the robustness trace for the GreaterThan formula.

        Args:
            signal (Union[torch.Tensor, Expression]): The input signal.
            kwargs: Other arguments.

        Returns:
            torch.Tensor: The robustness trace.
        """
        if isinstance(self.lhs, Predicate):
            return self.lhs(signal) - self.rhs
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return signal.value - self.rhs
        else:
            return signal - self.rhs

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.lhs, self.rhs]

    def __str__(self):
        """
        String representation of the GreaterThan formula.

        Returns:
            str: The string representation of the formula.
        """
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " > " + str(self.rhs)


class Equal(STLFormula):
    """
    The Equal STL formula.
    """

    def __init__(
        self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]
    ):
        """
        Initialize an Equal formula.

        Args:
            lhs (Union[Predicate, Expression, str]): The left-hand side of the comparison.
            rhs (Union[float, torch.Tensor]): The right-hand side of the comparison.
        """
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        """
        Compute the robustness trace for the Equal formula.

        Args:
            signal (Union[torch.Tensor, Expression]): The input signal.
            kwargs: Other arguments.

        Returns:
            torch.Tensor: The robustness trace.
        """
        if isinstance(self.lhs, Predicate):
            return -torch.abs((self.lhs(signal) - self.rhs))
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return -torch.abs(signal.value - self.rhs)
        else:
            return -torch.abs(signal - self.rhs)

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.lhs, self.rhs]

    def __str__(self):
        """
        String representation of the Equal formula.

        Returns:
            str: The string representation of the formula.
        """
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " == " + str(self.rhs)


class Negation(STLFormula):
    """
    The Negation STL formula.
    """

    def __init__(self, subformula: STLFormula):
        """
        Initialize a Negation formula.

        Args:
            subformula (STLFormula): The subformula to negate.
        """
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        """
        Compute the robustness trace for the Negation formula.

        Args:
            signal (Union[torch.Tensor, Expression]): The input signal.
            kwargs: Other arguments.

        Returns:
            torch.Tensor: The robustness trace.
        """
        return -self.subformula(signal, **kwargs)

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.subformula]

    def __str__(self):
        """
        String representation of the Negation formula.

        Returns:
            str: The string representation of the formula.
        """
        return "¬(" + str(self.subformula) + ")"


class And(STLFormula):
    """
    The And STL formula ∧ (subformula1 ∧ subformula2).
    Args:
        subformula1: subformula for lhs of the And operation.
        subformula2: subformula for rhs of the And operation.
    """

    def __init__(self, subformula1, subformula2):
        """
        Initialize an And formula.

        Args:
            subformula1 (STLFormula): The left-hand side subformula.
            subformula2 (STLFormula): The right-hand side subformula.
        """
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z))) where each x, y, z have expected size [time_dim].
            If using Predicates to define the formula, then inputs is just a single tensor -- no need for different signals for each subformula. Expected size [time_dim, state_dim] for Predicate-based formulas.
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.Tensor.
        """
        xx = separate_and(self, inputs, **kwargs)
        return minish(
            xx, dim=-1, keepdim=False, **kwargs
        )  # [batch_size, time_dim, ...]

    def _next_function(self):
        """
        Next function is the input subformulas. For visualization purposes.
        """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        """
        String representation of the And formula.

        Returns:
            str: The string representation of the formula.
        """
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"


class Or(STLFormula):
    """
    The Or STL formula ∧ (subformula1 ∧ subformula2).
    Args:
        subformula1: subformula for lhs of the Or operation.
        subformula2: subformula for rhs of the Or operation.
    """

    def __init__(self, subformula1, subformula2):
        """
        Initialize an Or formula.

        Args:
            subformula1 (STLFormula): The left-hand side subformula.
            subformula2 (STLFormula): The right-hand side subformula.
        """
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) V (ϕ₂(y) V ϕ₃(z)) would have inputs=(x, (y,z))) where each x, y, z have expected size [time_dim].
            If using Predicates to define the formula, then inputs is just a single tensor -- no need for different signals for each subformula. Expected size [time_dim, state_dim] for Predicate-based formulas.
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.Tensor.
        """
        xx = separate_or(self, inputs, **kwargs)
        return maxish(
            xx, dim=-1, keepdim=False, **kwargs
        )  # [batch_size, time_dim, ...]

    def _next_function(self):
        """
        Next function is the input subformulas. For visualization purposes.
        """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        """
        String representation of the Or formula.

        Returns:
            str: The string representation of the formula.
        """
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Implies(STLFormula):
    """
    The Implies STL formula ⇒. subformula1 ⇒ subformula2
    Args:
        subformula1: subformula for lhs of the Implies operation
        subformula2: subformula for rhs of the Implies operation
    """

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace, **kwargs):
        """
        Computing robustness trace of subformula1 ⇒ subformula2    max(-subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs.
            If using Predicates to define the formula, then inputs is just a single tensor -- no need for different signals for each subformula. Expected size [time_dim, state_dim] for Predicate-based formulas.
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.Tensor.
        """
        if isinstance(trace, tuple):
            trace1, trace2 = trace
            signal1 = self.subformula1(trace1, **kwargs)
            signal2 = self.subformula2(trace2, **kwargs)
        else:
            signal1 = self.subformula1(trace, **kwargs)
            signal2 = self.subformula2(trace, **kwargs)
        xx = torch.stack([-signal1, signal2], dim=-1)  # [time_dim, ..., 2]
        return maxish(xx, dim=-1, keepdim=False, **kwargs)  # [time_dim, ...]

    def _next_function(self):
        """next function is the input subformulas. For visualization purposes"""
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ⇒ (" + str(self.subformula2) + ")"


class Eventually(STLFormula):
    """
    The Eventually STL formula ♢ (subformula).
    """

    def __init__(self, subformula, interval=None):
        """
        Initialize an Eventually formula.

        Args:
            subformula (STLFormula): The subformula for the Eventually operation.
            interval (list, optional): The time interval for the Eventually operation. Defaults to [0, torch.inf].
        """
        super().__init__()
        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        """
        Compute the robustness trace for the Eventually formula.

        Args:
            signal (Union[torch.Tensor, tuple]): The input signal. Either a torch.Tensor or tuple depending on whether the formula is defined using Predicates or Expressions.
            padding (str, optional): The padding method. Defaults to None.
            large_number (float, optional): A large number used for masking. Defaults to 1e9.
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: The robustness trace.
        """
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(
            signal, padding=padding, large_number=large_number, **kwargs
        )
        T = signal.shape[time_dim]
        mask_value = -large_number
        offset = 0

        # if self.interval is None:
        #     interval = [0, T - 1]
        # elif self.interval[1] == torch.inf:
        #     interval = [self.interval[0], T - 1]
        #     offset = self.interval[0]
        # else:
        #     interval = self.interval
        def true_func(_interval, T):
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval, T):
            return _interval, 1.0, 0

        operands = (
            self._interval,
            T,
        )
        interval, _, offset = cond(
            self._interval[1] == torch.inf, true_func, false_func, *operands
        )

        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -large_number

        signal_pad = torch.ones([interval[1] + 1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[-1] - offset
        ) * torch.tril(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[0]
        )
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return maxish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.subformula]

    def __str__(self):
        """
        String representation of the Eventually formula.

        Returns:
            str: The string representation of the formula.
        """
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Always(STLFormula):
    def __init__(self, subformula, interval=None):
        """
        Initialize an Always formula.

        Args:
            subformula (STLFormula): The subformula for the Always operation.
            interval (list, optional): The time interval for the Always operation. Defaults to [0, torch.inf].
        """
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        """
        Compute the robustness trace for the Always formula.

        Args:
            signal (Union[torch.Tensor, tuple]): The input signal. Either a torch.Tensor or tuple depending on whether the formula is defined using Predicates or Expressions.
            padding (str, optional): The padding method. Defaults to None.
            large_number (float, optional): A large number used for masking. Defaults to 1e9.
            kwargs: Other arguments including time_dim, approx_method, temperature.

        Returns:
            torch.Tensor: The robustness trace for the Always formula.
        """
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(
            signal, padding=padding, large_number=large_number, **kwargs
        )
        T = signal.shape[time_dim]
        mask_value = large_number
        sign = 1.0
        offset = 0.0

        def true_func(_interval, T):
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval, T):
            return _interval, 1.0, 0

        operands = (
            self._interval,
            T,
        )
        interval, sign, offset = cond(
            self._interval[1] == torch.inf, true_func, false_func, *operands
        )

        # if self.interval is None:
        #     interval = [0,T-1]
        # elif self.interval[1] == torch.inf:
        #     interval = [self.interval[0], T-1]
        # else:
        #     interval = self.interval
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -large_number

        signal_pad = torch.cat(
            [
                torch.ones([interval[1], T], device=device) * sign * pad_value,
                torch.ones([1, T], device=device) * pad_value,
            ],
            dim=time_dim,
        )
        # signal_pad = torch.ones([interval[1]+1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[-1] - offset
        ) * torch.tril(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[0]
        )
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return minish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.subformula]

    def __str__(self):
        """
        String representation of the Always formula.

        Returns:
            str: The string representation of the formula.
        """
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Until(STLFormula):
    def __init__(self, subformula1, subformula2, interval=None):
        class Until:
            """
            Initialize an Until formula.

            Args:
                subformula1 (STLFormula): The first subformula.
                subformula2 (STLFormula): The second subformula.
                interval (list, optional): The time interval for the Until operation. Defaults to [0, torch.inf].
            """

        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        """
        Compute the robustness trace for the Until formula.

        Args:
            signal (Union[torch.Tensor, tuple]): The input signal. Either a torch.Tensor or a tuple of torch.Tensors depending on whether the formula is defined using Predicates or Expressions.
            padding (str, optional): The padding method to use for the signal. Options are 'last', 'mean', or None. Defaults to None.
            kwargs: Additional arguments including time_dim, approx_method, and temperature.

        Returns:
            torch.Tensor: The robustness trace of the Until formula.
        """
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        if isinstance(signal, tuple):
            signal1, signal2 = signal
            assert signal1.shape[time_dim] == signal2.shape[time_dim]
            signal1 = self.subformula1(
                signal1, padding=padding, large_number=large_number, **kwargs
            )
            signal2 = self.subformula2(
                signal2, padding=padding, large_number=large_number, **kwargs
            )
            T = signal1.shape[time_dim]
        else:
            signal1 = self.subformula1(
                signal, padding=padding, large_number=large_number, **kwargs
            )
            signal2 = self.subformula2(
                signal, padding=padding, large_number=large_number, **kwargs
            )
            T = signal.shape[time_dim]

        mask_value = large_number
        if self.interval is None:
            interval = [0, T - 1]
        elif self.interval[1] == torch.inf:
            interval = [self.interval[0], T - 1]
        else:
            interval = self.interval

        # Adding more memory efficiency (instead of creating a ones tensor and multiplying, expand will create views which are faster)
        signal1_matrix = signal1.unsqueeze(1).expand(-1, T)
        signal2_matrix = signal2.unsqueeze(1).expand(-1, T)

        if padding == "last":
            pad_value1 = signal1[-1]
            pad_value2 = signal2[-1]
        elif padding == "mean":
            pad_value1 = signal1.mean(dim=time_dim)
            pad_value2 = signal2.mean(dim=time_dim)
        else:
            pad_value1 = torch.tensor(-mask_value, device=device)
            pad_value2 = torch.tensor(-mask_value, device=device)

        # again instead of creating whole tensors, we just use expand which creates a view into it.
        signal1_pad = pad_value1.view(1, 1).expand(interval[1] + 1, T)
        signal2_pad = pad_value2.view(1, 1).expand(interval[1] + 1, T)
        signal1_padded = torch.cat([signal1_matrix, signal1_pad], dim=time_dim)
        signal2_padded = torch.cat([signal2_matrix, signal2_pad], dim=time_dim)

        rows = torch.arange(T + interval[1] + 1, device=device).view(
            -1, 1
        )  # Row indices
        cols = torch.arange(T, device=device).view(1, -1)  # Column indices

        # Generate masks directly without multiplying two subsequent masks
        phi1_mask = torch.stack(
            [
                ((cols - rows >= -end_idx) & (cols - rows <= 0))  # Row-bound growth
                for end_idx in range(interval[0], interval[-1] + 1)
            ],
            dim=0,
        )

        phi2_mask = torch.stack(
            [
                (
                    (cols - rows >= -end_idx) & (cols - rows <= -end_idx)
                )  # Row-bound growth
                for end_idx in range(interval[0], interval[-1] + 1)
            ],
            dim=0,
        )

        signal1_batched = signal1_padded.unsqueeze(0)  # [1, T + interval[1] + 1, T]
        signal2_batched = signal2_padded.unsqueeze(0)  # [1, T + interval[1] + 1, T]

        # Apply all masks in parallel using broadcasting
        phi1_masked_signal = torch.where(
            phi1_mask,  # [num_masks, T + interval[1] + 1, T]
            signal1_batched,  # [1, T + interval[1] + 1, T]
            mask_value,  # Scalar, broadcasted
        )  # Result: [num_masks, T + interval[1] + 1, T]

        phi2_masked_signal = torch.where(
            phi2_mask,  # [num_masks, T + interval[1] + 1, T]
            signal2_batched,  # [1, T + interval[1] + 1, T]
            mask_value,  # Scalar, broadcasted
        )

        return maxish(
            torch.stack(
                [
                    minish(
                        torch.stack(
                            [
                                minish(s1, dim=0, keepdim=False, **kwargs),
                                minish(s2, dim=0, keepdim=False, **kwargs),
                            ],
                            dim=0,
                        ),
                        dim=0,
                        keepdim=False,
                        **kwargs
                    )
                    for (s1, s2) in zip(phi1_masked_signal, phi2_masked_signal)
                ],
                dim=0,
            ),
            dim=0,
            keepdim=False,
            **kwargs
        )

    def _next_function(self):
        """
        Next function is the input subformula. For visualization purposes.
        """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        """
        String representation of the Until formula.

        Returns:
            str: The string representation of the formula.
        """
        return (
            "("
            + str(self.subformula1)
            + ") U "
            + str(self._interval)
            + "("
            + str(self.subformula2)
            + ")"
        )


class TemporalOperator(STLFormula):
    """
    A class to represent temporal operators in Signal Temporal Logic (STL).
    """

    def __init__(self, subformula, interval=None):
        """
        Initialize a Temporal Operator instance. Has subclasses AlwaysRecurrent and EventuallyRecurrent.

        Args:
            subformula (STLFormula): The subformula associated with the temporal operator.
            interval (list or tuple, optional): A list or tuple representing the interval for the temporal operator.
                                                If None, the interval is set to [0, torch.inf].
        """
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self.interval_str = interval

        if self.interval is None:
            self.hidden_dim = None
            self._interval = None
            self.interval_str = [0, torch.inf]
        elif interval[1] == torch.inf:
            self.hidden_dim = None
            self._interval = [interval[0], interval[1]]
        else:
            self.hidden_dim = interval[1] + 1
            self._interval = [interval[0], interval[1]]

        self.LARGE_NUMBER = 1e9
        self.operation = None

    def _get_interval_indices(self):
        """
        Get the start and end indices for the interval.

        Returns:
            tuple: The start and end indices for the interval.
        """
        start_idx = -self.hidden_dim
        end_idx = -self._interval[0]

        return start_idx, (None if end_idx == 0 else end_idx)

    def _run_cell(self, signal, padding=None, **kwargs):
        """
        Run the temporal operator on the given signal.

        Args:
            signal (torch.Tensor): The input signal.
            padding (str, optional): The padding method to use. Defaults to None.
            **kwargs: Additional arguments for the cell function.

        Returns:
            torch.Tensor: The output of the temporal operator.
        """
        hidden_state = self._initialize_hidden_state(
            signal, padding=padding
        )  # [hidden_dim]

        def f_(hidden, state):
            hidden, o = self._cell(state, hidden, **kwargs)
            return hidden, o

        _, outputs_stack = scan(f_, hidden_state, signal)
        return outputs_stack

    def _initialize_hidden_state(self, signal, padding=None):
        """
        Initialize the hidden state for the temporal operator.

        Args:
            signal (torch.Tensor): The input signal.
            padding (str, optional): The padding method to use. Defaults to None.

        Returns:
            torch.Tensor: The initialized hidden state.
        """
        device = signal.device

        if padding == "last":
            pad_value = (signal)[0].detach()
        elif padding == "mean":
            pad_value = (signal).mean(0).detach()
        else:
            pad_value = -self.LARGE_NUMBER

        n_time_steps = signal.shape[0]

        # compute hidden dim if signal length was needed
        if (self.interval is None) or (self.interval[1] == torch.inf):
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self._interval = [0, n_time_steps - 1]
        elif self.interval[1] == torch.inf:
            self._interval[1] = n_time_steps - 1

        # self.M = torch.diag(torch.ones(self.hidden_dim-1), k=1)
        # self.b = torch.zeros(self.hidden_dim)
        # self.b = self.b.at[-1].set(1)

        self.M = torch.diag(torch.ones(self.hidden_dim - 1, device=device), diagonal=1)
        self.b = torch.zeros(self.hidden_dim, device=device)
        self.b[-1] = 1.0

        if (self.interval is None) or (self.interval[1] == torch.inf):
            pad_value = torch.cat(
                [
                    torch.ones(self._interval[0] + 1, device=device) * pad_value,
                    torch.ones(self.hidden_dim - self._interval[0] - 1, device=device)
                    * self.sign
                    * pad_value,
                ]
            )

        h0 = torch.ones(self.hidden_dim, device=device) * pad_value

        return h0

    def _cell(self, state, hidden, **kwargs):
        """
        Perform a cell operation in the model.
        This function computes a new hidden state and an output based on the
        current state and hidden values. It applies a matrix multiplication
        and addition to update the hidden state, and then performs an operation
        on a specific interval of the new hidden state to produce the output.
        Args:
            state (Tensor): The current state tensor.
            hidden (Tensor): The current hidden state tensor.
            **kwargs: Additional keyword arguments for the operation.
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the new hidden state tensor
            and the output tensor.
        """
        h_new = self.M @ hidden + self.b * state
        start_idx, end_idx = self._get_interval_indices()
        output = self.operation(
            h_new[start_idx:end_idx], axis=0, keepdim=False, **kwargs
        )

        return h_new, output

    def robustness_trace(self, signal, padding=None, **kwargs):
        """
        Args:
            Computes the robustness of a given signal trace.
            Parameters:
            signal (array-like): The input signal trace to be evaluated.
            padding (optional): Additional padding to be applied to the signal trace.
            **kwargs: Additional keyword arguments to be passed to subformula and _run_cell methods.
        Returns:
            robustness_trace: torch.Tensor.
        """
        trace = self.subformula(signal, **kwargs)
        outputs = self._run_cell(trace, padding, **kwargs)
        return outputs

    def robustness(self, signal, **kwargs):
        return self.__call__(signal, **kwargs)[-1]

    def _next_function(self):
        return [self.subformula]


class AlwaysRecurrent(TemporalOperator):

    def __init__(self, subformula, interval=None):
        """
        Initializes an AlwaysRecurrent formula
        Args:
            subformula: The subformula to be used.
            interval (optional): The interval for the formula. Defaults to None.
        """
        super().__init__(subformula=subformula, interval=interval)
        self.operation = minish
        self.sign = -1.0

    def __str__(self):
        """
        String representation of the Always formula.

        Returns:
            str: The string representation of the formula.
        """
        return "◻ " + str(self.interval_str) + "( " + str(self.subformula) + " )"


class EventuallyRecurrent(TemporalOperator):

    def __init__(self, subformula, interval=None):
        """
        Initializes an EventuallyRecurrent formula
        Args:
            subformula: The subformula to be used.
            interval (optional): The interval for the formula. Defaults to None.
        """
        super().__init__(subformula=subformula, interval=interval)
        self.operation = maxish
        self.sign = 1.0

    def __str__(self):
        """
        String representation of the Eventually formula.

        Returns:
            str: The string representation of the formula.
        """
        return "♢ " + str(self.interval_str) + "( " + str(self.subformula) + " )"


class UntilRecurrent(STLFormula):

    def __init__(self, subformula1, subformula2, interval=None, overlap=True):
        """
        Initializes an UntilRecurrent formula
        Args:
            subformula1 (STLFormula): The first subformula.
            subformula2 (STLFormula): The second subformula.
            interval (list, optional): The time interval for the Until operation. Defaults to [0, torch.inf].
        """
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0, 1])
        self.LARGE_NUMBER = 1e9
        # self.Alw = AlwaysRecurrent(subformula=Identity(name=str(self.subformula1))
        self.Alw = AlwaysRecurrent(GreaterThan(Predicate("x", lambda x: x), 0.0))

        if self.interval is None:
            self.hidden_dim = None
        elif interval[1] == torch.inf:
            self.hidden_dim = None
        else:
            self.hidden_dim = interval[1] + 1

    def _initialize_hidden_state(self, signal, padding=None, **kwargs):
        """
        Initializes the hidden state for the formula based on the input signal.
        Args:
            signal (torch.Tensor or tuple): The input signal, which can be a tensor or a tuple of tensors.
            padding (optional): Padding value, if any. Default is None.
            **kwargs: Additional keyword arguments for subformula evaluation.
        Returns:
            tuple: A tuple containing:
            - h (tuple): A tuple of two tensors representing the hidden state.
            - trace1 (torch.Tensor): The result of applying subformula1 to the signal.
            - trace2 (torch.Tensor): The result of applying subformula2 to the signal.
        """

        time_dim = 0  # assuming signal is [time_dim,...]

        if isinstance(signal, tuple):
            # for formula defined using Expression
            assert signal[0].shape[time_dim] == signal[1].shape[time_dim]
            trace1 = self.subformula1(signal[0], **kwargs)
            trace2 = self.subformula2(signal[1], **kwargs)
            n_time_steps = signal[0].shape[time_dim]
            device = signal[0].device
        else:
            # for formula defined using Predicate
            trace1 = self.subformula1(signal, **kwargs)
            trace2 = self.subformula2(signal, **kwargs)
            n_time_steps = signal.shape[time_dim]
            device = signal.device

        # compute hidden dim if signal length was needed
        if self.hidden_dim is None:
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self.interval = [0, n_time_steps - 1]
        elif self.interval[1] == torch.inf:
            self.interval[1] = n_time_steps - 1

        self.ones_array = torch.ones(self.hidden_dim, device=device)

        # set shift operations given hidden_dim
        self.M = torch.diag(torch.ones(self.hidden_dim - 1, device=device), diagonal=1)
        self.b = torch.zeros(self.hidden_dim, device=device)
        self.b[-1] = 1.0

        if self.hidden_dim == n_time_steps:
            pad_value = self.LARGE_NUMBER
        else:
            pad_value = -self.LARGE_NUMBER

        h1 = pad_value * self.ones_array
        h2 = -self.LARGE_NUMBER * self.ones_array
        return (h1, h2), trace1, trace2

    def _get_interval_indices(self):
        """
        Calculate the start and end indices for an interval.
        The start index is calculated as the negative value of the hidden dimension.
        The end index is calculated as the negative value of the first element in the interval.
        If the end index is zero, it returns None instead.
        Returns:
            tuple: A tuple containing the start index and the end index (or None if the end index is zero).
        """

        start_idx = -self.hidden_dim
        end_idx = -self.interval[0]

        return start_idx, (None if end_idx == 0 else end_idx)

    def _cell(self, state, hidden, **kwargs):
        """
        Perform a single step of the cell computation.
        Args:
            state (tuple): A tuple containing the current state (x1, x2).
            hidden (tuple): A tuple containing the hidden states (h1, h2).
            **kwargs: Additional keyword arguments for the operations.
        Returns:
            tuple: A tuple containing the output and the new hidden states (h1_new, h2_new).
        The function performs the following steps:
        1. Computes new hidden states (h1_new, h2_new) using the current hidden states and the state inputs.
        2. Computes h1_min by applying the Alw operation on the flipped h1_new.
        3. Computes z by taking the minimum of h1_min and h2_new over the specified interval.
        4. Computes the output by applying the maxish operation on z.
        """

        x1, x2 = state
        h1, h2 = hidden
        h1_new = self.M @ h1 + self.b * x1
        h1_min = self.Alw(h1_new.flip(0), **kwargs).flip(0)
        h2_new = self.M @ h2 + self.b * x2
        start_idx, end_idx = self._get_interval_indices()
        z = minish(torch.stack([h1_min, h2_new]), axis=0, keepdim=False, **kwargs)[
            start_idx:end_idx
        ]

        # def g_(carry, x):
        #     carry = maxish(torch.tensor([carry, x]), axis=0, keepdim=False, **kwargs)
        #     return carry, carry

        # output, _ = scan(g_,  -self.LARGE_NUMBER, z)
        output = maxish(z, axis=0, keepdim=False, **kwargs)

        return output, (h1_new, h2_new)

    def robustness_trace(self, signal, padding=None, **kwargs):
        """
        Function to run a signal through a cell T times, where T is the length of the signal in the time dimension

        Args:
            signal: input signal, size = [time_dim,]
            time_dim: axis corresponding to time_dim. Default: 0
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return:
            outputs: list of outputs
            states: list of hidden_states
        """
        hidden_state, trace1, trace2 = self._initialize_hidden_state(
            signal, padding=padding, **kwargs
        )

        def f_(hidden, state):
            o, hidden = self._cell(state, hidden, **kwargs)
            return hidden, o

        _, outputs_stack = scan(f_, hidden_state, torch.stack([trace1, trace2], axis=1))
        return outputs_stack

    def robustness(self, signal, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal: torch.Tensor or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: torch.Tensor, same as input with the time_dim removed.
        """
        return self.__call__(signal, **kwargs)[-1]
        # return jnp.rollaxis(self.__call__(signal, **kwargs), time_dim)[-1]

    def _next_function(self):
        """next function is the input subformulas. For visualization purposes"""
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") U (" + str(self.subformula2) + ")"


class DifferentiableAlways(STLFormula):
    """
    Represents a differentiable 'Always' temporal operator in Signal Temporal Logic (STL).
    Attributes:
        interval (tuple or None): The time interval for the 'Always' operator. If None, defaults to [0, inf].
        subformula (STLFormula): The subformula to which the 'Always' operator is applied.
    """

    def __init__(self, subformula, interval=None):
        """
        Initialize a Formula object.
        Args:
            subformula: The subformula associated with this formula.
            interval (optional): The interval for the formula. Defaults to None.
        """

        super().__init__()

        self.interval = interval
        self.subformula = subformula
        # self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal,
        t_start,
        t_end,
        scale=1.0,
        padding=None,
        large_number=1e9,
        delta=1e-3,
        **kwargs
    ):
        """
        Computes the robustness trace of a given signal over a specified time interval.
        Args:
            signal (torch.Tensor): The input signal tensor.
            t_start (float): The start time of the interval.
            t_end (float): The end time of the interval.
            scale (float, optional): Scaling factor for the smooth time mask. Default is 1.0.
            padding (str, optional): Padding method for the signal. Options are 'last', 'mean', or None. Default is None.
            large_number (float, optional): A large number used for masking. Default is 1e9.
            delta (float, optional): A small number used for thresholding. Default is 1e-3.
            **kwargs: Additional keyword arguments passed to the subformula and minish functions.
        Returns:
            torch.Tensor: The robustness trace of the signal.
        """
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(
            signal, padding=padding, large_number=large_number, **kwargs
        )
        T = signal.shape[time_dim]
        mask_value = large_number
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -mask_value
        signal_pad = torch.ones([T, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(
            T, t_start, t_end, scale, device=device
        )  # * (1 - delta) + delta
        padded_smooth_time_mask = torch.stack(
            [
                torch.concat(
                    [
                        torch.zeros(i, device=device),
                        smooth_time_mask,
                        torch.zeros(T - i, device=device),
                    ]
                )
                for i in range(T)
            ],
            1,
        )
        masked_signal_matrix = torch.where(
            padded_smooth_time_mask > delta,
            signal_padded * padded_smooth_time_mask,
            mask_value,
        )
        return minish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """next function is the input subformula. For visualization purposes"""
        return [self.subformula]

    def __str__(self):
        return "◻ [a,b] ( " + str(self.subformula) + " )"


class DifferentiableEventually(STLFormula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal,
        t_start,
        t_end,
        scale=1.0,
        padding=None,
        large_number=1e9,
        delta=1e-3,
        **kwargs
    ):
        """
        Computes the robustness trace of a given signal over a specified time interval.
        Args:
            signal (torch.Tensor): The input signal tensor.
            t_start (float): The start time of the interval.
            t_end (float): The end time of the interval.
            scale (float, optional): Scaling factor for the smooth time mask. Default is 1.0.
            padding (str, optional): Padding method for the signal. Options are 'last', 'mean', or None. Default is None.
            large_number (float, optional): A large number used for masking. Default is 1e9.
            delta (float, optional): A small number used for thresholding. Default is 1e-3.
            **kwargs: Additional keyword arguments passed to the subformula and minish functions.
        Returns:
            torch.Tensor: The robustness trace of the signal.
        """
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(
            signal, padding=padding, large_number=large_number, **kwargs
        )
        T = signal.shape[time_dim]
        mask_value = -large_number
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = mask_value
        signal_pad = torch.ones([T, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(
            T, t_start, t_end, scale, device=device
        )  # * (1 - delta) + delta
        padded_smooth_time_mask = torch.stack(
            [
                torch.concat(
                    [
                        torch.zeros(i, device=device),
                        smooth_time_mask,
                        torch.zeros(T - i, device=device),
                    ]
                )
                for i in range(T)
            ],
            1,
        )
        masked_signal_matrix = torch.where(
            padded_smooth_time_mask > delta,
            signal_padded * padded_smooth_time_mask,
            mask_value,
        )
        return maxish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """
        Returns a list containing the subformula attribute.
        Returns:
            list: A list with a single element, the subformula attribute.
        """
        return [self.subformula]

    def __str__(self):
        """
        Returns a string representation of the formula object.
        The string representation is in the format: "♢ [a,b] ( <subformula> )".
        Returns:
            str: The string representation of the formula.
        """
        return "♢ [a,b] ( " + str(self.subformula) + " )"
