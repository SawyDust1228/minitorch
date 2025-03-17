from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    delta_f = f(*vals[0:arg], vals[arg] + epsilon, *vals[arg + 1:len(vals)]) - f(*vals[0:arg], vals[arg] - epsilon, *vals[arg + 1:len(vals)])
    return delta_f / (2 * epsilon)
variable_count = 1


class Variable(Protocol): #协议
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    
    result = []
    visited = set()
    def visit(variable : Variable):
        for p in variable.parents:
            if not p.is_constant():
                visit(p)
        if variable.unique_id not in visited:
            visited.add(variable.unique_id)
            result.append(variable)
    visit(variable);result.reverse()
    assert variable.unique_id == result[0].unique_id
    return result
    
    # perminentMarked = []
    # temporaryMarked = []
    # result = []
    
    # def visit(v: Variable):
    #     if v.is_constant():
    #         return
        
    #     if v.unique_id in perminentMarked:
    #         return
        
    #     if v.unique_id in temporaryMarked:
    #         raise ValueError("Cycle detected")
      
    #     temporaryMarked.append(v.unique_id)
    #     if not v.is_leaf():
    #         for p in v.parents:
    #             visit(p)
    #     perminentMarked.append(v.unique_id)
    #     temporaryMarked.remove(v.unique_id)
    #     result.append(v)
        
    # visit(variable)
    
    # result.reverse()
    # assert result[0].unique_id == variable.unique_id
    
    # return result
    

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    visited = dict();visited[variable.unique_id] = deriv
    for v in topological_sort(variable):
        if v.is_leaf():
            v.accumulate_derivative(visited[v.unique_id])
        else:
            for p, d in v.chain_rule(visited[v.unique_id]):
                if p.is_constant():
                    continue
                if p.unique_id not in visited.keys():
                    visited[p.unique_id] = 0
                visited[p.unique_id] += d
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
