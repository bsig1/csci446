from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Union, Dict, Tuple, Optional, Any, TypeAlias, Iterable, Callable, Set
import re

#Parameters
filepath = "Caves/easy/path_e2.txt"

# <editor-fold desc="Semantics and structure for FOL">

class TokenType(Enum):
    XOR = auto()
    OR = auto()
    AND = auto()
    NOT = auto()
    ALL = auto()
    ANY = auto()
    IMPLIES = auto()
    IFF = auto()
    IN = auto()

    TRUE = auto()
    FALSE = auto()
    IDENT = auto()

    LPAREN = auto()
    RPAREN = auto()
    COMMA  = auto()

    def __str__(self): return self.name


@dataclass(frozen=True)
class Token:
    type: TokenType
    name: Optional[str] = None  # used for variables quantifiers and predicates

class Lexer:
    _whitespace = re.compile(r"\s+")
    _identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    SYMBOLS = [
        ("(", TokenType.LPAREN),
        (")", TokenType.RPAREN),
        (",", TokenType.COMMA),
    ]

    # keywords, case-sensitive
    KEYWORDS = {
        "TRUE": TokenType.TRUE,
        "FALSE": TokenType.FALSE,

        "AND": TokenType.AND,
        "OR": TokenType.OR,
        "XOR": TokenType.XOR,
        "NOT": TokenType.NOT,

        "IMPLIES": TokenType.IMPLIES,
        "IFF": TokenType.IFF,

        "ALL": TokenType.ALL,
        "ANY": TokenType.ANY,
        "IN": TokenType.IN,
    }

    def tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        length = len(text)

        while i < length:
            # skip whitespace
            current = self._whitespace.match(text, i)
            if current:
                i = current.end()
                if i >= length:
                    break

            matched_symbol = False
            for symbol, token_type in self.SYMBOLS:
                if text.startswith(symbol, i):
                    tokens.append(Token(token_type))
                    i += len(symbol)
                    matched_symbol = True
                    break
            if matched_symbol:
                continue

            # Identifier / keyword
            current = self._identifier.match(text, i)
            if current:
                lex = current.group(0)
                i = current.end()

                # Keyword
                token_type = self.KEYWORDS.get(lex)
                if token_type is not None:
                    tokens.append(Token(token_type))
                else:
                    tokens.append(Token(TokenType.IDENT, name=lex))
                continue

            # fallback
            raise ValueError(f"Unexpected character at {i}: {repr(text[i])}")

        return tokens

# Terms
class LogicTerminal(Enum):
    U = auto()  # Unknown
    F = auto()  # False
    T = auto()  # True

    def __str__(self):
        if self is LogicTerminal.U: return "Unknown"
        if self is LogicTerminal.F: return "False"
        if self is LogicTerminal.T: return "True"
        return "NULL"

    def __bool__(self): return self is LogicTerminal.T


@dataclass(frozen=True)
class Variable:
    name: Any
    def __str__(self) -> str: return str(self.name)

@dataclass(frozen=True)
class Constant:
    value: Any = LogicTerminal.U
    def __str__(self) -> str: return str(self.value)
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Constant): return False
        return self.value == other.value

@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)  # must be immutable for hashing

    def __str__(self):
        return f"{self.name}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, Predicate): return False
        return self.name == other.name and self.args == other.args


class LogicOperator(Enum):
    XOR = auto()
    AND = auto()
    OR = auto()
    IFF = auto()
    IMPLIES = auto()

    def __str__(self): return self.name

class Quantifier(Enum):
    ANY = auto()
    ALL = auto()

@dataclass(frozen=True)
class Not:
    child: Any

    def __str__(self): return "Not "+str(self.child)

@dataclass(frozen=True)
class Operator:
    nodeType: LogicOperator
    children: List[Any] = field(default_factory=list)

@dataclass(frozen=True)
class QuantifierExpression:
    quantifier: Quantifier
    variables: Tuple[Variable, ...]
    domain: Any
    expression: 'Expression'

SimpleTerm: TypeAlias = Union[Variable, Constant, LogicTerminal, None]
Expression: TypeAlias = Union[Predicate, Not, Operator, QuantifierExpression, Constant, Variable]
Term = Union[SimpleTerm, Expression]


class Parser:
    def __init__(self):
        self.expression: List[Token] = []
        self.parse_index = 0

    def __call__(self, arg: Union[str,List[Token]]):
        if isinstance(arg, str):
            lex = Lexer()
            self.expression = lex.tokenize(arg)
        elif isinstance(arg, list):
            self.expression = arg
        return self.parse(self.expression)

    def parse(self, tokens: List[Token]) -> Term:
        self.expression = tokens
        self.parse_index = 0
        return self.parse_expression()

    def peek(self, k=0) -> Optional[Token]:
        i = self.parse_index + k
        return self.expression[i] if 0 <= i < len(self.expression) else None

    def peek_is(self, t: TokenType) -> bool:
        """
        Checks if next token exists and is of a certain type
        """
        tok = self.peek()
        return tok is not None and tok.type is t

    def eat(self) -> Optional[Token]:
        tok = self.peek()
        if tok is not None:
            self.parse_index += 1
        return tok

    def expect(self, token_type: TokenType) -> Token:
        tok = self.eat()
        if tok is None or tok.type is not token_type:
            raise ValueError(f"Expected {token_type}, got {tok}")
        return tok

    def parse_expression(self) -> Term:
        node = self._parse_iff()
        if self.peek() is not None:
            raise ValueError(f"Expression not empty after parsing")
        return node

    def _parse_iff(self) -> Term:
        node = self._parse_implies()
        while self.peek_is(TokenType.IFF):
            self.eat()
            rhs = self._parse_implies()
            node = self._reduce_iff(node, rhs)
        return node

    def _parse_implies(self) -> Term:
        left = self._parse_xor()
        if self.peek_is(TokenType.IMPLIES):
            self.eat()
            right = self._parse_implies()  # right-assoc
            return self.reduce_implies(left, right)
        return left

    def _parse_xor(self) -> Term:
        node = self._parse_or()
        while self.peek_is(TokenType.XOR):
            self.eat()
            rhs = self._parse_or()
            node = Operator(nodeType=LogicOperator.XOR, children=[node, rhs])
        return node

    def _parse_or(self) -> Term:
        node = self._parse_and()
        while self.peek_is(TokenType.OR):
            self.eat()
            rhs = self._parse_and()
            node = Operator(nodeType=LogicOperator.OR, children=[node, rhs])
        return node

    def _parse_and(self) -> Term:
        node = self._parse_not()
        while self.peek_is(TokenType.AND):
            self.eat()
            rhs = self._parse_not()
            node = Operator(nodeType=LogicOperator.AND, children=[node, rhs])
        return node

    def _parse_not(self) -> Term:
        if self.peek_is(TokenType.NOT):
            self.eat()
            return Not(child=self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> Term:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        # booleans
        if tok.type is TokenType.TRUE:
            self.eat()
            return Constant(LogicTerminal.T)
        if tok.type is TokenType.FALSE:
            self.eat()
            return Constant(LogicTerminal.F)

        #'(' expr ')'
        if tok.type is TokenType.LPAREN:
            self.eat()
            node = self._parse_iff()
            self.expect(TokenType.RPAREN)
            return node

        # Quantifier (ALL/ANY)
        if self.peek_is(TokenType.ALL) or self.peek_is(TokenType.ANY):
            return self._parse_quantifier()

        # Predicate or variable: IDENT [ '(' args ')' ]
        if tok.type is TokenType.IDENT:
            ident = self.eat()
            name = ident.name

            if self.peek_is(TokenType.LPAREN):
                self.eat()
                args: List[Term] = []
                if not self.peek_is(TokenType.RPAREN):
                    while True:
                        args.append(self._parse_term())
                        if self.peek_is(TokenType.COMMA):
                            self.eat()
                            continue
                        break
                self.expect(TokenType.RPAREN)
                return Predicate(name=name, args=tuple(args))

            # variable
            return Variable(name=name)

        raise ValueError(f"Unexpected token in atom: {tok}")

    def _parse_term(self) -> Term:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of arguments")
        if tok.type is TokenType.TRUE:
            self.eat()
            return Constant(LogicTerminal.T)
        if tok.type is TokenType.FALSE:
            self.eat()
            return Constant(LogicTerminal.F)
        if tok.type is TokenType.IDENT:
            return Variable(name=self.eat().name)
        if tok.type is TokenType.LPAREN:
            raise ValueError("Nested predicate arguments not supported")
        raise ValueError(f"Invalid token: {tok}")

    def _parse_quantifier(self) -> Term:
        quantifier_token = self.eat()
        quantifier = Quantifier.ALL if quantifier_token.type is TokenType.ALL else Quantifier.ANY

        # Variables: IDENT or '(' IDENT (',' IDENT)* ')'
        variables: List[Variable] = []
        if self.peek_is(TokenType.LPAREN):
            self.eat()
            while True:
                ident = self.expect(TokenType.IDENT)
                variables.append(Variable(name=ident.name))
                if self.peek_is(TokenType.COMMA):
                    self.eat()
                    continue
                break
            self.expect(TokenType.RPAREN)
        else:
            ident = self.expect(TokenType.IDENT)
            variables.append(Variable(name=ident.name))

        # IN domain
        self.expect(TokenType.IN)
        domain_token = self.expect(TokenType.IDENT)
        domain = domain_token.name

        if self.peek_is(TokenType.LPAREN):
            self.eat()
            body = self._parse_iff()
            self.expect(TokenType.RPAREN)
        else:
            body = self._parse_iff()

        return QuantifierExpression(quantifier=quantifier, variables=tuple(variables), domain=domain, expression=body)

    # Reductions for IMPLIES/IFF
    @staticmethod
    def reduce_implies(p: Term, q: Term) -> Term:
        # p -> q  ==  (!p) OR q
        return Operator(nodeType=LogicOperator.OR, children=[Not(p), q])

    def _reduce_iff(self, p: Term, q: Term) -> Term:
        # p <-> q  ==  (p -> q) AND (q -> p)
        return Operator(
            nodeType=LogicOperator.AND,
            children=[self.reduce_implies(p, q), self.reduce_implies(q, p)]
        )

    @staticmethod
    def pretty_print(node: Term, indent: str = "", is_last: bool = True):
        branch = "\\-- " if is_last else "| "
        next_indent = indent + ("    " if is_last else "|   ")

        # Operator
        if isinstance(node, Operator):
            label = str(node.nodeType.name)
            print(indent + branch + label)
            for i, child in enumerate(node.children):
                Parser.pretty_print(child, next_indent, i == len(node.children) - 1)
            return

        # NOT
        if isinstance(node, Not):
            print(indent + branch + "NOT")
            Parser.pretty_print(node.child, next_indent, True)
            return

        # Predicate
        if isinstance(node, Predicate):
            print(indent + branch + str(node))
            return

        # Constant (LogicTerminal)
        if isinstance(node, Constant):
            print(indent + branch + str(node.value))
            return

        # Variable
        if isinstance(node, Variable):
            print(indent + branch + f"Variable({node.name})")
            return

        # Quantifier
        if isinstance(node, QuantifierExpression):
            quantifier_name = str(node.quantifier.name)
            variables_str = ", ".join(v.name for v in node.variables)
            domain_str = node.domain
            label = f"{quantifier_name} {variables_str} IN {domain_str}"
            print(indent + branch + label)
            Parser.pretty_print(node.expression, next_indent, True)
            return

        # LogicTerminal
        if isinstance(node, LogicTerminal):
            print(indent + branch + str(node))
            return

        # Fallback
        print(indent + branch + f"{node}")



class ExpressionEvaluator:
    """
    Evaluate an AST
    - For propositional variables: look up in variable_environment (by variable name)
    - For predicates consult predicate_table
    - For quantifiers need domains
    """

    def __init__(
        self,
        root: Term,
        variable_environment: Optional[Dict[str, LogicTerminal]] = None,
        predicate_table: Optional[Dict[Tuple[str, Tuple[Any, ...]], LogicTerminal]] = None,
        domains: Optional[Dict[str, Union[Iterable,Callable]]] = None,
    ):
        self.root = root
        self.variable_environment = variable_environment or {}
        self.predicate_table = predicate_table or {}
        self.domains = domains or {}

        # current variable bindings
        self.bindings: Dict[str, Any] = {}

        self.evaluation: LogicTerminal = self.eval(root)

    def _resolve_domain(self, name: str) -> Iterable[Any]:
        if name not in self.domains:
            raise ValueError(f"Domain '{name}' not provided.")
        domain = self.domains[name]
        # If callable, pass current bindings so it can depend on bound vars
        return domain(self.bindings) if callable(domain) else domain

    @staticmethod
    def implies(p: Term, q: Term) -> Term:
        return Operator(LogicOperator.OR, [Not(p), q])

    @staticmethod
    def equivalence(p: Term, q: Term) -> LogicTerminal:
        """
        Evaluate (p -> q) ∧ (q -> p).
        """
        new_expression_tree = Operator(LogicOperator.AND, [ExpressionEvaluator.implies(p, q), ExpressionEvaluator.implies(q, p)])
        return ExpressionEvaluator(new_expression_tree).evaluation

    @staticmethod
    def _not(arg: LogicTerminal) -> LogicTerminal:
        match arg:
            case LogicTerminal.F: return LogicTerminal.T
            case LogicTerminal.T: return LogicTerminal.F
            case _: return LogicTerminal.U

    @staticmethod
    def _and(args: List[LogicTerminal]) -> LogicTerminal:
        """
        True IFF all entries are True. If at least one is false, false, if at least one is unknown, unknown
        """
        if any(arg is LogicTerminal.F for arg in args): return LogicTerminal.F
        if any(arg is LogicTerminal.U for arg in args): return LogicTerminal.U
        return LogicTerminal.T

    @staticmethod
    def _or(args: List[LogicTerminal]) -> LogicTerminal:
        """
        True IFF any entries are True. Then Unknown if any are unknown, false otherwise
        """
        if any(arg is LogicTerminal.T for arg in args): return LogicTerminal.T
        if any(arg is LogicTerminal.U for arg in args): return LogicTerminal.U
        return LogicTerminal.F

    @staticmethod
    def _xor(args: List[LogicTerminal]) -> LogicTerminal:
        """
        True if only one entry is true, with no unknown entries.
        If the number of true args is <=1 then the evaluation can be unknown if unknown is present.
        Otherwise false
        """
        true_count = sum(int(bool(arg)) for arg in args)
        if true_count > 1:
            return LogicTerminal.F
        if LogicTerminal.U in args:
            return LogicTerminal.U
        return LogicTerminal.T if true_count == 1 else LogicTerminal.F



    def eval(self, node: Term) -> LogicTerminal:
        # Constants / raw terminals
        if isinstance(node, Constant): return node.value
        if isinstance(node, LogicTerminal): return node

        # Propositional variable alone
        if isinstance(node, Variable):
            return self._eval_variable(node)

        # Predicates (possibly with vars)
        if isinstance(node, Predicate):
            return self._eval_predicate(node)

        # Unary NOT
        if isinstance(node, Not):
            return self._not(self.eval(node.child))

        # Operators (AND/OR/XOR/IMPLIES/IFF)
        if isinstance(node, Operator):
            op = node.nodeType
            vals = [self.eval(c) for c in node.children]
            if op is LogicOperator.AND:     return self._and(vals)
            if op is LogicOperator.OR:      return self._or(vals)
            if op is LogicOperator.XOR:     return self._xor(vals)
            if op is LogicOperator.IMPLIES: # reduce here just in case
                if len(vals) != 2:
                    return LogicTerminal.U
                return self._or([self._not(vals[0]), vals[1]])
            if op is LogicOperator.IFF:
                if len(vals) != 2:
                    return LogicTerminal.U
                # (a↔b) := (a->b) ∧ (b->a)
                ab = self._or([self._not(vals[0]), vals[1]])
                ba = self._or([self._not(vals[1]), vals[0]])
                return self._and([ab, ba])
            raise ValueError(f"Unsupported operator: {op}")

        # Quantifiers
        if isinstance(node, QuantifierExpression):
            return self._eval_quantifier(node)

        # Fallback
        raise ValueError(f"Cannot evaluate node of type {type(node).__name__}")

    def _eval_variable(self, variable: Variable) -> LogicTerminal:
        if variable.name in self.bindings:
            bound = self.bindings[variable.name]
            if isinstance(bound, bool):
                return LogicTerminal.T if bound else LogicTerminal.F
            if isinstance(bound, LogicTerminal):
                return bound
            return LogicTerminal.U
        # Propositional variable lookup by name
        return self.variable_environment.get(str(variable.name), LogicTerminal.U)

    def _eval_predicate(self, predicate: Predicate) -> LogicTerminal:
        # replace Variable by their bound values if present.
        predicate_args: Tuple[Any, ...] = tuple(self.bindings.get(arg.name, arg) if isinstance(arg, Variable) else arg for arg in predicate.args)

        # lookup in predicate table
        key = (predicate.name, predicate_args)
        if key in self.predicate_table:
            return self.predicate_table[key]

        # unknown if lookups do not yield truth value
        return LogicTerminal.U

    def _eval_quantifier(self, quantifier: QuantifierExpression) -> LogicTerminal:
        # get domain
        if isinstance(quantifier.domain, str):
            domain_iterable = self._resolve_domain(quantifier.domain)
        else:
            domain_iterable = quantifier.domain

        variables = list(quantifier.variables)
        if not variables:
            return self.eval(quantifier.expression)

        # tries combinations of facts to find truth value
        def assign_and_eval(arg_index: int) -> LogicTerminal:
            if arg_index == len(variables):
                return self.eval(quantifier.expression)  # all variables bound
            variable = variables[arg_index]
            result_accumulator: List[LogicTerminal] = []
            for element in domain_iterable:
                self.bindings[variable.name] = element
                value = assign_and_eval(arg_index + 1) # Recurse and assign next value
                result_accumulator.append(value)

                if quantifier.quantifier is Quantifier.ALL and value is LogicTerminal.F:
                    del self.bindings[variable.name]
                    return LogicTerminal.F
                if quantifier.quantifier is Quantifier.ANY and value is LogicTerminal.T:
                    del self.bindings[variable.name]
                    return LogicTerminal.T

            # Clean binding for this variable
            if variable.name in self.bindings:
                del self.bindings[variable.name]

            # Aggregate Unknowns
            if quantifier.quantifier is Quantifier.ALL:
                return self._and(result_accumulator)
            else:
                return self._or(result_accumulator)

        return assign_and_eval(0)


# </editor-fold>




# <editor-fold desc="Wumpis World">
class Safety(Enum):
    SAFE = auto()
    RISKY = auto()
    UNSAFE = auto()
    UNKNOWN = auto()

class PuzzleParser:
    def __init__(self):
        self.size: Tuple[int,int] = (-1, -1)
        self.arrows: int = -1
        self.path: Dict[Tuple[int, int], Dict[str, bool]] = {} # Relates Position to boolean values of Breeze and Stench
        self.query:Tuple[int,int] = (-1,-1)
        self.resolution: Safety = Safety.UNKNOWN
        self.file_read = False



        try:
            self.parse_puzzle()
            self.file_read = True
        except FileNotFoundError:
            print(f"File {filepath} not found")
            self.file_read = False
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Bad File: {filepath}")
            self.file_read = False

    def __bool__(self):
        return self.file_read

    def parse_puzzle(self):
        with open(filepath) as file:
            path: List[str] = []
            for raw in file.readlines():
                line = raw.strip()
                if line.startswith('GRID: '):
                    grid = line.replace('GRID: ', '')
                    self.size = tuple(map(int, grid.split('x')))
                if line.startswith('ARROWS: '):
                    self.arrows = int(line.replace('ARROWS: ', ''))
                if line.startswith('QUERY: '):
                    query = line.replace('QUERY: (', '')[:-1]
                    self.query = tuple(map(int, query.split(',')))
                if line.startswith('RESOLUTION: '):
                    self.resolution = Safety[line.replace('RESOLUTION: ', '')]
                if line.startswith('('):
                    path.append(line)

            for step in path:
                position, breeze, stench = tuple(step.split())
                position = position[1:-1]
                row, col = tuple(map(int, position.split(',')))
                breeze = breeze[-1] == 'T'
                stench = stench[-1] == 'T'

                self.path[(row, col)] = {"Breeze": breeze, "Stench": stench}

    def get_size(self):
        return self.size

    def get_path(self):
        return self.path



class KnowledgeBase:
    def __init__(self):
        self.rules: Set[Tuple[Term]] = set() # Each tuple is a set of disjuncts
        self.facts: Set[Tuple[Term]] = set()

        self.puzzle = PuzzleParser()
        if not self.puzzle: return

        self.logic_parser = Parser()

        # -- Safe iff (not wumpus and not pit)
        self.add_rule("NOT Safe(x)", "NOT Pit(x)") # Safe => Not Pit
        self.add_rule("NOT Safe(x)", "NOT Wumpus(x)") # Safe => Not Wumpus
        self.add_rule("NOT Wumpus(x)","NOT Safe(x)") # Wumpus => not safe
        self.add_rule("Wumpus(x)","Pit(x)","Safe(x)") # (not wumpus and not pit) => safe

        self.get_puzzle_facts()

    def __contains__(self, item: Tuple[Term]) -> bool:
        clauses = self.facts | self.rules
        if isinstance(item, tuple):
            return item in clauses
        else:
            return (item,) in clauses

    def add_rule(self,*rule):
        """
        :param rule: accepts any number of arguments, these are taken as disjuncts to each other
        """
        new_rule = []
        for arg in rule:
            if isinstance(arg, str):
                new_rule.append(self.logic_parser(arg))
            elif isinstance(arg, Term):
                new_rule.append(arg)

        self.rules.add(tuple(new_rule))

    def add_fact(self, *fact):
        """
        :param fact: accepts any number of arguments, these are taken as disjuncts to each other
        """
        new_fact = []
        for arg in fact:
            if isinstance(arg, str):
                new_fact.append(self.logic_parser(arg))
            elif isinstance(arg, Term):
                new_fact.append(arg)

        self.facts.add(tuple(new_fact))


    def get_neighbors(self,square: tuple)->List[Tuple[int,int]]:
        neighbors: List[Tuple[int,int]] = []
        xbounds,ybounds = zip((0,0),self.puzzle.get_size())

        for diff in [(1,0),(-1,0),(0,1),(0,-1)]:
            neighbor = (square[0] + diff[0], square[1] + diff[1])
            if neighbor[0] < xbounds[0] or neighbor[0] >= xbounds[1]:
                continue
            if neighbor[1] < ybounds[0] or neighbor[1] >= ybounds[1]:
                continue
            neighbors.append(neighbor)

        return neighbors

    def get_puzzle_facts(self):
        path = self.puzzle.get_path()
        for key in path:
            self.add_fact(Predicate("Safe",(Constant(value=key),)))
            for sense in ["Stench", "Breeze"]:
                fact = Predicate(sense, (Constant(value=key),))
                if path[key][sense]:
                    self.add_fact(fact)
                else:
                    self.add_fact(Not(fact))



class InferenceEngine:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.add_neighbor_info()
        self.answer = self.query(self.kb.puzzle.query)
        print(self.answer)

    def add_neighbor_info(self):
        hazards = {
            "Stench":"Wumpus",
            "Breeze":"Pit",
        }
        new_disjuncts = []
        for disjunct in self.kb.facts:
            if len(disjunct) !=1: continue
            negated = False
            fact = disjunct[0]
            if isinstance(fact, Not):
                negated = True
                fact = fact.child

            if (not isinstance(fact, Predicate)) or (not fact.name in hazards): continue

            neighbors = self.kb.get_neighbors(fact.args[0].value)
            if negated: # Not(stench) means all adjacent are safe
                for neighbor in neighbors:
                    new_disjuncts.append(Not(Predicate(hazards[fact.name], (Constant(neighbor),))))
                continue

            new_disjuncts.append( # stench means at least one adjacent is wumpis
                tuple([Predicate(hazards[fact.name], (Constant(neighbor),)) for neighbor in neighbors])
            )
        for clause in new_disjuncts:
            if isinstance(clause, tuple):
                self.kb.add_fact(*clause)
            else:
                self.kb.add_fact(clause)

    def occurs(self, variable: Variable, term: Term, theta: Dict[Term, Term]) -> bool:
        term = self.substitute(term, theta)
        if variable == term: return True # Checks if term is the same variable once walk is ran
        if isinstance(term, Not):
            return self.occurs(variable, term.child, theta)
        if isinstance(term, Predicate):
            return any(self.occurs(variable, a, theta) for a in term.args)
        return False

    def substitute(self,term: Term, theta: Dict[Term, Term]) -> Term:
        """
        digs through theta to find a definition for a variable
        """
        if isinstance(term, Variable) and term in theta:
            return self.substitute(theta[term], theta) # may need multiple layers of definitions
        if isinstance(term, Constant):
            return term
        if isinstance(term, Not):
            return Not(self.substitute(term.child, theta))
        if isinstance(term, Predicate):
            new_args = tuple(self.substitute(a, theta) for a in term.args)
            return Predicate(term.name, new_args)
        return term

    def unify_var(self, variable: Term, term: Term, theta: Dict[Term, Term]) -> Optional[Dict[Term, Term]]:
        term = self.substitute(term, theta)
        if variable == term:
            return theta
        if self.occurs(variable, term, theta):
            return None
        theta[variable] = term
        return theta

    def unify(self, a: Term, b: Term, theta: Optional[Dict[Term, Term]] = None) -> Optional[Dict[Term, Term]]:
        """
        unification for Variables, Constants, Predicates, and Not(Predicate).
        """
        if theta is None:
            theta = {}

        a = self.substitute(a, theta)
        b = self.substitute(b, theta)

        # identical after walking
        if a == b:
            return theta

        # variables
        if isinstance(a,Variable):
            return self.unify_var(a, b, theta)
        if isinstance(b,Variable):
            return self.unify_var(b, a, theta)

        # nots
        if isinstance(a,Not) and isinstance(b,Not):
            return self.unify(a.child, b.child, theta)

        # predicates
        if isinstance(a,Predicate) and isinstance(b,Predicate):
            function_a, args_a = a.name,a.args
            function_b, args_b = b.name,b.args
            if function_a != function_b or len(args_a) != len(args_b):
                return None
            for ai, bi in zip(args_a, args_b):
                theta = self.unify(ai, bi, theta)
                if theta is None:
                    return None
            return theta

        # constants must match exactly
        if isinstance(a,Constant) and isinstance(b,Constant):
            return theta if a == b else None

        # structure does not match
        return None

    @staticmethod
    def complements(a: Term,b: Term):
        if isinstance(a,Not) and (not isinstance(b,Not)):
            a = a.child
            return b,a
        if isinstance(b,Not) and (not isinstance(a,Not)):
            b = b.child
            return a,b
        return None


    def resolve_clauses(self, clause1, clause2):
        """
        Returns a list of resolvent clauses
        """
        outputs = []

        for left in clause1:
            for right in clause2:
                complements = self.complements(left, right)
                if complements is None: # needs opposite parity for resolution
                    continue

                pos, neg = complements  # neg is the Not clause
                theta = self.unify(pos,neg)
                if theta is None:
                    continue

                # Remove the resolved literals and apply theta to the rest
                new_literals = [self.substitute(term, theta) for term in (clause1 + clause2) if term not in (left, right)]
                # Avoid duplicates
                resolvent = tuple(set(new_literals))
                outputs.append(resolvent)

        return outputs

    def resolution(self, *assumptions):
        clauses = set(self.kb.facts) | set(self.kb.rules)
        for assumption in assumptions:
            clauses.add(assumption if isinstance(assumption, tuple) else (assumption,))

        while True:
            new_resolutions = set()
            clause_list = list(clauses)
            for i in range(len(clause_list)):
                for j in range(i + 1, len(clause_list)):
                    for resolution in self.resolve_clauses(clause_list[i], clause_list[j]):
                        if not resolution:
                            # self.print_knowledge(clauses | new_resolutions)
                            print("Derived empty clause!")
                            return True
                        if resolution not in clauses:
                            new_resolutions.add(resolution)
            if not new_resolutions:
                return False
            clauses |= new_resolutions

    def print_knowledge(self, knowledge):
        for clause in knowledge:
            print(" v ".join([str(subclause) for subclause in clause]))

    def query(self, cell: Tuple[int, int]) -> Safety:
        safe = Predicate("Safe",(Constant(cell),))
        unsafe = Predicate("Unsafe",(Constant(cell),))

        if (safe,) in self.kb:
            return Safety.SAFE
        if (unsafe,) in self.kb:
            return Safety.UNSAFE

        if self.resolution(safe): # If there is a contradiction
            return Safety.UNSAFE

        if self.resolution(unsafe):
            return Safety.SAFE

        return Safety.RISKY

class OutputWriter:
    # TODO: Write this class
    def __init__(self):
        self.engine = InferenceEngine()
        self.knowledge = self.engine.kb

    def write_result(self, kb: KnowledgeBase) -> None:
        """
        Emit a final report describing how the puzzle was solved:
        - metrics
        - facts used
        - (optionally) rules and/or proof traces
        """
        pass

# </editor-fold>

ie = InferenceEngine()

