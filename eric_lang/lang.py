import multiprocessing
import os
import re
import sys
import math

from tqdm import tqdm

from dataclasses import dataclass
from typing import List, Optional, Tuple


class ASTNode:
    pass


@dataclass(frozen=True)
class LiteralNode(ASTNode):
    value: any


@dataclass(frozen=True)
class IdentifierNode:
    name: str


@dataclass(frozen=True)
class ExpressionNode(ASTNode):
    identifier: IdentifierNode
    args: Tuple[ASTNode]


@dataclass(frozen=True)
class StatementNode(ASTNode):
    expr: ExpressionNode
    names: List[IdentifierNode]
    block: Optional[ASTNode]


@dataclass(frozen=True)
class BlockNode:
    stmts: List[StatementNode]


@dataclass(frozen=True)
class ModuleNode:
    blocks: List[BlockNode]


@dataclass(frozen=True)
class AssignmentNode(ASTNode):
    left: ExpressionNode
    right: ASTNode


@dataclass(frozen=True)
class CollectionItemNode(ASTNode):
    item: ASTNode
    expand: bool


@dataclass(frozen=True)
class CollectionNode(ASTNode):
    items: Tuple[CollectionItemNode]


def tokenize(input_string):
    # Tokens to return
    tokens = []
    # The stack to keep track of indentation levels; start with 0 (the base level)
    indent_stack = [0]
    # Split the input into lines to process indentation
    lines = input_string.split("\n")

    # Regex to capture tokens within lines
    line_token_pattern = re.compile(r'("[^"]*"|\w+|-?[0-9]+|\(|\)|\=|\#|\||\,|\.{3})')
    paren_level = 0

    for line in lines:
        # Check for empty or whitespace-only lines
        if not line.strip() and paren_level == 0:
            if tokens and tokens[-1] == "NEWLINE":
                tokens.pop()
            if tokens and tokens[-1] != "EMPTY":
                tokens.append("EMPTY")
            continue

        # Ignore indents when in parenthsis
        if paren_level == 0:
            # Measure indentation by the leading whitespace
            indentation = len(line) - len(line.lstrip(" "))

            # If this line's indentation is greater than the stack's, it's an indent
            if indentation > indent_stack[-1]:
                if tokens[-1] == "NEWLINE":
                    tokens.pop()
                indent_stack.append(indentation)
                tokens.append("INDENT")

            # If this line's indentation is less, it's one or more dedents
            while indentation < indent_stack[-1]:
                t = None
                if tokens[-1] in ("NEWLINE", "EMPTY"):
                    t = tokens.pop()
                indent_stack.pop()
                tokens.append("DEDENT")
                tokens.append(t or "NEWLINE")

        # Tokenize the rest of the line
        line_tokens = line_token_pattern.findall(line.lstrip(" "))

        for t in line_tokens:
            if t == "(":
                paren_level += 1
            elif t == ")":
                paren_level -= 1

        if line_tokens and not line_tokens[0] == "#":
            if "#" in line_tokens:
                line_tokens = line_tokens[: line_tokens.index("#")]

            for i in range(len(line_tokens)):
                if line_tokens[i] == "|":
                    line_tokens[i] = "NEWLINE"
            tokens.extend(line_tokens)

            # ignore newlines when in parenthesis
            if paren_level == 0:
                tokens.append("NEWLINE")

    if tokens[-1] == "NEWLINE":
        tokens.pop()
    # End of input - dedent back to the base level
    for _ in indent_stack[1:]:  # Skip the base level
        t = None
        if tokens[-1] in ("NEWLINE", "EMPTY"):
            t = tokens.pop()
        indent_stack.pop()
        tokens.append("DEDENT")
        tokens.append(t or "NEWLINE")

    while tokens[-1] in ("EMPTY", "NEWLINE"):
        del tokens[-1]
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.next()

    def next(self):
        self.current = next(self.tokens, None)

    def peek(self, token):
        return self.current == token

    def accept(self, token):
        if self.current == token:
            self.next()
            return True
        return False

    def expect(self, token):
        if self.current == token:
            self.next()
            return True
        raise Exception(f"Unexpected token: {self.current}")

    def identifier(self):
        token = self.current
        self.current = next(self.tokens)
        return IdentifierNode(name=token)

    def list(self):
        self.expect("(")
        items = []
        if self.accept(")"):
            return CollectionNode(items=tuple())

        if self.accept("..."):
            items.append(CollectionItemNode(item=self.expr(), expand=True))
        else:
            items.append(CollectionItemNode(item=self.expr(), expand=False))
        while self.accept(","):
            if self.current == ")":
                break
            if self.accept("..."):
                items.append(CollectionItemNode(item=self.expr(), expand=True))
            else:
                items.append(CollectionItemNode(item=self.expr(), expand=False))
        self.expect(")")
        return CollectionNode(items=tuple(items))

    def literal(self):
        token: str = self.current
        if not token:
            raise Exception("Error when parsing literal, end of file")
        if token == "true":
            node = LiteralNode(True)
        elif token == "false":
            node = LiteralNode(False)
        elif token.isnumeric():
            node = LiteralNode(int(token))
        elif token.startswith("-") and token[1:].isnumeric():
            node = LiteralNode(int(token))
        elif token.startswith('"') and token.endswith('"'):
            node = LiteralNode(token.strip('"'))
        else:
            node = IdentifierNode(token)
        self.next()
        return node

    def expr(self):
        if self.current == "(":
            literal = self.list()
            return literal

        literal = self.literal()

        if isinstance(literal, LiteralNode):
            return literal

        args = []
        if self.peek("("):
            args = [x.item for x in self.list().items]

        return ExpressionNode(identifier=literal, args=tuple(args))

    def stmt(self):
        block = None
        names = []
        expr = self.expr()
        if self.accept("="):
            if self.accept("INDENT"):
                expr = AssignmentNode(left=expr, right=self.module())
                self.expect("DEDENT")
            else:
                expr = AssignmentNode(left=expr, right=self.expr())
        if self.accept("as"):
            names.append(self.identifier())
            while self.accept(","):
                names.append(self.identifier())
        if self.accept("INDENT"):
            block = self.module()
            self.expect("DEDENT")
        return StatementNode(expr=expr, block=block, names=names)

    def block(self):
        stmts = []
        stmts.append(self.stmt())
        while self.accept("NEWLINE"):
            stmts.append(self.stmt())
        return BlockNode(stmts=stmts)

    def module(self):
        blocks = []
        blocks.append(self.block())
        while self.accept("EMPTY"):
            blocks.append(self.block())

        return ModuleNode(blocks=blocks)

    def parse(self):
        module = self.module()
        return module


def process(args):
    d, variables, stack, block = args
    sub_interpretor = Interpreter(variables.copy(), stack + [d])
    result = sub_interpretor.run(block)
    return result


memo = {}


class Interpreter:
    def __init__(self, variables=None, stack=None):
        if not variables:
            variables = dict()
        self.variables = variables.copy()

        if not stack:
            stack = []
        self.stack = stack.copy()

    def run(self, ast_node):
        match ast_node:
            case ModuleNode(blocks):
                for block in blocks:
                    self.run(block)
            case BlockNode(stmts):
                for stmt in stmts:
                    self.run(stmt)
            case StatementNode(expr, names, block):
                if (
                    isinstance(expr, ExpressionNode)
                    and isinstance(expr.identifier, IdentifierNode)
                    and expr.identifier.name == "reduce"
                ):
                    data = self.stack.pop()
                    if not block:
                        raise Exception("Block expected in reduce")
                    self.run(expr.args[0])
                    result = self.stack.pop()
                    for d in data:
                        sub_interpretor = Interpreter(
                            self.variables.copy(), self.stack + [(result, d)]
                        )
                        result = sub_interpretor.run(block)
                    self.stack.append(result)
                elif (
                    isinstance(expr, ExpressionNode)
                    and isinstance(expr.identifier, IdentifierNode)
                    and expr.identifier.name == "filter"
                ):
                    data = self.stack.pop()
                    if not block:
                        raise Exception("Block expected in reduce")
                    results = []
                    for d in data:
                        sub_interpretor = Interpreter(
                            self.variables.copy(), self.stack + [d]
                        )
                        result = sub_interpretor.run(block)
                        if result is True:
                            results.append(d)
                    self.stack.append(tuple(results))
                elif (
                    isinstance(expr, ExpressionNode)
                    and isinstance(expr.identifier, IdentifierNode)
                    and expr.identifier.name == "parallel"
                ):
                    data = self.stack.pop()
                    if not block:
                        raise Exception("Block expected in reduce")
                    self.run(expr.args[0].item)
                    cores = self.stack.pop()

                    args = [
                        (d, self.variables.copy(), self.stack.copy(), block)
                        for d in data
                    ]
                    pool = multiprocessing.Pool(cores)
                    results = pool.map(process, args)
                    self.stack.append(tuple(results))
                elif (
                    isinstance(expr, ExpressionNode)
                    and isinstance(expr.identifier, IdentifierNode)
                    and expr.identifier.name == "memoize"
                ):
                    if not block:
                        raise Exception("Block expected in memo")
                    self.run(expr.args[0].item)
                    key = self.stack.pop()

                    if key in memo:
                        self.stack.append(memo[key])
                    else:
                        sub_interpretor = Interpreter(
                            self.variables.copy(), self.stack.copy()
                        )
                        result = sub_interpretor.run(block)
                        self.stack.append(result)
                        memo[key] = result
                else:
                    self.run(expr)
                    if block:
                        results = []
                        for data in self.stack.pop():
                            sub_interpretor = Interpreter(
                                self.variables.copy(), self.stack + [data]
                            )
                            results.append(sub_interpretor.run(block))
                        self.stack.append(tuple(results))
                if names:
                    if len(names) == 1:
                        name = names[0]
                        self.variables[(name,)] = LiteralNode(value=self.stack[-1])
                    else:
                        values = self.stack[-1]

                        if len(values) != len(names):
                            raise Exception(
                                "As name expansion did not match number of values"
                            )

                        for name, value in zip(names, values):
                            self.variables[(name,)] = LiteralNode(value=value)

            case AssignmentNode(left, right):
                if isinstance(left, IdentifierNode):
                    self.variables[(left,)] = right
                elif isinstance(left, ExpressionNode):
                    full_ident = (left.identifier, *left.args)
                    self.variables[full_ident] = right

            case CollectionNode(items):
                results = []
                for item in items:
                    self.run(item.item)
                    result = self.stack.pop()
                    if item.expand:
                        results.extend(result)
                    else:
                        results.append(result)

                self.stack.append(tuple(results))

            case ExpressionNode(identifier, args):
                if isinstance(identifier, LiteralNode):
                    return self.run(identifier)
                elif (identifier,) in self.variables:
                    self.run(self.variables[(identifier,)])
                elif identifier.name == "if":
                    if len(args) == 3:
                        condition = self.run(args[0])
                        if condition:
                            self.run(args[1])
                        else:
                            self.run(args[2])
                    elif len(args) == 2:
                        condition = self.stack.pop()
                        if condition:
                            self.run(args[0])
                        else:
                            self.run(args[1])
                elif identifier.name == "cond":
                    for i in range(0, len(args), 2):
                        condition, body = args[i], args[i+1]
                        if self.run(condition):
                            self.run(body)
                            break
                    else:
                        raise Exception("No condition matches")

                else:
                    args_as_literals = []
                    top = None
                    if self.stack:
                        top = self.stack[-1]
                    for arg in args:
                        args_as_literals.append(LiteralNode(value=self.run(arg)))

                    matching_exact_vars = [
                        x
                        for x in self.variables
                        if x[0] == identifier
                        and all([isinstance(x, ExpressionNode) for x in x[1:]])
                        and (len(x) == len(args) + 1 or len(x) == len(args) + 2)
                    ]
                    if (identifier, *args_as_literals) in self.variables:
                        self.run(self.variables[(identifier, *args_as_literals)])
                    elif (identifier, *args_as_literals) in memo:
                        self.run(memo[(identifier, *args_as_literals)])
                    elif matching_exact_vars:
                        matching = matching_exact_vars[0]
                        func_vars = {}
                        if len(matching[1:]) > len(args):
                            args_as_literals.insert(0, LiteralNode(value=top))
                        for func_args, literal_arg in zip(
                            reversed(matching[1:]), reversed(args_as_literals)
                        ):
                            func_vars[(func_args.identifier,)] = literal_arg
                            self.stack.pop()

                        sub_interpretor = Interpreter(
                            {**self.variables, **func_vars}, self.stack
                        )
                        result = sub_interpretor.run(self.variables[matching])

                        self.stack.append(result)
                    else:
                        match identifier:
                            case IdentifierNode("_"):
                                pass
                            case IdentifierNode("_1"):
                                pass
                            case IdentifierNode("_2"):
                                self.stack.append(self.stack[-2])
                            case IdentifierNode("pyeval"):
                                code = self.stack.pop()
                                variables = {
                                    k[0].name: v.value
                                    for k, v in self.variables.items()
                                    if isinstance(v, LiteralNode)
                                }
                                result = eval(
                                    code, {"math": math, "tqdm": tqdm, **variables}
                                )
                                self.stack.append(result)
                            case IdentifierNode("stdin"):
                                self.stack.append(sys.stdin.read().strip())
                            case IdentifierNode("split"):
                                delimiter = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(
                                    tuple(
                                        data.split(
                                            delimiter.strip('"').replace("\\n", "\n")
                                        )
                                    )
                                )
                            case IdentifierNode("int"):
                                self.stack.append(int(self.stack.pop()))
                            case IdentifierNode("print"):
                                print(self.stack[-1])
                            case IdentifierNode("sum"):
                                self.stack.append(sum(self.stack.pop()))
                            case IdentifierNode("get"):
                                index = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(data[index])
                            case IdentifierNode("getd"):
                                default = self.stack.pop()
                                index = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(
                                    data[index] if index < len(data) else default
                                )
                            case IdentifierNode("set"):
                                item = self.stack.pop()
                                index = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(
                                    data[:index] + (item,) + data[index + 1 :]
                                )
                            case IdentifierNode("remove"):
                                index = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(data[:index] + data[index + 1 :])
                            case IdentifierNode("append"):
                                item = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(data + (item,))
                            case _:
                                raise NotImplementedError(
                                    (identifier, *args_as_literals)
                                )

            case LiteralNode(value):
                self.stack.append(value)

            case _:
                raise NotImplementedError(ast_node)

        if self.stack:
            return self.stack[-1]


def format_ast(node: ASTNode, indent=0):
    match node:
        case ModuleNode(blocks):
            return "\n\n".join([format_ast(b, indent) for b in blocks])
        case BlockNode(stmts):
            return "\n".join([format_ast(s, indent) for s in stmts])
        case StatementNode(expr, name, block):
            expr_str = format_ast(expr, indent)
            if name:
                expr_str += (
                    " " * (30 - indent - len(expr_str)) + " as " + str(name.name)
                )
            if block:
                expr_str += "\n" + format_ast(block, indent + 4)
            return " " * indent + expr_str
        case AssignmentNode(left, right):
            return format_ast(left, indent) + " = " + format_ast(right, indent)
        case ExpressionNode(literal, args):
            expr_str = format_ast(literal, indent)
            if args:
                expr_str += (
                    "(" + ", ".join([format_ast(arg, indent + 4) for arg in args]) + ")"
                )
            return expr_str
        case IdentifierNode(name):
            return name
        case LiteralNode(value):
            if isinstance(value, str):
                return repr(value).replace("'", '"')
            return str(value)
        case CollectionNode(items):
            formatted_str = "(" + ", ".join([format_ast(item) for item in items]) + ")"
            if len(formatted_str) > 80:
                formatted_str = (
                    "(\n    "
                    + ",\n    ".join([format_ast(item) for item in items])
                    + ",\n)"
                )
            return formatted_str

        case _:
            raise NotImplementedError(node)


def main():
    op = sys.argv[1]

    # Tokenize the input string:
    with open(sys.argv[2]) as f:
        tokens = tokenize(f.read())
        if op.startswith("token"):
            for t in tokens:
                print(t)
            exit(0)
        parser = Parser(tokens)
        ast = parser.parse()

        if op.startswith("format"):
            print(format_ast(ast))
            exit(0)

    # load stdlib
    with open(os.path.join(os.path.dirname(__file__), "./stdlib.eric")) as f:
        tokens = tokenize(f.read())
        stdlib_module = Parser(tokens).parse()

    interpreter = Interpreter()
    interpreter.run(ModuleNode(blocks=[*stdlib_module.blocks, *ast.blocks]))
