import os
import re
import sys

from dataclasses import dataclass
from typing import List, Optional, Tuple


class ASTNode:
    pass


@dataclass(frozen=True)
class LiteralNode(ASTNode):
    value: any


@dataclass(frozen=True)
class LiteralStringNode(LiteralNode):
    value: str


@dataclass(frozen=True)
class LiteralIntNode(LiteralNode):
    value: int


@dataclass(frozen=True)
class IdentifierNode:
    name: str


@dataclass(frozen=True)
class ExpressionNode(ASTNode):
    expr: ASTNode
    args: Tuple[ASTNode]


@dataclass(frozen=True)
class StatementNode(ASTNode):
    expr: ExpressionNode
    name: Optional[IdentifierNode]
    block: Optional[ASTNode]


@dataclass(frozen=True)
class BlockNode:
    stmts: List[StatementNode]


@dataclass(frozen=True)
class ModuleNode:
    blocks: List[BlockNode]


@dataclass(frozen=True)
class AssignmentNode(ASTNode):
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True)
class CollectionNode(ASTNode):
    items: Tuple[ASTNode]


def tokenize(input_string):
    # Tokens to return
    tokens = []
    # The stack to keep track of indentation levels; start with 0 (the base level)
    indent_stack = [0]
    # Split the input into lines to process indentation
    lines = input_string.split("\n")

    # Regex to capture tokens within lines
    line_token_pattern = re.compile(r'("[^"]*"|\w+|\(|\)|\=|\#)')
    paren_level = 0

    for line in lines:
        # Check for empty or whitespace-only lines
        if not line.strip():
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
                if tokens[-1] == "NEWLINE":
                    tokens.pop()
                indent_stack.pop()
                tokens.append("DEDENT")
                tokens.append("NEWLINE")

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

            tokens.extend(line_tokens)

            # ignore newlines when in parenthesis
            if paren_level == 0:
                tokens.append("NEWLINE")

    if tokens[-1] == "NEWLINE":
        tokens.pop()
    # End of input - dedent back to the base level
    for _ in indent_stack[1:]:  # Skip the base level
        tokens.append("DEDENT")

    while tokens[-1] == "EMPTY":
        del tokens[-1]
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.next()

    def next(self):
        self.current = next(self.tokens, None)

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
        items.append(self.expr())
        while not self.accept(")"):
            items.append(self.expr())

        return CollectionNode(items=tuple(items))

    def literal(self):
        token: str = self.current
        if token == "true":
            node = LiteralNode(True)
        elif token == "false":
            node = LiteralNode(False)
        elif token.isnumeric():
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
        if self.accept("("):
            while not self.accept(")"):
                args.append(self.expr())
                self.accept(",")

        return ExpressionNode(expr=literal, args=tuple(args))

    def stmt(self):
        block = None
        name = None
        expr = self.expr()
        if self.accept("="):
            expr = AssignmentNode(left=expr, right=self.expr())
        if self.accept("as"):
            name = self.identifier()
        if self.accept("INDENT"):
            block = self.module()
            self.expect("DEDENT")
        return StatementNode(expr=expr, block=block, name=name)

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


class Interpreter:
    def __init__(self, variables=None, stack=None):
        if not variables:
            variables = dict()
        self.variables = variables.copy()

        if not stack:
            stack = []
        self.stack = stack.copy()

        self.commands = ["split", "int"]

    def run(self, ast_node):
        match ast_node:
            case ModuleNode(blocks):
                for block in blocks:
                    self.run(block)
            case BlockNode(stmts):
                for stmt in stmts:
                    self.run(stmt)
            case StatementNode(expr, name, block):
                if (
                    isinstance(expr, ExpressionNode)
                    and isinstance(expr.expr, IdentifierNode)
                    and expr.expr.name == "reduce"
                ):
                    data = self.stack.pop()
                    if not block:
                        raise Exception("Block expected in reduce")
                    self.run(expr.args[0])
                    result = self.stack.pop()
                    for d in data:
                        sub_interpretor = Interpreter(
                            self.variables.copy(), self.stack + [result, d]
                        )
                        result = sub_interpretor.run(block)
                    self.stack.append(result)
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
                if name:
                    self.variables[(name,)] = LiteralNode(value=self.stack[-1])

            case AssignmentNode(left, right):
                if isinstance(left, IdentifierNode):
                    self.variables[(left,)] = right
                elif isinstance(left, ExpressionNode):
                    full_ident = (left.expr, *left.args)
                    self.variables[full_ident] = right

            case CollectionNode(items):
                results = []
                for item in items:
                    self.run(item)
                    result = self.stack.pop()
                    results.append(result)

                self.stack.append(tuple(results))

            case ExpressionNode(expr, args):
                if isinstance(expr, LiteralNode):
                    return self.run(expr)
                elif (expr,) in self.variables:
                    self.run(self.variables[(expr,)])
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
                        if x[0] == expr
                        and all([isinstance(x, ExpressionNode) for x in x[1:]])
                        and (len(x) == len(args) + 1 or len(x) == len(args) + 2)
                    ]

                    if (expr, *args_as_literals) in self.variables:
                        self.run(self.variables[(expr, *args_as_literals)])
                    elif matching_exact_vars:
                        matching = matching_exact_vars[0]
                        func_vars = {}
                        if len(matching[1:]) > len(args):
                            args_as_literals.insert(0, LiteralNode(value=top))
                        for func_args, literal_arg in zip(
                            reversed(matching[1:]), reversed(args_as_literals)
                        ):
                            func_vars[(func_args.expr,)] = literal_arg
                            self.stack.pop()

                        sub_interpretor = Interpreter(
                            {**self.variables, **func_vars}, self.stack
                        )
                        self.stack.append(sub_interpretor.run(self.variables[matching]))
                    else:
                        match expr:
                            case IdentifierNode("_"):
                                pass
                            case IdentifierNode("_1"):
                                pass
                            case IdentifierNode("_2"):
                                self.stack.append(self.stack[-2])
                            case IdentifierNode("pyeval"):
                                code = self.stack.pop()
                                result = eval(
                                    code,
                                    {
                                        k[0].name: v.value
                                        for k, v in self.variables.items()
                                        if isinstance(v, LiteralNode)
                                    },
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
                            case IdentifierNode("index"):
                                i = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(data[i])
                            case IdentifierNode("get"):
                                index = self.stack.pop()
                                data = self.stack.pop()
                                self.stack.append(data[index])
                            case IdentifierNode("set"):
                                item = self.stack.pop()
                                index = self.stack.pop()
                                data = list(self.stack.pop())

                                data[index] = item
                                self.stack.append(tuple(data))
                            case _:
                                raise NotImplementedError(expr)

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
