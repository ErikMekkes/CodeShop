

WHITESPACE_TYPE = 0
SEPARATOR_TYPE = 1
OPERATOR_TYPE = 2
LITERAL_TYPE = 3
KEYWORD_TYPE = 4

LANGUAGE_DEFINED = {
    SEPARATOR_TYPE : True,
    OPERATOR_TYPE : True,
    LITERAL_TYPE : True,
    KEYWORD_TYPE : True,
}

UNDEFINED_TOKEN_TYPE = {
    WHITESPACE_TYPE : True,
}

go_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "..." : SEPARATOR_TYPE,
    # Operator Tokens
    "+" : OPERATOR_TYPE,    
    "&" : OPERATOR_TYPE,     
    "+=" : OPERATOR_TYPE,    
    "&=" : OPERATOR_TYPE,     
    "&&" : OPERATOR_TYPE,    
    "==" : OPERATOR_TYPE,    
    "!=" : OPERATOR_TYPE,    
    "-" : OPERATOR_TYPE,    
    "|" : OPERATOR_TYPE,     
    "-=" : OPERATOR_TYPE,    
    "|=" : OPERATOR_TYPE,     
    "||" : OPERATOR_TYPE,    
    "<" : OPERATOR_TYPE,     
    "<=" : OPERATOR_TYPE,    
    "*" : OPERATOR_TYPE,    
    "^" : OPERATOR_TYPE,     
    "*=" : OPERATOR_TYPE,    
    "^=" : OPERATOR_TYPE,     
    "<-" : OPERATOR_TYPE,    
    ">" : OPERATOR_TYPE,     
    ">=" : OPERATOR_TYPE,    
    "/" : OPERATOR_TYPE,    
    "<<" : OPERATOR_TYPE,    
    "/=" : OPERATOR_TYPE,    
    "<<=" : OPERATOR_TYPE,    
    "++" : OPERATOR_TYPE,    
    "=" : OPERATOR_TYPE,     
    ":=" : OPERATOR_TYPE,    
    "%" : OPERATOR_TYPE,    
    ">>" : OPERATOR_TYPE,    
    "%=" : OPERATOR_TYPE,    
    ">>=" : OPERATOR_TYPE,    
    "--" : OPERATOR_TYPE,    
    "!" : OPERATOR_TYPE,     
    "&^" : OPERATOR_TYPE,          
    "&^=" : OPERATOR_TYPE,          
    "~" : OPERATOR_TYPE,
    # reserved literals
    "nil" : LITERAL_TYPE,
    "true" : LITERAL_TYPE,
    "false" : LITERAL_TYPE,
    # Reserved Keyword Tokens    "break" : True,
    "default" : KEYWORD_TYPE,
    "func" : KEYWORD_TYPE,
    "interface" : KEYWORD_TYPE,
    "select" : KEYWORD_TYPE,
    "case" : KEYWORD_TYPE,
    "defer" : KEYWORD_TYPE,
    "go" : KEYWORD_TYPE,
    "map" : KEYWORD_TYPE,
    "struct" : KEYWORD_TYPE,
    "chan" : KEYWORD_TYPE,
    "else" : KEYWORD_TYPE,
    "goto" : KEYWORD_TYPE,
    "package" : KEYWORD_TYPE,
    "switch" : KEYWORD_TYPE,
    "const" : KEYWORD_TYPE,
    "fallthrough" : KEYWORD_TYPE,
    "if" : KEYWORD_TYPE,
    "range" : KEYWORD_TYPE,
    "type" : KEYWORD_TYPE,
    "continue" : KEYWORD_TYPE,
    "for" : KEYWORD_TYPE,
    "import" : KEYWORD_TYPE,
    "return" : KEYWORD_TYPE,
    "var" : KEYWORD_TYPE,
}

java_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    "ĠĊ" : WHITESPACE_TYPE,
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    ");" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "();" : SEPARATOR_TYPE,
    "());" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "[]" : SEPARATOR_TYPE,
    #'("\\' : SEPARATOR_TYPE,   # user customized string
    #'Ġ"' : SEPARATOR_TYPE,     # 
    #'");' : SEPARATOR_TYPE,
    # "<" : SEPARATOR_TYPE,     # like List<Type>, also an operator
    # ">" : SEPARATOR_TYPE,
    "..." : SEPARATOR_TYPE,
    "@" : SEPARATOR_TYPE,
    "::" : SEPARATOR_TYPE,
    # Operator Tokens
    "=" : OPERATOR_TYPE,
    ">" : OPERATOR_TYPE,
    "<" : OPERATOR_TYPE,
    "!" : OPERATOR_TYPE,
    "~" : OPERATOR_TYPE,
    "?" : OPERATOR_TYPE,
    ":" : OPERATOR_TYPE,
    "->" : OPERATOR_TYPE,
    "==" : OPERATOR_TYPE,
    ">=" : OPERATOR_TYPE,
    "<=" : OPERATOR_TYPE,
    "!=" : OPERATOR_TYPE,
    "&&" : OPERATOR_TYPE,
    "||" : OPERATOR_TYPE,
    "++" : OPERATOR_TYPE,
    "--" : OPERATOR_TYPE,
    "+" : OPERATOR_TYPE,
    "-" : OPERATOR_TYPE,
    "*" : OPERATOR_TYPE,
    "/" : OPERATOR_TYPE,
    "&" : OPERATOR_TYPE,
    "|" : OPERATOR_TYPE,
    "^" : OPERATOR_TYPE,
    "%" : OPERATOR_TYPE,
    "<<" : OPERATOR_TYPE,
    ">>" : OPERATOR_TYPE,
    ">>>" : OPERATOR_TYPE,
    "+=" : OPERATOR_TYPE,
    "-=" : OPERATOR_TYPE,
    "*=" : OPERATOR_TYPE,
    "/=" : OPERATOR_TYPE,
    "&=" : OPERATOR_TYPE,
    "|=" : OPERATOR_TYPE,
    "^=" : OPERATOR_TYPE,
    "%=" : OPERATOR_TYPE,
    "<<=" : OPERATOR_TYPE,
    ">>=" : OPERATOR_TYPE,
    ">>>=" : OPERATOR_TYPE,
    # reserved literals
    "null" : LITERAL_TYPE,
    "true" : LITERAL_TYPE,
    "false" : LITERAL_TYPE,
    # Reserved Keyword Tokens
    "abstract" : KEYWORD_TYPE,
    "continue" : KEYWORD_TYPE,
    "for" : KEYWORD_TYPE,
    "new" : KEYWORD_TYPE,
    "switch" : KEYWORD_TYPE,
    "assert" : KEYWORD_TYPE,
    "default" : KEYWORD_TYPE,
    "if" : KEYWORD_TYPE,
    "package" : KEYWORD_TYPE,
    "synchronized" : KEYWORD_TYPE,
    "boolean" : KEYWORD_TYPE,
    "do" : KEYWORD_TYPE,
    "goto" : KEYWORD_TYPE,
    "private" : KEYWORD_TYPE,
    "this" : KEYWORD_TYPE,
    "break" : KEYWORD_TYPE,
    "double" : KEYWORD_TYPE,
    "implements" : KEYWORD_TYPE,
    "protected" : KEYWORD_TYPE,
    "throw" : KEYWORD_TYPE,
    "byte" : KEYWORD_TYPE,
    "else" : KEYWORD_TYPE,
    "import" : KEYWORD_TYPE,
    "public" : KEYWORD_TYPE,
    "throws" : KEYWORD_TYPE,
    "case" : KEYWORD_TYPE,
    "enum" : KEYWORD_TYPE,
    "instanceof" : KEYWORD_TYPE,
    "return" : KEYWORD_TYPE,
    "transient" : KEYWORD_TYPE,
    "catch" : KEYWORD_TYPE,
    "extends" : KEYWORD_TYPE,
    "int" : KEYWORD_TYPE,
    "short" : KEYWORD_TYPE,
    "try" : KEYWORD_TYPE,
    "char" : KEYWORD_TYPE,
    "final" : KEYWORD_TYPE,
    "interface" : KEYWORD_TYPE,
    "static" : KEYWORD_TYPE,
    "void" : KEYWORD_TYPE,
    "class" : KEYWORD_TYPE,
    "finally" : KEYWORD_TYPE,
    "long" : KEYWORD_TYPE,
    "strictfp" : KEYWORD_TYPE,
    "volatile" : KEYWORD_TYPE,
    "const" : KEYWORD_TYPE,
    "float" : KEYWORD_TYPE,
    "native" : KEYWORD_TYPE,
    "super" : KEYWORD_TYPE,
    "while" : KEYWORD_TYPE,
    "_" : KEYWORD_TYPE,
    "exports" : KEYWORD_TYPE,
    "opens" : KEYWORD_TYPE,
    "requires" : KEYWORD_TYPE,
    "uses" : KEYWORD_TYPE,
    "module" : KEYWORD_TYPE,
    "permits" : KEYWORD_TYPE,
    "sealed" : KEYWORD_TYPE,
    "var" : KEYWORD_TYPE,
    "non-sealed" : KEYWORD_TYPE,
    "provides" : KEYWORD_TYPE,
    "to" : KEYWORD_TYPE,
    "with" : KEYWORD_TYPE,
    "open" : KEYWORD_TYPE,
    "record" : KEYWORD_TYPE,
    "transitive" : KEYWORD_TYPE,
    "yield" : KEYWORD_TYPE,
}


julia_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    "ĠĊ" : WHITESPACE_TYPE,
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    ");" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "();" : SEPARATOR_TYPE,
    "());" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "[]" : SEPARATOR_TYPE,
    ".." : SEPARATOR_TYPE,
    "::" : SEPARATOR_TYPE,
    "<|" : SEPARATOR_TYPE,
    "|>" : SEPARATOR_TYPE,
    # Operator Tokens
    # arithmetic
    "+" : OPERATOR_TYPE,
    "-" : OPERATOR_TYPE,
    "*" : OPERATOR_TYPE,
    "/" : OPERATOR_TYPE,
    "÷" : OPERATOR_TYPE,
    "\\" : OPERATOR_TYPE,
    "//" : OPERATOR_TYPE,
    "^" : OPERATOR_TYPE,
    "%" : OPERATOR_TYPE,
    "√" : OPERATOR_TYPE,
    # boolean operators
    "!" : OPERATOR_TYPE,
    "&&" : OPERATOR_TYPE,
    "||" : OPERATOR_TYPE,
    # bitwise operators
    "~" : OPERATOR_TYPE,
    "&" : OPERATOR_TYPE,
    "|" : OPERATOR_TYPE,
    "⊻" : OPERATOR_TYPE,
    "xor" : OPERATOR_TYPE,
    "⊼" : OPERATOR_TYPE,
    "nand" : OPERATOR_TYPE,
    "⊽" : OPERATOR_TYPE,
    "nor" : OPERATOR_TYPE,
    ">>>" : OPERATOR_TYPE,
    "<<" : OPERATOR_TYPE,
    ">>" : OPERATOR_TYPE,
    # updating operators
    "=" : OPERATOR_TYPE,
    "+=" : OPERATOR_TYPE,
    "-=" : OPERATOR_TYPE,
    "*=" : OPERATOR_TYPE,
    "/=" : OPERATOR_TYPE,
    "\\=" : OPERATOR_TYPE,
    "÷=" : OPERATOR_TYPE,
    "%=" : OPERATOR_TYPE,
    "^=" : OPERATOR_TYPE,
    "&=" : OPERATOR_TYPE,
    "|=" : OPERATOR_TYPE,
    "⊻=" : OPERATOR_TYPE,
    ">>>=" : OPERATOR_TYPE,
    ">>=" : OPERATOR_TYPE,
    "<<=" : OPERATOR_TYPE,
    # comparison
    "=" : OPERATOR_TYPE,
    "~" : OPERATOR_TYPE,
    "?" : OPERATOR_TYPE,
    ":" : OPERATOR_TYPE,
    "->" : OPERATOR_TYPE,

    "==" : OPERATOR_TYPE,
    "!=" : OPERATOR_TYPE,
    "≠" : OPERATOR_TYPE,
    "<" : OPERATOR_TYPE,
    "<=" : OPERATOR_TYPE,
    "≤" : OPERATOR_TYPE,
    ">" : OPERATOR_TYPE,
    ">=" : OPERATOR_TYPE,
    "≥" : OPERATOR_TYPE,

    "=>" : OPERATOR_TYPE,
    # reserved literals
    "NaN" : LITERAL_TYPE,
    "Inf" : LITERAL_TYPE,
    "-Inf" : LITERAL_TYPE,      # Many more literal types
    # Reserved Keyword Tokens
    "baremodule": KEYWORD_TYPE,
    "begin": KEYWORD_TYPE,
    "break": KEYWORD_TYPE,
    "catch": KEYWORD_TYPE,
    "const": KEYWORD_TYPE,
    "continue": KEYWORD_TYPE,
    "do": KEYWORD_TYPE,
    "else": KEYWORD_TYPE,
    "elseif": KEYWORD_TYPE,
    "end": KEYWORD_TYPE,
    "export": KEYWORD_TYPE,
    "false": KEYWORD_TYPE,
    "finally": KEYWORD_TYPE,
    "for": KEYWORD_TYPE,
    "function": KEYWORD_TYPE,
    "global": KEYWORD_TYPE,
    "if": KEYWORD_TYPE,
    "import": KEYWORD_TYPE,
    "let": KEYWORD_TYPE,
    "local": KEYWORD_TYPE,
    "macro": KEYWORD_TYPE,
    "module": KEYWORD_TYPE,
    "quote": KEYWORD_TYPE,
    "return": KEYWORD_TYPE,
    "struct": KEYWORD_TYPE,
    "true": KEYWORD_TYPE,
    "try": KEYWORD_TYPE,
    "using": KEYWORD_TYPE,
    "while": KEYWORD_TYPE,
    # special value tests
    "isequal" : KEYWORD_TYPE,       # pre-defined function names
    "isfinite" : KEYWORD_TYPE,
    "isinf" : KEYWORD_TYPE,
    "isnan" : KEYWORD_TYPE,
    # annoying multi-word keywords
    # abstract, mutable, primitive, type are allowed as var names
    "abstract type": KEYWORD_TYPE,
    "mutable struct": KEYWORD_TYPE,
    "primitive type": KEYWORD_TYPE,
    # annoying depends on context, allowed as variable names
    "where" : KEYWORD_TYPE,
    "in" : KEYWORD_TYPE,
    "isa" : KEYWORD_TYPE,
    "outer" : KEYWORD_TYPE,
}


python_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    "ĠĊ" : WHITESPACE_TYPE,
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    ");" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "();" : SEPARATOR_TYPE,
    "());" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "[]" : SEPARATOR_TYPE,
    #'("\\' : SEPARATOR_TYPE,   # user customized string
    #'Ġ"' : SEPARATOR_TYPE,     # 
    #'");' : SEPARATOR_TYPE,
    # "<" : SEPARATOR_TYPE,     # like List<Type>, also an operator
    # ">" : SEPARATOR_TYPE,
    "..." : SEPARATOR_TYPE,
    "@" : SEPARATOR_TYPE,
    "::" : SEPARATOR_TYPE,
    # Operator Tokens
    "=" : OPERATOR_TYPE,
    ">" : OPERATOR_TYPE,
    "<" : OPERATOR_TYPE,
    "!" : OPERATOR_TYPE,
    "~" : OPERATOR_TYPE,
    "?" : OPERATOR_TYPE,
    ":" : OPERATOR_TYPE,
    "->" : OPERATOR_TYPE,
    "==" : OPERATOR_TYPE,
    ">=" : OPERATOR_TYPE,
    "<=" : OPERATOR_TYPE,
    "!=" : OPERATOR_TYPE,
    "&&" : OPERATOR_TYPE,
    "||" : OPERATOR_TYPE,
    "++" : OPERATOR_TYPE,
    "--" : OPERATOR_TYPE,
    "+" : OPERATOR_TYPE,
    "-" : OPERATOR_TYPE,
    "*" : OPERATOR_TYPE,
    "/" : OPERATOR_TYPE,
    "&" : OPERATOR_TYPE,
    "|" : OPERATOR_TYPE,
    "^" : OPERATOR_TYPE,
    "%" : OPERATOR_TYPE,
    "<<" : OPERATOR_TYPE,
    ">>" : OPERATOR_TYPE,
    ">>>" : OPERATOR_TYPE,
    "+=" : OPERATOR_TYPE,
    "-=" : OPERATOR_TYPE,
    "*=" : OPERATOR_TYPE,
    "/=" : OPERATOR_TYPE,
    "&=" : OPERATOR_TYPE,
    "|=" : OPERATOR_TYPE,
    "^=" : OPERATOR_TYPE,
    "%=" : OPERATOR_TYPE,
    "<<=" : OPERATOR_TYPE,
    ">>=" : OPERATOR_TYPE,
    ">>>=" : OPERATOR_TYPE,

    'False' : KEYWORD_TYPE,
    'None' : KEYWORD_TYPE,
    'True' : KEYWORD_TYPE,
    'and' : KEYWORD_TYPE,
    'as' : KEYWORD_TYPE,
    'assert' : KEYWORD_TYPE,
    'async' : KEYWORD_TYPE,
    'await' : KEYWORD_TYPE,
    'break' : KEYWORD_TYPE,
    'class' : KEYWORD_TYPE,
    'continue' : KEYWORD_TYPE,
    'def' : KEYWORD_TYPE,
    'del' : KEYWORD_TYPE,
    'elif' : KEYWORD_TYPE,
    'else' : KEYWORD_TYPE,
    'except' : KEYWORD_TYPE,
    'finally' : KEYWORD_TYPE,
    'for' : KEYWORD_TYPE,
    'from' : KEYWORD_TYPE,
    'global' : KEYWORD_TYPE,
    'if' : KEYWORD_TYPE,
    'import' : KEYWORD_TYPE,
    'in' : KEYWORD_TYPE,
    'is' : KEYWORD_TYPE,
    'lambda' : KEYWORD_TYPE,
    'nonlocal' : KEYWORD_TYPE,
    'not' : KEYWORD_TYPE,
    'or' : KEYWORD_TYPE,
    'pass' : KEYWORD_TYPE,
    'raise' : KEYWORD_TYPE,
    'return' : KEYWORD_TYPE,
    'try' : KEYWORD_TYPE,
    'while' : KEYWORD_TYPE,
    'with' : KEYWORD_TYPE,
    'yield' : KEYWORD_TYPE,

}

kotlin_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    "ĠĊ" : WHITESPACE_TYPE,
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    ");" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "();" : SEPARATOR_TYPE,
    "());" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "[]" : SEPARATOR_TYPE,
    #'("\\' : SEPARATOR_TYPE,   # user customized string
    #'Ġ"' : SEPARATOR_TYPE,     # 
    #'");' : SEPARATOR_TYPE,
    # "<" : SEPARATOR_TYPE,     # like List<Type>, also an operator
    # ">" : SEPARATOR_TYPE,
    "..." : SEPARATOR_TYPE,
    "@" : SEPARATOR_TYPE,
    "::" : SEPARATOR_TYPE,
    # Operator Tokens
    "=" : OPERATOR_TYPE,
    ">" : OPERATOR_TYPE,
    "<" : OPERATOR_TYPE,
    "!" : OPERATOR_TYPE,
    "~" : OPERATOR_TYPE,
    "?" : OPERATOR_TYPE,
    ":" : OPERATOR_TYPE,
    "->" : OPERATOR_TYPE,
    "==" : OPERATOR_TYPE,
    ">=" : OPERATOR_TYPE,
    "<=" : OPERATOR_TYPE,
    "!=" : OPERATOR_TYPE,
    "&&" : OPERATOR_TYPE,
    "||" : OPERATOR_TYPE,
    "++" : OPERATOR_TYPE,
    "--" : OPERATOR_TYPE,
    "+" : OPERATOR_TYPE,
    "-" : OPERATOR_TYPE,
    "*" : OPERATOR_TYPE,
    "/" : OPERATOR_TYPE,
    "&" : OPERATOR_TYPE,
    "|" : OPERATOR_TYPE,
    "^" : OPERATOR_TYPE,
    "%" : OPERATOR_TYPE,
    "<<" : OPERATOR_TYPE,
    ">>" : OPERATOR_TYPE,
    ">>>" : OPERATOR_TYPE,
    "+=" : OPERATOR_TYPE,
    "-=" : OPERATOR_TYPE,
    "*=" : OPERATOR_TYPE,
    "/=" : OPERATOR_TYPE,
    "&=" : OPERATOR_TYPE,
    "|=" : OPERATOR_TYPE,
    "^=" : OPERATOR_TYPE,
    "%=" : OPERATOR_TYPE,
    "<<=" : OPERATOR_TYPE,
    ">>=" : OPERATOR_TYPE,
    ">>>=" : OPERATOR_TYPE,
    "as" : KEYWORD_TYPE,
    "as?" : KEYWORD_TYPE,
    "break" : KEYWORD_TYPE,
    "class" : KEYWORD_TYPE,
    "continue" : KEYWORD_TYPE,
    "do" : KEYWORD_TYPE,
    "else" : KEYWORD_TYPE,
    "false" : KEYWORD_TYPE,
    "for" : KEYWORD_TYPE,
    "fun" : KEYWORD_TYPE,
    "if" : KEYWORD_TYPE,
    "in" : KEYWORD_TYPE,
    "!in" : KEYWORD_TYPE,
    "!is" : KEYWORD_TYPE,
    "null" : KEYWORD_TYPE,
    "object" : KEYWORD_TYPE,
    "package" : KEYWORD_TYPE,
    "return" : KEYWORD_TYPE,
    "super" : KEYWORD_TYPE,
    "this" : KEYWORD_TYPE,
    "throw" : KEYWORD_TYPE,
    "true" : KEYWORD_TYPE,
    "try" : KEYWORD_TYPE,
    "typealias" : KEYWORD_TYPE,
    "typeof" : KEYWORD_TYPE,
    "val" : KEYWORD_TYPE,
    "var" : KEYWORD_TYPE,
    "when" : KEYWORD_TYPE,
    "while" : KEYWORD_TYPE,
}

cpp_reserved_tokens = {
    # Whitespace Tokens
    "Ġ" : WHITESPACE_TYPE,     # space
    "Ċ" : WHITESPACE_TYPE,     # lf (\n) / crlf (\r\n)
    "ĉ" : WHITESPACE_TYPE,     # indents, set of multiple tabs / spaces
    "ĠĊ" : WHITESPACE_TYPE,
    # Separator Tokens
    "." : SEPARATOR_TYPE,
    "," : SEPARATOR_TYPE,
    ";" : SEPARATOR_TYPE,
    ":" : SEPARATOR_TYPE,
    "(" : SEPARATOR_TYPE,
    ")" : SEPARATOR_TYPE,
    ");" : SEPARATOR_TYPE,
    "((" : SEPARATOR_TYPE,
    "))" : SEPARATOR_TYPE,
    "()" : SEPARATOR_TYPE,
    "())" : SEPARATOR_TYPE,
    "();" : SEPARATOR_TYPE,
    "());" : SEPARATOR_TYPE,
    "{" : SEPARATOR_TYPE,
    "}" : SEPARATOR_TYPE,
    "[" : SEPARATOR_TYPE,
    "]" : SEPARATOR_TYPE,
    "[]" : SEPARATOR_TYPE,
    "::" : SEPARATOR_TYPE,
    # Operator Tokens
    "=" : OPERATOR_TYPE,
    "+=" : OPERATOR_TYPE,
    "-=" : OPERATOR_TYPE,
    "*=" : OPERATOR_TYPE,
    "/=" : OPERATOR_TYPE,
    "%=" : OPERATOR_TYPE,
    "&=" : OPERATOR_TYPE,
    "|=" : OPERATOR_TYPE,
    "^=" : OPERATOR_TYPE,
    "<<=" : OPERATOR_TYPE,
    ">>=" : OPERATOR_TYPE,
    # increment, c-unique: both prefix ++a and suffix versions a++ exist
    "++" : OPERATOR_TYPE,
    "--" : OPERATOR_TYPE,
    # arithmetic
    "+" : OPERATOR_TYPE,
    "-" : OPERATOR_TYPE,
    "*" : OPERATOR_TYPE,
    "/" : OPERATOR_TYPE,
    "%" : OPERATOR_TYPE,
    "~" : OPERATOR_TYPE,
    "&" : OPERATOR_TYPE,
    "|" : OPERATOR_TYPE,
    "^" : OPERATOR_TYPE,
    "<<" : OPERATOR_TYPE,
    ">>" : OPERATOR_TYPE,
    # logical
    "!" : OPERATOR_TYPE,
    "&&" : OPERATOR_TYPE,
    "||" : OPERATOR_TYPE,
    # comparison
    "==" : OPERATOR_TYPE,
    "!=" : OPERATOR_TYPE,
    "<" : OPERATOR_TYPE,
    ">" : OPERATOR_TYPE,
    "<=" : OPERATOR_TYPE,
    ">=" : OPERATOR_TYPE,
    "<=>" : OPERATOR_TYPE,
    # member access
    "*" : OPERATOR_TYPE,    # tricky prefix tokens like *var, &var
    "&" : OPERATOR_TYPE,
    "->" : OPERATOR_TYPE,
    "." : OPERATOR_TYPE,
    "->*" : OPERATOR_TYPE,
    ".*" : OPERATOR_TYPE,
    "..." : OPERATOR_TYPE,   # variadic functions
    # conditional
    "?" : OPERATOR_TYPE,
    ":" : OPERATOR_TYPE,
    # general keywords
    "alignas" : KEYWORD_TYPE,
    "alignof" : KEYWORD_TYPE,
    "and" : KEYWORD_TYPE,
    "and_eq" : KEYWORD_TYPE,
    "asm" : KEYWORD_TYPE,
    "atomic_cancel" : KEYWORD_TYPE,
    "atomic_commit" : KEYWORD_TYPE,
    "atomic_noexcept" : KEYWORD_TYPE,
    "auto" : KEYWORD_TYPE,
    "bitand" : KEYWORD_TYPE,
    "bitor" : KEYWORD_TYPE,
    "bool" : KEYWORD_TYPE,
    "break" : KEYWORD_TYPE,
    "case" : KEYWORD_TYPE,
    "catch" : KEYWORD_TYPE,
    "char" : KEYWORD_TYPE,
    "char8_t" : KEYWORD_TYPE,
    "char16_t" : KEYWORD_TYPE,
    "char32_t" : KEYWORD_TYPE,
    "class" : KEYWORD_TYPE,
    "compl" : KEYWORD_TYPE,
    "concept" : KEYWORD_TYPE,
    "const" : KEYWORD_TYPE,
    "consteval" : KEYWORD_TYPE,
    "constexpr" : KEYWORD_TYPE,
    "constinit" : KEYWORD_TYPE,
    "const_cast" : KEYWORD_TYPE,
    "continue" : KEYWORD_TYPE,
    "co_await" : KEYWORD_TYPE,
    "co_return" : KEYWORD_TYPE,
    "co_yield" : KEYWORD_TYPE,
    "decltype" : KEYWORD_TYPE,
    "default" : KEYWORD_TYPE,
    "delete" : KEYWORD_TYPE,
    "do" : KEYWORD_TYPE,
    "double" : KEYWORD_TYPE,
    "dynamic_cast" : KEYWORD_TYPE,
    "else" : KEYWORD_TYPE,
    "enum" : KEYWORD_TYPE,
    "explicit" : KEYWORD_TYPE,
    "export" : KEYWORD_TYPE,
    "extern" : KEYWORD_TYPE,
    "false" : KEYWORD_TYPE,
    "float" : KEYWORD_TYPE,
    "for" : KEYWORD_TYPE,
    "friend" : KEYWORD_TYPE,
    "goto" : KEYWORD_TYPE,
    "if" : KEYWORD_TYPE,
    "inline" : KEYWORD_TYPE,
    "int" : KEYWORD_TYPE,
    "long" : KEYWORD_TYPE,
    "mutable" : KEYWORD_TYPE,
    "namespace" : KEYWORD_TYPE,
    "new" : KEYWORD_TYPE,
    "noexcept" : KEYWORD_TYPE,
    "not" : KEYWORD_TYPE,
    "not_eq" : KEYWORD_TYPE,
    "nullptr" : KEYWORD_TYPE,
    "operator" : KEYWORD_TYPE,
    "or" : KEYWORD_TYPE,
    "or_eq" : KEYWORD_TYPE,
    "private" : KEYWORD_TYPE,
    "protected" : KEYWORD_TYPE,
    "public" : KEYWORD_TYPE,
    "reflexpr" : KEYWORD_TYPE,
    "register" : KEYWORD_TYPE,
    "reinterpret_cast" : KEYWORD_TYPE,
    "requires" : KEYWORD_TYPE,
    "return" : KEYWORD_TYPE,
    "short" : KEYWORD_TYPE,
    "signed" : KEYWORD_TYPE,
    "sizeof" : KEYWORD_TYPE,
    "static" : KEYWORD_TYPE,
    "static_assert" : KEYWORD_TYPE,
    "static_cast" : KEYWORD_TYPE,
    "struct" : KEYWORD_TYPE,
    "switch" : KEYWORD_TYPE,
    "synchronized" : KEYWORD_TYPE,
    "template" : KEYWORD_TYPE,
    "this" : KEYWORD_TYPE,
    "thread_local" : KEYWORD_TYPE,
    "throw" : KEYWORD_TYPE,
    "true" : KEYWORD_TYPE,
    "try" : KEYWORD_TYPE,
    "typedef" : KEYWORD_TYPE,
    "typeid" : KEYWORD_TYPE,
    "typename" : KEYWORD_TYPE,
    "union" : KEYWORD_TYPE,
    "unsigned" : KEYWORD_TYPE,
    "using" : KEYWORD_TYPE,
    "virtual" : KEYWORD_TYPE,
    "void" : KEYWORD_TYPE,
    "volatile" : KEYWORD_TYPE,
    "wchar_t" : KEYWORD_TYPE,
    "while" : KEYWORD_TYPE,
    "xor" : KEYWORD_TYPE,
    "xor_eq" : KEYWORD_TYPE,
    # context dependent
    "final" : KEYWORD_TYPE,
    "override" : KEYWORD_TYPE,
    "transaction_safe" : KEYWORD_TYPE,
    "transaction_safe_dynamic" : KEYWORD_TYPE,
    "import" : KEYWORD_TYPE,
    "module" : KEYWORD_TYPE,
    # preprocessor sepcific
    "if" : KEYWORD_TYPE,
    "elif" : KEYWORD_TYPE,
    "else" : KEYWORD_TYPE,
    "endif" : KEYWORD_TYPE,
    "ifdef" : KEYWORD_TYPE,
    "ifndef" : KEYWORD_TYPE,
    "elifdef" : KEYWORD_TYPE,
    "elifndef" : KEYWORD_TYPE,
    "define" : KEYWORD_TYPE,
    "undef" : KEYWORD_TYPE,
    "include" : KEYWORD_TYPE,
    "line" : KEYWORD_TYPE,
    "error" : KEYWORD_TYPE,
    "warning" : KEYWORD_TYPE,
    "pragma" : KEYWORD_TYPE,
    "defined" : KEYWORD_TYPE,
    "__has_include" : KEYWORD_TYPE,
    "__has_cpp_attribute" : KEYWORD_TYPE,
    "export" : KEYWORD_TYPE,
    "import" : KEYWORD_TYPE,
    "module" : KEYWORD_TYPE,
    "_Pragma" : KEYWORD_TYPE,

    
}