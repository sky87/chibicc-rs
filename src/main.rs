use std::env;

#[derive(Debug)]
pub enum TokenKind {
    Punct,
    Num { val: i32 },
    Eof
}

#[derive(Debug)]
pub struct Token {
    offset: usize,
    length: usize,
    kind: TokenKind
}

type P<A> = Box<A>;

pub enum NodeKind {
    Num { val: i32 },

    Add { lhs: P<Node>, rhs: P<Node> },
    Sub { lhs: P<Node>, rhs: P<Node> },
    Mul { lhs: P<Node>, rhs: P<Node> },
    Div { lhs: P<Node>, rhs: P<Node> },
    Neg { exp: P<Node> }
}

pub struct Node {
    kind: NodeKind
}

trait ErrorReporting {
    fn src(&self) -> &[u8];

    fn error_at(&self, offset: usize, msg: &str) -> ! {
        eprintln!("{}", String::from_utf8_lossy(&self.src()));
        eprint!("{: <1$}", "", offset);
        eprintln!("^ {}", msg);
        panic!();
    }
    
    fn error_tok(&self, tok: &Token, msg: &str) -> ! {
        self.error_at(tok.offset, msg);
    }
}

pub struct Lexer<'a> {
    src: &'a [u8],
}

impl<'a> ErrorReporting for Lexer<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self {
            src
        }
    }

    fn tokenize(&self) -> Vec<Token> {
        let mut toks = Vec::new();
        let mut offset = 0;
        let buf = &self.src;
    
        while offset < buf.len() {
            let c = buf[offset];
    
            if c.is_ascii_whitespace() {
                offset += 1;
            }
            else if c.is_ascii_digit() {
                let (val, count) = read_int(buf, offset);
                if count == 0 {
                    self.error_at(offset, "expected number")
                }
                toks.push(Token {
                    offset,
                    length: count,
                    kind: TokenKind::Num { val },
                });
                offset += count;
            }
            else if ispunct(c) {
                toks.push(Token {
                    offset,
                    length: 1,
                    kind: TokenKind::Punct,
                });
                offset += 1;
            }
            else {
                self.error_at(offset, "invalid token")
            }
        }
    
        toks.push(Token { offset, length: 0, kind: TokenKind::Eof });
        toks
    }
}

fn read_int(buf: &[u8], start_offset: usize) -> (i32, usize) {
    let mut acc: i32 = 0;
    let mut offset = start_offset;
    while offset < buf.len() {
        let b = buf[offset];
        if b.is_ascii_digit() {
            offset += 1;
            acc = acc * 10 + i32::from(b - b'0');
        }
        else {
            break;
        }
    }
    return (acc, offset - start_offset);
}

fn ispunct(c: u8) -> bool {
    return c == b'+' || c == b'-' || c == b'*' || c == b'/' || c == b'(' || c == b')';
}

pub struct Parser<'a> {
    src: &'a [u8],
    toks: &'a [Token],
    tok_index: usize
}

impl<'a> ErrorReporting for Parser<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8], toks: &'a [Token]) -> Self {
        if toks.is_empty() {
            panic!("Empty token array")
        }
        Self {
            src,
            toks,
            tok_index: 0,
        }
    }

    // expr = mul ("+" mul | "-" mul)*
    fn expr(&mut self) -> Node {
        let mut node = self.mul();
        
        loop {
            match self.peek().kind {
                TokenKind::Punct => {
                    if self.tok_is("+") {
                        self.advance();
                        node = Node {
                            kind: NodeKind::Add { 
                                lhs: P::new(node), 
                                rhs: P::new(self.mul()) 
                            }
                        }
                    }
                    else if self.tok_is("-") {
                        self.advance();
                        node = Node {
                            kind: NodeKind::Sub { 
                                lhs: P::new(node), 
                                rhs: P::new(self.mul()) 
                            }
                        }
                    }
                    else {
                        break;
                    }
                },
                _ => break
            }
        }

        node
    }

    // mul = unary ("*" unary | "/" unary)*
    fn mul(&mut self) -> Node {
        let mut node = self.unary();
        
        loop {
            match self.peek().kind {
                TokenKind::Punct => {
                    if self.tok_is("*") {
                        self.advance();
                        node = Node {
                            kind: NodeKind::Mul { 
                                lhs: P::new(node), 
                                rhs: P::new(self.unary()) 
                            }
                        }
                    }
                    else if self.tok_is("/") {
                        self.advance();
                        node = Node {
                            kind: NodeKind::Div { 
                                lhs: P::new(node), 
                                rhs: P::new(self.unary()) 
                            }
                        }
                    }
                    else {
                        break;
                    }
                },
                _ => break
            }
        }

        node
    }

    // unary = ("+" | "-") unary
    //       | primary
    fn unary(&mut self) -> Node {
        if self.tok_is("+") {
            self.advance();
            return self.unary()
        }
        
        if self.tok_is("-") {
            self.advance();
            return Node { kind: NodeKind::Neg { exp: P::new(self.unary()) }}
        }

        self.primary()
    }

    // primary = "(" expr ")" | num
    fn primary(&mut self) -> Node {
        match self.peek().kind {
            TokenKind::Num { val } => {
                self.advance();
                return Node { kind: NodeKind::Num { val } }
            }
            TokenKind::Punct => 
                if self.tok_is("(") {
                    self.advance();
                    let node = self.expr();
                    self.skip(")");
                    return node
                },
            _ => {}
        };
        self.error_tok(self.peek(), "expected an expression")
    }

    fn peek(&self) -> &Token { &self.toks[self.tok_index] }
    fn advance(&mut self) {
        if self.tok_index >= self.toks.len() {
            panic!("Unexpected end of file");
        }
        self.tok_index += 1;
    }

    fn tok_is(&self, s: &str) -> bool {
        let tok = self.peek();
        self.src[tok.offset..(tok.offset + tok.length)].eq(s.as_bytes())
    }

    fn skip(&mut self, s: &str) {
        if !self.tok_is(s) {
            self.error_tok(self.peek(), &format!("Expected {}", s));
        }
        self.advance();
    }

    fn ensure_done(&self) {
        match self.peek().kind {
            TokenKind::Eof => {},
            _ => self.error_tok(self.peek(), "extra token")
        }
    }
}

pub struct Codegen<'a> {
    src: &'a [u8],
    depth: i64,
}

impl<'a> ErrorReporting for Codegen<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Codegen<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            depth: 0
        }
    }

    fn program(&mut self, node: &Node) {
        println!("  .globl main");
        println!("main:");
        self.expr(node);
        println!("  ret");
    }

    fn push(&mut self) {
        println!("  push %rax");
        self.depth += 1;
    }

    fn pop(&mut self, arg: &str) {
        println!("  pop {}", arg);
        self.depth -= 1;
    }

    fn expr(&mut self, node: &Node) {
        match node.kind {
            NodeKind::Num { val } => println!("  mov ${}, %rax", val),
            NodeKind::Neg { exp: ref expr } => {
                self.expr(expr);
                println!("  neg %rax");
            }
            NodeKind::Add { ref lhs, ref rhs } => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  add %rdi, %rax");
            },
            NodeKind::Sub { ref lhs, ref rhs } => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  sub %rdi, %rax");
            },
            NodeKind::Mul { ref lhs, ref rhs } => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  imul %rdi, %rax");
            },
            NodeKind::Div { ref lhs, ref rhs } => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cqo");
                println!("  idiv %rdi, %rax");
            }
        };
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("{}: invalid number of arguments", args[0]);
    }

    let src = args[1].as_bytes();

    let lexer = Lexer::new(src);

    let toks = lexer.tokenize();

    let mut parser = Parser::new(src, &toks);

    let node = parser.expr();
    parser.ensure_done();

    let mut codegen = Codegen::new(src);
    codegen.program(&node);
    
    assert!(codegen.depth == 0);
}