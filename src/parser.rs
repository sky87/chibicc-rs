use crate::lexer::{Token, TokenKind};
use crate::errors::ErrorReporting;

pub type P<A> = Box<A>;

pub enum NodeKind {
    Num { val: i32 },

    Add { lhs: P<Node>, rhs: P<Node> },
    Sub { lhs: P<Node>, rhs: P<Node> },
    Mul { lhs: P<Node>, rhs: P<Node> },
    Div { lhs: P<Node>, rhs: P<Node> },
    Neg { expr: P<Node> },

    Eq { lhs: P<Node>, rhs: P<Node> },
    Ne { lhs: P<Node>, rhs: P<Node> },
    Lt { lhs: P<Node>, rhs: P<Node> },
    Le { lhs: P<Node>, rhs: P<Node> },
}

pub struct Node {
    pub kind: NodeKind
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
    pub fn new(src: &'a [u8], toks: &'a [Token]) -> Self {
        if toks.is_empty() {
            panic!("Empty token array")
        }
        Self {
            src,
            toks,
            tok_index: 0,
        }
    }
    
    // expr = equality
    pub fn expr(&mut self) -> Node {
        self.equality()
    }

    // equality = relational ("==" relational | "!=" relational)*
    fn equality(&mut self) -> Node {
        let mut node = self.relational();
        
        loop {
            if self.tok_is("==") {
                self.advance();
                node = Node {
                    kind: NodeKind::Eq { 
                        lhs: P::new(node), 
                        rhs: P::new(self.relational()) 
                    }
                };
            }
            else if self.tok_is("!=") {
                self.advance();
                node = Node {
                    kind: NodeKind::Ne { 
                        lhs: P::new(node), 
                        rhs: P::new(self.relational()) 
                    }
                };
            }
            else {
                break;
            }
        }

        node
    }

    // relational = add ("<" add | "<=" add | ">" add | ">=" add)*
    fn relational(&mut self) -> Node {
        let mut node = self.add();
        
        loop {
            if self.tok_is("<") {
                self.advance();
                node = Node {
                    kind: NodeKind::Lt { 
                        lhs: P::new(node), 
                        rhs: P::new(self.add()) 
                    }
                };
            }
            else if self.tok_is("<=") {
                self.advance();
                node = Node {
                    kind: NodeKind::Le { 
                        lhs: P::new(node), 
                        rhs: P::new(self.add()) 
                    }
                };
            }
            else if self.tok_is(">") {
                self.advance();
                node = Node {
                    kind: NodeKind::Lt { 
                        lhs: P::new(self.add()),
                        rhs: P::new(node)
                    }
                };
            }
            else if self.tok_is(">=") {
                self.advance();
                node = Node {
                    kind: NodeKind::Le { 
                        lhs: P::new(self.add()),
                        rhs: P::new(node)
                    }
                };
            }
            else {
                break;
            }
        }

        node
    }

    // add = mul ("+" mul | "-" mul)*
    fn add(&mut self) -> Node {
        let mut node = self.mul();
        
        loop {
            if self.tok_is("+") {
                self.advance();
                node = Node {
                    kind: NodeKind::Add { 
                        lhs: P::new(node), 
                        rhs: P::new(self.mul()) 
                    }
                };
            }
            else if self.tok_is("-") {
                self.advance();
                node = Node {
                    kind: NodeKind::Sub { 
                        lhs: P::new(node), 
                        rhs: P::new(self.mul()) 
                    }
                };
            }
            else {
                break;
            }
        }

        node
    }

    // mul = unary ("*" unary | "/" unary)*
    fn mul(&mut self) -> Node {
        let mut node = self.unary();
        
        loop {
            if self.tok_is("*") {
                self.advance();
                node = Node {
                    kind: NodeKind::Mul { 
                        lhs: P::new(node), 
                        rhs: P::new(self.unary()) 
                    }
                };
            }
            else if self.tok_is("/") {
                self.advance();
                node = Node {
                    kind: NodeKind::Div { 
                        lhs: P::new(node), 
                        rhs: P::new(self.unary()) 
                    }
                };
            }
            else {
                break;
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
            return Node { kind: NodeKind::Neg { expr: P::new(self.unary()) }}
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
        self.error_tok(self.peek(), "expected an expression");
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

    pub fn ensure_done(&self) {
        match self.peek().kind {
            TokenKind::Eof => {},
            _ => self.error_tok(self.peek(), "extra token")
        }
    }
}
