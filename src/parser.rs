use crate::lexer::{Token, TokenKind};
use crate::errors::ErrorReporting;

pub type P<A> = Box<A>;

#[derive(Debug)]
pub struct Node<Kind> {
    pub kind: Kind,
}

#[derive(Debug)]
pub enum ExprKind {
    Num(i32),

    Add(P<ExprNode>, P<ExprNode>),
    Sub(P<ExprNode>, P<ExprNode>),
    Mul(P<ExprNode>, P<ExprNode>),
    Div(P<ExprNode>, P<ExprNode>),
    Neg(P<ExprNode>),

    Eq(P<ExprNode>, P<ExprNode>),
    Ne(P<ExprNode>, P<ExprNode>),
    Lt(P<ExprNode>, P<ExprNode>),
    Le(P<ExprNode>, P<ExprNode>),
}

#[derive(Debug)]
pub enum StmtKind {
    Expr(ExprNode)
}

#[derive(Debug)]
pub enum TopLevelKind {
    Stmts(Vec<StmtNode>)
}

pub type ExprNode = Node<ExprKind>;
pub type StmtNode = Node<StmtKind>;
pub type TopLevelNode = Node<TopLevelKind>;

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
    
    // stmts = stmt+
    pub fn stmts(&mut self) -> TopLevelNode {
        let mut stmts = Vec::new();
        while !self.is_done() {
            stmts.push(self.stmt())
        }
        TopLevelNode { kind: TopLevelKind::Stmts(stmts) }
    }

    // stmt = expr-stmt
    fn stmt(&mut self) -> StmtNode {
        self.expr_stmt()
    }
    
    // expr-stmt = expr ";"
    fn expr_stmt(&mut self) -> StmtNode {
        let expr = self.expr();
        self.skip(";");
        StmtNode { kind: StmtKind::Expr(expr) }
    }

    // expr = equality
    fn expr(&mut self) -> ExprNode {
        self.equality()
    }

    // equality = relational ("==" relational | "!=" relational)*
    fn equality(&mut self) -> ExprNode {
        let mut node = self.relational();
        
        loop {
            if self.tok_is("==") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Eq(P::new(node), P::new(self.relational()))
                };
            }
            else if self.tok_is("!=") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Ne(P::new(node), P::new(self.relational()))
                };
            }
            else {
                break;
            }
        }

        node
    }

    // relational = add ("<" add | "<=" add | ">" add | ">=" add)*
    fn relational(&mut self) -> ExprNode {
        let mut node = self.add();
        
        loop {
            if self.tok_is("<") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(node), P::new(self.add()))
                };
            }
            else if self.tok_is("<=") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Le(P::new(node), P::new(self.add()) )
                };
            }
            else if self.tok_is(">") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(self.add()), P::new(node))
                };
            }
            else if self.tok_is(">=") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Le(P::new(self.add()), P::new(node))
                };
            }
            else {
                break;
            }
        }

        node
    }

    // add = mul ("+" mul | "-" mul)*
    fn add(&mut self) -> ExprNode {
        let mut node = self.mul();
        
        loop {
            if self.tok_is("+") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Add(P::new(node), P::new(self.mul()))
                };
            }
            else if self.tok_is("-") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Sub(P::new(node), P::new(self.mul()))
                };
            }
            else {
                break;
            }
        }

        node
    }

    // mul = unary ("*" unary | "/" unary)*
    fn mul(&mut self) -> ExprNode {
        let mut node = self.unary();
        
        loop {
            if self.tok_is("*") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Mul(P::new(node), P::new(self.unary()))
                };
            }
            else if self.tok_is("/") {
                self.advance();
                node = ExprNode {
                    kind: ExprKind::Div(P::new(node), P::new(self.unary()))
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
    fn unary(&mut self) -> ExprNode {
        if self.tok_is("+") {
            self.advance();
            return self.unary()
        }
        
        if self.tok_is("-") {
            self.advance();
            return ExprNode { kind: ExprKind::Neg(P::new(self.unary())) }
        }

        self.primary()
    }

    // primary = "(" expr ")" | num
    fn primary(&mut self) -> ExprNode {
        match self.peek().kind {
            TokenKind::Num { val } => {
                self.advance();
                return ExprNode { kind: ExprKind::Num(val) }
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

    fn is_done(&self) -> bool {
        match self.peek().kind {
            TokenKind::Eof => true,
            _ => false
        }
    }

    pub fn ensure_done(&self) {
        if !self.is_done() {
            self.error_tok(self.peek(), "extra token")
        }
    }
}
