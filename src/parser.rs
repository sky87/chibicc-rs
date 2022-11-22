use std::collections::HashSet;

use crate::lexer::{Token, TokenKind};
use crate::errors::ErrorReporting;

pub type P<A> = Box<A>;
pub type AsciiStr = Vec<u8>;

#[derive(Debug)]
pub struct Node<Kind> {
    pub kind: Kind,
}

#[derive(Debug)]
pub enum ExprKind {
    Num(i32),
    Var(AsciiStr),

    Add(P<ExprNode>, P<ExprNode>),
    Sub(P<ExprNode>, P<ExprNode>),
    Mul(P<ExprNode>, P<ExprNode>),
    Div(P<ExprNode>, P<ExprNode>),
    Neg(P<ExprNode>),

    Eq(P<ExprNode>, P<ExprNode>),
    Ne(P<ExprNode>, P<ExprNode>),
    Lt(P<ExprNode>, P<ExprNode>),
    Le(P<ExprNode>, P<ExprNode>),

    Assign(P<ExprNode>, P<ExprNode>),
}

#[derive(Debug)]
pub enum StmtKind {
    Expr(ExprNode),
    Return(ExprNode),
    Block(Vec<StmtNode>),
    If(P<ExprNode>, P<StmtNode>, Option<P<StmtNode>>),
    For(P<StmtNode>, Option<P<ExprNode>>, Option<P<ExprNode>>, P<StmtNode>)
}

#[derive(Debug)]
pub enum TopLevelKind {
    Function(Vec<AsciiStr>, Vec<StmtNode>)
}

pub type ExprNode = Node<ExprKind>;
pub type StmtNode = Node<StmtKind>;
pub type TopLevelNode = Node<TopLevelKind>;

pub struct Parser<'a> {
    src: &'a [u8],
    toks: &'a [Token],
    tok_index: usize,
    vars: HashSet<AsciiStr>,
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
            vars: HashSet::new()
        }
    }

    // function = stmt+
    pub fn function(&mut self) -> TopLevelNode {
        let mut stmts = Vec::new();
        while !self.is_done() {
            stmts.push(self.stmt())
        }
        TopLevelNode { kind: TopLevelKind::Function(self.vars.clone().into_iter().collect(), stmts) }
    }

    // stmt = "return" expr ";"
    //      | "if" "(" expr ")" stmt ("else" stmt)?
    //      | "for" "( expr-stmt ";" expr? ";" expr? ")" stmt
    //      | "{" compound-stmt
    //      | expr-stmt
    fn stmt(&mut self) -> StmtNode {
        if self.tok_is("return") {
            self.advance();
            let expr = self.expr();
            self.skip(";");
            return StmtNode { kind: StmtKind::Return(expr) }
        }

        if self.tok_is("if") {
            self.advance();
            self.skip("(");
            let cond = P::new(self.expr());
            self.skip(")");
            let then_stmt = P::new(self.stmt());
            let mut else_stmt = None;
            if self.tok_is("else") {
                self.advance();
                else_stmt = Some(P::new(self.stmt()));
            }
            return StmtNode { kind: StmtKind::If(cond, then_stmt, else_stmt) }
        }

        if self.tok_is("for") {
            self.advance();
            self.skip("(");
            let init = P::new(self.expr_stmt());

            let mut cond = None;
            if !self.tok_is(";") {
                cond = Some(P::new(self.expr()));
            }
            self.skip(";");

            let mut inc = None;
            if !self.tok_is(")") {
                inc = Some(P::new(self.expr()));
            }
            self.skip(")");

            let body = P::new(self.stmt());

            return StmtNode { kind: StmtKind::For(init, cond, inc, body) }
        }

        if self.tok_is("{") {
            self.advance();
            return self.compound_stmt()
        }

        self.expr_stmt()
    }

    fn compound_stmt(&mut self) -> StmtNode {
        let mut stmts = Vec::new();
        while !self.tok_is("}") {
            stmts.push(self.stmt());
        }
        self.advance();
        StmtNode { kind: StmtKind::Block(stmts) }
    }

    // expr-stmt = expr? ";"
    fn expr_stmt(&mut self) -> StmtNode {
        if self.tok_is(";") {
            self.advance();
            return StmtNode { kind: StmtKind::Block(Vec::new()) }
        }

        let expr = self.expr();
        self.skip(";");
        StmtNode { kind: StmtKind::Expr(expr) }
    }

    // expr = assign
    fn expr(&mut self) -> ExprNode {
        self.assign()
    }

    // assign = equality ("=" assign)?
    fn assign(&mut self) -> ExprNode {
        let mut node = self.equality();
        if self.tok_is("=") {
            self.advance();
            node = ExprNode {
                kind: ExprKind::Assign(P::new(node), P::new(self.assign()))
            };
        }
        node
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

    // primary = "(" expr ")" | ident | num
    fn primary(&mut self) -> ExprNode {
        match self.peek().kind {
            TokenKind::Num(val) => {
                self.advance();
                return ExprNode { kind: ExprKind::Num(val) }
            }
            TokenKind::Ident => {
                let name = self.tok_source(self.peek()).to_owned();
                let node = ExprNode { kind: ExprKind::Var(name.clone()) };
                self.vars.insert(name);
                self.advance();
                return node;
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

    fn tok_source(&self, tok: &Token) -> &[u8] {
        &self.src[tok.offset..(tok.offset + tok.length)]
    }

    fn tok_is(&self, s: &str) -> bool {
        self.tok_source(self.peek()).eq(s.as_bytes())
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
