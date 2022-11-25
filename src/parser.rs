use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::lexer::{Token, TokenKind};
use crate::errors::ErrorReporting;

pub type P<A> = Box<A>;
pub type SP<A> = Rc<RefCell<A>>;
pub type AsciiStr = Vec<u8>;

#[derive(Debug)]
pub struct Node<Kind> {
    pub kind: Kind,
    pub offset: usize
}

#[derive(Debug)]
pub struct VarData {
    pub name: AsciiStr,
    pub stack_offset: i64
}

#[derive(Debug)]
pub enum ExprKind {
    Num(i32),
    Var(SP<VarData>),

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
    For(Option<P<StmtNode>>, Option<P<ExprNode>>, Option<P<ExprNode>>, P<StmtNode>)
}

#[derive(Debug)]
pub enum TopLevelKind {
    SourceUnit(Vec<SP<VarData>>, Vec<StmtNode>, i64)
}

pub type ExprNode = Node<ExprKind>;
pub type StmtNode = Node<StmtKind>;
pub type TopLevelNode<'a> = Node<TopLevelKind>;

pub struct Parser<'a> {
    src: &'a [u8],
    toks: &'a [Token],
    tok_index: usize,
    vars: HashMap<AsciiStr, SP<VarData>>,
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
            vars: HashMap::new()
        }
    }

    // function = stmt+
    pub fn function(&mut self) -> TopLevelNode {
        let mut stmts = Vec::new();
        let offset = self.peek().offset;
        while !self.is_done() {
            stmts.push(self.stmt())
        }
        let mut locals = Vec::new();
        for el in self.vars.values() {
            locals.push(el.clone());
        }
        TopLevelNode { kind: TopLevelKind::SourceUnit(locals, stmts, -1), offset }
    }

    // stmt = "return" expr ";"
    //      | "if" "(" expr ")" stmt ("else" stmt)?
    //      | "for" "( expr-stmt ";" expr? ";" expr? ")" stmt
    //      | "while" "(" expr ")" stmt
    //      | "{" compound-stmt
    //      | expr-stmt
    fn stmt(&mut self) -> StmtNode {
        if self.tok_is("return") {
            let offset = self.advance().offset;
            let expr = self.expr();
            self.skip(";");
            return StmtNode { kind: StmtKind::Return(expr), offset }
        }

        if self.tok_is("if") {
            let offset = self.advance().offset;
            self.skip("(");
            let cond = P::new(self.expr());
            self.skip(")");
            let then_stmt = P::new(self.stmt());
            let mut else_stmt = None;
            if self.tok_is("else") {
                self.advance();
                else_stmt = Some(P::new(self.stmt()));
            }
            return StmtNode { kind: StmtKind::If(cond, then_stmt, else_stmt), offset }
        }

        if self.tok_is("for") {
            let offset = self.advance().offset;
            self.skip("(");
            let init = Some(P::new(self.expr_stmt()));

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

            return StmtNode { kind: StmtKind::For(init, cond, inc, body), offset }
        }

        if self.tok_is("while") {
            let offset = self.advance().offset;
            self.skip("(");
            let cond = Some(P::new(self.expr()));
            self.skip(")");
            let body = P::new(self.stmt());
            return StmtNode { kind: StmtKind::For(None, cond, None, body), offset }
        }

        if self.tok_is("{") {
            return self.compound_stmt()
        }

        self.expr_stmt()
    }

    // compound_stmt = "{" stmt+ "}
    fn compound_stmt(&mut self) -> StmtNode {
        let offset = self.skip("{").offset;
        let mut stmts = Vec::new();
        while !self.tok_is("}") {
            stmts.push(self.stmt());
        }
        self.advance();
        StmtNode { kind: StmtKind::Block(stmts), offset }
    }

    // expr-stmt = expr? ";"
    fn expr_stmt(&mut self) -> StmtNode {
        if self.tok_is(";") {
            let offset = self.advance().offset;
            return StmtNode { kind: StmtKind::Block(Vec::new()), offset }
        }

        let expr = self.expr();
        let offset = expr.offset;
        self.skip(";");
        StmtNode { kind: StmtKind::Expr(expr), offset }
    }

    // expr = assign
    fn expr(&mut self) -> ExprNode {
        self.assign()
    }

    // assign = equality ("=" assign)?
    fn assign(&mut self) -> ExprNode {
        let mut node = self.equality();
        if self.tok_is("=") {
            let offset = self.advance().offset;
            node = ExprNode {
                kind: ExprKind::Assign(P::new(node), P::new(self.assign())),
                offset
            };
        }
        node
    }

    // equality = relational ("==" relational | "!=" relational)*
    fn equality(&mut self) -> ExprNode {
        let mut node = self.relational();

        loop {
            if self.tok_is("==") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Eq(P::new(node), P::new(self.relational())),
                    offset
                };
            }
            else if self.tok_is("!=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Ne(P::new(node), P::new(self.relational())),
                    offset
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
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(node), P::new(self.add())),
                    offset
                };
            }
            else if self.tok_is("<=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(node), P::new(self.add())),
                    offset
                };
            }
            else if self.tok_is(">") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(self.add()), P::new(node)),
                    offset
                };
            }
            else if self.tok_is(">=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(self.add()), P::new(node)),
                    offset
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
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Add(P::new(node), P::new(self.mul())),
                    offset
                };
            }
            else if self.tok_is("-") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Sub(P::new(node), P::new(self.mul())),
                    offset
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
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Mul(P::new(node), P::new(self.unary())),
                    offset
                };
            }
            else if self.tok_is("/") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Div(P::new(node), P::new(self.unary())),
                    offset
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
            let offset = self.advance().offset;
            return ExprNode { kind: ExprKind::Neg(P::new(self.unary())), offset }
        }

        self.primary()
    }

    // primary = "(" expr ")" | ident | num
    fn primary(&mut self) -> ExprNode {
        match self.peek().kind {
            TokenKind::Num(val) => {
                let offset = self.advance().offset;
                return ExprNode { kind: ExprKind::Num(val), offset }
            }
            TokenKind::Ident => {
                let tok = self.peek();
                let offset = tok.offset;
                let name = self.tok_source(tok).to_owned();
                let var_data = self.vars.entry(name.clone()).or_insert_with(||
                    Rc::new(RefCell::new(VarData { name, stack_offset: -1 }))
                );
                let expr = ExprNode { kind: ExprKind::Var(var_data.clone()), offset };
                self.advance();
                return expr;
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
    fn advance(&mut self) -> &Token {
        if self.tok_index >= self.toks.len() {
            panic!("Unexpected end of file");
        }
        let tok = &self.toks[self.tok_index];
        self.tok_index += 1;
        tok
    }

    fn tok_source(&self, tok: &Token) -> &[u8] {
        &self.src[tok.offset..(tok.offset + tok.length)]
    }

    fn tok_is(&self, s: &str) -> bool {
        self.tok_source(self.peek()).eq(s.as_bytes())
    }

    fn skip(&mut self, s: &str) -> &Token {
        if !self.tok_is(s) {
            self.error_tok(self.peek(), &format!("Expected {}", s));
        }
        self.advance()
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
