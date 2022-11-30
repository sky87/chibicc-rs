use std::cell::RefCell;
use std::rc::Rc;

use crate::lexer::{Token, TokenKind};
use crate::errors::ErrorReporting;

pub type P<A> = Box<A>;
pub type SP<A> = Rc<RefCell<A>>;
pub type AsciiStr = Vec<u8>;

#[derive(Debug, Clone)]
pub enum Ty {
    Int,
    Ptr(P<Ty>),
    Fn(P<Ty>),
    Unit
}

#[derive(Debug, Clone)]
pub struct Node<Kind> {
    pub kind: Kind,
    pub offset: usize,
    pub ty: Ty
}

#[derive(Debug, Clone)]
pub struct VarData {
    pub name: AsciiStr,
    pub ty: Ty,
    pub stack_offset: i64
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Num(i64),
    Var(SP<VarData>),

    Addr(P<ExprNode>),
    Deref(P<ExprNode>),

    Funcall(AsciiStr, Vec<P<ExprNode>>),

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

#[derive(Debug, Clone)]
pub enum StmtKind {
    Expr(ExprNode),
    Return(ExprNode),
    Block(Vec<StmtNode>),
    If(P<ExprNode>, P<StmtNode>, Option<P<StmtNode>>),
    For(Option<P<StmtNode>>, Option<P<ExprNode>>, Option<P<ExprNode>>, P<StmtNode>)
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    Function(AsciiStr, Vec<SP<VarData>>, StmtNode, i64)
}

pub type ExprNode = Node<ExprKind>;
pub type StmtNode = Node<StmtKind>;
pub type DeclNode = Node<DeclKind>;
pub type SourceUnit = Vec<DeclNode>;

pub struct Parser<'a> {
    src: &'a [u8],
    toks: &'a [Token],
    tok_index: usize,
    vars: Vec<SP<VarData>>,
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
            vars: Vec::new()
        }
    }

    // source_unit = stmt+
    pub fn source_unit(&mut self) -> SourceUnit {
        let mut fns = Vec::new();
        loop {
            match self.peek().kind {
                TokenKind::Eof => break,
                _ => fns.push(self.function()),
            }
        }
        fns
    }

    pub fn function(&mut self) -> DeclNode {
        let offset = self.peek().offset;
        let ty = self.declspec();
        let (ty, name) = self.declarator(&ty);

        self.vars.clear();
        let body = self.compound_stmt();
        // Reverse them to keep the locals layout in line with chibicc
        let locals = self.vars.clone().into_iter().rev().collect();
        DeclNode { kind: DeclKind::Function(name, locals, body, -1), offset, ty }
    }

    // stmt = "return" expr ";"
    //      | "if" "(" expr ")" stmt ("else" stmt)?
    //      | "for" "( expr-stmt ";" expr? ";" expr? ")" stmt
    //      | "while" "(" expr ")" stmt
    //      | "{" compound-stmt
    //      | expr-stmt
    fn stmt(&mut self) -> StmtNode {
        if self.peek_is("return") {
            let offset = self.advance().offset;
            let expr = self.expr();
            self.skip(";");
            return StmtNode { kind: StmtKind::Return(expr), offset, ty: Ty::Unit }
        }

        if self.peek_is("if") {
            let offset = self.advance().offset;
            self.skip("(");
            let cond = P::new(self.expr());
            self.skip(")");
            let then_stmt = P::new(self.stmt());
            let mut else_stmt = None;
            if self.peek_is("else") {
                self.advance();
                else_stmt = Some(P::new(self.stmt()));
            }
            return StmtNode { kind: StmtKind::If(cond, then_stmt, else_stmt), offset, ty: Ty::Unit }
        }

        if self.peek_is("for") {
            let offset = self.advance().offset;
            self.skip("(");
            let init = Some(P::new(self.expr_stmt()));

            let mut cond = None;
            if !self.peek_is(";") {
                cond = Some(P::new(self.expr()));
            }
            self.skip(";");

            let mut inc = None;
            if !self.peek_is(")") {
                inc = Some(P::new(self.expr()));
            }
            self.skip(")");

            let body = P::new(self.stmt());

            return StmtNode { kind: StmtKind::For(init, cond, inc, body), offset, ty: Ty::Unit }
        }

        if self.peek_is("while") {
            let offset = self.advance().offset;
            self.skip("(");
            let cond = Some(P::new(self.expr()));
            self.skip(")");
            let body = P::new(self.stmt());
            return StmtNode { kind: StmtKind::For(None, cond, None, body), offset, ty: Ty::Unit }
        }

        if self.peek_is("{") {
            return self.compound_stmt()
        }

        self.expr_stmt()
    }

    // compound_stmt = "{" (declaration | stmt)* "}
    fn compound_stmt(&mut self) -> StmtNode {
        let offset = self.skip("{").offset;
        let mut stmts = Vec::new();
        while !self.peek_is("}") {
            if self.peek_is("int") {
                self.declaration(&mut stmts);
            }
            else {
                stmts.push(self.stmt());
            }
        }
        self.advance();
        StmtNode { kind: StmtKind::Block(stmts), offset, ty: Ty::Unit }
    }

    // declaration = declspec (declarator ("=" expr)? ("," declarator ("=" expr)?)*)? ";"
    fn  declaration(&mut self, stmts: &mut Vec<StmtNode>) {
        let base_ty = self.declspec();

        let mut count = 0;
        while !self.peek_is(";") {
            if count > 0 {
                self.skip(",");
            }
            count += 1;

            let offset = self.peek().offset;
            let (ty, name) = self.declarator(&base_ty);
            let var_data = Rc::new(RefCell::new(VarData { name, ty: ty.clone(), stack_offset: -1 }));
            self.vars.push(var_data.clone());

            if !self.peek_is("=") {
                continue;
            }

            self.advance();
            let lhs = ExprNode { kind: ExprKind::Var(var_data), offset, ty };
            let rhs = self.assign();
            let rhs_ty = rhs.ty.clone();
            stmts.push(StmtNode {
                kind: StmtKind::Expr(ExprNode {
                    kind: ExprKind::Assign(P::new(lhs), P::new(rhs)),
                    offset,
                    ty: rhs_ty,
                }),
                offset,
                ty: Ty::Unit
            });
        }
    }

    // declspec = "int"
    fn declspec(&mut self) -> Ty {
        self.skip("int");
        Ty::Int
    }

    // declarator = "*"* ident type-suffix
    fn declarator(&mut self, base_ty: &Ty) -> (Ty, AsciiStr) {
        let mut ty = base_ty.clone();
        while self.peek_is("*") {
            self.advance();
            ty = Ty::Ptr(P::new(ty));
        }

        match self.peek().kind {
            TokenKind::Ident => {
                let name = self.tok_source(self.peek()).to_owned();
                self.advance();
                (self.type_suffix(ty), name)
            },
            _ => self.error_tok(self.peek(), "expected a variable name")
        }
    }

    // type-suffix = ("(" func-params)?
    fn type_suffix(&mut self, ty: Ty) -> Ty {
        if self.peek_is("(") {
            self.advance();
            self.skip(")");
            return Ty::Fn(P::new(ty));
        }
        return ty;
    }

    // expr-stmt = expr? ";"
    fn expr_stmt(&mut self) -> StmtNode {
        if self.peek_is(";") {
            let offset = self.advance().offset;
            return StmtNode { kind: StmtKind::Block(Vec::new()), offset, ty: Ty::Unit }
        }

        let expr = self.expr();
        let offset = expr.offset;
        self.skip(";");
        StmtNode { kind: StmtKind::Expr(expr), offset, ty: Ty::Unit }
    }

    // expr = assign
    fn expr(&mut self) -> ExprNode {
        self.assign()
    }

    // assign = equality ("=" assign)?
    fn assign(&mut self) -> ExprNode {
        let mut node = self.equality();
        if self.peek_is("=") {
            let offset = self.advance().offset;
            let rhs = P::new(self.assign());
            let ty = node.ty.clone();
            node = ExprNode {
                kind: ExprKind::Assign(P::new(node), rhs),
                offset,
                ty
            };
        }
        node
    }

    // equality = relational ("==" relational | "!=" relational)*
    fn equality(&mut self) -> ExprNode {
        let mut node = self.relational();

        loop {
            if self.peek_is("==") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Eq(P::new(node), P::new(self.relational())),
                    offset,
                    ty: Ty::Int
                };
            }
            else if self.peek_is("!=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Ne(P::new(node), P::new(self.relational())),
                    offset,
                    ty: Ty::Int
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
            if self.peek_is("<") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(node), P::new(self.add())),
                    offset,
                    ty: Ty::Int
                };
            }
            else if self.peek_is("<=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(node), P::new(self.add())),
                    offset,
                    ty: Ty::Int
                };
            }
            else if self.peek_is(">") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(self.add()), P::new(node)),
                    offset,
                    ty: Ty::Int
                };
            }
            else if self.peek_is(">=") {
                let offset = self.advance().offset;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(self.add()), P::new(node)),
                    offset,
                    ty: Ty::Int
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
            if self.peek_is("+") {
                let offset = self.advance().offset;
                let rhs = P::new(self.mul());
                node = self.add_overload(P::new(node), rhs, offset);
            }
            else if self.peek_is("-") {
                let offset = self.advance().offset;
                let rhs = P::new(self.mul());
                node = self.sub_overload(P::new(node), rhs, offset);
            }
            else {
                break;
            }
        }

        node
    }

    fn add_overload(&self, lhs: P<ExprNode>, rhs: P<ExprNode>, offset: usize) -> ExprNode {
        let mut lhs = lhs;
        let mut rhs = rhs;

        if let Ty::Int = lhs.ty {
            if let Ty::Ptr(_) = rhs.ty {
                let tmp = lhs;
                lhs = rhs;
                rhs = tmp;
            }
        }

        match (&lhs.ty, &rhs.ty) {
            (Ty::Int, Ty::Int) => {
                ExprNode { kind: ExprKind::Add(lhs, rhs), offset, ty: Ty::Int }
            },
            (Ty::Ptr(_), Ty::Int) => {
                let rhs = P::new(ExprNode {
                    kind: ExprKind::Mul(
                        P::new(ExprNode { kind: ExprKind::Num(8), offset, ty: Ty::Int }),
                        rhs
                    ),
                    offset,
                    ty: Ty::Int,
                });
                let ty = lhs.ty.clone();
                ExprNode { kind: ExprKind::Add(lhs, rhs), offset, ty }
            },
            _ => self.error_at(offset, "invalid operands")
        }
    }

    fn sub_overload(&self, lhs: P<ExprNode>, rhs: P<ExprNode>, offset: usize) -> ExprNode {
        match (&lhs.ty, &rhs.ty) {
            (Ty::Int, Ty::Int) => {
                ExprNode { kind: ExprKind::Sub(lhs, rhs), offset, ty: Ty::Int }
            },
            (Ty::Ptr(_), Ty::Int) => {
                let rhs = P::new(ExprNode {
                    kind: ExprKind::Mul(
                        synth_num(8, offset),
                        rhs
                    ),
                    offset,
                    ty: Ty::Int,
                });
                let ty = lhs.ty.clone();
                ExprNode { kind: ExprKind::Sub(lhs, rhs), offset, ty }
            },
            (Ty::Ptr(_), Ty::Ptr(_)) => {
                let node = P::new(ExprNode { kind: ExprKind::Sub(lhs, rhs), offset, ty: Ty::Int });
                ExprNode { kind: ExprKind::Div(node, synth_num(8, offset)), offset, ty: Ty::Int }
            }
            _ => self.error_at(offset, "invalid operands")
        }
    }

    // mul = unary ("*" unary | "/" unary)*
    fn mul(&mut self) -> ExprNode {
        let mut node = self.unary();

        loop {
            if self.peek_is("*") {
                let offset = self.advance().offset;
                let ty = node.ty.clone();
                node = ExprNode {
                    kind: ExprKind::Mul(P::new(node), P::new(self.unary())),
                    offset,
                    ty
                };
            }
            else if self.peek_is("/") {
                let offset = self.advance().offset;
                let ty = node.ty.clone();
                node = ExprNode {
                    kind: ExprKind::Div(P::new(node), P::new(self.unary())),
                    offset,
                    ty
                };
            }
            else {
                break;
            }
        }

        node
    }

    // unary = ("+" | "-" | "*" | "&") unary
    //       | primary
    fn unary(&mut self) -> ExprNode {
        if self.peek_is("+") {
            self.advance();
            return self.unary()
        }

        if self.peek_is("-") {
            let offset = self.advance().offset;
            let node = P::new(self.unary());
            let ty = node.ty.clone();
            return ExprNode { kind: ExprKind::Neg(node), offset, ty }
        }

        if self.peek_is("&") {
            let offset = self.advance().offset;
            let node = P::new(self.unary());
            let ty = Ty::Ptr(P::new(node.ty.clone()));
            return ExprNode { kind: ExprKind::Addr(node), offset, ty }
        }

        if self.peek_is("*") {
            let offset = self.advance().offset;
            let node = P::new(self.unary());
            let ty = if let Ty::Ptr(ref base) = node.ty {
                *base.clone()
            } else {
                Ty::Int
            };
            return ExprNode { kind: ExprKind::Deref(node), offset, ty }
        }

        self.primary()
    }

    // primary = "(" expr ")" | funcall | num
    fn primary(&mut self) -> ExprNode {
        match self.peek().kind {
            TokenKind::Num(val) => {
                let offset = self.advance().offset;
                return ExprNode { kind: ExprKind::Num(val), offset, ty: Ty::Int }
            }
            TokenKind::Ident => {
                if self.la_is(1, "(") {
                    return self.funcall();
                }

                let tok = self.peek();
                let offset = tok.offset;
                let name = self.tok_source(tok).to_owned();
                self.advance();

                let var_data = self.vars.iter().find(|v| v.borrow().name == name);
                if let Some(var_data) = var_data {
                    let expr = ExprNode { kind: ExprKind::Var(var_data.clone()), offset, ty: Ty::Int };
                    return expr;
                }
                else {
                    self.error_at(offset, "undefined variable");
                }
            }
            TokenKind::Punct =>
                if self.peek_is("(") {
                    self.advance();
                    let node = self.expr();
                    self.skip(")");
                    return node
                },
            _ => {}
        };
        self.error_tok(self.peek(), "expected an expression");
    }

    // funcall = ident "(" (assign ("," assign)*)? ")"
    fn funcall(&mut self) -> ExprNode {
        let tok = self.peek();
        let offset = tok.offset;
        let fn_name = self.tok_source(tok).to_owned();
        self.advance();

        let mut args = Vec::new();
        self.skip("(");
        while !self.peek_is(")") {
            if args.len() > 0 {
                self.skip(",");
            }
            args.push(P::new(self.assign()));
        }
        self.skip(")");

        ExprNode {
            kind: ExprKind::Funcall(fn_name, args),
            offset,
            ty: Ty::Int,
        }
    }

    fn peek(&self) -> &Token { &self.toks[self.tok_index] }
    fn la(&self, n: usize) -> &Token { &self.toks[self.tok_index + n] }
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

    fn peek_is(&self, s: &str) -> bool {
        self.tok_source(self.peek()).eq(s.as_bytes())
    }

    fn la_is(&self, n: usize, s: &str) -> bool {
        self.tok_source(self.la(n)).eq(s.as_bytes())
    }

    fn skip(&mut self, s: &str) -> &Token {
        if !self.peek_is(s) {
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

fn synth_num(v: i64, offset: usize) -> P<ExprNode> {
    P::new(ExprNode { kind: ExprKind::Num(v), offset, ty: Ty::Int })
}
