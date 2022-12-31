use std::cell::RefCell;

use std::rc::Rc;

use crate::{lexer::{Token, TokenKind, SourceLocation}, context::{AsciiStr, Context, ascii}, codegen::align_to};

pub type P<A> = Box<A>;
pub type SP<A> = Rc<RefCell<A>>;

#[derive(Debug)]
pub enum TyKind {
    Char,
    Short,
    Int,
    Long,
    Ptr(Rc<Ty>),
    Fn(Rc<Ty>, Vec<Rc<Ty>>),
    Array(Rc<Ty>, usize),
    Struct(Vec<Rc<Member>>),
    Union(Vec<Rc<Member>>),
    Unit
}

#[derive(Debug)]
pub struct Member {
    pub name: AsciiStr,
    pub ty: Rc<Ty>,
    pub offset: usize
}

#[derive(Debug)]
pub struct Ty {
    pub kind: TyKind,
    pub size: usize,
    pub align: usize,
}

impl Ty {
    fn char() -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Char, size: 1, align: 1 }) }
    fn short() -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Short, size: 2, align: 2 }) }
    fn int() -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Int, size: 4, align: 4 }) }
    fn long() -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Long, size: 8, align: 8 }) }
    fn unit() -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Unit, size: 0, align: 1 }) }
    fn ptr(base: Rc<Ty>) -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Ptr(base), size: 8, align: 8 }) }
    fn func(ret: Rc<Ty>, params: Vec<Rc<Ty>>) -> Rc<Ty> { Rc::new(Ty { kind: TyKind::Fn(ret, params), size: 0, align: 1 }) }
    fn array(base: Rc<Ty>, len: usize) -> Rc<Ty> {
        let base_size = base.size;
        let base_align = base.align;
        Rc::new(Ty { kind: TyKind::Array(base, len), size: base_size*len, align: base_align })
    }
    fn strct(mut members: Vec<Member>) -> Rc<Ty> {
        let mut size = 0;
        let mut align = 1;
        for m in &mut members {
            size = align_to(size, m.ty.align);
            m.offset = size;
            size += m.ty.size;
            if align < m.ty.align {
                align = m.ty.align;
            }
        }
        size = align_to(size, align);
        Rc::new(Ty {
            kind: TyKind::Struct(members.into_iter().map(|m| Rc::new(m)).collect()),
            size, align
        })
    }
    fn union(members: Vec<Member>) -> Rc<Ty> {
        let size = members.iter().map(|m| m.ty.size).max().unwrap_or(0);
        let align = members.iter().map(|m| m.ty.align).max().unwrap_or(1);
        Rc::new(Ty { kind: TyKind::Union(members.into_iter().map(|m| Rc::new(m)).collect()), size, align })
    }

    fn is_integer_like(&self) -> bool {
        match &self.kind {
            TyKind::Char | TyKind::Short | TyKind::Int | TyKind::Long => true,
            _ => false,
        }
    }

    fn is_pointer_like(&self) -> bool {
        match &self.kind {
            TyKind::Ptr(_) | TyKind::Array(_, _) => true,
            _ => false
        }
    }

    fn base_ty(&self) -> Option<&Ty> {
        match &self.kind {
            TyKind::Ptr(base_ty) | TyKind::Array(base_ty, _) => Some(base_ty),
            _ => None
        }
    }
}

#[derive(Debug)]
pub struct Node<Kind> {
    pub kind: Kind,
    pub loc: SourceLocation,
    pub ty: Rc<Ty>
}

#[derive(Debug)]
pub struct Function {
    pub params: Vec<SP<Binding>>,
    pub locals: Vec<SP<Binding>>,
    pub body: P<StmtNode>,
    pub stack_size: i64
}

#[derive(Debug)]
pub enum BindingKind {
    Tag,
    GlobalVar { init_data: Option<Vec<u8>> },
    LocalVar { stack_offset: i64 },
    Function(Function),
}

#[derive(Debug)]
pub struct Binding {
    pub kind: BindingKind,
    pub name: AsciiStr,
    pub ty: Rc<Ty>,
    pub loc: SourceLocation,
}

#[derive(Debug)]
pub enum ExprKind {
    Num(i64),
    Var(SP<Binding>),

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

    MemberAccess(P<ExprNode>, Rc<Member>),

    Comma(Vec<P<ExprNode>>),
    StmtExpr(P<StmtNode>),

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

pub type ExprNode = Node<ExprKind>;
pub type StmtNode = Node<StmtKind>;
pub type SourceUnit = Vec<SP<Binding>>;

struct Scope {
    var_ns: Vec<SP<Binding>>,
    tag_ns: Vec<SP<Binding>>
}

impl Scope {
    fn new() -> Self {
        Scope {
            var_ns: Vec::new(),
            tag_ns: Vec::new()
        }
    }

    fn find(&self, name: &[u8]) -> Option<&SP<Binding>> {
        self.var_ns.iter().rfind(|v| v.borrow().name == name)
    }

    fn add(&mut self, binding: SP<Binding>) {
        self.var_ns.push(binding);
    }

    fn find_tag(&self, name: &[u8]) -> Option<&SP<Binding>> {
        self.tag_ns.iter().rfind(|v| v.borrow().name == name)
    }

    fn add_tag(&mut self, binding: SP<Binding>) {
        self.tag_ns.push(binding);
    }
}

pub struct Parser<'a> {
    ctx: &'a Context,
    toks: &'a [Token],
    tok_index: usize,

    scopes: Vec<Scope>,
    local_bindings: Vec<SP<Binding>>,
    global_bindings: Vec<SP<Binding>>,

    next_unique_id: u64,
}

impl<'a> Parser<'a> {
    pub fn new(ctx: &'a Context, toks: &'a [Token]) -> Self {
        if toks.is_empty() {
            panic!("Empty token array")
        }

        let global_scope = Scope::new();

        Self {
            ctx,
            toks,
            tok_index: 0,

            scopes: vec![global_scope],
            local_bindings: Vec::new(),
            global_bindings: Vec::new(),

            next_unique_id: 0
        }
    }

    // source_unit = stmt+
    pub fn source_unit(&mut self) -> SourceUnit {
        loop {
            match self.peek().kind {
                TokenKind::Eof => break,
                _ => {
                    if self.is_function() {
                        self.function();
                    }
                    else {
                        self.global_vars();
                    }
                },
            }
        }

        std::mem::replace(&mut self.global_bindings, Vec::new())
    }

    fn global_vars(&mut self) {
        let base_ty = self.declspec();

        let mut first = true;
        while !self.peek_is(";") {
            if !first {
                self.skip(",");
            }
            first = false;

            let loc = self.peek().loc;
            let (ty, name) = self.declarator(base_ty.clone());
            let gvar = Binding { kind: BindingKind::GlobalVar { init_data: None }, name, ty, loc };
            let binding = Rc::new(RefCell::new(gvar));
            self.add_global(binding);
        }
        self.skip(";");
    }

    fn is_function(&mut self) -> bool {
        if self.peek_is(";") {
            return false;
        }

        let idx = self.tok_index;
        let base_ty = self.declspec();
        let (ty, _) = self.declarator(base_ty);

        self.tok_index = idx;
        self.reset_locals();
        matches!(ty.kind, TyKind::Fn(_, _))
    }

    fn function(&mut self) {
        self.reset_locals();

        let loc = self.peek().loc;
        let base_ty = self.declspec();
        let (ty, name) = self.declarator(base_ty);

        let params = self.local_bindings.clone();

        let body = self.compound_stmt();
        // Reverse them to keep the locals layout in line with chibicc
        let locals: Vec<SP<Binding>> = self.local_bindings.clone().into_iter().rev().collect();
        self.add_global(Rc::new(RefCell::new(Binding {
            kind: BindingKind::Function(Function {
                params,
                locals,
                body: P::new(body),
                stack_size: -1
            }),
            name,
            ty,
            loc,
        })));
    }

    // stmt = "return" expr ";"
    //      | "if" "(" expr ")" stmt ("else" stmt)?
    //      | "for" "( expr-stmt ";" expr? ";" expr? ")" stmt
    //      | "while" "(" expr ")" stmt
    //      | "{" compound-stmt
    //      | expr-stmt
    fn stmt(&mut self) -> StmtNode {
        if self.peek_is("return") {
            let loc = self.advance().loc;
            let expr = self.expr();
            self.skip(";");
            return StmtNode { kind: StmtKind::Return(expr), loc, ty: Ty::unit() }
        }

        if self.peek_is("if") {
            let loc = self.advance().loc;
            self.skip("(");
            let cond = P::new(self.expr());
            self.skip(")");
            let then_stmt = P::new(self.stmt());
            let mut else_stmt = None;
            if self.peek_is("else") {
                self.advance();
                else_stmt = Some(P::new(self.stmt()));
            }
            return StmtNode { kind: StmtKind::If(cond, then_stmt, else_stmt), loc, ty: Ty::unit() }
        }

        if self.peek_is("for") {
            let loc = self.advance().loc;
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

            return StmtNode { kind: StmtKind::For(init, cond, inc, body), loc, ty: Ty::unit() }
        }

        if self.peek_is("while") {
            let loc = self.advance().loc;
            self.skip("(");
            let cond = Some(P::new(self.expr()));
            self.skip(")");
            let body = P::new(self.stmt());
            return StmtNode { kind: StmtKind::For(None, cond, None, body), loc, ty: Ty::unit() }
        }

        if self.peek_is("{") {
            return self.compound_stmt()
        }

        self.expr_stmt()
    }

    // compound_stmt = "{" (declaration | stmt)* "}
    fn compound_stmt(&mut self) -> StmtNode {
        let loc = self.skip("{").loc;
        let mut stmts = Vec::new();

        self.enter_scope();

        while !self.peek_is("}") {
            if self.peek_is_ty_name() {
                self.declaration(&mut stmts);
            }
            else {
                stmts.push(self.stmt());
            }
        }
        self.advance();


        self.exit_scope();

        StmtNode { kind: StmtKind::Block(stmts), loc, ty: Ty::unit() }
    }

    fn peek_is_ty_name(&self) -> bool {
        self.peek_is("char") || self.peek_is("short") || self.peek_is("int") || self.peek_is("long") ||
        self.peek_is("struct") || self.peek_is("union")
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

            let loc = self.peek().loc;
            let (ty, name) = self.declarator(base_ty.clone());
            let var_data = Rc::new(RefCell::new(Binding {
                kind: BindingKind::LocalVar { stack_offset: -1 },
                name,
                ty: ty.clone(),
                loc
            }));
            self.add_local(var_data.clone());

            if !self.peek_is("=") {
                continue;
            }

            self.advance();
            let lhs = ExprNode { kind: ExprKind::Var(var_data), loc, ty };
            let rhs = self.assign();
            let rhs_ty = rhs.ty.clone();
            stmts.push(StmtNode {
                kind: StmtKind::Expr(ExprNode {
                    kind: ExprKind::Assign(P::new(lhs), P::new(rhs)),
                    loc,
                    ty: rhs_ty,
                }),
                loc,
                ty: Ty::unit()
            });
        }
    }

    // declspec = "int" | "char" | "struct" struct-decl
    fn declspec(&mut self) -> Rc<Ty> {
        if self.peek_is("char") {
            self.advance();
            return Ty::char()
        }

        if self.peek_is("short") {
            self.advance();
            return Ty::short();
        }

        if self.peek_is("int") {
            self.advance();
            return Ty::int();
        }

        if self.peek_is("long") {
            self.advance();
            return Ty::long();
        }

        if self.peek_is("struct") || self.peek_is("union") {
            return self.struct_union_decl();
        }

        self.ctx.error_tok(self.peek(), "typename expected");
    }

    // declarator = "*"* ident type-suffix
    fn declarator(&mut self, base_ty: Rc<Ty>) -> (Rc<Ty>, AsciiStr) {
        let mut ty = base_ty;
        while self.peek_is("*") {
            self.advance();
            ty = Ty::ptr(ty);
        }

        let decl = match self.peek().kind {
            TokenKind::Ident => {
                let name = self.ctx.tok_source(self.peek()).to_owned();
                self.advance();
                (self.type_suffix(ty), name)
            },
            _ => self.ctx.error_tok(self.peek(), "expected a variable name")
        };

        //println!("# DECL {}: {:?}", ascii(&decl.1), decl.0);
        decl
    }

    // type-suffix = "(" func-params
    //             | "[" num "]" type-suffix
    //             | ε
    fn type_suffix(&mut self, ty: Rc<Ty>) -> Rc<Ty> {
        if self.peek_is("(") {
            return self.func_params(ty);
        }

        if self.peek_is("[") {
            self.advance();
            let len = self.get_number();
            self.skip("]");
            let ty = self.type_suffix(ty);
            return Ty::array(ty, len.try_into().unwrap());
        }
        return ty;
    }

    // func-params = (param ("," param)*)? ")"
    // param       = declspec declarator
    fn func_params(&mut self, ret_ty: Rc<Ty>) -> Rc<Ty> {
        let mut params = Vec::new();
        self.advance();
        while !self.peek_is(")") {
            if params.len() > 0 {
                self.skip(",");
            }
            let loc = self.peek().loc;
            let base_ty = self.declspec();
            let (ty, name) = self.declarator(base_ty);
            params.push(ty.clone());
            self.add_local(
                Rc::new(RefCell::new(Binding {
                    kind: BindingKind::LocalVar { stack_offset: -1 },
                    name,
                    ty,
                    loc
                }))
            );
        }
        self.skip(")");
        return Ty::func(ret_ty, params);
    }

    // struct-decl = ("struct" | "union") ident? ("{" (declspec declarator (","  declarator)* ";")* "}")?
    fn struct_union_decl(&mut self) -> Rc<Ty> {
        let mut tag = None;
        let is_struct = self.peek_is("struct");
        let loc = self.advance().loc;

        if let TokenKind::Ident = self.peek().kind {
            tag = Some(self.ctx.tok_source(self.advance()));
        }

        if tag.is_some() && !self.peek_is("{") {
            match self.find_tag(tag.unwrap()) {
                Some(binding) => {
                    let binding = binding.borrow();
                    let binding_is_struct = matches!(binding.ty.kind, TyKind::Struct(_));
                    if is_struct && !binding_is_struct {
                        self.ctx.error_at(&loc, "bound tag is a union")
                    }
                    if !is_struct && binding_is_struct {
                        self.ctx.error_at(&loc, "bound tag is a struct")
                    }
                    return binding.ty.clone();
                },
                None => {
                    self.ctx.error_at(&loc,
                        if is_struct { "unknown struct type" } else { "unknown union type" }
                    )
                }
            };
        }

        let mut members = Vec::new();

        self.skip("{");
        while !self.peek_is("}") {
            let base_ty = self.declspec();
            let mut i = 0;

            while !self.peek_is(";") {
                if i > 0 {
                    self.skip(",");
                }

                let (ty, name) = self.declarator(base_ty.clone());
                members.push(Member {
                    name,
                    ty,
                    offset: 0, // offsets are set by the type constructor
                });

                i+= 1;
            }
            self.advance(); // ;
        }
        self.advance(); // }

        let ty = if is_struct { Ty::strct(members) } else { Ty::union(members) };
        if tag.is_some() {
            self.add_tag(Rc::new(RefCell::new(Binding {
                kind: BindingKind::Tag,
                name: tag.unwrap().to_owned(),
                ty: ty.clone(),
                loc,
            })))
        }
        ty
    }

    // expr-stmt = expr? ";"
    fn expr_stmt(&mut self) -> StmtNode {
        if self.peek_is(";") {
            let loc = self.advance().loc;
            return StmtNode { kind: StmtKind::Block(Vec::new()), loc, ty: Ty::unit() }
        }

        let expr = self.expr();
        let loc = expr.loc;
        self.skip(";");
        StmtNode { kind: StmtKind::Expr(expr), loc, ty: Ty::unit() }
    }

    // expr = assign ("," expr)?
    fn expr(&mut self) -> ExprNode {
        let loc = self.peek().loc;
        let node = self.assign();
        if !self.peek_is(",") {
            return node;
        }
        let mut exprs = Vec::new();
        exprs.push(P::new(node));
        while self.peek_is(",") {
            self.advance();
            exprs.push(P::new(self.assign()));
        }

        let ty = exprs.last().unwrap().ty.clone();
        ExprNode {
            kind: ExprKind::Comma(exprs),
            loc,
            ty,
        }
    }

    // assign = equality ("=" assign)?
    fn assign(&mut self) -> ExprNode {
        let mut node = self.equality();
        if self.peek_is("=") {
            let loc = self.advance().loc;
            let rhs = P::new(self.assign());
            let ty = node.ty.clone();
            node = ExprNode {
                kind: ExprKind::Assign(P::new(node), rhs),
                loc,
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
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Eq(P::new(node), P::new(self.relational())),
                    loc,
                    ty: Ty::int()
                };
            }
            else if self.peek_is("!=") {
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Ne(P::new(node), P::new(self.relational())),
                    loc,
                    ty: Ty::int()
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
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(node), P::new(self.add())),
                    loc,
                    ty: Ty::int()
                };
            }
            else if self.peek_is("<=") {
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(node), P::new(self.add())),
                    loc,
                    ty: Ty::int()
                };
            }
            else if self.peek_is(">") {
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Lt(P::new(self.add()), P::new(node)),
                    loc,
                    ty: Ty::int()
                };
            }
            else if self.peek_is(">=") {
                let loc = self.advance().loc;
                node = ExprNode {
                    kind: ExprKind::Le(P::new(self.add()), P::new(node)),
                    loc,
                    ty: Ty::int()
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
                let loc = self.advance().loc;
                let rhs = P::new(self.mul());
                node = self.add_overload(P::new(node), rhs, loc);
            }
            else if self.peek_is("-") {
                let loc = self.advance().loc;
                let rhs = P::new(self.mul());
                node = self.sub_overload(P::new(node), rhs, loc);
            }
            else {
                break;
            }
        }

        node
    }

    fn add_overload(&self, lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
        let mut lhs = lhs;
        let mut rhs = rhs;

        if lhs.ty.is_integer_like() {
            if rhs.ty.is_pointer_like() {
                let tmp = lhs;
                lhs = rhs;
                rhs = tmp;
            }
        }

        if lhs.ty.is_integer_like() && rhs.ty.is_integer_like() {
            return synth_add(lhs, rhs, loc);
        }

        if lhs.ty.is_pointer_like() && rhs.ty.is_integer_like() {
            let base_ty = lhs.ty.base_ty().unwrap();
            let size = P::new(synth_num(base_ty.size.try_into().unwrap(), loc));
            let rhs = synth_mul(size, rhs, loc);
            return synth_add(lhs, P::new(rhs), loc)
        }

        self.ctx.error_at(&loc, "invalid operands");
    }

    fn sub_overload(&self, lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
        if lhs.ty.is_integer_like() && rhs.ty.is_integer_like() {
            return synth_sub(lhs, rhs, loc);
        }

        if lhs.ty.is_pointer_like() && rhs.ty.is_integer_like() {
            let base_ty = lhs.ty.base_ty().unwrap();
            let size = P::new(synth_num(base_ty.size.try_into().unwrap(), loc));
            let rhs = synth_mul(size, rhs, loc);
            return synth_sub(lhs, P::new(rhs), loc);
        }

        if lhs.ty.is_pointer_like() && rhs.ty.is_pointer_like() {
            let base_ty = lhs.ty.base_ty().unwrap();
            let size: i64 = base_ty.size.try_into().unwrap();
            let mut sub = synth_sub(lhs, rhs, loc);
            sub.ty = Ty::int();
            return synth_div(P::new(sub), P::new(synth_num(size, loc)), loc);
        }

        self.ctx.error_at(&loc, "invalid operands");
    }

    // mul = unary ("*" unary | "/" unary)*
    fn mul(&mut self) -> ExprNode {
        let mut node = self.unary();

        loop {
            if self.peek_is("*") {
                let loc = self.advance().loc;
                let ty = node.ty.clone();
                node = ExprNode {
                    kind: ExprKind::Mul(P::new(node), P::new(self.unary())),
                    loc,
                    ty
                };
            }
            else if self.peek_is("/") {
                let loc = self.advance().loc;
                let ty = node.ty.clone();
                node = ExprNode {
                    kind: ExprKind::Div(P::new(node), P::new(self.unary())),
                    loc,
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
    //       | postfix
    fn unary(&mut self) -> ExprNode {
        if self.peek_is("+") {
            self.advance();
            return self.unary()
        }

        if self.peek_is("-") {
            let loc = self.advance().loc;
            let node = P::new(self.unary());
            let ty = node.ty.clone();
            return ExprNode { kind: ExprKind::Neg(node), loc, ty }
        }

        if self.peek_is("&") {
            let loc = self.advance().loc;
            let node = P::new(self.unary());
            let ty = match &node.ty.kind {
                TyKind::Array(base_ty, _) => Ty::ptr(base_ty.clone()),
                _ => Ty::ptr(node.ty.clone())
            };
            return ExprNode { kind: ExprKind::Addr(node), loc, ty }
        }

        if self.peek_is("*") {
            let loc = self.advance().loc;
            let node = self.unary();
            return self.synth_deref(P::new(node), loc);
        }

        self.postfix()
    }

    // postfix = "primary" ("[" expr "]" | "." struct_ref | "->" struct_ref)*
    fn postfix(&mut self) -> ExprNode {
        let mut node = self.primary();
        loop {
            if self.peek_is("[") {
                let loc = self.advance().loc;
                let idx = self.expr();
                self.skip("]");
                let expr = self.add_overload(P::new(node), P::new(idx), loc);
                node = self.synth_deref(P::new(expr), loc);

                continue;
            }

            if self.peek_is(".") {
                self.advance();
                node = self.struct_ref(node);
                continue;
            }

            if self.peek_is("->") {
                // x -> y is short for (*x).y
                let loc = self.advance().loc;
                node = self.synth_deref(Box::new(node), loc);
                node = self.struct_ref(node);
                continue;
            }

            return node;
        }
    }

    // struct_ref = ident
    fn struct_ref(&mut self, struct_node: ExprNode) -> ExprNode {
        let loc = self.peek().loc;
        let name = {
            let name_tok = self.peek();
            if !matches!(name_tok.kind, TokenKind::Ident) {
                self.ctx.error_tok(name_tok, "expected struct member name");
            }
            self.ctx.tok_source(name_tok)
        };
        self.advance();
        let members = {
            match &struct_node.ty.kind {
                TyKind::Struct(members) => members,
                TyKind::Union(members) => members,
                _ => self.ctx.error_at(&struct_node.loc, "not a struct"),
            }
        };
        let member = members.iter().find(|m| m.name == name).unwrap_or_else(||
            self.ctx.error_at(&loc, "no such member")
        ).clone();

        let ty = member.ty.clone();
        ExprNode {
            kind: ExprKind::MemberAccess(Box::new(struct_node), member),
            loc,
            ty,
        }
    }

    // primary = "(" "{" stmt+ "}" ")"
    //         | "(" expr ")"
    //         | "sizeof" unary
    //         | ident func-args?
    //         | str
    //         | num
    fn primary(&mut self) -> ExprNode {
        match self.peek().kind {
            TokenKind::Num(val) => {
                let loc = self.advance().loc;
                return ExprNode { kind: ExprKind::Num(val), loc, ty: Ty::int() }
            },
            TokenKind::Keyword => {
                if self.peek_is("sizeof") {
                    self.advance();
                    let node = self.unary();
                    return synth_num(node.ty.size.try_into().unwrap(), node.loc);
                }
            }
            TokenKind::Str(ref str) => {
                let ty = Ty::array(Ty::char(), str.len());
                let init_data = Some(str.to_owned());
                let loc = self.advance().loc;
                let name = self.mk_unique_id(".L..");
                let binding = Rc::new(RefCell::new(Binding {
                    kind: BindingKind::GlobalVar { init_data },
                    name,
                    ty: ty.clone(),
                    loc
                }));
                self.add_hidden_global(binding.clone());
                return ExprNode {
                    kind: ExprKind::Var(binding),
                    loc,
                    ty,
                }
            }
            TokenKind::Ident => {
                if self.la_is(1, "(") {
                    return self.funcall();
                }

                let tok = self.peek();
                let loc = tok.loc;
                let name = self.ctx.tok_source(tok).to_owned();
                self.advance();

                if let Some(var_data) = self.find_binding(&name) {
                    let ty = var_data.borrow_mut().ty.clone();
                    let expr = ExprNode { kind: ExprKind::Var(var_data.clone()), loc, ty };
                    return expr;
                }
                else {
                    self.ctx.error_at(&loc, "undefined variable");
                }
            }
            TokenKind::Punct =>
                if self.peek_is("(") {
                    let loc = self.peek().loc;
                    self.advance();

                    let node = if self.peek_is("{") {
                        let body = self.compound_stmt();
                        let ty = if let StmtKind::Block(ref stmts) = body.kind {
                            if let Some(last) = stmts.last() {
                                if let StmtKind::Expr(exp) = &last.kind {
                                    exp.ty.clone()
                                }
                                else {
                                    self.ctx.error_at(&loc, "the last statement in a statement expression must be an expression");
                                }
                            }
                            else {
                                self.ctx.error_at(&loc, "statement expression cannot be empty");
                            }
                        }
                        else {
                            panic!("expected block")
                        };
                        ExprNode {
                            kind: ExprKind::StmtExpr(P::new(body)),
                            loc,
                            ty,
                        }
                    }
                    else {
                        self.expr()
                    };

                    self.skip(")");
                    return node
                },
            _ => {}
        };
        self.ctx.error_tok(self.peek(), "expected an expression");
    }

    // funcall = ident "(" (assign ("," assign)*)? ")"
    fn funcall(&mut self) -> ExprNode {
        let tok = self.peek();
        let loc = tok.loc;
        let fn_name = self.ctx.tok_source(tok).to_owned();
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
            loc,
            ty: Ty::long(),
        }
    }

    fn find_binding(&self, name: &[u8]) -> Option<&SP<Binding>> {
        for scope in self.scopes.iter().rev() {
            let binding = scope.find(name);
            if binding.is_some() {
                return binding;
            }
        }
        None
    }

    fn reset_locals(&mut self) {
        self.local_bindings.clear();
        self.scopes.truncate(1);
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_local(&mut self, binding: SP<Binding>) {
        self.local_bindings.push(binding.clone());
        self.scopes.last_mut().unwrap().add(binding);
    }

    fn add_global(&mut self, binding: SP<Binding>) {
        self.global_bindings.push(binding.clone());
        if self.scopes.len() > 1 {
            panic!("should not be adding to globals when nested scopes are present");
        }
        self.scopes.last_mut().unwrap().add(binding);
    }

    fn add_hidden_global(&mut self, binding: SP<Binding>) {
        self.global_bindings.push(binding);
    }

    fn add_tag(&mut self, binding: SP<Binding>) {
        self.scopes.last_mut().unwrap().add_tag(binding);
    }

    fn find_tag(&self, name: &[u8]) -> Option<&SP<Binding>> {
        for scope in self.scopes.iter().rev() {
            let binding = scope.find_tag(name);
            if binding.is_some() {
                return binding;
            }
        }
        None
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

    fn get_number(&mut self) -> i64 {
        if let TokenKind::Num(val) = self.peek().kind {
            self.advance();
            return val
        }
        self.ctx.error_tok(self.peek(), "expected a number");
    }
    fn peek_is(&self, s: &str) -> bool {
        self.ctx.tok_source(self.peek()).eq(s.as_bytes())
    }

    fn la_is(&self, n: usize, s: &str) -> bool {
        self.ctx.tok_source(self.la(n)).eq(s.as_bytes())
    }

    fn skip(&mut self, s: &str) -> &Token {
        if !self.peek_is(s) {
            self.ctx.error_tok(self.peek(), &format!("Expected {}", s));
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
            self.ctx.error_tok(self.peek(), "extra token")
        }
    }

    fn synth_deref(&self, expr: P<ExprNode>, loc: SourceLocation) -> ExprNode {
        let base_ty = get_base_ty(&expr.ty);
        let ty = match base_ty {
            None => self.ctx.error_at(&loc, "invalid pointer dereference"),
            Some(base_ty) => base_ty.clone()
        };
        ExprNode { kind: ExprKind::Deref(expr), loc, ty }
    }

    fn mk_unique_id(&mut self, prefix: &str) -> AsciiStr {
        let res = format!("{}{}", prefix, self.next_unique_id);
        self.next_unique_id += 1;
        res.into_bytes()
    }

    #[allow(dead_code)]
    fn src_rest(&self) -> std::borrow::Cow<str> {
        ascii(&self.ctx.src[self.peek().loc.offset..])
    }
}

fn synth_num(v: i64, loc: SourceLocation) -> ExprNode {
    ExprNode { kind: ExprKind::Num(v), loc, ty: Ty::int() }
}

fn synth_add(lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
    let ty = lhs.ty.clone();
    ExprNode { kind: ExprKind::Add(lhs, rhs), loc, ty }
}

fn synth_mul(lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
    let ty = lhs.ty.clone();
    ExprNode { kind: ExprKind::Mul(lhs, rhs), loc, ty }
}

fn synth_sub(lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
    let ty = lhs.ty.clone();
    ExprNode { kind: ExprKind::Sub(lhs, rhs), loc, ty }
}

fn synth_div(lhs: P<ExprNode>, rhs: P<ExprNode>, loc: SourceLocation) -> ExprNode {
    let ty = lhs.ty.clone();
    ExprNode { kind: ExprKind::Div(lhs, rhs), loc, ty }
}

fn get_base_ty(ty: &Rc<Ty>) -> Option<&Rc<Ty>> {
    match &ty.kind {
        TyKind::Ptr(bt) => Some(bt),
        TyKind::Array(bt, _) => Some(bt),
        _ => None
    }
}