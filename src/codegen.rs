use std::{io::Write, ops::{Add, Sub, Div, Mul}, fmt::Display};

use crate::{parser::{BindingKind, Function, StmtNode, StmtKind, ExprNode, ExprKind, SourceUnit, TyKind, Ty}, context::{Context, ascii}};

const ARG_REGS8: [&str;6] = [
    "%dil", "%sil", "%dl", "%cl", "%r8b", "%r9b"
];
const ARG_REGS16: [&str;6] = [
    "%di", "%si", "%dx", "%cx", "%r8w", "%r9w"
];
const ARG_REGS32: [&str;6] = [
    "%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"
];
const ARG_REGS64: [&str;6] = [
    "%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"
];

pub fn preprocess_source_unit(su: &SourceUnit) {
    for decl in su {
        let mut node = decl.borrow_mut();
        match node.kind {
            BindingKind::Function(Function {
                ref locals,
                ref mut stack_size,
                ..
            }) => {
                let mut offset: i64 = 0;
                for local in locals {
                    let mut local = local.borrow_mut();
                    let ty_size: i64 = local.ty.size.try_into().unwrap();
                    let ty_align: i64 = local.ty.align.try_into().unwrap();
                    if let BindingKind::LocalVar { stack_offset } = &mut local.kind {
                        offset += ty_size;
                        offset = align_to(offset, ty_align);
                        *stack_offset = -offset;
                    }
                }
                *stack_size = align_to(offset, 16);
            }
            _ => {}
        }
    }
}

pub struct Codegen<'a> {
    ctx: &'a Context,
    out: &'a mut dyn Write,
    su: &'a SourceUnit,
    depth: i64,
    id_count: usize,
    cur_ret_lbl: Option<String>
}

macro_rules! wln {
    ( $s:expr, $( $e:expr ),+ ) => {
        writeln!( $s.out, $( $e ),+ ).unwrap()
    };

    ( $s:expr ) => {
        writeln!( $s.out ).unwrap()
    }
}

macro_rules! w {
    ( $s:expr, $( $e:expr ),+ ) => {
        write!( $s.out, $( $e ),+ ).unwrap()
    }
}

impl<'a> Codegen<'a> {
    pub fn new(ctx: &'a Context, out: &'a mut dyn Write, su: &'a SourceUnit) -> Self {
        Self {
            ctx,
            out,
            su,
            depth: 0,
            id_count: 0,
            cur_ret_lbl: None
        }
    }

    pub fn program(&mut self) {
        self.data_sections();
        self.text_section();
    }

    fn data_sections(&mut self) {
        for binding in self.su {
            let binding = binding.borrow();
            if let BindingKind::GlobalVar { init_data } = &binding.kind {
                let name = ascii(&binding.name);
                wln!(self, "  .data");
                wln!(self, "  .globl {}", name);
                wln!(self, "{}:", name);
                if let Some(init_data) = init_data {
                    w!(self, "  .byte ");
                    let mut it = init_data.iter().peekable();
                    while let Some(b) = it.next() {
                        if it.peek().is_none() {
                            wln!(self, "{}", b);
                        }
                        else {
                            w!(self, "{},", b);
                        }
                    }
                }
                else {
                    wln!(self, "  .zero {}", binding.ty.size);
                }
            }
        }
    }

    fn text_section(&mut self) {
        wln!(self);
        wln!(self, "  .text");
        for decl in self.su.iter() {
            let decl = decl.borrow();
            if let BindingKind::Function(Function {
                ref params,
                ref locals,
                ref body,
                stack_size
            }) = decl.kind {
                let name = ascii(&decl.name);
                let ret_lbl = format!(".L.return.{}", name);
                self.cur_ret_lbl = Some(ret_lbl);

                wln!(self);
                wln!(self, "  .globl {}", name);
                for local in locals {
                    let local = local.borrow();
                    if let BindingKind::LocalVar { stack_offset } = local.kind {
                        wln!(self, "# var {} offset {}", ascii(&local.name), stack_offset);
                    }
                }
                wln!(self, "{}:", name);
                wln!(self, ".loc 1 {} {}", body.loc.line, body.loc.column);

                // Prologue
                wln!(self, "  push %rbp");
                wln!(self, "  mov %rsp, %rbp");
                wln!(self, "  sub ${}, %rsp", stack_size);
                wln!(self);

                for (i, param) in params.iter().enumerate() {
                    let param = param.borrow();
                    if let BindingKind::LocalVar { stack_offset } = param.kind {
                        self.store_gp(i, stack_offset, param.ty.size)
                    }
                }

                self.stmt(&body);
                self.sanity_checks();

                wln!(self);
                wln!(self, "{}:", self.cur_ret_lbl.as_ref().unwrap());
                wln!(self, "  mov %rbp, %rsp");
                wln!(self, "  pop %rbp");
                wln!(self, "  ret");
                wln!(self);
            };
        }
    }

    fn store_gp(&mut self, reg_idx: usize, stack_offset: i64, size: usize) {
        match size {
            1 => wln!(self, " mov {}, {}(%rbp)", ARG_REGS8[reg_idx], stack_offset),
            2 => wln!(self, " mov {}, {}(%rbp)", ARG_REGS16[reg_idx], stack_offset),
            4 => wln!(self, " mov {}, {}(%rbp)", ARG_REGS32[reg_idx], stack_offset),
            8 => wln!(self, " mov {}, {}(%rbp)", ARG_REGS64[reg_idx], stack_offset),
            _ => panic!("invalid size")
        }
    }

    fn stmt(&mut self, node: &StmtNode) {
        wln!(self, "  .loc 1 {} {}", node.loc.line, node.loc.column);
        match node.kind {
            StmtKind::Expr(ref expr) => self.expr(expr),
            StmtKind::Return(ref expr) => {
                self.expr(expr);
                let ret_lbl = self.cur_ret_lbl.as_ref().unwrap();
                wln!(self, "  jmp {}", ret_lbl);
            },
            StmtKind::Block(ref stmts) => {
                for stmt in stmts {
                    self.stmt(stmt)
                }
            },
            StmtKind::If(ref cond, ref then_stmt, ref else_stmt) => {
                let id = self.next_id();
                self.expr(cond);
                wln!(self, "  cmp $0, %rax");
                wln!(self, "  je .L.else.{}", id);
                self.stmt(then_stmt);
                wln!(self, "  jmp .L.end.{}", id);
                wln!(self, ".L.else.{}:", id);
                if let Some(else_stmt) = else_stmt {
                    self.stmt(else_stmt);
                }
                wln!(self, ".L.end.{}:", id);
            },
            StmtKind::For(ref init, ref cond, ref inc, ref body) => {
                let id = self.next_id();
                if let Some(init) = init {
                    self.stmt(init);
                }
                wln!(self, ".L.begin.{}:", id);
                if let Some(cond) = cond {
                    self.expr(cond);
                    wln!(self, "  cmp $0, %rax");
                    wln!(self, "  je .L.end.{}", id);
                }
                self.stmt(body);
                if let Some(inc) = inc {
                    self.expr(inc);
                }
                wln!(self, "  jmp .L.begin.{}", id);
                wln!(self, ".L.end.{}:", id);
            },
        }
    }

    fn expr(&mut self, node: &ExprNode) {
        wln!(self, "  .loc 1 {} {}", node.loc.line, node.loc.column);
        match &node.kind {
            ExprKind::Num(val) => wln!(self, "  mov ${}, %rax", val),
            ExprKind::Neg(expr) => {
                self.expr(expr);
                wln!(self, "  neg %rax");
            }
            ExprKind::Var(_) => {
                self.addr(node);
                self.load(&node.ty);
            }
            ExprKind::MemberAccess(_, _) => {
                self.addr(node);
                self.load(&node.ty);
            }
            ExprKind::Funcall(name, args) => {
                for arg in args {
                    self.expr(arg);
                    self.push();
                }
                for i in (0..args.len()).rev() {
                    self.pop(ARG_REGS64[i]);
                }
                wln!(self, "  mov $0, %rax");
                wln!(self, "  call {}", ascii(name));
            }
            ExprKind::Addr(expr) => {
                self.addr(expr);
            }
            ExprKind::Deref(expr) => {
                self.expr(expr);
                self.load(&node.ty);
            }
            ExprKind::Assign(lhs, rhs) => {
                self.addr(lhs);
                self.push();
                self.expr(rhs);
                self.store(&node.ty);
            }
            ExprKind::Add(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  add %rdi, %rax");
            }
            ExprKind::Sub(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  sub %rdi, %rax");
            }
            ExprKind::Mul(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  imul %rdi, %rax");
            }
            ExprKind::Div(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  cqo");
                wln!(self, "  idiv %rdi, %rax");
            }
            ExprKind::Eq(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  cmp %rdi, %rax");
                wln!(self, "  sete %al");
                wln!(self, "  movzb %al, %rax");
            }
            ExprKind::Ne(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  cmp %rdi, %rax");
                wln!(self, "  setne %al");
                wln!(self, "  movzb %al, %rax");
            }
            ExprKind::Le(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  cmp %rdi, %rax");
                wln!(self, "  setle %al");
                wln!(self, "  movzb %al, %rax");
            }
            ExprKind::Lt(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                wln!(self, "  cmp %rdi, %rax");
                wln!(self, "  setl %al");
                wln!(self, "  movzb %al, %rax");
            }
            ExprKind::Comma(exprs) => {
                for expr in exprs {
                    self.expr(expr);
                }
            }
            ExprKind::StmtExpr(body) => {
                if let StmtKind::Block(stmts) = &body.kind {
                    for stmt in stmts {
                        self.stmt(stmt);
                    }
                }
            }
        };
    }

    fn load(&mut self, ty: &Ty) {
        match ty.kind {
            TyKind::Array(_, _) | TyKind::Struct(_) | TyKind::Union(_) =>
                // If it is an array/struct/union, do not attempt to load a value to the
                // register because in general we can't load an entire array to a
                // register. As a result, the result of an evaluation of an array
                // becomes not the array itself but the address of the array.
                // This is where "array is automatically converted to a pointer to
                // the first element of the array in C" occurs.
                return,
            _ => {},
        }

        if ty.size == 1 {
            wln!(self, "  movsbq (%rax), %rax");
        }
        else if ty.size == 2 {
            wln!(self, "  movswq (%rax), %rax");
        }
        else if ty.size == 4 {
            wln!(self, "  movsxd (%rax), %rax");
        }
        else {
            wln!(self, "  mov (%rax), %rax");
        }
    }

    fn store(&mut self, ty: &Ty) {
        self.pop("%rdi");

        match &ty.kind {
            TyKind::Struct(_) | TyKind::Union(_) => {
                for i in 0..ty.size {
                    wln!(self, "  mov {}(%rax), %r8b", i);
                    wln!(self, "  mov %r8b, {}(%rdi)", i);
                }
                return;
            },
            _ => {}
        }

        if ty.size == 1 {
            wln!(self, "  mov %al, (%rdi)");
        }
        else if ty.size == 2 {
            wln!(self, "  mov %ax, (%rdi)");
        }
        else if ty.size == 4 {
            wln!(self, "  mov %eax, (%rdi)");
        }
        else {
            wln!(self, "  mov %rax, (%rdi)");
        }
    }

    fn push(&mut self) {
        wln!(self, "  push %rax");
        self.depth += 1;
    }

    fn pop(&mut self, arg: &str) {
        wln!(self, "  pop {}", arg);
        self.depth -= 1;
    }

    fn addr(&mut self, expr: &ExprNode) {
        match &expr.kind {
            ExprKind::Var(data) => {
                let data = data.upgrade().unwrap();
                let data = data.borrow();
                match &data.kind {
                    BindingKind::LocalVar { stack_offset } => {
                        wln!(self, "  lea {}(%rbp), %rax", stack_offset);
                    }
                    BindingKind::GlobalVar {..} => {
                        wln!(self, "  lea {}(%rip), %rax", ascii(&data.name));
                    }
                    _ => panic!("Unsupported")
                }
            },
            ExprKind::Deref(expr) => {
                self.expr(expr);
            },
            ExprKind::Comma(exprs) => {
                let mut it = exprs.iter().peekable();
                while let Some(expr) = it.next() {
                    if it.peek().is_none() {
                        self.addr(expr);
                    }
                    else {
                        self.expr(expr);
                    }
                }
            },
            ExprKind::MemberAccess(expr, member) => {
                self.addr(expr);
                wln!(self, "  add ${}, %rax", member.upgrade().unwrap().offset);
            }
            _ => self.ctx.error_at(&expr.loc, "not an lvalue")
        };
    }

    fn next_id(&mut self) -> usize {
        self.id_count += 1;
        return self.id_count;
    }

    pub fn sanity_checks(&self) {
        if self.depth != 0 {
            panic!("depth is not 0");
        }
    }
}

pub trait Alignable : Display + Copy + Add<Output=Self> + Sub<Output=Self> + Div<Output=Self> + Mul<Output=Self> {
    fn one() -> Self;
    fn is_zero(self) -> bool;
}

impl Alignable for i64 {
    fn one() -> Self { 1 }
    fn is_zero(self) -> bool { return self == 0 }
}

impl Alignable for usize {
    fn one() -> Self { 1 }
    fn is_zero(self) -> bool { return self == 0 }
}

// Round up `n` to the nearest multiple of `align`. For instance,
// align_to(5, 8) returns 8 and align_to(11, 8) returns 16.
pub fn align_to<T: Alignable>(n: T, align: T) -> T {
    if n.is_zero() {
        return n;
    }
    ((n + align - Alignable::one()) / align) * align
}
