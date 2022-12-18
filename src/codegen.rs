use std::io::Write;

use crate::{parser::{Binding, BindingKind, Function, StmtNode, StmtKind, ExprNode, ExprKind, SourceUnit, TyKind, Ty}, context::Context};

const ARG_REGS8: [&str;6] = [
    "%dil", "%sil", "%dl", "%cl", "%r8b", "%r9b"
];
const ARG_REGS64: [&str;6] = [
    "%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"
];

fn update_stack_info(node: &mut Binding) {
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
                if let BindingKind::LocalVar { stack_offset } = &mut local.kind {
                    offset -= ty_size;
                    *stack_offset = offset;
                }
            }
            *stack_size = align_to(-offset, 16);
        }
        _ => {}
    }
}

pub struct Codegen<'a> {
    ctx: &'a Context,
    out: &'a mut dyn Write,
    su: SourceUnit,
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
    pub fn new(ctx: &'a Context, out: &'a mut dyn Write, su: SourceUnit) -> Self {
        for decl in &su {
            update_stack_info(&mut decl.borrow_mut());
        }
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
        let len = self.su.len();
        // TODO The loopy ugliness is strong in this one
        // what one doesn't do to please the borrow checker...
        for ix in 0..len {
            let binding = self.su[ix].clone();
            let binding = binding.borrow();
            if let BindingKind::GlobalVar { init_data } = &binding.kind {
                let name = String::from_utf8_lossy(&binding.name);
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
        // TODO This loop still sucks too
        wln!(self);
        wln!(self, "  .text");
        for i in 0..self.su.len() {
            let decl = self.su[i].clone();
            let decl = decl.borrow();
            if let BindingKind::Function(Function {
                ref params,
                ref locals,
                ref body,
                stack_size
            }) = decl.kind {
                let name = String::from_utf8_lossy(&decl.name);
                let ret_lbl = format!(".L.return.{}", name);
                self.cur_ret_lbl = Some(ret_lbl);

                wln!(self);
                wln!(self, "  .globl {}", name);
                for local in locals {
                    let local = local.borrow();
                    if let BindingKind::LocalVar { stack_offset } = local.kind {
                        wln!(self, "# var {} offset {}", String::from_utf8_lossy(&local.name), stack_offset);
                    }
                }
                wln!(self, "{}:", name);

                // Prologue
                wln!(self, "  push %rbp");
                wln!(self, "  mov %rsp, %rbp");
                wln!(self, "  sub ${}, %rsp", stack_size);
                wln!(self);

                for (i, param) in params.iter().enumerate() {
                    let param = param.borrow();
                    if let BindingKind::LocalVar { stack_offset } = param.kind {
                        if param.ty.size == 1 {
                            wln!(self, "  mov {}, {}(%rbp)", ARG_REGS8[i], stack_offset);
                        }
                        else {
                            wln!(self, "  mov {}, {}(%rbp)", ARG_REGS64[i], stack_offset);
                        }
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

    fn stmt(&mut self, node: &StmtNode) {
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
            ExprKind::Funcall(name, args) => {
                for arg in args {
                    self.expr(arg);
                    self.push();
                }
                for i in (0..args.len()).rev() {
                    self.pop(ARG_REGS64[i]);
                }
                wln!(self, "  mov $0, %rax");
                wln!(self, "  call {}", String::from_utf8_lossy(name));
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
        if let TyKind::Array(_, _) = ty.kind {
            return;
        }

        if ty.size == 1 {
            wln!(self, "  movsbq (%rax), %rax");
        }
        else {
            wln!(self, "  mov (%rax), %rax");
        }
    }

    fn store(&mut self, ty: &Ty) {
        self.pop("%rdi");

        if ty.size == 1 {
            wln!(self, "  mov %al, (%rdi)");
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
                let data = data.borrow();
                match &data.kind {
                    BindingKind::LocalVar { stack_offset } => {
                        wln!(self, "  lea {}(%rbp), %rax", stack_offset);
                    }
                    BindingKind::GlobalVar {..} => {
                        wln!(self, "  lea {}(%rip), %rax", String::from_utf8_lossy(&data.name));
                    }
                    _ => panic!("Unsupported")
                }
            },
            ExprKind::Deref(expr) => {
                self.expr(expr);
            }
            _ => self.ctx.error_at(expr.offset, "not an lvalue")
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

fn align_to(n: i64, align: i64) -> i64 {
    ((n + align - 1) / align) * align
}
