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

    // TODO this with format! is ugly AF
    // using writeln! requires an .unwrap() on every call, also ugly
    // Ideally we have a macro version of what we would do with a vararg in C
    pub fn wln(&mut self, s: &str) {
        self.w(s);
        self.nl();
    }

    pub fn w(&mut self, s: &str) {
        self.out.write_all(s.as_bytes()).unwrap();
    }

    pub fn nl(&mut self) {
        self.out.write_all(&[b'\n']).unwrap();
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
                self.wln("  .data");
                self.wln(&format!("  .globl {}", name));
                self.wln(&format!("{}:", name));
                if let Some(init_data) = init_data {
                    self.w("  .byte ");
                    let mut it = init_data.iter().peekable();
                    while let Some(b) = it.next() {
                        if it.peek().is_none() {
                            self.wln(&format!("{}", b));
                        }
                        else {
                            self.w(&format!("{},", b));
                        }
                    }
                }
                else {
                    self.wln(&format!("  .zero {}", binding.ty.size));
                }
            }
        }
    }

    fn text_section(&mut self) {
        // TODO This loop still sucks too
        self.nl();
        self.wln("  .text");
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

                self.nl();
                self.wln(&format!("  .globl {}", name));
                for local in locals {
                    let local = local.borrow();
                    if let BindingKind::LocalVar { stack_offset } = local.kind {
                        self.wln(&format!("# var {} offset {}", String::from_utf8_lossy(&local.name), stack_offset));
                    }
                }
                self.wln(&format!("{}:", name));

                // Prologue
                self.wln("  push %rbp");
                self.wln("  mov %rsp, %rbp");
                self.wln(&format!("  sub ${}, %rsp", stack_size));
                self.nl();

                for (i, param) in params.iter().enumerate() {
                    let param = param.borrow();
                    if let BindingKind::LocalVar { stack_offset } = param.kind {
                        if param.ty.size == 1 {
                            self.wln(&format!("  mov {}, {}(%rbp)", ARG_REGS8[i], stack_offset));
                        }
                        else {
                            self.wln(&format!("  mov {}, {}(%rbp)", ARG_REGS64[i], stack_offset));
                        }
                    }
                }

                self.stmt(&body);
                self.sanity_checks();

                self.nl();
                self.wln(&format!("{}:", self.cur_ret_lbl.as_ref().unwrap()));
                self.wln("  mov %rbp, %rsp");
                self.wln("  pop %rbp");
                self.wln("  ret");
                self.nl();
            };
        }
    }

    fn stmt(&mut self, node: &StmtNode) {
        match node.kind {
            StmtKind::Expr(ref expr) => self.expr(expr),
            StmtKind::Return(ref expr) => {
                self.expr(expr);
                let ret_lbl = self.cur_ret_lbl.as_ref().unwrap();
                self.wln(&format!("  jmp {}", ret_lbl));
            },
            StmtKind::Block(ref stmts) => {
                for stmt in stmts {
                    self.stmt(stmt)
                }
            },
            StmtKind::If(ref cond, ref then_stmt, ref else_stmt) => {
                let id = self.next_id();
                self.expr(cond);
                self.wln("  cmp $0, %rax");
                self.wln(&format!("  je .L.else.{}", id));
                self.stmt(then_stmt);
                self.wln(&format!("  jmp .L.end.{}", id));
                self.wln(&format!(".L.else.{}:", id));
                if let Some(else_stmt) = else_stmt {
                    self.stmt(else_stmt);
                }
                self.wln(&format!(".L.end.{}:", id));
            },
            StmtKind::For(ref init, ref cond, ref inc, ref body) => {
                let id = self.next_id();
                if let Some(init) = init {
                    self.stmt(init);
                }
                self.wln(&format!(".L.begin.{}:", id));
                if let Some(cond) = cond {
                    self.expr(cond);
                    self.wln("  cmp $0, %rax");
                    self.wln(&format!("  je .L.end.{}", id));
                }
                self.stmt(body);
                if let Some(inc) = inc {
                    self.expr(inc);
                }
                self.wln(&format!("  jmp .L.begin.{}", id));
                self.wln(&format!(".L.end.{}:", id));
            },
        }
    }

    fn expr(&mut self, node: &ExprNode) {
        match &node.kind {
            ExprKind::Num(val) => self.wln(&format!("  mov ${}, %rax", val)),
            ExprKind::Neg(expr) => {
                self.expr(expr);
                self.wln("  neg %rax");
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
                self.wln("  mov $0, %rax");
                self.wln(&format!("  call {}", String::from_utf8_lossy(name)));
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
                self.wln("  add %rdi, %rax");
            }
            ExprKind::Sub(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  sub %rdi, %rax");
            }
            ExprKind::Mul(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  imul %rdi, %rax");
            }
            ExprKind::Div(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  cqo");
                self.wln("  idiv %rdi, %rax");
            }
            ExprKind::Eq(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  cmp %rdi, %rax");
                self.wln("  sete %al");
                self.wln("  movzb %al, %rax");
            }
            ExprKind::Ne(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  cmp %rdi, %rax");
                self.wln("  setne %al");
                self.wln("  movzb %al, %rax");
            }
            ExprKind::Le(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  cmp %rdi, %rax");
                self.wln("  setle %al");
                self.wln("  movzb %al, %rax");
            }
            ExprKind::Lt(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                self.wln("  cmp %rdi, %rax");
                self.wln("  setl %al");
                self.wln("  movzb %al, %rax");
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
        // self.w(&format!("LOAD {:?}", ty));
        if let TyKind::Array(_, _) = ty.kind {
            return;
        }

        if ty.size == 1 {
            self.wln("  movsbq (%rax), %rax");
        }
        else {
            self.wln("  mov (%rax), %rax");
        }
    }

    fn store(&mut self, ty: &Ty) {
        self.pop("%rdi");

        if ty.size == 1 {
            self.wln("  mov %al, (%rdi)");
        }
        else {
            self.wln("  mov %rax, (%rdi)");
        }
    }

    fn push(&mut self) {
        self.wln("  push %rax");
        self.depth += 1;
    }

    fn pop(&mut self, arg: &str) {
        self.wln(&format!("  pop {}", arg));
        self.depth -= 1;
    }

    fn addr(&mut self, expr: &ExprNode) {
        match &expr.kind {
            ExprKind::Var(data) => {
                let data = data.borrow();
                match &data.kind {
                    BindingKind::LocalVar { stack_offset } => {
                        self.wln(&format!("  lea {}(%rbp), %rax", stack_offset));
                    }
                    BindingKind::GlobalVar {..} => {
                        self.wln(&format!("  lea {}(%rip), %rax", String::from_utf8_lossy(&data.name)));
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
