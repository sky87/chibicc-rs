use crate::errors::ErrorReporting;
use crate::parser::{TopDeclNode, TopDeclKind, StmtNode, StmtKind, ExprNode, ExprKind, SourceUnit};

const ARG_REGS: [&str;6] = [
    "%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"
];

fn update_stack_info(node: &mut TopDeclNode) {
    match node.kind {
        TopDeclKind::Function {
            ref locals,
            ref mut stack_size,
            ..
        } => {
            let mut offset = 0;
            for local in locals {
                offset -= 8;
                let mut local = local.borrow_mut();
                local.stack_offset = offset;
            }
            *stack_size = align_to(-offset, 16);
        }
    }
}

pub struct Codegen<'a> {
    src: &'a [u8],
    su: SourceUnit,
    depth: i64,
    id_count: usize,
    cur_fn: Option<TopDeclNode>,
}

impl<'a> ErrorReporting for Codegen<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Codegen<'a> {
    pub fn new(src: &'a [u8], mut su: SourceUnit) -> Self {
        for decl in &mut su {
            update_stack_info(decl);
        }
        Self {
            src,
            su,
            depth: 0,
            id_count: 0,
            cur_fn: None
        }
    }

    pub fn program(&mut self) {
        // All the cloning sucks, but it will work until I figure out how to please the borrow checker...
        // maybe by splitting the state up?
        for decl in &self.su.clone() {
            match decl.kind {
                TopDeclKind::Function {
                    ref name,
                    ref params,
                    ref locals,
                    ref body,
                    stack_size
                } => {
                    self.cur_fn = Some(decl.clone());
                    let name = String::from_utf8_lossy(name);

                    println!();
                    println!("  .globl {}", name);
                    for local in locals {
                        // just borrow doesn't work anymore, god knows why...
                        // at some point I need to study how Rc and RefCell actually work...
                        let local = local.borrow_mut();
                        println!("# var {} offset {}", String::from_utf8_lossy(&local.name), local.stack_offset);
                    }
                    println!("{}:", name);

                    // Prologue
                    println!("  push %rbp");
                    println!("  mov %rsp, %rbp");
                    println!("  sub ${}, %rsp", stack_size);
                    println!();

                    for (i, param) in params.iter().enumerate() {
                        let param = param.borrow_mut();
                        println!("  mov {}, {}(%rbp)", ARG_REGS[i], param.stack_offset);
                    }

                    self.stmt(body);
                    self.sanity_checks();

                    println!();
                    println!(".L.return.{}:", name);
                    println!("  mov %rbp, %rsp");
                    println!("  pop %rbp");
                    println!("  ret");
                    println!();
                }
            }
        }
    }

    fn stmt(&mut self, node: &StmtNode) {
        match node.kind {
            StmtKind::Expr(ref expr) => self.expr(expr),
            StmtKind::Return(ref expr) => {
                self.expr(expr);
                let TopDeclKind::Function { ref name, .. } = &self.cur_fn.as_ref().unwrap().kind;
                println!("  jmp .L.return.{}", String::from_utf8_lossy(name));
            },
            StmtKind::Block(ref stmts) => {
                for stmt in stmts {
                    self.stmt(stmt)
                }
            },
            StmtKind::If(ref cond, ref then_stmt, ref else_stmt) => {
                let id = self.next_id();
                self.expr(cond);
                println!("  cmp $0, %rax");
                println!("  je .L.else.{}", id);
                self.stmt(then_stmt);
                println!("  jmp .L.end.{}", id);
                println!(".L.else.{}:", id);
                if let Some(else_stmt) = else_stmt {
                    self.stmt(else_stmt);
                }
                println!(".L.end.{}:", id);
            },
            StmtKind::For(ref init, ref cond, ref inc, ref body) => {
                let id = self.next_id();
                if let Some(init) = init {
                    self.stmt(init);
                }
                println!(".L.begin.{}:", id);
                if let Some(cond) = cond {
                    self.expr(cond);
                    println!("  cmp $0, %rax");
                    println!("  je .L.end.{}", id);
                }
                self.stmt(body);
                if let Some(inc) = inc {
                    self.expr(inc);
                }
                println!("  jmp .L.begin.{}", id);
                println!(".L.end.{}:", id);
            },
        }
    }

    fn push(&mut self) {
        println!("  push %rax");
        self.depth += 1;
    }

    fn pop(&mut self, arg: &str) {
        println!("  pop {}", arg);
        self.depth -= 1;
    }

    fn expr(&mut self, node: &ExprNode) {
        match node.kind {
            ExprKind::Num(val) => println!("  mov ${}, %rax", val),
            ExprKind::Neg(ref expr) => {
                self.expr(expr);
                println!("  neg %rax");
            }
            ExprKind::Var(_) => {
                self.addr(node);
                println!("  mov (%rax), %rax");
            }
            ExprKind::Funcall(ref name, ref args) => {
                for arg in args {
                    self.expr(arg);
                    self.push();
                }
                for i in (0..args.len()).rev() {
                    self.pop(ARG_REGS[i]);
                }

                println!("  mov $0, %rax");
                println!("  call {}", String::from_utf8_lossy(name));
            }
            ExprKind::Addr(ref expr) => {
                self.addr(expr);
            }
            ExprKind::Deref(ref expr) => {
                self.expr(expr);
                println!("  mov (%rax), %rax");
            }
            ExprKind::Assign(ref lhs, ref rhs) => {
                self.addr(lhs);
                self.push();
                self.expr(rhs);
                self.pop("%rdi");
                println!("  mov %rax, (%rdi)");
            }
            ExprKind::Add(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  add %rdi, %rax");
            }
            ExprKind::Sub(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  sub %rdi, %rax");
            }
            ExprKind::Mul(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  imul %rdi, %rax");
            }
            ExprKind::Div(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cqo");
                println!("  idiv %rdi, %rax");
            }
            ExprKind::Eq(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  sete %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Ne(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setne %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Le(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setle %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Lt(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setl %al");
                println!("  movzb %al, %rax");
            }
        };
    }

    fn addr(&mut self, expr: &ExprNode) {
        match expr.kind {
            ExprKind::Var(ref data) => {
                println!("  lea {}(%rbp), %rax", &data.borrow_mut().stack_offset);
            },
            ExprKind::Deref(ref expr) => {
                self.expr(expr);
            }
            _ => self.error_at(expr.offset, "not an lvalue"),
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
