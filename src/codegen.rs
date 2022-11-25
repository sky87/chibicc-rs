use crate::errors::ErrorReporting;
use crate::parser::{TopLevelNode, TopLevelKind, StmtNode, StmtKind, ExprNode, ExprKind};

fn update_stack_info(node: &mut TopLevelNode) {
    match node.kind {
        TopLevelKind::SourceUnit(ref mut locals, _, ref mut stack_size) => {
            let mut offset = 0;
            for local in locals {
                offset -= 8;
                local.borrow_mut().stack_offset = offset;
            }
            *stack_size = align_to(-offset, 16);
        }
    }
}

pub struct Codegen<'a> {
    src: &'a [u8],
    depth: i64,
    top_node: &'a TopLevelNode<'a>,
    id_count: usize,
}

impl<'a> ErrorReporting for Codegen<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Codegen<'a> {
    pub fn new(src: &'a [u8], node: &'a mut TopLevelNode) -> Self {
        update_stack_info(node);
        Self {
            src,
            depth: 0,
            top_node: node,
            id_count: 0
        }
    }

    pub fn program(&mut self) {
        match self.top_node.kind {
            TopLevelKind::SourceUnit(_, ref body, stack_size) => {
                println!("  .globl main");
                println!("main:");

                // Prologue
                println!("  push %rbp");
                println!("  mov %rsp, %rbp");
                println!("  sub ${}, %rsp", stack_size);
                println!();

                for stmt in body {
                    self.stmt(stmt)
                }

                println!();
                println!(".L.return:");
                println!("  mov %rbp, %rsp");
                println!("  pop %rbp");
                println!("  ret");
            }
        }
    }

    fn stmt(&mut self, node: &StmtNode) {
        match node.kind {
            StmtKind::Expr(ref expr) => self.expr(expr),
            StmtKind::Return(ref expr) => {
                self.expr(expr);
                println!("  jmp .L.return");
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
            },
            ExprKind::Sub(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  sub %rdi, %rax");
            },
            ExprKind::Mul(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  imul %rdi, %rax");
            },
            ExprKind::Div(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cqo");
                println!("  idiv %rdi, %rax");
            },
            ExprKind::Eq(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  sete %al");
                println!("  movzb %al, %rax");
            },
            ExprKind::Ne(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setne %al");
                println!("  movzb %al, %rax");
            },
            ExprKind::Le(ref lhs, ref rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setle %al");
                println!("  movzb %al, %rax");
            },
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

    fn addr(&self, expr: &ExprNode) {
        if let ExprKind::Var(ref data) = expr.kind {
            println!("  lea {}(%rbp), %rax", &data.borrow().stack_offset);
            return;
        }

        self.error_at(expr.offset, "not an lvalue");
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
