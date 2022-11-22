use std::collections::HashMap;

use crate::errors::ErrorReporting;
use crate::parser::{TopLevelNode, TopLevelKind, StmtNode, StmtKind, ExprNode, ExprKind, AsciiStr};


struct GenFunction {
    offsets: HashMap<AsciiStr, i64>,
    stack_size: usize
}

impl GenFunction {
    fn new(node: &TopLevelNode) -> Self {
        match node.kind {
            TopLevelKind::Function(ref locals, _) => {
                let mut offset = 0;
                let mut offsets = HashMap::new();
                for local in locals {
                    offset -= 8;
                    offsets.insert(local.to_owned(), offset);
                }
                Self {
                    offsets,
                    stack_size: align_to(-offset, 16),
                }
            }
        }
    }
}

pub struct Codegen<'a> {
    src: &'a [u8],
    depth: i64,
    top_node: &'a TopLevelNode,
    curr_gen_fn: GenFunction
}

impl<'a> ErrorReporting for Codegen<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Codegen<'a> {
    pub fn new(src: &'a [u8], node: &'a TopLevelNode) -> Self {
        Self {
            src,
            depth: 0,
            top_node: node,
            curr_gen_fn: GenFunction::new(node)
        }
    }

    pub fn program(&mut self) {
        println!("  .globl main");
        println!("main:");

        // Prologue
        println!("  push %rbp");
        println!("  mov %rsp, %rbp");
        println!("  sub ${}, %rsp", self.curr_gen_fn.stack_size);
        println!();

        match self.top_node.kind {
            TopLevelKind::Function(_, ref body) => {
                for stmt in body {
                    self.stmt(stmt)
                }
            }
        }

        println!();
        println!(".L.return:");
        println!("  mov %rbp, %rsp");
        println!("  pop %rbp");
        println!("  ret");
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
        if let ExprKind::Var(ref name) = expr.kind {
            let offset = self.curr_gen_fn.offsets.get(name).unwrap();
            println!("  lea {}(%rbp), %rax", offset);
            return;
        }

        panic!("not an lvalue: {:?}", expr);
    }

    pub fn sanity_checks(&self) {
        if self.depth != 0 {
            panic!("depth is not 0");
        }
    }
}

fn align_to(n: i64, align: i64) -> usize {
    (((n + align - 1) / align) * align).try_into().unwrap()
}
