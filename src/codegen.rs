use crate::errors::ErrorReporting;
use crate::parser::{Binding, BindingKind, Function, StmtNode, StmtKind, ExprNode, ExprKind, SourceUnit, TyKind, Ty};

const ARG_REGS: [&str;6] = [
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
    src: &'a [u8],
    su: SourceUnit,
    depth: i64,
    id_count: usize,
    cur_ret_lbl: Option<String>
}

impl<'a> ErrorReporting for Codegen<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Codegen<'a> {
    pub fn new(src: &'a [u8], su: SourceUnit) -> Self {
        for decl in &su {
            update_stack_info(&mut decl.borrow_mut());
        }
        Self {
            src,
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

    fn data_sections(&self) {
        for binding in &self.su {
            let binding = binding.borrow();
            if let BindingKind::GlobalVar = binding.kind {
                let name = String::from_utf8_lossy(&binding.name);
                println!("  .data");
                println!("  .globl {}", name);
                println!("{}:", name);
                println!("  .zero {}", binding.ty.size);
            }
        }
    }

    fn text_section(&mut self) {
        // This still sucks... just less than before
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

                println!();
                println!("  .globl {}", name);
                for local in locals {
                    let local = local.borrow();
                    if let BindingKind::LocalVar { stack_offset } = local.kind {
                        println!("# var {} offset {}", String::from_utf8_lossy(&local.name), stack_offset);
                    }
                }
                println!("{}:", name);

                // Prologue
                println!("  push %rbp");
                println!("  mov %rsp, %rbp");
                println!("  sub ${}, %rsp", stack_size);
                println!();

                for (i, param) in params.iter().enumerate() {
                    if let BindingKind::LocalVar { stack_offset } = param.borrow().kind {
                        println!("  mov {}, {}(%rbp)", ARG_REGS[i], stack_offset);
                    }
                }

                self.stmt(&body);
                self.sanity_checks();

                println!();
                println!("{}:", self.cur_ret_lbl.as_ref().unwrap());
                println!("  mov %rbp, %rsp");
                println!("  pop %rbp");
                println!("  ret");
                println!();
            };
        }
    }

    fn stmt(&mut self, node: &StmtNode) {
        match node.kind {
            StmtKind::Expr(ref expr) => self.expr(expr),
            StmtKind::Return(ref expr) => {
                self.expr(expr);
                let ret_lbl = self.cur_ret_lbl.as_ref().unwrap();
                println!("  jmp {}", ret_lbl);
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

    fn expr(&mut self, node: &ExprNode) {
        match &node.kind {
            ExprKind::Num(val) => println!("  mov ${}, %rax", val),
            ExprKind::Neg(expr) => {
                self.expr(expr);
                println!("  neg %rax");
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
                    self.pop(ARG_REGS[i]);
                }
                println!("  mov $0, %rax");
                println!("  call {}", String::from_utf8_lossy(name));
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
                self.store();
            }
            ExprKind::Add(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  add %rdi, %rax");
            }
            ExprKind::Sub(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  sub %rdi, %rax");
            }
            ExprKind::Mul(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  imul %rdi, %rax");
            }
            ExprKind::Div(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cqo");
                println!("  idiv %rdi, %rax");
            }
            ExprKind::Eq(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  sete %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Ne(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setne %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Le(lhs, rhs) => {
                self.expr(rhs.as_ref());
                self.push();
                self.expr(lhs.as_ref());
                self.pop("%rdi");
                println!("  cmp %rdi, %rax");
                println!("  setle %al");
                println!("  movzb %al, %rax");
            }
            ExprKind::Lt(lhs, rhs) => {
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

    fn load(&self, ty: &Ty) {
        // println!("LOAD {:?}", ty);
        if let TyKind::Array(_, _) = ty.kind {
            return;
        }
        println!("  mov (%rax), %rax");
    }

    fn store(&mut self) {
        self.pop("%rdi");
        println!("  mov %rax, (%rdi)");
    }

    fn push(&mut self) {
        println!("  push %rax");
        self.depth += 1;
    }

    fn pop(&mut self, arg: &str) {
        println!("  pop {}", arg);
        self.depth -= 1;
    }

    fn addr(&mut self, expr: &ExprNode) {
        match &expr.kind {
            ExprKind::Var(data) => {
                let data = data.borrow();
                match data.kind {
                    BindingKind::LocalVar { stack_offset } => {
                        println!("  lea {}(%rbp), %rax", stack_offset);
                    }
                    BindingKind::GlobalVar => {
                        println!("  lea {}(%rip), %rax", String::from_utf8_lossy(&data.name));
                    }
                    _ => panic!("Unsupported")
                }
            },
            ExprKind::Deref(expr) => {
                self.expr(expr);
            }
            _ => self.error_at(expr.offset, "not an lvalue")
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
