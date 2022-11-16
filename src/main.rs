use std::env;

#[derive(Debug)]
pub enum TokenKind {
    Punct,
    Num { val: i32 },
    Eof
}

#[derive(Debug)]
pub struct Token {
    offset: usize,
    length: usize,
    kind: TokenKind
}

pub struct Compiler {
    current_input: Vec<u8>,
}

impl Compiler {
    pub fn new(src: &str) -> Self {
        Self {
            current_input: src.as_bytes().to_vec()
        }
    }

    pub fn compile(&self) {
        let toks = self.tokenize();
        let mut toki = toks.iter();

        println!("  .globl main");
        println!("main:");
        println!("  mov ${}, %rax", self.get_number(toki.next().unwrap()));
    
        while let Some(tok) = toki.next() {
            if self.equal(tok, "+") {
                println!("  add ${}, %rax", self.get_number(toki.next().unwrap()));
            }
            else if self.equal(tok, "-") {
                println!("  sub ${}, %rax", self.get_number(toki.next().unwrap()));
            }
            else {
                match tok.kind {
                    TokenKind::Eof => break,
                    _ => self.error_tok(tok, "unexpected token")
                };
            }
        }
    
        println!("  ret");
    }

    fn tokenize(&self) -> Vec<Token> {
        let mut toks = Vec::new();
        let mut offset = 0;
        let buf = &self.current_input;
    
        while offset < buf.len() {
            let c = buf[offset];
    
            if c.is_ascii_whitespace() {
                offset += 1;
            }
            else if c.is_ascii_digit() {
                let (val, count) = read_int(buf, offset);
                if count == 0 {
                    self.error_at(offset, "expected number")
                }
                toks.push(Token {
                    offset,
                    length: count,
                    kind: TokenKind::Num { val },
                });
                offset += count;
            }
            else if c == b'+' || c == b'-' {
                toks.push(Token {
                    offset,
                    length: 1,
                    kind: TokenKind::Punct,
                });
                offset += 1;
            }
            else {
                self.error_at(offset, "invalid token")
            }
        }
    
        toks.push(Token { offset, length: 0, kind: TokenKind::Eof });
        toks
    }
    
    fn equal(&self, tok: &Token, s: &str) -> bool {
        self.current_input[tok.offset..(tok.offset + tok.length)].eq(s.as_bytes())
    }

    fn get_number(&self, tok: &Token) -> i32 {
        match tok.kind {
            TokenKind::Num { val } => val,
            _ => self.error_tok(tok, "expected a number")
        }
    }

    fn error_at(&self, offset: usize, msg: &str) -> ! {
        eprintln!("{}", String::from_utf8_lossy(&self.current_input));
        eprint!("{: <1$}", "", offset);
        eprintln!("^ {}", msg);
        panic!();
    }
    
    fn error_tok(&self, tok: &Token, msg: &str) -> ! {
        self.error_at(tok.offset, msg);
    }
}

fn read_int(buf: &[u8], start_offset: usize) -> (i32, usize) {
    let mut acc: i32 = 0;
    let mut offset = start_offset;
    while offset < buf.len() {
        let b = buf[offset];
        if b.is_ascii_digit() {
            offset += 1;
            acc = acc * 10 + i32::from(b - b'0');
        }
        else {
            break;
        }
    }
    return (acc, offset - start_offset);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("{}: invalid number of arguments", args[0]);
    }

    let compiler = Compiler::new(&args[1]);
    compiler.compile();
}