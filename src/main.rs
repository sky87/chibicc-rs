use std::env;

#[derive(Debug)]
enum TokenKind {
    Punct,
    Num { val: i32 },
    Eof
}

#[derive(Debug)]
struct Token {
    offset: usize,
    length: usize,
    kind: TokenKind
}

fn equal(tk: &Token, s: &str, src: &str) -> bool {
    src[tk.offset..(tk.offset + tk.length)].eq(s)
}

fn get_number(tk: &Token) -> i32 {
    match tk.kind {
        TokenKind::Num { val } => val,
        _ => panic!("expected a number, found {:?}", tk)
    }
}

fn tokenize(src: &str) -> Vec<Token> {
    let mut tks = Vec::new();
    let mut offset = 0;
    // TODO: figure out a way to sanely work with unicode cps
    // while keeping track of byte offsets...
    let buf = src.as_bytes();

    while offset < buf.len() {
        let c = buf[offset];

        if c.is_ascii_whitespace() {
            offset += 1;
        }
        else if c.is_ascii_digit() {
            let (val, count) = read_int(buf, offset);
            if count == 0 {
                panic!("Failed to read integer at: {}...", String::from_utf8_lossy(&buf[offset..(offset + 10)]))
            }
            tks.push(Token {
                offset,
                length: count,
                kind: TokenKind::Num { val },
            });
            offset += count;
        }
        else if c == b'+' || c == b'-' {
            tks.push(Token {
                offset,
                length: 1,
                kind: TokenKind::Punct,
            });
            offset += 1;
        }
    }

    tks.push(Token { offset, length: 0, kind: TokenKind::Eof });
    tks
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("{}: invalid number of arguments", args[0]);
    }

    let src = &args[1];
    let tks = tokenize(src);
    let mut tki = tks.iter();

    println!("  .globl main");
    println!("main:");
    println!("  mov ${}, %rax", get_number(tki.next().unwrap()));

    while let Some(tk) = tki.next() {
        if equal(tk, "+", src) {
            println!("  add ${}, %rax", get_number(tki.next().unwrap()));
        }
        else if equal(tk, "-", src) {
            println!("  sub ${}, %rax", get_number(tki.next().unwrap()));
        }
        else {
            match tk.kind {
                TokenKind::Eof => break,
                _ => panic!("unexpected token {:?}", tk)
            };
        }
    }

    println!("  ret");
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
