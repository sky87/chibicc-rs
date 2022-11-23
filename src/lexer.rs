use crate::errors::ErrorReporting;

#[derive(Debug)]
pub enum TokenKind {
    Punct,
    Ident,
    Keyword,
    Num(i32),
    Eof
}

#[derive(Debug)]
pub struct Token {
    pub offset: usize,
    pub length: usize,
    pub kind: TokenKind
}

const KEYWORDS: [&[u8];5] = [
    "return".as_bytes(),
    "if".as_bytes(), "else".as_bytes(),
    "for".as_bytes(), "while".as_bytes()
];

pub struct Lexer<'a> {
    src: &'a [u8],
}

impl<'a> ErrorReporting for Lexer<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self { src }
    }

    pub fn tokenize(&self) -> Vec<Token> {
        let mut toks = Vec::new();
        let mut offset = 0;
        let src = self.src;

        while offset < src.len() {
            let c = src[offset];

            if c.is_ascii_whitespace() {
                offset += 1;
            }
            else if c.is_ascii_digit() {
                let (val, count) = read_int(&src[offset..]);
                if count == 0 {
                    self.error_at(offset, "expected number")
                }
                toks.push(Token {
                    offset,
                    length: count,
                    kind: TokenKind::Num(val),
                });
                offset += count;
            }
            else if is_ident_start(c) {
              let start_offset = offset;
              loop {
                  offset += 1;
                  if !is_ident_cont(src[offset]) { break; }
              }
              let name = &src[start_offset..offset];
              let kind = if KEYWORDS.contains(&name) {
                  TokenKind::Keyword
              }
              else {
                  TokenKind::Ident
              };
              toks.push(Token {
                  offset: start_offset,
                  length: offset - start_offset,
                  kind,
              });
            }
            else {
                let punct_len = read_punct(&src[offset..]);
                if punct_len > 0 {
                    toks.push(Token {
                        offset,
                        length: punct_len,
                        kind: TokenKind::Punct,
                    });
                    offset += punct_len;
                }
                else {
                    self.error_at(offset, "invalid token");
                }
            }
        }

        toks.push(Token { offset, length: 0, kind: TokenKind::Eof });
        toks
    }
}

fn read_int(buf: &[u8]) -> (i32, usize) {
    let mut acc: i32 = 0;
    let mut offset = 0;
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
    return (acc, offset);
}

fn ispunct(c: u8) -> bool {
    return c == b';' || c == b'=' || c == b'{' || c == b'}' ||
        c == b'+' || c == b'-' || c == b'*' || c == b'/' ||
        c == b'(' || c == b')' || c == b'<' || c == b'>';
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

fn starts_with(src: &[u8], s: &str) -> bool {
    return src.starts_with(s.as_bytes());
}

fn read_punct(src: &[u8]) -> usize {
    if starts_with(src, "==") || starts_with(src, "!=")
       || starts_with(src, "<=") || starts_with(src, ">=") {
        2
    }
    else if ispunct(src[0]) {
        1
    }
    else {
        0
    }
}