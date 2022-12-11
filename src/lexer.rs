use std::collections::HashSet;

use crate::{errors::ErrorReporting, parser::AsciiStr};

#[derive(Debug)]
pub enum TokenKind {
    Punct,
    Ident,
    Keyword,
    Num(i64),
    Str(AsciiStr),
    Eof
}

#[derive(Debug)]
pub struct Token {
    pub offset: usize,
    pub length: usize,
    pub kind: TokenKind
}

lazy_static! {
    static ref KEYWORDS: HashSet<&'static [u8]> = {
        [
            "return",
            "if", "else",
            "for", "while",
            "sizeof",
            "int", "char"
        ].map(|k| k.as_bytes()).into()
    };
}

pub struct Lexer<'a> {
    src: &'a [u8]
}

impl<'a> ErrorReporting for Lexer<'a> {
    fn src(&self) -> &[u8] { self.src }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self { src }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut toks = Vec::new();
        let mut offset = 0;
        let src = self.src;

        while src[offset] != 0 {
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
            else if c == b'"' {
                let start_offset = offset;
                offset += 1;

                let mut str = Vec::new();
                while src[offset] != b'"' {
                    if src[offset] == b'\n' || src[offset] == 0 {
                        self.error_at(start_offset, "unclosed literal string");
                    }

                    if src[offset] == b'\\' {
                        let e = src[offset + 1];
                        let c = unescape(src[offset + 1])
                            // Keep behaviour the same as chibicc
                            // is this standard?
                            .unwrap_or(e);
                        str.push(c);
                        offset += 2;
                    }
                    else {
                        str.push(src[offset]);
                        offset += 1;
                    }
                }
                offset += 1;
                str.push(0);

                toks.push(Token {
                    offset: start_offset,
                    length: offset - start_offset,
                    kind: TokenKind::Str(str),
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

fn read_int(buf: &[u8]) -> (i64, usize) {
    let mut acc = 0;
    let mut offset = 0;
    while offset < buf.len() {
        let b = buf[offset];
        if b.is_ascii_digit() {
            offset += 1;
            acc = acc * 10 + i64::from(b - b'0');
        }
        else {
            break;
        }
    }
    return (acc, offset);
}

fn ispunct(c: u8) -> bool {
    return c == b';' || c == b'=' || c == b'(' || c == b')' ||
        c == b'{' || c == b'}' || c == b',' || c == b'[' || c == b']' ||
        c == b'+' || c == b'-' || c == b'*' || c == b'/' ||
        c == b'<' || c == b'>' || c == b'&';
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

fn unescape(b: u8) -> Option<u8> {
    match b {
        b'a' => Some(0x07),
        b'b' => Some(0x08),
        b't' => Some(0x09),
        b'n' => Some(0x0A),
        b'v' => Some(0x0B),
        b'f' => Some(0x0C),
        b'r' => Some(0x0D),
        // [GNU] \e for the ASCII escape character is a GNU C extension.
        b'e' => Some(0x1B),
        _ => None
    }
}