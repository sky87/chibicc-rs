use std::collections::HashSet;

use crate::context::{Context, AsciiStr};

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
    ctx: &'a Context,
}

impl<'a> Lexer<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut toks = Vec::new();
        let mut offset = 0;
        let src = &self.ctx.src;

        while src[offset] != 0 {
            let c = src[offset];

            if c.is_ascii_whitespace() {
                offset += 1;
            }
            else if c.is_ascii_digit() {
                let (val, count) = read_int(&src[offset..]);
                if count == 0 {
                    self.ctx.error_at(offset, "expected number")
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
                        self.ctx.error_at(start_offset, "unclosed literal string");
                    }

                    if src[offset] == b'\\' {
                        offset += 1;
                        let (c, len) = self.read_escaped_char(&src[offset..], offset - 1);
                        str.push(c);
                        offset += len;
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
                    self.ctx.error_at(offset, "invalid token");
                }
            }
        }

        toks.push(Token { offset, length: 0, kind: TokenKind::Eof });
        toks
    }

    fn read_escaped_char(&self, buf: &[u8], error_offset: usize) -> (u8, usize) {
        let mut oct = 0;
        let mut len = 0;
        while (len < 3 && len < buf.len()) &&
            (buf[len] >= b'0' && buf[len] <= b'7')
        {
            oct = 8*oct + (buf[len] - b'0');
            len += 1;
        }
        if len > 0 {
            return (oct, len)
        }

        if buf[0] == b'x' {
            if !buf[1].is_ascii_hexdigit() {
                self.ctx.error_at(error_offset, "invalid hex escape sequence");
            }
            let mut hex = 0;
            let mut len = 1;
            // The standard supports only 2 hex digits max, but chibicc
            // does allow an arbitrary number
            while len < buf.len() && buf[len].is_ascii_hexdigit() {
                hex = 16*hex + digit_to_number(buf[len]);
                len += 1;
            }
            return (hex, len);
        }

        match buf[0] {
            b'a' => (0x07, 1),
            b'b' => (0x08, 1),
            b't' => (0x09, 1),
            b'n' => (0x0A, 1),
            b'v' => (0x0B, 1),
            b'f' => (0x0C, 1),
            b'r' => (0x0D, 1),
            // [GNU] \e for the ASCII escape character is a GNU C extension.
            b'e' => (0x1B, 1),
            _ => (buf[0], 1)
        }
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

fn digit_to_number(digit: u8) -> u8 {
    if digit.is_ascii_digit() {
        return digit - b'0';
    }
    if digit.is_ascii_uppercase() {
        return digit - b'A' + 10;
    }
    if digit.is_ascii_lowercase() {
        return digit - b'a' + 10;
    }
    panic!("invalid digit");
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
