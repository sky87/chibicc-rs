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

#[derive(Debug, Clone, Copy)]
pub struct SourceLocation {
    pub offset: usize,
    pub line: u32,
    pub column: u32
}

#[derive(Debug)]
pub struct Token {
    pub loc: SourceLocation,
    pub length: usize,
    pub kind: TokenKind
}

lazy_static! {
    pub static ref TY_KEYWORDS: HashSet<&'static [u8]> = [
        "void", "_Bool", "char", "short", "int", "long", "struct", "union", "enum"
    ].map(|k| k.as_bytes()).into();

    static ref KEYWORDS: HashSet<&'static [u8]> = {
        let others: HashSet<&'static [u8]> = [
            "return",
            "if", "else",
            "for", "while",
            "sizeof",
            "typedef"
        ].map(|k| k.as_bytes()).into();
        others.union(&TY_KEYWORDS).cloned().collect()
    };

    static ref PUNCTUATION: Vec<&'static [u8]> = [
        // Longer strings should go first
        "==", "!=", "<=", ">=",
        "->",

        ";", "=", "(", ")",
        "{", "}", ",", ".", "[", "]",
        "+", "-", "*", "/",
        "<", ">", "&"
    ].map(|p| p.as_bytes()).into();
}

pub struct Lexer<'a> {
    ctx: &'a Context,
    offset: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx, offset: 0, line: 1, column: 1 }
    }

    pub fn advance(&mut self) {
        if self.ctx.src[self.offset] == b'\n' {
            self.line += 1;
            self.column = 0;
        }
        self.offset += 1;
        self.column += 1;
    }

    pub fn nadvance(&mut self, n: usize) {
        for _ in 0..n {
            self.advance()
        }
    }

    pub fn peek(&self) -> u8 {
        self.ctx.src[self.offset]
    }

    pub fn rest(&self) -> &[u8] {
        &self.ctx.src[self.offset..]
    }

    pub fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s.as_bytes())
    }

    pub fn loc(&self) -> SourceLocation {
        SourceLocation { offset: self.offset, line: self.line, column: self.column }
    }

    pub fn len(&self, start_loc: &SourceLocation) -> usize {
        self.offset - start_loc.offset
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut toks = Vec::new();

        loop {
            while self.consume_ws_and_comments() {}
            if self.peek() == 0 { break; }
            toks.push(self.next());
        }

        toks.push(Token { loc: self.loc(), length: 0, kind: TokenKind::Eof });
        toks
    }

    fn next(&mut self) -> Token {
        let c = self.peek();

        let loc = self.loc();

        if c.is_ascii_digit() {
            let (val, count) = read_int(self.rest());
            if count == 0 {
                self.ctx.error_at(&loc, "expected number")
            }
            self.nadvance(count);
            return Token {
                loc,
                length: count,
                kind: TokenKind::Num(val),
            };
        }
        else if is_ident_start(c) {
            loop {
                self.advance();
                if !is_ident_cont(self.peek()) { break; }
            }
            let name = &self.ctx.src[loc.offset..self.offset];
            let kind = if KEYWORDS.contains(&name) {
                TokenKind::Keyword
            }
            else {
                TokenKind::Ident
            };
            return Token {
                loc,
                length: name.len(),
                kind,
            };
        }
        else if c == b'"' {
            self.advance();

            let mut str = Vec::new();
            while self.peek() != b'"' {
                if self.peek() == b'\n' || self.peek() == 0 {
                    self.ctx.error_at(&loc, "unclosed literal string");
                }

                if self.peek() == b'\\' {
                    let escape_loc = self.loc(); // wastefull...
                    self.advance();
                    let (c, len) = self.read_escaped_char(self.rest(), &escape_loc);
                    str.push(c);
                    self.nadvance(len);
                }
                else {
                    str.push(self.peek());
                    self.advance();
                }
            }
            self.advance();
            str.push(0);

            return Token {
                loc,
                length: self.len(&loc),
                kind: TokenKind::Str(str),
            };
        }
        else if c == b'\'' {
            self.advance();

            if self.peek() == 0 {
                self.ctx.error_at(&loc, "unclosed charater literal");
            }

            let v = if self.peek() == b'\\' {
                let escape_loc = self.loc(); // wastefull...
                self.advance();
                let (v, len) = self.read_escaped_char(self.rest(), &escape_loc);
                self.nadvance(len);
                v
            }
            else {
                let v = self.peek();
                self.advance();
                v
            } as i8;

            if self.peek() != b'\'' {
                self.ctx.error_at(&loc, "unclosed charater literal");
            }
            self.advance();

            return Token {
                loc,
                length: self.len(&loc),
                kind: TokenKind::Num(v.into())
            }
        }
        else {
            let punct_len = read_punct(self.rest());
            if punct_len > 0 {
                self.nadvance(punct_len);
                return Token {
                    loc,
                    length: punct_len,
                    kind: TokenKind::Punct,
                };
            }
            else {
                self.ctx.error_at(&loc, "invalid token");
            }
        }
    }

    fn consume_ws_and_comments(&mut self) -> bool {
        if self.peek().is_ascii_whitespace() {
            while self.peek().is_ascii_whitespace() {
                self.advance();
            }
            return true;
        }

        if self.starts_with("//") {
            self.nadvance(2);
            while self.peek() != b'\n' && self.peek() != 0 {
                self.advance();
            }
            return true;
        }

        if self.starts_with("/*") {
            let loc = self.loc();
            self.nadvance(2);
            while !self.starts_with("*/") {
                if self.peek() == 0 {
                    self.ctx.error_at(&loc, "unclosed block comment");
                }
                self.advance();
            }
            self.nadvance(2);
            return true;
        }

        false
    }

    fn read_escaped_char(&self, buf: &[u8], escape_loc: &SourceLocation) -> (u8, usize) {
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
                self.ctx.error_at(escape_loc, "invalid hex escape sequence");
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

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

fn read_punct(src: &[u8]) -> usize {
    for p in PUNCTUATION.iter() {
        if src.starts_with(p) {
            return p.len();
        }
    }
    0
}
