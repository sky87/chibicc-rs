use std::borrow::Cow;

use crate::lexer::{Token, SourceLocation};

pub type AsciiStr = Vec<u8>;

pub struct Context {
    pub src: AsciiStr,
    pub filename: String,
}

pub fn ascii(s: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(s)
}

impl Context {
    pub fn error_at(&self, loc: &SourceLocation, msg: &str) -> ! {
        // TODO use column information
        let mut line_start = loc.offset;
        while line_start > 0 && self.src[line_start] != b'\n' {
            line_start -= 1;
        }
        let mut line_end = loc.offset;
        while self.src[line_end] != 0 && self.src[line_end] != b'\n' {
            line_end += 1;
        }

        let loc_str = format!("{}:{}: ", &self.filename, loc.line);

        eprintln!("{}{}", loc_str, ascii(&self.src[line_start + 1..line_end]));
        eprint!("{: <1$}", "", loc_str.len() + (loc.offset - line_start - 1));
        eprintln!("^ {}", msg);
        panic!();
    }

    pub fn error_tok(&self, tok: &Token, msg: &str) -> ! {
        self.error_at(&tok.loc, msg);
    }

    pub fn tok_source(&self, tok: &Token) -> &[u8] {
        &self.src[tok.loc.offset..(tok.loc.offset + tok.length)]
    }
}

