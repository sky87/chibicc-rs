use crate::lexer::Token;

pub type AsciiStr = Vec<u8>;

pub struct Context {
    pub src: AsciiStr,
    pub filename: String,
}

impl Context {
    pub fn error_at(&self, offset: usize, msg: &str) -> ! {
        let mut line_start = offset;
        while line_start > 0 && self.src[line_start] != b'\n' {
            line_start -= 1;
        }
        let mut line_end = line_start + 1;
        while self.src[line_end] != 0 && self.src[line_end] != b'\n' {
            line_end += 1;
        }
        let mut line_num = 1;
        for off in 0..=line_start {
            if self.src[off] == b'\n' {
                line_num += 1;
            }
        }

        let loc = format!("{}:{}: ", &self.filename, line_num);

        eprintln!("{}{}", loc, String::from_utf8_lossy(&self.src[line_start + 1..line_end]));
        eprint!("{: <1$}", "", loc.len() + (offset - line_start - 1));
        eprintln!("^ {}", msg);
        panic!();
    }

    pub fn error_tok(&self, tok: &Token, msg: &str) -> ! {
        self.error_at(tok.offset, msg);
    }
}

