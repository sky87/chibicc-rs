use crate::lexer::Token;

pub trait ErrorReporting {
  fn src(&self) -> &[u8];

  fn error_at(&self, offset: usize, msg: &str) -> ! {
      eprintln!("{}", String::from_utf8_lossy(&self.src()));
      eprint!("{: <1$}", "", offset);
      eprintln!("^ {}", msg);
      panic!();
  }
  
  fn error_tok(&self, tok: &Token, msg: &str) -> ! {
      self.error_at(tok.offset, msg);
  }
}
