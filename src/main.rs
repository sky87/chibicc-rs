#[macro_use]
extern crate lazy_static;

use std::{env, fs::read, io::{stdin, Read}};

use context::Context;

use crate::{lexer::Lexer, parser::Parser, codegen::Codegen};

pub mod context;
pub mod lexer;
pub mod parser;
pub mod codegen;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("{}: invalid number of arguments", args[0]);
    }

    let filename = &args[1];
    let mut src;

    if filename == "-" {
        src = Vec::new();
        stdin().read_to_end(&mut src).unwrap();
    }
    else {
        src = match read(&filename) {
            Ok(src) => src,
            Err(err) => panic!("Failed to open {}: {:?}", &filename, err),
        };
    }

    // It's nice to have a sentinel value so we don't have to keep checking bounds
    src.push(0);

    let ctx = Context { src, filename: filename.to_owned() };

    let mut lexer = Lexer::new(&ctx);

    let toks = lexer.tokenize();

    let mut parser = Parser::new(&ctx, &toks);

    let su = parser.source_unit();
    parser.ensure_done();

    let mut codegen = Codegen::new(&ctx, su);
    codegen.program();
}