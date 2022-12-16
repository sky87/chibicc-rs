#[macro_use]
extern crate lazy_static;

use std::{env, fs::{read, File}, io::{stdin, Read}, process::exit};

use context::Context;

use crate::{lexer::Lexer, parser::Parser, codegen::Codegen};

pub mod context;
pub mod lexer;
pub mod parser;
pub mod codegen;

fn main() {
    let mut args: Vec<String> = env::args().collect();
    args.push("".to_owned());

    let exec = &args[0];

    let mut arg_ix = 1;
    let mut in_filename: Option<&str> = None;
    let mut out_filename: Option<&str> = None;

    // who needs libraries
    while arg_ix < args.len() - 1 {
        let arg = &args[arg_ix];
        arg_ix += 1;

        if arg == "--help" {
            usage(exec, 0);
        }

        if arg == "-o" {
            out_filename = Some(&args[arg_ix]);
            arg_ix += 1;
            if out_filename == Some("") {
                usage(exec, 1);
            }
            continue;
        }

        if arg.starts_with("-o") {
            out_filename = Some(&arg[2..]);
            continue;
        }

        if arg.starts_with("-") && arg.len() > 1 {
            panic!("unknown argument: {}", arg);
        }

        in_filename = Some(arg);
    }

    let in_filename = match in_filename {
        Some(f) => f,
        None => panic!("no input file"),
    };

    let out_filename = match out_filename {
        Some(f) => f,
        None => panic!("no output file"),
    };

    let mut src;

    if in_filename == "-" {
        src = Vec::new();
        stdin().read_to_end(&mut src).unwrap();
    }
    else {
        src = match read(&in_filename) {
            Ok(src) => src,
            Err(err) => panic!("Failed to open {}: {:?}", &in_filename, err),
        };
    }

    // It's nice to have a sentinel value so we don't have to keep checking bounds
    src.push(0);

    let ctx = Context { src, filename: in_filename.to_owned() };

    let mut lexer = Lexer::new(&ctx);

    let toks = lexer.tokenize();

    let mut parser = Parser::new(&ctx, &toks);

    let su = parser.source_unit();
    parser.ensure_done();

    let mut out = File::create(out_filename).unwrap();

    let mut codegen = Codegen::new(&ctx, &mut out, su);
    codegen.program();
}

fn usage(exec: &str, exit_code: i32) -> ! {
    eprintln!("{} [ -o <path> ] <file>", exec);
    exit(exit_code);
}
