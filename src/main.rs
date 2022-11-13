use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("{}: invalid number of arguments", args[0]);
        process::exit(1);
    }

    println!("  .globl main");
    println!("main:");
    println!("  mov ${}, %rax", args[1].parse::<i32>().unwrap());
    println!("  ret");
}
