use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("{}: invalid number of arguments", args[0]);
    }

    let mut buf: &str = &args[1];
    let mut n: i32;

    println!("  .globl main");
    println!("main:");
    (n, buf) = read_int(buf);
    println!("  mov ${}, %rax", n);

    while buf.len() > 0 {
        let c = buf.as_bytes()[0];
        if c == b'+' {
            (n, buf) = read_int(&buf[1..]);
            println!("  add ${}, %rax", n);
        }
        else if c == b'-' {
            (n, buf) = read_int(&buf[1..]);
            println!("  sub ${}, %rax", n);
        }
        else {
            panic!("unexpected character: '{}'", c as char);
        }
    }

    println!("  ret");
}

// chibicc uses strtol instead of something home-cooked
fn read_int(buf: &str) -> (i32, &str) {
    let mut acc = 0;
    let mut sign = 1;
    for (i, c) in buf.chars().enumerate() {
        if c == '-' && i == 0 {
            sign = -1;
        }
        else if c == '+' && i == 0 {
            // do nothing, this is the default
        }
        else if c.is_numeric() {
            acc = acc * 10 + c.to_digit(10).unwrap();
        }
        else {
            if i == 0 {
                panic!("read_int failed on input: {}", buf);
            }
            return (sign*TryInto::<i32>::try_into(acc).unwrap(), &buf[i..]);
        }
    }
    return (sign*TryInto::<i32>::try_into(acc).unwrap(), &buf[buf.len()..]);
}
