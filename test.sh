#!/bin/bash
assert() {
  expected="$1"
  input="$2"

  ./target/debug/chibicc-rs "$input" > tmp.s || exit
  gcc -static -o tmp tmp.s
  ./tmp
  actual="$?"

  if [ "$actual" = "$expected" ]; then
    echo "$input => $actual"
  else
    echo "$input => $expected expected, but got $actual"
    exit 1
  fi
}

cargo build || exit

export RUST_BACKTRACE=1

assert 0 0
assert 42 42
assert 21 '5+20-4'
assert 41 ' 12 +  34 - 5'
assert 47 '5+6*7'
assert 15 '5*(9-6)'
assert 4 '(3+5)/2'

rm -f tmp tmp.s

echo OK
