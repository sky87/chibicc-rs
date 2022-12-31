CFLAGS=-std=c11 -g -fno-common

SRCS=$(wildcard src/*.rs)

TEST_SRCS=$(wildcard test/*.c)
TESTS_BUILD_DIR=target/tests-c
TESTS=${subst test,$(TESTS_BUILD_DIR),$(TEST_SRCS:.c=)}

BIN=./target/debug/chibicc-rs

$(BIN): $(SRCS)
	cargo build

$(TESTS_BUILD_DIR)/%: $(BIN) test/%.c
	@mkdir -p $(TESTS_BUILD_DIR)
	$(CC) -o $(TESTS_BUILD_DIR)/$*.p.c -E -P -C test/$*.c
	$(BIN) -o $(TESTS_BUILD_DIR)/$*.s $(TESTS_BUILD_DIR)/$*.p.c
	$(CC) -o $@ $(TESTS_BUILD_DIR)/$*.s -xc test/common

test: $(TESTS)
	@for i in $^; do echo $$i; ./$$i || exit 1; echo; done
	test/driver.sh

clean:
	rm -rf target

.PHONY: test clean
