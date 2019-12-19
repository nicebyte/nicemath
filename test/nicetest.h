#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <setjmp.h>

typedef struct nt_test {
  void          (*fn)();
  const char     *name;
  struct nt_test *next;
} nt_test;

extern __thread jmp_buf nt_jmpbuf;

bool     nt_run_tests(const nt_test*);
nt_test* nt_add_test (nt_test* n, nt_test **suite);

#define NTSUITE(NAME) nt_test *NAME = NULL

#define NTTEST(NAME, SUITE) \
  void NAME(); \
  nt_test NAME##_test = nt_test { .fn   = NAME, \
                                  .name = #NAME, \
                                  .next = nt_add_test(&NAME##_test, &SUITE) }; \
  void NAME()

#define NTASSERT(CONDITION) \
if (!(CONDITION)) { \
  fprintf(stderr, "  %s %s:%d assertion failed: [%s] \n", \
          __func__,  __FILE__, __LINE__, #CONDITION); \
  longjmp(nt_jmpbuf, 1); \
}

NTSUITE(global);

#if defined(NT_TEST_IMPL)

__thread jmp_buf nt_jmpbuf;
nt_test *nt_all_tests = NULL;

nt_test* nt_add_test(nt_test* n, nt_test** suite) {
  nt_test *tmp = *suite; *suite = n; return tmp;
}

bool nt_run_tests(const nt_test *suite) {
  int failed = 0, total = 0;
  for (const nt_test *t = suite; t; t = t->next, ++total) {
    fprintf(stderr, "\nrunning test %s\n", t->name);
    if (setjmp(nt_jmpbuf) == 0) { t->fn(); fprintf(stderr, "OK\n"); }
    else { fprintf(stderr, "FAIL\n"); ++failed; }
  }
  fprintf(stderr, "%d total %d failed\n", total, failed);
  return failed == 0;
}

#endif

