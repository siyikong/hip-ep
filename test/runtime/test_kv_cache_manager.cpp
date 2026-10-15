/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "kv_cache_manager.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

KvCachePlanInput base_input() {
  static int past_key;
  static int past_value;
  static int present_key;
  static int present_value;
  KvCachePlanInput input;
  input.past_key = &past_key;
  input.past_value = &past_value;
  input.present_key = &present_key;
  input.present_value = &present_value;
  input.past_len = 274;
  input.current_len = 1;
  input.past_buf_seq = 274;
  input.present_seq = 275;
  return input;
}

void test_separate_buffer_fallback() {
  const auto plan = build_gqa_decode_cache_plan(base_input());
  CHECK(plan.decision == KvCacheDecision::ConcatFallback);
  CHECK(plan.reason == KvCacheReason::PointerMismatch);
  CHECK(plan.total_seq == 275);
}

void test_alias_append() {
  auto input = base_input();
  input.present_key = const_cast<void *>(input.past_key);
  input.present_value = const_cast<void *>(input.past_value);
  input.past_buf_seq = 4096;
  input.present_seq = 4096;
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Append);
  CHECK(plan.reason == KvCacheReason::None);
  CHECK(plan.capacity == 4096);
  CHECK(plan.total_seq == 275);
}

void test_prefill_append() {
  auto input = base_input();
  input.past_len = 0;
  input.current_len = 274;
  input.past_buf_seq = 0;
  input.present_seq = 4096;
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Append);
  CHECK(plan.total_seq == 274);
}

void test_partial_alias_rejected() {
  auto input = base_input();
  input.present_key = const_cast<void *>(input.past_key);
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Reject);
  CHECK(plan.reason == KvCacheReason::PartialAlias);
}

void test_capacity_rejected() {
  auto input = base_input();
  input.past_len = 4096;
  input.current_len = 1;
  input.past_buf_seq = 4096;
  input.present_seq = 4096;
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Reject);
  CHECK(plan.reason == KvCacheReason::CapacityExceeded);
}

void test_geometry_rejected() {
  auto input = base_input();
  input.geometry_valid = false;
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Reject);
  CHECK(plan.reason == KvCacheReason::GeometryMismatch);
}

void test_stale_binding_rejected() {
  auto input = base_input();
  input.binding_stale = true;
  const auto plan = build_gqa_decode_cache_plan(input);
  CHECK(plan.decision == KvCacheDecision::Reject);
  CHECK(plan.reason == KvCacheReason::StaleBinding);
}

void test_unsupported_request_routes_safely() {
  auto separate = base_input();
  separate.request_supported = false;
  auto plan = build_gqa_decode_cache_plan(separate);
  CHECK(plan.decision == KvCacheDecision::ConcatFallback);
  CHECK(plan.reason == KvCacheReason::UnsupportedRequestShape);

  auto alias = base_input();
  alias.request_supported = false;
  alias.present_key = const_cast<void *>(alias.past_key);
  alias.present_value = const_cast<void *>(alias.past_value);
  plan = build_gqa_decode_cache_plan(alias);
  CHECK(plan.decision == KvCacheDecision::Reject);
  CHECK(plan.reason == KvCacheReason::UnsupportedRequestShape);
}

void test_status_strings() {
  CHECK(std::string(kv_cache_decision_string(KvCacheDecision::Append)) ==
        "append");
  CHECK(std::string(kv_cache_decision_string(
            KvCacheDecision::ConcatFallback)) == "concat_fallback");
  CHECK(std::string(kv_cache_reason_string(KvCacheReason::CapacityExceeded)) ==
        "capacity_exceeded");
}

void test_counters_and_summary() {
  KvCacheCounters counters;

  GqaDecodeCachePlan prefill{KvCacheDecision::Append, KvCacheReason::None, 0,
                             274, 4096};
  GqaDecodeCachePlan decode{KvCacheDecision::Append, KvCacheReason::None, 274,
                            275, 4096};
  GqaDecodeCachePlan fallback{KvCacheDecision::ConcatFallback,
                              KvCacheReason::PointerMismatch, 274, 275, 275};
  kv_cache_record_plan(counters, prefill, true);
  kv_cache_record_plan(counters, decode, false);
  kv_cache_record_plan(counters, fallback, false);
  for (int i = 0; i < 4; ++i)
    kv_cache_record_append_kernel(counters, true);
  for (int i = 0; i < 2; ++i)
    kv_cache_record_concat_kernel(counters, true);

  CHECK(counters.prefill_append_plans == 1);
  CHECK(counters.decode_append_plans == 1);
  CHECK(counters.fallback_plans == 1);
  CHECK(counters.first_reason == KvCacheReason::PointerMismatch);

  char buffer[1024];
  const size_t length =
      kv_cache_summary_json(counters, 7, 3, 4096, buffer, sizeof(buffer));
  CHECK(length > 0 && length < sizeof(buffer));
  const std::string json(buffer);
  CHECK(json.find("\"schema\":1") != std::string::npos);
  CHECK(json.find("\"status\":\"fallback\"") != std::string::npos);
  CHECK(json.find("\"session_generation\":7") != std::string::npos);
  CHECK(json.find("\"arena_generation\":3") != std::string::npos);
  CHECK(json.find("\"append_kernels\":4") != std::string::npos);
  CHECK(json.find("\"concat_kernels\":2") != std::string::npos);
  CHECK(json.find("\"first_reason\":\"pointer_mismatch\"") !=
        std::string::npos);
}

void test_kernel_failure_dominates_summary() {
  KvCacheCounters counters;
  char buffer[1024];
  kv_cache_record_append_kernel(counters, false);
  kv_cache_summary_json(counters, 1, 1, 4096, buffer, sizeof(buffer));
  const std::string json(buffer);
  CHECK(json.find("\"status\":\"runtime_error\"") != std::string::npos);
  CHECK(json.find("\"first_reason\":\"kernel_failure\"") != std::string::npos);
}

} // namespace

int main() {
  test_separate_buffer_fallback();
  test_alias_append();
  test_prefill_append();
  test_partial_alias_rejected();
  test_capacity_rejected();
  test_geometry_rejected();
  test_stale_binding_rejected();
  test_unsupported_request_routes_safely();
  test_status_strings();
  test_counters_and_summary();
  test_kernel_failure_dominates_summary();

  if (failures == 0) {
    std::printf("kv_cache_manager unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "kv_cache_manager unit test: %d FAILURE(S)\n", failures);
  return 1;
}
