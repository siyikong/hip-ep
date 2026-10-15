/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstddef>
#include <cstdint>

enum class KvCacheDecision : uint8_t {
  Append,
  ConcatFallback,
  Reject,
};

enum class KvCacheReason : uint8_t {
  None,
  CacheNotBound,
  PointerMismatch,
  UnsupportedRequestShape,
  PartialAlias,
  GeometryMismatch,
  InvalidLength,
  CapacityExceeded,
  StaleBinding,
  KernelFailure,
  OutputBindingFailed,
};

struct KvCachePlanInput {
  const void *past_key = nullptr;
  const void *past_value = nullptr;
  const void *present_key = nullptr;
  const void *present_value = nullptr;

  int64_t past_len = 0;
  int64_t current_len = 0;
  int64_t past_buf_seq = 0;
  int64_t present_seq = 0;

  bool geometry_valid = true;
  bool request_supported = true;
  bool concat_safe = true;
  bool binding_stale = false;
};

struct GqaDecodeCachePlan {
  KvCacheDecision decision = KvCacheDecision::Reject;
  KvCacheReason reason = KvCacheReason::OutputBindingFailed;
  int64_t past_len = 0;
  int64_t total_seq = 0;
  int64_t capacity = 0;
};

struct KvCacheCounters {
  uint64_t prefill_append_plans = 0;
  uint64_t decode_append_plans = 0;
  uint64_t concat_plans = 0;
  uint64_t fallback_plans = 0;
  uint64_t reject_plans = 0;
  uint64_t append_kernels = 0;
  uint64_t concat_kernels = 0;
  uint64_t kernel_failures = 0;
  KvCacheReason first_reason = KvCacheReason::None;
};

GqaDecodeCachePlan build_gqa_decode_cache_plan(const KvCachePlanInput &input);

const char *kv_cache_decision_string(KvCacheDecision decision);
const char *kv_cache_reason_string(KvCacheReason reason);

void kv_cache_record_plan(KvCacheCounters &counters,
                          const GqaDecodeCachePlan &plan, bool prefill);
void kv_cache_record_append_kernel(KvCacheCounters &counters, bool success);
void kv_cache_record_concat_kernel(KvCacheCounters &counters, bool success);

// Emits the stable schema=1 summary payload. Logging policy and line prefix
// belong to the caller so formatting can be tested without stderr capture.
// Returns the number of chars that would have been written, excluding NUL.
size_t kv_cache_summary_json(const KvCacheCounters &counters,
                             uint64_t session_generation,
                             uint64_t arena_generation, int64_t capacity,
                             char *output, size_t output_size);
