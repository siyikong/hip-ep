/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "kv_cache_manager.h"

#include <cstdio>

namespace {

GqaDecodeCachePlan reject(const KvCachePlanInput &input, KvCacheReason reason) {
  return {KvCacheDecision::Reject, reason, input.past_len, 0,
          input.present_seq};
}

GqaDecodeCachePlan fallback(const KvCachePlanInput &input,
                            KvCacheReason reason) {
  return {KvCacheDecision::ConcatFallback, reason, input.past_len,
          input.past_len + input.current_len, input.present_seq};
}

void record_first_reason(KvCacheCounters &counters, KvCacheReason reason) {
  if (reason != KvCacheReason::None &&
      counters.first_reason == KvCacheReason::None) {
    counters.first_reason = reason;
  }
}

const char *summary_status(const KvCacheCounters &counters) {
  if (counters.kernel_failures != 0)
    return "runtime_error";
  if (counters.reject_plans != 0)
    return "rejected";
  if (counters.fallback_plans != 0)
    return "fallback";
  return "ok";
}

} // namespace

GqaDecodeCachePlan build_gqa_decode_cache_plan(const KvCachePlanInput &input) {
  if (!input.present_key || !input.present_value)
    return reject(input, KvCacheReason::OutputBindingFailed);
  if (!input.geometry_valid)
    return reject(input, KvCacheReason::GeometryMismatch);
  if (input.binding_stale)
    return reject(input, KvCacheReason::StaleBinding);
  if (input.past_len < 0 || input.current_len <= 0 || input.present_seq <= 0 ||
      input.past_buf_seq < 0)
    return reject(input, KvCacheReason::InvalidLength);
  if (input.past_len > input.present_seq ||
      input.current_len > input.present_seq - input.past_len)
    return reject(input, KvCacheReason::CapacityExceeded);

  const bool has_past_key = input.past_key != nullptr;
  const bool has_past_value = input.past_value != nullptr;
  if (has_past_key != has_past_value)
    return reject(input, KvCacheReason::PartialAlias);

  // Prefill has no valid past tokens. OGA may still bind a C-sized past
  // view that aliases present, so pointer presence is not a length signal.
  if (input.past_len == 0) {
    if (!input.request_supported)
      return reject(input, KvCacheReason::UnsupportedRequestShape);
    return {KvCacheDecision::Append, KvCacheReason::None, 0, input.current_len,
            input.present_seq};
  }

  const bool key_alias = input.past_key == input.present_key;
  const bool value_alias = input.past_value == input.present_value;
  if (key_alias != value_alias)
    return reject(input, KvCacheReason::PartialAlias);

  if (!input.request_supported) {
    if (!key_alias && input.concat_safe)
      return fallback(input, KvCacheReason::UnsupportedRequestShape);
    return reject(input, KvCacheReason::UnsupportedRequestShape);
  }

  if (key_alias) {
    if (input.past_buf_seq != input.present_seq)
      return reject(input, KvCacheReason::GeometryMismatch);
    return {KvCacheDecision::Append, KvCacheReason::None, input.past_len,
            input.past_len + input.current_len, input.present_seq};
  }

  if (!input.concat_safe)
    return reject(input, KvCacheReason::OutputBindingFailed);
  return fallback(input, KvCacheReason::PointerMismatch);
}

const char *kv_cache_decision_string(KvCacheDecision decision) {
  switch (decision) {
  case KvCacheDecision::Append:
    return "append";
  case KvCacheDecision::ConcatFallback:
    return "concat_fallback";
  case KvCacheDecision::Reject:
    return "reject";
  }
  return "invalid";
}

const char *kv_cache_reason_string(KvCacheReason reason) {
  switch (reason) {
  case KvCacheReason::None:
    return "none";
  case KvCacheReason::CacheNotBound:
    return "cache_not_bound";
  case KvCacheReason::PointerMismatch:
    return "pointer_mismatch";
  case KvCacheReason::UnsupportedRequestShape:
    return "unsupported_request_shape";
  case KvCacheReason::PartialAlias:
    return "partial_alias";
  case KvCacheReason::GeometryMismatch:
    return "geometry_mismatch";
  case KvCacheReason::InvalidLength:
    return "invalid_length";
  case KvCacheReason::CapacityExceeded:
    return "capacity_exceeded";
  case KvCacheReason::StaleBinding:
    return "stale_binding";
  case KvCacheReason::KernelFailure:
    return "kernel_failure";
  case KvCacheReason::OutputBindingFailed:
    return "output_binding_failed";
  }
  return "invalid";
}

void kv_cache_record_plan(KvCacheCounters &counters,
                          const GqaDecodeCachePlan &plan, bool prefill) {
  record_first_reason(counters, plan.reason);
  switch (plan.decision) {
  case KvCacheDecision::Append:
    if (prefill)
      ++counters.prefill_append_plans;
    else
      ++counters.decode_append_plans;
    break;
  case KvCacheDecision::ConcatFallback:
    ++counters.concat_plans;
    ++counters.fallback_plans;
    break;
  case KvCacheDecision::Reject:
    ++counters.reject_plans;
    break;
  }
}

void kv_cache_record_append_kernel(KvCacheCounters &counters, bool success) {
  if (success)
    ++counters.append_kernels;
  else {
    ++counters.kernel_failures;
    record_first_reason(counters, KvCacheReason::KernelFailure);
  }
}

void kv_cache_record_concat_kernel(KvCacheCounters &counters, bool success) {
  if (success)
    ++counters.concat_kernels;
  else {
    ++counters.kernel_failures;
    record_first_reason(counters, KvCacheReason::KernelFailure);
  }
}

size_t kv_cache_summary_json(const KvCacheCounters &counters,
                             uint64_t session_generation,
                             uint64_t arena_generation, int64_t capacity,
                             char *output, size_t output_size) {
  const int n = std::snprintf(
      output, output_size,
      "{\"schema\":1,\"event\":\"summary\",\"status\":\"%s\","
      "\"session_generation\":%llu,\"arena_generation\":%llu,"
      "\"capacity\":%lld,\"prefill_append_plans\":%llu,"
      "\"decode_append_plans\":%llu,\"append_kernels\":%llu,"
      "\"concat_plans\":%llu,\"concat_kernels\":%llu,"
      "\"fallback_plans\":%llu,\"reject_plans\":%llu,"
      "\"kernel_failures\":%llu,\"first_reason\":\"%s\"}",
      summary_status(counters),
      static_cast<unsigned long long>(session_generation),
      static_cast<unsigned long long>(arena_generation),
      static_cast<long long>(capacity),
      static_cast<unsigned long long>(counters.prefill_append_plans),
      static_cast<unsigned long long>(counters.decode_append_plans),
      static_cast<unsigned long long>(counters.append_kernels),
      static_cast<unsigned long long>(counters.concat_plans),
      static_cast<unsigned long long>(counters.concat_kernels),
      static_cast<unsigned long long>(counters.fallback_plans),
      static_cast<unsigned long long>(counters.reject_plans),
      static_cast<unsigned long long>(counters.kernel_failures),
      kv_cache_reason_string(counters.first_reason));
  return n < 0 ? 0 : static_cast<size_t>(n);
}
