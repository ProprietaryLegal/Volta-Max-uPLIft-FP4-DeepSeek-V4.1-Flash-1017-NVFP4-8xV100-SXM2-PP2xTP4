// P3 speed bench, arm (b) host compute: V4.1 routed experts on the CPU straight
// from the NVFP4-8R pack records (row-major MXFP4, low nibble = even element,
// E8M0 per 32). Bench code only; not part of the engine. AVX2 + FMA (the 8-card host's
// Zen 2 has no AVX-512), OpenMP over output rows.
// Record (bytes): w1 codes [576][2560] | w1 E8M0 [576][160] | w3 codes | w3 E8M0 |
//                 w2 codes [5120][320] | w2 E8M0 [5120][20]; W2 K 576 padded to 640.
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define HIDDEN 5120
#define INTER 576
#define W13_CODES (INTER * HIDDEN / 2)
#define W13_SCALES (INTER * HIDDEN / 32)
#define W2_CODES (HIDDEN * 640 / 2)
#define W2_SCALES (HIDDEN * 640 / 32)
#define MAXM 16

static inline float e8m0(uint8_t s) {
  uint32_t bits = (uint32_t)s << 23;  // 2^(s-127), exact for s in 1..254
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// dot of one MXFP4 row (nblk blocks of 32) with m activation rows (stride ld).
static inline void row_dot(const uint8_t *codes, const uint8_t *scales, int nblk,
                           const float *x, int ld, int m, float *out) {
  // 2 x e2m1 as int8: 0,1,2,3,4,6,8,12 and negatives.
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i lo4 = _mm_set1_epi8(0x0f);
  __m256 acc[MAXM];
  for (int r = 0; r < m; r++) acc[r] = _mm256_setzero_ps();
  for (int b = 0; b < nblk; b++) {
    __m128i raw = _mm_loadu_si128((const __m128i *)(codes + 16 * b));
    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, lo4));
    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), lo4));
    __m128i v0 = _mm_unpacklo_epi8(lo, hi);  // elements 0..15
    __m128i v1 = _mm_unpackhi_epi8(lo, hi);  // elements 16..31
    __m256 w[4];
    w[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0));
    w[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8)));
    w[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1));
    w[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8)));
    __m256 sc = _mm256_set1_ps(0.5f * e8m0(scales[b]));
    for (int r = 0; r < m; r++) {
      const float *xr = x + (size_t)r * ld + 32 * b;
      __m256 s = _mm256_mul_ps(w[0], _mm256_loadu_ps(xr));
      s = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(xr + 8), s);
      s = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xr + 16), s);
      s = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xr + 24), s);
      acc[r] = _mm256_fmadd_ps(sc, s, acc[r]);
    }
  }
  for (int r = 0; r < m; r++) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[r]), _mm256_extractf128_ps(acc[r], 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    out[r] = _mm_cvtss_f32(h);
  }
}

// out[rows[i]][:] += weight[i] * expert(x[rows[i]]) for one expert TP slice.
// x: [*, 5120] fp32; act: scratch [MAXM][640]; gu: scratch [MAXM][1152].
void expert_slice(const uint8_t *rec, const float *x, const int *rows, const float *weight,
                  int m, float *out, float *xs, float *gu, float *act) {
  const uint8_t *w1c = rec, *w1s = w1c + W13_CODES, *w3c = w1s + W13_SCALES;
  const uint8_t *w3s = w3c + W13_CODES, *w2c = w3s + W13_SCALES, *w2s = w2c + W2_CODES;
  for (int i = 0; i < m; i++) memcpy(xs + (size_t)i * HIDDEN, x + (size_t)rows[i] * HIDDEN, HIDDEN * 4);
#pragma omp parallel
  {
    float tmp[MAXM];
#pragma omp for schedule(static)
    for (int n = 0; n < 2 * INTER; n++) {
      const uint8_t *c = n < INTER ? w1c + (size_t)n * (HIDDEN / 2)
                                   : w3c + (size_t)(n - INTER) * (HIDDEN / 2);
      const uint8_t *s = n < INTER ? w1s + (size_t)n * (HIDDEN / 32)
                                   : w3s + (size_t)(n - INTER) * (HIDDEN / 32);
      row_dot(c, s, HIDDEN / 32, xs, HIDDEN, m, tmp);
      for (int i = 0; i < m; i++) gu[i * 2 * INTER + n] = tmp[i];
    }
#pragma omp for schedule(static)
    for (int i = 0; i < m * INTER; i++) {
      int r = i / INTER, j = i % INTER;
      float g = fminf(gu[r * 2 * INTER + j], 10.0f);
      float u = fmaxf(fminf(gu[r * 2 * INTER + INTER + j], 10.0f), -10.0f);
      act[r * 640 + j] = g / (1.0f + expf(-g)) * u;
    }
#pragma omp for schedule(static)
    for (int n = 0; n < HIDDEN; n++) {
      row_dot(w2c + (size_t)n * 320, w2s + (size_t)n * 20, INTER / 32, act, 640, m, tmp);
      for (int i = 0; i < m; i++) out[(size_t)rows[i] * HIDDEN + n] += weight[i] * tmp[i];
    }
  }
}
