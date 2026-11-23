// src/models/inkling2/mxfp8_disk_test.cpp - the MXFP8 conversation namespace, disk side.
//
// The MXFP8 full-attention visits store E4M3 codes with one UE8M0 byte per 32 values, one aligned 1056-byte cell
// per position per K/V matrix (8 code rows of 128 bytes, then 8 rows of 4 exponent bytes). The disk tier writes a
// parked conversation under the identity `"basalt conversation state " + arch + " vision_in_engine " + N +
// rope + suffix`, where the model's disk_identity_suffix() for an MXFP8 engine is "mxfp8v1 cut=<n> drafter=<0|1>"
// (session_engine.cpp, model.hpp). This test pins that namespace:
//   1. a 131,072-position fixture with distinct scales on both K and V is written under the new identity and
//      round-trips bitwise after a fresh store (a cold read of the files);
//   2. it is not found under the legacy identity (no suffix);
//   3. an fp16 fixture (the old mode) loads in the legacy namespace and is invisible to the MXFP8 one.
//
// The fixture is constructed in host memory: the cell bytes are the same bytes the GPU caches hold, and the
// comparison is the cell-for-cell prefix of the fixture. The rings live in the running state, not in the cell
// records; the fixture's checkpoint carries distinct state bytes of the same shape.
#include "basalt/core/conversation_store.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_buffer.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace basalt::core;

int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

constexpr int64_t kCells = 131072;               // the fixture's positions, per the persistence gate
constexpr int kVisits = 8;                       // the live full-attention visits (K and V each)
constexpr int64_t kCheckpoint = 131071;          // the checkpoint at the live end: 131,071 tokens

// A deterministic byte per (visit-or-matrix, position, offset). The fixture's codes and exponent bytes are this
// value, so distinct positions, heads and groups produce distinct bytes.
static uint32_t mix32(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    return (uint32_t) (x ^ (x >> 31));
}
static uint8_t byte_at(int visit, int64_t position, size_t offset) {
    const uint64_t h = mix32((uint64_t) position * 0x9E3779B97F4A7C15ull ^ (uint64_t) visit * 0xD1B54A32D192ED03ull ^ (uint64_t) offset * 0x2545F4914F6CDD1Dull);
    return (uint8_t) (h ^ (h >> 8));
}
// One 1056-byte cell of one visit: 8 code rows (8 heads x 128 bytes), then 8 rows of 4 exponent bytes.
static void fill_cell(uint8_t* cell, int visit, int64_t position) {
    for (int head = 0; head < 8; ++head) {
        for (int dim = 0; dim < 128; ++dim)
            cell[128 * head + dim] = byte_at(visit, position, 128 * head + dim);
        for (int g = 0; g < 4; ++g)
            cell[1024 + 4 * head + g] = byte_at(visit, position, 1024 + 4 * head + g);
    }
}

// The new namespace exactly as the serve engine builds it (session_engine.cpp): the legacy text plus the suffix.
std::string identity_text(const std::string& suffix, const std::string& arch = "inkling2") {
    std::string t = "basalt conversation state " + arch + " vision_in_engine 0 ";
    if (!suffix.empty()) t += suffix;
    return t;
}
// The suffix the real Inkling model produces for its default, single-card config with no drafter (verified by code
// review against model.hpp). Other cuts/drafters produce different suffixes; the namespace property is what this
// test pins.
const std::string kSuffix = "mxfp8v1 cut=0 drafter=0";

static std::string temp_dir(const char* tag) {
    return std::string("/tmp/inkling2-disk-test-") + tag + "-" + std::to_string((long long) basalt::platform::process_id());
}
static void remove_tree(const std::string& dir) {
    std::vector<basalt::platform::DirFile> left;
    if (basalt::platform::list_files(dir, left))
        for (const auto& f : left) basalt::platform::remove_file(dir + "/" + f.name);
    basalt::platform::remove_file(dir);
}

SavedConversation fixture() {
    SavedConversation s;
    s.live.ids.reserve(kCells);
    for (int64_t i = 0; i < kCells; ++i) s.live.ids.push_back((int32_t) i);
    // the rings' running state, standing in for the rings (distinct bytes per checkpoint, as a checkpoint holds)
    s.live.state_a.resize(1u << 20);
    for (size_t i = 0; i < s.live.state_a.size(); ++i) s.live.state_a[i] = (uint8_t) (i * 31 + 7);
    s.live.state_b.resize(1024);
    for (size_t i = 0; i < s.live.state_b.size(); ++i) s.live.state_b[i] = (uint8_t) (i ^ (i << 3));
    s.geometry.fill(0);
    s.layer_lo = 0;
    s.layer_hi = 48;
    // one checkpoint at the live end: 131,071 tokens
    ConversationCheckpoint c;
    c.ids.assign(s.live.ids.begin(), s.live.ids.begin() + (std::ptrdiff_t) kCheckpoint);
    c.state_a = s.live.state_a;
    c.state_b = s.live.state_b;
    c.state_c.resize(3 * 512);
    for (size_t i = 0; i < c.state_c.size(); ++i) c.state_c[i] = (uint8_t) (i * 17 + 1);
    c.source = ConversationCheckpoint::kTurnEnd;
    s.checkpoints.push_back(std::move(c));
    // the 8 live full-attention visits, each with its own K and V
    for (int v = 0; v < kVisits; ++v) {
        ConversationKv kv;
        kv.format = 5;                  // the MXFP8 tag: 3 fp16, 1 fp8, 5 mxfp8
        kv.cells = kCells;
        kv.heads = 8;
        kv.head_dim = 128;
        kv.page_size = 1;
        kv.k.resize_for_overwrite((size_t) kCells * 1056);
        kv.k.visit(0, (size_t) kCells * 1056, [&](uint8_t* p, size_t n, size_t at) {
            for (size_t i = 0; i < n; ++i) p[i] = byte_at(v * 2, (int64_t) ((at + i) / 1056), (at + i) % 1056);
            return true;
        });
        kv.v.resize_for_overwrite((size_t) kCells * 1056);
        kv.v.visit(0, (size_t) kCells * 1056, [&](uint8_t* p, size_t n, size_t at) {
            for (size_t i = 0; i < n; ++i) p[i] = byte_at(v * 2 + 1, (int64_t) ((at + i) / 1056), (at + i) % 1056);
            return true;
        });
        s.kv.push_back(std::move(kv));
    }
    s.cvec = true;
    return s;
}

// rebuilds a visit's payload prefix exactly as the fixture filled it
static std::vector<uint8_t> reference(int visit, int64_t payload_cells) {
    std::vector<uint8_t> b((size_t) payload_cells * 1056);
    for (int64_t pos = 0; pos < payload_cells; ++pos)
        fill_cell(b.data() + (size_t) pos * 1056, visit, pos);
    return b;
}

}  // namespace

int main() {
    // ---- the namespace mechanics: distinct hashes, then the two stores ----
    const std::string legacy_text = identity_text("");
    const std::string mxfp8_text = identity_text(kSuffix);
    const uint64_t legacy_id = conversation_store_hash(legacy_text.data(), legacy_text.size(), 1);
    const uint64_t mxfp8_id = conversation_store_hash(mxfp8_text.data(), mxfp8_text.size(), 1);
    CHECK(legacy_id != mxfp8_id);
    std::printf("namespace: legacy %llu vs mxfp8 %llu - distinct\n",
                (unsigned long long) legacy_id, (unsigned long long) mxfp8_id);

    // ---- the fixture ----
    SavedConversation src = fixture();
    CHECK(src.kv.size() == 8);
    for (const auto& kv : src.kv) {
        CHECK(kv.format == 5 && kv.heads == 8 && kv.head_dim == 128 && kv.page_size == 1);
        CHECK(kv.k.size() == (size_t) kCells * 1056 && kv.v.size() == (size_t) kCells * 1056);
    }
    // distinct scales per position, on both K and V
    auto scale = [&](const ConversationKv& kv, const ConversationBuffer& buf, int64_t pos, int head) {
        std::vector<uint8_t> b(4);
        buf.read(b.data(), (size_t) pos * 1056 + 1024 + 4 * head, 4);
        return b;
    };
    bool distinct = true;
    for (int head = 0; head < 8; ++head) {
        auto a = scale(src.kv[0], src.kv[0].k, 0, head);
        auto b = scale(src.kv[0], src.kv[0].k, kCells - 1, head);
        auto c = scale(src.kv[0], src.kv[0].v, 0, head);
        if (a.size() != b.size() || std::memcmp(a.data(), b.data(), 4) == 0) distinct = false;
        if (a.size() != c.size() || std::memcmp(a.data(), c.data(), 4) == 0) distinct = false;
    }
    CHECK(distinct);
    if (!distinct) std::fprintf(stderr, "FAIL: the fixture's scales are not distinct across positions/K vs V\n");

    // ---- round-trip through the disk under the NEW namespace ----
    const std::string dir = temp_dir("mxfp8");
    remove_tree(dir);
    ConversationStore store;
    std::string err;
    CHECK(store.open(dir, mxfp8_id, 1ull << 42, err));
    if (!err.empty()) std::fprintf(stderr, "open: %s\n", err.c_str());
    CHECK(store.put(std::move(src), 1ull << 42));
    store.flush();
    CHECK(store.stats().files > 0);
    std::printf("cold read: wrote %zu files (%.1f MiB)\n", store.stats().files,
                (double) store.stats().bytes / 1048576.0);

    // a fresh store, in this process, re-reads the files (the cold-process check is the same read path)
    ConversationStore store2;
    CHECK(store2.open(dir, mxfp8_id, 1ull << 42, err));
    std::vector<int32_t> prompt(kCells);
    for (int64_t i = 0; i < kCells; ++i) prompt[i] = (int32_t) i;
    const ConversationStore::Match m = store2.best(prompt.data(), prompt.size(), {}, true);
    std::printf("match: %lld of %lld tokens\n", (long long) m.tokens, (long long) prompt.size());
    CHECK(m.tokens > 0);
    SavedConversation got;
    CHECK(store2.load(m, got, err));
    if (!err.empty()) std::fprintf(stderr, "load: %s\n", err.c_str());
    CHECK((int64_t) got.live.ids.size() == m.tokens);
    CHECK(got.checkpoints.size() == 1);
    if (!got.checkpoints.empty()) {
        CHECK((int64_t) got.checkpoints[0].ids.size() == m.tokens);
        // the rings' running state, bitwise
        std::vector<uint8_t> want(1u << 20);
        for (size_t i = 0; i < want.size(); ++i) want[i] = (uint8_t) (i * 31 + 7);
        CHECK(got.checkpoints[0].state_a.size() == want.size());
        CHECK(got.checkpoints[0].state_a.size() == want.size() &&
              std::memcmp(got.checkpoints[0].state_a.data(), want.data(), want.size()) == 0);
        std::vector<uint8_t> want_c(3 * 512);
        for (size_t i = 0; i < want_c.size(); ++i) want_c[i] = (uint8_t) (i * 17 + 1);
        CHECK(got.checkpoints[0].state_c.size() == want_c.size() &&
              std::memcmp(got.checkpoints[0].state_c.data(), want_c.data(), want_c.size()) == 0);
    }

    CHECK(got.kv.size() == kVisits);
    for (size_t i = 0; i < got.kv.size(); ++i) {
        const ConversationKv& k = got.kv[i];
        CHECK(k.format == 5);
        CHECK(k.cells == m.tokens);
        CHECK(k.heads == 8 && k.head_dim == 128 && k.page_size == 1);
        CHECK(k.k.size() == (size_t) m.tokens * 1056);
        CHECK(k.v.size() == (size_t) m.tokens * 1056);
        // the K and V payloads, bitwise, cell by cell
        auto ref_k = reference((int) i * 2, m.tokens);
        auto got_k = std::vector<uint8_t>(k.k.size());
        k.k.read(got_k.data(), 0, got_k.size());
        if (std::memcmp(ref_k.data(), got_k.data(), ref_k.size()) != 0) {
            std::fprintf(stderr, "FAIL: visit %zu k bytes differ\n", i);
            ++failures;
        } else {
            std::fprintf(stderr, "");
        }
        auto ref_v = reference((int) i * 2 + 1, m.tokens);
        auto got_v = std::vector<uint8_t>(k.v.size());
        k.v.read(got_v.data(), 0, got_v.size());
        if (std::memcmp(ref_v.data(), got_v.data(), ref_v.size()) != 0) {
            std::fprintf(stderr, "FAIL: visit %zu v bytes differ\n", i);
            ++failures;
        }
    }

    // ---- the legacy namespace must not find it ----
    ConversationStore legacy;
    CHECK(legacy.open(dir, legacy_id, 1ull << 42, err));
    const ConversationStore::Match lm = legacy.best(prompt.data(), prompt.size(), {}, true);
    CHECK(lm.tokens == 0);
    std::printf("legacy namespace: no match (rejected)\n");

    // ---- a pre-existing fp16 fixture (the old mode) ----
    const std::string old_dir = temp_dir("fp16");
    remove_tree(old_dir);
    SavedConversation old;
    old.live.ids.assign(prompt.begin(), prompt.begin() + 4097);
    old.live.state_a.resize(1u << 20);
    for (size_t i = 0; i < old.live.state_a.size(); ++i) old.live.state_a[i] = (uint8_t) (i * 13 + 5);
    old.geometry.fill(0);
    old.layer_lo = 0;
    old.layer_hi = 48;
    old.cvec = true;
    {
        ConversationCheckpoint c;
        c.ids = old.live.ids;
        c.state_a = old.live.state_a;
        c.source = ConversationCheckpoint::kTurnEnd;
        old.checkpoints.push_back(std::move(c));
        ConversationKv kv;
        kv.format = 3;                  // fp16
        kv.cells = 4097;
        kv.heads = 8;
        kv.head_dim = 128;
        kv.page_size = 1;
        kv.k.resize_for_overwrite((size_t) 4097 * 2048);
        kv.k.visit(0, (size_t) 4097 * 2048, [&](uint8_t* p, size_t n, size_t at) {
            for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) ((at + i) * 3 + 1);
            return true;
        });
        kv.v.resize_for_overwrite((size_t) 4097 * 2048);
        kv.v.visit(0, (size_t) 4097 * 2048, [&](uint8_t* p, size_t n, size_t at) {
            for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) ((at + i) * 5 + 9);
            return true;
        });
        old.kv.push_back(std::move(kv));
    }
    ConversationStore legacy2;
    CHECK(legacy2.open(old_dir, legacy_id, 1ull << 42, err));
    CHECK(legacy2.put(std::move(old), 1ull << 42));
    legacy2.flush();
    const ConversationStore::Match old_match = legacy2.best(prompt.data(), prompt.size(), {}, true);
    CHECK(old_match.tokens > 0);
    std::printf("fp16 fixture under the legacy namespace: matched at %lld\n", (long long) old_match.tokens);
    SavedConversation old_got;
    CHECK(legacy2.load(old_match, old_got, err));
    CHECK(old_got.kv.size() == 1 && old_got.kv[0].format == 3);
    {
        std::vector<uint8_t> got_k(old_got.kv[0].k.size());
        old_got.kv[0].k.read(got_k.data(), 0, got_k.size());
        bool ok = true;
        for (size_t i = 0; i < got_k.size(); ++i) ok = ok && got_k[i] == (uint8_t) ((i) * 3 + 1);
        CHECK(ok);
        (void) 0;
    }

    ConversationStore mxfp8_again;
    CHECK(mxfp8_again.open(old_dir, mxfp8_id, 1ull << 42, err));
    const ConversationStore::Match old_under_mxfp8 = mxfp8_again.best(prompt.data(), prompt.size(), {}, true);
    CHECK(old_under_mxfp8.tokens == 0);
    std::printf("fp16 fixture under the MXFP8 namespace: rejected\n");

    remove_tree(dir);
    remove_tree(old_dir);

    if (failures == 0) std::printf("inkling2_mxfp8_disk_test: ok\n");
    else std::printf("inkling2_mxfp8_disk_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
