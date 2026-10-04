/*
  Pikafish NNUE forward-evaluation pybind11 module.

  This module exposes Pikafish's NNUE forward evaluation to Python so that
  it can be used directly from a Texel-Tuning training pipeline.  It reuses
  the engine's own NNUE code path (network.evaluate / Eval::evaluate /
  UCIEngine::to_cp) and therefore produces values that are bit-for-bit
  identical to the UCI `eval` command.

  Python API:
      nnue_pybind.load(path)                 -> None
      nnue_pybind.evaluate(fen)              -> int   (raw NNUE, STM view)
      nnue_pybind.evaluate_scaled(fen)       -> int   (scaled,   STM view)
      nnue_pybind.evaluate_cp(fen)           -> int   (centipawn, white view)
      nnue_pybind.evaluate_batch([fen,...])  -> list[int]

  This file is compiled with -DTRAINING_TOOL (to unlock the network.h
  accessors, matching nnue_tool.cpp) and linked against the same .o files
  as the engine.  It must be built in WSL/MinGW because the engine relies
  on POSIX threads and the avxvnni code path.
*/

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector <-> list conversion

#include "../src/attacks.h"            // for Attacks::init()
#include "../src/evaluate.h"           // for Eval::evaluate()
#include "../src/misc.h"
#include "../src/nnue/network.h"       // for Network
#include "../src/nnue/nnue_accumulator.h"  // for AccumulatorStack / AccumulatorCaches
#include "../src/nnue/nnue_misc.h"     // for EvalFile
#include "../src/position.h"           // for Position, StateListPtr, StateInfo
#include "../src/types.h"
#include "../src/uci.h"                // for UCIEngine::to_cp()

namespace py = pybind11;

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

namespace {

// The network is ~64 MiB and must live on the heap (stack allocation would
// overflow).  Held as a unique_ptr so it is destroyed on module unload.
std::unique_ptr<Network> network;

// Saved from the most recent load() call so that save_network() can reuse the
// network description (header metadata) when re-serialising the network.
EvalFile g_evalFile;

// Attacks::init() / Position::init() populate global lookup tables that are
// required by Position::set() and the NNUE evaluation.  They are idempotent
// but relatively expensive, so we run them exactly once.
bool engineInitialized = false;

// Lazily initialise the engine-global lookup tables.  Called from load().
void ensure_engine_initialized() {
    if (engineInitialized)
        return;
    Attacks::init();
    Position::init();
    engineInitialized = true;
}

// Build a Position from a FEN string in place, throwing a Python-visible
// exception on failure.  Position is non-copyable/non-movable, so we populate
// it through an output reference.  The caller-supplied StateListPtr keeps the
// StateInfo chain alive for the lifetime of the position.
void make_position(const std::string& fen, StateListPtr& states, Position& pos) {
    states = StateListPtr(new std::deque<StateInfo>(1));
    auto    err = pos.set(fen, &states->back());
    if (err.has_value())
        throw std::runtime_error("Invalid FEN: " + std::string(err->what()));
}

}  // namespace

// ---------------------------------------------------------------------------
// load(path) — load a .nnue network file (zstd-compressed)
// ---------------------------------------------------------------------------
void load(const std::string& path) {
    ensure_engine_initialized();

    network = std::make_unique<Network>();

    // load_external() handles zstd decompression internally.  An empty
    // root directory makes dir/evalfilePath collapse to just evalfilePath,
    // so an absolute or cwd-relative path works directly.
    network->load_external(std::filesystem::path{}, std::filesystem::path(path), g_evalFile);

    if (!g_evalFile.current.has_value())
        throw std::runtime_error("Failed to load NNUE network from: " + path);
}

// ---------------------------------------------------------------------------
// evaluate(fen) — raw NNUE value (internal units, side-to-move perspective)
//
// This is the primary Texel-Tuning interface: it returns exactly what
// network.evaluate() returns, matching the UCI `eval` line
//   "NNUE evaluation <value> (side to move, internal units)".
// ---------------------------------------------------------------------------
int evaluate(const std::string& fen) {
    if (!network)
        throw std::runtime_error("No network loaded; call nnue_pybind.load(path) first.");

    StateListPtr states;
    Position     pos;
    make_position(fen, states, pos);

    auto accumulators = std::make_unique<AccumulatorStack>();
    auto caches       = std::make_unique<AccumulatorCaches>(*network);

    Value nnue = network->evaluate(pos, *accumulators, *caches);
    return int(nnue);
}

// ---------------------------------------------------------------------------
// evaluate_scaled(fen) — scaled NNUE value (internal units, STM perspective)
//
// Calls Eval::evaluate() which applies scale_evaluation() (material / optimism
// / rule60 scaling).  Eval::evaluate() has assert(!pos.checkers()), so for
// positions in check we fall back to the raw NNUE value (the assert would
// fire in a debug build and the scaling formula is not valid under check).
// optimism is fixed to 0 (no search context).
// ---------------------------------------------------------------------------
int evaluate_scaled(const std::string& fen) {
    if (!network)
        throw std::runtime_error("No network loaded; call nnue_pybind.load(path) first.");

    StateListPtr states;
    Position     pos;
    make_position(fen, states, pos);

    auto accumulators = std::make_unique<AccumulatorStack>();
    auto caches       = std::make_unique<AccumulatorCaches>(*network);

    if (pos.checkers())
        return int(network->evaluate(pos, *accumulators, *caches));

    Value scaled = Eval::evaluate(*network, pos, *accumulators, *caches, 0);
    return int(scaled);
}

// ---------------------------------------------------------------------------
// evaluate_cp(fen) — centipawn value (white perspective)
//
// Matches the UCI `eval` white-side output:
//   nnue = stm == WHITE ? nnue : -nnue
//   cp  = UCIEngine::to_cp(nnue, pos)
// ---------------------------------------------------------------------------
int evaluate_cp(const std::string& fen) {
    if (!network)
        throw std::runtime_error("No network loaded; call nnue_pybind.load(path) first.");

    StateListPtr states;
    Position     pos;
    make_position(fen, states, pos);

    auto accumulators = std::make_unique<AccumulatorStack>();
    auto caches       = std::make_unique<AccumulatorCaches>(*network);

    Value nnue = network->evaluate(pos, *accumulators, *caches);

    // Convert to white perspective before the win-rate-based cp transform.
    nnue = pos.side_to_move() == WHITE ? nnue : -nnue;

    return UCIEngine::to_cp(nnue, pos);
}

// ---------------------------------------------------------------------------
// evaluate_batch(fens) — batch raw NNUE (internal units, STM perspective)
//
// Convenience wrapper: loops over FENs reusing the single loaded network.
// Each position gets its own accumulator stack / cache (they are stateful).
// ---------------------------------------------------------------------------
std::vector<int> evaluate_batch(const std::vector<std::string>& fens) {
    if (!network)
        throw std::runtime_error("No network loaded; call nnue_pybind.load(path) first.");

    std::vector<int> results;
    results.reserve(fens.size());

    for (const auto& fen : fens)
    {
        StateListPtr states;
        Position     pos;
        make_position(fen, states, pos);

        auto accumulators = std::make_unique<AccumulatorStack>();
        auto caches       = std::make_unique<AccumulatorCaches>(*network);

        Value nnue = network->evaluate(pos, *accumulators, *caches);
        results.push_back(int(nnue));
    }
    return results;
}

// ---------------------------------------------------------------------------
// evaluate_with_trace(fen) — evaluation with intermediate activations
//
// Returns a Python dict with the final value plus all intermediate buffers
// (fc_0_out, fc_1_out, concat_buffer = fc_2 input, skip_0, psqt, bucket).
// These are needed by the fc_2 Texel-Tuning gradient computation.  The
// "value" entry is bit-for-bit identical to evaluate(fen).
// ---------------------------------------------------------------------------
py::dict evaluate_with_trace(const std::string& fen) {
    if (!network)
        throw std::runtime_error("No network loaded; call nnue_pybind.load(path) first.");

    StateListPtr states;
    Position     pos;
    make_position(fen, states, pos);

    auto accumulators = std::make_unique<AccumulatorStack>();
    auto caches       = std::make_unique<AccumulatorCaches>(*network);

    constexpr u64 alignment = CacheLineSize;
    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];
    NNZInfo<L1> nnzInfo;

    const int  bucket = PSQFeatureSet::make_layer_stack_bucket(pos);
    const auto psqt   = network->get_feature_transformer().transform(
        pos, *accumulators, *caches, transformedFeatures, bucket, nnzInfo);
    const auto trace  = network->get_network(bucket).propagate_with_trace(transformedFeatures, nnzInfo);

    py::dict result;
    result["value"]      = static_cast<int>(psqt / OutputScale + trace.output / OutputScale);
    result["bucket"]     = bucket;
    result["psqt"]       = static_cast<int>(psqt);
    result["positional"] = trace.output;
    result["skip_0"]     = trace.skip_0;
    result["fwd_out"]    = trace.fwd_out;  // raw fc_2_out + skip_0 (i32, pre 9600/16384 scale)

    py::list concat;
    for (int i = 0; i < NetworkArchitecture::FC_0_OUTPUTS * 2 + NetworkArchitecture::FC_1_OUTPUTS * 2; ++i)
        concat.append(static_cast<int>(trace.concat_buffer[i]));
    result["concat_buffer"] = concat;

    py::list fc0;
    for (int i = 0; i < NetworkArchitecture::FC_0_OUTPUTS; ++i)
        fc0.append(static_cast<int>(trace.fc_0_out[i]));
    result["fc_0_out"] = fc0;

    py::list fc1;
    for (int i = 0; i < NetworkArchitecture::FC_1_OUTPUTS; ++i)
        fc1.append(static_cast<int>(trace.fc_1_out[i]));
    result["fc_1_out"] = fc1;

    return result;
}

// ---------------------------------------------------------------------------
// fc_2 parameter accessors
//
// fc_2 is AffineTransform<FC_0_OUTPUTS*2 + FC_1_OUTPUTS*2, 1> = <128, 1>.
// Each of the LayerStacks (=16) layer stacks has its own fc_2 with 128 int8
// weights and 1 int32 bias.  Weights are clamped to [-128, 127] on set.
// ---------------------------------------------------------------------------
int get_fc2_weight(int bucket, int idx) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc2 = network->get_network(bucket).fc_2;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc2)>::InputDimensions)
        throw std::runtime_error("idx out of range");
    return static_cast<int>(fc2.get_weight(idx));
}

void set_fc2_weight(int bucket, int idx, int value) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc2 = network->get_network(bucket).fc_2;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc2)>::InputDimensions)
        throw std::runtime_error("idx out of range");
    fc2.set_weight(idx, static_cast<i8>(std::clamp(value, -128, 127)));
}

int get_fc2_bias(int bucket) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    return static_cast<int>(network->get_network(bucket).fc_2.get_bias(0));
}

void set_fc2_bias(int bucket, int value) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    network->get_network(bucket).fc_2.set_bias(0, value);
}

// ---------------------------------------------------------------------------
// fc_1 parameter accessors
//
// fc_1 is AffineTransform<FC_0_OUTPUTS*2, FC_1_OUTPUTS> = <64, 32>.
// Each of the LayerStacks (=16) layer stacks has its own fc_1 with 64*32=2048
// int8 weights and 32 int32 biases.  Weight index i = output*64 + input,
// i.e. i in [0, 2047].  Bias index k in [0, 31].
// Weights are clamped to [-128, 127] on set.
// ---------------------------------------------------------------------------
int get_fc1_weight(int bucket, int idx) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc1 = network->get_network(bucket).fc_1;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc1)>::num_weights())
        throw std::runtime_error("idx out of range");
    return static_cast<int>(fc1.get_weight(idx));
}

void set_fc1_weight(int bucket, int idx, int value) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc1 = network->get_network(bucket).fc_1;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc1)>::num_weights())
        throw std::runtime_error("idx out of range");
    fc1.set_weight(idx, static_cast<i8>(std::clamp(value, -128, 127)));
}

int get_fc1_bias(int bucket, int idx) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc1 = network->get_network(bucket).fc_1;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc1)>::num_biases())
        throw std::runtime_error("idx out of range");
    return static_cast<int>(fc1.get_bias(idx));
}

void set_fc1_bias(int bucket, int idx, int value) {
    if (!network)
        throw std::runtime_error("No network loaded.");
    if (bucket < 0 || bucket >= LayerStacks)
        throw std::runtime_error("bucket out of range");
    auto& fc1 = network->get_network(bucket).fc_1;
    if (idx < 0 || idx >= (int) std::remove_reference_t<decltype(fc1)>::num_biases())
        throw std::runtime_error("idx out of range");
    fc1.set_bias(idx, value);
}

// ---------------------------------------------------------------------------
// save_network(path) — write the current network to a zstd-compressed .nnue
//
// Mirrors nnue_tool.cpp: serialise to an uncompressed binary stream via
// tool_write_parameters(), write to a temp file, then call the `zstd` CLI
// (same external tool the engine relies on) to produce the final file.
// ---------------------------------------------------------------------------
void save_network(const std::string& path) {
    if (!network)
        throw std::runtime_error("No network loaded.");

    // 1. Serialise the network (header + parameters) to an in-memory buffer.
    std::ostringstream oss(std::ios::binary);
    if (!network->tool_write_parameters(oss, g_evalFile.netDescription))
        throw std::runtime_error("Failed to write network parameters.");

    // 2. Write the raw bytes to a temporary file.
    std::string tmp_path = path + ".tmp";
    std::ofstream fout(tmp_path, std::ios::binary);
    if (!fout)
        throw std::runtime_error("Failed to open temp file: " + tmp_path);
    std::string raw = oss.str();
    fout.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    fout.close();

    // 3. zstd-compress with the same high level the engine uses.
    std::string cmd = "zstd -f -q -19 " + tmp_path + " -o " + path;
    int         ret = std::system(cmd.c_str());
    std::remove(tmp_path.c_str());
    if (ret != 0)
        throw std::runtime_error("zstd compression failed with code " + std::to_string(ret));
}

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(nnue_pybind, m) {
    m.doc() = "Pikafish NNUE forward evaluation module";

    m.def("load", &load, py::arg("path"),
          "Load a .nnue network file (zstd-compressed). Must be called first.");

    m.def("evaluate", &evaluate, py::arg("fen"),
          "Raw NNUE evaluation in internal units (side-to-move perspective). "
          "Matches UCI 'eval' NNUE value line.");

    m.def("evaluate_scaled", &evaluate_scaled, py::arg("fen"),
          "Scaled NNUE evaluation in internal units (side-to-move perspective, optimism=0). "
          "Falls back to raw NNUE for positions in check.");

    m.def("evaluate_cp", &evaluate_cp, py::arg("fen"),
          "Centipawn evaluation (white perspective). Matches UCI 'eval' white-side cp output.");

    m.def("evaluate_batch", &evaluate_batch, py::arg("fens"),
          "Batch raw NNUE evaluation in internal units (side-to-move perspective).");

    m.def("evaluate_with_trace", &evaluate_with_trace, py::arg("fen"),
          "Evaluate with intermediate activations for training. "
          "Returns dict with value, bucket, concat_buffer (128-dim fc_2 input), etc.");

    m.def("get_fc2_weight", &get_fc2_weight, py::arg("bucket"), py::arg("idx"),
          "Get fc_2 weight (int8) for given layer stack bucket and input index (0-127).");

    m.def("set_fc2_weight", &set_fc2_weight, py::arg("bucket"), py::arg("idx"), py::arg("value"),
          "Set fc_2 weight (int8, clipped to [-128,127]) for given layer stack bucket and input index.");

    m.def("get_fc2_bias", &get_fc2_bias, py::arg("bucket"),
          "Get fc_2 bias (int32) for given layer stack bucket.");

    m.def("set_fc2_bias", &set_fc2_bias, py::arg("bucket"), py::arg("value"),
          "Set fc_2 bias (int32) for given layer stack bucket.");

    m.def("get_fc1_weight", &get_fc1_weight, py::arg("bucket"), py::arg("idx"),
          "Get fc_1 weight (int8) for given bucket. idx = output*64 + input, range [0, 2047].");

    m.def("set_fc1_weight", &set_fc1_weight, py::arg("bucket"), py::arg("idx"), py::arg("value"),
          "Set fc_1 weight (int8, clipped to [-128,127]) for given bucket. idx = output*64 + input.");

    m.def("get_fc1_bias", &get_fc1_bias, py::arg("bucket"), py::arg("idx"),
          "Get fc_1 bias (int32) for given bucket and output index [0, 31].");

    m.def("set_fc1_bias", &set_fc1_bias, py::arg("bucket"), py::arg("idx"), py::arg("value"),
          "Set fc_1 bias (int32) for given bucket and output index [0, 31].");

    m.def("save_network", &save_network, py::arg("path"),
          "Save the current network to a zstd-compressed .nnue file.");
}