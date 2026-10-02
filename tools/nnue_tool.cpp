/*
  Pikafish NNUE weight-level extraction / injection tool.

  This tool operates at the *weight* level: it serialises the full NNUE
  network to a plain (uncompressed) binary blob using the very same
  read_parameters/write_parameters code path as the engine, and can also
  inject such a blob back, producing a zstd-compressed .nnue file that
  the engine can load directly.

  Usage:
    nnue_tool extract <input.nnue> <output.bin>
        Extract all weights from a (zstd-compressed) .nnue file into a
        plain binary file in standard (unpermuted/unscrambled) layout.

    nnue_tool inject <input.bin> <output.nnue>
        Inject weights from a plain binary file and write a
        zstd-compressed .nnue file loadable by the engine.

  The round-trip
      extract pikafish.nnue -> inject -> .nnue
  is guaranteed to be byte-identical to the original network because we
  reuse the engine's own (de)serialisation logic.

  This file is compiled with -DTRAINING_TOOL which flips the
  #ifdef TRAINING_TOOL accessors in network.h.

  Note on zstd: the engine only embeds the zstd *decompression* sources,
  so for compression we shell out to the system `zstd` CLI. This keeps
  the tool self-contained and avoids symbol clashes with the embedded
  decompressor.
*/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/nnue/network.h"
#include "../src/nnue/nnue_common.h"
#include "../src/misc.h"
#include "../src/types.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

namespace {

// Print usage and exit with the given code.
[[noreturn]] void usage(int code) {
    std::cerr << "Pikafish NNUE weight-level tool\n"
              << "Usage:\n"
              << "  nnue_tool extract <input.nnue> <output.bin>\n"
              << "  nnue_tool inject  <input.bin>  <output.nnue>\n";
    std::exit(code);
}

// Read an entire file into a string. Returns false on failure.
bool read_file(const std::string& path, std::string& out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        return false;
    std::ostringstream ss;
    ss << fin.rdbuf();
    out = ss.str();
    return true;
}

// Write a string to a file. Returns false on failure.
bool write_file(const std::string& path, const std::string& data) {
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout)
        return false;
    fout.write(data.data(), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(fout);
}

// Compress <raw_path> into <out_path> using the system zstd CLI at the
// given level. Returns true on success.
bool zstd_compress_file(const std::string& raw_path,
                        const std::string& out_path,
                        int                level) {
    // zstd -f -q -<level> <input> -o <output>
    std::string cmd = "zstd -f -q -" + std::to_string(level)
                    + " " + raw_path + " -o " + out_path;
    return std::system(cmd.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// extract: .nnue (zstd) -> plain binary blob
// ---------------------------------------------------------------------------
int do_extract(const std::string& input_nnue, const std::string& output_bin) {
    // Network is ~64 MiB, so it must live on the heap.
    auto          network = std::make_unique<Network>();
    EvalFile      evalFile;

    // load_external handles zstd decompression internally via
    // read_compressed_nnue(). An empty directory makes dir/evalfilePath
    // collapse to just evalfilePath.
    network->load_external(std::filesystem::path{}, input_nnue, evalFile);

    if (!evalFile.current.has_value())
    {
        std::cerr << "Failed to load network from " << input_nnue << '\n';
        return 1;
    }

    std::ostringstream oss(std::ios::binary);
    if (!network->tool_write_parameters(oss, evalFile.netDescription))
    {
        std::cerr << "Failed to serialise network parameters.\n";
        return 1;
    }

    const std::string blob = oss.str();
    if (!write_file(output_bin, blob))
    {
        std::cerr << "Failed to write output file " << output_bin << '\n';
        return 1;
    }

    std::cout << "extract: read " << input_nnue
              << " (desc: \"" << evalFile.netDescription << "\")\n"
              << "extract: wrote " << blob.size() << " bytes to " << output_bin << '\n';
    return 0;
}

// ---------------------------------------------------------------------------
// inject: plain binary blob -> .nnue (zstd)
// ---------------------------------------------------------------------------
int do_inject(const std::string& input_bin, const std::string& output_nnue) {
    std::string blob;
    if (!read_file(input_bin, blob))
    {
        std::cerr << "Failed to read input file " << input_bin << '\n';
        return 1;
    }

    std::istringstream iss(blob, std::ios::binary);
    auto               network = std::make_unique<Network>();
    std::string        description;
    if (!network->tool_read_parameters(iss, description))
    {
        std::cerr << "Failed to parse network parameters from " << input_bin << '\n';
        return 1;
    }

    // Re-serialise to the canonical (uncompressed) representation, write
    // to a temporary file, then zstd-compress so the engine can load it
    // directly.
    std::ostringstream oss(std::ios::binary);
    if (!network->tool_write_parameters(oss, description))
    {
        std::cerr << "Failed to re-serialise network parameters.\n";
        return 1;
    }
    const std::string raw = oss.str();

    const std::string tmp_path = output_nnue + ".raw.tmp";
    if (!write_file(tmp_path, raw))
    {
        std::cerr << "Failed to write temporary file " << tmp_path << '\n';
        return 1;
    }

    if (!zstd_compress_file(tmp_path, output_nnue, 19))
    {
        std::cerr << "zstd compression failed.\n";
        std::remove(tmp_path.c_str());
        return 1;
    }
    std::remove(tmp_path.c_str());

    std::cout << "inject: read " << blob.size() << " bytes from " << input_bin
              << " (desc: \"" << description << "\")\n"
              << "inject: wrote zstd-compressed " << output_nnue
              << " (raw " << raw.size() << " bytes)\n";
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4)
        usage(2);

    const std::string cmd = argv[1];
    if (cmd == "extract")
        return do_extract(argv[2], argv[3]);
    if (cmd == "inject")
        return do_inject(argv[2], argv[3]);

    usage(2);
}