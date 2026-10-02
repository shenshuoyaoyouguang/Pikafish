/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <deque>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "shm.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

int MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

// The default configuration will attempt to group L3 domains up to 32 threads.
// This size was found to be a good balance between the Elo gain of increased
// history sharing and the speed loss from more cross-cache accesses (see
// PR#6526). The user can always explicitly override this behavior.
constexpr NumaAutoPolicy DefaultNumaPolicy = BundledL3Policy{32};

Engine::Engine(std::optional<std::filesystem::path> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : std::filesystem::path{}),
    numaContext(NumaConfig::from_system(DefaultNumaPolicy)),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networkFile{std::nullopt, ""},
    network(numaContext, get_default_network()) {

    pos.set(StartFEN, &states->back());

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          if (!set_numa_config_from_option(o))
              return "NumaPolicy: invalid value '" + std::string(o) + "', keeping previous config.";
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add("Move Overhead", Option(10, 0, 5000));

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_ShowWDL", Option(false));

    // LMR 连续化调优选项
    // 默认启用连续化路径——默认 θ₀ 与 legacy 逐位等价（1,484,403 节点），
    // 启用后便于后续 SPSA 调优 θ 偏离 θ₀ 时直接生效，无需额外 UCI 命令。
    options.add("LMR_Continuous", Option(true));

    // LMR 采样开关：开启后在 legacy 路径 Step 18 末尾向 stderr 输出 (φ(x), r) 样本
    options.add("LMR_Sample", Option(false));

    // θ₀ 默认值（Q16 定点逗号分隔串），与 init_lmr_theta() 一致
    // 26 维：合并原 4+5 → 新 4 (-1434*Q=-93978624)，新增交互项 5/21/23/25 初始为 0，
    //        特征 24 为 moveCount 线性项 θ₀=-64*Q=-4194304
    constexpr const char* LmrThetaDefault =
      "65536,73924608,21120,182779904,-93978624,0,-62849024,"
      "-73007104,-74448896,-60293120,65536,211419136,67895296,101777408,"
      "16973824,66781184,66453504,177668096,7648,196608,64225280,0,16646144,0,-4194304,0";

    options.add(  //
      "LMR_Theta", Option(LmrThetaDefault, [this](const Option& o) -> std::optional<std::string> {
          // 解析逗号分隔的 Q16 定点参数串（不使用异常，-fno-exceptions）
          std::string      s = std::string(o);
          std::stringstream ss(s);
          std::string       token;
          std::array<int, Search::Worker::LMR_THETA_SIZE> newTheta{};
          int  count = 0;
          bool ok    = true;
          while (std::getline(ss, token, ','))
          {
              if (count >= Search::Worker::LMR_THETA_SIZE)
              {
                  ok = false;
                  break;
              }
              // 用 strtol 解析，检测非法字符
              char* end = nullptr;
              long  val = std::strtol(token.c_str(), &end, 10);
              if (end == token.c_str() || *end != '\0')
              {
                  ok = false;
                  break;
              }
              newTheta[count++] = int(val);
          }
          if (!ok || count != Search::Worker::LMR_THETA_SIZE)
              return std::optional<std::string>(
                "LMR_Theta: expected "
                + std::to_string(Search::Worker::LMR_THETA_SIZE)
                + " comma-separated integers");
          // 更新所有 worker 的 lmrTheta
          for (auto&& t : threads)
              t->worker->lmrTheta = newTheta;
          return std::nullopt;
      }));

    // 联合搜索优化选项：Joint_Continuous 开关
    // jointContinuous=false 时使用 legacy 硬编码常数，true 时使用 jointTheta 参数化
    options.add("Joint_Continuous", Option(false));

    // Joint_Theta 默认值 = [lmrTheta默认值(26维), nm_legacy(6维), fut_legacy(5维), se_legacy(4维)]
    // 前 26 维与 LmrThetaDefault 一致（Q16 定点），后 15 维为 legacy 整数值
    constexpr const char* JointThetaDefault =
      "65536,73924608,21120,182779904,-93978624,0,-62849024,"
      "-73007104,-74448896,-60293120,65536,211419136,67895296,101777408,"
      "16973824,66781184,66453504,177668096,7648,196608,64225280,0,16646144,0,-4194304,0,"
      "8,51,8,282,3,188,"
      "41,33,2500,333,133448,"
      "45,72,69,176";

    options.add(  //
      "Joint_Theta", Option(JointThetaDefault, [this](const Option& o) -> std::optional<std::string> {
          // 解析逗号分隔的参数串（前 26 维 Q16 定点，后 15 维整数）
          std::string      s = std::string(o);
          std::stringstream ss(s);
          std::string       token;
          std::array<int, Search::Worker::JOINT_THETA_SIZE> newTheta{};
          int  count = 0;
          bool ok    = true;
          while (std::getline(ss, token, ','))
          {
              if (count >= Search::Worker::JOINT_THETA_SIZE)
              {
                  ok = false;
                  break;
              }
              // 用 strtol 解析，检测非法字符
              char* end = nullptr;
              long  val = std::strtol(token.c_str(), &end, 10);
              if (end == token.c_str() || *end != '\0')
              {
                  ok = false;
                  break;
              }
              newTheta[count++] = int(val);
          }
          if (!ok || count != Search::Worker::JOINT_THETA_SIZE)
              return std::optional<std::string>(
                "Joint_Theta: expected "
                + std::to_string(Search::Worker::JOINT_THETA_SIZE)
                + " comma-separated integers");
          // 更新所有 worker 的 jointTheta
          for (auto&& t : threads)
              t->worker->jointTheta = newTheta;
          return std::nullopt;
      }));

    options.add(  //
      "EvalFile", Option(EvalFileDefaultName, [this](const Option& o) {
          load_network(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();
}

std::variant<u64, PositionSetError> Engine::perft(const std::string& fen, Depth depth) {
    verify_network();

    return Benchmark::perft(fen, depth);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_network();

    threads.start_thinking(pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_start(std::function<void()>&& f) { updateContext.onStart = std::move(f); }

void Engine::set_on_verify_network(std::function<void(std::string_view)>&& f) {
    onVerifyNetwork = std::move(f);
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

std::optional<PositionSetError> Engine::set_position(const std::string&              fen,
                                                     const std::vector<std::string>& moves) {
    // Drop the old state and create a new one
    states   = StateListPtr(new std::deque<StateInfo>(1));
    auto err = pos.set(fen, &states->back());
    if (err.has_value())
        return err;

    for (const auto& move : moves)
    {
        auto m = UCIEngine::to_move(pos, move);

        if (m == Move::none())
            return PositionSetError("Illegal move: " + move);

        states->emplace_back();
        pos.do_move(m, states->back());
    }

    return std::nullopt;
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy));
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy, false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        auto parsed = NumaConfig::from_string(o);
        if (!parsed.has_value())
            return false;
        numaContext.set_numa_config(std::move(*parsed));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
    return true;
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists, network},
                updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(usize mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

std::array<int, Search::Worker::LMR_THETA_SIZE> Engine::get_lmr_theta() const {
    return threads.main_thread()->worker->lmrTheta;
}

// network related

void Engine::verify_network() const {
    const auto file = path_from_utf8(std::string(options["EvalFile"]));
    network->verify(onVerifyNetwork, networkFile, file);

    auto statuses = network.get_status_and_errors();
    for (usize i = 0; i < statuses.size(); ++i)
    {
        const auto [status, error] = statuses[i];
        std::string message        = "Network replica " + std::to_string(i + 1) + ": ";
        if (status == SystemWideSharedConstantAllocationStatus::NoAllocation)
        {
            message += "No allocation.";
        }
        else if (status == SystemWideSharedConstantAllocationStatus::LocalMemory)
        {
            message += "Local memory.";
        }
        else if (status == SystemWideSharedConstantAllocationStatus::SharedMemory)
        {
            message += "Shared memory.";
        }
        else
        {
            message += "Unknown status.";
        }

        if (error.has_value())
        {
            message += " " + *error;
        }

        onVerifyNetwork(message);
    }
}

std::unique_ptr<Eval::NNUE::Network> Engine::get_default_network() {

    auto network_ = std::make_unique<NN::Network>();

    network_->load(binaryDirectory, std::filesystem::path{}, networkFile);

    return network_;
}

void Engine::load_network(const std::filesystem::path& file) {
    network.modify_and_replicate(
      [this, &file](NN::Network& network_) { network_.load(binaryDirectory, file, networkFile); });
    threads.clear();
    threads.ensure_network_replicated();
}

void Engine::save_network(const std::optional<std::filesystem::path>& file) {
    network.modify_and_replicate(
      [&file, this](NN::Network& network_) { network_.save(networkFile, file); });
}

// utility functions

void Engine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), &trace_states->back());

    verify_network();

    sync_cout << "\n" << Eval::trace(p, *network) << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

std::optional<PositionSetError> Engine::flip() { return pos.flip(); }

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<usize, usize>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                 counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                    cfg    = numaContext.get_numa_config();
    std::vector<std::pair<usize, usize>> ratios;
    NumaIndex                            n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    usize threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
