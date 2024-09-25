#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "Tile.h"
#include "SimulationConfig.h"
#include "TMA.h"

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core() = default;
  bool running();
  bool can_issue(const std::shared_ptr<Tile>& op);
  void issue(std::shared_ptr<Tile> tile);
  std::shared_ptr<Tile> pop_finished_tile();
  void cycle();
  void compute_cycle();
  void dma_cycle();
  bool has_memory_request();
  void pop_memory_request();
  MemoryAccess* top_memory_request() { return _request_queue.front(); }
  void push_memory_response(MemoryAccess* response);
  void print_stats();
  void print_current_stats();
  void finish_instruction(std::shared_ptr<Instruction>& inst);
  cycle_type get_compute_cycles() { return _stat_tot_compute_cycle[SYSTOLIC_ARRAY]; }
  enum {
    VECTOR_UNIT,
    SYSTOLIC_ARRAY,
    NR_COMPUTE_UNIT
  };

 protected:
  bool can_issue_compute(std::shared_ptr<Instruction>& inst);
  void update_stats();

  /* Core id & config file */
  const uint32_t _id;   
  const SimulationConfig _config;
  size_t _sram_size;
  size_t _used_sram_size;

  /* TMA Unit */
  TMA _tma;

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _stat_tot_compute_cycle[NR_COMPUTE_UNIT] = {0, };
  cycle_type _stat_tot_tma_cycle = 0;
  cycle_type _stat_tot_tma_idle_cycle = 0;
  cycle_type _stat_tot_compute_idle_cycle[NR_COMPUTE_UNIT] = {0, };

  cycle_type _stat_compute_cycle[NR_COMPUTE_UNIT] = {0, };
  cycle_type _stat_tma_cycle = 0;
  cycle_type _stat_tma_idle_cycle = 0;
  cycle_type _stat_compute_idle_cycle[NR_COMPUTE_UNIT] = {0, };

  std::vector<std::shared_ptr<Tile>> _tiles;
  std::queue<std::shared_ptr<Tile>> _finished_tiles;

  std::vector<std::queue<std::shared_ptr<Instruction>>> _compute_pipeline;
  std::queue<std::shared_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::shared_ptr<Instruction>> _st_inst_queue;

  std::vector<std::shared_ptr<Instruction>> _dma_waiting_queue;
  /* Interconnect queue */
  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;
};