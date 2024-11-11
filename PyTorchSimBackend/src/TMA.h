#ifndef TMA_H
#define TMA_H

#include <cstdint>
#include <memory>
#include <map>
#include <vector>
#include "Instruction.h"
#include "SimulationConfig.h"
#include "Tile.h"

typedef struct {
  uint32_t id;
  addr_type dram_address;
  uint64_t size;
  bool write;
  bool request;
  uint32_t core_id;
  Instruction* owner_instruction;
  cycle_type start_cycle;
  cycle_type dram_enter_cycle;
  cycle_type dram_finish_cycle;
} MemoryAccess;

struct VectorCompare {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
        return a < b;
    }
};

class TMA {
 public:
  TMA(uint32_t id, uint32_t dram_req_size);

  void issue_tile(std::shared_ptr<Instruction> inst);
  bool is_finished() { return _finished; }
  bool empty() { return _current_inst==nullptr; }
  void register_tag(void* subgraph, std::vector<int> key) {
    if (!subgraph) {
      throw std::invalid_argument("subgraph cannot be null");
    }
    if (tag_table.find(subgraph) == tag_table.end()) {
      tag_table[subgraph] = std::map<std::vector<int>, bool, VectorCompare>();
    }
    tag_table[subgraph][key] = false;
  }
  void set_tag_finish(void* subgraph, std::vector<int> key) {
    if (!subgraph) {
      throw std::invalid_argument("subgraph cannot be null");
    }
    if (tag_table.find(subgraph) == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table[subgraph][key] = true;
  }
  bool get_tag_finish(void* subgraph, std::vector<int> key) {
    if (!subgraph) {
      throw std::invalid_argument("subgraph cannot be null");
    }
    auto subgraph_it = tag_table.find(subgraph);
    if (subgraph_it == tag_table.end()) {
      return false;
    }
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    return tag_table[subgraph][key];
  }
  void erase_tag_table(void* subgraph) {
    if (!subgraph) {
      throw std::invalid_argument("subgraph cannot be null");
    }
    auto subgraph_it = tag_table.find(subgraph);
    if (subgraph_it == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table.erase(subgraph);
  }

  std::shared_ptr<Instruction>& get_current_inst() { return _current_inst; }
  MemoryAccess* get_memory_access();
  uint32_t generate_mem_access_id();

 protected:
  uint32_t _id;
  std::shared_ptr<Instruction> _current_inst;
  uint32_t _dram_req_size;
  uint32_t _tile_size_x=0;
  uint32_t _tile_size_y=0;
  size_t _tile_idx_stride=1;
  uint32_t _tile_idx;
  bool _finished=true;
  std::map<void*, std::map<std::vector<int>, bool, VectorCompare>> tag_table;
};
#endif