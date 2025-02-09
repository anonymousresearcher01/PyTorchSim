#ifndef DRAM_H
#define DRAM_H
#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <utility>

#include "Common.h"
#include "TMA.h"
#include "ramulator2.hh"
#include "Hashing.h"
#include "Cache.h"

class Dram {
 public:
  virtual ~Dram() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual bool is_full(uint32_t cid, mem_fetch* request) = 0;
  virtual void push(uint32_t cid, mem_fetch* request) = 0;
  virtual bool is_empty(uint32_t cid) = 0;
  virtual mem_fetch* top(uint32_t cid) = 0;
  virtual void pop(uint32_t cid) = 0;
  uint32_t get_channel_id(mem_fetch* request);
  virtual void print_stat() {}

 protected:
  SimulationConfig _config;
  uint32_t _n_ch;
  uint32_t _n_partitions;
  uint32_t _n_ch_per_partition;
  cycle_type _cycles;
};

class DramRamulator2 : public Dram {
 public:
  DramRamulator2(SimulationConfig config);

  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, mem_fetch* request) override;
  virtual void push(uint32_t cid, mem_fetch* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual mem_fetch* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::vector<std::unique_ptr<Cache>> _m_caches;
  std::vector<std::unique_ptr<Ramulator2>> _mem;
  int _tx_ch_log2;
  int _tx_log2;
  int _req_size;
};

#endif