#include "Dram.h"

uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_id;
  if (_n_ch_per_partition >= 16)
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_config.dram_req_size, 0, _n_ch_per_partition);
  else
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_config.dram_req_size, 0, 16) % _n_ch_per_partition;
  
  channel_id += ((access->get_numa_id() % _n_partitions)* _n_ch_per_partition);
  return channel_id;
}

DramRamulator2::DramRamulator2(SimulationConfig config) {
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _n_partitions = config.dram_num_partitions;
  _n_ch_per_partition = _n_ch / _n_partitions;
  _config = config;
  _m_caches.resize(_n_ch);
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    //_m_caches = std::make_unique<ReadOnlyCache>("L2 RO cache");
    _mem[ch] = std::make_unique<Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, 1);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

bool DramRamulator2::running() {
  return false;
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();
  }
}

bool DramRamulator2::is_full(uint32_t cid, mem_fetch* request) {
  return _mem[cid]->full();
}

void DramRamulator2::push(uint32_t cid, mem_fetch* request) {
  addr_type atomic_bytes =_config.dram_req_size;
  addr_type target_addr = request->get_addr();
  // align address
  addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->get_data_size() == atomic_bytes);
  _mem[cid]->push(request);
}

bool DramRamulator2::is_empty(uint32_t cid) {
  return _mem[cid]->return_queue_top() == NULL;
}

mem_fetch* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  mem_fetch* mf = _mem[cid]->return_queue_top();
  return mf;
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  mem_fetch* mf = _mem[cid]->return_queue_pop();
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}
