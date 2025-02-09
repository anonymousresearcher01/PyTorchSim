#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "Cache_defs.h"

typedef unsigned long long new_addr_type;

enum mem_access_type {
  GLOBAL_ACC_R,
  GLOBAL_ACC_W,
  L2_CACHE_WA, /* Data L2 cache write alloc */
  L2_CACHE_WB, /* Data L2 cache write back */
  NUM_MEM_ACCESS_TYPE
};

static const char* mem_access_type_str[] = {
    "GLOBAL_ACC_R", "GLOBAL_ACC_W",  
    "L2_CACHE_WA", "L2_CACHE_WB"};
enum mf_type { READ_REQUEST = 0, WRITE_REQUEST, READ_REPLY, WRITE_ACK };

class mem_fetch {
 public:
  mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
            unsigned data_size, unsigned request_id, unsigned numa_id=-1,
            void* custom_data=NULL) :
            m_addr(addr), m_mem_access_type(acc_type),
            m_type(type), m_data_size(data_size),
            m_request_id(request_id), m_numa_id(numa_id),
            m_custom_data(custom_data) {}
  mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
            unsigned data_size) : m_addr(addr), m_mem_access_type(acc_type),
            m_type(type), m_data_size(data_size) {}
  mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
            unsigned data_size, SectorMask sector_mask) :
            m_addr(addr), m_mem_access_type(acc_type), m_type(type), m_data_size(data_size),
            m_sector_mask(sector_mask) {}
  mem_fetch(std::deque<mem_fetch*> mfs);  // for wrapping multiple mfs into one
  /* Src & Des */
  void set_core_id(int core_id) {m_core_id = core_id;}
  int get_core_id() { return m_core_id; }
  void set_channel(unsigned channel) { m_channel = channel; }
  unsigned get_channel() { return m_channel; }
  void set_numa_id(unsigned numa_id) { m_numa_id=numa_id; }
  unsigned get_numa_id() { return m_numa_id; }
  /* Data & size */
  void set_data(void* data) { m_data = data; }
  void* get_data() { return m_data; }
  void set_data_size(unsigned size) { m_data_size = size; }
  unsigned get_data_size() { return m_data_size; }
  new_addr_type get_addr() { return m_addr; }
  void set_addr(new_addr_type addr) { m_addr = addr; }
  /* Mem info */
  mem_access_type get_access_type() { return m_mem_access_type; }
  mf_type get_type() { return m_type; }
  void set_type(mf_type type) { m_type = type; }
  bool is_write() { return m_type == mf_type::WRITE_REQUEST || m_type == mf_type::WRITE_ACK; }
  void set_request_id(unsigned id) { m_request_id = id; }
  unsigned get_request_id() { return m_request_id; }
  SectorMask get_access_sector_mask() { return m_sector_mask; }
  void set_dirty_mask(SectorMask dirty_mask) { m_dirty_mask = dirty_mask; }
  SectorMask get_dirty_mask() { return m_dirty_mask; }
  mem_fetch* get_original_mf() { return m_original_mf; }
  bool is_atomic() { return false; }
  void set_custom_data(void* custom_data) { m_custom_data = custom_data; }
  void* get_custom_data() { return m_custom_data; }
  /* Stat */
  void set_start_cycle(uint64_t start_cycle) { m_start_cycle = start_cycle; }
  uint64_t get_start_cycle() { return m_start_cycle; } 

  std::string current_state = "NONE";
  uint64_t request_cycle;
  uint64_t response_cycle;
 private:
  uint64_t m_request_id;
  unsigned m_data_size;
  new_addr_type m_addr;
  void* m_data = NULL;
  mem_access_type m_mem_access_type;
  mf_type m_type;
  unsigned m_core_id;
  unsigned m_channel;
  unsigned m_numa_id;
  SectorMask m_sector_mask;
  SectorMask m_dirty_mask;
  mem_fetch* m_original_mf;
  void* m_custom_data = NULL;
  uint64_t m_start_cycle = 0ULL;
};

#endif