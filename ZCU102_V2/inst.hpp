#include "general.hpp"

struct inst
{
    uint8_t data[32];
    void load_data(const void* data);
    void set_out_addr(uint32_t addr);
    void set_weight_addr(uint32_t addr);
    void set_in_addr(uint32_t addr);
    uint32_t get_out_addr();
    uint32_t get_weight_addr();
    uint32_t get_in_addr();
};

std::vector<std::vector<inst>> load_inst_file_set(const std::string filename);
// void init_inst_addr(std::vector<std::vector<inst>> &inst_data_set,const uint32_t base_addr, const uint32_t weight_addr);
void init_info_addr(std::vector<inst> &inst_data, const uint32_t base_addr, const uint32_t weight_addr);
void load_weight(const std::string filename, void* dst);
std::vector<inst> load_inst_data(const std::string filename);
