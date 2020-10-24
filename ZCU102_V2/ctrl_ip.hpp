#include "general.hpp"

#include <sys/ioctl.h>

#define START 0
#define LAYER_INFO_0 1
#define LAYER_INFO_1 3
#define LAYER_INFO_2 4
#define LAYER_INFO_3 5
#define LAYER_INFO_4 6
#define R_ERROR_ADDR 7
#define W_ERROR_ADDR 8
#define R_ERROR_TYPE 9
#define W_ERROR_TYPE 10
#define MEM_ALLOC    11
#define MEM_FREE     12

struct layer_info
{
    uint8_t data[20];
    uint32_t get_tile_begin_addr() const;
    void run_inference(const int fd);
    void load_layer_info(const void* src);
    void set_tile_begin_addr(uint32_t addr);
    void init_addr(uint32_t base_addr);
    void get_layer_info();
};
std::vector<layer_info> parse_file(const std::string filename);
