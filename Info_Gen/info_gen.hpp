#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <bitset>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <cstring>
using json = nlohmann::json;

#define HW_INFO_MAX 1024   //Hardware store number of Tile Info
#define MAX_TILE_SIZE 34
#define BUS_WIDTH 16        //Byte
#define DATA_WIDTH 2        //Data 16-bit
#define DATA_DEPTH (BUS_WIDTH/DATA_WIDTH)
#define MAX_ICH (BUS_WIDTH/DATA_WIDTH)
#define MAX_OCH (BUS_WIDTH/DATA_WIDTH)

// enum HW_Pad_Side : size_t {Left, Bottom, Right, Top};

#define LEFT 0
#define BOTTOM 1
#define RIGHT 2
#define TOP 3

class Layer_Info{
    public:
        std::string node_name;
        std::string weight;
        size_t IF_Size = 0;
        size_t OF_Size = 0;
        size_t Kernel_Size = 0;
        size_t Stride = 0;
        size_t ICH = 0;
        size_t OCH = 0;
        size_t In_Pad_Size = 0;
        size_t Out_Pad_Size = 0;
        float Leaky_ReLU_alpha = 0;
        size_t Pool_Size = 0;
        size_t Pool_Stride = 0;
        size_t Input_Tile_Size = 0;
        size_t Output_Tile_Size = 0;
        size_t Input_Tile_Number = 0;
        size_t Output_Tile_Number = 0;
        size_t Next_Tile_Size = 0;
        std::vector<size_t> Input_Address;
        std::vector<size_t> Previous_node_OCH;
        size_t Output_Address = 0;
        size_t Weight_Address = 0;
        bool Have_ReLU = false;
        bool Have_MaxPool = false;
        bool Is_LeakyReLU = false;
        bool Have_BatchNormalization = false;
        bool Batch_First = false;
        bool Have_Upsample = false;
        bool Have_Bias = false;
        bool Is_Output_Layer = false;
        size_t quant_batch_bias = 0;
        int quant_finish = 0;
        size_t quant_batch = 0;
        size_t quant_word_size = 0;
        size_t quant_obuf = 0;
        size_t Tile_Info_Number = 0;
        size_t Tile_Info_Addr = 0;
        size_t Bit_Serial = 0;
        size_t Leaky_ReLU_alpha_FP = 0; //Fixed point
        std::vector<size_t> out_addr_set;
        std::vector<u_char> t_type; // Tile Padding Type and Valid
        std::vector<uint32_t> tile_info;

        Layer_Info() = default;
        std::vector<uint32_t> Gen_Layer_Info() const;
        void gen_out_addr();
        void gen_t_type_m();
        void gen_tile_info();
        void dump_for_sim(const size_t Data_Offset, const size_t Wegiht_Offset) const;
        // void const dump_bin(const std::string &filename) const;
};

std::vector<Layer_Info> parse_json(const std::string filename);
