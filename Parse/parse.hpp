#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <filesystem>

using json = nlohmann::json;

#define MAX_TILE_SIZE 34
#define BUS_WIDTH 16        //AXI BUS 16-Byte
#define DATA_WIDTH 2        //Data 16-bit
#define DATA_DEPTH (BUS_WIDTH/DATA_WIDTH)
#define MAX_ICH (BUS_WIDTH/DATA_WIDTH)
#define MAX_OCH (BUS_WIDTH/DATA_WIDTH)

enum class OP_Attribute_Type : size_t {None, Conv_Pool, Conv, MaxPool, ReLU, BatchNormalization, Concat, Upsample, InputLayer};
enum class Pad_Attribute_Type : size_t {Other, SAME, VALID};

struct model_op_node;
using tree_map = std::map<std::string, std::shared_ptr<model_op_node>>;

struct model_op_node
{
    std::vector<std::shared_ptr<model_op_node>> prev_node;
    std::vector<std::string> prev_node_name;
    std::string node_name;
    std::string weight;
    size_t IF_Size = 0;
    size_t OF_Size = 0;
    size_t Kernel_Size = 0;
    size_t Stride = 0;
    size_t ICH = 0;
    size_t OCH = 0;
    size_t In_Pad_Size = 0;
    int Out_Pad_Size = -1;
    float Leaky_ReLU_alpha = 0;
    size_t Pool_Size = 0;
    size_t Pool_Stride = 0;
    size_t Input_Tile_Size = 0;
    size_t Output_Tile_Size = 0;
    size_t Input_Tile_Number = 0;
    size_t Output_Tile_Number = 0;
    size_t Next_Tile_Size = 0;
    std::vector<size_t> Input_Address;
    size_t Output_Address = 0;
    size_t Weight_Address = 0;
    bool Have_ReLU = false;
    bool Have_MaxPool = false;
    bool Is_LeakyReLU = false;
    bool Have_BatchNormalization = false;
    bool Batch_First = false;
    bool Have_Upsample = false;
    // bool Has_Stored_DRAM = false;
    bool Is_Output_Layer = false;
    bool Have_Bias = true;
    OP_Attribute_Type OP_Attribute = OP_Attribute_Type::None;
    Pad_Attribute_Type Pad_Attribute = Pad_Attribute_Type::Other;
    void print_info() const;
};

std::tuple<json, json, json> read_json(const std::string filename);
tree_map create_table(const json &layers);
std::shared_ptr<model_op_node> create_tree(tree_map &table, const std::string &node_name);
bool compare_node(const std::shared_ptr<model_op_node> &node_a, const std::shared_ptr<model_op_node> &node_b);
std::shared_ptr<model_op_node> merge_dag (tree_map &table, tree_map &new_table, const std::string &node, std::shared_ptr<model_op_node> &compare_p);
bool gen_node_out_pad(tree_map &table, const std::string &node, const int out_pad_size);
size_t gen_layer_order(const std::shared_ptr<model_op_node> &p, std::map<std::string, size_t> &table, const size_t order);
size_t weight_offset(const std::shared_ptr<model_op_node> &p, std::map<std::string, size_t> &weight_table, const size_t addr_offset);
void gen_out_addr(tree_map &table, std::map<std::string, size_t> &order_table);
void gen_in_addr(std::shared_ptr<model_op_node> &p);
