#include "info_gen.hpp"
#include <filesystem>
#include <cassert>

#define ONCE_CAL_WEIGHT_NUM (DATA_DEPTH*DATA_DEPTH*9 + DATA_DEPTH*2)

// #define BATCH_MERGE_WEIGHT
#define DUMP_WEIGHT
#define DUMP_LAYER_INFO
#define DUMP_TILE_INFO
// #define DUMP_FOR_SIM

#ifdef DUMP_FOR_SIM
    #define DATA_BASE_ADDR   0x02001000
    #define WEIGHT_BASE_ADDR 0x01002000
    #define LAYER_ADDR_OFFSET 0x10000
#else
    #define DATA_BASE_ADDR   0
    #define WEIGHT_BASE_ADDR 0
    #define LAYER_ADDR_OFFSET 0
#endif

template<typename T>
size_t size_in_byte(const std::vector<T> &a){
    return  a.size()*sizeof(T);
}

enum File_Type : size_t {Weight, Bias, Scale, Var, Mean};

void dump_tile_bin(const std::string &filename, const std::vector<Layer_Info> &info_set){
    
    size_t total_tile_size = 0;
    std::vector<uint32_t> total_tile_tmp;

    for(auto &i : info_set){
        total_tile_size += size_in_byte(i.tile_info);
    }
    std::cout << total_tile_size << std::endl;
    total_tile_tmp.reserve(total_tile_size/(sizeof(uint32_t)));

    for(auto &i : info_set){
        total_tile_tmp.insert(total_tile_tmp.end(), i.tile_info.begin(), i.tile_info.end());
    }
    
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, total_tile_tmp.data(), total_tile_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}

void dump_layer_info_sim(const std::string &filename, const std::vector<uint32_t> &layer_info){
    const size_t LAYER_INFO_DATA_SIZE = 5; //160-bit
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr << fmt::format("File : {} can not open.\n",filename);
        throw std::invalid_argument("Can not open file");
    }
    if(layer_info.size() % LAYER_INFO_DATA_SIZE != 0){
        throw std::invalid_argument("Layer Info format error");
    }
    for(size_t i = 0; i < layer_info.size()/LAYER_INFO_DATA_SIZE; i++){
        for(int j = LAYER_INFO_DATA_SIZE-1; j >= 0; j--){
            out << fmt::format("{:08x}",layer_info[j+i*LAYER_INFO_DATA_SIZE]);
        }
        out << "\n";
    }
    out.close();
}

std::vector<uint32_t> gen_layer_info_data(std::vector<Layer_Info> &info_set){
    std::vector<uint32_t> layer_info_data;
    const size_t LAYER_INFO_DATA_SIZE = 5; //160-bit
    size_t tmp_addr = LAYER_ADDR_OFFSET;
    size_t tmp_layer_count = 0;
    bool first_tile_addr_op = false;
    
    for(auto &i : info_set){
        tmp_layer_count += std::ceil(i.Tile_Info_Number / (double)HW_INFO_MAX);
    }

    layer_info_data.reserve(info_set.size()*tmp_layer_count*LAYER_INFO_DATA_SIZE);
    
    for(auto&i : info_set){
        const size_t tile_num = i.Tile_Info_Number* (32/BUS_WIDTH);
        size_t tile_num_tmp = 0;
        i.Tile_Info_Addr += tmp_addr;
        for(size_t j = 0; j < tile_num; j+=tile_num_tmp){
            tile_num_tmp = tile_num-j;
            if(tile_num_tmp > HW_INFO_MAX*(32/BUS_WIDTH)){
                tile_num_tmp = HW_INFO_MAX*(32/BUS_WIDTH);
                tile_num_tmp = std::floor(tile_num_tmp/std::ceil(i.ICH/(double)DATA_DEPTH))*std::ceil(i.ICH/(double)DATA_DEPTH);
            }
            if(!first_tile_addr_op){
                first_tile_addr_op = true;
            }else{
                i.Tile_Info_Addr = tmp_addr;
            }
            i.Tile_Info_Number = tile_num_tmp - 1;
            const std::vector<uint32_t> layer_info = i.Gen_Layer_Info();
            layer_info_data.insert(layer_info_data.end(),layer_info.begin(), layer_info.end());
            tmp_addr += tile_num_tmp * BUS_WIDTH;
        }
    }
    return layer_info_data;
}

void dump_layer_info_data_bin(const std::string &filename, const std::vector<uint32_t> &layer_info_data_set){
    const size_t data_size = size_in_byte(layer_info_data_set);
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, layer_info_data_set.data(), data_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}

void dump_data_bin(const std::string &filename, const void *src, const size_t src_size){
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, src, src_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}

inline std::string weight_type(const size_t num){
    switch (num)
    {
    case 0:
        return "Kenerl";
    case 1:
        return "Bias";
    case 2:
        return "Scale";
    case 3:
        return "Variance";
    case 4:
        return "Mean";
    default:
        return "None";
    }
}

std::vector<float> get_weight_from_file(std::ifstream &in, const size_t need_num){
    std::vector<float> data;
    data.reserve(need_num);
    // std::string tmp;
    float tmp;
    for(size_t i = 0; i < need_num; i++){
        in >> tmp;
        if(in.eof()){
            throw std::invalid_argument("Weight need number greater then weight file");
        }
        data.push_back(tmp);
    }
    in >> tmp;
    if(!in.eof()){
        throw std::invalid_argument("Weight need number less then weight file");
    }
    return data;
}

std::vector<short> gen_layer_weight(const Layer_Info &info, const std::vector<std::string> &path_set){
    std::vector<std::vector<float>> data_in;
    const size_t OCH_NUM = std::ceil(info.OCH/(double)DATA_DEPTH);
    const size_t ICH_NUM = std::ceil(info.ICH/(double)DATA_DEPTH);
    const size_t weight_num = OCH_NUM*ICH_NUM*ONCE_CAL_WEIGHT_NUM;
    size_t kernel_count = 0;
    //TODO:Fix Kernel Size Only 9
    const size_t HW_KERNEL_SIZE = 9;
    std::vector<std::ifstream> file_set;
    const size_t need_file_num = info.Have_BatchNormalization ? 5 : (info.Have_Bias ? 2 : 1);
    std::vector<short> weight_data(weight_num, 0);
    // std::cout << fmt::format("{:04x}\n", (uint16_t)weight_data[0]);
    // weight_data.resize(weight_num);
    data_in.reserve(need_file_num);
    file_set.resize(need_file_num);

    if(path_set.size() != need_file_num){
        throw std::invalid_argument("path_set num not equal need_file_num");
    }
    for(size_t i = 0; i < need_file_num; i++){
        file_set[i].open(path_set[i]);
        if(!file_set[i].is_open()){
            std::cout << fmt::format("Can not open {} {} file\n", info.weight, weight_type(i));
            throw std::invalid_argument("Can not open weight file");
        }
    }

    for(size_t i = 0; i < need_file_num; i++){
        size_t tmp;
        if(i == 0)
            tmp = info.OCH * info.ICH * info.Kernel_Size * info.Kernel_Size;
        else
            tmp = info.OCH;
        data_in.push_back(get_weight_from_file(file_set[i], tmp));
        file_set[i].close();
    }
    

    for(size_t och_count = 0; och_count < OCH_NUM; och_count++){
        size_t index;
        for(size_t och = 0; och < DATA_DEPTH; och++){
            for(size_t ich = 0; ich < ICH_NUM; ich++){
                for(size_t ich_count = 0; ich_count < DATA_DEPTH; ich_count++){
                    for(size_t k = 0; k < HW_KERNEL_SIZE; k++){
                        short t = 0;
                        if(info.Kernel_Size == 1){
                            if(och_count*DATA_DEPTH+och < info.OCH){
                                if((ich*DATA_DEPTH+ich_count) < info.ICH){
                                    // t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                    #ifdef BATCH_MERGE_WEIGHT
                                        if(info.Have_BatchNormalization){
                                            const size_t och_index = och_count*DATA_DEPTH+och;
                                            float weight_tmp = data_in[Weight][kernel_count]*(data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index]));
                                            t = weight_tmp * std::pow(2,info.quant_batch);
                                            if(weight_tmp * (short)t < 0){
                                                std::cout << fmt::format("och_index : {}\n", och_index) << std::endl;
                                                std::cout << fmt::format("Weight : {}, Scale : {}, After : {}, Fix : {}\n",data_in[Weight][kernel_count], 
                                                (data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index])), weight_tmp, (short)t);
                                            }
                                            assert(weight_tmp * (short)t >= 0);
                                        }else{
                                            t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                            assert(data_in[Weight][kernel_count] * (short)t >= 0);
                                        }
                                    #else
                                        t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                        assert(data_in[Weight][kernel_count] * (short)t >= 0);
                                    #endif
                                    if(k == HW_KERNEL_SIZE-1){
                                        kernel_count++;
                                    }
                                }
                            }
                        }else{
                            if(och_count*DATA_DEPTH+och < info.OCH){
                                if((ich*DATA_DEPTH+ich_count) < info.ICH){
                                    // t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                    #ifdef BATCH_MERGE_WEIGHT
                                        if(info.Have_BatchNormalization){
                                            const size_t och_index = och_count*DATA_DEPTH+och;
                                            float weight_tmp = data_in[Weight][kernel_count]*(data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index]));
                                            t = weight_tmp * std::pow(2,info.quant_batch);
                                            if(weight_tmp * (short)t < 0){
                                                std::cout << fmt::format("och_index : {}\n", och_index) << std::endl;
                                                std::cout << fmt::format("Weight : {}, Scale : {}, After : {}, Fix : {}\n",data_in[Weight][kernel_count], 
                                                (data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index])), weight_tmp, (short)t);
                                            }
                                            assert(weight_tmp * (short)t >= 0);
                                        }else{
                                            t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                            assert(data_in[Weight][kernel_count] * (short)t >= 0);
                                        }
                                    #else
                                        t = data_in[Weight][kernel_count] * std::pow(2,info.quant_batch);
                                        assert(data_in[Weight][kernel_count] * (short)t >= 0);
                                    #endif
                                    kernel_count++;
                                }
                            }
                        }
                        index =  (och_count * ICH_NUM + ich) * ONCE_CAL_WEIGHT_NUM;
                        index += och * DATA_DEPTH * HW_KERNEL_SIZE;
                        index += k * DATA_DEPTH;
                        index += ich_count;
                        weight_data[index] = t;
                    }
                }
            }
        }
    }
    for(size_t i = 0; i < OCH_NUM; i++){
        const size_t bias_scale_num = (info.OCH > (i+1)*DATA_DEPTH )? DATA_DEPTH : (info.OCH - i*DATA_DEPTH);
        for(size_t j = 0; j < ICH_NUM; j++){
            const size_t index = (i * ICH_NUM + j) * ONCE_CAL_WEIGHT_NUM + DATA_DEPTH  * DATA_DEPTH * HW_KERNEL_SIZE;
            // for(size_t bias = 0; bias < (info.OCH > (i+1)*DATA_DEPTH ? DATA_DEPTH*2 : (info.OCH - i*DATA_DEPTH)*2);bias++){
            for(size_t bias = 0; bias < DATA_DEPTH*2;bias++){
                size_t och_index =  bias%DATA_DEPTH + i*DATA_DEPTH;
                short short_tmp = 0;
                if(bias < DATA_DEPTH){
                    if(bias < bias_scale_num){
                        if(info.Have_BatchNormalization){
                            #ifdef BATCH_MERGE_WEIGHT
                                short_tmp = std::pow(2,info.quant_batch);
                            #else
                                short_tmp = (short)((data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index])) * std::pow(2,info.quant_batch));
                                if(data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index]) * (short)short_tmp < 0){
                                    std::cerr << fmt::format("Scale {}, Var {}, Fix {}, Float {}\n", data_in[Scale][och_index],
                                    std::sqrt(data_in[Var][och_index]), (short)short_tmp, data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index]));
                                }
                                assert(data_in[Scale][och_index] / std::sqrt(data_in[Var][och_index]) * short_tmp >= 0);
                            #endif
                        }else if(info.Have_Bias){
                            short_tmp = std::pow(2,info.quant_batch);
                        }else{
                            short_tmp = 0;
                        }
                    }
                }
                else{
                    if(bias < bias_scale_num+DATA_DEPTH){
                        if(info.Have_BatchNormalization){
                            auto new_bias = data_in[Bias][och_index] - (data_in[Scale][och_index] * data_in[Mean][och_index] / std::sqrt(data_in[Var][och_index]));
                            short_tmp = (short)((new_bias) * std::pow(2,info.quant_batch));
                            assert((data_in[Bias][och_index] - (data_in[Scale][och_index] * data_in[Mean][och_index] / std::sqrt(data_in[Var][och_index]))) * (short)short_tmp >= 0);
                        }else if(info.Have_Bias){
                            short_tmp = (short)(data_in[Bias][och_index] * std::pow(2,info.quant_batch));
                        }else{
                            short_tmp = 0;
                        }
                    }
                }
                weight_data[index+bias] = short_tmp;
            }
        }
    }
    return weight_data;
}

std::vector<std::string> gen_weight_path(const std::string &dir_path, const Layer_Info &info){
    const size_t path_num = info.Have_BatchNormalization ? 5 : (info.Have_Bias ? 2 : 1);
    std::vector<std::string> path_set;
    path_set.reserve(path_num);
    std::string tmp;
    for(size_t i = 0; i < path_num; i++){
        switch (i){
        case Weight:
            tmp = info.weight + "_Weight.txt";
            break;
        case Bias:
            tmp = info.weight + "_Bias.txt";
            break;
        case Scale:
            tmp = info.weight + "_Scale.txt";
            break;
        case Var:
            tmp = info.weight + "_Variance.txt";
            break;
        case Mean:
            tmp = info.weight + "_Mean.txt";
            break;
        }
        path_set.push_back(dir_path+"/"+tmp);
    }
    return path_set;
}


std::vector<short> gen_total_weight(const std::string &dir_path, const std::vector<Layer_Info> &info_set){
    std::vector<short> weight_data;
    size_t total_weight_count = 0;
    std::map<std::string, bool> weight_table;
    for(auto &i : info_set){
        if(!weight_table[i.weight]){
            total_weight_count += std::ceil(i.ICH/(double)(DATA_DEPTH))*std::ceil(i.OCH/(double)(DATA_DEPTH))*ONCE_CAL_WEIGHT_NUM;
            weight_table[i.weight] = true;
        }
    }
    
    weight_data.reserve(total_weight_count);
    for(auto &i : info_set){
        if(!weight_table[i.weight])
            continue;
        weight_table[i.weight] = false;
        std::cout << fmt::format("Processing {} weight",i.node_name) << std::endl;
        const std::vector<std::string> path_set = gen_weight_path(dir_path, i);
        auto tmp = gen_layer_weight(i, path_set);
        // std::cout << fmt::format("{}\n",tmp.size());
        weight_data.insert(weight_data.end(), tmp.begin(), tmp.end());
    }
    
    return weight_data;
}

void dump_weight_sim(const std::string &filename, const std::vector<short> &weight_set){
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr << "Can not open" << filename << std::endl;
        throw std::invalid_argument("Open file failed.");
    }
    for(const auto &i : weight_set){
        out << fmt::format("{:04x}\n", (ushort)i);
    }
    // std::cout << fmt::format("Total Weight Number : {}\n", weight_set.size()) << std::endl;
    out.close();
}

void dump_total_tile_sim(const std::string &filename, const std::vector<Layer_Info> &info_set){
    std::ofstream out(filename);
    std::vector<uint32_t> data;
    const size_t PER_TILE_BYTE = 32;
    const size_t TILE_COUNT = PER_TILE_BYTE/(sizeof(uint32_t));
    if(!out.is_open()){
        std::cerr << "Can not open " << filename << std::endl;
        throw std::invalid_argument("Can not open file for dump tile");
    }
    for(const auto &info : info_set){
        if(info.tile_info.size() % TILE_COUNT != 0){
            throw std::invalid_argument("Tile Info format error");
        }
        for(size_t i = 0; i < info.tile_info.size()/TILE_COUNT; i++){
            for(int j = TILE_COUNT-1; j >= 0; j--){
                if(j < 3){
                    if(j == 1)
                        out << fmt::format("{:08x}",info.tile_info[i*TILE_COUNT+j]+WEIGHT_BASE_ADDR);
                    else
                        out << fmt::format("{:08x}",info.tile_info[i*TILE_COUNT+j]+DATA_BASE_ADDR);
                }
                else
                    out << fmt::format("{:08x}",info.tile_info[i*TILE_COUNT+j]);
            }
            out << "\n";
        }
    }

    out.close();
}

int main(int argc, char **argv){
    if(argc != 3){
        std::cerr << fmt::format("Input <json file name> <weight directory path>");
        return -1;
    }

    //Parse Json
    auto info_set = parse_json(argv[1]);
    
    //Generate Tile Info
    for(auto &i : info_set)
        i.gen_tile_info();
    

    // for(auto &i : info_set){
    //     i.dump_for_sim(DATA_BASE_ADDR, WEIGHT_BASE_ADDR);
    //     std::cout << i.node_name << std::endl;
    // }
    
    #ifdef DUMP_TILE_INFO
        #ifndef DUMP_FOR_SIM
            dump_tile_bin("tile_info_test.bin", info_set);
        #else
            dump_total_tile_sim("total_tile_info.txt", info_set);
            /*
            for(auto &i : info_set){
                i.dump_for_sim(DATA_BASE_ADDR, WEIGHT_BASE_ADDR);
            }
            */
        #endif
    #endif
    
    //Dump Layer Info
    #ifdef DUMP_LAYER_INFO
        auto layer_info_data = gen_layer_info_data(info_set);
        #ifndef DUMP_FOR_SIM
            dump_layer_info_data_bin("layer_info.bin", layer_info_data);
        #else
            dump_layer_info_sim("layer_info.txt", layer_info_data);
        #endif
    #endif
    
    //Dump Weight
    #ifdef DUMP_WEIGHT
        std::vector<short> total_weight = gen_total_weight(argv[2], info_set);
        #ifndef DUMP_FOR_SIM
            dump_data_bin("weight.bin", total_weight.data(), size_in_byte(total_weight));
        #else
            dump_weight_sim("total_weight.txt", total_weight);
        #endif
    #endif
    

    std::ofstream out("Output_Offset.txt");
    // std::cout << "Output Address Offset : ";
    for(const auto &i : info_set){
        #ifdef DUMP_FOR_SIM
            out << fmt::format("{:08x}\n", i.tile_info[0]+DATA_BASE_ADDR);
        #else
            if(i.Is_Output_Layer){
                out << fmt::format("{}\n", i.tile_info[0]);
            }
        #endif
    }
    out.close();
    //Print Layer Info
    /*
    for(auto &i : info_set){
        const size_t tile_num = i.Tile_Info_Number* (32/BUS_WIDTH);
        size_t tile_num_tmp = 0;
        std::cout << fmt::format("//{}\n",i.node_name);
        i.Tile_Info_Addr += tmp_addr;
        std::cout << fmt::format("//32'h{:08x}\n", i.Tile_Info_Addr);
        for(size_t j = 0; j < tile_num; j+=tile_num_tmp){
            std::cout << fmt::format("s_axi_start         = 32'd0;\n");
            std::cout << "{s_axi_inst_4,s_axi_inst_3,s_axi_inst_2,s_axi_inst_1,s_axi_inst_0} = 160'h";
            tile_num_tmp = tile_num-j;
            if(tile_num_tmp > HW_INFO_MAX*(32/BUS_WIDTH)){
                tile_num_tmp = HW_INFO_MAX*(32/BUS_WIDTH);
                // std::cout << tile_num_tmp << std::endl;
                tile_num_tmp = std::floor(tile_num_tmp/std::ceil(i.ICH/(double)DATA_DEPTH))*std::ceil(i.ICH/(double)DATA_DEPTH);
                // std::cout << tile_num_tmp << std::endl;
            }
            if(j != 0)
                i.Tile_Info_Addr += tile_num_tmp * BUS_WIDTH;
            // std::cout << tile_num_tmp << std::endl;
            i.Tile_Info_Number = tile_num_tmp - 1;
            auto layer_info = i.Gen_Layer_Info();
            for(int k = layer_info.size()-1; k >= 0; k--){
                std::cout << fmt::format("{:08x}",layer_info[k]);
            }
            std::cout << ";" << std::endl;
            std::cout << fmt::format("#(`period * 10.0)\n");
            std::cout << fmt::format("s_axi_start         = 32'd1;\n");
            std::cout << fmt::format("wait(IRQ);\n");
            tmp_addr += tile_num_tmp * BUS_WIDTH;
        }
    }
    */

    return 0;
}