#include "inst.hpp"
#include <cassert>

template<typename T>
size_t size_in_byte(const std::vector<T> &a){
    return  a.size()*sizeof(T);
}

void inst::load_data(const void* src){
    memcpy(data, src, 32);
}

void inst::set_out_addr(uint32_t addr){
    for(int i = 0; i < 4; i++){
        data[i] = (addr>>(i*8))&0xff;
    }
}

void inst::set_weight_addr(uint32_t addr){
    for(int i = 0; i < 4; i++){
        data[i+4] = (addr>>(i*8))&0xff;
    }
}

void inst::set_in_addr(uint32_t addr){
    for(int i = 0; i < 4; i++){
        data[i+8] = (addr>>(i*8))&0xff;
    }
}

uint32_t inst::get_out_addr(){
    uint32_t tmp = 0;
    for(int i = 0; i < 4; i++){
        tmp |= data[i] << (i*8);
    }
    return tmp;
}

uint32_t inst::get_weight_addr(){
    uint32_t tmp = 0;
    for(int i = 0; i < 4; i++){
        tmp |= data[i+4] << (i*8);
    }
    return tmp;
}

uint32_t inst::get_in_addr(){
    uint32_t tmp = 0;
    for(int i = 0; i < 4; i++){
        tmp |= data[i+8] << (i*8);
    }
    return tmp;
}

/*
void init_inst_addr(std::vector<std::vector<inst>> &inst_data_set,const uint32_t base_addr, const uint32_t weight_addr){
    for(size_t i = 0; i < inst_data_set.size(); i++){
        for(size_t j = 0; j < inst_data_set[i].size(); j++){
            inst tmp = inst_data_set[i].data()[j];
            tmp.set_in_addr(tmp.get_in_addr() + base_addr);
            tmp.set_out_addr(tmp.get_out_addr() + base_addr);
            tmp.set_weight_addr(tmp.get_weight_addr() + weight_addr);
            inst_data_set[i].data()[j] = tmp;
        }
    }
}
*/
void init_info_addr(std::vector<inst> &inst_data, const uint32_t base_addr, const uint32_t weight_addr){
    for(auto &i : inst_data){
        i.set_in_addr(i.get_in_addr()+ base_addr);
        i.set_out_addr(i.get_out_addr()+ base_addr);
        i.set_weight_addr(i.get_weight_addr()+ weight_addr);
    }
}

std::vector<inst> load_inst_data(const std::string filename){
    int fd;
    std::vector<inst> tmp;
    static_assert(std::is_pod_v<inst>);
    static_assert(sizeof(inst) == 32);

    fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("Can not open tile info file.");
    }

    const size_t file_size = (int)fsize(filename.c_str());
    void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    tmp.resize(file_size/sizeof(inst));
    assert(map_memory != nullptr);
    assert(file_size%sizeof(inst) == 0);
    memcpy(tmp[0].data, map_memory, file_size);
    munmap(map_memory, file_size);
    close(fd);

    return tmp;
}

/*
std::vector<std::vector<inst>> load_inst_file_set(const std::string filename){
    std::ifstream in(filename);
    
    if(!in.is_open())
        throw std::invalid_argument("Can not open inst_file_set.");

    int fd;
    int file_size;
    int file_num;
    std::string str;
    std::vector<std::vector<inst>> inst_data_set;
    
    file_num = std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n') + 1;
    in.seekg(in.beg);

    inst_data_set.reserve(file_num);

    for(int i = 0; i < file_num; i++){
        std::vector<inst> tmp;
        static_assert(std::is_pod_v<inst>);
        static_assert(sizeof(inst) == 32);

        std::getline(in, str);
        if(in.eof()){
            return inst_data_set;
        }

        fd = open(str.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
        if(fd < 0){
            throw std::invalid_argument("Can not open inst_bin_file.");
        }

        file_size = (int)fsize(str.c_str());
        void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
        tmp.resize(file_size/sizeof(inst));
        assert(map_memory != nullptr);
        assert(file_size%sizeof(inst) == 0);
        memcpy(tmp[0].data, map_memory, file_size);
        munmap(map_memory, file_size);
        close(fd);

        inst_data_set.push_back(tmp);
    }
    return inst_data_set;
}
*/

void load_weight(const std::string filename, void* dst){
    if(dst == nullptr){
        throw std::invalid_argument("dst pointer is null.");
    }
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("Can not open weight binaray file.");
    }
    int file_size = (int)fsize(filename.c_str());
    void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    memcpy(dst, map_memory, file_size);
    munmap(map_memory, file_size);
    close(fd);
}
