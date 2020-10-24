#include "ctrl_ip.hpp"

template<typename T>
size_t size_in_byte(const std::vector<T> &a){
    return  a.size()*sizeof(T);
}
void layer_info::load_layer_info(const void* src){
    memcpy(data, src, 20);
}

void layer_info::run_inference(const int fd){
    uint32_t *tmp = (uint32_t *)data;
    ioctl(fd, LAYER_INFO_0, tmp[0]);
    ioctl(fd, LAYER_INFO_1, tmp[1]);
    ioctl(fd, LAYER_INFO_2, tmp[2]);
    ioctl(fd, LAYER_INFO_3, tmp[3]);
    ioctl(fd, LAYER_INFO_4, tmp[4]);
    if(ioctl(fd, START) < 0){
        if(ioctl(fd, R_ERROR_TYPE) != 0){
            std::cerr << std::hex << "R Error : " << ioctl(fd, R_ERROR_ADDR) << std::endl;
        }
        if(ioctl(fd, W_ERROR_TYPE) != 0){
            std::cerr << std::hex << "W Error : " << ioctl(fd, W_ERROR_ADDR) << std::endl;
        }
    }
}

uint32_t layer_info::get_tile_begin_addr() const{
    uint32_t tmp = 0;
    for(int i = 0; i < 4; i++){
        tmp |= data[i+10] << (i*8);
    }
    return tmp;
}

void layer_info::set_tile_begin_addr(uint32_t addr){
    for(int i = 0; i < 4; i++){
        data[i+10] = (addr>>(i*8))&0xff;
    }
}

void layer_info::init_addr(uint32_t base_addr){
    set_tile_begin_addr(get_tile_begin_addr()+base_addr);
}

void layer_info::get_layer_info(){
    std::cout << std::dec;
    std::cout << "Have ReLU : ";
    if(data[0] & 0b10){
        std::cout << "True" << std::endl;
        std::cout << "\tReLU Type : ";
        if(data[0] & 0b1)
            std::cout << "Leaky ReLU";
        else
            std::cout  << "ReLU";
        std::cout << std::endl;
    }else{
        std::cout << "False" << std::endl;
    }
    
    std::cout << "Have Batch Normalization : ";
    if(data[0] & 0b100){
        std::cout << "True" << std::endl;
    }else{
        std::cout << "False" << std::endl;
    }

    std::cout << "Have Max Pooling : ";
    if(data[0] & 0b1000){
        std::cout << "True" << std::endl;
        std::cout << "\tMax Pooling Stride : " << (uint32_t)(data[1]&0b11) << std::endl;
        std::cout << "\tMax Pooling Size : " << (uint32_t)((data[1]&0b1100)>>2) << std::endl;
    }else{
        std::cout << "False" << std::endl;
    }

    std::cout << "Bit Serial : " << (uint32_t)(data[0] >> 5) << std::endl;

    std::cout << "Padding Size : " << (uint32_t)((data[1]&0b110000)>>4) << std::endl;

    std::cout << "Stride Size : " << (uint32_t)((data[1]&0b11000000)>>6) << std::endl;

    std::cout << "Kernel Size : " << (uint32_t)(data[2]&0b11) << std::endl;

    std::cout << "Output Tile Size : " << (uint32_t)((data[2]&0b11111100)>>2) << std::endl;

    std::cout << "Input Tile Size : " << (uint32_t)(data[3]&0b111111) << std::endl;

    std::cout << "Next Input Tile Size : " << (uint32_t)(data[14]&0b111111) << std::endl;

    std::cout << "Output Tile Number : " << (uint32_t)(((data[4]&0b111111) << 2) + ((data[3]&0b11000000)>>6)) << std::endl;

    std::cout << "quant_batch_bias : " << (uint32_t)(((data[5]&0b1111) << 2) + ((data[4]&0b11000000)>>6)) << std::endl;

    int8_t tmp = ((data[6]&0b11) << 4) + ((data[5]&0b11110000)>>4);
    if(tmp & 0b100000)
        tmp |= 0b11000000;

    std::cout << "quant_batch_finish : " << (int)tmp << std::endl;

    std::cout << "quant_batch : " << (uint32_t)((data[6]&0b11111100)>>2) << std::endl;

    std::cout << "quant_word_size : " << (uint32_t)(data[7]&0b111111) << std::endl;

    std::cout << "quant_obuf : " << (uint32_t)(((data[8]&0b1111) << 2) + ((data[7]&0b11000000)>>6)) << std::endl;

    std::cout << "Tile Info Number : " << (uint32_t)((data[9] << 4) + ((data[8]&0b11110000)>>4)) << std::endl;

    std::cout << "Is Upsample : " << ((data[14]&0b1000000) ? "True" : "False")<< std::endl;

    std::cout << "Leaky ReLU Factor : " <<  (uint32_t)(((data[14]&0b10000000) >> 7) | (data[15] << 1) | ((data[16] & 0b1111111) << 9)) << std::endl;

    std::cout << "Tile Info Address : " << std::hex << get_tile_begin_addr() << std::endl;

}

std::vector<layer_info> parse_file(const std::string filename){
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0)
        throw std::invalid_argument("Can not open layer bin file.");
    
    std::vector<layer_info> tmp;
    int file_size = (int) fsize(filename.c_str());

    if(file_size%20 != 0){
        throw "Layer bin file format error.";
    }

    tmp.resize(file_size/20);

    void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    memcpy(tmp[0].data, map_memory, file_size);
    munmap(map_memory, file_size);
    close(fd);
    
    return tmp;
}
