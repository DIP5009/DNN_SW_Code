#include "inst.hpp"
#include "ctrl_ip.hpp"
#include <fmt/format.h>
#include "pre_image.hpp"
#include <tuple>

#define OUTPUT_LAYER_NUM 2
#define OUTPUT_LAYER_1 9
#define OUTPUT_LAYER_2 14
#define BUS_WIDTH  8
#define DATA_WIDTH 2
#define DATA_DEPTH (BUS_WIDTH/DATA_WIDTH)

template<typename T>
size_t size_in_byte(const std::vector<T> &a){
    return  a.size()*sizeof(T);
}

#define MEM_SIZE (1024*1024*12)
#define PAGE_SIZE 4096

const uint32_t load_weight(void* dst, const std::string filename){
    int fd = open(filename.c_str(), O_RDWR);
    if(fd < 0)
        throw std::invalid_argument("Can not load weight.bin");
    const uint32_t file_size = (int)fsize(filename.c_str());
    void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    memcpy(dst, map_memory, file_size);
    munmap(map_memory, file_size);
    std::cout << 3 << std::endl;
    close(fd);
    return file_size;
}

bool check_have_final(const layer_info &layer, const void* baseaddr){
    const size_t HW_SHIFT = 2; //Bus data width 128 bit = 1, 64 bit = 2
    const uint32_t offset = ((uint32_t)((layer.data[9] << 4) + ((layer.data[8]&0b11110000)>>4)))>>HW_SHIFT;
    const uint32_t *tmp = (uint32_t *)((char *)baseaddr + layer.get_tile_begin_addr());
    return (tmp[offset*8+3]&0b1000)>>3;
}

std::tuple<std::vector<layer_info> ,const uint32_t> run_init(const std::string &weight,const std::string &tile_info_file, const std::string &layer_info_file, const u_int32_t phy_addr, uint8_t* dst){
    std::vector<inst> tile_info = load_inst_data(tile_info_file);
    std::vector<layer_info> layer =  parse_file(layer_info_file);
    const uint32_t tile_info_offset =  std::ceil(size_in_byte(tile_info)/(double)PAGE_SIZE)*PAGE_SIZE;
    const uint32_t wieght_offset = load_weight(dst+tile_info_offset, weight);
    const uint32_t input_addr  = std::ceil((phy_addr+tile_info_offset+wieght_offset)/(double)PAGE_SIZE)*PAGE_SIZE;
    const uint32_t input_offset = input_addr - phy_addr;
    const uint32_t weight_addr = phy_addr+tile_info_offset;
    std::cout << 2 << std::endl;
    init_info_addr(tile_info, input_addr, weight_addr);
    memcpy(dst, tile_info.data(), size_in_byte(tile_info));
    for(auto &i : layer){
        if(!check_have_final(i, dst)){
            std::cerr << "Tile infomation format error.\n";
        }
    }
    for(uint32_t i = 0; i < layer.size(); i++){
        layer[i].init_addr(phy_addr);
    }
    return std::tuple<std::vector<layer_info> ,const uint32_t>{layer, input_offset};
}

void spilt_channel(const void* src, void* dst, const uint32_t width, const uint32_t hight, const uint32_t channel){
    const short *tmp_src = (const short *) src;
    short *tmp_dst = (short *) dst; 
    const uint32_t channel_count = std::ceil(channel/(double)DATA_DEPTH);
    uint32_t src_index = 0;
    uint32_t dst_index = 0;

    for(uint32_t ch = 0; ch < channel_count; ch++){
        for(uint32_t c = 0; c < ((channel > (ch+1)*DATA_DEPTH)? DATA_DEPTH : (channel - ch*DATA_DEPTH)); c++){
            for(uint32_t y = 0; y < hight; y++){
                for(uint32_t x = 0; x < width; x++){
                    dst_index = (ch*DATA_DEPTH+c)*width*hight+y*width+x;
                    // std::cout << fmt::format("dst : {}, channel : {}, x : {}, y : {}\n", dst_index, ch*8+c, x, y);
                    src_index = (ch*width*hight+y*width+x)*DATA_DEPTH+c;
                    tmp_dst[dst_index] = tmp_src[src_index];
                }
            }
        }
    }
}

void dump_memory(const void * src, uint32_t size){
    int fd = open("Memory.bin", O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0)
        throw std::invalid_argument("Can not open Memory.bin");
    write(fd, src, size);
    close(fd);
}

int dump_data(const void * src, const std::string filename, const uint32_t w, const uint32_t h, const float f){
    const short *tmp = (const short *)src;
    std::ofstream out;
    out.open(filename);
    if(!out.is_open()){
        std::cout << "Can not open " << filename << std::endl;
        return -1;
    }
    for(uint32_t i = 0; i < h; i++){
        for(uint32_t j = 0; j < w; j++){
            out << tmp[i*w+j]/f;
            if(j != w-1){
                out << ",";
            }
        }
        out << std::endl;
    }
    out.close();
    return 0;
}

int main(int argc, char **argv){

    if(argc != 7){
        std::cout << 1 << std::endl;
        return -1;
    }
    
    const int fd = open(argv[1], O_RDWR);
    if(fd < 0){
        std::cerr << "Can not open " << argv[1] << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[5], cv::CAP_FFMPEG);
    if(!cap.isOpened()){
        close(fd);
        std::cerr << "Can not open " << argv[5] << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 20);

    const std::string layer_info_file = argv[2];
    const std::string tile_info_file = argv[3];
    const std::string weight_file = argv[4];
    // const std::string image_file = argv[5];
    std::ifstream in(argv[6]);
    size_t out_addr_tmp = 0;
    const int iou = 30;
    const int conf = 30;
    short *res_1 = (short *)malloc(8*8*18*sizeof(short));
    short *res_2 = (short *)malloc(16*16*18*sizeof(short));

    float draw_1 [8][8][18] = {0};
    float draw_2 [16][16][18] = {0};

    
    const uint32_t phy_addr = ioctl(fd, MEM_ALLOC, MEM_SIZE);
    uint8_t *virt_addr = (uint8_t *)mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd , 0);
    std::vector<uint32_t> out_addr;
    while (in >> out_addr_tmp){
        out_addr.push_back(out_addr_tmp);
    }
    in.close();
    auto [layer, input_offset] = run_init(weight_file, tile_info_file, layer_info_file, phy_addr, virt_addr);
	std::cout << "1" << std::endl;
    if(layer.data() == nullptr){
        munmap(virt_addr, MEM_SIZE);
        std::cout << "Error" << std::endl;
        ioctl(fd, MEM_FREE, MEM_SIZE);
        cap.release();
        close(fd);
        return -1;
    }

    std::cout << std::hex << "Phy addr = " << phy_addr << std::endl;
    std::cout << fmt::format("Input offset : {:08x}, Input address : {:08x}\n", input_offset, input_offset+phy_addr);


    cv::Mat pit;
    // cv::resize(cv::imread(image_file), pit, cv::Size(256,256));

    uint16_t *pre_image = (uint16_t *)malloc(34*34*64*DATA_DEPTH*2);
    memset(pre_image, 0, 34*34*64*DATA_DEPTH*2);

    cv::Mat frame;
    while(true){
        cap.read(frame);
        if(frame.empty()){
            break;
        }
        if(cv::waitKey(1) == 'q'){
            break;
        }

        cv::resize(frame, pit, cv::Size(256,256));
        image_pre_process(pit, pre_image, 8, 8, 2);
        memcpy(virt_addr+input_offset, pre_image, 34*34*64*DATA_DEPTH*2);

        for(auto &i : layer){
            i.run_inference(fd);
        }

        //std::cout << fmt::format();


        
        spilt_channel((virt_addr+out_addr[0]+input_offset), res_1, 8, 8, 18);
	spilt_channel((virt_addr+out_addr[1]+input_offset), res_2, 16, 16, 18);

        for(int ch = 0; ch < 18; ch++){
            for(int y = 0; y < 8; y++){
                for(int x = 0; x < 8; x++){
                    draw_1[y][x][ch] = res_1[ch*8*8+y*8+x]/1024.0;
                }
            }
        }

        for(int ch = 0; ch < 18; ch++){
            for(int y = 0; y < 16; y++){
                for(int x = 0; x < 16; x++){
                    draw_2[y][x][ch] = res_2[ch*16*16+y*16+x]/1024.0;
                }
            }
        }

        localization(pit, draw_1, draw_2, conf, iou);
    }

    /*
    image_pre_process(pit, pre_image, 8, 8, 2);
    memcpy(virt_addr+input_offset, pre_image, 34*34*64*DATA_DEPTH*2);

    for(auto &i : layer){
        i.run_inference(fd);
    }

    //std::cout << fmt::format();


    
    spilt_channel((virt_addr+out_addr[0]+input_offset), res_1, 8, 8, 18);
    spilt_channel((virt_addr+out_addr[1]+input_offset), res_2, 16, 16, 18);

    for(int ch = 0; ch < 18; ch++){
        for(int y = 0; y < 8; y++){
            for(int x = 0; x < 8; x++){
                draw_1[y][x][ch] = res_1[ch*8*8+y*8+x]/1024.0;
            }
        }
    }

    for(int ch = 0; ch < 18; ch++){
        for(int y = 0; y < 16; y++){
            for(int x = 0; x < 16; x++){
                draw_2[y][x][ch] = res_2[ch*16*16+y*16+x]/1024.0;
            }
        }
    }

    localization(pit, draw_1, draw_2, conf, iou);
    */

    free(res_1);
    free(res_2);
    free(pre_image);

    munmap(virt_addr, MEM_SIZE);
    ioctl(fd, MEM_FREE, MEM_SIZE);
    close(fd);
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
