#include "info_gen.hpp"

std::vector<uint32_t> Layer_Info::Gen_Layer_Info() const{
    std::vector<uint32_t> Inst(5,0);
    Inst[0] =  Is_LeakyReLU;
    Inst[0] |= (Have_ReLU&1) << 1;
    //TODO:Fine tune Hardware
    Inst[0] |= ((Have_BatchNormalization||Have_Bias)&1) << 2;
    Inst[0] |= (Have_MaxPool&1) << 3;
    Inst[0] |= (Batch_First&1) << 4;
    Inst[0] |= (Bit_Serial&0b111) << 5;
    Inst[0] |= (Pool_Stride&0b11) << 8;
    Inst[0] |= (Pool_Size&0b11) << 10;
    Inst[0] |= (Out_Pad_Size&0b11) << 12;
    Inst[0] |= (Stride&0b11) << 14;
    Inst[0] |= (Kernel_Size&0b11) << 16;
    Inst[0] |= (Output_Tile_Size&0b111111) << 18;
    Inst[0] |= (Input_Tile_Size&0b111111) << 24;
    Inst[0] |= (Output_Tile_Number&0b00000011) << 30;
    Inst[1] =  (Output_Tile_Number&0b11111100) >> 2;
    Inst[1] |= (quant_batch_bias&0b111111) << 6;
    Inst[1] |= (quant_finish&0b111111) << 12;
    Inst[1] |= (quant_batch&0b111111) << 18;
    Inst[1] |= (quant_word_size&0b111111) << 24;
    Inst[1] |= (quant_obuf&0b000011) << 30;
    Inst[2] =  (quant_obuf&0b111100) >> 2;
    Inst[2] |= (Tile_Info_Number&0xfff) << 4;
    Inst[2] |= (Tile_Info_Addr&0x0000ffff) << 16;
    Inst[3] =  (Tile_Info_Addr&0xffff0000) >> 16;
    Inst[3] |= (Next_Tile_Size&0x3f) << 16;
    Inst[3] |= (Have_Upsample&0x1) << 22;
    Inst[3] |= (Leaky_ReLU_alpha_FP&0x01ff) << 23;
    Inst[4] |= (Leaky_ReLU_alpha_FP&0xfe00) >> 9;
    return Inst;
}

void Layer_Info::gen_t_type_m(){
    std::bitset<4> valid_value;
    std::bitset<4> four_sided;
    t_type.reserve(out_addr_set.size());

    const size_t i_tile_num = Input_Tile_Number;
    const size_t next_tile_num = Output_Tile_Number;
    const size_t o_tile_count  = i_tile_num / next_tile_num;
    // std::cout << fmt::format("i_tile_num : {}, next_tile_num : {}, o_tile_count : {}",i_tile_num, next_tile_num
    // , o_tile_count) << std::endl;

    for(size_t j = 0; j < i_tile_num; j++){ //Only for max pooling
        for(size_t i = 0; i < i_tile_num; i++){
            four_sided = 0b0000;
            if(i%o_tile_count == 0){
                four_sided[LEFT] = 1;
            }else if(i%o_tile_count == o_tile_count-1){
                four_sided[RIGHT] = 1;
            }else{
                four_sided[RIGHT] = 0;
                four_sided[LEFT] = 0;
            }

            if(j%o_tile_count == 0){
                four_sided[TOP] = 1;
            }else if(j%o_tile_count == o_tile_count-1){
                four_sided[BOTTOM] = 1;
            }else{
                four_sided[TOP] = 0;
                four_sided[BOTTOM] = 0;
            }

            if(Input_Tile_Number == 1/*Next_Tile_Size <= MAX_TILE_SIZE*/){
                if(Out_Pad_Size != 0)
                    four_sided = 0b1111;
                else
                    four_sided = 0b0000;
            }

            if(i == 0){
                valid_value[LEFT] = 0;
            }else{
                valid_value[LEFT] = four_sided[LEFT];
            }

            if(i == i_tile_num-1){
                valid_value[RIGHT] = 0;
            }else{
                valid_value[RIGHT] = four_sided[RIGHT];
            }

            if(j == 0){
                valid_value[TOP] = 0;
            }else{
                valid_value[TOP] = four_sided[TOP];
            }

            if(j == i_tile_num-1){
                valid_value[BOTTOM] = 0;
            }else{
                valid_value[BOTTOM] = four_sided[BOTTOM];
            }

            if(Out_Pad_Size == 0){
                valid_value = 0b0000;
            }

            u_char tmp = four_sided.to_ulong();
            tmp = (tmp<<4) + valid_value.to_ulong();
            // std::cout <<  fmt::format("{:x}\n", tmp);
            t_type.push_back(tmp);
        }
    }
}

void Layer_Info::gen_out_addr(){
    const size_t i_tile_num = Input_Tile_Number;
    const size_t next_tile_size = Next_Tile_Size;
    const size_t next_tile_num  = Output_Tile_Number;
    const size_t o_tile_count  = i_tile_num / next_tile_num;
    const size_t o_tile_size = Output_Tile_Size;
    const size_t next_i_tile_n = Output_Tile_Number;
    const size_t channel_align = std::ceil(OCH/(double)DATA_DEPTH);
    out_addr_set.resize(i_tile_num*i_tile_num*channel_align,0);
    // std::cout << fmt::format("{:08x}\n", Output_Address);
    // std::cout << node_name << std::endl;
    // std::vector<size_t> next_in_addr;
    // next_in_addr.reserve(i_tile_num*i_tile_num*std::ceil(OCH/(double)DATA_DEPTH));
    for(size_t i = 0; i < i_tile_num; i++){
        size_t tmp =  Output_Address + (i/(o_tile_count)) * next_tile_size * next_tile_size * next_i_tile_n * BUS_WIDTH;
        // std::cout << fmt::format("{:08x}\n", tmp);
        tmp += (i % o_tile_count) * o_tile_size * next_tile_size * BUS_WIDTH;
        // std::cout << fmt::format("{:08x}\n", tmp);
        for(size_t j = 0; j < i_tile_num; j++){
            for(size_t och = 0; och < channel_align; och++){
                size_t index = (i*i_tile_num+j)*channel_align+och;
                out_addr_set[index] = tmp;
                out_addr_set[index] += (j/o_tile_count) * next_tile_size * next_tile_size * BUS_WIDTH;
                out_addr_set[index] += (j%o_tile_count) * o_tile_size *  BUS_WIDTH;
                out_addr_set[index] += och * next_tile_size * next_tile_size * next_i_tile_n * next_i_tile_n * BUS_WIDTH;
                // if((j%o_tile_count == 0) && (i%o_tile_count == 0))
                //     next_in_addr.push_back(out_addr[index]);
                // std::cout << fmt::format("{:08x}\n", out_addr_set[index]);
            }
        }
    }
}

void Layer_Info::gen_tile_info(){
    gen_out_addr();
    gen_t_type_m();
    const size_t i_tile_num = Input_Tile_Number;
    const size_t ICH_Round = std::ceil(ICH/(double)DATA_DEPTH);
    const size_t OCH_Round = std::ceil(OCH/(double)DATA_DEPTH);
    Tile_Info_Number = Input_Tile_Number*Input_Tile_Number*ICH_Round*OCH_Round;
    const size_t tile_info_num = Tile_Info_Number;
    tile_info.resize(tile_info_num*8); //256-bit
    // bool sel_buf = false;   //Select A B buffer A = 0, B = 1
    const size_t orig_ts = std::ceil((Input_Tile_Size-Kernel_Size+1)/(double)Stride);// have ceil determined by model
    const size_t out_ts = Output_Tile_Size;
    //Input tile count
    const size_t it_count = Input_Tile_Number;
    size_t tmp_in_addr = 0;
    size_t index = 0;
    size_t tile_num_count = 0;
    const size_t max_tile_num = std::floor(HW_INFO_MAX/(double)ICH_Round)*(ICH_Round);
    size_t tmp_wgt_addr;
    for(size_t i = 0; i < it_count*it_count; i++){
        for(size_t j = 0; j < OCH_Round; j++){
            size_t input_addr_count = 0;
            size_t ich_count = 0;
            for(const size_t layer_ich : Previous_node_OCH){
                for(size_t k = 0; k < std::ceil(layer_ich/(double)DATA_DEPTH); k++){
                    index = (i * OCH_Round + j) * ICH_Round + ich_count;
                    tile_info[index*8] = out_addr_set[i*OCH_Round+j];
                    tmp_in_addr = Input_Address[input_addr_count];
                    tmp_in_addr += k * Input_Tile_Size * Input_Tile_Size * i_tile_num * i_tile_num * BUS_WIDTH;
                    tmp_in_addr += i * Input_Tile_Size * Input_Tile_Size * BUS_WIDTH;
                    tile_info[index*8+2] = tmp_in_addr;
                    // size_t tmp_wgt_addr = Weight_Address;
                    //TODO: Kernel Size only for 3x3
                    // tmp_wgt_addr += (j * ICH_Round + ich_count) * (9 * DATA_DEPTH * DATA_DEPTH * DATA_WIDTH + DATA_DEPTH * 2 * DATA_WIDTH);
                    tmp_wgt_addr = (j * ICH_Round + ich_count) * (9 * DATA_DEPTH * DATA_DEPTH * DATA_WIDTH + DATA_DEPTH * 2 * DATA_WIDTH) + Weight_Address;
                    tile_info[index*8+1] = tmp_wgt_addr;

                    if(ich_count == ICH_Round-1)
                        tile_info[index*8+3] = 1;
                    else
                        tile_info[index*8+3] = 0;
                    
                    if(ich_count != 0)
                        tile_info[index*8+3] |= 1 << 1;
                    else
                        tile_info[index*8+3] |= 0 << 1;
                    
                    //TODO:Remove this(Hardware)
                    if(false)
                        tile_info[index*8+3] |= 1 << 2;
                    else
                        tile_info[index*8+3] |= 0 << 2;
                    
                    // sel_buf = !sel_buf;

                    if((i == it_count*it_count-1) && (j == OCH_Round-1) && (ich_count == ICH_Round-1))
                        tile_info[index*8+3] |= 1 << 3;
                    else
                        tile_info[index*8+3] |= 0 << 3;

                    if((tile_num_count%max_tile_num) == max_tile_num-1){
                        tile_info[index*8+3] |= 1 << 3;
                    }

                    tile_info[index*8+3] |= t_type[i] <<4;

                    tile_info[index*8+3] |= ((size_t)std::ceil(orig_ts*orig_ts/2.0)&0x3ff) << 12;

                    if(Have_MaxPool && (ich_count == ICH_Round-1))
                        tile_info[index*8+3] |= ((size_t)std::ceil(out_ts*out_ts)&0x3ff) << 22;
                    else
                        tile_info[index*8+3] |= ((size_t)std::ceil(orig_ts*orig_ts/2.0)&0x3ff) << 22;
                    
                    tile_info[index*8+4] =  orig_ts;
                    
                    if(Have_MaxPool&&(ich_count == ICH_Round-1))
                        tile_info[index*8+4] |= out_ts << 10;
                    else
                        tile_info[index*8+4] |= orig_ts << 10;
                    
                    ich_count++;
                    tile_num_count++;
                }
                input_addr_count++;
            }
        }
    }
}

void Layer_Info::dump_for_sim(const size_t Data_Offset, const size_t Wegiht_Offset) const{
    const std::string filename = node_name + ".txt";
    std::ofstream out;
    out.open(filename);
    if(!out.is_open()){
        throw std::invalid_argument("Can not output info txt");
    }
    size_t index;
    for(size_t i = 0; i < Tile_Info_Number; i++){
        for(int j = 7; j >= 0; j--){
            index = i*8+j;
            if(j == 2){
                out << fmt::format("{:08x}",tile_info[index]+Data_Offset);
            }else if(j == 1){
                out << fmt::format("{:08x}",tile_info[index]+Wegiht_Offset);
            }else if(j == 0){
                out << fmt::format("{:08x}",tile_info[index]+Data_Offset);
            }else{
                out << fmt::format("{:08x}",tile_info[index]);
            }
        }
        out << std::endl;
    }

    out.close();
}

std::vector<Layer_Info> parse_json(const std::string filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::invalid_argument("Can not open json file");
    }
    const json parse_data = json::parse(in);
    std::vector<Layer_Info> info_set;
    info_set.reserve(parse_data.size());
    for(const auto &i : parse_data){
        Layer_Info tmp;
        tmp.node_name = i["Node_name"];
        tmp.weight = i["Weight_name"];
        tmp.IF_Size = i["IF_Size"];
        tmp.OF_Size = i["OF_Size"];
        tmp.Kernel_Size = i["Kernel_Size"];
        tmp.Stride = i["Stride"];
        tmp.ICH = i["Input_Channel"];
        tmp.OCH = i["Output_Channel"];
        tmp.In_Pad_Size = i["Input_Padding_Size"];
        tmp.Out_Pad_Size = i["Output_Padding_Size"];
        tmp.Have_MaxPool = i["Have_MaxPool"];
        tmp.Pool_Size = i["Pool_Size"];
        tmp.Pool_Stride = i["Pool_Stride"];
        tmp.Input_Tile_Size = i["Input_Tile_Size"];
        tmp.Output_Tile_Size = i["Output_Tile_Size"];
        tmp.Next_Tile_Size = i["Next_Tile_Size"];
        tmp.Input_Tile_Number = i["Input_Tile_Number"];
        tmp.Output_Tile_Number = i["Output_Tile_Number"];
        tmp.Have_ReLU = i["Have_ReLU"];
        tmp.Is_LeakyReLU = i["Is_LeakyReLU"];
        tmp.Have_BatchNormalization = i["Have_BatchNormalization"];
        tmp.Have_Bias = i["Have_Bias"];
        tmp.Batch_First = i["Batch_First"];
        tmp.Have_Upsample = i["Have_Upsample"];
        tmp.Leaky_ReLU_alpha = i["LeakyReLU_Alpha"];
        tmp.quant_batch_bias = i["quant_batch_bias"];
        tmp.quant_finish = i["quant_finish"];
        tmp.quant_batch = i["quant_batch"];
        tmp.quant_word_size = i["quant_word_size"];
        tmp.quant_obuf = i["quant_obuf"];
        tmp.Is_Output_Layer = i ["Is_Output_Layer"];
        for(const size_t &j : i["Input_Address"]){
            tmp.Input_Address.push_back(j);
        }
        tmp.Output_Address = i["Output_Address"];
        tmp.Weight_Address = i["Weight_Address"];
        for(const size_t &j : i["Previous_node_OCH"]){
            tmp.Previous_node_OCH.push_back(j);
        }
        tmp.Leaky_ReLU_alpha_FP = tmp.Leaky_ReLU_alpha*std::pow(2, tmp.quant_batch);
        info_set.push_back(tmp);
    }
    in.close();
    return info_set;
}