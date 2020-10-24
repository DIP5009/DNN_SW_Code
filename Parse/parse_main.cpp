#include "parse.hpp"

void dag_dump_json(tree_map &table, std::map<std::string, size_t> &order_table, const std::string &filename, const size_t default_quant){
    if(order_table.size() != table.size())
        throw std::invalid_argument("Order table not equal tree map");
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr << "Can not open " << filename << std::endl;
        return;
    }
    json j;
    std::vector<std::string> order_trans(order_table.size(), "0");
    for(auto &i : order_table){
        order_trans[i.second] = i.first;
    }
    for(size_t i = 0; i < order_trans.size(); i++){
        j[i]["Node_name"] = table[order_trans[i]]->node_name;
        j[i]["Weight_name"] = table[order_trans[i]]->weight;
        j[i]["IF_Size"] = table[order_trans[i]]->IF_Size;
        j[i]["OF_Size"] = table[order_trans[i]]->OF_Size;
        j[i]["Kernel_Size"] = table[order_trans[i]]->Kernel_Size;
        j[i]["Stride"] = table[order_trans[i]]->Stride;
        j[i]["Input_Channel"] = table[order_trans[i]]->ICH;
        j[i]["Output_Channel"] = table[order_trans[i]]->OCH;
        j[i]["Input_Padding_Size"] = table[order_trans[i]]->In_Pad_Size;
        j[i]["Output_Padding_Size"] = table[order_trans[i]]->Out_Pad_Size;
        j[i]["Have_MaxPool"] = table[order_trans[i]]->Have_MaxPool;
        j[i]["Pool_Size"] = table[order_trans[i]]->Pool_Size;
        j[i]["Pool_Stride"] = table[order_trans[i]]->Pool_Stride;
        j[i]["Input_Tile_Size"] = table[order_trans[i]]->Input_Tile_Size;
        j[i]["Output_Tile_Size"] = table[order_trans[i]]->Output_Tile_Size;
        j[i]["Next_Tile_Size"] = table[order_trans[i]]->Next_Tile_Size;
        j[i]["Input_Tile_Number"] = table[order_trans[i]]->Input_Tile_Number;
        j[i]["Output_Tile_Number"] = table[order_trans[i]]->Output_Tile_Number;
        j[i]["Have_ReLU"] = table[order_trans[i]]->Have_ReLU;
        j[i]["Is_LeakyReLU"] = table[order_trans[i]]->Is_LeakyReLU;
        j[i]["Have_BatchNormalization"] = table[order_trans[i]]->Have_BatchNormalization;
        j[i]["Batch_First"] = table[order_trans[i]]->Batch_First;
        j[i]["Have_Upsample"] = table[order_trans[i]]->Have_Upsample;
        j[i]["LeakyReLU_Alpha"] = table[order_trans[i]]->Leaky_ReLU_alpha;
        j[i]["Have_Bias"] = table[order_trans[i]]->Have_Bias;
        j[i]["Is_Output_Layer"] = table[order_trans[i]]->Is_Output_Layer;
        j[i]["quant_batch_bias"] = default_quant;
        j[i]["quant_finish"] = 0;
        j[i]["quant_batch"] = default_quant;
        j[i]["quant_word_size"] = 0;
        j[i]["quant_obuf"] = default_quant;
        if(table[order_trans[i]]->prev_node.size() == 0){
            j[i]["Previous_node_name"][0] = "InputLayer";
            j[i]["Previous_node_OCH"][0] = table[order_trans[i]]->ICH;
            j[i]["Input_Address"][0] = 0;
        }else{
            for(size_t k = 0; k < table[order_trans[i]]->prev_node.size();k++){
                j[i]["Previous_node_name"][k] = table[order_trans[i]]->prev_node[k]->node_name;
                j[i]["Previous_node_OCH"][k] = table[order_trans[i]]->prev_node[k]->OCH;
                j[i]["Input_Address"][k] = table[order_trans[i]]->prev_node[k]->Output_Address;
            }
        }
        // for(size_t k = 0; k < table[order_trans[i]]->Input_Address.size(); k++){
        //     j[i]["Input_Address"][k] = table[order_trans[i]]->Input_Address[k];
        // }
        j[i]["Output_Address"] = table[order_trans[i]]->Output_Address;
        j[i]["Weight_Address"] = table[order_trans[i]]->Weight_Address;

    }
    out << j;
    out.close();
}

inline size_t round(const size_t ch){
    return std::ceil(ch/(double)DATA_DEPTH);
}

int main(int argc, char** argv){

    auto [layers, input_layers, output_layers] = read_json(argv[1]);
    tree_map table = create_table(layers);

    for(auto &i : output_layers){
        for(auto &j : table[i[0]]->prev_node_name){
            table[i[0]]->prev_node.push_back(create_tree(table, j));
        }
    }

    tree_map merge_table;
    std::shared_ptr<model_op_node> compare_p = std::make_shared<model_op_node> ();
    for(auto &i : output_layers){
        merge_table[i[0]] = merge_dag(table, merge_table, i[0], compare_p);    
    }

    for(auto &i : output_layers){
        gen_node_out_pad(merge_table, i[0], 0);
    }

    for(auto &i : merge_table)
        if(i.second == nullptr)
            merge_table.erase(i.first);

    std::map<std::string, size_t> order_table;

    size_t tmp = 0;
    size_t compare_tmp = 0;

    for(auto &i : output_layers){
        tmp = gen_layer_order(merge_table[i[0]], order_table, compare_tmp);
        if(compare_tmp < tmp)
            compare_tmp = tmp;
    }

    gen_out_addr(merge_table, order_table);

    // gen_in_addr(merge_table[output_layers[0][0]]);
    // gen_in_addr(merge_table[output_layers[1][0]]);
    for(auto &i : output_layers){
        gen_in_addr(merge_table[i[0]]);
    }

    tmp = 0;
    std::map<std::string, size_t> weight_offset_table;
    for(auto &i : output_layers){
        const size_t Kernel_Size = 3 > merge_table[i[0]]->Kernel_Size ? 3 : merge_table[i[0]]->Kernel_Size;
        tmp = weight_offset(merge_table[i[0]], weight_offset_table, tmp);
        // std::cout << fmt::format("{:08x}",tmp) << std::endl;
        tmp += (round(merge_table[i[0]]->ICH)*round(merge_table[i[0]]->OCH)*std::pow(Kernel_Size, 2)*DATA_DEPTH*DATA_DEPTH)*DATA_WIDTH;
        tmp += round(merge_table[i[0]]->ICH)*round(merge_table[i[0]]->OCH)*DATA_DEPTH*2*DATA_WIDTH;
        // std::cout << fmt::format("{:08x}",tmp) << std::endl;
    }

    // for(auto&i : weight_offset_table){
    //     std::cout << i.first << std::endl;
    //     std::cout << fmt::format("{:08x}",i.second) << std::endl;
    //     std::cout << std::endl;
    // }

    for(auto&i : merge_table){
        i.second->Weight_Address = weight_offset_table[i.second->weight];
    }

    // for(auto &i : merge_table){
    //     std::cout << i.first << std::endl;
    //     i.second->print_info();
    //     std::cout << std::endl;
    // }

    for(auto &i : output_layers){
        merge_table[i[0]]->Is_Output_Layer = true;
    }

    dag_dump_json(merge_table, order_table, "merge.json", std::stoi(argv[2]));

    return 0;
}