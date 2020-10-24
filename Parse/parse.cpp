#include "parse.hpp"

std::tuple<json, json, json> read_json(const std::string filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::invalid_argument("Can not open json file.");
    }
    json tmp = json::parse(in);
    json layers = tmp["config"]["layers"];
    json input_layers = tmp["config"]["input_layers"];
    json output_layers = tmp["config"]["output_layers"];
    return std::tuple<json, json, json>(layers, input_layers, output_layers);
}

tree_map create_table(const json &layers){
    tree_map table;
    std::string tmp;
    for(auto i : layers){
        std::shared_ptr<model_op_node> p = std::make_shared<model_op_node>();
        tmp = i["name"];
        p->node_name = tmp;

        //Get pervious layer
        for(auto &j : i["inbound_nodes"][0]){
            p->prev_node_name.push_back(j[0]);
        }

        //Set layer infomation
        if(i["class_name"] == "InputLayer"){
            if(i["config"]["batch_input_shape"][1] != i["config"]["batch_input_shape"][2]){
                throw std::invalid_argument("Input shape[0] not equal shape[1]");
            }
            p->IF_Size = i["config"]["batch_input_shape"][1];
            p->ICH = i["config"]["batch_input_shape"][3];
            p->OP_Attribute = OP_Attribute_Type::InputLayer;

        }else if(i["class_name"] == "Conv2D"){
            if(i["config"]["kernel_size"][0] != i["config"]["kernel_size"][1]){
                throw std::invalid_argument("Kernel size shape[0] not equal shape[1]");
            }else if(i["config"]["strides"][0] != i["config"]["strides"][1]){
                throw std::invalid_argument("Kernel stride shape[0] not equal shape[1]");
            }
            p->Kernel_Size = i["config"]["kernel_size"][0];
            p->Stride = i["config"]["strides"][0];
            p->OCH = i["config"]["filters"];
            p->OP_Attribute = OP_Attribute_Type::Conv;
            p->weight = tmp;
            if(i["config"]["padding"] == "valid")
                p->Pad_Attribute = Pad_Attribute_Type::VALID;
            else if(i["config"]["padding"] == "same")
                p->Pad_Attribute = Pad_Attribute_Type::SAME;
            else
                p->Pad_Attribute = Pad_Attribute_Type::Other;

        }else if(i["class_name"] == "MaxPooling2D"){
            if(i["config"]["pool_size"][0] != i["config"]["pool_size"][1]){
                throw std::invalid_argument("Pool size shape[0] not equal shape[1]");
            }else if(i["config"]["strides"][0] != i["config"]["strides"][1]){
                throw std::invalid_argument("Pool stride shape[0] not equal shape[1]");
            }
            p->Pool_Size = i["config"]["pool_size"][0];
            p->Pool_Stride = i["config"]["strides"][0];
            p->OP_Attribute = OP_Attribute_Type::MaxPool;
            if(i["config"]["padding"] == "valid")
                p->Pad_Attribute = Pad_Attribute_Type::VALID;
            else if(i["config"]["padding"] == "same")
                p->Pad_Attribute = Pad_Attribute_Type::SAME;
            else
                p->Pad_Attribute = Pad_Attribute_Type::Other;

        }else if(i["class_name"] == "BatchNormalization"){
            p->OP_Attribute = OP_Attribute_Type::BatchNormalization;

        }else if(i["class_name"] == "LeakyReLU"){
            p->Is_LeakyReLU = true;
            p->Leaky_ReLU_alpha = i["config"]["alpha"];
            p->OP_Attribute = OP_Attribute_Type::ReLU;

        }else if(i["class_name"] == "ReLU"){
            p->Is_LeakyReLU = false;
            p->OP_Attribute = OP_Attribute_Type::ReLU;

        }else if(i["class_name"] == "Concatenate"){
            p->OP_Attribute = OP_Attribute_Type::Concat;
        }else if(i["class_name"] == "UpSampling2D"){
            p->OP_Attribute = OP_Attribute_Type::Upsample;
        }else{
            p->OP_Attribute = OP_Attribute_Type::None;
        }

        table[tmp] = p;
    }
    return table;
}

std::shared_ptr<model_op_node> create_tree(tree_map &table, const std::string &node_name){
    if(table[node_name]->prev_node.size() != 0){
        return table[node_name];
    }else{
        if(table[node_name]->prev_node_name.size() == 0){
            return table[node_name];
        }else{
            for(auto &i :table[node_name]->prev_node_name)
                table[node_name]->prev_node.push_back(create_tree(table, i));
            return table[node_name];
        }
    }
}

bool compare_node(const std::shared_ptr<model_op_node> &node_a, const std::shared_ptr<model_op_node> &node_b){
    if(node_a->Have_MaxPool != node_b->Have_MaxPool){
        return false;
    }
    if(node_a->Have_ReLU != node_b->Have_ReLU){
        return false;
    }
    if(node_a->Is_LeakyReLU != node_b->Is_LeakyReLU){
        return false;
    }
    if(node_a->Have_BatchNormalization != node_b->Have_BatchNormalization){
        return false;
    }
    if(node_a->Batch_First != node_b->Batch_First){
        return false;
    }
    if(node_a->Have_Upsample != node_b->Have_Upsample){
        return false;
    }
    return true;
}

std::shared_ptr<model_op_node> merge_dag (tree_map &table, tree_map &new_table, const std::string &node, std::shared_ptr<model_op_node> &compare_p){
    bool same_node_flag = false;
    if(new_table[node] != nullptr){
        same_node_flag = compare_node(new_table[node], compare_p);
    }

    if(same_node_flag){
        if(new_table[node]->prev_node.size() != 0)
            return new_table[node];
        std::cout << "Error" << std::endl;
    }
    
    if(table[node]->OP_Attribute == OP_Attribute_Type::InputLayer){
        std::shared_ptr<model_op_node> p = std::make_shared<model_op_node> ();
        //set info
        p->IF_Size = table[node]->IF_Size;
        p->ICH = table[node]->ICH;
        p->OP_Attribute = OP_Attribute_Type::InputLayer;
        return p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::Conv){
        if((!same_node_flag) && (new_table[node] != nullptr)){
            if(new_table[node]->prev_node.size() != 0){
                const std::string new_node = node + "_2";
                if(new_table[new_node] != nullptr){
                    return merge_dag(table, new_table, new_node, compare_p);
                }
                std::shared_ptr<model_op_node> add_p = std::make_shared<model_op_node> (*new_table[node]);
                add_p->node_name = new_node;
                add_p->Have_ReLU = false;
                add_p->Have_MaxPool = false;
                add_p->Pool_Size = false;
                add_p->Leaky_ReLU_alpha = 0;
                add_p->Is_LeakyReLU = false;
                add_p->Have_BatchNormalization = false;
                add_p->Batch_First = false;
                add_p->Have_Upsample = false;
                new_table[new_node] = add_p;
                switch (add_p->Pad_Attribute){
                    case Pad_Attribute_Type::Other:{
                        std::cerr << "Hardware can not support" << std::endl;
                        break;
                    }
                    case Pad_Attribute_Type::SAME:{
                        add_p->OF_Size = add_p->IF_Size / add_p->Stride;
                        size_t tmp_of_size = (size_t)std::ceil((add_p->IF_Size - add_p->Kernel_Size + 1) / (double)add_p->Stride);
                        add_p->In_Pad_Size = (add_p->OF_Size - tmp_of_size)/2;
                        break;
                    }
                    case Pad_Attribute_Type::VALID:{
                        add_p->OF_Size = (size_t)std::ceil((add_p->IF_Size - add_p->Kernel_Size + 1) / (double)add_p->Stride);
                        add_p->In_Pad_Size = 0;
                        break;
                    }
                    default:{
                        std::cerr << "Can not parse Pad_Attribute_Type " << std::endl;
                        break;
                    }
                }
                return add_p;
            }
        }

        std::shared_ptr<model_op_node> new_node_p = std::make_shared<model_op_node> ();
        for(auto i : table[node]->prev_node_name){
            if(table[i]->OP_Attribute == OP_Attribute_Type::Concat){
                for(auto j : table[i]->prev_node_name)
                    new_node_p->prev_node_name.push_back(j);
            }else{
                new_node_p->prev_node_name.push_back(i);
            }
        }
        for(auto i : new_node_p->prev_node_name){
            std::shared_ptr<model_op_node> compare_p = std::make_shared<model_op_node> ();
            std::shared_ptr<model_op_node> p = merge_dag(table, new_table, i, compare_p);
            if(p->OP_Attribute != OP_Attribute_Type::InputLayer)
                new_node_p->prev_node.push_back(p);
            if(p->OP_Attribute == OP_Attribute_Type::InputLayer){
                //set info
                new_node_p->ICH = p->ICH;
                new_node_p->IF_Size = p->IF_Size;
            }else{
                new_node_p->ICH += p->OCH;
                if(new_node_p->IF_Size != 0)
                    if(new_node_p->IF_Size != p->OF_Size){
                        std::cout << new_node_p->IF_Size << std::endl;
                        std::cout << p->OF_Size << std::endl;
                        std::cout << p->node_name << std::endl;
                        throw std::invalid_argument("Concat different layer shape is different.");
                    }
                new_node_p->IF_Size = p->OF_Size;
            }
        }

        new_node_p->OP_Attribute = table[node]->OP_Attribute;
        new_node_p->node_name = table[node]->node_name;
        new_node_p->Stride = table[node]->Stride;
        new_node_p->OCH = table[node]->OCH;
        new_node_p->Pad_Attribute = table[node]->Pad_Attribute;
        new_node_p->Kernel_Size = table[node]->Kernel_Size;
        new_node_p->weight = table[node]->weight;

        switch (new_node_p->Pad_Attribute){
            case Pad_Attribute_Type::Other:{
                std::cerr << "Hardware can not support" << std::endl;
                break;
            }
            case Pad_Attribute_Type::SAME:{
                new_node_p->OF_Size = new_node_p->IF_Size / new_node_p->Stride;
                size_t tmp_of_size = (size_t)std::ceil((new_node_p->IF_Size - new_node_p->Kernel_Size + 1) / (double)new_node_p->Stride);
                new_node_p->In_Pad_Size = (new_node_p->OF_Size - tmp_of_size)/2;
                break;
            }
            case Pad_Attribute_Type::VALID:{
                new_node_p->OF_Size = (size_t)std::ceil((new_node_p->IF_Size - new_node_p->Kernel_Size + 1) / (double)new_node_p->Stride);
                new_node_p->In_Pad_Size = 0;
                break;
            }
            default:{
                std::cerr << "Can not parse Pad_Attribute_Type " << std::endl;
                break;
            }
        }
        new_table[node] = new_node_p;
        return new_node_p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::MaxPool){
        compare_p->Have_MaxPool = true;
        if(table[node]->prev_node_name.size() > 1)
            throw std::invalid_argument("MaxPooling have greater 2 inbound_node");
        std::shared_ptr<model_op_node> p = merge_dag(table, new_table, table[node]->prev_node_name[0], compare_p);
        p->Have_MaxPool = true;
        p->Pool_Size = table[node]->Pool_Size;
        p->Pool_Stride = table[node]->Pool_Stride;
        switch (table[node]->Pad_Attribute){
            case Pad_Attribute_Type::Other:{
                std::cerr << "Hardware can not support" << std::endl;
                break;
            }
            case Pad_Attribute_Type::SAME:{
                p->OF_Size = p->IF_Size / table[node]->Pool_Stride;
                break;
            }
            case Pad_Attribute_Type::VALID:{
                p->OF_Size = (size_t)std::ceil((p->IF_Size - table[node]->Pool_Size + 1) / (double)table[node]->Pool_Stride);
                break;
            }
            default:{
                std::cerr << "Can not parse Pad_Attribute_Type " << std::endl;
                break;
            }
        }
        return p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::ReLU){
        compare_p->Have_ReLU = true;
        compare_p->Is_LeakyReLU = table[node]->Is_LeakyReLU;
        // if(compare_p->Have_BatchNormalization)
        //     compare_p->Batch_First = true;
        std::shared_ptr<model_op_node> p = merge_dag(table, new_table, table[node]->prev_node_name[0], compare_p);
        if(table[node]->Is_LeakyReLU){
            p->Is_LeakyReLU = true;
            p->Leaky_ReLU_alpha = table[node]->Leaky_ReLU_alpha;
        }
        p->Have_ReLU = true;
        return p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::BatchNormalization){
        compare_p->Have_BatchNormalization = true;
        if(compare_p->Have_ReLU)
            compare_p->Batch_First = true;
        std::shared_ptr<model_op_node> p = merge_dag(table, new_table, table[node]->prev_node_name[0], compare_p);
        p->Have_BatchNormalization = true;
        if(!p->Have_ReLU)
            p->Batch_First = true;
        return p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::Upsample){
        compare_p->Have_Upsample = true;
        std::shared_ptr<model_op_node> p = merge_dag(table, new_table, table[node]->prev_node_name[0], compare_p);
        p->Have_Upsample = true;
        p->OF_Size = p->OF_Size*2;
        return p;
    }else if(table[node]->OP_Attribute == OP_Attribute_Type::Concat){
        throw std::invalid_argument("Concat only immediately previous Convluation.");
    }else{
        throw std::invalid_argument("Operation not support by hardware");
    }
}

bool gen_node_out_pad(tree_map &table, const std::string &node, const int out_pad_size){
    if(table[node]->prev_node.size() == 0){
        table[node]->Out_Pad_Size = out_pad_size;
        return true;
    }else{
        if(table[node]->Out_Pad_Size >= 0){
            if(table[node]->Out_Pad_Size == out_pad_size)
                return true;
            else
                return false;
        }else{
            table[node]->Out_Pad_Size = out_pad_size;
            for(auto &i : table[node]->prev_node){
                if(!(gen_node_out_pad(table, i->node_name, table[node]->In_Pad_Size))){
                    std::shared_ptr<model_op_node> p = std::make_shared<model_op_node> (*i);
                    const std::string new_node_name = p->node_name + "_2";
                    p->node_name = new_node_name;
                    p->Out_Pad_Size = i->In_Pad_Size;
                    table[new_node_name] = p;
                    i = p;

                }
            }
            return true;
        }
    }
}

inline size_t round(const size_t ch){
    return std::ceil(ch/(double)DATA_DEPTH);
}

size_t gen_layer_order(const std::shared_ptr<model_op_node> &p, std::map<std::string, size_t> &table, const size_t order){
    if(table[p->node_name] != 0){
        // std::cout << p->node_name << std::endl;
        // std::cout << order << std::endl;
        return order;
    }
    if(p->prev_node.size() == 0){
        table[p->node_name] = 0;
        return 0;
    }else{
        size_t compare_tmp = order;
        size_t tmp = 0;
        for(const auto &i : p->prev_node){
            tmp = gen_layer_order(i, table, compare_tmp);
            if(compare_tmp < tmp)
                compare_tmp = tmp;
        }
        table[p->node_name] = tmp + 1;
        // std::cout << fmt::format("{} : {}\n", p->node_name, table[p->node_name]);
        return table[p->node_name];
    }
}

size_t weight_offset(const std::shared_ptr<model_op_node> &p, std::map<std::string, size_t> &weight_table, const size_t addr_offset){
    if((weight_table[p->weight] != 0 && p->prev_node.size() != 0)){
        const size_t Kernel_Size = 3 > p->Kernel_Size ? 3 : p->Kernel_Size;
        size_t tmp = (round(p->ICH)*round(p->OCH)*std::pow(Kernel_Size, 2)*DATA_DEPTH*DATA_DEPTH)*DATA_WIDTH;
        tmp += round(p->ICH)*round(p->OCH)*DATA_DEPTH*2*DATA_WIDTH;
        return addr_offset - tmp;
    }
    weight_table[p->weight] = 0;
    // std::cout << p->node_name << std::endl;
    if(p->prev_node.size() == 1){
        const auto &i = p->prev_node[0];
        const size_t Kernel_Size = 3 > i->Kernel_Size ? 3 : i->Kernel_Size;
        weight_table[p->weight] += (round(i->ICH)*round(i->OCH)*std::pow(Kernel_Size, 2)*DATA_DEPTH*DATA_DEPTH)*DATA_WIDTH;
        weight_table[p->weight] += round(i->ICH)*round(i->OCH)*DATA_DEPTH*2*DATA_WIDTH;
        weight_table[p->weight] += weight_offset(i, weight_table, addr_offset);
        return weight_table[p->weight];
    }
    if(p->prev_node.size() > 1){
        size_t compare_tmp = 0;
        size_t compare_offset_tmp = 0;
        const auto &i = p->prev_node[0];
        const size_t Kernel_Size = 3 > i->Kernel_Size ? 3 : i->Kernel_Size;
        weight_table[p->weight] += (round(i->ICH)*round(i->OCH)*std::pow(Kernel_Size, 2)*DATA_DEPTH*DATA_DEPTH)*DATA_WIDTH;
        weight_table[p->weight] += round(i->ICH)*round(i->OCH)*DATA_DEPTH*2*DATA_WIDTH;
        for(const auto &j : p->prev_node){
            if(compare_tmp > addr_offset)
                compare_offset_tmp = compare_tmp;
            else
                compare_offset_tmp = addr_offset;
            
            size_t tmp = weight_offset(j, weight_table, compare_offset_tmp);
            if(compare_tmp < tmp)
                compare_tmp = tmp;
        }
        weight_table[p->weight] += compare_tmp;
        return weight_table[p->weight];
    }
    return weight_table[p->weight];
}

inline size_t cal_in_tile_num(const std::shared_ptr<model_op_node> &p){
    const size_t tile_size = MAX_TILE_SIZE - p->In_Pad_Size*2; //Not include pad
    const size_t tmp = std::ceil(p->IF_Size/(double)tile_size);
    return (tmp > 0) ? tmp : 1;
}

inline size_t cal_out_tile_num(const std::shared_ptr<model_op_node> &p){
    const size_t tile_size = MAX_TILE_SIZE - p->Out_Pad_Size*2; //Not include pad
    const size_t tmp = std::ceil(p->OF_Size/(double)tile_size);
    return (tmp > 0) ? tmp : 1;
}

inline size_t cal_layer_tile_size(const std::shared_ptr<model_op_node> &p){
    const size_t tmp = p->IF_Size + p->In_Pad_Size*2;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}

inline size_t cal_next_layer_tile_size(const std::shared_ptr<model_op_node> &p){
    const size_t tmp = p->OF_Size + p->Out_Pad_Size*2;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}

inline size_t cal_out_layer_tile_size(const std::shared_ptr<model_op_node> &p){
    size_t tmp = (p->Input_Tile_Size - (p->In_Pad_Size<<1))/p->Stride;
    if(p->Have_MaxPool)
        tmp = tmp/p->Pool_Stride;
    if(p->Have_Upsample)
        tmp = tmp << 1;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}

//Generate Output Address and misc info
void gen_out_addr(tree_map &table, std::map<std::string, size_t> &order_table){
    const size_t node_num = order_table.size();
    std::vector<std::string> node_name_set;
    node_name_set.resize(node_num);

    for(auto &i : order_table){
        node_name_set[i.second] = i.first;
    }
    size_t tmp = 0;
    for(size_t i = 0; i < node_name_set.size(); i++){
        const size_t in_tile_num = cal_in_tile_num(table[node_name_set[i]]);
        const size_t out_tile_num = cal_out_tile_num(table[node_name_set[i]]);
        const size_t layer_tile_size = cal_layer_tile_size(table[node_name_set[i]]);
        table[node_name_set[i]]->Input_Tile_Size = layer_tile_size;
        const size_t out_tile_size = cal_out_layer_tile_size(table[node_name_set[i]]);
        const size_t next_tile_size = cal_next_layer_tile_size(table[node_name_set[i]]);
        const size_t layer_ich = std::ceil(table[node_name_set[i]]->ICH/(double)DATA_DEPTH)*DATA_DEPTH;
        const size_t output_size = std::pow(in_tile_num, 2)*std::pow(layer_tile_size, 2)*layer_ich*DATA_WIDTH;
        table[node_name_set[i]]->Output_Address = output_size + tmp;
        table[node_name_set[i]]->Output_Tile_Size = out_tile_size;
        table[node_name_set[i]]->Next_Tile_Size = next_tile_size;
        table[node_name_set[i]]->Input_Tile_Number = in_tile_num;
        table[node_name_set[i]]->Output_Tile_Number = out_tile_num;
        tmp = table[node_name_set[i]]->Output_Address;
    }
}

void gen_in_addr(std::shared_ptr<model_op_node> &p){
    if(p->prev_node.size() == 0){
        p->Input_Address.push_back(0);
    }else{
        if(p->Input_Address.size() == 0){
            for(auto &i : p->prev_node){
                p->Input_Address.push_back(i->Output_Address);
                gen_in_addr(i);
            }
        }
    }
}

void model_op_node::print_info() const{
    std::cout << fmt::format("Node name : {}", node_name) << std::endl;
    std::cout << fmt::format("Weight Name : {}", weight) << std::endl;
    std::cout << fmt::format("Input Feature Size : {}", IF_Size) << std::endl;
    std::cout << fmt::format("Output Feature Size : {}", OF_Size) << std::endl;
    std::cout << fmt::format("Kernel Size : {}", Kernel_Size) << std::endl;
    std::cout << fmt::format("Stride : {}", Stride) << std::endl;
    std::cout << fmt::format("Input Channel : {}", ICH) << std::endl;
    std::cout << fmt::format("Output Channel : {}", OCH) << std::endl;
    std::cout << fmt::format("Input Padding Size : {}", In_Pad_Size) << std::endl;
    std::cout << fmt::format("Output Padding Size : {}", Out_Pad_Size) << std::endl;
    std::cout << fmt::format("Have MaxPool: {}", Have_MaxPool) << std::endl;
    if(Have_MaxPool){
        std::cout << fmt::format("\tMaxPool Size: {}", Pool_Size) << std::endl;
        std::cout << fmt::format("\tMaxPool Stride: {}", Pool_Stride) << std::endl;
    }
    std::cout << fmt::format("Have ReLU: {}", Have_ReLU) << std::endl;
    std::cout << fmt::format("Is LeakyReLU: {}", Is_LeakyReLU) << std::endl;
    std::cout << fmt::format("Have BatchNormalization: {}", Have_BatchNormalization) << std::endl;
    std::cout << fmt::format("Batch First: {}", Batch_First) << std::endl;
    std::cout << fmt::format("Have Upsample: {}", Have_Upsample) << std::endl;
    std::cout << fmt::format("LeakyReLU alpha: {}", Leaky_ReLU_alpha) << std::endl;
    std::cout << fmt::format("Input Tile Size: {}", Input_Tile_Size) << std::endl;
    std::cout << fmt::format("Output Tile Size {}", Output_Tile_Size) << std::endl;
    std::cout << fmt::format("Input Tile Number: {}", Input_Tile_Number) << std::endl;
    std::cout << fmt::format("Output Tile Number: {}", Output_Tile_Number) << std::endl;
    std::cout << fmt::format("Next Tile Size: {}", Next_Tile_Size) << std::endl;
    
    if(Input_Address.size() != 0){
        std::cout << fmt::format("Input Address:");
        for(auto &i : Input_Address)
            std::cout << fmt::format(" {:08x}", i);
        std::cout << std::endl;
    }

    std::cout << fmt::format("Output Address: {:08x}", Output_Address) << std::endl;
    std::cout << fmt::format("Weight Address: {:08x}", Weight_Address) << std::endl;
    
    if(prev_node.size() != 0){
        std::cout << "Previous Node Name : ";
        for(auto i : prev_node)
            std::cout << i->node_name << " ";
        std::cout << std::endl;
    }
}