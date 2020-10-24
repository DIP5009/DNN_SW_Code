#include "pre_image.hpp"
#define BUS_WIDTH  8
#define DATA_WIDTH 2
#define DATA_DEPTH (BUS_WIDTH/DATA_WIDTH)

void image_pre_process(const cv::Mat &src, uint16_t *dst, const uint32_t count_tile_x, const uint32_t count_tile_y, const int s){
    uint32_t src_index;
    uint32_t dst_index;
    std::vector<cv::Mat> tmp;
    tmp.resize(3);
    cv::split(src, tmp);
    cv::copyMakeBorder(tmp[0], tmp[0], 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(tmp[1], tmp[1], 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(tmp[2], tmp[2], 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    for(uint32_t ch = 0; ch < 3; ch++){
        for(uint32_t tile_y = 0; tile_y < count_tile_y; tile_y++){
            for(uint32_t tile_x = 0; tile_x < count_tile_x; tile_x++){
                for(uint32_t y = 0; y < 34; y++){
                    for(uint32_t x = 0; x < 34; x++){
                        src_index = ((tile_y*32)+y)*((count_tile_x*32)+2)+((tile_x*32)+x);
                        dst_index = (((tile_y*count_tile_x+tile_x)*34*34+y*34+x)*DATA_DEPTH)+ch;
                        dst[dst_index] = (uint16_t)tmp[2-ch].data[src_index] << s;
                        //dst[dst_index] = (uint16_t)tmp[ch].data[src_index] << s;
                    }
                }
            }
        }
    }
}

float sigmoid(float x) {
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = std::exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

void BubbleSort (int nSize , float A[] , int B[]) {
    int i , j , t_i;
    float t;

    for (i = 0 ; i< nSize-1 ;i++) {
        for (j = nSize-1 ; j>i ;j--) {
            if (A[j] > A[j-1]) {
                t        = A[j] ;
                t_i      = B[j] ;
                A[j]     = A[j-1] ;
                B[j]     = B[j-1] ;
                A[j-1]   = t;
                B[j-1]   = t_i;
                //print
                //PrintArray(nSize , A) ;
            }
        }
    }
}

float _iou(float* box1, float* box2) {
    float b1_x0 = box1[0];    float b1_y0 = box1[1];    float b1_x1 = box1[2];    float b1_y1 = box1[3];
    float b2_x0 = box2[0];    float b2_y0 = box2[1];    float b2_x1 = box2[2];    float b2_y1 = box2[3];
    
    float int_x0 = (b1_x0>b2_x0) ? b1_x0:b2_x0 ; 
    float int_y0 = (b1_y0>b2_y0) ? b1_y0:b2_y0 ; 
    float int_x1 = (b1_x1<b2_x1) ? b1_x1:b2_x1 ; 
    float int_y1 = (b1_y1<b2_y1) ? b1_y1:b2_y1 ; 

    float int_area = (int_x1 - int_x0) * (int_y1 - int_y0);
    float b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0);
    float b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0);

    // we add small epsilon of 1e-05 to avoid division by 0
    float iou = int_area / (b1_area + b2_area - int_area + 0.00001);
    
    return iou;
}

void _detection_layer0(float inputs[lyr0_grid_size][lyr0_grid_size][lyr_grid_ch], 
					   int anchors[3][2], int grid_size, float predictions[lyr0_size][lyr_ch]) { 
    const int stride = img_size/grid_size;	
    for(int m=0; m<grid_size; m++) {
        for(int n=0; n<grid_size; n++) {
            for(int x=0; x<num_anchors; x++) {
                for(int y=0; y<lyr_ch; y++) {
					switch (y) {
						//sig_box_centers
						case 0: 
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = (sigmoid(inputs[m][n][x*lyr_ch + y]) + n)*stride ;
							break;
						case 1:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = (sigmoid(inputs[m][n][x*lyr_ch + y]) + m)*stride ;
							break;
						//exp_box_sizes
						case 2: case 3:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = ( exp(inputs[m][n][x*lyr_ch +y ])*anchors[x][y-2] );
							break;
						//sig_confidence
						case 4:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = sigmoid(inputs[m][n][x*lyr_ch + y]);
							break;
						//sig_classes
						default:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = sigmoid(inputs[m][n][x*lyr_ch + y]);
					}
                }
            }
        }
    }
}

void _detection_layer1(float inputs[lyr1_grid_size][lyr1_grid_size][lyr_grid_ch],
                       int anchors[3][2], int grid_size, float predictions[lyr1_size][lyr_ch] ) {

    const int stride = img_size/grid_size;
		
    for(int m=0; m<grid_size; m++){
        for(int n=0; n<grid_size; n++){
            for(int x=0; x<num_anchors; x++){
                for(int y=0; y<lyr_ch; y++){
					switch (y)
					{
						//sig_box_centers
						case 0:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = (sigmoid(inputs[m][n][x*lyr_ch + y]) + n)*stride ;
							break;
						case 1:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = (sigmoid(inputs[m][n][x*lyr_ch + y]) + m)*stride ;
							break;							
						//exp_box_sizes
						case 2: case 3:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = ( exp(inputs[m][n][x*lyr_ch +y ])*anchors[x][y-2] );
							break;
						//sig_confidence
						case 4:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = sigmoid(inputs[m][n][x*lyr_ch + y]);
							break;
						//sig_classes
						default:
							predictions[ m*grid_size*num_anchors + n*num_anchors + x ][y]  = sigmoid(inputs[m][n][x*lyr_ch + y]);
					}
                }
            }
        }
    }
}

/*=======non_max_suppression===========================================================*/
int non_max_suppression(float detected_boxes[][lyr_ch], 
						float confidence_threshold, float iou_threshold, float out_filtered_boxes[][5]) {
    /*
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 2D  array, first 4 values in 2rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]          // = [n][(box, score)] = size(n,5)
    */

    //mask for confidence >  confidence_threshold
    float predictions_with_mask [lyr_total_size][lyr_ch]   = {0};
    int   predictions_cls_idx[num_classes][lyr_total_size] = {0};
    float predictions_cls_cof[num_classes][lyr_total_size] = {0};
	int   cls_cnt[num_classes]				  			   = {0};
	
	float max_value;
	int   max_idx;
    int cnt = 0;//nonzero count

    for(int i=0; i<lyr_total_size; i++)
	{
        if((detected_boxes[i][4]) > confidence_threshold)
		{
            for(int j=0; j<lyr_ch; j++)
			{
                predictions_with_mask[cnt][j] = detected_boxes[i][j];               
            }           
            cnt++;        
        }
    }
    //cout << "BBox cnt : "<< cnt << endl;


    for(int i=0; i<cnt; i++) {    
		//find the max index of class
		max_value = predictions_with_mask[i][5];
		max_idx   = 5;
		for(int j=5; j<5+num_classes;++j) {
			if(predictions_with_mask[i][j]>max_value) {
				max_value = predictions_with_mask[i][j];
				max_idx   = j;
			}
		}
	    
		max_idx -= 5;
		
		predictions_cls_idx[max_idx][cls_cnt[max_idx]] =  i;          				// index
        predictions_cls_cof[max_idx][cls_cnt[max_idx]] =  predictions_with_mask[i][4];  // conficidence      
        ++cls_cnt[max_idx];	
    }
  
	for(int i=0;i<num_classes;++i)
		BubbleSort(cls_cnt[i] , predictions_cls_cof[i] , predictions_cls_idx[i]);
    
    int   cls_best_idx ;
    //float cls_best_cof ;
    int   larger_than_iou, little_than_iou ;
    float predictions_best_bbox[4] = {0};
    float predictions_other_bbox[4] = {0};
    float IOU;

    int result_cnt 				   = 0 ;
    int result_idx[lyr_total_size] = {0} ;

	for(int j=0;j<num_classes;++j) {	
		while(cls_cnt[j]>0) {
			cls_best_idx = predictions_cls_idx[j][0];
			//cls_best_cof = predictions_cls_cof[j][0];
			
			larger_than_iou = 0;
			little_than_iou = 0;
			
			if(cls_cnt[j]!=1) {
				for(int i=1; i<cls_cnt[j]; i++) {
					predictions_best_bbox[0] = predictions_with_mask[cls_best_idx][0];
					predictions_best_bbox[1] = predictions_with_mask[cls_best_idx][1];
					predictions_best_bbox[2] = predictions_with_mask[cls_best_idx][2];
					predictions_best_bbox[3] = predictions_with_mask[cls_best_idx][3];
					predictions_other_bbox[0] = predictions_with_mask[predictions_cls_idx[j][i]][0];
					predictions_other_bbox[1] = predictions_with_mask[predictions_cls_idx[j][i]][1];
					predictions_other_bbox[2] = predictions_with_mask[predictions_cls_idx[j][i]][2];
					predictions_other_bbox[3] = predictions_with_mask[predictions_cls_idx[j][i]][3];

					IOU = _iou(predictions_best_bbox, predictions_other_bbox);

					if(IOU < iou_threshold) {
						predictions_cls_cof[j][little_than_iou] = predictions_cls_cof[j][i];
						predictions_cls_idx[j][little_than_iou] = predictions_cls_idx[j][i];
						little_than_iou++;
					}
					else {
						larger_than_iou++;
					}

				}
			}
			cls_cnt[j] -= larger_than_iou;  // take out the high IOU bbox
			cls_cnt[j] --; //take the highest confidence into result buffer
			
			result_idx[result_cnt] = cls_best_idx;
			result_cnt ++;
		}
	}
    
    for(int i=0; i<result_cnt; i++) {		
		//find the max index of class
		max_value = predictions_with_mask[i][5];
		max_idx   = 5;
		for(int j=5; j<5+num_classes;++j) {
			if(predictions_with_mask[i][j]>max_value) {
				max_value = predictions_with_mask[i][j];
				max_idx   = j;
			}
		}
		
		max_idx -= 5 ;
		out_filtered_boxes[i][4] = max_idx; 

        //TODO: The fifth element of array uses to index which class
        for(int j=0; j<6; j++) {
        //for(int j=0; j<4; j++) {
            out_filtered_boxes[i][j] = predictions_with_mask[result_idx[i]][j];   
        }
    }
    return result_cnt;
}

void _detections_boxes(float detections[lyr_total_size][lyr_ch]) {
  float center_x, center_y, width_div2, height_div2;

  for (int i = 0 ; i< lyr_total_size ;i++) {
        center_x = detections[i][0];
        center_y = detections[i][1];
        width_div2  = detections[i][2] / 2;
        height_div2 = detections[i][3] / 2;      
        
        detections[i][0] = center_x - width_div2 ;
        detections[i][1] = center_y - height_div2;
        detections[i][2] = center_x + width_div2 ;
        detections[i][3] = center_y + height_div2 ;
  }  
}

void localization(cv::Mat img, float inputs_arr0[lyr0_grid_size][lyr0_grid_size][lyr_grid_ch], 
	float inputs_arr1[lyr1_grid_size][lyr1_grid_size][lyr_grid_ch] , int conf, int iou) {

    float conf_threshold = conf / 100.0;
    float iou_threshold = iou / 100.0;
	//string category[num_classes] = {"shrimp" ,"food"};
	std::string category[num_classes] = {"shrimp"};
	//int anchors[2][3][2] = { {{101,160},{170,134},{156,210}} , {{79,51},{64,121},{133,84}} };
	//!!!!Anchor box need to swap.
	int anchors[2][3][2] = { {{49, 50},{83, 104},{211, 196}}, {{6, 8},{14, 16},{22, 35}} };
	
    /*=======detection_layer for 16*16*21/32*32*21 feature=====================================*/
    float predictions_0[lyr0_size][lyr_ch] = {0};
    float predictions_1[lyr1_size][lyr_ch] = {0};
	
    _detection_layer0(inputs_arr0 , anchors[0], lyr0_grid_size, predictions_0);
    _detection_layer1(inputs_arr1 , anchors[1], lyr1_grid_size, predictions_1);

    /*=======NMS=====================================*/

    //concate the feature (768*7) + (3072*7)  -> (3840*7)
    float detected_boxes[lyr_total_size][lyr_ch] = {0};
 
    for(int i=0; i<lyr_total_size; i++) {
        for(int j=0; j<lyr_ch; j++) {
            if(i<lyr0_size)
				detected_boxes[i][j] = predictions_0[i][j];
            else        
				detected_boxes[i][j] = predictions_1[i-lyr0_size][j];
			//cout << detected_boxes[i][j] <<" ";
        }
		//cout<<endl;
	}

    //find the Actual coordinates
    _detections_boxes(detected_boxes);

    /*=======draw the box=====================================*/
    float out_filtered_boxes[lyr_total_size][5];
	
    int box_number = non_max_suppression(detected_boxes, conf_threshold, iou_threshold, out_filtered_boxes);

    // std::cout << "Box number : " << box_number << "\n";

	// for(int i=0; i<*box_number; ++i) {
	// 	YoloData_tmp.x = (out_filtered_boxes[i][0]) / img_size;
	// 	YoloData_tmp.y = (out_filtered_boxes[i][1]) / img_size;
	// 	YoloData_tmp.w = (out_filtered_boxes[i][2] - out_filtered_boxes[i][0]) / img_size;
	// 	YoloData_tmp.h = (out_filtered_boxes[i][3] - out_filtered_boxes[i][1]) / img_size;
	// 	YoloData_tmp.name = "shrimp";
	// 	YoloData_tmp.prob = out_filtered_boxes[i][4];
	// }
    
    cv::Scalar color[2] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0)};

    for(int m=0; m<box_number; m++)
    {
        for(int n=0; n<num_classes; n++)
        {  
			// if(out_filtered_boxes[m][4] > iou/100.0)
			// {
			// 	cv::rectangle(img, cv::Point(out_filtered_boxes[m][0],out_filtered_boxes[m][1]), cv::Point(out_filtered_boxes[m][2],out_filtered_boxes[m][3]), color[n], 2);
			// 	cv::putText(img, category[n], cv::Point(out_filtered_boxes[m][0],out_filtered_boxes[m][1]), 3, 0.7, color[n], 2);
			// 	break;
			// }
            cv::rectangle(img, cv::Point(out_filtered_boxes[m][0],out_filtered_boxes[m][1]), cv::Point(out_filtered_boxes[m][2],out_filtered_boxes[m][3]), color[n], 1);
            cv::putText(img, category[n], cv::Point(out_filtered_boxes[m][0],out_filtered_boxes[m][1]), 3, 0.5, color[n], 1);
            break;
        }
    }
	
    cv::resize(img, img, cv::Size(512, 512));
    //imwrite("output.jpg",img); 
	
    cv::imshow("window", img);
    
}
