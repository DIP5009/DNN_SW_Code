#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#define img_size        256
#define num_anchors		3
#define num_classes     1
#define lyr_ch          (5 + num_classes)
#define lyr0_grid_size  8
#define lyr1_grid_size  16
#define lyr_grid_ch     (lyr_ch * num_anchors)
#define lyr0_size       (lyr0_grid_size * lyr0_grid_size * num_anchors)
#define lyr1_size       (lyr1_grid_size * lyr1_grid_size * num_anchors)
#define lyr_total_size  (lyr0_size + lyr1_size)

void image_pre_process(const cv::Mat &src, uint16_t *dst, const uint32_t count_tile_x, const uint32_t count_tile_y, const int s);
void localization(cv::Mat img, float inputs_arr0[lyr0_grid_size][lyr0_grid_size][lyr_grid_ch], float inputs_arr1[lyr1_grid_size][lyr1_grid_size][lyr_grid_ch] , int conf, int iou);