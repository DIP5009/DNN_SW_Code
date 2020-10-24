#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

std::mutex img_pre_process_m, HW_m;
std::condition_variable prev_img_cond_v, HW_cond_v;
std::string data;
bool HW_L2_processed = true;
bool img_processed = false;
bool processed = false;
bool ready = false;
const std::string file_name = "aaaa";
const int fd = 10;

void read_image(std::unique_lock<std::mutex> lk){
    bool first = true;
    while (true)
    {
        if(first)
            first = false;
        else
            lk.lock();
        prev_img_cond_v.wait(lk, []{return HW_L2_processed;});
        std::cout << "Open " << file_name << std::endl;
        img_processed = false;
        std::cout << "Layer " << 0 << std::endl;
        img_processed = true;
        HW_L2_processed = false;
        lk.unlock();
        prev_img_cond_v.notify_one();
    }
}

void HW(){
    while (true)
    {
        std::unique_lock<std::mutex> lk(img_pre_process_m);
        std::unique_lock<std::mutex> HW_lk(HW_m);
        prev_img_cond_v.wait(lk, []{return img_processed;});
        std::cout << fd << std::endl;
        std::cout << "Layer " << 1 << std::endl;
        HW_L2_processed = true;
        lk.unlock();
        prev_img_cond_v.notify_one();
        for(int i = 2; i < 13 ; i++){
            std::cout << "Layer " << i << std::endl;
        }
        processed = true;
        HW_lk.unlock();
        HW_cond_v.notify_one();
    }
}

void write_image(){
    while(true){
        std::unique_lock<std::mutex> lk(HW_m);
        HW_cond_v.wait(lk, []{return processed;});
        processed = false;
        std::cout << "Finish" << std::endl;
        lk.unlock();
    }
}
/*
void worker_thread()
{
    // Wait until main() sends data
    std::unique_lock<std::mutex> lk(m);
    cond_v.wait(lk, []{return ready;});
 
    // after the wait, we own the lock.
    std::cout << "Worker thread is processing data\n";
    data += " after processing";
 
    // Send data back to main()
    processed = true;
    std::cout << "Worker thread signals data processing completed\n";
 
    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lk.unlock();
    cond_v.notify_one();
}
*/
 
int main(int argc, char **argv)
{
    
    std::unique_lock<std::mutex> lk(img_pre_process_m);
    prev_img_cond_v.notify_one();
    std::thread worker([&](){read_image(std::move(lk));});
    std::thread worker2(HW);
    std::thread worker3(write_image);
    
    data = "Example data";/*
    // send data to the worker thread
    {
        std::lock_guard<std::mutex> lk(m);
        ready = true;
        std::cout << "main() signals data ready for processing\n";
    }
    cond_v.notify_one();
 
    // wait for the worker
    {
        std::unique_lock<std::mutex> lk(m);
        cond_v.wait(lk, []{return processed;});
    }
    std::cout << "Back in main(), data = " << data << '\n';
    */
   
    worker.join();
    worker2.join();
    worker3.join();

}