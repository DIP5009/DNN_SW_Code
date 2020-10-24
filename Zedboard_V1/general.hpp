#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>

off_t fsize(const char *filename);