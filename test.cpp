#include <iostream>
#include <vector> // 额外测试 C++ 标准库其他头文件
int main() {
    std::vector<int> vec = {1, 2, 3};
    std::cout << "GCC 15 编译成功！" << std::endl;
    std::cout << "vector 大小：" << vec.size() << std::endl;
    return 0;
}