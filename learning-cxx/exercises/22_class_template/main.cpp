#include "../exercise.h"
#include <cstring> 
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<typename T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        std::memcpy(shape, shape_, sizeof(shape));
        for (int i = 0; i < 4; ++i) {
            size *= shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D&& other) noexcept {
       *this = std::move(other);
    }

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        // 初始化索引和步长数组
        unsigned int this_idx[4] = {0};
        unsigned int others_idx[4] = {0};
        unsigned int this_stride[4] = {1};
        unsigned int others_stride[4] = {1};

        // 计算步长
        for (int i = 3; i >= 0; --i) {
            this_stride[i] = (i == 3) ? 1 : this_stride[i + 1] * shape[i + 1];
            others_stride[i] = (i == 3) ? 1 : others_stride[i + 1] * others.shape[i + 1];
        }

        // 遍历 this 和 others 的所有元素
        while (true) {
            // 检查是否超出了 this 的边界
            bool this_done = false;
            for (int i = 0; i < 4; ++i) {
                if (this_idx[i] >= shape[i]) {
                    this_done = true;
                    break;
                }
            }
            if (this_done) break;

            // 检查是否超出了 others 的边界，并处理广播
            bool others_done = false;
            for (int i = 0; i < 4; ++i) {
                if (others_idx[i] >= others.shape[i]) {
                    if (others.shape[i] == 1) {
                        others_idx[i] = 0;// 重置索引以进行广播
                    } else {
                        others_done = true;
                        break;
                    }
                }
            }
            if (others_done) break;

            // 计算线性索引
            unsigned int this_lin_idx = 0;
            unsigned int others_lin_idx = 0;
            for (int i = 0; i < 4; ++i) {
                this_lin_idx += this_idx[i] * this_stride[i];
                others_lin_idx += others_idx[i] * others_stride[i];
            }

            // 执行加法
            data[this_lin_idx] += others.data[others_lin_idx];

            // 更新索引
            for (int i = 0; i < 4; ++i) {
                ++this_idx[i];
                if (this_idx[i] < shape[i]) break;
                this_idx[i] = 0;
            }
            for (int i = 0; i < 4; ++i) {
                ++others_idx[i];
                if (others_idx[i] < others.shape[i]) break;
                others_idx[i] = 0;
            }
        }

        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D<int>(shape, data);
        auto t1 = Tensor4D<int>(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D<float>(s0, d0);
        auto t1 = Tensor4D<float>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D<double>(s0, d0);
        auto t1 = Tensor4D<double>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
