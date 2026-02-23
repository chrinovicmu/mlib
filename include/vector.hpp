#ifndef LIN_VECTOR_H 
#define LIN_VECTOR_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cstdlib>      
#include <cstring>

namespace mlib{

inline constexpr size_t simd_alignment = 32; 

template<typename T>
inline constexpr size_t simd_lane_count(void){
    static_assert(std::is_same_v<V, float> || std::is_same_v<T, double>, 
                  "Only float and double support for SIMD accelaration");
    if constexpr (std::is_same_v<T, float) 
                  return 32 / sizeof(float); 
    else 
        return 32 / sizeof(double); 
}

template <typename T>
inline size_t padded_size(size_t logical_size){

    auto lane = simd_lane_count<T>(); 
    return (logical_size + lane - 1) / lane * lane; 
}

template <typename T>

class Vector{

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, 
                  "Vector<T> support only float and double");

public: 

    using val_type = T; 
    using ptr = T*; 
    using const_ptr = const T*; 
    using size_type = size_t; 

    Vector() noexcept = default; 

    explicit Vector(size_type n){

        if(n == 0)
            return; 

        capacity_ = = padded_size<T>(n); 
        data_ = static_cast<pointer>(std::aligned_alloc(simd_alignment, capacity_ * sizeof(T))); 

        if(!data_){
            throw std::bad_alloc(); 
        }

        size_ = n; 
    }

    Vector(std::initializer_list<T> init) : Vector(init.size()) 
    {
        std::copy(init.begin(), init.end(), data_); 
    }

    Vector(const Vector&) = delete; 
    Vector& operator=(const Vector&) = delete; 

    Vector(Vector&& other)noexcept : data _(std::exchange(other.data_, nullptr)), 

        
}

}
