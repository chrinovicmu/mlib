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

    Vector(Vector&& other)noexcept 
        : data _(std::exchange(other.data_, nullptr)), 
        size_(std::exchange(other.size_,0)), 
        capacity_(std::exchange(other.capacity_,0)) 
    {

    }

    Vector& operator=(Vector&& other) noexcept{
        if(this != &other){
            destroy(); 
            data_ = std::exchange(other.data_, nullptr); 
            size_ = std::exchange(other.size_, 0); 
            capacity_ = std::exchange(other.capacity_, 0); 
        }

        return *this; 
    }

    ~Vector(){
        destroy(); 
    }

    size_type size() const noexcept {return size_;}
    size_type capacity() const noexcept {return capacity_;} 
    bool empty() const noexcept {return size_ == 0;}

    ptr data() noexcept {return data_;}
    const_ptr data()  const noexcept{return data_;} 

    ptr aligned_data() noexcept {return data_;} 
    const_ptr aligned_data() noexcept {return data_;}

    bool is_aligned() const noexcept{
        return (size == 0) || (reinterpret_cast<uintptr_t>(data) % simd_alignment == 0); 
    }

    T& operator[](size_type i) noexcept{
        return data_[i]; 
    }

    const T& operator[](size_type i) const noexcept{
        return data_[i]; 
    }

    void resize(size_type new_size){
        if(new_size == size_)
            return; 

        size_type new_capacity = padded_size<T>(new_size); 

        if(new_capacity <= capacity_){
            size_ = new_size; 
            return; 
        }

        ptr new_data = static_cast(ptr)(
            std::aligned_alloc(simd_alignment, new_capacity * sizeof(T))); 

        if(!new_data){
            throw std::bad_alloc(); 
        }

        if(data_)
        {
            std::memcpy(new_data, data_, size_ * sizeof(T)); 
            std::free(data_); 
        }
        data_ = new_data; 
        size_ = new_size; 
        capacity_ = new_capacity; 
    }

    void refill(T value){
        for(size_type = 0; i < size_; ++i){
            data_[i] = value;
        }
    }

    void clear() noexcept{
        size_ = 0; 
    }

private:

    void destroy() noexcept{
        if(data_){
            std::free(data_); 
            data_ = nullptr; 
        }
        size_ = 0; 
        capacity_ = 0; 
    }

    ptr data_ = nullptr; 
    size_type size_ = 0; 
    size_type capacity_ = 0; 
}; 

}
#endif 
