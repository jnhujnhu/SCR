#include "global_def.hpp"

using namespace global_def;

void Tuple::clean_up() {
   delete _w;
   delete _b;
}

void Batch::clean_up() {
    delete _X;
    delete _Y;
}

void Tuple::print_all() {
    std::cout << "Weights(" << _w->rows() <<"x"<< _w->cols() <<"):\n";
    std::cout << (*_w) << std::endl;
    std::cout << "Biases(" << _b->rows() <<"x1):\n";
    std::cout << (*_b) << std::endl;
}

Tuple& Tuple::operator +=(const Tuple& rhs) {
    if(this->_w->cols() != rhs._w->cols()
    || this->_w->rows() != rhs._w->rows()
    || this->_b->cols() != rhs._b->cols()
    || this->_b->rows() != rhs._b->rows()) {
        throw std::string("Unaligned tuple sizes for add.");
    }
    *(this->_w) += *rhs._w;
    *(this->_b) += *rhs._b;
    return *this;
}

Tuple& Tuple::operator -=(const Tuple& rhs) {
    if(this->_w->cols() != rhs._w->cols()
    || this->_w->rows() != rhs._w->rows()
    || this->_b->cols() != rhs._b->cols()
    || this->_b->rows() != rhs._b->rows()) {
        throw std::string("Unaligned tuple sizes for add.");
    }
    *(this->_w) -= *rhs._w;
    *(this->_b) -= *rhs._b;
    return *this;
}

Tuple& Tuple::operator *=(const double rhs) {
    *(this->_w) *= rhs;
    *(this->_b) *= rhs;
    return *this;
}

double Tuple::l2_norm_square() {
    return (*_w).squaredNorm() + (*_b).squaredNorm();
}
