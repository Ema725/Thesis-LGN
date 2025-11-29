//	CGP++: Modern C++ Implementation of Cartesian Genetic Programming
// ===============================================================================
//	File BooleanFunctions.h 
// ===============================================================================
//
// ===============================================================================
//  Copyright (C) 2024
//
//
//	License: Academic Free License v. 3.0
// ================================================================================

#ifndef FUNCTIONS_BOOLEANFUNCTIONS_H_
#define FUNCTIONS_BOOLEANFUNCTIONS_H_

#include "Functions.h"
#include "../parameters/Parameters.h"

#include <string>
#include <sstream>

/// @brief Class to represent a function set of Boolean functions 
/// @details Ensures that only datatypes such as long, unsigned int and unsigned long 
/// that fit the domain of the considered Boolean functions are used with this class
/// @tparam E Evaluation type 
template<class E>
class FunctionsBoolean: public Functions<E> {
public:
	FunctionsBoolean(std::shared_ptr<Parameters> p_parameters);
	virtual ~FunctionsBoolean() = default;

	E call_function(E inputs[], int function) override;
	std::string input_name(int input) override;
	std::string function_name(int function) override;

	int arity_of(int function) override;

};

template<class E>
FunctionsBoolean<E>::FunctionsBoolean(std::shared_ptr<Parameters> p_parameters) :
		Functions<E>(p_parameters) {

	if constexpr (!std::is_same<int, E>::value) {
		if constexpr (!std::is_same<long, E>::value) {
			if constexpr (!std::is_same<unsigned int, E>::value) {
				if constexpr (!std::is_same<unsigned long, E>::value) {
					throw std::invalid_argument(
							"This class only supports (signed/unsigned) long and int!");
				}
			}
		}
	}

}

/// @brief Provides a Boolean functions such as AND, OR, NAND, NOR.
/// @details Configuration is commonly used for a reduced function set in logic synthesis 
/// as approached with genetic programming. 
/// @param inputs pair of inputs values
/// @param function index of the functions to call 
/// @return result of the function call
template<class E>
E FunctionsBoolean<E>::call_function(E inputs[], int function) {

	E result;

	switch (function) {

	case 0:
		result = inputs[0] & inputs[1];
		break;

	case 1:
		result = inputs[0] | inputs[1];
		break;

	case 2:
		result = ~(inputs[0] & inputs[1]);
		break;

	case 3:
		result = ~(inputs[0] | inputs[1]);
		break;

	case 4:
		result = inputs[0];
		break;
	
	case 5:
		result = ~inputs[0];
		break;

	case 6:
		// XOR
		result = inputs[0] ^ inputs[1];
		break;
	
	case 7:
		// XNOR
		result = ~(inputs[0] ^ inputs[1]);
		break;

	default:
		throw std::invalid_argument("Illegal function number!");

	}

	return result;

}


template<class E>
std::string FunctionsBoolean<E>::FunctionsBoolean::function_name(int function) {

	std::string name = "";

	switch (function) {

	case 0:
		name = "AND";
		break;

	case 1:
		name = "OR";
		break;

	case 2:
		name = "NAND";
		break;

	case 3:
		name = "NOR";
		break;

	case 4:
		name = "BUF";
		break;

	case 5:
		name = "NOT";
		break;

	case 6:
		name = "XOR";
		break;

	case 7:
		name = "XNOR";
		break;

	default:
		throw std::invalid_argument("Illegal function number!");

	}

	return name;
}

template<class E>
std::string FunctionsBoolean<E>::input_name(int input) {
	std::string input_name = "x" + std::to_string(input);
	return input_name;
}

template<class E>
int FunctionsBoolean<E>::arity_of(int function) {
	switch (function) {
        case 0: // AND
        case 1: // OR
        case 2: // NAND
        case 3: // NOR
        case 6: // XOR
        case 7: // XNOR
            return 2;
        
        case 4: // BUF
        case 5: // NOT
            return 1;
            
        default:
            throw std::invalid_argument("Illegal function number in arity_of!");
    }
}

#endif /* FUNCTIONS_BOOLEANFUNCTIONS_H_ */
