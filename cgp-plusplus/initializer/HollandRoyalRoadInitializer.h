// In GIST/cgp-plusplus/initializer/HollandRoyalRoadInitializer.h

#ifndef INITIALIZER_HOLLANDROYALROADINITIALIZER_H_
#define INITIALIZER_HOLLANDROYALROADINITIALIZER_H_

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

#include "BlackBoxInitializer.h"
#include "../problems/HollandRoyalRoadProblem.h" 
#include "../functions/BooleanFunctions.h"     

template<class E, class G, class F>
class HollandRoyalRoadInitializer : public BlackBoxInitializer<E, G, F> {

public:
    HollandRoyalRoadInitializer(const std::string &p_benchmark_file)
        : BlackBoxInitializer<E, G, F>(p_benchmark_file) {
        // p_benchmark_file is ignored for HRR, but the base constructor requires it
    }

    ~HollandRoyalRoadInitializer() = default;

    /**
     * @brief Overrides read_data to manually set HRR parameters.
     * HRR does not read a data file, so we set here 
     * the number of instances, inputs, and outputs.
     */
    void read_data() override {
        
        // 1. Set fixed parameters of HRR
        this->num_instances = 1; // We run the network only once to get the string
        this->parameters->set_num_variables(2);  // 2 input
        this->parameters->set_num_outputs(240); // The HRR base requires 240 outputs

        // 2. Create dummy input/output vectors
        // The BlackBoxProblem constructor expects them, 
        // even though HollandRoyalRoadProblem will ignore them.
        this->inputs = std::make_shared<std::vector<std::vector<E>>>(this->num_instances);
        this->outputs = std::make_shared<std::vector<std::vector<E>>>(this->num_instances);

        (*this->inputs)[0].resize(this->parameters->get_num_variables());
        (*this->outputs)[0].resize(this->parameters->get_num_outputs());
        
        // initialize with 1 and 0
        (*this->inputs)[0][0] = static_cast<E>(0);
        (*this->inputs)[0][1] = static_cast<E>(1);

        std::fill((*this->outputs)[0].begin(), (*this->outputs)[0].end(), static_cast<E>(0));
    }

    /**
     * @brief Initializes the set of functions (boolean logic).
     * The network must produce bits (0/1), so boolean functions are suitable.
     */
    void init_functions() override {
        // We assume EVALUATION_TYPE (E) is unsigned int for 0/1
        this->functions = std::make_shared<FunctionsBoolean<E>>(this->parameters);
    }

    /**
     * @brief Initializes the HRR problem instance.
     */
    void init_problem() override {
        this->problem = std::make_shared<HollandRoyalRoadProblem<E, G, F>>(
            this->parameters, 
            this->evaluator, 
            this->inputs,
            this->outputs,     
            this->constants, 
            this->num_instances
        );
        
        // Ensure evolution is a MAXIMIZATION. (for now doesn't work)
        this->parameters->set_minimizing_fitness(false);

        // Set the ideal (maximum) fitness if not provided
        if (this->parameters->get_ideal_fitness() == -1) {
             this->parameters->set_ideal_fitness(12.8);
        }

        this->composite->set_problem(this->problem);
    }
};

#endif /* INITIALIZER_HOLLANDROYALROADINITIALIZER_H_ */