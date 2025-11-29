// In GIST/cgp-plusplus/problems/HollandRoyalRoadProblem.h

#ifndef PROBLEMS_HOLLANDROYALROADPROBLEM_H_
#define PROBLEMS_HOLLANDROYALROADPROBLEM_H_

#include "../problems/BlackBoxProblem.h"
#include <vector>
#include <memory>
#include <cmath>
#include <numeric>

template<class E, class G, class F>
class HollandRoyalRoadProblem : public BlackBoxProblem<E, G, F> {
private:
    // Default parameters of the HRR
    const int k = 4;
    const int b = 8;
    const int g = 7;
    const int m_star = 4;
    const double v = 0.02;
    const double u_star = 1.0;
    const double u = 0.3;

    const int num_regions = 16; // 2^k
    const int region_length = 15; // b+g
    const int string_length = 240; // 2^k * (b+g)

    // Helper function for calculating PART
    double calculate_part_fitness(const std::shared_ptr<std::vector<E>>& bit_string) {
        double total_part_fitness = 0.0;
        
        for (int i = 0; i < num_regions; i++) {
            int block_start = i * region_length;
            int ones_in_block = 0;

            // Calculate "ones" only in the block (length b), ignoring the gap
            for (int j = 0; j < b; j++) {
                if (bit_string->at(block_start + j) != 0) {
                    ones_in_block++;
                }
            }

            // Apply PART fitness rules
            switch (ones_in_block) {
                case 0: total_part_fitness += 0.00; break;
                case 1: total_part_fitness += 0.02; break;
                case 2: total_part_fitness += 0.04; break;
                case 3: total_part_fitness += 0.06; break;
                case 4: total_part_fitness += 0.08; break;
                case 5: total_part_fitness -= 0.02; break;
                case 6: total_part_fitness -= 0.04; break;
                case 7: total_part_fitness -= 0.06; break;
                case 8: total_part_fitness += 0.00; break; // 'b' ones, handled by BONUS
                default: break; // Should never happen
            }
        }
        return total_part_fitness;
    }

    // Helper function for calculating BONUS
    double calculate_bonus_fitness(const std::shared_ptr<std::vector<E>>& bit_string) {
        double total_bonus_fitness = 0.0;
        
        std::vector<bool> complete_blocks(num_regions, false);
        
        // 1. Find all complete blocks
        for (int i = 0; i < num_regions; i++) {
            int block_start = i * region_length;
            bool is_complete = true;
            for (int j = 0; j < b; j++) {
                if (bit_string->at(block_start + j) == 0) {
                    is_complete = false;
                    break;
                }
            }
            if (is_complete) {
                complete_blocks[i] = true;
            }
        }

        // 2. Calculate bonus for all levels, from 0 to k
        for (int level = 0; level <= k; level++) {
            int sets_found = 0;
            int set_size = 1 << level; // 2^level
            int num_sets_at_level = num_regions / set_size; // 2^(k-l)

            for (int i = 0; i < num_sets_at_level; i++) {
                // Check if the set B_{i*2^l} is complete 
                bool set_is_complete = true;
                int start_block = i * set_size;
                int end_block = start_block + set_size;

                for (int block_idx = start_block; block_idx < end_block; block_idx++) {
                    if (!complete_blocks[block_idx]) {
                        set_is_complete = false;
                        break;
                    }
                }

                if (set_is_complete) {
                    if (sets_found == 0) {
                        total_bonus_fitness += u_star; // First set found
                    } else {
                        total_bonus_fitness += u; // Subsequent sets
                    }
                    sets_found++;
                }
            }
        }
        
        return total_bonus_fitness;
    }

public:
    // Constructor
    HollandRoyalRoadProblem(std::shared_ptr<Parameters> p_parameters,
                            std::shared_ptr<Evaluator<E, G, F>> p_evaluator,
                            std::shared_ptr<std::vector<std::vector<E>>> p_inputs,
                            std::shared_ptr<std::vector<std::vector<E>>> p_outputs,
                            std::shared_ptr<std::vector<E>> p_constants,
                            int p_num_instances)
        : BlackBoxProblem<E, G, F>(p_parameters, p_evaluator, p_inputs, p_outputs, p_constants, p_num_instances) {
        this->name = "Holland's Royal Road Problem";
        
        static_assert(std::is_floating_point<F>::value, "Fitness type F must be floating point for HRRProblem");
    }

    // Destructor
    ~HollandRoyalRoadProblem() = default;

    F evaluate(std::shared_ptr<std::vector<E>> outputs_real,
               std::shared_ptr<std::vector<E>> outputs_individual) override {
        
        // 'outputs_real' is ignored. There is no "objective"
        // 'outputs_individual' is our phenotype, the 240-bit string.
        
        if (outputs_individual->size() != string_length) {
            // Throw an exception if the CGP is not producing 240 outputs
            throw std::length_error("HRRProblem expects a 240-bit string from the individual.");
        }

        // Calculate total fitness as PART + BONUS 
        double part_score = calculate_part_fitness(outputs_individual);
        double bonus_score = calculate_bonus_fitness(outputs_individual);
        F total_fitness = static_cast<F>(part_score + bonus_score);
        
        float max_rr = 12.8; // Max theoretical fitness

        return max_rr - total_fitness;
    }

    // Implement the virtual clone function
    HollandRoyalRoadProblem<E, G, F>* clone() override {
        return new HollandRoyalRoadProblem<E, G, F>(*this);
    }
};

#endif /* PROBLEMS_HOLLANDROYALROADPROBLEM_H_ */