#ifndef PROBLEMS_MNISTLOGICPROBLEM_H_
#define PROBLEMS_MNISTLOGICPROBLEM_H_

#include "BlackBoxProblem.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

template<class E, class G, class F>
class MnistLogicProblem : public BlackBoxProblem<E, G, F> {
private:
    int bits_per_class;
    const int NUM_CLASSES = 10;

    std::vector<std::vector<E>> full_inputs;
    std::vector<std::vector<E>> full_outputs;

public:
    MnistLogicProblem(std::shared_ptr<Parameters> p_parameters,
                      std::shared_ptr<Evaluator<E, G, F>> p_evaluator,
                      std::shared_ptr<std::vector<std::vector<E>>> p_inputs,
                      std::shared_ptr<std::vector<std::vector<E>>> p_outputs,
                      std::shared_ptr<std::vector<E>> p_constants,
                      int p_num_instances)
        : BlackBoxProblem<E, G, F>(p_parameters, p_evaluator, p_inputs, p_outputs, p_constants, p_num_instances) {
        
        this->name = "MNIST Logic Problem";
        
        // Es: 500 output / 10 classes = 50 bit per class
        if (this->parameters->get_num_outputs() % NUM_CLASSES != 0) {
            throw std::invalid_argument("Total outputs must be a multiple of 10 (classes)!");
        }
        this->bits_per_class = this->parameters->get_num_outputs() / NUM_CLASSES;
    }

    // Upload full dataset
    void set_full_dataset(const std::vector<std::vector<E>>& all_inputs, 
                          const std::vector<std::vector<E>>& all_outputs) {
        this->full_inputs = all_inputs;
        this->full_outputs = all_outputs;
    }

    // change batch
    void load_batch(int start_index, int batch_size) {
        // Verifies limits
        if (start_index + batch_size > (int)this->full_inputs.size()) {
            start_index = 0; 
        }

        // updates active vectors
        this->inputs->clear();
        this->outputs->clear();
        
        // Copy the window
        for(int i = 0; i < batch_size; i++) {
            this->inputs->push_back(this->full_inputs[start_index + i]);
            this->outputs->push_back(this->full_outputs[start_index + i]);
        }
        
        this->num_instances = batch_size;
    }

    ~MnistLogicProblem() = default;
    
    /**
     * @brief Calculates the exact number of correctly classified images.
     * This method is separate from the fitness and is only used for reporting.
     */
    int validate_individual(std::shared_ptr<Individual<G, F>> individual) override {
        int hits = 0;
        
        // Temporary vectors for the individual's output
        std::shared_ptr<std::vector<E>> outputs_ind = std::make_shared<std::vector<E>>();
        
        // Loop over all images (instances)
        for (int i = 0; i < this->num_instances; i++) {
            
            // 1. Run the network on the current image
            // Note: we need to reconstruct the full input for the evaluator
            std::shared_ptr<std::vector<E>> input_instance = std::make_shared<std::vector<E>>(this->inputs->at(i));
            
            // If there are constants (in your config they are 0, but we keep for compatibility)
            if (this->constants != nullptr && !this->constants->empty()) {
                input_instance->insert(std::end(*input_instance), std::begin(*this->constants), std::end(*this->constants));
            }

            outputs_ind->clear();
            
            //evaluate
            this->evaluator->evaluate_iterative(individual, input_instance, outputs_ind);

            // Classification logic (Bit Counting)
            int true_label = static_cast<int>(this->outputs->at(i)[0]);
            int best_class = -1;
            int max_bits_on = -1;

            for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
                int current_bits_on = 0;
                int start_idx = class_idx * this->bits_per_class;

                for (int bit = 0; bit < this->bits_per_class; ++bit) {
                    if (outputs_ind->at(start_idx + bit) != 0) {
                        current_bits_on++;
                    }
                }

                if (current_bits_on > max_bits_on) {
                    max_bits_on = current_bits_on;
                    best_class = class_idx;
                }
            }

            if (best_class == true_label) {
                hits++;
            }
        }
        return hits;
    }

    /**
     * @brief Evaluates a single prediction.
     * @param outputs_real Contains the true LABEL as the only element (see Initializer).
     * @param outputs_individual Contains the bit string produced by the network (e.g., 500 bits).
     * @return 0.0 if the prediction is correct, 1.0 if it is incorrect (Minimization).
     */
    F evaluate(std::shared_ptr<std::vector<E>> outputs_real,
               std::shared_ptr<std::vector<E>> outputs_individual) override {

        // 1. Retrieve the true label (Saved in element 0 by our Initializer)
        int true_label = static_cast<int>(outputs_real->at(0));

        // 2. Classification logic (Bit Counting)
        int best_class = -1;
        int max_bits_on = -1;
        int prediction_strength = 0;
        int max_incorrect_bits = 0;
        int sum_incorrect_bits = 0;


        // Iterates for each class (0-9)
        for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
            int current_bits_on = 0;
            int start_idx = class_idx * this->bits_per_class;

            // Count the '1' bits in the block dedicated to this class
            for (int bit = 0; bit < this->bits_per_class; ++bit) {
                // Consider any non-zero value as a logical '1' (for safety)
                if (outputs_individual->at(start_idx + bit) != 0) {
                    current_bits_on++;
                }
            }

            // Note: if there is a tie, the class with the smaller index wins.
            if (current_bits_on > max_bits_on) {
                max_bits_on = current_bits_on;
                best_class = class_idx;
            }
            if (class_idx == true_label) {
                prediction_strength = current_bits_on;
            } else {
                // Statistics on incorrect classes
                sum_incorrect_bits += current_bits_on;
                if (current_bits_on > max_incorrect_bits) {
                    max_incorrect_bits = current_bits_on;
                }
            }
        }

        // the more bits on for the correct class, the better
        int func_type = this->parameters->get_fitness_function();

        switch (func_type) {
            case 0: //minfitness = -50 * nclasses
                if (best_class == true_label) {
                    return 0.0 - prediction_strength;
                } else {
                    return 50.0 + (max_bits_on - prediction_strength);
                }

            case 1: //minfitness = 0 maybe better case 0
                if (best_class == true_label) {
                    return 0.0;
                } else {
                    return 50.0 + (max_bits_on - prediction_strength);
                }

            case 2: //minfitness = -50 * nclasses bocciata
                return (max_incorrect_bits - prediction_strength);
            
            case 3: //minfitness = -50 * nclasses bocciata
                return static_cast<F>(sum_incorrect_bits - prediction_strength);
            
            case 4: //minfitness = 0 maybe better case 5
            if (best_class == true_label) {
                return 0.0;
            } else {
                return 50.0 + (max_bits_on - prediction_strength) + (sum_incorrect_bits) * 0.1;
            }

            case 5: //minfitness = 0
            if (best_class == true_label) {
                return 0.0;
            } else {
                return 50.0 + (max_bits_on - prediction_strength) + (sum_incorrect_bits) * 0.05;
            }

            case 6: //minfitness = -500
            if (best_class == true_label) {
                return 0.0 - (prediction_strength - max_incorrect_bits);
            } else {
                return 50.0 + (max_bits_on - prediction_strength) + (sum_incorrect_bits) * 0.05;
            }

            case 7: //minfitness = -50 * nclasses
                if (best_class == true_label) {
                    return 0.0 - (prediction_strength - max_incorrect_bits);
                } else {
                    // Penalizza in base alla distanza dal vincitore attuale
                    return 50.0 + (max_bits_on - prediction_strength);
                }

            default:
                throw std::invalid_argument("Unknown fitness_function type!");
        }
    }

    MnistLogicProblem<E, G, F>* clone() override {
        return new MnistLogicProblem<E, G, F>(*this);
    }
};

#endif /* PROBLEMS_MNISTLOGICPROBLEM_H_ */