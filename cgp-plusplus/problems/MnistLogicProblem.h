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

public:
    MnistLogicProblem(std::shared_ptr<Parameters> p_parameters,
                      std::shared_ptr<Evaluator<E, G, F>> p_evaluator,
                      std::shared_ptr<std::vector<std::vector<E>>> p_inputs,
                      std::shared_ptr<std::vector<std::vector<E>>> p_outputs,
                      std::shared_ptr<std::vector<E>> p_constants,
                      int p_num_instances)
        : BlackBoxProblem<E, G, F>(p_parameters, p_evaluator, p_inputs, p_outputs, p_constants, p_num_instances) {
        
        this->name = "MNIST Logic Problem";
        
        // Calcoliamo i bit per classe dinamicamente dal numero totale di output
        // Es: 500 output / 10 classi = 50 bit per classe
        if (this->parameters->get_num_outputs() % NUM_CLASSES != 0) {
            throw std::invalid_argument("Total outputs must be a multiple of 10 (classes)!");
        }
        this->bits_per_class = this->parameters->get_num_outputs() / NUM_CLASSES;
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
            
            // Valuta l'individuo (senza lock mtx qui, siamo nel thread principale di report)
            this->evaluator->evaluate_iterative(individual, input_instance, outputs_ind);

            // 2. Logica di Classificazione (Bit Counting)
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

            // 3. Incrementa Hits se corretto
            if (best_class == true_label) {
                hits++;
            }
        }
        return hits;
    }

    /**
     * @brief Valuta una singola predizione.
     * @param outputs_real Contiene la LABEL vera come unico elemento (vedi Initializer).
     * @param outputs_individual Contiene la stringa di bit prodotta dalla rete (es. 500 bit).
     * @return 0.0 se la predizione è corretta, 1.0 se è errata (Minimizzazione).
     */
    F evaluate(std::shared_ptr<std::vector<E>> outputs_real,
               std::shared_ptr<std::vector<E>> outputs_individual) override {
        
        // 1. Recupera la vera etichetta (Salvata nell'elemento 0 dal nostro Initializer)
        int true_label = static_cast<int>(outputs_real->at(0));

        // 2. Logica di Bit-Counting (Population Count)
        int best_class = -1;
        int max_bits_on = -1;
        int prediction_strength = 0;


        // Itera su ogni classe (0-9)
        for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
            int current_bits_on = 0;
            int start_idx = class_idx * this->bits_per_class;

            // Conta i bit a '1' nel blocco dedicato a questa classe
            for (int bit = 0; bit < this->bits_per_class; ++bit) {
                // Consideriamo qualsiasi valore != 0 come '1' logico (per sicurezza)
                if (outputs_individual->at(start_idx + bit) != 0) {
                    current_bits_on++;
                }
            }

            // ArgMax: Se questo blocco ha più '1' del precedente record, diventa il vincitore.
            // Nota: In caso di pareggio, qui vince la classe con indice minore.
            if (current_bits_on > max_bits_on) {
                max_bits_on = current_bits_on;
                best_class = class_idx;
            }
            if (class_idx == true_label) {
                prediction_strength = current_bits_on;
            }
        }

        // the more bits on for the correct class, the better
        if (best_class == true_label) {
            return 0 - prediction_strength;
        } else {
            return 50.0 + (max_bits_on - prediction_strength); // the more its wrong, the worse
        }
    }

    MnistLogicProblem<E, G, F>* clone() override {
        return new MnistLogicProblem<E, G, F>(*this);
    }
};

#endif /* PROBLEMS_MNISTLOGICPROBLEM_H_ */