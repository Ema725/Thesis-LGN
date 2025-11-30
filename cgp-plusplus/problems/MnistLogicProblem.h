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
        }

        // 3. Calcolo Fitness (Errore)
        if (best_class == true_label) {
            return 0.0; // Corretto -> 0 errori
        } else {
            return 1.0; // Sbagliato -> 1 errore
        }
    }

    MnistLogicProblem<E, G, F>* clone() override {
        return new MnistLogicProblem<E, G, F>(*this);
    }
};

#endif /* PROBLEMS_MNISTLOGICPROBLEM_H_ */