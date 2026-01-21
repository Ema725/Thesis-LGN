#ifndef PROBLEMS_MNISTLOGICPROBLEM_H_
#define PROBLEMS_MNISTLOGICPROBLEM_H_

#include "BlackBoxProblem.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <numeric>

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

    // Upload full dataset (Move semantics)
    void set_full_dataset(std::vector<std::vector<E>>&& all_inputs, 
                          std::vector<std::vector<E>>&& all_outputs) {
        std::cout << "[DEBUG] MnistLogicProblem: Moving full dataset..." << std::endl;
        this->full_inputs = std::move(all_inputs);
        this->full_outputs = std::move(all_outputs);
        std::cout << "[DEBUG] MnistLogicProblem: Dataset moved. Sizes: " 
                  << this->full_inputs.size() << " / " << this->full_outputs.size() << std::endl;
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

    // Aggiungi override nella classe MnistLogicProblem
    void load_random_batch(int batch_size, std::shared_ptr<Random> rng) override {
        this->inputs->clear();
        this->outputs->clear();
        
        int total_samples = this->full_inputs.size();
        
        // Creiamo un vettore di indici [0, 1, ..., total-1]
        static std::vector<int> indices;
        if (indices.size() != (size_t)total_samples) {
            indices.resize(total_samples);
            std::iota(indices.begin(), indices.end(), 0); // Richiede <numeric>
        }

        // Fisher-Yates shuffle parziale per selezionare 'batch_size' elementi casuali
        for (int i = 0; i < batch_size; i++) {
            int j = rng->random_integer(i, total_samples - 1);
            std::swap(indices[i], indices[j]);
            
            int idx = indices[i];
            // Carica l'immagine corrispondente all'indice scelto
            this->inputs->push_back(this->full_inputs[idx]);
            this->outputs->push_back(this->full_outputs[idx]);
        }
        
        this->num_instances = batch_size;
        
        std::cout << ">>> LOADED RANDOM BATCH (" << batch_size << " samples)" << std::endl;
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
            int ones_prediction = -1;

            for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
                int current_bits_on = 0;
                int start_idx = class_idx * this->bits_per_class;

                for (int bit = 0; bit < this->bits_per_class; ++bit) {
                    if (outputs_ind->at(start_idx + bit) != 0) {
                        current_bits_on++;
                    }
                }

                if (current_bits_on > ones_prediction) {
                    ones_prediction = current_bits_on;
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
     * @brief Override di evaluate_individual per gestire la fitness globale P2 (Fitness 23).
     */
    void evaluate_individual(std::shared_ptr<Individual<G, F>> individual) override {

        // Se NON stiamo usando la fitness 24, usa il comportamento standard
        if (this->parameters->get_fitness_function() != 24) {
            BlackBoxProblem<E, G, F>::evaluate_individual(individual);
            return;
        }

        if (individual->is_evaluated()) {
            return;
        }

        F total_fitness = 0;
        int error_count = 0; // Contatore errori accumulati nella batch
        
        // Definiamo i quartili per la penalità P2 in base alla dimensione della batch corrente
        int limit_q1 = this->num_instances * 0.25;
        int limit_q2 = this->num_instances * 0.50;
        int limit_q3 = this->num_instances * 0.75;
        
        // Valore base di P2 (come da formula nel PDF può essere un fattore di scala, qui 1.0)
        double P2_base = 1.0; 

        std::shared_ptr<std::vector<E>> outputs_ind = std::make_shared<std::vector<E>>();
        std::shared_ptr<std::vector<E>> input_instance;

        // Iterazione su tutte le istanze della batch
        for (int i = 0; i < this->num_instances; i++) {
            
            // 1. Preparazione Input (copia e aggiunta costanti)
            input_instance = std::make_shared<std::vector<E>>(this->inputs->at(i));
            if (this->num_constants > 0) {
                input_instance->insert(std::end(*input_instance),
                        std::begin(*this->constants), std::end(*this->constants));
            }
            outputs_ind->clear();

            // 2. Valutazione Rete
            mtx.lock();
            this->evaluator->evaluate_iterative(individual, input_instance, outputs_ind);
            mtx.unlock();

            // 3. Logica di Classificazione (Bit Counting)
            int true_label = static_cast<int>(this->outputs->at(i)[0]);
            int best_class = -1;
            int ones_prediction = -1;
            int ones_ground_truth = 0;
            int sum_incorrect_bits = 0;

            for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
                int current_bits_on = 0;
                int start_idx = class_idx * this->bits_per_class;

                for (int bit = 0; bit < this->bits_per_class; ++bit) {
                    if (outputs_ind->at(start_idx + bit) != 0) {
                        current_bits_on++;
                    }
                }

                if (current_bits_on > ones_prediction) {
                    ones_prediction = current_bits_on;
                    best_class = class_idx;
                }
                
                if (class_idx == true_label) {
                    ones_ground_truth = current_bits_on;
                } else {
                    sum_incorrect_bits += current_bits_on;
                }
            }

            // 4. Calcolo Componenti Fitness (e^j, P0, P1)
            double p_true = static_cast<double>(ones_ground_truth) / this->bits_per_class;
            double e_j = 1.0 - p_true; // Errore probabilistico [cite: 29]

            // P0: Penalità Normalizzazione 
            int total_ones = ones_ground_truth + sum_incorrect_bits;
            double P0 = 1.0;
            if (total_ones != this->bits_per_class) {
                P0 = 1.0 + (std::abs(total_ones - this->bits_per_class) * 0.1);
            }

            // P1: Penalità Errata Classificazione 
            double P1 = 1.0;
            bool is_error = (best_class != true_label);
            if (is_error) {
                P1 = 5.0; 
            }

            // 5. Calcolo P2 (Penalità Crescente a Quartili)
            double P2_multiplier = 1.0;
            
            if (is_error) {
                error_count++; // Incrementa il contatore degli errori commessi finora
                
                if (error_count <= limit_q1) {
                    P2_multiplier = 1.0; // Primo 25% errori: 1 * P2
                } else if (error_count <= limit_q2) {
                    P2_multiplier = 2.0; // Successivo 25%: 2 * P2
                } else if (error_count <= limit_q3) {
                    P2_multiplier = 4.0; // Successivo 25%: 4 * P2
                } else {
                    P2_multiplier = 8.0; // Ultimo 25%: 8 * P2
                }
            }

            // Formula: E = P2_mult * P2_base * (P1 * P0 * e^j)
            double instance_fitness = P2_multiplier * P2_base * (P1 * P0 * e_j);
            total_fitness += static_cast<F>(instance_fitness);
        }

        individual->set_fitness(total_fitness);
        individual->set_evaluated(true);
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

        int ones_prediction = -1;
        //num bits at 1 for the ground truth class
        int ones_ground_truth = 0;
        int max_incorrect_bits = 0;
        //sum of bits at 1 for the incorrect classes
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
            if (current_bits_on > ones_prediction) {
                ones_prediction = current_bits_on;
                best_class = class_idx;
            }
            if (class_idx == true_label) {
                ones_ground_truth = current_bits_on;
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
                    return 0.0 - ones_ground_truth;
                } else {
                    return 50.0 + (ones_prediction - ones_ground_truth);
                }

            case 1: //minfitness = 0 maybe better case 0
                if (best_class == true_label) {
                    return 0.0;
                } else {
                    return 50.0 + (ones_prediction - ones_ground_truth);
                }

            case 2: //minfitness = -50 * nclasses bocciata
                return (max_incorrect_bits - ones_ground_truth);
            
            case 3: //minfitness = -50 * nclasses bocciata
                return static_cast<F>(sum_incorrect_bits - ones_ground_truth);
            
            case 4: //minfitness = 0 maybe better case 5
            if (best_class == true_label) {
                return 0.0;
            } else {
                return 50.0 + (ones_prediction - ones_ground_truth) + (sum_incorrect_bits) * 0.1;
            }

            case 5: //minfitness = 0
            if (best_class == true_label) {
                return 0.0;
            } else {
                return 50.0 + (ones_prediction - ones_ground_truth) + (sum_incorrect_bits) * 0.05;
            }

            case 6: //minfitness = -500
            if (best_class == true_label) {
                return 0.0 - (ones_ground_truth - max_incorrect_bits);
            } else {
                return 50.0 + (ones_prediction - ones_ground_truth) + (sum_incorrect_bits) * 0.05;
            }

            case 7:
            if (best_class == true_label) {
                return 0.0;
            } else {
                return (50 - ones_ground_truth) * 2 + (sum_incorrect_bits);
            }

            case 8: //somma dei bit incorretti + somma dei bit a 0 nella classe corretta
            return sum_incorrect_bits + (50 - ones_ground_truth);

            case 9: //somma dei bit incorretti + somma dei bit a 0 nella classe corretta
            return sum_incorrect_bits + (50 - ones_ground_truth) * 2;

            case 10: //somma dei bit incorretti + somma dei bit a 0 nella classe corretta
            return sum_incorrect_bits + (50 - ones_ground_truth) * 5;

            case 11: //somma dei bit incorretti + somma dei bit a 0 nella classe corretta
            return sum_incorrect_bits + (50 - ones_ground_truth) * 10;

            case 12: //somma dei bit incorretti + somma dei bit a 0 nella classe corretta
            return sum_incorrect_bits + (50 - ones_ground_truth) * 100;

            case 13: //9 mod
            if (best_class == true_label) {
                return (50 - ones_ground_truth) + (sum_incorrect_bits) * 0.5;
            } else {
                return (50 - ones_ground_truth) * 2 + (sum_incorrect_bits);
            }

            case 14: //10 mod
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 5 + (sum_incorrect_bits)) * 0.5;
            } else {
                return (50 - ones_ground_truth) * 5 + (sum_incorrect_bits);
            }

            case 15: //10 mod 2
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 5 + (sum_incorrect_bits)) * 0.75;
            } else {
                return (50 - ones_ground_truth) * 5 + (sum_incorrect_bits);
            }

            case 16: //10 mod 3
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 5 + (sum_incorrect_bits)) * 0.25;
            } else {
                return (50 - ones_ground_truth) * 5 + (sum_incorrect_bits);
            }

            case 17: //11 mod
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 10 + (sum_incorrect_bits)) * 0.5;
            } else {
                return (50 - ones_ground_truth) * 10 + (sum_incorrect_bits);
            }

            case 18: //11 mod 2
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 10 + (sum_incorrect_bits)) * 0.75;
            } else {
                return (50 - ones_ground_truth) * 10 + (sum_incorrect_bits);
            }

            case 19: //11 mod 3
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 10 + (sum_incorrect_bits)) * 0.25;
            } else {
                return (50 - ones_ground_truth) * 10 + (sum_incorrect_bits);
            }

            case 20: //12 mod
            if (best_class == true_label) {
                return ((50 - ones_ground_truth) * 100 + (sum_incorrect_bits)) * 0.5;
            } else {
                return (50 - ones_ground_truth) * 100 + (sum_incorrect_bits);
            }

            case 21: //jaccard
            {
                int M01 = ones_ground_truth - ones_prediction;
                int denominator = ones_ground_truth + sum_incorrect_bits;
                int numerator = M01 + sum_incorrect_bits;
                //M10 = sum_incorrect_bits
                if(numerator == 0) {
                    return 0.0; //perfect classification
                }else {
                    return static_cast<F>(numerator/ static_cast<F>(denominator));
                }
            }

            case 22: // boosted jaccard
            {
                int M01 = ones_ground_truth - ones_prediction;
                int denominator = ones_ground_truth + sum_incorrect_bits;
                int numerator = M01 + sum_incorrect_bits;
                //M10 = sum_incorrect_bits
                if(numerator == 0) {
                    return 0.0; //perfect classification
                } else if (best_class != true_label) {
                    return static_cast<F>((numerator/ static_cast<F>(denominator)) * 500);
                } else {
                    return static_cast<F>((numerator/ static_cast<F>(denominator))* 500 ) * 0.5;
                }
            }

            case 23: // Alberto Incompleta
            {
                // 1. Calcolo di e^j (Errore di Probabilità)
                // p_i^j = bit_attivi_classe_corretta / L_c
                // e^j = 1 - p_i^j
                double p_true = static_cast<double>(ones_ground_truth) / this->bits_per_class;
                double e_j = 1.0 - p_true;

                // 2. Calcolo di P0 (Penalità di Normalizzazione)
                // Vincolo: Somma di tutti i bit a 1 (su tutte le classi) deve essere pari a L_c (50)
                // P0 è un fattore moltiplicativo > 1.0 se il vincolo è violato
                int total_ones = ones_ground_truth + sum_incorrect_bits;
                double P0 = 1.0;
                
                if (total_ones != this->bits_per_class) {
                    // Esempio: Aumenta la penalità del 10% per ogni bit di deviazione
                    // Questo forza la rete a spegnere i bit delle classi sbagliate per bilanciare
                    P0 = 1.0 + (std::abs(total_ones - this->bits_per_class) * 0.1); 
                }

                // 3. Calcolo di P1 (Penalità Errata Classificazione)
                // Se la classe predetta è sbagliata, moltiplichiamo l'errore
                double P1 = 1.0;
                if (best_class != true_label) {
                    P1 = 5.0; // Fattore di penalità configurabile (es. 5x)
                }

                // Formula finale: E = P1 * (P0 * e^j)
                // Nota: P2 (logica dei quartili) è omesso perché richiede valutazione globale
                return static_cast<F>(P1 * P0 * e_j);
            }

            case 24: // Alberto completa
            {
                // 1. Calcolo di e^j (Errore di Probabilità)
                // p_i^j = bit_attivi_classe_corretta / L_c
                // e^j = 1 - p_i^j
                double p_true = static_cast<double>(ones_ground_truth) / this->bits_per_class;
                double e_j = 1.0 - p_true;

                // 2. Calcolo di P0 (Penalità di Normalizzazione)
                // Vincolo: Somma di tutti i bit a 1 (su tutte le classi) deve essere pari a L_c (50)
                // P0 è un fattore moltiplicativo > 1.0 se il vincolo è violato
                int total_ones = ones_ground_truth + sum_incorrect_bits;
                double P0 = 1.0;
                
                if (total_ones != this->bits_per_class) {
                    // Esempio: Aumenta la penalità del 10% per ogni bit di deviazione
                    // Questo forza la rete a spegnere i bit delle classi sbagliate per bilanciare
                    P0 = 1.0 + (std::abs(total_ones - this->bits_per_class) * 0.1); 
                }

                // 3. Calcolo di P1 (Penalità Errata Classificazione)
                // Se la classe predetta è sbagliata, moltiplichiamo l'errore
                double P1 = 1.0;
                if (best_class != true_label) {
                    P1 = 5.0; // Fattore di penalità configurabile (es. 5x)
                }

                // Formula finale: E = P1 * (P0 * e^j)
                // Nota: P2 (logica dei quartili) è omesso perché richiede valutazione globale
                return static_cast<F>(P1 * P0 * e_j);
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