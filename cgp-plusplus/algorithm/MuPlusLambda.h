//	CGP++: Modern C++ Implementation of Cartesian Genetic Programming
// ===============================================================================
//	File: MuPlusLambda.h 
// ===============================================================================
//
// ===============================================================================
//  Copyright (C) 2024
//
//
//	License: Academic Free License v. 3.0
// ================================================================================

#ifndef ALGORITHM_MUPLUSLAMBDA_H_
#define ALGORITHM_MUPLUSLAMBDA_H_

#include "EvolutionaryAlgorithm.h"

/// @brief Provides a mu+lambda ES that is used to enable crossover-based CGP. 

/// @details This implementation follows the defintion of a mu+lambda EA 
/// by Beyer and Schwefel (2002)

/// @see Beyer, Schwefel: Evolution strategies -  A comprehensive introduction.
/// Natural Computing 1, 3–52 (2002). 
/// https://doi.org/10.1023/A:1015059928466

/// @tparam E Evaluation Type
/// @tparam G Genotype Type
/// @tparam F Fitness Type 
// 
template<class E, class G, class F>
class MuPlusLambda: public EvolutionaryAlgorithm<E, G, F> {
private:
	int mu;
	int lambda;

	int select_parent();
	void breed(int num_offspring) override;
public:
	MuPlusLambda(std::shared_ptr<Composite<E, G, F>> p_composite);
	virtual ~MuPlusLambda() = default;

	std::pair<int, F> evolve() override;

};

template<class E, class G, class F>
MuPlusLambda<E, G, F>::MuPlusLambda(
		std::shared_ptr<Composite<E, G, F>> p_composite) :
		EvolutionaryAlgorithm<E, G, F>(p_composite) {
	this->name = "mu-plus-lambda";
	mu = this->parameters->get_mu();
	lambda = this->parameters->get_lambda();
	this->parameters->set_population_size(mu + lambda);
}

template<class E, class G, class F>
int MuPlusLambda<E, G, F>::select_parent() {
	return this->random->random_integer(0, this->mu - 1);
}

/// @brief Breeds new offspring by recombination and mutation by 
/// selecting from the parent population. 
template<class E, class G, class F>
void MuPlusLambda<E, G, F>::breed(int num_offspring) {

	for (int i = 0; i < num_offspring; i++) {

		int idx1 = this->select_parent();
		int idx2 = this->select_parent();

		std::shared_ptr<Individual<G, F>> p1 = this->population->get_individual(
				idx1);
		std::shared_ptr<Individual<G, F>> p2 = this->population->get_individual(
				idx2);

		std::shared_ptr<Individual<G, F>> o1 =
				std::make_shared<Individual<G, F>>(p1);

		std::shared_ptr<Individual<G, F>> o2 =
				std::make_shared<Individual<G, F>>(p2);

		this->recombination->crossover(o1, o2);

		this->mutation->mutate(o1);
		o1->set_evaluated(false);

		this->population->set_individual(o1, this->mu + i);
	}
}

/// @brief Evolves the population in the mu+lambda fashion 
/// @return number of fitness evaluations, best fitness 
template<class E, class G, class F>
std::pair<int, F> MuPlusLambda<E, G, F>::evolve() {

	this->best_fitness = this->fitness->worst_value();
	this->is_ideal = false;

	// <<< BATCH TRACKING >>>
    double current_batch_min_acc = 0.0; // Accuracy at the beginning of the batch
	F current_batch_min_loss = 0.0; // Loss at the beginning of the batch
    int current_batch_idx = 0;          // Index for current batch
    bool first_eval_of_batch = true;    // Flag for the first evaluation of the batch

	// <<< 1/5th RULE VARIABLES >>>
    int rule_k_counter = 0;         // Conta le generazioni
    int rule_success_counter = 0;   // Conta le mutazioni di successo
    const int RULE_K = this->parameters->get_K();         // Finestra di k generazioni
    const float RULE_C = this->parameters->get_C();       // Fattore di moltiplicazione/divisione

	while (this->generation_number <= this->max_generations && !this->is_ideal) {

		// <<< INIZIO NUOVA LOGICA BATCH >>>
        bool batch_switched = false;

		if (this->parameters->get_batch_training() > 0 && 
            this->generation_number > 0 && 
            this->generation_number % this->parameters->get_batch_gen() == 0) {
				
			// A. FINE BATCH PRECEDENTE: Calcola Max Accuracy e Scrivi Report
            // Recupera il miglior individuo corrente (che è il risultato finale di questa batch)
            this->population->sort(); // Assicuriamoci che sia ordinata
            auto best_ind_end = this->population->get_individual(0);

			this->evaluator->decode_path(best_ind_end);

			auto problem = this->composite->get_problem();
            int total_samples = problem->get_num_instances(); // Ritorna 2000 se hai fatto la mod precedente
            int hits_end = problem->validate_individual(best_ind_end);

			double batch_max_acc = (double)hits_end / total_samples * 100.0;
			F batch_max_loss = best_ind_end->get_fitness();

            // Scrivi nel file .stat se disponibile
            if (this->stat_stream) {
                *this->stat_stream << "Batch " << current_batch_idx 
                                   << " :: Min Accuracy: " << current_batch_min_acc << "%"
                                   << " :: Max Accuracy: " << batch_max_acc << "%" 
								   << " minloss: " << current_batch_min_loss  // [NEW] Stampa minloss
                                   << " maxloss: " << batch_max_loss          // [NEW] Stampa maxloss
                                   << std::endl;
                this->stat_stream->flush();
            }

            // B. CARICAMENTO NUOVA BATCH
            if (this->parameters->get_batch_training() == 1) {
                // --- MODE 1: SEQUENTIAL ---
                int file_size = this->parameters->get_file_size();
                int batch_size = this->parameters->get_batch_size();
                int batch_gen = this->parameters->get_batch_gen();
                
                int total_batches = file_size / batch_size;
                int batch_idx = (this->generation_number / batch_gen) % total_batches;
                int start_idx = batch_idx * batch_size;

                std::cout << ">>> BATCH SWITCH (SEQ) at Gen " << this->generation_number 
                          << " -> Loading Batch " << batch_idx 
                          << " (Range: " << start_idx << "-" << start_idx + batch_size << ")" << std::endl;
                        
                current_batch_idx = batch_idx;
                problem->load_batch(start_idx, batch_size);

            } else {
                // --- MODE 2: RANDOM ---
                int batch_size = this->parameters->get_batch_size();
                
                std::cout << ">>> BATCH SWITCH (RND) at Gen " << this->generation_number << std::endl;
                
                current_batch_idx = -1; // -1 indica random
                // Chiama il nuovo metodo passando il generatore random
                problem->load_random_batch(batch_size, this->random);
            }

            // 3. Resetta la fitness di TUTTA la popolazione (Genitori inclusi)
            // Poiché i dati sono cambiati, la fitness calcolata prima non è più valida.
            // Dobbiamo costringere l'algoritmo a ricalcolare quanto sono bravi i genitori sui nuovi dati.
            for (int i = 0; i < this->population->size(); i++) {
                this->population->get_individual(i)->set_evaluated(false);
            }
            
            batch_switched = true;
			first_eval_of_batch = true;
			rule_k_counter = 0;
            rule_success_counter = 0;
        }
		
		// Trigger the evaluation process
		this->evaluate();

		// 2. Applicazione Regola 1/5 (PRIMA del sort)
        int rule_mode = this->parameters->get_one_fifth_rule();
        
        if (rule_mode > 0) {
            int mu = this->parameters->get_mu();
            int lambda = this->parameters->get_lambda();
            
            // Il miglior genitore è sempre all'indice 0
            F parent_fitness = this->population->get_individual(0)->get_fitness();
            
            if (rule_mode == 1) {
                // --- MODALITA' 1: STANDARD ---
                // Conta ogni singolo figlio che batte il genitore.
                // Success Rate = Successi / (K * Lambda)
                for (int i = mu; i < mu + lambda; i++) {
                    if (this->population->get_individual(i)->get_fitness() < parent_fitness) {
                        rule_success_counter++;
                    }
                }
            } 
            else if (rule_mode == 2) {
                // --- MODALITA' 2: BEST OFFSPRING ONLY ---
                // Conta successo solo se il MIGLIOR figlio della covata batte il genitore.
                // Success Rate = Successi / K
                
                F best_offspring_fitness = this->population->get_individual(mu)->get_fitness();
                
                // Trova il fitness migliore tra i figli (range [mu, mu+lambda-1])
                for (int i = mu + 1; i < mu + lambda; i++) {
                    F fit = this->population->get_individual(i)->get_fitness();
                    if (fit < best_offspring_fitness) {
                        best_offspring_fitness = fit;
                    }
                }

                // Se il campione dei figli ha superato il genitore, è un successo per questa generazione
                if (best_offspring_fitness < parent_fitness) {
                    rule_success_counter++;
                }
            }
            
            rule_k_counter++;

            // Se abbiamo raggiunto k generazioni, calcoliamo e resettiamo
            if (rule_k_counter >= RULE_K) {
                
                double total_trials;
                if (rule_mode == 1) {
                    total_trials = (double)RULE_K * lambda; // Tutti i figli contano
                } else {
                    total_trials = (double)RULE_K;          // Un tentativo (di gruppo) per generazione
                }

                double success_rate = (double)rule_success_counter / total_trials;
                float current_mut_rate = this->parameters->get_mutation_rate();
                float current_func_rate = this->parameters->get_function_mutation_rate();
                
                // Applicazione Regola: > 0.2 aumenta mutazione, < 0.2 diminuisce
                if (success_rate > 0.2) {
                    float new_rate = current_mut_rate / RULE_C;
                    if (new_rate > 1.0) new_rate = 1.0; // Cap a 1.0
                    this->parameters->set_mutation_rate(new_rate);

                    // [NEW] Aumenta Function Mutation Rate (se attiva)
                    if (current_func_rate > 0.0) {
                        float new_func_rate = current_func_rate / RULE_C;
                        if (new_func_rate > 1.0) new_func_rate = 1.0;
                        this->parameters->set_function_mutation_rate(new_func_rate);
                    }

                } else if (success_rate < 0.2) {
                    this->parameters->set_mutation_rate(current_mut_rate * RULE_C);

                    // [NEW] Diminuisci Function Mutation Rate (se attiva)
                    if (current_func_rate > 0.0) {
                        this->parameters->set_function_mutation_rate(current_func_rate * RULE_C);
                    }
                }

				// Debug opzionale
                
                if(current_mut_rate > 0.0) {
                    std::cout << "[1/5 Rule] Gen " << this->generation_number 
                              << " Success Rate: " << success_rate 
                              << " New Mut Rate: " << this->parameters->get_mutation_rate() 
                              << std::endl;
                }
                if(current_func_rate > 0.0) {
                    std::cout << "[1/5 Rule] Gen " << this->generation_number 
                              << " Success Rate: " << success_rate 
                              << " New Func Mut Rate: " << this->parameters->get_function_mutation_rate() 
                              << std::endl;
                }
                

                // Reset contatori per la prossima finestra
                rule_k_counter = 0;
                rule_success_counter = 0;
            }
        }

		// Increase the number of fitness evaluations by the number
		// that has been used in the evaluation procedure
		this->fitness_evaluations += this->lambda;

		// Sort population for the selection process
		this->population->sort();
		auto best_ind_current = this->population->get_individual(0);

		// Obtain best fitness from the sorted population
		this->best_fitness = this->population->get_individual(0)->get_fitness();

		// <<< CATTURA MIN ACCURACY (INIZIO BATCH) >>>
        // Se siamo alla prima generazione assoluta (Gen 1) O appena dopo uno switch
        if (this->generation_number == 1 || first_eval_of_batch) {
            auto problem = this->composite->get_problem();
            int total_samples = problem->get_num_instances();
            int hits_start = problem->validate_individual(best_ind_current);
            
            current_batch_min_acc = (double)hits_start / total_samples * 100.0;
			current_batch_min_loss = best_ind_current->get_fitness();
            
            first_eval_of_batch = false; // Fatto, non ricalcolare fino al prossimo switch
        }

		// Trigger reporting intermediate result results
		this->report(this->generation_number);

		// Check for ideal fitness
		this->check_ideal(this->generation_number);

		// Check for checkpoint modulo 
		this->check_checkpoint();

		// Breed lambda offspring 
		this->breed(lambda);

		this->generation_number++;

	}

	return std::pair<int, F> { this->fitness_evaluations, this->best_fitness };
}

#endif /* ALGORITHM_MUPLUSLAMBDA_H_ */
