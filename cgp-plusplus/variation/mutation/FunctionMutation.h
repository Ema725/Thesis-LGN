//	CGP++: Modern C++ Implementation of Cartesian Genetic Programming
// ===============================================================================
//	File: FunctionMutation.h
// ===============================================================================

#ifndef VARIATION_FUNCTIONMUTATION_H_
#define VARIATION_FUNCTIONMUTATION_H_

#include "../UnaryOperator.h"

/// @brief Mutation operator that targets only function genes.
template <class G, class F>
class FunctionMutation : public UnaryOperator<G,F> {
private:
	float mutation_rate;
public:
	FunctionMutation(std::shared_ptr<Parameters> p_parameters,
			std::shared_ptr<Random> p_random,
			std::shared_ptr<Species<G>> p_species);
	virtual ~FunctionMutation() = default;

	void variate(std::shared_ptr<Individual<G, F>> individual) override;
};

template <class G, class F>
FunctionMutation<G, F>::FunctionMutation(std::shared_ptr<Parameters> p_parameters,
		std::shared_ptr<Random> p_random,
		std::shared_ptr<Species<G>> p_species) : UnaryOperator<G, F>(p_parameters, p_random, p_species ) {

		this->name = "Function Mutation";
		this->mutation_rate = this->parameters->get_function_mutation_rate();
}

template <class G, class F>
void FunctionMutation<G, F>::variate(std::shared_ptr<Individual<G, F>> individual) {

	std::shared_ptr<G[]> genome = individual->get_genome();

	int num_function_nodes = this->parameters->get_num_function_nodes();
	int max_arity = this->parameters->get_max_arity();

	float current_rate = this->parameters->get_function_mutation_rate();
	
    // Calcola il numero di mutazioni basato sulla percentuale dei nodi funzione
	int num_mutations = (int)(current_rate * num_function_nodes);
    
    // Se il rate è molto basso ma > 0, garantiamo almeno 1 mutazione? 
    // Opzionale, ma solitamente preferibile se num_mutations arrotonda a 0.
    if (num_mutations == 0 && this->mutation_rate > 0) num_mutations = 1;

	for (int i = 0; i < num_mutations; i++) {
        // 1. Seleziona un nodo funzione a caso (indice 0 .. N-1)
		int random_node_idx = this->random->random_integer(0, num_function_nodes - 1);
        
        // 2. Calcola la posizione del gene funzione nel genoma
        // In CGP standard, il gene funzione è il primo di ogni blocco nodo
        // Blocco = [Func, In1, In2, ...] -> Dimensione = max_arity + 1
        int gene_pos = random_node_idx * (max_arity + 1);

        // 3. Muta il valore
		if (this->species->is_real_valued()) {
			genome[gene_pos] = this->random->random_float(0.0, 1.0);
		} else {
            // Usa Species per ottenere i limiti corretti per i geni funzione
			int min_gene = this->species->min_gene(gene_pos);
			int max_gene = this->species->max_gene(gene_pos);
            
            // Assicura che il nuovo valore sia diverso dal precedente (opzionale ma utile)
            int current_val = genome[gene_pos];
            int new_val = current_val;
            
            // Se c'è solo 1 funzione disponibile, non può cambiare
            if (max_gene > min_gene) {
                while (new_val == current_val) {
                    new_val = this->random->random_integer(min_gene, max_gene);
                }
                genome[gene_pos] = new_val;
            }
		}
	}
}

#endif /* VARIATION_FUNCTIONMUTATION_H_ */