/**
* Name: PGGxp
* Based on the internal empty template. 
* Author: kevinchapuis
* Tags: 
*/


model PGGxp

import "PPG.gaml"

global {
	
	// Number of max cycle in a simulation
	int LIMIT <- 1000;
	
	// OUTPUT VARIABLES
	float prey_nrj -> sum(prey collect each.energy);
	float prey_nb -> length(prey);
	float predator_nrj -> sum(predator collect each.energy);
	float predator_nb -> length(predator);
	float grass_nrj -> sum(vegetation_cell collect each.food);
	
	// INPUT VECTOR
	list params <- [
			nb_preys_init,pp_ratio,predator_proba_reproduce,prey_nb_max_offsprings,
			prey_energy_reproduce,prey_proba_reproduce,predator_nb_max_offsprings,
			predator_energy_reproduce
		];
	
	// BATCH RECORD PER STEPS
	bool batch_record <- false;
	string batch_xp;
	reflex record when:batch_record {
		save params+[cycle,prey_nrj,prey_nb,predator_nrj,predator_nb,grass_nrj] 
			to:"../Results/"+batch_xp+"_"+int(self)+".csv" rewrite:false;
	}
	
}

experiment ppg_batch type:batch virtual:true {
	
	// Full analysis
	parameter "Prey number" var: nb_preys_init min: 100 max: 2000;
	parameter "Predator prey ratio" var: pp_ratio min: 0.01 max: 0.2;
	parameter 'Predator probability reproduce' var: predator_proba_reproduce min:0.01 max:0.2;
	
	// Secondary parameters
	parameter 'Prey nb max offsprings: ' var: prey_nb_max_offsprings min:1 max:5;
	parameter 'Prey energy reproduce: ' var: prey_energy_reproduce min:0.5 max:0.95;
	parameter 'Prey probability reproduce: ' var: prey_proba_reproduce min:0.01 max:0.2;
	parameter 'Predator nb max offsprings: ' var: predator_nb_max_offsprings min:1 max:5;
	parameter 'Predator energy reproduce: ' var: predator_energy_reproduce min:0.5 max:0.95;
	
}

experiment ppd_stoch parent:ppg_batch type:batch until: length(prey)=0 or length(predator)=0 or cycle=LIMIT 
	repeat:50 keep_simulations:false {
	method stochanalyse outputs:["prey_nrj", "prey_nb", "predator_nrj", "predator_nb", "grass_nrj", "cycle"] 
		report:"Results/stochanalysis.csv" results:"Results/stochanalysis_raw.csv" sample:50;
}

experiment ppd_lhc parent:ppg_batch type:batch until: length(prey)=0 or length(predator)=0 or cycle=LIMIT 
	repeat:100 keep_simulations:false {
	method exploration sampling:"latinhypercube" sample:1000;
	init {batch_record <- true;}
}

