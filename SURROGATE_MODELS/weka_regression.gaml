/**
* Name: multi_linear_regression
* 
* This file contains a function to train different regression models from the gama plugin weka.
* It had only been tested with CART (train_rep_tree) model but seems to work the exact same way with other models.
* To use it, you need to update the init block with the values corresponding to your model and then run the main method
* 
* Author: Dupont Raphaël
* Tags: surrogate model
* 
 */

model weka_regression

global{
	init{
		do regression_model(
			"./Results/COMOKIT_run.csv",
			",",
			[
				"nb_init_infected",
				"density_ref_contact",
				"init_all_ages_successful_contact_rate_human",
				"init_all_ages_factor_contact_rate_asymptomatic",
				"init_all_ages_proportion_asymptomatic",
				"init_all_ages_proportion_hospitalisation",
				"init_all_ages_proportion_icu",
				"init_all_ages_proportion_dead_symptomatic"
			],
			map([]),			
			"nb_dead",
			"train_reptree", // train_smo_reg, train_reptree, train_gauss, train_rbf
			true,
			"./Results/weka_reg.txt"
			);
	}

	
	/**
	 * Compute a regression model using weka library
	 */
	action regression_model(
		string csv_path, // Path to the .csv file of format : X0, ..., XN | Y0, ... YK
		string separator, // Separator of the csv values
		list variables, // List of the variables ex: ["infection_probability", "nb_infected_init", ..]
		map desc,// decription for nominale variables ex: ["age"::["H","F"]]
		string output,  // Name of the output variable
		string model_type, // train_smo_reg, train_reptree, train_gauss, train_rbf
		bool print_model, // If true print the model
		string result_path // Path to write the report
	){	
		// Open the csv
		csv_file my_csv_file <- csv_file(csv_path, separator, string, false);
		matrix mat <- matrix<string>(my_csv_file);
		
		// Get the output column
		int output_index;
		loop j from:0 to: mat.columns - 1 {
			if (mat[j, 0] = output){
				output_index <- j;
			}
		}
		
		// Convert data
		list data;
		list rows <- rows_list(mat);
		loop i from:1 to: length(rows) - 1 {
			map sample;
			loop j from:0 to: length(variables) - 1{
				add variables[j]::rows[i][j] to: sample;
			}
			add output::rows[i][output_index] to: sample;			
			add sample to: data;
		}
	
		data <- shuffle(data);
		
		
		// Build test and training set
		int train_size <- int(floor(0.9 * length(data)));
		
		list train;
		loop i from:0 to: train_size - 1{
			add data[i] to: train;
		}
		
		list test;
		loop i from:train_size to: length(data) - 1{
			add data[i] to: test;
		}
		
		// Train classifier
		classifier rf;
		switch model_type{
			match "train_smo_reg"{ rf<- train_smo_reg(data, variables, output, desc); }
			match "train_reptree"{ rf<- train_reptree(data, variables, output, desc); }
			match "train_gauss"{ rf<- train_gauss(data, variables, output, desc); }
			match "train_rbf"{ rf<- train_rbf(data, variables, output, desc); }
		} 
		// Test Classifier
		
		// predictions
		list<float> predictions;
		list<float> actuals;
		loop i from:0 to: length(test) - 1{
			map sample <- test[i];
			add float(sample[output]) to: actuals;
			remove key: output from: sample;
			add float(rf classify(sample)) to: predictions;
		}
		
		// compute error
		float error <- MAPE(predictions, actuals);
		
		
		// output
		string s <- "";
		if(print_model){
			s <- s + rf + "\n";
		}
		s <- s + "Size of training set : " + string(train_size) + "\n";
		s <- s + "Size of test set : " + string(length(data) - train_size) + "\n";
		s <- s + "Mean Absolute Percentage Error : " + string(error * 100) + "%" + "\n";
		s <- s + "Accuracy : " + string(100 * (1.0 - error))+ "\n";
		
		write s;
		save s to:result_path;
	}
	
	float MAPE(list<float> pred, list<float> actual){
		float mape <- 0.0;
		loop i from:0 to: length(pred) - 1{
			mape <- mape + abs(pred[i] - actual[i]) / sum(actual);
		}
		return mape;
	}
}

experiment main type:gui;