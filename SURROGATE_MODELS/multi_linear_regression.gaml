/**
* Name: multi_linear_regression
* 
* This file contain a function to train a multilenear model for regression.
* It uses a .csv file as an input with the model inputs and outputs as columns.
* (first columns of the .csv should be the inputs)
* You need to update the init block with the values corresponding to your model and then run the main method
* You can visualize the residual graphs of each output to see if the multi linear model fits well you data or if 
* you need a more complex model.
* 
* 
* Author: Dupont Raphaël
* Tags: surrogate model
*/


model multi_linear_regression

global {	
	init{
		do multi_linear_regression(
			"./Results/COMOKIT_run.csv",		//Input path
			",",								//Separator
			8,									//Number of variables
			"./Results/linear_regression.txt"	//Output report path
		);
	}
	
	action multi_linear_regression(string csv_path, string separator, int nb_variables, string result_path){
		//CONST
		csv_file my_csv_file <- csv_file(csv_path, separator, string, false);
			
		//VAR
		matrix<float> X;
		matrix<float> Y;
		list<regression> reg_functions;
		list<float> r_square;
		list<float> p_values;
		
		//#################################################
		//############# get X and Y matrices ##############
		//#################################################		
		
		//convert the file into a matrix
		matrix<string> data <- matrix<string>(my_csv_file);
		
		//write sample(data);
		
		//init variables
		bound <- nb_variables;
		X <- matrix<float>({bound, data.rows-1} matrix_with 0);
		Y <- matrix<float>({data.columns - bound, data.rows-1} matrix_with 0);
		var_names <- list_with(data.columns, "");
		write "Number of sample : " + X.rows;
		write "Number of input variables : " + X.columns;
		write "Number of output variables : " + Y.columns;
		
		
		//header line
		loop j from: 0 to: data.columns - 1{
			var_names[j] <- data[j,0];
		}
		write "variable names : " + var_names;
		
		
		//loop on the matrix rows (skip the first header line)
		loop i from: 1 to: data.rows - 1{
			//loop on the matrix variables
			loop j from: 0 to:bound - 1{
				X[j,i-1] <- float(data[j,i]);
			}
			//loop on the matrix outputs
			loop j from:bound to: data.columns -1{
				Y[j-bound,i-1] <- float(data[j,i]);
			}
		}
		
		
		//regressions for each ouput
		loop output from: 0 to: Y.columns - 1{
			
			//#################################################
			//############## Compute regression ###############
			//#################################################		
			
			matrix y <- (Y column_at output) as_matrix {1,Y.rows};
			matrix instance <- append_horizontally(y, X);
			
 
			regression reg <- build(instance);
			write "\nOutput :" + var_names[bound + output];	
			write "regression : " + reg; // y_pred = const + coef_1 * x1 + ... + coef_n * xn
			add reg to: reg_functions;
			
			
			//#################################################
			//############## Evaluate regression ##############
			//#################################################
			float r2 <- rSquare(reg);
			write "adjusted R² = " + r2;
			add r2 to:r_square;
			
			
			list<float> y_pred;
			loop i from: 0 to:X.rows - 1{
				add predict(reg, X row_at i) to: y_pred;
			}
			add list<float>(residuals(reg)) to: residuals;
			add y_pred to: y_fit;
			
			//write sample(y_fit);
			//write sample(residuals(reg));
			
			
		}
		
		
		//#################################################
		//################# Save output ###################
		//#################################################	
		write "\n";
		string s <- "=================REPORT=================";
		s <- s + "\n----------------------------------------\nvariables :\n----------------------------------------\n";
		loop i from: 0 to: bound - 1{
			s <- s + "x" + (i+1) + " = " + var_names[i] + "\n";
		}
		s <- s + "\n----------------------------------------\nregression functions :\n----------------------------------------\n";
		loop i from: 0 to: length(reg_functions) - 1{
			s <- s + "output - " + var_names[bound + i] + " : \n";
			s <- s + string(reg_functions[i]) + "\n";
			s <- s + "adjusted R² = " + string(r_square[i]) + "\n\n";
		}
		write s;
		save s to: result_path rewrite:true;
	}
	
	//For residual graphs
	list<list<float>> y_fit;
	list<list<float>> residuals;
	list<string> var_names;
	float xplot;
	float yplot;
	string current_output <- "";
	int bound;
	int output_index <- 0;
	int y_index <- 0;
	bool end <- false;
	
	reflex plot_all_residual when: !end{
		write string(output_index) + " " + string(y_index) + " " + string(y_fit[0][0]) + " " + string(residuals[0][0]);
		xplot <- y_fit[output_index][y_index];
		yplot <- residuals[output_index][y_index];
		if(length(first(y_fit)) - 1 = y_index or length(first(residuals)) - 1 = y_index){
			write "play for next output";
			do pause;
			
			if(length(y_fit) - 1 = output_index and length(residuals) - 1 = output_index){
				end <- true;
				write "end";
			}else{
			 	y_index <- -1;
				output_index <- output_index + 1;	
			}
		}
		
		if (!end) { y_index <- y_index + 1; }
	}
	
}


experiment main type:gui;

experiment residual_graphs type: gui{
	output{
		display residuals refresh:every(5#cycles) and cycle > 0 {
			chart "residuals vs fited" type: xy {
				data "residuals" value: {xplot, yplot} line_visible:false;
			}
		}
	}
}
