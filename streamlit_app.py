#=========================================================================================================================================================
# Author: Luis E C Rocha  - Ghent University, Belgium  - 26.09.2022
#
# Description:  This file contains the implementation of the Schelling Segregation Model, including alternative implementations
#               1. first install streamlit using "pip install streamlit" 
#               2. when you run streamlit, it will open a tab in your default browser with the streamlit application *it works as a webpage hosted at the following URL:  Local URL: http://localhost:8501
#
#=========================================================================================================================================================

# Import essential modules/libraries
import numpy as np
import random as rg
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import sleep
import scipy.signal as sg

#===============================================================================================================
# THIS IS THE DEFINITION OF THE CLASS FOR THE MODEL
class SchellingModel:

    #===================================================================
    # This method initialises the system
    def __init__(self, N, empty_ratio, threshold, boundary):

    	# These methods are used to fix the random number generator
#        np.random.seed(0)
        # Fixed seeds used for testing. Add None to have different seeds every time
        np.random.seed(None)

        # These are the parameters of the model, input by the user
        self.N = N
        self.empty_ratio = empty_ratio
        self.threshold = threshold
        self.boundary = boundary
        
        # This is the size of the system - Ps: this little trick just makes sure the system (grid) will have integer length for a square grid
        NN = int(np.sqrt(self.N))**2
        self.grid_size = int(np.sqrt(NN))

        # Initialise 3 grids for the neighbours
        # One for the number of neighbours of type A
        self.grid_neigh_A = np.zeros([self.grid_size, self.grid_size])
        # One for the number of neighbours of type B
        self.grid_neigh_B = np.zeros([self.grid_size, self.grid_size])
        # One for the number of empty neighbours of both types
        self.grid_neigh_E = np.zeros([self.grid_size, self.grid_size])

	#--------------------------------------------------------------------------------------------------
        # INITIAL GRID CONFIGURATION
        # Ratio of cell of type A, cell of type B, cell of type C (empty)
        # We define the number of empty cells (0) and then divide the remaining cells "1-empty_ratio" equally into type A (-1) and type B (+1)
        p = [(1.0-empty_ratio)/2.0, (1.0-empty_ratio)/2.0, empty_ratio]

        # Generates a random sample as an 1d array for types -1, 1, and 0, taking into account the fractions p defined above
        self.grid = np.random.choice([-1, 1, 0], size = NN, p = p)

        # Reshape the 1D array "self.grid" to a 2D array with size "(self.grid_size, self.grid_size)"
        self.grid = np.reshape( self.grid, (self.grid_size, self.grid_size) )

    	# Total number of agents in the system
        # Here I "cheat" and write a piece of code in a Pythonic way (using np.where to find all entries equal to -1)
        self.no_agents = len( np.where(self.grid == -1)[0] ) + len( np.where(self.grid == 1)[0] )

        # initialise the boundary conditions
        self.init_boundary(boundary)

#------------------------------------------------------------------------------------------------------------------
#       # ALTERNATIVE way to initialise the cells with random states instead of using the method "np.random.choice" and "np.reshape" above
#       # Monte Carlo implementation from scratch, see week 2
#       # define a 2d array of zeros
#       self.grid = np.zeros([self.grid_size, self.grid_size])
#       # scan all entries of the matrix
#       for i in range(self.grid_size):
#           for j in range(self.grid_size):
#               p = np.random.random()
#               if p <= (1-empty_ratio)/2:
#                  self.grid[i, j] = -1
#               elif p > (1-empty_ratio)/2 & p <= (1-empty_ratio):
#                  self.grid[i, j] = 1
#               else:
#                  self.grid[i, j] = 0
#------------------------------------------------------------------------------------------------------------------
    #====================================================================
    # Function to initialise the boundary conditions
    def init_boundary(self, boundary):            
        
	    # We use the Moore neighbourhood here
        # You could also use Von Neuman, but the original model uses Moore
        self.kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.kernel_norm = 8
        
        # Call the boundary conditions
        if self.boundary == "Periodic":
            self.periodic_boundary()
        elif self.boundary == "Finite":
            self.finite_boundary()            

    #====================================================================
    # MODEL: This method updates the state of all the unhappy agents (grid) at each time step - but it does not guarantee that the agents will be happy after the move
    def run(self):

        #-----------------------------------------------------
        # Call the boundary conditions
        if self.boundary == "Periodic":
            self.periodic_boundary()
        elif self.boundary == "Finite":
            self.finite_boundary()
        
        # Initialises a list of empty cells and a list of unhappy agents - to be used and reset at each time step
        temp_empty_cells = []
        unhappy = []

        # The loop checks one cell per time step. It updates a list of unhappy agents (with positions) and a list of empty cells (with positions) for this time step
        for i in range(self.grid_size):
            for j in range(self.grid_size):

                # Check if the cell type is -1
                if self.grid[i, j] == -1:
                    # Check if the fraction of neighbours of the cell is lower than the threshold - note that I removed the empty cells... - you can do it without removing them, but the result is partially correct
                    if self.kernel_norm != self.grid_neigh_E[i,j]:
                        if (self.grid_neigh_A[i, j]/(self.kernel_norm-self.grid_neigh_E[i, j]) <= self.threshold):
                            unhappy.append([i,j,-1])

                # Check if the cell type is 1
                elif self.grid[i, j] == 1:
                    # Check if the fraction of neighbours of the cell is lower than the threshold
                    if self.kernel_norm != self.grid_neigh_E[i,j]:
                        if (self.grid_neigh_B[i, j]/(self.kernel_norm-self.grid_neigh_E[i, j]) <= self.threshold):
                            unhappy.append([i,j,1])
        
                # Check if the cell type is 0 (empty). If so, store in the temporary list of empty cells
                elif self.grid[i, j] == 0:
                    temp_empty_cells.append([i,j])

        # this line shuffles the list of unhappy agents - to avoid keeping correlations in the sequence of updates
        # If you do not reshuffle, the next loop will always update the unhappy cells in the matrix sequentially
        np.random.shuffle(unhappy)
        
    	# this loop will change the position of all the unhappy agents at least once in this time step
    	# note that I do not guarantee that the agent is moving to a cell that keeps it happy - that is OK
    	# note that the number of unhappy agents is likely higher than the number of empty cells -that will change at every time an agents moves-, therefore, I loop over the unhappy agents
        for i in range( len(unhappy) ):

            # Select one empty cell at random
            pos = np.random.randint(0, len(temp_empty_cells))

            # Retrieve the x,y coordinates of the selected empty cell
            pos_empty_x = temp_empty_cells[pos][0]
            pos_empty_y = temp_empty_cells[pos][1]

            # Move the unhappy agent to this empty cell
            self.grid[ pos_empty_x, pos_empty_y ] = unhappy[i][2]

            # Make the cell of the unhappy agent empty
            self.grid[ unhappy[i][0], unhappy[i][1] ] = 0

            # Add the x,y coordinates of the unhappy agent -that just moved away- to the list of empty cells
            temp_empty_cells[pos][0] = unhappy[i][0]
            temp_empty_cells[pos][1] = unhappy[i][1]

        # Calculate the fraction of unhappy agents in this time step
        FUA = len(unhappy)/self.no_agents
       
        return( FUA )

    #====================================================================
    # Define boundary conditions - periodic
    def periodic_boundary(self):

        # This line makes the convolution of the kernel and the grid considering only the cell with value -1. The result is a matrix of the same size where the cells contain the number of neighbours
        self.grid_neigh_A = sg.convolve2d(self.grid == -1, self.kernel, mode='same', boundary='wrap')
        
        # This line makes the convolution of the kernel and the grid considering only the cell with value 1. The result is a matrix of the same size where the cells contain the number of neighbours
        self.grid_neigh_B = sg.convolve2d(self.grid == 1, self.kernel, mode='same', boundary='wrap')
        
    	# (trick) This line makes the convolution of the kernel and the grid considering only the cell with value (abs(grid)-1)==-1. That's a way to detect the number of empty cells around cell 1 or -1
        self.grid_neigh_E = sg.convolve2d(abs(self.grid)-1 == -1, self.kernel, mode='same', boundary='wrap')
        
        # This line returns a list of the positions of all cells with value 0
        self.empty_cells = np.where(self.grid == 0)

    #====================================================================
    # Define boundary conditions - finite
    def finite_boundary(self):

        # This line makes the convolution of the kernel and the grid considering only the cell with value -1. The result is a matrix of the same size where the cells contain the number of neighbours
        self.grid_neigh_A = sg.convolve2d(self.grid == -1, self.kernel, mode='same', boundary='fill')
        
        # This line makes the convolution of the kernel and the grid considering only the cell with value 1. The result is a matrix of the same size where the cells contain the number of neighbours
        self.grid_neigh_B = sg.convolve2d(self.grid == 1, self.kernel, mode='same', boundary='fill')

    	# (trick) This line makes the convolution of the kernel and the grid considering only the cell with value (abs(grid)-1)==-1. That's a way to detect the number of empty cells around cell 1 or -1
        self.grid_neigh_E = sg.convolve2d(abs(self.grid)-1 == -1, self.kernel, mode='same', boundary='fill')
         
        # This line returns a list of the positions of all cells with value 0
        self.empty_cells = np.where(self.grid == 0)

    #====================================================================
    # Calculates the Freeman Segregation Index, see slides week 4
    def freeman_segregation_index(self):

        # This is the Freeman Segregation Index
        n_AA = 0
        n_BB = 0
        n_AB = 0
        n_BA = 0
        for i in range( self.grid_size ):
            for j in range( self.grid_size ):
                
                if self.grid[i,j] == -1:
                    n_AA += self.grid_neigh_A[i,j]
                    n_AB += self.grid_neigh_B[i,j]
                elif self.grid[i,j] == 1:
                    n_BB += self.grid_neigh_B[i,j]
                    n_BA += self.grid_neigh_A[i,j]

        n_Ao = n_AA + n_AB
        n_Bo = n_BA + n_BB
        n_oA = n_AA + n_BA
        n_oB = n_AB + n_BB
        n_oo = n_AA + n_AB + n_BA + n_BB
        E_n_AB = 0
        E_n_BA = 0
        if n_oo != 0:
            E_n_AB = ( n_Ao * n_oB ) / n_oo
            E_n_BA = ( n_Bo * n_oA ) / n_oo
        
        FSI = 0
        if (E_n_AB != 0) & (E_n_BA != 0):
            FSI = ( (E_n_AB + E_n_BA) - (n_AB + n_BA) ) / (E_n_AB + E_n_BA)
            if FSI < 0:
                FSI = 0

        return(FSI)

    #====================================================================
    # Calculates the Fraction of neighbours of the same type as the focal cell, see slides week 4
    def fraction_same_type_neighbours(self):

        # This is the Fraction of the neighbours of the same type as the focal cell
        FSTN = 0
        for i in range( self.grid_size ):
            for j in range( self.grid_size ):
                
                if self.grid[i,j] == -1:
                    FSTN += self.grid_neigh_A[i,j] / self.kernel_norm
                elif self.grid[i,j] == 1:
                    FSTN += self.grid_neigh_B[i,j] / self.kernel_norm

        # I normalised by the sum of all coloured cells. Skip the white cells because they are empty
        # This sum is equivalent to "(1.0-self.empty_ratio) * self.grid_size*self.grid_size"
        FSTN /= self.no_agents

        return(FSTN)
    
#===============================================================================================================
# VISUALISATION OF THE MODEL DYNAMICS USING THE streamlit FRAMEWORK (see more on https://streamlit.io/)

#--------------------------------------------------------------------------------------------------
# Title of the visualisation - shows on screen

st.title("Schelling Model of Segregation")

#--------------------------------------------------------------------------------------------------
# Methods to interactively collect input variables

N = st.sidebar.slider("Population Size", 100, 10000, 1000)
empty_ratio = st.sidebar.slider("Empty households", 0.0, 1.0, 0.2)
threshold = st.sidebar.slider("Tolerance Threshold", 0.0, 1.0, 0.5)
boundary = st.sidebar.radio("Boundary Conditions", ('Periodic', 'Finite'))
no_iter = st.sidebar.number_input("Number of Iterations", 10)
no_sim = st.sidebar.number_input("Number of Realisations", 1)

#--------------------------------------------------------------------------------------------------
# Initialise the object   - Note that when one runs the code, the selected parameters will be passed here during the initialisation of the object via "self" method

schelling = SchellingModel(N, empty_ratio, threshold, boundary)

#===============================================================================================================
# VISUALISE THE DYNAMICS (IF THE USER CLICKS ON "Visualise") - This option is for visualisation
# CALCULATE THE DYNAMICS (IF THE USER CLICKS ON "Calculate") - no visualisation in this case, it runs several times to get averages

# This splits the screen into two parts (or containers) of relative size 5:1
col1, col2 = st.columns([5,1])

# In the first column
# Visualisation of the evolution

with col1:

    if st.button('Visualise'):

        cmap = ListedColormap(['red', 'white', 'blue'])
        progress_bar = st.progress(0)
        
        iteration_text = st.empty()
        plot_container = st.empty()
    
        # Initialise a list measure_FSI to store the Freeman Segregation Index over time
        measureFSI = []

        # Initialise a list measure_FSTN to store the Fraction of Same-Type Neighbours over time
        measureFSTN = []

        # Initialise a list measure_UA to store the Fraction of Unhappy Agents over time
        measureFUA = []
    
        # measures at t=0
        measureFSI.append(schelling.freeman_segregation_index())
        # Measure FSTN and append the results on the list "measureFSTN"
        measureFSTN.append(schelling.fraction_same_type_neighbours())
    
        # Run the simulation for "no_iter" iterations, i.e. total time of the simulation
        # Repeat routines below for each time step i
        for i in range(no_iter):
                
            # Call the method "Run" with the interaction rules - update the states in the grid
            FUA = schelling.run()
            # Return and store the number of unhappy agents at time 0 (t)
            measureFUA.append( FUA )
                
            # Store the other measures in the updated grid at time 1 (t+1)
            # Measure FSI and append the results on the list "measureFSI" - Ps: By the end of the simulation, this list contains all FSI for all time steps
            measureFSI.append(schelling.freeman_segregation_index())
            # Measure FSTN and append the results on the list "measureFSTN"
            measureFSTN.append(schelling.fraction_same_type_neighbours())

            # Create figure with subplots
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            
            #--------------------------------------------------------------------
            # left top of the grid - graph with the calculated values of FSI
            ax[0, 0].pcolormesh(schelling.grid, cmap=cmap, edgecolors='k', linewidths=0.0)
            ax[0, 0].axis('off') 
        
            # right top of the grid - graph with the calculated values of FSI
            ax[0, 1].set_title("Freeman Segregation Index")
            ax[0, 1].set_xlabel("Iterations")
            ax[0, 1].set_ylabel("FSI")
            ax[0, 1].set_xlim(0, no_iter)
            ax[0, 1].set_ylim(0, 1)
            # Draw the current value of measure 1 on top of the plot
            ax[0, 1].text(0.05, 0.95, "FSI: %.2f" %schelling.freeman_segregation_index(), fontsize=10)

            # Plot the entire list with the values of measure_1 (from time 0 to current time) on the graph
            ax[0, 1].plot(range(len(measureFSI)), measureFSI)

            # Left bottom of the grid - graph with results measure_2
            ax[1, 0].set_title("Fraction Same-Type Neighbours")
            ax[1, 0].set_xlabel("Iterations")
            ax[1, 0].set_ylabel("Fraction")
            ax[1, 0].set_xlim(0, no_iter)
            ax[1, 0].set_ylim(0, 1)       
            # Draw the current value of measure 2 on top of the plot
            ax[1, 0].text(0.05, 0.95, "FSTN: %.2f" %schelling.fraction_same_type_neighbours(), fontsize=10)

            # Plot the entire list with the values of measure_2 (from time 0 to current time) on the graph
            ax[1, 0].plot(range(len(measureFSTN)), measureFSTN)
        
            # Right bottom of the grid - graph with results measure_3
            ax[1, 1].set_title("Fraction Unhappy Agents")
            ax[1, 1].set_xlabel("Iterations")
            ax[1, 1].set_ylabel("Fraction")
            ax[1, 1].set_xlim(0, no_iter)
            ax[1, 1].set_ylim(0, 1)
            
            # Plot the entire list with the values of measure_3 (from time 0 to current time) on the graph
            ax[1, 1].plot(range(len(measureFUA)), measureFUA)
        
            # This is a call for the dummy method to count colours
            #ax[1, 1].text(1, 0.95, "FSI: %.2f" % schelling.counter_colour(0), fontsize=10)
        
            plot_container.pyplot(fig)     
        
            # Closes all the figure instances (to replot them the next time step)
            plt.close(fig)

            # Updates the progress bar
            iteration_text.text("Step %d" %(i+1))
            progress_bar.progress( (i+1.0)/no_iter )
            #--------------------------------------------------------------------

# In the second column
# Make calculations using the simulations
with col2:

    if st.button('Calculate'):
   
        # initialise the dataframe to store the output of the simulations
        # Here I implemented the saving routine only for FSI. You can create other dataframes for the other measures
        df_FSI = pd.DataFrame(index=range(no_iter+1),columns=range(no_sim))

        df_FSTN = pd.DataFrame(index=range(no_iter+1),columns=range(no_sim))
    
        # Why FUA has one less row?
        df_FUA = pd.DataFrame(index=range(no_iter),columns=range(no_sim))

        show_iteration = st.empty()
        show_iteration.text("Simulation 0")
        progress_bar = st.progress(0)

        # repeat the entire simulation "no_sim" times - input from the user via screen interface
        for m in range(no_sim):

            # Updates the progress bar
            show_iteration.text("Simulation %d" %(m+1))
            progress_bar.progress( (m+1.0)/no_sim )
        
            # Restart the grid
            p = [(1.0-empty_ratio)/2.0, (1.0-empty_ratio)/2.0, empty_ratio]
            schelling.grid = np.random.choice([-1, 1, 0], size = schelling.grid_size*schelling.grid_size, p = p)
            schelling.grid = np.reshape( schelling.grid, (schelling.grid_size, schelling.grid_size) )        
        
            # Initialise a list measure_FSI to store the Freeman Segregation Index over time
            measureFSI = []
            # Initialise a list measure_FSTN to store the Fraction of Same-Type Neighbours over time
            measureFSTN = []
            # Initialise a list measure_UA to store the Fraction of Unhappy Agents over time
            measureFUA = []

            # Measures at time = 0
            measureFSI.append(schelling.freeman_segregation_index())
            measureFSTN.append(schelling.fraction_same_type_neighbours())
        
            # Run the simulation for no_iter iterations, i.e. total time of the simulation
            # Repeat routines below for each time step i
            for i in range(no_iter):

                # Call the method "Run" with the interaction rules
                FUA = schelling.run()
                # Store the number of unhappy agents at time step t
                measureFUA.append( FUA )

                # Measure FSI and append the results on the list "measureFSI" at time step t+1
                measureFSI.append(schelling.freeman_segregation_index())
            
                # Measure FSTN and append the results on the list "measureFSTN" at time step t+1
                measureFSTN.append(schelling.fraction_same_type_neighbours())
 
            #--------------------------------------------------------------------
            # After each simulation, store all the values of the FSI measure
            df_FSI.iloc[:,m] = measureFSI
        
            # ======================================================================
            # ROUTINE TO SAVE THE CALCULATED MEASURES OF EACH SIMULATION INTO A DIFFERENT FILE
            # This can be useful if you have to analyse the distribution of outputs, for example
            #
            # Convert the list "measureFSI" to a pandas dataframe
            dfp = pd.DataFrame( measureFSI )

            # Save the dataframe to a "m" .csv file with data separated by space
            dfp.to_csv("./results_FSI_s"+str(m)+".csv", index = False, header = False, sep = ' ')
            # ======================================================================

        # ======================================================================
        # ROUTINE TO SAVE THE MEAN AND ST_DEV OF THE CALCULATED MEASURE OF ALL SIMULATION INTO A DIFFERENT FILE
        # After completing all simulations, calculate the mean and variance
        # Another more efficient strategy is to make online calculations (see week 5)
        time_step = pd.DataFrame( range(0, no_iter+1) )
        mean = df_FSI.mean(axis = 1)
        stdev = df_FSI.std(axis = 1)

        # Save the mean and variance, for each time step, in a file
        df_all = pd.concat([time_step, mean, stdev], axis = 1)
        df_all.to_csv("./results_FSI_mean.csv", index = False, header = False, sep = ' ')