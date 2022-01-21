#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 18:25:39 2022

@author: hemerson
"""

"""
Functions for evaluating algorithmic performance and displaying it to the user.
"""

import matplotlib.pyplot as plt
import numpy as np

# TODO: add the testing function
# will need to incorporate a way of getting the agent action

def test_algorithm(env, model, seed=0, max_timesteps=480, sequence_length=80,
                   data_processing="condensed", pid_run=False, params):
    
    # Unpack the params
    default_basal = params.get("default_basal")  
    state_mean = params.get("state_mean")  
    state_std = params.get("state_std")  
    
    
    # initialise the arrays for data collection    
    rl_reward, rl_blood_glucose, rl_action = 0, [], []
    pid_reward, pid_blood_glucose, pid_action = 0, [], []
    rl_action, rl_meals = [], []

    # select the number of iterations    
    if pid_run: runs = 2
    else: runs = 1
    
    for ep in range(runs):
        
        # set the seed for the environment
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        random.seed(seed)    
        
        # Initialise the environment --------------------------------------
        
        # get the state
        insulin_dose = 1/3 * default_basal
        meal, done, bg_val = 0, False, env.reset()
        time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
        state = np.array([bg_val[0], meal, insulin_dose, time], dtype=np.float32)

        # get a suitable input
        state_stack = np.tile(state, (sequence_length + 1, 1))
        state_stack[:, 3] = (state_stack[:, 3] - np.arange(((sequence_length + 1) / 479), 0, -(1 / 479))[:sequence_length + 1]) * 479            
        state_stack[:, 3] = (np.around(state_stack[:, 3], 0) % 480) / 479         

        # get the action and reward stack
        action_stack = np.tile(np.array([insulin_dose], dtype=np.float32), (sequence_length + 1, 1))        
        reward_stack = np.tile(-calculate_risk(bg_val), (sequence_length + 1, 1))

        # get the meal history
        meal_history = np.zeros(int((3 * 60) / 3), dtype=np.float32)
        
        # intiialise pid parameters
        integrated_state = 0
        previous_error = 0
        timesteps = 0
        
        
        while not done and timesteps < max_timesteps:
            
            # Run the RL algorithm ------------------------------------------------------
            if ep == 0:
                
                # condense the state
                if data_processing == "condensed":
                                    
                    # Unpack the state
                    bg_vals, meal_vals, insulin_vals = state_stack[:, 0][::10], state_stack[:, 1], state_stack[:, 2]
                    
                    # calculate insulin and meals on board
                    decay_factor = np.arange(1 / (sequence_length + 2), 1, 1 / (sequence_length + 2))
                    meals_on_board, insulin_on_board = np.sum(meal_vals * decay_factor), np.sum(insulin_vals * decay_factor) 
                    
                    # create the state
                    state = np.concatenate([bg_vals, meals_on_board.reshape(1), insulin_on_board.reshape(1)])
                
                # get the state a sequence of specified length
                elif data_processing == "sequence":
                    state = state_stack[1:, :3].reshape(1, sequence_length, 3) 
                                            
                # Normalise the current state
                state = (state - state_mean) / state_std
                
                if self.data_processing == "sequence":
                    model.init_hidden(1)
                
                with torch.no_grad():
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    action = model(state)
                                        
                # Unnormalise action output  
                action_pred = (action.cpu().data.numpy().flatten() * self.action_std + self.action_mean)[0]
                
                # to stop subtracting from bolus when -ve
                action_pred = max(0, action_pred)
                player_action = action_pred
                

            # Run the pid algorithm ------------------------------------------------------
            else:                            
                player_action = self.PID_action(bg_val)

            # update the chosen action
            chosen_action = np.copy(player_action)

            # take meal bolus
            if meal > 0:                             
                chosen_action = float(chosen_action) + self.calculate_bolus(bg_val, meal_history, meal)

            # append the basal and bolus action
            action_stack = np.delete(action_stack, 0, 0)
            action_stack = np.vstack([action_stack, player_action])

            # step the simulator
            next_bg_val, _, done, info = env.step(chosen_action)
            reward = -self.calculate_risk(next_bg_val) 

            # get the rnn array format for state
            input_time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
            next_state = np.array([float(next_bg_val[0]), float(info['meal']), float(chosen_action), input_time], dtype=np.float32)   

            # update the state stacks
            next_state_stack = np.delete(state_stack, 0, 0)
            next_state_stack = np.vstack([next_state_stack, next_state]) 
            reward_stack = np.delete(reward_stack, 0, 0)
            reward_stack = np.vstack([reward_stack, np.array([reward], dtype=np.float32)])

            # add a termination penalty
            if done: 
                reward = -1e5
                break
        

if not training:  
            
            rnn = ""
            if self.data_processing == "sequence":
                rnn = "_RNN" 
            
            # load the learned model
            self.load_model('./Models/BC' + rnn + '_weights')            
            
        if self.data_processing == "condensed":
            model = self.model
            
        elif self.data_processing == "sequence":
            model = self.RNN_model            
            
        # TESTING --------------------------------------------------------------------------------------------

        test_seed = 0
        env = gym.make('simglucose-child1-v0')
        max_timesteps = 480

        runs = 2
        if self.pid_run:
            runs = 1

        # get the rl arrays
        rl_reward = 0
        rl_bg = []
        rl_insulin = []
        rl_action = []
        rl_meals = []

        # stop the pid re-running
        self.pid_run = True  
        
        for ep in range(runs):

            # reset the seed
            env.seed(test_seed)  
            np.random.seed(test_seed)
            torch.manual_seed(test_seed) 
            random.seed(test_seed)
            
            # Preprocess the state ----------------------------------------------------------

            # get the state
            insulin_dose = 1/3 * self.bas
            meal, done, bg_val = 0, False, env.reset()
            input_time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
            state = [bg_val[0], meal, insulin_dose, input_time]    
            state = np.array(state, dtype=np.float32)

            # get a suitable input
            state_stack = np.tile(state, (self.sequence_length + 1, 1))

            # ensure that the time is correct
            state_stack[:, 3] = (state_stack[:, 3] - np.arange(((self.sequence_length + 1) / 479), 0, -(1 / 479))[:self.sequence_length + 1]) * 479            
            state_stack[:, 3] = (np.around(state_stack[:, 3], 0) % 480) / 479         

            # get the action and reward stack
            action_stack = np.tile(np.array([insulin_dose], dtype=np.float32), (self.sequence_length + 1, 1))        
            reward_stack = np.tile(-self.calculate_risk(bg_val), (self.sequence_length + 1, 1))

            # get the meal history
            meal_history = np.zeros(int((3 * 60) / 3), dtype=np.float32)

            self.integrated_state = 0
            self.previous_error = 0
            timesteps = 0

            while not done and timesteps < max_timesteps:

                # Run the RL algorithm ------------------------------------------------------
                if ep == 0:
                    
                    if self.data_processing == "condensed":
                                        
                        # Process the current state
                        bg_vals = state_stack[:, 0][::10]                    
                        meal_vals = state_stack[:, 1]
                        insulin_vals = state_stack[:, 2]
                        meals_on_board = np.sum(meal_vals * np.arange(1 / (self.sequence_length + 2), 1, 1/(self.sequence_length + 2)))
                        insulin_on_board = np.sum(insulin_vals * np.arange(1 / (self.sequence_length + 2), 1, 1/(self.sequence_length + 2)))                     
                        state = np.concatenate([bg_vals, meals_on_board.reshape(1), insulin_on_board.reshape(1)])
                        
                    elif self.data_processing == "sequence":
                        state = state_stack[1:, :3].reshape(1, self.sequence_length, 3) 
                                                
                    # Normalise the current state
                    state = (state - self.state_mean) / self.state_std
                    
                    if self.data_processing == "sequence":
                        model.init_hidden(1)
                    
                    with torch.no_grad():
                        state = torch.tensor(state, dtype=torch.float32, device=self.device)
                        action = model(state)
                                            
                    # Unnormalise action output  
                    action_pred = (action.cpu().data.numpy().flatten() * self.action_std + self.action_mean)[0]
                    
                    # to stop subtracting from bolus when -ve
                    action_pred = max(0, action_pred)
                    player_action = action_pred

                # Run the pid algorithm ------------------------------------------------------
                else:                            
                    player_action = self.PID_action(bg_val)

                # update the chosen action
                chosen_action = np.copy(player_action)

                # take meal bolus
                if meal > 0:                             
                    chosen_action = float(chosen_action) + self.calculate_bolus(bg_val, meal_history, meal)

                # append the basal and bolus action
                action_stack = np.delete(action_stack, 0, 0)
                action_stack = np.vstack([action_stack, player_action])

                # step the simulator
                next_bg_val, _, done, info = env.step(chosen_action)
                reward = -self.calculate_risk(next_bg_val) 

                # get the rnn array format for state
                input_time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
                next_state = np.array([float(next_bg_val[0]), float(info['meal']), float(chosen_action), input_time], dtype=np.float32)   

                # update the state stacks
                next_state_stack = np.delete(state_stack, 0, 0)
                next_state_stack = np.vstack([next_state_stack, next_state]) 
                reward_stack = np.delete(reward_stack, 0, 0)
                reward_stack = np.vstack([reward_stack, np.array([reward], dtype=np.float32)])

                # add a termination penalty
                if done: 
                    reward = -1e5
                    break

                # ---------------------------------------

                # for RL agent
                if ep == 0:
                    rl_bg.append(next_bg_val[0])
                    rl_action.append(player_action)
                    rl_insulin.append(chosen_action)
                    rl_reward += reward
                    rl_meals.append(info['meal'])

                # for pid agent
                else:
                    self.pid_bg.append(next_bg_val[0])
                    self.pid_insulin.append(chosen_action)
                    self.pid_action.append(player_action)
                    self.pid_reward += reward
                    
                # -----------------------------------------

                # update the meal history
                meal_history = np.append(meal_history, meal)
                meal_history = np.delete(meal_history, 0)   

                # update the state stacks
                state_stack = next_state_stack

                # update the state
                bg_val = next_bg_val
                state = next_state     
                meal = info['meal']
                timesteps += 1 

"""
Plot a four-tiered graph comparing the blood glucose control of a PID 
and RL algorithm, showing the blood glucose, insulin doses and meal 
carbohyrdates.
"""

def create_graph(rl_reward, rl_blood_glucose, rl_action, rl_insulin, rl_meals,
                 pid_reward, pid_blood_glucose, pid_action, default_basal):
    
    # TODO: add the function for printing metrics like TIR for PID vs RL
    
    # Display the reward results
    print('PID Reward: {} - RL Reward: {}'.format(pid_reward, rl_reward))
    
    # Check that the rl algorithm completed the full episode
    if len(pid_blood_glucose) == len(rl_blood_glucose):
                
        # get the x-axis 
        x = list(range(len(pid_reward)))
        
        # Initialise the plot and specify the title
        fig = plt.figure(dpi=160)
        gs = fig.add_gridspec(4, hspace=0.0)
        axs = gs.subplots(sharex=True, sharey=False)        
        fig.suptitle('Blood Glucose Control Algorithm Comparison')
        
        # define the hypo, eu and hyper regions
        axs[0].axhspan(180, 500, color='lightcoral', alpha=0.6, lw=0)
        axs[0].axhspan(70, 180, color='#c1efc1', alpha=1.0, lw=0)
        axs[0].axhspan(0, 70, color='lightcoral', alpha=0.6, lw=0)
        
        # plot the blood glucose values
        axs[0].plot(x, pid_blood_glucose, label='pid', color='orange')
        axs[0].plot(x, rl_blood_glucose, label='rl', color='dodgerblue')
        axs[0].legend(bbox_to_anchor=(1.0, 1.0))
        
        # specify the limits and the axis lables
        axs[0].axis(ymin=50, ymax=500)
        axs[0].axis(xmin=0.0, xmax=len(pid_reward))
        axs[0].set_ylabel("BG \n(mg/dL)")
        axs[0].set_xlabel("Time \n(mins)")
        
        # show the basal doses
        axs[1].plot(x, pid_action, label='pid', color='orange')
        axs[1].plot(x, rl_action, label='rl', color='dodgerblue')
        axs[1].axis(ymin=0.0, ymax=(default_basal * 1.4))
        axs[1].set_ylabel("Basal \n(U/min)")

        # show the bolus doses
        axs[2].plot(x, rl_insulin)
        axs[2].axis(ymin=0.01, ymax=0.99)
        axs[2].set_ylabel("Bolus \n(U/min)")

        # show the scheduled meals
        axs[3].plot(x, rl_meals)
        axs[3].axis(ymin=0, ymax=29.9)
        axs[3].set_ylabel("CHO \n(g/min)")

        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()

        plt.show()
    
    # specify the timesteps before termination
    else: print('Terminated after: {} timesteps.'.format(len(rl_blood_glucose)))