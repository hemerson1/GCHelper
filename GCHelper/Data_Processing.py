#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:48:31 2022

@author: hemerson
"""

"""
Functions for converting the replay output into the correct form for the 
chosen agent.
"""

import numpy as np
import random, torch

"""
Converts a list of trajectories gathered from the data_colleciton algorithm into
a replay with samples appropriate for model training along with state and action
means and stds.
"""
def unpackage_replay(trajectories, empty_replay, data_processing="condensed", sequence_length=80):
    
    # initialise the data lists
    states, rewards, actions, dones = [], [], [], []
    
    for path in trajectories: 
        
        # states include blood glucose, meal carbs, insulin dose, time 
        states += path['state']
        rewards += path['reward']
        actions += path['action']  
        dones += path['done']
        
        # ensure that the last state is always a terminal state
        dones[-1] = True             
        
    # initialise the lists
    processed_states, processed_next_states, processed_rewards, processed_actions, processed_dones  = [], [], [], [], []
    decay_state = np.arange(1 / (sequence_length + 2), 1, 1 / (sequence_length + 2))
    counter = 0 

    # Condense the state -------------------------------------------------

    # 4hr | 3.5hr | 3hr | 2.5hr | 2hr | 1.5hr | 1hr | 0.5hr | 0hr | meal_on_board | insulin_on_board
    if data_processing == "condensed":

        for idx, state in enumerate(states):

            # if there are 80 states previously
            if counter >= (sequence_length) and idx + 1 != len(states):

                # add rewards, actions and dones
                processed_rewards.append(rewards[idx])
                processed_actions.append(actions[idx])
                processed_dones.append(dones[idx])

                # current state -----------------------------------------

                # unpackage the values
                related_states = states[idx - sequence_length: idx + 1]  
                related_bgs, related_meals, related_insulins, _ = zip(*related_states)

                # extract the correct metrics
                extracted_bg = related_bgs[::10]
                meals_on_board = [np.sum(np.array(related_meals) * decay_state)]
                insulin_on_board = [np.sum(np.array(related_insulins) * decay_state)]

                # append the state
                processed_state = list(extracted_bg) + meals_on_board + insulin_on_board
                processed_states.append(processed_state)   

                # next state -----------------------------------------

                # unpackage the values
                related_next_states = states[(idx - sequence_length) + 1: idx + 1 + 1]  
                related_next_bgs, related_next_meals, related_next_insulins, _ = zip(*related_next_states)

                # extract the correct metrics
                extracted_next_bg = related_next_bgs[::10]
                next_meals_on_board = [np.sum(np.array(related_next_meals) * decay_state)]
                next_insulin_on_board = [np.sum(np.array(related_next_insulins) * decay_state)]  

                # append the state
                processed_next_state = list(extracted_next_bg) + next_meals_on_board + next_insulin_on_board
                processed_next_states.append(processed_next_state) 

            # update the counter
            counter += 1
            if dones[idx]:
                counter = 0   


    # Create a sequence -------------------------------------------------

    elif data_processing == "sequence":        

        for idx, state in enumerate(states):

            # if there are 80 states previously
            if counter >= (sequence_length) and idx + 1 != len(states):

                # add rewards, actions and dones
                processed_rewards.append(rewards[idx - sequence_length:idx])
                processed_actions.append(actions[idx - sequence_length:idx])
                processed_dones.append(dones[idx - sequence_length:idx])

                extracted_states = [state[:3] for state in states[idx - sequence_length:idx]]
                processed_states.append(extracted_states)

            # update the counter
            counter += 1
            if dones[idx]:
                counter = 0   


    # Normalisation ------------------------------------------------------
    array_states = np.array(processed_states)
    array_actions = np.array(processed_actions)        

    if data_processing == "condensed":

        # ensure the state mean and std are consistent across blood glucose
        state_mean, state_std = np.mean(array_states, axis=0), np.std(array_states, axis=0)
        action_mean, action_std = np.mean(array_actions, axis=0), np.std(array_actions, axis=0)             
        state_mean[:-2], state_std[:-2]  = state_mean[0], state_std[0]    

    elif data_processing == "sequence":

        # reshape array and calculate mean and std
        state_size, action_size = array_states.shape[2], array_actions.shape[2] 
        state_mean = np.mean(array_states.reshape(-1, state_size), axis=0)
        state_std = np.std(array_states.reshape(-1, state_size), axis=0)
        action_mean = np.mean(array_actions.reshape(-1, action_size), axis=0)
        action_std = np.std(array_actions.reshape(-1, action_size), axis=0)                     

    # load in new replay ----------------------------------------------------

    for idx, state in enumerate(processed_states):
        empty_replay.append((state, processed_actions[idx], processed_rewards[idx], processed_next_states[idx], processed_dones[idx]))
    full_replay = empty_replay

    return full_replay, state_mean, state_std, action_mean, action_std

"""
Extracts a batch of data from the full replay and puts it in an appropriate form
"""    
def get_batch(replay, batch_size, data_processing="condensed", sequence_length=80, device='cpu', params=None):
    
    # Environment
    state_size = params.get("state_size")  
    state_mean = params.get("state_mean")  
    state_std = params.get("state_std")  
    action_mean = params.get("action_mean")  
    action_std = params.get("action_std")  
    
    # sample a minibatch
    minibatch = random.sample(replay, batch_size)
    
    if data_processing == "condensed":
        state = np.zeros((batch_size, state_size), dtype=np.float32)
        action = np.zeros(batch_size, dtype=np.float32)        
        reward = np.zeros(batch_size, dtype=np.float32)
        next_state = np.zeros((batch_size, state_size), dtype=np.float32)
        done = np.zeros(batch_size, dtype=np.uint8)
                
    elif data_processing == "sequence":
        
        state = np.zeros((batch_size, sequence_length, state_size), dtype=np.float32)
        action = np.zeros(batch_size, sequence_length, dtype=np.float32)        
        reward = np.zeros(batch_size, sequence_length, dtype=np.float32)
        next_state = np.zeros((batch_size, sequence_length, state_size), dtype=np.float32)
        done = np.zeros(batch_size, sequence_length, dtype=np.uint8)            

    # unpack the batch
    for i in range(len(minibatch)):
        state[i], action[i], reward[i], next_state[i], done[i] = minibatch[i]  
    
    # convert to torch
    state = torch.FloatTensor((state - state_mean) / state_std).to(device)
    action = torch.FloatTensor((action - action_mean) / action_std).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_state = torch.FloatTensor((next_state - state_mean) / state_std).to(device)
    done = torch.FloatTensor(1 - done).to(device)
                
    # Modify Dimensions
    action = action.unsqueeze(1)
    reward = reward.unsqueeze(1)
    
    return state, action, reward, next_state, done
            
            
        
    
    
    
    
    
    
