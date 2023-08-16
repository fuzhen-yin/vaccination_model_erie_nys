# -*- coding: utf-8 -*-
"""
ABM Vaccination Model for Erie County

Agent-based model using MESA framework

Script 1/1: ABM model 
Created on Tue July. 2023
"""
#%% Library
# working directory 
from os import chdir, getcwd
# specify working directory 
# chdir('FILE LOC OF DATA')

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn-v0_8-white')

from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import random
import networkx as nx
import datetime
from datetime import date

## MESA
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation
from mesa.space import MultiGrid, NetworkGrid
from mesa.space import ContinuousSpace
# analysis modules
from mesa.datacollection import DataCollector 
from mesa.batchrunner import BatchRunner # vary model parameter (e.g., n_agents)
# visualization modules
from mesa.visualization.modules import CanvasGrid 
from mesa.visualization.ModularVisualization import ModularServer

# Mesa Local Module: for visualization
from eg_mesa.boid_flockers.SimpleContinuousModule import SimpleCanvas

## Setting
pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.4f}'.format # specifies default number format to 4 decimal places
plt.style.use('ggplot') # specifies that graphs should use ggplot styling

#%% DATA1 date tick correpondence. 
# Generate date vs tick. Temporal resolution = week (Laurin 2018)
tick_day = 7
df_tick = pd.DataFrame(
    {"date": pd.date_range("2021-01-01", "2022-05-19", freq=(str(tick_day)+"D")).strftime("%Y-%m-%d"),
      "tick":range(int(np.ceil(500/tick_day))),
     "age_5":0,
     "age_12":0,
     "age_16":0,
     "age_30":0,
     "age_50":0,
     "age_60":0,
     "age_65":0,
     "age_75":0      }
    )

# NYS vaccine adminstration plan (NYS 2021a,b)
df_tick.loc[df_tick["date"] >= "2021-12-01", "age_5"] = 1
df_tick.loc[df_tick["date"] >= "2021-05-19", "age_12"] = 1
df_tick.loc[df_tick["date"] >= "2021-04-06", "age_16"] = 1
df_tick.loc[df_tick["date"] >= "2021-03-30", "age_30"] = 1
df_tick.loc[df_tick["date"] >= "2021-03-22", "age_50"] = 1
df_tick.loc[df_tick["date"] >= "2021-03-10", "age_60"] = 1
df_tick.loc[df_tick["date"] >= "2021-01-23", "age_65"] = 1
df_tick.loc[df_tick["date"] >= "2021-01-11", "age_75"] = 1

## get the tick when different age groups are eligible for vaccine
df_tick_age = pd.DataFrame({
    "age" : range (85),
    "tick" : 999
    })

df_tick_age.age = 1 + df_tick_age.age
for col in df_tick.iloc[:, 2:].columns:
    print(col)
    x = int(col.split("_")[1])
    xx = min(df_tick.loc[df_tick[col] >0, "tick"])
    df_tick_age.loc[df_tick_age['age'] >= x, "tick"] = xx
#%% DATA2 Syn ind-hh-ntwk

# Read in data: synthetic population data & hybrid space network data
syn_ind = pd.read_csv ('data/00_model_input_data/00_finl_ind_NAD83_20230718.csv', dtype=object)
syn_ind = syn_ind.rename({'id': 'ind_id', 'sex':'gender', 'urban':'urban_rural'}, axis='columns')

ntwk_all = pd.read_csv("data/00_model_input_data/00_finl_ntwk_20230718.csv", dtype = object)

syn_ind.age = pd.to_numeric(syn_ind.age)
syn_ind.long = pd.to_numeric(syn_ind.long)
syn_ind.lat = pd.to_numeric(syn_ind.lat)

syn_ind["xlab"] = syn_ind.long - min(syn_ind.long)
syn_ind["ylab"] = max(syn_ind.lat) - syn_ind.lat

# identify employed individual 
syn_ind['working_status'] = syn_ind["wp"].str.slice(start = 11, stop = 12)
syn_ind['if_employed'] = 0 
syn_ind.loc[syn_ind['working_status']=="w", "if_employed"] = 1
x = syn_ind.loc[syn_ind['working_status']=="w"]

#%% Functions 

# Random samples X% of syn_ind, and designate Y% working adults as essential workers who are eligble for vaccination since day 0 
def agent_network_generator(sample_size_perg, essential_wrker_perg):
    lt = random.sample(range(len(syn_ind)), round(sample_size_perg * len(syn_ind)))
    eg_syn_ind = syn_ind.iloc[lt].reset_index(drop=True)
    eg_syn_ind['ind_new_id'] = range(len(eg_syn_ind))
    eg_syn_ind["if_ntwk"] = 0
    
    # Random 18.2% of workers as essential worker 
    eg_syn_ind['if_essential_worker'] = 0
    lt_employed = list(eg_syn_ind.loc[(eg_syn_ind['if_employed']=="1") & (eg_syn_ind['age']>= 18)].index)
    lt_essential = random.sample(lt_employed, round(essential_wrker_perg * len(lt_employed)))
    eg_syn_ind.loc[lt_essential, "if_essential_worker"] = 1
    
    ## Tick: when individual is eligible for vax 
    eg_syn_ind = eg_syn_ind.merge(df_tick_age, on="age", how="left")
    eg_syn_ind.loc[eg_syn_ind['if_essential_worker'] > 0, "tick"] = 1
    
    # Network df
    eg_ntwk = ntwk_all.loc[(ntwk_all['Source'].isin(eg_syn_ind.ind_id)) & 
                           (ntwk_all['Target'].isin(eg_syn_ind.ind_id))].reset_index(drop=True)
    eg_ntwk = eg_ntwk.merge(eg_syn_ind[['ind_id','ind_new_id']], left_on='Source', right_on='ind_id', how='left')
    eg_ntwk['source_reindex'] = eg_ntwk['ind_new_id']
    eg_ntwk = eg_ntwk.drop(columns=['ind_id',"ind_new_id"])
    
    eg_ntwk = eg_ntwk.merge(eg_syn_ind[['ind_id','ind_new_id']], left_on="Target", right_on="ind_id", how='left')
    eg_ntwk['target_reindex'] = eg_ntwk['ind_new_id']
    eg_ntwk = eg_ntwk.drop(columns=['ind_id',"ind_new_id"])
    return eg_syn_ind, eg_ntwk

## F1 - Agent - Attributes Initialization 
# agent's susceptibility score (0, 1). uniform distribution
def ini_suscptby ():
    a = random.uniform(0,1) 
    return a

# y1, initial opinion towards vaccine 
def ini_opinin_vx():
    a = random.uniform(-1, 1)
    return a

## F2 - Model Level Functions
# Multimodel Networks: split network dataframe and produce hybrid space networks 
# physical (family+group quarter), relational (school+work), cyber (socialmedia)
def init_empty_agent_list(graph):
    for node_id in graph.nodes:
        graph.nodes[node_id]["agent"]=list()
    return graph

def multimodel_net(df_ntwk):
    # split networks based on relations: family, school and social media network
    df_family = df_ntwk[df_ntwk.Relation.isin(['Family','gq','hhold'])][['source_reindex', 'target_reindex', 'Relation']]
    df_work = df_ntwk[df_ntwk.Relation.isin(['School','Work','daycare','school','work'])][['source_reindex', 'target_reindex', 'Relation']]
    df_smedia = df_ntwk[df_ntwk.Relation.isin(['SocialMedia_teen','SocialMedia'])][['source_reindex', 'target_reindex', 'Relation']]
    
    # convert from df to networkx.graph object    
    g_family, g_work, g_smedia = nx.Graph(), nx.Graph(), nx.Graph()
    
    g_family = nx.from_pandas_edgelist(df_family, source='source_reindex', target='target_reindex', edge_attr=True)
    g_family = init_empty_agent_list(g_family)

    g_work = nx.from_pandas_edgelist(df_work, source='source_reindex', target='target_reindex', edge_attr=True)
    g_work = init_empty_agent_list(g_work)
    
    g_smedia = nx.from_pandas_edgelist(df_smedia, source='source_reindex', target='target_reindex', edge_attr=True)
    g_smedia = init_empty_agent_list(g_smedia)
    
    return g_family, g_work, g_smedia

## F3 - Analysis of Modeling Result
# Plot simulated vaccination rate of all population
# df_model: simulated vaccination rate
# df_x:     ground truth vaccination rate 
def plot_all_simulated_vax_rate(df_model, df_x, file_name):
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot('x','p_all_vaxed', data = df_model,color="green", label="simulated")
    plt.plot('x','pop_pct_MA', data = df_x,color="red", label="observed")
    plt.ylabel("Simulated Vaccinated Rate (%)")
    plt.tight_layout()
    plt.legend()
    plt.ylim(0, 100)
    title = ("All_pop_Family_Work_SocialMedia____" + str(file_name))
    plt.suptitle(title)
    
    loc = ("plot/xx_"+ date.today().strftime("%Y_%m_%d") + "_" + title + ".pdf")
    plt.savefig(loc)  

# Plot simulated vaccination rate of different age groups: all, 12+, 18+, 65+
def plot_group_simulated_vax_rate(df_model, df_x, file_name):
    title = ("Family_Work_SocialMedia____" + str(file_name))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(title)
    
    ax1.plot(df_model.x, df_model.p_5_11_vaxed, color="green", label="predicted")
    ax1.plot(df_x.x, df_x.pct_5_11_MA, color="red", label="observed")
    ax1.set_title("age 5-11")
     
    ax2.plot(df_model.x, df_model.p_12_17_vaxed, color="green", label="predicted")
    ax2.plot(df_x.x, df_x.pct_12_17_MA, color="red", label="observed")
    ax2.set_title("age 12-17")
    
    ax3.plot(df_model.x, df_model.p_18_64_vaxed, color="green", label="predicted")
    ax3.plot(df_x.x, df_x.pct_18_64_MA, color="red", label="observed")
    ax3.set_title("age 18-64")
    
    ax4.plot(df_model.x, df_model.p_65plus_vaxed, color="green", label="predicted")
    ax4.plot(df_x.x, df_x.pct_65plus_MA, color="red", label="observed")
    ax4.set_title("age 65+")
    
        
    
    title = ("Family_Work_SocialMedia____" + str(file_name))
    plt.suptitle(title)
    plt.legend()
    
    loc = ("plot/xx_only_one_group_"+ date.today().strftime("%Y_%m_%d") +"_"+title+".pdf")
    plt.savefig(loc) 
#%% Read in ground truth vaccination timeseries
df_vax_real = pd.read_csv("data/01_vax_record/00_Erie_COVID19_Vaccinations_rate_pop_group.csv")
df_vax_real['Date'] = pd.to_datetime(df_vax_real['Date'],format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

df_vax_real = df_vax_real.sort_values(by=['Date'])

# simulation time period: Jan. 01, 2021 until May 15, 2022
df_vax_real = df_vax_real.loc[df_vax_real['Date']<="2022-05-15"] 
df_vax_real = df_vax_real.loc[df_vax_real['Date']>="2021-01-01"] 
df_vax_real = df_vax_real.reset_index(drop=True) 
df_vax_real['x'] = df_vax_real.index

## moving average - smooth
# time period for moving average 
moving_average_period = 7
# ground truth 
df_x = df_vax_real.loc[:, ('x',"Date","Dose1_Recip_pop_pct","Dose1_Recip_5_11_pct","Dose1_Recip_12_17_pct","Dose1_Recip_18_64_pct","Dose1_Recip_65Plus_pct")]

# calculate moving average 
df_x['pop_pct_MA'] = df_x.Dose1_Recip_pop_pct.rolling(moving_average_period, min_periods = 1).mean()
df_x['pct_5_11_MA'] = df_x.Dose1_Recip_5_11_pct.rolling(moving_average_period, min_periods = 1).mean()
df_x['pct_12_17_MA'] = df_x.Dose1_Recip_12_17_pct.rolling(moving_average_period, min_periods = 1).mean()
df_x['pct_18_64_MA'] = df_x.Dose1_Recip_18_64_pct.rolling(moving_average_period, min_periods = 1).mean()
df_x['pct_65plus_MA'] = df_x.Dose1_Recip_65Plus_pct.rolling(moving_average_period, min_periods = 1).mean()

#%% Agent Class
## Agent's opinion dynamics are defined based on social influence theory  (Friedkin & Johnsen, 1990) 
## Equation: y_i^{(t)} =  a_i \sum_{j=1}^{N}w_{ij} (y_j^{(t)}?y_j^{(t)}:y_j^{(t-1)}) + (1-a_i)y_i^{(0)}
class VaxAgent (Agent):
    # Initialization of agents
    def __init__(self, ind_id, model):
        super().__init__(ind_id, model)
        self.id = ind_id
        # opinion dynamics
        self.v_y1 = ini_opinin_vx()   # initial opinions about vax. [-1, 1]
        self.v_suscep = ini_suscptby() # susceptibility to neighbors' opinions
        # dependent variable
        self.v_status = 0
        self.v_yt_1 = self.v_y1
        self.v_yt = self.v_y1 # intention to vaccinate at time t
        # network
        self.n_total_neighbors = 0
        self.n_family_neighbors = 0
        self.n_wrk_neighbors = 0
        self.n_smedia_neighbors = 0
        self.avg_family_yt = 0
        self.avg_coworker_yt = 0
        self.avg_smedia_friend_yt = 0
        # weights of opinion sources
        self.w_y1 = 1
        self.w_glb_family = 0       # global weight at model level
        self.w_glb_work = 0         
        self.w_glb_smedia = 0       
        
        self.w_family = 0           # individual agent's weight
        self.w_work = 0
        self.w_smedia = 0
        
       
        # demographic attributes 
        self.age = []
        self.gender = []
        self.urban_rural = []
        # location
        self.home_xlab = []
        self.home_ylab = []
        
        # ticks when eligible 
        self.a_tick_eligible = 999
        
    def get_ntwk_neigbrs_id_list(self):
        ''' Return network neighbors list in order (1) family, (2) work and (3) social media '''
        lt_f, lt_w, lt_s = [], [], []
        ## family network 
        if self.model.G_family.has_node(self.id):
            lt_f = list(self.model.G_family.neighbors(self.id))
        ## work network 
        if self.model.G_work.has_node(self.id):
            lt_w = list(self.model.G_work.neighbors(self.id))
        ## social media network
        if self.model.G_smedia.has_node(self.id):
            lt_s = list(self.model.G_smedia.neighbors(self.id))
        return lt_f, lt_w, lt_s
    
    def get_ntwk_neigbrs_agent_list(self):
        """Return three lists of agent object: neighboring agents in family, work and social media networks"""
        # first get neighboring agents' id in the three networks 
        lt_f, lt_w, lt_s = self.get_ntwk_neigbrs_id_list()
        lt_agent_f, lt_agent_w, lt_agent_s = [],[],[]
        
        # get a list of agent objects based on their id
        # family -- neighboring agents 
        if len(lt_f) > 0:
            lt_agent_f = list(self.model.G_family.nodes[node_id]['agent'] for node_id in lt_f)
            lt_agent_f = [item for sublist in lt_agent_f for item in sublist]
        # work -- neighboring agents
        if len(lt_w) > 0:
            lt_agent_w = list(self.model.G_work.nodes[node_id]['agent'] for node_id in lt_w)
            lt_agent_w = [item for sublist in lt_agent_w for item in sublist]
        # social media -- neighboring agents
        if len(lt_s) > 0:
            lt_agent_s = list(self.model.G_smedia.nodes[node_id]['agent'] for node_id in lt_s)
            lt_agent_s = [item for sublist in lt_agent_s for item in sublist]
        
        return lt_agent_f, lt_agent_w, lt_agent_s

    
    def get_ntwk_neigbrs_avg_opinion(self):
        """ Return the average opinions of neighboring agents in the three networks.
        In order: family, work, and social media """
        # get neighboring agent objects in networks
        lt_agent_f, lt_agent_w, lt_agent_s = self.get_ntwk_neigbrs_agent_list()
        f_avg_score, w_avg_score, s_avg_score = 0, 0, 0
        
        # calculate mean opinions of neighboring agents at previous time (t-1)
        if len(lt_agent_f) > 0:
            f_avg_score = np.mean([agent.v_yt_1 for agent in lt_agent_f])
        if len(lt_agent_w) > 0:
            w_avg_score = np.mean([agent.v_yt_1 for agent in lt_agent_w])
        if len(lt_agent_s) > 0:
            s_avg_score = np.mean([agent.v_yt_1 for agent in lt_agent_s])
        return f_avg_score, w_avg_score, s_avg_score 
    
    def action_after_vaccination(self):
        self.v_yt = 1    
        self.v_yt_1 = 1
    
    def minor_both_parents_vaxed(self):
        if_parents_allow = False
        # get neighbors in family network 
        if self.model.G_family.has_node(self.id):
            lt_parent = list(self.model.G_family.neighbors(self.id))
            lt_agent_parent = list(self.model.G_family.nodes[node_id]['agent'] for node_id in lt_parent)
            lt_agent_parent = [item for sublist in lt_agent_parent for item in sublist]
            parent_vax_status = list(agent.v_status for agent in lt_agent_parent if (agent.age >= 18))
            parent_vax_status = np.mean(pd.to_numeric(parent_vax_status))
            if parent_vax_status == 1:
                if_parents_allow = True
        return if_parents_allow
            

    def step(self):
        # Only update those agents' opinions who didn't vaccinate
        if self.v_status == 0:
            self.avg_family_yt, self.avg_coworker_yt, self.avg_smedia_friend_yt = self.get_ntwk_neigbrs_avg_opinion()
            ## calculate weighted score 
            self.v_yt = np.sum([
                self.w_family * self.avg_family_yt,
                self.w_work * self.avg_coworker_yt,
                self.w_smedia * self.avg_smedia_friend_yt,
                self.w_y1 * self.v_y1  ])
            
            # check if me (agent) is eligible to take vaccine 
            if self.a_tick_eligible <= self.model.m_tick:
                
                # Adult agent (>=18) can make decision independently, otherwise, need both parents' permissions
                if self.age >=18:
                    # bernouli - binomial distribution
                    self.v_status = np.random.binomial(1,self.v_yt) if (self.v_yt > 0) else 0  
                else:
                    """need new test condition"""   
                    if self.minor_both_parents_vaxed():
                        self.v_status = np.random.binomial(1,self.v_yt) if (self.v_yt > 0) else 0  
                
                if self.v_status == 1:
                    self.action_after_vaccination()
            self.v_yt_1 = self.v_yt

#%% Model Class
class VaxModel (Model):
    def __init__(self, n_agents, width, height, 
                 weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11,
                 weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17,
                 weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64,
                 weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder ):
        super().__init__()
        # Model schedule & tick 
        self.m_tick = 0        
        self.schedule = RandomActivation(self)
        
        # Environment: Continuous Space
        self.space = ContinuousSpace(width, height, torus = False)
        
        # Initialize multimodel network with empty agent list 
        self.G_family, self.G_work, self.G_smedia, = multimodel_net(eg_ntwk)            
                
        # fill agents' attributes
        for i in range(n_agents):
            a = VaxAgent(eg_syn_ind.ind_new_id[i], self)
            # fill in attributes 
            a.age, a.gender, a.urban_rural = eg_syn_ind.age[i], eg_syn_ind.gender[i], eg_syn_ind.urban_rural[i]
            a.home_xlab, a.home_ylab = eg_syn_ind.xlab[i], eg_syn_ind.ylab[i]
            a.a_tick_eligible = eg_syn_ind.tick[i]
            
            # save global weight 
            if (a.age <= 11):
                a.w_glb_family = weight_family_ntwk_5_11
                a.w_glb_work = weight_work_ntwk_5_11
                a.w_glb_smedia = weight_smedia_ntwk_5_11
            if (a.age >= 12) & (a.age <= 17):
                a.w_glb_family = weight_family_ntwk_12_17
                a.w_glb_work = weight_work_ntwk_12_17
                a.w_glb_smedia = weight_smedia_ntwk_12_17
            if (a.age >= 18) & (a.age <= 64):
                a.w_glb_family = weight_family_ntwk_18_64
                a.w_glb_work = weight_work_ntwk_18_64
                a.w_glb_smedia = weight_smedia_ntwk_18_64
            if (a.age >= 65):
                a.w_glb_family = weight_family_ntwk_65_elder
                a.w_glb_work = weight_work_ntwk_65_elder
                a.w_glb_smedia = weight_smedia_ntwk_65_elder
            
            # location
            coords = (a.home_xlab, a.home_ylab)
            self.space.place_agent(a, coords)
            
            # Place the agent into the network 
            ## family 
            if self.G_family.has_node(a.id):
                self.G_family.nodes[a.id]["agent"].append(a)
                # print("add agent {} to family network".format(a.id))                
            ## work 
            if self.G_work.has_node(a.id):
                self.G_work.nodes[a.id]["agent"].append(a)
                # print("add agent {} to work network".format(a.id))
            ## social media
            if self.G_smedia.has_node(a.id):
                self.G_smedia.nodes[a.id]['agent'].append(a)
                # print("add agent {} to social media network".format(a.id))
            
            ## Initialization: count neighbors in family, work, and social media network 
            alt_f, alt_w, alt_s = a.get_ntwk_neigbrs_id_list()
            a.n_family_neighbors, a.n_wrk_neighbors, a.n_smedia_neighbors = len(alt_f), len(alt_w), len(alt_s)
            a.n_total_neighbors = np.sum([a.n_family_neighbors, a.n_wrk_neighbors, a.n_smedia_neighbors])
            ## Weight multimodal networks at agent level
            if a.n_total_neighbors > 0:
                weight_1 = a.w_glb_family if (a.n_family_neighbors > 0) else 0
                weight_2 = a.w_glb_work if (a.n_wrk_neighbors > 0) else 0
                weight_3 = a.w_glb_smedia if (a.n_smedia_neighbors > 0) else 0
                
                if np.sum([weight_1, weight_2, weight_3]) > 0:                
                    a.w_family = a.v_suscep * weight_1 / np.sum([weight_1, weight_2, weight_3])
                    a.w_work = a.v_suscep * weight_2 / np.sum([weight_1, weight_2, weight_3])
                    a.w_smedia = a.v_suscep * weight_3 / np.sum([weight_1, weight_2, weight_3])
                    a.w_y1 = 1-a.v_suscep

            # agent's schedule
            self.schedule.add(a)
        
        # data collector 
        self.datacollector = DataCollector(
            model_reporters={"model_tick":lambda m: m.m_tick,
                             'fmily_n_nodes': lambda m: len(m.G_family.nodes),
                             'fmily_n_edges': lambda m: len(m.G_family.edges),
                             'work_n_nodes': lambda m: len(m.G_work.nodes),
                             'work_n_edges': lambda m: len(m.G_work.edges),
                             'smedia_n_nodes': lambda m: len(m.G_smedia.nodes),
                             'smedia_n_edges': lambda m: len(m.G_smedia.edges),
                             'count_12_17':lambda m: m.n_agent_12_17,
                             'count_18_64':lambda m: m.n_agent_18_64,
                             'count_65plus':lambda m: m.n_agent_65plus,
                             
                             'n_all_vaxed': lambda m: m.vax_n,
                             'n_5_11_vaxed': lambda m: m.vax_5_11_n,
                             'n_12_17_vaxed': lambda m: m.vax_12_17_n,
                             'n_18_64_vaxed': lambda m: m.vax_18_64_n,
                             'n_65plus_vaxed': lambda m: m.vax_65plus_n,
                             
                             'p_all_vaxed': lambda m: m.vax_rate,
                             'p_5_11_vaxed': lambda m: m.vax_5_11_rate,
                             'p_12_17_vaxed': lambda m: m.vax_12_17_rate,
                             'p_18_64_vaxed': lambda m: m.vax_18_64_rate,
                             'p_65plus_vaxed': lambda m: m.vax_65plus_rate,
                             },
            agent_reporters={"agent_age":"age",
                             'tick_eligibility':'a_tick_eligible',
                             'vax_status':'v_status',
                             'susceptibility':'v_suscep',
                             'initial_opinion':'v_y1',                             
                             'opinion_time_t':'v_yt',
                             'total_degree':'n_total_neighbors',
                             'family_degree':'n_family_neighbors',
                             'work_degree':'n_wrk_neighbors',
                             'smedia_degree':'n_smedia_neighbors',
                             'global_w_family':'w_glb_family',
                             'global_w_work':'w_glb_work',
                             'global_w_smedia':'w_glb_smedia',
                             'initial_opinion_weight':'w_y1',
                             'fmily_ntwk_weight':'w_family',
                             'work_ntwk_weight':'w_work',
                             'smedia_ntwk_weight':'w_smedia',
                             'fmily_avg_opinion':'avg_family_yt',
                             'cowork_avg_opinion':'avg_coworker_yt',
                             'smedia_avg_opinion':'avg_smedia_friend_yt'})
        
        # Initialize global variable 
        self.n_agent_5_11 = len(list(agent for agent in self.schedule.agents if (agent.age >= 5) & (agent.age <= 11)))
        self.n_agent_12_17 = len(list(agent for agent in self.schedule.agents if (agent.age >= 12) & (agent.age <= 17)))
        self.n_agent_18_64 = len(list(agent for agent in self.schedule.agents if (agent.age >= 18) & (agent.age <= 64)))
        self.n_agent_65plus = len(list(agent for agent in self.schedule.agents if (agent.age >= 65)))
        
        # global vars to update every step - vaccinated pop 
        self.vax_n = 0
        self.vax_5_11_n = 0
        self.vax_12_17_n = 0
        self.vax_18_64_n = 0
        self.vax_65plus_n = 0
        
        self.vax_rate = 0
        self.vax_5_11_rate = 0
        self.vax_12_17_rate = 0
        self.vax_18_64_rate = 0
        self.vax_65plus_rate = 0
        
    def step(self):
        # Step
        print("Step: "+str(self.m_tick))
        self.schedule.step()
                
        # update model level results
        self.vax_n = np.sum([agent.v_status for agent in self.schedule.agents])
        self.vax_5_11_n = np.sum([agent.v_status for agent in self.schedule.agents if (agent.age >= 5) & (agent.age <= 11)])
        self.vax_12_17_n = np.sum([agent.v_status for agent in self.schedule.agents if (agent.age >= 12) & (agent.age <= 17)])
        self.vax_18_64_n = np.sum([agent.v_status for agent in self.schedule.agents if (agent.age >= 18) & (agent.age <= 64)])
        self.vax_65plus_n = np.sum([agent.v_status for agent in self.schedule.agents if agent.age >= 65])
        
        self.vax_rate = self.vax_n / n_agents
        self.vax_5_11_rate = self.vax_5_11_n / self.n_agent_5_11
        self.vax_12_17_rate = self.vax_12_17_n / self.n_agent_12_17
        self.vax_18_64_rate = self.vax_18_64_n / self.n_agent_18_64
        self.vax_65plus_rate = self.vax_65plus_n / self.n_agent_65plus
        # collect data
        self.datacollector.collect(self)
        # tick 
        self.m_tick = self.m_tick + 1    

#%% Function: Running the model 
def run_vax_model(n_agent, canvas_w, canvas_h, n_step,
                  weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11,
                  weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17,
                  weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64,
                  weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder ):
    # initialize model
    my_model = VaxModel(n_agent,canvas_w,canvas_h, 
                        weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11,
                        weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17,
                        weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64,
                        weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder)
    # run model
    for i in range(n_step):
        my_model.step()
    # get model output 
    df_m = my_model.datacollector.get_model_vars_dataframe()
    df_m['tick'] = df_m.index
    #df_a = my_model.datacollector.get_agent_vars_dataframe()
    # return model object, model level outputs and agent level output
    return my_model, df_m  #, df_a

#%% Parameter initialization
eg_syn_ind, eg_ntwk = agent_network_generator(sample_size_perg = 1, essential_wrker_perg = 0.182)
n_agents = len(eg_syn_ind)

# ==== Initial model parameter: Test 3:1:1 ====
weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11 = 3,1,1
weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17 = 3,1,1
weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64 = 3,1,1
weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder = 3,1,1

# ## ==== final model parameter: a_100_b_113_c_311_d_131 ====
# weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11 = 1,0,0
# weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17 = 1,1,3
# weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64 = 3,1,1
# weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder = 1,3,1

n_step = 73 # each step represents 7 days
#%% Model Run 1 times
model_test, df_m_output = run_vax_model(n_agents,60000,74000,n_step, 
                                                     weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11,
                                                     weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17,
                                                     weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64,
                                                     weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder)
# , df_a_output
# xx = df_a_output
# xx = xx.loc[xx.index.get_level_values("Step") == n_step]
# xx["ind_new_id"] = xx.index.get_level_values("AgentID")
# xx = xx.merge(eg_syn_ind[["ind_id","ind_new_id"]], on="ind_new_id", how="left", indicator=False)

filename = ("a_"+str(weight_family_ntwk_5_11)+str(weight_work_ntwk_5_11)+str(weight_smedia_ntwk_5_11)+ "_" + 
            "b_"+str(weight_family_ntwk_12_17)+str(weight_work_ntwk_12_17)+str(weight_smedia_ntwk_12_17)+ "_" + 
            "c_"+str(weight_family_ntwk_18_64)+str(weight_work_ntwk_18_64)+str(weight_smedia_ntwk_18_64)+ "_" + 
            "d_"+str(weight_family_ntwk_65_elder)+str(weight_work_ntwk_65_elder)+str(weight_smedia_ntwk_65_elder))

df_m_output.to_csv(("plot/01_test/xx_model_"+str(filename)+".csv"))
# xx.to_csv(("plot/01_test/xx_agent_"+str(filename)+".csv"))

# for visualization 
yy = df_m_output.loc[:, ["tick","p_all_vaxed","p_5_11_vaxed","p_12_17_vaxed","p_18_64_vaxed","p_65plus_vaxed"]]
yy.loc[:, ["p_all_vaxed","p_5_11_vaxed","p_12_17_vaxed","p_18_64_vaxed","p_65plus_vaxed"]] = 100 * yy.loc[:, ["p_all_vaxed","p_5_11_vaxed","p_12_17_vaxed","p_18_64_vaxed","p_65plus_vaxed"]]
yy['x'] = yy.tick * tick_day

# calculate error
error_simu_groundtruth(yy, df_x, tick_day)
plot_all_simulated_vax_rate(yy, df_x,filename)
plot_group_simulated_vax_rate(yy, df_x,filename)

#%% （Stochastisity） Model Run XX times
lt_timestep = [11, 14, 19, 48] # correspondent global vax rates are: 15%, 30%, 45%, 60%

for j in range (1):
    print("Step_"+str(j))
    
    # prepare dataset 
    eg_syn_ind, eg_ntwk = agent_network_generator(sample_size_perg = 1, essential_wrker_perg = 0.182)
    n_agents = len(eg_syn_ind)
    
    ## Model Run
    model_test, df_m_output, df_a_output = run_vax_model(n_agents,60000,64000,n_step, 
                                                         weight_family_ntwk_5_11, weight_work_ntwk_5_11, weight_smedia_ntwk_5_11,
                                                         weight_family_ntwk_12_17, weight_work_ntwk_12_17, weight_smedia_ntwk_12_17,
                                                         weight_family_ntwk_18_64, weight_work_ntwk_18_64, weight_smedia_ntwk_18_64,
                                                         weight_family_ntwk_65_elder, weight_work_ntwk_65_elder, weight_smedia_ntwk_65_elder )
    ## a new df saving AGENT attributes at the last time step 
    xx = pd.DataFrame()
    xx = df_a_output.loc[df_a_output.index.get_level_values("Step") == n_step]
    xx["ind_new_id"] = xx.index.get_level_values("AgentID")
    xx = xx.merge(eg_syn_ind[["ind_id","ind_new_id"]], on="ind_new_id", how="left", indicator=False)

    ## prepare file name
    filename = ("weight_a_"+str(weight_family_ntwk_5_11)+str(weight_work_ntwk_5_11)+str(weight_smedia_ntwk_5_11)+ "_" + 
                "b_"+str(weight_family_ntwk_12_17)+str(weight_work_ntwk_12_17)+str(weight_smedia_ntwk_12_17)+ "_" + 
                "c_"+str(weight_family_ntwk_18_64)+str(weight_work_ntwk_18_64)+str(weight_smedia_ntwk_18_64)+ "_" + 
                "d_"+str(weight_family_ntwk_65_elder)+str(weight_work_ntwk_65_elder)+str(weight_smedia_ntwk_65_elder)+ 
                 "_run_"+ str(j) + "_tstep_"+str(tick_day)+"_days")
    ## EXPORT 
    ## export MODEL outcomes: vaccination rate of different age groups
    df_m_output.to_csv(("plot/11_model_0409_verification_validation/model_"+str(filename)+".csv"))
    ## export AGENT outcomes at the last time step: a1, y0, yt, degrees... 
    xx.to_csv(("plot/11_model_0409_verification_validation/agent_simplified_"+str(filename)+".csv"))
    ## export AGENT outcomes of all time step: a1, y0, yt, degrees... ONLY RUN THIS IF NEED. Very time consuming. 
    
    for k in range(4):
        p = lt_timestep[k]+1      
        tt = df_a_output.loc[df_a_output.index.get_level_values("Step") == p]
        tt["ind_new_id"] = tt.index.get_level_values("AgentID")
        tt = tt.merge(eg_syn_ind[["ind_id","ind_new_id"]], on="ind_new_id", how="left", indicator=False)
        tt.to_csv(("plot/11_model_0409_verification_validation/agent_simplified_step"+ str(p)+ "_"+str(filename)+".csv"))  


#%% Visualization Server 
## agent's potrayal
def agent_portrayal(agent):
    portrayal = {
        "Shape":"circle",
        "Filled":"true",
        "Layer":0,
        "r":0.2
        }
    if agent.v_status ==1:
        portrayal["Color"] = "green"
    else:
        portrayal["Color"] = "red"        
    return portrayal 

# space set-up
space = SimpleCanvas(agent_portrayal,750, 750)

# model parameter 
model_params = {
    "n_agents": n_agents,
    "width": 60000,
    "height": 64000,
    'weight_family_ntwk':w_ntwk_family,
    'weight_work_ntwk':w_ntwk_work,
    'weight_smedia_ntwk':w_ntwk_socialmedia
    }

# chart 
chart = mesa.visualization.ChartModule([{"Label":"p_all_vaxed",
                                         "Color":"Black"}], 
                                       canvas_height = 50,
                                       canvas_width = 100,
                                       data_collector_name="datacollector")

server = ModularServer(VaxModel, [space,chart], "Social Influence Network on Vaccination", model_params)
server.launch()


#%% References 

# Laurin, K. (2018). Inaugurating rationalization: Three field studies find increased rationalization when anticipated realities become current. Psychological science, 29 (4), 483–495.
# NYS. (2021a). Governor Cuomo Announces Additional New Yorkers, Individuals 75 and Older Can Begin Scheduling with Providers COVID-19 Vaccination Appointments. NYS Governor’s Press Office. Retrieved 2023-02-28, from https://www.governor.ny.gov/news/governor-cuomo-announces-additional-new-yorkers-individuals-75-and-older-can-begin-scheduling 
# NYS. (2021b). Statement From Governor Kathy Hochul on CDC’s Recommendation of Pfizer Vaccine for 5-11 Year Olds. NYS Governor’s Press Office. Retrieved 2023-02-28, from https://www.governor.ny.gov/news/statement-governor-kathy-hochul-cdcs-recommendation-pfizer-vaccine-5-11-year-olds 
# Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. Journal of Mathematical Sociology, 15 (3-4), 193–206.

