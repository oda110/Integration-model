# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 23:51:25 2021

@author: Admin
"""


from gurobipy import GRB
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel(r'C:\Users\Admin\Desktop\\PhD\Integration models\Model\Data\H2_Profile.xls',sheet_name='Sheet1')

#parameters

beta    = 5.71*(10**-3)            #Constant in power generation efficiency equation (MW)
ya      = 0.9              #CO2 removal rate of scrubber
etago   = 0.74             #base case gross efficiency
muao    = 0.02             #base case efficiency penalty of absorption
mubo    = 0.01             #base case basic efficieny penalty
muco    = 0.02             #base case efficieny penalty of compression
mudo    = 0.04             #base case efficiency penalty of desorption
ple     = 40               #Price of CEC in carbon market (€/ton)
plg     = 40               #price of power to grid (€/MWh)
pfg     = 200              #price of power from grid (€/MWh)
w       = 0.974            #Quadratic coefficient in power generation efficieny equation (MW-2)
cgo     = 32.3             #base case power generation cost pet unit of gross energy output (€/MWh)
ctss    = 7                #Transport and storage cost of CO2 (€/ton)
go      = 50.45             #Base case (maximum) gross power output in (MW)
eo      = go/etago         #Power input to the plant under base case conditions (MW)
ego     = 0.9              #Base case CO2 emission intensity (tons/MWh)
elg     = 460000              #CEC allocated for one day of operation (tons)
ecgmax  = 0.3              #Maximum CO2 emission intensity (tons/MWh)
glt     = 40               #refinery demand (MW)
lo      = 892              #Base case inlet flowrate in lean solvent storage tank (m3/hr)
lototal = 892              #Initial solvent volume in lean solvent storage tank (m3)
ro      = 892              #Base case inlet flowrate in rich solvent storage tank (m3/hr)
rototal = 892              #Initial solvent volume in rich solvent storage tank (m3)
alphaa  = eo*muao          #Absorption energy penalty (MW)
alphad  = eo*(mudo+muco)   #Combined desorption and compression energy penalty (MW)

#Variable bounds

deltagr     = 50.45         #Maximum ramping rate of power generation (MW/hr)
deltaramax  = 1            #Maximum ramping rate of scrubber (hr-1)
deltardmax  = 1.25         #Maximum ramping rate of stripper (hr-1)
gmin        = 32           #Minimum gross power output (MW)
lmax        = 1800         #Maximum capacity of lean solvent storage tank (m3)
ramax       = 1            #Maximum CO2 absorption rate in scrubber
rdmax       = 1.25         #Maximum CO2 desorption rate in stripper
rmax        = 1800         #Maximum capacity of rich solvent storage tank (m3)

m = gp.Model('CC Integration with carbon captured')

#variables

T       = 8759          # time horizon in h
NT      = range(T)    # range time
psgt    = 45 #The purchase cost is 85 and selling price 40

h2 = df['CO2_tons'].tolist()


gt      = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'gt')     #gross power output at hour t in MW
gnt     = m.addVars(T, vtype = GRB.CONTINUOUS, lb=41, name = 'gnt')    #net power output at hour t in MW
gcc     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'gcc')    #power to capture plant
rat     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'rat')    #rate of CO2 absorbed in scrubber at hour t
rdt     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'rdt')    #rate of CO2 desorbed in stripper at hour t
rct     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'rct')    #rate of CO2 compression in compressor at hour t
etagt   = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'etagt')  #gross efficiency at hout t
etangt  = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'etangt') #net efficiency at hour t
cgt     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'cgt')    #energy generation cost per unit of total energy output at hour t in $/MWh
egt     = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'egt')    #CO2 emissions intensity (emissions per unit of gross energy output) at hour t in tons/MWh
engt    = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'engt')   #net CO2 emissions at hour t in tons
esgt    = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'esgt')   #amount of compressed CO2 exiting the capture system at hour t in tons
lt      = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'lt')     #solvent volume in lean solvent storage tank at hour t in m3
ltin    = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'ltin')   #inlet volumetric flowrate in lean solvent storage tank at hour t in m3/hr
rt      = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'rt')     #solvent volume in rich solvent storage tank at hour t in m3
rtin    = m.addVars(T, vtype = GRB.CONTINUOUS, name = 'rtin')   #inlet volumetric flowrate in rich solvent storage tank at hour t in m3/hr
z       = m.addVars(T, vtype = GRB.CONTINUOUS, name ='z')       #Dummy variable as Gurobi does not allow variable division
y       = m.addVars(T, vtype = GRB.CONTINUOUS, name ='y') 
e       = m.addVars(T, vtype = GRB.CONTINUOUS, name ='e') 

#constraints

m.addConstrs(gt[t]  >= gmin for t in NT ) #lower bound of gross power output
m.addConstrs(gt[t] <= go for t in NT) #higher bound of gross power output based on maximum gross power output

m.addConstrs(gnt[t]  == gt[t] - (alphaa*rat[t])-(alphad*rdt[t]) for t in NT) #net power output

m.addConstrs(gcc[t] == gt[t] - gnt[t] for t in NT) #CC power = gross power output - net power

m.addConstrs(rat[t] >= 0 for t in NT) # lower bound of absorption 
m.addConstrs(rat[t] <= ramax for t in NT) # upper bound of absorption

m.addConstrs(rdt[t] >= 0 for t in NT) # lower bound of desorption 
m.addConstrs(rdt[t] <= rdmax for t in NT) # upper bound of desorption 


m.addConstrs(rtin[t] == ro*rat[t] for t in NT) #volumetric flow rate in rich tank as function of base case flow rate and amount of CO2 processed
m.addConstrs(ltin[t] == lo*rdt[t] for t in NT) #volumetric flow rate in lean tank as function of base case flow rate and amount of CO2 processed
 

m.addConstrs(rt[t] >= 0 for t in NT) #rich tank capacity constraint
m.addConstrs(rt[t] <= rmax for t in NT) #rich tank capacity constraint
m.addConstrs(lt[t] >= 0 for t in NT) #lean tank capacity constraint
m.addConstrs(lt[t] <= lmax for t in NT) #lean tank capacity constraint

m.addConstr(rt [0] == rototal + (rtin[0]-ltin[0])) #volumetric flow balance
m.addConstrs(rt [t] == rt[t-1] + (rtin[t]-ltin[t]) for t in range(1,T)) 
m.addConstr(lt [0] == lototal + (ltin[0]-rtin[0])) 
m.addConstrs(lt [t] == lt[t-1] + (ltin[t]-rtin[t]) for t in range(1,T)) 


m.addConstrs(etagt[t] == (beta*gt[t])+ w for t in NT)

m.addConstrs(z[t] * etagt[t] == 1 for t in NT) # dummy variable as gurobi does not divide by variables
m.addConstrs(y[t] * e[t] == 1 for t in NT) 
m.addConstrs(e[t] == gt[t] * z[t] for t in NT) 

m.addConstrs(engt[t] == egt[t]*gt[t]- go*ego*ya*rdt[t] for t in NT) #net CO2 emissions at time t

#m.addConstr(gp.quicksum(engt[t] for t in NT)-(ecgmax*gp.quicksum(gnt[t] for t in NT)) <= 0) #emissions constraint where the total co2 emissions in one day should below a threshold

m.addConstrs(gt[t+1] - gt[t] >= -deltagr for t in range(T-1)) #ramping constraint for the generation system
m.addConstrs(gt[t+1] - gt[t] <=  deltagr for t in range(T-1)) #ramping constraint for the generation system
m.addConstrs(rat[t+1] - rat[t] >= -deltaramax for t in range(T-1)) #ramping constraint for the absorption capture system
m.addConstrs(rat[t+1] - rat[t] <=  deltaramax for t in range(T-1)) #ramping constraint for the absorption capture system
m.addConstrs(rdt[t+1] - rdt[t] >= -deltardmax for t in range(T-1)) #ramping constraint for the desorption capture system
m.addConstrs(rdt[t+1] - rdt[t] <=  deltardmax for t in range(T-1)) #ramping constraint for the desorption capture system
m.addConstrs(rct[t] == rdt[t] for t in NT) #the amount of co2 desorbed will equal the amount compressed as there is no co2 storage
m.addConstrs(esgt[t] == rct[t] *go *ego*ya for t in NT) #amount of compressed co2 exiting the capture system at time t
m.addConstrs(cgt[t] == etago*cgo*z[t] for t in NT) #energy generation cost per unit of total energy output at time t
m.addConstrs(egt[t] == etago*ego*z[t]for t in NT) #emissions intensity per unit of power generation at time t
m.addConstrs(etangt[t]== gnt[t]/eo for t in NT) #net efficiency at hour t
m.addConstrs(esgt[t]  == h2 [t] for t in NT ) #hydrogen tracking


#Objective function
  
day_ahead = gp.quicksum((gnt[t]-glt)*psgt for t in NT)
generation_cost = gp.quicksum(gt[t]*cgt[t] for t in NT)
CEC1 = gp.quicksum(engt[t]for t in NT)
CEC = ple * CEC1
Transport = gp.quicksum(esgt[t]*ctss for t in NT)

m.setObjective(day_ahead - generation_cost - CEC - Transport, GRB.MAXIMIZE)
m.setParam("NonConvex", 2)
# m.write('model.lp')
m.optimize()

print('Number of Varibales in the model =', m.numVars)
print('The maximum revenue =' , m.ObjVal, '€')
print('Revenue in electricity spot market =',day_ahead.getValue(), '€')
print('Generation costs =',generation_cost.getValue(), '€')
print('carbon cost =',CEC.getValue(), '€')
print('Transportation and storage costs =',Transport.getValue(), '€')

m.printStats()


# for v in m.getVars():
#   print(v.varName, v.x)

# m.printAttr('X')



# gt_ = []
# for i in range(8759):
#     gt_.append(gt[i].x)

# gnt_ = []
# for i in range(8759):
#     gnt_.append(gnt[i].x)
    
# rat_ = []
# for i in range(8759):
#     rat_.append(rat[i].x)
    
# rdt_ = []
# for i in range(8759):
#     rdt_.append(rdt[i].x)

# gcc_ = []
# for i in range(8759):
#     gcc_.append(gcc[i].x)
    
# esgt_ = []
# for i in range(8759):
#     esgt_.append(esgt[i].x)
    
# engt_ = []
# for i in range(8759):
#     engt_.append(engt[i].x)

# rt_ = []
# for i in range(8759):
#     rt_.append(rt[i].x)

# lt_ = []
# for i in range(8759):
#     lt_.append(lt[i].x)

# marks_data = pd.DataFrame({'gt':gt_,
#                             'gnt':gnt_,
#                             'rat':rat_,
#                             'rdt':rdt_,
#                             'gcc':gcc_,
#                             'engt':engt_,
#                             'esgt':esgt_,
#                             'rt':rt_,
#                             'lt':lt_})

# file_name = 'heide_data.xlsx'

# marks_data.to_excel(file_name)





#---------------------------------------------------------- 
# plt.plot(psgt)
# plt.ylabel('Electricity Price $/MWh')
# plt.xlabel('Time, hr')
# plt.show()

#---------------------------------------------------------- 

rat = [rat[t].x for t in NT]

plt.figure()  
plt.step(NT, rat)
plt.xlim(0,8000)
plt.ylim(0, 1.25)
plt.xlabel('time, hr')
plt.ylabel('Absorption rate')

#---------------------------------------------------------- 

# fig, ax = plt.subplots()
# Time = list(NT)
# gnt = [gnt[t].x for t in NT]
# gcc = [gcc[t].x for t in NT]
# Net_power = list(gnt)
# Power_to_capture_plant = list(gcc)
# width = 0.35
# bar1 = ax.bar(Time, Net_power, width,color='tomato', hatch='//', label='Net Power')
# bar2 = ax.bar(Time, Power_to_capture_plant, width, bottom =Net_power, color='limegreen', label='CC')
# plt.ylim(0, 630)
# plt.xlabel("Time, hr")
# plt.ylabel("Power, MW")
# plt.legend(loc='best')
# plt.show()

#---------------------------------------------------------- 


# fig = plt.figure(figsize = (10, 5))
# n_groups = 24
# index = np.arange(n_groups)
# rat = [rat[t].x for t in NT]
# rdt = [rdt[t].x for t in NT]
# Absorption = list(rat)
# Desorption = list(rdt)
# bar_width = 0.35
# bar1 = plt.bar(index, Absorption, bar_width,color='tomato', hatch='//', label='Absorption')
# bar2 = plt.bar(index + bar_width, Desorption, bar_width, color='limegreen', label='Desorption')
# plt.xlabel("Time, hr")
# plt.ylabel("Absorption/Desorption rate")
# #plt.title("Absorption/Desorption schedule")
# plt.legend(loc='best')
# plt.show()


#---------------------------------------------------------- 

# rdt = [rdt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, rdt)
# plt.xlim(0,23)
# plt.ylim(0, 1.3)
# plt.xlabel('time, hr')
# plt.ylabel('Desorption rate')

#---------------------------------------------------------- 

rdt = [rdt[t].x for t in NT]
Desorption = list(rdt)
Time = list(NT)
fig = plt.figure(figsize = (10, 5))
plt.bar(Time, Desorption, color ='maroon',width = 0.4)
plt.xlabel("Time, hr")
plt.ylabel("Desorption")
plt.title("Desorption schedule")
plt.show()

#---------------------------------------------------------- 

# gt = [gt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, gt)
# plt.xlim(0,23)
# plt.ylim(0, 650)
# plt.xlabel('Time, hr')
# plt.ylabel('Gross power output, MW')

#---------------------------------------------------------- 

# gnt = [gnt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, gnt)
# plt.xlim(-1,23)
# plt.ylim(0, 650)
# plt.xlabel('Time, hr')
# plt.ylabel('Net power output, MW')

#---------------------------------------------------------- 


# gcc = [gcc[t].x for t in NT]

# plt.figure()  
# plt.step(NT, gcc)
# plt.xlim(0,23)
# plt.ylim(-5, 300)
# plt.xlabel('Time, hr')
# plt.ylabel('Capture plant power, MW')

#---------------------------------------------------------- 

# rt = [rt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, rt)
# plt.xlim(0,8759)
# plt.ylim(0, 16000)
# plt.xlabel('time, hr')
# plt.ylabel('Rich Tank Volume, m\u00b3')

#---------------------------------------------------------- 


# lt = [lt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, lt)
# plt.xlim(0,24)
# plt.ylim(0, 15000)
# plt.xlabel('time, hr')
# plt.ylabel('lean tank volume, m\u00b3')

#---------------------------------------------------------- 

engt = [engt[t].x for t in NT]

plt.figure()  
plt.step(NT, engt)
plt.xlim(-1,720)
plt.ylim(-10, 25)
plt.xlabel('Time, hr')
plt.ylabel('Net CO\u2082 emissions tons')

#---------------------------------------------------------- 

# esgt = [esgt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, esgt)
# plt.xlim(0,24)
# plt.ylim(0, 500)
# plt.xlabel('time, hr')
# plt.ylabel('compressed CO\u2082 tons')

#---------------------------------------------------------- 

# etagt = [etagt[t].x for t in NT]

# plt.figure()  
# plt.step(NT, etagt)
# plt.xlim(0,24)
# plt.ylim(0, 1)
# plt.xlabel('time, hr')
# plt.ylabel('gross efficiency')

#---------------------------------------------------------- 


# v = [v[t].x for t in NT]

# plt.figure()  
# plt.step(NT, v)
# plt.xlim(0,24)
# plt.ylim(0, 1)
# plt.xlabel('time, hr')
# plt.ylabel('venting rate')

#---------------------------------------------------------- 


# e = [e[t].x for t in NT]

# plt.figure()  
# plt.step(NT, e)
# plt.xlim(0,24)
# plt.ylim(0, 1400)
# plt.xlabel('time, hr')
# plt.ylabel('imported energy MW')

#---------------------------------------------------------- 

# muccs = [muccs[t].x for t in NT]

# plt.figure()  
# plt.step(NT, muccs)
# plt.xlim(0,24)
# plt.ylim(0, 0.15)
# plt.xlabel('time, hr')
# plt.ylabel('energy penalty')

#---------------------------------------------------------- 





