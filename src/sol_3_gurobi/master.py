import gurobipy as gb   
import numpy as np
from datetime import datetime
import plotly.express as px
import pandas as pd
import math
from datetime import datetime
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Reader:

    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        patients = pd.DataFrame(pd.read_excel(self.file_path, "patients")).fillna(0).astype('int')
        nurses = pd.DataFrame(pd.read_excel(self.file_path, "nurses")).fillna(0).astype('int')
        return patients, nurses

def distance(x, y, bigM):
    x = np.array(x)
    y = np.array(y)
    dist_grid = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    np.fill_diagonal(dist_grid, bigM)
    return dist_grid

class MasterProblemManager:
    def __init__(self, patients, nurses):
        self.model = gb.Model("NurseAssignment")
        self.patients = patients
        self.nurses = nurses
        self.number_of_days = 5
        self.number_of_patients = self.patients.shape[0] - 1
        self.frequency = list(self.patients["f"].astype('int'))
        self.number_of_nurses_required = list(self.patients["nN"].astype('int'))
        self.earliest_service_start_time = list(self.patients["et"].astype('int'))
        self.latest_service_start_time = list(self.patients["lt"].astype('int'))
        self.service_duration = list(self.patients["sd"].astype('int'))
        self.qualification_first_nurse = list(self.patients["Q'"].astype('int'))
        self.qualification_second_nurse = list(self.patients["Q'2"].astype('int'))
        self.qualification_third_nurse = list(self.patients["Q'3"].astype('int'))
        self.qualification = list(self.nurses["Q"].astype('int'))
        self.number_of_nurses = self.nurses.shape[0]
        self.nurse_time = list(self.nurses["Time"].astype('int'))
        self.bigM = 1000
        self.X, self.Y = list(self.patients["x"]), list(self.patients["y"])
        self.depot = [self.X[0], self.Y[0]]
        self.start_time = datetime.now()
        self.grid = distance(self.X, self.Y, self.bigM)
        self.decision_variables = {}
        self.add_decision_variables()
        self.add_objective_function()
        self.add_constraints()
        self.add_sub_constraints_decision_variables()
        self.add_sub_constraints()
        self.optimize()
        self.end_time = datetime.now()
        time_in_milliseconds = (self.end_time - self.start_time).total_seconds() * 1000
        print(f"Total time taken for optimization is {time_in_milliseconds} milliseconds")


    def add_decision_variables(self):
        self.decision_variables["d"] = {}
        self.decision_variables["x"] = {}
        self.decision_variables["y"] = {}

        for i in range(self.number_of_patients):
            self.decision_variables["d"][i] = self.model.addVar(name=f"d_{i}", vtype=gb.GRB.BINARY)
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                self.decision_variables["x"][i, j] = self.model.addVar(name=f"x_{i}_{j}", vtype=gb.GRB.BINARY)
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for k in range(self.number_of_days):
                    self.decision_variables["y"][i, j, k] = self.model.addVar(name=f"y_{i}_{j}_{k}", vtype=gb.GRB.BINARY)
        self.model.update()

    def add_objective_function(self):
        self.model.setObjective(gb.quicksum(self.decision_variables["d"][i] for i in range(self.number_of_patients)), gb.GRB.MAXIMIZE)
        self.model.update()


    def add_constraints(self):
        for j in range(self.number_of_patients):   
            self.model.addConstr(gb.quicksum(self.decision_variables["x"][i, j] for i in range(self.number_of_nurses)) == \
                                  self.decision_variables["d"][j] * self.number_of_nurses_required[j+1])
        self.model.update()

        for j in range(self.number_of_patients):
            self.model.addConstr(gb.quicksum(gb.quicksum(self.decision_variables["y"][i, j + 1, k] for k in range(self.number_of_days)) \
                                              for i in range(self.number_of_nurses) ) == \
                                                  self.frequency[j + 1] * self.decision_variables["d"][j] * self.number_of_nurses_required[j + 1])
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    self.model.addConstr(self.decision_variables["y"][i, j + 1, k] <= self.decision_variables["x"][i, j])
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                if self.qualification_first_nurse[j+1] != self.qualification[i] \
                      and self.qualification_second_nurse[j+1] != self.qualification[i] \
                        and self.qualification_third_nurse[j+1] != self.qualification[i]:
                    self.model.addConstr(self.decision_variables["x"][i, j] == 0)
        self.model.update()

        for i in range(self.number_of_nurses):
            for k in range(self.number_of_days):
                self.model.addConstr(self.decision_variables["y"][i, 0, k] == 1)
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    for tou in range(1, (4 - (self.frequency[j+1]) + 1)):
                        if (k + tou) <= 4:
                            self.model.addConstr(self.decision_variables["y"][i, j + 1, k] + \
                                                  self.decision_variables["y"][i, j + 1, k + tou] <= 1)
        self.model.update()
    

    def add_sub_constraints_decision_variables(self):
        self.decision_variables["z"] = {} 
        self.decision_variables["s"] = {}

        # z[i, j, l, k] = 1 if nurse i travels from patient j to patient l on day k
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for l in range(self.number_of_patients + 1):
                    for k in range(self.number_of_days):
                        self.decision_variables["z"][i, j, l, k] = self.model.addVar(name=f"z_{i}_{j}_{l}_{k}", vtype=gb.GRB.BINARY)
        
        self.model.update()
        # service start time for patient j on day k
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for k in range(self.number_of_days):
                    self.decision_variables["s"][i, j, k] = self.model.addVar(name=f"s_{i}_{j}_{k}", vtype=gb.GRB.CONTINUOUS)
        self.model.update()
    
    def add_sub_constraints(self):
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for k in range(self.number_of_days):
                    # makes sure that the arrival to and the departure from every patient nodes
                    #for a nurse on a day is equal to the assignment decision of that patient for the same
                    #nurse on the same day.
                    self.model.addConstr(gb.quicksum(self.decision_variables["z"][i, j, l, k] for l in range(self.number_of_patients + 1)) == self.decision_variables["y"][i, j, k])
                    self.model.addConstr(gb.quicksum(self.decision_variables["z"][i, l, j, k] for l in range(self.number_of_patients + 1)) == self.decision_variables["y"][i, j, k])
                
                    # time window
                    #  multiple nurses are required for the same patient simultaneously, ensuring that their service start times are synchronized.
                    if j <= self.number_of_patients - 1:
                        self.model.addConstr(self.decision_variables["s"][i, j + 1, k] + self.bigM * (1 - self.decision_variables["y"][i, j + 1, k]) >= self.earliest_service_start_time[j + 1])
                        self.model.addConstr(self.decision_variables["s"][i, j + 1, k] + self.service_duration[j + 1] - self.bigM * (1 - self.decision_variables["y"][i, j + 1, k]) <= self.latest_service_start_time[j + 1])
                    
                    # loop elimination
                    if j != 0:
                        self.model.addConstr(self.decision_variables["z"][i, j, j, k] == 0)
                    
                    # service start time of patient needing multiple nurses at a same time
                    if j <= self.number_of_patients - 1:
                        for p in range(self.number_of_nurses):
                            if self.number_of_nurses_required[j + 1] != 1 and i != p:
                                self.model.addConstr(self.decision_variables["s"][i, j + 1, k] + self.bigM * (2 - self.decision_variables["y"][i, j + 1, k] - self.decision_variables["y"][p, j + 1, k]) >= self.decision_variables["s"][p, j + 1, k])
                                self.model.addConstr(self.decision_variables["s"][i, j + 1, k] <= self.decision_variables["s"][p, j + 1, k] + self.bigM * (2 - self.decision_variables["y"][i, j + 1, k] - self.decision_variables["y"][p, j + 1, k]))
        self.model.update()

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for l in range(self.number_of_patients):
                    for k in range(self.number_of_days):
                        self.model.addConstr(self.decision_variables["s"][i, j, k] <= self.bigM * self.decision_variables["y"][i, j, k])
                        self.model.addConstr(self.decision_variables["z"][i, 0, j + 1, k] * self.grid[0, j + 1] + self.decision_variables["s"][i, l + 1, k] + (self.grid[l + 1, 0] + self.service_duration[l + 1]) * self.decision_variables["z"][i, l + 1, 0, k] <= \
                            self.nurse_time[i] / self.number_of_days + self.bigM * (1 - self.decision_variables["z"][i, l + 1, 0, k]) + self.bigM * (1 - self.decision_variables["z"][i, 0, j + 1, k]))
        
        self.model.update()
        # i k j l
        for i in range(self.number_of_nurses):
            for k in range(self.number_of_days):
                for j in range(self.number_of_patients):
                    for l in range(self.number_of_patients):
                        self.model.addConstr(self.decision_variables["s"][i, j + 1, k] + (self.service_duration[j + 1] + self.grid[j + 1, l]) * self.decision_variables["z"][i, j + 1, l, k] <= self.decision_variables["s"][i, l, k] + self.bigM * (1 - self.decision_variables["z"][i, j + 1, l, k]))
        self.model.update()


    def optimize(self):
        self.model.optimize()
        print(f"Solution status: {self.model.status}")
        
        if self.model.status == gb.GRB.Status.OPTIMAL:
            print(f"Optimal solution found")
            solution = self.model.getAttr('x', self.decision_variables["d"])
            for i in range(self.number_of_patients):
                print(f"Patient {i}: {solution[i]}")
                
            obj = self.model.getObjective()
            print(f"Objective value: {obj.getValue()}")

    def save_solution(self, path):
        """
                'Nurse': 1,
                'Patient': 1
                'Day': 1,
                'start_time': 10,
                'duration': 10
        """
        file = open(path, "w")
        file.write("Nurse,Patient,Day,Start Time,Duration\n")
        sol_y = self.model.getAttr('x', self.decision_variables["y"])
        sol_s = self.model.getAttr('x', self.decision_variables["s"])
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    #if self.decision_variables["y"][i, j + 1, k].varValue == 1:
                    if sol_y[i, j + 1, k] == 1:
                        #start_time = self.decision_variables["s"][i, j + 1, k].varValue
                        start_time = sol_s[i, j + 1, k]
                        file.write(f"{i},{j},{k},{start_time},{math.ceil(self.service_duration[j + 1])}\n")
        file.close()

        #save a file for each nurse
        """ 
         nurse_0.csv
         day,patient,x,y,start_time,duration

        """
        sol_z = self.model.getAttr('x', self.decision_variables["z"])
        for i in range(self.number_of_nurses):
            file = open(f"nurse_{i}.csv", "w")
            file.write("Day,Patient,x,y,Start Time,Duration\n")
            for k in range(self.number_of_days):
                for j in range(self.number_of_patients + 1):
                    for l in range(self.number_of_patients + 1):
                        if sol_z[i, j, l, k] == 1:
                            #start_time = self.decision_variables["s"][i, j, k].varValue
                            start_time = sol_s[i, j, k]
                            file.write(f"{k},{j},{self.X[j]},{self.Y[j]},{start_time},{math.ceil(self.service_duration[j])}\n")




    def print_solution(self):
        pass


class PlotResults:
    def __init__(self, file, patients, nurses):
        self.file = file
        self.data = pd.read_csv(file)
        self.data_grouped = []
        self.patients = patients
        self.nurses = nurses

    def group_data(self):
     
        for i in range(self.data.shape[0]):
            nurse = self.data["Nurse"][i]
            patient = self.data["Patient"][i]
            day = datetime.now() + pd.Timedelta(days=self.data["Day"][i])
            start_time = day + pd.Timedelta(minutes=self.data['Start Time'][i])
            end_time = start_time + pd.Timedelta(minutes=self.data['Duration'][i])
            self.data_grouped.append({"Nurse": nurse, "Patient": patient, "Day": day, "start_time": start_time, "end_time": end_time})
        return self
    
    def plot_nurse_road(self):
        fig = make_subplots(rows=1, cols=5)
        for day in range(5):
            for nurse in range(len(self.nurses)):
                data = pd.read_csv(f"nurse_{nurse}.csv")
                data = data[data["Day"] == day]
                fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="lines+markers", name=f"Nurse {nurse}"), row=1, col=day + 1)
            for i in range(self.patients.shape[0]):
                if i == 0:
                    # depot
                    fig.add_trace(go.Scatter(x=[self.patients["x"][i]], y=[self.patients["y"][i]], mode="markers", name=f"Depot", marker=dict(size=10, color="black"), showlegend=False), row=1, col=day + 1)
                else:
                    fig.add_trace(go.Scatter(x=[self.patients["x"][i]], y=[self.patients["y"][i]], mode="markers", name=f"Patient {i}", marker=dict(size=3, color="red"), showlegend=False), row=1, col=day + 1)
        fig.show()
       

    def plot(self):
        print(self.data_grouped[0])
        category_order = sorted(list(set([i['Nurse'] for i in self.data_grouped])))
        fig = px.timeline(self.data_grouped, x_start="start_time", x_end="end_time", y="Nurse", color="Patient", labels={"Nurse": "Nurse", "Patient": "Patient", "start_time": "Start Time", "end_time": "End Time"}, category_orders={"Nurse": category_order})
        fig.show()


if __name__ == "__main__":
    s = 50
    reader = Reader(f"./DB/{s}P.xlsx")
    patients, nurses = reader.read()
    manager = MasterProblemManager(patients, nurses)
    manager.save_solution(f"solution_{s}.csv")
   
    plot = PlotResults(f"solution_{s}.csv", patients, nurses)
    plot.group_data().plot()
    plot.plot_nurse_road()

    

