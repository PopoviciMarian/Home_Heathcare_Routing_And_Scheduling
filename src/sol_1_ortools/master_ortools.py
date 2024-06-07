from ortools.sat.python import cp_model
import numpy as np
from  reader import Reader
from datetime import datetime


class MasterProblemManager:
    def __init__(self, patients, nurses):
        self.model = cp_model.CpModel()

        self.patients = patients
        self.nurses = nurses
        self.number_of_days = 5
        self.number_of_patients = self.patients.shape[0] - 1
        self.frequency = list(self.patients["f"].astype('int'))
        self.qualification_first_nurse = list(self.patients["Q'"].astype('int'))
        self.qualification = list(self.nurses["Q"].astype('int'))
        self.X, self.Y = list(self.patients["x"]), list(self.patients["y"])
        self.number_of_nurses = self.nurses.shape[0]
        self.start_time = datetime.now()
        self.decision_variables = {}
        self.add_decision_variables()
        self.add_objective_function()
        self.add_constraints()
        self.optimize()
        self.end_time = datetime.now()
        time_in_milliseconds = (self.end_time - self.start_time).total_seconds() * 1000
        print(f"Total time taken for optimization is {time_in_milliseconds} milliseconds")


    def add_decision_variables(self):
        self.decision_variables["d"] = {}
        self.decision_variables["x"] = {}
        self.decision_variables["y"] = {}

        # Decision variable d_i is a binary variable that is equal to 1 if patient i is assigned to a nurse and 0 otherwise.
        for i in range(self.number_of_patients):
            self.decision_variables["d"][i] = self.model.NewBoolVar(f"d_{i}")

        # Decision variable x_ij is a binary variable that is equal to 1 if patient j is assigned to nurse i and 0 otherwise.
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                self.decision_variables["x"][i, j] =self.model.NewBoolVar(f"x_{i}_{j}")

        # Decision variable y_ijk is a binary variable that is equal to 1 if patient j is assigned to nurse i on day k and 0 otherwise.
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for k in range(self.number_of_days):
                    self.decision_variables["y"][i, j, k] = self.model.NewBoolVar(f"y_{i}_{j}_{k}")


    def add_objective_function(self):
        #maximize the number of patients assigned to nurses
        self.model.Maximize(sum(self.decision_variables["d"][i] for i in range(self.number_of_patients)))


    def add_constraints(self):
        for j in range(self.number_of_patients):
            self.model.Add(sum(self.decision_variables["x"][i, j] for i in range(self.number_of_nurses)) == self.decision_variables["d"][j])

        for j in range(self.number_of_patients):
            self.model.Add(sum(sum(self.decision_variables["y"][i, j + 1, k] for k in range(self.number_of_days)) for i in range(self.number_of_nurses)) \
                            == self.frequency[j+1] * self.decision_variables["d"][j])

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    self.model.Add(self.decision_variables["y"][i, j + 1, k] <= self.decision_variables["x"][i, j])

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                if self.qualification_first_nurse[j+1] != self.qualification[i]:
                    self.model.Add(self.decision_variables["x"][i, j] == 0)

        for i in range(self.number_of_nurses):
            for k in range(self.number_of_days):
                self.model.Add(self.decision_variables["y"][i, 0, k] == 1)


        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    for tou in range(1, (4 - (self.frequency[j + 1]) + 1)):
                        if (k + tou) <= 4:
                            self.model.Add(self.decision_variables["y"][i, j + 1, k] + self.decision_variables["y"][i, j + 1, k + tou] <= 1)


    def optimize(self):
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        if status ==  cp_model.OPTIMAL:
            print(f"Optimal solution found")
            for i in range(self.number_of_patients):

                if solver.Value(self.decision_variables["d"][i]) != 1:
                    print(f"Patient {i} not assigned to any nurse")
                    continue
                for j in range(self.number_of_nurses):
                    if solver.Value(self.decision_variables["x"][j, i]) == 1:
                        print(f"Patient {i} assigned to nurse {j}")
            print(f"Objective value: {solver.ObjectiveValue()}")
            #print(f"Objective value: {solver.Objective().Value()}")



    def save_solution(self, file_path):
        file = open(file_path, "w")
        file.write("nurse,patient,x,y\n")
        # save nurse,patient,x,y
        for i in range(self.number_of_patients):
            if self.decision_variables["d"][i].solution_value() != 1:
               # print(f"Patient {i} not assigned to any nurse")
                continue
            for j in range(self.number_of_nurses):
                if self.decision_variables["x"][j, i] == 1:
                    if j == 0:
                        file.write(f"{j},{i},{self.X[i]},{self.Y[i]}\n")
        file.close()



    def print_solution(self):
        pass




if __name__ == "__main__":
    reader = Reader("./DB/30P.xlsx")
    patients, nurses = reader.read()
    manager = MasterProblemManager(patients, nurses)
   # manager.save_solution("solution.csv")


