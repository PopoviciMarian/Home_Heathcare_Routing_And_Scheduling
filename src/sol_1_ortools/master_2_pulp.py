from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, LpStatus, lpSum
from reader import Reader

class MasterProblemManager:
    def __init__(self, patients, nurses):
        self.model = LpProblem("NurseAssignment", LpMaximize)

        self.patients = patients
        self.nurses = nurses
        self.number_of_days = 5
        self.number_of_patients = self.patients.shape[0] - 1
        self.frequency = patients["f"].astype(int).tolist()
        self.qualification_first_nurse = patients["Q'"].astype(int).tolist()
        self.qualification = nurses["Q"].astype(int).tolist()
        self.X, self.Y = patients["x"], patients["y"]
        self.number_of_nurses = self.nurses.shape[0]

        self.decision_variables = {}
        self.add_decision_variables()
        self.add_objective_function()
        self.add_constraints()
        self.solve()

    def add_decision_variables(self):
        self.decision_variables["d"] = {}
        self.decision_variables["x"] = {}
        self.decision_variables["y"] = {}

    # Decision variable d_i (binary)
        for i in range(self.number_of_patients):
            self.decision_variables["d"][i] = LpVariable(f"d_{i}", cat=LpBinary)
            self.model += self.decision_variables["d"][i]

    # Decision variable x_ij (binary)
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                self.decision_variables["x"][i, j] = LpVariable(f"x_{i}_{j}", cat=LpBinary)
                self.model += self.decision_variables["x"][i, j]

        # Decision variable y_ijk (binary)
        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients + 1):
                for k in range(self.number_of_days):
                    self.decision_variables["y"][i, j, k] = LpVariable(f"y_{i}_{j}_{k}", cat=LpBinary)
                    self.model += self.decision_variables["y"][i, j, k]

    def add_objective_function(self):
    # Maximize the number of patients assigned to nurses
        self.model += lpSum([self.decision_variables["d"][i] for i in range(self.number_of_patients)])
    
    def add_constraints(self):
        for j in range(self.number_of_patients):
            self.model += lpSum(self.decision_variables["x"][i, j] for i in range(self.number_of_nurses)) == self.decision_variables["d"][j]

        for j in range(self.number_of_patients):
            self.model += lpSum(lpSum(self.decision_variables["y"][i, j + 1, k] for k in range(self.number_of_days)) for i in range(self.number_of_nurses)) == self.frequency[j+1] * self.decision_variables["d"][j]

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    self.model += self.decision_variables["y"][i, j + 1, k] <= self.decision_variables["x"][i, j]

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                if self.qualification_first_nurse[j+1] != self.qualification[i]:
                    self.model += self.decision_variables["x"][i, j] == 0

        for i in range(self.number_of_nurses):
            for k in range(self.number_of_days):
                self.model += self.decision_variables["y"][i, 0, k] == 1

        for i in range(self.number_of_nurses):
            for j in range(self.number_of_patients):
                for k in range(self.number_of_days):
                    for tou in range(1, (4 - (self.frequency[j + 1]) + 1)):
                        if (k + tou) <= 4:
                            self.model += self.decision_variables["y"][i, j + 1, k] + self.decision_variables["y"][i, j + 1, k + tou] <= 1

    def solve(self):
        self.model.solve()
        print(f"Solution status: {LpStatus[self.model.status]}")
        if LpStatus[self.model.status] == "Optimal":
            
            for i in range(self.number_of_patients):
                if self.decision_variables["d"][i].varValue != 1:
                    print(f"Patient {i} not assigned to any nurse")
                    continue
                for j in range(self.number_of_nurses):
                    if self.decision_variables["x"][j, i].varValue == 1:
                        print(f"Patient {i} assigned to nurse {j}")
            print(f"Objective value: {self.model.objective.value()}")

        else:
            print(f"Solution status: {self.model.status}")

if __name__ == "__main__":
    reader = Reader("./DB/100P.xlsx")
    patients, nurses = reader.read()
    manager = MasterProblemManager(patients, nurses)