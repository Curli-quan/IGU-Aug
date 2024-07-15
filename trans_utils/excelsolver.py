import pandas as pd



class ExcelSolver:
    def __init__(self, filename:str = None) -> None:
        # self.file_type = "xls" # or "xlsx"
        if filename is not None:
            self.set_file(filename)
    
    def set_file(self, filename):
        ext = filename.split(".")[-1]
        # if ext in ['xls']:
        file_type = ext
        self.filename = filename

    def read(self):
        content = pd.read_excel(io=self.filename, sheet_name=None)
        # return content

        # Specific Solving
        sheet_names = list(content.keys())
        # print(sheet_names)
        pd_data = content[sheet_names[0]]
        # return pd_data
        self.cnames = pd_data.columns.values
        self.cvalues = pd_data.to_numpy()
        # print("debug00 before update", self.cvalues[:2])
        # return cnames, cvalues

    def __len__(self):
        return len(self.cvalues)

    def get_label(self, index):
        # ['11182324_08.jpg', '(130, 212)', '(238, 164)', '(180, 246)',
        # '(189, 240)', '(197, 234)', '(205, 227)', '(214, 220)']
        values = self.cvalues[index]
        res = {}
        res['name'] = values[0]

        for i, v in enumerate(values[1:3]):
            v = v.split(",")
            num1, num2 = int(v[0].strip()[1:]), int(v[1].strip()[:-1])
            res[f'landmark_ring_{i}'] = (num1, num2)
        
        for i, v in enumerate(values[3:]):
            v = v.split(",")
            num1, num2 = int(v[0].strip()[1:]), int(v[1].strip()[:-1])
            res[f'landmark_argin_{i}'] = (num1, num2)

        return res
    
    def remove_by_index(self, index):
        if isinstance(index, list):
            # print("debug before update", self.cvalues[:2])
            new_list = []
            for i, v in enumerate(self.cvalues):
                if i not in index:
                    new_list.append(v)
            self.cvalues = new_list
            # print("debug after update", self.cvalues[:2])
        else:
            if index == len(self.cvalues) - 1:
                self.cvalues = self.cvalues[:-1]
            elif index == 0:
                self.cvalues = self.cvalues[1:]
            else:
                self.cvalues = self.cvalues[:index] + self.cvalues[index+1:]
        

if __name__ == "__main__":
    name = "./landmark_gangmen.xlsx"
    solver = ExcelSolver()
    solver.set_file(name)
    content = solver.read()
    for i in range(len(solver)):
        print("Processing ", i)
        print(solver.get_label(i))
    import ipdb; ipdb.set_trace()