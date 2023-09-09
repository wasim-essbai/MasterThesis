class ResultEval:
    def __init__(self, name):
        self.accuracy = []
        self.steps = []
        self.unkn = []
        self.aleatoric = []
        self.epistemic = []
        self.name = name
