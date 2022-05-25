Notes:

Model.train()
    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
        
Model.eva()
    def eval(self):
        r"""Sets the module in evaluation mode."""
        return self.train(False)
        

* Compute edge probablity *

    