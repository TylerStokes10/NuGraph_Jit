import nugraph as ng

# Assuming NuGraph3 is the model class
Model = ng.models.NuGraph3

# Print the model's forward method annotations to understand the input structure
print(Model.forward.__annotations__)

