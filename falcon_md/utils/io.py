import os
import copy

def save_models(model_list, folder_path='models'):
    os.makedirs(folder_path, exist_ok=True)
    for idx, model in enumerate(model_list):
        model_path = os.path.join(folder_path, f"model_{idx}.h5")
        model.save(model_path)
        print(f"Saved model {idx} to {model_path}")


def load_models(basis_model, folder_path='models'):
    model_files = sorted([f for f in os.listdir(folder_path) if f.startswith("model_") and f.endswith(".h5")])
    models = []
    for file in model_files:
        full_path = os.path.join(folder_path, file)
        model = copy.deepcopy(basis_model)
        model.load(full_path)
        models.append(model)
        print(f"Loaded {full_path} into new model object")
    return models
